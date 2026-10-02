"""
Shared, cached object tracking for PhysicsLENS Stage 2+.
-------------------------------------------------------
Shi-Tomasi + Lucas-Kanade sparse tracking, clustered into persistent object
tracks. Previously this logic lived inside object_tracker.py, forcing every
downstream stage (trajectory_extractor, the specialists) to re-run it — wasting
compute AND, worse, producing *different* tracks per stage so their evidence
disagreed about what the objects even were.

`get_tracks(video_path)` centralises it: one canonical track set per
(video, params), memoised by content hash. Tracks use ABSOLUTE frame indices
(into the returned frame list) so frame references are consistent across stages.

Also the home for the geometry/colour helpers shared by the Stage 2 pipelines.
"""
from typing import Optional

import cv2
import numpy as np

from tools.video import load_frames, frame_to_gray
from tools.flow  import detect_keypoints, track_keypoints
from tools.evidence import file_hash

_PALETTE = [
    '#1a54c4', '#c05621', '#7c3aed', '#1a7a3c',
    '#e24b4a', '#d97706', '#0891b2', '#be185d',
    '#0f766e', '#7e22ce',
]

CROP_PAD    = 0.10   # fractional padding around bounding box
MIN_CROP_PX = 12     # ignore degenerate boxes smaller than this

# Canonical defaults — stages that keep these share one cache entry (and thus
# an identical, consistent track set).
DEFAULT_NUM_KP      = 60
DEFAULT_MAX_OBJECTS = 8

_TRACK_CACHE: dict = {}     # (hash, num_kp, sample_every, max_objects) -> (tracks, meta)


# ── Geometry / colour helpers ─────────────────────────────────────────────────

def _box_from_points(pts: np.ndarray, H: int, W: int) -> Optional[tuple]:
    """Padded XYXY box for a set of (x, y) points, or None if degenerate."""
    if len(pts) == 0:
        return None
    x0, x1 = int(pts[:, 0].min()), int(pts[:, 0].max())
    y0, y1 = int(pts[:, 1].min()), int(pts[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    if bw < MIN_CROP_PX:
        pad = (MIN_CROP_PX - bw) // 2 + 4
        x0, x1 = x0 - pad, x1 + pad
        bw = x1 - x0
    if bh < MIN_CROP_PX:
        pad = (MIN_CROP_PX - bh) // 2 + 4
        y0, y1 = y0 - pad, y1 + pad
        bh = y1 - y0
    if bw < MIN_CROP_PX or bh < MIN_CROP_PX:
        return None
    px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
    return (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))


def _hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _cluster_points(pts: np.ndarray, dist_thresh: float) -> list[list[int]]:
    """Greedy single-linkage clustering by pairwise Euclidean distance."""
    assigned = [False] * len(pts)
    clusters: list[list[int]] = []
    for i in range(len(pts)):
        if assigned[i]:
            continue
        grp = [i]; assigned[i] = True
        for j in range(i + 1, len(pts)):
            if not assigned[j] and np.linalg.norm(pts[i] - pts[j]) < dist_thresh:
                grp.append(j); assigned[j] = True
        clusters.append(grp)
    return clusters


# ── Core extraction (LK tracking + clustering) ────────────────────────────────

def _extract(frames: list, num_kp: int, max_objects: int):
    """Return (tracks, meta). Tracks carry ABSOLUTE frame indices. Pure-CPU."""
    n = len(frames)
    H, W = frames[0].shape[:2]

    # Seek the first frame with detectable corners (skip featureless lead-in).
    start, pts0, gray0 = 0, None, None
    while start <= n - 3:
        gray0 = frame_to_gray(frames[start])
        pts0 = detect_keypoints(gray0, n=num_kp)
        if pts0 is not None and len(pts0) > 0:
            break
        start += 1
    meta = {"n_frames": n, "H": H, "W": W, "start": start,
            "nk": 0, "kp_loss_pct": 0.0,
            "num_kp": num_kp, "max_objects": max_objects}
    if pts0 is None or len(pts0) == 0:
        return [], meta

    work = frames[start:]
    nw = len(work)
    nk = len(pts0)
    tracks = [[pts0[i, 0].tolist()] for i in range(nk)]
    active = [pts0[i:i + 1] for i in range(nk)]

    prev_gray = gray0
    for f in range(1, nw):
        curr_gray = frame_to_gray(work[f])
        for ki in range(nk):
            if active[ki] is None:
                tracks[ki].append(None)
                continue
            _, good = track_keypoints(prev_gray, curr_gray, active[ki])
            if len(good) == 0:
                active[ki] = None
                tracks[ki].append(None)
            else:
                active[ki] = good.reshape(-1, 1, 2)
                tracks[ki].append(good.reshape(-1, 2)[0].tolist())
        prev_gray = curr_gray

    kp_lost = sum(1 for ki in range(nk) if active[ki] is None)
    meta["nk"] = nk
    meta["kp_loss_pct"] = kp_lost / max(nk, 1)

    dist_thresh = max(40.0, min(W, H) * 0.08)
    clusters = _cluster_points(pts0[:, 0, :], dist_thresh)

    obj_tracks: list[dict] = []
    for idx_list in clusters:
        f_idxs, boxes, cx, cy = [], [], [], []
        for f in range(nw):
            pts = [tracks[ki][f] for ki in idx_list if tracks[ki][f] is not None]
            if not pts:
                continue
            box = _box_from_points(np.array(pts), H, W)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            f_idxs.append(f + start)                 # absolute frame index
            boxes.append(box)
            cx.append((x0 + x1) / 2.0)
            cy.append((y0 + y1) / 2.0)
        if len(f_idxs) < 3:
            continue
        obj_tracks.append({
            "id": 0, "frames": f_idxs, "boxes": boxes,
            "cx": cx, "cy": cy, "n_kp": len(idx_list),
        })

    obj_tracks.sort(key=lambda t: -len(t["frames"]))
    obj_tracks = obj_tracks[:max_objects]
    for new_id, ct in enumerate(obj_tracks):
        ct["id"] = new_id
    return obj_tracks, meta


# ── Public, cached entry point ────────────────────────────────────────────────

def get_tracks(video_path: str, *, num_kp: int = DEFAULT_NUM_KP,
               sample_every: int = 1, max_objects: int = DEFAULT_MAX_OBJECTS) -> dict:
    """Load frames and return canonical object tracks for the video.

    Returns {"frames": [...], "fps": eff_fps, "meta": {...}, "tracks": [...]}.
    The expensive LK pass is memoised by (content hash, params); frames are
    (re)loaded each call (cheap relative to tracking) and never cached.
    """
    num_kp = max(5, int(num_kp))
    sample_every = max(1, int(sample_every))
    frames, raw_fps = load_frames(video_path, step=sample_every)
    eff_fps = (raw_fps or 30.0) / sample_every

    if len(frames) < 3:
        return {"frames": frames, "fps": eff_fps,
                "meta": {"n_frames": len(frames), "H": 0, "W": 0, "start": 0,
                         "nk": 0, "kp_loss_pct": 0.0,
                         "num_kp": num_kp, "max_objects": max_objects},
                "tracks": []}

    key = (file_hash(video_path), num_kp, sample_every, max_objects)
    hit = _TRACK_CACHE.get(key)
    if hit is None:
        tracks, meta = _extract(frames, num_kp, max_objects)
        _TRACK_CACHE[key] = (tracks, meta)
    else:
        tracks, meta = hit

    return {"frames": frames, "fps": eff_fps, "meta": meta, "tracks": tracks}
