"""
Stage 2 · Step 2 — Trajectory Extractor
------------------------------------------
Converts raw bounding-box tracks (re-derived from the same Shi-Tomasi + LK
pipeline as object_tracker.py) into higher-level kinematic descriptors and
flags statistically anomalous motion — the kind that betrays AI-generated
physics (teleport spikes, force-free velocity reversals, non-smooth jerk).

Because the harness passes only `video_path` (no inter-pipeline state), this
module re-runs sparse tracking internally, reusing object_tracker's geometry
helpers, then builds kinematics on top of the resulting tracks.

Kinematics (per track, on its sampled frame grid)
  • Position      — smoothed box centroid (px)
  • Velocity      — d/dt position           (px/s)  via uneven-grid gradient
  • Acceleration  — d/dt velocity           (px/s²)
  • Jerk          — d/dt acceleration       (px/s³)  (smoothness probe)
  • Angular proxy — d/dt box aspect ratio   (1/s)
  • Contacts      — frames where two tracks' boxes overlap / touch

Anomaly detection (calibration-free, statistical)
  • accel_spike       — robust z-score of |a| exceeds threshold
  • velocity_reversal — direction flip while moving (force-free reversal)
  • contact           — inter-object collision frames

Emits a Stage-1-compatible `signal` event so the Event Localizer can crop
windows around trajectory anomalies, plus plots, metrics and a severity score.

Robust by design: no scipy/torch hard dependency, guards every divide, and
always terminates with a `done` (or `error`) event.
"""
import asyncio, base64, json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video    import encode_video_browser
from tools.tracking import get_tracks, _hex_to_bgr, _PALETTE
from tools.evidence import EVIDENCE, video_id


# ── Numeric helpers (all NaN/zero-safe) ───────────────────────────────────────

def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing with a moving-average fallback. Window is
    clamped to an odd value ≤ len(y); short series are returned untouched."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3 or window < 3:
        return y
    w = min(window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y
    try:
        from scipy.signal import savgol_filter
        return np.asarray(savgol_filter(y, w, min(2, w - 1)), dtype=float)
    except Exception:
        kernel = np.ones(w) / w
        pad = w // 2
        padded = np.pad(y, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")


def _grad(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """First derivative on a possibly uneven time grid. Returns zeros for
    series too short to differentiate."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.zeros_like(v)
    t = np.asarray(t, dtype=float)
    # Guard against duplicate timestamps (zero dt → inf gradient).
    if np.any(np.diff(t) <= 0):
        t = np.arange(len(v), dtype=float)
    return np.gradient(v, t, edge_order=1)


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score — outlier-resistant. Falls back to std, then to a
    flat-zero array when the series has no spread."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(x))
    if scale < 1e-9:
        return np.zeros_like(x)
    return (x - med) / scale


def _box_iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def _box_gap(a: tuple, b: tuple) -> float:
    """Minimum edge-to-edge distance between two XYXY boxes; 0 if they touch
    or overlap."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(np.hypot(dx, dy))


# ── Kinematics + anomaly scoring per track ────────────────────────────────────

def _compute_kinematics(ct: dict, fps: float, smooth_w: int, z_thresh: float) -> dict:
    """Attach smoothed position, velocity, acceleration, jerk, angular proxy
    and per-frame anomaly flags to a track. Pure-numeric, no I/O."""
    frames = np.asarray(ct["frames"], dtype=float)
    t = frames / max(fps, 1e-6)                       # seconds (uneven if gaps)

    x = _smooth(np.asarray(ct["cx"], float), smooth_w)
    y = _smooth(np.asarray(ct["cy"], float), smooth_w)

    vx, vy = _grad(x, t), _grad(y, t)                 # px/s
    speed = np.hypot(vx, vy)
    ax, ay = _grad(vx, t), _grad(vy, t)               # px/s²
    accel = np.hypot(ax, ay)
    jerk = np.abs(_grad(accel, t))                    # px/s³

    boxes = np.asarray(ct["boxes"], float)
    w_box = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    h_box = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    aspect = w_box / h_box
    ang_rate = np.abs(_grad(aspect, t))               # 1/s (deformation proxy)

    # Dynamic significance floors keep noise on near-static tracks from firing.
    speed_floor = max(1.0, 0.15 * float(np.percentile(speed, 75)) if len(speed) else 1.0)
    accel_floor = max(1.0, float(np.median(accel)))

    z_acc = _robust_z(accel)
    acc_raw = set(int(i) for i in np.where((z_acc > z_thresh) & (accel > accel_floor))[0])

    # Direction reversals: a component sign flip is only real if THAT component
    # carries meaningful speed on both sides of the flip — otherwise it's LK
    # jitter crossing zero (e.g. horizontal motion with vertical noise).
    rev_raw = set()
    for comp in (vx, vy):
        s = np.sign(comp)
        c = np.abs(comp)
        real = (s[:-1] * s[1:] < 0) & (c[:-1] > speed_floor) & (c[1:] > speed_floor)
        rev_raw.update(int(i) for i in (np.where(real)[0] + 1))

    # ── Physics reconciliation ────────────────────────────────────────────────
    # A reversal that coincides with an acceleration impulse is a real IMPACT
    # (bounce / collision) — physically valid, not an anomaly. What's suspicious
    # is the *unexplained* remainder:
    #   • accel spike with NO reversal nearby → teleport / force-without-contact
    #   • reversal with NO accel impulse      → force-free direction change
    gate = max(1, int(round(0.06 * max(fps, 1e-6))))   # ≈60 ms coincidence window

    def _near(i, pool):
        return any(abs(i - j) <= gate for j in pool)

    impacts   = sorted(i for i in (acc_raw | rev_raw)
                       if _near(i, acc_raw) and _near(i, rev_raw))
    acc_flags = sorted(i for i in acc_raw if not _near(i, rev_raw))
    rev_flags = sorted(i for i in rev_raw if not _near(i, acc_raw))

    ct.update({
        "t": t, "x": x, "y": y, "vx": vx, "vy": vy, "speed": speed,
        "ax": ax, "ay": ay, "accel": accel, "jerk": jerk, "ang_rate": ang_rate,
        "z_acc": z_acc, "acc_flags": acc_flags, "rev_flags": rev_flags,
        "impacts": impacts, "gate": gate,
        "speed_floor": speed_floor, "accel_floor": accel_floor,
    })
    return ct


def _detect_contacts(obj_tracks: list[dict], fps: float, pad_px: float) -> list[dict]:
    """Per-frame box-overlap detection between track pairs, collapsed into
    discrete contact events (rising edge of each contiguous contact run)."""
    # frame_idx -> {track_id: box}
    by_frame: dict[int, dict] = {}
    for ct in obj_tracks:
        for f, box in zip(ct["frames"], ct["boxes"]):
            by_frame.setdefault(int(f), {})[ct["id"]] = box

    active_pairs: set = set()
    events: list[dict] = []
    for f in sorted(by_frame):
        present = by_frame[f]
        ids = sorted(present)
        touching = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = present[ids[i]], present[ids[j]]
                if _box_iou(a, b) > 0.0 or _box_gap(a, b) <= pad_px:
                    pair = (ids[i], ids[j])
                    touching.add(pair)
                    if pair not in active_pairs:     # rising edge = new contact
                        events.append({
                            "frame": f, "t": f / max(fps, 1e-6),
                            "track_a": ids[i], "track_b": ids[j],
                            "iou": round(_box_iou(a, b), 3),
                        })
        active_pairs = touching
    return events


def _gate_by_contacts(obj_tracks: list[dict], contacts: list[dict]) -> None:
    """Demote any residual accel/reversal flags that sit on a tracked
    inter-object contact — those are explained collisions, not anomalies.
    Mutates each track's flag lists in place."""
    by_track: dict[int, list[int]] = {}
    for ev in contacts:
        by_track.setdefault(ev["track_a"], []).append(ev["frame"])
        by_track.setdefault(ev["track_b"], []).append(ev["frame"])
    for ct in obj_tracks:
        cframes = by_track.get(ct["id"])
        if not cframes:
            continue
        gate = ct["gate"]

        def explained(pos: int) -> bool:
            fr = ct["frames"][pos]
            return any(abs(fr - c) <= gate for c in cframes)

        moved = [i for i in ct["acc_flags"] if explained(i)]
        ct["acc_flags"] = [i for i in ct["acc_flags"] if not explained(i)]
        ct["rev_flags"] = [i for i in ct["rev_flags"] if not explained(i)]
        ct["impacts"] = sorted(set(ct["impacts"]) | set(moved))


# ── Annotated overlay video ───────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, org, color, scale=0.5):
    """Draw text with a filled background chip for legibility on busy frames."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    y = max(th + 4, y)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 3), color, -1)
    cv2.putText(img, text, (x, y), _FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _render_trajectory_overlay(frames, obj_tracks, contacts, fps,
                               max_frames=240, max_w=720, trail_len=45):
    """Overlay motion trails, velocity arrows and anomaly callouts onto the
    video. Returns (annotated_frames, subsample_step)."""
    n = len(frames)
    step = max(1, -(-n // max_frames))   # ceil division

    # Per-track lookups keyed by ORIGINAL frame index.
    tinfo = []
    for ct in obj_tracks:
        col = _hex_to_bgr(_PALETTE[ct["id"] % len(_PALETTE)])
        fr = np.asarray(ct["frames"], dtype=int)
        tinfo.append({
            "ct": ct, "color": col, "frames": fr,
            "x": np.asarray(ct["x"], float), "y": np.asarray(ct["y"], float),
            "vx": ct["vx"], "vy": ct["vy"],
            "f2i": {int(f): i for i, f in enumerate(fr)},
            "acc": {int(ct["frames"][i]) for i in ct["acc_flags"]},
            "rev": {int(ct["frames"][i]) for i in ct["rev_flags"]},
            "imp": {int(ct["frames"][i]) for i in ct["impacts"]},
        })

    contacts_by_f: dict[int, list] = {}
    for ev in contacts:
        contacts_by_f.setdefault(int(ev["frame"]), []).append((ev["track_a"], ev["track_b"]))
    box_at = {ti["ct"]["id"]: ti for ti in tinfo}

    out = []
    for f in range(0, n, step):
        img = frames[f].copy()
        for ti in tinfo:
            ct, color = ti["ct"], ti["color"]
            seen = ti["frames"] <= f
            if not seen.any():
                continue
            xs, ys = ti["x"][seen], ti["y"][seen]
            pts = np.stack([xs, ys], axis=1).astype(int)

            # Fading motion trail (recent = brighter / thicker).
            tail = pts[-trail_len:]
            for k in range(1, len(tail)):
                a = 0.25 + 0.75 * k / len(tail)
                c = tuple(int(v * a) for v in color)
                cv2.line(img, tuple(tail[k - 1]), tuple(tail[k]), c, 2, cv2.LINE_AA)

            idx = ti["f2i"].get(f)
            cur = (int(pts[-1][0]), int(pts[-1][1]))
            if idx is not None:
                x0, y0, x1, y1 = ct["boxes"][idx]
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                cur = (int(ti["x"][idx]), int(ti["y"][idx]))
                # Velocity arrow = predicted displacement over the next 150 ms.
                ex = int(cur[0] + ti["vx"][idx] * 0.15)
                ey = int(cur[1] + ti["vy"][idx] * 0.15)
                if (ex - cur[0]) ** 2 + (ey - cur[1]) ** 2 > 9:
                    cv2.arrowedLine(img, cur, (ex, ey), (0, 255, 255), 2,
                                    cv2.LINE_AA, tipLength=0.3)
                _label(img, f"obj {ct['id']}", (x0 + 2, y0 - 4), color)

            # Anomaly / impact callouts at this exact frame.
            if f in ti["acc"]:
                cv2.circle(img, cur, 27, (40, 40, 230), 3, cv2.LINE_AA)
                _label(img, "ACCEL SPIKE", (cur[0] - 24, cur[1] - 30), (40, 40, 230))
            elif f in ti["rev"]:
                cv2.circle(img, cur, 27, (200, 60, 200), 3, cv2.LINE_AA)
                _label(img, "REVERSAL", (cur[0] - 20, cur[1] - 30), (200, 60, 200))
            elif f in ti["imp"]:
                cv2.circle(img, cur, 22, (70, 200, 70), 2, cv2.LINE_AA)

        # Contact links between colliding objects.
        for ta, tb in contacts_by_f.get(f, []):
            ia, ib = box_at.get(ta), box_at.get(tb)
            if ia is None or ib is None:
                continue
            pa, pb = ia["f2i"].get(f), ib["f2i"].get(f)
            if pa is None or pb is None:
                continue
            ca = (int(ia["x"][pa]), int(ia["y"][pa]))
            cb = (int(ib["x"][pb]), int(ib["y"][pb]))
            cv2.line(img, ca, cb, (0, 200, 255), 2, cv2.LINE_AA)
            mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
            _label(img, "CONTACT", (mid[0] - 18, mid[1]), (0, 165, 255))

        # HUD.
        _label(img, f"t = {f / max(fps, 1e-6):.2f}s", (8, 20), (40, 40, 40), 0.55)
        out.append(img)
    return out, step


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg       = json.loads(settings) if settings else {}
    num_kp    = max(10, int(cfg.get("num_keypoints", 60)))
    smooth_w  = max(1,  int(cfg.get("smoothing_window", 7)))
    z_thresh  = max(1.5, float(cfg.get("z_threshold", 3.5)))
    fps_over  = float(cfg.get("fps", 0) or 0)
    render_video = str(cfg.get("render_video", "true")).lower() not in ("false", "0", "no")

    try:
        yield {"type": "log", "level": "info", "text": "Loading video & tracking…"}
        await asyncio.sleep(0)

        # ── Phase 1: tracks. Prefer the Object Tracker's SAM3 mask tracks on
        # the evidence bus (already computed, labeled, centroid-accurate —
        # zero added compute). Fall back to shared cached LK tracks.
        loop = asyncio.get_event_loop()
        obj_tracks, track_src = [], "lk"
        ev_tr = EVIDENCE.get(video_id(video_path), "s2_object_tracker")
        if ev_tr and ev_tr.get("mode") == "sam3" and ev_tr.get("tracks"):
            from tools.video import load_frames as _load_frames
            frames, _raw_fps = await loop.run_in_executor(
                None, _load_frames, video_path)
            if len(frames) == int(ev_tr["n_frames"]):
                obj_tracks = [dict(ct) for ct in ev_tr["tracks"]
                              if len(ct.get("frames", [])) >= 3]
                if obj_tracks:
                    track_src = "sam3"
                    H, W = frames[0].shape[:2]
                    n = len(frames)
                    nk, kp_loss_pct, start = 0, 0.0, 0
                    fps = fps_over if fps_over > 0 else float(ev_tr["fps"])
                    names = ", ".join(f"“{ct['label']}”" for ct in obj_tracks
                                      if ct.get("label"))
                    yield {"type": "log", "level": "info",
                           "text": f"Reusing {len(obj_tracks)} SAM3 mask track(s) "
                                   f"from the Object Tracker ({names})."}
            else:
                yield {"type": "log", "level": "warn",
                       "text": "Object Tracker evidence has a different frame grid "
                               "— falling back to LK tracks."}

        if not obj_tracks:
            tr = await loop.run_in_executor(
                None, lambda: get_tracks(video_path, num_kp=num_kp, max_objects=8))
            frames = tr["frames"]
            meta   = tr["meta"]
            # Shallow-copy so kinematics keys we attach don't leak into the
            # shared track cache (the lists inside are read-only here).
            obj_tracks = [dict(ct) for ct in tr["tracks"]]
            n  = meta["n_frames"]
            H, W = meta["H"], meta["W"]
            nk = meta["nk"]
            kp_loss_pct = meta["kp_loss_pct"]
            start = meta["start"]
            fps = fps_over if fps_over > 0 else float(tr["fps"])

            # Static-track filter: Shi-Tomasi corners latch onto textured
            # background, producing tracks that never move — junk for
            # trajectory analysis. Keep movers; if nothing moves enough, keep
            # the single most-moving track rather than none.
            def _disp(ct):
                cx, cy = np.asarray(ct["cx"], float), np.asarray(ct["cy"], float)
                return float(np.hypot(cx.max() - cx.min(), cy.max() - cy.min()))
            if obj_tracks:
                min_disp = max(8.0, 0.02 * float(np.hypot(W, H)))
                movers = [ct for ct in obj_tracks if _disp(ct) >= min_disp]
                if not movers:
                    movers = [max(obj_tracks, key=_disp)]
                    yield {"type": "log", "level": "warn",
                           "text": "No track moved significantly — keeping only "
                                   "the most-moving one (video may be near-static)."}
                elif len(movers) < len(obj_tracks):
                    yield {"type": "log", "level": "info",
                           "text": f"Filtered {len(obj_tracks) - len(movers)} static "
                                   f"background track(s) (moved < {min_disp:.0f}px); "
                                   f"{len(movers)} moving track(s) kept."}
                obj_tracks = movers

        n_obj = len(obj_tracks)
        if n < 3:
            yield {"type": "error", "text": f"Video too short ({n} frames)."}
            yield {"type": "done"}
            return

        yield {"type": "log", "level": "info",
               "text": f"{n} frames @ {fps:.1f} fps — {n_obj} track(s) "
                       f"({'SAM3 masks' if track_src == 'sam3' else 'LK keypoints'})."}
        await asyncio.sleep(0)

        if n_obj == 0:
            yield {"type": "log", "level": "warn",
                   "text": "No trackable objects found — nothing to analyze."}
            yield {"type": "metric", "label": "Objects analyzed", "value": "0", "sub": "no tracks"}
            yield {"type": "severity", "label": "Trajectory anomaly score",
                   "value": 0, "color": "#4CAF50"}
            yield {"type": "done"}
            return

        if start > 0:
            yield {"type": "log", "level": "info",
                   "text": f"Skipped {start} featureless lead-in frame(s)."}
        yield {"type": "log", "level": "info",
               "text": f"{n_obj} track(s) extracted — computing kinematics…"}
        await asyncio.sleep(0)

        # ── Phase 2: kinematics + statistical anomalies ───────────────────────
        for ct in obj_tracks:
            _compute_kinematics(ct, fps, smooth_w, z_thresh)
        contacts = _detect_contacts(obj_tracks, fps, pad_px=max(2.0, 0.01 * min(W, H)))
        _gate_by_contacts(obj_tracks, contacts)
        await asyncio.sleep(0)

        # ── Phase 3: assemble signals for the Event Localizer ─────────────────
        signals: list[dict] = []
        n_acc = n_rev = 0
        for ct in obj_tracks:
            for i in ct["acc_flags"]:
                z = float(ct["z_acc"][i]) if i < len(ct["z_acc"]) else 0.0
                signals.append({"frame": int(ct["frames"][i]), "signal_type": "accel_spike",
                                "score": round(min(z / max(z_thresh, 1e-6), 3.0), 3)})
                n_acc += 1
            for i in ct["rev_flags"]:
                sp = float(ct["speed"][i]) if i < len(ct["speed"]) else 0.0
                fl = max(ct["speed_floor"], 1e-6)
                signals.append({"frame": int(ct["frames"][i]), "signal_type": "velocity_reversal",
                                "score": round(min(sp / fl, 3.0), 3)})
                n_rev += 1
        for ev in contacts:
            signals.append({"frame": int(ev["frame"]), "signal_type": "contact",
                            "score": round(0.5 + 0.5 * ev["iou"], 3)})

        total_kpts  = sum(len(ct["frames"]) for ct in obj_tracks)
        n_impacts   = sum(len(ct["impacts"]) for ct in obj_tracks)
        anom_frames = len({s["frame"] for s in signals
                           if s["signal_type"] != "contact"})

        # Severity reflects *unexplained* motion only — impacts and contacts are
        # physically valid events and never inflate the anomaly score.
        accel_sev = min(100, int(n_acc / max(total_kpts, 1) * 250))
        rev_sev   = min(100, int(n_rev / max(total_kpts, 1) * 250))
        contact_sev = min(100, int(len(contacts) * 12))   # reported, not summed in
        severity  = min(100, int(anom_frames / max(total_kpts, 1) * 250))
        color = "#E24B4A" if severity > 40 else "#EF9F27" if severity > 15 else "#4CAF50"

        # ── Phase 4: plots ────────────────────────────────────────────────────
        yield {"type": "log", "level": "info", "text": "Building kinematic plots…"}
        await asyncio.sleep(0)

        kin = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("Speed |v| (px/s)", "Acceleration |a| (px/s²) — flagged spikes"),
        )
        for ct in obj_tracks:
            color_t = _PALETTE[ct["id"] % len(_PALETTE)]
            ts = ct["t"].tolist()
            kin.add_trace(go.Scatter(
                x=ts, y=ct["speed"].tolist(), mode="lines", name=f"obj {ct['id']}",
                line=dict(color=color_t, width=1.6),
                legendgroup=f"obj{ct['id']}",
                hovertemplate="<b>t=%{x:.2f}s</b><br>speed=%{y:.0f}px/s<extra></extra>",
            ), row=1, col=1)
            # reversal markers on the speed lane
            if ct["rev_flags"]:
                kin.add_trace(go.Scatter(
                    x=[ts[i] for i in ct["rev_flags"] if i < len(ts)],
                    y=[ct["speed"][i] for i in ct["rev_flags"] if i < len(ts)],
                    mode="markers", name="velocity reversal", showlegend=(ct["id"] == 0),
                    legendgroup="rev",
                    marker=dict(color="#7c3aed", size=8, symbol="x"),
                    hovertemplate="<b>reversal</b><br>t=%{x:.2f}s<extra></extra>",
                ), row=1, col=1)

            kin.add_trace(go.Scatter(
                x=ts, y=ct["accel"].tolist(), mode="lines", showlegend=False,
                line=dict(color=color_t, width=1.4), legendgroup=f"obj{ct['id']}",
                hovertemplate="<b>t=%{x:.2f}s</b><br>accel=%{y:.0f}px/s²<extra></extra>",
            ), row=2, col=1)
            if ct["acc_flags"]:
                kin.add_trace(go.Scatter(
                    x=[ts[i] for i in ct["acc_flags"] if i < len(ts)],
                    y=[ct["accel"][i] for i in ct["acc_flags"] if i < len(ts)],
                    mode="markers", name="accel spike", showlegend=(ct["id"] == 0),
                    legendgroup="acc",
                    marker=dict(color="#e24b4a", size=8, symbol="circle-open",
                                line=dict(width=2)),
                    hovertemplate="<b>accel spike</b><br>t=%{x:.2f}s<br>"
                                  "|a|=%{y:.0f}px/s²<extra></extra>",
                ), row=2, col=1)

        for ev in contacts:
            kin.add_vline(x=ev["t"], line=dict(color="#1a1917", dash="dot", width=1),
                          row=2, col=1)

        _grid = dict(showgrid=True, gridcolor="#ebebeb")
        kin.update_yaxes(title_text="px/s", row=1, col=1, **_grid)
        kin.update_yaxes(title_text="px/s²", row=2, col=1, rangemode="tozero", **_grid)
        kin.update_xaxes(title_text="Time (s)", row=2, col=1, **_grid)
        kin.update_xaxes(row=1, col=1, **_grid)
        kin.update_layout(
            title=dict(text="Trajectory Kinematics & Motion Anomalies", font=dict(size=15)),
            height=520, legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=65, r=40, t=100, b=55),
            font=dict(family="IBM Plex Sans, sans-serif", size=13),
        )
        yield {"type": "plotly", "data": kin.to_json(),
               "caption": ("Top: per-object speed (× = force-free direction reversal). "
                           "Bottom: acceleration with flagged spikes; dotted lines = contacts.")}
        await asyncio.sleep(0)

        # Spatial trajectory map (pixel space, y flipped so up = up).
        spatial = go.Figure()
        for ct in obj_tracks:
            color_t = _PALETTE[ct["id"] % len(_PALETTE)]
            xs = ct["x"].tolist()
            ys = [H - v for v in ct["y"].tolist()]
            spatial.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=f"obj {ct['id']}",
                line=dict(color=color_t, width=1.6), marker=dict(size=3, color=color_t),
                hovertemplate="x=%{x:.0f}px<br>y=%{y:.0f}px<extra></extra>",
            ))
            spatial.add_trace(go.Scatter(
                x=[xs[0]], y=[ys[0]], mode="markers", showlegend=False,
                marker=dict(color=color_t, size=11, symbol="circle",
                            line=dict(color="white", width=2)),
                hovertemplate=f"obj {ct['id']} start<extra></extra>"))
            if ct["acc_flags"]:
                spatial.add_trace(go.Scatter(
                    x=[xs[i] for i in ct["acc_flags"] if i < len(xs)],
                    y=[ys[i] for i in ct["acc_flags"] if i < len(ys)],
                    mode="markers", showlegend=False,
                    marker=dict(color="#e24b4a", size=9, symbol="x"),
                    hovertemplate="accel spike<extra></extra>"))
        spatial.update_xaxes(title_text="X (px)", range=[0, W], showgrid=True, gridcolor="#ebebeb")
        spatial.update_yaxes(title_text="Y (px, up = up)", range=[0, H],
                             showgrid=True, gridcolor="#ebebeb", scaleanchor="x", scaleratio=1)
        spatial.update_layout(
            title=dict(text="Spatial Trajectories (○ = start, × = accel spike)", font=dict(size=15)),
            height=460, legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=65, r=40, t=90, b=55),
            font=dict(family="IBM Plex Sans, sans-serif", size=13),
        )
        yield {"type": "plotly", "data": spatial.to_json(),
               "caption": "Top-down centroid paths in pixel space; red × marks acceleration spikes."}
        await asyncio.sleep(0)

        # ── Phase 4b: annotated overlay video ─────────────────────────────────
        if render_video:
            yield {"type": "log", "level": "info",
                   "text": "Rendering kinematic overlay video…"}
            await asyncio.sleep(0)
            try:
                ann, ov_step = await loop.run_in_executor(
                    None, _render_trajectory_overlay, frames, obj_tracks, contacts, fps)
                data, mime = await loop.run_in_executor(
                    None, encode_video_browser, ann, fps / ov_step)
                yield {
                    "type": "video", "data": base64.b64encode(data).decode(), "mime": mime,
                    "caption": ("Motion trails + velocity arrows (yellow). Callouts: "
                                "ACCEL SPIKE / REVERSAL = anomalies, green ring = physical "
                                "impact, CONTACT = inter-object collision."),
                }
                yield {"type": "log", "level": "info",
                       "text": f"Overlay video ready ({len(data)/1024:.0f} KB, {mime})."}
            except Exception as exc:
                yield {"type": "log", "level": "warn",
                       "text": f"Overlay rendering failed ({exc}) — continuing."}
            await asyncio.sleep(0)

        # ── Phase 5: signal event for downstream localization ─────────────────
        yield {"type": "signal", "source": "s2_trajectory_extractor",
               "source_name": "Trajectory Extractor", "fps": float(fps),
               "n_frames": int(n), "severity": severity,
               "type_severities": {"accel_spike": accel_sev,
                                   "velocity_reversal": rev_sev,
                                   "contact": contact_sev},
               "signals": signals}

        # Publish kinematic trajectories to the evidence bus for Stage 3/4.
        def _ev_traj(ct):
            return {
                "track_id": int(ct["id"]),
                "frames": [int(f) for f in ct["frames"]],
                "positions":     [[float(a), float(b)] for a, b in zip(ct["x"], ct["y"])],
                "velocities":    [[float(a), float(b)] for a, b in zip(ct["vx"], ct["vy"])],
                "accelerations": [[float(a), float(b)] for a, b in zip(ct["ax"], ct["ay"])],
                "speed":  [float(v) for v in ct["speed"]],
                "accel":  [float(v) for v in ct["accel"]],
                "acc_flags": [int(i) for i in ct["acc_flags"]],
                "rev_flags": [int(i) for i in ct["rev_flags"]],
                "impacts":   [int(i) for i in ct["impacts"]],
            }
        EVIDENCE.put(video_id(video_path), "s2_trajectory_extractor", {
            "fps": float(fps), "n_frames": int(n), "severity": severity,
            "type_severities": {"accel_spike": accel_sev,
                                "velocity_reversal": rev_sev, "contact": contact_sev},
            "trajectories": [_ev_traj(ct) for ct in obj_tracks],
            "contacts": contacts,
            "signals": signals,
        })

        # ── Phase 6: metrics ──────────────────────────────────────────────────
        peak_speed = max((float(np.max(ct["speed"])) for ct in obj_tracks), default=0.0)
        peak_accel = max((float(np.max(ct["accel"])) for ct in obj_tracks), default=0.0)
        mean_speed = float(np.mean(np.concatenate([ct["speed"] for ct in obj_tracks])))

        yield {"type": "metric", "label": "Objects analyzed", "value": str(n_obj),
               "sub": ("SAM3 mask tracks (Object Tracker)" if track_src == "sam3"
                       else f"{kp_loss_pct:.0%} keypoint loss")}
        yield {"type": "metric", "label": "Peak speed", "value": f"{peak_speed:.0f}",
               "sub": "px/s (fastest track)"}
        yield {"type": "metric", "label": "Peak acceleration", "value": f"{peak_accel:.0f}",
               "sub": "px/s² (largest motion change)"}
        yield {"type": "metric", "label": "Accel spikes", "value": str(n_acc),
               "sub": f"robust z > {z_thresh:g}"}
        yield {"type": "metric", "label": "Velocity reversals", "value": str(n_rev),
               "sub": "force-free direction flips"}
        yield {"type": "metric", "label": "Physical impacts", "value": str(n_impacts),
               "sub": "explained collisions (not anomalies)"}
        yield {"type": "metric", "label": "Contact events", "value": str(len(contacts)),
               "sub": "inter-object box overlaps"}
        yield {"type": "severity", "label": "Trajectory anomaly score",
               "value": severity, "color": color}

        parts = []
        if n_acc:          parts.append(f"{n_acc} accel spike(s)")
        if n_rev:          parts.append(f"{n_rev} velocity reversal(s)")
        if contacts:       parts.append(f"{len(contacts)} contact(s)")
        msg = (", ".join(parts) + " detected.") if parts else \
            "No trajectory anomalies detected — motion is kinematically smooth."
        yield {"type": "log", "level": "warn" if parts else "success", "text": msg}
        yield {"type": "done"}

    except Exception as exc:
        import traceback
        yield {"type": "log", "level": "warn",
               "text": f"Trajectory extraction failed: {exc}"}
        yield {"type": "error", "text": f"{type(exc).__name__}: {exc}\n"
               f"{traceback.format_exc(limit=3)}"}
        yield {"type": "done"}
