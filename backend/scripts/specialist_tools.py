"""
Per-specialist PHYSICAL measurements — tools, not judgements.

Motivation. The per-specialist ablation splits cleanly: probes that detect
APPEARANCE anomalies work (fluid 0.88, friction 0.71, permanence 0.68) and
probes that need DYNAMICS do not (collision 0.59, deformation 0.58, gravity
0.56, momentum 0.56 — the last two at chance). That is the downstream
consequence of the shuffle ablation: these models read static appearance, not
motion. Asking the VLM harder cannot fix it; measuring the motion can.

Each weak specialist has an exact geometric signature:

  gravity     an unsupported object must accelerate downward at a constant rate.
              Violation = sustained near-zero vertical acceleration while
              airborne (hovering), or upward acceleration with no cause.
  momentum    speed changes only when something acts on the object. Violation =
              impulsive speed change, or speed/energy INCREASING with no
              converging object.
  collision   the classic tell is action at a distance: a large velocity change
              at a moment when the nearest other object is still far away.
  deformation a rigid body keeps its internal distances. Violation = pairwise
              distances between points on one object drifting apart after
              global scale is normalised out.
  friction    contact should dissipate. Violation = a surface-contacting object
              sliding with no deceleration, or stopping instantly.
  permanence  tracks should persist. Violation = mass track death mid-clip.

CAMERA MOTION IS THE MAIN CONFOUND. A pan makes every track accelerate; a zoom
makes every pairwise distance grow. Both would masquerade as physics violations.
So all features are computed AFTER subtracting the per-frame median translation
and dividing out the per-frame median scale change — i.e. in a
camera-compensated frame.

Usage:
  python backend/scripts/specialist_tools.py --data data/videophy1200 --workers 8
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

G_DOWN = 1.0          # image y grows downward, so gravity is +y


def _read(path, max_frames=64, max_side=360):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 1:
        cap.release()
        return []
    want = set(int(t) for t in np.linspace(0, total - 1,
                                           min(max_frames, total)).astype(int))
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            h, w = fr.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                fr = cv2.resize(fr, (int(w * s), int(h * s)))
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return out


def build_tracks(g, max_pts=250):
    """-> P (T, N, 2) float array with NaN where a point is lost, plus alive mask."""
    p0 = cv2.goodFeaturesToTrack(g[0], maxCorners=max_pts, qualityLevel=0.01,
                                 minDistance=6)
    if p0 is None or len(p0) < 8:
        return None, None
    T, N = len(g), len(p0)
    P = np.full((T, N, 2), np.nan, dtype=np.float32)
    P[0] = p0.reshape(-1, 2)
    cur = p0
    idx = np.arange(N)
    for t in range(1, T):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(g[t - 1], g[t], cur, None)
        if nxt is None:
            break
        ok = st.reshape(-1) == 1
        P[t, idx[ok]] = nxt[ok].reshape(-1, 2)
        cur = nxt[ok].reshape(-1, 1, 2)
        idx = idx[ok]
        if len(idx) < 4:
            break
    return P, ~np.isnan(P[:, :, 0])


def compensate(P):
    """Remove per-frame median translation and median scale change.

    Without this a camera pan gives every track a constant acceleration and a
    zoom makes every internal distance grow — both read as physics violations.
    """
    T = P.shape[0]
    Q = P.copy()
    for t in range(1, T):
        d = Q[t] - Q[t - 1]
        m = np.isfinite(d[:, 0])
        if m.sum() < 4:
            continue
        Q[t] -= np.nanmedian(d[m], axis=0)          # translation
    # scale: normalise spread relative to frame 0
    ref = np.nanstd(Q[0][np.isfinite(Q[0][:, 0])], axis=0).mean()
    for t in range(1, T):
        m = np.isfinite(Q[t][:, 0])
        if m.sum() < 4 or ref < 1e-6:
            continue
        s = np.nanstd(Q[t][m], axis=0).mean()
        if s > 1e-6:
            c = np.nanmean(Q[t][m], axis=0)
            Q[t] = (Q[t] - c) * (ref / s) + c
    return Q


def features(path):
    g = _read(Path(path))
    if len(g) < 8:
        return None
    P, alive = build_tracks(g)
    if P is None:
        return None
    Q = compensate(P)
    T, N, _ = Q.shape
    V = np.diff(Q, axis=0)                       # velocity (T-1, N, 2)
    A = np.diff(V, axis=0)                       # acceleration (T-2, N, 2)
    spd = np.linalg.norm(V, axis=2)
    fin = np.isfinite(spd)

    f = {}

    # ── gravity ───────────────────────────────────────────────────────────────
    ay = A[:, :, 1]
    ax = A[:, :, 0]
    moving = np.isfinite(spd[:-1]) & (spd[:-1] > 0.35)
    m = np.isfinite(ay) & moving
    if m.sum() > 20:
        # airborne-ish points: moving, not near the bottom edge of the frame
        f["grav_ay_mean"] = float(np.nanmean(ay[m]))
        f["grav_ay_std"] = float(np.nanstd(ay[m]))
        # hover: moving points whose vertical acceleration is ~0 (should be +g)
        f["grav_hover_frac"] = float(np.mean(np.abs(ay[m]) < 0.06))
        # sustained upward acceleration with no cause
        f["grav_up_frac"] = float(np.mean(ay[m] < -0.25))
        # a real ballistic arc has ay dominating ax
        f["grav_aniso"] = float(np.nanmean(np.abs(ay[m])) /
                                (np.nanmean(np.abs(ax[np.isfinite(ax) & moving])) + 1e-6))
    # ── momentum ──────────────────────────────────────────────────────────────
    ds = np.diff(spd, axis=0)
    dm = np.isfinite(ds)
    if dm.sum() > 20:
        f["mom_impulse_p99"] = float(np.nanpercentile(np.abs(ds[dm]), 99))
        # speed GAINS are the suspicious direction: losing speed has many causes
        f["mom_gain_frac"] = float(np.mean(ds[dm] > 0.30))
        f["mom_gain_mean"] = float(np.nanmean(np.clip(ds[dm], 0, None)))
        # direction reversals without a contact event
        c = np.sum(V[:-1] * V[1:], axis=2)
        cm = np.isfinite(c)
        f["mom_reversal_frac"] = float(np.mean(c[cm] < 0)) if cm.sum() > 10 else 0.0
    # ── collision: action at a distance ───────────────────────────────────────
    # For the frame with the largest velocity change, how far was the nearest
    # other tracked point? A real impact happens at near-zero separation.
    best = 0.0
    for t in range(min(len(ds), T - 2)):
        row = np.abs(ds[t])
        if not np.isfinite(row).any():
            continue
        j = int(np.nanargmax(row))
        if not np.isfinite(row[j]) or row[j] < 0.25:
            continue
        pts = Q[t + 1]
        ok = np.isfinite(pts[:, 0])
        ok[j] = False
        if ok.sum() < 2 or not np.isfinite(pts[j][0]):
            continue
        d = np.linalg.norm(pts[ok] - pts[j], axis=1)
        best = max(best, float(row[j] * np.min(d)))    # big kick + far away
    f["coll_action_at_distance"] = best
    # ── deformation: internal distances of a rigid neighbourhood ──────────────
    base = Q[0]
    ok0 = np.isfinite(base[:, 0])
    ids = np.where(ok0)[0]
    cvs = []
    if len(ids) > 6:
        rng = np.random.RandomState(0)
        for _ in range(min(120, len(ids) * 2)):
            i, j = rng.choice(ids, 2, replace=False)
            d = np.linalg.norm(Q[:, i] - Q[:, j], axis=1)
            d = d[np.isfinite(d)]
            if len(d) > 6 and d.mean() > 3 and d.mean() < 60:   # local pairs only
                cvs.append(float(d.std() / (d.mean() + 1e-6)))
    f["deform_pair_cv"] = float(np.median(cvs)) if cvs else 0.0
    f["deform_pair_cv_p90"] = float(np.percentile(cvs, 90)) if cvs else 0.0
    # ── friction: does surface-contacting motion decay? ───────────────────────
    h = g[0].shape[0]
    low = np.nanmean(Q[:, :, 1], axis=0) > 0.6 * h        # lower part of frame
    if low.sum() > 3:
        s = spd[:, low]
        sm = np.isfinite(s)
        if sm.sum() > 15:
            t0 = np.nanmean(s[:max(1, len(s) // 3)])
            t1 = np.nanmean(s[-max(1, len(s) // 3):])
            f["fric_decay_ratio"] = float(t1 / (t0 + 1e-6))
            f["fric_abrupt_stop"] = float(np.nanmax(np.abs(np.diff(
                np.nanmean(s, axis=1)))) if s.shape[0] > 2 else 0.0)
    # ── permanence: track mortality ───────────────────────────────────────────
    n0 = int(alive[0].sum())
    f["perm_survival"] = float(alive[-1].sum() / max(n0, 1))
    deaths = -np.diff(alive.sum(axis=1).astype(float))
    f["perm_max_death_burst"] = float(np.max(deaths) / max(n0, 1)) if len(deaths) else 0.0
    return f


def _job(t):
    cid, p = t
    try:
        return cid, features(p)
    except Exception as e:  # noqa: BLE001
        return cid, {"_error": str(e)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    data = ROOT / a.data
    clips = json.loads((data / "manifest.json").read_text())["clips"]
    from videophy_prepare import ensure_videos, index_videos
    idx = index_videos(ensure_videos())
    jobs = []
    for c in clips:
        for stem, p in idx.items():
            if f"{c['generator']}__{stem}"[:110] == c["clip_id"]:
                jobs.append((c["clip_id"], str(p)))
                break
    print(f"matched {len(jobs)}/{len(clips)} videos", flush=True)

    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (cid, s) in enumerate(ex.map(_job, jobs), 1):
            if s and "_error" not in s:
                out[cid] = s
            if k % 100 == 0:
                print(f"    {k}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    outp = data / "specialist_tools.json"
    outp.write_text(json.dumps({"n": len(out), "features": out}, indent=1))
    names = sorted({k for v in out.values() for k in v})
    print(f"\n  {len(out)}/{len(jobs)} clips, {len(names)} features "
          f"({time.time()-t0:.0f}s)")
    print("  " + ", ".join(names))
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()
