"""
Object-level physical measurements — v2.

WHY v2. The v1 features (specialist_tools.py) scored ~0.50 for gravity and
momentum, and the features were NOT degenerate — ~890 distinct values each. The
error was what they measured, not how. v1 tracked Shi-Tomasi corners across the
whole frame, which are mostly background texture, then averaged. A gravity
violation involves ONE object; averaging its vertical acceleration together with
250 static background points erases it.

v2 segments the moving object first, then measures physics on that object's
centroid trajectory:

  1. track corners as before
  2. estimate BACKGROUND motion as the per-frame median velocity
  3. a track is FOREGROUND if its motion deviates persistently from background
  4. keep the largest coherent foreground cluster, take its centroid
  5. compute gravity / momentum / collision features on that ONE trajectory

The centroid of a moving object is what has a ballistic arc, an impact, a speed
profile. That is the object physics actually applies to.

Also fixes v1's silent 25% missing-feature rate (guards were too strict, so a
quarter of clips fell back to the column median and diluted every AUC).

Usage:
  python backend/scripts/specialist_tools_v2.py --data data/videophy1200 --workers 10
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
from specialist_tools import _read, build_tracks       # noqa: E402


def foreground_centroid(P, alive):
    """-> (T,2) centroid of the dominant moving object, in background-relative
    coordinates, plus the fraction of tracks judged foreground.

    Background-relative matters: if the camera pans, every track moves. What
    identifies an object is moving DIFFERENTLY from everything else.
    """
    T, N, _ = P.shape
    V = np.diff(P, axis=0)
    bg = np.nanmedian(V, axis=1, keepdims=True)          # (T-1,1,2)
    rel = V - bg                                          # background-removed
    dev = np.nansum(np.linalg.norm(rel, axis=2), axis=0)  # persistence of deviation
    cnt = np.sum(np.isfinite(rel[:, :, 0]), axis=0)
    dev = dev / np.maximum(cnt, 1)
    ok = np.isfinite(dev) & (cnt > max(3, T // 4))
    if ok.sum() < 4:
        return None, 0.0
    thr = np.nanpercentile(dev[ok], 70)                   # top 30% = object
    sel = ok & (dev >= thr)
    if sel.sum() < 3:
        return None, 0.0

    # Integrate background-relative velocity into a displacement trajectory,
    # so a camera pan cannot appear as object motion.
    C = np.zeros((T, 2), dtype=np.float64)
    C[0] = np.nanmean(P[0, sel], axis=0)
    for t in range(1, T):
        step = rel[t - 1, sel]
        step = step[np.isfinite(step[:, 0])]
        C[t] = C[t - 1] + (np.nanmean(step, axis=0) if len(step) else 0.0)
    return C, float(sel.sum() / max(N, 1))


def feats_from_traj(C, h):
    """Physics of a single object trajectory."""
    v = np.diff(C, axis=0)
    a = np.diff(v, axis=0)
    spd = np.linalg.norm(v, axis=1)
    f = {}
    if len(a) < 3:
        return f
    ay, ax = a[:, 1], a[:, 0]
    mov = spd[:-1] > 0.25
    f["o_ay_mean"] = float(np.mean(ay))
    f["o_ay_std"] = float(np.std(ay))
    # A real ballistic arc has near-constant downward ay. Two tells:
    #   consistency  — how constant is ay
    #   hover        — moving object with ~zero vertical acceleration
    f["o_ay_consistency"] = float(np.std(ay) / (abs(np.mean(ay)) + 1e-3))
    f["o_hover"] = float(np.mean(np.abs(ay[mov]) < 0.05)) if mov.sum() else 0.0
    f["o_up_accel"] = float(np.mean(ay[mov] < -0.15)) if mov.sum() else 0.0
    # fit ay to a constant: residual is how non-ballistic the vertical motion is
    f["o_ballistic_resid"] = float(np.mean((ay - np.mean(ay)) ** 2))
    # momentum
    ds = np.diff(spd)
    f["o_impulse_max"] = float(np.max(np.abs(ds))) if len(ds) else 0.0
    f["o_speed_gain"] = float(np.max(ds)) if len(ds) else 0.0
    f["o_gain_frac"] = float(np.mean(ds > 0.15)) if len(ds) else 0.0
    f["o_speed_cv"] = float(np.std(spd) / (np.mean(spd) + 1e-6))
    # direction changes
    if len(v) > 2:
        c = np.sum(v[:-1] * v[1:], axis=1)
        nn = np.linalg.norm(v[:-1], axis=1) * np.linalg.norm(v[1:], axis=1) + 1e-6
        f["o_turn_max"] = float(np.max(1.0 - c / nn))
        f["o_turn_mean"] = float(np.mean(1.0 - c / nn))
    # vertical extent relative to frame — is the object airborne at all
    f["o_y_range"] = float((C[:, 1].max() - C[:, 1].min()) / max(h, 1))
    f["o_path_len"] = float(np.sum(spd) / max(h, 1))
    return f


def features(path):
    g = _read(Path(path))
    if len(g) < 8:
        return None
    P, alive = build_tracks(g)
    if P is None:
        return None
    C, frac = foreground_centroid(P, alive)
    if C is None:
        return None
    f = feats_from_traj(C, g[0].shape[0])
    f["o_fg_frac"] = frac
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
    ap.add_argument("--workers", type=int, default=10)
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
    print(f"matched {len(jobs)}/{len(clips)}", flush=True)

    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (cid, s) in enumerate(ex.map(_job, jobs), 1):
            if s and "_error" not in s:
                out[cid] = s
            if k % 200 == 0:
                print(f"    {k}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    names = sorted({k for v in out.values() for k in v})
    cov = {t: sum(1 for v in out.values() if t in v) for t in names}
    outp = data / "specialist_tools_v2.json"
    outp.write_text(json.dumps({"n": len(out), "features": out}, indent=1))
    print(f"\n  {len(out)}/{len(jobs)} clips, {len(names)} features "
          f"({time.time()-t0:.0f}s)")
    print("  coverage: " + ", ".join(f"{t}={100*c/max(len(out),1):.0f}%"
                                     for t, c in sorted(cov.items())))
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()
