"""
Compute the Stage-1 and Stage-2 signals for every clip in a staged dataset.

PURPOSE. The specialists currently see 8 JPEGs and one sentence. Stages 1 and 2
already compute a great deal more than that — motion, tracks, where the clip goes
wrong — and none of it reaches Stage 3. This script produces those signals so we
can measure, per specialist, which upstream evidence is actually worth passing
down and which is decoration.

READ FROM THE ORIGINAL VIDEO, NOT THE 8 FROZEN FRAMES. Optical flow and track
kinematics over 8 samples spanning 5 seconds are meaningless — 0.7 s between
samples is longer than most of the events we care about. Everything here decodes
up to 48 frames from the source file. The 8 frozen frames stay the VLM's input;
these signals are a parallel channel, which is the point.

SIGNAL GROUPS
  s1_*   cheap screening, no tracking: frame differences, global flow, camera
         motion, flow entropy. This is what Stage 1 emits today.
  s2_*   tracking-derived: Lucas-Kanade tracks and per-track kinematics, plus a
         localized peak-anomaly time. This is what Stage 2 emits today.
  sp_*   specialist-targeted derivations built from the same tracks — vertical
         acceleration for gravity, speed gain across an impact for momentum,
         track death for permanence, and so on. These exist to test a sharper
         question than "does motion help": does the RIGHT motion statistic help
         the specialist it was designed for?

Camera motion is estimated as the median flow vector and subtracted before any
object statistic, because a panning camera moves every pixel and would otherwise
register as every object accelerating at once.

Usage:
  python backend/scripts/stage_signals.py --data data/robotbench \
      --videos data/robotbench_videos --workers 8
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

MAX_SIDE = 320
SAMPLE_FPS = 8.0      # samples per SECOND, not per clip
WINDOW_S = 5.0        # seconds of video to analyse, from the start
MAX_FRAMES = int(SAMPLE_FPS * WINDOW_S)


def read_gray(path, max_side=MAX_SIDE, fps=SAMPLE_FPS, window=WINDOW_S):
    """Frames on a fixed WALL-CLOCK grid, not a fixed count per clip.

    This is load-bearing for any real-vs-AI comparison. The real demos run ~370
    frames at 30 fps (~12 s); the generated clips are 121 frames at 24 fps
    (5.04 s). Taking a fixed 48 samples from each puts the AI samples 0.10 s
    apart and the real ones 0.26 s apart, so every velocity, acceleration and
    jerk statistic comes out ~2.5x smaller for AI purely from sampling. A first
    pass with fixed-count sampling "separated" real from AI at AUC 0.83 on jerk
    alone — that was the frame rate, not the physics.

    Fixing the sample rate AND the analysed duration makes a pixel displacement
    mean the same thing in both. Resolution is already normalised by max_side.
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if total <= 1:
        cap.release()
        return []
    if src_fps <= 1:
        src_fps = 24.0
    step = max(src_fps / fps, 1.0)                  # source frames per sample
    last = min(total - 1, int(window * src_fps))    # same duration for everyone
    want = set(int(round(x)) for x in np.arange(0, last + 1, step))
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok or i > last:
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


def _entropy(a, bins=16):
    if len(a) == 0:
        return 0.0
    h, _ = np.histogram(a, bins=bins, range=(-np.pi, np.pi))
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))


def stage1(g):
    """Screening signals: no tracking, no correspondence, just pixels and flow."""
    out = {}
    d = [np.abs(g[i + 1].astype(np.float32) - g[i].astype(np.float32))
         for i in range(len(g) - 1)]
    if d:
        m = np.array([x.mean() for x in d])
        out["s1_diff_mean"] = float(m.mean())
        out["s1_diff_p95"] = float(np.percentile(m, 95))
        out["s1_diff_max"] = float(m.max())
        # roughness of the difference curve: a real clip's motion energy varies
        # smoothly, generated video often steps or flickers
        out["s1_diff_jerk"] = float(np.abs(np.diff(m)).mean()) if len(m) > 1 else 0.0
        out["s1_diff_cv"] = float(m.std() / (m.mean() + 1e-6))

    mags, angs, cams, res = [], [], [], []
    for i in range(len(g) - 1):
        f = cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = f[..., 0], f[..., 1]
        # median vector == dominant global motion == camera
        cx, cy = float(np.median(fx)), float(np.median(fy))
        cams.append(float(np.hypot(cx, cy)))
        rx, ry = fx - cx, fy - cy          # residual == object motion
        rm = np.hypot(rx, ry)
        res.append(float(rm.mean()))
        mags.append(float(np.hypot(fx, fy).mean()))
        sel = rm > max(np.percentile(rm, 90), 1e-3)
        if sel.any():
            angs.append(_entropy(np.arctan2(ry[sel], rx[sel])))
    if mags:
        mg = np.array(mags)
        out["s1_flow_mean"] = float(mg.mean())
        out["s1_flow_p95"] = float(np.percentile(mg, 95))
        out["s1_flow_accel_p95"] = float(np.percentile(np.abs(np.diff(mg)), 95)) \
            if len(mg) > 1 else 0.0
        out["s1_cam_motion"] = float(np.mean(cams))
        out["s1_obj_motion"] = float(np.mean(res))
        # how much of the motion is the camera rather than the scene
        out["s1_cam_frac"] = float(np.mean(cams) / (mg.mean() + 1e-6))
        out["s1_flow_entropy"] = float(np.mean(angs)) if angs else 0.0
    return out


def _tracks(g, n_kp=120):
    """LK tracks as (T, N, 2) with NaN where a point was lost."""
    p0 = cv2.goodFeaturesToTrack(g[0], maxCorners=n_kp, qualityLevel=0.01,
                                 minDistance=7)
    if p0 is None or len(p0) < 4:
        return None
    T, N = len(g), len(p0)
    P = np.full((T, N, 2), np.nan, np.float32)
    P[0] = p0[:, 0, :]
    cur, idx = p0, np.arange(N)
    for t in range(1, T):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(g[t - 1], g[t], cur, None)
        if nxt is None:
            break
        st = st.reshape(-1).astype(bool)
        P[t, idx[st]] = nxt[st][:, 0, :]
        cur, idx = nxt[st], idx[st]
        if len(cur) < 3:
            break
    return P


def stage2(g):
    """Tracking-derived signals plus specialist-targeted derivations."""
    out = {}
    P = _tracks(g)
    if P is None:
        return out
    T, N, _ = P.shape
    alive = ~np.isnan(P[..., 0])
    out["s2_n_tracks"] = float(N)
    out["s2_track_survival"] = float(alive[-1].sum() / max(N, 1))
    # objects that vanish rather than leave frame: a track dying away from the
    # border has nothing physical to explain it
    H, W = g[0].shape
    died_inner = 0
    for j in range(N):
        a = np.where(alive[:, j])[0]
        if len(a) and a[-1] < T - 1:
            x, y = P[a[-1], j]
            if 0.08 * W < x < 0.92 * W and 0.08 * H < y < 0.92 * H:
                died_inner += 1
    out["s2_inner_death_frac"] = float(died_inner / max(N, 1))

    # camera-compensated velocities
    V = np.diff(P, axis=0)                              # (T-1, N, 2)
    cam = np.nanmedian(V, axis=1, keepdims=True)        # global per frame
    Vr = V - cam
    sp = np.linalg.norm(Vr, axis=2)                     # residual speed
    with np.errstate(invalid="ignore"):
        out["s2_speed_mean"] = float(np.nanmean(sp))
        out["s2_speed_p95"] = float(np.nanpercentile(sp, 95))
        A = np.diff(Vr, axis=0)
        am = np.linalg.norm(A, axis=2)
        out["s2_accel_p95"] = float(np.nanpercentile(am, 95))
        J = np.diff(A, axis=0)
        out["s2_jerk_p95"] = float(np.nanpercentile(np.linalg.norm(J, axis=2), 95))

        # ── specialist-targeted ────────────────────────────────────────────
        # gravity: real free fall accelerates DOWNWARD (+y in image coords).
        vy = Vr[..., 1]
        ay = np.diff(vy, axis=0)
        down = vy[:-1] > 0.5
        out["sp_gravity_accel"] = float(np.nanmean(ay[down])) if down.any() else 0.0
        # a falling thing whose vertical speed stays flat is the defect
        out["sp_gravity_flat"] = float(np.nanmean(np.abs(ay[down]))) \
            if down.any() else 0.0

        # momentum: speed after the sharpest slowdown vs speed before it.
        # >1 means something LEFT an interaction faster than it entered.
        gain = []
        for j in range(N):
            s = sp[:, j]
            s = s[~np.isnan(s)]
            if len(s) < 6:
                continue
            k = int(np.argmin(np.diff(s))) + 1
            pre, post = s[max(0, k - 3):k], s[k + 1:k + 4]
            if len(pre) and len(post) and pre.mean() > 0.3:
                gain.append(post.mean() / pre.mean())
        out["sp_momentum_gain"] = float(np.nanpercentile(gain, 90)) if gain else 0.0

        # momentum / causality: direction reversals with nothing to bounce off
        cs = np.einsum("tni,tni->tn", Vr[:-1], Vr[1:])
        nrm = np.linalg.norm(Vr[:-1], axis=2) * np.linalg.norm(Vr[1:], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosang = cs / (nrm + 1e-6)
        out["sp_reversals"] = float(np.nanmean(cosang < -0.5))

        # deformation: how much the track cloud's spread changes. A rigid object
        # translates without its points spreading apart.
        spread = np.array([np.nanstd(P[t], axis=0).mean() for t in range(T)])
        spread = spread[np.isfinite(spread)]
        if len(spread) > 2:
            out["sp_deform_spread"] = float(spread.max() / (spread.min() + 1e-6))
            out["sp_deform_drift"] = float(np.abs(np.diff(spread)).mean())

        # collision: closest approach between the two densest track clusters
        try:
            from scipy.cluster.vq import kmeans2
            pts = P[0][~np.isnan(P[0][:, 0])]
            if len(pts) >= 8:
                cent, lab = kmeans2(pts.astype(float), 2, minit="++", seed=0)
                d = []
                for t in range(T):
                    q = P[t]
                    m0 = (lab == 0) & ~np.isnan(q[:, 0])
                    m1 = (lab == 1) & ~np.isnan(q[:, 0])
                    if m0.any() and m1.any():
                        d.append(float(np.linalg.norm(
                            q[m0].mean(0) - q[m1].mean(0))))
                if len(d) > 2:
                    out["sp_min_approach"] = float(min(d) / (max(d) + 1e-6))
        except Exception:  # noqa: BLE001
            pass

        # friction: does translation match rotation? proxy = spin of the track
        # cloud about its centroid vs how far the centroid travelled
        cen = np.nanmean(P, axis=1)
        travel = float(np.nansum(np.linalg.norm(np.diff(cen, axis=0), axis=1)))
        rel = P - cen[:, None, :]
        ang = np.arctan2(rel[..., 1], rel[..., 0])
        spin = float(np.nansum(np.abs(np.diff(np.nanmean(np.unwrap(
            np.where(np.isnan(ang), 0, ang), axis=0), axis=1)))))
        out["sp_slide_ratio"] = float(travel / (spin + 1e-3))

        # event localization: when is the anomaly, as a fraction of the clip
        energy = np.nan_to_num(np.nanmean(am, axis=1))
        if len(energy) > 2:
            out["s2_event_t"] = float(np.argmax(energy) / max(len(energy) - 1, 1))
            out["s2_event_strength"] = float(
                energy.max() / (np.median(energy) + 1e-6))
    return out


def one(job):
    cid, path = job
    try:
        g = read_gray(path)
        if len(g) < 6:
            return cid, {}
        d = stage1(g)
        d.update(stage2(g))
        return cid, {k: (float(v) if np.isfinite(v) else 0.0) for k, v in d.items()}
    except Exception as e:  # noqa: BLE001
        print(f"  fail {cid[:40]}: {str(e)[:60]}", file=sys.stderr)
        return cid, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    vdir = Path(a.videos) if Path(a.videos).is_absolute() else ROOT / a.videos
    clips = json.loads((data / "manifest.json").read_text())["clips"]

    jobs, miss = [], 0
    for c in clips:
        p = vdir / f"{c['clip_id']}.mp4"
        if p.exists():
            jobs.append((c["clip_id"], str(p)))
        else:
            miss += 1
    print(f"{len(jobs)} clips ({miss} without a source video), "
          f"{MAX_FRAMES} frames each, {a.workers} workers", flush=True)

    t0, out = time.time(), {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (cid, d) in enumerate(ex.map(one, jobs)):
            out[cid] = d
            if (i + 1) % 40 == 0:
                el = time.time() - t0
                print(f"   {i+1}/{len(jobs)} ({el:.0f}s, "
                      f"eta {el/(i+1)*(len(jobs)-i-1)/60:.1f}m)", flush=True)

    keys = sorted({k for v in out.values() for k in v})
    ok = sum(1 for v in out.values() if v)
    print(f"\n{ok}/{len(out)} clips produced signals | {len(keys)} signals")
    for k in keys:
        v = np.array([out[c][k] for c in out if k in out[c]])
        print(f"   {k:22s} n={len(v):4d}  mean {v.mean():10.4f}  "
              f"sd {v.std():9.4f}")
    p = data / "stage_signals.json"
    p.write_text(json.dumps({"n": len(out), "keys": keys,
                             "max_frames": MAX_FRAMES, "signals": out}, indent=1))
    print(f"\n-> {p}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
