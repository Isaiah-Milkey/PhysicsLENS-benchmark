"""
Domain rules — fit the physical LAW per object, measure the violation.

DIFFERENT FROM specialist_tools*.py, which computed statistical aggregates
(mean vertical acceleration, 95th-percentile impulse). Those describe motion;
they do not check a law. Here each rule states what physics REQUIRES, fits that
model, and reports how badly the clip breaks it.

  R1 ballistic     free flight is parabolic. Fit y(t)=½at²+v₀t+y₀ over the
                   longest airborne run. Violation = a <= 0 (not accelerating
                   downward) or a large residual against ANY constant a.
  R2 hover         longest run with |v_y| ~ 0 while moving horizontally and
                   nothing beneath — gravity forbids sustained hovering.
  R3 restitution   at each impact (local speed minimum) the speed after must be
                   <= speed before. Violation = max(v_after/v_before) - 1.
  R4 uncaused_dv   every speed change needs something nearby to cause it.
                   Violation = largest |Δv| whose nearest other object is far.
  R5 contact_gap   at the single largest |Δv|, the minimum separation to another
                   object should be ~0. Violation = that separation.
  R6 rigidity      a rigid body preserves internal pairwise distances.
                   Violation = dispersion of within-object distances over time.
  R7 fric_decay    an object sliding on a surface must lose speed.
                   Violation = speed ratio end/start for surface-contacting objects.

THE FIX THAT MATTERS — object identity. specialist_tools_v2 fitted one
"foreground centroid", which is a motion-weighted average of everything moving.
Fitting a parabola to a blend of the ball AND the thrower measures nothing; that
is the likely reason its ballistic residual failed to discriminate gravity.
Here tracks are CLUSTERED into coherent moving groups, every rule is fitted per
cluster, and the clip score is the WORST violation over clusters — because one
object breaking gravity is a gravity violation regardless of what else is in
frame.

Clustering is k-means on motion+position rather than SAM3 masks: it costs
nothing, runs in seconds, and answers whether per-object law-fitting helps at
all before paying for segmentation.

Usage:
  python backend/scripts/physics_rules.py --data data/videophy1200 --workers 10
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_tools import _read, build_tracks, compensate     # noqa: E402


def cluster_tracks(Q, k=4, min_pts=4):
    """Group tracks into coherent moving objects. k-means over mean velocity and
    mean position — velocity separates the ball from the thrower, position keeps
    spatially distinct objects apart when their speeds happen to match."""
    T, N, _ = Q.shape
    V = np.diff(Q, axis=0)
    mv = np.nanmean(V, axis=0)                       # (N,2) mean velocity
    mp = np.nanmean(Q, axis=0)                       # (N,2) mean position
    ok = np.isfinite(mv[:, 0]) & np.isfinite(mp[:, 0])
    idx = np.where(ok)[0]
    if len(idx) < min_pts * 2:
        return [idx] if len(idx) >= min_pts else []
    F = np.hstack([mv[idx] / (np.nanstd(mv[idx]) + 1e-6),
                   mp[idx] / (np.nanstd(mp[idx]) + 1e-6) * 0.5])
    rng = np.random.RandomState(0)
    C = F[rng.choice(len(F), min(k, len(F)), replace=False)]
    lab = np.zeros(len(F), dtype=int)
    for _ in range(15):
        d = ((F[:, None, :] - C[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(len(C)):
            if (lab == j).any():
                C[j] = F[lab == j].mean(0)
    return [idx[lab == j] for j in range(len(C)) if (lab == j).sum() >= min_pts]


def centroid(Q, members):
    T = Q.shape[0]
    C = np.full((T, 2), np.nan)
    for t in range(T):
        p = Q[t, members]
        p = p[np.isfinite(p[:, 0])]
        if len(p):
            C[t] = p.mean(0)
    return C


def rules_for(C, Others, h):
    """Violation magnitudes for one object trajectory."""
    good = np.isfinite(C[:, 0])
    if good.sum() < 6:
        return {}
    t = np.where(good)[0]
    P = C[good]
    v = np.diff(P, axis=0)
    spd = np.linalg.norm(v, axis=1)
    out = {}

    # R1 ballistic: fit y = ½a t² + v0 t + y0, report a and normalised residual
    if len(P) >= 6:
        A = np.stack([0.5 * t[:len(P)] ** 2, t[:len(P)], np.ones(len(P))], 1).astype(float)
        sol, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
        resid = float(np.mean((A @ sol - P[:, 1]) ** 2)) / (np.var(P[:, 1]) + 1e-6)
        out["r1_accel_sign"] = float(-sol[0])        # >0 means NOT falling: wrong
        out["r1_fit_resid"] = float(min(resid, 5.0))
    # R2 hover: horizontal motion with no vertical change
    if len(v) >= 4:
        moving_x = np.abs(v[:, 0]) > 0.3
        flat_y = np.abs(v[:, 1]) < 0.06
        run = best = 0
        for m, f in zip(moving_x, flat_y):
            run = run + 1 if (m and f) else 0
            best = max(best, run)
        out["r2_hover_run"] = float(best / max(len(v), 1))
    # R3 restitution: speed after an impact vs before
    if len(spd) >= 5:
        best = 0.0
        for i in range(1, len(spd) - 1):
            if spd[i] < spd[i - 1] and spd[i] <= spd[i + 1]:      # local minimum
                before = float(np.max(spd[max(0, i - 2):i + 1]))
                after = float(np.max(spd[i + 1:i + 4]))
                if before > 0.3:
                    best = max(best, after / before - 1.0)
        out["r3_restitution"] = float(min(max(best, 0.0), 5.0))
    # R4/R5 causation: big speed changes need something close by
    if len(spd) >= 3 and Others:
        ds = np.abs(np.diff(spd))
        j = int(np.argmax(ds))
        tt = t[min(j + 1, len(t) - 1)]
        here = C[tt]
        gaps = []
        for O in Others:
            if np.isfinite(O[tt][0]) and np.isfinite(here[0]):
                gaps.append(float(np.linalg.norm(O[tt] - here)))
        if gaps:
            out["r5_contact_gap"] = float(min(gaps) / max(h, 1))
            out["r4_uncaused_dv"] = float(ds[j] * min(gaps) / max(h, 1))
    # R7 friction: surface-contacting objects must slow
    if len(spd) >= 6 and np.nanmean(P[:, 1]) > 0.6 * h:
        a0 = float(np.mean(spd[:max(1, len(spd) // 3)]))
        a1 = float(np.mean(spd[-max(1, len(spd) // 3):]))
        out["r7_fric_decay"] = float(min(a1 / (a0 + 1e-6), 5.0))
    return out


def features(path):
    g = _read(Path(path))
    if len(g) < 8:
        return None
    P, alive = build_tracks(g)
    if P is None:
        return None
    Q = compensate(P)
    groups = cluster_tracks(Q)
    if not groups:
        return None
    cents = [centroid(Q, m) for m in groups]
    h = g[0].shape[0]

    per = []
    for i, C in enumerate(cents):
        others = [c for j, c in enumerate(cents) if j != i]
        r = rules_for(C, others, h)
        if r:
            per.append(r)
    if not per:
        return None
    # Worst violation over objects: one object breaking a law breaks the law.
    keys = sorted({k for r in per for k in r})
    out = {k: float(np.nanmax([r[k] for r in per if k in r])) for k in keys}
    # R6 rigidity: dispersion of within-object pairwise distances
    cvs = []
    rng = np.random.RandomState(0)
    for m in groups:
        if len(m) < 4:
            continue
        for _ in range(24):
            i, j = rng.choice(m, 2, replace=False)
            d = np.linalg.norm(Q[:, i] - Q[:, j], axis=1)
            d = d[np.isfinite(d)]
            if len(d) > 5 and d.mean() > 2:
                cvs.append(float(d.std() / (d.mean() + 1e-6)))
    out["r6_rigidity"] = float(np.median(cvs)) if cvs else 0.0
    out["n_objects"] = float(len(groups))
    return out


def _job(t):
    cid, p = t
    try:
        return cid, features(p)
    except Exception as e:  # noqa: BLE001
        return cid, {"_error": str(e)[:70]}


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
    outp = data / "physics_rules.json"
    outp.write_text(json.dumps({"n": len(out), "features": out}, indent=1))
    print(f"\n  {len(out)}/{len(jobs)} clips, {len(names)} rules "
          f"({time.time()-t0:.0f}s)")

    from vlm_rapidata_eval import spearman
    y = [6 - c["pc"] for c in clips if c["clip_id"] in out]
    print("\n  univariate rho vs human pc:")
    for nm in names:
        v = [out[c["clip_id"]][nm] for c in clips if c["clip_id"] in out
             and nm in out[c["clip_id"]]]
        yy = [6 - c["pc"] for c in clips if c["clip_id"] in out
              and nm in out[c["clip_id"]]]
        print(f"    {nm:18s} n={len(v):4d}  rho={spearman(v, yy):+.3f}")
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()
