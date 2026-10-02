"""
Table 7 (tab:backbones-obs-unobs) as mean +/- sd over three runs that differ in
which frames the VLM sees:

  run 0  data/consol       8 frames evenly spaced, first to last (the paper's run)
  run 1  data/consol_fs1   same, shifted by 1/3 of the frame spacing
  run 2  data/consol_fs2   same, shifted by 2/3 of the frame spacing

(make_frame_seeds.py builds runs 1-2; vlm_plausibility.py --only
plaus,plaus_debias,hidden scores them.) Per run and backbone: standard and
physics-error plausibility AUC on observable / unobservable videos, the
physics-error score fused with Stage-1/2 signals (weight fitted on training
folds, fusion_utils), and the VLM-only hidden-property AUC. Signals-only has no
VLM, so it is identical across runs.

python backend/scripts/backbones_3run.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
with contextlib.redirect_stdout(io.StringIO()):
    import paper_tables as P  # noqa: E402
from fusion_utils import make_fused  # noqa: E402
from specialist_accuracy import auc  # noqa: E402

ROOT = P.ROOT
ids = P.ids
N = len(ids)
fz = make_fused(P.X, P.G, P.folds(P.G), N)
YP = (P.Y >= 3).astype(int)
OBS, UN = P.OBS, ~P.OBS
HM = UN & np.isfinite(P.HF)
HB = np.where(np.isfinite(P.HF), (P.HF >= 3).astype(int), 0)
FR = {"llama4-scout-17b": 4}


def vec(scores, key):
    v = np.array([scores.get(c, {}).get(key, np.nan) for c in ids], float)
    return np.nan_to_num(v, nan=np.nanmean(v)), int(np.isfinite(v).sum())


def run_scores(m, r):
    f = FR.get(m, 8)
    if r == 0:
        std = json.loads((P.D / f"vlmplaus_{m}__f{f}.json").read_text())["scores"]
        deb = json.loads((P.D / f"vlmplaus_{m}__f8__variant-plaus_debias.json").read_text())["scores"] \
            if (P.D / f"vlmplaus_{m}__f8__variant-plaus_debias.json").exists() else \
            json.loads((P.D / f"vlmplaus_{m}__f{f}__variant-plaus_debias.json").read_text())["scores"]
        return vec(std, "plaus"), vec(deb, "plaus_debias"), vec(std, "hidden")
    p = ROOT / f"data/consol_fs{r}/vlmplaus_{m}__f{f}__variant-plaus+plaus_debias+hidden.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())["scores"]
    return vec(s, "plaus"), vec(s, "plaus_debias"), vec(s, "hidden")


def cols(std, deb, hid):
    fu = fz(deb, P.Y, YP)
    return [auc(YP[OBS], std[OBS]), auc(YP[UN], std[UN]), auc(YP[OBS], deb[OBS]), auc(YP[UN], deb[UN]),
            auc(YP[OBS], fu[OBS]), auc(YP[UN], fu[UN]), auc(HB[HM], hid[HM])]


out, missing = {}, []
for m in P.LIVE:
    if m not in P.DEB:
        continue
    runs = []
    for r in range(3):
        sc = run_scores(m, r)
        if sc is None:
            missing.append(f"{m} run{r}")
            continue
        (std, n1), (deb, n2), (hid, n3) = sc
        runs.append(cols(std, deb, hid))
        print(f"{m:24s} run{r} n={n1}/{n2}/{n3} " + " ".join(f"{x:.3f}" for x in runs[-1]))
    R = np.array(runs)
    out[m] = {"n_runs": len(runs), "mean": R.mean(0).tolist(), "sd": R.std(0, ddof=1).tolist() if len(runs) > 1 else None,
              "runs": R.tolist()}
# Signals only has no VLM and the signals are deterministic, so its three runs
# are three cross-validation fold splits (signal selection happens on training folds).
from system_eval import tool_oof  # noqa: E402
S = []
for seed in (0, 1, 2):
    tp = tool_oof(P.X, P.Y, P.G, seed=seed)
    th = np.full(N, np.nan)
    th[UN] = tool_oof(P.X[UN], P.HF[UN], P.G[UN], seed=seed)
    S.append([auc(YP[OBS], tp[OBS]), auc(YP[UN], tp[UN]), auc(HB[HM], th[HM])])
S = np.array(S)
out["Signals only"] = {"n_runs": 3, "mean": S.mean(0).tolist(), "sd": S.std(0, ddof=1).tolist(),
                       "runs": S.tolist(), "note": "three fold splits (seeds 0-2)"}
json.dump(out, open(P.D / "backbones_3run.json", "w"), indent=1)
print("\nmissing:", missing or "none")
for m, v in out.items():
    sd = v["sd"] or [0] * len(v["mean"])
    print(f"{m:24s} ({v['n_runs']} runs) " + " ".join(f"{a:.2f}±{b:.2f}" for a, b in zip(v["mean"], sd)))
