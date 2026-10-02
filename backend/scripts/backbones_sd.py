"""
Table 7 (tab:backbones-obs-unobs) with a bootstrap sd for every cell.
Same scores as the table: standard and physics-error (debiased) plausibility,
physics-error + Stage-1/2 signals (weight fitted on training folds, as in
main_table_full.py), and the VLM-only hidden-property answer. sd = std of the
AUC over 500 bootstrap resamples of the videos.

python backend/scripts/backbones_sd.py
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

N = len(P.ids)
F = P.folds(P.G)
fz = make_fused(P.X, P.G, F, N)
YP = (P.Y >= 3).astype(int)
OBS, UN = P.OBS, ~P.OBS
HM = UN & np.isfinite(P.HF)
HB = np.where(np.isfinite(P.HF), (P.HF >= 3).astype(int), 0)
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, N, N) for _ in range(500)]


def A(y, s, m, idx=None):
    idx = np.arange(N) if idx is None else idx
    mm = m[idx]
    return auc(y[idx][mm], s[idx][mm])


def cell(y, s, m):
    return A(y, s, m), float(np.std([A(y, s, m, b) for b in BOOT]))


out = {}
sig_pl, sig_hid = P.T_plaus, P.T_hid
out["Signals only"] = {"sig_obs": cell(YP, sig_pl, OBS), "sig_un": cell(YP, sig_pl, UN),
                       "hid": cell(HB, np.nan_to_num(sig_hid), HM)}
for m in P.LIVE:
    if m not in P.DEB:
        continue
    std = P.col(P.PL, m, "plaus")
    deb = P.col(P.DEB, m, "plaus_debias")
    fu = fz(deb, P.Y, YP)
    hid = P.col(P.PL, m, "hidden")
    out[m] = {"std_obs": cell(YP, std, OBS), "std_un": cell(YP, std, UN),
              "deb_obs": cell(YP, deb, OBS), "deb_un": cell(YP, deb, UN),
              "sig_obs": cell(YP, fu, OBS), "sig_un": cell(YP, fu, UN),
              "hid": cell(HB, hid, HM)}
    print(m, {k: f"{v[0]:.2f}±{v[1]:.2f}" for k, v in out[m].items()})
print("signals", {k: f"{v[0]:.2f}±{v[1]:.2f}" for k, v in out["Signals only"].items()})
json.dump(out, open(P.D / "backbones_sd.json", "w"), indent=1)
