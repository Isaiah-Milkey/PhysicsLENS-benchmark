"""
Final PhysicsLENS system for the main table, fixed in advance of this run:
ten-VLM ensemble (physics-error question for plausibility; the same completion,
specialist and hidden-property questions as elsewhere) combined with the
Stage-1/2 + temporal-embedding signals, weight fitted on training folds
("ens+sig+T" in leader_search2.py). No backbone is selected.

Also reports WITHIN-GENERATOR AUC (mean over the four generators of the AUC
computed inside each one), where generator identity is 0.5 by construction.

python backend/scripts/leader_final.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import leader_search as L  # noqa: E402
import leader_search2 as L2  # noqa: E402
from fusion_utils import make_fused  # noqa: E402
from specialist_accuracy import auc  # noqa: E402

fzT = make_fused(L.XS, L.G, L.F, L.N)
S = L2.ens_sig(fzT)
pt, sd = L.with_sd(S)
gen_pt, gen_sd = L.with_sd((L.gen_identity(L.Y, L.ALL), L.gen_identity(L.ACT, L.ALL),
                            L.gen_identity(L.DET.astype(float), L.ALL),
                            {c: L.gen_identity(L.CATY[c], L.VIOL) for c in L.CATS},
                            L.gen_identity(L.HF, L.HM)))


def within(s, y, mask):
    v = []
    for g in L.P.GENS:
        m = mask & (L.GEN == g) & np.isfinite(s) & np.isfinite(y)
        if m.sum() > 10 and len(set(y[m])) == 2:
            v.append(auc(y[m].astype(int), s[m]))
    return float(np.mean(v))


plaus, act, det, cat, hid = S
W = {"plaus_obs": within(plaus, L.YP, L.OBS), "plaus_unobs": within(plaus, L.YP, L.UN),
     "plaus_all": within(plaus, L.YP, L.ALL), "compl": within(act, L.ACT, L.ALL),
     "detect": within(det, L.DET.astype(float), L.ALL), "hidden": within(hid, L.HB, L.HM)}
print("PhysicsLENS full:", " ".join(f"{p:.3f}±{s:.2f}" for p, s in zip(pt, sd)))
print("generator id    :", " ".join(f"{p:.3f}±{s:.2f}" for p, s in zip(gen_pt, gen_sd)))
print("within-generator AUC (generator identity = 0.5):", {k: round(v, 3) for k, v in W.items()})
json.dump({"full": [list(map(float, pt)), list(map(float, sd))], "within_generator": W},
          open(L.P.D / "leader_final.json", "w"), indent=1)
