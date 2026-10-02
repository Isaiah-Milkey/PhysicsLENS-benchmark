"""
Second round of the leader search (same held-out protocol as leader_search.py).

  ens+sig      ten-VLM ensemble answer + Stage-1/2 signals, weight fitted on
               training folds (fusion_utils.fused)
  ens+sig+T    same, signals also include the temporal-embedding features
  ensK         ensemble of the K backbones that score best on the TRAINING
               folds (K fixed at 5), per target and per fold
  ensK+sig     ensK + signals, fitted weight

python backend/scripts/leader_search2.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import leader_search as L  # noqa: E402
from fusion_utils import make_fused  # noqa: E402
from system_eval import rank01  # noqa: E402
from specialist_accuracy import auc  # noqa: E402

N, F, G = L.N, L.F, L.G
fz28 = make_fused(L.X, G, F, N)
fzT = make_fused(L.XS, G, F, N)
C = L.CATS
QPL, YP, ALL, HM, VIOL = "deb", L.YP, L.ALL, L.HM, L.VIOL
DETf = L.DET.astype(float)
HBi = np.nan_to_num(L.HB).astype(int)


def ens_sig(fz):
    E = L.ENS
    return (fz(E["deb"], L.Y, YP.astype(int)),
            fz(E["act"], L.ACT, L.ACT.astype(int)),
            fz(E["none"], DETf, L.DET),
            {c: fz(E["cat_" + c], L.CATY[c], L.CATY[c].astype(int), VIOL) for c in C},
            fz(np.where(HM, E["hid"], 0.0), L.HF, HBi, HM))


def topk(q, y, mask, K=5):
    """Per fold: rank-mean of the K backbones with the best training-fold AUC."""
    out = np.full(N, np.nan)
    for i in range(5):
        tr, te = mask & (F != i) & np.isfinite(y), mask & (F == i)
        sc = {m: auc(y[tr].astype(int), L.A[m][q][tr]) for m in L.LIVE if q in L.A[m]}
        best = sorted(sc, key=lambda m: -sc[m])[:K]
        out[te] = np.mean([rank01(L.A[m][q])[te] for m in best], axis=0)
    return out


def ensK():
    return (topk("deb", YP, ALL), topk("act", L.ACT, ALL), topk("none", DETf, ALL),
            {c: topk("cat_" + c, L.CATY[c], VIOL) for c in C}, topk("hid", L.HB, HM))


def ensK_sig(fz):
    p, a, d, cat, h = ensK()
    fill = lambda v: np.nan_to_num(v, nan=np.nanmean(v))  # noqa: E731
    return (fz(fill(p), L.Y, YP.astype(int)), fz(fill(a), L.ACT, L.ACT.astype(int)),
            fz(fill(d), DETf, L.DET),
            {c: fz(fill(cat[c]), L.CATY[c], L.CATY[c].astype(int), VIOL) for c in C},
            fz(np.where(HM, fill(h), 0.0), L.HF, HBi, HM))


def main():
    RES = {}
    for name, fn in [("ens+sig", lambda: ens_sig(fz28)), ("ens+sig+T", lambda: ens_sig(fzT)),
                     ("ensK", ensK), ("ensK+sig", lambda: ensK_sig(fz28))]:
        RES[name] = L.with_sd(fn())
        print(name, "done", file=sys.stderr)

    names = ["Obs", "Unobs", "All", "Compl", "Detect", "Family", "Hidden"]
    print(f"{'':16s}" + "".join(f"{n:>13s}" for n in names))
    for k, (pt, sd) in RES.items():
        print(f"{k:16s}" + "".join(f"{p:8.3f}±{s:.2f}" for p, s in zip(pt, sd)))
    json.dump({k: [list(map(float, v[0])), list(map(float, v[1]))] for k, v in RES.items()},
              open(L.P.D / "leader_search2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
