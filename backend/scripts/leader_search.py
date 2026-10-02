"""
Search for stronger PhysicsLENS configurations on the main-table targets, with
the same held-out protocol as the paper: 5 folds grouped by scenario; every
choice (signal selection, fusion weights, regularisation) is fitted on the
training folds only and scored on the held-out fold.

Systems (all columns: plausibility obs/unobs/all, completion, detection,
constraint family, hidden property):

  gen        generator identity (baseline, training-fold mean per generator)
  ens        rank-mean of the same question over all ten VLMs (nothing fitted)
  ens+sig    ens combined with Stage-1/2 signals, weight fitted on train folds
  stack1     L2-logistic on one VLM's answers to ALL questions + signals
             + temporal-embedding features (fitted on train folds)
  stackE     same, with the ten-VLM ensemble answers as the VLM features

python backend/scripts/leader_search.py
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
from system_eval import rank01, spear, tool_oof  # noqa: E402
from specialist_accuracy import auc, cats_of  # noqa: E402

ids, X, G, GEN, OBS, Y, ACT, HF, DET = P.ids, P.X, P.G, P.GEN, P.OBS, P.Y, P.ACT, P.HF, P.DET
N = len(ids)
UN = ~OBS
F = P.folds(G)
YP = (Y >= 3).astype(float)
HB = np.where(np.isfinite(HF), (HF >= 3).astype(float), np.nan)
TE = json.loads((P.D / "temporal_embed.json").read_text())["features"]
TKEYS = sorted(next(iter(TE.values())).keys())
XT = np.array([[TE.get(c, {}).get(k, np.nan) for k in TKEYS] for c in ids], float)
XT = np.where(np.isfinite(XT), XT, np.nanmean(XT, axis=0))
XS = np.hstack([X, XT])                           # Stage-1/2 + temporal signals

VIOL = np.array([(P.clips[c].get("has_violation") is not False
                  and len(str(P.clips[c].get("violated_rules") or "")) > 4) for c in ids])
CATS = [c for c in P.CATS if sum(c in cats_of(P.clips[ids[i]]) for i in np.where(VIOL)[0]) >= 8]
CATY = {c: np.array([1.0 if c in cats_of(P.clips[x]) else 0.0 for x in ids]) for c in CATS}
LIVE = [m for m in P.LIVE if m in P.DEB and m in P.MQ]
HSIG = P.load("vlmplaus_*__variant-hidden_sig.json", "scores")


# ── per-model answers ───────────────────────────────────────────────────────
def answers(m):
    a = {"plaus": P.col(P.PL, m, "plaus"), "deb": P.col(P.DEB, m, "plaus_debias"),
         "act": P.col(P.PL, m, "action"), "none": P.pnone(m),
         "hid": P.col(P.PL, m, "hidden")}
    if m in HSIG:
        v = np.array([HSIG[m].get(c, {}).get("hidden_sig", np.nan) for c in ids], float)
        if np.isfinite(v).sum() > 50:
            a["hsig"] = np.nan_to_num(v, nan=np.nanmean(v))
    for c in CATS:
        a["cat_" + c] = np.nan_to_num(np.array(
            [P.MQ[m].get(x, {}).get("probs", {}).get(c, np.nan) for x in ids]), nan=0.0)
    return a


A = {m: answers(m) for m in LIVE}
QKEYS = ["plaus", "deb", "act", "none", "hid"] + ["cat_" + c for c in CATS]
ENS = {q: np.mean([rank01(A[m][q]) for m in LIVE], axis=0) for q in QKEYS}
ENS["hsig"] = np.mean([rank01(A[m]["hsig"]) for m in LIVE if "hsig" in A[m]], axis=0)


# ── fitting helpers (train folds only) ──────────────────────────────────────
def zfit(Xtr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return lambda Z: (Z - mu) / sd


def logreg(Xtr, ytr, lam):
    Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    w = np.zeros(Xb.shape[1])
    R = lam * np.eye(Xb.shape[1]); R[-1, -1] = 0
    for _ in range(50):                            # Newton / IRLS
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - ytr) + R @ w
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return lambda Z: np.hstack([Z, np.ones((len(Z), 1))]) @ w


LAMS = [1, 10, 100, 1000]


def stack_oof(Fm, y, mask):
    """Out-of-fold L2-logistic score. Fm: feature matrix (N x d). lambda picked
    by inner grouped CV on the training folds."""
    out = np.full(N, np.nan)
    for i in range(5):
        tr = mask & (F != i) & np.isfinite(y)
        te = mask & (F == i)
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        # inner CV for lambda
        gi = G[tr]
        fi = P.folds(gi)
        best, bl = -1, LAMS[0]
        for lam in LAMS:
            s = np.full(tr.sum(), np.nan)
            Xtr, ytr = Fm[tr], y[tr]
            for j in range(5):
                a, b = fi != j, fi == j
                if b.sum() == 0 or len(set(ytr[a])) < 2:
                    continue
                z = zfit(Xtr[a])
                s[b] = logreg(z(Xtr[a]), ytr[a], lam)(z(Xtr[b]))
            ok = np.isfinite(s)
            sc = auc(ytr[ok].astype(int), s[ok])
            if sc > best:
                best, bl = sc, lam
        z = zfit(Fm[tr])
        out[te] = logreg(z(Fm[tr]), y[tr], bl)(z(Fm[te]))
    return out


def gen_identity(target, mask):
    out = np.full(N, np.nan)
    for i in range(5):
        for g in P.GENS:
            tr = mask & (F != i) & (GEN == g) & np.isfinite(target)
            te = mask & (F == i) & (GEN == g)
            if tr.sum():
                out[te] = np.nanmean(target[tr])
    return out


# ── the seven targets ───────────────────────────────────────────────────────
ALL = np.ones(N, bool)
HM = UN & np.isfinite(HF)
TARGETS = {  # name: (binary label, continuous target for signal selection, mask, question keys)
    "plaus": (YP, Y, ALL, ["deb", "plaus"]),
    "act": (ACT, ACT, ALL, ["act"]),
    "det": (DET.astype(float), DET.astype(float), ALL, ["none", "deb"]),
    "hid": (HB, HF, HM, ["hid", "hsig"]),
}


def cols(s_plaus, s_act, s_det, s_cat, s_hid, idx=None):
    idx = np.arange(N) if idx is None else idx

    def Au(y, s, m):
        m = m[idx]
        yy, ss = y[idx][m], s[idx][m]
        ok = np.isfinite(ss) & np.isfinite(yy)
        return auc(yy[ok].astype(int), ss[ok]) if ok.sum() > 5 and len(set(yy[ok])) == 2 else np.nan
    r = [Au(YP, s_plaus, OBS), Au(YP, s_plaus, UN), Au(YP, s_plaus, ALL),
         Au(ACT, s_act, ALL), Au(DET.astype(float), s_det, ALL),
         np.nanmean([Au(CATY[c], s_cat[c], VIOL) for c in CATS]), Au(HB, s_hid, HM)]
    return np.array(r)


rng = np.random.default_rng(0)
BOOT = [rng.integers(0, N, N) for _ in range(500)]


def with_sd(sc):
    pt = cols(*sc)
    return pt, np.nanstd([cols(*sc, idx=b) for b in BOOT], axis=0)


def main():
    RES = {}
    RES["gen"] = with_sd((gen_identity(Y, ALL), gen_identity(ACT, ALL), gen_identity(DET.astype(float), ALL),
                          {c: gen_identity(CATY[c], VIOL) for c in CATS}, gen_identity(HF, HM)))
    print("gen done", file=sys.stderr)

    # ens: nothing fitted
    RES["ens"] = with_sd((ENS["deb"], ENS["act"], ENS["none"], {c: ENS["cat_" + c] for c in CATS}, ENS["hid"]))
    print("ens done", file=sys.stderr)


    def feats(src, keys):
        return np.column_stack([rank01(src[k]) for k in keys])


    def stack_system(src, with_sig=True):
        """src: dict of answer arrays (one model's, or the ensemble's)."""
        allq = [k for k in ["deb", "plaus", "act", "none", "hid", "hsig"] if k in src]
        base = feats(src, allq)
        Fm = np.hstack([base, XS]) if with_sig else base
        s_pl = stack_oof(Fm, YP, ALL)
        s_act = stack_oof(Fm, ACT, ALL)
        s_det = stack_oof(np.hstack([Fm, feats(src, ["cat_" + c for c in CATS])]), DET.astype(float), ALL)
        Fc = np.hstack([Fm, feats(src, ["cat_" + c for c in CATS])])
        s_cat = {c: stack_oof(Fc, CATY[c], VIOL) for c in CATS}
        s_hid = stack_oof(Fm, HB, HM)
        return s_pl, s_act, s_det, s_cat, s_hid


    RES["stackE_nosig"] = with_sd(stack_system(ENS, with_sig=False))
    print("stackE_nosig done", file=sys.stderr)
    RES["stackE"] = with_sd(stack_system(ENS))
    print("stackE done", file=sys.stderr)
    RES["stack1_qwen32"] = with_sd(stack_system(A["qwen3-vl-32b-instruct"]))
    print("stack1 done", file=sys.stderr)
    per = {}
    for m in LIVE:
        per[m] = cols(*stack_system(A[m]))
        print(m, np.round(per[m], 3), file=sys.stderr)
    RES["stack1_mean"] = (np.nanmean([per[m] for m in LIVE], 0), np.nanstd([per[m] for m in LIVE], 0))

    names = ["Obs", "Unobs", "All", "Compl", "Detect", "Family", "Hidden"]
    print(f"{'':16s}" + "".join(f"{n:>13s}" for n in names))
    for k, (pt, sd) in RES.items():
        print(f"{k:16s}" + "".join(f"{p:8.3f}±{s:.2f}" for p, s in zip(pt, sd)))
    json.dump({k: [list(map(float, v[0])), list(map(float, v[1]))] for k, v in RES.items()}
              | {"per_model_stack1": {m: list(map(float, v)) for m, v in per.items()}},
              open(P.D / "leader_search.json", "w"), indent=1)
    print("->", P.D / "leader_search.json")


if __name__ == "__main__":
    main()
