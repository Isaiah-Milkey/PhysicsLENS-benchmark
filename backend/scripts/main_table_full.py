"""
The paper's main table (tab:main-results) with every cell filled.

Differences from paper_tables.main_table():

  * FUSION WEIGHT FITTED ON TRAINING FOLDS. "VLM + signals" was an unweighted
    rank mean. Here the weight w in rank(VLM)*w + rank(signals)*(1-w) is chosen
    per outer fold from a grid, by AUC on the training folds only. The signal
    scores the weight is fitted on are themselves re-derived inside the training
    folds (nested tool_oof), so no test-fold label reaches either the signal
    selection or the weight. Same 5 folds, grouped by scenario, as everywhere.
  * EVERY COLUMN FOR EVERY ROW. The debiased prompt replaces only the
    plausibility question; completion, detection, family and hidden-property use
    the same questions as the standard rows. Generator identity is computed for
    every target the same way it is for plausibility (training-fold mean of the
    target per generator).
  * SPREAD. Rows averaged over backbones report the sd across the 10 backbones;
    single rows (generator identity, signals only, one backbone) report the
    bootstrap sd over videos (500 resamples).

python backend/scripts/main_table_full.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
with contextlib.redirect_stdout(io.StringIO()):
    import paper_tables as P  # noqa: E402  (builds the shared arrays)
from system_eval import tool_oof, rank01  # noqa: E402
from specialist_accuracy import auc, cats_of  # noqa: E402

ids, X, G, GEN, OBS, Y, ACT, HF, DET = P.ids, P.X, P.G, P.GEN, P.OBS, P.Y, P.ACT, P.HF, P.DET
un = ~OBS
YP = (Y >= 3).astype(int)
F = P.folds(G)
GRID = np.linspace(0, 1, 11)
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(ids), len(ids)) for _ in range(500)]


def fused(v, target, ybin, mask=None):
    """Out-of-fold fusion of VLM score v with Stage-1/2 signals, weight fitted
    on training folds. target: what the signals are selected against;
    ybin: binary label the weight is tuned for. Returns a score for every clip
    in mask (NaN elsewhere)."""
    mask = np.ones(len(ids), bool) if mask is None else mask
    out = np.full(len(ids), np.nan)
    rv = rank01(v)
    for i in range(5):
        tr, te = mask & (F != i), mask & (F == i)
        if te.sum() == 0:
            continue
        # signals for the TEST fold: selected on all training folds
        t_sel = tool_oof_fit_apply(X[tr], target[tr], X[te])
        # signals for the TRAINING folds, out-of-fold within them (nested)
        t_tr = tool_oof(X[tr], target[tr], G[tr])
        rt_tr, rv_tr = rank01(t_tr), rank01(v[tr])
        yb = ybin[tr]
        best = max(GRID, key=lambda w: auc(yb, w * rv_tr + (1 - w) * rt_tr))
        # rank the test fold's signal score against the training folds' scores
        ref = np.sort(tool_oof_fit_apply(X[tr], target[tr], X[tr]))
        rt_te = np.searchsorted(ref, t_sel) / max(len(ref) - 1, 1)
        rv_te = np.searchsorted(np.sort(v[tr]), v[te]) / max(tr.sum() - 1, 1)
        out[te] = best * rv_te + (1 - best) * rt_te
    return out


def tool_oof_fit_apply(Xtr, ytr, Xte, k=5):
    """Select + orient the top-k signals on (Xtr, ytr); score Xte with them.
    Ranks are taken relative to the training rows."""
    from system_eval import spear
    R = np.column_stack([rank01(Xtr[:, j]) for j in range(Xtr.shape[1])])
    ok = np.isfinite(ytr)
    r = np.nan_to_num(np.array([spear(R[ok, j], ytr[ok]) for j in range(R.shape[1])]))
    top = np.argsort(-np.abs(r))[:k]
    cols = []
    for j in top:
        ref = np.sort(Xtr[:, j])
        cols.append(np.sign(r[j]) * np.searchsorted(ref, Xte[:, j]) / max(len(ref) - 1, 1))
    return np.mean(cols, axis=0)


def gen_identity(target, mask=None):
    mask = np.ones(len(ids), bool) if mask is None else mask
    out = np.full(len(ids), np.nan)
    for i in range(5):
        for g in P.GENS:
            tr = mask & (F != i) & (GEN == g) & np.isfinite(target)
            te = mask & (F == i) & (GEN == g)
            if tr.sum():
                out[te] = np.nanmean(target[tr])
    return out


VIOL = np.array([clips_has for clips_has in
                 [(P.clips[c].get("has_violation") is not False
                   and len(str(P.clips[c].get("violated_rules") or "")) > 4) for c in ids]])
CATS = [c for c in P.CATS if sum(c in cats_of(P.clips[ids[i]]) for i in np.where(VIOL)[0]) >= 8]
CATY = {c: np.array([1.0 if c in cats_of(P.clips[x]) else 0.0 for x in ids]) for c in CATS}


def metrics(s_plaus, s_act, s_det, s_cat, s_hid, idx=None):
    """The seven columns. s_cat: dict family -> score (used on violating clips)."""
    idx = np.arange(len(ids)) if idx is None else idx

    def A(y, s, m):
        m = m[idx]
        yy, ss = y[idx][m], s[idx][m]
        ok = np.isfinite(ss)
        return auc(yy[ok].astype(int), ss[ok]) if ok.sum() > 5 and len(set(yy[ok])) == 2 else np.nan
    allm = np.ones(len(ids), bool)
    cols = [A(YP, s_plaus, OBS), A(YP, s_plaus, un), A(YP, s_plaus, allm),
            A(ACT, s_act, allm), A(DET, s_det, allm)]
    if s_cat is None:
        cols.append(np.nan)
    else:
        cols.append(np.nanmean([A(CATY[c], s_cat[c], VIOL) for c in CATS]))
    hy = np.where(np.isfinite(HF), (HF >= 3).astype(float), np.nan)
    cols.append(A(hy, s_hid, un & np.isfinite(HF)))
    return np.array(cols)


def with_sd(fn_scores):
    """(point, bootstrap sd) for one system."""
    pt = metrics(*fn_scores)
    bs = np.array([metrics(*fn_scores, idx=b) for b in BOOT])
    return pt, np.nanstd(bs, axis=0)


# ── signals-only and generator-identity scores (model independent) ──────────
T = {"plaus": P.T_plaus, "act": P.T_act, "det": P.T_det, "hid": P.T_hid}
T_cat = {c: np.full(len(ids), np.nan) for c in CATS}
for c in CATS:
    T_cat[c][VIOL] = tool_oof(X[VIOL], CATY[c][VIOL], G[VIOL])
hid_bin = np.where(np.isfinite(HF), (HF >= 3).astype(float), np.nan)

rows = {}
rows["Generator identity"] = with_sd((
    gen_identity(Y), gen_identity(ACT), gen_identity(DET.astype(float)),
    {c: gen_identity(CATY[c], VIOL) for c in CATS}, gen_identity(HF, un)))
rows["Signals only (no VLM)"] = with_sd((T["plaus"], T["act"], T["det"], T_cat, T["hid"]))


def vlm_scores(m, debiased, fuse):
    p = P.col(P.DEB, m, "plaus_debias") if debiased else P.col(P.PL, m, "plaus")
    a = P.col(P.PL, m, "action")
    d = P.pnone(m)
    h = P.col(P.PL, m, "hidden")
    cat = {c: np.nan_to_num(np.array([P.MQ[m].get(x, {}).get("probs", {}).get(c, np.nan)
                                      for x in ids]), nan=0.0) for c in CATS}
    if not fuse:
        return p, a, d, cat, np.where(un, h, np.nan)
    return (fused(p, Y, YP),
            fused(a, ACT, ACT.astype(int)),
            fused(d, DET.astype(float), DET),
            {c: fused(cat[c], CATY[c], CATY[c].astype(int), VIOL) for c in CATS},
            fused(np.where(un, h, 0.0), HF, np.nan_to_num(hid_bin).astype(int),
                  un & np.isfinite(HF)))


SYSTEMS = [("VLM only", False, False), ("VLM + signals", False, True),
           ("Debiased VLM", True, False), ("Debiased VLM + signals", True, True)]
live = [m for m in P.LIVE if m in P.DEB and m in P.MQ]
per_model = {}
for m in live:
    for name, deb, fu in SYSTEMS:
        per_model[(m, name)] = metrics(*vlm_scores(m, deb, fu))
    print(m, "done", file=sys.stderr)
best_m = "qwen3-vl-32b-instruct"
best_rows = {name: with_sd(vlm_scores(best_m, deb, fu)) for name, deb, fu in SYSTEMS}

mean_rows = {}
for name, _, _ in SYSTEMS:
    M = np.array([per_model[(m, name)] for m in live])
    mean_rows[name] = (np.nanmean(M, axis=0), np.nanstd(M, axis=0))

# ── print + LaTeX ───────────────────────────────────────────────────────────
COLS = ["Obs", "Unobs", "All", "Compl", "Detect", "Family", "Hidden"]
print(f"{'':28s}" + "".join(f"{c:>13s}" for c in COLS))
for blk, R in (("single", rows), ("mean over %d" % len(live), mean_rows), ("Qwen3-VL-32B", best_rows)):
    print(f"-- {blk}")
    for k, (pt, sd) in R.items():
        print(f"{k:28s}" + "".join(f"{p:8.3f}±{s:.2f}" for p, s in zip(pt, sd)))
print("\nper-backbone (all columns):")
for m in live:
    for name, _, _ in SYSTEMS:
        print(f"{m:24s} {name:24s}" + "".join(f"{x:7.3f}" for x in per_model[(m, name)]))

json.dump({"single": {k: [list(map(float, a)) for a in v] for k, v in rows.items()},
           "mean": {k: [list(map(float, a)) for a in v] for k, v in mean_rows.items()},
           "best": {k: [list(map(float, a)) for a in v] for k, v in best_rows.items()},
           "per_model": {f"{m}|{n}": list(map(float, per_model[(m, n)])) for m, n in per_model}},
          open(P.D / "main_table_full.json", "w"), indent=1)
print("->", P.D / "main_table_full.json")


# ── LaTeX: tab:main-results, every column filled ─────────────────────────────
# This is now the sole writer of paper/tables/main.tex. paper_tables.py's own
# main_table() (unweighted rank-mean fusion, plausibility-only debiased rows)
# is superseded and removed; the "Bold: best in each column among the first
# six rows" rule below matches that rule as stated in the paper caption.
def cell(x, s=None, bold=False):
    if not np.isfinite(x):
        return "---"
    out = f"{x:.2f}"
    if s is not None:
        out += rf"{{\scriptsize$\pm${s:.2f}}}"
    return r"\textbf{" + out + "}" if bold else out


six = [rows["Generator identity"], rows["Signals only (no VLM)"]] + \
      [mean_rows[name] for name, _, _ in SYSTEMS]
colmax = np.nanmax(np.vstack([pt for pt, _ in six]), axis=0)


def row_cells(pt, sd):
    return " & ".join(cell(pt[j], sd[j], bool(abs(pt[j] - colmax[j]) < 1e-9))
                       for j in range(len(pt)))


L = [r"\begin{table*}[t]", r"    \centering", r"    \small",
     r"    \setlength{\tabcolsep}{2.4pt}",
     r"    \begin{tabular}{@{}lccc|cccc@{}}", r"        \toprule",
     r"        & \multicolumn{3}{c|}{Physical plausibility} & Task & Violation"
     r" & Constr. & Hidden \\",
     r"        System & Obs. & Unobs. & All & compl. & detection & family & property \\",
     r"        \midrule"]
for name in ("Generator identity", "Signals only (no VLM)"):
    pt, sd = rows[name]
    L.append(f"        {name} & " + row_cells(pt, sd) + r" \\")
L += [r"        \midrule",
      rf"        \multicolumn{{8}}{{@{{}}l}}{{\emph{{Mean $\pm$ sd over {len(live)} VLM backbones}}}} \\"]
for name, _, _ in SYSTEMS:
    pt, sd = mean_rows[name]
    L.append(f"        {name} & " + row_cells(pt, sd) + r" \\")
BEST_DISPLAY = "Qwen3-VL-32B"  # best_m == "qwen3-vl-32b-instruct"
L += [r"        \midrule", rf"        \multicolumn{{8}}{{@{{}}l}}{{\emph{{{BEST_DISPLAY}}}}} \\"]
for name, _, _ in SYSTEMS:
    pt, sd = best_rows[name]
    # per-backbone block never bolds (bold is only among the first six rows)
    L.append(f"        {name} & "
              + " & ".join(cell(x) for x in pt) + r" \\")
L += [r"        \bottomrule", r"    \end{tabular}",
      r"    \caption{PhysicsLENS against human judgment (AUC; 439 videos). "
      r"\emph{Plausibility}: human rating 3--4 vs.\ 1--2. \emph{Task compl.}: "
      r"completed vs.\ not. \emph{Violation detection}: any recorded violation "
      r"vs.\ none. \emph{Constr.\ family}: naming which physics family was "
      r"violated. \emph{Hidden property}: whether the stated property was "
      r"followed (unobservable videos). \emph{Signals}: PhysicsLENS Stage-1/2 "
      r"measurements, combined with the VLM using a weight fitted on training "
      r"folds. \emph{Debiased}: the PhysicsLENS physics-error question, which "
      r"replaces only the plausibility question, so the other columns match "
      r"the rows above it. \emph{Generator identity} scores each video by its "
      r"generator's average on the training folds. $\pm$: sd across the ten "
      r"VLMs for the averaged rows, bootstrap sd over videos otherwise. Bold: "
      r"best in each column among the first six rows.}",
      r"    \label{tab:main-results}", r"\end{table*}"]
(P.PAPER / "tables/main.tex").write_text("\n".join(L) + "\n")
print("->", P.PAPER / "tables/main.tex")
