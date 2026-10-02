"""
Numbers for the paper's automated-diagnosis results, written as LaTeX.

Produces (paper/tables/):
  agreement.tex     tab:pipeline-human-agreement — per generator, weighted
                    Cohen's kappa for plausibility and hidden property, kappa
                    for completion, plus AUC
  stagewise.tex     tab:stage-ablation — cumulative stage prefixes x backbone
  models.tex        results across VLM backbones, observable vs unobservable
  categories.tex    physical-category accuracy per specialist family
and paper/fig/models_obs_unobs.pdf.

KAPPA NEEDS RATINGS, NOT SCORES. Every system emits a continuous score, so it is
mapped to 1-4 by quantile matching fitted on TRAINING folds only (5-fold,
grouped by task so an obs/unobs twin and the other generators' versions of a
task never straddle a fold): a held-out clip gets the rating found at its
score's percentile in the training fold's rating distribution. Binary
completion uses the training-fold base rate as the threshold. Nothing about the
test fold's labels is used.

CONFIGURATIONS are fixed by earlier, separately validated results, not chosen
here: plausibility = debiased prompt (validated on held-out VideoPhy-2);
completion and detection = VLM + Stage-1/2 tool (significant for 8/10 and 10/10
judges); category attribution = VLM specialist alone.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from system_eval import tool_oof, rank01, spear  # noqa: E402
from specialist_accuracy import auc, cats_of  # noqa: E402

D = ROOT / "data" / "consol"
PAPER = ROOT / "paper"
(PAPER / "tables").mkdir(parents=True, exist_ok=True)

GEN_NAME = {"wan": "Wan 2.2", "cosmos": "Cosmos-nano 1", "hunyuan": "HunyuanVideo 1.5",
            "magi": "MAGI 4.5B distill"}
GENS = ["wan", "cosmos", "hunyuan", "magi"]
MODEL_NAME = {"qwen3-vl-32b-instruct": "Qwen3-VL-32B", "qwen3-vl-8b": "Qwen3-VL-8B",
              "qwen2.5-vl-32b": "Qwen2.5-VL-32B", "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
              "internvl3-14b": "InternVL3-14B", "internvl3-8b": "InternVL3-8B",
              "gemma3-12b": "Gemma-3-12B", "llama4-scout-17b": "Llama-4-Scout-17B",
              "llava-ov-7b": "LLaVA-OV-7B", "mistral-small-24b": "Mistral-Small-3.1-24B"}
SHORT = {"qwen3-vl-32b-instruct": "Q3-32B", "qwen3-vl-8b": "Q3-8B", "qwen2.5-vl-32b": "Q2.5-32B",
         "qwen2.5-vl-7b": "Q2.5-7B", "internvl3-14b": "IV-14B", "internvl3-8b": "IV-8B",
         "gemma3-12b": "G3-12B", "llama4-scout-17b": "L4S", "llava-ov-7b": "LOV-7B",
         "mistral-small-24b": "MS-24B"}
CATS = ["collision", "deformation", "causality", "momentum", "gravity", "fluid", "friction"]

clips = {c["clip_id"]: c for c in json.loads((D / "manifest.json").read_text())["clips"]}
S = json.loads((D / "stage_signals.json").read_text())
SIG, KEYS = S["signals"], S["keys"]
ids = sorted(c for c in clips if SIG.get(c))
X = np.array([[SIG[c].get(k, 0.0) for k in KEYS] for c in ids], float)
S1 = [i for i, k in enumerate(KEYS) if k.startswith("s1_")]
G = np.array([clips[c]["testset_id"] for c in ids])
GEN = np.array([clips[c]["generator"] for c in ids])
OBS = np.array([clips[c]["observability"] == "observable" for c in ids])
Y = np.array([clips[c]["pc"] for c in ids], float)
ACT = np.array([1.0 if clips[c]["action_completed"] == "yes" else 0.0 for c in ids])
HF = np.array([float(clips[c]["hidden_followed"]) if not OBS[i] else np.nan
               for i, c in enumerate(ids)])
DET = np.array([0 if clips[c]["has_violation"] is False else 1 for c in ids])


def load(pattern, key):
    out = {}
    for f in sorted(D.glob(pattern)):
        d = json.loads(f.read_text())
        if "shuffled" in f.name or "__f16" in f.name or "__f2." in f.name:
            continue
        out[d["model"]] = d[key]
    return out


PL = {m: v for m, v in load("vlmplaus_*__f[48].json", "scores").items()}
DEB = load("vlmplaus_*__variant-plaus_debias.json", "scores")
MQ = load("mcq_*__f[48].json", "mcq")
LIVE = [m for m in MODEL_NAME if m in PL]


def col(src, m, key, idl=ids):
    v = np.array([src[m].get(c, {}).get(key, np.nan) for c in idl], float)
    return np.nan_to_num(v, nan=np.nanmean(v))


def pnone(m):
    v = np.array([MQ[m].get(c, {}).get("probs", {}).get("none", np.nan) for c in ids], float)
    return 1 - np.nan_to_num(v, nan=np.nanmean(v))


def folds(groups, k=5, seed=0):
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    f = {g: i % k for i, g in enumerate(ug)}
    return np.array([f[g] for g in groups])


def to_rating(score, y, groups, mask=None, k=5):
    """CV quantile matching -> integer ratings (NaN outside mask)."""
    mask = np.ones(len(score), bool) if mask is None else mask
    f = folds(groups)
    pred = np.full(len(score), np.nan)
    for i in range(k):
        tr, te = mask & (f != i), mask & (f == i)
        if tr.sum() < 5 or te.sum() == 0:
            continue
        q = np.sort(y[tr])
        pct = np.array([(score[tr] < s).mean() for s in score[te]])
        pred[te] = q[np.clip((pct * len(q)).astype(int), 0, len(q) - 1)]
    return pred


def to_binary(score, y, groups, k=5):
    f = folds(groups)
    pred = np.zeros(len(score))
    for i in range(k):
        tr, te = f != i, f == i
        thr = np.quantile(score[tr], 1 - y[tr].mean())
        pred[te] = (score[te] > thr).astype(float)
    return pred


def kappa(a, b, weights=None, cats=(1, 2, 3, 4)):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m].astype(int), b[m].astype(int)
    idx = {c: i for i, c in enumerate(cats)}
    n = len(cats)
    O = np.zeros((n, n))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    W = (np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / (n - 1)
         if weights == "linear" else 1 - np.eye(n))
    return float(1 - (W * O).sum() / (W * E).sum())


def boot_kappa(a, b, weights=None, cats=(1, 2, 3, 4), B=1000):
    rng = np.random.default_rng(0)
    m = np.where(np.isfinite(a) & np.isfinite(b))[0]
    v = [kappa(a[i], b[i], weights, cats) for i in (rng.choice(m, len(m)) for _ in range(B))]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def fmt(x, d=2):
    return "---" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# ── system scores (continuous) ───────────────────────────────────────────────
T_plaus = tool_oof(X, Y, G)
T_act = tool_oof(X, ACT, G)
T_det = tool_oof(X, DET.astype(float), G)
un = ~OBS
T_hid = np.full(len(ids), np.nan)
T_hid[un] = tool_oof(X[un], HF[un], G[un])

PLAUS_M = "qwen3-vl-32b-instruct"   # debiased, validated on VideoPhy-2
ACT_M = "qwen3-vl-8b"
HID_M = "llama4-scout-17b"

s_plaus = col(DEB, PLAUS_M, "plaus_debias")
s_act = rank01(col(PL, ACT_M, "action")) + rank01(T_act)
s_hid = np.full(len(ids), np.nan)
s_hid[un] = rank01(col(PL, HID_M, "hidden")[un]) + rank01(T_hid[un])

r_plaus = to_rating(s_plaus, Y, G)
r_hid = to_rating(np.nan_to_num(s_hid), HF, G, mask=un)
b_act = to_binary(s_act, ACT, G)

# ── Table: agreement per generator ───────────────────────────────────────────
rows = []
for g in GENS + ["all"]:
    m = np.ones(len(ids), bool) if g == "all" else GEN == g
    mu = m & un
    kp = kappa(Y[m], r_plaus[m], "linear")
    kh = kappa(HF[mu], r_hid[mu], "linear")
    kc = kappa(ACT[m], b_act[m], None, cats=(0, 1))
    ap = auc((Y[m] >= 3).astype(int), s_plaus[m])
    ah = auc((HF[mu] >= 3).astype(int), s_hid[mu]) if 0 < (HF[mu] >= 3).sum() < mu.sum() else np.nan
    ac = auc(ACT[m].astype(int), s_act[m]) if 0 < ACT[m].sum() < m.sum() else np.nan
    rows.append((g, m.sum(), mu.sum(), kp, ap, kh, ah, kc, ac))
ci_p = boot_kappa(Y, r_plaus, "linear")
ci_h = boot_kappa(HF, r_hid, "linear")
ci_c = boot_kappa(ACT, b_act, None, (0, 1))

L = [r"\begin{table}[t]", r"    \centering", r"    \small",
     r"    \setlength{\tabcolsep}{4pt}",
     r"    \begin{tabular}{@{}lrcccccc@{}}", r"        \toprule",
     r"        & & \multicolumn{2}{c}{Plausibility} & \multicolumn{2}{c}{Hidden property}"
     r" & \multicolumn{2}{c}{Completion} \\",
     r"        \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"        Generator & $n$ & $\kappa_w$ & AUC & $\kappa_w$ & AUC & $\kappa$ & AUC \\",
     r"        \midrule"]
for g, n, nu, kp, ap, kh, ah, kc, ac in rows:
    name = "Overall" if g == "all" else GEN_NAME[g]
    if g == "all":
        L.append(r"        \midrule")
    L.append(f"        {name} & {n} & {fmt(kp)} & {fmt(ap)} & {fmt(kh)} & {fmt(ah)} "
             f"& {fmt(kc)} & {fmt(ac)} \\\\")
L += [r"        \bottomrule", r"    \end{tabular}",
      r"    \caption{Agreement between automated diagnosis and human annotation. "
      r"Ratings are obtained from continuous scores by quantile matching fitted on "
      r"training folds only (5-fold cross-validation grouped by task, so the "
      r"observable and unobservable versions of a scenario never straddle a fold). "
      r"$\kappa_w$ is linearly weighted Cohen's $\kappa$ on the 1--4 scale; "
      r"completion uses unweighted $\kappa$. AUC separates ratings 3--4 from 1--2 "
      r"(completion: yes vs.\ no). Plausibility: debiased Qwen3-VL-32B prompt; "
      r"hidden property (unobservable clips only, $n{=}119$): Llama-4-Scout with the "
      r"stated property plus Stage-1/2 signals; completion: Qwen3-VL-8B plus "
      rf"Stage-1/2 signals. Overall 95\% bootstrap CIs: plausibility "
      rf"[{ci_p[0]:.2f}, {ci_p[1]:.2f}], hidden property [{ci_h[0]:.2f}, {ci_h[1]:.2f}], "
      rf"completion [{ci_c[0]:.2f}, {ci_c[1]:.2f}].}}",
      r"    \label{tab:pipeline-human-agreement}", r"\end{table}"]
(PAPER / "tables/agreement.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))

# ── Table: stage-wise ────────────────────────────────────────────────────────
T1_plaus = tool_oof(X[:, S1], Y, G)
T1_det = tool_oof(X[:, S1], DET.astype(float), G)
yp = (Y >= 3).astype(int)
L = [r"\begin{table}[t]", r"    \centering", r"    \small",
     r"    \setlength{\tabcolsep}{5pt}",
     r"    \begin{tabular}{@{}lcccc|cccc@{}}", r"        \toprule",
     r"        & \multicolumn{4}{c|}{Violation detection (AUC)} "
     r"& \multicolumn{4}{c}{Plausibility (AUC)} \\",
     r"        MLLM backbone & VLM & S1 & S1+2 & S1+2+3 & VLM & S1 & S1+2 & S1+2+3 \\",
     r"        \midrule",
     rf"        No VLM (signals only) & --- & {fmt(auc(DET, T1_det))} & "
     rf"{fmt(auc(DET, T_det))} & --- & --- & {fmt(auc(yp, T1_plaus))} & "
     rf"{fmt(auc(yp, T_plaus))} & --- \\"]
stage_rows = {}
for m in ["qwen3-vl-32b-instruct", "qwen3-vl-8b", "internvl3-14b", "internvl3-8b",
          "gemma3-12b", "qwen2.5-vl-7b"]:
    if m not in DEB or m not in MQ:
        continue
    v = col(DEB, m, "plaus_debias")
    scr = rank01(-v)                         # screening: low plausibility = suspicious
    spec = rank01(pnone(m))                  # specialist MCQ: 1 - P(no problem)
    d_row = [auc(DET, scr), auc(DET, scr + rank01(T1_det)),
             auc(DET, scr + rank01(T_det)), auc(DET, scr + rank01(T_det) + spec)]
    p_row = [auc(yp, v), auc(yp, rank01(v) + rank01(T1_plaus)),
             auc(yp, rank01(v) + rank01(T_plaus)),
             auc(yp, rank01(v) + rank01(T_plaus) + rank01(1 - pnone(m)))]
    stage_rows[m] = (d_row, p_row)
    L.append(f"        {MODEL_NAME[m]} & " + " & ".join(fmt(x) for x in d_row) + " & "
             + " & ".join(fmt(x) for x in p_row) + r" \\")
L += [r"        \bottomrule", r"    \end{tabular}",
      r"    \caption{Cumulative stage ablation on the 439 annotated clips. "
      r"\emph{VLM}: the backbone's debiased screening question alone. \emph{S1}: "
      r"fused with Stage-1 screening signals (frame differences, optical flow, "
      r"camera motion). \emph{S1+2}: additionally Stage-2 track kinematics "
      r"(camera-compensated Lucas--Kanade tracks, track loss, per-object speed and "
      r"acceleration). \emph{S1+2+3}: additionally the Stage-3 specialist "
      r"judgment ($1-P(\text{no violation})$ from a single forced-choice query over "
      r"the seven constraint families). Numeric signals are selected and oriented on "
      r"training folds only; channels are combined by an unweighted rank mean. "
      r"Violation detection separates clips with any human-recorded violation "
      r"($n{=}333$) from clips with none ($n{=}106$).}",
      r"    \label{tab:stage-ablation}", r"\end{table}"]
(PAPER / "tables/stagewise.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))

# ── Table: across backbones, observable vs unobservable ──────────────────────
L = [r"\begin{table}[t]", r"    \centering", r"    \small",
     r"    \setlength{\tabcolsep}{3.5pt}",
     r"    \begin{tabular}{@{}lcccccc|c@{}}", r"        \toprule",
     r"        & \multicolumn{2}{c}{VLM, standard prompt} & \multicolumn{2}{c}{VLM, debiased}"
     r" & \multicolumn{2}{c|}{Debiased + S1/2} & Hidden \\",
     r"        \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"        Backbone & Obs. & Unobs. & Obs. & Unobs. & Obs. & Unobs. & AUC \\",
     r"        \midrule",
     rf"        Signals only (S1/2) & --- & --- & --- & --- & "
     rf"{fmt(auc(yp[OBS], T_plaus[OBS]))} & {fmt(auc(yp[un], T_plaus[un]))} & "
     rf"{fmt(auc((HF[un] >= 3).astype(int), T_hid[un]))} \\"]
fig_rows = []
order = sorted([m for m in LIVE if m in DEB], key=lambda m: -auc(yp, col(DEB, m, "plaus_debias")))
for m in order:
    p0, p1 = col(PL, m, "plaus"), col(DEB, m, "plaus_debias")
    fz = rank01(p1) + rank01(T_plaus)
    h = col(PL, m, "hidden")
    vals = [auc(yp[OBS], p0[OBS]), auc(yp[un], p0[un]), auc(yp[OBS], p1[OBS]),
            auc(yp[un], p1[un]), auc(yp[OBS], fz[OBS]), auc(yp[un], fz[un]),
            auc((HF[un] >= 3).astype(int), h[un])]
    fig_rows.append((MODEL_NAME[m], vals))
    L.append(f"        {MODEL_NAME[m]} & " + " & ".join(fmt(x) for x in vals) + r" \\")
L += [r"        \bottomrule", r"    \end{tabular}",
      r"    \caption{Plausibility AUC (human 3--4 vs.\ 1--2) across VLM backbones on "
      r"observable ($n{=}320$) and unobservable ($n{=}119$) videos. The debiased "
      r"prompt asks the judge to count only physics errors and to ignore task "
      r"completion and the amount of motion. \emph{Hidden}: AUC for hidden-property "
      r"adherence (3--4 vs.\ 1--2) on unobservable videos when the judge is told the "
      r"property and its expected outcome. The generator-identity baseline "
      r"(cross-validated mean rating per generator) reaches 0.65.}",
      r"    \label{tab:backbones-obs-unobs}", r"\end{table}"]
(PAPER / "tables/models.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))

# ── Table: physical categories ───────────────────────────────────────────────
viol = [i for i, c in enumerate(ids) if clips[c].get("has_violation") is not False
        and len(str(clips[c].get("violated_rules") or "")) > 4]
clean = [i for i in range(len(ids)) if DET[i] == 0]
Xv = X[viol]
L = [r"\begin{table}[t]", r"    \centering", r"    \small",
     r"    \setlength{\tabcolsep}{4pt}",
     r"    \begin{tabular}{@{}lrccc|cc@{}}", r"        \toprule",
     r"        & & \multicolumn{3}{c|}{Attribution AUC} & \multicolumn{2}{c}{Detection AUC} \\",
     r"        Constraint family & Pos. & Best VLM & Mean$\pm$sd (10) & Signals & Best VLM & +S1/2 \\",
     r"        \midrule"]
catres = {}
for cat in CATS:
    yv = np.array([cat in cats_of(clips[ids[i]]) for i in viol], int)
    if yv.sum() < 8:
        continue
    per = {}
    for m in LIVE:
        if m in MQ:
            v = np.array([MQ[m].get(ids[i], {}).get("probs", {}).get(cat, np.nan) for i in viol])
            per[m] = auc(yv, np.nan_to_num(v, nan=np.nanmean(v)))
    bm = max(per, key=per.get)
    tv = tool_oof(Xv, yv.astype(float), G[viol])
    pos = [i for i in viol if cat in cats_of(clips[ids[i]])]
    ii = pos + clean
    yd = np.array([1] * len(pos) + [0] * len(clean))
    dper = {m: auc(yd, np.array([MQ[m].get(ids[i], {}).get("probs", {}).get(cat, 0.0) for i in ii]))
            for m in per}
    dm = max(dper, key=dper.get)
    tcat = tool_oof(X, np.array([1. if cat in cats_of(clips[c]) else 0. for c in ids]), G)
    dfz = auc(yd, rank01(np.array([MQ[dm].get(ids[i], {}).get("probs", {}).get(cat, 0.0) for i in ii]))
              + rank01(tcat[ii]))
    vals = list(per.values())
    catres[cat] = (int(yv.sum()), per[bm], bm, np.mean(vals), np.std(vals), auc(yv, tv), dper[dm], dm, dfz)
    L.append(f"        {cat.capitalize()} & {yv.sum()} & {per[bm]:.2f} {{\\scriptsize {SHORT[bm]}}} & "
             f"{np.mean(vals):.2f}$\\pm${np.std(vals):.2f} & {auc(yv, tv):.2f} & "
             f"{dper[dm]:.2f} {{\\scriptsize {SHORT[dm]}}} & {dfz:.2f} \\\\")
L += [r"        \bottomrule", r"    \end{tabular}",
      r"    \caption{Physical-category accuracy. \emph{Attribution}: among the 333 "
      r"videos with a recorded violation, AUC for separating videos labelled with "
      r"this family from videos with a \emph{different} violation (multi-label; "
      r"69\% of violating videos carry two or more families; Contact is merged into Collision). \emph{Detection}: "
      r"videos with this family vs.\ the 106 videos with no recorded violation. "
      r"VLM scores are the family's probability in a single forced-choice query "
      r"with shuffled option order; \emph{Signals} are Stage-1/2 statistics selected "
      r"on training folds. Best VLM is the best of ten backbones and is therefore "
      r"optimistic; the mean$\pm$sd column is not. Backbones: Q3 = Qwen3-VL, "
      r"Q2.5 = Qwen2.5-VL, IV = InternVL3, G3 = Gemma-3.}",
      r"    \label{tab:category-accuracy}", r"\end{table}"]
(PAPER / "tables/categories.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))

json.dump({"fig_rows": fig_rows, "stage_rows": {k: v for k, v in stage_rows.items()},
           "catres": {k: list(map(lambda x: x if isinstance(x, str) else float(x), v))
                      for k, v in catres.items()}},
          open(PAPER / "tables/_numbers.json", "w"), indent=1)


# NOTE: tab:main-results (paper/tables/main.tex) is no longer written from
# here. This module's own main_table() (unweighted rank-mean fusion, only the
# plausibility columns filled for the debiased rows) has been superseded by
# main_table_full.py, which fits the fusion weight on training folds and fills
# every column for every row; run that script to regenerate main.tex.
