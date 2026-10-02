"""
Everything else the consolidated benchmark can show, beyond the system table.

Each section answers one question and writes a markdown table; the whole run
produces eval_reports/CONSOLIDATED_ANALYSES.md. No model calls — everything is
computed from the staged labels, stage_signals.json, and the per-model
vlmplaus_*.json / mcq_*.json already on disk (plus data/consol_real if present).

Sections
  A  generator leaderboard — does the tool rank generators the way humans do?
  B  leave-one-generator-out — does the no-VLM tool transfer to an unseen one?
  C  which generator breaks which physics law (human labels)
  D  what hiding a property does, per generator and per property type
  E  ensembles and inter-judge agreement
  F  predicting the 1-4 rating itself (MAE, confusion)
  G  annotator effects
  H  task difficulty — do generators fail the same tasks?
  I  cost vs accuracy per judge
  J  error analysis — the most confident disagreements, with human text
  K  real demonstrations vs generated video

Usage:
  python backend/scripts/consol_analyses.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from system_eval import tool_oof, rank01, spear  # noqa: E402
from specialist_accuracy import auc, cats_of, spread  # noqa: E402

D = ROOT / "data" / "consol"
DR = ROOT / "data" / "consol_real"
OUT = ROOT / "eval_reports" / "CONSOLIDATED_ANALYSES.md"
SP = ROOT / "data" / "consol" / "logs"   # optional per-model run logs (c_<model>_plaus.log); timing column is NaN without them
GENS = ["cosmos", "wan", "hunyuan", "magi"]
CATS = ["collision", "deformation", "causality", "momentum", "gravity", "fluid", "friction"]
DEAD = {"idefics3-8b", "smolvlm2-2.2b"}

L = []


def md(s=""):
    L.append(s)
    print(s)


def table(hdr, rows):
    md("| " + " | ".join(hdr) + " |")
    md("|" + "---|" * len(hdr))
    for r in rows:
        md("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |")
    md()


def boot_mean(v, B=2000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ── load ──────────────────────────────────────────────────────────────────────
clips = {c["clip_id"]: c for c in json.loads((D / "manifest.json").read_text())["clips"]}
S = json.loads((D / "stage_signals.json").read_text())
SIG, KEYS = S["signals"], S["keys"]
ids = sorted(c for c in clips if SIG.get(c))
X = np.array([[SIG[c].get(k, 0.0) for k in KEYS] for c in ids], float)
G = np.array([clips[c]["testset_id"] for c in ids])
GEN = np.array([clips[c]["generator"] for c in ids])
OBS = np.array([clips[c]["observability"] == "observable" for c in ids])
Y = np.array([clips[c]["pc"] for c in ids], float)
ACT = np.array([1.0 if clips[c]["action_completed"] == "yes" else 0.0 for c in ids])


def is_control(f, d):
    """shuffled / non-default frame-count runs are controls: they must never be
    loaded as a model's main scores (a shuffled file silently overwrote the real
    one once, making ordered and shuffled columns identical)."""
    return (d.get("order", "temporal") != "temporal" or "shuffled" in f.name
            or "__variant" in f.name
            or "__f16" in f.name or "__f2." in f.name)

PL, MQ = {}, {}
for f in sorted(D.glob("vlmplaus_*.json")):
    d = json.loads(f.read_text())
    if is_control(f, d):
        continue
    PL[d["model"]] = d["scores"]
for f in sorted(D.glob("mcq_*.json")):
    d = json.loads(f.read_text())
    if "mcq" in d:
        MQ[d["model"]] = d["mcq"]
LIVE = [m for m in sorted(PL) if m not in DEAD]


def vlm(m, key="plaus", idl=None):
    idl = ids if idl is None else idl
    v = np.array([PL[m].get(c, {}).get(key, np.nan) for c in idl], float)
    return np.nan_to_num(v, nan=np.nanmean(v))


T = tool_oof(X, Y, G)
md("# Consolidated benchmark — further analyses")
md()
md(f"_{len(ids)} annotated clips · 4 generators · {len(PL)} VLM judges "
   f"({len(LIVE)} live) · no new model calls unless stated._")
md()

# ── A. generator leaderboard ─────────────────────────────────────────────────
md("## A. Generator leaderboard — does each system rank generators like humans?")
md()
rows, human_rank = [], {}
for g in GENS:
    m = GEN == g
    lo, hi = boot_mean(Y[m])
    row = [g, int(m.sum()), float(Y[m].mean()), f"[{lo:.2f},{hi:.2f}]",
           float(Y[m & OBS].mean()), float(Y[m & ~OBS].mean()),
           float((Y[m] >= 3).mean()), float(ACT[m].mean())]
    rows.append(row)
    human_rank[g] = Y[m].mean()
table(["generator", "n", "mean rating", "95% CI", "obs", "unobs", "% rated 3-4",
       "% action done"], sorted(rows, key=lambda r: -r[2]))
hr = [human_rank[g] for g in GENS]
agree = []
for name, s in [("tool only", T)] + [(m, vlm(m)) for m in LIVE] + \
        [(m + " + tool", rank01(vlm(m)) + rank01(T)) for m in LIVE]:
    gm = [s[GEN == g].mean() for g in GENS]
    order = "".join(g[0].upper() for g in np.array(GENS)[np.argsort(gm)[::-1]])
    # only 4 generators: spear() needs >=5 points, so rank-correlate directly
    agree.append([name, float(np.corrcoef(rank01(gm), rank01(hr))[0, 1]), order])
md("Human order (best→worst): **" + "".join(g[0].upper() for g in
   sorted(GENS, key=lambda g: -human_rank[g])) + "** (C=cosmos W=wan H=hunyuan M=magi)")
md()
table(["system", "ρ with human generator ranking", "its order"],
      sorted(agree, key=lambda r: -np.nan_to_num(r[1], nan=-9)))

# ── B. leave-one-generator-out ───────────────────────────────────────────────
md("## B. Leave-one-generator-out — does the no-VLM tool transfer to an unseen generator?")
md()
md("Tool signals selected/oriented on 3 generators, applied to the 4th. Compared "
   "with the in-distribution grouped-CV score on the same clips. The VLM needs "
   "no fitting, so its column is unchanged; the fused column uses the LOGO tool.")
md()


def logo_tool(y, k=5):
    out = np.full(len(ids), np.nan)
    R = np.column_stack([rank01(X[:, j]) for j in range(X.shape[1])])
    for g in GENS:
        tr, te = GEN != g, GEN == g
        r = np.nan_to_num([spear(R[tr, j], y[tr]) for j in range(R.shape[1])])
        top = np.argsort(-np.abs(r))[:k]
        out[te] = np.mean([np.sign(r[j]) * R[te, j] for j in top], axis=0)
    return out


TL = logo_tool(Y)
best = "internvl3-8b" if "internvl3-8b" in PL else LIVE[0]
rows = []
for g in GENS:
    m = GEN == g
    yy = (Y[m] >= 3).astype(int)
    v = vlm(best)[m]
    rows.append([g, auc(yy, T[m]), auc(yy, TL[m]), auc(yy, v),
                 auc(yy, rank01(v) + rank01(TL[m]))])
rows.append(["**mean**"] + [float(np.mean([r[i] for r in rows])) for i in range(1, 5)])
table(["held-out generator", "tool (in-dist CV)", "tool (LOGO)", f"{best} alone",
       f"{best} + LOGO tool"], rows)

# ── C. generator x physics law ───────────────────────────────────────────────
md("## C. Which generator breaks which physics law (human labels)")
md()
md("Share of each generator's clips carrying the category (multi-label).")
md()
rows = []
for g in GENS:
    m = [c for c in ids if clips[c]["generator"] == g]
    rows.append([g] + [float(np.mean([cat in cats_of(clips[c]) for c in m])) for cat in CATS]
                + [float(np.mean([not cats_of(clips[c]) for c in m]))])
table(["generator"] + CATS + ["clean"], rows)

# ── D. hiding a property ─────────────────────────────────────────────────────
md("## D. What hiding a property does")
md()
pairs = defaultdict(dict)
for i, c in enumerate(ids):
    pairs[clips[c]["pair_id"]][clips[c]["observability"]] = i
pairs = {k: v for k, v in pairs.items() if len(v) == 2}
rows = []
for g in GENS + ["all"]:
    ps = [v for k, v in pairs.items() if g == "all" or k.endswith("__" + g)]
    d = np.array([Y[v["observable"]] - Y[v["unobservable"]] for v in ps])
    hf = np.array([float(clips[ids[v["unobservable"]]]["hidden_followed"]) for v in ps])
    lo, hi = boot_mean(d)
    rows.append([g, len(ps), float(d.mean()), f"[{lo:+.2f},{hi:+.2f}]",
                 f"{int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}",
                 float(hf.mean()), float((hf >= 3).mean())])
table(["generator", "pairs", "rating drop obs→unobs", "95% CI", "obs>/=/<unobs",
       "hidden-followed mean (1-4)", "% followed (3-4)"], rows)
by = defaultdict(list)
for v in pairs.values():
    c = clips[ids[v["unobservable"]]]
    by[c.get("hidden_property") or "?"].append(float(c["hidden_followed"]))
rows = [[k, len(v), float(np.mean(v)), float(np.mean(np.array(v) >= 3))]
        for k, v in sorted(by.items(), key=lambda kv: np.mean(kv[1]))]
md("By hidden-property type (worst-followed first):")
md()
table(["hidden property", "clips", "followed mean", "% followed"], rows)

# ── E. ensembles and agreement ───────────────────────────────────────────────
md("## E. Ensembles and inter-judge agreement")
md()
Vm = {m: vlm(m) for m in LIVE}
C = np.array([[spear(Vm[a], Vm[b]) for b in LIVE] for a in LIVE])
off = C[~np.eye(len(LIVE), dtype=bool)]
hum = [spear(Vm[m], Y) for m in LIVE]
md(f"Mean judge–judge Spearman (plausibility): **{np.nanmean(off):.3f}** · "
   f"mean judge–human: **{np.nanmean(hum):.3f}**. When judges agree with each "
   f"other more than with people, averaging them cancels little.")
md()
yy = (Y >= 3).astype(int)
ens = np.mean([rank01(Vm[m]) for m in LIVE], axis=0)
rows = [["best single judge (in-sample pick)",
         max(auc(yy, Vm[m]) for m in LIVE)],
        [f"mean of {len(LIVE)} live judges", auc(yy, ens)],
        ["tool only", auc(yy, T)],
        [f"{len(LIVE)} judges + tool", auc(yy, ens + rank01(T))],
        ["generator-id baseline (see system table)", 0.650]]
md("Plausibility AUC (all clips):")
md()
table(["system", "AUC"], rows)
det = np.array([0 if clips[c]["has_violation"] is False else 1 for c in ids])
Pn = {m: np.array([MQ[m].get(c, {}).get("probs", {}).get("none", np.nan) for c in ids])
      for m in LIVE if m in MQ}
ensd = np.mean([rank01(1 - np.nan_to_num(v, nan=np.nanmean(v))) for v in Pn.values()], axis=0)
Td = tool_oof(X, det.astype(float), G)
table(["detection (violation vs clean)", "AUC"],
      [["best single judge", max(auc(det, 1 - np.nan_to_num(v, nan=np.nanmean(v))) for v in Pn.values())],
       [f"mean of {len(Pn)} judges", auc(det, ensd)],
       ["tool only", auc(det, Td)],
       ["judges + tool", auc(det, ensd + rank01(Td))]])

# ── F. predicting the rating itself ──────────────────────────────────────────
md("## F. Predicting the 1–4 rating itself")
md()
md("Score → rating by quantile-matching on training folds (grouped by task): "
   "the held-out clip gets the rating at its score's percentile in the "
   "training fold's rating distribution. No labels from the test fold are used.")
md()


def to_rating(score, k=5):
    ug = np.unique(G)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    f = np.array([{g: i % k for i, g in enumerate(ug)}[g] for g in G])
    pred = np.zeros(len(ids))
    for i in range(k):
        tr, te = f != i, f == i
        q = np.sort(Y[tr])
        pct = np.array([(score[tr] < s).mean() for s in score[te]])
        pred[te] = q[np.clip((pct * len(q)).astype(int), 0, len(q) - 1)]
    return pred


def rating_row(name, pred):
    cm = np.zeros((4, 4), int)
    for t, p in zip(Y.astype(int), pred.astype(int)):
        cm[t - 1, p - 1] += 1
    return [name, float(np.abs(pred - Y).mean()), float((pred == Y).mean()),
            float((np.abs(pred - Y) <= 1).mean())], cm


maj = np.full(len(ids), Counter(Y).most_common(1)[0][0])
gm = np.array([Y[GEN == g].mean() for g in GEN])
rows, cms = [], {}
for name, pred in [("always majority class", maj),
                   ("generator mean (rounded)", np.round(gm)),
                   ("tool only", to_rating(T)),
                   (f"{best} alone", to_rating(vlm(best))),
                   (f"{best} + tool", to_rating(rank01(vlm(best)) + rank01(T))),
                   (f"{len(LIVE)} judges + tool", to_rating(ens + rank01(T)))]:
    r, cm = rating_row(name, pred)
    rows.append(r)
    cms[name] = cm
table(["predictor", "MAE", "exact", "within ±1"], rows)
cm = cms[f"{len(LIVE)} judges + tool"]
md(f"Confusion, {len(LIVE)} judges + tool (rows = human, cols = predicted):")
md()
table(["human \\ pred", "1", "2", "3", "4"], [[str(i + 1)] + list(map(int, cm[i])) for i in range(4)])

# ── G. annotators ────────────────────────────────────────────────────────────
md("## G. Annotator effects")
md()
ann = np.array([str(clips[c].get("annotator") or "?") for c in ids])
if len(set(ann)) <= 1:
    import pandas as pd
    lab = pd.read_csv(ROOT / "data/consolidated_labels.csv").set_index("id")
    ann = np.array([str(lab.loc[c, "annotator"]) for c in ids])
rows = []
for a in sorted(set(ann)):
    m = ann == a
    if m.sum() < 8:
        continue
    gs = Counter(GEN[m])
    rows.append([a, int(m.sum()), float(Y[m].mean()),
                 ", ".join(f"{g}:{n}" for g, n in gs.most_common()),
                 auc((Y[m] >= 3).astype(int), rank01(vlm(best)[m]) + rank01(T[m]))
                 if 0 < (Y[m] >= 3).sum() < m.sum() else float("nan")])
table(["annotator", "clips", "mean rating", "generators rated",
       f"{best}+tool AUC on their clips"], rows)
# offsets controlling for generator: OLS rating ~ generator + annotator
A = sorted(set(ann))
Xd = np.column_stack([np.ones(len(ids))] + [(GEN == g).astype(float) for g in GENS[1:]]
                     + [(ann == a).astype(float) for a in A[1:]])
beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
off = dict(zip(A, [0.0] + list(beta[len(GENS):])))
md("Annotator offset after controlling for generator (OLS, relative to "
   f"`{A[0]}`): " + ", ".join(f"{a} {v:+.2f}" for a, v in off.items()))
md()

# ── H. task difficulty ───────────────────────────────────────────────────────
md("## H. Task difficulty — do generators fail the same tasks?")
md()
task = {}
for i, c in enumerate(ids):
    if OBS[i]:
        task.setdefault(clips[c]["testset_id"], {})[clips[c]["generator"]] = Y[i]
full = {t: v for t, v in task.items() if len(v) == 4}
M = np.array([[full[t][g] for g in GENS] for t in sorted(full)])
cc = [spear(M[:, a], M[:, b]) for a in range(4) for b in range(a + 1, 4)]
md(f"{len(full)} tasks rated for all 4 generators. Mean between-generator "
   f"Spearman on the SAME task: **{np.nanmean(cc):.3f}** — "
   + ("tasks are hard for everyone." if np.nanmean(cc) > 0.3 else
      "difficulty is mostly generator-specific, not task-intrinsic."))
md()
names = {}
for c in clips.values():
    names[c["testset_id"]] = c.get("task") or ""
tm = sorted(full, key=lambda t: np.mean(list(full[t].values())))
table(["task id", "task", "mean over 4 gens", "cosmos", "wan", "hunyuan", "magi"],
      [[t, names[t][:55], float(np.mean(list(full[t].values())))] +
       [float(full[t][g]) for g in GENS] for t in tm[:8]])
md("Easiest:")
md()
table(["task id", "task", "mean over 4 gens"],
      [[t, names[t][:55], float(np.mean(list(full[t].values())))] for t in tm[-5:][::-1]])

# ── I. cost vs accuracy ──────────────────────────────────────────────────────
md("## I. Cost vs accuracy per judge")
md()
KEYMAP = {"qwen3-vl-32b-instruct": "qwen3vl", "llama4-scout-17b": "llama4",
          "internvl3-8b": "iv8", "qwen2.5-vl-7b": "q7"}
PARAMS = {"qwen3-vl-32b-instruct": 32, "llama4-scout-17b": 109, "internvl3-8b": 8,
          "qwen2.5-vl-7b": 7, "internvl3-14b": 14, "qwen3-vl-8b": 8, "llava-ov-7b": 7,
          "gemma3-12b": 12, "idefics3-8b": 8, "smolvlm2-2.2b": 2.2,
          "mistral-small-24b": 24, "qwen2.5-vl-32b": 32}
rows = []
for m in sorted(PL):
    k = KEYMAP.get(m, m)
    lg = SP / f"c_{k}_plaus.log"
    sec = float("nan")
    if lg.exists():
        t = re.findall(r"answers \((\d+)s\)", lg.read_text())
        if t:
            sec = int(t[-1]) / len(ids)
    rows.append([m + (" ⚠dead" if m in DEAD else ""), PARAMS.get(m, "?"),
                 "API" if m in ("qwen3-vl-32b-instruct", "llama4-scout-17b") else "local",
                 sec, auc(yy, vlm(m)), auc(yy, rank01(vlm(m)) + rank01(T))])
table(["judge", "params (B)", "where", "sec / clip (3 questions)",
       "plaus AUC alone", "plaus AUC + tool"], sorted(rows, key=lambda r: -r[5]))

# ── J. error analysis ────────────────────────────────────────────────────────
md("## J. Error analysis — most confident disagreements")
md()
fused = ens + rank01(T)
z = rank01(fused)
import pandas as pd
lab = pd.read_csv(ROOT / "data/consolidated_labels.csv").fillna("").set_index("id")
fp = [i for i in np.argsort(-z) if Y[i] <= 1][:6]
fn = [i for i in np.argsort(z) if Y[i] >= 4][:6]
md("**System says plausible, human rated 1:**")
md()
table(["clip", "system pct", "human", "what the annotator wrote"],
      [[ids[i], float(z[i]), int(Y[i]), str(lab.loc[ids[i], "rules"])[:90]] for i in fp])
md("**System says implausible, human rated 4:**")
md()
table(["clip", "system pct", "human", "task"],
      [[ids[i], float(z[i]), int(Y[i]), str(lab.loc[ids[i], "task"])[:70]] for i in fn])


# ── L. the static-video confound ─────────────────────────────────────────────
md("## L. Does 'nothing happens' get rated as 'plausible'?")
md()
mot = np.array([SIG[c].get("s1_obj_motion", np.nan) for c in ids])
rows = []
for g in GENS:
    m = GEN == g
    rows.append([g, float(np.nanmedian(mot[m])), float(ACT[m].mean()), float(Y[m].mean())])
table(["generator", "median object motion", "% action done", "mean rating"], rows)
md(f"Spearman(object motion, human rating) over all clips: **{spear(mot, Y):+.3f}**; "
   f"within action-completed clips only: **{spear(mot[ACT==1], Y[ACT==1]):+.3f}**; "
   f"Spearman(action done, rating): **{spear(ACT, Y):+.3f}**.")
md()
md("Re-scoring plausibility on **action-completed clips only** — the clips where "
   "the robot actually did something, so 'plausible' cannot come from standing still:")
md()
a = ACT == 1
ya = (Y[a] >= 3).astype(int)
Ta = tool_oof(X[a], Y[a], G[a])
rows = [["tool only", auc(ya, Ta)]]
for m in LIVE:
    rows.append([m, auc(ya, vlm(m)[a])])
    rows.append([m + " + tool", auc(ya, rank01(vlm(m)[a]) + rank01(Ta))])
gma = np.array([Y[a][GEN[a] == g].mean() for g in GEN[a]])
rows.append(["generator-id baseline (in-sample)", auc(ya, gma)])
table(["system (action-completed clips, n=%d)" % a.sum(), "plaus AUC"],
      sorted(rows, key=lambda r: -r[1]))

# ── S. is tool + VLM really better? paired bootstrap ─────────────────────────
md("## S. Is 'tool + VLM' significantly better than 'VLM alone'?")
md()
md("Paired bootstrap over clips (1,000 resamples): CI of AUC(tool+VLM) − AUC(VLM). "
   "✓ = CI excludes 0.")
md()


def paired_ci(y, a, b, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(auc(y[i], b[i]) - auc(y[i], a[i]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


Td2 = tool_oof(X, det.astype(float), G)
TA2 = tool_oof(X, ACT, G)
rows, wins = [], Counter()
for m in LIVE:
    row = [m]
    for lab, y, v, t in [("plaus", yy, vlm(m), T), ("action", ACT.astype(int), vlm(m, "action"), TA2),
                         ("detect", det, 1 - np.nan_to_num(Pn.get(m, np.full(len(ids), np.nan)), nan=0.5), Td2)]:
        f = rank01(v) + rank01(t)
        lo, hi = paired_ci(y, v, f)
        dlt = auc(y, f) - auc(y, v)
        ok = lo > 0
        wins[lab] += ok
        row.append(f"{dlt:+.3f} [{lo:+.2f},{hi:+.2f}]" + (" ✓" if ok else ""))
    rows.append(row)
table(["judge", "plausibility Δ", "action Δ", "detection Δ"], rows)
md(f"Significant gains: plausibility {wins['plaus']}/{len(LIVE)}, action "
   f"{wins['action']}/{len(LIVE)}, detection {wins['detect']}/{len(LIVE)}.")
md()

# ── U. fusion weight ─────────────────────────────────────────────────────────
md("## U. How much should the tool weigh against the VLM?")
md()
md("score = w·rank(VLM) + (1−w)·rank(tool). Descriptive sweep (in-sample; the "
   "default w=0.5 was fixed in advance, not chosen from this).")
md()
ws = [0.0, 0.25, 0.5, 0.75, 1.0]
rows = []
for m in ["internvl3-8b", "gemma3-12b", "qwen3-vl-8b", "qwen3-vl-32b-instruct"]:
    if m in PL:
        rows.append([m] + [auc(yy, w * rank01(vlm(m)) + (1 - w) * rank01(T)) for w in ws])
table(["judge (plausibility AUC)"] + [f"w={w}" for w in ws], rows)

# ── V. hidden property -> specialist ─────────────────────────────────────────
md("## V. When a hidden property is ignored, does the matching specialist fire?")
md()
PMAP = {"viscosity": "fluid", "surface_condition": "friction",
        "elasticity": "deformation", "mass": "momentum"}
md("Unobservable clips only. Property → specialist mapping fixed a priori: "
   + ", ".join(f"{k}→{v}" for k, v in PMAP.items())
   + ". AUC of the mapped specialist's MCQ probability for 'property NOT followed' "
     "(human hidden-followed ≤ 2) vs followed.")
md()
un = [i for i, c in enumerate(ids) if not OBS[i]]
rows = []
for m in LIVE:
    if m not in MQ:
        continue
    row = [m]
    allv, ally = [], []
    for prop, cat in PMAP.items():
        ii = [i for i in un if clips[ids[i]].get("hidden_property") == prop]
        y = np.array([float(clips[ids[i]]["hidden_followed"]) <= 2 for i in ii], int)
        v = np.array([MQ[m].get(ids[i], {}).get("probs", {}).get(cat, np.nan) for i in ii])
        v = np.nan_to_num(v, nan=np.nanmean(v))
        row.append(auc(y, v) if 0 < y.sum() < len(y) else float("nan"))
    rows.append(row)
table(["judge"] + [f"{k}→{v}" for k, v in PMAP.items()], rows)
md("Note: 'not followed' is 76% of unobservable clips, and per-property n is "
   "20–40, so single cells are noisy; read rows, not cells.")
md()

# ── W. specialist attribution per generator ──────────────────────────────────
md("## W. Specialist attribution per generator (best attribution judge)")
md()
bj = "qwen2.5-vl-7b" if "qwen2.5-vl-7b" in MQ else sorted(MQ)[0]
viol = [i for i, c in enumerate(ids) if clips[c].get("has_violation") is not False
        and len(str(clips[c].get("violated_rules") or "")) > 4]
rows = []
for g in GENS + ["all"]:
    vi = [i for i in viol if g == "all" or GEN[i] == g]
    row = [g, len(vi)]
    for cat in CATS:
        y = np.array([cat in cats_of(clips[ids[i]]) for i in vi], int)
        v = np.array([MQ[bj].get(ids[i], {}).get("probs", {}).get(cat, 0.0) for i in vi])
        row.append(auc(y, v) if 5 <= y.sum() <= len(y) - 5 else "—")
    rows.append(row)
md(f"Judge: `{bj}` (VLM alone — the system that wins attribution). '—' = fewer "
   "than 5 positives in that generator.")
md()
table(["generator", "violation clips"] + CATS, rows)

# ── P. pairwise comparison ───────────────────────────────────────────────────
PW = {}
for f in sorted(D.glob("pairwise_*.json")):
    d = json.loads(f.read_text())
    PW[d["model"] + ("  [debiased prompt]" if d.get("prompt") == "debias" else "")] = d["pairs"]
if PW:
    md("## P. Pairwise judgement — show two videos, ask which is more plausible")
    md()
    md("Each pair asked in both orders and averaged, so an always-'A' model scores "
       "exactly 50%. Accuracy over pairs whose human ratings differ. For comparison, "
       "the absolute-score methods are applied to the same pairs: the pair is "
       "ordered by which clip got the higher standalone score.")
    md()
    Yd = {c: Y[i] for i, c in enumerate(ids)}
    Tsc = {c: T[i] for i, c in enumerate(ids)}
    rows = []
    for kind in ("twins", "gens"):
        for m, pr in sorted(PW.items()):
            ks = [k for k, v in pr.items() if v["kind"] == kind]
            lab = [(k, Yd[k.split("|")[0]] - Yd[k.split("|")[1]]) for k in ks]
            lab = [(k, d) for k, d in lab if d != 0]
            if len(lab) < 10:
                continue
            acc = np.mean([(pr[k]["p"] > 0.5) == (d > 0) for k, d in lab])
            cons = np.mean([(pr[k]["raw"][0] > 0.5) != (pr[k]["raw"][1] > 0.5) for k, _ in lab])
            va = vlm(m.split("  [")[0]) if m.split("  [")[0] in PL else None
            vs = ({c: va[i] for i, c in enumerate(ids)} if va is not None else None)
            abs_acc = (np.mean([(vs[k.split("|")[0]] > vs[k.split("|")[1]]) == (d > 0)
                                for k, d in lab]) if vs else float("nan"))
            tool_acc = np.mean([(Tsc[k.split("|")[0]] > Tsc[k.split("|")[1]]) == (d > 0)
                                for k, d in lab])
            rng = np.random.default_rng(0)
            hits = np.array([(pr[k]["p"] > 0.5) == (d > 0) for k, d in lab], float)
            bs = [rng.choice(hits, len(hits)).mean() for _ in range(2000)]
            rows.append([kind, m, len(lab), float(acc),
                         f"[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]",
                         float(cons), abs_acc, float(tool_acc)])
    table(["pair set", "judge", "untied pairs", "PAIRWISE acc", "95% CI",
           "order-consistent", "same judge, absolute scores", "tool, absolute"], rows)
    md("`order-consistent` = the judge picks the same video in both orders; a "
       "position-biased judge scores low here even when its averaged accuracy looks fine.")
    md()

# ── X. frame-order control and frame count ───────────────────────────────────
SH, FC = {}, {}
for f in sorted(D.glob("vlmplaus_*__shuffled.json")):
    d = json.loads(f.read_text())
    SH[d["model"]] = d["scores"]
for f in sorted(D.glob("vlmplaus_*__f2.json")) + sorted(D.glob("vlmplaus_*__f16.json")):
    d = json.loads(f.read_text())
    FC[(d["model"], d["frames"])] = d["scores"]
if SH:
    md("## X. Frame-order control — do the judges read motion at all?")
    md()
    md("Same frames, shuffled into a random order fixed at staging. If a judge "
       "reads motion, shuffling should hurt it; if its score barely moves, it is "
       "judging frames one at a time.")
    md()
    rows = []
    for m, sc in sorted(SH.items()):
        vs = np.array([sc.get(c, {}).get("plaus", np.nan) for c in ids])
        vs = np.nan_to_num(vs, nan=np.nanmean(vs))
        vt = vlm(m)
        hs = np.array([sc.get(c, {}).get("hidden", np.nan) for c in ids])
        ht = vlm(m, "hidden")
        un_ = ~OBS
        rows.append([m, auc(yy, vt), auc(yy, vs), spear(vt, vs),
                     auc(ACT.astype(int), vlm(m, "action")),
                     auc(ACT.astype(int), np.nan_to_num(np.array([sc.get(c, {}).get("action", np.nan) for c in ids]), nan=0.5))])
    table(["judge", "plaus AUC ordered", "plaus AUC SHUFFLED", "ρ(ordered, shuffled scores)",
           "action AUC ordered", "action AUC SHUFFLED"], rows)
if FC:
    md("## Frame count")
    md()
    rows = []
    for (m, k), sc in sorted(FC.items()):
        v = np.nan_to_num(np.array([sc.get(c, {}).get("plaus", np.nan) for c in ids]), nan=2.5)
        a = np.nan_to_num(np.array([sc.get(c, {}).get("action", np.nan) for c in ids]), nan=1.5)
        rows.append([m, k, auc(yy, v), auc(ACT.astype(int), a)])
    for m in sorted({m for m, _ in FC}):
        rows.append([m, 8, auc(yy, vlm(m)), auc(ACT.astype(int), vlm(m, "action"))])
    table(["judge", "frames", "plaus AUC", "action AUC"], sorted(rows, key=lambda r: (r[0], r[1])))

# ── Y. which signals does the tool use? ──────────────────────────────────────
md("## Y. What the no-VLM tool is actually measuring")
md()
md("Signals selected in each training fold (top-5 by |Spearman| with the target), "
   "over 20 random fold assignments. `sign` > 0 means more of the signal → the "
   "target is higher (more plausible / action done / has a violation).")
md()
R0 = np.column_stack([rank01(X[:, j]) for j in range(X.shape[1])])


def sel_counts(y, reps=20, k=5):
    cnt, sgn = Counter(), defaultdict(list)
    for rep in range(reps):
        ug = np.unique(G)
        rng = np.random.default_rng(rep)
        rng.shuffle(ug)
        f = np.array([{g: i % 5 for i, g in enumerate(ug)}[g] for g in G])
        for i in range(5):
            tr = f != i
            r = np.nan_to_num([spear(R0[tr, j], y[tr]) for j in range(R0.shape[1])])
            for j in np.argsort(-np.abs(r))[:k]:
                cnt[KEYS[j]] += 1
                sgn[KEYS[j]].append(np.sign(r[j]))
    return cnt, sgn


for lab, y in [("plausibility", Y), ("action completed", ACT), ("has a violation", det.astype(float))]:
    cnt, sgn = sel_counts(y)
    md(f"**{lab}**")
    md()
    table(["signal", "selected (of 100 folds)", "sign"],
          [[k, v, f"{np.mean(sgn[k]):+.0f}"] for k, v in cnt.most_common(7)])

# ── Z. debiased wording ──────────────────────────────────────────────────────
VD = {}
for f in sorted(D.glob("vlmplaus_*__variant-plaus_debias.json")):
    d = json.loads(f.read_text())
    VD[d["model"]] = d["scores"]
if VD:
    md("## Z. Debiased wording — tell the judge to ignore task success and motion")
    md()
    md("Plain question: 'how physically plausible is this video?'. Debiased: count "
       "only named physics errors, explicitly ignoring whether the task finished and "
       "how much happens. If judges were rewarding activity, debiasing should "
       "weaken the score's link to motion and bring it closer to humans.")
    md()
    mot_ = np.array([SIG[c].get("s1_obj_motion", np.nan) for c in ids])
    rows = []
    for m, sc in sorted(VD.items()):
        v = np.array([sc.get(c, {}).get("plaus_debias", np.nan) for c in ids])
        v = np.nan_to_num(v, nan=np.nanmean(v))
        p0 = vlm(m) if m in PL else np.full(len(ids), np.nan)
        rows.append([m, auc(yy, p0), auc(yy, v), spear(p0, mot_), spear(v, mot_),
                     spear(p0, Y), spear(v, Y), auc(yy, rank01(v) + rank01(T))])
    table(["judge", "plain AUC", "DEBIASED AUC", "ρ(plain, motion)", "ρ(debiased, motion)",
           "ρ(plain, human)", "ρ(debiased, human)", "debiased + tool AUC"], rows)
    md(f"For reference: ρ(human rating, motion) = {spear(Y, mot_):+.3f}.")
    md()

# ── K. real vs generated ─────────────────────────────────────────────────────
if (DR / "stage_signals.json").exists():
    md("## K. Real demonstrations vs generated video")
    md()
    rc = {c["clip_id"]: c for c in json.loads((DR / "manifest.json").read_text())["clips"]}
    RS = json.loads((DR / "stage_signals.json").read_text())["signals"]
    rid = sorted(c for c in rc if RS.get(c))
    RX = np.array([[RS[c].get(k, 0.0) for k in KEYS] for c in rid], float)
    RP = {}
    for f in sorted(DR.glob("vlmplaus_*.json")):
        d = json.loads(f.read_text())
        if is_control(f, d):
            continue
        RP[d["model"]] = d["scores"]
    RM = {}
    for f in sorted(DR.glob("mcq_*.json")):
        d = json.loads(f.read_text())
        if "mcq" in d:
            RM[d["model"]] = d["mcq"]
    # tool: the recipe learned on generated clips (all of them), applied to real
    R_all = np.vstack([X, RX])
    Rk = np.column_stack([rank01(R_all[:, j]) for j in range(R_all.shape[1])])
    r = np.nan_to_num([spear(Rk[:len(ids), j], Y) for j in range(Rk.shape[1])])
    top = np.argsort(-np.abs(r))[:5]
    tool_all = np.mean([np.sign(r[j]) * Rk[:, j] for j in top], axis=0)
    t_gen, t_real = tool_all[:len(ids)], tool_all[len(ids):]
    md(f"{len(rid)} real demos (the source videos every generated clip started from). "
       "Real = physically correct by construction. Scores are 'higher = more plausible'.")
    md()
    rows = []
    good = Y >= 4
    for name, sg, sr in [("tool only", t_gen, t_real)] + \
            [(m, np.nan_to_num(np.array([PL[m].get(c, {}).get("plaus", np.nan) for c in ids])),
              np.nan_to_num(np.array([RP[m].get(c, {}).get("plaus", np.nan) for c in rid])))
             for m in sorted(RP) if m in PL]:
        yb = np.r_[np.ones(len(sr)), np.zeros(len(sg))]
        rows.append([name, float(np.mean(sr)), float(np.mean(sg[good])), float(np.mean(sg[Y <= 1])),
                     auc(yb.astype(int), np.r_[sr, sg]),
                     auc(np.r_[np.ones(len(sr)), np.zeros(good.sum())].astype(int), np.r_[sr, sg[good]])])
    table(["system", "mean on REAL", "mean on gen rated 4", "mean on gen rated 1",
           "AUC real vs all gen", "AUC real vs gen-rated-4"], rows)
    md("`AUC real vs gen-rated-4` is the sharp one: both sides are physically fine "
       "per humans, so a high value means the system is detecting *generated video*, "
       "not *broken physics*.")
    md()
    rows = []
    for m in sorted(RM):
        if m not in MQ:
            continue
        pr = np.array([RM[m][c]["probs"]["none"] for c in rid if c in RM[m]])
        pg = np.array([MQ[m][c]["probs"]["none"] for c in ids if c in MQ[m]])
        am = np.array([max(RM[m][c]["probs"], key=RM[m][c]["probs"].get) for c in rid if c in RM[m]])
        rows.append([m, float(pr.mean()), float(pg.mean()), float((am != "none").mean())])
    if rows:
        md("False alarms on real footage (MCQ):")
        md()
        table(["judge", "mean P(none) real", "mean P(none) generated",
               "% real clips accused of a violation"], rows)

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(L))
print(f"\n-> {OUT}")
