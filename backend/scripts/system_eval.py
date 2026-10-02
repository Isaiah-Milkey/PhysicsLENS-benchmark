"""
Three systems x several VLMs on the consolidated benchmark.

  VLM alone      the judge's own answers: 1-4 plausibility, P(action done),
                 hidden-property rating, MCQ P(category).
  Tool, no VLM   Stage-1/2 signals only (stage_signals.json, 28 per clip). No
                 model calls.
  Tool + VLM     rank-mean of the two above.

HOW THE NO-VLM TOOL IS SCORED WITHOUT CHEATING. The signals have no natural
polarity and there are 28 of them, so something must choose direction and
subset. That choice is made on TRAINING folds only: grouped 5-fold CV, groups =
testset_id, so an observable clip and its unobservable twin (same source frame)
never straddle a fold, and neither do the four generators' versions of the same
demo. Within each training fold the top-k signals by |Spearman| with the target
are kept, oriented, rank-averaged; the held-out fold is scored with that recipe.
The whole procedure repeats per target, so the tool gets the same chance to fit
each question as the VLM gets to answer it. Selection cannot borrow last week's
RobotBench choices: the Cosmos/Wan observable clips here ARE those videos.

The VLM side fits nothing. Fusion is an unweighted rank-mean of the VLM score and
the tool's out-of-fold score, so it too is fitted on nothing held-out.

SCOPES. `pairs` = the 119 obs/unobs pairs (238 clips), `all` = every staged clip.

Usage:
  python backend/scripts/system_eval.py --data data/consol --scope pairs
  python backend/scripts/system_eval.py --data data/consol --scope all
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, _avg_rank, cats_of, tie_frac, spread  # noqa: E402

CATS = ["collision", "deformation", "causality", "momentum", "gravity",
        "fluid", "friction"]


def rank01(v):
    v = np.asarray(v, float)
    return (_avg_rank(v) - 1) / max(len(v) - 1, 1)


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(_avg_rank(a[m]), _avg_rank(b[m]))[0, 1])


def boot(fn, n, B=1000, seed=0):
    """95% CI of a metric by resampling clip indices."""
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        x = fn(rng.integers(0, n, n))
        if np.isfinite(x):
            v.append(x)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def tool_oof(X, y, groups, k=5, nfold=5, seed=0):
    """Out-of-fold tool score: select + orient top-k signals on train folds."""
    y = np.asarray(y, float)
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    fold = {g: i % nfold for i, g in enumerate(ug)}
    f = np.array([fold[g] for g in groups])
    R = np.column_stack([rank01(X[:, j]) for j in range(X.shape[1])])
    out = np.full(len(y), np.nan)
    for i in range(nfold):
        tr, te = f != i, f == i
        ok = tr & np.isfinite(y)
        if ok.sum() < 10 or te.sum() == 0:
            continue
        r = np.array([spear(R[ok, j], y[ok]) for j in range(R.shape[1])])
        r = np.nan_to_num(r)
        top = np.argsort(-np.abs(r))[:k]
        out[te] = np.mean([np.sign(r[j]) * R[te, j] for j in top], axis=0)
    return out


def metric_rows(name, score, clips, sub, want):
    """score: higher = MORE plausible / more likely yes. Returns dict of metrics."""
    s = np.asarray(score, float)
    row = {"system": name}
    pc = np.array([clips[c]["pc"] for c in sub], float)
    m = np.isfinite(s)
    if "plaus" in want:
        row["rho"] = spear(s, pc)
        bad = pc <= 2
        yy = (~bad).astype(int)
        row["auc_plaus"] = auc(yy[m], s[m])
        row["auc_ci"] = boot(lambda i: auc(yy[i][m[i]], s[i][m[i]]), len(s))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/consol")
    ap.add_argument("--scope", default="pairs", choices=["pairs", "all"])
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    clips = {c["clip_id"]: c for c in json.loads((data / "manifest.json").read_text())["clips"]}
    sig = json.loads((data / "stage_signals.json").read_text())
    S, keys = sig["signals"], sig["keys"]

    if a.scope == "pairs":
        pc = {}
        for c in clips.values():
            pc.setdefault(c["pair_id"], set()).add(c["observability"])
        ids = sorted(c for c in clips if len(pc[clips[c]["pair_id"]]) == 2)
    else:
        ids = sorted(clips)
    ids = [c for c in ids if S.get(c)]
    X = np.array([[S[c].get(k, 0.0) for k in keys] for c in ids], float)
    G = np.array([clips[c]["testset_id"] for c in ids])
    pcv = np.array([clips[c]["pc"] for c in ids], float)

    models = {}
    for f in sorted(data.glob("vlmplaus_*.json")):
        if "shuffled" in f.name or "__f16" in f.name or "__f2." in f.name or "__variant" in f.name:
            continue  # control variants must never overwrite the main run
        d = json.loads(f.read_text())
        models.setdefault(d["model"], {})["plaus"] = d["scores"]
    # debiased-wording variants enter as their own judges ("<model> [debiased]"),
    # plausibility only — they have no MCQ, so the category sections skip them
    for f in sorted(data.glob("vlmplaus_*__variant-plaus_debias.json")):
        d = json.loads(f.read_text())
        models.setdefault(d["model"] + " [debiased]", {})["plaus"] = {
            c: {"plaus": v["plaus_debias"]} for c, v in d["scores"].items()
            if "plaus_debias" in v}
    for f in sorted(data.glob("mcq_*.json")):
        if "shuffled" in f.name or "__f16" in f.name or "__f2." in f.name or "__variant" in f.name:
            continue  # control variants must never overwrite the main run
        d = json.loads(f.read_text())
        if "mcq" in d:
            models.setdefault(d["model"], {})["mcq"] = d["mcq"]

    R = {"scope": a.scope, "n": len(ids), "models": {}}
    print(f"scope={a.scope} | {len(ids)} clips | models: {', '.join(models)}\n")

    # ── tool-only scores (model independent) ────────────────────────────────
    T_plaus = tool_oof(X, pcv, G, a.k)
    act = np.array([1.0 if clips[c].get("action_completed") == "yes" else
                    0.0 if clips[c].get("action_completed") == "no" else np.nan
                    for c in ids])
    T_act = tool_oof(X, act, G, a.k)
    unob = np.array([clips[c]["observability"] == "unobservable" for c in ids])
    hf = np.array([float(clips[c]["hidden_followed"]) if unob[i] and str(clips[c]["hidden_followed"]) not in ("", "nan") else np.nan
                   for i, c in enumerate(ids)])
    T_hid = np.full(len(ids), np.nan)
    if unob.sum() > 20:
        T_hid[unob] = tool_oof(X[unob], hf[unob], G[unob], a.k)

    def evalset(label, score_fn):
        pass

    # ── 1. plausibility ─────────────────────────────────────────────────────
    def plaus_metrics(s):
        s = np.asarray(s, float)
        m = np.isfinite(s)
        yy = (pcv >= 3).astype(int)
        lo, hi = boot(lambda i: auc(yy[i][m[i]], s[i][m[i]]), len(s))
        return spear(s, pcv), auc(yy[m], s[m]), lo, hi

    print("=" * 96)
    print("1. PHYSICAL PLAUSIBILITY vs human 1-4   (rho = Spearman; AUC = good(3-4) vs bad(1-2))")
    print("=" * 96)
    print(f"{'model':26s} {'system':14s} {'rho':>7s} {'AUC':>7s} {'95% CI':>15s}")
    print("-" * 96)
    # BASELINE: generator identity alone, as a CV'd mean rating per generator.
    # Generators differ in average quality, so any score that merely recognises
    # the generator gets credit for it. A system must beat this row to be
    # detecting physics rather than provenance.
    gen = np.array([clips[c]["generator"] for c in ids])
    B = np.full(len(ids), np.nan)
    ug = np.unique(G); rng = np.random.default_rng(0); rng.shuffle(ug)
    fold = {g: i % 5 for i, g in enumerate(ug)}; f = np.array([fold[g] for g in G])
    for i in range(5):
        tr, te = f != i, f == i
        for g in np.unique(gen):
            B[te & (gen == g)] = pcv[tr & (gen == g)].mean()
    r, au, lo, hi = plaus_metrics(B)
    print(f"{'(baseline)':26s} {'generator id':14s} {r:7.3f} {au:7.3f} {f'[{lo:.2f},{hi:.2f}]':>15s}")
    R["generator_baseline"] = dict(rho=r, auc=au, ci=[lo, hi])
    r, au, lo, hi = plaus_metrics(T_plaus)
    print(f"{'(none)':26s} {'tool only':14s} {r:7.3f} {au:7.3f} {f'[{lo:.2f},{hi:.2f}]':>15s}")
    R["tool_only"] = {"plaus": dict(rho=r, auc=au, ci=[lo, hi])}
    for mdl, D in models.items():
        P = D.get("plaus", {})
        v = np.array([P.get(c, {}).get("plaus", np.nan) for c in ids], float)
        if np.isfinite(v).sum() < 0.5 * len(v):
            continue
        fused = rank01(np.nan_to_num(v, nan=np.nanmean(v))) + rank01(T_plaus)
        # a judge that gives (nearly) every clip the same rating is not judging;
        # flag it so its row — and its "tool + VLM" row, which is then just the
        # tool — cannot be read as a result
        vv = v[np.isfinite(v)]
        dead = tie_frac(np.round(vv, 3)) >= 0.75 or spread(vv) < 0.05
        out = {}
        for sysn, s in (("VLM only", v), ("tool + VLM", fused)):
            r, au, lo, hi = plaus_metrics(s)
            out[sysn] = dict(rho=r, auc=au, ci=[lo, hi], dead=bool(dead))
            print(f"{mdl[:26]:26s} {sysn:14s} {r:7.3f} {au:7.3f} {f'[{lo:.2f},{hi:.2f}]':>15s}"
                  + ("   DEAD judge (constant rating)" if dead else ""))
        R["models"].setdefault(mdl, {})["plaus"] = out

    # ── 2. per-generator plausibility AUC ───────────────────────────────────
    gens = sorted({clips[c]["generator"] for c in ids})
    print(f"\n{'='*96}\n2. PLAUSIBILITY AUC PER GENERATOR\n{'='*96}")
    print(f"{'model':26s} {'system':14s}" + "".join(f"{g:>10s}" for g in gens))
    print("-" * 96)

    def per_gen(s):
        s = np.asarray(s, float)
        out = []
        for g in gens:
            m = np.array([clips[c]["generator"] == g for c in ids]) & np.isfinite(s)
            out.append(auc((pcv[m] >= 3).astype(int), s[m]) if m.sum() > 10 else np.nan)
        return out
    print(f"{'(none)':26s} {'tool only':14s}" + "".join(f"{x:10.3f}" for x in per_gen(T_plaus)))
    for mdl, D in models.items():
        P = D.get("plaus", {})
        v = np.array([P.get(c, {}).get("plaus", np.nan) for c in ids], float)
        if np.isfinite(v).sum() < 0.5 * len(v):
            continue
        fused = rank01(np.nan_to_num(v, nan=np.nanmean(v))) + rank01(T_plaus)
        for sysn, s in (("VLM only", v), ("tool + VLM", fused)):
            print(f"{mdl[:26]:26s} {sysn:14s}" + "".join(f"{x:10.3f}" for x in per_gen(s)))

    # ── 3. category attribution ─────────────────────────────────────────────
    viol = [i for i, c in enumerate(ids) if clips[c].get("has_violation") is not False
            and len(str(clips[c].get("violated_rules") or "")) > 4]
    print(f"\n{'='*96}\n3. SPECIALIST ATTRIBUTION AUC (this violation vs a different one, "
          f"{len(viol)} clips)\n{'='*96}")
    cat_rows = {}
    usable = []
    for cat in CATS:
        y = np.array([cat in cats_of(clips[ids[i]]) for i in viol], int)
        if 8 <= y.sum() <= len(y) - 8:
            usable.append(cat)
    print(f"{'model':26s} {'system':14s}" + "".join(f"{c[:9]:>10s}" for c in usable) + f"{'mean':>8s}")
    print("-" * 96)
    Xv, Gv = X[viol], G[viol]
    Tcat = {}
    for cat in usable:
        y = np.array([cat in cats_of(clips[ids[i]]) for i in viol], float)
        Tcat[cat] = tool_oof(Xv, y, Gv, a.k)
    line = [auc(np.array([cat in cats_of(clips[ids[i]]) for i in viol], int), Tcat[cat]) for cat in usable]
    print(f"{'(none)':26s} {'tool only':14s}" + "".join(f"{x:10.3f}" for x in line) + f"{np.nanmean(line):8.3f}")
    cat_rows["tool only"] = dict(zip(usable, line))
    for mdl, D in models.items():
        M = D.get("mcq")
        if not M:
            continue
        for sysn in ("VLM only", "tool + VLM"):
            line = []
            for cat in usable:
                y = np.array([cat in cats_of(clips[ids[i]]) for i in viol], int)
                v = np.array([M.get(ids[i], {}).get("probs", {}).get(cat, np.nan) for i in viol], float)
                if np.isfinite(v).sum() < 0.5 * len(v):
                    line.append(np.nan)
                    continue
                v = np.nan_to_num(v, nan=np.nanmean(v))
                s = v if sysn == "VLM only" else rank01(v) + rank01(Tcat[cat])
                line.append(auc(y, s))
            print(f"{mdl[:26]:26s} {sysn:14s}" + "".join(f"{x:10.3f}" for x in line) + f"{np.nanmean(line):8.3f}")
            R["models"].setdefault(mdl, {}).setdefault("category", {})[sysn] = dict(zip(usable, line))
    R["tool_only"]["category"] = cat_rows["tool only"]

    # ── 3b. detection: violation vs CLEAN clips ─────────────────────────────
    # Attribution only uses the clips that HAVE a violation, which leaves the
    # clean clips out of every specialist test. This puts them back: can the
    # system tell "something is broken" from "nothing is broken", overall
    # (1 - P(none) for the VLM) and per specialist (P(cat), this violation vs
    # clean clips).
    clean = [i for i, c in enumerate(ids) if clips[c].get("has_violation") is False]
    anyv = np.array([0 if clips[c].get("has_violation") is False else 1 for c in ids])
    if len(clean) >= 10:
        T_any = tool_oof(X, anyv.astype(float), G, a.k)
        print(f"\n{'='*96}\n3b. DETECTION — violation vs CLEAN ({len(viol)} violation, "
              f"{len(clean)} clean)\n{'='*96}")
        print(f"{'model':26s} {'system':14s} {'ANY':>8s}" + "".join(f"{c[:9]:>10s}" for c in usable))
        print("-" * 96)

        def det_line(anyscore, catscore):
            row = [auc(anyv, anyscore)]
            for cat in usable:
                pos = [i for i in viol if cat in cats_of(clips[ids[i]])]
                ii = pos + clean
                yy = np.array([1] * len(pos) + [0] * len(clean))
                row.append(auc(yy, np.asarray(catscore(cat))[ii]))
            return row
        Tc = {cat: tool_oof(X, np.array([1. if cat in cats_of(clips[c]) else 0. for c in ids]), G, a.k)
              for cat in usable}
        line = det_line(T_any, lambda cat: Tc[cat])
        print(f"{'(none)':26s} {'tool only':14s}" + "".join(f"{x:8.3f}" if j == 0 else f"{x:10.3f}" for j, x in enumerate(line)))
        for mdl, D in models.items():
            M = D.get("mcq")
            if not M:
                continue
            pn = np.array([M.get(c, {}).get("probs", {}).get("none", np.nan) for c in ids], float)
            if np.isfinite(pn).sum() < 0.5 * len(pn):
                continue
            pn = np.nan_to_num(pn, nan=np.nanmean(pn))
            pcat = {cat: np.nan_to_num(np.array([M.get(c, {}).get("probs", {}).get(cat, np.nan) for c in ids], float), nan=0.0) for cat in usable}
            for sysn in ("VLM only", "tool + VLM"):
                if sysn == "VLM only":
                    line = det_line(1 - pn, lambda cat: pcat[cat])
                else:
                    line = det_line(rank01(1 - pn) + rank01(T_any), lambda cat: rank01(pcat[cat]) + rank01(Tc[cat]))
                print(f"{mdl[:26]:26s} {sysn:14s}" + "".join(f"{x:8.3f}" if j == 0 else f"{x:10.3f}" for j, x in enumerate(line)))
                R["models"].setdefault(mdl, {}).setdefault("detection", {})[sysn] = line

    # ── 4. action completed ─────────────────────────────────────────────────
    ma = np.isfinite(act)
    print(f"\n{'='*96}\n4. ACTION COMPLETED — AUC vs annotator yes/no ({int(ma.sum())} clips, "
          f"{int(np.nansum(act))} yes)\n{'='*96}")
    print(f"{'(none)':26s} {'tool only':14s} {auc(act[ma].astype(int), T_act[ma]):7.3f}")
    for mdl, D in models.items():
        P = D.get("plaus", {})
        v = np.array([P.get(c, {}).get("action", np.nan) for c in ids], float)
        if np.isfinite(v[ma]).sum() < 0.5 * ma.sum():
            continue
        v = np.nan_to_num(v, nan=np.nanmean(v))
        for sysn, s in (("VLM only", v), ("tool + VLM", rank01(v) + rank01(T_act))):
            print(f"{mdl[:26]:26s} {sysn:14s} {auc(act[ma].astype(int), s[ma]):7.3f}")

    # ── 5. unobservable: hidden property followed ───────────────────────────
    mh = np.isfinite(hf)
    if mh.sum() > 20:
        yh = (hf[mh] >= 3).astype(int)
        print(f"\n{'='*96}\n5. HIDDEN PROPERTY FOLLOWED (unobservable clips, n={int(mh.sum())}, "
              f"{int(yh.sum())} rated 3-4)\n{'='*96}")
        print(f"{'model':26s} {'system':22s} {'rho':>7s} {'AUC':>7s}")
        print("-" * 96)
        print(f"{'(none)':26s} {'tool only':22s} {spear(T_hid[mh], hf[mh]):7.3f} {auc(yh, T_hid[mh]):7.3f}")
        for mdl, D in models.items():
            P = D.get("plaus", {})
            for key, lab in (("hidden", "VLM + hint"), ("hidden_nohint", "VLM, no hint")):
                v = np.array([P.get(c, {}).get(key, np.nan) for c in ids], float)[mh]
                if np.isfinite(v).sum() < 0.5 * len(v):
                    continue
                v = np.nan_to_num(v, nan=np.nanmean(v))
                print(f"{mdl[:26]:26s} {lab:22s} {spear(v, hf[mh]):7.3f} {auc(yh, v):7.3f}")
                if key == "hidden":
                    fz = rank01(v) + rank01(T_hid[mh])
                    print(f"{mdl[:26]:26s} {'tool + VLM + hint':22s} {spear(fz, hf[mh]):7.3f} {auc(yh, fz):7.3f}")

    # ── 6. paired obs/unobs ─────────────────────────────────────────────────
    idx = {c: i for i, c in enumerate(ids)}
    pairs = {}
    for c in ids:
        pairs.setdefault(clips[c]["pair_id"], {})[clips[c]["observability"]] = c
    pairs = {k: v for k, v in pairs.items() if len(v) == 2}
    if pairs:
        print(f"\n{'='*96}\n6. PAIRED obs vs unobs ({len(pairs)} pairs, same source frame & "
              f"generator)\n{'='*96}")
        dh = np.array([clips[p["observable"]]["pc"] - clips[p["unobservable"]]["pc"] for p in pairs.values()], float)
        print(f"human: obs rated higher in {int((dh>0).sum())}, lower in {int((dh<0).sum())}, "
              f"tied {int((dh==0).sum())}   (mean obs-unobs = {dh.mean():+.2f})")
        print(f"{'model':26s} {'system':14s} {'rho(dScore,dHuman)':>19s} {'pair order acc':>15s}")
        print("-" * 96)

        def paired(s):
            ds = np.array([s[idx[p["observable"]]] - s[idx[p["unobservable"]]] for p in pairs.values()])
            nz = dh != 0
            accy = np.mean(np.sign(ds[nz]) == np.sign(dh[nz])) if nz.any() else np.nan
            return spear(ds, dh), accy
        r, acc = paired(T_plaus)
        print(f"{'(none)':26s} {'tool only':14s} {r:19.3f} {acc:15.1%}")
        for mdl, D in models.items():
            P = D.get("plaus", {})
            v = np.array([P.get(c, {}).get("plaus", np.nan) for c in ids], float)
            if np.isfinite(v).sum() < 0.5 * len(v):
                continue
            v = np.nan_to_num(v, nan=np.nanmean(v))
            for sysn, s in (("VLM only", v), ("tool + VLM", rank01(v) + rank01(T_plaus))):
                r, acc = paired(s)
                print(f"{mdl[:26]:26s} {sysn:14s} {r:19.3f} {acc:15.1%}")

    out = data / f"system_eval_{a.scope}.json"
    out.write_text(json.dumps(R, indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
