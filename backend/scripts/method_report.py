"""
Held-out comparison of every strategy tried for improving the physics judge,
plus fusion over them.

Everything is reported on the SAME 160-clip test split, paired against the same
baseline (the hand-written likert prompt), so the numbers are directly
comparable. Anything requiring a fit uses TRAIN ONLY; val is available for
selection; test is never fit on.

Conditions:
  baseline            hand-written likert prompt              (prior best)
  gepa / random       best evolved prompt from each arm
  probes              6 defect probes, unweighted mean
  probes_learned      6 defect probes, weights fit on train
  fewshot             4 labeled train exemplars in context
  pairwise            mean P(worse) vs 6 train anchors
  rankfuse_models     mean rank over the 7 existing model/prompt judges,
                      UNSUPERVISED — uses no labels, so it can be computed
                      directly on test with no leakage
  fuse_learned        z-scored linear combination over every available signal,
                      weights fit on train

MULTIPLE COMPARISONS: ~9 conditions are compared against one baseline. Nominal
95% intervals are therefore optimistic — with 9 comparisons the family-wise
threshold is roughly a 99.4% interval. Both are printed; the Bonferroni column
is the one to believe when claiming a win.

Usage:
  python backend/scripts/method_report.py
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman                          # noqa: E402
from videophy_eval import auc_extremes                          # noqa: E402
from gepa_optimize import load, split_clips                     # noqa: E402

# The 7 saved temporal-order judges (shuffled runs excluded — they are the
# ablation, not candidate judges).
MODEL_RUNS = [
    "scores_gemma4-31b-it_binary_temporal.json",
    "scores_gemma4-31b-it_caption_temporal.json",
    "scores_gemma4-31b-it_likert_temporal.json",
    "scores_internvl3-8b_binary_temporal.json",
    "scores_internvl3-8b_likert_temporal.json",
    "scores_internvl3-14b_binary_temporal.json",
    "scores_qwen2.5-vl-7b_binary_temporal.json",
]


def rank01(vals):
    """Average-rank transform to [0,1], NaN-safe for None entries.

    Fusing raw scores would let one judge dominate purely by having a wider
    numeric range — gemma4 sits near 0.82 and InternVL3-8B near 0.098 on the
    same clips. Ranks put every judge on the same footing.
    """
    idx = [i for i, v in enumerate(vals) if v is not None]
    out = [None] * len(vals)
    order = sorted(idx, key=lambda i: vals[i])
    n = len(order)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = r / max(n - 1, 1)
        i = j + 1
    return out


def zfit(vals):
    v = [x for x in vals if x is not None]
    m, s = float(np.mean(v)), float(np.std(v))
    return m, (s if s > 1e-9 else 1.0)


def zapply(vals, m, s):
    return [None if x is None else (x - m) / s for x in vals]


def boot_ci_rho(xs, ys, n=4000, seed=0, alpha=0.05):
    rnd = random.Random(seed)
    m = len(xs)
    out = []
    for _ in range(n):
        k = [rnd.randrange(m) for _ in range(m)]
        r = spearman([xs[i] for i in k], [ys[i] for i in k])
        if r == r:
            out.append(r)
    if len(out) < n * 0.5:
        return None, None
    out.sort()
    return out[int(alpha / 2 * len(out))], out[int((1 - alpha / 2) * len(out))]


def paired_boot(xa, xb, ys, n=4000, seed=0, alpha=0.05):
    rnd = random.Random(seed)
    m = len(ys)
    out = []
    for _ in range(n):
        k = [rnd.randrange(m) for _ in range(m)]
        ra = spearman([xa[i] for i in k], [ys[i] for i in k])
        rb = spearman([xb[i] for i in k], [ys[i] for i in k])
        if ra == ra and rb == rb:
            out.append(ra - rb)
    if len(out) < n * 0.5:
        return None, None
    out.sort()
    return out[int(alpha / 2 * len(out))], out[int((1 - alpha / 2) * len(out))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy300")
    ap.add_argument("--n-train", type=int, default=90)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data, clips = load(a.data)
    train, val, test = split_clips(clips, a.seed, a.n_train, a.n_val)
    tr_ids = [c["clip_id"] for c in train]
    te_ids = [c["clip_id"] for c in test]
    tr_y = [6 - c["pc"] for c in train]
    te_y = [6 - c["pc"] for c in test]
    te_pc = [c["pc"] for c in test]

    def get(d, ids):
        return [d.get(i) for i in ids]

    conds = {}       # name -> (train_scores, test_scores)
    bname = "baseline (likert prompt)"

    base = json.loads(
        (data / "scores_gemma4-31b-it_likert_temporal.json").read_text())["scores"]
    conds[bname] = (get(base, tr_ids), get(base, te_ids))

    for mode in ("gepa", "random"):
        f = data / "gepa_final.json"
        if f.exists():
            d = json.loads(f.read_text())["conditions"]
            k = f"{mode} (evolved)"
            if k in d:
                s = d[k]["scores"]
                conds[f"{mode} evolved prompt"] = (None, get(s, te_ids))

    probe_names, probe_tr, probe_te = [], [], []
    for meth in ("probes", "fewshot", "pairwise"):
        f = data / f"method_{meth}.json"
        if not f.exists():
            print(f"  (skip {meth}: not run yet)")
            continue
        d = json.loads(f.read_text())
        if d["n"] < 100:
            print(f"  (skip {meth}: only {d['n']} clips — smoke run, not full)")
            continue
        s = d["scores"]
        conds[{"probes": "probes (6 defects, mean)",
               "fewshot": "fewshot (4 exemplars)",
               "pairwise": "pairwise (6 anchors)"}[meth]] = (get(s, tr_ids), get(s, te_ids))
        if meth == "probes":
            ps = d.get("probe_scores", {})
            probe_names = [n for n, _ in __import__("physics_probes").PROBES]
            for nm in probe_names:
                probe_tr.append([(ps.get(i) or {}).get(nm) for i in tr_ids])
                probe_te.append([(ps.get(i) or {}).get(nm) for i in te_ids])

    # ── learned probe weighting (fit on train only) ───────────────────────────
    if probe_tr:
        keep = [k for k in range(len(tr_ids))
                if all(p[k] is not None for p in probe_tr)]
        X = np.array([[p[k] for p in probe_tr] for k in keep])
        y = np.array([tr_y[k] for k in keep], dtype=float)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
        # Ridge, not OLS: 6 correlated probes on ~90 rows overfits badly under
        # OLS and the weights flip sign between splits.
        w = np.linalg.solve(Xz.T @ Xz + 5.0 * np.eye(Xz.shape[1]), Xz.T @ (y - y.mean()))
        print("\nlearned probe weights (train, ridge):")
        for nm, wi in sorted(zip(probe_names, w), key=lambda t: -abs(t[1])):
            print(f"    {nm:12s} {wi:+.3f}")

        def apply_probes(cols):
            out = []
            for k in range(len(cols[0])):
                v = [c[k] for c in cols]
                if any(x is None for x in v):
                    out.append(None)
                else:
                    z = (np.array(v) - X.mean(0)) / (X.std(0) + 1e-9)
                    out.append(float(z @ w))
            return out
        conds["probes (learned weights)"] = (apply_probes(probe_tr),
                                             apply_probes(probe_te))

    # ── non-VLM motion channel ────────────────────────────────────────────────
    mot_tr, mot_te, mot_names = [], [], []
    fmot = data / "method_motion.json"
    if fmot.exists():
        ms = json.loads(fmot.read_text())["signals"]
        mot_names = sorted(next(iter(ms.values())).keys())
        for nm in mot_names:
            mot_tr.append([(ms.get(i) or {}).get(nm) for i in tr_ids])
            mot_te.append([(ms.get(i) or {}).get(nm) for i in te_ids])

        # Pick the single best motion channel ON TRAIN. Choosing it by its test
        # correlation would be selecting on the held-out split and would inflate
        # whatever it is then used for.
        univ = []
        for j, nm in enumerate(mot_names):
            k = [i for i in range(len(tr_ids)) if mot_tr[j][i] is not None]
            univ.append((abs(spearman([mot_tr[j][i] for i in k],
                                      [tr_y[i] for i in k])), nm, j))
        univ.sort(reverse=True)
        print("\nmotion channels by |rho| on TRAIN:")
        for r, nm, _ in univ:
            print(f"    {nm:16s} {r:.3f}")
        bt, bnm, bj = univ[0]
        conds[f"motion: {bnm} (best on train)"] = (mot_tr[bj], mot_te[bj])

        # Unsupervised 2-channel fusion: no weights are fit, so there is nothing
        # to overfit — the only choice made was which motion channel, made on
        # train. This is the cleanest test of "do the channels complement".
        vlm_tr, vlm_te = conds[bname]

        def rankmix(a, b):
            ra, rb = rank01(a), rank01(b)
            return [None if (ra[i] is None or rb[i] is None)
                    else (ra[i] + rb[i]) / 2 for i in range(len(a))]
        conds[f"VLM + {bnm} (rank mean, unsup.)"] = (rankmix(vlm_tr, mot_tr[bj]),
                                                     rankmix(vlm_te, mot_te[bj]))

    # ── unsupervised rank fusion over existing judges (no labels used) ────────
    judges_tr, judges_te, jn = [], [], []
    for f in MODEL_RUNS:
        p = data / f
        if not p.exists():
            continue
        s = json.loads(p.read_text())["scores"]
        judges_tr.append(get(s, tr_ids))
        judges_te.append(get(s, te_ids))
        jn.append(f.replace("scores_", "").replace("_temporal.json", ""))
    if len(judges_te) >= 2:
        def meanrank(cols):
            rk = [rank01(c) for c in cols]
            out = []
            for k in range(len(cols[0])):
                v = [r[k] for r in rk if r[k] is not None]
                out.append(float(np.mean(v)) if v else None)
            return out
        conds[f"rankfuse {len(jn)} judges (unsup.)"] = (meanrank(judges_tr),
                                                        meanrank(judges_te))

    # ── learned fusion over every available signal (fit on train) ─────────────
    sig_tr, sig_te, sn = list(judges_tr), list(judges_te), list(jn)
    for nm in ("probes (6 defects, mean)", "fewshot (4 exemplars)",
               "pairwise (6 anchors)"):
        if nm in conds and conds[nm][0] is not None:
            sig_tr.append(conds[nm][0])
            sig_te.append(conds[nm][1])
            sn.append(nm.split(" ")[0])
    for j, nm in enumerate(mot_names):
        sig_tr.append(mot_tr[j])
        sig_te.append(mot_te[j])
        sn.append(f"motion:{nm}")
    if len(sig_te) >= 3:
        keep = [k for k in range(len(tr_ids))
                if all(c[k] is not None for c in sig_tr)]
        if len(keep) > 20:
            stats = [zfit([c[k] for k in keep]) for c in sig_tr]
            X = np.array([[(sig_tr[j][k] - stats[j][0]) / stats[j][1]
                           for j in range(len(sig_tr))] for k in keep])
            y = np.array([tr_y[k] for k in keep], dtype=float)
            w = np.linalg.solve(X.T @ X + 10.0 * np.eye(X.shape[1]),
                                X.T @ (y - y.mean()))
            print(f"\nlearned fusion weights over {len(sn)} signals (train, ridge):")
            for nm, wi in sorted(zip(sn, w), key=lambda t: -abs(t[1])):
                print(f"    {nm:34s} {wi:+.3f}")

            def apply_fuse(cols):
                out = []
                for k in range(len(cols[0])):
                    v = [c[k] for c in cols]
                    if any(x is None for x in v):
                        out.append(None)
                    else:
                        z = np.array([(v[j] - stats[j][0]) / stats[j][1]
                                      for j in range(len(v))])
                        out.append(float(z @ w))
                return out
            conds[f"fuse_learned ({len(sn)} signals)"] = (apply_fuse(sig_tr),
                                                          apply_fuse(sig_te))

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\nheld-out test: {len(test)} clips\n")
    print(f"{'condition':34s} {'n':>4s} {'rho':>8s} {'95% CI':>18s} {'AUC':>7s} {'distinct':>9s}")
    print("-" * 88)
    rows = {}
    for nm, (_, te) in conds.items():
        keep = [i for i, v in enumerate(te) if v is not None]
        xs = [te[i] for i in keep]
        ys = [te_y[i] for i in keep]
        r = spearman(xs, ys)
        lo, hi = boot_ci_rho(xs, ys)
        auc, _, _ = auc_extremes(xs, [te_pc[i] for i in keep])
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "-"
        print(f"{nm:34s} {len(xs):4d} {r:+8.3f} {ci:>18s} {auc or 0:7.3f} "
              f"{len(set(xs)):9d}")
        rows[nm] = {"n": len(xs), "rho": round(r, 3), "auc": round(auc, 3) if auc else None,
                    "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
                    "distinct": len(set(xs))}

    bname = "baseline (likert prompt)"
    others = [n for n in conds if n != bname]
    kfam = max(len(others), 1)
    alpha_b = 0.05 / kfam
    print(f"\n{'paired vs baseline (same clips)':34s} {'delta':>8s} {'95% CI':>18s}"
          f" {'Bonf %.1f%% CI' % (100*(1-alpha_b)):>20s}")
    print("-" * 88)
    stats_out = {}
    for nm in others:
        te_a, te_b = conds[nm][1], conds[bname][1]
        keep = [i for i in range(len(te_ids))
                if te_a[i] is not None and te_b[i] is not None]
        xa = [te_a[i] for i in keep]
        xb = [te_b[i] for i in keep]
        ys = [te_y[i] for i in keep]
        d = spearman(xa, ys) - spearman(xb, ys)
        lo, hi = paired_boot(xa, xb, ys)
        blo, bhi = paired_boot(xa, xb, ys, alpha=alpha_b)
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "-"
        bci = f"[{blo:+.2f}, {bhi:+.2f}]" if blo is not None else "-"
        mark = "  **" if (blo is not None and blo > 0) else (
            "  *" if (lo is not None and lo > 0) else "")
        print(f"{nm:34s} {d:+8.3f} {ci:>18s} {bci:>20s}{mark}")
        stats_out[nm] = {"delta": round(d, 3),
                         "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
                         "ci_bonf": [round(blo, 3), round(bhi, 3)] if blo is not None else None,
                         "sig_nominal": bool(lo is not None and lo > 0),
                         "sig_bonferroni": bool(blo is not None and blo > 0)}
    print("\n  *  beats baseline at nominal 95%")
    print("  ** survives Bonferroni correction for %d comparisons" % kfam)

    outp = data / "method_report.json"
    outp.write_text(json.dumps({"n_test": len(test), "conditions": rows,
                                "paired_vs_baseline": stats_out}, indent=1))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
