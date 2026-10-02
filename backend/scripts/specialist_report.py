"""
Per-specialist ablation: does each defect probe detect ITS OWN physics category,
across several VLMs, and does retrieval-based test-time adaptation help?

Ground truth comes from VideoPhy-2's `human_violated_rules`, mapped onto the
PhysicsLENS specialist categories by rule_taxonomy.py.

TWO DETECTION METRICS, and the gap between them is the whole point:

  AUC vs all       positives = clips whose human rule is in category S
                   negatives = every other clip (mostly clean)
                   A probe that merely detects "something is wrong here" scores
                   well on this without being a specialist at all.

  AUC vs violators positives = same
                   negatives = clips that DO have a violation, just not S
                   This is the specificity test: can the gravity probe tell a
                   gravity failure from a deformation failure? Only this number
                   supports the claim that specialists are doing separate jobs.

CONFUSION: for each category, which probe actually fires hardest. If one probe
wins every category, the probes are reading a single "brokenness" axis and the
specialist decomposition is cosmetic.

TEST-TIME ADAPTATION, per specialist: a k-NN over DINOv2 clip embeddings that
predicts P(category S violated) from neighbours' category labels — 5-fold CV,
same-caption neighbours excluded (VideoPhy-2 renders one caption with several
generators, so without that exclusion the neighbour can be the same prompt and
the model is recalling prompt difficulty, not judging the clip). Then fused with
the probe by rank mean.

Usage:
  python backend/scripts/specialist_report.py --data data/videophy1200
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman                       # noqa: E402
from gepa_optimize import load                               # noqa: E402
from method_report import rank01                             # noqa: E402
from rule_taxonomy import CATEGORIES, PROBE_FOR              # noqa: E402
from retrieval_memory import folds                           # noqa: E402


def auc(pos, neg):
    """P(a positive scores above a negative), ties at 0.5."""
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def auc_ci(pos, neg, n=2000, seed=0):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        p = [pos[rnd.randrange(len(pos))] for _ in pos]
        q = [neg[rnd.randrange(len(neg))] for _ in neg]
        v = auc(p, q)
        if v is not None:
            out.append(v)
    if len(out) < n * 0.5:
        return None, None
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def knn_category(X, lab, tr, te, k=25, exclude=None):
    """P(category violated) = similarity-weighted fraction of neighbours with it."""
    out = {}
    for i in te:
        pool = tr if exclude is None else [j for j in tr if exclude[j] != exclude[i]]
        if len(pool) < 3:
            out[i] = None
            continue
        sims = np.array([float(X[i] @ X[j]) for j in pool])
        top = np.argsort(-sims)[:min(k, len(pool))]
        w = np.clip(sims[top], 0, None) ** 8
        if w.sum() < 1e-8:
            w = np.ones_like(w)
        out[i] = float(np.average([lab[pool[t]] for t in top], weights=w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--k", type=int, default=25)
    a = ap.parse_args()

    data, clips = load(a.data)
    n = len(clips)
    cats = json.loads((data / "rule_categories.json").read_text())["per_clip"]
    caps = [c["caption"] for c in clips]
    has_rule = [len(c.get("violated_rules") or "") > 4 for c in clips]
    cat_of = [set(cats.get(c["clip_id"], [])) for c in clips]

    # which models have probe results
    models = {}
    for f in sorted(data.glob("method_probes*.json")):
        d = json.loads(f.read_text())
        if d.get("n", 0) < 100 or "probe_scores" not in d:
            continue
        models[d.get("model", f.stem)] = d["probe_scores"]
    if not models:
        sys.exit("no probe result files yet")
    print(f"{n} clips | models: {', '.join(models)}")

    order = [c for c in CATEGORIES if PROBE_FOR.get(c)]
    counts = {c: sum(1 for s in cat_of if c in s) for c in order}
    print("positives per specialist:",
          ", ".join(f"{c}={counts[c]}" for c in order))

    # ── per-model, per-specialist detection ───────────────────────────────────
    results = {}
    for mname, ps in models.items():
        print(f"\n{'='*94}\n{mname}\n{'='*94}")
        print(f"{'specialist':13s} {'probe':12s} {'n+':>4s} "
              f"{'AUC vs all':>18s} {'AUC vs violators':>18s} {'best probe':>14s}")
        print("-" * 94)
        results[mname] = {}
        for cat in order:
            pname = PROBE_FOR[cat]
            sc = [(ps.get(c["clip_id"]) or {}).get(pname) for c in clips]
            idx = [i for i in range(n) if sc[i] is not None]
            pos = [sc[i] for i in idx if cat in cat_of[i]]
            neg_all = [sc[i] for i in idx if cat not in cat_of[i]]
            neg_vio = [sc[i] for i in idx
                       if cat not in cat_of[i] and has_rule[i] and cat_of[i]]
            a1 = auc(pos, neg_all)
            a2 = auc(pos, neg_vio)
            l1, h1 = auc_ci(pos, neg_all) if pos and neg_all else (None, None)
            l2, h2 = auc_ci(pos, neg_vio) if pos and neg_vio else (None, None)

            # which probe separates this category best (confusion check)
            best, bestv = None, -1
            for other, _ in [(k, 0) for k in
                             {PROBE_FOR[c] for c in order if PROBE_FOR[c]}]:
                o = [(ps.get(c["clip_id"]) or {}).get(other) for c in clips]
                oi = [i for i in range(n) if o[i] is not None]
                v = auc([o[i] for i in oi if cat in cat_of[i]],
                        [o[i] for i in oi if cat not in cat_of[i] and
                         has_rule[i] and cat_of[i]])
                if v is not None and v > bestv:
                    best, bestv = other, v
            c1 = f"{a1:.3f} [{l1:.2f},{h1:.2f}]" if a1 and l1 else "-"
            c2 = f"{a2:.3f} [{l2:.2f},{h2:.2f}]" if a2 and l2 else "-"
            flag = "" if best == pname else f"  <- {best}"
            print(f"{cat:13s} {pname:12s} {len(pos):4d} {c1:>18s} {c2:>18s} "
                  f"{bestv:8.3f}{flag}")
            results[mname][cat] = {
                "probe": pname, "n_pos": len(pos),
                "auc_vs_all": round(a1, 3) if a1 else None,
                "auc_vs_violators": round(a2, 3) if a2 else None,
                "ci_vs_violators": [round(l2, 3), round(h2, 3)] if l2 else None,
                "best_probe": best, "best_probe_auc": round(bestv, 3)}

        # holistic: probe-mean vs human pc
        mean = []
        for c in clips:
            v = list((ps.get(c["clip_id"]) or {}).values())
            mean.append(float(np.mean(v)) if v else None)
        ok = [i for i in range(n) if mean[i] is not None]
        rho = spearman([mean[i] for i in ok], [6 - clips[i]["pc"] for i in ok])
        print(f"{'holistic':13s} {'probe-mean':12s} {len(ok):4d} "
              f"  rho vs human pc = {rho:+.3f}")
        results[mname]["_holistic_rho"] = round(rho, 3)

    # ── test-time adaptation per specialist ───────────────────────────────────
    embf = data / "emb_dinov2.npy"
    if embf.exists():
        X = np.load(embf)
        F = folds(n, 5)
        print(f"\n{'='*94}\nTEST-TIME ADAPTATION — retrieval per specialist "
              f"(5-fold CV, same-caption neighbours excluded)\n{'='*94}")
        best_model = max(models, key=lambda m: results[m]["_holistic_rho"])
        ps = models[best_model]
        print(f"probe column uses {best_model} (best holistic)\n")
        print(f"{'specialist':13s} {'probe AUC':>11s} {'retrieval AUC':>15s} "
              f"{'FUSED AUC':>12s} {'gain':>8s}")
        print("-" * 94)
        for cat in order:
            lab = [1.0 if cat in s else 0.0 for s in cat_of]
            pred = {}
            for f in range(5):
                te = F[f]
                tr = [i for g in range(5) if g != f for i in F[g]]
                pred.update(knn_category(X, lab, tr, te, a.k, exclude=caps))
            pname = PROBE_FOR[cat]
            sc = [(ps.get(c["clip_id"]) or {}).get(pname) for c in clips]
            idx = [i for i in range(n) if sc[i] is not None and pred.get(i) is not None
                   and (cat in cat_of[i] or (has_rule[i] and cat_of[i]))]
            if len(idx) < 40:
                continue
            s = set(idx)
            rp = rank01([sc[i] if i in s else None for i in range(n)])
            rr = rank01([pred[i] if i in s else None for i in range(n)])
            fu = {i: (rp[i] + rr[i]) / 2 for i in idx}
            def split(d):
                return ([d[i] for i in idx if cat in cat_of[i]],
                        [d[i] for i in idx if cat not in cat_of[i]])
            ap_, an_ = split({i: sc[i] for i in idx})
            rp_, rn_ = split(pred)
            fp_, fn_ = split(fu)
            a1, a2, a3 = auc(ap_, an_), auc(rp_, rn_), auc(fp_, fn_)
            lo, hi = auc_ci(fp_, fn_)
            g = (a3 - a1) if (a3 and a1) else 0
            print(f"{cat:13s} {a1 or 0:11.3f} {a2 or 0:15.3f} "
                  f"{a3 or 0:8.3f} [{lo:.2f},{hi:.2f}] {g:+8.3f}")
            results.setdefault("_tta", {})[cat] = {
                "probe_auc": round(a1, 3) if a1 else None,
                "retrieval_auc": round(a2, 3) if a2 else None,
                "fused_auc": round(a3, 3) if a3 else None,
                "gain": round(g, 3)}

    outp = data / "specialist_report.json"
    outp.write_text(json.dumps({"counts": counts, "results": results}, indent=1))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
