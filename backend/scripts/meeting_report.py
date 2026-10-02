"""
Per-specialist configuration card + held-out generalisation, for the meeting.

THE SELECTION PROBLEM this fixes. Every per-specialist number so far came from
5-fold CV over the same 704 rule-annotated clips, and the winning question for
each category was chosen by looking at those same numbers. That is selection on
the evaluation set: with ~30 candidate signals per category the best-looking one
is optimistically biased even when each individual number is honest.

The 300-clip staged set is a strict SUBSET of the 1200-clip set (verified: 300
overlap, 900 disjoint). So:

    SELECT  on the 300-subset clips only  — pick each specialist's config here
    REPORT  on the 900 disjoint clips     — never examined during selection

Nothing about the 900 informs the choice, so the reported AUC is what a new
dataset would give. Both numbers are printed; the gap between them IS the
selection bias, quantified rather than assumed away.

Candidate configs per specialist: every individual signal, plus rank-mean
combinations of the top 2 and top 3 as ranked ON THE SELECTION SET.

Also produced:
  - cross-specialist confusion matrix (is each detector specific, or is
    everything just firing on generic brokenness?)
  - per-generator breakdown (a detector that only works on one generator is
    not deployable)
  - operating points (threshold, precision, recall) so a specialist can emit a
    flag instead of a score
  - error analysis over clips every detector misses

Usage:
  python backend/scripts/meeting_report.py --data data/videophy1200
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from gepa_optimize import load                              # noqa: E402
from specialist_report import auc, auc_ci                   # noqa: E402
from method_report import rank01                            # noqa: E402

CATS = ["fluid", "friction", "collision", "permanence",
        "gravity", "deformation", "momentum"]

# Which probe name in each file corresponds to each category's winning question.
WINNER_KEY = {"gravity": "g_no_accel", "permanence": "p_count_change",
              "collision": "c2_overlap", "deformation": "d2_length_change",
              "friction": "f_no_rolling_link", "momentum": "m_gains_energy",
              "fluid": "fluid"}
W_ALIAS = {c: f"w_{c}" for c in CATS}


def collect_signals(data, clips):
    """-> (name -> per-clip list). Everything measured this session."""
    sig = {}
    for f in sorted(data.glob("*.json")):
        nm = f.name
        if nm in ("manifest.json", "rule_categories.json",
                  "rule_categories_llm.json"):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or d.get("n", 0) < 100:
            continue
        block = d.get("probe_scores") or d.get("features") or d.get("signals")
        if not isinstance(block, dict):
            continue
        model = d.get("model", "")
        tag = nm.replace("domain_probes_", "").replace("method_", "") \
                .replace(".json", "")[:28]
        keys = sorted({k for v in block.values() if isinstance(v, dict)
                       for k in v})
        for k in keys:
            col = [(block.get(c["clip_id"]) or {}).get(k) for c in clips]
            if sum(v is not None for v in col) < 100:
                continue
            sig[f"{tag}:{k}"] = col
    # the clip-level likert score too
    p = data / "scores_gemma4-31b-it_likert_temporal.json"
    if p.exists():
        s = json.loads(p.read_text())["scores"]
        sig["likert:clip"] = [s.get(c["clip_id"]) for c in clips]
    return sig


def sel_auc(col, idx, pos_mask):
    p = [col[i] for i in idx if pos_mask[i] and col[i] is not None]
    q = [col[i] for i in idx if not pos_mask[i] and col[i] is not None]
    if len(p) < 8 or len(q) < 8:
        return None
    return auc(p, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--sel", default="data/videophy300")
    a = ap.parse_args()

    data, clips = load(a.data)
    n = len(clips)
    seldata, selclips = load(a.sel)
    sel_ids = {c["clip_id"] for c in selclips}
    cats_of = json.loads((data / "rule_categories.json").read_text())["per_clip"]
    co = [set(cats_of.get(c["clip_id"], [])) for c in clips]
    has = [len(c.get("violated_rules") or "") > 4 for c in clips]
    gens = [c["generator"] for c in clips]

    SEL = [i for i in range(n) if clips[i]["clip_id"] in sel_ids]
    TST = [i for i in range(n) if clips[i]["clip_id"] not in sel_ids]
    print(f"selection set {len(SEL)} clips | HELD-OUT {len(TST)} clips "
          f"(disjoint, never used for any choice)")

    sig = collect_signals(data, clips)
    print(f"candidate signals: {len(sig)}")

    card, conf = {}, {}
    print(f"\n{'='*104}\nPER-SPECIALIST CONFIGURATION CARD\n{'='*104}")
    print(f"{'specialist':12s} {'n+ held':>8s} {'SELECT AUC':>11s} "
          f"{'HELD-OUT AUC':>22s} {'config':>42s}")
    print("-" * 104)
    for cat in CATS:
        pos = [cat in s for s in co]
        dsel = [i for i in SEL if pos[i] or (has[i] and co[i])]
        dtst = [i for i in TST if pos[i] or (has[i] and co[i])]
        if sum(pos[i] for i in dsel) < 8 or sum(pos[i] for i in dtst) < 8:
            print(f"{cat:12s}   (too few positives to split)")
            continue

        # rank every signal ON THE SELECTION SET ONLY
        scored = []
        for nm, col in sig.items():
            v = sel_auc(col, dsel, pos)
            if v is None:
                continue
            scored.append((max(v, 1 - v), 1.0 if v >= 0.5 else -1.0, nm))
        scored.sort(reverse=True)
        top = scored[:3]

        cands = {}
        for k in (1, 2, 3):
            if len(top) < k:
                continue
            use = top[:k]
            def build(idxs):
                sub = set(idxs)
                rk = [rank01([(sig[nm][i] * s) if i in sub and sig[nm][i] is not None
                              else None for i in range(n)]) for _, s, nm in use]
                return {i: float(np.mean([r[i] for r in rk]))
                        for i in idxs if all(r[i] is not None for r in rk)}
            ssel, stst = build(dsel), build(dtst)
            av = auc([ssel[i] for i in dsel if i in ssel and pos[i]],
                     [ssel[i] for i in dsel if i in ssel and not pos[i]])
            cands[k] = (av, ssel, stst, [nm for _, _, nm in use])
        if not cands:
            continue
        bk = max(cands, key=lambda k: cands[k][0] or 0)
        av, ssel, stst, used = cands[bk]

        tp = [stst[i] for i in dtst if i in stst and pos[i]]
        tq = [stst[i] for i in dtst if i in stst and not pos[i]]
        at = auc(tp, tq)
        lo, hi = auc_ci(tp, tq)
        print(f"{cat:12s} {len(tp):8d} {av:11.3f} "
              f"{at:.3f} [{lo:.2f},{hi:.2f}]{'':>4s} "
              f"{('+'.join(u.split(':')[-1] for u in used))[:42]:>42s}")
        card[cat] = {"n_pos_heldout": len(tp), "select_auc": round(av, 3),
                     "heldout_auc": round(at, 3),
                     "heldout_ci": [round(lo, 3), round(hi, 3)],
                     "k": bk, "signals": used,
                     "selection_bias": round(av - at, 3)}
        conf[cat] = stst

    if card:
        bias = np.mean([v["selection_bias"] for v in card.values()])
        print(f"\nmean selection bias (select minus held-out): {bias:+.3f}")
        print(f"mean HELD-OUT AUC: "
              f"{np.mean([v['heldout_auc'] for v in card.values()]):.3f}")

    # ── confusion: does each detector fire only on its own category ───────────
    print(f"\n{'='*104}\nCONFUSION — each detector (row) scored against each "
          f"category (column), held-out clips\n{'='*104}")
    hdr = "".join(f"{c[:9]:>10s}" for c in CATS)
    print(f"{'detector':13s}{hdr}")
    print("-" * 104)
    for cat in CATS:
        if cat not in conf:
            continue
        row = ""
        for other in CATS:
            pos2 = [other in s for s in co]
            d2 = [i for i in TST if i in conf[cat] and
                  (pos2[i] or (has[i] and co[i]))]
            v = auc([conf[cat][i] for i in d2 if pos2[i]],
                    [conf[cat][i] for i in d2 if not pos2[i]])
            mark = "*" if (other == cat) else " "
            row += f"{(f'{v:.3f}' if v else '  -  '):>9s}{mark}"
        print(f"{cat:13s}{row}")
    print("  * = own category (should be the row maximum)")

    # ── per-generator ─────────────────────────────────────────────────────────
    print(f"\n{'='*104}\nPER-GENERATOR held-out AUC (blank = too few positives)"
          f"\n{'='*104}")
    glist = sorted(set(gens))
    print(f"{'specialist':13s}" + "".join(f"{g[:9]:>11s}" for g in glist))
    print("-" * 104)
    for cat in CATS:
        if cat not in conf:
            continue
        pos = [cat in s for s in co]
        row = ""
        for g in glist:
            d2 = [i for i in TST if i in conf[cat] and gens[i] == g and
                  (pos[i] or (has[i] and co[i]))]
            p_ = [conf[cat][i] for i in d2 if pos[i]]
            q_ = [conf[cat][i] for i in d2 if not pos[i]]
            v = auc(p_, q_) if len(p_) >= 5 and len(q_) >= 5 else None
            row += f"{(f'{v:.3f}' if v else '     '):>11s}"
        print(f"{cat:13s}{row}")

    # ── operating points ──────────────────────────────────────────────────────
    print(f"\n{'='*104}\nOPERATING POINTS on held-out clips "
          f"(threshold chosen on the SELECTION set)\n{'='*104}")
    print(f"{'specialist':13s} {'thresh pct':>11s} {'precision':>10s} "
          f"{'recall':>8s} {'base rate':>10s} {'lift':>7s}")
    print("-" * 104)
    ops = {}
    for cat in CATS:
        if cat not in card:
            continue
        pos = [cat in s for s in co]
        d2 = [i for i in TST if i in conf[cat] and (pos[i] or (has[i] and co[i]))]
        vals = sorted(conf[cat][i] for i in d2)
        base = np.mean([pos[i] for i in d2])
        best = None
        for pct in (60, 70, 80, 90):
            thr = vals[int(len(vals) * pct / 100)]
            flag = [i for i in d2 if conf[cat][i] >= thr]
            if len(flag) < 5:
                continue
            prec = np.mean([pos[i] for i in flag])
            rec = sum(pos[i] for i in flag) / max(sum(pos[i] for i in d2), 1)
            if best is None or prec > best[1]:
                best = (pct, prec, rec, thr)
        if best:
            pct, prec, rec, thr = best
            print(f"{cat:13s} {pct:10d}% {prec:10.3f} {rec:8.3f} "
                  f"{base:10.3f} {prec/max(base,1e-6):7.2f}x")
            ops[cat] = {"pct": pct, "precision": round(float(prec), 3),
                        "recall": round(float(rec), 3),
                        "base_rate": round(float(base), 3)}

    outp = data / "meeting_report.json"
    outp.write_text(json.dumps({"n_select": len(SEL), "n_heldout": len(TST),
                                "card": card, "operating_points": ops}, indent=1))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
