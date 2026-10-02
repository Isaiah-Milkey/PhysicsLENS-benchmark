"""
Does val rho predict test rho? — a diagnostic on the SELECTION step, not a
second bite at model selection.

Motivation: GEPA lifted val rho from +0.265 to +0.450 (n_val=49) but only
+0.264 -> +0.281 on the 160-clip held-out test. Either (a) the evolved prompts
are genuinely no better and val was measuring noise, or (b) some candidate in
the pool IS better and picking-the-val-max chose the wrong one. Those have
opposite fixes — (a) needs a bigger val set, (b) needs a better selection rule
— so it is worth resolving.

Method: score EVERY candidate in the pool on the held-out clips and correlate
val rho against test rho across candidates. A near-zero correlation means the
val set at this size cannot rank prompts, and every "improvement" the optimiser
promoted was selection noise.

The reported headline result stays the pre-registered one (best-on-val scored
on test). Nothing here re-selects a winner on test — it measures whether the
selection procedure itself has any predictive validity.

Usage:
  python backend/scripts/gepa_diagnose.py --mode gepa
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman                                  # noqa: E402
from videophy_eval import auc_extremes, client                          # noqa: E402
from gepa_optimize import (TASK_MODEL, Scorer, load, split_clips)       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy300")
    ap.add_argument("--mode", default="gepa", choices=["gepa", "random"])
    ap.add_argument("--n-train", type=int, default=90)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data, clips = load(a.data)
    _, val, test = split_clips(clips, a.seed, a.n_train, a.n_val)
    d = json.loads((data / f"gepa_{a.mode}.json").read_text())
    pool = d["pool"]
    print(f"{a.mode}: {len(pool)} candidates | val {len(val)} | test {len(test)}")

    c = client()
    sc = Scorer(c, data, TASK_MODEL, a.workers)
    ys = [6 - cl["pc"] for cl in test]

    rows = []
    for i, p in enumerate(pool):
        evs = sc.run(p["prompt"], test)
        keep = [k for k, e in enumerate(evs) if e is not None]
        xs = [(5.0 - evs[k]) / 4.0 for k in keep]
        tr = spearman(xs, [ys[k] for k in keep])
        auc, _, _ = auc_extremes(xs, [test[k]["pc"] for k in keep])
        rows.append({"id": p["id"], "iter": p["iter"], "parent": p["parent"],
                     "val_rho": p["val_rho"], "test_rho": round(tr, 3),
                     "test_auc": round(auc, 3) if auc else None,
                     "n_scored": len(keep)})
        print(f"  cand {p['id']:2d} (it{p['iter']:02d}) val {p['val_rho']:+.3f} "
              f"-> test {tr:+.3f}  AUC {auc or 0:.3f}", flush=True)

    v = [r["val_rho"] for r in rows]
    t = [r["test_rho"] for r in rows]
    rank = spearman(v, t)
    best_val = max(rows, key=lambda r: r["val_rho"])
    best_test = max(rows, key=lambda r: r["test_rho"])

    print(f"\n  corr(val rho, test rho) across {len(rows)} candidates: "
          f"rho = {rank:+.3f}")
    print(f"  val range  {min(v):+.3f} .. {max(v):+.3f}   (spread {max(v)-min(v):.3f})")
    print(f"  test range {min(t):+.3f} .. {max(t):+.3f}   (spread {max(t)-min(t):.3f})")
    print(f"  picked-by-val  : cand {best_val['id']} -> test {best_val['test_rho']:+.3f}")
    print(f"  best-on-test   : cand {best_test['id']} -> test {best_test['test_rho']:+.3f}"
          f"  (val {best_test['val_rho']:+.3f})")
    print(f"  regret from selection: "
          f"{best_test['test_rho'] - best_val['test_rho']:+.3f}")
    print(f"  mean test rho across pool: {np.mean(t):+.3f}")

    out = data / f"gepa_diagnose_{a.mode}.json"
    out.write_text(json.dumps(
        {"mode": a.mode, "n_val": len(val), "n_test": len(test),
         "val_test_rank_corr": round(rank, 3),
         "picked_by_val_test_rho": best_val["test_rho"],
         "best_on_test_rho": best_test["test_rho"],
         "selection_regret": round(best_test["test_rho"] - best_val["test_rho"], 3),
         "mean_test_rho": round(float(np.mean(t)), 3), "candidates": rows}, indent=1))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
