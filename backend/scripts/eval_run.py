"""
Run the full evaluation on a staged dataset, then report.

One command. It executes every scorer in dependency order, runs the controls that
have repeatedly caught us out, and produces the held-out tables.

WHAT IT RUNS AND WHY EACH IS HERE
---------------------------------
  clip-level channels
    vlm        gemma4-31b-it, 1-5 likert question              (1 API call/clip)
    retrieval  DINOv2 k-NN over labelled clips                 (GPU, no API)
    temporal   d1_max, largest frame-to-frame DINOv2 change    (GPU, no API)
    These three fuse to +0.383 vs +0.309 for the VLM alone, because they are
    mechanistically different and so make different mistakes. Adding more VLM
    judges instead gained nothing (they agree with each other at 0.43 and with
    humans at 0.18).

  specialists  one fixed question per category, gemma4-31b-it  (1 API call each)
    Fixed by design, NOT chosen by search. Automatic per-category config search
    scored 0.618 where always-gemma4 scored 0.637 — with ~100 positives per
    category, search fits noise.

  controls     shuffled frames, caption removed
    Not optional. The shuffle control is what proved our best question ignores
    frame order despite being worded as a cross-frame comparison. Any claim
    about temporal reasoning is unsafe without it.

METHODOLOGY THIS ENFORCES
-------------------------
  * a selection split and a held-out split, fixed before anything runs
  * every reported number computed on held-out clips only
  * selection bias measured and printed, not assumed away
Cross-validated numbers on the same clips used to pick the config overstated our
specialist AUCs by +0.140. The split is the only defence.

Usage:
  python backend/scripts/eval_run.py --data data/newbench
  python backend/scripts/eval_run.py --data data/newbench --stages clip,specialists
  python backend/scripts/eval_run.py --data data/newbench --dry-run
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SC = Path(__file__).parent

STAGES = {
    "embed": [
        ("DINOv2 clip embeddings + retrieval channel", "method_retrieval.json",
         [sys.executable, str(SC / "retrieval_memory.py"), "--data", "{data}"]),
        ("DINOv2 per-frame -> temporal channel", "temporal_embed.json",
         [sys.executable, str(SC / "temporal_embed.py"), "--data", "{data}"]),
    ],
    "clip": [
        ("VLM likert judge (clip-level score)",
         "scores_gemma4-31b-it_likert_temporal.json",
         [sys.executable, str(SC / "videophy_eval.py"), "--data", "{data}",
          "--prompt", "likert", "--workers", "{workers}"]),
    ],
    "taxonomy": [
        ("map rule text -> specialist categories", "rule_categories.json",
         [sys.executable, str(SC / "rule_taxonomy.py"), "--data", "{data}"]),
    ],
    "specialists": [
        ("specialist questions (7 categories)",
         "domain_probes_winners.json",
         [sys.executable, str(SC / "domain_probes.py"), "--data", "{data}",
          "--cats", "winners", "--workers", "{workers}"]),
    ],
    "controls": [
        ("CONTROL: shuffled frames", "domain_probes_winners__shuffled.json",
         [sys.executable, str(SC / "domain_probes.py"), "--data", "{data}",
          "--cats", "winners", "--order", "shuffled", "--workers", "{workers}"]),
        ("CONTROL: caption removed", "domain_probes_winners__nocap.json",
         [sys.executable, str(SC / "domain_probes.py"), "--data", "{data}",
          "--cats", "winners", "--no-caption", "--workers", "{workers}"]),
    ],
    "report": [
        ("held-out config card + confusion + operating points", "meeting_report.json",
         [sys.executable, str(SC / "meeting_report.py"), "--data", "{data}",
          "--sel", "{sel}"]),
        ("control ablations", "ablation_report.json",
         [sys.executable, str(SC / "ablation_report.py"), "--data", "{data}",
          "--sel", "{sel}"]),
    ],
}
ORDER = ["embed", "clip", "taxonomy", "specialists", "controls", "report"]


def make_split(data: Path, frac: float, seed: int):
    """Fix the selection / held-out split ONCE and write it to disk.

    Written as its own file so every downstream script reads the same split and
    it cannot drift between runs. Stratified by rating so both halves span the
    full quality range — an unstratified split can hand one side a narrow range,
    which both depresses correlation and makes selection noisy.
    """
    import random
    from collections import defaultdict
    man = json.loads((data / "manifest.json").read_text())
    by = defaultdict(list)
    for c in man["clips"]:
        by[c["pc"]].append(c["clip_id"])
    rnd = random.Random(seed)
    sel = []
    for pc in sorted(by):
        v = by[pc][:]
        rnd.shuffle(v)
        sel += v[:max(1, int(round(len(v) * frac)))]
    out = {"seed": seed, "frac": frac, "n_total": len(man["clips"]),
           "n_selection": len(sel), "selection_ids": sorted(sel)}
    (data / "split.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sel", default=None,
                    help="separate selection dataset; default = split --data internally")
    ap.add_argument("--stages", default=",".join(ORDER))
    ap.add_argument("--workers", type=int, default=3,
                    help="gateway concurrency; >6 total triggers 503 load shedding")
    ap.add_argument("--split-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run steps whose output exists")
    a = ap.parse_args()

    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    if not (data / "manifest.json").exists():
        sys.exit(f"ERROR: {data}/manifest.json missing — run eval_prepare.py first")
    man = json.loads((data / "manifest.json").read_text())
    n = len(man["clips"])
    n_rules = sum(1 for c in man["clips"] if len(c.get("violated_rules") or "") > 4)

    print(f"dataset: {data.name} | {n} clips | {n_rules} with rule text")
    if n_rules < 50:
        print("  WARNING: too few rule annotations for per-specialist evaluation.")
        print("           Clip-level scoring will still work.")

    if not (data / "split.json").exists() or a.force:
        sp = make_split(data, a.split_frac, a.seed)
    else:
        sp = json.loads((data / "split.json").read_text())
    print(f"split: {sp['n_selection']} selection / "
          f"{sp['n_total'] - sp['n_selection']} HELD-OUT (fixed, seed {sp['seed']})")

    want = [s.strip() for s in a.stages.split(",") if s.strip() in STAGES]
    plan = []
    for st in ORDER:
        if st not in want:
            continue
        for label, produces, cmd in STAGES[st]:
            done = bool(produces) and (data / produces).exists() and not a.force
            plan.append((st, label, produces, cmd, done))

    print(f"\n{'stage':12s} {'step':52s} {'status'}")
    print("-" * 88)
    for st, label, produces, cmd, done in plan:
        print(f"{st:12s} {label[:52]:52s} {'SKIP (exists)' if done else 'run'}")
    if a.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    sel_arg = a.sel or a.data
    t0 = time.time()
    failed = []
    for st, label, produces, cmd, done in plan:
        if done:
            continue
        c = [x.replace("{data}", a.data).replace("{workers}", str(a.workers))
              .replace("{sel}", sel_arg) for x in cmd]
        print(f"\n{'='*88}\n[{st}] {label}\n{'='*88}", flush=True)
        r = subprocess.run(c, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"  !! FAILED (exit {r.returncode}) — continuing", file=sys.stderr)
            failed.append(label)

    print(f"\n{'='*88}")
    print(f"finished in {(time.time()-t0)/60:.1f} min"
          + (f" — {len(failed)} step(s) failed: {failed}" if failed else " — all steps ok"))
    print(f"results in {data}/  (meeting_report.json, ablation_report.json)")


if __name__ == "__main__":
    main()
