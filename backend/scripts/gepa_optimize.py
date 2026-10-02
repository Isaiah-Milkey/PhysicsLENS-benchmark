"""
GEPA-style reflective prompt evolution for the VLM physics judge.

Optimises the *instruction text* handed to gemma4-31b-it (our best-measured
judge, rho=+0.286 with the hand-written likert prompt) against human physical-
commonsense ratings, then measures the gain on a held-out split.

WHY THIS AND NOT dspy.GEPA (3.3.0 is installed): DSPy's GEPA mutates
`predictor.signature.instructions` and builds its reflective dataset from the
trace of predictor calls. Our "predictor" never emits text — it emits a single
token whose *probability distribution* is the score (see `_token_probs`). A
DSPy module wrapping that would have an empty trace and nothing to reflect on,
so the optimiser would be a shell around our own loop. The algorithm below is
GEPA's, reimplemented against the logprob scorer.

ALGORITHM (Agrawal et al., "GEPA: Reflective Prompt Evolution"), and where this
deviates:

  1. Seed the candidate pool with the current hand-written prompt.
  2. Evaluate the seed on the val set, keeping PER-INSTANCE scores.
  3. Until the rollout budget is spent:
       a. Sample a parent from the PARETO FRONTIER — the candidates that are
          best-on-at-least-one val instance, weighted by how many they win.
          (Sampling by aggregate score instead would collapse diversity onto
          one lineage; the frontier is the part of GEPA that matters most.)
       b. Draw a minibatch from train; evaluate the parent on it, producing a
          scalar AND a natural-language critique per clip.
       c. A reflection LM reads (instruction, clips, model output, human label,
          human-annotated violated rule, critique) and writes a new instruction.
       d. Evaluate the child on the SAME minibatch. Promote to the pool only if
          it beats the parent there — the minibatch is a cheap filter so the
          expensive val pass runs rarely.
  4. Select the candidate with the best val Spearman, score it ONCE on test.

  DEVIATION 1 — per-instance score vs reported metric. GEPA needs a decomposable
  per-instance scalar; Spearman is set-level and has no per-clip value. So the
  frontier uses per-clip accuracy, 1-|predicted_pc - human_pc|/4, while
  candidate SELECTION uses val Spearman, which is what we actually report.
  Optimising calibration to select for ranking is a real gap, and is why val
  rho is tracked separately rather than assumed to follow.

  DEVIATION 2 — frozen format contract. `FORMAT_LOCK` is appended to every
  candidate and is not mutable. A mutation that dropped "answer with one digit"
  would put zero mass on 1-5 and score ~0: self-correcting but wasteful, and it
  would let the optimiser wander out of the space where the metric is defined.

CONTROL: --mode random runs the identical loop with the feedback stripped out
of the mutation prompt ("rewrite this differently"). Same budget, same
reflection LM, same promotion rule. If GEPA does not beat it, the reflective
feedback is contributing nothing and the gain is prompt lottery.

Usage:
  python backend/scripts/gepa_optimize.py --mode gepa   --budget 900
  python backend/scripts/gepa_optimize.py --mode random --budget 900
  python backend/scripts/gepa_optimize.py --final            # held-out test + stats
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman                                # noqa: E402
from videophy_eval import (LIKERT_Q, _retry, _token_probs, auc_extremes,  # noqa: E402
                           boot_ci, client, load_frames)

TASK_MODEL = "gemma4-31b-it"
REFLECT_MODEL = "qwen3-235b-a22b-instruct-2507"

# Appended verbatim to every candidate; never mutated. See DEVIATION 2.
FORMAT_LOCK = "\nAnswer with exactly one digit, 1 to 5. No other text."

# The seed is the hand-written prompt that scored rho=+0.286, minus its own
# format line (FORMAT_LOCK supplies it) so the seed is not double-instructed.
SEED = LIKERT_Q.replace("\nAnswer with exactly one digit, 1 to 5.", "")


# ── data ──────────────────────────────────────────────────────────────────────

def split_clips(clips, seed=0, n_train=90, n_val=50):
    """Stratify by pc so every split spans the full 1-5 range.

    An unstratified split can hand the val set a narrow pc range, which both
    depresses Spearman (less spread to rank) and makes selection noisy.
    """
    by_pc = defaultdict(list)
    for c in clips:
        by_pc[c["pc"]].append(c)
    rnd = random.Random(seed)
    for v in by_pc.values():
        rnd.shuffle(v)

    train, val, test = [], [], []
    for pc in sorted(by_pc):
        v = by_pc[pc]
        n_tr = round(n_train * len(v) / len(clips))
        n_va = round(n_val * len(v) / len(clips))
        train += v[:n_tr]
        val += v[n_tr:n_tr + n_va]
        test += v[n_tr + n_va:]
    for s in (train, val, test):
        rnd.shuffle(s)
    return train, val, test


# ── scoring ───────────────────────────────────────────────────────────────────

class Scorer:
    """Evaluates a candidate instruction on clips, with a memo.

    The memo matters: a parent is re-evaluated on every minibatch it is sampled
    for, and candidates share val clips. Without it roughly half the rollout
    budget would be spent re-scoring pairs already seen.
    """

    def __init__(self, c, data, model, workers=4):
        self.c, self.data, self.model, self.workers = c, data, model, workers
        self.memo = {}
        self.calls = 0

    def _one(self, prompt, clip):
        key = (hash(prompt), clip["clip_id"])
        if key in self.memo:
            return self.memo[key]
        imgs = load_frames(self.data, clip, "temporal")
        # .replace not .format — an evolved prompt may contain stray braces,
        # and .format would raise on them. Placeholders are opt-in for the
        # optimiser: it may keep {n}, drop it, or start using {caption}.
        q = (prompt.replace("{n}", str(len(imgs)))
                   .replace("{caption}", clip.get("caption", "")) + FORMAT_LOCK)
        try:
            p = _token_probs(self.c, self.model, imgs, q)
            mass = {int(k): v for k, v in p.items() if len(k) == 1 and k in "12345"}
            tot = sum(mass.values())
            ev = sum(k * v for k, v in mass.items()) / tot if tot > 1e-4 else None
        except Exception as e:  # noqa: BLE001
            print(f"    score fail {clip['clip_id'][:32]}: {str(e)[:60]}",
                  file=sys.stderr)
            ev = None
        self.calls += 1
        self.memo[key] = ev
        return ev

    def run(self, prompt, clips):
        """-> (list of predicted pc in 1..5 or None, aligned with clips)."""
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda cl: self._one(prompt, cl), clips))


def per_instance(evs, clips):
    """1 - |predicted pc - human pc| / 4, in [0,1]. None -> 0 (a candidate that
    breaks the output format must be penalised, not silently skipped)."""
    return [0.0 if e is None else 1.0 - abs(e - c["pc"]) / 4.0
            for e, c in zip(evs, clips)]


def rho_of(evs, clips):
    """Spearman between implausibility (5-ev) and (6-pc). Both inverted, so the
    sign convention matches videophy_eval.py and a good judge scores positive."""
    pairs = [(5.0 - e, 6 - c["pc"]) for e, c in zip(evs, clips) if e is not None]
    if len(pairs) < 8 or len({p for p, _ in pairs}) < 2:
        return 0.0
    r = spearman([p for p, _ in pairs], [h for _, h in pairs])
    return r if r == r else 0.0


# ── reflection ────────────────────────────────────────────────────────────────

REFLECT_SYS = (
    "You improve instructions given to a vision-language model that rates the "
    "physical realism of AI-generated video, shown as still frames. You will see "
    "the current instruction and how it performed on real clips with human "
    "ratings. Write a better instruction."
)

REFLECT_TMPL = """Current instruction given to the vision model:
---
{prompt}
---

It was run on {k} video clips. The model sees {n} still frames from each clip.
For each clip below: what the video was supposed to show, the rating the model
produced, the rating human annotators gave, and — where available — the exact
physical rule the humans said was broken.

{examples}

The model's ratings should correlate with the human ratings. Study where it went
wrong. Common failure patterns worth addressing: rating everything near the same
value (no discrimination), being systematically too generous or too harsh, or
ignoring the specific categories of physical rule the humans actually cared about.

Write a NEW instruction that would make the model's ratings track the human
ratings more closely. You may use the placeholder {{n}} for the number of frames
and {{caption}} for what the video is supposed to show. Do NOT include any
output-format instruction — the digit-only format is appended automatically.

Reply with ONLY the new instruction text, nothing else."""

RANDOM_TMPL = """Current instruction given to a vision model that rates the physical
realism of AI-generated video shown as {n} still frames:
---
{prompt}
---

Write a NEW, meaningfully different instruction for the same task. Vary the
framing, emphasis, or level of detail. You may use the placeholder {{n}} for the
number of frames and {{caption}} for what the video is supposed to show. Do NOT
include any output-format instruction — it is appended automatically.

Reply with ONLY the new instruction text, nothing else."""


def build_examples(clips, evs):
    out = []
    for cl, ev in zip(clips, evs):
        if ev is None:
            out.append(f"- \"{cl['caption'][:110]}\"\n  model: FAILED to produce a "
                       f"valid rating | human: {cl['pc']}/5")
            continue
        err = ev - cl["pc"]
        d = ("OVERRATED (too generous)" if err > 0.5 else
             "UNDERRATED (too harsh)" if err < -0.5 else "close")
        line = (f"- \"{cl['caption'][:110]}\"\n"
                f"  model: {ev:.2f}/5 | human: {cl['pc']}/5 | {d} by {abs(err):.2f}")
        rules = (cl.get("violated_rules") or "").strip()
        if len(rules) > 4:
            line += f"\n  human-annotated broken rule: {rules[:180]}"
        out.append(line)
    return "\n".join(out)


def reflect(c, prompt, clips, evs, mode, n_frames=8):
    if mode == "gepa":
        msg = REFLECT_TMPL.format(prompt=prompt, k=len(clips), n=n_frames,
                                  examples=build_examples(clips, evs))
    else:
        msg = RANDOM_TMPL.format(prompt=prompt, n=n_frames)
    r = _retry(lambda: c.chat.completions.create(
        model=REFLECT_MODEL, temperature=1.0, max_tokens=900,
        messages=[{"role": "system", "content": REFLECT_SYS},
                  {"role": "user", "content": msg}]))
    txt = (r.choices[0].message.content or "").strip()
    for fence in ("```text", "```"):
        if txt.startswith(fence):
            txt = txt[len(fence):].strip().removesuffix("```").strip()
    return txt if 40 < len(txt) < 4000 else None


# ── the optimiser ─────────────────────────────────────────────────────────────

def pareto_parents(pool):
    """Candidates that achieve the best per-instance val score on >=1 clip.

    This is GEPA's central mechanism: a candidate that is mediocre on average
    but uniquely good on a handful of clips stays in the gene pool, where a
    pure best-aggregate rule would discard it.
    """
    n = len(pool[0]["val_inst"])
    wins = defaultdict(int)
    for i in range(n):
        col = [cand["val_inst"][i] for cand in pool]
        best = max(col)
        for j, v in enumerate(col):
            if v >= best - 1e-9:
                wins[j] += 1
    idx = [j for j in wins if wins[j] > 0]
    return idx, [wins[j] for j in idx]


def optimize(c, sc, train, val, mode, budget, mb_size, seed, log):
    rnd = random.Random(seed)
    t0 = time.time()

    evs = sc.run(SEED, val)
    pool = [{"prompt": SEED, "val_inst": per_instance(evs, val),
             "val_rho": rho_of(evs, val), "parent": None, "iter": 0}]
    log(f"[{mode}] seed val rho = {pool[0]['val_rho']:+.3f}  "
        f"(mean per-instance {np.mean(pool[0]['val_inst']):.3f})")

    it = 0
    while sc.calls < budget:
        it += 1
        idx, w = pareto_parents(pool)
        pj = rnd.choices(idx, weights=w, k=1)[0]
        parent = pool[pj]

        mb = rnd.sample(train, min(mb_size, len(train)))
        p_evs = sc.run(parent["prompt"], mb)
        p_score = float(np.mean(per_instance(p_evs, mb)))

        try:
            child = reflect(c, parent["prompt"], mb, p_evs, mode)
        except Exception as e:  # noqa: BLE001
            log(f"  it{it:02d} reflection failed: {str(e)[:70]}")
            continue
        if not child or any(child == p["prompt"] for p in pool):
            log(f"  it{it:02d} no usable child")
            continue

        c_evs = sc.run(child, mb)
        c_score = float(np.mean(per_instance(c_evs, mb)))
        tag = f"  it{it:02d} parent#{pj} mb {p_score:.3f} -> child {c_score:.3f}"

        if c_score <= p_score:
            log(tag + "   rejected")
            continue

        v_evs = sc.run(child, val)                     # the expensive step
        cand = {"prompt": child, "val_inst": per_instance(v_evs, val),
                "val_rho": rho_of(v_evs, val), "parent": pj, "iter": it}
        pool.append(cand)
        best = max(p["val_rho"] for p in pool)
        log(tag + f"   PROMOTED  val rho {cand['val_rho']:+.3f}"
                  f"  (best {best:+.3f})  [{sc.calls}/{budget} rollouts,"
                  f" {time.time()-t0:.0f}s]")

    for i, p in enumerate(pool):
        p["id"] = i
    return pool


# ── entrypoints ───────────────────────────────────────────────────────────────

def load(data_dir):
    data = (ROOT / data_dir) if not Path(data_dir).is_absolute() else Path(data_dir)
    clips = json.loads((data / "manifest.json").read_text())["clips"]
    return data, clips


def cmd_optimize(a):
    data, clips = load(a.data)
    train, val, test = split_clips(clips, a.seed, a.n_train, a.n_val)
    print(f"split: train {len(train)} | val {len(val)} | test {len(test)} (held out)")

    c = client()
    sc = Scorer(c, data, TASK_MODEL, a.workers)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    pool = optimize(c, sc, train, val, a.mode, a.budget, a.mb, a.seed, log)
    best = max(pool, key=lambda p: p["val_rho"])
    out = data / f"gepa_{a.mode}.json"
    out.write_text(json.dumps(
        {"mode": a.mode, "task_model": TASK_MODEL, "reflect_model": REFLECT_MODEL,
         "budget": a.budget, "rollouts_used": sc.calls, "mb": a.mb, "seed": a.seed,
         "n_train": len(train), "n_val": len(val), "n_test": len(test),
         "seed_val_rho": pool[0]["val_rho"], "best_val_rho": best["val_rho"],
         "best_prompt": best["prompt"], "best_id": best["id"],
         "pool": [{k: v for k, v in p.items() if k != "val_inst"} for p in pool],
         "log": lines}, indent=1))
    print(f"\n[{a.mode}] {len(pool)} candidates, {sc.calls} rollouts")
    print(f"  seed val rho {pool[0]['val_rho']:+.3f} -> best {best['val_rho']:+.3f}")
    print(f"  -> {out}")


def paired_boot(xa, xb, ys, n=4000, seed=0):
    """CI on rho(a) - rho(b) resampling CLIPS, so both prompts are compared on
    the identical clip in every draw. Unpaired CIs would be far wider and would
    miss a real difference between two similar prompts."""
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
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def cmd_final(a):
    data, clips = load(a.data)
    train, val, test = split_clips(clips, a.seed, a.n_train, a.n_val)
    ids = {c["clip_id"] for c in test}
    print(f"held-out test: {len(test)} clips "
          f"(pc {dict(sorted(__import__('collections').Counter(c['pc'] for c in test).items()))})")

    # Baseline = the saved 300-clip likert run, subset to test. GEPA seeds from
    # that same prompt, so this is the exact paired baseline with zero re-scoring.
    saved = json.loads(
        (data / "scores_gemma4-31b-it_likert_temporal.json").read_text())["scores"]
    runs = {"baseline (hand-written likert)":
            [saved.get(c["clip_id"]) for c in test]}

    c = client()
    sc = Scorer(c, data, TASK_MODEL, a.workers)
    for mode in ("gepa", "random"):
        f = data / f"gepa_{mode}.json"
        if not f.exists():
            print(f"  (skip {mode}: {f.name} missing)")
            continue
        d = json.loads(f.read_text())
        print(f"\nscoring {mode} best prompt on {len(test)} held-out clips "
              f"(val rho was {d['best_val_rho']:+.3f}) …", flush=True)
        t0 = time.time()
        evs = sc.run(d["best_prompt"], test)
        # -> implausibility, same transform as score_likert, so directly
        # comparable to the saved baseline scores.
        runs[f"{mode} (evolved)"] = [None if e is None else (5.0 - e) / 4.0
                                     for e in evs]
        print(f"  done in {time.time()-t0:.0f}s")

    print(f"\n{'condition':34s} {'n':>4s} {'rho':>8s} {'95% CI':>18s} {'AUC':>7s}")
    print("-" * 76)
    summary = {}
    for name, xs in runs.items():
        keep = [i for i, x in enumerate(xs) if x is not None]
        sx = [xs[i] for i in keep]
        sy = [6 - test[i]["pc"] for i in keep]
        r = spearman(sx, sy)
        lo, hi = boot_ci(lambda rn: spearman(
            *zip(*[(sx[k], sy[k]) for k in (rn.randrange(len(sx)) for _ in sx)])))
        auc, nb, ng = auc_extremes(sx, [test[i]["pc"] for i in keep])
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "-"
        print(f"{name:34s} {len(sx):4d} {r:+8.3f} {ci:>18s} {auc or 0:7.3f}")
        summary[name] = {"n": len(sx), "rho": round(r, 3),
                         "ci95": [round(lo, 3), round(hi, 3)] if lo else None,
                         "auc": round(auc, 3) if auc else None,
                         "scores": {test[i]["clip_id"]: xs[i] for i in keep}}

    print(f"\n{'paired comparison (same clips)':44s} {'delta':>8s} {'95% CI':>18s}")
    print("-" * 76)
    base = "baseline (hand-written likert)"
    cmps = [(k, base) for k in runs if k != base]
    if "gepa (evolved)" in runs and "random (evolved)" in runs:
        cmps.append(("gepa (evolved)", "random (evolved)"))
    stats = {}
    for na, nb_ in cmps:
        keep = [i for i in range(len(test))
                if runs[na][i] is not None and runs[nb_][i] is not None]
        xa = [runs[na][i] for i in keep]
        xb = [runs[nb_][i] for i in keep]
        ys = [6 - test[i]["pc"] for i in keep]
        d = spearman(xa, ys) - spearman(xb, ys)
        lo, hi = paired_boot(xa, xb, ys)
        sig = "SIGNIFICANT" if lo is not None and lo > 0 else "not significant"
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "-"
        print(f"{na + ' vs ' + nb_:44s} {d:+8.3f} {ci:>18s}  {sig}")
        stats[f"{na} vs {nb_}"] = {"delta": round(d, 3),
                                   "ci95": [round(lo, 3), round(hi, 3)] if lo else None,
                                   "significant": bool(lo is not None and lo > 0)}

    outp = data / "gepa_final.json"
    outp.write_text(json.dumps({"n_test": len(test), "seed": a.seed,
                                "conditions": summary, "paired": stats}, indent=1))
    print(f"\n-> {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy300")
    ap.add_argument("--mode", default="gepa", choices=["gepa", "random"])
    ap.add_argument("--budget", type=int, default=900, help="rollout budget")
    ap.add_argument("--mb", type=int, default=10, help="minibatch size")
    ap.add_argument("--n-train", type=int, default=90)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true",
                    help="score evolved prompts on the held-out test split")
    a = ap.parse_args()
    (cmd_final if a.final else cmd_optimize)(a)


if __name__ == "__main__":
    main()
