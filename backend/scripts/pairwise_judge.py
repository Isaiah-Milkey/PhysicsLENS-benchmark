"""
Pairwise VLM judgement: show two videos, ask which is more physically plausible.

WHY. Every system so far scores each video ALONE and compares scores afterwards.
On the obs/unobs twins that sits at chance: the twins share scene, objects and
generator, and a 1-4 absolute rating is too coarse to express "slightly worse".
A direct comparison is a different measurement — the model sees both at once and
only has to order them, which is what the human rating difference encodes.

TWO PAIR SETS
  twins    obs vs unobs, same source frame and generator (119 pairs)
  gens     same task (observable), two different generators (up to 6 per task)
Only pairs whose human ratings differ are scored against labels; ties are kept
in the output but excluded from accuracy.

POSITION BIAS. Models prefer "A" (or the first video). Every pair is asked in
BOTH orders and the two probabilities are combined as
    score(first beats second) = ( P(A | first=A) + P(B | first=B) ) / 2,
so a model that always says "A" lands at exactly 0.5 instead of looking right on
half the pairs.

Frames: k per video (default 4 -> 8 images, within every local model's budget;
the API path uses --frames 2 because gateway models cap at 4 images).

Usage:
  python backend/scripts/pairwise_judge.py --data data/consol --model qwen3-vl-8b --device cuda:1
  python backend/scripts/pairwise_judge.py --data data/consol --model qwen3-vl-32b-instruct --api --frames 2
"""
import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from gepa_optimize import load  # noqa: E402

Q = ('You will see two videos of a robot doing the same task: "{task}".\n'
     'Video A is the first {k} frames, Video B is the next {k} frames, each in '
     'time order.\n\nWhich video is MORE PHYSICALLY PLAUSIBLE — objects, contact, '
     'motion, liquids and deformation behaving as they would in the real world?\n'
     'Answer with exactly one letter: A or B.')


# DEBIASED wording. With the plain question, judges prefer the video where MORE
# happens (rho +0.45 with the motion difference) while humans prefer the one
# where less breaks — they read "plausible" as "successful". This version names
# the error types and explicitly rules out task success and activity as criteria.
Q_DEBIAS = ('You will see two videos of a robot attempting: "{task}".\n'
            'Video A is the first {k} frames, Video B is the next {k} frames, each '
            'in time order.\n\nIgnore whether the robot finishes the task and ignore '
            'how much motion there is — a video where little happens can be the '
            'better one.\nCount PHYSICS ERRORS only: objects passing through each '
            'other, appearing or vanishing, floating without support, changing shape '
            'impossibly, or moving with nothing touching them.\n\nWhich video has '
            'FEWER physics errors? Answer with exactly one letter: A or B.')


def pairs_for(clips):
    by_pair, by_task = {}, {}
    for c in clips:
        by_pair.setdefault(c["pair_id"], {})[c["observability"]] = c
        if c["observability"] == "observable":
            by_task.setdefault(c["testset_id"], []).append(c)
    twins = [(v["observable"], v["unobservable"], "twins")
             for v in by_pair.values() if len(v) == 2]
    gens = [(a, b, "gens") for t in by_task.values()
            for a, b in itertools.combinations(sorted(t, key=lambda c: c["generator"]), 2)]
    return twins + gens


def pab(probs):
    """P(A) renormalised over the two letters, case/space insensitive."""
    a = sum(v for k, v in probs.items() if str(k).strip().upper() == "A")
    b = sum(v for k, v in probs.items() if str(k).strip().upper() == "B")
    return a / (a + b) if a + b > 1e-6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/consol")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sets", default="twins,gens")
    ap.add_argument("--prompt", default="plain", choices=["plain", "debias"])
    a = ap.parse_args()
    data, clips = load(a.data)
    want = set(a.sets.split(","))
    P = [p for p in pairs_for(clips) if p[2] in want]
    print(f"[{a.model}] {len(P)} pairs x 2 orders = {2*len(P)} calls", flush=True)

    def prompt(x):
        return (Q_DEBIAS if a.prompt == "debias" else Q).format(
            task=x.get("task") or "a manipulation task", k=a.frames)

    res, t0 = {}, time.time()
    if a.api:
        from videophy_eval import _token_probs, client
        from domain_probes import frames_ordered
        c = client()
        fc = {}

        def fr(x):
            if x["clip_id"] not in fc:
                fc[x["clip_id"]] = frames_ordered(data, x, a.frames, "temporal")
            return fc[x["clip_id"]]

        def work(p):
            x, y, kind = p
            try:
                p1 = pab(_token_probs(c, a.model, fr(x) + fr(y), prompt(x)))
                p2 = pab(_token_probs(c, a.model, fr(y) + fr(x), prompt(x)))
                if p1 is not None and p2 is not None:
                    res[f"{x['clip_id']}|{y['clip_id']}"] = dict(kind=kind, p=(p1 + (1 - p2)) / 2,
                                                                 raw=[p1, p2])
            except Exception as e:  # noqa: BLE001
                print(f"  fail {x['clip_id']}|{y['clip_id']}: {str(e)[:60]}", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, P))
    else:
        import torch
        from videophy_eval_local import load_model
        from probes_local import frames_pil
        _, model, proc = load_model(a.model, a.device)
        tok = getattr(proc, "tokenizer", proc)
        ids_ = {L: [tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)
                    if tok.encode(v, add_special_tokens=False)] for L in "AB"}

        def ask(imgs, text):
            content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": text}]
            inp = proc.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
            for k_, v_ in list(inp.items()):
                if hasattr(v_, "is_floating_point") and v_.is_floating_point():
                    inp[k_] = v_.to(model.dtype)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=1, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True)
            pr = torch.softmax(o.scores[0][0].float(), dim=-1)
            pa = sum(float(pr[i]) for i in set(ids_["A"]))
            pb = sum(float(pr[i]) for i in set(ids_["B"]))
            return pa / (pa + pb) if pa + pb > 1e-6 else None
        fc = {}

        def fr(x):
            if x["clip_id"] not in fc:
                fc[x["clip_id"]] = frames_pil(data, x, a.frames)
            return fc[x["clip_id"]]
        for i, (x, y, kind) in enumerate(P):
            p1, p2 = ask(fr(x) + fr(y), prompt(x)), ask(fr(y) + fr(x), prompt(x))
            if p1 is not None and p2 is not None:
                res[f"{x['clip_id']}|{y['clip_id']}"] = dict(kind=kind, p=(p1 + (1 - p2)) / 2, raw=[p1, p2])
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"   {i+1}/{len(P)} ({el:.0f}s, eta {el/(i+1)*(len(P)-i-1)/60:.0f}m)", flush=True)

    out = data / (f"pairwise_{a.model}__f{a.frames}"
                  + ("__debias" if a.prompt == "debias" else "") + ".json")
    out.write_text(json.dumps({"model": a.model, "frames": a.frames, "prompt": a.prompt,
                               "n": len(res),
                               "pairs": res}, indent=1))
    print(f"\n  {len(res)}/{len(P)} pairs ({time.time()-t0:.0f}s) -> {out}")
    if len(res) < 0.5 * len(P):
        sys.exit("ERROR: under half the pairs produced an A/B distribution")


if __name__ == "__main__":
    main()
