"""
One multiple-choice question instead of eight independent probes.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE TRIED. The eight specialists are
currently scored independently: each gets its own call, its own 1-5 scale, and
nothing forces them to disagree. That produces the exact pathology measured in
the failure analysis — collision answers ~0.97 on every clip (grippers do touch
things) and deformation answers 0.000 on every clip. Independent scales let every
specialist be simultaneously right and useless.

A single forced-choice question makes the options COMPETE. The probabilities over
the letters are normalised by construction, so "everything is a collision
failure" is no longer expressible: saying collision costs probability mass that
must come from somewhere else. That is precisely the discrimination the
attribution test measures, asked directly instead of reconstructed from eight
separate opinions.

It is also ~8x cheaper: one call per clip rather than one per specialist.

POSITION BIAS IS CONTROLLED. Language models favour the first option and the
letter A. The option order is therefore permuted per clip with a fixed seed, and
the permutation is stored, so a preference for "whatever is listed first" shows
up as noise across clips rather than as a win for whichever specialist happened
to be printed first. Without this the experiment would mostly measure list order.

A NONE option is included. Clean clips exist in this dataset (20 annotated clean,
plus 82 real demonstrations), and without an explicit "no physics problem" escape
the model is forced to allege a defect on every clip, which is both wrong and
makes the clean-vs-broken test unmeasurable.

Usage:
  python backend/scripts/mcq_probe.py --data data/robotbench --model internvl3-8b
  python backend/scripts/mcq_probe.py --data data/robotbench --model gemma4-31b-it --api
"""
import argparse
import json
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from domain_probes import caption_for  # noqa: E402
from gepa_optimize import load  # noqa: E402

# Wording taken from the robot battery, compressed to one line each. These are
# the descriptions that beat the VideoPhy-2 phrasing by +0.082, so the options
# inherit the one intervention that demonstrably worked.
OPTIONS = [
    ("collision", "Two things overlap or pass through each other instead of "
                  "meeting at their surfaces, or the gripper grasps without "
                  "closing on the object"),
    ("deformation", "A rigid object changes its shape, length or thickness"),
    ("causality", "An object moves or changes on its own, with nothing visibly "
                  "touching it"),
    ("gravity", "Something hangs in the air, floats, or fails to fall when "
                "nothing is holding it"),
    ("momentum", "Something speeds up after being released or struck, or stops "
                 "dead for no reason"),
    ("permanence", "An object appears from nowhere or vanishes"),
    ("fluid", "Liquid appears, vanishes, or holds an impossible rigid shape"),
    ("friction", "Something slides when it should grip, or keeps sliding with "
                 "nothing pushing it"),
    ("none", "No physics problem — everything behaves as it should"),
]
NAMES = [n for n, _ in OPTIONS]

PROMPT = ('Look at these {n} frames, sampled in order from a video of: '
          '"{caption}".\n\n'
          'Which ONE of these best describes the main physics problem in this '
          'video?\n\n{opts}\n\n'
          'Answer with exactly one letter.')


def build(clip, nframes, capmode, seed=0):
    """Prompt text plus the letter -> specialist mapping for THIS clip."""
    rng = np.random.default_rng(abs(hash(clip["clip_id"])) % (2 ** 31) + seed)
    order = list(rng.permutation(len(OPTIONS)))
    letters = string.ascii_uppercase[:len(OPTIONS)]
    lines, mapping = [], {}
    for pos, idx in enumerate(order):
        name, desc = OPTIONS[idx]
        lines.append(f"{letters[pos]}) {desc}")
        mapping[letters[pos]] = name
    cap = caption_for(clip, capmode)
    head = PROMPT.format(n=nframes, caption=cap, opts="\n".join(lines)) if cap \
        else PROMPT.format(n=nframes, caption="a robot performing a task",
                           opts="\n".join(lines))
    return head, mapping


def normalise(probs, mapping):
    """letter probabilities -> specialist probabilities, renormalised.

    Case- and whitespace-insensitive, and it SUMS the variants rather than
    picking one. gemma4-31b-it returns lowercase first tokens ('g', 'd') through
    this gateway while the local models return uppercase, so exact matching on
    'A'-'I' silently scored 0/226 clips for gemma4 — the run "completed" with an
    empty file. Different tokenizers also split a bare letter from a
    space-prefixed one, and both spellings are the same answer.
    """
    agg = {}
    for k, v in probs.items():
        key = str(k).strip().upper()
        if key in mapping:
            agg[mapping[key]] = agg.get(mapping[key], 0.0) + v
    tot = sum(agg.values())
    if tot <= 1e-6:
        return None
    return {n: agg.get(n, 0.0) / tot for n in NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="internvl3-8b")
    ap.add_argument("--api", action="store_true",
                    help="route through the OpenAI-compatible gateway instead of local")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--capmode", default="task")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--all-clips", action="store_true", default=True)
    a = ap.parse_args()

    data, clips = load(a.data)
    print(f"MCQ [{a.model}]: {len(OPTIONS)} options x {len(clips)} clips "
          f"= {len(clips)} calls (vs {8*len(clips)} for separate probes)",
          flush=True)

    out, t0 = {}, time.time()
    if a.api:
        from videophy_eval import _token_probs, client
        from domain_probes import frames_ordered
        c = client()
        res = [None] * len(clips)

        def work(i):
            cl = clips[i]
            head, mapping = build(cl, a.frames, a.capmode)
            try:
                p = _token_probs(c, a.model,
                                 frames_ordered(data, cl, a.frames, "temporal"),
                                 head)
                res[i] = (cl["clip_id"], normalise(p, mapping), mapping)
            except Exception as e:  # noqa: BLE001
                print(f"   fail {cl['clip_id'][:26]}: {str(e)[:50]}",
                      file=sys.stderr)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, range(len(clips))))
        for r in res:
            if r and r[1]:
                out[r[0]] = {"probs": r[1], "map": r[2]}
    else:
        import torch
        from videophy_eval_local import load_model
        from probes_local import frames_pil
        torch_, model, proc = load_model(a.model, a.device)
        tok = getattr(proc, "tokenizer", proc)
        for i, cl in enumerate(clips):
            imgs = frames_pil(data, cl, a.frames)
            head, mapping = build(cl, a.frames, a.capmode)
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": head})
            inputs = proc.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            # some processors (Mistral-3) emit float32 pixels for bf16 weights
            for _k, _v in list(inputs.items()):
                if hasattr(_v, "is_floating_point") and _v.is_floating_point():
                    inputs[_k] = _v.to(model.dtype)
            with torch_.no_grad():
                o = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                   output_scores=True,
                                   return_dict_in_generate=True)
            pr = torch_.softmax(o.scores[0][0].float(), dim=-1)
            probs = {}
            for L in mapping:
                # both bare and space-prefixed forms; models differ on which
                # they emit first and the mass belongs to the same answer
                for v in (L, " " + L):
                    e = tok.encode(v, add_special_tokens=False)
                    if e:
                        probs[L] = probs.get(L, 0.0) + float(pr[e[0]].item())
            n = normalise(probs, mapping)
            if n:
                out[cl["clip_id"]] = {"probs": n, "map": mapping}
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"    {i+1}/{len(clips)} ({el:.0f}s, eta "
                      f"{el/(i+1)*(len(clips)-i-1)/60:.0f}m)", flush=True)

    p = data / f"mcq_{a.model}__f{a.frames}.json"
    p.write_text(json.dumps({"model": a.model, "frames": a.frames,
                             "capmode": a.capmode, "options": NAMES,
                             "n": len(out), "mcq": out}, indent=1))
    print(f"\n  {len(out)}/{len(clips)} clips ({time.time()-t0:.0f}s)")
    if out:
        top = [max(v["probs"], key=v["probs"].get) for v in out.values()]
        from collections import Counter
        print("  argmax answer distribution:",
              dict(Counter(top).most_common()))
    print(f"  -> {p}")
    if len(out) < 0.5 * len(clips):
        print(f"\nERROR: only {len(out)}/{len(clips)} clips produced a usable "
              f"answer distribution. The model is probably not emitting an "
              f"option letter as its first token — inspect the raw logprobs "
              f"before trusting any downstream table.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
