"""
VLM-alone judgements on the consolidated benchmark, on the annotators' own scales.

Three questions per clip, each scored by token probability over the answer
digits (expected value over 1-4, renormalised over the digit tokens so preamble
and whitespace mass does not leak in):

  plaus    "How physically plausible is this video?" 1-4 — the exact scale of
           physical_plausibility_1_4, so VLM and human are directly comparable.
  action   "Did the robot complete the task?" P(yes) — vs action_completed.
  hidden   UNOBSERVABLE clips only. The judge is TOLD the hidden property and
           the expected outcome ("the table is hydrophobic; liquid should bead
           and roll off"), then asked 1-4 whether the video follows it — the
           scale of hidden_property_followed_1_4. The property is invisible by
           construction, so without the hint the question is unanswerable.
  hidden_nohint
           the same question with the hint removed. The gap between the two is
           how much of any score comes from the text rather than the pixels.

Backends: --api (OpenAI-compatible gateway) or local HF models, same estimator either way.

Usage:
  python backend/scripts/vlm_plausibility.py --data data/consol --model qwen3-vl-32b-instruct --api --frames 8
  python backend/scripts/vlm_plausibility.py --data data/consol --model internvl3-8b --device cuda:0
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from gepa_optimize import load  # noqa: E402

HEAD = 'Look at these {n} frames, sampled in order from a video of a robot. The task: "{task}".\n\n'

Q = {
    "plaus": ("Rate how PHYSICALLY PLAUSIBLE this video is — do objects, contact, "
              "motion, liquids and deformation behave as they would in the real "
              "world?\n1 = clearly impossible physics, 2 = noticeable physics "
              "errors, 3 = minor oddities, 4 = fully plausible.\n"
              "Answer with exactly one digit, 1 to 4.", "1234"),
    # debiased: the plain question lets judges reward activity/task success
    "plaus_debias": ("Ignore whether the robot finishes the task and ignore how much "
                     "motion there is. Count only PHYSICS ERRORS: objects passing "
                     "through each other, appearing or vanishing, floating without "
                     "support, changing shape impossibly, or moving with nothing "
                     "touching them.\n1 = many clear physics errors, 2 = some errors, "
                     "3 = minor oddities, 4 = no physics errors.\n"
                     "Answer with exactly one digit, 1 to 4.", "1234"),
    "action": ("Did the robot actually COMPLETE the task shown?\n"
               "Answer with exactly one digit: 1 = no, 2 = yes.", "12"),
    "hidden": ("Hidden property of this scene (not visible in the frames): "
               "{hp} — {hv}.\nIf the video respects it, this should happen: "
               "{exp}\n\nRate how well the video FOLLOWS that hidden property.\n"
               "1 = contradicts it, 2 = mostly contradicts, 3 = mostly follows, "
               "4 = clearly follows.\nAnswer with exactly one digit, 1 to 4.",
               "1234"),
    # concrete failure: name what a violation LOOKS like (failure_signature from
    # the prompt tables) instead of asking abstractly whether a property "holds"
    "hidden_sig": ("Hidden property of this scene (not visible in the frames): "
                   "{hp} — {hv}.\nIgnore whether the robot finishes the task and "
                   "how much motion there is.\nIf the video VIOLATES that property, "
                   "it would look like this: {sig}\n\nHow clearly does the video show "
                   "that violation?\n1 = not at all, 2 = slightly, 3 = clearly, "
                   "4 = blatantly.\nAnswer with exactly one digit, 1 to 4.", "1234"),
    "hidden_nohint": ("Does the video behave consistently with the physical "
                      "properties of the objects and surfaces in the scene?\n"
                      "1 = contradicts them, 2 = mostly contradicts, 3 = mostly "
                      "consistent, 4 = clearly consistent.\n"
                      "Answer with exactly one digit, 1 to 4.", "1234"),
}


def ev(probs, digits):
    """Expected value over the allowed digits, case/space-insensitive, summing
    surface-form variants. Returns (ev, mass) — mass lets a run fail loudly when
    a model is not answering with digits at all."""
    m = {}
    for k, v in probs.items():
        k = str(k).strip()
        # len guard: "" is a substring of every string, so a bare `in` let an
        # empty token through to int("") and killed that call
        if len(k) == 1 and k in digits:
            m[k] = m.get(k, 0.0) + v
    tot = sum(m.values())
    if tot <= 1e-6:
        return None, 0.0
    return sum(int(k) * v for k, v in m.items()) / tot, tot


HEAD_GENERIC = 'Look at these {n} frames, sampled in order from a video of: "{cap}".\n\n'


def prompts(cl, nframes, only=None):
    task = cl.get("task") or cl.get("action") or "a manipulation task"
    if only and "hidden_sig" in only:
        if cl.get("observability") != "unobservable" or not cl.get("failure_signature"):
            return {}
        return {"hidden_sig": HEAD.format(n=nframes, task=task) + Q["hidden_sig"][0].format(
            hp=cl["hidden_property"].replace("_", " "), hv=cl.get("hidden_value", ""),
            sig=cl["failure_signature"])}
    if only:
        # non-robot datasets (VideoPhy-2) have a caption, not a robot task
        h = (HEAD.format(n=nframes, task=task) if cl.get("task") or cl.get("action")
             else HEAD_GENERIC.format(n=nframes, cap=(cl.get("caption") or "")[:300]))
        q = {k: Q[k][0].replace("Ignore whether the robot finishes the task",
                                 "Ignore whether the action succeeds")
                        .replace("did the robot", "did it") for k in only}
        out = {}
        for k in only:
            if k == "hidden":   # needs the scene's property; unobservable clips only
                if cl.get("observability") == "unobservable" and cl.get("hidden_property"):
                    out[k] = HEAD.format(n=nframes, task=task) + Q["hidden"][0].format(
                        hp=cl["hidden_property"].replace("_", " "),
                        hv=cl.get("hidden_value", ""), exp=cl.get("expected_outcome", ""))
            else:
                out[k] = h + q[k]
        return out
    out = {k: HEAD.format(n=nframes, task=task) + Q[k][0] for k in ("plaus", "action")}
    if cl.get("observability") == "unobservable" and cl.get("hidden_property"):
        out["hidden"] = HEAD.format(n=nframes, task=task) + Q["hidden"][0].format(
            hp=cl["hidden_property"].replace("_", " "), hv=cl.get("hidden_value", ""),
            exp=cl.get("expected_outcome", ""))
        out["hidden_nohint"] = HEAD.format(n=nframes, task=task) + Q["hidden_nohint"][0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default=None,
                    help="comma list of questions; output goes to a __variant file "
                         "so it never replaces the main run")
    ap.add_argument("--order", default="temporal", choices=["temporal", "shuffled"],
                    help="shuffled = the frame-order control: same frames, "
                         "random order fixed at staging")
    a = ap.parse_args()
    data, clips = load(a.data)
    only = a.only.split(",") if a.only else None
    jobs = [(cl, k, t) for cl in clips for k, t in prompts(cl, a.frames, only).items()]
    print(f"[{a.model}] {len(clips)} clips, {len(jobs)} calls", flush=True)
    res, t0 = {}, time.time()

    def store(cl, k, p):
        v, mass = ev(p, Q[k][1])
        if v is not None:
            res.setdefault(cl["clip_id"], {})[k] = v

    if a.api:
        from videophy_eval import _token_probs, client
        from domain_probes import frames_ordered
        c = client()
        cache = {}

        def work(j):
            cl, k, text = j
            try:
                imgs = cache.get(cl["clip_id"]) or frames_ordered(data, cl, a.frames, a.order)
                cache[cl["clip_id"]] = imgs
                store(cl, k, _token_probs(c, a.model, imgs, text))
            except Exception as e:  # noqa: BLE001
                print(f"  fail {cl['clip_id']} {k}: {str(e)[:70]}", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, jobs))
    else:
        import torch
        from videophy_eval_local import load_model
        from probes_local import frames_pil
        _, model, proc = load_model(a.model, a.device)
        tok = getattr(proc, "tokenizer", proc)
        last, imgs = None, None
        for i, (cl, k, text) in enumerate(jobs):
            if cl["clip_id"] != last:
                imgs, last = frames_pil(data, cl, a.frames), cl["clip_id"]
                if a.order == "shuffled":
                    idx = [i for i in cl["order_shuffled"] if i < len(imgs)]
                    imgs = [imgs[i] for i in idx]
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": text})
            inp = proc.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
            # some processors (Mistral-3) emit float32 pixels for bf16 weights
            for _k, _v in list(inp.items()):
                if hasattr(_v, "is_floating_point") and _v.is_floating_point():
                    inp[_k] = _v.to(model.dtype)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=1, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True)
            pr = torch.softmax(o.scores[0][0].float(), dim=-1)
            p = {}
            for d in Q[k][1]:
                for s in (d, " " + d):
                    e = tok.encode(s, add_special_tokens=False)
                    if e:
                        p[d] = p.get(d, 0.0) + float(pr[e[0]].item())
            store(cl, k, p)
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"   {i+1}/{len(jobs)} ({el:.0f}s, eta {el/(i+1)*(len(jobs)-i-1)/60:.0f}m)", flush=True)

    tag = a.model.replace("/", "_")
    out = data / (f"vlmplaus_{tag}__f{a.frames}"
                  + ("__shuffled" if a.order == "shuffled" else "")
                  + (f"__variant-{a.only.replace(',', '+')}" if a.only else "") + ".json")
    out.write_text(json.dumps({"model": a.model, "frames": a.frames, "order": a.order,
                               "n": len(res),
                               "scores": res}, indent=1))
    cells = sum(len(v) for v in res.values())
    print(f"\n  {cells}/{len(jobs)} answers ({time.time()-t0:.0f}s) -> {out}")
    if cells < 0.5 * len(jobs):
        sys.exit("ERROR: under half the questions produced a digit distribution")


if __name__ == "__main__":
    main()
