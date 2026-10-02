"""
Score LOCAL VLMs on the staged VideoPhy-2 subset — same frozen frames, same
prompts, same metrics as videophy_eval.py, so API and local numbers are
directly comparable.

Exists because (a) the API gateway rate-limits hard, and (b) the best-measured
judges so far are local (InternVL3-8B, Qwen2.5-VL-7B). Also settles the
"bigger is worse" question at n=300 — the original claim rested on n=10, where
the confidence intervals overlapped from 0.24 to 1.00 and supported nothing.

Usage:
  python backend/scripts/videophy_eval_local.py --model internvl3-8b --device cuda:1
  python backend/scripts/videophy_eval_local.py --model qwen2.5-vl-7b --order shuffled
  python backend/scripts/videophy_eval_local.py --model internvl3-8b --prompt likert
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "backend"))
from vlm_rapidata_eval import spearman                       # noqa: E402
from videophy_eval import (BINARY_Q, CAPTION_Q, LIKERT_Q,    # noqa: E402
                           auc_extremes, boot_ci)

MODELS = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl3-8b":  "OpenGVLab/InternVL3-8B-hf",
    "internvl3-14b": "OpenGVLab/InternVL3-14B-hf",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    # third local family (neither Qwen nor InternVL) — architectural
    # diversity is what could make per-specialist judge choice real
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    # new families (Sep 23) — architectural diversity, not more of the same
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma3-12b": "google/gemma-3-12b-it",
    "llava-ov-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "idefics3-8b": "HuggingFaceM4/Idefics3-8B-Llama3",
    "mistral-small-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",

}


def load_model(key, device):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    hf = MODELS.get(key, key)
    print(f"[local] loading {hf} on {device} …", flush=True)
    proc = AutoProcessor.from_pretrained(hf)
    model = AutoModelForImageTextToText.from_pretrained(
        hf, dtype=torch.bfloat16, device_map=device).eval()
    return torch, model, proc


def frames_pil(data: Path, clip: dict, order: str):
    from PIL import Image
    idx = clip["order_shuffled"] if order == "shuffled" else clip["order_temporal"]
    d = data / "frames" / clip["clip_id"]
    return [Image.open(d / f"{i:03d}.jpg").convert("RGB")
            for i in idx if (d / f"{i:03d}.jpg").exists()]


def token_probs(torch, model, proc, imgs, question):
    """{token -> summed prob} for the first generated token. Same summing rule
    as the API path: surface forms of one word must accumulate, not overwrite."""
    content = [{"type": "image", "image": im} for im in imgs]
    content.append({"type": "text", "text": question})
    inputs = proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
    tok = getattr(proc, "tokenizer", proc)
    res = {}
    for word in ("Yes", "No", "1", "2", "3", "4", "5"):
        total = 0.0
        seen = set()
        for v in (word, " " + word, word.lower(), " " + word.lower(), word.upper()):
            enc = tok.encode(v, add_special_tokens=False)
            if enc and enc[0] not in seen:
                seen.add(enc[0])
                total += float(probs[enc[0]].item())
        res[word.lower()] = total
    return res


def score(torch, model, proc, imgs, clip, prompt):
    n = len(imgs)
    if prompt in ("binary", "caption"):
        q = (CAPTION_Q if prompt == "caption" else BINARY_Q).format(
            n=n, caption=clip.get("caption", ""))
        p = token_probs(torch, model, proc, imgs, q)
        y, no = p["yes"], p["no"]
        return y / (y + no) if (y + no) > 1e-6 else None
    if prompt == "likert":
        p = token_probs(torch, model, proc, imgs, LIKERT_Q.format(n=n))
        mass = {int(k): p[k] for k in "12345" if p.get(k, 0) > 0}
        tot = sum(mass.values())
        if tot < 1e-6:
            return None
        ev = sum(k * v for k, v in mass.items()) / tot
        return (5.0 - ev) / 4.0
    raise ValueError(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy300")
    ap.add_argument("--model", default="internvl3-8b")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--order", default="temporal", choices=["temporal", "shuffled"])
    ap.add_argument("--prompt", default="binary", choices=["binary", "caption", "likert"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    data = (ROOT / a.data) if not Path(a.data).is_absolute() else Path(a.data)
    meta = json.loads((data / "manifest.json").read_text())
    clips = meta["clips"][:a.limit] if a.limit else meta["clips"]
    torch, model, proc = load_model(a.model, a.device)

    print(f"\n=== {a.model} | prompt={a.prompt} | order={a.order} | {len(clips)} clips",
          flush=True)
    t0 = time.time()
    scores = []
    for i, cl in enumerate(clips):
        try:
            scores.append(score(torch, model, proc,
                                frames_pil(data, cl, a.order), cl, a.prompt))
        except Exception as e:  # noqa: BLE001
            print(f"    fail {cl['clip_id'][:40]}: {str(e)[:70]}", file=sys.stderr)
            scores.append(None)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(clips)} ({time.time()-t0:.0f}s)", flush=True)

    pcs = [c["pc"] for c in clips]
    ok = [(s, 6 - p, p) for s, p in zip(scores, pcs) if s is not None]
    xs = [o[0] for o in ok]
    ys = [o[1] for o in ok]
    rho = spearman(xs, ys)
    lo, hi = boot_ci(lambda r: spearman(*zip(*[(xs[k], ys[k]) for k in
                     (r.randrange(len(xs)) for _ in xs)])))
    auc, nb, ng = auc_extremes([o[0] for o in ok], [o[2] for o in ok])
    res = {"model": a.model, "prompt": a.prompt, "order": a.order, "local": True,
           "n": len(clips), "n_scored": len(ok), "spearman": round(rho, 3),
           "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
           "auc_pc12_vs_pc45": round(auc, 3) if auc else None,
           "distinct_scores": len(set(xs)),
           "mean_score": round(float(np.mean(xs)), 3),
           "seconds": round(time.time() - t0, 1),
           "scores": {c["clip_id"]: s for c, s in zip(clips, scores)}}
    outp = data / f"scores_{a.model}_{a.prompt}_{a.order}.json"
    outp.write_text(json.dumps(res, indent=1))
    ci = f"[{res['ci95'][0]:+.2f}, {res['ci95'][1]:+.2f}]" if res["ci95"] else "-"
    print(f"\n  rho : {rho:+.3f}  {ci}")
    print(f"  AUC : {res['auc_pc12_vs_pc45']}  ({nb} bad / {ng} good)")
    print(f"  distinct {res['distinct_scores']}/{len(ok)}   mean {res['mean_score']}")
    print(f"  -> {outp}\n")


if __name__ == "__main__":
    main()
