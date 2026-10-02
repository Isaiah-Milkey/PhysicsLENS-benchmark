"""
Run the specialist defect probes on LOCAL open-weight VLMs.

Same PROBES list, same frozen frames, same P(Yes)/(P(Yes)+P(No)) token-probability
scoring as the API path in physics_probes.py — so per-specialist numbers are
directly comparable across hosted and local models rather than being a separate
incomparable eval.

Exists because the ablation needs several VLMs and the gateway is the bottleneck:
the local models cost nothing and run on an idle GPU in parallel with the API runs.

Usage:
  python backend/scripts/probes_local.py --model qwen2.5-vl-7b --device cuda:1
  python backend/scripts/probes_local.py --model internvl3-8b --limit 400
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman                       # noqa: E402
from videophy_eval import auc_extremes                       # noqa: E402
from videophy_eval_local import MODELS, load_model           # noqa: E402
from physics_probes import PROBES                            # noqa: E402
from gepa_optimize import load                               # noqa: E402


def frames_pil(data: Path, clip: dict, k: int = 8):
    from PIL import Image
    idx = clip["order_temporal"]
    if k < len(idx):
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, k).astype(int)]
    d = data / "frames" / clip["clip_id"]
    return [Image.open(d / f"{i:03d}.jpg").convert("RGB")
            for i in idx if (d / f"{i:03d}.jpg").exists()]


def yes_no(torch, model, proc, imgs, question):
    """P(Yes)/(P(Yes)+P(No)) from the first generated token.

    Surface forms are SUMMED, not assigned — "Yes", " Yes" and "YES" are distinct
    token ids and letting one overwrite another was a real bug in the API path
    (it produced yes=7e-08 on every clip).
    """
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
    tot = {}
    for word in ("Yes", "No"):
        s, seen = 0.0, set()
        for v in (word, " " + word, word.lower(), " " + word.lower(), word.upper()):
            enc = tok.encode(v, add_special_tokens=False)
            if enc and enc[0] not in seen:
                seen.add(enc[0])
                s += float(probs[enc[0]].item())
        tot[word.lower()] = s
    y, n = tot["yes"], tot["no"]
    return y / (y + n) if (y + n) > 1e-6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--model", default="qwen2.5-vl-7b", choices=list(MODELS))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    data, clips = load(a.data)
    if a.limit:
        clips = clips[:a.limit]
    torch, model, proc = load_model(a.model, a.device)

    print(f"probes [{a.model}]: {len(clips)} clips x {len(PROBES)} probes",
          flush=True)
    t0 = time.time()
    out = {}
    for i, cl in enumerate(clips):
        imgs = frames_pil(data, cl, 8)
        r = {}
        for name, q in PROBES:
            try:
                v = yes_no(torch, model, proc, imgs,
                           q.format(n=len(imgs), caption=cl.get("caption", "")))
                if v is not None:
                    r[name] = v
            except Exception as e:  # noqa: BLE001
                print(f"    {name} fail {cl['clip_id'][:28]}: {str(e)[:50]}",
                      file=sys.stderr)
        out[cl["clip_id"]] = r
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"    {i+1}/{len(clips)}  ({el:.0f}s, eta "
                  f"{el/(i+1)*(len(clips)-i-1)/60:.0f}m)", flush=True)

    scores = {k: (float(np.mean(list(v.values()))) if v else None)
              for k, v in out.items()}
    outp = data / f"method_probes_{a.model}.json"
    outp.write_text(json.dumps({"method": "probes", "model": a.model,
                                "local": True, "n": len(clips),
                                "seconds": round(time.time() - t0, 1),
                                "scores": scores, "probe_scores": out}, indent=1))

    sub = [(scores[c["clip_id"]], c["pc"]) for c in clips
           if scores.get(c["clip_id"]) is not None]
    if len(sub) > 8:
        xs = [s for s, _ in sub]
        r = spearman(xs, [6 - p for _, p in sub])
        auc, _, _ = auc_extremes(xs, [p for _, p in sub])
        print(f"\n  [{a.model}] probe-mean rho={r:+.3f} AUC={auc or 0:.3f} "
              f"(n={len(sub)})")
    print(f"  -> {outp}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
