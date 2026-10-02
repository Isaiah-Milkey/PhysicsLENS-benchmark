"""
Verify whether OPEN-SOURCE VLMs can characterize video-model performance and
identify failure modes — across architecturally-distinct model families.

Usage:
  python vlm_multimodel_eval.py                       # default 3 models, JSON score
  python vlm_multimodel_eval.py --logprob             # also token-prob Yes/No AUC
  python vlm_multimodel_eval.py Qwen/Qwen2.5-VL-32B-Instruct OpenGVLab/InternVL3-14B-hf

Protocol: multi-frame (8 frames in temporal order, one holistic physics judgement),
which an earlier single-vs-multi study showed is the configuration that works.
Each model scores 5 AI-generated vs 5 real physics videos. We report, per model:
  • AI vs real mean suspicion + rank AUC (P[AI scored > real]) for the JSON score
  • with --logprob: the same AUC for the token-probability score, P(model answers
    "Yes" to "does this violate physics?") read from the next-token logits
Results merge into scripts/vlm_multimodel_results.json (per-model keys updated,
existing entries for other models preserved).
"""
import gc, json, re, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from tools.video import load_frames, sample_frames

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "OpenGVLab/InternVL3-8B-hf",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
]

VIDEOS = {
    "ai": [
        "test_videos/ai/sora/basketball-explosion.mp4",
        "test_videos/ai/sora/petri-dish-pandas.mp4",
        "test_videos/ai/sora/ships-in-coffee.mp4",
        "test_videos/ai/lumiere/beer_pouring.mp4",
        "test_videos/ai/videopoet/tidal_wave.mp4",
    ],
    "real": [
        "test_videos/real/physics_iq/balls-collide.mp4",
        "test_videos/real/physics_iq/ball-ramp.mp4",
        "test_videos/real/physics_iq/ball-and-block-fall.mp4",
        "test_videos/real/physics_iq/block-domino.mp4",
        "test_videos/real/wikimedia/bouncing_ball.webm",
    ],
}

PROMPT = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video. Judge whether the MOTION and INTERACTIONS across frames obey "
    "real-world physics (gravity, momentum, object permanence, rigid-body and fluid "
    "behavior). AI-generated videos often show objects that morph, appear/disappear, "
    "move without forces, or deform implausibly.\n"
    "Reply with VALID JSON ONLY - no markdown fences:\n"
    '{{"suspicion_score": <float 0.0-1.0>, "suspected_failure": "<brief label or null>", '
    '"explanation": "<one or two sentences>", "confidence": <float 0.0-1.0>}}'
)


def to_pil(frame_bgr, max_side=512):
    h, w = frame_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def parse_json(raw):
    raw = raw.replace("{{", "{").replace("}}", "}")  # some models echo literal {{ }}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"suspicion_score": None, "suspected_failure": None,
                "explanation": f"unparseable: {raw[:100]}", "confidence": 0.0}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate trailing junk / minor issues
        try:
            return json.loads(m.group(0)[:m.group(0).rfind("}") + 1])
        except Exception:
            return {"suspicion_score": None, "suspected_failure": None,
                    "explanation": f"badjson: {raw[:100]}", "confidence": 0.0}


def _inputs(model, processor, images, prompt):
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)


def ask(model, processor, images, prompt):
    inputs = _inputs(model, processor, images, prompt)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    return parse_json(raw)


YESNO_PROMPT = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video. AI-generated videos often show objects that morph, appear/disappear, "
    "float, pass through each other, or move without forces. Judging the MOTION and "
    "INTERACTIONS across frames: does this video violate real-world physics?\n"
    "Answer with exactly one word: Yes or No."
)


def ask_logprob(model, processor, images):
    """Token-probability score: P(Yes) / (P(Yes)+P(No)) from the next-token logits."""
    inputs = _inputs(model, processor, images,
                     YESNO_PROMPT.format(n=len(images)))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
    tok = getattr(processor, "tokenizer", processor)

    def mass(word):
        ids = set()
        for v in (word, " " + word, word.lower(), " " + word.lower(), word.upper()):
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return float(sum(probs[i].item() for i in ids))

    p_yes, p_no = mass("Yes"), mass("No")
    if p_yes + p_no < 1e-4:
        return None
    return p_yes / (p_yes + p_no)


def rank_auc(ai, real):
    pairs = [(a, r) for a in ai for r in real if a is not None and r is not None]
    if not pairs:
        return float("nan")
    wins = sum(1.0 if a > r else 0.5 if a == r else 0.0 for a, r in pairs)
    return round(wins / len(pairs), 3)


def eval_model(model_id, clips, do_logprob=False):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"\n### Loading {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    rows = []
    for label, name, imgs in clips:
        r = ask(model, processor, imgs, PROMPT.replace("{n}", str(len(imgs))))
        s = r.get("suspicion_score")
        s = float(s) if isinstance(s, (int, float)) else None
        lp = ask_logprob(model, processor, imgs) if do_logprob else None
        rows.append({"label": label, "video": name, "score": s,
                     "logprob_score": round(lp, 4) if lp is not None else None,
                     "failure": r.get("suspected_failure"),
                     "explanation": r.get("explanation")})
        lp_txt = f" lp={lp:.3f}" if lp is not None else ""
        print(f"  [{label:4s}] {name:28s} score={s}{lp_txt} fail={r.get('suspected_failure')}", flush=True)

    del model, processor
    gc.collect(); torch.cuda.empty_cache()

    def split(key):
        ai = [r[key] for r in rows if r["label"] == "ai"]
        re_ = [r[key] for r in rows if r["label"] == "real"]
        return ai, re_

    ai, re_ = split("score")
    ai_v = [x for x in ai if x is not None]; re_v = [x for x in re_ if x is not None]
    summary = {
        "ai_mean": round(float(np.mean(ai_v)), 3) if ai_v else None,
        "real_mean": round(float(np.mean(re_v)), 3) if re_v else None,
        "auc": rank_auc(ai, re_),
        "n_parsed": len(ai_v) + len(re_v),
    }
    if do_logprob:
        ai_l, re_l = split("logprob_score")
        summary["logprob_auc"] = rank_auc(ai_l, re_l)
        summary["logprob_ai_mean"] = round(float(np.mean([x for x in ai_l if x is not None])), 3) \
            if any(x is not None for x in ai_l) else None
        summary["logprob_real_mean"] = round(float(np.mean([x for x in re_l if x is not None])), 3) \
            if any(x is not None for x in re_l) else None
    print(f"  => AI {summary['ai_mean']} vs real {summary['real_mean']} | AUC {summary['auc']}"
          + (f" | logprob AUC {summary.get('logprob_auc')}" if do_logprob else ""), flush=True)
    return {"rows": rows, "summary": summary}


def main():
    args = [a for a in sys.argv[1:]]
    do_logprob = "--logprob" in args
    models = [a for a in args if not a.startswith("--")] or DEFAULT_MODELS

    # Pre-load all clips once (8 frames each)
    clips = []
    for label, paths in VIDEOS.items():
        for rel in paths:
            vp = ROOT / rel
            if not vp.exists():
                print(f"SKIP {rel}", flush=True); continue
            frames, _ = load_frames(str(vp))
            if not frames:
                print(f"SKIP unreadable {rel}", flush=True); continue
            imgs = [to_pil(f) for f in sample_frames(frames, 8)]
            clips.append((label, Path(rel).name, imgs))
    print(f"Loaded {len(clips)} clips. Models: {models} logprob={do_logprob}", flush=True)

    # Merge into the existing results file so per-model entries accumulate.
    out_path = Path(__file__).parent / "vlm_multimodel_results.json"
    out = {"protocol": "multi-frame (8 frames, temporal order)", "models": {}}
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text())
            out.setdefault("models", {})
        except Exception:
            pass

    for mid in models:
        try:
            out["models"][mid] = eval_model(mid, clips, do_logprob=do_logprob)
        except Exception as exc:
            print(f"  !! {mid} FAILED: {exc}", flush=True)
            out["models"][mid] = {"error": str(exc)}
        out_path.write_text(json.dumps(out, indent=2))   # save after each model

    print("\n==== SEPARATION SUMMARY (AUC = P[AI scored more suspicious than real]) ====", flush=True)
    for mid, res in out["models"].items():
        s = res.get("summary")
        if s:
            lp = f" logprobAUC={s['logprob_auc']}" if s.get("logprob_auc") is not None else ""
            print(f"  {mid:42s} AI={s['ai_mean']} real={s['real_mean']} AUC={s['auc']}{lp} (n={s['n_parsed']}/10)", flush=True)
        else:
            print(f"  {mid:42s} ERROR: {res.get('error')}", flush=True)


if __name__ == "__main__":
    main()
