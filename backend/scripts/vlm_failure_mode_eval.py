"""
Empirical test: can a VLM characterize video-model physics performance and
identify failure modes?

Runs Qwen2.5-VL-7B-Instruct locally (no API key needed) over labeled
AI-generated vs real test videos, in two conditions:

  1. single-frame — the exact approach of pipelines/stage1/vlm_suspicion.py:
     score 5 uniformly-sampled keyframes independently, aggregate mean/peak.
  2. multi-frame  — 8 frames in one prompt with temporal instructions,
     one holistic suspicion score + failure-mode label.

Outputs per-video scores, AI-vs-real separation (mean gap + rank AUC), and
the failure labels the VLM produced, to scripts/vlm_eval_results.json.
"""
import json, re, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from tools.video import load_frames, sample_frames

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

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

# Same intent as tools/vlm.py SUSPICION_PROMPT (single frame)
SINGLE_PROMPT = (
    "You are a physics expert reviewing a video frame.\n\n"
    "Does the interaction shown violate physical common sense?\n\n"
    "Reply with VALID JSON ONLY - no markdown fences, no extra text:\n"
    '{"suspicion_score": <float 0.0-1.0>, '
    '"suspected_failure": "<brief label or null>", '
    '"explanation": "<one sentence>", '
    '"confidence": <float 0.0-1.0>}'
)

MULTI_PROMPT = (
    "You are a physics expert. These {n} frames are sampled in temporal order "
    "from one video. Judge whether the MOTION and INTERACTIONS across frames "
    "obey real-world physics (gravity, momentum, object permanence, rigid-body "
    "behavior, fluid behavior). AI-generated videos often show objects that "
    "morph, appear/disappear, move without forces, or deform implausibly.\n\n"
    "Reply with VALID JSON ONLY - no markdown fences, no extra text:\n"
    '{"suspicion_score": <float 0.0-1.0 that this video violates physics>, '
    '"suspected_failure": "<brief label or null>", '
    '"explanation": "<one or two sentences>", '
    '"confidence": <float 0.0-1.0>}'
)


def to_pil(frame: np.ndarray, max_side: int = 640) -> Image.Image:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"suspicion_score": None, "suspected_failure": None,
                "explanation": f"unparseable: {raw[:120]}", "confidence": 0.0}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"suspicion_score": None, "suspected_failure": None,
                "explanation": f"bad json: {raw[:120]}", "confidence": 0.0}


def ask(model, processor, images: list, prompt: str) -> dict:
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    return parse_json(raw)


def rank_auc(ai_scores: list, real_scores: list) -> float:
    """P(random AI video scores higher than random real video)."""
    pairs = [(a, r) for a in ai_scores for r in real_scores
             if a is not None and r is not None]
    if not pairs:
        return float("nan")
    wins = sum(1.0 if a > r else 0.5 if a == r else 0.0 for a, r in pairs)
    return wins / len(pairs)


def main():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"Loading {MODEL_ID}…", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    print("Model loaded.", flush=True)

    results = []
    for label, paths in VIDEOS.items():
        for rel in paths:
            vp = ROOT / rel
            if not vp.exists():
                print(f"SKIP missing {rel}", flush=True)
                continue
            frames, fps = load_frames(str(vp))
            if not frames:
                print(f"SKIP unreadable {rel}", flush=True)
                continue

            # Condition 1: single-frame x5 (pipeline approach)
            kf5 = sample_frames(frames, 5)
            single = []
            for i, f in enumerate(kf5):
                r = ask(model, processor, [to_pil(f)], SINGLE_PROMPT)
                single.append(r)
                print(f"  [{label}] {Path(rel).name} frame {i+1}/5: "
                      f"score={r.get('suspicion_score')} "
                      f"fail={r.get('suspected_failure')}", flush=True)

            # Condition 2: multi-frame x8, one shot
            kf8 = [to_pil(f) for f in sample_frames(frames, 8)]
            multi = ask(model, processor, kf8,
                        MULTI_PROMPT.replace("{n}", str(len(kf8))))
            print(f"  [{label}] {Path(rel).name} MULTI: "
                  f"score={multi.get('suspicion_score')} "
                  f"fail={multi.get('suspected_failure')} "
                  f"expl={multi.get('explanation')}", flush=True)

            ss = [r.get("suspicion_score") for r in single
                  if isinstance(r.get("suspicion_score"), (int, float))]
            results.append({
                "video": rel, "label": label, "n_frames": len(frames),
                "single_scores": [r.get("suspicion_score") for r in single],
                "single_mean": float(np.mean(ss)) if ss else None,
                "single_peak": float(np.max(ss)) if ss else None,
                "single_failures": [r.get("suspected_failure") for r in single],
                "multi_score": multi.get("suspicion_score"),
                "multi_failure": multi.get("suspected_failure"),
                "multi_explanation": multi.get("explanation"),
                "multi_confidence": multi.get("confidence"),
            })

    # ── Separation summary ────────────────────────────────────────────────────
    def split(key):
        ai = [r[key] for r in results if r["label"] == "ai"]
        re_ = [r[key] for r in results if r["label"] == "real"]
        return ai, re_

    summary = {}
    for key in ("single_mean", "single_peak", "multi_score"):
        ai, re_ = split(key)
        ai_v = [v for v in ai if v is not None]
        re_v = [v for v in re_ if v is not None]
        summary[key] = {
            "ai_mean": float(np.mean(ai_v)) if ai_v else None,
            "real_mean": float(np.mean(re_v)) if re_v else None,
            "auc": rank_auc(ai, re_),
        }

    out = {"model": MODEL_ID, "results": results, "summary": summary}
    out_path = Path(__file__).parent / "vlm_eval_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
