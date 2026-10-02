"""VLM suspicion scoring via OpenRouter."""
import base64, json, re
from typing import Any, Dict, List

import cv2
import numpy as np

# Fixed seed so repeated calls on the same frames are reproducible.
SUSPICION_SEED = 42

# Map of friendly key → OpenRouter model ID. Only vision-capable models that
# accept multiple images in one request belong here (the suspicion payload sends
# all sampled frames as image blocks in a single message). Model IDs verified
# against the live OpenRouter /models list.
#
# Reliability note (from the matched real/AI A/B eval, 2026-06): default is
# gemini-2.5-pro (user choice) — after the truncation fix it separated the matched
# pair cleanly (real 0.0 / AI 1.0) with real explanations. gpt-4o is the most
# calibrated alternate; gemini-flash tends to under-call (≈0 for everything) and
# claude-sonnet gives rich explanations but can over-flag real footage.
OPENROUTER_MODELS: Dict[str, str] = {
    "gpt-4o":            "openai/gpt-4o",
    "gpt-4.1":           "openai/gpt-4.1",
    "gpt-5.1":           "openai/gpt-5.1",
    "gemini-2.5-flash":  "google/gemini-2.5-flash",
    "gemini-2.5-pro":    "google/gemini-2.5-pro",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
}

SUSPICION_PROMPT = (
    "You are a physics expert reviewing a video frame.\n\n"
    "Does the interaction shown violate physical common sense?\n\n"
    "Reply with VALID JSON ONLY — no markdown fences, no extra text:\n"
    '{"suspicion_score": <float 0.0-1.0>, '
    '"suspected_failure": "<brief label or null>", '
    '"time_interval": "<rough description or null>", '
    '"explanation": "<one sentence>", '
    '"confidence": <float 0.0-1.0>}'
)

# Multi-frame temporal prompt. Physics violations live in MOTION across frames,
# not in any single still — so all sampled frames go to the model in one call and
# get one holistic judgement (validated in scripts/vlm_failure_mode_eval.py,
# AI-vs-real AUC 0.90 vs 0.58 for single-frame).
#
# This prompt is the result of an A/B eval on a matched real/AI video set. Two
# things mattered: (1) telling the model frames are SPARSELY sampled — otherwise
# it reads the large position jump between consecutive frames as "teleportation"
# and false-flags real footage (observed: a real ball-in-basket clip jumping from
# 0.10→0.00 once this clause was added); (2) explicit score anchors, so different
# models land on a comparable scale instead of defaulting to 0 or 1.
SUSPICION_PROMPT_MULTI = (
    "You are a physics expert deciding whether a short video is REAL footage or "
    "AI-GENERATED (AI clips often break physics).\n\n"
    "You are given {n} still frames taken at EQUAL time steps from one video, in "
    "chronological order. They are SPARSELY sampled: an object can legitimately "
    "move a large distance between two consecutive frames. A large but smooth, "
    "consistent shift of position is NORMAL and is NOT teleportation — do NOT "
    "flag it. Infer the underlying trajectory across all frames before judging.\n\n"
    "Flag the video as suspicious ONLY for genuine physics violations:\n"
    "- Object permanence: things appear, vanish, split, or merge with no cause\n"
    "- Gravity/trajectory: floating, wrong acceleration, motion against gravity, impossible paths\n"
    "- Momentum/collisions: bounces that gain energy, objects passing through each other, motion with no cause\n"
    "- Rigid bodies: solid objects morph, melt, warp, or change size/shape without a force\n"
    "- Fluids/soft bodies behaving impossibly; textures, identities, or object counts that flicker or swap\n\n"
    "Score calibration (use the full range; reserve extremes for clear cases):\n"
    "  0.0-0.2  plausible, consistent with real footage\n"
    "  0.3-0.5  mostly plausible, minor oddities\n"
    "  0.6-0.8  at least one clear physics violation\n"
    "  0.9-1.0  multiple or blatant violations\n\n"
    "Reply with VALID JSON ONLY - no markdown, no extra text. Put the score and "
    "explanation FIRST and keep the explanation to one or two sentences:\n"
    '{"suspicion_score": <float 0.0-1.0>, '
    '"explanation": "<one or two sentences citing concrete visual evidence>", '
    '"suspected_failure": "<brief label or null>", '
    '"confidence": <float 0.0-1.0>, '
    '"violations": [<short strings, [] if none>]}'
)


def frame_to_b64(frame: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _extract_fields(raw: str) -> Dict[str, Any]:
    """Pull individual fields out of malformed/truncated JSON by regex.

    Verbose or reasoning models sometimes exceed max_tokens and get cut off
    before the closing brace — the score is present but json.loads fails. This
    recovers the numeric score (and any quoted string fields) field-by-field so
    a real verdict isn't thrown away as "unparseable".
    """
    out: Dict[str, Any] = {}
    sm = re.search(r'"suspicion_score"\s*:\s*(-?\d*\.?\d+)', raw)
    if sm:
        out["suspicion_score"] = float(sm.group(1))
    cm = re.search(r'"confidence"\s*:\s*(-?\d*\.?\d+)', raw)
    if cm:
        out["confidence"] = float(cm.group(1))
    for key in ("suspected_failure", "explanation"):
        km = re.search(rf'"{key}"\s*:\s*"([^"]*)"', raw)
        if km:
            out[key] = km.group(1)
        else:
            # Truncated mid-string (opening quote, no closing quote): keep what we
            # have so a real explanation isn't lost to "recovered from truncated
            # reply". Verbose/reasoning models (e.g. gemini-2.5-pro) hit this.
            km2 = re.search(rf'"{key}"\s*:\s*"([^"]+)$', raw)
            if km2 and km2.group(1).strip():
                out[key] = km2.group(1).strip()
    return out


def parse_vlm_json(raw: str) -> Dict[str, Any]:
    """Extract the verdict from a model reply, robustly.

    Tries, in order: (1) a complete JSON object (plain, ```json fenced, or
    prose-wrapped), (2) field-by-field regex recovery for truncated/malformed
    JSON. Never raises — if no score can be found, returns suspicion_score=None
    so the caller treats it as "no signal" rather than crashing the pipeline.
    """
    raw = raw or ""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: recover fields from truncated / malformed JSON.
    recovered = _extract_fields(raw)
    if recovered.get("suspicion_score") is not None:
        recovered.setdefault("suspected_failure", None)
        recovered.setdefault("explanation", "recovered from truncated reply")
        recovered.setdefault("confidence", 0.0)
        return recovered
    return {"suspicion_score": None, "suspected_failure": None,
            "explanation": f"unparseable: {raw[:120]}", "confidence": 0.0}


def build_suspicion_payload(
    frames: List[np.ndarray],
    model_id: str,
    *,
    seed: int = SUSPICION_SEED,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """OpenRouter payload: all frames in ONE message, deterministic decoding.

    Frames are sent as `image_url` blocks, NOT as a single `video_url`. This is
    deliberate: OpenRouter does support native video, but only on the Gemini /
    Qwen / Nova families — the strongest physics judges here (GPT-4o etc.) are
    image-only, and native video-LM calls are markedly more expensive. Sampling
    a handful of frames keeps cost low and lets us use the best image models. If
    native video is ever wanted, send {"type": "video_url", "video_url": {"url":
    "data:video/mp4;base64,..."}} and restrict to a video-capable model.
    """
    content: List[Dict[str, Any]] = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{frame_to_b64(fr)}"}}
        for fr in frames
    ]
    content.append({"type": "text",
                    "text": SUSPICION_PROMPT_MULTI.replace("{n}", str(len(frames)))})
    # temperature=0 (universal) + seed (best-effort, ignored by providers that
    # don't support it) drive determinism. We deliberately do NOT send
    # response_format: many OpenRouter models 400 on it, and parse_vlm_json
    # already recovers JSON from prose — robustness over JSON-mode.
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
    }


async def chat_vision(
    prompt: str,
    frame_bgr: np.ndarray,
    model_key: str = "gpt-4o",
    api_key: str = "",
    timeout: float = 60.0,
) -> str:
    """Generic single-image OpenRouter chat → reply text (no fixed prompt).

    Used by tools.vlm_router so any pipeline can send an arbitrary prompt +
    image to an OpenRouter model. `model_key` may be a friendly key from
    OPENROUTER_MODELS or a full "vendor/model" id (passed through).
    """
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp not installed — pip install aiohttp")

    model_id = OPENROUTER_MODELS.get(model_key, model_key)
    payload = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{frame_to_b64(frame_bgr)}"}},
            ],
        }],
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://physicslens.local"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            data = await resp.json()
    if "choices" not in data:
        raise RuntimeError(f"OpenRouter error: {str(data)[:300]}")
    return data["choices"][0]["message"]["content"]


async def score_frame(
    frame: np.ndarray,
    model_key: str = "gpt-4o",
    api_key: str = "",
) -> Dict[str, Any]:
    """Send one frame to OpenRouter and parse the JSON response."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp not installed — pip install aiohttp")

    model_id = OPENROUTER_MODELS.get(model_key, OPENROUTER_MODELS["gpt-4o"])
    payload = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": SUSPICION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_to_b64(frame)}"},
                },
            ],
        }],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://physicslens.local",
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(total=45),
        ) as resp:
            data = await resp.json()

    raw = data["choices"][0]["message"]["content"]
    return parse_vlm_json(raw)


async def score_frames(
    frames: List[np.ndarray],
    model_key: str = "gpt-4o",
    api_key: str = "",
) -> Dict[str, Any]:
    """Send all frames in ONE multi-frame call and parse the holistic verdict."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp not installed — pip install aiohttp")

    model_id = OPENROUTER_MODELS.get(model_key, OPENROUTER_MODELS["gpt-4o"])
    payload  = build_suspicion_payload(frames, model_id)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://physicslens.local",
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as resp:
            data = await resp.json()

    raw = data["choices"][0]["message"]["content"]
    return parse_vlm_json(raw)
