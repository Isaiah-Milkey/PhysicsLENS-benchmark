"""
Async client for a standard OpenAI-compatible chat API — used by Stage 1/3/4
for VLM adjudication and by the Stage 4 report's text summary.

Credentials come from .env at the PhysicsLENS root:
  OPENAI_API_KEY, OPENAI_BASE_URL (optional — omit to hit api.openai.com;
  set it to point at any other OpenAI-compatible endpoint instead)
These are the same names the `openai` package itself reads by default, so a
plain `openai.OpenAI()` picks them up with no extra plumbing.
(scripts/openai_vision.py is the sync/manual-test counterpart.)
"""
import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass                                    # fall back to plain os.environ

DEFAULT_MODEL = "gpt-4o-mini"


def frame_to_data_url(frame_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def credentials() -> tuple[Optional[str], Optional[str]]:
    return os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_BASE_URL")


def _client(token: Optional[str], base_url: Optional[str]):
    from openai import AsyncOpenAI
    env_token, env_base = credentials()
    token = token or env_token
    base_url = base_url or env_base or None  # None -> SDK default (api.openai.com)
    if not token:
        raise RuntimeError(
            "OpenAI API key missing — enter one in the pipeline's API key "
            "field, or set OPENAI_API_KEY in the PhysicsLENS .env file."
        )
    return AsyncOpenAI(api_key=token, base_url=base_url)


async def query_vision(query: str, frame_bgr: np.ndarray, *,
                       model: str = DEFAULT_MODEL,
                       system_prompt: Optional[str] = None,
                       thinking_level: Optional[str] = None,
                       timeout_s: float = 60.0,
                       token: Optional[str] = None,
                       base_url: Optional[str] = None) -> Dict[str, Any]:
    """One image + prompt → chat completion, as a dict `response_text()` reads.

    `token`/`base_url` override the .env credentials when provided (e.g. a key
    entered in the UI); otherwise they fall back to OPENAI_API_KEY /
    OPENAI_BASE_URL. `thinking_level` is accepted for interface parity with
    earlier callers but has no standard Chat Completions equivalent, so it
    is currently a no-op. Raises RuntimeError on missing credentials or an API
    error.
    """
    client = _client(token, base_url)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": query},
        {"type": "image_url", "image_url": {"url": frame_to_data_url(frame_bgr)}},
    ]})
    resp = await client.chat.completions.create(
        model=model, messages=messages, timeout=timeout_s)
    return {"response": resp.choices[0].message.content or ""}


async def query_vision_token_probs(prompt: str, frames_bgr: list, *,
                                   model: str = DEFAULT_MODEL,
                                   top_logprobs: int = 20,
                                   timeout_s: float = 60.0,
                                   token: Optional[str] = None,
                                   base_url: Optional[str] = None) -> Dict[str, float]:
    """Multiple images + prompt → {token: probability} for the first
    generated token, read from top_logprobs rather than a greedy decode.
    Used for forced-choice (MCQ) scoring, where the answer is exactly one
    option letter and its probability is directly informative.

    `top_logprobs` must stay at 20: at lower values some OpenAI-compatible
    gateways return a malformed distribution (probabilities not summing to
    1, competing tokens assigned identical mass) — verified against the
    gateway this was built for. Tokens are summed by their stripped,
    uppercased surface form, since "A", " A" and "a" are the same answer and
    a plain overwrite would let whichever form is emitted last erase the
    real mass.
    """
    import math
    client = _client(token, base_url)
    content = [{"type": "image_url", "image_url": {"url": frame_to_data_url(f)}}
              for f in frames_bgr]
    content.append({"type": "text", "text": prompt})
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        max_tokens=1, temperature=0, logprobs=True, top_logprobs=top_logprobs,
        timeout=timeout_s)
    lp = resp.choices[0].logprobs
    if not lp or not lp.content:
        return {}
    out: Dict[str, float] = {}
    for t in lp.content[0].top_logprobs:
        key = t.token.strip().upper()
        if key:
            out[key] = out.get(key, 0.0) + math.exp(t.logprob)
    return out


async def query_text(query: str, *,
                     model: str = DEFAULT_MODEL,
                     system_prompt: Optional[str] = None,
                     thinking_level: Optional[str] = None,
                     timeout_s: float = 90.0,
                     token: Optional[str] = None,
                     base_url: Optional[str] = None) -> Dict[str, Any]:
    """Text-only prompt → chat completion, as a dict `response_text()` reads.

    Same credential handling as `query_vision`, minus the image.
    """
    client = _client(token, base_url)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    resp = await client.chat.completions.create(
        model=model, messages=messages, timeout=timeout_s)
    return {"response": resp.choices[0].message.content or ""}


_SUBJECTS_PROMPT = (
    "The image shows sampled frames (side by side) from one video. If a tile "
    "is labeled MOTION, it is a zoomed crop of the region with the MOST "
    "MOTION in the video — the moving object shown there is the most "
    "important subject: name it FIRST, as a specific visual noun phrase.\n"
    "Then list the other primary physical objects — moving/acting subjects, "
    "then key interacting surfaces. Skip pure background. Use short visual "
    'noun phrases a segmentation model can ground (e.g. "basketball", '
    '"wooden crate"). At most {k}. Reply with ONLY strict JSON: '
    '{{"subjects": ["...", "..."]}}'
)


async def name_subjects(frames_bgr: list, *, max_subjects: int = 3,
                        model: str = DEFAULT_MODEL,
                        motion_crop=None) -> list[str]:
    """Ask the VLM to name the primary subjects across sampled frames.

    `frames_bgr`: 1–2 frames (e.g. first + middle) tiled side by side so
    subjects that only appear mid-action are still named. `motion_crop`, if
    given, is appended as a labeled MOTION tile and the prompt requires the
    moving object shown there to be named first — this keeps a fast-moving
    subject (which static frames under-represent) from being missed. Returns
    a list of short noun phrases (may be empty). Raises RuntimeError on
    missing credentials / API failure — callers degrade to their fallback.
    """
    from tools.vlm import parse_vlm_json

    tiles = list(frames_bgr) if isinstance(frames_bgr, list) else [frames_bgr]
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (max(2, int(t.shape[1] * h / t.shape[0])), h))
             for t in tiles]
    if motion_crop is not None and motion_crop.size:
        mc = cv2.resize(motion_crop,
                        (max(2, int(motion_crop.shape[1] * h / motion_crop.shape[0])), h))
        cv2.rectangle(mc, (0, 0), (mc.shape[1] - 1, mc.shape[0] - 1), (0, 0, 255), 4)
        cv2.putText(mc, "MOTION", (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(mc)
    gap = np.full((h, 12, 3), 255, np.uint8)
    composite = tiles[0]
    for t in tiles[1:]:
        composite = np.concatenate([composite, gap, t], axis=1)

    data = await query_vision(_SUBJECTS_PROMPT.format(k=max_subjects),
                              composite, model=model)
    parsed = parse_vlm_json(response_text(data) or "")
    subjects = parsed.get("subjects") if isinstance(parsed, dict) else None
    if not isinstance(subjects, list):
        return []
    out = [str(s).strip() for s in subjects if str(s).strip()]
    return out[:max_subjects]


def response_text(data: Dict[str, Any]) -> str:
    """Best-effort extraction of the model's text from a query_* response."""
    for key in ("response", "answer", "output", "text", "result", "content"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = response_text(v)
            if inner:
                return inner
    return ""
