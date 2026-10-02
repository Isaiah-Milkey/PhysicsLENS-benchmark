"""
Unified VLM router — one model key selects both provider and model.
-------------------------------------------------------------------
Every VLM-using pipeline offers a single provider-tagged model dropdown and one
"API key" field. The dropdown value encodes the provider (``openai:...`` or
``openrouter:...``); this module routes the call to the right backend:

  * OpenAI     → tools.llm_api  (standard OpenAI-compatible chat API; key from
                                 setting or OPENAI_API_KEY in .env, base URL
                                 from OPENAI_BASE_URL — omit to hit
                                 api.openai.com, or point it at any other
                                 OpenAI-compatible endpoint)
  * OpenRouter → tools.vlm      (openrouter.ai; key from setting or
                                 OPENROUTER_API_KEY)

Pipelines call `ask_vision_json(prompt, image_bgr, model_key, api_key)` and get
parsed JSON back regardless of provider. `key_status()` reports whether the
selected provider has usable credentials (so a pipeline can degrade cleanly).
"""
import os
from typing import Any, Optional

import cv2
import numpy as np

# ── Registry: dropdown value → provider + provider-native model id ────────────
# Keep labels in "<Model> — <Provider>" form so the single dropdown reads clearly.
_MODELS: dict[str, dict] = {
    # OpenAI — key/base URL from .env by default, no per-request override needed.
    "openai:gpt-4o-mini":           {"provider": "openai", "model": "gpt-4o-mini",
                                     "label": "GPT-4o mini — OpenAI"},
    "openai:gpt-4o":                {"provider": "openai", "model": "gpt-4o",
                                     "label": "GPT-4o — OpenAI"},
    # OpenRouter — "model" is the friendly key tools.vlm maps to a full id.
    "openrouter:gemini-2.5-flash":  {"provider": "openrouter", "model": "gemini-2.5-flash",
                                     "label": "Gemini 2.5 Flash — OpenRouter"},
    "openrouter:gemini-2.5-pro":    {"provider": "openrouter", "model": "gemini-2.5-pro",
                                     "label": "Gemini 2.5 Pro — OpenRouter"},
    "openrouter:gpt-4o":            {"provider": "openrouter", "model": "gpt-4o",
                                     "label": "GPT-4o — OpenRouter"},
    "openrouter:claude-sonnet-4.5": {"provider": "openrouter", "model": "claude-sonnet-4.5",
                                     "label": "Claude Sonnet 4.5 — OpenRouter"},
}

DEFAULT_MODEL_KEY = "openai:gpt-4o-mini"


def model_options() -> list[dict]:
    """UI select options for a provider-tagged VLM model dropdown."""
    return [{"value": k, "label": v["label"]} for k, v in _MODELS.items()]


def resolve(model_key: str) -> tuple[str, str]:
    """(provider, provider_model) for a dropdown value. Tolerant of a bare
    OpenAI model name (e.g. "gpt-4o") and explicit "provider:model"."""
    if model_key in _MODELS:
        e = _MODELS[model_key]
        return e["provider"], e["model"]
    if model_key.startswith("openrouter:"):
        return "openrouter", model_key.split(":", 1)[1]
    if model_key.startswith("openai:"):
        return "openai", model_key.split(":", 1)[1]
    # Legacy bare key: a known OpenRouter friendly id routes to OpenRouter;
    # anything else is treated as an OpenAI model name.
    try:
        from tools.vlm import OPENROUTER_MODELS
        if model_key in OPENROUTER_MODELS:
            return "openrouter", model_key
    except Exception:                                    # noqa: BLE001
        pass
    return "openai", (model_key or DEFAULT_MODEL_KEY.split(":", 1)[1])


def key_status(model_key: str, api_key: str = "") -> tuple[bool, str]:
    """(has_credentials, human description) for the selected model's provider."""
    provider, _ = resolve(model_key)
    if provider == "openrouter":
        ok = bool(api_key or os.environ.get("OPENROUTER_API_KEY"))
        return ok, "OpenRouter key (settings field or OPENROUTER_API_KEY)"
    from tools.llm_api import credentials
    tok, _base = credentials()   # base URL defaults in llm_api, so a key is enough
    return bool(api_key or tok), \
        "OpenAI API key (settings field or .env OPENAI_API_KEY)"


async def ask_vision(prompt: str, image_bgr: np.ndarray, model_key: str,
                     api_key: str = "", *, timeout: float = 60.0) -> str:
    """Prompt + one image → model reply text, via the model's provider."""
    provider, model = resolve(model_key)
    if provider == "openrouter":
        from tools.vlm import chat_vision
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OpenRouter API key missing — enter it in the "
                               "pipeline's API key field or set OPENROUTER_API_KEY.")
        return await chat_vision(prompt, image_bgr, model_key=model,
                                 api_key=key, timeout=timeout)
    from tools.llm_api import query_vision, response_text
    data = await query_vision(prompt, image_bgr, model=model,
                              token=(api_key or None), timeout_s=timeout)
    return response_text(data)


async def ask_vision_json(prompt: str, image_bgr: np.ndarray, model_key: str,
                          api_key: str = "", *, timeout: float = 60.0) -> dict:
    """Same as ask_vision but robustly parsed to a JSON dict."""
    from tools.vlm import parse_vlm_json
    txt = await ask_vision(prompt, image_bgr, model_key, api_key, timeout=timeout)
    return parse_vlm_json(txt or "")


# ── Subject naming (provider-agnostic) — used by the tracker + inline fallback ─

_SUBJECTS_PROMPT = (
    "The image shows sampled frames (side by side) from one video. If a tile "
    "is labeled MOTION, it is a zoomed crop of the region with the MOST MOTION "
    "in the video — the moving object shown there is the most important "
    "subject: name it FIRST, as a specific visual noun phrase.\n"
    "Then list the other primary physical objects — moving/acting subjects, "
    "then key interacting surfaces. Skip pure background. Use short visual noun "
    'phrases a segmentation model can ground (e.g. "basketball", "wooden '
    'crate"). At most {k}. Reply with ONLY strict JSON: '
    '{{"subjects": ["...", "..."]}}'
)


async def name_subjects(frames_bgr: list, *, max_subjects: int = 3,
                        model_key: str = DEFAULT_MODEL_KEY, api_key: str = "",
                        motion_crop: Optional[np.ndarray] = None) -> list[str]:
    """Name the primary subjects across sampled frames, via any provider.

    `frames_bgr`: 1–2 frames tiled side by side. `motion_crop`, if given, is
    appended as a red-bordered MOTION tile and must be named first.
    """
    tiles = list(frames_bgr) if isinstance(frames_bgr, list) else [frames_bgr]
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (max(2, int(t.shape[1] * h / t.shape[0])), h)) for t in tiles]
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

    parsed = await ask_vision_json(_SUBJECTS_PROMPT.format(k=max_subjects),
                                   composite, model_key, api_key)
    subjects = parsed.get("subjects") if isinstance(parsed, dict) else None
    if not isinstance(subjects, list):
        return []
    return [str(s).strip() for s in subjects if str(s).strip()][:max_subjects]
