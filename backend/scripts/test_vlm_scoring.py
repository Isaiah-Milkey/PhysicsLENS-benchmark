"""Unit tests for the redesigned VLM suspicion scoring (tools/vlm.py).

Run directly:  python backend/scripts/test_vlm_scoring.py

Covers the two pure, deterministic units behind the multi-frame + determinism
redesign (the model call itself is validated end-to-end by
scripts/vlm_failure_mode_eval.py, AUC 0.90):

  1. parse_vlm_json   — robust JSON extraction (fenced / prose-wrapped / plain)
  2. build_suspicion_payload — one multi-frame call, deterministic decoding
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vlm import parse_vlm_json, build_suspicion_payload


def test_parse_plain_json():
    out = parse_vlm_json('{"suspicion_score": 0.3, "confidence": 0.8}')
    assert out["suspicion_score"] == 0.3, out
    assert out["confidence"] == 0.8, out


def test_parse_markdown_fenced():
    out = parse_vlm_json('```json\n{"suspicion_score": 0.7}\n```')
    assert out["suspicion_score"] == 0.7, out


def test_parse_prose_wrapped():
    # The old lstrip("```json") approach mangles this; regex extraction must not.
    raw = 'Sure! Here is the result: {"suspicion_score": 0.9} Hope this helps.'
    out = parse_vlm_json(raw)
    assert out["suspicion_score"] == 0.9, out


def test_parse_unparseable_is_safe():
    out = parse_vlm_json("the image looks fine, no json here")
    assert out["suspicion_score"] is None, out  # must not raise, must signal failure


def test_parse_truncated_json_recovers_score():
    # Real failure mode observed live: max_tokens cut the reply off before the
    # closing brace. The score is right there — the parser must recover it, not
    # throw it away as "unparseable" (gemini-2.5-pro / gemini-3.5-flash did this).
    raw = '{"suspicion_score": 0.3, "suspected_failure": "gravity", "explanat'
    out = parse_vlm_json(raw)
    assert out["suspicion_score"] == 0.3, out


def test_parse_fenced_truncated_recovers_score():
    raw = '```json\n{\n  "suspicion_score": 0.85,\n  "explanation": "the ball app'
    out = parse_vlm_json(raw)
    assert out["suspicion_score"] == 0.85, out


def test_parse_truncated_keeps_explanation():
    raw = '{"suspicion_score": 0.7, "explanation": "objects float upward", "conf'
    out = parse_vlm_json(raw)
    assert out["suspicion_score"] == 0.7, out
    assert "float" in (out.get("explanation") or ""), out


def test_parse_recovers_unterminated_explanation():
    # Reply cut off WHILE writing the explanation (no closing quote). We must
    # still surface the partial reasoning, not the "truncated reply" placeholder.
    raw = '{"suspicion_score": 0.6, "explanation": "the ball passes through the floor and'
    out = parse_vlm_json(raw)
    assert out["suspicion_score"] == 0.6, out
    assert "passes through the floor" in (out.get("explanation") or ""), out
    assert "truncated" not in (out.get("explanation") or "").lower(), out


def test_payload_is_deterministic():
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(8)]
    p = build_suspicion_payload(frames, "qwen/qwen-2.5-vl")
    assert p["temperature"] == 0, p.get("temperature")
    assert "seed" in p, "payload must pin a seed for reproducibility"


def test_payload_is_multi_frame_single_call():
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(8)]
    p = build_suspicion_payload(frames, "qwen/qwen-2.5-vl")
    assert len(p["messages"]) == 1, "all frames go in ONE call, not N calls"
    content = p["messages"][0]["content"]
    imgs = [c for c in content if c["type"] == "image_url"]
    texts = [c for c in content if c["type"] == "text"]
    assert len(imgs) == len(frames), f"{len(imgs)} image blocks for {len(frames)} frames"
    assert len(texts) == 1, "exactly one prompt text block"
    assert "physics" in texts[0]["text"].lower(), "prompt must be the physics judgment prompt"


def test_payload_carries_model_id():
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    p = build_suspicion_payload(frames, "openai/gpt-4o")
    assert p["model"] == "openai/gpt-4o", p.get("model")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
