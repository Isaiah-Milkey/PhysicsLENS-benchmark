"""Integration test for stage1/vlm_suspicion.run().

Run directly:  python backend/scripts/test_vlm_pipeline.py

Stubs the only external IO (video file read + network VLM call) and asserts the
redesigned pipeline's behavioral contract: ONE holistic multi-frame call (not N
single-frame calls), and a well-formed event stream ending in `done`.
"""
import asyncio, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipelines.stage1.vlm_suspicion as vs

calls = {"multi": 0, "single": 0}


async def fake_multi(frames, model_key="gpt-4o", api_key=""):
    calls["multi"] += 1
    assert isinstance(frames, list) and len(frames) > 1, "expected multiple frames in one call"
    return {"suspicion_score": 0.8, "suspected_failure": "gravity",
            "explanation": "objects accelerate upward", "confidence": 0.9}


async def fake_single(frame, model_key="gpt-4o", api_key=""):
    calls["single"] += 1
    return {"suspicion_score": 0.5, "suspected_failure": None,
            "explanation": "", "confidence": 0.5}


def _drain():
    calls["multi"] = calls["single"] = 0
    vs.score_frames = fake_multi          # bound to module namespace
    vs.score_frame = fake_single          # must NOT be used by the new pipeline
    vs.load_frames = lambda p: ([np.zeros((16, 16, 3), np.uint8) for _ in range(20)], 10.0)

    async def go():
        out = []
        settings = json.dumps({"api_key": "sk-test", "model": "gpt-4o", "num_frames": 8})
        async for ev in vs.run("dummy.mp4", settings):
            out.append(ev)
        return out

    return asyncio.run(go())


def test_makes_one_multiframe_call():
    _drain()
    assert calls["multi"] == 1, f'expected 1 multi-frame call, got {calls["multi"]}'
    assert calls["single"] == 0, f'single-frame scorer must not be used, got {calls["single"]}'


def test_emits_well_formed_stream():
    events = _drain()
    types = [e["type"] for e in events]
    assert "plotly" in types, types
    assert "severity" in types, types
    assert types[-1] == "done", types


def test_severity_in_range():
    events = _drain()
    sev = [e for e in events if e["type"] == "severity"][0]
    assert 0 <= sev["value"] <= 100, sev
    # verdict score 0.8 -> suspicious -> severity should be high
    assert sev["value"] >= 50, sev


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
