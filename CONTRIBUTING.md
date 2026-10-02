# Developer documentation

How the interactive tool is put together and how to extend it. For the
benchmark, the paper's evaluation, and limitations, see [README.md](README.md).

## Adding a pipeline

### 1. Create the module

```python
# backend/pipelines/stageN/my_test.py
import asyncio, json
from typing import AsyncGenerator

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}

    yield {"type": "log", "level": "info", "text": "Starting…"}
    await asyncio.sleep(0)

    # --- your implementation here ---

    yield {"type": "metric",   "label": "Score",         "value": "0.85", "sub": "description"}
    yield {"type": "severity", "label": "Physics score",  "value": 85,     "color": "#1a7a3c"}
    yield {"type": "done"}
```

### 2. Register it in `backend/main.py`

```python
from pipelines.stageN.my_test import run as run_my_test

PIPELINES["my_test"] = {
    "id":           "my_test",
    "name":         "My Test Name",
    "desc":         "Short description shown on the test card.",
    "badge":        "medium",       # cheap | medium | expensive | output
    "dummy":        True,           # True = STUB badge + dashed border in UI
    "requires_pair": False,
    "settings": [
        {"id": "threshold", "label": "Threshold", "type": "number",
         "default": 0.5, "min": 0.0, "max": 1.0},
    ],
    "run": run_my_test,
}
```

Register new, incomplete pipelines with `"dummy": True`. Set it to `False`
only after the pipeline is implemented and tested; changing the flag itself
does not make a pipeline ready. Use the shared settings builders at the top of
`main.py` (`_vlm_model_setting`, `_vlm_key_setting`, `_openai_key_setting`)
for model and API-key fields.

### 3. Add the pipeline id to `STAGES` in `frontend/index.html`

```js
{ id: 'specialist', ..., pipelines: [..., 'my_test'] },
```

The UI loads the pipeline list from `GET /pipelines` on startup; `STAGES`
decides which stage tab it appears under.

### Stage 3 and the report

Stage 3 is a single pipeline, `s3_specialist`
(`backend/pipelines/stage3/specialist_mcq.py`). It publishes one entry to the
evidence bus:
`{"family_probs", "top_family", "p_violation", "task_completed_p", "hidden_property_score", "explanation"}`.
Stage 4 (`pipelines/stage4/diagnostic_report.py`) reads it via
`SPECIALIST_DISPLAY` and `_collect_semantic_findings()`. A new Stage 3 test
must publish under its own key, be added to `SPECIALIST_DISPLAY`, and have its
shape handled in `_collect_semantic_findings()`, or the report will ignore it.

## Event schema (pipeline → frontend)

| `type`     | Required keys                              | Notes                          |
|------------|--------------------------------------------|--------------------------------|
| `log`      | `level` (info/warn/error/success), `text`  | Appears in live log stream     |
| `metric`   | `label`, `value`, `sub`                    | `"PASS"` / `"FAIL"` colored    |
| `severity` | `label`, `value` (0–100), `color` (hex)    | Renders a progress bar         |
| `image`    | `data` (base64), `mime`, `caption`         | Rendered inline                |
| `plotly`   | `data` (JSON string), `caption`            | Interactive Plotly chart       |
| `video`    | `data` (base64), `mime`, `caption`         | Inline video player            |
| `result`   | free-form structured payload               | Shown as JSON; used by Stage 4 |
| `done`     | —                                          | Sets status to Done            |
| `error`    | `text`                                     | Logs error + sets status       |

The frontend's `handleEvent()` and `applyEventToEntry()` in
`frontend/index.html` are the authoritative renderers; a new event type must
be handled in both.

## UI overview

The tool has a single **Dataset** view; one video is a batch of one.

- **Add videos:** upload files or a folder, or download every video from a
  Hugging Face dataset repo.
- **Select tests:** a checklist of registered (non-stub) pipelines by stage.
- **Run batch:** each video runs the selected pipelines in stage order. The
  Hypothesis Generator can auto-run the Stage 3 test afterwards (⚙ setting).
- **Per-video report:** click a card to see every test that ran on it and
  re-run any test individually.

Pipelines in a batch share the per-video track cache and evidence bus, since
they run on the same file. The batch API is in `backend/dataset_api.py`.

## Smoke tests

`backend/scripts/test_object_tracker.py`, `vlm_failure_mode_eval.py`,
`vlm_multimodel_eval.py` and `eval_framework.py` read short clips from
`test_videos/` at the repository root. Those clips are not distributed (they
come from third-party sources); supply your own, keeping the subfolder paths
listed in each script. `test_videos/` is gitignored. `sam3_smoke.py` takes a
video path as its first argument.

## Running pipelines without the UI

```bash
cd backend
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py <pipeline_id> <video_path> --set key=val --prereq <stage2_id>
```

Pipelines that use a local VLM load several GB of weights onto the GPU on
first run.
