"""
Stage 1 · Test 4 — VLM Physics Anomaly Detector
------------------------------------------------
Sample keyframes and ask a vision-language model whether the MOTION across the
video obeys real-world physics. All frames are sent in a single multi-frame
call and the model returns one holistic verdict plus a list of concrete
violations — which physical law is broken, what is observed, and how it
contradicts real physics. Multi-frame beats per-frame scoring decisively
(AUC 0.90 vs 0.58, scripts/vlm_failure_mode_eval.py).

Model dropdown (measured AI-vs-real separation, scripts/vlm_multimodel_eval.py):
  • Local open-weight, no API key: Qwen2.5-VL-7B (AUC 0.92) · InternVL3-8B
    (AUC 0.70) · SmolVLM2-2.2B (AUC 0.50)
  • API via OpenRouter (needs key): Gemini 2.5 Pro/Flash, GPT-4o/4.1/5.1,
    Claude Sonnet 4.5

Without a key, API models fall back to demo mode (placeholder score); local
models always run for real on the GPU.
"""
import asyncio, json, os
from typing import AsyncGenerator

import cv2
import numpy as np
import plotly.graph_objects as go

from tools.video     import load_frames, sample_frames
from tools.vlm       import score_frames, OPENROUTER_MODELS, SUSPICION_PROMPT_MULTI
from tools.vlm_local import LOCAL_VLMS
from tools.vlm_router import resolve as _resolve, key_status as _key_status, model_options


def _zone_color(score: float) -> str:
    return "#E24B4A" if score > 0.5 else "#EF9F27" if score > 0.25 else "#4CAF50"


def _normalize_violations(result: dict) -> list[dict]:
    """Coerce local (rich dict) and OpenRouter (short string) violations into
    one shape: {law, observation, why_impossible, severity}."""
    out = []
    for v in result.get("violations") or []:
        if isinstance(v, dict):
            out.append({
                "law":            v.get("law") or "unspecified",
                "observation":    v.get("observation") or "",
                "why_impossible": v.get("why_impossible") or "",
                "severity":       v.get("severity"),
            })
        elif isinstance(v, str) and v.strip():
            out.append({"law": v.strip(), "observation": "", "why_impossible": "",
                        "severity": None})
    return out


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg        = json.loads(settings) if settings else {}
    model_key  = cfg.get("model", "qwen2.5-vl-7b")
    num_frames = max(2, int(cfg.get("num_frames", 8)))
    n_consist  = min(5, max(1, int(cfg.get("consistency_samples", 1))))
    use_logprob = str(cfg.get("token_prob_score", "true")).lower() not in ("false", "0", "no")
    api_key    = str(cfg.get("api_key", "")).strip()

    is_local  = model_key in LOCAL_VLMS
    provider  = "local" if is_local else _resolve(model_key)[0]   # openai | openrouter
    have_api  = is_local or _key_status(model_key, api_key)[0]
    demo_mode = (not is_local) and (not have_api)

    if is_local:
        info = LOCAL_VLMS[model_key]
        auc_txt = f"measured AUC {info['auc']:.2f}" if info.get("auc") else "AUC not yet measured"
        yield {"type": "log", "level": "info",
               "text": (f"Local model: {info['label']} ({info['hf_id']}) — "
                        f"~{info['vram_gb']} GB VRAM, {auc_txt}, no API key needed.")}
    elif demo_mode:
        yield {"type": "log", "level": "warn",
               "text": "No API key — running in demo mode (placeholder score)."}
        yield {"type": "log", "level": "info",
               "text": f"Enter a {provider} key in settings (or set the provider's env "
                       "var), or pick a local model (Qwen2.5-VL, InternVL3, SmolVLM2) "
                       "that needs none."}
    else:
        yield {"type": "log", "level": "info",
               "text": f"API model via {provider}: {model_key}"}

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    n = len(frames)
    if n == 0:
        yield {"type": "error", "text": "Could not read any frames from the video."}
        return

    keyframes = sample_frames(frames, num_frames)
    duration  = n / fps if fps else 0.0

    yield {"type": "log", "level": "info",
           "text": f"Analyzing {len(keyframes)} frames from {n} total in one multi-frame pass…"}
    await asyncio.sleep(0)

    if demo_mode:
        await asyncio.sleep(0.4)
        result = {
            "suspicion_score":    float(np.random.beta(2, 5)),
            "overall_assessment": "Demo mode — no API key provided.",
            "confidence":         0.0,
            "violations":         [],
        }
    elif is_local:
        try:
            from tools.vlm_local import analyze_physics, plausibility_logprob
            loop = asyncio.get_event_loop()
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in keyframes]
            yield {"type": "log", "level": "info",
                   "text": "Loading model (first run downloads/loads weights — may take a minute)…"}
            await asyncio.sleep(0)
            if n_consist > 1:
                yield {"type": "log", "level": "info",
                       "text": f"Self-consistency: sampling the judgement {n_consist}× "
                               "to MEASURE agreement (median score, spread → consistency)."}
                await asyncio.sleep(0)
            result = await loop.run_in_executor(
                None, lambda: analyze_physics(rgb, n_sample=len(rgb),
                                              model_key=model_key, samples=n_consist))
            if use_logprob:
                try:
                    result["logprob_score"] = await loop.run_in_executor(
                        None, lambda: plausibility_logprob(rgb, n_sample=len(rgb),
                                                           model_key=model_key))
                except Exception as exc:  # noqa: BLE001
                    yield {"type": "log", "level": "warn",
                           "text": f"Token-probability score failed ({exc}) — skipping."}
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "error", "text": f"Local VLM failed: {exc}"}
            result = {"suspicion_score": None, "overall_assessment": str(exc),
                      "confidence": 0.0, "violations": []}
    elif provider == "openrouter":
        # OpenRouter sends each keyframe as its own image block (measured AUC 0.90).
        or_model = _resolve(model_key)[1]
        or_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        try:
            result = await score_frames(keyframes, model_key=or_model, api_key=or_key)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "error", "text": f"VLM call failed: {exc}"}
            result = {"suspicion_score": None, "overall_assessment": str(exc),
                      "confidence": 0.0, "violations": []}
    else:
        # The OpenAI chat API takes ONE image per call here, so tile the keyframes into a labeled
        # strip and send the same multi-frame suspicion prompt.
        from tools.vlm_router import ask_vision_json
        tiles = []
        for i, f in enumerate(keyframes, 1):
            s = 260 / f.shape[0]
            t = cv2.resize(f, (max(2, int(f.shape[1] * s)), 260))
            band = np.full((26, t.shape[1], 3), 20, np.uint8)
            cv2.putText(band, f"frame {i}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.concatenate([band, t], axis=0))
        gap = np.full((tiles[0].shape[0], 8, 3), 255, np.uint8)
        composite = tiles[0]
        for t in tiles[1:]:
            composite = np.concatenate([composite, gap, t], axis=1)
        prompt = SUSPICION_PROMPT_MULTI.replace("{n}", str(len(keyframes)))
        try:
            result = await ask_vision_json(prompt, composite, model_key, api_key, timeout=90)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "error", "text": f"VLM call failed: {exc}"}
            result = {"suspicion_score": None, "overall_assessment": str(exc),
                      "confidence": 0.0, "violations": []}

    raw_score  = result.get("suspicion_score")
    parsed_ok  = isinstance(raw_score, (int, float))
    score      = min(max(float(raw_score), 0.0), 1.0) if parsed_ok else 0.0
    assessment = (result.get("overall_assessment") or result.get("explanation") or "").strip()
    confidence = float(result.get("confidence", 0) or 0)
    logprob    = result.get("logprob_score")          # measured P(Yes|violation?)
    consist    = result.get("consistency")            # measured sample agreement
    violations = _normalize_violations(result)
    fail_label = (result.get("suspected_failure") or "").strip()
    if fail_label and fail_label.lower() not in ("null", "none") and not violations:
        violations = [{"law": fail_label, "observation": "", "why_impossible": "",
                       "severity": None}]

    if not parsed_ok and not demo_mode:
        yield {"type": "log", "level": "warn",
               "text": "Model did not return a usable score — reporting 0 (see explanation)."}

    level = "warn" if score > 0.5 else "info"
    yield {"type": "log", "level": level,
           "text": f"Verdict: score={score:.2f}  ({len(violations)} violation(s))  {assessment}"}

    # ── What physics are broken, and how ────────────────────────────────────────
    for i, v in enumerate(violations, 1):
        parts = [f"Violation {i}: {v['law']}"]
        if v["observation"]:
            parts.append(f"observed: {v['observation']}")
        if v["why_impossible"]:
            parts.append(f"why impossible: {v['why_impossible']}")
        if v["severity"] is not None:
            parts.append(f"severity {v['severity']:.1f}")
        yield {"type": "log", "level": "warn", "text": " — ".join(parts)}

    # ── Gauge: one holistic suspicion score ─────────────────────────────────────
    color = _zone_color(score)
    model_label = (LOCAL_VLMS[model_key]["label"] if is_local else model_key)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(valueformat=".2f", font=dict(size=42)),
        gauge=dict(
            axis=dict(range=[0, 1], tickwidth=1, tickcolor="#888",
                      tickvals=[0, 0.25, 0.5, 0.75, 1]),
            bar=dict(color=color, thickness=0.3),
            borderwidth=0,
            steps=[
                dict(range=[0,    0.25], color="#E8F5E9"),
                dict(range=[0.25, 0.5 ], color="#FFF3E0"),
                dict(range=[0.5,  1.0 ], color="#FFEBEE"),
            ],
            threshold=dict(line=dict(color="#E24B4A", width=3), thickness=0.85, value=0.5),
        ),
        title=dict(
            text=f"VLM Physics Suspicion — {model_label}" + ("  [DEMO]" if demo_mode else ""),
            font=dict(size=15),
        ),
    ))
    fig.update_layout(
        height=360, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=20),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": (f"Holistic suspicion over {len(keyframes)} frames "
                    f"({duration:.1f}s of video). "
                    "Red ≥ 0.5 (suspicious), amber ≥ 0.25 (uncertain), green < 0.25 (plausible)."),
    }

    # ── Violations table ─────────────────────────────────────────────────────────
    if violations:
        tbl = go.Figure(data=[go.Table(
            columnwidth=[22, 39, 39],
            header=dict(values=["<b>Broken law / principle</b>", "<b>What is observed</b>",
                                "<b>Why it's physically impossible</b>"],
                        fill_color="#1a54c4", font=dict(color="white", size=13),
                        align="left", height=30),
            cells=dict(values=[
                [v["law"] for v in violations],
                [v["observation"] or "—" for v in violations],
                [v["why_impossible"] or "—" for v in violations]],
                fill_color=[["#f6f8fe", "#ffffff"] * len(violations)],
                align="left", font=dict(size=12.5), height=28),
        )])
        tbl.update_layout(
            title=dict(text="Physics Violations Extracted by the Model", font=dict(size=15)),
            height=120 + 34 * max(len(violations), 1),
            margin=dict(l=20, r=20, t=55, b=15),
            paper_bgcolor="white",
            font=dict(family="IBM Plex Sans, sans-serif"),
        )
        yield {"type": "plotly", "data": tbl.to_json(),
               "caption": "Each row: the physical law the model believes is broken, "
                          "the visual evidence, and the reasoning."}

    # ── Score comparison: self-reported vs measured signals ─────────────────────
    if isinstance(logprob, (int, float)):
        names  = ["Self-reported score (JSON)", "Token-probability P(violates)"]
        vals   = [score, float(logprob)]
        colors = [_zone_color(score), _zone_color(float(logprob))]
        if isinstance(consist, (int, float)):
            names.append("Self-consistency (measured)")
            vals.append(float(consist))
            colors.append("#1a54c4")
        cmp_fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors, text=[f"{v:.2f}" for v in vals],
            textposition="outside",
        ))
        cmp_fig.update_layout(
            title=dict(text="Suspicion Signals — self-written vs read from the model",
                       font=dict(size=15)),
            xaxis=dict(range=[0, 1.12], showgrid=True, gridcolor="#ebebeb"),
            height=110 + 55 * len(names), paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=10, r=30, t=55, b=25),
            font=dict(family="IBM Plex Sans, sans-serif", size=13),
        )
        yield {"type": "plotly", "data": cmp_fig.to_json(),
               "caption": ("JSON score: the float the model wrote. Token-probability: "
                           "P(model answers 'Yes' to 'does this violate physics?'), read "
                           "from the next-token distribution — continuous and not "
                           "self-flattering. Consistency: agreement across resampled "
                           "judgements (1.0 = same verdict every time).")}

    severity = min(int(score * 100), 100)
    top_law = violations[0]["law"] if violations else "—"
    yield {"type": "metric", "label": "Suspicion score",    "value": f"{score:.2f}",
           "sub": "0–1 (whole video)"}
    if isinstance(logprob, (int, float)):
        yield {"type": "metric", "label": "Token-prob score", "value": f"{logprob:.2f}",
               "sub": "P(violates) from Yes/No logits"}
    yield {"type": "metric", "label": "Physics violations", "value": str(len(violations)),
           "sub": top_law if len(violations) else "none detected"}
    yield {"type": "metric", "label": "Model confidence",   "value": f"{confidence:.0%}",
           "sub": "self-reported"}
    if isinstance(consist, (int, float)):
        samples_txt = ", ".join(f"{s:.2f}" for s in result.get("score_samples", []))
        yield {"type": "metric", "label": "Self-consistency", "value": f"{consist:.2f}",
               "sub": f"measured over samples [{samples_txt}]"}
    yield {"type": "severity", "label": "VLM suspicion level", "value": severity, "color": color}

    if assessment and not demo_mode:
        yield {"type": "log", "level": "info", "text": f"Assessment: {assessment}"}
    if demo_mode:
        yield {"type": "log", "level": "warn",
               "text": "Result is a demo placeholder. Pick a local model or provide an "
                       "OpenRouter API key for real scoring."}
    yield {"type": "done"}
