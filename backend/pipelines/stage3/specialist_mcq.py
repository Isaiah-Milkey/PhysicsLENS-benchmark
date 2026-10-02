"""
Stage 3 · VLM verification — the paper's three questions, one VLM.
-------------------------------------------------------------------
Replaces the seven separate per-category specialists (collision, gravity,
momentum, friction, deformation, fluid, causality; see git history for the
retired implementations). Given eight evenly spaced frames and the task, one
VLM answers, each read from the probabilities of its first output token:

  (i)   violation type — a forced choice over eight failure descriptions
        (seven constraint families plus object permanence) and "none";
  (ii)  task completion — did the robot finish the task, P(yes);
  (iii) hidden-property adherence — 1-4, only when a property is supplied.

This is the design evaluated in the paper's automated-diagnosis results
(backend/scripts/mcq_probe.py and vlm_plausibility.py compute the same
questions offline over the benchmark). Prompt wording matches those scripts.

WHY ONE FORCED CHOICE INSTEAD OF SEVEN INDEPENDENT SPECIALISTS. Scored
independently, each specialist had its own scale and nothing forced them to
disagree — collision answered "yes" on nearly every clip while deformation
answered "no" on nearly every clip. A single question makes the options
compete: probabilities are normalised by construction, so alleging every
failure is collision costs mass that must come from another option. It is
also one call instead of seven.

POSITION BIAS is controlled the way the evaluation does it: options are
shuffled per video with a fixed seed (a CRC of the video's content hash, so
it is identical across processes — Python's built-in hash() is salted per
process and would not be), and the mapping is stored in the result.

No Contact option: Contact is merged into Collision when the paper scores
attribution, and "permanence" (an object appearing or vanishing) has no
human label. Both are documented in the paper's Appendix D.

Backends, matching the evaluation scripts:
  • local  (tools.vlm_local.mcq_probs) — no API key, runs on the GPU.
  • openai (tools.llm_api.query_vision_token_probs) — needs OPENAI_API_KEY;
    the same Chat Completions top-logprobs read as the offline scripts.
    OpenRouter is not wired here: logprob support is inconsistent across the
    models it proxies, and it is not what the design was evaluated against.

Frames: 8 by default, evenly spaced first to last, longer side at most 512 px.
"""
import asyncio
import functools
import json
import zlib
from typing import AsyncGenerator

import cv2
import numpy as np
import plotly.graph_objects as go

from tools.evidence import EVIDENCE, video_id
from tools.video import load_frames, sample_frames

MAX_SIDE = 512

# Keep in sync with backend/scripts/mcq_probe.py's OPTIONS — same wording, so
# a re-evaluation of the live tool is directly comparable to the paper.
OPTIONS = [
    ("collision", "Two things overlap or pass through each other instead of "
                  "meeting at their surfaces, or the gripper grasps without "
                  "closing on the object"),
    ("deformation", "A rigid object changes its shape, length or thickness"),
    ("causality", "An object moves or changes on its own, with nothing visibly "
                  "touching it"),
    ("gravity", "Something hangs in the air, floats, or fails to fall when "
                "nothing is holding it"),
    ("momentum", "Something speeds up after being released or struck, or stops "
                 "dead for no reason"),
    ("permanence", "An object appears from nowhere or vanishes"),
    ("fluid", "Liquid appears, vanishes, or holds an impossible rigid shape"),
    ("friction", "Something slides when it should grip, or keeps sliding with "
                 "nothing pushing it"),
    ("none", "No physics problem — everything behaves as it should"),
]
NAMES = [n for n, _ in OPTIONS]
LETTERS = "ABCDEFGHI"  # one per option, in NAMES order before shuffling

MCQ_PROMPT = ('Look at these {n} frames, sampled in order from a video of: '
              '"{task}".\n\n'
              'Which ONE of these best describes the main physics problem in this '
              'video?\n\n{opts}\n\n'
              'Answer with exactly one letter.')

# Rating questions (same wording as scripts/vlm_plausibility.py).
HEAD = ('Look at these {n} frames, sampled in order from a video of a robot. '
        'The task: "{task}".\n\n')
Q_ACTION = ("Did the robot actually COMPLETE the task shown?\n"
            "Answer with exactly one digit: 1 = no, 2 = yes.")
Q_HIDDEN = ("Hidden property of this scene (not visible in the frames): "
            "{hp} — {hv}.\nIf the video respects it, this should happen: "
            "{exp}\n\nRate how well the video FOLLOWS that hidden property.\n"
            "1 = contradicts it, 2 = mostly contradicts, 3 = mostly follows, "
            "4 = clearly follows.\nAnswer with exactly one digit, 1 to 4.")


def _build_mcq(video_key: str, nframes: int, task: str):
    """Prompt text + letter->family mapping, options shuffled per video."""
    seed = zlib.crc32(video_key.encode("utf-8"))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(OPTIONS)))
    lines, mapping = [], {}
    for pos, idx in enumerate(order):
        name, desc = OPTIONS[idx]
        letter = LETTERS[pos]
        lines.append(f"{letter}) {desc}")
        mapping[letter] = name
    prompt = MCQ_PROMPT.format(n=nframes, task=task or "a robot performing a task",
                               opts="\n".join(lines))
    return prompt, mapping


def _normalise(raw_probs: dict, mapping: dict) -> dict | None:
    """Raw {token: probability} -> {family: probability}, renormalised.
    Sums surface-form variants the same way mcq_probe.py's normalise() does."""
    agg: dict[str, float] = {}
    for k, v in raw_probs.items():
        key = str(k).strip().upper()
        if key in mapping:
            agg[mapping[key]] = agg.get(mapping[key], 0.0) + v
    tot = sum(agg.values())
    if tot <= 1e-6:
        return None
    return {n: agg.get(n, 0.0) / tot for n in NAMES}


def _digit_mass(raw_probs: dict, digits: str) -> dict[str, float] | None:
    """Probability of each allowed digit, renormalised over the allowed set."""
    m = {d: 0.0 for d in digits}
    for k, v in raw_probs.items():
        key = str(k).strip()
        if key in m:
            m[key] += v
    tot = sum(m.values())
    if tot <= 1e-6:
        return None
    return {d: v / tot for d, v in m.items()}


def _expected(raw_probs: dict, digits: str) -> float | None:
    """Expected rating over the allowed digits (the 1-4 questions)."""
    p = _digit_mass(raw_probs, digits)
    return None if p is None else sum(int(d) * v for d, v in p.items())


def _p_yes(raw_probs: dict) -> float | None:
    """Task completion: '1 = no, 2 = yes' -> P(2) renormalised over {1,2}."""
    p = _digit_mass(raw_probs, "12")
    return None if p is None else p["2"]


def _shrink(frame_bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return frame_bgr
    return cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _sev_color(sev: float) -> str:
    return "#E24B4A" if sev > 60 else "#EF9F27" if sev > 30 else "#4CAF50"


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg        = json.loads(settings) if settings else {}
    model_key  = str(cfg.get("model") or "qwen2.5-vl-7b")
    api_key    = str(cfg.get("api_key", "")).strip()
    num_frames = max(4, min(16, int(cfg.get("num_frames", 8))))
    task       = str(cfg.get("task_description", "")).strip()
    hidden_prop = str(cfg.get("hidden_property", "")).strip()
    hidden_val  = str(cfg.get("hidden_value", "")).strip()
    hidden_exp  = str(cfg.get("expected_outcome", "")).strip()

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames_bgr, _fps = await loop.run_in_executor(None, load_frames, video_path)
    if len(frames_bgr) < 2:
        yield {"type": "error", "text": f"Video too short ({len(frames_bgr)} frames)."}
        return
    sampled = [_shrink(f) for f in sample_frames(frames_bgr, num_frames)]
    n = len(sampled)

    from tools.vlm_local import LOCAL_VLMS
    is_local = model_key in LOCAL_VLMS

    # ── Backend: one async `ask(prompt, allowed_chars) -> {token: prob}` ──────
    if is_local:
        from tools.vlm_local import mcq_probs
        sampled_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled]

        async def ask(prompt: str, chars: str) -> dict:
            return await loop.run_in_executor(
                None, functools.partial(mcq_probs, sampled_rgb, prompt, chars, n,
                                        model_key, MAX_SIDE))
    else:
        from tools.vlm_router import resolve, key_status
        provider, provider_model = resolve(model_key)
        if provider != "openai":
            yield {"type": "error",
                   "text": f"Provider '{provider}' isn't wired for Stage 3 (only "
                           "local models or an openai:… model key are supported — "
                           "scoring needs a Chat Completions logprobs response)."}
            return
        have_key, key_desc = key_status(model_key, api_key)
        if not have_key:
            yield {"type": "log", "level": "warn",
                   "text": f"No API key ({key_desc}) — running in demo mode "
                           "(placeholder score)."}
            yield {"type": "metric", "label": "Top family", "value": "demo",
                   "sub": "Demo mode — no API key provided."}
            yield {"type": "severity", "label": "Specialist violation score",
                   "value": 0, "color": "#4CAF50"}
            yield {"type": "done"}
            return
        from tools.llm_api import query_vision_token_probs

        async def ask(prompt: str, chars: str) -> dict:
            return await query_vision_token_probs(
                prompt, sampled, model=provider_model, token=(api_key or None))

    yield {"type": "log", "level": "info",
           "text": f"VLM verification over {n} frame(s) via {model_key}"
                   f"{' (local)' if is_local else ''}: violation type, "
                   "task completion"
                   + (", hidden property" if hidden_prop else "") + "…"}

    # ── (i) violation type: forced choice ─────────────────────────────────────
    prompt, mapping = _build_mcq(vid, n, task)
    try:
        raw_probs = await ask(prompt, "".join(mapping))
    except Exception as exc:                                        # noqa: BLE001
        yield {"type": "error", "text": f"VLM call failed: {str(exc)[:300]}"}
        return
    probs = _normalise(raw_probs, mapping)
    if not probs:
        yield {"type": "error",
               "text": "Model did not answer with a recognizable option letter — "
                       "no usable probability distribution."}
        return

    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top_family, top_p = ranked[0]
    p_violation = 1.0 - probs.get("none", 0.0)
    desc_by_name = dict(OPTIONS)

    yield {"type": "log", "level": "success" if top_family == "none" else "warn",
           "text": f"Top answer: {top_family} ({top_p:.0%})"
                   + (f" — {desc_by_name[top_family]}" if top_family != "none" else "")}
    for name, p in ranked[:5]:
        yield {"type": "metric", "label": name.capitalize(), "value": f"{p:.0%}",
               "sub": desc_by_name[name]}

    fig = go.Figure(go.Bar(
        x=[p for _, p in ranked][::-1], y=[nm.capitalize() for nm, _ in ranked][::-1],
        orientation="h", marker_color=[_sev_color(p * 100) for _, p in ranked][::-1],
        text=[f"{p:.0%}" for _, p in ranked][::-1], textposition="outside"))
    fig.update_xaxes(range=[0, 1.05], title_text="Probability",
                     showgrid=True, gridcolor="#ebebeb")
    fig.update_layout(
        title=dict(text="Failure type (forced choice)", font=dict(size=15)),
        height=160 + 26 * len(ranked), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=110, r=60, t=60, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13))
    yield {"type": "plotly", "data": fig.to_json(),
           "caption": "One question, nine options (eight failure descriptions "
                      "plus none), shuffled per video. Probabilities sum to 1, so "
                      "naming one family costs mass that must come from another."}

    # ── (ii) task completion ──────────────────────────────────────────────────
    head = HEAD.format(n=n, task=task or "a manipulation task")
    task_done = None
    try:
        task_done = _p_yes(await ask(head + Q_ACTION, "12"))
    except Exception as exc:                                        # noqa: BLE001
        yield {"type": "log", "level": "warn",
               "text": f"Task-completion question failed: {str(exc)[:160]}"}
    if task_done is not None:
        yield {"type": "metric", "label": "Task completed", "value": f"{task_done:.0%}",
               "sub": "P(yes) that the robot finished the task"}

    # ── (iii) hidden-property adherence (only when a property is supplied) ────
    hidden = None
    if hidden_prop:
        q = Q_HIDDEN.format(hp=hidden_prop.replace("_", " "), hv=hidden_val,
                            exp=hidden_exp)
        try:
            hidden = _expected(await ask(head + q, "1234"), "1234")
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "log", "level": "warn",
                   "text": f"Hidden-property question failed: {str(exc)[:160]}"}
        if hidden is not None:
            yield {"type": "metric", "label": "Hidden-property adherence",
                   "value": f"{hidden:.2f} / 4",
                   "sub": "expected rating that the video follows the stated property"}

    yield {"type": "severity", "label": "Specialist violation score",
           "value": round(p_violation * 100, 1), "color": _sev_color(p_violation * 100)}

    finding = {
        "family_probs": {k: round(v, 4) for k, v in probs.items()},
        "top_family": top_family,
        "p_violation": round(p_violation, 4),
        "task_completed_p": None if task_done is None else round(task_done, 4),
        "hidden_property_score": None if hidden is None else round(hidden, 3),
        "explanation": (f"Forced-choice VLM judged \"{top_family}\" the most likely "
                        f"failure ({top_p:.0%}): {desc_by_name[top_family]}."
                        if top_family != "none" else
                        "Forced-choice VLM found no likely physics failure "
                        f"({probs['none']:.0%} confidence)."),
    }
    yield {"type": "result", "status": "ok", "verification": finding, "mapping": mapping}
    EVIDENCE.put(vid, "s3_specialist", finding)
    yield {"type": "done"}
