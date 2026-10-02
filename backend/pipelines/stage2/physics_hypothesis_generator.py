"""
Stage 2 · Step 4 — Physics Hypothesis Generator
--------------------------------------------------
Decides WHICH Stage 3 specialists are worth running on this video, and where
to look. Ranking is built in (this absorbed the former hypothesis_ranker).

Two evidence sources, combined:

1. HEURISTIC PRIORS (free) — quantitative signals already on the evidence bus:
   trajectory contacts/spikes/reversals, event-localizer marker types, and the
   tracked subject labels. Each maps to specialist priors.
2. VLM TRIAGE (one API call) — keyframes around the strongest events are
   tiled into a composite; the VLM is shown the specialist catalog plus the
   quantitative evidence summary and returns a ranked JSON list of hypotheses
   with time windows and rationale.

Final confidence = max(heuristic prior, VLM confidence) per specialist, so a
strong quantitative signal can't be talked away, and the VLM can surface
categories the heuristics have no probe for (fluid, deformation). Degrades to
heuristics-only when no API model is available.
"""
import asyncio
import base64
import json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go

from tools.video import load_frames
from tools.evidence import EVIDENCE, video_id
from tools.vlm import parse_vlm_json

# Specialist catalog: id → (UI name, what it verifies). Shown to the VLM and
# used to sanitize its answers. Keep in sync with main.py's Stage 3 registry.
SPECIALISTS = {
    "collision":   ("Collision & Contact",
                    "impacts between objects: interpenetration, restitution/energy "
                    "gain, phantom bounces off nothing"),
    "gravity":     ("Gravity",
                    "free-fall or projectile motion with wrong acceleration, "
                    "floating/levitating objects"),
    "deformation": ("Deformation",
                    "implausible shape/appearance change: squash/stretch, rigid "
                    "objects wobbling, soft objects not deforming at impact or not "
                    "recovering, or an object morphing / changing identity / "
                    "vanishing/reappearing over time"),
    "friction":    ("Friction",
                    "sliding/rolling objects decelerating wrongly, accelerating "
                    "on flat surfaces, or changing friction character mid-motion"),
    "fluid":       ("Fluid",
                    "liquids, splashes, pouring, buoyancy behaving impossibly"),
    "causality":   ("Causality",
                    "effects before causes: objects moving before being hit, "
                    "response lag, time-reversed motion"),
    "momentum":    ("Momentum",
                    "motion changes with no visible cause, or contacts where "
                    "momentum transfer is one-sided / appears from nowhere"),
}

TRIAGE_PROMPT = (
    "You are triaging a video for physics defects (it may be AI-generated). "
    "The image shows keyframes from one video, sampled around the moments "
    "automated screening flagged (timestamps printed on each tile).\n"
    "Quantitative evidence already gathered:\n{evidence}\n"
    "Available specialist tests:\n{catalog}\n"
    "Rank which specialists are worth running on THIS video: judge from what "
    "the scene physically contains (e.g. no liquid visible → fluid is "
    "pointless) and what the keyframes/evidence suggest is suspicious. "
    "3 or fewer strong picks beat a long hedge list.\n"
    "Reply with ONLY strict JSON: {{\"hypotheses\": [{{\"specialist\": "
    "\"<id>\", \"confidence\": <0..1>, \"reason\": \"<one sentence>\", "
    "\"t_window\": [<start_s>, <end_s>]}}]}}"
)


def _sev_color(sev: float) -> str:
    return "#E24B4A" if sev > 60 else "#EF9F27" if sev > 30 else "#4CAF50"


def _tile(img: np.ndarray, label: str, tile_h: int = 300) -> np.ndarray:
    s = tile_h / img.shape[0]
    img = cv2.resize(img, (max(2, int(img.shape[1] * s)), tile_h))
    band = np.full((28, img.shape[1], 3), 20, np.uint8)
    cv2.putText(band, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([band, img], axis=0)


def _heuristic_priors(ev_traj: Optional[dict], ev_loc: Optional[dict],
                      ev_tr: Optional[dict]) -> tuple[dict, list[str]]:
    """Map bus evidence to specialist priors. Returns (priors, summary lines)."""
    priors = {k: 0.0 for k in SPECIALISTS}
    lines: list[str] = []

    if ev_tr:
        labels = [t.get("label") for t in ev_tr.get("tracks", []) if t.get("label")]
        if labels:
            lines.append(f"- tracked subjects: {', '.join(labels)}")

    if ev_traj:
        ts = ev_traj.get("type_severities", {})
        n_contacts = len(ev_traj.get("contacts", []))
        if n_contacts:
            priors["collision"] = max(priors["collision"], 0.55)
            priors["deformation"] = max(priors["deformation"], 0.3)
            lines.append(f"- {n_contacts} inter-object contact event(s)")
        if ts.get("accel_spike", 0) > 20:
            priors["collision"] = max(priors["collision"], 0.5)
            priors["causality"] = max(priors["causality"], 0.35)
            lines.append(f"- acceleration spikes (severity {ts['accel_spike']})")
        if ts.get("velocity_reversal", 0) > 20:
            priors["causality"] = max(priors["causality"], 0.5)
            priors["gravity"] = max(priors["gravity"], 0.4)
            lines.append(f"- force-free velocity reversal(s) (severity {ts['velocity_reversal']})")

    if ev_loc:
        by_type: dict[str, int] = {}
        for m in ev_loc.get("markers", []):
            by_type[m.get("type", "?")] = by_type.get(m.get("type", "?"), 0) + 1
        if by_type:
            lines.append("- event-localizer markers: "
                         + ", ".join(f"{k}×{v}" for k, v in by_type.items()))
        if by_type.get("embedding_jump"):
            priors["deformation"] = max(priors["deformation"], 0.6)
        if by_type.get("temporal_anomaly"):
            priors["causality"] = max(priors["causality"], 0.45)
        if by_type.get("flow_spike"):
            priors["collision"] = max(priors["collision"], 0.4)
            priors["gravity"] = max(priors["gravity"], 0.3)

    if not lines:
        lines.append("- (no upstream evidence on the bus — run Stage 1/2 "
                     "pipelines first for better triage)")
    return priors, lines


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg            = json.loads(settings) if settings else {}
    model          = str(cfg.get("model") or "openai:gpt-4o-mini")
    api_key        = str(cfg.get("api_key", "")).strip()
    max_hypotheses = max(1, int(cfg.get("max_hypotheses", 4)))
    max_keyframes  = max(2, min(6, int(cfg.get("max_keyframes", 4))))

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, raw_fps = await loop.run_in_executor(None, load_frames, video_path)
    n = len(frames)
    if n < 3:
        yield {"type": "error", "text": f"Video too short ({n} frames)."}
        return
    eff_fps = raw_fps or 30.0

    # ── 1. Gather evidence + heuristic priors ─────────────────────────────────
    ev_tr   = EVIDENCE.get(vid, "s2_object_tracker")
    ev_traj = EVIDENCE.get(vid, "s2_trajectory_extractor")
    ev_loc  = EVIDENCE.get(vid, "s2_event_localizer")
    priors, ev_lines = _heuristic_priors(ev_traj, ev_loc, ev_tr)
    yield {"type": "log", "level": "info",
           "text": "Evidence summary: " + "; ".join(l.lstrip("- ") for l in ev_lines)}
    await asyncio.sleep(0)

    # ── 2. Keyframes around the strongest flagged moments ────────────────────
    key_ts: list[float] = []
    if ev_loc and ev_loc.get("markers"):
        ms = sorted(ev_loc["markers"], key=lambda m: -m.get("severity", 0))
        key_ts = [m["t_center"] for m in ms[:max_keyframes]]
    elif ev_traj and ev_traj.get("signals"):
        fps_tr = float(ev_traj.get("fps") or eff_fps)
        key_ts = [s["frame"] / fps_tr for s in ev_traj["signals"][:max_keyframes]]
    if not key_ts:
        key_ts = list(np.linspace(0, (n - 1) / eff_fps, max_keyframes + 2)[1:-1])
        yield {"type": "log", "level": "warn",
               "text": "No flagged moments upstream — using evenly spaced keyframes."}
    key_ts = sorted(set(round(float(t), 2) for t in key_ts))[:max_keyframes]

    tiles = []
    for t in key_ts:
        fi = min(n - 1, max(0, int(round(t * eff_fps))))
        tiles.append(_tile(frames[fi], f"t={t:.2f}s"))
    gap = np.full((tiles[0].shape[0], 10, 3), 255, np.uint8)
    composite = tiles[0]
    for tl in tiles[1:]:
        composite = np.concatenate([composite, gap, tl], axis=1)

    # ── 3. VLM triage (heuristics-only fallback) ──────────────────────────────
    from tools.vlm_router import ask_vision_json, key_status
    have_api, key_desc = key_status(model, api_key)
    vlm_hyps: list[dict] = []
    if have_api:
        catalog = "\n".join(f"- {k}: {d}" for k, (_nm, d) in SPECIALISTS.items())
        prompt = TRIAGE_PROMPT.format(evidence="\n".join(ev_lines), catalog=catalog)
        try:
            yield {"type": "log", "level": "info",
                   "text": f"VLM triage over {len(tiles)} keyframe(s) via {model}…"}
            parsed = await ask_vision_json(prompt, composite, model, api_key)
            for h in (parsed.get("hypotheses") or []):
                sid = str(h.get("specialist", "")).lower().strip()
                if sid not in SPECIALISTS:
                    continue
                tw = h.get("t_window") or []
                vlm_hyps.append({
                    "specialist": sid,
                    "confidence": float(np.clip(float(h.get("confidence", 0.5)), 0, 1)),
                    "reason": str(h.get("reason", ""))[:250],
                    "t_window": ([round(float(tw[0]), 2), round(float(tw[1]), 2)]
                                 if len(tw) == 2 else None),
                    "source": "vlm",
                })
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "log", "level": "warn",
                   "text": f"VLM triage failed ({str(exc)[:160]}) — using "
                           "heuristic priors only."}
    else:
        yield {"type": "log", "level": "warn",
               "text": f"No VLM credentials ({key_desc}) — heuristic priors only."}

    # ── 4. Merge: per specialist, max(heuristic prior, VLM confidence) ───────
    merged: dict[str, dict] = {}
    for sid, p in priors.items():
        if p > 0:
            merged[sid] = {"specialist": sid, "confidence": round(p, 3),
                           "reason": "quantitative evidence (see summary)",
                           "t_window": None, "source": "heuristic"}
    for h in vlm_hyps:
        cur = merged.get(h["specialist"])
        if cur is None or h["confidence"] >= cur["confidence"]:
            if cur and cur["source"] == "heuristic":
                h = dict(h, confidence=max(h["confidence"], cur["confidence"]),
                         source="heuristic+vlm")
            merged[h["specialist"]] = h
        else:
            cur["reason"] = (cur["reason"] + f" | VLM: {h['reason']}")[:300]
            cur["source"] = "heuristic+vlm"

    ranked = sorted(merged.values(), key=lambda h: -h["confidence"])[:max_hypotheses]

    if not ranked:
        yield {"type": "log", "level": "success",
               "text": "No physics-failure hypotheses — nothing suspicious upstream "
                       "and the VLM found no likely failure modes."}
    for i, h in enumerate(ranked):
        name = SPECIALISTS[h["specialist"]][0]
        win = (f" (t={h['t_window'][0]}–{h['t_window'][1]}s)" if h.get("t_window") else "")
        lvl = "warn" if h["confidence"] >= 0.5 else "info"
        yield {"type": "log", "level": lvl,
               "text": f"#{i + 1} {name} — {h['confidence']:.0%}{win}: {h['reason']}"}

    ok, buf = cv2.imencode(".jpg", composite, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        yield {"type": "image", "mime": "image/jpeg",
               "data": base64.b64encode(buf).decode(),
               "caption": "Keyframes shown to the VLM for triage "
                          "(sampled at flagged moments)."}

    # Confidence bar chart.
    if ranked:
        names = [SPECIALISTS[h["specialist"]][0] for h in ranked][::-1]
        confs = [h["confidence"] for h in ranked][::-1]
        colors = [_sev_color(c * 100) for c in confs]
        fig = go.Figure(go.Bar(x=confs, y=names, orientation="h",
                               marker_color=colors,
                               text=[f"{c:.0%}" for c in confs],
                               textposition="outside"))
        fig.update_xaxes(range=[0, 1.05], title_text="Confidence",
                         showgrid=True, gridcolor="#ebebeb")
        fig.update_layout(
            title=dict(text="Recommended Stage 3 Specialists", font=dict(size=15)),
            height=160 + 40 * len(ranked),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=170, r=60, t=60, b=50),
            font=dict(family="IBM Plex Sans, sans-serif", size=13),
        )
        yield {"type": "plotly", "data": fig.to_json(),
               "caption": "Run the top specialists in Stage 3; confidence blends "
                          "quantitative priors with VLM triage."}

    for h in ranked:
        yield {"type": "metric", "label": SPECIALISTS[h["specialist"]][0],
               "value": f"{h['confidence']:.0%}",
               "sub": f"{h['source']}" + (f" · t={h['t_window'][0]}–{h['t_window'][1]}s"
                                          if h.get("t_window") else "")}

    # No severity is emitted: this is a router/ranker, not a suspicion detector.
    # Its "top-hypothesis confidence" was effectively always ~100% and polluted
    # the cross-video benchmark's severity column, so it's reported only as
    # per-specialist confidence metrics and the result payload — never a % score.

    yield {"type": "result", "status": "ok",
           "hypotheses": ranked, "evidence_summary": ev_lines,
           "keyframe_ts": key_ts}
    EVIDENCE.put(vid, "s2_hypothesis_generator", {
        "hypotheses": ranked, "evidence_summary": ev_lines,
    })
    yield {"type": "done"}
