"""
Stage 2 · Step 3 — Event Localizer
--------------------------------------
Consume the structured anomaly *signals* emitted by Stage 1 screening tests
and crop the video timeline into per-diagnostic **failure markers** — the
search space later handed to the physics specialists.

Design
------
Earlier versions merged every signal (regardless of type) into one window with
a wide context pad.  Dense diagnostics (e.g. directional chaos firing on most
frames) then chained into a single block that swallowed everything.

This version instead:
  1. Groups signals by diagnostic type — each diagnostic gets its own lane and
     is never merged with another.
  2. Buckets each diagnostic's signals into fixed `window_seconds` windows, so a
     pervasive diagnostic produces many distinct per-second markers (its lane
     fills up) while a sparse diagnostic produces a few isolated markers.
  3. Locates *where* on the frame each marker's motion sits (frame-difference
     ROI) for the viewer overlay.

Output (per marker, inside the marker_video event):
  {
    "id": int, "lane": int,
    "type": str, "type_label": str, "color": "#hex",
    "frame_start": int, "frame_end": int,
    "t_start": float, "t_end": float, "t_center": float,
    "severity": int, "n_signals": int,
    "region": {"x","y","w","h"} | None
  }

Stage 1 signals arrive in `settings.stage1_signals` (forwarded by the UI).
"""
import asyncio, base64, json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go

from tools.video import load_frames, frame_to_gray
from tools.evidence import EVIDENCE, video_id


def _encode_frames(frames: list, idxs: list, fps: float, max_w: int, quality: int):
    """Shared JPEG encoder. Returns (b64_list, time_list)."""
    if not frames or not idxs:
        return [], []
    H, W = frames[0].shape[:2]
    scale = min(1.0, max_w / W)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    out, times = [], []
    for i in idxs:
        img = frames[i]
        if scale < 1.0:
            img = cv2.resize(img, (max(2, int(W * scale)), max(2, int(H * scale))))
        ok, buf = cv2.imencode(".jpg", img, enc)
        if not ok:
            continue
        out.append(base64.b64encode(buf).decode())
        times.append(round(i / max(fps, 1e-6), 4))
    return out, times


def _build_frame_strip(frames: list, fps: float,
                       max_frames: int = 300, max_w: int = 440, quality: int = 52):
    """Sparse overview strip covering the full timeline (~1–2 fps).

    Used for timeline scrubbing between events. Low quality keeps payload small.
    """
    n = len(frames)
    if n == 0:
        return [], []
    idxs = (list(range(n)) if n <= max_frames
            else [int(i) for i in np.linspace(0, n - 1, max_frames)])
    return _encode_frames(frames, idxs, fps, max_w, quality)


def _build_event_clip(frames: list, fps: float, lo: int, hi: int,
                      context_s: float = 1.5, target_fps: float = 10.0,
                      max_frames: int = 60, max_w: int = 440, quality: int = 68):
    """Dense clip for a single event window with context padding on both sides.

    Samples at target_fps (or native fps if lower) so playback looks smooth
    near each marker, without encoding the entire video at high density.
    """
    n = len(frames)
    if n == 0:
        return [], []
    ctx = max(0, int(round(context_s * fps)))
    f0 = max(0, lo - ctx)
    f1 = min(n - 1, hi + ctx)
    step = max(1, int(round(fps / max(target_fps, 0.5))))
    seg = list(range(f0, f1 + 1, step))
    # Always include the marker center frame so the key moment is never skipped.
    ctr = (lo + hi) // 2
    if ctr not in seg:
        import bisect; bisect.insort(seg, ctr)
    if len(seg) > max_frames:
        idxs = [int(i) for i in np.linspace(0, len(seg) - 1, max_frames)]
        seg = [seg[i] for i in idxs]
    return _encode_frames(frames, seg, fps, max_w, quality)

# ── Friendly names + a stable color per Stage 1 signal type ───────────────────
_TRIGGER_LABELS = {
    "flow_spike":       "Flow spike",
    "flow_chaos":       "Erratic motion direction",
    "temporal_anomaly": "Temporal jump",
    "embedding_jump":   "Appearance jump",
    "camera_motion":    "Camera motion",
}
_TYPE_COLORS = {
    "flow_spike":       "#e24b4a",
    "flow_chaos":       "#d97706",
    "temporal_anomaly": "#7c3aed",
    "embedding_jump":   "#1a54c4",
    "camera_motion":    "#0891b2",
}
# Lane display order (any unknown types are appended after these).
_LANE_ORDER = ["flow_spike", "flow_chaos", "temporal_anomaly", "embedding_jump", "camera_motion"]
_EXTRA_COLORS = ["#0f766e", "#be185d", "#7e22ce", "#1a7a3c", "#c05621"]


def _sev_color(sev: int) -> str:
    return "#E24B4A" if sev > 60 else "#EF9F27" if sev > 30 else "#4CAF50"


def _trigger_label(sig_type: str) -> str:
    return _TRIGGER_LABELS.get(sig_type, sig_type.replace("_", " ").title())


def _type_color(sig_type: str, fallback_idx: int = 0) -> str:
    return _TYPE_COLORS.get(sig_type, _EXTRA_COLORS[fallback_idx % len(_EXTRA_COLORS)])


def _collect_signals(cfg: dict, eff_fps: float, n: int) -> list[dict]:
    """Normalise forwarded Stage 1 signals into {t, frame, signal_type, score}.

    Accepts a flat list of signals or the grouped {source: {fps, signals}} shape.
    Frame indices are remapped onto this run's timeline via timestamps so that
    sources sampled at different rates still line up.
    """
    raw = cfg.get("stage1_signals", [])
    groups: list[dict] = []
    if isinstance(raw, dict):
        groups = list(raw.values())
    elif isinstance(raw, list):
        if raw and isinstance(raw[0], dict) and "signals" in raw[0]:
            groups = raw
        else:
            groups = [{"fps": eff_fps, "signals": raw}]

    out: list[dict] = []
    for g in groups:
        g_fps    = float(g.get("fps") or eff_fps) or eff_fps
        src_name = g.get("source_name") or g.get("source") or "Stage 1 test"
        g_sev    = g.get("severity")
        type_sev = g.get("type_severities") or {}
        for s in g.get("signals", []):
            if "frame" not in s and "t" not in s:
                continue
            t = float(s["t"]) if "t" in s else float(s["frame"]) / max(g_fps, 1e-6)
            frame = max(0, min(int(round(t * eff_fps)), n - 1))
            stype = s.get("signal_type", "anomaly")
            tsev  = type_sev.get(stype, g_sev)
            out.append({
                "t": t, "frame": frame,
                "signal_type": stype,
                "score": float(s.get("score", 1.0)),
                "source_name": src_name,
                "test_severity": (float(tsev) if tsev is not None else None),
            })
    out.sort(key=lambda d: d["t"])
    return out


def _motion_roi(frames: list, lo: int, hi: int, max_probe: int = 48) -> Optional[dict]:
    """Bounding box (source px) of the dominant inter-frame motion within a
    window, via accumulated absolute frame differencing. None if no motion."""
    hi = max(hi, lo + 1)
    idxs = list(range(lo, min(hi + 1, len(frames))))
    if len(idxs) < 2:
        return None
    if len(idxs) > max_probe:
        idxs = list(np.linspace(idxs[0], idxs[-1], max_probe, dtype=int))

    H, W = frames[lo].shape[:2]
    accum = np.zeros((H, W), np.float32)
    prev = frame_to_gray(frames[idxs[0]])
    for fi in idxs[1:]:
        cur = frame_to_gray(frames[fi])
        accum += cv2.absdiff(cur, prev).astype(np.float32)
        prev = cur
    if accum.max() < 1e-3:
        return None

    norm = (accum / accum.max() * 255).astype(np.uint8)
    _, mask = cv2.threshold(norm, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w * h < (W * H) * 0.0008:
        return None
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg              = json.loads(settings) if settings else {}
    window_seconds   = max(0.1, float(cfg.get("window_seconds", 0.5)))
    min_signals      = max(1, int(cfg.get("min_signals_per_window", 1)))
    min_test_severity = float(cfg.get("min_test_severity", 15))   # %; gate weak diagnostics

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    n = len(frames)
    if n < 2:
        yield {"type": "error", "text": f"Video too short ({n} frames)."}
        return
    H, W = frames[0].shape[:2]
    eff_fps = fps or 30.0
    duration = n / eff_fps

    signals = _collect_signals(cfg, eff_fps, n)
    if not signals:
        yield {"type": "log", "level": "warn",
               "text": ("No Stage 1 signals received — run a Stage 1 screening test "
                        "(e.g. Optical Flow or Temporal Smoothness) first, then re-run "
                        "the Event Localizer to populate defect markers.")}
    else:
        by_type: dict[str, int] = {}
        for s in signals:
            by_type[s["signal_type"]] = by_type.get(s["signal_type"], 0) + 1
        breakdown = ", ".join(f"{_trigger_label(k)}×{v}" for k, v in by_type.items())
        yield {"type": "log", "level": "info",
               "text": f"{len(signals)} Stage 1 signal(s) across {len(by_type)} "
                       f"diagnostic(s): {breakdown}."}
    await asyncio.sleep(0)

    # ── Group by diagnostic, then drop low-significance ones ──────────────────
    type_groups: dict[str, list[dict]] = {}
    for s in signals:
        type_groups.setdefault(s["signal_type"], []).append(s)

    # Per-diagnostic source test + its reported severity (significance gate).
    src_of:  dict[str, str] = {}
    sev_of:  dict[str, Optional[float]] = {}
    for t, ss in type_groups.items():
        src_of[t] = next((x.get("source_name") for x in ss if x.get("source_name")), "Stage 1 test")
        sev_of[t] = next((x["test_severity"] for x in ss if x.get("test_severity") is not None), None)

    dropped = [(t, sev_of[t]) for t in type_groups
               if sev_of[t] is not None and sev_of[t] < min_test_severity]
    for t, _ in dropped:
        type_groups.pop(t, None)
    if dropped:
        listing = ", ".join(f"{_trigger_label(t)} ({src_of[t]}, {sev:.0f}%)" for t, sev in dropped)
        yield {"type": "log", "level": "info",
               "text": f"Skipped {len(dropped)} low-significance diagnostic(s) "
                       f"below {min_test_severity:.0f}% test severity: {listing}."}

    lane_types = ([t for t in _LANE_ORDER if t in type_groups]
                  + [t for t in type_groups if t not in _LANE_ORDER])
    lanes = [{"type": t, "label": _trigger_label(t), "color": _type_color(t, i),
              "source": src_of[t],
              "severity": (round(sev_of[t]) if sev_of[t] is not None else None)}
             for i, t in enumerate(lane_types)]
    lane_index = {t: i for i, t in enumerate(lane_types)}

    # ── Cluster each diagnostic into one marker per contiguous event ──────────
    # Group consecutive signals (gap ≤ window_seconds) so a single spike that
    # spans several frames — or a clock boundary — stays ONE marker, matching how
    # the Stage 1 test counts its spikes. Distinct events stay separate.
    gap_frames = max(1, int(round(window_seconds * eff_fps)))
    raw_markers: list[dict] = []
    for t in lane_types:
        sigs = sorted(type_groups[t], key=lambda s: s["frame"])
        clusters: list[list[dict]] = []
        for s in sigs:
            if clusters and s["frame"] - clusters[-1][-1]["frame"] <= gap_frames:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        for members in clusters:
            if len(members) < min_signals:
                continue
            mframes = [m["frame"] for m in members]
            lo, hi  = min(mframes), max(mframes)
            peak    = max(m["score"] for m in members)
            sev     = int(min(100, 20 + 12 * len(members) + 25 * min(peak, 2.0)))
            raw_markers.append({
                "lane": lane_index[t], "type": t, "type_label": _trigger_label(t),
                "source": src_of[t], "color": _type_color(t, lane_index[t]),
                "frame_start": int(lo), "frame_end": int(hi),
                "t_start": round(lo / eff_fps, 3),
                "t_end":   round(max(hi, lo + 1) / eff_fps, 3),
                "t_center": round(sum(mframes) / len(mframes) / eff_fps, 3),
                "severity": sev, "n_signals": len(members),
                "_lohi": (lo, hi),
            })

    raw_markers.sort(key=lambda m: (m["t_start"], m["lane"]))

    yield {"type": "log", "level": "info",
           "text": f"Localized {len(raw_markers)} event marker(s) across {len(lanes)} "
                   f"diagnostic lane(s) (signals merged within {window_seconds:g}s)."}
    await asyncio.sleep(0)

    # ── Assign ids + per-marker motion ROI ────────────────────────────────────
    markers: list[dict] = []
    for i, m in enumerate(raw_markers):
        lo, hi = m.pop("_lohi")
        m["id"] = i
        m["region"] = _motion_roi(frames, lo, hi)
        markers.append(m)
        if i % 6 == 0:
            await asyncio.sleep(0)

    # ── Build two-tier JPEG frame data for the viewer ─────────────────────────
    # Overview: sparse (~1–2 fps) strip covering the full timeline for scrubbing.
    # Event clips: dense (≤10 fps) windows around each marker for smooth playback.
    # No ffmpeg needed; frame-exact seeking; never hangs on malformed clips.
    yield {"type": "log", "level": "info", "text": "Building viewer frames…"}
    await asyncio.sleep(0)
    try:
        loop = asyncio.get_event_loop()

        # Overview: target ~2 fps, capped at 300 frames regardless of duration.
        overview_count = max(60, min(300, int(duration * 2) + 1))
        strip_frames, strip_times = await loop.run_in_executor(
            None, _build_frame_strip, frames, eff_fps, overview_count)

        # Event clips: one dense clip per marker, all built in a single executor call.
        def _build_all_event_clips():
            for m in markers:
                ef, et = _build_event_clip(frames, eff_fps,
                                           m["frame_start"], m["frame_end"])
                m["event_frames"] = ef
                m["event_times"]  = et

        await loop.run_in_executor(None, _build_all_event_clips)

        event_total  = sum(len(m.get("event_frames", [])) for m in markers)
        payload_kb   = (
            sum(len(f) for f in strip_frames) +
            sum(len(f) for m in markers for f in m.get("event_frames", []))
        ) / 1024

        yield {
            "type": "marker_video",
            "mode": "frames",
            "frames": strip_frames,
            "frame_times": strip_times,
            "fps": round(eff_fps, 3),
            "duration": round(duration, 3),
            "src_width": W, "src_height": H,
            "lanes": lanes,
            "markers": markers,
            "caption": ("Defect timeline — one lane per diagnostic. Colored bands mark "
                        "when each error fires. Use ◀ ▶ to skip between markers; click a "
                        "band or list row to jump."),
        }
        yield {"type": "log", "level": "info",
               "text": (f"Viewer ready — {len(strip_frames)} overview + "
                        f"{event_total} event frame(s) across {len(markers)} clip(s), "
                        f"{payload_kb:.0f} KB total.")}
    except Exception as exc:
        yield {"type": "log", "level": "warn",
               "text": f"Frame strip build failed ({exc}) — showing diagnostics only."}
    await asyncio.sleep(0)

    # ── Diagnostics: per-lane timeline plot ───────────────────────────────────
    yield {"type": "log", "level": "info", "text": "Building diagnostics timeline…"}
    fig = go.Figure()
    for lane in lanes:
        ms = [m for m in markers if m["type"] == lane["type"]]
        # Thick colored line segment for each marker window.
        xs, ys = [], []
        for m in ms:
            xs += [m["t_start"], m["t_end"], None]
            ys += [lane["label"], lane["label"], None]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=lane["color"], width=11),
                name=lane["label"], showlegend=False, hoverinfo="skip",
            ))
        # Center points carry the hover detail.
        fig.add_trace(go.Scatter(
            x=[m["t_center"] for m in ms], y=[lane["label"]] * len(ms),
            mode="markers",
            marker=dict(color=lane["color"], size=9, line=dict(color="white", width=1.2)),
            name=lane["label"], legendgroup=lane["type"],
            customdata=[[m["id"] + 1, m["severity"], m["n_signals"]] for m in ms],
            hovertemplate=(f"<b>{lane['label']}</b><br>"
                           "marker #%{customdata[0]} · t = %{x:.3f}s<br>"
                           "severity %{customdata[1]} · %{customdata[2]} signal(s)"
                           "<extra></extra>"),
        ))
    # Adaptive tick spacing so 1–2 s clips show sub-second detail.
    xaxis_kw = dict(title_text="Time (s)", range=[0, duration],
                    showgrid=True, gridcolor="#ebebeb", hoverformat=".3f")
    if   duration <= 3:  xaxis_kw.update(dtick=0.25, tickformat=".2f")
    elif duration <= 8:  xaxis_kw.update(dtick=0.5,  tickformat=".1f")
    elif duration <= 30: xaxis_kw.update(dtick=2.0,  tickformat=".0f")
    fig.update_xaxes(**xaxis_kw)
    fig.update_yaxes(categoryorder="array",
                     categoryarray=[l["label"] for l in lanes][::-1],
                     showgrid=True, gridcolor="#f2f2f2")
    fig.update_layout(
        title=dict(text="Event Localizer — Defect Markers per Diagnostic", font=dict(size=15)),
        height=170 + 46 * max(1, len(lanes)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=130, r=40, t=70, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        showlegend=False,
    )
    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": "Each row is one Stage 1 diagnostic; colored bands are the seconds it flagged.",
    }

    # ── Metrics + severity ────────────────────────────────────────────────────
    flagged_frames = sum(m["frame_end"] - m["frame_start"] + 1 for m in markers)
    coverage = min(1.0, flagged_frames / max(n, 1))
    peak_sev = max((m["severity"] for m in markers), default=0)

    yield {"type": "metric", "label": "Defect markers", "value": str(len(markers)),
           "sub": f"across {len(lanes)} diagnostic(s)"}
    for lane in lanes:
        cnt = sum(1 for m in markers if m["type"] == lane["type"])
        sev_txt = f" · test {lane['severity']}%" if lane["severity"] is not None else ""
        yield {"type": "metric", "label": f"{lane['label']}", "value": f"{cnt} marker(s)",
               "sub": f"from “{lane['source']}”{sev_txt}"}
    yield {"type": "metric", "label": "Flagged span", "value": f"{coverage:.0%}",
           "sub": f"{flagged_frames}/{n} frames of timeline"}

    overall = int(min(100, peak_sev * 0.6 + min(len(markers), 12) / 12 * 40)) if markers else 0
    yield {"type": "severity", "label": "Localization confidence",
           "value": overall, "color": _sev_color(overall)}

    # Publish markers on the evidence bus so Stage 3 specialists can read them
    # server-side (frame data stripped — downstream stages reload frames).
    EVIDENCE.put(video_id(video_path), "s2_event_localizer", {
        "fps": eff_fps, "n_frames": n, "duration": duration,
        "markers": [{k: v for k, v in m.items()
                     if k not in ("event_frames", "event_times")} for m in markers],
        "lanes": lanes,
    })

    msg = (f"{len(markers)} marker(s) localized across {len(lanes)} diagnostic(s); "
           f"peak severity {peak_sev}." if markers
           else "No markers — no Stage 1 signals to localize.")
    yield {"type": "log", "level": "success" if markers else "warn", "text": msg}
    yield {"type": "done"}
