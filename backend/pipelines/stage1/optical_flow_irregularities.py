"""
Stage 1 · Test 2 — Optical Flow Irregularities
-----------------------------------------------
Track Shi-Tomasi corners with Lucas-Kanade across every frame pair.
Detect flow magnitude spikes, directional chaos, and flow discontinuities.
"""
import asyncio, json
from typing import AsyncGenerator

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video import load_frames, frame_to_gray
from tools.flow  import detect_keypoints, track_keypoints, flow_vectors, direction_consistency


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg           = json.loads(settings) if settings else {}
    num_kp        = max(1,   int(cfg.get("num_keypoints",        30)))
    spike_thresh  = float(cfg.get("spike_threshold",       20.0))
    cons_floor    = float(cfg.get("consistency_threshold",  0.5))

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    n = len(frames)
    if n < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return

    yield {"type": "log", "level": "info",
           "text": f"{n} frames @ {fps:.1f} fps — initialising {num_kp} tracks…"}
    await asyncio.sleep(0)

    gray0    = frame_to_gray(frames[0])
    curr_pts = detect_keypoints(gray0, n=num_kp)
    if curr_pts is None or len(curr_pts) == 0:
        yield {"type": "error", "text": "No keypoints detected in first frame."}
        return

    mag_series, cons_series = [], []
    flagged_mag, flagged_cons = [], []
    prev_gray = gray0

    for f in range(1, n):
        curr_gray = frame_to_gray(frames[f])

        good_prev, good_curr = track_keypoints(prev_gray, curr_gray, curr_pts)
        if len(good_curr) < 2:
            curr_pts = detect_keypoints(curr_gray, n=num_kp)
            mag_series.append(0.0)
            cons_series.append(0.0)
            prev_gray = curr_gray
            continue

        curr_pts = good_curr.reshape(-1, 1, 2)
        mag, angle = flow_vectors(good_prev.reshape(-1, 2), good_curr.reshape(-1, 2))
        mean_mag   = float(mag.mean())
        cons       = direction_consistency(angle)

        mag_series.append(mean_mag)
        cons_series.append(cons)

        if mean_mag > spike_thresh:
            flagged_mag.append(f)
        if cons < cons_floor and mean_mag > 2.0:
            flagged_cons.append(f)

        prev_gray = curr_gray

        if f % 60 == 0:
            yield {"type": "log", "level": "info", "text": f"Optical flow… {f}/{n} frames"}
            await asyncio.sleep(0)

    yield {"type": "log", "level": "info", "text": "Plotting results…"}
    await asyncio.sleep(0)

    time_ax = [i / fps for i in range(1, n)]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=("Flow Magnitude (px/frame)", "Direction Consistency (0–1)"),
    )

    fig.add_trace(go.Scatter(
        x=time_ax, y=mag_series, mode="lines", name="mean flow magnitude",
        line=dict(color="#1a54c4", width=1.5),
        fill="tozeroy", fillcolor="rgba(26,84,196,0.08)",
    ), row=1, col=1)
    fig.add_hline(
        y=spike_thresh, line=dict(color="red", dash="dash", width=1.2),
        annotation_text=f"spike threshold = {spike_thresh} px",
        annotation_position="top right", row=1, col=1,
    )
    if flagged_mag:
        fm_x = [fi / fps for fi in flagged_mag]
        fm_y = [mag_series[fi - 1] if 0 < fi <= len(mag_series) else 0.0 for fi in flagged_mag]
        fig.add_trace(go.Scatter(
            x=fm_x, y=fm_y, mode="markers", name="flow spike",
            marker=dict(color="red", size=6, symbol="circle"),
            hovertemplate="<b>t=%{x:.2f}s</b><br>mag=%{y:.1f}px<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_ax, y=cons_series, mode="lines", name="direction consistency",
        line=dict(color="#7c3aed", width=1.5),
    ), row=2, col=1)
    fig.add_hline(
        y=cons_floor, line=dict(color="orange", dash="dash", width=1.2),
        annotation_text=f"floor = {cons_floor}",
        annotation_position="top right", row=2, col=1,
    )
    if flagged_cons:
        fc_x = [fi / fps for fi in flagged_cons]
        fc_y = [cons_series[fi - 1] if 0 < fi <= len(cons_series) else 0.0 for fi in flagged_cons]
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y, mode="markers", name="erratic direction",
            marker=dict(color="orange", size=6, symbol="circle"),
            hovertemplate="<b>t=%{x:.2f}s</b><br>cons=%{y:.2f}<extra></extra>",
        ), row=2, col=1)

    fig.update_yaxes(title_text="px / frame", row=1, col=1, showgrid=True, gridcolor="#ebebeb")
    fig.update_yaxes(title_text="consistency", range=[0, 1], row=2, col=1,
                     showgrid=True, gridcolor="#ebebeb")
    fig.update_xaxes(title_text="Time (s)", row=2, col=1, showgrid=True, gridcolor="#ebebeb")
    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb", row=1, col=1)
    fig.update_layout(
        title=dict(text="Optical Flow Irregularities", font=dict(size=15, color="#1a1917")),
        height=460, legend=dict(orientation="h", y=1.07),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=40, t=90, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )

    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": "Top: flow magnitude (red spike lines). Bottom: directional consistency.",
    }

    # ── Structured signals for Stage 2 Event Localizer ────────────────────────
    # Each flagged frame becomes a defect signal the localizer can segment around.
    signals = []
    for fi in flagged_mag:
        mag = mag_series[fi - 1] if 0 < fi <= len(mag_series) else 0.0
        signals.append({"frame": int(fi), "signal_type": "flow_spike",
                        "score": round(float(mag) / max(spike_thresh, 1e-6), 3)})
    for fi in flagged_cons:
        cons = cons_series[fi - 1] if 0 < fi <= len(cons_series) else 0.0
        signals.append({"frame": int(fi), "signal_type": "flow_chaos",
                        "score": round(float(1.0 - cons), 3)})
    peak_mag  = float(max(mag_series,  default=0))
    mean_mag  = float(np.mean(mag_series)  if mag_series  else 0)
    mean_cons = float(np.mean(cons_series) if cons_series else 0)
    n_flagged = len(set(flagged_mag + flagged_cons))
    spike_score = min(int(len(flagged_mag)  / max(n, 1) * 250), 100)
    chaos_score = min(int(len(flagged_cons) / max(n, 1) * 250), 100)
    score     = min(int(n_flagged / max(n, 1) * 250), 100)
    color     = "#E24B4A" if score > 40 else "#EF9F27" if score > 15 else "#4CAF50"

    # Emit signals with per-diagnostic severity so the Event Localizer can drop
    # weak diagnostics (e.g. a 2% directional-flag) instead of marking them.
    yield {"type": "signal", "source": "s1_optical_flow", "source_name": "Optical Flow",
           "fps": float(fps), "n_frames": int(n), "severity": score,
           "type_severities": {"flow_spike": spike_score, "flow_chaos": chaos_score},
           "signals": signals}

    yield {"type": "metric", "label": "Flagged frames",       "value": str(n_flagged),     "sub": "spike or chaotic direction"}
    yield {"type": "metric", "label": "Peak flow",            "value": f"{peak_mag:.1f}",   "sub": "px/frame"}
    yield {"type": "metric", "label": "Mean flow",            "value": f"{mean_mag:.1f}",   "sub": "px/frame"}
    yield {"type": "metric", "label": "Mean dir consistency", "value": f"{mean_cons:.2f}",  "sub": "0 = random · 1 = uniform"}
    yield {"type": "severity", "label": "Optical flow anomaly score", "value": score, "color": color}

    parts = []
    if flagged_mag:  parts.append(f"{len(flagged_mag)} flow-spike frame(s)")
    if flagged_cons: parts.append(f"{len(flagged_cons)} direction-chaos frame(s)")
    msg = (", ".join(parts) + " detected.") if parts else "No optical flow irregularities detected."
    yield {"type": "log", "level": "warn" if parts else "success", "text": msg}
    yield {"type": "done"}
