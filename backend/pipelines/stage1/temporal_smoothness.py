"""
Stage 1 · Test 1 — Temporal Smoothness Anomalies
-------------------------------------------------
Measures MOTION INCOHERENCE: the share of frame-to-frame motion that cannot be
explained by a single global (camera) transform.

Why not raw acceleration (the previous approach): differentiating keypoint
positions twice amplifies tracking noise by fps², and a fixed px/s² threshold is
neither resolution- nor fps-invariant. On the bundled corpus that scored *real*
handheld footage at a saturated 100 — it was measuring camera shake and pixel
jitter, not physics. Worse, the severity formula (`n_flagged/n * 300`) hit its
ceiling once a third of frames were flagged, so nearly every clip returned
exactly 100 and the test carried no information.

What actually separates AI-generated from real footage is *coherence*, not
magnitude. A real scene is rigid: however much the camera moves, one global
affine explains almost all background motion, and the few genuinely-moving
objects deviate smoothly. Generative video has no such constraint — textures
swim, background points drift independently, and detail is re-synthesised each
frame, so points scatter off any global model.

Per consecutive frame pair we therefore:
  1. detect corners and track them with Lucas-Kanade,
  2. fit a robust global affine (RANSAC) — this absorbs all real camera motion,
  3. measure each point's residual from that global model.

The residual is normalised to **frame-diagonals per second**, making it
invariant to both resolution and frame rate. Severity is the geometric mean of
the median (typical incoherence) and 90th-percentile (worst-case) residual,
mapped through a log ramp — incoherence spans four orders of magnitude across
real and generated clips, so a linear scale would be useless.

Separately, frames whose incoherence spikes far above the clip's *own* robust
baseline are flagged as localised glitches and handed to the Stage 2 Event
Localizer.
"""
import asyncio
import json
import math
from typing import AsyncGenerator

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video import frame_to_gray, iter_frames, probe_video

# Calibration in frame-diagonals/second, from the bundled AI/real corpus
# (21 generated vs 11 real clips). Below CLEAN a clip is indistinguishable from
# rigid real footage; at/above BROKEN the scene is not moving rigidly at all.
CLEAN_INCOHERENCE = 3e-4
BROKEN_INCOHERENCE = 3e-2

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
_MIN_POINTS = 8          # fewest tracked points that still support an affine fit


def _score(x: float) -> int:
    """Map incoherence (frame-diagonals/s) to 0-100 on a log ramp."""
    if x <= 0 or not math.isfinite(x):
        return 0
    lo, hi = math.log10(CLEAN_INCOHERENCE), math.log10(BROKEN_INCOHERENCE)
    frac = (math.log10(x) - lo) / (hi - lo)
    return int(round(100 * min(max(frac, 0.0), 1.0)))


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg       = json.loads(settings) if settings else {}
    num_kp    = max(_MIN_POINTS, int(cfg.get("num_keypoints", 120)))
    max_pairs = max(4, int(cfg.get("max_pairs", 120)))
    spike_k   = float(cfg.get("spike_sensitivity", 6.0))
    work_size = max(240, int(cfg.get("work_size", 1280)))

    yield {"type": "log", "level": "info", "text": "Opening video…"}
    # Streamed, not materialised: only two grayscale frames are ever held, so a
    # long 4K clip costs ~2 MB here instead of the ~11 GB `load_frames` would
    # need. Corner detection also dominates runtime and scales with pixel count
    # (~125 ms/frame at 4K vs 10 ms at 720p), and the metric is normalised by
    # the frame diagonal — so decoding straight to `work_size` is both far
    # cheaper and measures the same scale-free quantity.
    fps, n_est, (src_h, src_w) = probe_video(video_path)
    fps = float(fps) if fps and fps > 0 else 30.0
    # Analyse every `stride`-th consecutive pair so cost stays bounded on long
    # clips while Δt within a pair remains exactly one frame (keeping the
    # measurement comparable across videos). n_est comes from container
    # metadata and is occasionally wrong, so it only sets the stride — the loop
    # itself is driven by the actual stream and capped by max_pairs.
    stride = max(1, (n_est - 1) // max_pairs) if n_est > 1 else 1
    est_pairs = min(max_pairs, max(1, (n_est - 1) // stride)) if n_est > 1 else max_pairs
    yield {"type": "log", "level": "info",
           "text": f"~{n_est or '?'} frames @ {fps:.1f} fps ({src_w}×{src_h}) — "
                   f"streaming ~{est_pairs} frame pair(s) at ≤{work_size}px…"}
    await asyncio.sleep(0)

    W = H = diag = ransac_thr = None
    t_s, med_s, p90_s, inlier_s, glob_s, frame_s = [], [], [], [], [], []
    prev = None            # previous frame's grayscale, the only frame retained
    n_seen = 0
    for idx, bgr in enumerate(iter_frames(video_path, max_dim=work_size)):
        n_seen = idx + 1
        gray = frame_to_gray(bgr)
        if diag is None:
            H, W = gray.shape[:2]
            diag = float(np.hypot(H, W))
            # RANSAC tolerance scales with resolution so "inlier" means the same
            # thing on a 320p clip and a 4K one.
            ransac_thr = max(1.0, 0.0015 * diag)

        # Only the frame that opens a sampled pair needs its successor kept.
        if prev is not None:
            g0, g1 = prev, gray
            prev = None
            p = cv2.goodFeaturesToTrack(g0, maxCorners=num_kp, qualityLevel=0.01,
                                        minDistance=10, blockSize=7)
            if p is not None and len(p) >= _MIN_POINTS:
                q, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p.astype(np.float32),
                                                    None, **_LK)
                st = st.ravel().astype(bool)
                if st.sum() >= _MIN_POINTS:
                    a = p[st, 0, :].astype(np.float32)
                    b = q[st, 0, :].astype(np.float32)
                    M, mask = cv2.estimateAffinePartial2D(
                        a, b, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr,
                        maxIters=2000, confidence=0.995)
                    # No consensus at all is itself evidence of incoherence: fall
                    # back to "no global motion" so raw displacement is the residual.
                    pred = a if M is None else (a @ M[:, :2].T) + M[:, 2]
                    r = np.linalg.norm(b - pred, axis=1) * fps / diag   # diag/second
                    t_s.append((idx - 0.5) / fps)
                    frame_s.append(idx - 1)        # frame that opens this pair
                    med_s.append(float(np.median(r)))
                    p90_s.append(float(np.percentile(r, 90)))
                    inlier_s.append(0.0 if mask is None else float(mask.mean()))
                    glob_s.append(float(np.linalg.norm(pred - a, axis=1).mean())
                                  * fps / diag)
            if len(med_s) >= max_pairs:
                break
            if len(med_s) and len(med_s) % 40 == 0:
                yield {"type": "log", "level": "info",
                       "text": f"Analysed {len(med_s)} frame pair(s)…"}
                await asyncio.sleep(0)
        elif idx % stride == 0:
            prev = gray        # this frame opens the next sampled pair

    n = n_seen
    if n < 3:
        yield {"type": "error", "text": "Video too short (need ≥ 3 frames)."}
        return

    if len(med_s) < 4:
        yield {"type": "error",
               "text": "Not enough trackable texture in this video to assess "
                       "temporal coherence."}
        return

    med = np.asarray(med_s)
    p90 = np.asarray(p90_s)
    inl = np.asarray(inlier_s)
    glob = np.asarray(glob_s)

    typical = float(np.median(med))          # typical background incoherence
    worst = float(np.median(p90))            # worst-case (per frame) incoherence
    incoherence = math.sqrt(max(typical, 0.0) * max(worst, 0.0))
    score = _score(incoherence)

    # ── Localised glitches: frames spiking above this clip's OWN baseline ─────
    # MAD is used instead of std so a few large spikes don't inflate the very
    # threshold meant to catch them.
    base = float(np.median(med))
    mad = float(np.median(np.abs(med - base)))
    spike_thr = base + spike_k * (mad * 1.4826 if mad > 0 else (med.std() or 1e-9))
    spike_idx = np.where(med > spike_thr)[0]

    yield {"type": "log", "level": "info", "text": "Building plot…"}
    await asyncio.sleep(0)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
        subplot_titles=(
            "Motion incoherence — residual after removing global camera motion",
            "Global (camera) motion, for context — large values here are fine",
        ),
    )
    fig.add_trace(go.Scatter(
        x=t_s, y=med.tolist(), mode="lines", name="typical (median)",
        line=dict(color="#1a54c4", width=1.8),
        hovertemplate="<b>t = %{x:.2f}s</b><br>incoherence = %{y:.5f} diag/s<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_s, y=p90.tolist(), mode="lines", name="worst-case (p90)",
        line=dict(color="#7c3aed", width=1.2, dash="dot"),
        hovertemplate="<b>t = %{x:.2f}s</b><br>p90 = %{y:.5f} diag/s<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=CLEAN_INCOHERENCE, line=dict(color="#4CAF50", dash="dash", width=1.3),
                  annotation_text="clean (rigid scene)", annotation_position="bottom right",
                  annotation_font=dict(color="#4CAF50", size=11), row=1, col=1)
    fig.add_hline(y=BROKEN_INCOHERENCE, line=dict(color="#E24B4A", dash="dash", width=1.3),
                  annotation_text="incoherent", annotation_position="top right",
                  annotation_font=dict(color="#E24B4A", size=11), row=1, col=1)
    for si in spike_idx:
        hw = max(0.5 / fps, 0.02)
        fig.add_vrect(x0=t_s[si] - hw, x1=t_s[si] + hw, fillcolor="#E24B4A",
                      opacity=0.18, line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_s, y=glob.tolist(), mode="lines", name="global motion",
        line=dict(color="#8a8580", width=1.4), showlegend=False,
        hovertemplate="<b>t = %{x:.2f}s</b><br>camera = %{y:.5f} diag/s<extra></extra>",
    ), row=2, col=1)

    _grid = dict(showgrid=True, gridcolor="#ebebeb", gridwidth=1)
    fig.update_xaxes(**_grid)
    fig.update_yaxes(**_grid, zeroline=False)
    # Log scale: incoherence spans several decades across clean vs broken clips.
    fig.update_yaxes(type="log", title_text="diagonals / s", row=1, col=1)
    fig.update_yaxes(title_text="diagonals / s", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(
        title=dict(text="Temporal Smoothness — Motion Coherence Analysis",
                   font=dict(size=15)),
        height=540,
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=65, r=45, t=110, b=55),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        hovermode="x unified",
    )
    yield {"type": "plotly", "data": fig.to_json(),
           "caption": "Top: motion that a single global camera transform cannot "
                      "explain (log scale); red bands are localised spikes. "
                      "Bottom: the global camera motion that was removed."}

    # ── Structured signals for the Stage 2 Event Localizer ───────────────────
    signals = [
        {"frame": int(frame_s[si]), "signal_type": "temporal_anomaly",
         "score": round(float(med[si] / spike_thr), 3)}
        for si in spike_idx
    ]
    yield {"type": "signal", "source": "s1_temporal",
           "source_name": "Temporal Smoothness",
           "fps": fps, "n_frames": int(n), "severity": score,
           "type_severities": {"temporal_anomaly": score}, "signals": signals}

    yield {"type": "metric", "label": "Motion incoherence",
           "value": f"{incoherence:.5f}",
           "sub": f"frame-diagonals/s not explained by camera motion "
                  f"(clean < {CLEAN_INCOHERENCE:g})"}
    yield {"type": "metric", "label": "Rigid-scene agreement",
           "value": f"{100 * float(inl.mean()):.1f}%",
           "sub": "keypoints consistent with one global transform"}
    yield {"type": "metric", "label": "Glitch frames", "value": str(len(spike_idx)),
           "sub": f"of {len(med)} analysed, spiking above this clip's own baseline"}
    yield {"type": "metric", "label": "Camera motion",
           "value": f"{float(np.median(glob)):.5f}",
           "sub": "median global motion (diagonals/s) — removed before scoring"}

    color = "#E24B4A" if score > 60 else "#EF9F27" if score > 30 else "#4CAF50"
    yield {"type": "severity", "label": "Temporal anomaly score",
           "value": score, "color": color}

    if score > 60:
        msg = (f"Severe motion incoherence ({incoherence:.5f} diag/s) — the scene "
               "does not move as one rigid world; typical of generated video.")
        lvl = "warn"
    elif score > 30:
        msg = (f"Moderate motion incoherence ({incoherence:.5f} diag/s) — some "
               "motion is unexplained by camera movement.")
        lvl = "warn"
    else:
        msg = (f"Motion is coherent ({incoherence:.5f} diag/s) — consistent with a "
               "rigid scene under normal camera motion.")
        lvl = "success"
    yield {"type": "log", "level": lvl, "text": msg}
    if len(spike_idx):
        yield {"type": "log", "level": "warn",
               "text": f"{len(spike_idx)} frame(s) spike above this clip's own "
                       "coherence baseline — handed to the Event Localizer."}
    yield {"type": "done"}
