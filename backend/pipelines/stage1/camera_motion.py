"""
Stage 1 — Camera Motion Detector  (KLT Pipeline, v2)
-----------------------------------------------------------
Key principle: shake is TEMPORALLY ERRATIC; intentional camera moves
(pans, tilts, orbits around a subject) are TEMPORALLY SMOOTH — even when fast.

Earlier versions measured the spatial spread of residual motion after
subtracting the median translation.  That falsely fires on smooth orbits:
rotation + depth parallax make points at different depths move different
amounts, inflating "residual" variance even though the motion is silky smooth.

v2 pipeline:

  1. Preprocessing: grayscale, optional downscale, brightness normalisation
     (flicker fix).
  2. Zone-based Shi-Tomasi detection (n_zones × n_zones grid, fresh each pair).
  3. Pyramidal LK optical flow.
  4. Camera velocity v[r] = MEDIAN point displacement (deterministic — RANSAC
     velocity jumps between motion populations under parallax and fakes
     jitter).  A RANSAC similarity transform (translation+rotation+scale) is
     fitted separately, used only for residual analysis.
  5. PRIMARY signal — high-frequency camera jitter (stabilisation decomposition):
        v_smooth = low-pass(v)          # the intentional camera path
        jitter[r] = |v[r] − v_smooth[r]|   # px/frame of tremor
     minus a 0.3 px tracking-noise deadband, discounted by camera speed
     (tremor riding a fast move is imperceptible and mostly measurement noise).
     A fast smooth orbit has near-constant v → jitter ≈ 0.
     Handheld shake oscillates v → jitter large.
  6. SECONDARY signal — model residual: mean reprojection error of RANSAC
     inliers vs the similarity transform, divided by (1 + 0.5·|v_smooth|)
     to discount parallax, which grows with camera speed.
  7. Scores are calibrated in ABSOLUTE pixels (jitter_px maps to score 1.0),
     not per-video percentiles — a smooth video genuinely scores ~0.
  8. Fusion + temporal Gaussian smoothing → per-frame 0–1 score.
  9. Output: time-series plot, zone heatmap (model residual per zone, absolute
     scale), worst-frame motion-vector image.
"""
import asyncio
import base64
import json
from typing import AsyncGenerator, List, Optional, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go

from tools.video import load_frames, frame_to_gray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_brightness(gray: np.ndarray) -> np.ndarray:
    """Rescale frame to mean=128, std=64 — suppresses flicker artefacts."""
    f = gray.astype(np.float32)
    s = f.std()
    if s < 1.0:
        return gray
    return np.clip((f - f.mean()) / s * 64.0 + 128.0, 0, 255).astype(np.uint8)


def _detect_zone_points(
    gray: np.ndarray, n_zones: int, pts_per_zone: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect Shi-Tomasi corners in each cell of an n_zones × n_zones grid.
    Returns:
      pts      — (N, 1, 2) float32 in full-frame coordinates
      zone_ids — (N,) int; zone_ids[i] = flat zone index (row*n_zones+col)
    """
    H, W = gray.shape
    all_pts: List[np.ndarray] = []
    all_ids: List[int] = []
    for zy in range(n_zones):
        for zx in range(n_zones):
            zi  = zy * n_zones + zx
            y0  = int(zy * H / n_zones);  y1 = int((zy + 1) * H / n_zones)
            x0  = int(zx * W / n_zones);  x1 = int((zx + 1) * W / n_zones)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            pts = cv2.goodFeaturesToTrack(
                patch, maxCorners=pts_per_zone,
                qualityLevel=0.01, minDistance=5, blockSize=7,
            )
            if pts is not None:
                pts[:, 0, 0] += x0
                pts[:, 0, 1] += y0
                all_pts.append(pts)
                all_ids.extend([zi] * len(pts))
    if all_pts:
        return np.vstack(all_pts).astype(np.float32), np.array(all_ids, dtype=int)
    return np.zeros((0, 1, 2), dtype=np.float32), np.array([], dtype=int)


_LK = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def _gaussian_smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """Convolve a 1-D array with a Gaussian kernel of the given width.

    Uses edge-padding so the start/end of the video aren't dragged toward
    zero (mode='same' zero-padding would create artificial jitter at clip
    boundaries).
    """
    if window < 3 or len(signal) < 3:
        return signal.astype(float).copy()
    window = min(window, len(signal))
    sigma  = window / 4.0
    half   = window // 2
    x      = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(signal.astype(float), half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _track_pair(
    prev_norm: np.ndarray,
    curr_norm: np.ndarray,
    n_zones: int,
    pts_per_zone: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Zone-detect + LK-track one frame pair.
    Returns (prev_pts (N,2), curr_pts (N,2), zone_ids (N,)) or None."""
    pts, zone_ids = _detect_zone_points(prev_norm, n_zones, pts_per_zone)
    if len(pts) < 8:
        return None
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_norm, curr_norm, pts, None, **_LK
    )
    mask = status.ravel() == 1
    if mask.sum() < 8:
        return None
    return pts[mask].reshape(-1, 2), curr_pts[mask].reshape(-1, 2), zone_ids[mask]


def _fit_global_motion(
    prev_pts: np.ndarray, curr_pts: np.ndarray
) -> Tuple[float, float, Optional[np.ndarray], np.ndarray]:
    """
    Estimate per-frame camera velocity and the global motion model.
    Returns:
      (vx, vy)     — median point displacement.  Deliberately NOT taken from
                     the RANSAC transform: under parallax (e.g. orbiting a
                     subject) RANSAC randomly locks onto a different motion
                     population each frame, making its velocity jump even when
                     the true motion is smooth.  The median over all points is
                     deterministic and varies smoothly for smooth motion.
      M            — 2×3 RANSAC similarity transform for residual analysis
                     (or None if the fit failed)
      inlier_mask  — boolean (N,) RANSAC inlier mask (all-True on fallback)
    """
    disp = curr_pts - prev_pts
    vx   = float(np.median(disp[:, 0]))
    vy   = float(np.median(disp[:, 1]))
    M, inliers = cv2.estimateAffinePartial2D(
        prev_pts.reshape(-1, 1, 2).astype(np.float32),
        curr_pts.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99,
    )
    if M is not None and inliers is not None and inliers.sum() >= 8:
        return vx, vy, M, inliers.ravel().astype(bool)
    return vx, vy, None, np.ones(len(prev_pts), dtype=bool)


def _model_residuals(
    prev_pts: np.ndarray, curr_pts: np.ndarray,
    M: Optional[np.ndarray], vx: float, vy: float,
) -> np.ndarray:
    """Per-point distance between observed motion and the global model's
    prediction. With a similarity transform this is near-zero for any rigid
    camera move (pan / rotation / zoom); only parallax, moving objects and
    AI warping artefacts remain."""
    if M is not None:
        pred = cv2.transform(
            prev_pts.reshape(-1, 1, 2).astype(np.float32), M
        ).reshape(-1, 2)
    else:
        pred = prev_pts + np.array([vx, vy])
    return np.linalg.norm(curr_pts - pred, axis=1)


def _draw_motion_vectors(
    frame_bgr: np.ndarray,
    pts_prev: np.ndarray,       # (N, 2)
    pts_curr: np.ndarray,       # (N, 2)
    global_dx: float,
    global_dy: float,
    residual_mags: np.ndarray,  # (N,)
) -> str:
    """
    Draw per-point motion arrows coloured by model residual
    (green = matches camera model, red = outlier / artefact).
    Cyan bold arrow = global camera motion (3× scaled for visibility).
    Returns base64-encoded PNG.
    """
    img   = frame_bgr.copy()
    H, W  = img.shape[:2]
    max_r = max(float(residual_mags.max()), 1e-6)

    for i in range(len(pts_prev)):
        px, py = int(round(pts_prev[i, 0])), int(round(pts_prev[i, 1]))
        qx, qy = int(round(pts_curr[i, 0])), int(round(pts_curr[i, 1]))
        t = float(residual_mags[i]) / max_r            # 0 = green → 1 = red
        color = (30, int(180 * (1 - t)), int(200 * t)) # BGR
        cv2.circle(img, (px, py), 2, color, -1)
        if abs(qx - px) + abs(qy - py) > 0:
            cv2.arrowedLine(img, (px, py), (qx, qy), color, 1, tipLength=0.35)

    # Global motion — bold cyan arrow from frame centre, 3× magnified
    cx, cy = W // 2, H // 2
    ex = cx + int(global_dx * 3)
    ey = cy + int(global_dy * 3)
    cv2.arrowedLine(img, (cx, cy), (ex, ey), (255, 210, 0), 3, tipLength=0.2)
    cv2.putText(img, "Global motion (3x)",
                (cx + 6, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 210, 0), 1)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg              = json.loads(settings) if settings else {}
    num_points       = max(20,    int(cfg.get("num_points",       300)))
    n_zones          = max(1,     int(cfg.get("n_zones",            3)))
    smooth_frames    = max(3,     int(cfg.get("smooth_frames",     15)))
    shake_threshold  = float(np.clip(cfg.get("motion_threshold", 0.60), 0.01, 0.99))
    jitter_px        = max(0.2,   float(cfg.get("jitter_px",       2.0)))
    jitter_weight    = float(np.clip(cfg.get("jitter_weight",     0.80), 0.0, 1.0))
    min_inlier_ratio = float(np.clip(cfg.get("min_inlier_ratio",  0.30), 0.05, 0.95))
    max_height       = int(cfg.get("max_height", 720))

    # ── Load ──────────────────────────────────────────────────────────────────
    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    n = len(frames)
    if n < 3:
        yield {"type": "error", "text": "Video too short — need at least 3 frames."}
        return
    yield {"type": "log", "level": "info", "text": f"{n} frames @ {fps:.1f} fps"}
    await asyncio.sleep(0)

    # ── Optional downscale (also keeps jitter_px calibration consistent) ──────
    if max_height > 0:
        H0, W0 = frames[0].shape[:2]
        if H0 > max_height:
            nw = int(W0 * max_height / H0)
            frames = [cv2.resize(f, (nw, max_height)) for f in frames]
            yield {"type": "log", "level": "info",
                   "text": f"Downscaled to {nw}×{max_height}"}

    gray_f = [frame_to_gray(f) for f in frames]
    norm_f = [_normalize_brightness(g) for g in gray_f]

    pts_per_zone = max(4, num_points // (n_zones * n_zones))
    n_cells      = n_zones * n_zones
    yield {"type": "log", "level": "info",
           "text": (f"Zone grid: {n_zones}×{n_zones}, {pts_per_zone} pts/zone "
                    f"→ ~{pts_per_zone * n_cells} pts/frame")}
    await asyncio.sleep(0)

    # ── Per-frame arrays ──────────────────────────────────────────────────────
    vel       = np.zeros((n, 2))         # camera velocity (frame-centre motion)
    resid_raw = np.zeros(n)              # mean model residual of inliers (px)
    zone_raw  = np.zeros((n, n_cells))   # mean model residual per zone (px)
    tracked   = np.zeros(n, dtype=bool)  # frames with a usable track

    # ── Main tracking loop ────────────────────────────────────────────────────
    for r in range(1, n):
        pair = _track_pair(norm_f[r - 1], norm_f[r], n_zones, pts_per_zone)
        if pair is None:
            vel[r] = vel[r - 1]          # hold velocity through dropouts
            continue
        prev_good, curr_good, good_zones = pair

        vx, vy, M, inlier_mask = _fit_global_motion(prev_good, curr_good)
        vel[r]     = (vx, vy)
        tracked[r] = True

        residuals = _model_residuals(prev_good, curr_good, M, vx, vy)

        # Inlier consensus too low → likely flicker / scene cut; skip residual
        inlier_ratio = inlier_mask.mean()
        if inlier_ratio >= min_inlier_ratio:
            resid_raw[r] = float(residuals[inlier_mask].mean())

        # Per-zone mean residual (all points — outliers ARE the artefacts
        # we want to localise spatially)
        for zi in range(n_cells):
            in_zi = good_zones == zi
            if in_zi.sum() >= 2:
                zone_raw[r, zi] = float(residuals[in_zi].mean())

        if r % 60 == 0:
            yield {"type": "log", "level": "info",
                   "text": f"Processing… {r}/{n - 1} frames"}
            await asyncio.sleep(0)

    yield {"type": "log", "level": "info", "text": "Computing motion scores…"}
    await asyncio.sleep(0)

    # ── Stabilisation decomposition: intentional path vs jitter ──────────────
    # Frame 0 has no measurement (vel[0] = 0); back-fill it so a video that is
    # already moving at frame 1 doesn't get a fake jitter spike at the start
    # when the zero contaminates the smoothed path.
    if n > 1:
        vel[0] = vel[1]

    # Low-pass the velocity over ~half a second: that's the intentional camera
    # move (pan / orbit / zoom). What's left is high-frequency tremor.
    traj_window = max(smooth_frames, int(round(fps / 2)) | 1)
    vx_smooth   = _gaussian_smooth(vel[:, 0], traj_window)
    vy_smooth   = _gaussian_smooth(vel[:, 1], traj_window)
    speed_smooth = np.hypot(vx_smooth, vy_smooth)        # intentional px/frame
    jitter_raw   = np.hypot(vel[:, 0] - vx_smooth,
                            vel[:, 1] - vy_smooth)       # tremor px/frame
    jitter_raw[~tracked] = 0.0

    # ── Absolute calibration (no per-video percentile scaling) ───────────────
    # Deadband: sub-0.3 px deviations are tracking noise, not visible shake.
    # Speed discount: measurement noise grows with camera speed (motion blur,
    # larger search windows), and a small tremor riding on a fast intentional
    # move is perceptually invisible anyway.
    jitter_eff   = np.maximum(jitter_raw - 0.3, 0.0)
    jitter_score = np.clip(
        jitter_eff / (jitter_px * (1.0 + 0.15 * speed_smooth)), 0.0, 1.0
    )

    # Model residual, heavily discounted for parallax: residual error grows
    # with camera speed for any scene with depth, real shake does not.
    resid_score = np.clip(
        resid_raw / (jitter_px * (1.0 + 0.5 * speed_smooth)), 0.0, 1.0
    )

    fused_raw    = jitter_weight * jitter_score + (1.0 - jitter_weight) * resid_score
    fused_smooth = _gaussian_smooth(fused_raw, smooth_frames)

    # Zone averages on the same absolute scale
    zone_avg_norm = np.clip(
        zone_raw.mean(axis=0) / jitter_px, 0.0, 1.0
    ).reshape(n_zones, n_zones)

    # ── Classify shaky periods ────────────────────────────────────────────────
    is_shaky = fused_smooth >= shake_threshold
    intervals: List[Tuple[int, int]] = []
    in_s, s0 = False, 0
    for r in range(n):
        if is_shaky[r] and not in_s:
            in_s, s0 = True, r
        elif not is_shaky[r] and in_s:
            intervals.append((s0, r))
            in_s = False
    if in_s:
        intervals.append((s0, n))

    n_shaky_frames = int(is_shaky.sum())
    worst_idx      = int(np.argmax(fused_smooth))

    time_ax = [r / max(fps, 1) for r in range(n)]

    # ── Chart 1: Shakiness over time ─────────────────────────────────────────
    fig1 = go.Figure()

    for s0i, s1i in intervals:
        t1 = time_ax[s1i - 1] if s1i < n else time_ax[-1]
        label = "EXCESSIVE" if (s1i - s0i) / max(fps, 1) > 0.4 else ""
        fig1.add_vrect(
            x0=time_ax[s0i], x1=t1,
            fillcolor="#E24B4A", opacity=0.12, line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font=dict(size=10, color="#c0392b"),
        )

    # Camera speed for context (own scale, so the viewer can see "fast but
    # smooth" — high teal line with a low blue score line = intentional move)
    speed_rel = (speed_smooth / max(float(speed_smooth.max()), 1e-6)).tolist()
    fig1.add_trace(go.Scatter(
        x=time_ax, y=speed_rel, mode="lines",
        name="Camera speed (relative)",
        line=dict(color="rgba(0,160,120,0.35)", width=1.0), hoverinfo="skip",
    ))
    fig1.add_trace(go.Scatter(
        x=time_ax, y=jitter_score.tolist(), mode="lines", name="Jitter signal",
        line=dict(color="rgba(26,84,196,0.35)", width=1.0), hoverinfo="skip",
    ))
    fig1.add_trace(go.Scatter(
        x=time_ax, y=resid_score.tolist(), mode="lines", name="Model residual",
        line=dict(color="rgba(150,50,220,0.35)", width=1.0), hoverinfo="skip",
    ))
    fig1.add_trace(go.Scatter(
        x=time_ax, y=fused_smooth.tolist(), mode="lines",
        name="Motion score (smoothed)",
        line=dict(color="#1a54c4", width=2.5),
        fill="tozeroy", fillcolor="rgba(26,84,196,0.07)",
        hovertemplate="<b>t = %{x:.2f}s</b><br>Motion score: %{y:.3f}<extra></extra>",
    ))
    fig1.add_hline(
        y=shake_threshold,
        line=dict(color="#E24B4A", dash="dash", width=1.5),
        annotation_text=f"Motion threshold ({shake_threshold:.2f})",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#c0392b"),
    )
    fig1.add_vline(
        x=time_ax[worst_idx],
        line=dict(color="#EF9F27", dash="dot", width=1.5),
        annotation_text="Peak movement",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#b07000"),
    )
    fig1.update_layout(
        title=dict(text="Camera Motion Over Time",
                   font=dict(size=15, color="#1a1917"),
                   x=0, xanchor="left", pad=dict(b=10)),
        xaxis=dict(title=dict(text="Time (seconds)", standoff=12),
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title=dict(text="Motion score  (0 = stable → 1 = excessive)", standoff=14),
                   range=[0, 1.12], showgrid=True, gridcolor="#ebebeb"),
        height=350,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=86, r=60, t=76, b=66),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=11)),
        hovermode="x unified",
    )
    yield {
        "type": "plotly", "data": fig1.to_json(),
        "caption": (
            "Per-frame camera motion score (bold blue). The score measures "
            "high-frequency tremor in the camera path — smooth pans, orbits and "
            "zooms score near zero even when fast. Compare with the faint teal "
            "camera-speed line: high speed + low score = intentional movement. "
            "Red shading = frames above the threshold (excessive movement)."
        ),
    }
    await asyncio.sleep(0)

    # ── Chart 2: Zone heatmap ─────────────────────────────────────────────────
    # Flip rows so zone (row 0, col 0) appears at the top-left
    z_display  = zone_avg_norm[::-1].tolist()
    text_cells = [[f"{zone_avg_norm[n_zones - 1 - zy, zx]:.2f}"
                   for zx in range(n_zones)]
                  for zy in range(n_zones)]

    fig2 = go.Figure(go.Heatmap(
        z=z_display,
        colorscale=[[0.0, "#4CAF50"], [0.5, "#EF9F27"], [1.0, "#E24B4A"]],
        zmin=0.0, zmax=1.0,
        text=text_cells, texttemplate="%{text}",
        textfont=dict(size=15, color="white"),
        showscale=True,
        colorbar=dict(
            title=dict(text="Avg residual", side="right", font=dict(size=12)),
            thickness=14,
        ),
    ))
    fig2.update_layout(
        title=dict(text="Where Does Motion Break the Camera Model?",
                   font=dict(size=15, color="#1a1917"),
                   x=0, xanchor="left", pad=dict(b=10)),
        xaxis=dict(
            tickmode="array", tickvals=list(range(n_zones)),
            ticktext=[f"Col {i + 1}" for i in range(n_zones)],
            showgrid=False, zeroline=False,
            title=dict(text="Horizontal zone", standoff=10),
        ),
        yaxis=dict(
            tickmode="array", tickvals=list(range(n_zones)),
            ticktext=[f"Row {n_zones - i}" for i in range(n_zones)],
            showgrid=False, zeroline=False,
            title=dict(text="Vertical zone", standoff=12),
        ),
        height=340,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=86, r=95, t=76, b=72),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    yield {
        "type": "plotly", "data": fig2.to_json(),
        "caption": (
            f"Average motion-model residual per zone ({n_zones}×{n_zones} grid, "
            "absolute pixel scale — all-green means the whole frame moves like "
            "one rigid camera). Localised red zones = motion that no camera "
            "move explains: moving subjects, parallax, or AI warping artefacts."
        ),
    }
    await asyncio.sleep(0)

    # ── Chart 3: Worst-frame motion-vector image ──────────────────────────────
    if worst_idx > 0:
        pair_wf = _track_pair(norm_f[worst_idx - 1], norm_f[worst_idx],
                              n_zones, pts_per_zone)
        if pair_wf is not None:
            prev_wf, curr_wf, _ = pair_wf
            vx_wf, vy_wf, M_wf, _ = _fit_global_motion(prev_wf, curr_wf)
            r_wf = _model_residuals(prev_wf, curr_wf, M_wf, vx_wf, vy_wf)
            img_b64 = _draw_motion_vectors(
                frames[worst_idx], prev_wf, curr_wf, vx_wf, vy_wf, r_wf
            )
            if img_b64:
                t_worst = worst_idx / max(fps, 1)
                yield {
                    "type": "image", "data": img_b64, "mime": "image/png",
                    "caption": (
                        f"Worst frame at t = {t_worst:.2f}s "
                        f"(frame {worst_idx}, score = {fused_smooth[worst_idx]:.3f}). "
                        "Arrows = per-point optical flow: "
                        "green = fits the global camera model, "
                        "red = outlier (subject motion / parallax / artefact). "
                        "Cyan bold arrow = global camera motion (3× scaled)."
                    ),
                }
                await asyncio.sleep(0)

    # ── Structured signals for Stage 2 Event Localizer ────────────────────────
    # One signal per excessive-movement interval, placed at its peak frame.
    signals = []
    for s0i, s1i in intervals:
        seg = fused_smooth[s0i:s1i]
        if len(seg) == 0:
            continue
        peak_fr = s0i + int(np.argmax(seg))
        signals.append({"frame": int(peak_fr), "signal_type": "camera_motion",
                        "score": round(float(seg.max()) / max(shake_threshold, 1e-6), 3)})
    # severity computed below (needs peak/shaky stats); emit signal after it.

    # ── Metrics ───────────────────────────────────────────────────────────────
    mean_score  = float(fused_smooth[1:].mean())
    peak_score  = float(fused_smooth.max())
    shaky_pct   = n_shaky_frames / max(n, 1)
    n_segs      = len(intervals)
    mean_jitter = float(jitter_raw[tracked].mean()) if tracked.any() else 0.0
    mean_speed  = float(speed_smooth[1:].mean())

    yield {"type": "metric", "label": "Mean motion score",
           "value": f"{mean_score:.3f}",
           "sub": "0 = perfectly stable · 1 = excessive movement"}
    yield {"type": "metric", "label": "Peak motion score",
           "value": f"{peak_score:.3f}",
           "sub": f"at t = {worst_idx / max(fps, 1):.2f}s (frame {worst_idx})"}
    yield {"type": "metric", "label": "Excessive motion duration",
           "value": f"{n_shaky_frames / max(fps, 1):.1f}s",
           "sub": f"{shaky_pct:.0%} of video · {n_segs} segment(s)"}
    yield {"type": "metric", "label": "Camera tremor",
           "value": f"{mean_jitter:.2f} px/frame",
           "sub": f"high-frequency jitter (intentional speed: {mean_speed:.1f} px/frame)"}

    if intervals:
        segs = [f"{s / max(fps, 1):.1f}–{e / max(fps, 1):.1f}s"
                for s, e in intervals[:6]]
        suffix = " …" if n_segs > 6 else ""
        yield {"type": "log", "level": "warn",
               "text": f"{n_segs} excessive movement segment(s): {', '.join(segs)}{suffix}"}
    else:
        yield {"type": "log", "level": "success",
               "text": "No excessive camera movement detected."}

    # Severity from sustained shake, not a single peak frame
    severity = min(100, int((0.4 * peak_score + 0.6 * shaky_pct * 2.0) * 100))
    color    = ("#E24B4A" if severity > 55
                else "#EF9F27" if severity > 25
                else "#4CAF50")

    yield {"type": "signal", "source": "s1_camera_motion", "source_name": "Camera Motion",
           "fps": float(fps), "n_frames": int(n), "severity": severity,
           "type_severities": {"camera_motion": severity}, "signals": signals}
    yield {"type": "severity", "label": "Camera motion severity",
           "value": severity, "color": color}

    yield {"type": "done"}
