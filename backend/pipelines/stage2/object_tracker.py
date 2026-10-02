"""
Stage 2 · Step 1 — Object Tracker
------------------------------------
Segment, name, and track the physically-interacting objects in a video, then
measure each object's appearance stability over time.

Default method ("sam3"):
  1. A local VLM (Qwen2.5-VL) names the distinct objects in the scene
     (human-readable labels, e.g. "ball", "wooden block").
  2. SAM 3 segments + tracks every instance of each named concept across frames.
  3. DINOv2 embeds each tracked object's masked crop → appearance-drift curve
     (flat = stable identity; spikes = morphing / flicker — an AI biomarker).
  4. The GUI gets a *labeled segmented video* (colored masks + names that match
     the plot legend) plus trajectory & drift plots.

Fallback method ("lk"): the original Shi-Tomasi + Lucas-Kanade corner-cluster
tracker (shared, cached implementation in tools.tracking). Used when SAM3/GPU
are unavailable, or when explicitly selected.

Either path publishes its canonical tracks to the evidence bus for downstream
stages.

Metrics: objects found (named), per-object presence, mean/peak appearance drift,
overall tracking-instability score (0–100).
"""
import asyncio, base64, json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video    import load_frames_rgb, encode_video_browser
from tools.tracking import get_tracks, _PALETTE, _hex_to_bgr
from tools.evidence import EVIDENCE, video_id

# Concepts tried when VLM naming is unavailable / returns nothing.
_FALLBACK_CONCEPTS = ["ball", "block", "cup", "bottle", "person"]

CROP_PAD    = 0.10   # fractional padding around bounding box before DINOv2
MIN_CROP_PX = 12     # ignore degenerate boxes smaller than this


def _drift_curve(embeddings: np.ndarray) -> np.ndarray:
    """Cosine-distance drift to the first embedding: drift[t] = 1 − <e_t, e_0>."""
    ref = embeddings[0]
    return 1.0 - embeddings @ ref


def _publish_tracks(video_path: str, objects: list[dict], eff_fps: float, n: int) -> None:
    """Publish canonical tracks to the evidence bus for downstream stages.

    For the SAM3 path (objects carry ``inst["masks"]``), the per-object masks are
    also published PNG-encoded so the Stage 3 Consistency / Collision specialists
    can reuse them instead of re-segmenting. Masks are keyed by this tracker's
    frame index (0..n-1, from ``load_frames_rgb(video_path, n)`` at target_h=480);
    consumers decode identically so indices align — see ``mode`` / ``num_frames``.
    """
    payload = {
        "fps": float(eff_fps), "n_frames": int(n),
        "tracks": [{"id": oi, "label": o["label"], "frames": o["frames"],
                    "boxes": o["boxes"], "cx": o["cx"], "cy": o["cy"],
                    "n_kp": o.get("n_kp", 0)}
                   for oi, o in enumerate(objects)],
    }

    has_masks = any(o.get("inst", {}).get("masks") for o in objects)
    if has_masks:
        from tools.sam3 import encode_mask_png
        masks_png: dict[str, dict[int, bytes]] = {}
        for oi, o in enumerate(objects):
            masks = o.get("inst", {}).get("masks") or {}
            if not masks:
                continue
            key = o["label"] or f"obj {oi}"
            masks_png[key] = {int(fi): encode_mask_png(m) for fi, m in masks.items()}
        payload.update({
            "mode": "sam3",
            "masks_png": masks_png,
            "mask_scale": 1.0,          # masks are at load_frames_rgb (target_h=480) resolution
            "num_frames": int(n),       # consumers: load_frames_rgb(video_path, num_frames)
            "sampled_frames": sorted({fi for m in masks_png.values() for fi in m}),
        })
    else:
        payload["mode"] = "lk"

    EVIDENCE.put(video_id(video_path), "s2_object_tracker", payload)


# ════════════════════════════════════════════════════════════════════════════
# SAM3 path
# ════════════════════════════════════════════════════════════════════════════

def _assign_labels(instances: list[dict]) -> None:
    """Give each instance a human-readable `label`; number repeats within a concept."""
    by_concept: dict[str, list[dict]] = {}
    for inst in instances:
        by_concept.setdefault(inst["concept"], []).append(inst)
    for concept, group in by_concept.items():
        if len(group) == 1:
            group[0]["label"] = concept
        else:
            for i, inst in enumerate(group, 1):
                inst["label"] = f"{concept} #{i}"


def _dedupe_instances(instances: list[dict], iou_thresh: float = 0.6) -> list[dict]:
    """Drop instances that overlap an already-kept one (e.g. two concepts hit the
    same object). Keeps the more persistent / confident of the pair."""
    from tools.sam3 import mask_iou
    instances = sorted(instances, key=lambda d: (-d["n_frames"], -d["mean_score"]))
    kept: list[dict] = []
    for inst in instances:
        if all(mask_iou(inst["masks"], k["masks"]) < iou_thresh for k in kept):
            kept.append(inst)
    return kept


def _object_geometry(inst: dict, n: int) -> dict:
    """Per-frame centroid + bbox for a tracked instance."""
    from tools.sam3 import mask_bbox
    frame_idxs, boxes, cx_list, cy_list = [], [], [], []
    for f in sorted(inst["masks"]):
        mask = inst["masks"][f]
        box = mask_bbox(mask, pad=CROP_PAD, min_px=MIN_CROP_PX)
        if box is None:
            continue
        ys, xs = np.where(mask)
        frame_idxs.append(f)
        boxes.append(box)
        cx_list.append(float(xs.mean()))
        cy_list.append(float(ys.mean()))
    return {"frames": frame_idxs, "boxes": boxes, "cx": cx_list, "cy": cy_list}


def _render_segmented(frames_rgb: list, objects: list[dict], max_frames: int = 240) -> tuple:
    """Overlay each object's mask (tinted fill + contour + readable label) on every
    frame. Returns (annotated BGR frames, subsample step)."""
    n = len(frames_rgb)
    step = max(1, -(-n // max_frames))
    out = []
    for f in range(0, n, step):
        img = cv2.cvtColor(frames_rgb[f], cv2.COLOR_RGB2BGR)
        for oi, obj in enumerate(objects):
            mask = obj["inst"]["masks"].get(f)
            if mask is None or not mask.any():
                continue
            color = _hex_to_bgr(_PALETTE[oi % len(_PALETTE)])
            tint = np.array(color, np.uint8)
            img[mask] = (0.45 * tint + 0.55 * img[mask]).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 2)
            ys, xs = np.where(mask)
            lx, ly = int(xs.min()), int(ys.min())
            label = obj["label"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx = max(0, min(lx, img.shape[1] - tw - 8))   # keep label box on-screen
            ty = ly - 6 if ly - th - 10 >= 0 else min(ly + th + 16, img.shape[0] - 4)
            cv2.rectangle(img, (lx, ty - th - 4), (lx + tw + 6, ty + 4), color, -1)
            cv2.putText(img, label, (lx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
        out.append(img)
    return out, step


async def _run_sam3(video_path: str, cfg: dict) -> AsyncGenerator[dict, None]:
    num_frames   = max(8, int(cfg.get("num_frames", 48)))
    use_dinov2   = str(cfg.get("use_dinov2", "true")).lower() not in ("false", "0", "no")
    render_video = str(cfg.get("render_video", "true")).lower() not in ("false", "0", "no")
    use_naming   = str(cfg.get("use_vlm_naming", "true")).lower() not in ("false", "0", "no")
    naming_model = str(cfg.get("naming_model", "local")).strip() or "local"
    api_key      = str(cfg.get("api_key", "")).strip()
    concepts_in  = str(cfg.get("concepts", "")).strip()
    loop = asyncio.get_event_loop()

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, eff_fps = await loop.run_in_executor(None, load_frames_rgb, video_path, num_frames)
    n = len(frames)
    if n < 3:
        yield {"type": "error", "text": f"Video too short ({n} frames)."}
        return
    H, W = frames[0].shape[:2]
    yield {"type": "log", "level": "info",
           "text": f"{n} frames @ {eff_fps:.1f} fps ({W}×{H}) loaded."}
    await asyncio.sleep(0)

    # ── Step 1: name the objects ──────────────────────────────────────────────
    if concepts_in:
        concepts = [c.strip().lower() for c in concepts_in.split(",") if c.strip()]
        yield {"type": "log", "level": "info", "text": f"Using provided concepts: {', '.join(concepts)}"}
    elif use_naming:
        _nm = "local Qwen2.5-VL" if naming_model == "local" else naming_model
        yield {"type": "log", "level": "info", "text": f"Naming scene objects with VLM ({_nm})…"}
        await asyncio.sleep(0)
        try:
            if naming_model == "local":
                from tools.vlm_local import name_objects
                concepts = await loop.run_in_executor(None, name_objects, frames)
            else:
                # frames are RGB (load_frames_rgb); the router encodes JPEG → BGR.
                from tools.vlm_router import name_subjects
                bgr = [cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR),
                       cv2.cvtColor(frames[n // 2], cv2.COLOR_RGB2BGR)]
                concepts = await name_subjects(bgr, max_subjects=6,
                                               model_key=naming_model, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "warn", "text": f"VLM naming failed ({exc}); using fallback vocabulary."}
            concepts = []
        if concepts:
            yield {"type": "log", "level": "success", "text": f"VLM identified: {', '.join(concepts)}"}
        else:
            concepts = _FALLBACK_CONCEPTS
            yield {"type": "log", "level": "info", "text": f"Falling back to: {', '.join(concepts)}"}
    else:
        concepts = _FALLBACK_CONCEPTS
        yield {"type": "log", "level": "info", "text": f"Trying concept vocabulary: {', '.join(concepts)}"}
    await asyncio.sleep(0)

    # ── Step 2: SAM3 segment + track each concept ─────────────────────────────
    yield {"type": "log", "level": "info", "text": "Loading SAM3 and segmenting…"}
    await asyncio.sleep(0)
    try:
        from tools.sam3 import segment_concept
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "text": f"SAM3 import failed: {exc}"}
        return

    instances: list[dict] = []
    for concept in concepts:
        try:
            found = await loop.run_in_executor(None, segment_concept, frames, concept)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "warn", "text": f'"{concept}" segmentation error: {exc}'}
            continue
        if found:
            yield {"type": "log", "level": "info",
                   "text": f'"{concept}" → {len(found)} instance(s) '
                           f'(best present {found[0]["n_frames"]}/{n} frames).'}
        instances.extend(found)
        await asyncio.sleep(0)

    if not instances:
        yield {"type": "error",
               "text": "SAM3 found no trackable objects for the named concepts. "
                       "Try entering concepts manually in test settings."}
        return

    instances = _dedupe_instances(instances)
    instances = sorted(instances, key=lambda d: (-d["n_frames"], -d["mean_score"]))[:6]
    _assign_labels(instances)

    objects = [{"label": inst["label"], "inst": inst, **_object_geometry(inst, n)}
               for inst in instances]
    objects = [o for o in objects if len(o["frames"]) >= 2]
    n_objects = len(objects)
    yield {"type": "log", "level": "success",
           "text": f"Tracking {n_objects} object(s): {', '.join(o['label'] for o in objects)}"}
    await asyncio.sleep(0)

    _publish_tracks(video_path, objects, eff_fps, n)

    # ── Step 3: labeled segmented video ───────────────────────────────────────
    if render_video and n_objects > 0:
        yield {"type": "log", "level": "info", "text": "Rendering labeled segmented video…"}
        await asyncio.sleep(0)
        try:
            ann, step = _render_segmented(frames, objects)
            data, mime = await loop.run_in_executor(None, encode_video_browser, ann, eff_fps / step)
            yield {"type": "video", "data": base64.b64encode(data).decode(), "mime": mime,
                   "caption": "SAM3 segmentation — mask colors & labels match the plot legend below."}
            yield {"type": "log", "level": "info", "text": f"Segmented video ready ({len(data)/1024:.0f} KB, {mime})."}
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "warn", "text": f"Segmented video failed ({exc}) — continuing."}
        await asyncio.sleep(0)

    # ── Step 4: DINOv2 appearance drift per object ────────────────────────────
    dino_drifts: dict[int, np.ndarray] = {}
    dino_available = False
    if use_dinov2 and n_objects > 0:
        yield {"type": "log", "level": "info", "text": "DINOv2 appearance-drift analysis…"}
        await asyncio.sleep(0)
        try:
            from tools.embeddings import embed_crops_dinov2_hf
            for oi, obj in enumerate(objects):
                crops = []
                for f, box in zip(obj["frames"], obj["boxes"]):
                    x0, y0, x1, y1 = box
                    crop = frames[f][y0:y1, x0:x1]   # RGB
                    if crop.size > 0:
                        crops.append(crop)
                if len(crops) < 2:
                    continue
                embs = await loop.run_in_executor(None, embed_crops_dinov2_hf, crops)
                dino_drifts[oi] = _drift_curve(embs)
            dino_available = True
            yield {"type": "log", "level": "info",
                   "text": f"Appearance drift computed for {len(dino_drifts)} object(s)."}
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "warn", "text": f"DINOv2 unavailable ({exc}) — skipping drift."}
    await asyncio.sleep(0)

    # ── Step 5: plots ─────────────────────────────────────────────────────────
    async for ev in _emit_plots_and_metrics(objects, dino_drifts, dino_available,
                                             eff_fps, H, n, kp_loss_pct=None):
        yield ev


# ════════════════════════════════════════════════════════════════════════════
# Shared plotting + metrics (used by both methods)
# ════════════════════════════════════════════════════════════════════════════

async def _emit_plots_and_metrics(objects, dino_drifts, dino_available,
                                  eff_fps, H, n, kp_loss_pct) -> AsyncGenerator[dict, None]:
    yield {"type": "log", "level": "info", "text": "Building plots…"}
    await asyncio.sleep(0)

    n_rows = 2 if dino_available else 1
    subtitles = ["Object Trajectories (centroid Y, px)"]
    if dino_available:
        subtitles.append("DINOv2 Appearance Drift per Object — cosine dist to frame 0")
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.12, subplot_titles=subtitles)

    for oi, obj in enumerate(objects):
        color = _PALETTE[oi % len(_PALETTE)]
        t_secs = [f / eff_fps for f in obj["frames"]]
        fig.add_trace(go.Scatter(
            x=t_secs, y=[H - cy for cy in obj["cy"]],
            mode="lines+markers", marker=dict(size=4, color=color),
            line=dict(color=color, width=1.6), name=obj["label"],
            hovertemplate=("<b>t = %{x:.2f}s</b><br>y-centroid = %{y:.0f}px<br>"
                           f"{obj['label']}<extra></extra>"),
        ), row=1, col=1)

        if dino_available and oi in dino_drifts:
            drift = dino_drifts[oi]
            t_drift = t_secs[:len(drift)]
            fig.add_trace(go.Scatter(
                x=t_drift, y=drift.tolist(), mode="lines+markers",
                marker=dict(size=4, color=color), line=dict(color=color, width=1.6),
                name=f"{obj['label']} drift", showlegend=False,
                hovertemplate=("<b>t = %{x:.2f}s</b><br>cosine dist = %{y:.3f}<br>"
                               f"{obj['label']}<extra></extra>"),
            ), row=2, col=1)
            for si in np.where(drift > 0.35)[0]:
                if si < len(t_drift):
                    hw = 0.5 / eff_fps
                    fig.add_vrect(x0=t_drift[si] - hw, x1=t_drift[si] + hw,
                                  fillcolor=color, opacity=0.12, line_width=0, row=2, col=1)

    if dino_available:
        fig.add_hline(y=0.35, line=dict(color="#E24B4A", dash="dash", width=1.2),
                      annotation_text="appearance jump threshold (0.35)",
                      annotation_font=dict(color="#E24B4A", size=11),
                      annotation_position="top right", row=2, col=1)

    _grid = dict(showgrid=True, gridcolor="#ebebeb", gridwidth=1)
    fig.update_xaxes(**_grid, title_text="Time (s)", row=n_rows, col=1)
    fig.update_xaxes(**_grid)
    fig.update_yaxes(**_grid, zeroline=False)
    fig.update_yaxes(title_text="Y position (px, bottom=0)", row=1, col=1)
    if dino_available:
        fig.update_yaxes(title_text="Cosine distance to frame 0", range=[0, None], row=2, col=1)
    fig.update_layout(
        title=dict(text="Object Tracker — Segment Trajectories & DINOv2 Appearance Drift",
                   font=dict(size=15)),
        height=380 + (260 if dino_available else 0),
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=65, r=40, t=110, b=55),
        font=dict(family="IBM Plex Sans, sans-serif", size=13), hovermode="x unified",
    )
    yield {"type": "plotly", "data": fig.to_json(),
           "caption": ("Centroid-Y trajectory of each named object (top). "
                       + ("DINOv2 appearance drift — flat = stable, spikes = identity jump (bottom)."
                          if dino_available else ""))}

    # ── Metrics ───────────────────────────────────────────────────────────────
    durations  = [len(o["frames"]) for o in objects]
    mean_dur   = float(np.mean(durations)) if durations else 0.0
    persistent = sum(1 for d in durations if d >= n * 0.5)
    n_objects  = len(objects)

    if dino_drifts:
        all_drift = np.concatenate(list(dino_drifts.values()))
        mean_drift, peak_drift = float(all_drift.mean()), float(all_drift.max())
    else:
        mean_drift = peak_drift = float("nan")

    names = ", ".join(o["label"] for o in objects) or "—"
    yield {"type": "metric", "label": "Objects tracked", "value": str(n_objects), "sub": names}
    yield {"type": "metric", "label": "Persistent objects", "value": str(persistent),
           "sub": f"present ≥50% of video"}
    yield {"type": "metric", "label": "Mean track length", "value": f"{mean_dur:.0f}",
           "sub": f"frames (video = {n})"}
    if kp_loss_pct is not None:
        yield {"type": "metric", "label": "Keypoint loss", "value": f"{kp_loss_pct:.0%}",
               "sub": "LK corners lost"}
    if not np.isnan(mean_drift):
        yield {"type": "metric", "label": "Mean appearance drift", "value": f"{mean_drift:.3f}",
               "sub": "cosine dist to frame 0 (DINOv2)"}
        yield {"type": "metric", "label": "Peak drift", "value": f"{peak_drift:.3f}",
               "sub": "max identity deviation across objects"}

    drift_contrib = (min(mean_drift, 0.6) / 0.6 * 70) if not np.isnan(mean_drift) else 35
    loss_contrib  = (kp_loss_pct * 30) if kp_loss_pct is not None else 0
    instability   = min(int(drift_contrib + loss_contrib), 100)
    sev_color = "#E24B4A" if instability > 50 else "#EF9F27" if instability > 25 else "#4CAF50"
    yield {"type": "severity", "label": "Tracking instability score", "value": instability, "color": sev_color}
    msg = (f"{n_objects} object(s) tracked ({names}); {persistent} persistent"
           + (f"; mean drift {mean_drift:.3f}" if not np.isnan(mean_drift) else "") + ".")
    yield {"type": "log", "level": "success" if instability < 30 else "warn", "text": msg}
    yield {"type": "done"}


# ════════════════════════════════════════════════════════════════════════════
# Lucas-Kanade fallback path (shared, cached tracker in tools.tracking)
# ════════════════════════════════════════════════════════════════════════════

async def _run_lk(video_path: str, cfg: dict) -> AsyncGenerator[dict, None]:
    num_kp       = max(5, int(cfg.get("num_keypoints", 50)))
    sample_every = max(1, int(cfg.get("sample_every", 1)))
    use_dinov2   = str(cfg.get("use_dinov2", "true")).lower() not in ("false", "0", "no")
    render_video = str(cfg.get("render_video", "true")).lower() not in ("false", "0", "no")
    loop = asyncio.get_event_loop()

    yield {"type": "log", "level": "info", "text": "Loading video & tracking keypoints (LK method)…"}
    await asyncio.sleep(0)

    tr = await loop.run_in_executor(
        None, lambda: get_tracks(video_path, num_kp=num_kp,
                                 sample_every=sample_every, max_objects=8))
    frames     = tr["frames"]
    eff_fps    = tr["fps"]
    meta       = tr["meta"]
    obj_tracks = tr["tracks"]
    n  = meta["n_frames"]
    H, W = meta["H"], meta["W"]
    nk = meta["nk"]
    kp_loss_pct = meta["kp_loss_pct"]

    if n < 3:
        yield {"type": "error", "text": f"Video too short ({n} frames)."}
        return
    if not obj_tracks:
        yield {"type": "error", "text": "No keypoints detected in any frame."}
        return
    if meta["start"] > 0:
        yield {"type": "log", "level": "warn",
               "text": f"First {meta['start']} frame(s) featureless — tracking started later."}

    for ct in obj_tracks:
        ct["label"] = f"obj {ct['id']}"
    n_objects = len(obj_tracks)
    yield {"type": "log", "level": "info",
           "text": f"{n} frames @ {eff_fps:.1f} fps — {n_objects} object track(s) "
                   f"from {nk} keypoints."}
    await asyncio.sleep(0)

    _publish_tracks(video_path, obj_tracks, eff_fps, n)

    # Labeled (box) video for parity with the SAM3 path.
    if render_video and obj_tracks:
        yield {"type": "log", "level": "info", "text": "Rendering labeled object video…"}
        await asyncio.sleep(0)
        try:
            by_frame: dict[int, list] = {}
            for oi, ct in enumerate(obj_tracks):
                for f, box in zip(ct["frames"], ct["boxes"]):
                    by_frame.setdefault(f, []).append((oi, box))
            step = max(1, -(-n // 240))
            ann = []
            for f in range(0, n, step):
                img = frames[f].copy()
                for oi, (x0, y0, x1, y1) in by_frame.get(f, []):
                    color = _hex_to_bgr(_PALETTE[oi % len(_PALETTE)])
                    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                    label = obj_tracks[oi]["label"]
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    ty = y0 - 6 if y0 - th - 10 >= 0 else min(y1 + th + 6, img.shape[0] - 4)
                    cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), color, -1)
                    cv2.putText(img, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 2, cv2.LINE_AA)
                ann.append(img)
            data, mime = await loop.run_in_executor(
                None, encode_video_browser, ann, eff_fps / step)
            yield {"type": "video", "data": base64.b64encode(data).decode(), "mime": mime,
                   "caption": "LK corner-cluster tracks (boxes & labels match the plot legend)."}
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "warn", "text": f"LK video failed ({exc})."}
        await asyncio.sleep(0)

    dino_drifts, dino_available = {}, False
    if use_dinov2 and obj_tracks:
        yield {"type": "log", "level": "info", "text": "DINOv2 appearance-drift analysis…"}
        await asyncio.sleep(0)
        try:
            from tools.embeddings import load_dinov2, embed_frames_dinov2
            model = load_dinov2()
            for oi, ct in enumerate(obj_tracks):
                crops = [frames[f][y0:y1, x0:x1]
                         for f, (x0, y0, x1, y1) in zip(ct["frames"], ct["boxes"])
                         if frames[f][y0:y1, x0:x1].size > 0]
                if len(crops) < 2:
                    continue
                embs = await loop.run_in_executor(None, embed_frames_dinov2, crops, model)
                dino_drifts[oi] = _drift_curve(embs)
            dino_available = True
        except Exception as exc:  # noqa: BLE001
            yield {"type": "log", "level": "warn", "text": f"DINOv2 unavailable ({exc})."}
    await asyncio.sleep(0)

    async for ev in _emit_plots_and_metrics(obj_tracks, dino_drifts, dino_available,
                                            eff_fps, H, n, kp_loss_pct=kp_loss_pct):
        yield ev


# ════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ════════════════════════════════════════════════════════════════════════════

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    method = str(cfg.get("method", "sam3")).lower()

    if method == "lk":
        async for ev in _run_lk(video_path, cfg):
            yield ev
        return

    # Default: SAM3. If SAM3/GPU init fails, degrade to LK so the test still runs.
    try:
        sam3_failed = False
        async for ev in _run_sam3(video_path, cfg):
            if ev.get("type") == "error":
                sam3_failed = True
                yield {"type": "log", "level": "warn",
                       "text": f"SAM3 path failed ({ev['text']}); falling back to LK tracker."}
                break
            yield ev
        if not sam3_failed:
            return
    except Exception as exc:  # noqa: BLE001
        yield {"type": "log", "level": "warn",
               "text": f"SAM3 path crashed ({exc}); falling back to LK tracker."}

    async for ev in _run_lk(video_path, cfg):
        yield ev
