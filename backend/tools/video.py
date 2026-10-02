"""Shared video I/O and frame utilities."""
import io, base64
from typing import List, Optional, Tuple

import cv2
import numpy as np


def load_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    step: int = 1,
) -> Tuple[List[np.ndarray], float]:
    """Return (frames, fps). `step` keeps every Nth frame.

    Animated GIFs are decoded via Pillow — OpenCV's VideoCapture is unreliable
    on GIFs across platforms (often only the first frame, or none, on Windows).
    The returned fps is always the *source* fps (callers divide by `step`).
    """
    if str(video_path).lower().endswith(".gif"):
        return _load_gif_frames(video_path, max_frames, step)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames, fps


def _load_gif_frames(
    gif_path: str,
    max_frames: Optional[int],
    step: int,
) -> Tuple[List[np.ndarray], float]:
    """Decode an animated GIF to a list of BGR uint8 frames + its source fps."""
    from PIL import Image, ImageSequence

    im = Image.open(gif_path)
    frames: List[np.ndarray] = []
    durations: List[float] = []        # ms per ORIGINAL frame (for fps)
    for i, fr in enumerate(ImageSequence.Iterator(im)):
        durations.append(float(fr.info.get("duration", 0) or 0))
        if i % step == 0:
            rgb = np.asarray(fr.convert("RGB"))         # (H, W, 3) RGB
            frames.append(np.ascontiguousarray(rgb[:, :, ::-1]))  # → BGR
            if max_frames and len(frames) >= max_frames:
                break

    valid = [d for d in durations if d > 0]
    fps = (1000.0 / (sum(valid) / len(valid))) if valid else 10.0
    fps = max(1.0, min(60.0, fps))
    return frames, fps


def probe_video(video_path: str) -> Tuple[float, int, Tuple[int, int]]:
    """(fps, n_frames, (H, W)) read from container metadata — no full decode.

    `n_frames` is the container's own count, which some encoders get wrong (or
    report as 0); treat it as an estimate and tolerate a short/long stream.
    """
    if str(video_path).lower().endswith(".gif"):
        from PIL import Image
        im = Image.open(video_path)
        n = getattr(im, "n_frames", 1)
        dur = float(im.info.get("duration", 0) or 0)
        fps = max(1.0, min(60.0, 1000.0 / dur)) if dur > 0 else 10.0
        return fps, int(n), (im.size[1], im.size[0])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cap.release()
    return float(fps), max(0, n), (h, w)


def iter_frames(video_path: str, max_dim: Optional[int] = None):
    """Yield BGR frames one at a time — O(1) memory instead of O(n_frames).

    `load_frames` materialises the whole clip (a 450-frame 4K video is ~11 GB);
    use this when a pipeline only needs a sliding window. `max_dim` downscales
    each frame as it is decoded, so full-resolution frames never accumulate.
    Same container handling as `load_frames`: Pillow for GIFs (OpenCV's
    VideoCapture is unreliable on them), OpenCV/ffmpeg for everything else.
    """
    def _shrink(bgr):
        if not max_dim:
            return bgr
        h, w = bgr.shape[:2]
        s = min(1.0, max_dim / max(h, w))          # only ever downscale
        if s >= 1.0:
            return bgr
        return cv2.resize(bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                          interpolation=cv2.INTER_AREA)

    if str(video_path).lower().endswith(".gif"):
        from PIL import Image, ImageSequence
        im = Image.open(video_path)
        for fr in ImageSequence.Iterator(im):      # already lazy
            rgb = np.asarray(fr.convert("RGB"))
            yield _shrink(np.ascontiguousarray(rgb[:, :, ::-1]))
        return

    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            yield _shrink(bgr)
    finally:
        cap.release()


def load_frames_rgb(
    video_path: str,
    max_frames: int = 48,
    target_h: int = 480,
) -> Tuple[List[np.ndarray], float]:
    """
    Read a video into RGB uint8 frames, downscaled to `target_h` and uniformly
    subsampled to at most `max_frames`. Returns (frames, effective_fps) where
    effective_fps accounts for subsampling so playback speed is preserved.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    raw: List[np.ndarray] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        h, w = bgr.shape[:2]
        if h > target_h:
            bgr = cv2.resize(bgr, (int(target_h * w / h), target_h))
        raw.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not raw:
        return [], fps
    if len(raw) > max_frames:
        idx = np.linspace(0, len(raw) - 1, max_frames).astype(int)
        eff_fps = fps * max_frames / len(raw)
        return [raw[i] for i in idx], eff_fps
    return raw, fps


def sample_frames(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    """Uniformly sample exactly n frames (or all if len ≤ n)."""
    if len(frames) <= n:
        return frames
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in idxs]


def frame_to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def encode_video_browser(
    frames: List[np.ndarray],
    fps: float,
    max_width: int = 640,
) -> Tuple[bytes, str]:
    """
    Encode BGR frames to a browser-playable clip. Tries H.264 MP4 via the
    system ffmpeg first, falls back to animated WebP via Pillow.
    Returns (bytes, mime_type).
    """
    import os, shutil, subprocess, tempfile

    if not frames:
        raise RuntimeError("No frames to encode.")

    h, w = frames[0].shape[:2]
    scale = min(1.0, max_width / w)
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
        frames = [cv2.resize(f, (w, h)) for f in frames]
    # libx264 yuv420p requires even dimensions
    w -= w % 2
    h -= h % 2
    frames = [np.ascontiguousarray(f[:h, :w]) for f in frames]
    fps = float(max(1.0, min(60.0, fps)))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
                input=b"".join(f.tobytes() for f in frames),
                capture_output=True, check=True, timeout=120,
            )
            with open(out_path, "rb") as fh:
                data = fh.read()
            if len(data) > 100:
                return data, "video/mp4"
        except Exception:
            pass
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    # Fallback: animated WebP (no codec needed; frontend renders as <img>)
    from PIL import Image
    if len(frames) > 120:
        idx = np.linspace(0, len(frames) - 1, 120, dtype=int)
        fps = fps * 120 / len(frames)
        frames = [frames[i] for i in idx]
    pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    buf = io.BytesIO()
    pil[0].save(
        buf, format="WEBP", save_all=True, append_images=pil[1:],
        loop=0, duration=max(33, int(1000 / fps)), method=3,
    )
    return buf.getvalue(), "image/webp"


def fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
