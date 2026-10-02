"""Frame-level visual embeddings: DINOv2, CLIP, SigLIP.

All loaders return a cached (model, processor) bundle — repeated calls within
one server process reuse the same instance. All embedders accept a list of
BGR uint8 frames, run batched on CUDA when available, and return
L2-normalised float32 arrays of shape (T, D).

DINOv2 is facebook/dinov2-base via HuggingFace: the same checkpoint and
extraction path (pooler output == CLS token, L2-normalised) used by the
world_models_evaluation notebooks, the Physics-IQ baseline statistics,
scripts/check_models.py, and the archived SAM3 pipeline. Keep them aligned —
latent-kinematics thresholds and baselines are only meaningful in this space.
"""
import threading
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np

BATCH_SIZE = 16


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_pil_rgb(frames: List[np.ndarray]) -> list:
    from PIL import Image
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def _l2_normalize(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms < 1e-8, 1.0, norms)


def _embed_batched(frames: List[np.ndarray], encode) -> np.ndarray:
    """Run `encode(pil_batch) -> tensor` over BGR frames in batches."""
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(frames), BATCH_SIZE):
            batch = _to_pil_rgb(frames[i : i + BATCH_SIZE])
            out.append(encode(batch).float().cpu().numpy())
    return _l2_normalize(np.concatenate(out, axis=0).astype(np.float32))


# ── DINOv2 ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def load_dinov2(variant: str = "facebook/dinov2-base") -> Tuple:
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(variant)
    model = AutoModel.from_pretrained(variant).to(device()).eval()
    return model, processor


# ── DINOv2 via transformers (GPU singleton, 768-d CLS) ────────────────────────
# Used by the SAM3 object tracker: fast batched crop embedding on-device.
DINOV2_HF_ID = "facebook/dinov2-base"
_DINO_HF: dict = {}
_DINO_LOCK = threading.Lock()
_DINO_GPU_LOCK = threading.Lock()


def load_dinov2_hf(device: str = "cuda:0"):
    if _DINO_HF:
        return _DINO_HF
    with _DINO_LOCK:
        if _DINO_HF:
            return _DINO_HF
        import torch
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(DINOV2_HF_ID)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModel.from_pretrained(
            DINOV2_HF_ID, dtype=dtype, attn_implementation="sdpa"
        ).to(device).eval()
        _DINO_HF.update(torch=torch, model=model, proc=proc, device=device, dtype=dtype)
        return _DINO_HF


def embed_crops_dinov2_hf(crops_rgb: List[np.ndarray], batch: int = 64) -> np.ndarray:
    """L2-normalised (K, 768) DINOv2 CLS descriptors for a list of RGB uint8 crops."""
    from PIL import Image
    m = load_dinov2_hf()
    torch, model, proc, device, dtype = (
        m["torch"], m["model"], m["proc"], m["device"], m["dtype"]
    )
    pil = [Image.fromarray(c) for c in crops_rgb]
    feats = []
    with _DINO_GPU_LOCK, torch.inference_mode():
        for i in range(0, len(pil), batch):
            inputs = proc(images=pil[i:i + batch], return_tensors="pt").to(device, dtype)
            cls = model(**inputs).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
            feats.append(cls.float().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.empty((0, 768), np.float32)


def embed_frames_dinov2(frames: List[np.ndarray], bundle) -> np.ndarray:
    """(T, 768) L2-normalised DINOv2 descriptors (pooler output == CLS token)."""
    model, processor = bundle
    dev = next(model.parameters()).device

    def encode(batch):
        inputs = processor(images=batch, return_tensors="pt").to(dev)
        out = model(**inputs)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    return _embed_batched(frames, encode)


# ── CLIP ──────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def load_clip(variant: str = "openai/clip-vit-base-patch32") -> Tuple:
    from transformers import AutoImageProcessor, CLIPModel
    processor = AutoImageProcessor.from_pretrained(variant)
    model = CLIPModel.from_pretrained(variant).to(device()).eval()
    return model, processor


def embed_frames_clip(frames: List[np.ndarray], bundle) -> np.ndarray:
    """(T, 512) L2-normalised CLIP image features."""
    model, processor = bundle
    dev = next(model.parameters()).device

    def encode(batch):
        inputs = processor(images=batch, return_tensors="pt").to(dev)
        return model.get_image_features(**inputs)

    return _embed_batched(frames, encode)


# ── SigLIP ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def load_siglip(variant: str = "google/siglip-base-patch16-224") -> Tuple:
    from transformers import AutoImageProcessor, SiglipModel
    processor = AutoImageProcessor.from_pretrained(variant)
    model = SiglipModel.from_pretrained(variant).to(device()).eval()
    return model, processor


def embed_frames_siglip(frames: List[np.ndarray], bundle) -> np.ndarray:
    """(T, 768) L2-normalised SigLIP image features."""
    model, processor = bundle
    dev = next(model.parameters()).device

    def encode(batch):
        inputs = processor(images=batch, return_tensors="pt").to(dev)
        return model.get_image_features(**inputs)

    return _embed_batched(frames, encode)
