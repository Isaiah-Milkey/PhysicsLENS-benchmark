#!/usr/bin/env python
"""
Model sanity-check for the SAM3 + DINOv2 (`sam_track`) pipeline.

Verifies, in order:
  1. torch + CUDA are available
  2. the SAM3 / DINOv2 classes import (transformers is new enough)
  3. DINOv2 (non-gated) loads and embeds a dummy crop  -> proves the embed path
  4. SAM3 access (gated) — loads only the small processor/config so it 401s fast
     if you have not been granted access / logged in (no 3.4 GB download).

Run inside the GPU env:
    python backend/scripts/check_models.py
"""
import sys
import traceback

OK, BAD, WARN = "\033[92m✓\033[0m", "\033[91m✗\033[0m", "\033[93m!\033[0m"


def main() -> int:
    rc = 0

    # 1. torch + CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "—"
        print(f"{OK if cuda else BAD} torch {torch.__version__}  CUDA available={cuda}  device={dev}")
        rc |= 0 if cuda else 1
    except Exception:
        print(f"{BAD} torch not importable")
        traceback.print_exc()
        return 1

    # 2. transformers + SAM3/DINOv2 classes
    try:
        import transformers
        from transformers import (  # noqa: F401
            AutoImageProcessor, AutoModel, Sam3VideoModel, Sam3VideoProcessor,
        )
        print(f"{OK} transformers {transformers.__version__}  (Sam3VideoModel + Dinov2 import OK)")
    except Exception:
        print(f"{BAD} transformers missing SAM3 classes — need transformers>=5.5")
        traceback.print_exc()
        return 1

    # 3. DINOv2 — non-gated, prove the embedding path end to end
    try:
        import numpy as np
        from PIL import Image
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        dummy = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype("uint8"))
        inputs = proc(images=dummy, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        with torch.inference_mode():
            cls = model(**inputs).last_hidden_state[:, 0]
        print(f"{OK} DINOv2 embed OK — descriptor dim={cls.shape[-1]}")
    except Exception:
        print(f"{BAD} DINOv2 load/embed failed")
        traceback.print_exc()
        rc |= 1

    # 4. SAM3 — gated. Load only the processor (small) to test access cheaply.
    try:
        Sam3VideoProcessor.from_pretrained("facebook/sam3")
        print(f"{OK} SAM3 access OK — facebook/sam3 is reachable (weights download on first pipeline run)")
    except Exception as exc:
        msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"{WARN} SAM3 not accessible yet: {msg}")
        print("    → Request access at https://huggingface.co/facebook/sam3 then run "
              "`hf auth login` (or set HF_TOKEN) with an approved account.")
        # not fatal — this is the expected state before you authenticate

    print("\nDone." if rc == 0 else "\nDone (with errors above).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
