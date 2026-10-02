"""
Frame sets for the 3-run std: run 0 is data/consol (8 frames evenly spaced from
first to last frame). Runs 1 and 2 take 8 evenly spaced frames shifted by 1/3
and 2/3 of the frame spacing, same resolution and JPEG quality. The manifest is
shared, so every run scores the same clips with the same questions.

python backend/scripts/make_frame_seeds.py
"""
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/consol"
VID = ROOT / "data/consolidated_videos"
man = json.loads((SRC / "manifest.json").read_text())
MAXS = man.get("max_side", 512)


def extract(cid, out_dir, run, k=8):
    cap = cv2.VideoCapture(str(VID / f"{cid}.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = (total - 1) / (k - 1)
    lo, hi = run * step / 3, (total - 1) - (2 - run) * step / 3
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for j, i in enumerate(np.linspace(lo, hi, k).round().astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(i, total - 1)))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = MAXS / max(h, w)
        if s < 1:
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
        cv2.imwrite(str(out_dir / f"{j:03d}.jpg"), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        n += 1
    cap.release()
    return n


for run in (1, 2):
    dst = ROOT / f"data/consol_fs{run}"
    dst.mkdir(exist_ok=True)
    shutil.copy(SRC / "manifest.json", dst / "manifest.json")
    ids = [c["clip_id"] for c in man["clips"]]
    with ThreadPoolExecutor(16) as ex:
        got = list(ex.map(lambda c: extract(c, dst / "frames" / c, run), ids))
    print(f"run {run}: {len(ids)} clips, {sum(g == 8 for g in got)} with 8 frames, "
          f"{sum(g == 0 for g in got)} empty")
