"""
Stage a VideoPhy-2 subset and FREEZE the frame sequences used for scoring.

Why freeze frames instead of decoding on the fly: every model, prompt variant
and frame-order condition must see byte-identical pixels, or a score difference
could be a decode difference. Extracting once also makes re-scoring free — the
expensive part becomes the API calls, not the video I/O.

Output layout:
    <out>/manifest.json                     labels + per-clip frame ordering
    <out>/frames/<clip_id>/000.jpg ... N.jpg   temporal order, one folder per clip

The manifest stores BOTH orderings as index lists into those files, so the
shuffle is fixed at staging time and reproducible across every later run:

    {"clip_id": ..., "pc": 4, "sa": 3, "generator": "wan",
     "order_temporal": [0,1,2,...], "order_shuffled": [5,2,7,...]}

VideoPhy-2 labels (human-annotated):
    pc  physical commonsense, 1-5   <- the target for physics scoring
    sa  semantic adherence,   1-5   (does it match the caption)
    human_violated_rules            natural-language rule(s) broken -> the
                                    per-specialist supervision signal

Usage:
  python backend/scripts/videophy_prepare.py --n 300 --out data/videophy300
  python backend/scripts/videophy_prepare.py --n 60 --out data/videophy60 --frames 8
"""
import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV_REPO = "videophysics/videophy2_test"
CSV_FILE = "videophy2_test.csv"
VID_REPO = "videophysics/videophy2_test_videos"
VID_FILE = "videophy2_test .zip"


def load_table():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    return pd.read_csv(hf_hub_download(CSV_REPO, CSV_FILE, repo_type="dataset"))


def ensure_videos() -> Path:
    """Unzip the video archive once into the HF cache dir beside the zip."""
    from huggingface_hub import hf_hub_download
    z = Path(hf_hub_download(VID_REPO, VID_FILE, repo_type="dataset"))
    dest = z.parent / "extracted"
    if not dest.exists():
        print(f"[prepare] extracting {z.name} …", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
    return dest


def index_videos(root: Path) -> dict:
    """basename (no extension) -> path, for matching CSV rows to files."""
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}:
            idx.setdefault(p.stem, p)
    return idx


def stratified(df, idx, n, seed=0):
    """Balance across the pc scale AND across generators.

    Stratifying on pc alone would let one generator dominate a rating bucket,
    and 'our detector is bad' would be indistinguishable from 'this generator
    fails in a characteristic way'.
    """
    rnd = random.Random(seed)
    df = df[df["pc"].notna()].copy()
    df["_stem"] = df["video_url"].map(lambda u: Path(str(u).split("?")[0]).stem)
    df = df[df["_stem"].isin(idx)]
    if df.empty:
        sys.exit("ERROR: no CSV rows matched extracted video files")

    buckets = defaultdict(list)
    for _, r in df.iterrows():
        buckets[(int(r["pc"]), str(r["model_name"]))].append(r)
    for v in buckets.values():
        rnd.shuffle(v)

    picked, keys = [], sorted(buckets)
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:                      # round-robin keeps strata even
            if buckets[k] and len(picked) < n:
                picked.append(buckets[k].pop())
    return picked


def extract(video: Path, out_dir: Path, n_frames: int, max_side: int) -> int:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    for i in np.linspace(0, total - 1, min(n_frames, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
        cv2.imwrite(str(out_dir / f"{k:03d}.jpg"), fr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        k += 1
    cap.release()
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/videophy300")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    df = load_table()
    print(f"[prepare] VideoPhy-2 test: {len(df)} rows")
    idx = index_videos(ensure_videos())
    print(f"[prepare] {len(idx)} video files on disk")

    rows = stratified(df, idx, a.n, a.seed)
    print(f"[prepare] selected {len(rows)} clips")

    shutil.rmtree(out, ignore_errors=True)
    (out / "frames").mkdir(parents=True)
    rnd = random.Random(a.seed)
    clips, dropped = [], 0
    for r in rows:
        stem = Path(str(r["video_url"]).split("?")[0]).stem
        cid = f"{r['model_name']}__{stem}"[:110]
        k = extract(idx[stem], out / "frames" / cid, a.frames, a.max_side)
        if k < 4:
            dropped += 1
            continue
        order = list(range(k))
        shuf = order[:]
        # Reject the identity permutation — a "shuffled" condition that happens
        # to equal temporal order would silently weaken the ablation.
        while k > 2 and shuf == order:
            rnd.shuffle(shuf)
        clips.append({
            "clip_id": cid, "n_frames": k, "generator": str(r["model_name"]),
            "pc": int(r["pc"]), "sa": int(r["sa"]),
            "joint": int(r["joint"]) if str(r.get("joint")) not in ("nan", "None") else None,
            "is_hard": int(r["is_hard"]) if str(r.get("is_hard")) not in ("nan", "None") else None,
            "caption": str(r["caption"])[:400],
            "violated_rules": str(r.get("human_violated_rules") or ""),
            "order_temporal": order, "order_shuffled": shuf,
        })
        if len(clips) % 50 == 0:
            print(f"[prepare]   {len(clips)}/{len(rows)}", flush=True)

    meta = {"dataset": CSV_REPO, "n_clips": len(clips), "frames_per_clip": a.frames,
            "max_side": a.max_side, "seed": a.seed,
            "label": "pc = physical commonsense 1-5 (higher = MORE physically correct)",
            "clips": clips}
    (out / "manifest.json").write_text(json.dumps(meta, indent=1))

    print(f"\n[prepare] {len(clips)} clips staged -> {out}  ({dropped} dropped)")
    print("  pc distribution :", dict(sorted(Counter(c['pc'] for c in clips).items())))
    print("  generators      :", dict(Counter(c['generator'] for c in clips).most_common()))
    print("  with rule labels:", sum(1 for c in clips if len(c['violated_rules']) > 4))
    print(f"\nNext: python backend/scripts/videophy_eval.py --data {a.out}")


if __name__ == "__main__":
    main()
