"""
Temporal embedding features — motion in DINOv2 space.

WHY. Every representation tried so far is motion-blind in the same way:
  - the VLM reads static appearance (proved by the shuffle ablation: temporal vs
    shuffled frames is null across 3 datasets and 2 model families)
  - the retrieval embedding is DINOv2 MEAN-POOLED over 8 frames, which discards
    order by construction
  - the hand-built kinematics measure real motion but on sparse corner tracks,
    which turned out too noisy for per-category discrimination
So the "give it motion information" idea has never actually been tested in a
LEARNED representation. That is what this does.

Per-frame DINOv2 embeddings, then statistics of how the embedding MOVES:
  d1_*   consecutive-frame embedding distance — appearance change rate
  d2_*   second difference — jerk in feature space; smooth real motion has low
         d2, teleporting/morphing generated motion has high d2
  cos_*  cosine between successive change VECTORS — a scene changing coherently
         keeps direction; incoherent generation flips direction frame to frame
  drift  distance from first to last frame vs summed path length. Near 1 means
         the clip moved somewhere; near 0 means it churned in place, which is
         the signature of an object morphing without going anywhere.

These are cheap (one extra forward pass over frames already on disk) and they
are the natural thing to try before reaching for SAM3.

Usage:
  python backend/scripts/temporal_embed.py --data data/videophy1200 --device cuda:1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "backend"))
from gepa_optimize import load                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    data, clips = load(a.data)

    cache = data / "emb_dinov2_perframe.npy"
    if cache.exists():
        P = np.load(cache)
        print(f"loaded per-frame embeddings {P.shape}")
    else:
        from tools import embeddings as E
        bundle = E.load_dinov2()
        print("embedding per frame …", flush=True)
        t0 = time.time()
        seqs = []
        for i, c in enumerate(clips):
            d = data / "frames" / c["clip_id"]
            fr = [cv2.imread(str(d / f"{j:03d}.jpg")) for j in c["order_temporal"]
                  if (d / f"{j:03d}.jpg").exists()]
            fr = [f for f in fr if f is not None]
            e = E.embed_frames_dinov2(fr, bundle) if fr else None
            seqs.append(e if e is not None and len(e) >= 4 else None)
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(clips)} ({time.time()-t0:.0f}s)", flush=True)
        T = max(len(s) for s in seqs if s is not None)
        D = next(s for s in seqs if s is not None).shape[1]
        P = np.full((len(clips), T, D), np.nan, dtype=np.float32)
        for i, s in enumerate(seqs):
            if s is not None:
                P[i, :len(s)] = s
        np.save(cache, P)
        print(f"  cached {P.shape}")

    feats = {}
    for i, c in enumerate(clips):
        S = P[i]
        S = S[np.isfinite(S[:, 0])]
        if len(S) < 4:
            continue
        d1 = np.linalg.norm(np.diff(S, axis=0), axis=1)
        d2 = np.abs(np.diff(d1))
        V = np.diff(S, axis=0)
        nv = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        U = V / nv
        cos = np.sum(U[:-1] * U[1:], axis=1)
        path = float(d1.sum())
        net = float(np.linalg.norm(S[-1] - S[0]))
        feats[c["clip_id"]] = {
            "d1_mean": float(d1.mean()), "d1_std": float(d1.std()),
            "d1_max": float(d1.max()), "d1_cv": float(d1.std() / (d1.mean() + 1e-8)),
            "d2_mean": float(d2.mean()) if len(d2) else 0.0,
            "d2_max": float(d2.max()) if len(d2) else 0.0,
            "cos_mean": float(cos.mean()) if len(cos) else 0.0,
            "cos_min": float(cos.min()) if len(cos) else 0.0,
            "cos_neg_frac": float(np.mean(cos < 0)) if len(cos) else 0.0,
            "drift_ratio": float(net / (path + 1e-8)),
        }
    outp = data / "temporal_embed.json"
    outp.write_text(json.dumps({"n": len(feats), "features": feats}, indent=1))

    from vlm_rapidata_eval import spearman
    y = [6 - c["pc"] for c in clips if c["clip_id"] in feats]
    names = sorted(next(iter(feats.values())).keys())
    print(f"\n  {len(feats)} clips | univariate rho vs human pc:")
    for nmm in names:
        v = [feats[c["clip_id"]][nmm] for c in clips if c["clip_id"] in feats]
        print(f"    {nmm:14s} {spearman(v, y):+.3f}")
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()
