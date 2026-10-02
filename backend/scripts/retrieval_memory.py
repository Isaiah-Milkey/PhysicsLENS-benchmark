"""
"Store what's wrong and what's correct" — as a RETRIEVAL DATABASE, not as
prompt exemplars.

The in-prompt version of this idea is already tested and failed: fewshot showed
4 labeled cases (rating + the rule humans said was broken) before the target and
scored -0.014 vs baseline at n=1059. It failed for a mechanical reason — the
VLM's judgement barely moves with prompt content at all (one model under three
very different prompts self-agrees at rho~0.85). Anything routed through that
channel inherits the ceiling.

This is the other version: embed every labeled clip, store it, and predict a new
clip's rating from its NEAREST NEIGHBOURS' labels. The VLM is never asked to
judge, so the rho~0.85 ceiling does not apply. That makes it a genuinely
untested mechanism rather than a restatement of fewshot.

EVALUATION — 5-fold cross-validation over all clips. Each clip is predicted from
a database of the other folds only, so every clip gets an out-of-fold prediction
and the comparison against the VLM covers the identical set. No train/test
asymmetry to argue about.

THE CONTROL THAT MATTERS — generator leakage. The 7 generators have distinctive
visual signatures AND different average quality, so a visual k-NN can score well
by recognising "this looks like VideoCrafter" and recalling that VideoCrafter is
usually bad. That is dataset bookkeeping, not physics. Three controls separate
them:
  gen_mean        predict from generator identity ALONE (out-of-fold). This is
                  the shortcut's own score. If k-NN ~ gen_mean, k-NN learned the
                  shortcut and nothing else.
  knn_within_gen  neighbours restricted to the SAME generator, which removes the
                  shortcut entirely. Whatever survives here is real.
  knn_motion      retrieval in the 7-dim motion-signal space instead of visual
                  space — much less able to encode generator identity.

Usage:
  python backend/scripts/retrieval_memory.py --data data/videophy1200 --device cuda:1
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "backend"))
from vlm_rapidata_eval import spearman                     # noqa: E402
from videophy_eval import auc_extremes                     # noqa: E402
from gepa_optimize import load                             # noqa: E402
from method_report import boot_ci_rho, paired_boot         # noqa: E402


def embed_clips(data: Path, clips, device: str, kind: str = "dinov2"):
    """Mean-pooled frame embedding per clip, from the frozen JPEGs."""
    from tools import embeddings as E
    bundle = (E.load_dinov2() if kind == "dinov2" else E.load_clip())
    fn = E.embed_frames_dinov2 if kind == "dinov2" else E.embed_frames_clip
    out = []
    for i, c in enumerate(clips):
        d = data / "frames" / c["clip_id"]
        frames = []
        for j in c["order_temporal"]:
            p = d / f"{j:03d}.jpg"
            if p.exists():
                frames.append(cv2.imread(str(p)))
        if not frames:
            out.append(None)
            continue
        e = fn(frames, bundle)                    # (T, D), L2-normalised
        v = e.mean(0)
        out.append(v / max(np.linalg.norm(v), 1e-8))
        if (i + 1) % 200 == 0:
            print(f"    embedded {i+1}/{len(clips)}", flush=True)
    return out


def folds(n, k=5, seed=0):
    idx = list(range(n))
    rnd = np.random.RandomState(seed)
    rnd.shuffle(idx)
    return [idx[i::k] for i in range(k)]


def knn_predict(X, y, tr, te, k=25, same=None, exclude=None):
    """Similarity-weighted mean of neighbour labels.

    `same`    restricts the database to entries sharing the query's group
              (the within-generator control).
    `exclude` DROPS database entries sharing the query's key — used to remove
              same-caption neighbours. VideoPhy-2 renders one caption with
              several generators, so without this a neighbour can be literally
              the same prompt rendered by another model, and the k-NN would be
              recalling "this action is hard to generate" rather than judging
              the clip in front of it.
    """
    out = {}
    for i in te:
        pool = tr if same is None else [j for j in tr if same[j] == same[i]]
        if exclude is not None:
            pool = [j for j in pool if exclude[j] != exclude[i]]
        if len(pool) < 3:
            out[i] = None
            continue
        sims = np.array([float(X[i] @ X[j]) for j in pool])
        top = np.argsort(-sims)[:min(k, len(pool))]
        w = np.clip(sims[top], 0, None) ** 8       # sharpen: nearest dominate
        if w.sum() < 1e-8:
            w = np.ones_like(w)
        out[i] = float(np.average([y[pool[t]] for t in top], weights=w))
    return out


def report(name, pred, y, pcs, base=None):
    keep = [i for i in range(len(y)) if pred.get(i) is not None]
    xs = [pred[i] for i in keep]
    ys = [y[i] for i in keep]
    r = spearman(xs, ys)
    lo, hi = boot_ci_rho(xs, ys)
    auc, _, _ = auc_extremes(xs, [pcs[i] for i in keep])
    ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "-"
    extra = ""
    if base is not None:
        kp = [i for i in keep if base[i] is not None]
        d = spearman([pred[i] for i in kp], [y[i] for i in kp]) - \
            spearman([base[i] for i in kp], [y[i] for i in kp])
        blo, bhi = paired_boot([pred[i] for i in kp], [base[i] for i in kp],
                               [y[i] for i in kp])
        sig = "SIG" if (blo is not None and blo > 0) else "ns"
        extra = (f"  |  vs VLM {d:+.3f} "
                 f"[{blo:+.2f}, {bhi:+.2f}] {sig}" if blo is not None else "")
    print(f"{name:32s} n={len(xs):5d} rho={r:+.3f} {ci:>16s} AUC={auc or 0:.3f}{extra}")
    return {"n": len(xs), "rho": round(r, 3), "auc": round(auc, 3) if auc else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--kind", default="dinov2", choices=["dinov2", "clip"])
    a = ap.parse_args()

    data, clips = load(a.data)
    n = len(clips)
    y = [6 - c["pc"] for c in clips]            # implausibility, matches VLM sign
    pcs = [c["pc"] for c in clips]
    gens = [c["generator"] for c in clips]
    print(f"{n} clips | {len(set(gens))} generators | k={a.k}")

    vlmf = data / "scores_gemma4-31b-it_likert_temporal.json"
    vlm_raw = json.loads(vlmf.read_text())["scores"] if vlmf.exists() else {}
    vlm = [vlm_raw.get(c["clip_id"]) for c in clips]

    cache = data / f"emb_{a.kind}.npy"
    if cache.exists():
        X = np.load(cache)
        print(f"\nloaded cached {a.kind} embeddings {X.shape}")
    else:
        print(f"\nembedding with {a.kind} …", flush=True)
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.device.split(":")[-1])
        emb = embed_clips(data, clips, a.device, a.kind)
        ok = [i for i, e in enumerate(emb) if e is not None]
        X = np.zeros((n, len(emb[ok[0]])), dtype=np.float32)
        for i in ok:
            X[i] = emb[i]
        np.save(cache, X)
        print(f"  embedded {len(ok)}/{n} -> cached")

    mo = {}
    fm = data / "method_motion.json"
    if fm.exists():
        ms = json.loads(fm.read_text())["signals"]
        names = sorted(next(iter(ms.values())).keys())
        M = np.array([[float((ms.get(c["clip_id"]) or {}).get(nm, 0.0))
                       for nm in names] for c in clips], dtype=np.float32)
        M = (M - M.mean(0)) / (M.std(0) + 1e-9)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        mo = {"M": M, "names": names}

    caps = [c["caption"] for c in clips]
    F = folds(n, 5)
    knn, knn_wg, knn_mo, genm, knn_nc = {}, {}, {}, {}, {}
    for f in range(5):
        te = F[f]
        tr = [i for g in range(5) if g != f for i in F[g]]
        knn.update(knn_predict(X, y, tr, te, a.k))
        knn_wg.update(knn_predict(X, y, tr, te, a.k, same=gens))
        knn_nc.update(knn_predict(X, y, tr, te, a.k, exclude=caps))
        if mo:
            knn_mo.update(knn_predict(mo["M"], y, tr, te, a.k))
        # generator-mean control, fit out-of-fold
        gm = defaultdict(list)
        for i in tr:
            gm[gens[i]].append(y[i])
        allm = float(np.mean([y[i] for i in tr]))
        for i in te:
            genm[i] = float(np.mean(gm[gens[i]])) if gm.get(gens[i]) else allm

    print(f"\n{'method':32s} {'':7s} {'rho':>6s} {'95% CI':>17s} {'AUC':>10s}")
    print("-" * 96)
    res = {}
    res["vlm"] = report("VLM baseline (likert)",
                        {i: vlm[i] for i in range(n)}, y, pcs)
    res["knn"] = report(f"knn {a.kind} (5-fold CV)", knn, y, pcs, base=vlm)
    if mo:
        res["knn_motion"] = report("knn motion-space", knn_mo, y, pcs, base=vlm)
    print("-" * 96)
    print("CONTROLS — is it just recognising the generator?")
    res["gen_mean"] = report("  generator mean ONLY", genm, y, pcs, base=vlm)
    res["knn_within_gen"] = report("  knn within-generator", knn_wg, y, pcs, base=vlm)
    res["knn_no_same_caption"] = report("  knn EXCLUDING same caption",
                                        knn_nc, y, pcs, base=vlm)

    # How much of the visual k-NN is the generator shortcut?
    kp = [i for i in range(n) if knn.get(i) is not None and genm.get(i) is not None]
    print(f"\n  corr(knn, generator-mean) = "
          f"{spearman([knn[i] for i in kp], [genm[i] for i in kp]):+.3f}"
          "   <- high means the k-NN is mostly recalling generator identity")
    uc = len(set(caps))
    print(f"  {uc} unique captions / {n} clips — {n-uc} clips share a caption, "
          "which is what the exclusion control removes")

    # ── fusion: retrieval + VLM ───────────────────────────────────────────────
    print("\n" + "-" * 96)
    print("FUSION (rank mean, unsupervised — no weights fit, nothing to overfit)")
    from method_report import rank01
    for tag, pred in (("knn", knn), ("knn_no_same_caption", knn_nc)):
        kp = [i for i in range(n) if pred.get(i) is not None and vlm[i] is not None]
        s = set(kp)
        rk = rank01([pred.get(i) if i in s else None for i in range(n)])
        rv = rank01([vlm[i] if i in s else None for i in range(n)])
        fu = {i: (rk[i] + rv[i]) / 2 for i in kp}
        print(f"  corr({tag}, VLM) = "
              f"{spearman([pred[i] for i in kp], [vlm[i] for i in kp]):+.3f}"
              "   <- low means independent channels")
        res[f"fuse_{tag}_vlm"] = report(f"  VLM + {tag}", fu, y, pcs, base=vlm)

    outp = data / "method_retrieval.json"
    outp.write_text(json.dumps({"k": a.k, "kind": a.kind, "results": res,
                                "knn": {clips[i]["clip_id"]: knn.get(i)
                                        for i in range(n)},
                                "knn_no_same_caption": {
                                    clips[i]["clip_id"]: knn_nc.get(i)
                                    for i in range(n)}}, indent=1))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
