"""
Evaluate local VLMs against Rapidata's human-rated Sora physics-plausibility
dataset (huggingface.co/datasets/Rapidata/sora-video-generation-physics-
likert-scoring): ~200 Sora clips, each Likert-scored 1 ("makes total sense") to
5 ("doesn't make any sense") by a crowd of human raters, aggregated into
LikertScoreNormalized in [0, 1] where HIGHER = humans found it MORE physically
implausible.

Unlike scripts/vlm_multimodel_eval.py (5 AI vs 5 real, binary), this dataset is
ALL AI-generated with a CONTINUOUS human implausibility label, so we report two
metrics per model:
  • Spearman correlation  — does the model's suspicion score rank videos the
    way humans did? (the statistically appropriate metric for a continuous
    label; 1.0 = perfect agreement, 0.0 = no relationship)
  • Median-split AUC      — bucket videos into the human-implausible top half
    vs bottom half, then compute the same rank AUC used in the AI-vs-real
    benchmark, so the two evals are comparable at a glance.

Usage:
  python vlm_rapidata_eval.py                                   # default: 60 videos, 3 default models
  python vlm_rapidata_eval.py --n 198                           # full dataset
  python vlm_rapidata_eval.py --logprob Qwen/Qwen2.5-VL-7B-Instruct

Results -> scripts/vlm_rapidata_results.json (merged per model, like vlm_multimodel_eval.py)
"""
import gc, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from tools.video import load_frames, sample_frames

# Re-use the exact prompt + parsing + logprob machinery from the AI-vs-real eval
# so scores are directly comparable across both benchmarks.
sys.path.insert(0, str(Path(__file__).parent))
from vlm_multimodel_eval import (
    PROMPT, to_pil, ask, ask_logprob, parse_json,
)

REPO = "Rapidata/sora-video-generation-physics-likert-scoring"
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "OpenGVLab/InternVL3-8B-hf",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
]


def load_labels():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    p = hf_hub_download(REPO, "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    return df[["FileName", "Prompt", "LikertScoreNormalized"]]


def stratified_sample(df, n):
    """Evenly cover the label range rather than a random/first-N sample, so a
    small n still spans plausible -> implausible."""
    if n >= len(df):
        return df
    ordered = df.sort_values("LikertScoreNormalized").reset_index(drop=True)
    idx = np.linspace(0, len(ordered) - 1, n).astype(int)
    return ordered.iloc[idx].reset_index(drop=True)


def spearman(x, y):
    """Spearman rank correlation with no scipy dependency."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        # average ranks for ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts)); np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def median_split_auc(scores, labels):
    """Bucket by the label's median, then rank-AUC the model scores across
    buckets — same metric as vlm_multimodel_eval.rank_auc, for comparability."""
    from vlm_multimodel_eval import rank_auc
    med = float(np.median(labels))
    hi = [s for s, l in zip(scores, labels) if l > med and s is not None]
    lo = [s for s, l in zip(scores, labels) if l <= med and s is not None]
    if not hi or not lo:
        return float("nan")
    return rank_auc(hi, lo)


def eval_model(model_id, clips, do_logprob=False):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"\n### Loading {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    rows = []
    for name, prompt_text, human_score, imgs in clips:
        r = ask(model, processor, imgs, PROMPT.replace("{n}", str(len(imgs))))
        s = r.get("suspicion_score")
        s = float(s) if isinstance(s, (int, float)) else None
        lp = ask_logprob(model, processor, imgs) if do_logprob else None
        rows.append({"video": name, "human_score": round(float(human_score), 4),
                     "model_score": s,
                     "logprob_score": round(lp, 4) if lp is not None else None})
        lp_txt = f" lp={lp:.3f}" if lp is not None else ""
        print(f"  {name:28s} human={human_score:.3f} model={s}{lp_txt}", flush=True)

    del model, processor
    gc.collect(); torch.cuda.empty_cache()

    human = [r["human_score"] for r in rows]
    model_scores = [r["model_score"] for r in rows]
    paired = [(m, h) for m, h in zip(model_scores, human) if m is not None]
    summary = {
        "n": len(rows), "n_parsed": len(paired),
        "spearman": round(spearman([m for m, _ in paired], [h for _, h in paired]), 3) if paired else None,
        "median_split_auc": round(median_split_auc(model_scores, human), 3),
    }
    if do_logprob:
        lp_scores = [r["logprob_score"] for r in rows]
        paired_lp = [(m, h) for m, h in zip(lp_scores, human) if m is not None]
        summary["logprob_spearman"] = round(spearman([m for m, _ in paired_lp], [h for _, h in paired_lp]), 3) if paired_lp else None
        summary["logprob_median_split_auc"] = round(median_split_auc(lp_scores, human), 3)
    print(f"  => Spearman {summary['spearman']} | median-split AUC {summary['median_split_auc']}"
          + (f" | logprob Spearman {summary.get('logprob_spearman')} AUC {summary.get('logprob_median_split_auc')}" if do_logprob else ""),
          flush=True)
    return {"rows": rows, "summary": summary}


def main():
    from huggingface_hub import hf_hub_download
    args = sys.argv[1:]
    do_logprob = "--logprob" in args
    args = [a for a in args if a != "--logprob"]
    n = 60
    if "--n" in args:
        n = int(args[args.index("--n") + 1])
        args = args[:args.index("--n")] + args[args.index("--n") + 2:]
    models = [a for a in args if not a.startswith("--")] or DEFAULT_MODELS

    df = load_labels()
    sample = stratified_sample(df, n)
    print(f"Dataset has {len(df)} videos; evaluating a stratified sample of {len(sample)} "
          f"spanning the full human-implausibility range (this is NOT the full dataset "
          f"unless --n {len(df)} was passed).", flush=True)

    clips = []
    for _, row in sample.iterrows():
        try:
            local = hf_hub_download(REPO, f"Videos/{row['FileName']}", repo_type="dataset")
            frames, _ = load_frames(local)
            if not frames:
                print(f"SKIP unreadable {row['FileName']}", flush=True); continue
            imgs = [to_pil(f) for f in sample_frames(frames, 8)]
            clips.append((row["FileName"], row["Prompt"], row["LikertScoreNormalized"], imgs))
        except Exception as exc:
            print(f"SKIP {row['FileName']}: {exc}", flush=True)
    print(f"Loaded {len(clips)} clips. Models: {models} logprob={do_logprob}", flush=True)

    out_path = Path(__file__).parent / "vlm_rapidata_results.json"
    out = {"dataset": REPO, "protocol": f"stratified n={len(clips)} of {len(df)}, "
           "label=LikertScoreNormalized (higher=more physically implausible)",
           "models": {}}
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text()); out.setdefault("models", {})
        except Exception:
            pass

    for mid in models:
        try:
            out["models"][mid] = eval_model(mid, clips, do_logprob=do_logprob)
        except Exception as exc:
            print(f"  !! {mid} FAILED: {exc}", flush=True)
            out["models"][mid] = {"error": str(exc)}
        out_path.write_text(json.dumps(out, indent=2))

    print("\n==== HUMAN-AGREEMENT SUMMARY (vs Rapidata Likert physics ratings) ====", flush=True)
    for mid, res in out["models"].items():
        s = res.get("summary")
        if s:
            lp = f" logprobAUC={s.get('logprob_median_split_auc')}" if s.get("logprob_median_split_auc") is not None else ""
            print(f"  {mid:42s} spearman={s['spearman']} median-split-AUC={s['median_split_auc']}{lp} (n={s['n_parsed']}/{s['n']})", flush=True)
        else:
            print(f"  {mid:42s} ERROR: {res.get('error')}", flush=True)


if __name__ == "__main__":
    main()
