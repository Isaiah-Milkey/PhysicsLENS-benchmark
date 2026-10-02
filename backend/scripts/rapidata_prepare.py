"""
Stage the Rapidata Sora physics set so eval_framework.py can run on it.

Downloads the clips named in
huggingface.co/datasets/Rapidata/sora-video-generation-physics-likert-scoring
into a flat directory and writes a labels file mapping each clip to its human
score, so the existing full-framework harness needs no dataset-specific code —
only `--videos-dir` and `--labels`.

The dataset is ~198 Sora clips, each Likert-rated 1 ("makes total sense") to 5
("doesn't make any sense") by a crowd, aggregated into LikertScoreNormalized in
[0, 1] where HIGHER = humans found it MORE physically implausible. PhysicsLENS
severity is also higher = worse, so the two should correlate POSITIVELY.

Reuses load_labels / stratified_sample from vlm_rapidata_eval.py so the clip set
is identical to the VLM-only benchmark already in this repo, which is what makes
"does the full pipeline beat one VLM prompt" a fair question.

Usage (from repo root):
  python backend/scripts/rapidata_prepare.py                 # all 198
  python backend/scripts/rapidata_prepare.py --n 60          # same 60 as the VLM eval
  python backend/scripts/rapidata_prepare.py --n 60 --out data/rapidata60
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from vlm_rapidata_eval import REPO, load_labels, stratified_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="0 = the whole dataset; otherwise a stratified sample "
                         "spanning the full human-score range (use 60 to match "
                         "the existing VLM benchmark exactly)")
    ap.add_argument("--out", default="data/rapidata",
                    help="directory to stage clips into, relative to repo root")
    ap.add_argument("--copy", action="store_true",
                    help="copy clips instead of symlinking into the HF cache")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download

    out_dir = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    vid_dir = out_dir / "ai"          # eval_framework labels by top dir; all AI here
    vid_dir.mkdir(parents=True, exist_ok=True)

    df = load_labels()
    sample = df if a.n <= 0 else stratified_sample(df, a.n)
    print(f"[prepare] dataset has {len(df)} clips; staging {len(sample)}"
          f"{'' if a.n <= 0 else ' (stratified across the human-score range)'}",
          flush=True)

    labels, missing = {}, []
    for i, row in enumerate(sample.itertuples(index=False), 1):
        name = row.FileName
        dest = vid_dir / name
        try:
            if not dest.exists():
                src = Path(hf_hub_download(REPO, f"Videos/{name}", repo_type="dataset"))
                if a.copy:
                    shutil.copy2(src, dest)
                else:
                    dest.symlink_to(src)
            # eval_framework.slug() = path parts under the videos root, joined by "__"
            labels[f"ai__{Path(name).stem}"] = {
                "file": name,
                "human_score": float(row.LikertScoreNormalized),
                "prompt": str(row.Prompt),
            }
        except Exception as exc:                                  # noqa: BLE001
            missing.append(name)
            print(f"[prepare] SKIP {name}: {type(exc).__name__}: {exc}", flush=True)
        if i % 25 == 0:
            print(f"[prepare]   {i}/{len(sample)}", flush=True)

    lab_path = out_dir / "labels.json"
    lab_path.write_text(json.dumps({
        "dataset": REPO,
        "label_field": "LikertScoreNormalized",
        "direction": "higher = humans found it MORE physically implausible; "
                     "PhysicsLENS severity is also higher = worse, so expect a "
                     "POSITIVE correlation",
        "n_total_in_dataset": int(len(df)),
        "n_staged": len(labels),
        "stratified": bool(a.n > 0),
        "clips": labels,
    }, indent=1))

    print(f"[prepare] {len(labels)} clips -> {vid_dir}")
    print(f"[prepare] labels -> {lab_path}")
    if missing:
        print(f"[prepare] {len(missing)} unavailable: {missing[:5]}")
    print(f"\nNext:\n"
          f"  python backend/scripts/eval_framework.py --videos-dir {a.out} "
          f"--out eval_reports/rapidata\n"
          f"  python backend/scripts/eval_framework.py --aggregate "
          f"--labels {a.out}/labels.json --out eval_reports/rapidata")


if __name__ == "__main__":
    main()
