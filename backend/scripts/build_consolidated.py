"""
Ingest the PhysicsLENS dataset (4 generators x observable/unobservable) as
released on Hugging Face (swiftrando/PhysicsLENS). Download it so that
`physicslens_robot_data/` sits under the gitignored data/ directory:

  data/physicslens_robot_data/
    consolidated_annotations.csv     human annotations, one row per video
    consolidated_prompts_80.csv      prompt table, one row per scenario
                                     (observable fields + unobs_* fields)
    <generator>/  <generator>_unobs/ generated videos per model and condition
      generator in: Wan2.2_TI2V-5B, cosmos3-nano, hunyuan15, magi

Pass --root to point at a different location of physicslens_robot_data/.

Resolves each annotation row to its video by (model, observability) -> directory,
not by filename alone: Wan and Cosmos reuse the source demo's filename, and the
updated drop suffixes Cosmos files with `_cosmos3-nano` while the CSV does not.

Joins the prompt tables from the prompt_generation branch so every clip carries
what it was SUPPOSED to show. For unobservable clips this is essential, not
decoration: the hidden property (e.g. "hydrophobic layer, repels liquid") is by
design not visible in the frames, so no judge can grade
`hidden_property_followed_1_4` without being told what the property is.

`pair_id` = "<testset_id>__<generator>" links an observable clip to its
unobservable twin — same source frame, same generator, only the hidden property
differs — which is what makes the paired obs/unobs analysis possible.

Output: data/consolidated_labels.csv + data/consolidated_videos/<clip_id>.mp4
symlinks, ready for eval_prepare.py.
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "physicslens_robot_data"   # overridable with --root
LINK = ROOT / "data" / "consolidated_videos"
OUT = ROOT / "data" / "consolidated_labels.csv"

GEN = {  # annotation model name -> (directory stem, short id)
    "Wan2.2": ("Wan2.2_TI2V-5B", "wan"),
    "cosmos3 nano": ("cosmos3-nano", "cosmos"),
    "Hunyuan 1.5": ("hunyuan15", "hunyuan"),
    "MAGI 4.5B distill": ("magi", "magi"),
}


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def load_prompts(path: Path):
    """Split the merged prompt table into the observable and unobservable
    tables the rest of this script reads. Both are indexed by scenario id and
    expose: prompt, task, action, scene, and (unobservable only)
    hidden_property, hidden_property_value, expected_outcome,
    failure_signature."""
    m = pd.read_csv(path)
    po = m[["id", "task", "scene", "action", "prompt"]].set_index("id")
    u = m[m["has_unobservable"].map(_truthy)]
    pu = pd.DataFrame({
        "id": u["id"], "task": u["task"],
        "scene": u["unobs_scene"], "action": u["unobs_action"],
        "prompt": u["unobs_prompt"],
        "hidden_property": u["hidden_property"],
        "hidden_property_value": u["hidden_property_value"],
        "expected_outcome": u["expected_outcome"],
        "failure_signature": u["failure_signature"],
    }).set_index("id")
    return po, pu


def resolve(r):
    d, _ = GEN[r.model]
    d = RAW / (d + ("_unobs" if r.observability == "unobservable" else ""))
    f = d / r.video_file
    if f.exists():
        return f
    alt = d / (Path(r.video_file).stem + f"_{GEN[r.model][0]}.mp4")
    return alt if alt.exists() else None


def main():
    global RAW
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="path to physicslens_robot_data/ "
                         "(default: data/physicslens_robot_data)")
    args = ap.parse_args()
    if args.root:
        RAW = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    if not RAW.exists():
        sys.exit(f"ERROR: {RAW} not found — download the dataset from "
                 "https://huggingface.co/datasets/swiftrando/PhysicsLENS into data/")

    a = pd.read_csv(RAW / "consolidated_annotations.csv")
    po, pu = load_prompts(RAW / "consolidated_prompts_80.csv")
    # The benchmark is the scenarios in the prompt table. The annotation file
    # also holds rows for scenarios dropped from the benchmark (scenario 91,
    # 3 rows), which are excluded here: 442 annotation rows -> 439 videos.
    extra = sorted(set(a.testset_id) - set(po.index))
    if extra:
        n_extra = int(a.testset_id.isin(extra).sum())
        print(f"excluding {n_extra} annotation row(s) for scenario(s) not in the "
              f"benchmark prompt table: {extra}")
        a = a[~a.testset_id.isin(extra)]

    LINK.mkdir(parents=True, exist_ok=True)
    for p in LINK.iterdir():
        p.unlink()

    rows, miss = [], []
    for _, r in a.iterrows():
        src = resolve(r)
        if src is None:
            miss.append(f"{r.model}/{r.video_file}")
            continue
        g = GEN[r.model][1]
        unobs = r.observability == "unobservable"
        cid = f"{g}__{'unobs' if unobs else 'obs'}__{int(r.testset_id):03d}"
        (LINK / f"{cid}.mp4").symlink_to(src.resolve())
        P = pu if unobs else po
        pr = P.loc[r.testset_id] if r.testset_id in P.index else None
        get = (lambda k: "" if pr is None or pd.isna(pr.get(k)) else str(pr.get(k)))
        issue = "" if pd.isna(r.description_of_issue) else str(r.description_of_issue).strip()
        cat = "" if pd.isna(r.physics_category) else str(r.physics_category).strip()
        # a row can carry a category with a blank description (3 in the current
        # file); it is still a violation, so fall back to the category text
        # rather than silently dropping it from every per-specialist test
        if not issue and cat:
            issue = f"[category only] {cat}"
        rows.append(dict(
            id=cid, generator=g, observability=r.observability,
            testset_id=int(r.testset_id), pair_id=f"{int(r.testset_id):03d}__{g}",
            rating=r.physical_plausibility_1_4,
            hidden_followed=r.hidden_property_followed_1_4,
            action_completed=str(r.action_completed).strip().lower(),
            has_violation=1 if issue else 0, rules=issue,
            category="" if pd.isna(r.physics_category) else r.physics_category,
            annotator=r.annotator,
            caption=get("prompt"), task=get("task"), action=get("action"),
            hidden_property=get("hidden_property"),
            hidden_value=get("hidden_property_value"),
            expected_outcome=get("expected_outcome"),
            failure_signature=get("failure_signature")))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"{len(df)}/{len(a)} rows staged -> {OUT}  ({len(miss)} missing: {miss})")
    print(pd.crosstab(df.generator, df.observability))
    both = df.groupby("pair_id").observability.nunique()
    print(f"obs/unobs pairs: {(both == 2).sum()}")
    print(f"unobs with hidden_property text: "
          f"{(df[df.observability=='unobservable'].hidden_property != '').sum()}")
    print(f"captions present: {(df.caption != '').sum()}/{len(df)}")


if __name__ == "__main__":
    sys.exit(main())
