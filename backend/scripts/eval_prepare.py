"""
Stage a NEW evaluation dataset into the frozen-frame format the whole harness uses.

This is the adapter between whatever format a dataset arrives in and the internal
layout every scorer expects. Point it at a folder of videos plus a labels file and
it produces the same manifest.json + frames/ tree that videophy_prepare.py builds,
so every existing script runs unchanged.

WHAT THE HARNESS NEEDS FROM A LABELS FILE
-----------------------------------------
Required per clip:
  id      something that identifies the clip and maps to a video file
  rating  human physics score, any numeric scale (mapped to 1-5 internally)

Strongly recommended (unlocks per-specialist evaluation — without it you can only
measure whole-clip quality, not whether the gravity specialist finds gravity bugs):
  rules   free text naming the rule(s) broken

Optional but valuable (see docs/eval_dataset_spec.md for why each matters):
  caption       what the video was supposed to show
  category      pre-assigned failure category, skips our keyword taxonomy
  has_violation } clean-vs-broken flag. VideoPhy-2 lacked it, so our negatives are
  is_clean      } "a different violation" and we have never measured clean vs broken
  t_start       } if the annotation localises the failure in time, we can finally
  t_end         } evaluate temporal accuracy instead of only whole-clip detection
  object        which object broke the rule — enables object-conditioned questions
  severity      lets us weight or stratify by how bad the failure is
  rating2       second annotator's rating on a subset -> inter-annotator ceiling

FRAMES ARE FROZEN ON PURPOSE. Extracting once and reusing the JPEGs means a score
difference between two runs is caused by the thing under test, not by a decode
difference. It also makes re-scoring free, which matters because the expensive
part is model calls, not video I/O.

Usage:
  python backend/scripts/eval_prepare.py \
      --videos /path/to/videos --labels /path/to/labels.json \
      --out data/newbench --map id=clip_id,rating=score,rules=violations

  python backend/scripts/eval_prepare.py --videos vids/ --labels labels.csv \
      --out data/newbench --frames 8 --inspect
"""
import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}
# field name -> the names we will accept for it without an explicit --map
ALIASES = {
    "id":       ["id", "clip_id", "video_id", "name", "file", "filename", "video"],
    "rating":   ["rating", "pc", "score", "human_score", "quality", "label"],
    "rules":    ["rules", "violated_rules", "human_violated_rules", "violations",
                 "description", "failure", "annotation"],
    "caption":  ["caption", "prompt", "text", "instruction", "task"],
    "category": ["category", "dimension", "type", "failure_type", "class"],
    "t_start":  ["t_start", "start", "start_time", "begin"],
    "t_end":    ["t_end", "end", "end_time", "finish"],
    "object":   ["object", "entity", "target", "subject"],
    "severity": ["severity", "sev", "impact"],
    "generator": ["generator", "model", "model_name", "source", "system"],
    # clean-vs-broken. Two spellings with OPPOSITE polarity, normalised below into a
    # single has_violation. Without this the only negatives available are clips with a
    # different violation, so "detect any failure" is unmeasurable.
    "has_violation": ["has_violation", "violation", "is_violation", "violated",
                      "broken", "fail", "failed"],
    "is_clean": ["is_clean", "clean", "no_violation", "ok", "valid", "correct"],
    "rating2": ["rating2", "rating_b", "score2", "second_rating", "annotator2"],
}
# our specialists; anything else an annotator writes lands in `other`
CATEGORIES = ["collision", "gravity", "momentum", "friction", "deformation",
              "fluid", "permanence", "causality"]
# Annotators reach for the pipeline's specialist names, or plain English, rather
# than the eval taxonomy's spelling. Fold those in instead of dropping them to
# `other` — a mis-spelled category is a lost clip, and we do not have spares.
SYNONYMS = {
    "contact": "collision",       # contact_specialist was merged into collision
    "impact": "collision",
    "penetration": "collision",
    "rigidity": "deformation",
    "shape": "deformation",
    "morphing": "deformation",
    "object permanence": "permanence",
    "disappearance": "permanence",
    "vanishing": "permanence",
    "cause": "causality",
    "temporal": "causality",
    "liquid": "fluid",
    "inertia": "momentum",
    "velocity": "momentum",
    "support": "gravity",
    "floating": "gravity",
}

TRUE = {"1", "true", "yes", "y", "t"}
FALSE = {"0", "false", "no", "n", "f"}


def as_bool(v):
    """Parse a flag cell. Returns True/False, or None when genuinely absent."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    return None


def split_cats(v):
    """'Collision, Deformation' -> ['collision', 'deformation']. Unknown -> 'other'."""
    if v is None:
        return []
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    out = []
    for p in re.split(r"[,;/|]", s):
        p = p.strip().lower()
        if not p:
            continue
        p = SYNONYMS.get(p, p)
        out.append(p if p in CATEGORIES else "other")
    return out


def auc_ci(n_pos, n_neg, auc=0.70):
    """Hanley-McNeil 95% half-width. Used to tell annotators, before they spend a
    week labelling, whether a category will be able to say anything at all."""
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                  + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))
    return 1.96 * float(se)


def read_labels(path: Path):
    """JSON (list, or dict keyed by id), CSV/TSV, or XLSX. Returns list of dicts.

    XLSX is supported because annotation actually happens in a spreadsheet —
    asking a team to export to CSV first is one more step to get wrong, and the
    export silently drops which sheet the data was on.
    """
    if path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        xl = pd.ExcelFile(path)
        if len(xl.sheet_names) > 1:
            print(f"   note: {len(xl.sheet_names)} sheets {xl.sheet_names}, "
                  f"reading '{xl.sheet_names[0]}'")
        df = xl.parse(xl.sheet_names[0])
        # NaN -> "" so downstream "is this cell empty" checks behave like CSV
        return df.where(df.notna(), "").to_dict("records")
    if path.suffix.lower() in (".csv", ".tsv"):
        import csv
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open() as f:
            return list(csv.DictReader(f, delimiter=delim))
    d = json.loads(path.read_text())
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        # {"clips": [...]} or {id: {...}}
        for k in ("clips", "data", "annotations", "items", "records"):
            if isinstance(d.get(k), list):
                return d[k]
        out = []
        for k, v in d.items():
            if isinstance(v, dict):
                v = dict(v)
                v.setdefault("id", k)
                out.append(v)
        if out:
            return out
    sys.exit(f"ERROR: could not interpret {path} as a list of labelled clips")


def build_mapping(rows, explicit):
    """Work out which source column feeds which internal field.

    Two passes. Exact lowercase match first, then substring, because real
    spreadsheets name columns things like `physical_plausibility_1_4` and
    `description_of_issue` — recognisable, but never equal to a bare alias. The
    substring pass only uses aliases of 5+ characters and never reuses a column
    already claimed, so it cannot quietly hijack a field that matched exactly.
    """
    keys = set()
    for r in rows[:50]:
        keys |= set(map(str, r))
    mapping = {}
    for field, aliases in ALIASES.items():
        if field in explicit:
            mapping[field] = explicit[field]
            continue
        for a in aliases:
            hit = next((k for k in keys if k.lower() == a), None)
            if hit:
                mapping[field] = hit
                break
    used = set(mapping.values())
    fuzzy = {}
    for field, aliases in ALIASES.items():
        if field in mapping:
            continue
        for a in (x for x in aliases if len(x) >= 5):
            hit = next((k for k in sorted(keys)
                        if a in k.lower() and k not in used), None)
            if hit:
                mapping[field] = hit
                fuzzy[field] = hit
                used.add(hit)
                break
    return mapping, sorted(keys), fuzzy


ALLOW_CONSTANT = False


def to_1_5(values):
    """Map any numeric rating onto 1-5, preserving order.

    Datasets arrive on 1-5, 0-1, 0-100 or 1-10 scales. Every metric we compute is
    rank-based, so the mapping only needs to be monotonic — but normalising here
    means the same thresholds and the same 'pc<=2 vs pc>=4' extreme split work
    across datasets without per-dataset tuning.
    """
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {}
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        if ALLOW_CONSTANT:        # reference sets (e.g. real demos, all "4")
            return None
        sys.exit("ERROR: all ratings identical — nothing to rank "
                 "(pass --allow-constant-rating for a reference-only set)")
    # already an integer 1-5 scale: leave alone
    if lo >= 1 and hi <= 5 and np.allclose(v, np.round(v)):
        return None
    return (lo, hi)


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


def get_violation(row, mapping):
    """Normalise the two opposite-polarity spellings into one has_violation bool."""
    if mapping.get("has_violation"):
        b = as_bool(row.get(mapping["has_violation"]))
        if b is not None:
            return b
    if mapping.get("is_clean"):
        b = as_bool(row.get(mapping["is_clean"]))
        if b is not None:
            return not b
    return None


def audit(matched, mapping, ratings):
    """Everything flagged here is cheaper to fix during annotation than after.

    Each check corresponds to a way we have actually been burned:
      * category counts -> friction (35) and permanence (21) came out with CIs wider
        than their distance from chance, so those numbers say nothing
      * clean fraction  -> without clean clips only failure-vs-failure is measurable
      * caption reuse   -> caption identity was worth 0.043 AUC of fake performance
      * generator skew  -> generator label alone scored 0.527 with no video content
    """
    print("\n" + "=" * 74)
    print("PRE-FLIGHT AUDIT  (see docs/eval_dataset_spec.md)")
    print("=" * 74)
    n = len(matched)
    problems = []

    # --- clean vs broken -----------------------------------------------------
    flags = [get_violation(r, mapping) for r, _ in matched]
    known = [f for f in flags if f is not None]
    if not known:
        print("clean/broken flag : ABSENT")
        print("   -> only failure-vs-different-failure is measurable, never")
        print("      clean-vs-broken. Add has_violation (1 checkbox per clip).")
        problems.append("no clean/broken flag")
    else:
        clean = sum(1 for f in known if not f)
        pct = 100 * clean / len(known)
        print(f"clean/broken flag : {len(known)}/{n} labelled | {clean} clean ({pct:.0f}%)")
        if pct < 25:
            print("   -> WARNING: target is 25-35% clean; below that the clean-vs-broken")
            print("      test has too few negatives to be worth reporting.")
            problems.append(f"only {pct:.0f}% clean clips")

    # --- per-category power --------------------------------------------------
    cats = Counter()
    if mapping.get("category"):
        for r, _ in matched:
            for c in set(split_cats(r.get(mapping["category"]))):
                cats[c] += 1
    if not cats:
        print("categories        : ABSENT -> will be guessed from rule text by "
              "rule_taxonomy.py")
        if not mapping.get("rules"):
            print("   -> and there is no rule text either: NO per-specialist "
                  "evaluation is possible.")
            problems.append("no category and no rule text")
    else:
        print(f"\n{'category':14s} {'pos':>5s} {'neg':>5s}  {'95% CI':>8s}  verdict")
        print("-" * 60)
        for c in CATEGORIES + ["other"]:
            p = cats.get(c, 0)
            if not p:
                continue
            hw = auc_ci(p, n - p)
            if p < 40:
                v = "UNUSABLE — cannot beat chance"
                problems.append(f"{c} has only {p} positives")
            elif p < 60:
                v = "marginal"
            elif p < 80:
                v = "ok"
            else:
                v = "good"
            print(f"{c:14s} {p:5d} {n-p:5d}  {'±%.3f' % hw:>8s}  {v}")
        oth = cats.get("other", 0)
        if oth > 0.10 * max(sum(cats.values()), 1):
            print(f"   -> WARNING: 'other' is {100*oth/sum(cats.values()):.0f}% of "
                  "labels; our 7-category taxonomy may be wrong for this data.")
            problems.append("'other' category over 10%")

    # --- temporal coverage ---------------------------------------------------
    if mapping.get("t_start"):
        have = sum(1 for r, _ in matched
                   if str(r.get(mapping["t_start"], "")).strip() not in ("", "None"))
        print(f"\ntemporal spans    : {have}/{n} rows have t_start")
        if have:
            print("   -> within-clip negatives are possible. This is the only design")
            print("      that holds scene/caption/generator constant, and the only")
            print("      way to tell motion reading from single-frame appearance")
            print("      (5 of 6 specialists scored identically on shuffled frames).")
    else:
        print("\ntemporal spans    : ABSENT")
        print("   -> cannot measure WHEN a failure was detected, only whether.")
        problems.append("no t_start/t_end")

    # --- shortcut audit: caption ---------------------------------------------
    ok_r = [(r, x) for (r, _), x in zip(matched, ratings) if x is not None]
    if mapping.get("caption") and ok_r:
        by_cap = {}
        for r, x in ok_r:
            by_cap.setdefault(str(r.get(mapping["caption"], ""))[:200], []).append(x)
        dup = {k: v for k, v in by_cap.items() if len(v) > 1 and k}
        mixed = sum(1 for v in dup.values() if max(v) - min(v) >= 2)
        print(f"\ncaption reuse     : {len(dup)} captions used by >1 clip, "
              f"{mixed} of those span both good and bad")
        if dup and mixed < 0.5 * len(dup):
            print("   -> WARNING: caption mostly predicts outcome on its own. Caption")
            print("      identity was worth +0.043 AUC of fake performance for us.")
            print("      Generate each prompt until you have both a good and a bad clip.")
            problems.append("caption predicts outcome")

    # --- shortcut audit: generator -------------------------------------------
    if mapping.get("generator") and ok_r:
        by_gen = {}
        for r, x in ok_r:
            by_gen.setdefault(str(r.get(mapping["generator"], "?")), []).append(x)
        print("\ngenerator balance :")
        for g, v in sorted(by_gen.items(), key=lambda kv: -len(kv[1]))[:10]:
            print(f"   {g[:22]:22s} n={len(v):4d}  mean rating {np.mean(v):.2f}"
                  f"  spread {min(v):.0f}-{max(v):.0f}")
        means = [np.mean(v) for v in by_gen.values() if len(v) >= 5]
        if len(means) > 1 and max(means) - min(means) > 1.0:
            print("   -> WARNING: generators differ by >1.0 rating point. Generator")
            print("      identity alone scored 0.527 AUC for us with no video content.")
            problems.append("generator predicts rating")

    # --- second annotator ----------------------------------------------------
    if mapping.get("rating2"):
        k = sum(1 for r, _ in matched
                if str(r.get(mapping["rating2"], "")).strip() not in ("", "None"))
        print(f"\nsecond annotator  : {k} clips double-rated"
              + ("" if k >= 100 else "  -> want ~150 for a stable ceiling"))
    else:
        print("\nsecond annotator  : ABSENT")
        print("   -> no inter-annotator ceiling, so every score is compared only to")
        print("      chance. ~150 double-rated clips fixes this.")
        problems.append("no second-annotator subset")

    print("\n" + "-" * 74)
    if problems:
        print(f"{len(problems)} issue(s) to fix before this dataset carries a claim:")
        for p in problems:
            print(f"   - {p}")
    else:
        print("no blocking issues found.")
    print("-" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder of video files (searched recursively)")
    ap.add_argument("--labels", required=True, help="JSON or CSV of annotations")
    ap.add_argument("--out", required=True, help="output dataset folder")
    ap.add_argument("--map", default="", help="explicit field mapping, e.g. id=clip,rating=score")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", action="store_true",
                    help="report what was detected and exit WITHOUT extracting frames")
    ap.add_argument("--allow-constant-rating", action="store_true",
                    help="stage a set whose ratings are all equal, e.g. real "
                         "demonstrations used only as clean references")
    a = ap.parse_args()
    global ALLOW_CONSTANT
    ALLOW_CONSTANT = a.allow_constant_rating

    rows = read_labels(Path(a.labels))
    explicit = dict(kv.split("=", 1) for kv in a.map.split(",") if "=" in kv)
    mapping, all_keys, fuzzy = build_mapping(rows, explicit)

    print(f"{len(rows)} label rows | columns present: {', '.join(all_keys)}")
    NOTE = {
        "id": "(REQUIRED)", "rating": "(REQUIRED)",
        "rules": "(unlocks per-specialist eval)",
        "has_violation": "(unlocks clean-vs-broken detection)",
        "is_clean": "(alt spelling of has_violation)",
        "t_start": "(unlocks temporal localisation)",
        "t_end": "(unlocks temporal localisation)",
        "rating2": "(unlocks inter-annotator ceiling)",
    }
    print("\nfield mapping:")
    for f in ALIASES:
        src = mapping.get(f)
        tag = "  [fuzzy match — check this]" if f in fuzzy else ""
        print(f"   {f:14s} <- {src if src else '— not found —'} {NOTE.get(f, '')}{tag}")
    for req in ("id", "rating"):
        if req not in mapping:
            sys.exit(f"\nERROR: no column found for '{req}'. "
                     f"Pass it explicitly, e.g. --map {req}=<column>")

    # index videos by several keys so ids match loosely
    vids = {}
    for p in Path(a.videos).rglob("*"):
        if p.suffix.lower() in VIDEO_EXT:
            vids.setdefault(p.stem, p)
            vids.setdefault(p.name, p)
    print(f"\n{len(set(map(str, vids.values())))} video files found")

    matched, unmatched = [], []
    for r in rows:
        rid = str(r.get(mapping["id"], "")).strip()
        p = vids.get(rid) or vids.get(Path(rid).stem) or vids.get(Path(rid).name)
        (matched if p else unmatched).append((r, p))
    print(f"{len(matched)}/{len(rows)} rows matched to a video file")
    if unmatched:
        print(f"   unmatched examples: "
              f"{[str(r.get(mapping['id']))[:40] for r, _ in unmatched[:3]]}")

    ratings = []
    for r, _ in matched:
        try:
            ratings.append(float(r[mapping["rating"]]))
        except Exception:  # noqa: BLE001
            ratings.append(None)
    rng = to_1_5(ratings)
    good = [x for x in ratings if x is not None]
    print(f"ratings: {len(good)} numeric, range {min(good):.3g}–{max(good):.3g}"
          + (f"  -> rescaling to 1–5" if rng else "  -> already 1–5, kept as is"))
    n_rules = sum(1 for r, _ in matched
                  if mapping.get("rules") and len(str(r.get(mapping["rules"], ""))) > 4)
    print(f"rows with rule text: {n_rules}"
          + ("" if n_rules else "   <-- per-specialist eval NOT possible without this"))
    has_time = mapping.get("t_start") and mapping.get("t_end")
    print(f"temporal spans: {'YES — temporal accuracy can be measured' if has_time else 'no'}")
    print(f"object names:   {'YES — object-conditioned questions possible' if mapping.get('object') else 'no'}")

    audit(matched, mapping, ratings)

    if a.inspect:
        print("\n--inspect set: stopping before frame extraction.")
        return

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    shutil.rmtree(out, ignore_errors=True)
    (out / "frames").mkdir(parents=True)
    rnd = random.Random(a.seed)
    clips, dropped = [], 0
    for i, ((r, p), raw) in enumerate(zip(matched, ratings)):
        if raw is None:
            dropped += 1
            continue
        pc = int(round(raw)) if rng is None else int(round(
            1 + 4 * (raw - rng[0]) / (rng[1] - rng[0])))
        pc = max(1, min(5, pc))
        cid = str(r.get(mapping["id"]))[:110].replace("/", "_")
        k = extract(p, out / "frames" / cid, a.frames, a.max_side)
        if k < 4:
            dropped += 1
            continue
        order = list(range(k))
        shuf = order[:]
        # never let "shuffled" equal temporal — it would silently void the control
        while k > 2 and shuf == order:
            rnd.shuffle(shuf)
        def num(field):
            try:
                return float(r.get(mapping.get(field, ""), ""))
            except (TypeError, ValueError):
                return None

        clips.append({
            "clip_id": cid, "n_frames": k, "pc": pc, "rating_raw": raw,
            "generator": str(r.get(mapping.get("generator", ""), "unknown")),
            "caption": str(r.get(mapping.get("caption", ""), ""))[:400],
            "violated_rules": str(r.get(mapping.get("rules", ""), "")),
            "category_given": str(r.get(mapping.get("category", ""), "")),
            # normalised multi-label form; downstream code should prefer this over
            # re-parsing category_given
            "categories": split_cats(r.get(mapping.get("category", ""), "")),
            # None means "not annotated", which is NOT the same as False
            "has_violation": get_violation(r, mapping),
            "object": str(r.get(mapping.get("object", ""), "")),
            "t_start": num("t_start"),
            "t_end": num("t_end"),
            "severity": str(r.get(mapping.get("severity", ""), "")),
            "rating2": num("rating2"),
            "order_temporal": order, "order_shuffled": shuf,
        })
        if len(clips) % 100 == 0:
            print(f"   staged {len(clips)}", flush=True)

    meta = {"dataset": str(a.labels), "n_clips": len(clips),
            "frames_per_clip": a.frames, "max_side": a.max_side, "seed": a.seed,
            "field_mapping": mapping,
            "label": "pc = human physics rating 1-5, higher = MORE correct",
            "clips": clips}
    (out / "manifest.json").write_text(json.dumps(meta, indent=1))
    print(f"\n{len(clips)} clips staged -> {out}  ({dropped} dropped)")
    print("  pc distribution:", dict(sorted(Counter(c["pc"] for c in clips).items())))
    print("  generators     :", dict(Counter(c["generator"] for c in clips).most_common(8)))
    print("  categories     :", dict(Counter(
        c for cl in clips for c in set(cl["categories"])).most_common()))
    print("  clean clips    :", sum(1 for c in clips if c["has_violation"] is False))
    print("  with timestamps:", sum(1 for c in clips if c["t_start"] is not None))
    print(f"\nNext: python backend/scripts/eval_run.py --data {a.out}")


if __name__ == "__main__":
    main()
