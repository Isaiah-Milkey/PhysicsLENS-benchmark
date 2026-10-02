"""
Caption and frame-order ablations on the winning per-specialist questions.

Two claims in the meeting report depend on these, and both are the kind that
look fine until someone asks:

  CAPTION   Every domain question embeds the clip's caption ("...from a video of:
            <caption>"). A detector could therefore be scoring the PROMPT TEXT
            rather than the pixels — captions describe the intended action, and
            harder actions correlate with worse generations. If the caption-free
            scores hold up, the detector is visual. If they collapse, part of
            what we are calling physics detection is text priors.

  ORDER     `g_no_accel` is worded as an explicit comparison ACROSS frames
            ("the gap between later frames must be LARGER than between earlier
            frames"). Shuffling the frames destroys that ordering. If the score
            does not drop, the question is being answered from a static cue and
            the "compares acceleration across frames" story is wrong — which
            matters, because that question is our best domain result and the
            report currently explains it that way.

Both are scored on the 900 held-out clips only, against the same fixed questions,
so the comparison is like-for-like.

Usage:
  python backend/scripts/ablation_report.py --data data/videophy1200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from gepa_optimize import load                          # noqa: E402
from specialist_report import auc, auc_ci               # noqa: E402
from vlm_rapidata_eval import spearman                  # noqa: E402

WIN = {"gravity": "g_no_accel", "permanence": "p_count_change",
       "collision": "c2_overlap", "deformation": "d2_length_change",
       "friction": "f_no_rolling_link", "momentum": "m_gains_energy"}


def load_variant(data, suffix):
    """Load one condition, keyed by FILENAME rather than metadata.

    The metadata fields (`caption`, `order`) were meant to record the condition
    but the writer patch did not take, so they come back absent on the ablation
    files. The filename suffix is authoritative and unambiguous, so use it.

    The ablation runs used the "winners" battery, which names its probes
    `w_<category>`; the baseline came from the original per-category batteries,
    which use the question's own name (`g_no_accel`, ...). Both are normalised to
    the category name here so the three conditions are directly comparable.
    """
    out = {}
    for f in sorted(data.glob("domain_probes_*.json")):
        nm = f.name
        if suffix == "base":
            # gemma4, temporal, with caption: the original per-category files
            if "__" in nm:
                continue
        elif not nm.endswith(f"__{suffix}.json"):
            continue
        d = json.loads(f.read_text())
        if d.get("n", 0) < 100 or d.get("model", "") != "gemma4-31b-it":
            continue
        for cid, r in d["probe_scores"].items():
            for cat, key in WIN.items():
                v = r.get(f"w_{cat}", r.get(key))
                if v is not None:
                    out.setdefault(cid, {})[cat] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--sel", default="data/videophy300")
    a = ap.parse_args()
    data, clips = load(a.data)
    n = len(clips)
    sel = {c["clip_id"] for c in load(a.sel)[1]}
    TST = [i for i in range(n) if clips[i]["clip_id"] not in sel]
    co = [set(json.loads((data / "rule_categories.json").read_text())["per_clip"]
              .get(c["clip_id"], [])) for c in clips]
    has = [len(c.get("violated_rules") or "") > 4 for c in clips]

    base = load_variant(data, "base")
    nocap = load_variant(data, "nocap")
    shuf = load_variant(data, "shuffled")
    print(f"clips scored — baseline {len(base)}, no-caption {len(nocap)}, "
          f"shuffled {len(shuf)}")
    if not nocap and not shuf:
        sys.exit("neither ablation has full results yet")

    print(f"\n{'specialist':13s} {'n+':>4s} {'baseline':>18s} {'NO CAPTION':>18s} "
          f"{'SHUFFLED':>18s}")
    print("-" * 78)
    rows = {}
    for cat, key in WIN.items():
        pos = [cat in s for s in co]
        cells, vals = [], {}
        for tag, src in (("base", base), ("nocap", nocap), ("shuf", shuf)):
            if not src:
                cells.append(f"{'(not run)':>18s}")
                continue
            ds = [i for i in TST if (pos[i] or (has[i] and co[i]))
                  and cat in (src.get(clips[i]["clip_id"]) or {})]
            p_ = [src[clips[i]["clip_id"]][cat] for i in ds if pos[i]]
            q_ = [src[clips[i]["clip_id"]][cat] for i in ds if not pos[i]]
            if len(p_) < 8 or len(q_) < 8:
                cells.append(f"{'(n<8)':>18s}")
                continue
            v = auc(p_, q_)
            lo, hi = auc_ci(p_, q_)
            vals[tag] = v
            cells.append(f"{v:.3f} [{lo:.2f},{hi:.2f}]")
        npos = sum(1 for i in TST if pos[i] and has[i])
        print(f"{cat:13s} {npos:4d} " + " ".join(f"{c:>18s}" for c in cells))
        rows[cat] = vals

    # agreement between conditions on the SAME clips — the sharper test
    print(f"\n{'specialist':13s} {'rho(base,nocap)':>17s} {'rho(base,shuf)':>17s}"
          "   interpretation")
    print("-" * 78)
    for cat, key in WIN.items():
        line = []
        for src in (nocap, shuf):
            if not src:
                line.append(float("nan"))
                continue
            kp = [i for i in TST
                  if cat in (base.get(clips[i]["clip_id"]) or {})
                  and cat in (src.get(clips[i]["clip_id"]) or {})]
            line.append(spearman([base[clips[i]["clip_id"]][cat] for i in kp],
                                 [src[clips[i]["clip_id"]][cat] for i in kp])
                        if len(kp) > 20 else float("nan"))
        note = ""
        if line[1] == line[1] and line[1] > 0.8:
            note = "order irrelevant -> static cue"
        elif line[1] == line[1] and line[1] < 0.5:
            note = "order matters -> reads sequence"
        print(f"{cat:13s} {line[0]:17.3f} {line[1]:17.3f}   {note}")

    outp = data / "ablation_report.json"
    outp.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
