"""
Map VideoPhy-2 `human_violated_rules` text onto PhysicsLENS specialist categories.

This is the bridge that makes per-specialist evaluation possible at all: the
dataset labels a violation in free text ("The axe must make contact with the tree
to cause it to split"), and PhysicsLENS organises its Stage-3 specialists by
physics category. Without this mapping there is no way to ask "is the gravity
specialist good at gravity?".

Keyword-based on purpose, not LLM-based: the mapping is the measurement
instrument here, so it must be deterministic, inspectable, and stable across
runs. An LLM classifier would put a second stochastic component inside the thing
being measured.

Rules are MULTI-LABEL — one clip can break contact and deformation at once, and
forcing a single label would corrupt the negatives for every other specialist.

Usage:
  python backend/scripts/rule_taxonomy.py --data data/videophy1200
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

# Ordered specialist categories. Patterns are matched against the lowercased
# rule text; a rule may match several.
CATEGORIES = {
    "collision": [
        r"\bcontact\b", r"\bstrike[sd]?\b", r"\bstruck\b", r"\bhit(s|ting)?\b",
        r"\bimpact\b", r"\bcollide|collision", r"\btouch(es|ing|ed)?\b",
        r"\bconnect(s|ing|ed)?\b", r"\bmake contact\b",
    ],
    "gravity": [
        r"\bgravity\b", r"\bfall(s|ing|en)?\b", r"\bfloat(s|ing)?\b",
        r"\bhover(s|ing)?\b", r"\bsuspend(ed|s)?\b", r"\bmid-?air\b",
        r"\bsupport(ed|s|ing)?\b", r"\bdrop(s|ping|ped)?\b", r"\bdescend",
        r"\bsink(s|ing)?\b", r"\bweight\b",
    ],
    "deformation": [
        r"\bdeform", r"\bshape\b", r"\bbend(s|ing)?\b", r"\bstructural\b",
        r"\bintegrity\b", r"\brigid\b", r"\bstretch", r"\bcompress",
        r"\bbreak(s|ing)?\b", r"\bshatter", r"\bmelt", r"\bwarp",
    ],
    "momentum": [
        r"\bmomentum\b", r"\bvelocity\b", r"\btrajectory\b", r"\baccelerat",
        r"\bdecelerat", r"\bspeed\b", r"\brebound", r"\bbounce",
        r"\bconserv", r"\binertia\b", r"\bforward motion\b", r"\bpropel",
        # gap-fill from observed unmatched rules: elastic/reflective collisions
        # and oscillatory decay are momentum statements in other words.
        r"\boscillat", r"\bdamp(s|ing|ed)?\b", r"\belastic", r"\breflect",
        r"\bangle of incidence\b", r"\bswing(s|ing)?\b", r"\bpendulum\b",
    ],
    "friction": [
        r"\bfriction\b", r"\bslip(s|ping|ped)?\b", r"\bslide|sliding\b",
        r"\bgrip\b", r"\btraction\b", r"\broll(s|ing)?\b", r"\bstatic\b",
        r"\bresistance\b", r"\bskid",
    ],
    "fluid": [
        r"\bwater\b", r"\bliquid\b", r"\bfluid\b", r"\bsplash", r"\bpour",
        r"\bflow(s|ing)?\b", r"\bwave(s)?\b", r"\bdrip", r"\bspill",
        r"\bbubble", r"\bsmoke\b", r"\bsteam\b", r"\bfoam\b", r"\bripple",
    ],
    "permanence": [
        r"\bdisappear", r"\bvanish", r"\bappear(s|ing)? from\b",
        r"\bmaterializ", r"\bduplicate", r"\bmerge(s|d)?\b",
        r"\bremain(s|ing)? (visible|intact|present)\b", r"\bpersist",
        r"\bidentity\b", r"\bchanging into\b", r"\bturn(s|ing)? into\b",
    ],
}

# Gap-fill for deformation added separately to keep the block above readable.
CATEGORIES["deformation"] += [r"\bcurvature\b", r"\bflat\b", r"\btension\b",
                              r"\bcrumple", r"\bfold(s|ing)?\b"]

# Probe name (physics_probes.PROBES) -> specialist category it is meant to catch.
PROBE_FOR = {
    "collision": "contact",
    "gravity": "support",
    "deformation": "integrity",
    "momentum": "trajectory",
    "permanence": "permanence",
    "friction": "friction",
    "fluid": "fluid",
}


def parse_rules(raw: str) -> list[str]:
    """The field is a stringified python list; fall back to raw text."""
    raw = (raw or "").strip()
    if len(raw) < 5:
        return []
    try:
        v = json.loads(raw.replace("'", '"'))
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:  # noqa: BLE001
        pass
    return re.findall(r"'([^']{8,})'", raw) or [raw]


def categorize(raw: str) -> list[str]:
    """-> sorted list of matching categories (multi-label, possibly empty)."""
    txt = " ".join(parse_rules(raw)).lower()
    if not txt:
        return []
    hits = set()
    for cat, pats in CATEGORIES.items():
        if any(re.search(p, txt) for p in pats):
            hits.add(cat)
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--show", type=int, default=0, help="print N examples/category")
    a = ap.parse_args()

    data = ROOT / a.data
    clips = json.loads((data / "manifest.json").read_text())["clips"]

    labeled, cats = 0, Counter()
    per_clip, examples = {}, defaultdict(list)
    unmatched = []
    # A dataset annotated per docs/eval_dataset_spec.md ships `categories` directly.
    # Human labels win — the keyword pass is a fallback, not a second opinion. We
    # still run it on those clips so we can report how far off the guesser is.
    n_given, agree, both = 0, 0, 0
    for c in clips:
        guess = categorize(c.get("violated_rules", ""))
        given = [x for x in (c.get("categories") or []) if x != "other"]
        if given:
            n_given += 1
            if guess:
                both += 1
                agree += bool(set(given) & set(guess))
        cs = given or guess
        per_clip[c["clip_id"]] = cs
        if len(c.get("violated_rules") or "") > 4:
            labeled += 1
            if not cs:
                unmatched.append(c["violated_rules"][:100])
        for x in cs:
            cats[x] += 1
            if len(examples[x]) < 3:
                examples[x].append(" ".join(parse_rules(c["violated_rules"]))[:95])

    print(f"{len(clips)} clips | {labeled} carry a rule annotation")
    if n_given:
        print(f"human categories supplied for {n_given} clips — used directly")
        if both:
            print(f"  keyword taxonomy agrees with the human label on "
                  f"{agree}/{both} ({100*agree/both:.0f}%) of clips having both")
    print(f"unmatched by the taxonomy: {len(unmatched)} "
          f"({100*len(unmatched)/max(labeled,1):.0f}% of labeled)\n")
    print(f"{'specialist':14s} {'clips':>6s} {'% of labeled':>13s}  probe")
    print("-" * 62)
    for cat in CATEGORIES:
        n = cats[cat]
        pr = PROBE_FOR.get(cat) or "— none —"
        print(f"{cat:14s} {n:6d} {100*n/max(labeled,1):12.1f}%  {pr}")
    multi = sum(1 for v in per_clip.values() if len(v) > 1)
    print(f"\nmulti-label clips: {multi} (a clip can break several rule types)")

    if a.show:
        print()
        for cat in CATEGORIES:
            print(f"[{cat}]")
            for e in examples[cat][:a.show]:
                print(f"   {e}")
    if unmatched[:5]:
        print("\nsample unmatched rules (taxonomy gaps):")
        for u in unmatched[:5]:
            print(f"   {u}")

    outp = data / "rule_categories.json"
    outp.write_text(json.dumps({"categories": list(CATEGORIES),
                                "probe_for": PROBE_FOR,
                                "n_labeled": labeled,
                                "counts": dict(cats),
                                "per_clip": per_clip}, indent=1))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
