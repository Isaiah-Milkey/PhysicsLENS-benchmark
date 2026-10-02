"""
Three test-time strategies for making the physics judge better, beyond prompt
wording (which GEPA showed is not the lever: +0.017 held-out, indistinguishable
from blind paraphrase).

Each writes per-clip scores for all 300 staged clips so that fusion weights can
be FIT on train and reported on the held-out test split.

  --method probes    Decompose one holistic rating into 6 targeted binary
                     defect probes, each scored by token probability. Probe
                     categories are mined from the 181 human `violated_rules`
                     annotations, whose dominant terms are contact/impact,
                     structural integrity, support/floating, motion without
                     force, deformation, trajectory. This is the "specialist"
                     framing: k cheap focused judgements instead of one vague
                     one. Per-probe scores are kept separately so a learned
                     weighting can be fit later.

  --method fewshot   In-context learning. Four LABELED exemplar clips drawn
                     from TRAIN ONLY (never val/test — that would leak) are
                     shown with their human rating and the rule humans said was
                     broken, then the target. Tests whether the judge's problem
                     is not knowing what a "2" looks like on this scale.

  --method pairwise  Ranking by comparison instead of by absolute rating. The
                     target is compared against 6 fixed TRAIN anchors of known
                     rating; score = mean P(target is worse). Our metric IS a
                     ranking, and pointwise absolute ratings are known to be
                     poorly calibrated for ranking, so asking the question in
                     the form we actually score should help.
                     Anchor order alternates A/B by index so position bias
                     cancels across anchors instead of accumulating.

Usage:
  python backend/scripts/physics_probes.py --method probes
  python backend/scripts/physics_probes.py --method fewshot
  python backend/scripts/physics_probes.py --method pairwise
"""
import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman                                  # noqa: E402
from videophy_eval import _retry, _token_probs, auc_extremes, client    # noqa: E402
from gepa_optimize import load, split_clips                             # noqa: E402

TASK_MODEL = "gemma4-31b-it"

# Mined from the 181 human violated_rules. "Yes" = the defect IS present, so
# every probe points the same direction (higher = more broken) and they can be
# averaged without sign bookkeeping.
PROBES = [
    ("contact",
     "Look at these {n} frames from a video showing: \"{caption}\".\n"
     "In a real version of this action, one object must physically touch another "
     "for the effect to happen (a bat must strike the ball, an axe must reach the "
     "tree, a hand must reach the object).\n"
     "Does an effect happen WITHOUT the required physical contact ever being "
     "visible — objects reacting at a distance, or a gap remaining at the moment "
     "of impact?\nAnswer exactly one word: Yes or No."),
    ("integrity",
     "Look at these {n} frames from a video showing: \"{caption}\".\n"
     "Does any solid object lose its structural integrity in a way real materials "
     "would not — changing shape, bending, melting, merging into another object, "
     "or having parts detach without a force that would cause it?\n"
     "Answer exactly one word: Yes or No."),
    ("support",
     "Look at these {n} frames from a video showing: \"{caption}\".\n"
     "Is any object unsupported yet not falling — hovering, floating, resting on "
     "nothing, or staying suspended in mid-air when gravity should pull it down?\n"
     "Answer exactly one word: Yes or No."),
    ("spontaneous",
     "Look at these {n} frames from a video showing: \"{caption}\".\n"
     "Does anything start moving, change direction, or stop with no visible cause "
     "— no push, no collision, no applied force to explain it?\n"
     "Answer exactly one word: Yes or No."),
    ("permanence",
     "Look at these {n} frames from a video showing: \"{caption}\".\n"
     "Does any object or body part appear from nowhere, vanish, duplicate, or "
     "change into a different object between frames?\n"
     "Answer exactly one word: Yes or No."),
    ("trajectory",
     "Look at these {n} frames from a video showing: \"{caption}\".\n"
     "Does any moving object follow an impossible path — an arc that bends the "
     "wrong way, motion that speeds up with nothing driving it, or a bounce that "
     "returns more energy than it received?\n"
     "Answer exactly one word: Yes or No."),
    # Added for the per-specialist ablation: `fluid` had no matched probe at all,
    # and `friction` was being scored by the `spontaneous` probe, which is about
    # uncaused motion rather than contact resistance.
    ("fluid",
     "Look at these {n} frames from a video showing: \"{caption}\".\n"
     "Does any liquid, smoke, spray or other fluid behave impossibly — water "
     "that vanishes or appears instantly, splashes that do not match the impact, "
     "flow running the wrong way, or fluid holding a shape it could not hold?\n"
     "Answer exactly one word: Yes or No."),
    ("friction",
     "Look at these {n} frames from a video showing: \"{caption}\".\n"
     "Does contact between surfaces behave impossibly — something sliding when "
     "it should grip, rolling without the surface driving it, starting to slide "
     "with no force applied, or stopping instantly with nothing to stop it?\n"
     "Answer exactly one word: Yes or No."),
]

FEWSHOT_INTRO = (
    "You are grading how well AI-generated videos obey real-world physics, on the "
    "same 1-5 scale human annotators used (1 = badly broken physics, 5 = fully "
    "physically correct). Here are {k} graded examples showing how humans applied "
    "that scale, then the video you must grade."
)

PAIRWISE_Q = (
    "Two videos, each shown as {f} frames in temporal order.\n"
    "VIDEO A ({ca}) is the first {f} images. VIDEO B ({cb}) is the next {f} images.\n"
    "Which video's MOTION and INTERACTIONS break real-world physics MORE "
    "(impossible contact, floating, morphing, motion without force)?\n"
    "Answer with exactly one letter: A or B."
)


def frames(data: Path, clip: dict, k: int = 8) -> list[str]:
    """k evenly-spaced frames from the frozen temporal ordering, as data URLs."""
    idx = clip["order_temporal"]
    if k < len(idx):
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, k).astype(int)]
    d = data / "frames" / clip["clip_id"]
    out = []
    for i in idx:
        p = d / f"{i:03d}.jpg"
        if p.exists():
            out.append("data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode())
    return out


def _content(imgs, text):
    return [{"type": "image_url", "image_url": {"url": u}} for u in imgs] + \
           [{"type": "text", "text": text}]


# ── method: probes ────────────────────────────────────────────────────────────

def run_probes(c, data, clip, model=TASK_MODEL):
    """-> {probe_name: P(defect present)}. Missing probes are dropped, not
    zero-filled: a failed call is absence of evidence, and zero would read as
    confident 'no defect' and drag the mean down."""
    imgs = frames(data, clip, 8)
    out = {}
    for name, q in PROBES:
        try:
            p = _token_probs(c, model, imgs,
                             q.format(n=len(imgs), caption=clip.get("caption", "")))
            yes, no = p.get("yes", 0.0), p.get("no", 0.0)
            if yes + no > 1e-4:
                out[name] = yes / (yes + no)
        except Exception as e:  # noqa: BLE001
            print(f"    probe {name} fail {clip['clip_id'][:28]}: {str(e)[:50]}",
                  file=sys.stderr)
    return out


# ── method: fewshot ───────────────────────────────────────────────────────────

def pick_exemplars(train, seed=0):
    """Four labeled exemplars spanning the scale: pc 1, 2, 4, 5.

    Deliberately skips pc 3 — the exemplars exist to anchor the ENDS of the
    scale, and a mid exemplar mostly teaches the model to answer 3.
    Preference for clips that carry a human rule annotation, since that text is
    the part that teaches what counts as a violation.
    """
    import random
    rnd = random.Random(seed)
    out = []
    for pc in (1, 2, 4, 5):
        pool = [c for c in train if c["pc"] == pc]
        withr = [c for c in pool if len(c.get("violated_rules") or "") > 4]
        cand = withr if (pc <= 2 and withr) else pool
        if cand:
            out.append(rnd.choice(cand))
    return out


def run_fewshot(c, data, clip, exemplars, model=TASK_MODEL):
    content = [{"type": "text", "text": FEWSHOT_INTRO.format(k=len(exemplars))}]
    for j, ex in enumerate(exemplars, 1):
        content += [{"type": "image_url", "image_url": {"url": u}}
                    for u in frames(data, ex, 3)]
        t = (f"EXAMPLE {j} — supposed to show: \"{ex['caption'][:150]}\"\n"
             f"Human rating: {ex['pc']}/5")
        r = (ex.get("violated_rules") or "").strip()
        if len(r) > 4:
            t += f"\nRule humans said was broken: {r[:160]}"
        content.append({"type": "text", "text": t})

    tgt = frames(data, clip, 8)
    content.append({"type": "text", "text":
                    f"NOW GRADE THIS VIDEO — supposed to show: "
                    f"\"{clip.get('caption','')[:150]}\"\n"
                    f"({len(tgt)} frames, temporal order)"})
    content += [{"type": "image_url", "image_url": {"url": u}} for u in tgt]
    content.append({"type": "text", "text":
                    "Rate this video 1-5 on the same scale the humans used.\n"
                    "Answer with exactly one digit, 1 to 5. No other text."})

    r = _retry(lambda: c.chat.completions.create(
        model=model, max_tokens=1, temperature=0, logprobs=True,
        top_logprobs=20, messages=[{"role": "user", "content": content}]))
    lp = r.choices[0].logprobs
    if not lp or not lp.content:
        return None
    import math
    p = {}
    for t in lp.content[0].top_logprobs:
        k = t.token.strip().lower()
        p[k] = p.get(k, 0.0) + math.exp(t.logprob)
    mass = {int(k): v for k, v in p.items() if len(k) == 1 and k in "12345"}
    tot = sum(mass.values())
    if tot < 1e-4:
        return None
    ev = sum(k * v for k, v in mass.items()) / tot
    return (5.0 - ev) / 4.0                      # -> implausibility 0..1


# ── method: pairwise ──────────────────────────────────────────────────────────

def pick_anchors(train, seed=0):
    """Six train anchors at the ends of the scale (1,1,2,4,5,5)."""
    import random
    rnd = random.Random(seed + 7)
    out = []
    for pc in (1, 1, 2, 4, 5, 5):
        pool = [c for c in train if c["pc"] == pc and c not in out]
        if pool:
            out.append(rnd.choice(pool))
    return out


def run_pairwise(c, data, clip, anchors, f=4, model=TASK_MODEL):
    """mean P(target is the more-broken video), over anchors.

    Position alternates so bias cancels: on even anchors the target is A, on
    odd it is B. Reading P(A) vs P(B) accordingly means a model that simply
    favours the first slot contributes ~0.5 to every clip and shifts nothing.
    """
    tgt = frames(data, clip, f)
    vals = []
    for i, an in enumerate(anchors):
        if an["clip_id"] == clip["clip_id"]:
            continue
        af = frames(data, an, f)
        tgt_is_a = (i % 2 == 0)
        imgs = (tgt + af) if tgt_is_a else (af + tgt)
        ca = "target" if tgt_is_a else "reference"
        cb = "reference" if tgt_is_a else "target"
        try:
            p = _token_probs(c, model, imgs,
                             PAIRWISE_Q.format(f=f, ca=ca, cb=cb))
            pa, pb = p.get("a", 0.0), p.get("b", 0.0)
            if pa + pb > 1e-4:
                vals.append((pa if tgt_is_a else pb) / (pa + pb))
        except Exception as e:  # noqa: BLE001
            print(f"    pair fail {clip['clip_id'][:28]}: {str(e)[:50]}",
                  file=sys.stderr)
    return float(np.mean(vals)) if vals else None


# ── driver ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy300")
    ap.add_argument("--method", required=True,
                    choices=["probes", "fewshot", "pairwise"])
    ap.add_argument("--model", default=TASK_MODEL,
                    help="gateway model id; the probe set is identical across "
                         "models so per-specialist numbers stay comparable")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=90)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    data, clips = load(a.data)
    train, val, test = split_clips(clips, a.seed, a.n_train, a.n_val)
    if a.limit:
        clips = clips[:a.limit]
    test_ids = {c["clip_id"] for c in test}

    c = client()
    ex = an = None
    if a.method == "fewshot":
        ex = pick_exemplars(train, a.seed)
        print("exemplars (train only): " +
              ", ".join(f"pc{e['pc']}:{e['caption'][:30]}" for e in ex))
    if a.method == "pairwise":
        an = pick_anchors(train, a.seed)
        print("anchors (train only): " + ", ".join(f"pc{x['pc']}" for x in an))

    print(f"{a.method} [{a.model}]: scoring {len(clips)} clips, {a.workers} workers", flush=True)
    t0 = time.time()
    res = [None] * len(clips)
    done = [0]

    def work(i):
        cl = clips[i]
        if a.method == "probes":
            res[i] = run_probes(c, data, cl, a.model)
        elif a.method == "fewshot":
            res[i] = run_fewshot(c, data, cl, ex, model=a.model)
        else:
            res[i] = run_pairwise(c, data, cl, an, model=a.model)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"    {done[0]}/{len(clips)}  ({time.time()-t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as exr:
        list(exr.map(work, range(len(clips))))

    # Collapse to one score per clip. For probes the unweighted mean is only a
    # provisional readout — fusion fits per-probe weights on train later.
    def collapse(r):
        if r is None:
            return None
        if a.method == "probes":
            return float(np.mean(list(r.values()))) if r else None
        return r

    scores = {cl["clip_id"]: collapse(r) for cl, r in zip(clips, res)}
    out = {"method": a.method, "model": a.model, "n": len(clips),
           "seconds": round(time.time() - t0, 1),
           "exemplar_ids": [e["clip_id"] for e in ex] if ex else None,
           "anchor_ids": [x["clip_id"] for x in an] if an else None,
           "scores": scores}
    if a.method == "probes":
        out["probe_scores"] = {cl["clip_id"]: r for cl, r in zip(clips, res)}

    tag = a.method if a.model == TASK_MODEL else f"{a.method}_{a.model}"
    outp = data / f"method_{tag}.json"
    outp.write_text(json.dumps(out, indent=1))

    # Provisional held-out readout. The authoritative table comes from
    # method_report.py, which also fits the learned fusion.
    sub = [(scores[cl["clip_id"]], cl["pc"]) for cl in test
           if scores.get(cl["clip_id"]) is not None]
    if len(sub) > 8:
        xs = [s for s, _ in sub]
        r = spearman(xs, [6 - p for _, p in sub])
        auc, _, _ = auc_extremes(xs, [p for _, p in sub])
        print(f"\n  [{a.method}] held-out test n={len(sub)}  rho={r:+.3f}  "
              f"AUC={auc or 0:.3f}   (baseline rho=+0.264 AUC=0.660)")
    print(f"  -> {outp}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
