"""
Score a VLM on the staged VideoPhy-2 subset, with frame-order and prompt
ablations built in.

Reads the FROZEN frames from videophy_prepare.py, so every condition sees
byte-identical pixels and any score difference is attributable to the change
under test (frame order, or prompt).

Label: `pc` = human physical-commonsense rating, 1-5, HIGHER = MORE physically
correct. Our model score is an implausibility (higher = worse), so we correlate
against `implausibility = 6 - pc` and expect a POSITIVE rho. Getting this sign
backwards silently inverts every conclusion, so it is asserted in one place.

Metrics
  rho    Spearman vs implausibility (all clips)
  AUC    P(model scores a pc<=2 clip above a pc>=4 clip) — a clean binary with
         a real gap, unlike a median split on a compressed scale.

Prompt variants (--prompt):
  binary   Yes/No "does this violate physics" + P(Yes) logprob      [baseline]
  likert   "rate 1-5 how physically correct" + expected value over digit
           logprobs — matches how the ground truth was actually collected
  caption  binary, but told what the video is SUPPOSED to depict
  cot      brief reasoning then a rating (generated only, no logprob)

Usage:
  python backend/scripts/videophy_eval.py --data data/videophy300
  python backend/scripts/videophy_eval.py --data data/videophy300 --order shuffled
  python backend/scripts/videophy_eval.py --data data/videophy300 --prompt likert
"""
import argparse
import base64
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman  # noqa: E402

DEFAULT_MODEL = "gemma4-31b-it"

BINARY_Q = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video. AI-generated videos often show objects that morph, appear/disappear, "
    "float, pass through each other, or move without forces. Judging the MOTION and "
    "INTERACTIONS across frames: does this video violate real-world physics?\n"
    "Answer with exactly one word: Yes or No."
)
CAPTION_Q = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video that is supposed to show: \"{caption}\"\n"
    "Judging the MOTION and INTERACTIONS across frames — do the objects behave the "
    "way real physics would make them behave during that action?\n"
    "Answer with exactly one word: Yes if the physics is WRONG, No if it is correct."
)
LIKERT_Q = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video. Rate how well the MOTION and INTERACTIONS obey real-world physics "
    "(gravity, momentum, object permanence, rigid-body and fluid behavior).\n"
    "1 = badly broken physics, 5 = fully physically correct.\n"
    "Answer with exactly one digit, 1 to 5."
)
COT_Q = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video showing: \"{caption}\"\n"
    "First name the single most important physical rule this action must obey. "
    "Then state whether the frames obey it. Then give a rating.\n"
    'Reply with ONLY strict JSON: {{"rule": "...", "obeyed": true/false, '
    '"rating": <1-5, 1=broken, 5=correct>}}'
)


def client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    # OPENAI_BASE_URL is optional — omit it to hit api.openai.com directly, or
    # point it at any other OpenAI-compatible endpoint (self-hosted gateway,
    # vLLM/LiteLLM serving the open-weight backbones, etc.).
    k = os.environ.get("OPENAI_API_KEY")
    b = os.environ.get("OPENAI_BASE_URL") or None
    if not k:
        sys.exit("ERROR: OPENAI_API_KEY missing from .env")
    return OpenAI(base_url=b, api_key=k, timeout=180, max_retries=3)


def load_frames(data: Path, clip: dict, order: str) -> list[str]:
    """Frozen JPEGs as data URLs, in the ordering fixed at staging time."""
    idx = clip["order_shuffled"] if order == "shuffled" else clip["order_temporal"]
    d = data / "frames" / clip["clip_id"]
    out = []
    for i in idx:
        p = d / f"{i:03d}.jpg"
        if p.exists():
            out.append("data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode())
    return out


def _retry(call, tries=6, base=2.0):
    """Retry on the gateway's 503/429. LiteLLM in front of a busy backend sheds
    load aggressively under concurrency — without backoff a 300-clip sweep loses
    most of its calls (observed: 5/300 scored). Jittered exponential backoff."""
    import time as _t
    last = None
    for k in range(tries):
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            last = e
            code = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None)
            if code not in (429, 500, 502, 503, 504):
                raise
            _t.sleep(base * (2 ** k) * (0.5 + random.random()))
    raise last


def _msg(imgs, text):
    return [{"role": "user", "content":
             [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
             + [{"type": "text", "text": text}]}]


def _token_probs(c, model, imgs, question, top=20):
    """{normalised token -> summed probability} for the first generated token.

    Two traps, both verified against this gateway on 2026-08-21:

    1. Probabilities must be SUMMED across surface forms, not assigned. "Yes",
       " Yes" and "YES" all normalise to "yes"; a dict comprehension would let
       the last (usually ~0) overwrite the real mass.
    2. `top_logprobs` must be 20. At 2/5/10 this gateway returns a malformed
       distribution — probabilities summing to 1.36/1.68/1.56 with Yes and No
       assigned identical values. Only top=20 sums to 1.0.
    """
    r = _retry(lambda: c.chat.completions.create(
        model=model, max_tokens=1, temperature=0,
        logprobs=True, top_logprobs=top, messages=_msg(imgs, question)))
    lp = r.choices[0].logprobs
    if not lp or not lp.content:
        return {}
    out: dict[str, float] = {}
    for t in lp.content[0].top_logprobs:
        out[t.token.strip().lower()] = out.get(t.token.strip().lower(), 0.0) \
            + math.exp(t.logprob)
    return out


def score_binary(c, model, imgs, clip, caption=False):
    q = (CAPTION_Q if caption else BINARY_Q).format(
        n=len(imgs), caption=clip.get("caption", ""))
    p = _token_probs(c, model, imgs, q)
    yes, no = p.get("yes", 0.0), p.get("no", 0.0)
    return yes / (yes + no) if (yes + no) > 1e-4 else None


def score_likert(c, model, imgs, clip):
    """Expected value over the digit distribution, mapped to implausibility.

    Using the full distribution rather than argmax keeps the score continuous —
    argmax would reproduce the 5-distinct-values quantisation that caps rank
    correlation regardless of how accurate the model is.
    """
    p = _token_probs(c, model, imgs, LIKERT_Q.format(n=len(imgs)))
    # `k in "12345"` alone is a trap: "" is a substring of every string, and the
    # gateway does return empty tokens, so int("") blows up. Require length 1.
    mass = {int(k): v for k, v in p.items() if len(k) == 1 and k in "12345"}
    tot = sum(mass.values())
    if tot < 1e-4:
        return None
    ev = sum(k * v for k, v in mass.items()) / tot      # expected pc, 1..5
    return (5.0 - ev) / 4.0                             # -> implausibility 0..1


def score_cot(c, model, imgs, clip):
    import re
    r = _retry(lambda: c.chat.completions.create(
        model=model, max_tokens=250, temperature=0,
        messages=_msg(imgs, COT_Q.format(n=len(imgs), caption=clip.get("caption", "")))))
    raw = r.choices[0].message.content or ""
    for cand in (raw, *re.findall(r"\{.*\}", raw, re.S)):
        try:
            d = json.loads(cand.strip().strip("`").removeprefix("json").strip())
            return (5.0 - float(np.clip(float(d["rating"]), 1, 5))) / 4.0
        except Exception:  # noqa: BLE001
            continue
    return None


SCORERS = {
    "binary":  lambda c, m, i, cl: score_binary(c, m, i, cl, caption=False),
    "caption": lambda c, m, i, cl: score_binary(c, m, i, cl, caption=True),
    "likert":  score_likert,
    "cot":     score_cot,
}


# ── metrics ───────────────────────────────────────────────────────────────────

def auc_extremes(scores, pcs, lo=2, hi=4):
    """P(model rates a pc<=lo clip more implausible than a pc>=hi clip)."""
    bad = [s for s, p in zip(scores, pcs) if s is not None and p <= lo]
    good = [s for s, p in zip(scores, pcs) if s is not None and p >= hi]
    if not bad or not good:
        return None, 0, 0
    w = sum((b > g) + 0.5 * (b == g) for b in bad for g in good)
    return w / (len(bad) * len(good)), len(bad), len(good)


def boot_ci(fn, n=2000, seed=0):
    rnd = random.Random(seed)
    out = [v for _ in range(n) if (v := fn(rnd)) is not None and v == v]
    if len(out) < n * 0.5:
        return None, None
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy300")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--order", default="temporal", choices=["temporal", "shuffled"])
    ap.add_argument("--prompt", default="binary", choices=list(SCORERS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    data = (ROOT / a.data) if not Path(a.data).is_absolute() else Path(a.data)
    meta = json.loads((data / "manifest.json").read_text())
    clips = meta["clips"][:a.limit] if a.limit else meta["clips"]
    c = client()
    scorer = SCORERS[a.prompt]

    print(f"\n=== {a.model} | prompt={a.prompt} | order={a.order} | {len(clips)} clips",
          flush=True)
    t0 = time.time()
    scores = [None] * len(clips)

    def work(i):
        try:
            imgs = load_frames(data, clips[i], a.order)
            scores[i] = scorer(c, a.model, imgs, clips[i])
        except Exception as e:  # noqa: BLE001
            print(f"    fail {clips[i]['clip_id'][:40]}: {str(e)[:60]}", file=sys.stderr)
        done = sum(1 for s in scores if s is not None)
        if done and done % 50 == 0:
            print(f"    {done}/{len(clips)} ({time.time()-t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, range(len(clips))))

    pcs = [cl["pc"] for cl in clips]
    impl = [6 - p for p in pcs]                       # higher = more implausible
    ok = [(s, im, p) for s, im, p in zip(scores, impl, pcs) if s is not None]
    if len(ok) < 10:
        sys.exit(f"only {len(ok)} clips scored — aborting")
    xs = [o[0] for o in ok]
    ys = [o[1] for o in ok]
    rho = spearman(xs, ys)
    lo, hi = boot_ci(lambda r: spearman(*zip(*[(xs[k], ys[k]) for k in
                     (r.randrange(len(xs)) for _ in xs)])))
    auc, n_bad, n_good = auc_extremes([o[0] for o in ok], [o[2] for o in ok])

    res = {"model": a.model, "prompt": a.prompt, "order": a.order,
           "n": len(clips), "n_scored": len(ok),
           "spearman": round(rho, 3),
           "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
           "auc_pc12_vs_pc45": round(auc, 3) if auc else None,
           "n_bad": n_bad, "n_good": n_good,
           "distinct_scores": len(set(xs)),
           "mean_score": round(float(np.mean(xs)), 3),
           "seconds": round(time.time() - t0, 1),
           "scores": {cl["clip_id"]: s for cl, s in zip(clips, scores)}}

    tag = a.tag or f"{a.model}_{a.prompt}_{a.order}"
    outp = data / f"scores_{tag}.json"
    outp.write_text(json.dumps(res, indent=1))

    ci = f"[{res['ci95'][0]:+.2f}, {res['ci95'][1]:+.2f}]" if res["ci95"] else "-"
    print(f"\n  rho vs human implausibility : {rho:+.3f}  {ci}")
    print(f"  AUC (pc<=2 vs pc>=4)        : {res['auc_pc12_vs_pc45']}  "
          f"(n={n_bad} bad / {n_good} good)")
    print(f"  distinct scores             : {res['distinct_scores']}/{len(ok)}")
    print(f"  mean score                  : {res['mean_score']}")
    print(f"  -> {outp}\n")


if __name__ == "__main__":
    main()
