"""
Domain-knowledge probe batteries — what a physicist would actually check.

WHY THIS IS NOT A REPEAT. The earlier probes asked ONE abstract question per
category ("is any object unsupported yet not falling"). Five attempts to lift
the weak specialists failed, and a ceiling test over 55 engineered signals plus
768 embedding dims put gravity at 0.516. But that ceiling only covers signals
already measured — a battery of NEW, specific questions is genuinely new
measurement, not a recombination of the old ones.

The change is decomposition plus concreteness. Instead of "does gravity look
wrong", ask the individual things that make it wrong:

  - free fall must ACCELERATE (later gaps bigger than earlier gaps)
  - an unsupported object must not hold its height
  - nothing rises without a visible driver
  - landings dissipate: bounce, squash, or settle — never a dead stop
  - support must be plausible: not air, not something far too weak

Each is a separate yes/no with its own token-probability score, so a category
becomes a 5-dimensional measurement instead of a scalar. Weak-but-real
sub-signals can then combine even when the holistic question is uninformative.

Run over the 704 rule-annotated clips only — that set IS the discriminative
comparison (positives plus other-category violators), so scoring the other 496
clean clips would spend budget without sharpening any AUC.

Usage:
  python backend/scripts/domain_probes.py --cats gravity,momentum
  python backend/scripts/domain_probes.py --cats gravity,momentum,deformation,permanence
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from videophy_eval import _token_probs, client        # noqa: E402
from physics_probes import frames                     # noqa: E402
import base64 as _b64
import numpy as _np


def frames_ordered(data, clip, k=8, order="temporal"):
    """k frames in the frozen temporal OR shuffled ordering.

    The shuffled ordering is the one fixed at staging time, so the control is
    reproducible and identical to the ablation used on the holistic score.
    """
    if order == "temporal":
        return frames(data, clip, k)
    idx = clip["order_shuffled"]
    if k < len(idx):
        idx = [idx[i] for i in _np.linspace(0, len(idx) - 1, k).astype(int)]
    d = data / "frames" / clip["clip_id"]
    out = []
    for i in idx:
        f = d / f"{i:03d}.jpg"
        if f.exists():
            out.append("data:image/jpeg;base64," + _b64.b64encode(f.read_bytes()).decode())
    return out
from gepa_optimize import load                        # noqa: E402

MODEL = "gemma4-31b-it"
_P = ("Look at these {n} frames, sampled in order from a video of: \"{caption}\".\n")

# DEGREE, not yes/no. These questions are specific enough that the model answers
# "No" confidently and "Yes" falls outside the top-20 logprobs entirely — so
# p.get("yes", 0.0) returns the default and EVERY clip ties at exactly 0.0, with
# no ranking information whatever. Verified on a 3-clip smoke run: 13 of 15
# sub-scores were exactly 0.0. A 1-5 degree scale spreads the mass over five
# tokens that are all in-distribution, which is the same fix that gave the
# likert prompt 300/300 distinct values.
_A = ("\nHow clearly does this happen in these frames?\n"
      "1 = not at all, 2 = barely, 3 = somewhat, 4 = clearly, 5 = blatantly.\n"
      "Answer with exactly one digit, 1 to 5.")

# "Yes" always means THE DEFECT IS PRESENT, so sub-scores combine without sign
# bookkeeping.
BATTERIES = {
    "gravity": [
        ("g_no_accel",
         "Find anything falling. Real falling speeds UP: the gap it moves "
         "between later frames must be LARGER than between earlier frames.\n"
         "Does something fall at a constant speed, or drift down too slowly, "
         "instead of accelerating?"),
        ("g_holds_height",
         "Is there an object in the air that keeps the SAME height across "
         "several frames, when nothing is holding it up?"),
        ("g_rises",
         "Does anything move upward without a visible cause — no throw, no jet, "
         "no lift, no one pushing it?"),
        ("g_dead_landing",
         "When something lands or comes to rest, does it stop dead — with no "
         "bounce, no squash, no wobble, no settling of any kind?"),
        ("g_bad_support",
         "Is anything resting on empty space, or held up by something far too "
         "thin or weak to support its weight?"),
    ],
    "momentum": [
        ("m_gains_energy",
         "After a hit, bounce or collision, does anything end up moving FASTER "
         "than it was before? A bounce must always return less speed."),
        ("m_stops_dead",
         "Does a moving object stop suddenly without hitting anything and "
         "without slowing down first?"),
        ("m_no_recoil",
         "When one object strikes another, does the STRIKING object continue "
         "as if nothing happened — no slowing, no bounce back, no recoil?"),
        ("m_midair_turn",
         "Does anything change direction while in mid-air, with nothing "
         "touching it?"),
        ("m_mass_mismatch",
         "Does a heavy object get thrown or deflected by something much "
         "lighter, or a light object shrug off a heavy impact?"),
    ],
    "deformation": [
        ("d_shape_drift",
         "Does a solid, rigid object slowly change its shape, length or "
         "proportions across the frames while nothing is squeezing it?"),
        ("d_bend_no_force",
         "Does something bend, twist or fold where no force is being applied "
         "to bend it?"),
        ("d_no_deform_on_impact",
         "At a hard impact, do the objects stay perfectly rigid — no squash, "
         "dent, compression or give at all, where real materials would show it?"),
        ("d_merge",
         "Do two separate objects blend, fuse or pass into one another as if "
         "they were made of liquid?"),
        ("d_texture_swim",
         "Does the surface pattern, print or texture on an object slide around "
         "or crawl across it instead of staying fixed to the surface?"),
    ],
    "permanence": [
        ("p_vanish",
         "Does any object, limb or body part disappear between frames?"),
        ("p_appear",
         "Does any object, limb or body part appear out of nowhere?"),
        ("p_count_change",
         "Does the NUMBER of a repeated thing change — fingers, legs, wheels, "
         "people, objects in a set?"),
        ("p_identity_swap",
         "Does something turn into a different object, or swap its identity, "
         "between frames?"),
        ("p_occlusion_fail",
         "When something passes behind another object, does it fail to come "
         "back out correctly — wrong place, wrong shape, or not at all?"),
    ],
    "collision": [
        ("c_gap_at_impact",
         "At the moment of contact, is there still a visible GAP between the "
         "two objects?"),
        ("c_early_reaction",
         "Does the target start reacting BEFORE it is actually touched?"),
        ("c_pass_through",
         "Does anything pass through a solid object instead of being stopped "
         "by it?"),
        ("c_no_contact_effect",
         "Does the intended effect happen even though the two things never "
         "visibly meet?"),
    ],
    "friction": [
        ("f_slides_forever",
         "Does something slide or roll along a surface without slowing down at "
         "all?"),
        ("f_grip_fail",
         "Does something slip on a surface where it should grip, or grip where "
         "it should slip?"),
        ("f_no_rolling_link",
         "Does a wheel or ball move across the ground without its spin matching "
         "how far it travels — sliding rather than rolling?"),
    ],
    "fluid": [
        ("fl_vanish",
         "Does liquid, spray or smoke appear or vanish instantly instead of "
         "flowing, draining or dispersing?"),
        ("fl_shape",
         "Does a fluid hold a shape it could not hold — a rigid blob, a frozen "
         "splash, a wall of water standing up?"),
        ("fl_no_splash",
         "Does something enter or strike liquid without the splash, ripple or "
         "disturbance it should cause?"),
    ],
}

# ── v2 batteries ──────────────────────────────────────────────────────────────
# Designed from what actually separated in v1. Across all 30 sub-checks, the
# ones that worked asked the model to COMPARE two things it can see, or COUNT
# something discrete:
#     g_no_accel     0.661  compare gap sizes between frames
#     p_count_change 0.665  count a repeated thing
# The ones at chance asked it to judge an ABSENCE or ATTRIBUTE A CAUSE:
#     g_rises        0.498  "without a visible cause"
#     d_no_deform    0.497  "no squash where real materials would show it"
#     m_no_recoil    0.458  "as if nothing happened"
#     c_gap_at_impact 0.491 requires precise timing perception
# momentum, deformation and collision were written entirely in the failing
# style. These rewrite them as comparisons and counts.
BATTERIES_V2 = {
    "momentum_v2": [
        ("m2_faster_after",
         "Compare how far the moving object travels between frames BEFORE the "
         "hit versus AFTER it. Does it cover MORE distance per frame after the "
         "hit than before?"),
        ("m2_speed_jump",
         "Compare the spacing of the object between each pair of frames. Is "
         "there one place where the spacing suddenly jumps much larger, with no "
         "other object touching it at that point?"),
        ("m2_stops_early",
         "Compare the object's spacing across the last few frames. Does the "
         "spacing go from large straight to zero in a single step, rather than "
         "shrinking gradually?"),
        ("m2_both_move",
         "After two things meet, compare how much EACH of them moves. Does only "
         "one of them change its motion while the other keeps going exactly as "
         "before?"),
    ],
    "deformation_v2": [
        ("d2_length_change",
         "Pick one solid object. Compare its length end-to-end in the first "
         "frame and in the last frame. Has that length visibly changed?"),
        ("d2_straight_edges",
         "Count the straight edges or sharp corners on the main solid object in "
         "the first frame, then count them again in the last frame. Do the "
         "counts differ?"),
        ("d2_proportion",
         "Compare the width-to-height proportion of the main object between the "
         "first and last frame. Has that proportion changed?"),
        ("d2_impact_squash",
         "Compare the object's shape in the frame just before contact and the "
         "frame just after. For a hard impact, is it EXACTLY the same shape?"),
    ],
    "collision_v2": [
        ("c2_gap_size",
         "At the frame where the two objects are closest, compare the gap "
         "between them to the size of the smaller object. Is the gap a "
         "noticeable fraction of that object rather than zero?"),
        ("c2_order",
         "Compare WHEN the target starts moving to WHEN the two objects touch. "
         "Does the target start moving in an EARLIER frame than the touch?"),
        ("c2_overlap",
         "At the closest frame, do the two objects visibly OVERLAP — occupying "
         "the same space — rather than meeting at their surfaces?"),
        ("c2_count_contacts",
         "Count how many times the two objects actually touch across these "
         "frames. Is that count ZERO even though an effect clearly happens?"),
    ],
}
BATTERIES.update(BATTERIES_V2)

# The single best-discriminating question per category, from the v1/v2 sweep.
# Fluid reached 0.944 by fusing the SAME question across three VLMs, while
# fusing seven models on the vague holistic question gained nothing — a specific
# question makes model noise independent, a vague one leaves shared bias. These
# winners were only ever run on gemma4, so the ensemble trick is untested
# exactly where it is most needed.
BATTERIES["winners"] = [
    ("w_gravity",     dict(BATTERIES["gravity"])["g_no_accel"]),
    ("w_permanence",  dict(BATTERIES["permanence"])["p_count_change"]),
    ("w_collision",   dict(BATTERIES["collision_v2"])["c2_overlap"]),
    ("w_deformation", dict(BATTERIES["deformation_v2"])["d2_length_change"]),
    ("w_friction",    dict(BATTERIES["friction"])["f_no_rolling_link"]),
    ("w_momentum",    dict(BATTERIES["momentum"])["m_gains_energy"]),
    # fl_shape won the fluid sweep on VideoPhy-2 at 0.867 (vs fl_vanish 0.846,
    # fl_no_splash 0.812). Added here so `winners` covers all seven; the choice
    # was made on VideoPhy-2, so it stays pre-registered w.r.t. any new dataset.
    ("w_fluid",       dict(BATTERIES["fluid"])["fl_shape"]),
]

# Causality had no probe: VideoPhy-2's rule text almost never described one, so
# the category never appeared in our taxonomy counts. Robot-manipulation data is
# full of it — "rag moves across the table without the arm actually moving it".
#
# NOT PRE-REGISTERED. Written after seeing this dataset's category list (not its
# scores), so its number is a first look, not a held-out result. Worded as
# "what do you see" rather than "what caused what" on purpose: across 30
# sub-checks the questions that worked asked the model to compare or count
# something visible, and the ones asking about cause or timing did not.
BATTERIES["causality"] = [
    ("cz_uncaused_motion",
     "Does an object move, change shape or change state on its own, without "
     "anything visibly touching it or acting on it?"),
    ("cz_no_effect",
     "Does something make contact with an object without producing the effect "
     "that contact should have caused?"),
]
BATTERIES["winners"].append(
    ("w_causality", dict(BATTERIES["causality"])["cz_uncaused_motion"]))

# ── robot-manipulation rewrite of the same eight ──────────────────────────────
# The `winners` set was tuned on VideoPhy-2, where failures are loud: a ball
# through a wall, a body folding in half. On robot data five of eight probes
# returned one constant value for ~every clip — the model answers "nothing
# wrong" because nothing dramatic happens.
#
# These are rewritten from the annotators' own 124 issue descriptions, whose
# vocabulary is overwhelmingly about CONTACT and APPEARANCE, not rates:
# "moves on its own no contact" (x11), "appears out of nowhere" (x9),
# "deforms/grew/morphs" (x20), "grasp without touching" (x6).
#
# Same design rule that separated the live probes from the dead ones: ask the
# model to COMPARE two named frames or COUNT something visible. Never ask about
# speed, rate, or what caused what — those are exactly the probes that flatlined,
# and the shuffle control says the model cannot read frame order anyway.
# Deliberately mentions the gripper/hand, because in every one of these clips the
# only legitimate cause of motion is the robot touching something.
BATTERIES["robot"] = [
    ("w_gravity",
     "Look for an object that is resting on nothing — not on a surface, not "
     "held by the gripper or hand.\n"
     "Is such an object hanging in the air across several frames instead of "
     "dropping to the surface below it?"),
    ("w_permanence",
     "Count the separate objects on the work surface in the FIRST frame, then "
     "count them again in the LAST frame.\n"
     "Is an object present in one of those frames and simply absent in the "
     "other, with no hand having carried it away?"),
    ("w_collision",
     "Look at the gripper or hand and the object it is working on, at the frame "
     "where they are closest.\n"
     "Is there still a visible GAP between them, or do they overlap into the "
     "same space, rather than meeting cleanly at their surfaces?"),
    ("w_deformation",
     "Pick one rigid object — a bottle, cup, tool, box or the gripper itself.\n"
     "Compare its outline in the first frame and in the last frame. Has its "
     "shape, length or thickness visibly changed?"),
    ("w_friction",
     "Look for an object sliding across the surface.\n"
     "Does it keep sliding while nothing is pushing it, or slide underneath a "
     "gripper that is holding it still?"),
    ("w_momentum",
     "Find an object the hand or gripper has let go of, or knocked.\n"
     "Compare how far it moves between frames just before and just after that "
     "moment. Does it visibly travel FARTHER per frame afterwards?"),
    ("w_fluid",
     "Look at any liquid, spray, foam or smoke.\n"
     "Compare its amount and its outline between frames. Does it appear from "
     "nowhere, vanish, or hold a stiff unmoving shape?"),
    ("w_causality",
     "Find an object that changes position between two frames.\n"
     "In those frames, are the gripper and both hands clearly somewhere else — "
     "not touching it — so that nothing visible moved it?"),
]

# ── Stage-1/2 evidence injected into the prompt ───────────────────────────────
# Stages 1 and 2 already measure motion, tracks and where a clip goes wrong, and
# none of it currently reaches Stage 3. This turns those numbers into a short
# factual block the VLM can read.
#
# Values are expressed as DATASET PERCENTILES, not raw units. "0.83 px/frame of
# residual motion" is meaningless to a language model and unanchored across
# datasets; "more motion than 90% of clips" is a judgement it can actually use.
# Only signals above/below a decisive percentile are mentioned at all — listing
# every signal on every clip would make the block constant, and a constant
# preamble carries no information while still costing tokens and attention.
INJECT_TEMPLATES = {
    "s1_obj_motion":       ("the objects move much more than usual",
                            "the objects barely move"),
    "s2_accel_p95":        ("motion speeds up and slows down far more sharply "
                            "than usual", "motion is unusually steady"),
    "s2_jerk_p95":         ("movement is unusually jerky",
                            "movement is unusually smooth"),
    "s2_inner_death_frac": ("many tracked points disappear in the middle of the "
                            "frame, away from any edge", ""),
    "s2_track_survival":   ("", "most tracked points are lost before the end"),
    "sp_deform_spread":    ("the outline of the tracked object changes size a "
                            "lot", ""),
    "sp_deform_drift":     ("object outlines drift and wobble more than usual",
                            ""),
    "sp_momentum_gain":    ("something moves faster after an interaction than "
                            "before it", ""),
    "sp_reversals":        ("tracked points reverse direction unusually often",
                            ""),
    "sp_gravity_flat":     ("downward motion is unusually steady rather than "
                            "speeding up", ""),
    "sp_min_approach":     ("two groups of tracked points come unusually close "
                            "together", ""),
    "s1_cam_frac":         ("most of the motion is the camera moving, not the "
                            "scene", ""),
    "s1_flow_entropy":     ("motion directions are unusually disordered", ""),
}
HI, LO = 85.0, 15.0


def build_injection(cid, sigstats):
    """One short evidence block for a clip, or '' if nothing stands out."""
    if not sigstats:
        return ""
    pct, bits = sigstats.get(cid) or {}, []
    for k, (hi_txt, lo_txt) in INJECT_TEMPLATES.items():
        p = pct.get(k)
        if p is None:
            continue
        if p >= HI and hi_txt:
            bits.append(hi_txt)
        elif p <= LO and lo_txt:
            bits.append(lo_txt)
    if not bits:
        return ""
    if len(bits) > 4:
        bits = bits[:4]
    return ("Automated motion analysis of this clip reports that "
            + "; ".join(bits) + ".\n")


def percentile_table(sig_path):
    """signal -> per-clip percentile within this dataset."""
    d = json.loads(Path(sig_path).read_text())
    sig, keys = d["signals"], d["keys"]
    ids = [c for c in sig if sig[c]]
    out = {c: {} for c in ids}
    for k in keys:
        v = np.array([sig[c].get(k, 0.0) for c in ids], float)
        order = v.argsort().argsort()
        p = 100.0 * order / max(len(v) - 1, 1)
        for c, x in zip(ids, p):
            out[c][k] = float(x)
    return out


def caption_for(clip, mode="full"):
    """Caption variants, cut from the UNTRUNCATED caption.

    Every run before this one effectively used `scene`: staging capped the
    caption at 400 chars and the prompt cut it again at 200, so the "Action:"
    line — the only part saying what was supposed to HAPPEN — reached the model
    on almost no clip. The specialists were told what the scene looked like and
    never what it was meant to do.

    scene  = appearance only. A specialist that scores well on this alone is
             probably reading render quality, not physics.
    action = the intended event, from the generation prompt.
    task   = the robot's goal, one line, from the source dataset.
    """
    full = (clip.get("caption_full") or clip.get("caption") or "")
    if mode == "none":
        return ""
    if mode == "task":
        return (clip.get("task") or "")[:200]
    m_s = re.search(r"Scene:\s*(.*?)(?:\n\s*Action:|$)", full, re.S)
    m_a = re.search(r"Action:\s*(.*)$", full, re.S)
    scene = (m_s.group(1).strip() if m_s else full).strip()
    action = (m_a.group(1).strip() if m_a else "").strip()
    if mode == "scene":
        return scene[:280]
    if mode == "action":
        return action[:200] or (clip.get("task") or "")[:200]
    # full: keep the action even when the scene is long — the action is the
    # short, high-value half and truncation used to delete exactly it
    return (scene[:260] + (". " + action[:200] if action else ""))[:480]


def score_clip(c, data, clip, probes, model=MODEL, caption=True,
               order="temporal", nframes=8, sigstats=None, capmode="full"):
    imgs = frames_ordered(data, clip, nframes, order)
    cap = caption_for(clip, capmode) if caption else ""
    if cap:
        head = _P.format(n=len(imgs), caption=cap)
    else:
        # Caption-free control: every domain question embeds the clip caption,
        # so a detector could be scoring the PROMPT TEXT rather than the pixels.
        head = (f"Look at these {len(imgs)} frames, sampled in order from a "
                "video.\n")
    if sigstats is not None:
        head += build_injection(clip["clip_id"], sigstats)
    out, ans = {}, {}
    for name, q in probes:
        try:
            p = _token_probs(c, model, imgs, head + q + _A)
            # len(k)==1 guard: "" is a substring of every string, and the gateway
            # does emit empty tokens, so a bare `k in "12345"` lets int("") raise.
            mass = {int(k): v for k, v in p.items() if len(k) == 1 and k in "12345"}
            if mass:
                # The DISCRETE answer: the digit the model would actually emit
                # under greedy decoding, i.e. argmax of this same distribution.
                # Stored alongside the continuous score so the two readings of
                # one identical call can be compared without re-querying.
                ans[name] = int(max(mass.items(), key=lambda kv: kv[1])[0])
            if sum(mass.values()) > 1e-4:
                # Score = 1 - P("1"), i.e. P(the defect is present to ANY degree).
                # Not the expected value over digits: when the model is confident
                # the answer is "1", digits 2-5 fall outside the top-20 logprobs,
                # the EV comes back exactly 1.0, and the clip pins to 0.0 with no
                # gradation (55% of sub-scores did this on a smoke run). P("1") is
                # a true probability from the raw distribution, so 1 - P("1") stays
                # precise however hard the tail is truncated, and it is monotonic
                # in "how much defect the model sees" — which is all AUC needs.
                out[name] = float(min(max(1.0 - p.get("1", 0.0), 0.0), 1.0))
        except Exception as e:  # noqa: BLE001
            print(f"    {name} fail {clip['clip_id'][:26]}: {str(e)[:45]}",
                  file=sys.stderr)
    return {"scores": out, "answers": ans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--cats", default="gravity,momentum,deformation,permanence")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default=MODEL,
                    help="gateway model id; the question set is identical across "
                         "models so the ensemble stays apples-to-apples")
    ap.add_argument("--inject", default=None,
                    help="stage_signals.json — inject a Stage-1/2 evidence "
                         "block into every prompt")
    ap.add_argument("--all-clips", action="store_true",
                    help="also score clips with no rule text — enables the "
                         "clean-vs-broken detection test, not just attribution")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--capmode", default="full",
                    choices=["full", "scene", "action", "task", "none"],
                    help="which part of the caption to show")
    ap.add_argument("--no-caption", action="store_true",
                    help="strip the caption from the prompt (ablation)")
    ap.add_argument("--order", default="temporal",
                    choices=["temporal", "shuffled"],
                    help="frame order; shuffled is the motion-blindness control")
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()

    data, clips = load(a.data)
    # rule-annotated clips only: that is exactly the discriminative set
    if not a.all_clips:
        # default: rule-annotated clips only, so the negatives are "a clip with a
        # DIFFERENT violation" — the attribution test.
        clips = [c for c in clips if len(c.get("violated_rules") or "") > 4]
    else:
        # keep unannotated clips too. On a dataset that marks clean clips, this
        # makes the same run support the detection test as well (this violation
        # vs no violation at all), which the rule-text filter silently prevents.
        print("   --all-clips: clean clips kept as negatives")
    if a.limit:
        clips = clips[:a.limit]

    cats = [x.strip() for x in a.cats.split(",") if x.strip() in BATTERIES]
    probes = [(nm, q) for cat in cats for nm, q in BATTERIES[cat]]
    print(f"domain probes: {len(cats)} categories, {len(probes)} sub-probes, "
          f"{len(clips)} clips = {len(probes)*len(clips)} calls", flush=True)

    SIG = percentile_table(data / a.inject) if a.inject else None
    if SIG:
        n_any = sum(1 for cid in SIG if build_injection(cid, SIG))
        print(f"   --inject: evidence block on {n_any}/{len(SIG)} clips")

    c = client()
    t0 = time.time()
    res = [None] * len(clips)
    done = [0]

    def work(i):
        res[i] = score_clip(c, data, clips[i], probes, a.model,
                            caption=not a.no_caption, order=a.order,
                            nframes=a.frames, sigstats=SIG,
                            capmode=a.capmode)
        done[0] += 1
        if done[0] % 50 == 0:
            el = time.time() - t0
            print(f"    {done[0]}/{len(clips)} ({el:.0f}s, eta "
                  f"{el/done[0]*(len(clips)-done[0])/60:.0f}m)", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, range(len(clips))))

    out = {cl["clip_id"]: r["scores"] for cl, r in zip(clips, res) if r and r["scores"]}
    answers = {cl["clip_id"]: r["answers"] for cl, r in zip(clips, res) if r and r["answers"]}
    tag = '_'.join(cats) + ('' if a.model == MODEL else f"__{a.model}")
    if a.no_caption:
        tag += "__nocap"
    if a.order != "temporal":
        tag += f"__{a.order}"
    if a.frames != 8:
        tag += f"__f{a.frames}"
    if a.inject:
        tag += "__inject"
    if a.capmode != "full":
        tag += f"__cap{a.capmode}"
    outp = data / f"domain_probes_{tag}.json"
    # a.model, NOT the module default. Writing the default here mislabelled every
    # non-default run: domain_probes_winners__qwen3-vl-32b-instruct.json recorded
    # "model": "gemma4-31b-it" while actually holding qwen3-vl scores, so the
    # filename and the provenance field disagreed and only the filename was right.
    outp.write_text(json.dumps({"model": a.model, "cats": cats, "n": len(out),
                                "probe_scores": out,
                                "probe_answers": answers}, indent=1))
    print(f"\n  {len(out)}/{len(clips)} clips ({time.time()-t0:.0f}s)")
    print(f"  -> {outp}")

    # Fail loudly on a mostly-empty run. This silently produced six 0-cell files
    # that exited 0 and looked "complete" in 2.5 minutes: the gateway had begun
    # rejecting 8-image prompts with
    #   "At most 4 image(s) may be provided in one prompt"
    # on some backends of the gemma4-31b-it model group but not others, so the
    # failure was partial, intermittent, and invisible in the exit status.
    cells = sum(len(v) for v in out.values())
    want = len(clips) * len(probes)
    if want and cells < 0.5 * want:
        print(f"\nERROR: only {cells}/{want} cells ({100*cells/want:.0f}%) were "
              f"scored — refusing to report this as a completed condition.\n"
              f"       Check the log for the failure mode; if it is the image "
              f"cap, re-run with --frames 4.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
