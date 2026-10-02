"""
Per-specialist accuracy on any staged dataset, with honest error bars.

Reports TWO different tests, because they answer different questions and a single
"accuracy" number hides which one you got:

  ATTRIBUTION   positives = clips with THIS violation
                negatives = clips with a DIFFERENT violation
                -> can the specialist tell its own failure type from another's?
                This is the only test VideoPhy-2 could support, because it never
                labelled clean clips. 0.5 means "no better than guessing which
                kind of broken this is".

  DETECTION     positives = clips with THIS violation
                negatives = clips with NO violation at all
                -> can the specialist tell broken from clean?
                Needs a dataset that marks clean clips. Easier than attribution,
                and the number most people assume they are being shown.

Every AUC carries a bootstrap CI. At the sample sizes small datasets provide,
the CI is usually the whole story: an AUC of 0.72 with 11 positives is not a
result, and printing it without the interval invites a claim the data cannot
support. Rows whose CI covers 0.5 are marked `ns` (not significant).

Usage:
  python backend/scripts/specialist_accuracy.py --data data/teambench_staged
  python backend/scripts/specialist_accuracy.py --data data/teambench_staged \
      --probes domain_probes_winners.json --compare data/videophy1200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# probe key -> specialist. The winners battery prefixes with w_; keep the map
# explicit so a renamed probe fails loudly instead of silently scoring nothing.
PROBE_CAT = {
    "w_gravity": "gravity", "w_permanence": "permanence",
    "w_collision": "collision", "w_deformation": "deformation",
    "w_friction": "friction", "w_momentum": "momentum",
    "w_fluid": "fluid", "w_causality": "causality",
}


def _avg_rank(v):
    """Ranks with ties averaged. Plain argsort-of-argsort assigns tied values
    arbitrary distinct ranks in array order, which on this data is catastrophic:
    several probes return 0.0 for 60-99% of clips, and with positives listed
    first every tie broke against them, reporting AUC 0.000 for a probe that is
    actually constant (a constant score is AUC 0.5 by definition)."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1, dtype=float)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    r = _avg_rank(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def tie_frac(s):
    """Share of clips sitting on the single most common score. A probe that puts
    most of the dataset on one value cannot rank it, whatever its AUC says."""
    s = np.asarray(s, float)
    if len(s) == 0:
        return float("nan")
    _, cnt = np.unique(s, return_counts=True)
    return float(cnt.max() / len(s))


def spread(s):
    """Interdecile range (p90 - p10). Catches the OTHER degenerate mode.

    tie_frac only sees exact duplicates. gemma4 fails that way — 100% of clips
    on exactly 0.0. qwen2.5-vl-7b fails differently: every one of 226 values is
    distinct (tie_frac 0%) but they all sit between 0.749 and 0.925, an
    interdecile range of ~0.06. That is 226 distinct numbers carrying almost no
    decision, and a tie-based check calls it perfectly healthy.

    Rank-based AUC is scale-free, so a narrow band is not automatically useless —
    but combined with a near-chance AUC it means the probe is not committing to
    anything, and that is worth seeing next to the number.
    """
    s = np.asarray(s, float)
    if len(s) < 3:
        return float("nan")
    return float(np.percentile(s, 90) - np.percentile(s, 10))


def boot_ci(y, s, n_boot=4000, seed=0):
    """Stratified bootstrap: resample positives and negatives separately so a
    draw can never produce an empty class, which would silently drop samples and
    bias the interval inward."""
    y, s = np.asarray(y), np.asarray(s, float)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(ip) < 2 or len(ineg) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        a = rng.choice(ip, len(ip), replace=True)
        b = rng.choice(ineg, len(ineg), replace=True)
        idx = np.concatenate([a, b])
        out.append(auc(y[idx], s[idx]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def load(data: Path, probe_file: str):
    man = json.loads((data / "manifest.json").read_text())
    clips = {c["clip_id"]: c for c in man["clips"]}
    pf = data / probe_file
    if not pf.exists():
        sys.exit(f"ERROR: {pf} missing — run domain_probes.py --cats winners first")
    ps = json.loads(pf.read_text())
    return clips, ps.get("probe_scores", ps), ps.get("model", "?")


def cats_of(c):
    """Prefer the human category; fall back to the keyword taxonomy's output."""
    v = c.get("categories")
    return set(v) if v else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probes", default="domain_probes_winners.json")
    ap.add_argument("--compare", default=None,
                    help="a second staged dataset to print alongside, e.g. "
                         "data/videophy1200 — makes domain transfer visible")
    ap.add_argument("--min-pos", type=int, default=8,
                    help="below this many positives, do not print an AUC at all")
    a = ap.parse_args()

    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    clips, ps, model = load(data, a.probes)

    scored = [cid for cid in ps if cid in clips]
    broken = [c for c in scored if clips[c].get("has_violation") is not False]
    clean = [c for c in scored if clips[c].get("has_violation") is False]
    print(f"dataset: {data.name} | judge: {model} | {len(scored)} clips scored")
    print(f"  {len(broken)} with a violation, {len(clean)} clean")
    if not clean:
        print("  NOTE: no clean clips -> detection test unavailable, "
              "attribution only")

    rows = []
    for probe, cat in PROBE_CAT.items():
        if not any(probe in ps[c] for c in scored):
            continue
        pos = [c for c in broken if cat in cats_of(clips[c]) and probe in ps[c]]
        # attribution: negatives are OTHER violations
        neg_a = [c for c in broken if cat not in cats_of(clips[c]) and probe in ps[c]]
        # detection: negatives are clean clips
        neg_d = [c for c in clean if probe in ps[c]]

        def run(p, n):
            if len(p) < a.min_pos or len(n) < 2:
                return None
            y = [1] * len(p) + [0] * len(n)
            s = [ps[c][probe] for c in p] + [ps[c][probe] for c in n]
            lo, hi = boot_ci(y, s)
            return dict(auc=auc(y, s), lo=lo, hi=hi, npos=len(p),
                        nneg=len(n), tie=tie_frac(s))

        rows.append(dict(cat=cat, probe=probe, n_pos=len(pos),
                         attr=run(pos, neg_a), det=run(pos, neg_d)))

    rows.sort(key=lambda r: -(r["attr"]["auc"] if r["attr"] else -1))

    def fmt(d):
        if not d:
            return "  too few  ", ""
        # saturation is checked BEFORE significance: a probe sitting on one
        # value for most of the set is not ranking anything, and its AUC is
        # tie-breaking noise however tight the interval looks.
        if d["tie"] >= 0.75:
            note = f"  DEAD ({100*d['tie']:.0f}% one value)"
        elif d["tie"] >= 0.5:
            note = f"  saturated ({100*d['tie']:.0f}%)"
        elif d["lo"] > 0.5:
            note = ""
        else:
            note = "  ns"
        return f"{d['auc']:.3f} [{d['lo']:.2f},{d['hi']:.2f}]", note

    print(f"\n{'='*78}")
    print("ATTRIBUTION — this violation vs a DIFFERENT violation")
    print("=" * 78)
    print(f"{'specialist':13s} {'pos':>4s} {'neg':>4s}  {'AUC [95% CI]':>22s}  note")
    print("-" * 78)
    for r in rows:
        d = r["attr"]
        v, sig = fmt(d)
        n = f"{d['nneg']:>4d}" if d else "   -"
        print(f"{r['cat']:13s} {r['n_pos']:4d} {n}  {v:>22s}{sig}")

    if clean:
        print(f"\n{'='*78}")
        print("DETECTION — this violation vs CLEAN clips")
        print("=" * 78)
        print(f"{'specialist':13s} {'pos':>4s} {'neg':>4s}  {'AUC [95% CI]':>22s}  note")
        print("-" * 78)
        for r in rows:
            d = r["det"]
            v, sig = fmt(d)
            n = f"{d['nneg']:>4d}" if d else "   -"
            print(f"{r['cat']:13s} {r['n_pos']:4d} {n}  {v:>22s}{sig}")

    ok = [r for r in rows if r["attr"] and r["attr"]["tie"] < 0.75]
    if ok:
        m = float(np.mean([r["attr"]["auc"] for r in ok]))
        n_sig = sum(1 for r in ok if r["attr"]["lo"] > 0.5)
        print(f"\nmean attribution AUC over {len(ok)} live (non-dead) specialists: {m:.3f}")
        print(f"significantly above chance: {n_sig}/{len(ok)}")

    if a.compare:
        cd = Path(a.compare) if Path(a.compare).is_absolute() else ROOT / a.compare
        try:
            c2, p2, _ = load(cd, a.probes)
        except SystemExit:
            print(f"\n(compare skipped: {a.compare} has no {a.probes})")
            c2 = None
        if c2:
            print(f"\n{'='*78}")
            print(f"TRANSFER — {data.name} vs {cd.name} (attribution, same probes)")
            print("=" * 78)
            print(f"{'specialist':13s} {cd.name[:14]:>14s} {data.name[:16]:>16s}  {'delta':>8s}")
            print("-" * 78)
            s2 = [c for c in p2 if c in c2]
            for r in rows:
                pos = [c for c in s2 if r["cat"] in cats_of(c2[c]) and r["probe"] in p2[c]]
                neg = [c for c in s2 if r["cat"] not in cats_of(c2[c]) and r["probe"] in p2[c]]
                if len(pos) < a.min_pos or not neg:
                    continue
                y = [1] * len(pos) + [0] * len(neg)
                sc = [p2[c][r["probe"]] for c in pos] + [p2[c][r["probe"]] for c in neg]
                base = auc(y, sc)
                here = r["attr"]["auc"] if r["attr"] else float("nan")
                print(f"{r['cat']:13s} {base:14.3f} {here:16.3f}  {here-base:+8.3f}")

    # ── every specialist vs the OVERALL human rating, on every clip ──────────
    # Per-category AUC subsets to clips carrying that category, which on a small
    # dataset leaves 3-5 positives and nothing measurable. This section asks a
    # different question that every clip can answer: does this specialist's score
    # track how bad a human said the clip was? n = the whole dataset for all
    # eight, so the categories with too few positives still get a number here.
    #
    # Sign convention: the human rating is higher = MORE correct, the specialist
    # score is higher = MORE broken, so a working specialist gives NEGATIVE
    # Spearman. Reported flipped ("vs badness") so positive = working.
    import itertools
    rat = {c: clips[c].get("pc") for c in scored if clips[c].get("pc") is not None}
    if rat:
        def spear(x, y):
            return float(np.corrcoef(_avg_rank(x), _avg_rank(y))[0, 1])

        def spear_ci(x, y, n_boot=4000, seed=0):
            rng = np.random.default_rng(seed)
            v = [spear(np.take(x, i), np.take(y, i))
                 for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
            v = [z for z in v if np.isfinite(z)]
            # A constant probe makes every resample's correlation undefined, so v
            # comes back empty. Return NaN rather than letting percentile raise —
            # "no interval" is the correct answer for a probe with no variance.
            if not v:
                return (float("nan"), float("nan"))
            return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

        print(f"\n{'='*78}")
        print(f"ALL {len(rat)} CLIPS — each specialist vs the overall human rating")
        print("(positive rho = the specialist scores bad clips higher; "
              "no category subsetting)")
        print("=" * 78)
        print(f"{'specialist':13s} {'n':>4s} {'rho vs badness [95% CI]':>28s} "
              f"{'AUC bad-vs-good':>16s}  note")
        print("-" * 78)
        allrows = []
        cols = {}
        for probe, cat in PROBE_CAT.items():
            ids = [c for c in rat if probe in ps[c]]
            if len(ids) < 20:
                continue
            s = np.array([ps[c][probe] for c in ids], float)
            r = np.array([rat[c] for c in ids], float)
            cols[cat] = (ids, s)
            rho = -spear(s, r) if tie_frac(s) < 1.0 else float("nan")
            lo, hi = spear_ci(s, -r)
            # extreme split: clearly-bad clips vs clearly-good ones
            bad = [i for i, c in enumerate(ids) if rat[c] <= 2]
            good = [i for i, c in enumerate(ids) if rat[c] >= 4]
            bg = auc([1]*len(bad) + [0]*len(good),
                     list(s[bad]) + list(s[good])) if bad and good else float("nan")
            tf = tie_frac(s)
            note = (f"DEAD ({100*tf:.0f}% one value)" if tf >= 0.75 else
                    f"saturated ({100*tf:.0f}%)" if tf >= 0.5 else
                    "" if lo > 0 else "ns")
            print(f"{cat:13s} {len(ids):4d} {f'{rho:+.3f} [{lo:+.2f},{hi:+.2f}]':>28s} "
                  f"{bg:16.3f}  {note}")
            allrows.append(dict(cat=cat, n=len(ids), rho=rho, lo=lo, hi=hi,
                                auc_bad_good=bg, tie=tf))

        # does firing ANY specialist beat every single one of them?
        if len(cols) >= 2:
            ids0 = sorted(set.intersection(*[set(v[0]) for v in cols.values()]))
            probes_used = [p for p, c in PROBE_CAT.items() if c in cols]
            M = np.array([[ps[c][p] for c in ids0] for p in probes_used], float)
            r0 = np.array([rat[c] for c in ids0], float)
            print("-" * 78)
            for nm, agg in (("MAX over all", M.max(0)), ("MEAN over all", M.mean(0))):
                rho = -spear(agg, r0)
                lo, hi = spear_ci(agg, -r0)
                bad = [i for i, c in enumerate(ids0) if rat[c] <= 2]
                good = [i for i, c in enumerate(ids0) if rat[c] >= 4]
                bg = auc([1]*len(bad) + [0]*len(good),
                         list(agg[bad]) + list(agg[good])) if bad and good else float("nan")
                print(f"{nm:13s} {len(ids0):4d} {f'{rho:+.3f} [{lo:+.2f},{hi:+.2f}]':>28s} "
                      f"{bg:16.3f}  {'' if lo > 0 else 'ns'}")

        # full per-video matrix, so the raw scores can be eyeballed
        csv = data / "specialist_scores_per_video.csv"
        pk = [p for p in PROBE_CAT if any(p in ps[c] for c in scored)]
        with csv.open("w") as f:
            f.write("clip_id,pc,has_violation,categories," +
                    ",".join(PROBE_CAT[p] for p in pk) + "\n")
            for c in sorted(scored, key=lambda x: (clips[x].get("pc") or 0)):
                f.write(f'"{c}",{clips[c].get("pc")},'
                        f'{clips[c].get("has_violation")},'
                        f'"{"|".join(clips[c].get("categories") or [])}",'
                        + ",".join(f'{ps[c].get(p, float("nan")):.4f}' for p in pk) + "\n")
        print(f"\nper-video matrix -> {csv}")

    # ── scoring-mode comparison: token probability vs the literal digit ──────
    # Same 640 calls, two readings of the same distribution:
    #   token-prob  1 - P("1"), a continuous value using the whole distribution
    #   discrete    argmax digit, i.e. what greedy decoding would print
    # The discrete reading is what you get from any API that returns text only,
    # so this measures what the continuous score is actually buying us.
    raw = json.loads((data / a.probes).read_text())
    ans = raw.get("probe_answers")
    if ans and rat:
        print(f"\n{'='*78}")
        print("FINAL — token probability vs discrete answer (same 640 calls)")
        print("=" * 78)
        print(f"{'specialist':13s} {'pos':>4s} | {'ATTRIBUTION AUC':^21s} | "
              f"{'CLIP-LEVEL rho':^19s} | {'levels':>7s}")
        print(f"{'':13s} {'':>4s} | {'tok-prob':>10s} {'digit':>10s} | "
              f"{'tok-prob':>9s} {'digit':>9s} | {'tok/dig':>7s}")
        print("-" * 78)
        final = []
        for probe, cat in PROBE_CAT.items():
            ids = [c for c in rat if probe in ps.get(c, {}) and probe in ans.get(c, {})]
            if len(ids) < 20:
                continue
            sv = np.array([ps[c][probe] for c in ids], float)
            dv = np.array([float(ans[c][probe]) for c in ids], float)
            r = np.array([rat[c] for c in ids], float)

            pos = [i for i, c in enumerate(ids)
                   if cat in cats_of(clips[c])
                   and clips[c].get("has_violation") is not False]
            neg = [i for i, c in enumerate(ids)
                   if cat not in cats_of(clips[c])
                   and clips[c].get("has_violation") is not False]

            def a_auc(v):
                if len(pos) < a.min_pos or len(neg) < 2:
                    return float("nan")
                return auc([1]*len(pos) + [0]*len(neg),
                           list(v[pos]) + list(v[neg]))

            def a_rho(v):
                return -spear(v, r) if tie_frac(v) < 1.0 else float("nan")

            row = dict(cat=cat, n_pos=len(pos),
                       auc_tok=a_auc(sv), auc_dig=a_auc(dv),
                       rho_tok=a_rho(sv), rho_dig=a_rho(dv),
                       lv_tok=len(np.unique(sv)), lv_dig=len(np.unique(dv)))
            final.append(row)

            def f(x):
                return "   —  " if not np.isfinite(x) else f"{x:.3f}"
            print(f"{cat:13s} {len(pos):4d} | {f(row['auc_tok']):>10s} "
                  f"{f(row['auc_dig']):>10s} | {f(row['rho_tok']):>9s} "
                  f"{f(row['rho_dig']):>9s} | "
                  f"{row['lv_tok']:>3d}/{row['lv_dig']:<3d}")
        print("-" * 78)
        for key, lab in (("auc", "attribution AUC"), ("rho", "clip-level rho")):
            t = [r[f"{key}_tok"] for r in final if np.isfinite(r[f"{key}_tok"])
                 and np.isfinite(r[f"{key}_dig"])]
            d = [r[f"{key}_dig"] for r in final if np.isfinite(r[f"{key}_tok"])
                 and np.isfinite(r[f"{key}_dig"])]
            if t:
                print(f"mean {lab:16s} tok-prob {np.mean(t):+.3f}   "
                      f"digit {np.mean(d):+.3f}   delta {np.mean(t)-np.mean(d):+.3f}"
                      f"   (n={len(t)})")
        # how often does the model literally say "1" (= nothing wrong)?
        flat = [v for c in ans for v in ans[c].values()]
        from collections import Counter as _C
        cc = _C(flat)
        print("\ndigit the model actually emits, over all "
              f"{len(flat)} calls: "
              + "  ".join(f"{k}:{cc.get(k,0)}" for k in (1, 2, 3, 4, 5)))
        json.dump(final, (data / "scoring_mode_comparison.json").open("w"), indent=1)

    out = data / "specialist_accuracy.json"
    out.write_text(json.dumps({"model": model, "n_scored": len(scored),
                               "n_clean": len(clean), "rows": rows}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
