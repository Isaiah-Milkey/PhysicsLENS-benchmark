"""
Full-framework evaluation: run every implemented pipeline, in stage order, on
every video under test_videos/, saving each pipeline's event stream (heavy
payloads stripped) plus a per-video summary. Aggregate afterwards with
--aggregate to get real-vs-AI separation per pipeline.

All pipelines run in ONE process, in stage order, so the in-process evidence
bus (stage 2 → 3 → 4) works and the local Qwen VLM loads once (shared by
tracker naming, s1_vlm and s3_causality).

Usage (from repo root, physics-lens env, CUDA_VISIBLE_DEVICES pinned):
  python backend/scripts/eval_framework.py --out eval_reports/2026-07-23
  python backend/scripts/eval_framework.py --aggregate --out eval_reports/2026-07-23
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif"}

ORDER = [
    "s1_temporal", "s1_optical_flow", "s1_embeddings", "s1_camera_motion",
    "s1_vlm",
    "s2_object_tracker", "s2_trajectory_extractor", "s2_event_localizer",
    "s2_hypothesis_generator",
    "s3_collision", "s3_gravity", "s3_momentum", "s3_friction",
    "s3_deformation", "s3_fluid", "s3_causality",
    "s4_report",
]

# Eval-mode overrides: skip rendered overlay videos (time + useless in saved
# streams); gravity deps run explicitly in stage order, so no planner call.
# All VLM calls route to the strongest OpenAI model (user choice 2026-07-23:
# "bigger VLMs via API wherever possible" — the run this config originally
# produced used a Gemini model via a since-removed proxy; re-running today
# uses OpenAI instead, so treat historical numbers as not directly
# reproducible bit-for-bit) — keeps the local 17 GB Qwen off the GPU entirely.
# Exception: s3_causality NEEDS local Yes/No token logits, so it runs as a
# separate pass (--only s3_causality) when VRAM allows.
_PRO = "openai:gpt-4o"
OVERRIDES = {
    "s1_vlm":                  {"model": _PRO},
    "s2_object_tracker":       {"render_video": "false", "naming_model": _PRO},
    "s2_trajectory_extractor": {"render_video": "false"},
    "s2_hypothesis_generator": {"model": _PRO},
    "s3_collision":            {"model": _PRO},
    "s3_gravity":              {"auto_deps": "off"},          # model already the strong tier
    "s3_momentum":             {"model": _PRO},
    "s3_friction":             {"model": _PRO},
    "s3_deformation":          {"model": _PRO},
    "s3_fluid":                {"model": _PRO},
    "s4_report":               {"summary_model": "gpt-4o"},
}

PIPELINE_TIMEOUT_S = 1800

_STAGES = {"s1": (1, "Stage 1 — Screening"),
           "s2": (2, "Stage 2 — Differential Diagnosis"),
           "s3": (3, "Stage 3 — Specialist Evaluation"),
           "s4": (4, "Stage 4 — Final Diagnosis")}


def prev_results(summary: dict) -> list[dict]:
    """Per-test results in the shape s4_report's `previous_results` expects
    (mirrors what the frontend passes after running earlier stages)."""
    out = []
    for pid, p in summary["pipelines"].items():
        if pid == "s4_report":
            continue
        num, name = _STAGES[pid[:2]]
        out.append({"id": pid, "pipelineId": pid, "stageId": pid[:2],
                    "stageNum": num, "stageName": name,
                    "status": "done" if p["status"] == "ok" else p["status"],
                    "severities": p["severities"], "metrics": p["metrics"],
                    "logs": p.get("logs", [])})
    return out


# Set by main() so a run can point at any staged video set (e.g. the Rapidata
# Sora clips) without a second harness.
VIDEO_ROOT = ROOT / "test_videos"


def find_videos():
    vids = [p for p in sorted(VIDEO_ROOT.rglob("*"))
            if p.suffix.lower() in VIDEO_EXTS]
    return [(p, "real" if "real" in p.relative_to(VIDEO_ROOT).parts[:1]
             else "ai") for p in vids]


def slug(video: Path) -> str:
    return "__".join(video.relative_to(VIDEO_ROOT).with_suffix("").parts)


def slim(obj, limit=2048):
    """Recursively truncate long strings (base64 images/videos/plotly blobs)."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else f"<stripped {len(obj)} chars>"
    if isinstance(obj, dict):
        return {k: slim(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [slim(v, limit) for v in obj]
    return obj


async def run_pipeline(run_fn, video: Path, settings: dict | None, out_path: Path):
    events, status, err = [], "ok", ""
    t0 = time.time()

    async def consume():
        async for ev in run_fn(str(video), json.dumps(settings) if settings else None):
            events.append(ev)

    try:
        await asyncio.wait_for(consume(), PIPELINE_TIMEOUT_S)
    except asyncio.TimeoutError:
        status, err = "timeout", f"exceeded {PIPELINE_TIMEOUT_S}s"
    except Exception as e:                                   # noqa: BLE001
        status, err = "error", f"{type(e).__name__}: {e}"
    wall = time.time() - t0

    with out_path.open("w") as f:
        for ev in events:
            f.write(json.dumps(slim(ev)) + "\n")

    summ = {
        "status": status, "error": err, "wall_s": round(wall, 1),
        "n_events": len(events),
        "severities": [{"label": e.get("label"), "value": e.get("value")}
                       for e in events if e.get("type") == "severity"],
        "metrics": [{"label": e.get("label"), "value": slim(e.get("value")),
                     "sub": e.get("sub")}
                    for e in events if e.get("type") == "metric"],
        "stream_errors": [e.get("text") or e.get("message") or str(slim(e))
                          for e in events if e.get("type") == "error"],
        "logs": [{"level": e.get("level"), "text": e.get("text")}
                 for e in events if e.get("type") == "log"],
    }
    if any(e.get("type") == "error" for e in events) and status == "ok":
        summ["status"] = "stream_error"
    return summ


async def run_phase(phase_ids, out_dir: Path, only, video_filter, limit):
    from main import PIPELINES                       # heavy import (torch etc.)

    ids = [i for i in phase_ids if not only or i in only]
    videos = find_videos()
    if video_filter:
        videos = [(v, l) for v, l in videos if video_filter in str(v)]
    if limit:
        videos = videos[:limit]
    print(f"[eval] {len(videos)} videos x {len(ids)} pipelines -> {out_dir}",
          flush=True)

    for n, (video, label) in enumerate(videos, 1):
        vdir = out_dir / slug(video)
        vdir.mkdir(parents=True, exist_ok=True)
        spath = vdir / "summary.json"
        summary = json.loads(spath.read_text()) if spath.exists() else {
            "video": str(video.relative_to(ROOT)), "label": label, "pipelines": {}}
        for pid in ids:
            if summary["pipelines"].get(pid, {}).get("status") == "ok":
                continue                              # resumable: retry non-ok
            print(f"[eval] ({n}/{len(videos)}) {slug(video)} :: {pid}", flush=True)
            settings = dict(OVERRIDES.get(pid, {}))
            if pid == "s4_report":
                settings["previous_results"] = prev_results(summary)
            summ = await run_pipeline(PIPELINES[pid]["run"], video,
                                      settings or None, vdir / f"{pid}.jsonl")
            summary["pipelines"][pid] = summ
            spath.write_text(json.dumps(summary, indent=1))
            print(f"[eval]    {summ['status']} {summ['wall_s']}s "
                  f"sev={[s['value'] for s in summ['severities']]} "
                  f"{summ['error'][:120]}", flush=True)
    print("[eval] phase complete", flush=True)


def aggregate(out_dir: Path):
    rows = []
    for spath in sorted(out_dir.glob("*/summary.json")):
        s = json.loads(spath.read_text())
        for pid, p in s["pipelines"].items():
            sev = max((x["value"] for x in p["severities"]
                       if isinstance(x["value"], (int, float))), default=None)
            rows.append({"video": s["video"], "label": s["label"], "pipeline": pid,
                         "status": p["status"], "wall_s": p["wall_s"],
                         "max_severity": sev, "error": p["error"][:200]})
    import csv
    with (out_dir / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-pipeline real-vs-AI separation (max severity as the score)
    pids = sorted({r["pipeline"] for r in rows})
    lines = ["pipeline, n_ok, n_err/timeout, real_mean, ai_mean, auc"]
    for pid in pids:
        pr = [r for r in rows if r["pipeline"] == pid]
        ok = [r for r in pr if r["max_severity"] is not None]
        bad = len(pr) - len(ok)
        real = [r["max_severity"] for r in ok if r["label"] == "real"]
        ai = [r["max_severity"] for r in ok if r["label"] == "ai"]
        auc = ""
        if real and ai:
            wins = sum((a > r) + 0.5 * (a == r) for a in ai for r in real)
            auc = f"{wins / (len(ai) * len(real)):.2f}"
        fm = lambda xs: f"{sum(xs)/len(xs):.1f}" if xs else "-"
        lines.append(f"{pid}, {len(ok)}, {bad}, {fm(real)}, {fm(ai)}, {auc}")
    (out_dir / "separation.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ====================================================================== #
# Human agreement (Rapidata Sora physics set)
# ====================================================================== #
#
# The real-vs-AI aggregate above asks "can it tell generated from real". This
# asks the harder question: given only AI clips, does it ORDER them the way a
# crowd of humans did? That is what the tool claims to do.
#
# Reported per candidate score, because the pipeline emits per-failure severities
# rather than one scalar and there is no a-priori right way to reduce them.
# `s1_vlm` is included specifically so the full pipeline can be compared against
# the single-VLM benchmark already in this repo (vlm_rapidata_results.json).

def _severity_values(pipe: dict) -> list[float]:
    return [x["value"] for x in (pipe.get("severities") or [])
            if isinstance(x.get("value"), (int, float))]


def candidate_scores(pipelines: dict) -> dict:
    """Every plausible reduction of a clip's pipeline output to one number.

    All computed from the SAVED summary, so adding a candidate later costs
    nothing — the expensive part is running the pipeline, and this is a pure
    re-scoring of what is already on disk."""
    s1 = [p for pid, p in pipelines.items() if pid.startswith("s1_")]
    s3 = [p for pid, p in pipelines.items() if pid.startswith("s3_")]
    all_p = list(pipelines.values())

    def mx(ps):
        vals = [v for p in ps for v in _severity_values(p)]
        return max(vals) if vals else None

    def mean(ps):
        vals = [v for p in ps for v in _severity_values(p)]
        return sum(vals) / len(vals) if vals else None

    vlm = pipelines.get("s1_vlm", {})
    rep = pipelines.get("s4_report", {})
    all_vals = [v for p in all_p for v in _severity_values(p)]
    return {
        "max_all":    mx(all_p),
        "max_s1":     mx(s1),
        "max_s3":     mx(s3),
        "mean_s3":    mean(s3),
        "s1_vlm":     mx([vlm]) if vlm else None,
        "s4_report":  mx([rep]) if rep else None,
        "n_flagged":  float(sum(1 for v in all_vals if v >= 50)) if all_vals else None,
    }


def _bootstrap_ci(xs, ys, n_boot, seed=0):
    """Percentile bootstrap CI for Spearman. The existing VLM benchmark reports
    a point estimate only; at n=60 the interval is roughly +/-0.25, which is
    wider than most differences anyone will want to argue about."""
    import numpy as np
    from vlm_rapidata_eval import spearman
    if n_boot <= 0 or len(xs) < 8:
        return None, None
    rng = np.random.default_rng(seed)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xs), len(xs))
        r = spearman(xs[idx], ys[idx])
        if r == r:                                   # skip NaN (degenerate resample)
            stats.append(r)
    if not stats:
        return None, None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def human_agreement(out_dir: Path, labels_path: Path, n_boot: int = 2000):
    sys.path.insert(0, str(Path(__file__).parent))
    from vlm_rapidata_eval import spearman, median_split_auc

    meta = json.loads(labels_path.read_text())
    clips = meta["clips"]

    rows = []
    for spath in sorted(out_dir.glob("*/summary.json")):
        key = spath.parent.name
        if key not in clips:
            continue
        s = json.loads(spath.read_text())
        pipelines = s.get("pipelines", {})
        n_ok = sum(1 for p in pipelines.values() if p.get("status") == "ok")
        rows.append({
            "clip": key,
            "human": float(clips[key]["human_score"]),
            "scores": candidate_scores(pipelines),
            "n_pipelines_ok": n_ok,
            "n_pipelines": len(pipelines),
            "wall_s": round(sum(p.get("wall_s", 0) or 0 for p in pipelines.values()), 1),
        })

    if len(rows) < 8:
        print(f"[human] only {len(rows)} scored clips — too few to correlate.")
        return

    human = [r["human"] for r in rows]
    names = list(rows[0]["scores"].keys())

    print("\n" + "=" * 92)
    print(f"  HUMAN AGREEMENT — {meta['dataset']}")
    print(f"  {len(rows)} clips scored of {meta.get('n_staged', '?')} staged "
          f"({meta.get('n_total_in_dataset', '?')} in the dataset)")
    print("  label: higher = humans found it MORE implausible; severity is also")
    print("  higher = worse, so a POSITIVE correlation is the expected direction.")
    print("=" * 92)
    print(f"  {'score':<12}{'n':>5}{'spearman':>10}{'95% CI':>18}"
          f"{'median-split AUC':>19}")
    print("  " + "-" * 88)

    results = {}
    for nm in names:
        pairs = [(r["scores"][nm], r["human"]) for r in rows
                 if r["scores"][nm] is not None]
        if len(pairs) < 8:
            print(f"  {nm:<12}{len(pairs):>5}   (too few clips produced this score)")
            results[nm] = {"n": len(pairs)}
            continue
        xs = [p for p, _ in pairs]
        ys = [h for _, h in pairs]
        rho = spearman(xs, ys)
        lo, hi = _bootstrap_ci(xs, ys, n_boot)
        auc = median_split_auc([r["scores"][nm] for r in rows], human)
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "-"
        print(f"  {nm:<12}{len(pairs):>5}{rho:>10.3f}{ci:>18}{auc:>19.3f}")
        results[nm] = {"n": len(pairs), "spearman": round(rho, 3),
                       "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
                       "median_split_auc": round(auc, 3)}

    print("  " + "-" * 88)
    print("  Seven candidate reductions are reported because the pipeline has no single")
    print("  scalar output. Testing seven inflates the chance one looks good by luck —")
    print("  read the CI, and treat `max_all` as the pre-registered headline.")

    # Baseline from the VLM-only benchmark already in this repo.
    base = Path(__file__).parent / "vlm_rapidata_results.json"
    if base.exists():
        try:
            b = json.loads(base.read_text())
            print("\n  Single-VLM baseline on this dataset (scripts/vlm_rapidata_eval.py):")
            for mid, res in b.get("models", {}).items():
                sm = res.get("summary") or {}
                if sm.get("spearman") is not None:
                    lp = sm.get("logprob_spearman")
                    extra = f"   logprob rho={lp}" if lp is not None else ""
                    print(f"    {mid:<42} rho={sm['spearman']:<6} "
                          f"AUC={sm['median_split_auc']}{extra}  (n={sm.get('n')})")
            print("  If the full pipeline does not clear these, that is the finding.")
        except Exception:
            pass

    # Per-pipeline: which single stage carries the signal.
    print("\n  Per-pipeline severity vs human score (which stage is doing the work):")
    print(f"    {'pipeline':<26}{'n':>5}{'spearman':>10}{'coverage':>11}")
    pids = sorted({pid for spath in out_dir.glob("*/summary.json")
                   for pid in json.loads(spath.read_text()).get("pipelines", {})})
    per_pipe = {}
    for pid in pids:
        pairs = []
        for spath in sorted(out_dir.glob("*/summary.json")):
            key = spath.parent.name
            if key not in clips:
                continue
            p = json.loads(spath.read_text()).get("pipelines", {}).get(pid)
            if not p:
                continue
            vals = _severity_values(p)
            if vals:
                pairs.append((max(vals), float(clips[key]["human_score"])))
        if len(pairs) >= 8:
            rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
            cov = len(pairs) / len(rows)
            print(f"    {pid:<26}{len(pairs):>5}{rho:>10.3f}{cov:>10.0%}")
            per_pipe[pid] = {"n": len(pairs), "spearman": round(rho, 3),
                             "coverage": round(cov, 3)}
        else:
            print(f"    {pid:<26}{len(pairs):>5}         -  "
                  f"{len(pairs)/max(1,len(rows)):>9.0%}   (no usable severity)")
            per_pipe[pid] = {"n": len(pairs), "spearman": None,
                             "coverage": round(len(pairs) / max(1, len(rows)), 3)}
    print("=" * 92 + "\n")

    # Largest disagreements — the input to failure analysis.
    head = [r for r in rows if r["scores"]["max_all"] is not None]
    if head:
        import numpy as np
        sc = np.asarray([r["scores"]["max_all"] for r in head], float)
        hu = np.asarray([r["human"] for r in head], float)
        # compare on a common scale: rank-normalise both, then diff
        def rk(v):
            o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
            return r / max(1, len(v) - 1)
        d = rk(sc) - rk(hu)
        order = np.argsort(-np.abs(d))
        print("  Largest disagreements (rank-normalised; + = tool harsher than humans):")
        for i in order[:10]:
            print(f"    {head[i]['clip']:<44} human={hu[i]:.3f} "
                  f"sev={sc[i]:<6.1f} delta={d[i]:+.2f}")
        for i, r in enumerate(head):
            r["rank_delta"] = round(float(d[i]), 3)

    payload = {"dataset": meta["dataset"], "n_clips": len(rows),
               "n_bootstrap": n_boot, "by_score": results,
               "by_pipeline": per_pipe, "rows": rows}
    (out_dir / "human_agreement.json").write_text(json.dumps(payload, indent=1))
    print(f"  -> {out_dir / 'human_agreement.json'}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_reports/run")
    ap.add_argument("--only", help="comma-separated pipeline ids")
    ap.add_argument("--skip", help="comma-separated pipeline ids to exclude")
    ap.add_argument("--videos", help="substring filter on video path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--videos-dir", default="test_videos",
                    help="video root, relative to repo root "
                         "(e.g. data/rapidata, staged by rapidata_prepare.py)")
    ap.add_argument("--labels",
                    help="JSON from rapidata_prepare.py: continuous human score "
                         "per clip. With --aggregate, switches the report from "
                         "real-vs-AI separation to human-agreement.")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="bootstrap resamples for the Spearman CI (0 to skip)")
    a = ap.parse_args()

    global VIDEO_ROOT
    VIDEO_ROOT = (ROOT / a.videos_dir) if not Path(a.videos_dir).is_absolute() \
        else Path(a.videos_dir)

    out_dir = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.aggregate:
        aggregate(out_dir)
        if a.labels:
            lab_path = (ROOT / a.labels) if not Path(a.labels).is_absolute() \
                else Path(a.labels)
            human_agreement(out_dir, lab_path, a.bootstrap)
        return
    only = set(a.only.split(",")) if a.only else None
    order = [i for i in ORDER if i not in set((a.skip or "").split(","))]
    asyncio.run(run_phase(order, out_dir, only, a.videos, a.limit))


if __name__ == "__main__":
    main()
