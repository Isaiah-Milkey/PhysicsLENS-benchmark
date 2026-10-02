"""
Run any PhysicsLENS pipeline from the command line (no server needed).

Usage:
  python scripts/run_pipeline.py <pipeline_id> <video_path> [settings_json]
  python scripts/run_pipeline.py <pipeline_id> <video_path> --set key=val --set key=val
  python scripts/run_pipeline.py --list                 # show all pipeline ids
  python scripts/run_pipeline.py <pipeline_id> <video> --prereq s2_trajectory_extractor

Stage 3/4 tests read earlier results from the in-process evidence bus. Since a
CLI run is a fresh process, pass --prereq <pipeline_id> (repeatable) to run those
first and populate the bus before the main test.

Examples:
  # Stage 3 VLM verification on one clip (local model, no API key)
  python scripts/run_pipeline.py s3_specialist path/to/clip.mp4 --set model=qwen2.5-vl-7b

  # Stage 4 report with Stage 1/2 evidence on the bus
  python scripts/run_pipeline.py s4_report path/to/clip.mp4 \
      --prereq s1_temporal --prereq s2_trajectory_extractor

  # Override a setting
  python scripts/run_pipeline.py s1_temporal path/to/clip.mp4 --set key=value
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import PIPELINES  # noqa: E402


def _parse_args(argv):
    if not argv or argv[0] in ("--list", "-l"):
        return None
    pid = argv[0]
    video = None
    settings = {}
    prereqs = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--set":
            k, _, v = argv[i + 1].partition("=")
            # best-effort typing: int, float, else string
            for cast in (int, float):
                try:
                    v = cast(v); break
                except ValueError:
                    pass
            settings[k] = v
            i += 2
        elif a == "--prereq":
            prereqs.append(argv[i + 1]); i += 2
        elif a.startswith("{"):
            settings.update(json.loads(a)); i += 1
        elif video is None:
            video = a; i += 1
        else:
            i += 1
    return pid, video, settings, prereqs


async def _drive(pid, video, settings):
    p = PIPELINES[pid]
    gen = p["run"](video, settings=json.dumps(settings)) if p.get("settings") \
        else p["run"](video)
    async for ev in gen:
        t = ev.get("type")
        if t == "log":
            print(f"  [{ev.get('level', 'info'):7s}] {ev.get('text', '')}")
        elif t == "metric":
            print(f"  METRIC  {ev.get('label')}: {ev.get('value')}  ({ev.get('sub', '')})")
        elif t == "severity":
            print(f"  SEVERITY {ev.get('label')}: {ev.get('value')}/100")
        elif t == "plotly":
            print(f"  [plot]  {ev.get('caption', '')[:90]}")
        elif t in ("video", "image", "marker_video"):
            print(f"  [{t}]   {ev.get('caption', '')[:90]}")
        elif t == "error":
            print(f"  ERROR   {ev.get('text')}")
        elif t == "done":
            print("  done.")


async def main():
    argv = sys.argv[1:]
    parsed = _parse_args(argv)
    if parsed is None:
        print("Pipelines:")
        for pid, p in PIPELINES.items():
            stub = " (stub)" if p.get("dummy") else ""
            print(f"  {pid:28s} {p['name']}{stub}")
        return
    pid, video, settings, prereqs = parsed
    if pid not in PIPELINES:
        print(f"Unknown pipeline '{pid}'. Use --list to see options."); return
    if not video or not Path(video).exists():
        print(f"Video not found: {video}"); return

    for pre in prereqs:
        if pre not in PIPELINES:
            print(f"Skipping unknown prereq '{pre}'"); continue
        print(f"\n### prereq: {pre} — populating evidence bus")
        await _drive(pre, video, settings if PIPELINES[pre].get("settings") else {})

    print(f"\n### {pid}: {PIPELINES[pid]['name']}")
    await _drive(pid, video, settings)


if __name__ == "__main__":
    asyncio.run(main())
