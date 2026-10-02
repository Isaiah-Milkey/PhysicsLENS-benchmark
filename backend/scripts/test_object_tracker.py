"""Smoke test: run the stage-2 object tracker on real + AI videos, print all events."""
import asyncio, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.stage2.object_tracker import run as track


async def run_one(video: str, settings: dict | None = None):
    print(f"\n{'='*70}\n{video}\n{'='*70}")
    t0 = time.time()
    events = {"log": 0, "metric": 0, "plotly": 0, "severity": 0, "error": 0, "done": 0}
    async for ev in track(video, json.dumps(settings) if settings else None):
        et = ev.get("type", "?")
        events[et] = events.get(et, 0) + 1
        if et == "log":
            print(f"  [{ev['level']:7s}] {ev['text']}")
        elif et == "metric":
            print(f"  METRIC   {ev['label']}: {ev['value']}  ({ev.get('sub','')})")
        elif et == "severity":
            print(f"  SEVERITY {ev['label']}: {ev['value']}")
        elif et == "plotly":
            fig = json.loads(ev["data"])
            print(f"  PLOTLY   {len(fig.get('data', []))} traces")
        elif et == "error":
            print(f"  ERROR    {ev['text']}")
    print(f"  -- {time.time()-t0:.1f}s  events: {events}")
    assert events["done"] == 1 or events["error"] > 0, "pipeline ended without done/error"
    return events


async def main():
    root = Path(__file__).resolve().parents[2] / "test_videos"
    vids = [
        (root / "real/wikimedia/bouncing_ball.webm", None),
        (root / "real/physics_iq/balls-collide.mp4", None),
        (root / "ai/sora/basketball-explosion.mp4", None),
        (root / "ai/lumiere/beer_pouring.mp4", None),
        # no-dinov2 fallback path
        (root / "real/physics_iq/ball-ramp.mp4", {"use_dinov2": "false"}),
    ]
    results = {}
    for v, cfg in vids:
        if not v.exists():
            print(f"SKIP missing {v}")
            continue
        results[v.name + (" (no-dino)" if cfg else "")] = await run_one(str(v), cfg)

    print("\n\nSUMMARY")
    for name, ev in results.items():
        ok = ev["done"] == 1 and ev["error"] == 0
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {ev}")


asyncio.run(main())
