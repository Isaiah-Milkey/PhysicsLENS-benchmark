"""
Shared evidence bus for the PhysicsLENS multi-stage pipeline.
------------------------------------------------------------
Each pipeline's run() receives only `(video_path, settings)`, so there is no
native way to pass results downstream (Stage 2 trajectories → Stage 3
specialists → Stage 4 scorer). This module is the server-side data bus that
closes that gap: pipelines write their structured outputs keyed by a content
hash of the video, and downstream stages read them back.

It also exposes `file_hash`, the canonical per-video identity used both here
and by the cached tracker (`tools.tracking`), so every stage agrees on which
video — and which cache entry — it is operating on.

In-process and thread-safe; an LRU cap keeps memory bounded across many
uploads. Evidence is intentionally ephemeral (cleared on server restart).
"""
import hashlib
import os
import threading
from collections import OrderedDict
from typing import Any, Optional

_LOCK = threading.RLock()
_MAX_VIDEOS = 16          # LRU cap on distinct videos retained


# ── Canonical video identity ──────────────────────────────────────────────────

def file_hash(path: str, _memo: dict = {}) -> str:
    """Stable content id for a video file. Hashes size + head/tail 1 MiB (fast
    on large files) and memoises by (path, size, mtime) within the process."""
    st = os.stat(path)
    memo_key = (path, st.st_size, st.st_mtime_ns)
    cached = _memo.get(memo_key)
    if cached is not None:
        return cached

    h = hashlib.sha1()
    h.update(str(st.st_size).encode())
    chunk = 1 << 20
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    digest = h.hexdigest()[:16]
    _memo[memo_key] = digest
    return digest


def video_id(video_path: str) -> str:
    """Public alias — the evidence-bus key for a given video path."""
    return file_hash(video_path)


# ── Evidence store ────────────────────────────────────────────────────────────

class EvidenceStore:
    """video_id → {stage_id → payload}. LRU-bounded, thread-safe."""

    def __init__(self, max_videos: int = _MAX_VIDEOS):
        self._d: "OrderedDict[str, dict]" = OrderedDict()
        self._max = max_videos

    def put(self, vid: str, stage_id: str, payload: Any) -> None:
        with _LOCK:
            bucket = self._d.get(vid)
            if bucket is None:
                bucket = {}
                self._d[vid] = bucket
            bucket[stage_id] = payload
            self._d.move_to_end(vid)
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def get(self, vid: str, stage_id: str, default: Any = None) -> Any:
        with _LOCK:
            return self._d.get(vid, {}).get(stage_id, default)

    def get_all(self, vid: str) -> dict:
        """Shallow copy of every stage payload recorded for a video."""
        with _LOCK:
            return dict(self._d.get(vid, {}))

    def has(self, vid: str, stage_id: str) -> bool:
        with _LOCK:
            return stage_id in self._d.get(vid, {})

    def clear(self, vid: Optional[str] = None) -> None:
        with _LOCK:
            if vid is None:
                self._d.clear()
            else:
                self._d.pop(vid, None)


# Process-wide singleton — import and use directly.
EVIDENCE = EvidenceStore()
