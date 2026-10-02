"""Shared sparse optical-flow utilities (Lucas-Kanade)."""
from typing import Optional, Tuple

import cv2
import numpy as np

_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def detect_keypoints(gray: np.ndarray, n: int = 20) -> Optional[np.ndarray]:
    """Shi-Tomasi corners. Returns (N,1,2) float32 or None."""
    return cv2.goodFeaturesToTrack(
        gray, maxCorners=n, qualityLevel=0.01, minDistance=10, blockSize=7
    )


def track_keypoints(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """LK tracking; returns only points where status == 1."""
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **_LK
    )
    mask = status.ravel() == 1
    return prev_pts[mask], curr_pts[mask]


def flow_vectors(
    prev_pts: np.ndarray, curr_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point (magnitude px, angle rad) from two (N,2) arrays."""
    disp  = (curr_pts - prev_pts).reshape(-1, 2)
    mag   = np.linalg.norm(disp, axis=1)
    angle = np.arctan2(disp[:, 1], disp[:, 0])
    return mag, angle


def direction_consistency(angles: np.ndarray) -> float:
    """Mean resultant length: 1 = all same direction, 0 = isotropic."""
    if len(angles) == 0:
        return 0.0
    cos_m = np.cos(angles).mean()
    sin_m = np.sin(angles).mean()
    return float(np.sqrt(cos_m ** 2 + sin_m ** 2))
