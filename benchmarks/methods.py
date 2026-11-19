"""The methods under test, behind one signature: `fn(pair) -> flow`.

Each method gets the whole `Pair`, so it can pick the input format it wants (RGB,
BGR or gray). It returns float32 (H, W, 2) flow, with NaN wherever it gives no answer.
Only pixelmap ever returns NaN. OpenCV's methods answer everywhere.

`gt` and `zero` are not methods. They check the loaders and metrics: `gt` must score
EPE 0 and coverage 1, and `zero` must score EPE equal to the mean ground-truth
magnitude.

Derived methods (`*_filled`) post-process another method's output, so `run.py`
computes their base once per pair and reuses it.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import pixelmap

from datasets import Pair

SEED = None  # None = the crate's fixed DEFAULT_SEED, so runs are reproducible


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _bgr(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, :, ::-1])


def _pixelmap(quality: str) -> Callable[[Pair], np.ndarray]:
    def run(pair: Pair) -> np.ndarray:
        mapping = pixelmap.correspond(pair.img1, pair.img2, quality=quality, seed=SEED)
        return np.asarray(mapping.flow(), dtype=np.float32)

    return run


def _dis(preset: int) -> Callable[[Pair], np.ndarray]:
    def run(pair: Pair) -> np.ndarray:
        dis = cv2.DISOpticalFlow_create(preset)
        return dis.calc(_gray(pair.img1), _gray(pair.img2), None)

    return run


def farneback(pair: Pair) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        _gray(pair.img1), _gray(pair.img2), None,
        pyr_scale=0.5, levels=6, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )


def deepflow(pair: Pair) -> np.ndarray:
    return cv2.optflow.createOptFlow_DeepFlow().calc(_gray(pair.img1), _gray(pair.img2), None)


def rlof(pair: Pair) -> np.ndarray:
    return cv2.optflow.calcOpticalFlowDenseRLOF(_bgr(pair.img1), _bgr(pair.img2), None)


def pcaflow(pair: Pair) -> np.ndarray:
    return cv2.optflow.createOptFlow_PCAFlow().calc(_gray(pair.img1), _gray(pair.img2), None)


def gt(pair: Pair) -> np.ndarray:
    flow = pair.gt_flow.copy()
    flow[~pair.gt_valid] = np.nan
    return flow


def zero(pair: Pair) -> np.ndarray:
    return np.zeros(pair.gt_flow.shape, np.float32)


def fill_nearest(flow: np.ndarray) -> np.ndarray:
    """Fill each NaN with the vector of the nearest answered pixel (L2 distance)."""
    answered = np.all(np.isfinite(flow), axis=-1)
    if answered.all() or not answered.any():
        return flow
    # distanceTransform measures distance to the nearest *zero* pixel. With
    # DIST_LABEL_PIXEL every zero pixel gets its own label, numbered 1.. in raster order.
    src = (~answered).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    flat = np.flatnonzero(answered)
    nearest = flat[labels.ravel() - 1]
    out = flow.reshape(-1, 2)[nearest].reshape(flow.shape)
    return np.ascontiguousarray(out, dtype=np.float32)


METHODS: dict[str, Callable[[Pair], np.ndarray]] = {
    "pixelmap_low": _pixelmap("low"),
    "pixelmap_medium": _pixelmap("medium"),
    "pixelmap_high": _pixelmap("high"),
    "pixelmap_ultra": _pixelmap("ultra"),
    "dis_fast": _dis(cv2.DISOPTICAL_FLOW_PRESET_FAST),
    "dis_medium": _dis(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM),
    "farneback": farneback,
    "deepflow": deepflow,
    "rlof": rlof,
    "pcaflow": pcaflow,
    "gt": gt,
    "zero": zero,
}

# name -> (base method, post-processing step)
DERIVED: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    f"{base}_filled": (base, fill_nearest)
    for base in ("pixelmap_low", "pixelmap_medium", "pixelmap_high", "pixelmap_ultra")
}

ALL = list(METHODS) + list(DERIVED)
# `ultra` differs from `high` only on photos wider than 1600 px, which none of the
# benchmark datasets have, so a default run leaves it out rather than repeat `high`.
DEFAULT = [
    m
    for m in ALL
    if m not in ("gt", "zero", "pixelmap_low_filled", "pixelmap_ultra", "pixelmap_ultra_filled")
]

# The method whose answered pixels define the "shared mask" every method is also
# scored on: is pixelmap more accurate *where it answers*?
REFERENCE = "pixelmap_medium"
