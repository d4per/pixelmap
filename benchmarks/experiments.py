"""Experimental pixelmap variants, for tuning a preset above `high`.

Imported by `run.py --experiments`. The variants drive hidden options of a pixelmap
build with the `bench` feature through environment variables (see
`Builder::with_env_overrides` in the crate), so they need a development build of the
binding on `PYTHONPATH`; against the published wheel they would all silently run
plain `high`. Write their results to a separate CSV (`--out results/experiments.csv`).

    PIXELMAP_SCHEDULE     width:clean_max_dist,...   replaces the preset's schedule
    PIXELMAP_FILL         off | plain | checked      fill final holes from earlier widths
    PIXELMAP_RADIUS       r                          scoring disc radius of every step (default 5)
    PIXELMAP_INIT_RADIUS  r                          ... of the init solve alone
    PIXELMAP_LOCAL_CONSISTENCY  off | on | flow,jacobian,anisotropy   `Builder::local_consistency`

A `PIXELMAP_SCHEDULE` step may also name its own radius, `width:dist:radius`.

These are the variants that led to `ultra` and to the source-width cap and hole filling in
`medium` and `high`. A variant that names no fill mode runs with it off, so
each still measures what it measured then. The multi-resolution init variants (extra
feature-matching widths at init) are gone along with the crate option: they changed
nothing measurable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pixelmap

import cv2

import datasets
import methods
from datasets import Pair

Schedule = Sequence[tuple[int, float]]

MEDIUM: list[tuple[int, float]] = [
    (400, 3.0), (400, 2.0), (400, 1.0), (400, 0.5), (400, 1.0), (400, 2.0), (400, 2.0),
    (800, 3.0), (800, 1.0), (800, 2.0),
]
HIGH: list[tuple[int, float]] = MEDIUM + [(1600, 1.0), (1600, 2.0), (1600, 2.0)]

_ENV = ("PIXELMAP_SCHEDULE", "PIXELMAP_FILL", "PIXELMAP_RADIUS", "PIXELMAP_INIT_RADIUS",
        "PIXELMAP_LOCAL_CONSISTENCY")


def _variant(
    quality: str,
    schedule: Union[None, Schedule, Callable[[int], Schedule]] = None,
    fill: str = "off",
    finish: float = 2.0,
    seed: Optional[int] = None,
) -> Callable[[Pair], np.ndarray]:
    def run(pair: Pair) -> np.ndarray:
        w = pair.img1.shape[1]
        # The presets' old schedules: they now cap their widths at the photo's own.
        steps = schedule(w) if callable(schedule) else (schedule or {"medium": MEDIUM, "high": HIGH}[quality])
        env = {
            "PIXELMAP_SCHEDULE": ",".join(f"{sw}:{d}" for sw, d in steps) if steps else "",
            "PIXELMAP_FILL": fill,
        }
        os.environ.update(env)
        try:
            mapping = pixelmap.correspond(
                pair.img1, pair.img2, quality=quality, seed=methods.SEED if seed is None else seed, max_round_trip_error=finish
            )
        finally:
            for k in _ENV:
                os.environ.pop(k, None)
        return np.asarray(mapping.flow(), dtype=np.float32)

    return run


def _src(w: int) -> list[tuple[int, float]]:
    """High with its 1600 px steps at the source width instead (no upsampling)."""
    return MEDIUM + [(w, 1.0), (w, 2.0), (w, 2.0)]


def _ultra(w: int) -> list[tuple[int, float]]:
    """High with every step capped at the source width, plus two steps at up to 3200 px
    for sources wider than 1600."""
    steps = [(min(sw, w), d) for sw, d in HIGH]
    if w > 1600:
        g = min(w, 3200)
        steps += [(g, 1.0), (g, 2.0)]
    return steps


def _radius(
    quality: str,
    radius: Optional[int] = None,
    per_width: Optional[Callable[[int, bool], int]] = None,
    init_radius: Optional[int] = None,
    seed: Optional[int] = None,
) -> Callable[[Pair], np.ndarray]:
    """The shipped `quality` preset (its own capped schedule and fill) with another scoring
    disc radius. `radius` sets every step and the init solve. `per_width(width, final)` picks
    a step's radius from its working width and whether that is the schedule's last width,
    and then the schedule is rebuilt here exactly as `Quality::steps_for` builds it."""

    def run(pair: Pair) -> np.ndarray:
        env = {"PIXELMAP_RADIUS": "" if radius is None else str(radius),
               "PIXELMAP_INIT_RADIUS": "" if init_radius is None else str(init_radius)}
        if per_width is not None:
            w = pair.img1.shape[1]
            steps = [(min(sw, w), d) for sw, d in {"medium": MEDIUM, "high": HIGH}[quality]]
            last = steps[-1][0]
            env["PIXELMAP_SCHEDULE"] = ",".join(f"{sw}:{d}:{per_width(sw, sw == last)}" for sw, d in steps)
        os.environ.update(env)
        try:
            mapping = pixelmap.correspond(
                pair.img1, pair.img2, quality=quality, seed=methods.SEED if seed is None else seed
            )
        finally:
            for k in _ENV:
                os.environ.pop(k, None)
        return np.asarray(mapping.flow(), dtype=np.float32)

    return run


def _shipped(quality: str, fill: str = "", lc: str = "off", finish: float = 2.0) -> Callable[[Pair], np.ndarray]:
    """The shipped `quality` preset, with its fill mode replaced unless `fill` is empty, the
    local-consistency filter set by `lc` (see `PIXELMAP_LOCAL_CONSISTENCY`), and `finish`
    as the final round-trip threshold."""

    def run(pair: Pair) -> np.ndarray:
        os.environ.update({"PIXELMAP_FILL": fill, "PIXELMAP_LOCAL_CONSISTENCY": lc})
        try:
            mapping = pixelmap.correspond(
                pair.img1, pair.img2, quality=quality, seed=methods.SEED, max_round_trip_error=finish
            )
        finally:
            for k in _ENV:
                os.environ.pop(k, None)
        return np.asarray(mapping.flow(), dtype=np.float32)

    return run


def _split(coarse: int, final: int) -> Callable[[int, bool], int]:
    """`coarse` at every width before the last, `final` at the last."""
    return lambda _w, is_final: final if is_final else coarse


VARIANTS: dict[str, Callable[[Pair], np.ndarray]] = {
    # idea 2: final round-trip threshold (grid cells at the final working width)
    "x_high_fin3": _variant("high", finish=3.0),
    "x_high_fin4": _variant("high", finish=4.0),
    "x_high_fin6": _variant("high", finish=6.0),
    "x_medium_fin3": _variant("medium", finish=3.0),
    "x_medium_fin4": _variant("medium", finish=4.0),
    "x_high_last4": _variant("high", schedule=HIGH[:-1] + [(1600, 4.0)], finish=4.0),
    # idea 1: resolution
    "x_high_more1600": _variant("high", schedule=HIGH + [(1600, 1.0), (1600, 2.0)]),
    "x_high_3200": _variant("high", schedule=HIGH + [(3200, 1.0), (3200, 2.0)]),
    "x_high_src": _variant("high", schedule=_src),
    # idea 4: fill final holes from earlier widths
    "x_high_fill": _variant("high", fill="plain"),
    "x_high_fillchk": _variant("high", fill="checked"),
    # combinations of the winners
    "x_src_fill": _variant("high", schedule=_src, fill="plain"),
    "x_src_fillchk": _variant("high", schedule=_src, fill="checked"),
    "x_src_fill_fin3": _variant("high", schedule=_src, fill="plain", finish=3.0),
    "x_srcmore": _variant("high", schedule=lambda w: _src(w) + [(w, 1.0), (w, 2.0)]),
    "x_srcmore_fill": _variant("high", schedule=lambda w: _src(w) + [(w, 1.0), (w, 2.0)], fill="plain"),
    "x_ultra": _variant("high", schedule=_ultra, fill="plain"),
    "x_ultra_chk": _variant("high", schedule=_ultra, fill="checked"),
    "x_ultra_s2": _variant("high", schedule=_ultra, fill="plain", seed=2),
    "x_high_s2": _variant("high", seed=2),
    "x_high_3200_fill": _variant("high", schedule=HIGH + [(3200, 1.0), (3200, 2.0)], fill="plain"),
    # scoring disc radius, on the shipped presets (2026-11-21)
    **{f"x_r{r}": _radius("high", radius=r) for r in (3, 4, 5, 6, 7, 9)},
    **{f"x_med_r{r}": _radius("medium", radius=r) for r in (4, 5, 7)},
    "x_r5_s2": _radius("high", radius=5, seed=2),
    **{f"x_r{r}": _radius("high", radius=r) for r in (11, 13)},
    # a different radius at the coarse widths (and the init solve) than at the final one
    **{f"x_c{c}_f{f}": _radius("high", per_width=_split(c, f), init_radius=c)
       for c, f in ((7, 5), (9, 5), (13, 5), (9, 7), (5, 9))},
    "x_i9": _radius("high", init_radius=9),
    "x_med_c9_f5": _radius("medium", per_width=_split(9, 5), init_radius=9),
    # the init solve's radius on its own and combined
    "x_i13": _radius("high", init_radius=13),
    "x_i9_s2": _radius("high", init_radius=9, seed=2),
    "x_i9_c7_f5": _radius("high", per_width=_split(7, 5), init_radius=9),
    "x_i9_c5_f7": _radius("high", per_width=_split(5, 7), init_radius=9),
    "x_i13_c7_f7": _radius("high", per_width=_split(7, 7), init_radius=13),
    "x_med_i9": _radius("medium", init_radius=9),
    # the local-consistency filter, on High with its own fill and with fill off (2026-11-22)
    "x_high_base": _shipped("high"),
    "x_high_lc": _shipped("high", lc="on"),
    "x_high_off": _shipped("high", fill="off"),
    "x_high_off_lc": _shipped("high", fill="off", lc="on"),
    # the existing knob, for comparison at equal coverage
    "x_high_off_rt1": _shipped("high", fill="off", finish=1.0),
    "x_high_off_rt07": _shipped("high", fill="off", finish=0.7),
    "x_high_off_rt07_lc": _shipped("high", fill="off", lc="on", finish=0.7),
}


def hires(stride: int = 1, width: int = 3200):
    """Real 4128 px photos (the tri1/tri2 sets) scaled to `width` and warped by known
    homographies: exact ground truth on real high-resolution texture, to see whether
    working above 1600 px pays off for photos that have the detail. No parallax or
    occlusion, so this says nothing about those."""
    rng = np.random.default_rng(4321)
    photos = sorted((Path.home() / "Pictures" / "evert").glob("tri[12]/*.jpg"))
    if not photos:
        raise datasets.DatasetMissing("hires needs ~/Pictures/evert/tri1 and tri2")
    idx = 0
    for path in photos:
        full = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h = round(width * full.shape[0] / full.shape[1])
        base = cv2.resize(full, (width, h), interpolation=cv2.INTER_AREA)
        for k in range(2):
            corners = np.float32([[0, 0], [width, 0], [width, h], [0, h]])
            jitter = rng.uniform(-0.10, 0.10, size=(4, 2)) * np.float32([width, h])
            H = cv2.getPerspectiveTransform(corners, (corners + jitter).astype(np.float32))
            if idx % stride == 0:
                warped = cv2.warpPerspective(base, H, (width, h), flags=cv2.INTER_LINEAR)
                flow, valid = datasets._homography_flow(H, h, width, h, width, h, width)
                yield Pair("hires", f"{path.stem}_{k}", base, warped, flow, valid)
            idx += 1


def register() -> None:
    datasets.DATASETS["hires"] = hires
    methods.METHODS.update(VARIANTS)
    for name in VARIANTS:
        methods.DERIVED[f"{name}_filled"] = (name, methods.fill_nearest)
    methods.ALL[:] = list(methods.METHODS) + list(methods.DERIVED)


register()
