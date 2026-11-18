#!/usr/bin/env python3
"""Run methods over datasets and append one CSV row per (pair, method, threads).

    python run.py --datasets synthetic
    python run.py --datasets hpatches eth3d --threads 1
    python run.py --datasets kitti15 --methods pixelmap_medium dis_medium --limit 20

Runs can be resumed: a row that already exists in the output CSV is not computed
again. Run `report.py` on the CSV to get the tables.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    # Import-free, so the thread count can be set in the environment before rayon and
    # OpenCV start their pools.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["synthetic"])
    p.add_argument("--methods", nargs="+", default=None, help="default: every method except gt/zero/high")
    p.add_argument("--stride", type=int, default=None, help="take every N-th pair (overrides dataset default)")
    p.add_argument("--limit", type=int, default=None, help="at most N pairs per dataset")
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="threads for both pixelmap and OpenCV")
    p.add_argument("--save-vis", type=int, default=0, metavar="N", help="save flow images for the first N pairs per dataset")
    p.add_argument("--out", type=Path, default=HERE / "results" / "results.csv")
    p.add_argument("--experiments", action="store_true", help="register the variants in experiments.py")
    return p.parse_args()


args = parse_args()
os.environ["RAYON_NUM_THREADS"] = str(args.threads)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import datasets  # noqa: E402
import methods  # noqa: E402
from metrics import PCK, score  # noqa: E402

if args.experiments:
    import experiments  # noqa: E402,F401  (registers its variants into `methods`)

cv2.setNumThreads(args.threads)

_METRICS = ["gt_pixels", "gt_mag", "coverage", "epe", "epe_med", "fl"] + [f"pck{k}" for k in PCK]
FIELDS = (
    ["dataset", "pair", "method", "threads", "width", "height", "seconds"]
    + _METRICS
    + [f"{m}_noc" for m in ("coverage", "epe", "fl")] + [f"pck{k}_noc" for k in PCK]
    + ["epe_shared", "shared_frac", "error"]
)


def flow_to_color(flow: np.ndarray, max_mag: float) -> np.ndarray:
    """HSV flow colouring (hue = direction, value = magnitude); no answer is black."""
    answered = np.all(np.isfinite(flow), axis=-1)
    f = np.where(answered[..., None], flow, 0)
    mag, ang = cv2.cartToPolar(f[..., 0].astype(np.float32), f[..., 1].astype(np.float32), angleInDegrees=True)
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max(max_mag, 1e-6), 0, 1) * 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[~answered] = 0
    return bgr


def main() -> int:
    wanted = args.methods or methods.DEFAULT
    unknown = [m for m in wanted if m not in methods.ALL]
    if unknown:
        sys.exit(f"unknown methods {unknown}; choose from {methods.ALL}")
    bad = [d for d in args.datasets if d not in datasets.DATASETS]
    if bad:
        sys.exit(f"unknown datasets {bad}; choose from {list(datasets.DATASETS)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, str, str]] = set()
    if args.out.exists():
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["pair"], r["method"], r["threads"]))
    new_file = not args.out.exists()
    out = open(args.out, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
    if new_file:
        writer.writeheader()

    warmed: set[str] = set()
    threads = str(args.threads)

    for ds_name in args.datasets:
        loader = datasets.DATASETS[ds_name]
        try:
            pairs = loader(args.stride) if args.stride else loader()
            n_seen = 0
            for pair in pairs:
                if args.limit is not None and n_seen >= args.limit:
                    break
                n_seen += 1
                vis = n_seen <= args.save_vis
                todo = [m for m in wanted if (ds_name, pair.name, m, threads) not in done]
                if not todo and not vis:
                    continue
                run_pair(pair, todo if not vis else wanted, todo, writer, warmed, vis)
                out.flush()
        except datasets.DatasetMissing as e:
            print(f"skipping: {e}", file=sys.stderr)
    out.close()
    return 0


def run_pair(pair, compute, record, writer, warmed, vis) -> None:
    """Compute `compute` methods on one pair; write rows for the ones in `record`."""
    cache: dict[str, tuple[np.ndarray, float]] = {}

    def get(name: str):
        if name in cache:
            return cache[name]
        try:
            if name in methods.DERIVED:
                base, post = methods.DERIVED[name]
                base_flow, base_s = get(base)
                t = time.perf_counter()
                flow = post(base_flow)
                result = (flow, base_s + time.perf_counter() - t)
            else:
                fn = methods.METHODS[name]
                if name not in warmed:  # first call pays for pool start-up, JIT-free but cold caches
                    fn(pair)
                    warmed.add(name)
                t = time.perf_counter()
                flow = fn(pair)
                result = (flow, time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001 — one failing method must not end the run
            result = (None, float("nan"), f"{type(e).__name__}: {e}")
        cache[name] = result
        return result

    ref = get(methods.REFERENCE)
    shared = None
    if ref[0] is not None:
        shared = np.all(np.isfinite(ref[0]), axis=-1) & pair.gt_valid

    h, w = pair.gt_valid.shape
    for name in compute:
        res = get(name)
        if name not in record:
            continue
        row = {"dataset": pair.dataset, "pair": pair.name, "method": name, "threads": args.threads,
               "width": w, "height": h, "seconds": f"{res[1]:.4f}"}
        if res[0] is None:
            row["error"] = res[2]
            print(f"{pair.dataset}/{pair.name} {name}: ERROR {res[2]}", file=sys.stderr)
        else:
            row.update(score(res[0], pair.gt_flow, pair.gt_valid, pair.gt_noc, shared))
            print(f"{pair.dataset}/{pair.name} {name:24s} cov {row['coverage']:.3f}  "
                  f"epe {row['epe']:7.2f}  pck3 {row['pck3']:.3f}  {res[1]:6.2f}s", flush=True)
        writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})

    if vis:
        d = args.out.parent / "vis" / pair.dataset / pair.name
        d.mkdir(parents=True, exist_ok=True)
        max_mag = float(np.linalg.norm(pair.gt_flow, axis=-1)[pair.gt_valid].max(initial=1.0))
        cv2.imwrite(str(d / "img1.png"), pair.img1[:, :, ::-1])
        cv2.imwrite(str(d / "img2.png"), pair.img2[:, :, ::-1])
        gt_vis = pair.gt_flow.copy()
        gt_vis[~pair.gt_valid] = np.nan
        cv2.imwrite(str(d / "gt.png"), flow_to_color(gt_vis, max_mag))
        for name in compute:
            if cache[name][0] is not None:
                cv2.imwrite(str(d / f"{name}.png"), flow_to_color(cache[name][0], max_mag))

    # `get` calls itself, so it sits in a reference cycle (get -> its closure cell ->
    # get) that keeps everything it closes over, `cache` and `pair`, alive until the
    # cyclic GC runs, which can take many pairs. That piled up 100+ MB per pair until
    # the machine ran out of memory and froze. Deleting the name breaks the cycle.
    del get


if __name__ == "__main__":
    sys.exit(main())
