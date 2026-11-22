"""Write pixelmap results in the official formats of the public leaderboards.

    python submit.py sintel  --split train --fill off   --densify nearest
    python submit.py kitti15 --split test  --fill plain --densify none
    python submit.py eth3d   --split train --fill plain --densify nearest --evaluate

`--split train` writes the training pairs, so the output can be scored against the ground
truth (`--evaluate`, or KITTI's own devkit) and the configuration chosen there. `--split
test` writes the hidden test pairs for upload. Choose on train, then run test once with
that configuration: every benchmark forbids tuning on the test set.

Output goes to `submissions/<benchmark>-<split>-<fill>-<densify>/`:

    sintel   clean/<scene>/frame_NNNN.flo, final/...      pack with the Bundler (--pack)
    kitti15  flow/NNNNNN_10.png                           zip with flow/ at the root
    eth3d    low_res_two_view/<scene>.pfm + <scene>.txt   zip with low_res_two_view/ at the root

plus `runs.csv` (seconds per pair) and `submission.json` (versions, hardware, settings) for
the web form. An interrupted run resumes: pairs whose output exists are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


class Frame(NamedTuple):
    out: str  # output path relative to the submission root
    img1: Path
    img2: Path


# --- writers ------------------------------------------------------------------------


def write_flo(path: Path, flow: np.ndarray) -> None:
    """Middlebury .flo: magic 202021.25, width, height, then (u, v) per pixel."""
    if not np.isfinite(flow).all():
        raise ValueError(f"{path}: .flo results must be dense")
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        np.array([202021.25], "<f4").tofile(f)
        np.array([w, h], "<i4").tofile(f)
        flow.astype("<f4").tofile(f)


def write_kitti_png(path: Path, flow: np.ndarray) -> None:
    """KITTI 16-bit PNG: R = u*64 + 2^15, G = v*64 + 2^15, B = 1 where valid."""
    valid = np.all(np.isfinite(flow), axis=-1)
    enc = np.zeros(flow.shape[:2] + (3,), np.uint16)
    uv = np.clip(np.nan_to_num(flow) * 64.0 + 2**15, 0, 2**16 - 1).astype(np.uint16)
    enc[..., 2] = uv[..., 0]  # cv2 writes BGR, so R is the last channel
    enc[..., 1] = uv[..., 1]
    enc[..., 0] = valid
    enc[~valid, 1:] = 0
    if not cv2.imwrite(str(path), enc):
        raise IOError(f"cannot write {path}")


def write_pfm(path: Path, image: np.ndarray) -> None:
    """Single-channel little-endian PFM, rows bottom to top."""
    h, w = image.shape
    with open(path, "wb") as f:
        f.write(f"Pf\n{w} {h}\n-1\n".encode())
        np.flipud(image).astype("<f4").tofile(f)


# --- pairs --------------------------------------------------------------------------


def _data() -> Path:
    import datasets

    return datasets.DATA


def sintel_frames(split: str) -> Iterator[Frame]:
    root = _data() / "sintel" / ("training" if split == "train" else "test")
    for render in ("clean", "final"):
        if not (root / render).is_dir():
            raise SystemExit(f"missing {root / render}; run ./download.sh sintel_test")
        for scene in sorted(p for p in (root / render).iterdir() if p.is_dir()):
            pngs = sorted(scene.glob("frame_*.png"))
            for a, b in zip(pngs, pngs[1:]):
                yield Frame(f"{render}/{scene.name}/{a.stem}.flo", a, b)


def kitti_frames(split: str) -> Iterator[Frame]:
    root = _data() / "kitti15" / ("training" if split == "train" else "testing") / "image_2"
    if not root.is_dir():
        raise SystemExit(f"missing {root}; run ./download.sh kitti15_test")
    for i in range(200):
        n = f"{i:06d}"
        yield Frame(f"flow/{n}_10.png", root / f"{n}_10.png", root / f"{n}_11.png")


def eth3d_frames(split: str) -> Iterator[Frame]:
    root = _data() / "eth3d" / ("two_view_training" if split == "train" else "two_view_test")
    if not root.is_dir():
        raise SystemExit(f"missing {root}; run ./download.sh eth3d_test")
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        yield Frame(f"low_res_two_view/{scene.name}.pfm", scene / "im0.png", scene / "im1.png")


FRAMES = {"sintel": sintel_frames, "kitti15": kitti_frames, "eth3d": eth3d_frames}


# --- running ------------------------------------------------------------------------


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _fill_nearest(flow: np.ndarray) -> np.ndarray:
    import methods

    return methods.fill_nearest(flow)


def compute(frame: Frame, quality: str, fill: str, densify: str) -> tuple[np.ndarray, float]:
    import pixelmap

    img1, img2 = _read_rgb(frame.img1), _read_rgb(frame.img2)
    start = time.perf_counter()
    mapping = pixelmap.correspond(img1, img2, quality=quality, fill_mode=fill)
    flow = np.asarray(mapping.flow(), dtype=np.float32)
    if densify == "nearest":
        flow = _fill_nearest(flow)
    return flow, time.perf_counter() - start


def write(benchmark: str, path: Path, flow: np.ndarray, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.part{path.suffix}")
    if benchmark == "sintel":
        write_flo(tmp, flow)
    elif benchmark == "kitti15":
        write_kitti_png(tmp, flow)
    else:
        # Left-to-right flow (u, 0) is disparity -u. Missing pixels are inf, which ETH3D
        # reads as "no answer".
        disp = np.where(np.isfinite(flow[..., 0]), -flow[..., 0], np.inf).astype(np.float32)
        write_pfm(tmp, disp)
        path.with_suffix(".txt").write_text(f"runtime {seconds:.3f}\n")
    tmp.replace(path)  # the result file appears last, so it marks a finished pair


def environment(args: argparse.Namespace, runs: list[dict]) -> dict:
    import pixelmap

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    secs = [float(r["seconds"]) for r in runs]
    return {
        "benchmark": args.benchmark,
        "split": args.split,
        "pairs": len(runs),
        "quality": args.quality,
        "fill_mode": args.fill,
        "densify": args.densify,
        "seed": "default",
        "pixelmap_python": getattr(pixelmap, "__version__", "unknown"),
        "pixelmap_module": pixelmap.__file__,
        "threads": int(os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1)),
        "cpu": cpu,
        "seconds_mean": round(sum(secs) / len(secs), 3) if secs else None,
        "seconds_total": round(sum(secs), 1),
    }


# --- evaluation on the training split -----------------------------------------------


def kitti_interpolate_background(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The KITTI devkit's `FlowImage::interpolateBackground` (devkit/cpp/io_flow.h),
    which the server applies to sparse results before scoring them.

    Per row, a gap with valid pixels on both sides takes the smaller of its two
    neighbours, per component. Then each row is extended outwards from its first and last
    valid pixel, and each column likewise, which fills rows that had nothing."""
    f = flow.astype(np.float32).copy()
    ok = valid.copy()
    h, w = ok.shape
    for v in range(h):
        row_ok = ok[v]
        idx = np.flatnonzero(row_ok)
        if idx.size == 0:
            continue
        # interior gaps: between consecutive valid pixels a and b with b > a + 1
        for a, b in zip(idx[:-1], idx[1:]):
            if b > a + 1:
                f[v, a + 1 : b] = np.minimum(f[v, a], f[v, b])
                ok[v, a + 1 : b] = True
        f[v, : idx[0]] = f[v, idx[0]]
        f[v, idx[-1] + 1 :] = f[v, idx[-1]]
        ok[v, :] = True
    for u in range(w):
        idx = np.flatnonzero(ok[:, u])
        if idx.size == 0:
            continue
        f[: idx[0], u] = f[idx[0], u]
        f[idx[-1] + 1 :, u] = f[idx[-1], u]
    return f



def evaluate(benchmark: str, root: Path) -> dict:
    """The leaderboard's own headline numbers, from the written files and the training
    ground truth. Missing pixels (sparse output) count as wrong."""
    import datasets

    rows = []
    if benchmark == "sintel":
        for render in ("clean", "final"):
            gt_root = datasets.DATA / "sintel" / "training" / "flow"
            for flo in sorted((root / render).glob("*/frame_*.flo")):
                gt = datasets.read_flo(gt_root / flo.parent.name / flo.name)
                epe = np.linalg.norm(datasets.read_flo(flo) - gt, axis=-1)
                rows.append({"set": render, "sum": float(epe.sum()), "n": epe.size})
        out = {}
        for render in ("clean", "final"):
            r = [x for x in rows if x["set"] == render]
            # The leaderboard's "EPE all" is over every pixel of every frame.
            out[f"{render}_epe_all"] = sum(x["sum"] for x in r) / max(1, sum(x["n"] for x in r))
            out[f"{render}_frames"] = len(r)
        return out
    if benchmark == "kitti15":
        tot = {"occ": [0, 0], "noc": [0, 0]}
        gt_root = datasets.DATA / "kitti15" / "training"
        for png in sorted((root / "flow").glob("*_10.png")):
            est, est_valid = datasets.read_kitti_flow(png)
            est = kitti_interpolate_background(est, est_valid)
            for key in tot:
                gt, valid = datasets.read_kitti_flow(gt_root / f"flow_{key}" / png.name)
                err = np.linalg.norm(gt - est, axis=-1)
                mag = np.linalg.norm(gt, axis=-1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    wrong = (err > 3.0) & (err / mag > 0.05)
                tot[key][0] += int(wrong[valid].sum())
                tot[key][1] += int(valid.sum())
        return {f"fl_{k}": 100.0 * e / max(1, n) for k, (e, n) in tot.items()} | {
            "fl_all_is": "fl_occ (the leaderboard's 'all pixels')"
        }
    if benchmark == "eth3d":
        gt_root = datasets.DATA / "eth3d" / "two_view_training_gt"
        tot = {"n": 0, "bad1": 0, "bad2": 0, "bad4": 0, "err": 0.0, "answered": 0}
        for pfm in sorted((root / "low_res_two_view").glob("*.pfm")):
            gt = datasets.read_pfm(gt_root / pfm.stem / "disp0GT.pfm")
            nocc = cv2.imread(str(gt_root / pfm.stem / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255
            m = nocc & np.isfinite(gt)
            est = datasets.read_pfm(pfm)[m]
            ans = np.isfinite(est)
            err = np.full(est.shape, np.inf, np.float32)  # a missing pixel is wrong
            err[ans] = np.abs(est[ans] - gt[m][ans])
            tot["n"] += int(m.sum())
            tot["answered"] += int(ans.sum())
            for k, t in (("bad1", 1), ("bad2", 2), ("bad4", 4)):
                tot[k] += int((err > t).sum())
            tot["err"] += float(err[ans].sum())
        n = max(1, tot["n"])
        return {
            "nocc_bad1": 100 * tot["bad1"] / n,
            "nocc_bad2": 100 * tot["bad2"] / n,
            "nocc_bad4": 100 * tot["bad4"] / n,
            "nocc_avgerr_answered": tot["err"] / max(1, tot["answered"]),
            "nocc_coverage": 100 * tot["answered"] / n,
        }
    raise ValueError(benchmark)


# --- packing ------------------------------------------------------------------------


def pack(benchmark: str, root: Path) -> Path:
    if benchmark == "sintel":
        data = _data() / "sintel"
        # The zip ships one Bundler per platform (linux-x64/, osx/, win/), all executable.
        system = platform.system()
        sub = {"Linux": "linux-x64/bundler", "Darwin": "osx/bundler", "Windows": "win/bundler.exe"}.get(system)
        bundler = data / "bundler" / sub if sub else None
        if bundler is None or not bundler.is_file():
            raise SystemExit(f"no Sintel bundler for {system} under {data / 'bundler'}")
        target = root.with_suffix(".lzma")
        subprocess.run([str(bundler), str(root / "clean"), str(root / "final"), str(target)], check=True)
        return target
    top = "flow" if benchmark == "kitti15" else "low_res_two_view"
    archive = shutil.make_archive(str(root), "zip", root_dir=root, base_dir=top)
    return Path(archive)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=list(FRAMES))
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--quality", default="high")
    p.add_argument("--fill", choices=["off", "plain", "checked"], required=True)
    p.add_argument("--densify", choices=["none", "nearest"], required=True)
    p.add_argument("--limit", type=int, help="at most N pairs (for trying the pipeline out)")
    p.add_argument("--evaluate", action="store_true", help="score the training output")
    p.add_argument("--pack", action="store_true", help="build the upload archive")
    p.add_argument("--out", type=Path, help="defaults to submissions/<benchmark>-<split>-<fill>-<densify>")
    args = p.parse_args()

    if args.benchmark == "sintel" and args.densify == "none":
        p.error("Sintel needs dense results: use --densify nearest")
    if args.evaluate and args.split != "train":
        p.error("--evaluate needs --split train: the test split has no ground truth here")
    sys.path.insert(0, str(HERE))
    root = args.out or HERE / "submissions" / f"{args.benchmark}-{args.split}-{args.fill}-{args.densify}"
    root.mkdir(parents=True, exist_ok=True)

    log_path = root / "runs.csv"
    runs = list(csv.DictReader(open(log_path))) if log_path.exists() else []
    with open(log_path, "a", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=["out", "seconds"])
        if not runs:
            writer.writeheader()
        frames = list(FRAMES[args.benchmark](args.split))[: args.limit]
        for i, frame in enumerate(frames):
            target = root / frame.out
            if target.exists():
                continue
            flow, seconds = compute(frame, args.quality, args.fill, args.densify)
            write(args.benchmark, target, flow, seconds)
            row = {"out": frame.out, "seconds": f"{seconds:.3f}"}
            writer.writerow(row)
            log.flush()
            runs.append(row)
            cov = float(np.all(np.isfinite(flow), axis=-1).mean())
            print(f"[{i + 1}/{len(frames)}] {frame.out}  cov {cov:.3f}  {seconds:.2f}s", flush=True)

    info = environment(args, runs)
    if args.evaluate:
        info["training_scores"] = evaluate(args.benchmark, root)
        print(json.dumps(info["training_scores"], indent=2))
    (root / "submission.json").write_text(json.dumps(info, indent=2) + "\n")
    if args.pack:
        print("packed", pack(args.benchmark, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
