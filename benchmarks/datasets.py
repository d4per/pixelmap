"""Ground-truth image pairs for the benchmark.

Every loader yields `Pair`s with the same layout, whatever the dataset stores on disk:

- `img1`, `img2`: uint8 RGB arrays of identical shape (H, W, 3). Pixelmap and OpenCV's
  dense flow methods both require equal sizes. Where a dataset's two images differ in
  size (HPatches), both are padded on the right and bottom, which leaves pixel
  coordinates unchanged.
- `gt_flow`: float32 (H, W, 2) holding (dx, dy), such that pixel (x, y) of img1 lands
  at (x + dx, y + dy) in img2.
- `gt_valid`: bool (H, W), where `gt_flow` is defined.
- `gt_noc`: bool (H, W) or None, the subset of `gt_valid` that is visible in both
  images (non-occluded). Only some datasets have it.

Datasets are read from `$PIXELMAP_BENCH_DATA` (default `~/datasets/pixelmap-bench`),
laid out as `download.sh` extracts them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("PIXELMAP_BENCH_DATA", "~/datasets/pixelmap-bench")).expanduser()


@dataclass
class Pair:
    dataset: str
    name: str
    img1: np.ndarray
    img2: np.ndarray
    gt_flow: np.ndarray
    gt_valid: np.ndarray
    gt_noc: Optional[np.ndarray] = None


class DatasetMissing(Exception):
    pass


def _require(path: Path, dataset: str) -> Path:
    if not path.exists():
        raise DatasetMissing(
            f"{dataset}: {path} not found. Run `benchmarks/download.sh {dataset.split('_')[0]}` "
            f"or set PIXELMAP_BENCH_DATA."
        )
    return path


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # grayscale files come back as 3 channels
    if img is None:
        raise IOError(f"cannot read {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs, ys


def _pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ph, pw = h - img.shape[0], w - img.shape[1]
    if ph == 0 and pw == 0:
        return img
    pad = [(0, ph), (0, pw)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad)


def _homography_flow(H: np.ndarray, h1: int, w1: int, h2: int, w2: int, h: int, w: int):
    """Dense flow of the homography H (img1 → img2 coordinates) on an (h, w) canvas.

    Valid where the source pixel lies inside img1's real (h1, w1) area and its image
    lands inside img2's real (h2, w2) area.
    """
    xs, ys = _grid(h, w)
    pts = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ H.T.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = pts[..., 0] / pts[..., 2]
        my = pts[..., 1] / pts[..., 2]
    flow = np.stack([mx - xs, my - ys], axis=-1).astype(np.float32)
    valid = (
        (xs < w1) & (ys < h1)
        & (pts[..., 2] > 0)
        & (mx >= 0) & (mx <= w2 - 1) & (my >= 0) & (my <= h2 - 1)
    )
    flow[~valid] = 0
    return flow, valid


# --- synthetic ----------------------------------------------------------------------


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    best, start, run_start = (0, 0), None, None
    for i, m in enumerate(np.append(mask, False)):
        if m and run_start is None:
            run_start = i
        elif not m and run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    return best


def _photo_bbox(region: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the photo inside a figure panel on a white background."""
    ink = (region < 235).any(axis=-1)
    y0, y1 = _longest_run(ink.mean(axis=1) > 0.3)
    x0, x1 = _longest_run(ink[y0:y1].mean(axis=0) > 0.6)
    my, mx = (y1 - y0) // 50, (x1 - x0) // 50  # trim antialiased edges
    return y0 + my, y1 - my, x0 + mx, x1 - mx


def synthetic(stride: int = 1) -> Iterator[Pair]:
    """Photos from the repo's images/, warped by known random homographies.

    Needs no download. Each source figure's top-left photo is cut out (without the
    figure's white margin and caption) and scaled to 960 px wide. It is then warped by
    a homography that moves each corner by up to 12% of the image size. img2 is the
    warp of img1, so the ground truth is exact wherever the warped point stays inside
    the frame.
    """
    rng = np.random.default_rng(1234)
    sources = ["apa_3.png", "staty_3.png", "trappa_3.png"]
    idx = 0
    for src in sources:
        full = _read_rgb(REPO / "images" / src)
        # The figures are composites with the two input photos on the top row.
        panel = full[: full.shape[0] // 2, : full.shape[1] // 2]
        y0, y1, x0, x1 = _photo_bbox(panel)
        crop = panel[y0:y1, x0:x1]
        w = 960
        h = round(w * crop.shape[0] / crop.shape[1])
        base = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        for k in range(3):
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            jitter = rng.uniform(-0.12, 0.12, size=(4, 2)) * np.float32([w, h])
            H = cv2.getPerspectiveTransform(corners, (corners + jitter).astype(np.float32))
            if idx % stride == 0:
                warped = cv2.warpPerspective(base, H, (w, h), flags=cv2.INTER_LINEAR)
                flow, valid = _homography_flow(H, h, w, h, w, h, w)
                yield Pair("synthetic", f"{Path(src).stem}_{k}", base, warped, flow, valid)
            idx += 1


# --- HPatches -----------------------------------------------------------------------


def hpatches(stride: int = 1) -> Iterator[Pair]:
    """HPatches viewpoint sequences (v_*): image 1 against images 2..6.

    Ground truth is the provided homography H_1_k, so it is exact for these planar
    scenes. Because the five pairs of a sequence span increasingly large viewpoint
    changes, the pair names end in the target index (2 = mildest, 6 = widest).
    """
    root = _require(DATA / "hpatches-sequences-release", "hpatches")
    idx = 0
    for seq in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("v_")):
        img1 = _read_rgb(seq / "1.ppm")
        for k in range(2, 7):
            if idx % stride == 0:
                img2 = _read_rgb(seq / f"{k}.ppm")
                H = np.loadtxt(seq / f"H_1_{k}")
                (h1, w1), (h2, w2) = img1.shape[:2], img2.shape[:2]
                h, w = max(h1, h2), max(w1, w2)
                flow, valid = _homography_flow(H, h1, w1, h2, w2, h, w)
                yield Pair("hpatches", f"{seq.name}_{k}", _pad_to(img1, h, w), _pad_to(img2, h, w), flow, valid)
            idx += 1


# --- ETH3D two-view -----------------------------------------------------------------


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        if header not in ("Pf", "PF"):
            raise IOError(f"{path}: not a PFM file")
        w, h = map(int, f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
    channels = 3 if header == "PF" else 1
    data = data.reshape(h, w, channels) if channels == 3 else data.reshape(h, w)
    return np.flipud(data).astype(np.float32)  # PFM rows run bottom to top


def eth3d(stride: int = 1) -> Iterator[Pair]:
    """ETH3D low-res two-view stereo, training split (27 rectified pairs).

    The left image's disparity d maps pixel x to x - d in the right image, so the flow
    is (-d, 0). mask0nocc marks the non-occluded pixels with 255.
    """
    img_root = _require(DATA / "eth3d" / "two_view_training", "eth3d")
    gt_root = _require(DATA / "eth3d" / "two_view_training_gt", "eth3d")
    for i, scene in enumerate(sorted(p for p in img_root.iterdir() if p.is_dir())):
        if i % stride:
            continue
        img1 = _read_rgb(scene / "im0.png")
        img2 = _read_rgb(scene / "im1.png")
        disp = read_pfm(gt_root / scene.name / "disp0GT.pfm")
        valid = np.isfinite(disp) & (disp > 0)
        flow = np.zeros(disp.shape + (2,), np.float32)
        flow[..., 0] = np.where(valid, -disp, 0)
        noc_path = gt_root / scene.name / "mask0nocc.png"
        noc = None
        if noc_path.exists():
            noc = (cv2.imread(str(noc_path), cv2.IMREAD_GRAYSCALE) == 255) & valid
        yield Pair("eth3d", scene.name, img1, img2, flow, valid, noc)


# --- KITTI 2015 ---------------------------------------------------------------------


def read_kitti_flow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # uint16, BGR order
    if raw is None:
        raise IOError(f"cannot read {path}")
    flow = (raw[:, :, [2, 1]].astype(np.float32) - 2**15) / 64.0
    valid = raw[:, :, 0] > 0
    flow[~valid] = 0
    return flow, valid


def kitti15(stride: int = 1) -> Iterator[Pair]:
    """KITTI 2015 scene flow, training split: frame 10 → 11 of 200 scenes.

    flow_occ is the evaluation ground truth (all pixels with a measurement); flow_noc
    is its non-occluded subset.
    """
    root = _require(DATA / "kitti15" / "training", "kitti15")
    for i in range(0, 200, stride):
        name = f"{i:06d}"
        flow, valid = read_kitti_flow(root / "flow_occ" / f"{name}_10.png")
        _, noc = read_kitti_flow(root / "flow_noc" / f"{name}_10.png")
        img1 = _read_rgb(root / "image_2" / f"{name}_10.png")
        img2 = _read_rgb(root / "image_2" / f"{name}_11.png")
        yield Pair("kitti15", name, img1, img2, flow, valid, noc & valid)


# --- MPI-Sintel ---------------------------------------------------------------------


def read_flo(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = np.fromfile(f, "<f4", 1)[0]
        if magic != 202021.25:
            raise IOError(f"{path}: bad .flo magic")
        w, h = np.fromfile(f, "<i4", 2)
        return np.fromfile(f, "<f4", int(w) * int(h) * 2).reshape(int(h), int(w), 2)


def _sintel(render: str, stride: int) -> Iterator[Pair]:
    root = _require(DATA / "sintel" / "training", f"sintel_{render}")
    idx = 0
    for scene in sorted(p.name for p in (root / "flow").iterdir() if p.is_dir()):
        for flo in sorted((root / "flow" / scene).glob("frame_*.flo")):
            if idx % stride == 0:
                n = int(re.search(r"(\d+)", flo.stem).group(1))
                img1 = _read_rgb(root / render / scene / f"frame_{n:04d}.png")
                img2 = _read_rgb(root / render / scene / f"frame_{n + 1:04d}.png")
                flow = read_flo(flo)
                valid = np.all(np.isfinite(flow), axis=-1) & (np.abs(flow).max(axis=-1) < 1e9)
                occ_path = root / "occlusions" / scene / f"frame_{n:04d}.png"
                noc = None
                if occ_path.exists():
                    noc = (cv2.imread(str(occ_path), cv2.IMREAD_GRAYSCALE) == 0) & valid
                yield Pair(f"sintel_{render}", f"{scene}_{n:04d}", img1, img2, flow, valid, noc)
            idx += 1


def sintel_clean(stride: int = 10) -> Iterator[Pair]:
    """MPI-Sintel training, clean pass. Every `stride`-th of the 1041 pairs."""
    return _sintel("clean", stride)


def sintel_final(stride: int = 10) -> Iterator[Pair]:
    """MPI-Sintel training, final pass (motion blur, fog). Every `stride`-th pair."""
    return _sintel("final", stride)


# Default strides are the loaders' own defaults; `--stride` overrides them.
DATASETS: dict[str, Callable[..., Iterator[Pair]]] = {
    "synthetic": synthetic,
    "hpatches": hpatches,
    "eth3d": eth3d,
    "kitti15": kitti15,
    "sintel_clean": sintel_clean,
    "sintel_final": sintel_final,
}
