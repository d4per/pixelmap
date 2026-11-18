# Benchmarks

This folder compares pixelmap with the dense optical flow methods in OpenCV on public datasets with ground truth: DIS, Farneback, DeepFlow, RLOF and PCAFlow. It exists to answer one question with numbers: is pixelmap good enough, and different enough, to be worth contributing to OpenCV?

Nothing in this folder is packaged into the crates.

## Setup

```bash
cd benchmarks
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./download.sh hpatches eth3d          # ~1.3 GB; `all` adds KITTI 2015 and Sintel (~7.3 GB more)
```

Datasets go to `$PIXELMAP_BENCH_DATA`, which defaults to `~/datasets/pixelmap-bench`. Each dataset has its own licence; `download.sh` links them all.

## Run

```bash
.venv/bin/python run.py --datasets synthetic                       # seconds, no download
.venv/bin/python run.py --datasets hpatches eth3d kitti15 sintel_clean sintel_final
.venv/bin/python run.py --datasets hpatches --threads 1           # single-threaded timings
.venv/bin/python report.py                                         # markdown tables
```

`run.py` appends rows to `results/results.csv` and skips rows it already has, so an interrupted run resumes where it stopped. `--save-vis N` writes flow images (hue = direction, brightness = length, black = no answer) for the first N pairs of each dataset to `results/vis/`.

| dataset | pairs | what it tests |
|---|---|---|
| `synthetic` | 9 | photos from `../images` under random homographies. Exact ground truth, needs no download; use it as the smoke test |
| `hpatches` | 295 | the viewpoint sequences (`v_*`), image 1 against 2–6 of the same sequence. Planar scenes with exact homography ground truth and large viewpoint changes |
| `eth3d` | 27 | rectified two-view stereo in real indoor and outdoor scenes. Disparities are small (under ~25 px) |
| `kitti15` | 200 | driving scenes with large forward motion; sparse LiDAR ground truth |
| `sintel_clean`, `sintel_final` | 105 each | rendered film frames with every 10th pair taken; the final pass adds motion blur and fog |

## Methods

| method | notes |
|---|---|
| `pixelmap_low` / `_medium` / `_high` / `_ultra` | `pixelmap.correspond`, default seed and round-trip threshold. Leaves the pixels pixelmap did not map empty (above `low`, it fills its own holes from coarser steps first). `_ultra` differs from `_high` only on photos wider than 1600 px, so it is not in the default run |
| `pixelmap_*_filled` | the same, with holes filled from the nearest answered pixel, for comparison against methods that answer at every pixel |
| `dis_fast`, `dis_medium` | `cv2.DISOpticalFlow` presets |
| `farneback` | `cv2.calcOpticalFlowFarneback`, 6 levels, window 15 |
| `deepflow`, `pcaflow` | `cv2.optflow`, defaults |
| `rlof` | `cv2.optflow.calcOpticalFlowDenseRLOF`, defaults. It writes zero flow where it finds no match |
| `gt`, `zero` | checks, not methods. `gt` must score 0 EPE and 100 % coverage; `zero` must score an EPE equal to `gt_mag` |

The OpenCV baselines run with their default parameters, which is what an OpenCV user would get.

## Metrics

Each metric is computed per pair and then averaged over the dataset.

- **coverage**: the share of ground-truth pixels the method answered. Every OpenCV method answers 100 %.
- **EPE**: mean end-point error in pixels over the answered pixels. **EPE med** is the median.
- **EPE shared**: EPE over the pixels where *pixelmap_medium* answered. It shows how accurate each method is on the same pixels.
- **Fl**: the KITTI outlier rate, i.e. the share of pixels with error > 3 px and > 5 % of the true motion.
- **PCK@k**: the share of *all* ground-truth pixels whose error is below k px, with unanswered pixels counted as misses. This is the fairest single number, because declining to answer cannot raise it.
- **noc**: the same metrics restricted to non-occluded pixels, where the dataset provides them.
- **s/pair**: wall time per pair, measured after a warm-up call. Both libraries use `--threads` threads.
