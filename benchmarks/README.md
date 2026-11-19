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

## Results: HPatches

HPatches has no submission server, so these numbers are published here. The protocol is the
one dense-matching papers use for HPatches at original resolution: the 59 viewpoint
sequences (`v_*`), image 1 matched against images 2–6, ground truth from the provided
homographies. Each number is a mean over the 295 pairs. pixelmap ran at its defaults
(pixelmap 0.3.1 plus the Unreleased changes: `High` capped at the photo width and filling its
holes); the OpenCV methods ran at theirs, with 12 threads on an i7-9750H.

| method | pairs | coverage | AEPE | PCK-1 | PCK-3 | PCK-5 | s/pair |
|---|---|---|---|---|---|---|---|
| `pixelmap_high` | 295 | 77.1% | 7.14 | 37.7% | 65.1% | 70.9% | 3.21 |
| `pixelmap_medium` | 295 | 76.3% | 16.83 | 34.7% | 64.0% | 69.8% | 1.46 |
| `pixelmap_low` | 295 | 70.8% | 12.96 | 26.7% | 58.6% | 64.6% | 0.43 |
| `pixelmap_high_filled` | 295 | 90.8% | 14.97 | 38.1% | 67.0% | 74.0% | 3.24 |
| `pixelmap_medium_filled` | 295 | 92.2% | 27.15 | 35.1% | 66.0% | 73.0% | 1.49 |
| `dis_medium` | 295 | 100.0% | 154.62 | 14.1% | 25.1% | 28.1% | 0.05 |
| `dis_fast` | 295 | 100.0% | 166.73 | 7.9% | 16.6% | 19.5% | 0.02 |
| `deepflow` | 295 | 100.0% | 178.42 | 9.6% | 13.3% | 14.2% | 2.25 |
| `farneback` | 295 | 100.0% | 190.40 | 2.5% | 4.1% | 4.7% | 0.26 |
| `pcaflow` | 295 | 100.0% | 190.50 | 0.7% | 2.1% | 3.0% | 0.07 |
| `rlof` | 169 (126 failed) | 100.0% | 157.22 | 4.2% | 9.4% | 10.6% | 0.46 |

How to read it:

- **PCK-k** is the share of all ground-truth pixels within k px, with unanswered pixels
  counted as misses. It is the column to compare across methods.
- **AEPE** is over the pixels each method answered. A method that declines hard pixels gets a
  lower AEPE than one that answers everything, so compare it only between methods with the
  same coverage.
- **pixelmap gives no answer at all on 27 of the 295 pairs** (High), mostly the
  strongest viewpoint changes (images 4–6). Those pairs count as zero in PCK, are left out of
  AEPE, and are why the `_filled` rows reach 90.8% coverage rather than 100%: with nothing
  answered there is nothing to fill from.
- The OpenCV methods are optical flow methods built for small motion, so HPatches' large
  viewpoint changes are far outside what they were designed for. Learned dense matchers
  report on this same protocol in their papers. Those numbers come from the papers, not from
  runs here, and their image sizes and averaging may differ, so check before comparing.

Regenerate the numbers with `python run.py --datasets hpatches` and `python report.py`.

## Submitting to the leaderboards

MPI Sintel, KITTI 2015 and ETH3D rank results on hidden test sets through their own
websites. `submit.py` writes pixelmap's output in each one's official format and packs it;
`python submit.py --help` lists the layouts, and `test_submit.py` checks the writers.
Download the test splits with `./download.sh test`.

The configuration was chosen on the **training** splits with each leaderboard's own metric,
`quality="high"` throughout:

| fill / densify | Sintel EPE clean / final (1041 pairs each) | KITTI Fl-all | ETH3D bad-2 (nocc) |
|---|---|---|---|
| **off / nearest** | **3.00 / 4.74** | **26.4%** | 14.68% |
| plain / nearest | 3.08 / 4.80 | 27.2% | **14.64%** |
| off / none | — (must be dense) | 28.9% | 22.7% |
| plain / none | — (must be dense) | 28.5% | 18.5% |

- **Submit `--fill off --densify nearest` to all three.** It wins on Sintel and KITTI and ties
  on ETH3D, where the 0.04-point gap is noise over 27 pairs. One configuration also keeps
  the method description the same on every leaderboard. pixelmap's own hole filling (`plain`,
  the High default) adds answers that are less accurate than the nearest pixel it would
  otherwise be filled from, so it helps PCK and hurts EPE and Fl.
- Filling ourselves beats leaving holes to the server: KITTI's own background interpolation
  scores 28.9% against 26.4%, and ETH3D counts missing pixels as wrong.
- KITTI Fl-all is computed by `submit.py`'s port of the devkit's interpolation and scoring,
  because the devkit's evaluator only runs in the server's directory layout.
- About 1 s per pair at these sizes, 12 threads on an i7-9750H.

Run the test split **once** with that configuration, then upload:

```bash
.venv/bin/python submit.py sintel  --split test --fill off --densify nearest --pack
.venv/bin/python submit.py kitti15 --split test --fill off --densify nearest --pack
.venv/bin/python submit.py eth3d   --split test --fill off --densify nearest --pack
```

Every benchmark forbids tuning on the test set, and KITTI and ETH3D accept a single upload
per method, so do not change the configuration after seeing a test score.
