"""Round trips of submit.py's writers through the readers datasets.py scores with.

    .venv/bin/python test_submit.py
"""

import tempfile
from pathlib import Path

import numpy as np

import datasets
import submit


def _flow(h=7, w=9, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-300, 300, (h, w, 2))).astype(np.float32)


def test_flo_round_trip(tmp: Path):
    flow = _flow()
    submit.write_flo(tmp / "a.flo", flow)
    assert np.array_equal(datasets.read_flo(tmp / "a.flo"), flow)


def test_flo_refuses_holes(tmp: Path):
    flow = _flow()
    flow[2, 3] = np.nan
    try:
        submit.write_flo(tmp / "b.flo", flow)
    except ValueError:
        return
    raise AssertionError("a .flo with holes was written")


def test_kitti_png_round_trip_within_its_precision(tmp: Path):
    flow = _flow()
    flow[1, 1] = np.nan
    submit.write_kitti_png(tmp / "a.png", flow)
    back, valid = datasets.read_kitti_flow(tmp / "a.png")
    assert not valid[1, 1] and valid.sum() == valid.size - 1
    assert np.abs(back[valid] - flow[valid]).max() <= 1 / 64


def test_pfm_round_trip(tmp: Path):
    disp = _flow()[..., 0]
    disp[0, 0] = np.inf
    submit.write_pfm(tmp / "a.pfm", disp)
    assert np.array_equal(datasets.read_pfm(tmp / "a.pfm"), disp)


def test_eth3d_disparity_is_minus_u(tmp: Path):
    flow = np.zeros((4, 5, 2), np.float32)
    flow[..., 0] = -3.5
    flow[0, 0] = np.nan
    submit.write("eth3d", tmp / "low_res_two_view" / "s.pfm", flow, 1.25)
    disp = datasets.read_pfm(tmp / "low_res_two_view" / "s.pfm")
    assert disp[1, 1] == 3.5 and np.isinf(disp[0, 0])
    assert (tmp / "low_res_two_view" / "s.txt").read_text() == "runtime 1.250\n"
    assert sorted(p.name for p in (tmp / "low_res_two_view").iterdir()) == ["s.pfm", "s.txt"]


def test_kitti_background_interpolation_matches_the_devkit(tmp: Path):
    # Row 0: a gap between 5 and 2 takes the smaller, 2; the ends extend outwards.
    # Row 1 has nothing, so it takes the column's value from row 0.
    flow = np.zeros((2, 6, 2), np.float32)
    valid = np.zeros((2, 6), bool)
    flow[0, 1], flow[0, 4] = (5, -1), (2, 3)
    valid[0, [1, 4]] = True
    out = submit.kitti_interpolate_background(flow, valid)
    assert out[0, :, 0].tolist() == [5, 5, 2, 2, 2, 2]
    assert out[0, :, 1].tolist() == [-1, -1, -1, -1, 3, 3]
    assert np.array_equal(out[1], out[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print("ok", name)
