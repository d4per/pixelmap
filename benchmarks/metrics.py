"""Per-pair scores.

Notation: G = pixels with ground truth, P = pixels the method answered (finite flow),
S = the shared mask, i.e. the reference method's P ∩ G.

- coverage      |P∩G| / |G|
- epe, epe_med  mean / median end-point error over P∩G
- fl            KITTI outlier rate over P∩G: EPE > 3 px and > 5 % of |gt|
- pck1..pck10   fraction of G with EPE below k px, where unanswered pixels count as
                misses. This is the headline number: refusing to answer cannot inflate it.
- epe_shared    mean EPE over S ∩ P. For a method that answers everywhere, that is S.
- *_noc         the same over the non-occluded subset, where the dataset has one
- gt_mag        mean |gt| over G, which the `zero` method's EPE must equal
"""

from __future__ import annotations

from typing import Optional

import numpy as np

PCK = (1, 3, 5, 10)


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


def _median(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


def score(flow: np.ndarray, gt: np.ndarray, valid: np.ndarray,
          noc: Optional[np.ndarray] = None, shared: Optional[np.ndarray] = None) -> dict:
    answered = np.all(np.isfinite(flow), axis=-1)
    err = np.linalg.norm(np.where(answered[..., None], flow, 0) - gt, axis=-1)
    mag = np.linalg.norm(gt, axis=-1)

    def block(region: np.ndarray, suffix: str = "") -> dict:
        n = int(region.sum())
        hit = answered & region
        e = err[hit]
        out = {
            f"coverage{suffix}": hit.sum() / n if n else float("nan"),
            f"epe{suffix}": _mean(e),
            f"epe_med{suffix}": _median(e),
            f"fl{suffix}": _mean(((e > 3) & (e > 0.05 * mag[hit])).astype(np.float32)),
        }
        for k in PCK:
            out[f"pck{k}{suffix}"] = ((err < k) & hit).sum() / n if n else float("nan")
        return out

    row = {"gt_pixels": int(valid.sum()), "gt_mag": _mean(mag[valid])}
    row.update(block(valid))
    if noc is not None:
        row.update({k: v for k, v in block(noc, "_noc").items() if not k.startswith("epe_med")})
    if shared is not None:
        s = shared & answered & valid
        row["epe_shared"] = _mean(err[s])
        row["shared_frac"] = s.sum() / max(int(valid.sum()), 1)
    return row
