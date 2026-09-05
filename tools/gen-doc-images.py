#!/usr/bin/env python3
"""Generate the figures embedded in pixelmap's rustdoc.

Reads the full-size PNGs in images/, writes web-sized WebP derivatives to
images/docs/, and emits one markdown fragment per figure into
rust/pixelmap/doc/. Each fragment carries the image inline as a data: URI so the
published crate is self-contained: the docs render on docs.rs, in an offline
`cargo doc`, and in every archived version, with nothing fetched over the network.

This script lives outside rust/pixelmap/ on purpose, so it is not packaged into
the crate. Re-run it after changing a source image, then commit both the .webp
under images/docs/ and the regenerated .md under rust/pixelmap/doc/.
"""

import base64
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "images"
WEBP_DIR = ROOT / "images" / "docs"
DOC_DIR = ROOT / "rust" / "pixelmap" / "doc"

# Anything above this is too heavy to embed; see the size budget in the README.
MAX_WEBP_BYTES = 200 * 1024

FIGURES = [
    {
        "source": "apa_3.png",
        "stem": "apa",
        "doc": "results.md",
        "width": 1400,
        "alt": (
            "Two photographs of a monkey statue taken from different positions, above "
            "three renderings of the correspondence recovered between them"
        ),
        "caption": (
            "Two views of the same statue (top) and the correspondence recovered "
            "between them, rendered at three settings (bottom). Regions left blank "
            "are those with no accepted match."
        ),
    },
    {
        "source": "tree_scale.png",
        "stem": "tree_scale",
        "doc": "quality-scales.md",
        "width": 900,
        "alt": (
            "The same photo pair solved at several correspondence grid sizes, coarse "
            "at the top and fine at the bottom"
        ),
        "caption": (
            "One photo pair solved across a range of correspondence grid sizes. "
            "Coverage fills in as the grid is refined; the gaps are cells left "
            "without an accepted correspondence."
        ),
    },
    {
        "source": "model3D.png",
        "stem": "model_3d",
        "doc": "model-3d.md",
        "width": 750,
        "alt": (
            "A shaded 3D surface of a monkey statue reconstructed from a single "
            "correspondence map"
        ),
        "caption": (
            "The surface [`Model3D`](crate::model_3d::Model3D) recovers from one "
            "correspondence map, shaded to show the geometry alone. Each grid point "
            "also carries the texture coordinates it came from."
        ),
    },
]


def trim_white(im, tolerance=12, pad=8):
    """Crop the uniform white border off a figure.

    Several sources are white-page exports with a lot of empty margin -
    tree_scale.png is nearly half blank - and every trimmed pixel is base64 we
    would otherwise ship inside the crate.
    """
    blank = Image.new("RGB", im.size, (255, 255, 255))
    mask = ImageChops.difference(im, blank).convert("L").point(lambda p: 255 if p > tolerance else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return im
    left, upper, right, lower = bbox
    return im.crop((
        max(left - pad, 0),
        max(upper - pad, 0),
        min(right + pad, im.width),
        min(lower + pad, im.height),
    ))


def encode(fig):
    """Downscale one source image to WebP and return its bytes."""
    src = SRC_DIR / fig["source"]
    out = WEBP_DIR / f"{fig['stem']}.webp"

    # The sources are RGBA with an opaque alpha channel; dropping it saves bytes
    # and avoids a pointless alpha plane in the WebP.
    im = Image.open(src).convert("RGB")
    im = trim_white(im)
    if im.width > fig["width"]:
        height = round(im.height * fig["width"] / im.width)
        im = im.resize((fig["width"], height), Image.LANCZOS)

    # method=6 is the slowest, smallest setting; this runs by hand, not in CI.
    im.save(out, "WEBP", quality=80, method=6)
    data = out.read_bytes()
    print(f"  {src.name:16} -> {out.name:16} {im.width}x{im.height}  {len(data) / 1024:6.1f} KiB")
    return data


def write_fragment(fig, data):
    """Write the markdown fragment that include_str! pulls into the docs.

    Deliberately free of fenced code blocks: markdown included with
    #[doc = include_str!(..)] has its code blocks compiled as doctests.
    """
    uri = "data:image/webp;base64," + base64.b64encode(data).decode("ascii")
    path = DOC_DIR / fig["doc"]
    path.write_text(
        f'<img src="{uri}"\n'
        f'     alt="{fig["alt"]}"\n'
        f'     style="max-width:100%">\n'
        f"\n"
        f"{fig['caption']}\n"
    )
    return path


def main():
    WEBP_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    print("Encoding figures:")
    encoded = [(fig, encode(fig)) for fig in FIGURES]

    print("\nWriting fragments:")
    total = 0
    oversized = []
    for fig, data in encoded:
        path = write_fragment(fig, data)
        total += path.stat().st_size
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:6.1f} KiB")
        if len(data) > MAX_WEBP_BYTES:
            oversized.append((fig["stem"], len(data)))

    print(f"\nTotal embedded in the crate: {total / 1024:.1f} KiB")
    for stem, size in oversized:
        print(f"WARNING: {stem}.webp is {size / 1024:.1f} KiB, over the {MAX_WEBP_BYTES / 1024:.0f} KiB budget")


if __name__ == "__main__":
    main()
