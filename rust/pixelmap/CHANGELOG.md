# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

Initial release: the reference implementation of the PIXELMAP framework for dense image
correspondence.

- `Correspondence` and its `Builder` as the high-level entry point, with `correspond` as
  the one-call shorthand.
- `Quality` presets (`Low`, `Medium`, `High`) over an explicit `IterationParams` schedule,
  and `PixelMapProcessor` for callers driving the solver step by step.
- `DensePhotoMap` for the raw forward and backward correspondence grids, including a
  versioned serialization format that validates on decode.
- `Photo` as a plain RGBA buffer, with `From` conversions for the `image` crate's types
  behind the `image` feature.
- Deterministic results from a seed, identical with and without the `parallel` feature.
