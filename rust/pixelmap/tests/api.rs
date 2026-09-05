//! Integration tests exercising the crate the way a downstream user would: only through
//! its public API, with no access to internals.

use std::sync::Arc;

use pixelmap::{
    correspond, Correspondence, DecodeError, DensePhotoMap, Error, Photo, Quality, DEFAULT_SEED,
};

/// Deterministic xorshift, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
}

/// A texture with enough local structure for the feature matcher to lock onto: a smooth
/// interference pattern so neighbouring pixels differ, plus noise so distant regions do
/// not look alike.
fn texture(width: usize, height: usize) -> Vec<[u8; 4]> {
    let mut rng = Rng(0x1234_5678);
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let wave = 127.0
                + 100.0 * ((x as f32) / 9.0).sin() * ((y as f32) / 11.0).cos()
                + (rng.next_u32() % 70) as f32
                - 35.0;
            let v = wave.clamp(0.0, 255.0) as u8;
            pixels.push([
                v,
                (v as f32 * 0.7 + 40.0).clamp(0.0, 255.0) as u8,
                255 - v,
                255,
            ]);
        }
    }
    pixels
}

/// Two windows onto the same texture, offset by `(dx, dy)`.
///
/// `b` samples the base image `(dx, dy)` further along than `a` does, so whatever `a`
/// shows at `(x, y)` sits at `(x - dx, y - dy)` in `b`: the forward mapping is a
/// translation of `(-dx, -dy)`.
fn shifted_pair(width: usize, height: usize, dx: usize, dy: usize) -> (Photo, Photo) {
    let (bw, bh) = (width + dx, height + dy);
    let base = texture(bw, bh);

    let window = |ox: usize, oy: usize| {
        let mut data = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&base[(y + oy) * bw + (x + ox)]);
            }
        }
        Photo::from_rgba(width, height, data).expect("buffer matches the dimensions")
    };

    (window(0, 0), window(dx, dy))
}

fn run(photo1: Photo, photo2: Photo, seed: u64) -> Correspondence {
    Correspondence::builder()
        .quality(Quality::Low)
        .seed(seed)
        .run(photo1, photo2)
        .expect("synthetic photos are valid input")
}

/// The headline claim: given a known warp, the pipeline recovers it.
#[test]
fn recovers_a_known_translation() {
    let (dx, dy) = (6usize, 3usize);
    let (photo1, photo2) = shifted_pair(420, 300, dx, dy);
    let mapping = run(photo1, photo2, DEFAULT_SEED);

    // `lookup` answers in source coordinates, so the expected shift is the one we built
    // the pair with — no working-resolution conversion for the caller to get right.
    let (expected_dx, expected_dy) = (-(dx as f64), -(dy as f64));

    let (mut sum_dx, mut sum_dy, mut mapped) = (0.0f64, 0.0f64, 0usize);
    let (w, h) = (420usize, 300usize);
    for y in (0..h).step_by(3) {
        for x in (0..w).step_by(3) {
            let Some((mx, my)) = mapping.lookup(x as f32, y as f32) else {
                continue;
            };
            sum_dx += (mx - x as f32) as f64;
            sum_dy += (my - y as f32) as f64;
            mapped += 1;
        }
    }

    let sampled = (0..h).step_by(3).len() * (0..w).step_by(3).len();
    let coverage = mapped as f32 / sampled as f32;
    assert!(
        coverage > 0.8,
        "only {:.1}% of the image was mapped",
        coverage * 100.0
    );

    let (mean_dx, mean_dy) = (sum_dx / mapped as f64, sum_dy / mapped as f64);
    assert!(
        (mean_dx - expected_dx).abs() < 0.5,
        "recovered dx {mean_dx:.2}, expected {expected_dx:.2}"
    );
    assert!(
        (mean_dy - expected_dy).abs() < 0.5,
        "recovered dy {mean_dy:.2}, expected {expected_dy:.2}"
    );
}

/// Reproducibility is a contract, not an accident: same input, same seed, same mapping.
#[test]
fn the_same_seed_reproduces_the_same_mapping() {
    let (a1, b1) = shifted_pair(200, 150, 5, 2);
    let (a2, b2) = shifted_pair(200, 150, 5, 2);

    let first = run(a1, b1, 4242).forward().serialize();
    let second = run(a2, b2, 4242).forward().serialize();

    assert_eq!(
        first, second,
        "identical inputs and seed produced different mappings"
    );
}

/// ...and that the seed is actually load-bearing, so the test above is not passing for
/// the trivial reason that the solver ignores it.
#[test]
fn different_seeds_explore_differently() {
    let (a1, b1) = shifted_pair(200, 150, 5, 2);
    let (a2, b2) = shifted_pair(200, 150, 5, 2);

    assert_ne!(
        run(a1, b1, 1).forward().serialize(),
        run(a2, b2, 999_999).forward().serialize(),
        "the seed had no effect on the result"
    );
}

/// The whole point of `Arc` over `Rc`: a caller can move the work onto another thread.
#[test]
fn a_mapping_can_be_computed_on_a_worker_thread() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}

    assert_send_sync::<Photo>();
    assert_send_sync::<DensePhotoMap>();
    assert_send_sync::<Correspondence>();
    assert_send_sync::<Error>();
    assert_send::<pixelmap::PixelMapProcessor>();

    let (photo1, photo2) = shifted_pair(200, 150, 4, 2);
    let handle = std::thread::spawn(move || run(photo1, photo2, DEFAULT_SEED).coverage());
    let coverage = handle.join().expect("worker thread panicked");
    assert!(coverage > 0.5, "coverage was only {coverage:.2}");
}

/// Two pairs mapped concurrently must not interfere. This is what the old process-global
/// seed counter made impossible to guarantee.
#[test]
fn concurrent_runs_do_not_influence_each_other() {
    let alone = {
        let (a, b) = shifted_pair(200, 150, 5, 2);
        run(a, b, 77).forward().serialize()
    };

    let handles: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(move || {
                let (a, b) = shifted_pair(200, 150, 5, 2);
                run(a, b, 77).forward().serialize()
            })
        })
        .collect();

    for handle in handles {
        let concurrent = handle.join().expect("worker thread panicked");
        assert_eq!(concurrent, alone, "a concurrent run changed the result");
    }
}

/// A round trip through the serialization format must preserve every mapped coordinate.
#[test]
fn serialization_round_trips() {
    let (photo1, photo2) = shifted_pair(200, 150, 5, 2);
    let (p1, p2) = (Arc::new(photo1), Arc::new(photo2));
    let map = Correspondence::builder()
        .seed(DEFAULT_SEED)
        .run(p1.clone(), p2.clone())
        .expect("synthetic photos are valid input")
        .into_parts()
        .0;

    let restored = DensePhotoMap::deserialize(&map.serialize(), p1, p2)
        .expect("a freshly serialized mapping round trips");

    assert_eq!(restored.grid_dimensions(), map.grid_dimensions());
    assert_eq!(restored.grid_cell_size(), map.grid_cell_size());
    let (grid_width, grid_height) = map.grid_dimensions();
    for y in 0..grid_height {
        for x in 0..grid_width {
            let (ax, ay) = map.grid_coordinates(x, y);
            let (bx, by) = restored.grid_coordinates(x, y);
            assert_eq!(ax.is_nan(), bx.is_nan(), "validity differs at ({x}, {y})");
            if !ax.is_nan() {
                assert_eq!((ax, ay), (bx, by), "coordinates differ at ({x}, {y})");
            }
        }
    }
}

/// A small hand-built mapping, cheap enough to serialize in every decoder test below.
fn tiny_map() -> DensePhotoMap {
    let photo = Arc::new(Photo::from_rgba(64, 48, vec![0; 64 * 48 * 4]).unwrap());
    let mut map = DensePhotoMap::new(photo.clone(), photo, 5, 4);
    map.set_grid_coordinates(1, 1, 12.0, 34.0);
    map
}

fn photos() -> (Arc<Photo>, Arc<Photo>) {
    let photo = Arc::new(Photo::from_rgba(64, 48, vec![0; 64 * 48 * 4]).unwrap());
    (photo.clone(), photo)
}

fn decode(bytes: &[u8]) -> Result<DensePhotoMap, Error> {
    let (p1, p2) = photos();
    DensePhotoMap::deserialize(bytes, p1, p2)
}

/// The error from decoding `bytes`, which every test here expects there to be.
fn decode_err(bytes: &[u8]) -> Error {
    decode(bytes).expect_err("expected a decode error")
}

/// The serialized bytes must be self-describing: anything that is not a mapping written
/// by this crate has to be rejected before it is read as a grid dimension.
#[test]
fn decoding_rejects_data_that_is_not_a_mapping() {
    assert_eq!(decode_err(b""), Error::Decode(DecodeError::NotAMapping));
    assert_eq!(decode_err(b"PX"), Error::Decode(DecodeError::NotAMapping));
    assert_eq!(
        decode_err(b"not a mapping at all, but long enough to hold a header"),
        Error::Decode(DecodeError::NotAMapping)
    );

    // The old headerless format, which began directly with the grid width.
    let mut headerless = Vec::new();
    headerless.extend_from_slice(&5u64.to_le_bytes());
    headerless.extend_from_slice(&4u64.to_le_bytes());
    headerless.extend_from_slice(&16u64.to_le_bytes());
    assert_eq!(
        decode_err(&headerless),
        Error::Decode(DecodeError::NotAMapping)
    );
}

/// A future version of the format must be refused by name, not misread as this one.
#[test]
fn decoding_rejects_an_unsupported_version() {
    let mut bytes = tiny_map().serialize();
    bytes[4..6].copy_from_slice(&99u16.to_le_bytes());

    assert_eq!(
        decode_err(&bytes),
        Error::Decode(DecodeError::UnsupportedVersion {
            found: 99,
            supported: 1,
        })
    );
}

/// Every truncation of a valid encoding must come back as an error. This is the property
/// `DecodeError`'s documentation promises: no input, however malformed, can panic.
#[test]
fn decoding_never_panics_on_a_truncated_mapping() {
    let bytes = tiny_map().serialize();

    for len in 0..bytes.len() {
        let err = decode(&bytes[..len]).expect_err("a truncated mapping is not decodable");
        assert!(
            matches!(
                err,
                Error::Decode(DecodeError::NotAMapping | DecodeError::Truncated { .. })
            ),
            "truncating to {len} bytes gave {err}"
        );
    }

    // Whole and untruncated, it still decodes.
    assert!(decode(&bytes).is_ok());
}

/// Trailing bytes mean the payload does not match the header that describes it, which is
/// a corrupt stream rather than a mapping with something appended.
#[test]
fn decoding_rejects_a_payload_that_does_not_match_the_header() {
    let mut bytes = tiny_map().serialize();
    bytes.push(0);

    assert!(matches!(
        decode_err(&bytes),
        Error::Decode(DecodeError::InvalidHeader { .. })
    ));
}

/// A header can be complete and still describe a grid that cannot exist. Those dimensions
/// are used for indexing, so they are validated before anything is allocated from them.
#[test]
fn decoding_rejects_impossible_grid_dimensions() {
    let with_dimensions = |width: u64, height: u64| {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PXMP");
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&16u64.to_le_bytes());
        bytes
    };

    for (width, height) in [(0, 4), (1, 4), (5, 0)] {
        assert!(
            matches!(
                decode_err(&with_dimensions(width, height)),
                Error::Decode(DecodeError::InvalidHeader { .. })
            ),
            "a {width}x{height} grid was accepted"
        );
    }

    // A grid whose cell count overflows must be rejected on the header alone, without
    // trying to allocate for it.
    assert!(matches!(
        decode_err(&with_dimensions(u64::MAX, u64::MAX)),
        Error::Decode(DecodeError::InvalidHeader { .. })
    ));
}

/// Reading past the right edge of the grid must report "no mapping", not the first cell
/// of the next row. Interpolation samples the corner at `x + 1`, so a flat-index-only
/// bounds check wrapped the whole right-hand column onto the opposite side of the image.
#[test]
fn the_right_edge_does_not_wrap_onto_the_next_row() {
    let photo = Arc::new(Photo::from_rgba(64, 48, vec![0; 64 * 48 * 4]).unwrap());
    let mut map = DensePhotoMap::new(photo.clone(), photo, 9, 7);

    // A distinctive value in the first cell of row 1 — the cell a wrapped read lands on.
    map.set_grid_coordinates(0, 1, 1234.0, 5678.0);

    let (grid_width, _) = map.grid_dimensions();
    let (x, y) = map.grid_coordinates(grid_width, 0);
    assert!(
        x.is_nan() && y.is_nan(),
        "reading past the right edge returned ({x}, {y})"
    );

    // Out-of-range writes are dropped rather than corrupting a wrapped cell.
    map.set_grid_coordinates(grid_width, 0, -1.0, -1.0);
    assert_eq!(map.grid_coordinates(0, 1), (1234.0, 5678.0));
}

/// Bad input is reported, not panicked on. These are the paths a caller is most likely to
/// hit first, so they have to fail in a way that can be handled.
#[test]
fn invalid_input_is_reported_as_an_error() {
    let (good, _) = shifted_pair(64, 64, 2, 2);

    // A buffer that does not match the dimensions it claims.
    assert_eq!(
        Photo::from_rgba(4, 4, vec![0; 10]),
        Err(Error::BufferLength {
            expected: 64,
            actual: 10
        })
    );

    // Photos of different sizes have no common grid to map over.
    let (other, _) = shifted_pair(80, 64, 2, 2);
    assert_eq!(
        correspond(good.clone(), other).unwrap_err(),
        Error::SizeMismatch {
            first: (64, 64),
            second: (80, 64)
        }
    );

    // Too small for the feature detector's sampling disc.
    let tiny = Photo::from_rgba(8, 8, vec![0; 8 * 8 * 4]).unwrap();
    assert_eq!(
        correspond(tiny.clone(), tiny).unwrap_err(),
        Error::PhotoTooSmall {
            dimensions: (8, 8),
            minimum: pixelmap::MIN_DIMENSION
        }
    );

    // A photo with no pixels at all.
    let empty = Photo::from_rgba(0, 0, Vec::new()).unwrap();
    assert_eq!(
        correspond(empty.clone(), empty).unwrap_err(),
        Error::EmptyPhoto
    );

    // Errors are worth printing.
    assert!(!Error::EmptyPhoto.to_string().is_empty());
}

/// `from_rgb` fills in an opaque alpha channel rather than making the caller do it.
#[test]
fn rgb_input_gains_an_opaque_alpha_channel() {
    let photo = Photo::from_rgb(2, 1, &[1, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(photo.as_rgba(), &[1, 2, 3, 255, 4, 5, 6, 255]);
    assert_eq!(photo.pixel(1, 0), Some([4, 5, 6, 255]));
    assert_eq!(
        photo.pixel(2, 0),
        None,
        "out-of-bounds reads are None, not a sentinel"
    );
    assert!(Photo::from_rgb(2, 1, &[1, 2, 3]).is_err());
}

/// Upscaling used to divide by zero at a factor of two or more, which the `Low` schedule
/// reaches for any photo under 200 px wide.
#[test]
fn small_photos_do_not_panic_when_the_schedule_upscales_them() {
    let (photo1, photo2) = shifted_pair(64, 48, 3, 2);
    let mapping = correspond(photo1, photo2).expect("small photos are still valid input");
    assert!(
        mapping.working_scale() > 1.0,
        "this case should be an upscale"
    );
}

/// The progress callback fires once per schedule step, and reaches 1.0.
#[test]
fn progress_is_reported_for_every_step() {
    let (photo1, photo2) = shifted_pair(64, 48, 3, 2);
    let mut seen = Vec::new();
    Correspondence::builder()
        .run_with_progress(photo1, photo2, |p| {
            seen.push((p.step, p.total, p.fraction()))
        })
        .unwrap();

    assert_eq!(
        seen.len(),
        seen[0].1,
        "expected one callback per reported step"
    );
    assert_eq!(seen.first().unwrap().0, 1);
    assert_eq!(
        seen.last().unwrap().2,
        1.0,
        "the last callback should report completion"
    );
    assert!(
        seen.windows(2).all(|w| w[0].0 < w[1].0),
        "steps must increase"
    );
}
