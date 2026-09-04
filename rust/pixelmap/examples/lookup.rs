//! Query a correspondence field at individual points, with no image codecs involved.
//!
//! ```sh
//! cargo run --release --example lookup
//! ```
//!
//! Builds two synthetic photos related by a known shift, maps them, and prints what the
//! algorithm recovered — a self-contained tour of the API that needs no input files.

use pixelmap::{correspond, Photo};

const WIDTH: usize = 240;
const HEIGHT: usize = 180;
const SHIFT: usize = 5;

fn main() -> Result<(), pixelmap::Error> {
    let (photo1, photo2) = shifted_pair();

    let mapping = correspond(photo1, photo2)?;
    println!(
        "mapped {:.1}% of the image in {} comparisons",
        mapping.coverage() * 100.0,
        mapping.comparisons()
    );
    println!("working scale: {:.3}", mapping.working_scale());
    println!("\n  point        maps to        (expected)");

    for (x, y) in [(40.0, 40.0), (120.0, 90.0), (200.0, 140.0)] {
        match mapping.lookup(x, y) {
            Some((mx, my)) => println!(
                "  ({x:5.0},{y:5.0}) -> ({mx:6.1},{my:6.1})   ({:6.1},{:6.1})",
                x - SHIFT as f32,
                y - SHIFT as f32
            ),
            None => println!("  ({x:5.0},{y:5.0}) -> unmapped"),
        }
    }

    Ok(())
}

/// Two windows onto one texture, offset by `SHIFT` in both axes.
fn shifted_pair() -> (Photo, Photo) {
    let (bw, bh) = (WIDTH + SHIFT, HEIGHT + SHIFT);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut base = Vec::with_capacity(bw * bh * 4);
    for y in 0..bh {
        for x in 0..bw {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 56) as f32 / 255.0 * 70.0 - 35.0;
            let wave = 127.0 + 100.0 * (x as f32 / 9.0).sin() * (y as f32 / 11.0).cos() + noise;
            let v = wave.clamp(0.0, 255.0) as u8;
            base.extend_from_slice(&[v, (v / 2).wrapping_add(40), 255 - v, 255]);
        }
    }

    let window = |ox: usize, oy: usize| {
        let mut data = Vec::with_capacity(WIDTH * HEIGHT * 4);
        for y in 0..HEIGHT {
            let row = (y + oy) * bw + ox;
            data.extend_from_slice(&base[row * 4..(row + WIDTH) * 4]);
        }
        Photo::from_rgba(WIDTH, HEIGHT, data).expect("buffer matches the dimensions")
    };

    (window(0, 0), window(SHIFT, SHIFT))
}
