//! Dumps the mapping's coverage mask after each step of a quality preset's schedule.
//!
//! White is a mapped grid cell, black an unmapped one, so the "dark bands" the solver
//! sometimes produces show up directly as bands in the mask instead of having to be
//! spotted in a morph frame. Run it as
//!
//! ```text
//! cargo run --release -p pixelmap_command_line_tool --example coverage_mask -- a.jpg b.jpg medium out_dir
//! ```
use image::open;
use pixelmap::{Correspondence, Photo, Quality};
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (a, b) = (&args[1], &args[2]);
    let quality: Quality = args
        .get(3)
        .map(|s| s.parse().expect("low, medium or high"))
        .unwrap_or(Quality::Low);
    let out_dir = args.get(4).cloned().unwrap_or_else(|| ".".to_string());
    let max_width: usize = args
        .get(5)
        .map(|s| s.parse().expect("a width"))
        .unwrap_or(1200);
    std::fs::create_dir_all(&out_dir).expect("could not create the output directory");

    let load = |p: &str| {
        let photo = Photo::from(open(p).expect("could not load image"));
        if photo.width() > max_width {
            photo.scaled_to_width(max_width)
        } else {
            photo
        }
    };
    let (photo1, photo2) = (Arc::new(load(a)), Arc::new(load(b)));

    let steps = quality.steps();
    for prefix in 1..=steps.len() {
        let mapping = Correspondence::builder()
            .quality(quality)
            .schedule(&steps[..prefix])
            .run(photo1.clone(), photo2.clone())
            .expect("valid photos");

        for (name, map) in [("fwd", mapping.forward()), ("bwd", mapping.backward())] {
            let (gw, gh) = map.grid_dimensions();
            let mut mask = vec![0u8; gw * gh];
            let mut mapped = 0usize;
            for y in 0..gh {
                for x in 0..gw {
                    if !map.grid_coordinates(x, y).0.is_nan() {
                        mask[y * gw + x] = 255;
                        mapped += 1;
                    }
                }
            }
            let path = format!("{out_dir}/step{prefix:02}_{name}.png");
            image::GrayImage::from_raw(gw as u32, gh as u32, mask)
                .expect("the buffer is exactly gw*gh")
                .save(&path)
                .expect("could not write the mask");
            if name == "fwd" {
                println!(
                    "step {prefix:2}: grid {gw}x{gh}, coverage {:.4}  ({} of {})",
                    mapped as f32 / (gw * gh) as f32,
                    mapped,
                    gw * gh
                );
            }
        }
    }
}
