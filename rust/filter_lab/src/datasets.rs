//! The benchmark datasets, read the way `benchmarks/datasets.py` reads them, so the
//! numbers here describe the same pixels as the Python harness's.

use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use pixelmap::Photo;

pub type Res<T> = Result<T, Box<dyn Error>>;

/// One image pair with dense ground truth in the first photo's pixel grid.
pub struct Pair {
    pub name: String,
    pub img1: Photo,
    pub img2: Photo,
    pub width: usize,
    pub height: usize,
    /// Flow per pixel of `img1`, row-major; meaningful where `valid`.
    pub gt: Vec<[f32; 2]>,
    pub valid: Vec<bool>,
    /// The subset of `valid` visible in both photos, where the dataset says.
    pub noc: Option<Vec<bool>>,
}

/// A pair not yet read from disk.
pub enum Spec {
    Hpatches { seq: PathBuf, k: usize },
    Eth3d { scene: PathBuf, gt: PathBuf },
    Kitti { root: PathBuf, index: usize },
    Sintel { root: PathBuf, scene: String, frame: usize },
}

pub const NAMES: [&str; 4] = ["hpatches", "eth3d", "kitti15", "sintel_clean"];

pub fn data_root() -> PathBuf {
    std::env::var_os("PIXELMAP_BENCH_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("datasets/pixelmap-bench"))
}

fn sorted_dirs(dir: &Path) -> Res<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    v.sort();
    Ok(v)
}

/// The pairs of `dataset`, with the Python harness's default strides (Sintel: every 10th).
pub fn list(dataset: &str) -> Res<Vec<Spec>> {
    let root = data_root();
    let mut out = Vec::new();
    match dataset {
        "hpatches" => {
            for seq in sorted_dirs(&root.join("hpatches-sequences-release"))? {
                if seq.file_name().unwrap().to_string_lossy().starts_with("v_") {
                    for k in 2..=6 {
                        out.push(Spec::Hpatches { seq: seq.clone(), k });
                    }
                }
            }
        }
        "eth3d" => {
            let gt = root.join("eth3d/two_view_training_gt");
            for scene in sorted_dirs(&root.join("eth3d/two_view_training"))? {
                let g = gt.join(scene.file_name().unwrap());
                out.push(Spec::Eth3d { scene, gt: g });
            }
        }
        "kitti15" => {
            let r = root.join("kitti15/training");
            out.extend((0..200).map(|index| Spec::Kitti { root: r.clone(), index }));
        }
        "sintel_clean" => {
            let r = root.join("sintel/training");
            let mut idx = 0;
            for scene in sorted_dirs(&r.join("flow"))? {
                let mut flos: Vec<PathBuf> = fs::read_dir(&scene)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e == "flo"))
                    .collect();
                flos.sort();
                for flo in flos {
                    if idx % 10 == 0 {
                        let stem = flo.file_stem().unwrap().to_string_lossy().to_string();
                        let frame: usize = stem.trim_start_matches("frame_").parse()?;
                        let scene = scene.file_name().unwrap().to_string_lossy().to_string();
                        out.push(Spec::Sintel { root: r.clone(), scene, frame });
                    }
                    idx += 1;
                }
            }
        }
        _ => return Err(format!("unknown dataset {dataset}; choose from {NAMES:?}").into()),
    }
    Ok(out)
}

fn read_rgba(path: &Path) -> Res<image::RgbaImage> {
    Ok(image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgba8())
}

fn photo(img: image::RgbaImage) -> Res<Photo> {
    let (w, h) = img.dimensions();
    Ok(Photo::from_rgba(w as usize, h as usize, img.into_raw())?)
}

/// `img` on a black (opaque) canvas of `w`×`h`, top-left aligned.
fn padded(img: image::RgbaImage, w: u32, h: u32) -> image::RgbaImage {
    if img.dimensions() == (w, h) {
        return img;
    }
    let mut canvas = image::RgbaImage::from_pixel(w, h, image::Rgba([0, 0, 0, 255]));
    image::imageops::overlay(&mut canvas, &img, 0, 0);
    canvas
}

pub fn load(spec: &Spec) -> Res<Pair> {
    match spec {
        Spec::Hpatches { seq, k } => {
            let a = read_rgba(&seq.join("1.ppm"))?;
            let b = read_rgba(&seq.join(format!("{k}.ppm")))?;
            let (w1, h1) = a.dimensions();
            let (w2, h2) = b.dimensions();
            let (w, h) = (w1.max(w2), h1.max(h2));
            let text = fs::read_to_string(seq.join(format!("H_1_{k}")))?;
            let hm: Vec<f64> = text.split_whitespace().map(|t| t.parse()).collect::<Result<_, _>>()?;
            if hm.len() != 9 {
                return Err(format!("{}: H_1_{k} is not 3x3", seq.display()).into());
            }
            let (w, h, w1, h1, w2, h2) = (w as usize, h as usize, w1 as f64, h1 as f64, w2 as f64, h2 as f64);
            let mut gt = vec![[0.0f32; 2]; w * h];
            let mut valid = vec![false; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (xf, yf) = (x as f64, y as f64);
                    let z = hm[6] * xf + hm[7] * yf + hm[8];
                    let mx = (hm[0] * xf + hm[1] * yf + hm[2]) / z;
                    let my = (hm[3] * xf + hm[4] * yf + hm[5]) / z;
                    let ok = xf < w1 && yf < h1 && z > 0.0 && mx >= 0.0 && mx <= w2 - 1.0 && my >= 0.0 && my <= h2 - 1.0;
                    if ok {
                        gt[y * w + x] = [(mx - xf) as f32, (my - yf) as f32];
                        valid[y * w + x] = true;
                    }
                }
            }
            let name = format!("{}_{k}", seq.file_name().unwrap().to_string_lossy());
            Ok(Pair {
                name,
                img1: photo(padded(a, w as u32, h as u32))?,
                img2: photo(padded(b, w as u32, h as u32))?,
                width: w,
                height: h,
                gt,
                valid,
                noc: None,
            })
        }
        Spec::Eth3d { scene, gt: gt_dir } => {
            let a = read_rgba(&scene.join("im0.png"))?;
            let b = read_rgba(&scene.join("im1.png"))?;
            let (w, h, disp) = read_pfm(&gt_dir.join("disp0GT.pfm"))?;
            let valid: Vec<bool> = disp.iter().map(|d| d.is_finite() && *d > 0.0).collect();
            let gt = disp.iter().zip(&valid).map(|(d, v)| [if *v { -d } else { 0.0 }, 0.0]).collect();
            let noc_path = gt_dir.join("mask0nocc.png");
            let noc = if noc_path.exists() {
                let m = image::open(&noc_path)?.to_luma8();
                Some(m.as_raw().iter().zip(&valid).map(|(m, v)| *m == 255 && *v).collect())
            } else {
                None
            };
            Ok(Pair {
                name: scene.file_name().unwrap().to_string_lossy().to_string(),
                img1: photo(a)?,
                img2: photo(b)?,
                width: w,
                height: h,
                gt,
                valid,
                noc,
            })
        }
        Spec::Kitti { root, index } => {
            let name = format!("{index:06}");
            let (gt, valid, w, h) = read_kitti_flow(&root.join(format!("flow_occ/{name}_10.png")))?;
            let (_, noc, _, _) = read_kitti_flow(&root.join(format!("flow_noc/{name}_10.png")))?;
            let noc = noc.iter().zip(&valid).map(|(n, v)| *n && *v).collect();
            Ok(Pair {
                img1: photo(read_rgba(&root.join(format!("image_2/{name}_10.png")))?)?,
                img2: photo(read_rgba(&root.join(format!("image_2/{name}_11.png")))?)?,
                name,
                width: w,
                height: h,
                gt,
                valid,
                noc: Some(noc),
            })
        }
        Spec::Sintel { root, scene, frame } => {
            let n = *frame;
            let (w, h, flow) = read_flo(&root.join(format!("flow/{scene}/frame_{n:04}.flo")))?;
            let valid: Vec<bool> = flow
                .iter()
                .map(|f| f[0].is_finite() && f[1].is_finite() && f[0].abs().max(f[1].abs()) < 1e9)
                .collect();
            let occ_path = root.join(format!("occlusions/{scene}/frame_{n:04}.png"));
            let noc = if occ_path.exists() {
                let m = image::open(&occ_path)?.to_luma8();
                Some(m.as_raw().iter().zip(&valid).map(|(m, v)| *m == 0 && *v).collect())
            } else {
                None
            };
            Ok(Pair {
                name: format!("{scene}_{n:04}"),
                img1: photo(read_rgba(&root.join(format!("clean/{scene}/frame_{n:04}.png")))?)?,
                img2: photo(read_rgba(&root.join(format!("clean/{scene}/frame_{:04}.png", n + 1)))?)?,
                width: w,
                height: h,
                gt: flow,
                valid,
                noc,
            })
        }
    }
}

/// KITTI's 16-bit flow PNG: `u = (R - 2^15) / 64`, `v` likewise from G, valid where B > 0.
fn read_kitti_flow(path: &Path) -> Res<(Vec<[f32; 2]>, Vec<bool>, usize, usize)> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb16();
    let (w, h) = img.dimensions();
    let mut flow = Vec::with_capacity((w * h) as usize);
    let mut valid = Vec::with_capacity((w * h) as usize);
    for p in img.pixels() {
        let ok = p[2] > 0;
        valid.push(ok);
        flow.push(if ok {
            [(p[0] as f32 - 32768.0) / 64.0, (p[1] as f32 - 32768.0) / 64.0]
        } else {
            [0.0, 0.0]
        });
    }
    Ok((flow, valid, w as usize, h as usize))
}

fn read_flo(path: &Path) -> Res<(usize, usize, Vec<[f32; 2]>)> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.read_to_end(&mut bytes)?;
    let f32_at = |i: usize| f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    if bytes.len() < 12 || f32_at(0) != 202021.25 {
        return Err(format!("{}: bad .flo magic", path.display()).into());
    }
    let w = i32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let h = i32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if bytes.len() < 12 + w * h * 8 {
        return Err(format!("{}: truncated", path.display()).into());
    }
    let flow = (0..w * h).map(|i| [f32_at(12 + 8 * i), f32_at(16 + 8 * i)]).collect();
    Ok((w, h, flow))
}

/// A single-channel PFM, with its bottom-to-top rows flipped into the usual order.
fn read_pfm(path: &Path) -> Res<(usize, usize, Vec<f32>)> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut pos = 0;
    let mut line = || {
        let start = pos;
        while pos < bytes.len() && bytes[pos] != b'\n' {
            pos += 1;
        }
        pos += 1;
        String::from_utf8_lossy(&bytes[start..pos - 1]).trim().to_string()
    };
    if line() != "Pf" {
        return Err(format!("{}: not a single-channel PFM", path.display()).into());
    }
    let dims: Vec<usize> = line().split_whitespace().map(|t| t.parse()).collect::<Result<_, _>>()?;
    let scale: f32 = line().parse()?;
    let (w, h) = (dims[0], dims[1]);
    let data = &bytes[pos..];
    if data.len() < w * h * 4 {
        return Err(format!("{}: truncated", path.display()).into());
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = 4 * (y * w + x);
            let b: [u8; 4] = data[i..i + 4].try_into().unwrap();
            let v = if scale < 0.0 { f32::from_le_bytes(b) } else { f32::from_be_bytes(b) };
            out[(h - 1 - y) * w + x] = v;
        }
    }
    Ok((w, h, out))
}
