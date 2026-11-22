//! Which signals tell pixelmap's wrong answers from its right ones?
//!
//! Runs pixelmap (`Quality::High`, fill off, default seed) on benchmark pairs, computes
//! the candidate features of [`features`] at every node of the final forward map, labels
//! each node that has ground truth as bad when its answer is more than 3 source pixels
//! off, and reports, per dataset, how well each feature ranks the bad answers first.
//!
//! ```text
//! cargo run --release -p filter_lab -- --datasets kitti15,sintel_clean --limit 20
//! cargo run --release -p filter_lab -- --datasets kitti15 --reuse     # re-analyse the cache
//! cargo run --release -p filter_lab -- --reuse --rules                 # fixed thresholds
//! ```
//!
//! `--rules` asks the question a shipped filter has to answer: one threshold per
//! feature, the same on every dataset, chosen so that it removes `GOOD_BUDGET` of the
//! correct answers (the median over the datasets of each one's own quantile), and then
//! what share of the wrong answers it removes on each dataset. Unlike the ranking, a
//! rule never flags a `NaN`: it acts only on what it could measure.
//!
//! Samples are cached per dataset under `--cache` (default `target/filter_lab`), so the
//! analysis can be changed and re-run without solving again.

mod datasets;
mod features;
mod fundamental;

use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use datasets::Res;
use pixelmap::{FillMode, PixelMapProcessor, Quality, DEFAULT_SEED};

const BAD_PX: f32 = 3.0;

/// One answered grid node with ground truth.
struct Sample {
    feats: [f32; features::N],
    err: f32,
    occ: bool,
}

struct Args {
    datasets: Vec<String>,
    limit: Option<usize>,
    cache: PathBuf,
    reuse: bool,
    rules: bool,
    verify: bool,
    two_stage: bool,
}

fn parse_args() -> Res<Args> {
    let mut a = Args {
        datasets: vec!["kitti15".into(), "sintel_clean".into(), "eth3d".into(), "hpatches".into()],
        limit: None,
        cache: PathBuf::from("target/filter_lab"),
        reuse: false,
        rules: false,
        verify: false,
        two_stage: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--datasets" => a.datasets = it.next().ok_or("--datasets needs a list")?.split(',').map(String::from).collect(),
            "--limit" => a.limit = Some(it.next().ok_or("--limit needs N")?.parse()?),
            "--cache" => a.cache = it.next().ok_or("--cache needs a dir")?.into(),
            "--reuse" => a.reuse = true,
            "--rules" => a.rules = true,
            "--verify" => a.verify = true,
            "--two-stage" => a.two_stage = true,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    Ok(a)
}

/// Solves one pair and returns its samples.
fn run_pair(pair: datasets::Pair) -> Vec<Sample> {
    let (w, h) = (pair.width, pair.height);
    let quality = Quality::High;
    let mut p = PixelMapProcessor::with_seed(Arc::new(pair.img1), Arc::new(pair.img2), quality.photo_width(), DEFAULT_SEED);
    p.set_fill_mode(FillMode::Off);
    p.init();
    let steps = quality.steps_for(w);
    let mut prev = None;
    for (k, step) in steps.iter().enumerate() {
        step.apply(&mut p);
        if steps.get(k + 1).is_some_and(|n| n.photo_width != step.photo_width) {
            prev = Some(p.finish(2.0).0);
        }
    }
    let (fwd, bwd) = p.finish(2.0);
    let radius = steps.last().map_or(5, |s| s.neighborhood_radius);
    let feats = features::compute(&fwd, &bwd, prev.as_ref(), radius);

    let (gw, gh) = fwd.grid_dimensions();
    let c = fwd.grid_cell_size() as f32;
    let scale = fwd.dimensions().0 as f32 / w as f32;
    let mut out = Vec::new();
    for gy in 0..gh {
        for gx in 0..gw {
            let (tx, ty) = fwd.grid_coordinates(gx, gy);
            if tx.is_nan() || ty.is_nan() {
                continue;
            }
            let (x, y) = ((gx as f32 * c / scale).round() as usize, (gy as f32 * c / scale).round() as usize);
            if x >= w || y >= h || !pair.valid[y * w + x] {
                continue;
            }
            let g = pair.gt[y * w + x];
            let err = (tx / scale - (x as f32 + g[0])).hypot(ty / scale - (y as f32 + g[1]));
            let occ = pair.noc.as_ref().is_some_and(|n| !n[y * w + x]);
            out.push(Sample { feats: feats[gy * gw + gx], err, occ });
        }
    }
    out
}

fn save(path: &PathBuf, samples: &[Sample]) -> Res<()> {
    let mut f = BufWriter::new(fs::File::create(path)?);
    f.write_all(&(features::N as u32).to_le_bytes())?;
    for s in samples {
        for v in s.feats.iter().chain([s.err, s.occ as u8 as f32].iter()) {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn load(path: &PathBuf) -> Res<Vec<Sample>> {
    let mut bytes = Vec::new();
    BufReader::new(fs::File::open(path)?).read_to_end(&mut bytes)?;
    let n = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
    if n != features::N {
        return Err(format!("{}: cached with {n} features, this build has {}", path.display(), features::N).into());
    }
    let vals: Vec<f32> = bytes[4..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    Ok(vals
        .chunks_exact(n + 2)
        .map(|r| Sample { feats: r[..n].try_into().unwrap(), err: r[n], occ: r[n + 1] != 0.0 })
        .collect())
}

/// SplitMix64, for a deterministic tie-break that is not the raster order.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sample indices from most to least suspect under `key` (NaN most suspect).
fn ranking(samples: &[Sample], key: impl Fn(&Sample) -> f32) -> Vec<usize> {
    let mut idx: Vec<(f32, u64, usize)> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let k = key(s);
            (if k.is_nan() { f32::INFINITY } else { k }, mix(i as u64), i)
        })
        .collect();
    idx.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    idx.into_iter().map(|t| t.2).collect()
}

/// Probability that a random bad sample ranks above a random good one, ties counted half.
fn auc(samples: &[Sample], key: impl Fn(&Sample) -> f32) -> f64 {
    let mut v: Vec<(f32, bool)> = samples
        .iter()
        .map(|s| {
            let k = key(s);
            (if k.is_nan() { f32::INFINITY } else { k }, s.err > BAD_PX)
        })
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut rank_sum, mut nb) = (0.0f64, 0usize);
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0; // ranks i+1..=j
        for item in &v[i..j] {
            if item.1 {
                rank_sum += avg;
                nb += 1;
            }
        }
        i = j;
    }
    let ng = v.len() - nb;
    if nb == 0 || ng == 0 {
        return f64::NAN;
    }
    (rank_sum - (nb * (nb + 1)) as f64 / 2.0) / (nb as f64 * ng as f64)
}

const QS: [f64; 3] = [0.05, 0.10, 0.20];

fn report(dataset: &str, samples: &[Sample]) {
    let n = samples.len();
    let nb = samples.iter().filter(|s| s.err > BAD_PX).count();
    let no = samples.iter().filter(|s| s.occ).count();
    let nob = samples.iter().filter(|s| s.occ && s.err > BAD_PX).count();
    println!("\n### {dataset}: {n} answered nodes with ground truth, {:.1}% bad (> {BAD_PX} px)", 100.0 * nb as f64 / n as f64);
    if no > 0 {
        println!("occluded: {:.1}% of the answers, {:.1}% of them bad", 100.0 * no as f64 / n as f64, 100.0 * nob as f64 / no as f64);
    }
    println!();
    let qh: Vec<String> = QS.iter().map(|q| format!("bad cut @{:.0}%", q * 100.0)).collect();
    println!("| feature | AUC | {} | good cut @10% | occ cut @10% | NaN |", qh.join(" | "));
    println!("|---|---|{}---|---|---|", "---|".repeat(QS.len()));

    let oracle: Vec<String> = QS.iter().map(|q| format!("{:.1}%", 100.0 * (q * n as f64 / nb.max(1) as f64).min(1.0))).collect();
    println!("| *oracle* | 1.000 | {} | | | |", oracle.join(" | "));
    let random: Vec<String> = QS.iter().map(|q| format!("{:.1}%", 100.0 * q)).collect();
    println!("| *random* | 0.500 | {} | 10.0% | 10.0% | |", random.join(" | "));

    for (f, name) in features::NAMES.iter().enumerate() {
        let key = |s: &Sample| s.feats[f];
        let order = ranking(samples, key);
        let cells: Vec<String> = QS
            .iter()
            .map(|q| {
                let cut = (q * n as f64) as usize;
                let bad = order[..cut].iter().filter(|&&i| samples[i].err > BAD_PX).count();
                format!("{:.1}%", 100.0 * bad as f64 / nb.max(1) as f64)
            })
            .collect();
        let cut = n / 10;
        let good = order[..cut].iter().filter(|&&i| samples[i].err <= BAD_PX).count();
        let occ = order[..cut].iter().filter(|&&i| samples[i].occ).count();
        let nan = samples.iter().filter(|s| s.feats[f].is_nan()).count();
        println!(
            "| {name} | {:.3} | {} | {:.1}% | {} | {:.1}% |",
            auc(samples, key),
            cells.join(" | "),
            100.0 * good as f64 / (n - nb).max(1) as f64,
            if no > 0 { format!("{:.1}%", 100.0 * occ as f64 / no as f64) } else { "".into() },
            100.0 * nan as f64 / n as f64,
        );
    }
}

/// The share of correct answers each rule may remove.
const GOOD_BUDGET: f64 = 0.02;

/// The `q` quantile of the measured (non-NaN) values of feature `f` over correct answers.
fn good_quantile(samples: &[Sample], f: usize, q: f64) -> f32 {
    let mut v: Vec<f32> = samples.iter().filter(|s| s.err <= BAD_PX && !s.feats[f].is_nan()).map(|s| s.feats[f]).collect();
    if v.is_empty() {
        return f32::INFINITY;
    }
    v.sort_by(f32::total_cmp);
    v[((q * v.len() as f64) as usize).min(v.len() - 1)]
}

fn rules(all: &[(String, Vec<Sample>)]) {
    // Per feature: the median over datasets of the (1 - budget) quantile of good values.
    let threshold = |f: usize, budget: f64| -> f32 {
        let mut t: Vec<f32> = all.iter().map(|(_, s)| good_quantile(s, f, 1.0 - budget)).collect();
        median_f32(&mut t)
    };
    let fmt = |flag: &dyn Fn(&Sample) -> bool, samples: &[Sample]| -> String {
        let (mut nb, mut cb, mut ng, mut cg, mut no, mut co) = (0, 0, 0, 0, 0, 0);
        for s in samples {
            let hit = flag(s);
            if s.err > BAD_PX { nb += 1; cb += hit as usize } else { ng += 1; cg += hit as usize }
            if s.occ { no += 1; co += hit as usize }
        }
        let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
        let occ = if no > 0 { format!(", occ {:.0}%", pct(co, no)) } else { String::new() };
        format!("{:.1}% / {:.1}%{occ}", pct(cb, nb), pct(cg, ng))
    };
    println!("\n### Fixed thresholds: bad cut / good cut (and occluded cut) per dataset\n");
    let names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
    println!("| rule | threshold | {} |", names.join(" | "));
    println!("|---|---|{}", "---|".repeat(names.len()));
    let mut thresholds = [0.0f32; features::N];
    for (f, name) in features::NAMES.iter().enumerate() {
        let t = threshold(f, GOOD_BUDGET);
        thresholds[f] = t;
        let cells: Vec<String> = all.iter().map(|(_, s)| fmt(&|x: &Sample| x.feats[f] > t, s)).collect();
        println!("| {name} | {t:.3} | {} |", cells.join(" | "));
    }
    let combos: [&[&str]; 4] = [
        &["aniso", "j_dev"],
        &["aniso", "j_dev", "flow_dev"],
        &["aniso", "j_dev", "flow_dev", "sampson"],
        &["aniso", "j_dev", "flow_dev", "prev_width", "island"],
    ];
    for combo in combos {
        // Each member gets an equal share of the budget, so the OR stays near it.
        let idx: Vec<(usize, f32)> = combo
            .iter()
            .map(|n| {
                let f = features::NAMES.iter().position(|m| m == n).unwrap();
                (f, threshold(f, GOOD_BUDGET / combo.len() as f64))
            })
            .collect();
        let flag = |x: &Sample| idx.iter().any(|&(f, t)| x.feats[f] > t);
        let cells: Vec<String> = all.iter().map(|(_, s)| fmt(&flag, s)).collect();
        let ts: Vec<String> = idx.iter().map(|(_, t)| format!("{t:.3}")).collect();
        println!("| {} | {} | {} |", combo.join(" ∨ "), ts.join(", "), cells.join(" | "));
    }
}

fn median_f32(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Checks that `pixelmap::LocalConsistency` removes exactly the nodes the lab's own
/// features flag under the same thresholds, so what the lab measured is what ships.
fn verify(pair: datasets::Pair) -> (usize, usize) {
    let quality = Quality::High;
    let w = pair.width;
    let mut p = PixelMapProcessor::with_seed(Arc::new(pair.img1), Arc::new(pair.img2), quality.photo_width(), DEFAULT_SEED);
    p.init();
    let steps = quality.steps_for(w);
    for step in &steps {
        step.apply(&mut p);
    }
    let (fwd, bwd) = p.finish(2.0);
    let feats = features::compute(&fwd, &bwd, None, 5);
    let lc = pixelmap::LocalConsistency::default();
    let idx = |n: &str| features::NAMES.iter().position(|m| *m == n).unwrap();
    let (fa, fj, ff) = (idx("aniso"), idx("j_dev"), idx("flow_dev"));
    p.set_local_consistency(Some(lc));
    let (filtered, _) = p.finish(2.0);
    let (gw, gh) = fwd.grid_dimensions();
    let (mut removed, mut mismatched) = (0, 0);
    for i in 0..gw * gh {
        let before = !fwd.grid_coordinates(i % gw, i / gw).0.is_nan();
        if !before {
            continue;
        }
        let gone = filtered.grid_coordinates(i % gw, i / gw).0.is_nan();
        let f = feats[i];
        let flag = f[ff] > lc.max_flow_deviation || f[fa] > lc.max_anisotropy.ln() || f[fj] > lc.max_jacobian_deviation;
        removed += gone as usize;
        mismatched += (gone != flag) as usize;
    }
    (removed, mismatched)
}

/// Bad/good node counts removed by the one-pass rule and by a two-stage one that drops
/// the flow outliers first and only then judges the Jacobians, on a map without them.
/// Returns `[bad, good, removed bad 1-pass, removed good 1-pass, removed bad 2-stage,
/// removed good 2-stage]`.
fn two_stage(pair: datasets::Pair) -> [usize; 6] {
    let quality = Quality::High;
    let (w, h) = (pair.width, pair.height);
    let mut p = PixelMapProcessor::with_seed(Arc::new(pair.img1), Arc::new(pair.img2), quality.photo_width(), DEFAULT_SEED);
    p.init();
    for step in &quality.steps_for(w) {
        step.apply(&mut p);
    }
    let (fwd, bwd) = p.finish(2.0);
    let lc = pixelmap::LocalConsistency::default();
    let idx = |n: &str| features::NAMES.iter().position(|m| *m == n).unwrap();
    let (fa, fj, ff) = (idx("aniso"), idx("j_dev"), idx("flow_dev"));
    let la = lc.max_anisotropy.ln();
    let f1 = features::compute(&fwd, &bwd, None, 5);
    let (gw, gh) = fwd.grid_dimensions();
    let mut reduced = fwd.clone();
    for i in 0..gw * gh {
        if f1[i][ff] > lc.max_flow_deviation {
            reduced.set_grid_coordinates(i % gw, i / gw, f32::NAN, f32::NAN);
        }
    }
    let f2 = features::compute(&reduced, &bwd, None, 5);
    let c = fwd.grid_cell_size() as f32;
    let scale = fwd.dimensions().0 as f32 / w as f32;
    let mut out = [0usize; 6];
    for gy in 0..gh {
        for gx in 0..gw {
            let i = gy * gw + gx;
            let (tx, ty) = fwd.grid_coordinates(gx, gy);
            if tx.is_nan() {
                continue;
            }
            let (x, y) = ((gx as f32 * c / scale).round() as usize, (gy as f32 * c / scale).round() as usize);
            if x >= w || y >= h || !pair.valid[y * w + x] {
                continue;
            }
            let g = pair.gt[y * w + x];
            let bad = (tx / scale - (x as f32 + g[0])).hypot(ty / scale - (y as f32 + g[1])) > BAD_PX;
            let one = f1[i][ff] > lc.max_flow_deviation || f1[i][fa] > la || f1[i][fj] > lc.max_jacobian_deviation;
            let two = f1[i][ff] > lc.max_flow_deviation || f2[i][fa] > la || f2[i][fj] > lc.max_jacobian_deviation;
            let k = if bad { 0 } else { 1 };
            out[k] += 1;
            out[2 + k] += one as usize;
            out[4 + k] += two as usize;
        }
    }
    out
}

fn main() -> Res<()> {
    let args = parse_args()?;
    if args.two_stage {
        for ds in &args.datasets {
            let mut specs = datasets::list(ds)?;
            let n = args.limit.unwrap_or(20);
            let stride = (specs.len() / n).max(1);
            specs = specs.into_iter().step_by(stride).take(n).collect();
            let mut t = [0usize; 6];
            for spec in &specs {
                let r = two_stage(datasets::load(spec)?);
                for k in 0..6 {
                    t[k] += r[k];
                }
            }
            let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
            println!(
                "{ds} ({} pairs): one pass removes {:.1}% of bad / {:.1}% of good; two-stage {:.1}% / {:.1}%",
                specs.len(),
                pct(t[2], t[0]),
                pct(t[3], t[1]),
                pct(t[4], t[0]),
                pct(t[5], t[1])
            );
        }
        return Ok(());
    }
    if args.verify {
        for ds in &args.datasets {
            let mut specs = datasets::list(ds)?;
            specs.truncate(args.limit.unwrap_or(5));
            for spec in &specs {
                let pair = datasets::load(spec)?;
                let name = pair.name.clone();
                let (removed, mismatched) = verify(pair);
                println!("{ds} {name}: filter removed {removed} nodes, {mismatched} differ from the lab's rule");
            }
        }
        return Ok(());
    }
    fs::create_dir_all(&args.cache)?;
    let mut all = Vec::new();
    for ds in &args.datasets {
        let path = args.cache.join(format!("{ds}.bin"));
        let samples = if args.reuse && path.exists() {
            load(&path)?
        } else {
            let mut specs = datasets::list(ds)?;
            if let Some(l) = args.limit {
                specs.truncate(l);
            }
            let mut all = Vec::new();
            let t0 = Instant::now();
            for (k, spec) in specs.iter().enumerate() {
                let pair = datasets::load(spec)?;
                let name = pair.name.clone();
                let s = run_pair(pair);
                eprintln!("{ds} {}/{} {name}: {} samples ({:.0} s)", k + 1, specs.len(), s.len(), t0.elapsed().as_secs_f64());
                all.extend(s);
            }
            save(&path, &all)?;
            all
        };
        if args.rules {
            all.push((ds.clone(), samples));
        } else {
            report(ds, &samples);
        }
    }
    if args.rules {
        rules(&all);
    }
    Ok(())
}
