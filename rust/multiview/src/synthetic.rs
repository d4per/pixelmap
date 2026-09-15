//! Synthetic scenes with known geometry, for testing every stage against an exact answer.
//!
//! A pipeline like this one usually fails by running cleanly and producing garbage, and
//! with real photos there is no telling which stage is at fault. Here the scene, the
//! cameras and every correspondence are known exactly, so each stage can be checked on
//! its own. [`SyntheticPair`] answers lookups straight from the geometry, with optional
//! noise and outliers, and [`SyntheticSet::render`] produces photos for exercising
//! pixelmap itself.
//!
//! World coordinates have y up. The cameras stand on the negative-z side of the scene
//! and look towards positive z.

use nalgebra::{Point3, Rotation3, Vector3};
use pixelmap::Photo;

use crate::calib::{FocalSource, Intrinsics};
use crate::lookup::PairLookup;
use crate::pose::Pose;
use crate::rng::{hash3, unit};
use crate::types::{PairId, PhotoPx, ViewId, World};

const EPSILON: f64 = 1e-9;

/// A textured surface.
#[derive(Clone, Debug)]
pub enum Surface {
    /// The rectangle `origin + a·u + b·v` for `a, b` in `[0, 1]`. `u` and `v` must be
    /// perpendicular.
    Rectangle {
        /// One corner.
        origin: Point3<f64>,
        /// The first edge.
        u: Vector3<f64>,
        /// The second edge.
        v: Vector3<f64>,
    },
    /// A sphere.
    Sphere {
        /// Its centre.
        centre: Point3<f64>,
        /// Its radius.
        radius: f64,
    },
}

impl Surface {
    /// The distance along a ray with unit direction `dir` to where it first hits this
    /// surface, if it does.
    fn intersect(&self, origin: &Point3<f64>, dir: &Vector3<f64>) -> Option<f64> {
        match self {
            Surface::Rectangle { origin: o, u, v } => {
                let normal = u.cross(v);
                let denominator = dir.dot(&normal);
                if denominator.abs() < 1e-12 {
                    return None;
                }
                let t = (o - origin).dot(&normal) / denominator;
                if t <= EPSILON {
                    return None;
                }
                let offset = origin + dir * t - o;
                let a = offset.dot(u) / u.norm_squared();
                let b = offset.dot(v) / v.norm_squared();
                ((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b)).then_some(t)
            }
            Surface::Sphere { centre, radius } => {
                let oc = origin - centre;
                let b = oc.dot(dir);
                let discriminant = b * b - (oc.norm_squared() - radius * radius);
                if discriminant < 0.0 {
                    return None;
                }
                let root = discriminant.sqrt();
                [-b - root, -b + root].into_iter().find(|&t| t > EPSILON)
            }
        }
    }
}

/// A set of textured surfaces, and where to stand to look at them.
#[derive(Clone, Debug)]
pub struct Scene {
    surfaces: Vec<Surface>,
    target: Point3<f64>,
    distance: f64,
    seed: u64,
}

impl Scene {
    /// The names [`Self::named`] accepts.
    pub const NAMES: [&'static str; 3] = ["plane", "sphere", "corner"];

    /// A scene built from `surfaces`, viewed from `distance` away from `target`.
    pub fn new(surfaces: Vec<Surface>, target: Point3<f64>, distance: f64) -> Self {
        Scene {
            surfaces,
            target,
            distance,
            seed: 0x5EED,
        }
    }

    /// A single flat rectangle, facing the cameras. Degenerate for the 8-point
    /// algorithm.
    pub fn plane() -> Self {
        Scene::new(
            vec![Surface::Rectangle {
                origin: Point3::new(-3.0, -3.0, 0.0),
                u: Vector3::new(6.0, 0.0, 0.0),
                v: Vector3::new(0.0, 6.0, 0.0),
            }],
            Point3::origin(),
            5.0,
        )
    }

    /// A unit sphere in front of a flat backdrop.
    pub fn sphere() -> Self {
        Scene::new(
            vec![
                Surface::Sphere {
                    centre: Point3::origin(),
                    radius: 1.0,
                },
                Surface::Rectangle {
                    origin: Point3::new(-6.0, -6.0, 2.5),
                    u: Vector3::new(12.0, 0.0, 0.0),
                    v: Vector3::new(0.0, 12.0, 0.0),
                },
            ],
            Point3::origin(),
            5.0,
        )
    }

    /// The inside corner of a room: back wall, floor and one side wall.
    pub fn corner() -> Self {
        Scene::new(
            vec![
                Surface::Rectangle {
                    origin: Point3::new(-3.0, -1.5, 1.5),
                    u: Vector3::new(6.0, 0.0, 0.0),
                    v: Vector3::new(0.0, 4.0, 0.0),
                },
                Surface::Rectangle {
                    origin: Point3::new(-3.0, -1.5, -3.0),
                    u: Vector3::new(6.0, 0.0, 0.0),
                    v: Vector3::new(0.0, 0.0, 4.5),
                },
                Surface::Rectangle {
                    origin: Point3::new(2.0, -1.5, -3.0),
                    u: Vector3::new(0.0, 4.0, 0.0),
                    v: Vector3::new(0.0, 0.0, 4.5),
                },
            ],
            Point3::new(0.3, -0.3, 0.5),
            5.0,
        )
    }

    /// One of the scenes in [`Self::NAMES`].
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "plane" => Some(Scene::plane()),
            "sphere" => Some(Scene::sphere()),
            "corner" => Some(Scene::corner()),
            _ => None,
        }
    }

    /// The distance to the nearest surface along a ray with unit direction `dir`.
    pub fn intersect(&self, origin: &Point3<f64>, dir: &Vector3<f64>) -> Option<f64> {
        self.surfaces
            .iter()
            .filter_map(|s| s.intersect(origin, dir))
            .min_by(f64::total_cmp)
    }

    /// The texture colour at a point: a few octaves of value noise per channel. Fixed to
    /// the surface, so every camera sees the same colour at the same point.
    pub fn colour(&self, p: &Point3<f64>) -> [u8; 4] {
        let channel = |salt: u64| {
            let seed = self.seed.wrapping_add(salt << 8);
            0.5 * value_noise(p, 0.4, seed)
                + 0.3 * value_noise(p, 0.15, seed + 1)
                + 0.2 * value_noise(p, 0.06, seed + 2)
        };
        let byte = |v: f64| (((v - 0.5) * 2.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
        [byte(channel(1)), byte(channel(2)), byte(channel(3)), 255]
    }
}

/// Smoothly interpolated random values on a lattice of spacing `cell`, in `[0, 1]`.
fn value_noise(p: &Point3<f64>, cell: f64, seed: u64) -> f64 {
    let q = p.coords / cell;
    let base = q.map(f64::floor);
    let f = (q - base).map(|t| t * t * (3.0 - 2.0 * t));
    let (ix, iy, iz) = (base.x as i64, base.y as i64, base.z as i64);
    let corner = |dx: i64, dy: i64, dz: i64| {
        let key = hash3((ix + dx) as u64, (iy + dy) as u64, (iz + dz) as u64);
        unit(hash3(seed, key, 0))
    };
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let plane = |dz| {
        lerp(
            lerp(corner(0, 0, dz), corner(1, 0, dz), f.x),
            lerp(corner(0, 1, dz), corner(1, 1, dz), f.x),
            f.y,
        )
    };
    lerp(plane(0), plane(1), f.z)
}

/// A scene photographed by several cameras that share one set of intrinsics.
#[derive(Clone, Debug)]
pub struct SyntheticSet {
    /// What is being photographed.
    pub scene: Scene,
    /// One pose per view.
    pub poses: Vec<Pose>,
    /// The shared camera.
    pub intrinsics: Intrinsics,
    /// Photo width in pixels.
    pub width: usize,
    /// Photo height in pixels.
    pub height: usize,
}

impl SyntheticSet {
    /// `views` cameras on a horizontal arc around the scene's target, spanning
    /// `spread_deg` degrees in total, all looking at the target.
    pub fn orbit(scene: Scene, views: usize, spread_deg: f64, width: usize, height: usize) -> Self {
        let elevation = 8f64.to_radians();
        let poses = (0..views)
            .map(|i| {
                let theta = arc_angle(i, views, spread_deg);
                let direction = Vector3::new(
                    theta.sin() * elevation.cos(),
                    elevation.sin(),
                    -theta.cos() * elevation.cos(),
                );
                let eye = scene.target + direction * scene.distance;
                Pose::look_at(&eye, &scene.target, &Vector3::y())
            })
            .collect();
        Self::with_poses(scene, poses, width, height)
    }

    /// `views` cameras at one position, panning across `spread_deg` degrees. There is no
    /// parallax between these views, so no depth can be recovered from them.
    pub fn rotation_only(
        scene: Scene,
        views: usize,
        spread_deg: f64,
        width: usize,
        height: usize,
    ) -> Self {
        let eye = scene.target - Vector3::z() * scene.distance;
        let poses = (0..views)
            .map(|i| {
                let turn =
                    Rotation3::from_axis_angle(&Vector3::y_axis(), arc_angle(i, views, spread_deg));
                let look = eye + turn * Vector3::z();
                Pose::look_at(&eye, &look, &Vector3::y())
            })
            .collect();
        Self::with_poses(scene, poses, width, height)
    }

    /// Cameras at the given poses, with a focal length equal to the photo width.
    pub fn with_poses(scene: Scene, poses: Vec<Pose>, width: usize, height: usize) -> Self {
        SyntheticSet {
            scene,
            poses,
            intrinsics: Intrinsics::from_focal(width as f64, width, height, FocalSource::Provided),
            width,
            height,
        }
    }

    /// How many views there are.
    pub fn views(&self) -> usize {
        self.poses.len()
    }

    /// The ground-truth pose of `pair`'s second camera in its first camera's frame, with
    /// the translation at its true length.
    pub fn relative_pose(&self, pair: PairId) -> Pose {
        self.poses[pair.b().index()].relative_to(&self.poses[pair.a().index()])
    }

    /// The scene point seen at pixel `p` of `view`.
    pub fn surface_point(&self, view: ViewId, p: PhotoPx) -> Option<World> {
        let pose = &self.poses[view.index()];
        let n = self.intrinsics.normalize(p);
        let dir = pose.rotation.inverse() * Vector3::new(n.x(), n.y(), 1.0).normalize();
        let origin = pose.centre();
        let t = self.scene.intersect(&origin, &dir)?;
        Some(World(origin + dir * t))
    }

    /// The depth (camera-frame z) of the scene at pixel `p` of `view`.
    pub fn depth(&self, view: ViewId, p: PhotoPx) -> Option<f64> {
        let x = self.surface_point(view, p)?;
        Some(self.poses[view.index()].to_camera(&x.0).z)
    }

    /// Where `point` lands in `view`, if it is in front of the camera and inside the photo.
    /// Does not check occlusion; see [`Self::visible`].
    pub fn project(&self, view: ViewId, point: &World) -> Option<PhotoPx> {
        let n = self.poses[view.index()].project(point)?;
        let p = self.intrinsics.denormalize(n);
        let inside = (0.0..=(self.width - 1) as f32).contains(&p.x())
            && (0.0..=(self.height - 1) as f32).contains(&p.y());
        inside.then_some(p)
    }

    /// Whether `point`, which must lie on a surface, is the first thing `view`'s camera
    /// sees in its direction.
    pub fn visible(&self, view: ViewId, point: &World) -> bool {
        let centre = self.poses[view.index()].centre();
        let offset = point.0 - centre;
        let distance = offset.norm();
        self.scene
            .intersect(&centre, &(offset / distance))
            .is_some_and(|t| (t - distance).abs() <= 1e-6 * distance.max(1.0))
    }

    /// The exact correspondence of pixel `p` of `from` in `to`: `None` where the point is
    /// occluded, outside the photo, or where `from` sees nothing at all.
    pub fn transfer(&self, from: ViewId, to: ViewId, p: PhotoPx) -> Option<PhotoPx> {
        let point = self.surface_point(from, p)?;
        if !self.visible(to, &point) {
            return None;
        }
        self.project(to, &point)
    }

    /// Renders `view` as a photo, with 2 × 2 supersampling. Pixels that see no surface
    /// are flat grey.
    pub fn render(&self, view: ViewId) -> Photo {
        const OFFSETS: [(f32, f32); 4] =
            [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)];
        let mut data = Vec::with_capacity(self.width * self.height * 4);
        for y in 0..self.height {
            for x in 0..self.width {
                let mut sum = [0u32; 3];
                for (dx, dy) in OFFSETS {
                    let p = PhotoPx::new(x as f32 + dx, y as f32 + dy);
                    let colour = self
                        .surface_point(view, p)
                        .map_or([128, 128, 128, 255], |point| self.scene.colour(&point.0));
                    for (s, c) in sum.iter_mut().zip(colour) {
                        *s += c as u32;
                    }
                }
                data.extend(sum.map(|s| (s / 4) as u8));
                data.push(255);
            }
        }
        Photo::from_rgba(self.width, self.height, data)
            .expect("buffer is exactly width × height × 4")
    }

    /// An exact lookup for `pair`, with no noise and no outliers until configured.
    pub fn pair(&self, pair: PairId) -> SyntheticPair<'_> {
        let mut lookup = SyntheticPair {
            set: self,
            pair,
            noise_px: 0.0,
            outlier_fraction: 0.0,
            seed: 0,
            coverage: 0.0,
        };
        lookup.coverage = lookup.measure_coverage();
        lookup
    }
}

fn arc_angle(index: usize, views: usize, spread_deg: f64) -> f64 {
    let t = if views > 1 {
        index as f64 / (views - 1) as f64 - 0.5
    } else {
        0.0
    };
    (spread_deg * t).to_radians()
}

/// A [`PairLookup`] that answers from the scene geometry.
#[derive(Clone, Debug)]
pub struct SyntheticPair<'a> {
    set: &'a SyntheticSet,
    pair: PairId,
    noise_px: f32,
    outlier_fraction: f32,
    seed: u64,
    coverage: f32,
}

impl SyntheticPair<'_> {
    /// Adds Gaussian noise with standard deviation `sigma_px` to every answer.
    pub fn with_noise(mut self, sigma_px: f32) -> Self {
        self.noise_px = sigma_px;
        self
    }

    /// Replaces a `fraction` of answers with a uniformly random point in the photo.
    pub fn with_outliers(mut self, fraction: f32) -> Self {
        self.outlier_fraction = fraction;
        self
    }

    /// Seeds the noise and outliers. Answers are a pure function of the seed and the
    /// query point, so the same query always gets the same answer.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn measure_coverage(&self) -> f32 {
        let (a, b) = (self.pair.a(), self.pair.b());
        let (mut hits, mut total) = (0, 0);
        for y in (0..self.set.height).step_by(16) {
            for x in (0..self.set.width).step_by(16) {
                total += 1;
                if self
                    .set
                    .transfer(a, b, PhotoPx::new(x as f32, y as f32))
                    .is_some()
                {
                    hits += 1;
                }
            }
        }
        hits as f32 / total.max(1) as f32
    }

    fn perturb(&self, answer: Option<PhotoPx>, query: PhotoPx, stream: u64) -> Option<PhotoPx> {
        let answer = answer?;
        let h = hash3(
            hash3(self.seed, stream, 0),
            query.x().to_bits() as u64,
            query.y().to_bits() as u64,
        );
        if unit(hash3(h, 1, 0)) < self.outlier_fraction as f64 {
            return Some(PhotoPx::new(
                (unit(hash3(h, 2, 0)) * (self.set.width - 1) as f64) as f32,
                (unit(hash3(h, 3, 0)) * (self.set.height - 1) as f64) as f32,
            ));
        }
        if self.noise_px == 0.0 {
            return Some(answer);
        }
        // Box–Muller.
        let (u1, u2) = (unit(hash3(h, 4, 0)), unit(hash3(h, 5, 0)));
        let radius = (-2.0 * (1.0 - u1).ln()).sqrt() * self.noise_px as f64;
        let angle = std::f64::consts::TAU * u2;
        Some(PhotoPx::new(
            answer.x() + (radius * angle.cos()) as f32,
            answer.y() + (radius * angle.sin()) as f32,
        ))
    }
}

impl PairLookup for SyntheticPair<'_> {
    fn a_to_b(&self, p: PhotoPx) -> Option<PhotoPx> {
        let exact = self.set.transfer(self.pair.a(), self.pair.b(), p);
        self.perturb(exact, p, 0)
    }

    fn b_to_a(&self, p: PhotoPx) -> Option<PhotoPx> {
        let exact = self.set.transfer(self.pair.b(), self.pair.a(), p);
        self.perturb(exact, p, 1)
    }

    fn coverage(&self) -> f32 {
        self.coverage
    }

    fn native_stride(&self) -> f32 {
        6.0
    }

    fn precision_px(&self) -> f32 {
        self.noise_px.max(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correspondences_round_trip() {
        let set = SyntheticSet::orbit(Scene::sphere(), 3, 30.0, 160, 120);
        let pair = set.pair(PairId::new(ViewId(0), ViewId(2)).unwrap());
        assert!(pair.coverage() > 0.3, "coverage {}", pair.coverage());

        // Interior points only: a point on the border can round to just outside the photo
        // on the way back. A ray grazing the sphere's silhouette can also miss it after the
        // round trip through f32 pixels, so a few misses are allowed.
        let (mut checked, mut missed) = (0, 0);
        for y in (3..117).step_by(7) {
            for x in (3..157).step_by(7) {
                let p = PhotoPx::new(x as f32, y as f32);
                let Some(q) = pair.a_to_b(p) else { continue };
                checked += 1;
                match pair.b_to_a(q) {
                    Some(back) => assert!(
                        (back.x() - p.x()).abs() < 1e-2 && (back.y() - p.y()).abs() < 1e-2,
                        "{p:?} came back as {back:?}"
                    ),
                    None => missed += 1,
                }
            }
        }
        assert!(checked > 100, "only {checked} points mapped");
        assert!(
            missed * 50 < checked,
            "{missed} of {checked} points did not map back"
        );
    }

    #[test]
    fn the_sphere_hides_the_backdrop_behind_it() {
        let set = SyntheticSet::orbit(Scene::sphere(), 2, 40.0, 160, 120);
        let centre = PhotoPx::new(80.0, 60.0);
        let depth = set.depth(ViewId(0), centre).unwrap();
        assert!(
            (depth - 4.0).abs() < 0.1,
            "front of the sphere, got {depth}"
        );

        let behind = World(Point3::new(0.0, 0.0, 2.5));
        assert!(!set.visible(ViewId(0), &behind));
    }

    #[test]
    fn renders_a_textured_photo() {
        let set = SyntheticSet::orbit(Scene::corner(), 2, 20.0, 64, 48);
        let photo = set.render(ViewId(1));
        assert_eq!((photo.width(), photo.height()), (64, 48));
        let values: std::collections::BTreeSet<u8> = photo
            .as_rgba()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0])
            .collect();
        assert!(values.len() > 40, "only {} distinct values", values.len());
    }
}
