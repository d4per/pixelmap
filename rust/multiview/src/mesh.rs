//! A triangle mesh, and the clean-up operations on it.

use nalgebra::{Point3, Vector3};

/// Triangles over shared vertices, in the reconstruction's frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions.
    pub positions: Vec<Point3<f64>>,
    /// One unit normal per vertex, pointing towards the side the cameras saw. Empty until
    /// [`Self::compute_normals`] has run.
    pub normals: Vec<Vector3<f64>>,
    /// Vertex indices, counter-clockwise seen from the side the normal points to.
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    /// Sets each vertex normal to the area-weighted mean of its triangles' normals.
    pub fn compute_normals(&mut self) {
        let mut normals = vec![Vector3::zeros(); self.positions.len()];
        for t in &self.triangles {
            let [a, b, c] = t.map(|i| self.positions[i as usize]);
            // Twice the triangle's area times its unit normal.
            let n = (b - a).cross(&(c - a));
            for &i in t {
                normals[i as usize] += n;
            }
        }
        self.normals = normals
            .into_iter()
            .map(|n| n.try_normalize(1e-300).unwrap_or_else(Vector3::zeros))
            .collect();
    }

    /// The connected piece each triangle belongs to, numbered from 0 in order of first
    /// appearance, and each piece's triangle count.
    pub fn components(&self) -> (Vec<usize>, Vec<usize>) {
        let mut parent: Vec<usize> = (0..self.positions.len()).collect();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        for t in &self.triangles {
            let a = root(&mut parent, t[0] as usize);
            for &v in &t[1..] {
                let b = root(&mut parent, v as usize);
                if a != b {
                    parent[b] = a;
                }
            }
        }

        let mut number = vec![usize::MAX; self.positions.len()];
        let mut sizes = Vec::new();
        let labels = self
            .triangles
            .iter()
            .map(|t| {
                let r = root(&mut parent, t[0] as usize);
                if number[r] == usize::MAX {
                    number[r] = sizes.len();
                    sizes.push(0);
                }
                sizes[number[r]] += 1;
                number[r]
            })
            .collect();
        (labels, sizes)
    }

    /// Drops every connected piece with fewer than `min_fraction` of all triangles, and
    /// any vertices left unused. Returns the fraction of triangles in the largest piece,
    /// measured before dropping anything.
    pub fn keep_large_components(&mut self, min_fraction: f64) -> f64 {
        if self.triangles.is_empty() {
            return 0.0;
        }
        let (labels, sizes) = self.components();
        let total = self.triangles.len() as f64;
        let largest = sizes.iter().copied().max().unwrap_or(0) as f64 / total;
        let keep: Vec<bool> = sizes
            .iter()
            .map(|&s| s as f64 / total >= min_fraction)
            .collect();
        let triangles: Vec<[u32; 3]> = self
            .triangles
            .iter()
            .zip(&labels)
            .filter(|(_, &label)| keep[label])
            .map(|(t, _)| *t)
            .collect();
        self.triangles = triangles;
        self.remove_unused_vertices();
        largest
    }

    fn remove_unused_vertices(&mut self) {
        let mut remap = vec![u32::MAX; self.positions.len()];
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        for t in &mut self.triangles {
            for v in t.iter_mut() {
                let old = *v as usize;
                if remap[old] == u32::MAX {
                    remap[old] = positions.len() as u32;
                    positions.push(self.positions[old]);
                    if let Some(n) = self.normals.get(old) {
                        normals.push(*n);
                    }
                }
                *v = remap[old];
            }
        }
        self.positions = positions;
        self.normals = normals;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(offset: f64) -> (Vec<Point3<f64>>, Vec<[u32; 3]>) {
        (
            vec![
                Point3::new(offset, 0.0, 0.0),
                Point3::new(offset + 1.0, 0.0, 0.0),
                Point3::new(offset + 1.0, 1.0, 0.0),
                Point3::new(offset, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn normals_follow_the_winding() {
        let (positions, triangles) = square(0.0);
        let mut mesh = Mesh {
            positions,
            triangles,
            normals: Vec::new(),
        };
        mesh.compute_normals();
        for n in &mesh.normals {
            assert!((n - Vector3::z()).norm() < 1e-12);
        }
    }

    #[test]
    fn drops_small_pieces() {
        // One piece of two triangles, and one of a single triangle far away.
        let (mut positions, mut triangles) = square(0.0);
        positions.extend([
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(11.0, 0.0, 0.0),
            Point3::new(10.0, 1.0, 0.0),
        ]);
        triangles.push([4, 5, 6]);
        let mut mesh = Mesh {
            positions,
            triangles,
            normals: Vec::new(),
        };
        assert_eq!(mesh.components().1, [2, 1]);

        let largest = mesh.keep_large_components(0.5);
        assert!((largest - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(mesh.triangles.len(), 2);
        assert_eq!(mesh.positions.len(), 4);
    }
}
