use crate::circular_feature_grid::CircularFeatureGrid;
use crate::circular_feature_descriptor::CircularFeatureDescriptor;
use kd_tree::{KdPoint, KdTree};

impl KdPoint for CircularFeatureDescriptor {
    type Scalar = i64;
    type Dim = typenum::U6;
    fn at(&self, k: usize) -> i64 {
        self.feature_vector[k]
    }
}

/// Provides a way to match circular feature descriptors between two images.
///
/// This struct uses a [KdTree] (built from the `kd_tree` crate) to find the
/// nearest neighbor from one set of `CircularFeatureDescriptor` objects
/// (`CircularFeatureGrid`) for each descriptor in another. This allows for
/// quick and efficient matching of similar descriptors across two images.
///
pub struct CircularFeatureDescriptorMatcher;

impl CircularFeatureDescriptorMatcher {

    pub fn new() -> Self {
        CircularFeatureDescriptorMatcher { }
    }

    pub fn match_areas(
        &self,
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
    ) -> Vec<(CircularFeatureDescriptor, CircularFeatureDescriptor)> {
        let kdtree = KdTree::build(img1.get_infos().clone());
        let mut ans: Vec<(CircularFeatureDescriptor, CircularFeatureDescriptor)> = Vec::new();
        for cai2 in img2.get_infos() {
            let nearest = kdtree.nearest(cai2);
            if nearest.is_some() {
                let cai1 = nearest.unwrap().item;
                ans.push((*cai1, *cai2));
            }
        }
        ans
    }
}
