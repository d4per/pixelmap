//! Feature pair type for feature matching

use crate::circular_feature_descriptor::CircularFeatureDescriptor;

/// A pair of matching features from two different images
#[derive(Clone)]
pub struct FeaturePair(pub CircularFeatureDescriptor, pub CircularFeatureDescriptor);

impl FeaturePair {
    /// Create a new feature pair
    pub fn new(desc1: CircularFeatureDescriptor, desc2: CircularFeatureDescriptor) -> Self {
        Self(desc1, desc2)
    }
    
    /// Get the first descriptor
    pub fn first(&self) -> &CircularFeatureDescriptor {
        &self.0
    }
    
    /// Get the second descriptor
    pub fn second(&self) -> &CircularFeatureDescriptor {
        &self.1
    }
    
    /// Get both descriptors as a tuple
    pub fn as_tuple(&self) -> (&CircularFeatureDescriptor, &CircularFeatureDescriptor) {
        (&self.0, &self.1)
    }
    
    /// Convert to a tuple (consumes the FeaturePair)
    pub fn into_tuple(self) -> (CircularFeatureDescriptor, CircularFeatureDescriptor) {
        (self.0, self.1)
    }
}

// Add this IntoIterator implementation to make `for (p1, p2) in pairs` work
impl IntoIterator for Vec<FeaturePair> {
    type Item = (CircularFeatureDescriptor, CircularFeatureDescriptor);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    
    fn into_iter(self) -> Self::IntoIter {
        let tuples: Vec<_> = self.into_iter().map(|pair| pair.into_tuple()).collect();
        tuples.into_iter()
    }
}

// Also implement for &'a Vec<FeaturePair> for cases where we don't want to consume the vector
impl<'a> IntoIterator for &'a Vec<FeaturePair> {
    type Item = (&'a CircularFeatureDescriptor, &'a CircularFeatureDescriptor);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    
    fn into_iter(self) -> Self::IntoIter {
        let tuples: Vec<_> = self.iter().map(|pair| pair.as_tuple()).collect();
        tuples.into_iter()
    }
}
