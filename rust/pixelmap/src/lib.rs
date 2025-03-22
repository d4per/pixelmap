//! # PixelMap Library
//!
//! The `pixelmap` library provides a set of tools for image processing, feature matching,
//! and 2D/3D correspondence mapping. It is designed to handle tasks such as
//! finding local alignments between two images (e.g., for stitching, registration, or
//! morphing), computing dense transformation fields, and performing a light 3D
//! reconstruction via dimensionality reduction.
//!
//! ## Overview of Modules
//!
//! - **`pixelmap_processor`**: Orchestrates the high-level workflow of matching two images
//!   by creating feature descriptors, matching them, and refining local transforms.
//!
//! - **`photo`**: Defines a basic `Photo` struct for storing pixel data (RGBA format) along
//!   with methods for scaling, pixel access, and other common image operations.
//!
//! - **`correspondence_mapping_algorithm`**: Implements a grid-based method to iteratively
//!   refine an affine transformation for each cell, guided by a scoring function that measures
//!   how well one region of an image aligns with another.
//!
//! - **`correspondence_scoring`** (private): Contains logic to compare corresponding regions
//!   in two photos (e.g., color differences), returning a similarity score for an affine
//!   mapping.
//!
//! - **`affine_transform`**: Provides the `AffineTransform` struct for 2D transformations,
//!   including translation, rotation, and scale, along with utility methods.
//!
//! - **`affine_transform_cell`**: Wraps an `AffineTransform` and its score in a cell-like
//!   structure, allowing interior mutability and easy storage in grids.
//!
//! - **`dense_photo_map`**: Stores a dense mapping between two photos, in which each cell
//!   in a grid maps a coordinate from `photo1` to a corresponding coordinate in `photo2`.
//!   Supports methods for smoothing, outlier removal, and interpolation of the dense mapping.
//!
//! - **`circular_feature_grid`** (private): Builds a grid of local circular descriptors
//!   for an image, typically used for initial matching or feature extraction.
//!
//! - **`circular_feature_descriptor`** (private): Defines the structure of a circular descriptor
//!   capturing color and orientation data around a region of an image.
//!
//! - **`circular_feature_descriptor_matcher`** (private): Matches circular feature descriptors
//!   between two images, providing candidate correspondences for initialization or refinement.
//!
//! - **`model_3d`**: Creates a 3D mesh/model from a dense mapping (e.g., after PCA-like
//!   dimensionality reduction), along with texture coordinates referencing the original `Photo`.
//!
//! - **`ac_grid`**: Implements a grid of `AffineTransformCell`s, used by the correspondence
//!   mapping algorithm to store and update the best transform found for each grid cell.

pub mod pixelmap_processor;

// Add public modules
pub mod photo;
pub mod circular_feature_descriptor;
pub mod circular_feature_grid;
pub mod dense_photo_map;
pub mod correspondence_mapping_algorithm;
pub mod affine_transform;
pub mod affine_transform_cell;

// Conditional compilation for CUDA support
#[cfg(feature = "cuda")]
pub mod cuda2;

#[cfg(feature = "cuda")]
pub mod cuda_bindings;

pub use cuda_bindings::FeaturePair;

mod cuda {
    pub mod memory;
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Internal Modules
mod correspondence_scoring;
mod circular_feature_descriptor_matcher;

pub mod model_3d;
pub mod ac_grid;
pub mod core {
    /// Module containing core functionality
    // Core implementations will go here
    pub mod core {
        // Core implementations will go here
    }
}
pub mod utils {
    /// Module containing utility functions
    // Utility implementations will go here
    pub mod utils {
        // Utility implementations will go here
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}