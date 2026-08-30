/// The signature of one circular neighbourhood: where it is, how the combined colour
/// centre of mass is oriented, and the quantized 6-D key the matcher searches on.
///
/// One of these exists per *pixel* of each working image, and both images' grids are
/// live at once, so its size is a hard constraint rather than a detail: at a working
/// width of 1600 the two grids hold about 2.9 million descriptors. The struct used to
/// carry the intermediate values `finish_descriptor` computes on the way to the key —
/// the per-channel sums, radii and unquantized aligned coordinates — at 96 bytes, or
/// ~276 MB for the pair. Nothing outside `circular_feature_grid` ever read them, and
/// keeping only what the matcher uses brings that to 20 bytes and ~58 MB, which is the
/// difference between fitting and not fitting in a wasm32 heap.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct CircularFeatureDescriptor {
    /// The x-coordinate of the descriptor's center in the image grid.
    pub center_x: u16,

    /// The y-coordinate of the descriptor's center in the image grid.
    pub center_y: u16,

    /// The combined angle (using `atan2`) for the total color center of mass in this
    /// region. The difference of two of these is the rotation between a matched pair.
    pub total_angle: f32,

    /// The 6-D search key: the three channels' aligned centres of mass, scaled by 100
    /// and rounded.
    ///
    /// `i16` because an aligned centre of mass cannot leave the disc, so at radius 10 the
    /// magnitude stays around 1000 — two orders of magnitude inside the type. A value
    /// that somehow did overflow saturates rather than wraps (Rust's float-to-int `as`),
    /// so `FeaturePoint::assert_key_range` still catches it loudly.
    pub feature_vector: [i16; 6],
}
