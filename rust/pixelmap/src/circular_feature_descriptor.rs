/// Stores information about the "center of mass" of color channels in a circular region
/// of an image. Each channel (R, G, B) has its total sums recorded as well as a rotated
/// alignment relative to the overall (combined) center of mass.
#[derive(Default, Debug, Clone, Copy)]
pub struct CircularFeatureDescriptor {
    /// The x-coordinate of the descriptor's center in the image grid.
    pub center_x: u16,

    /// The y-coordinate of the descriptor's center in the image grid.
    pub center_y: u16,

    /// The combined angle (using `atan2`) for the total color center of mass in this region.
    pub total_angle: f32,

    /// The magnitude (radius) of the total color center of mass in this region.
    pub total_radius: f32,

    /// The total sum of red values within the circular region.
    pub sum_red: i32,

    /// The total sum of green values within the circular region.
    pub sum_green: i32,

    /// The total sum of blue values within the circular region.
    pub sum_blue: i32,

    /// The "center of mass" X coordinate for the red channel, aligned relative to the `total_angle`.
    pub aligned_red_x: f32,

    /// The "center of mass" Y coordinate for the red channel, aligned relative to the `total_angle`.
    pub aligned_red_y: f32,

    /// The "center of mass" X coordinate for the green channel, aligned relative to the `total_angle`.
    pub aligned_green_x: f32,

    /// The "center of mass" Y coordinate for the green channel, aligned relative to the `total_angle`.
    pub aligned_green_y: f32,

    /// The "center of mass" X coordinate for the blue channel, aligned relative to the `total_angle`.
    pub aligned_blue_x: f32,

    /// The "center of mass" Y coordinate for the blue channel, aligned relative to the `total_angle`.
    pub aligned_blue_y: f32,

    /// An integer-based feature vector (scaled by 100) derived from the aligned color coordinates.
    pub feature_vector: [i64; 6],
}

impl CircularFeatureDescriptor {
    /// Computes the Euclidean distance between two `CircularFeatureDescriptor` objects
    /// based on their aligned R/G/B coordinates (e.g., `aligned_red_x`, `aligned_red_y`, etc.).
    /// This provides a simple measure of how "similar" or "different" two descriptors are.
    pub fn distance(&self, other: &CircularFeatureDescriptor) -> f32 {
        let diff_red_x = self.aligned_red_x - other.aligned_red_x;
        let diff_red_y = self.aligned_red_y - other.aligned_red_y;
        let diff_green_x = self.aligned_green_x - other.aligned_green_x;
        let diff_green_y = self.aligned_green_y - other.aligned_green_y;
        let diff_blue_x = self.aligned_blue_x - other.aligned_blue_x;
        let diff_blue_y = self.aligned_blue_y - other.aligned_blue_y;

        (diff_red_x * diff_red_x
            + diff_red_y * diff_red_y
            + diff_green_x * diff_green_x
            + diff_green_y * diff_green_y
            + diff_blue_x * diff_blue_x
            + diff_blue_y * diff_blue_y)
            .sqrt()
    }
}
