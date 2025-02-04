import math

class CircularFeatureDescriptor:
    """
    Stores information about the center of mass of color channels in a circular region.
    Each channel (R, G, B) has:
      - total color sums (sum_red, sum_green, sum_blue),
      - a rotated alignment relative to the overall (combined) center of mass (aligned_* fields),
      - and a feature vector (feature_vector) scaled by 100.
    """

    def __init__(self,
                 center_x=0,
                 center_y=0,
                 total_angle=0.0,
                 total_radius=0.0,
                 sum_red=0,
                 sum_green=0,
                 sum_blue=0,
                 aligned_red_x=0.0,
                 aligned_red_y=0.0,
                 aligned_green_x=0.0,
                 aligned_green_y=0.0,
                 aligned_blue_x=0.0,
                 aligned_blue_y=0.0,
                 feature_vector=None):
        """
        If no feature_vector is provided, it defaults to a 6-element list of zeros.
        """
        self.center_x = center_x               # (u16 in Rust)
        self.center_y = center_y               # (u16 in Rust)
        self.total_angle = total_angle         # (f32 in Rust)
        self.total_radius = total_radius       # (f32)
        self.sum_red = sum_red                 # (i32)
        self.sum_green = sum_green             # (i32)
        self.sum_blue = sum_blue               # (i32)

        self.aligned_red_x = aligned_red_x     # (f32)
        self.aligned_red_y = aligned_red_y     # (f32)
        self.aligned_green_x = aligned_green_x # (f32)
        self.aligned_green_y = aligned_green_y # (f32)
        self.aligned_blue_x = aligned_blue_x   # (f32)
        self.aligned_blue_y = aligned_blue_y   # (f32)

        if feature_vector is None:
            self.feature_vector = [0]*6       # (i64 in Rust)
        else:
            self.feature_vector = feature_vector

    def distance(self, other) -> float:
        """
        Computes the Euclidean distance between 'self' and 'other'
        based on aligned color coordinates in R/G/B.
        """
        diff_red_x = self.aligned
