import math

from src.circlualr_feature_descriptor import CircularFeatureDescriptor


class CircularFeatureGrid:
    """
    Computes and stores CircularFeatureDescriptor values for each (x, y) in an image.
    The descriptors capture the color channel 'center of mass' in a circular neighborhood.
    """

    def __init__(self, image_data, width, height, circle_radius):
        """
        Creates a new CircularFeatureGrid (by default using rotation-based computations).

        :param image_data: a flat list/array of RGBA bytes, length = width*height*4
                           (or at least 3 channels for R,G,B).
        :param width:      the image width in pixels
        :param height:     the image height in pixels
        :param circle_radius: the radius of the circular neighborhood
        """
        # Default to 'rotation=True' for consistency with the Rust code's `new()`.
        self.__init_with_rotation(image_data, width, height, circle_radius, rotation=True)

    def __init_with_rotation(self, image_data, width, height, circle_radius, rotation):
        """
        Mirrors the Rust `new_with_rotation(...)` constructor.
        """
        self.image_width = width
        self.image_height = height
        self.circle_radius = circle_radius
        self.max_color_value = 0

        # Python version of: vec![default(); width*height]
        self.feature_descriptors = [
            CircularFeatureDescriptor() for _ in range(width * height)
        ]

        # Copy the raw image data.
        # In Rust: `photo.img_data` -> `.to_vec()`
        # In Python, we assume `image_data` is already a 1D array of bytes, length=width*height*4
        self.image_pixel_data = list(image_data)  # or just store as-is if it's already a list

        # Populate the descriptors (with or without rotation).
        # (In the Rust code, the branches are identical, so we do the same logic either way.)
        self.populate_feature_descriptors(rotation)

        # Compute the maximum possible sum of color values in a circle of radius circle_radius
        pixel_count_in_circle = 0
        for y in range(-circle_radius, circle_radius + 1):
            row_width = int(round(math.sqrt(circle_radius**2 - y**2)))
            pixel_count_in_circle += (row_width * 2) + 1

        # If each pixel can contribute 255 in a channel, multiply by 255
        # (the Rust code uses this for normalization if needed).
        self.max_color_value = pixel_count_in_circle * 255

    def get_infos(self):
        """
        Returns the list of CircularFeatureDescriptor objects for the entire grid.
        """
        return self.feature_descriptors

    def populate_feature_descriptors(self, rotation):
        """
        Fills in the feature_descriptors for every (x, y) position in the grid.
        If rotation=False, you could skip some advanced logic.
        But in the Rust code, both branches are identical.
        """
        for y in range(self.image_height):
            for x in range(self.image_width):
                idx = x + y * self.image_width
                # If we had different logic for rotation, we might do that here.
                # But the Rust code calls the same method in both branches.
                self.feature_descriptors[idx] = self.compute_circular_feature_descriptor(
                    x, y, self.circle_radius
                )

    def compute_circular_feature_descriptor(self, center_x, center_y, radius):
        """
        Computes a CircularFeatureDescriptor for the circular region around (center_x, center_y),
        with wrap-around / toroidal addressing.
        """
        # Accumulators for total R/G/B.
        sum_red = 0
        sum_green = 0
        sum_blue = 0

        # Weighted sums for center of mass calculations.
        sum_weighted_red_x = 0
        sum_weighted_red_y = 0
        sum_weighted_green_x = 0
        sum_weighted_green_y = 0
        sum_weighted_blue_x = 0
        sum_weighted_blue_y = 0

        # We'll count how many pixels we've processed (though not used in final code).
        pixel_count = 0

        for dy in range(-radius, radius + 1):
            # row_radius: how far we can go in x for this row
            row_radius = int(round(math.sqrt(radius**2 - dy**2)))

            for dx in range(-row_radius, row_radius + 1):
                # Wrap around edges (toroidal).
                wrapped_x = (center_x + dx + self.image_width) % self.image_width
                wrapped_y = (center_y + dy + self.image_height) % self.image_height

                # In Rust, pixel_index = 4 * (wrapped_x + wrapped_y * self.image_width)
                # We'll replicate that:
                pixel_index = 4 * (wrapped_x + wrapped_y * self.image_width)

                # R, G, B from the flattened array
                red_val   = int(self.image_pixel_data[wrapped_y][wrapped_x][0])
                green_val = int(self.image_pixel_data[wrapped_y][wrapped_x][1])
                blue_val  = int(self.image_pixel_data[wrapped_y][wrapped_x][2])
                # alpha_val = self.image_pixel_data[pixel_index + 3] if needed

                # Accumulate color sums.
                sum_red += red_val
                sum_green += green_val
                sum_blue += blue_val

                # Weighted sums
                sum_weighted_red_x += dx * red_val
                sum_weighted_red_y += dy * red_val

                sum_weighted_green_x += dx * green_val
                sum_weighted_green_y += dy * green_val

                sum_weighted_blue_x += dx * blue_val
                sum_weighted_blue_y += dy * blue_val

                pixel_count += 1

        # Build the descriptor (we'll fill fields similarly to the Rust code).
        descriptor = CircularFeatureDescriptor()

        # For each color, compute center-of-mass X and Y offsets.
        # If sum_red == 0, it means no red in that region, so handle divide-by-zero safely.
        def safe_div(a, b):
            return 0.0 if b == 0 else (a / b)

        red_cm_x = safe_div(sum_weighted_red_x, sum_red)
        red_cm_y = safe_div(sum_weighted_red_y, sum_red)
        green_cm_x = safe_div(sum_weighted_green_x, sum_green)
        green_cm_y = safe_div(sum_weighted_green_y, sum_green)
        blue_cm_x = safe_div(sum_weighted_blue_x, sum_blue)
        blue_cm_y = safe_div(sum_weighted_blue_y, sum_blue)

        # Angles & radii for each color
        red_angle = math.atan2(red_cm_y, red_cm_x) if sum_red != 0 else 0.0
        red_radius = math.hypot(red_cm_x, red_cm_y)

        green_angle = math.atan2(green_cm_y, green_cm_x) if sum_green != 0 else 0.0
        green_radius = math.hypot(green_cm_x, green_cm_y)

        blue_angle = math.atan2(blue_cm_y, blue_cm_x) if sum_blue != 0 else 0.0
        blue_radius = math.hypot(blue_cm_x, blue_cm_y)

        # Total color sum
        sum_all = sum_red + sum_green + sum_blue
        total_cm_x = safe_div(
            sum_weighted_red_x + sum_weighted_green_x + sum_weighted_blue_x,
            sum_all
        )
        total_cm_y = safe_div(
            sum_weighted_red_y + sum_weighted_green_y + sum_weighted_blue_y,
            sum_all
        )

        total_angle = math.atan2(total_cm_y, total_cm_x) if sum_all != 0 else 0.0

        # Fill descriptor
        descriptor.center_x = center_x
        descriptor.center_y = center_y
        descriptor.total_angle = total_angle
        descriptor.total_radius = math.hypot(total_cm_x, total_cm_y)

        # Align each color channel relative to total_angle:
        #   color_angle_relative = color_angle - total_angle
        #   descriptor.aligned_<color>_x = cos(color_angle_relative) * color_radius
        #   descriptor.aligned_<color>_y = sin(color_angle_relative) * color_radius
        def align_coords(color_angle, color_radius):
            rel = color_angle - total_angle
            return (math.cos(rel)*color_radius, math.sin(rel)*color_radius)

        # Red
        (rx, ry) = align_coords(red_angle, red_radius)
        descriptor.aligned_red_x = rx
        descriptor.aligned_red_y = ry

        # Green
        (gx, gy) = align_coords(green_angle, green_radius)
        descriptor.aligned_green_x = gx
        descriptor.aligned_green_y = gy

        # Blue
        (bx, by) = align_coords(blue_angle, blue_radius)
        descriptor.aligned_blue_x = bx
        descriptor.aligned_blue_y = by

        # Fill sums
        descriptor.sum_red = sum_red
        descriptor.sum_green = sum_green
        descriptor.sum_blue = sum_blue

        # Build the feature_vector
        # (6 integers scaled by 100 from the aligned X/Y coords.)
        def scaled_round(val):
            return int(round(val*100.0))

        descriptor.feature_vector[0] = scaled_round(rx)
        descriptor.feature_vector[1] = scaled_round(ry)
        descriptor.feature_vector[2] = scaled_round(gx)
        descriptor.feature_vector[3] = scaled_round(gy)
        descriptor.feature_vector[4] = scaled_round(bx)
        descriptor.feature_vector[5] = scaled_round(by)

        return descriptor
