import math

class CorrespondenceScoring:
    """
    Evaluates similarity between two images using pixel comparisons within a circular neighborhood.
    Scores a given AffineTransform by sampling color differences (R, G, B).
    """

    def __init__(self, photo1, photo2, neighborhood_radius):
        """
        Creates a new CorrespondenceScoring instance.

        :param photo1: A 'Photo'-like object with width, height, img_data.
        :param photo2: Another Photo-like object.
        :param neighborhood_radius: The radius of the circular neighborhood used for comparisons.
        """
        self.photo1 = photo1
        self.photo2 = photo2

        self.neighborhood_radius = neighborhood_radius
        self.total_invocations = 0  # Replace Cell<usize> from Rust

        # Precompute sqrt_table for y offsets in [-radius, radius].
        diameter = (2 * neighborhood_radius + 1)
        self.sqrt_table = [0]*(diameter)
        radius2 = neighborhood_radius * neighborhood_radius

        # Fill in sqrt_table. For each y in the circle, compute max x extent.
        idx = 0
        for y in range(-neighborhood_radius, neighborhood_radius + 1):
            # index = y + neighborhood_radius
            # to handle negative y, shift by neighborhood_radius
            table_index = y + neighborhood_radius
            yy = y*y
            # floor(sqrt(radius^2 - y^2))
            self.sqrt_table[table_index] = int(math.floor(math.sqrt(radius2 - yy)))
            idx += 1

    def calculate_similarity_score(self, circle_mapping):
        """
        Calculates a variance-based color difference score over a circular neighborhood
        defined by 'circle_mapping' (an AffineTransform).

        :param circle_mapping: An AffineTransform with fields:
                              origin_x, origin_y, translate_x, translate_y,
                              a11, a12, a21, a22
        :return: A float score (lower is better). If no pixels are compared, returns float('inf').
        """
        count = 0
        sum_dr = 0
        sum_dg = 0
        sum_db = 0
        sum_dr2 = 0
        sum_dg2 = 0
        sum_db2 = 0

        # Extract fields from transform
        x1 = circle_mapping.origin_x
        y1 = circle_mapping.origin_y
        kx1 = circle_mapping.a11
        kx2 = circle_mapping.a12
        ky1 = circle_mapping.a21
        ky2 = circle_mapping.a22
        x2_offset = circle_mapping.translate_x
        y2_offset = circle_mapping.translate_y

        # For indexing
        photo1_width = self.photo1.shape[1]
        photo2_width = self.photo2.shape[1]
        photo1_data = self.photo1
        photo2_data = self.photo2
        radius = self.neighborhood_radius

        for dy in range(-radius, radius+1):
            # The table_index for dy is (dy + radius)
            table_index = (dy + radius)
            xx = self.sqrt_table[table_index]
            yy1 = y1 + dy

            # For x in [-xx, xx]
            for dx in range(-xx, xx+1):
                xx1 = x1 + dx

                # Check bounds in photo1 (we need pixel1index..pixel1index+3 in range).
                if xx1 < 0 or xx1 >= photo1_data.shape[1] or yy1 < 0 or yy1 >= photo1_data.shape[0]:
                    continue

                # Inline extrapolate (no separate function call).
                # x2f = (dx)*kx1 + (dy)*kx2 + x2_offset
                # y2f = (dx)*ky1 + (dy)*ky2 + y2_offset
                x2f = dx*kx1 + dy*kx2 + x2_offset
                y2f = dx*ky1 + dy*ky2 + y2_offset
                x2 = int(round(x2f))
                y2 = int(round(y2f))

                # Check bounds in photo2
                if x2 < 0 or x2 >= photo2_data.shape[1] or y2 < 0 or y2 >= photo2_data.shape[0]:
                    continue

                # Access R, G, B from each image
                r1 = int(photo1_data[yy1][xx1][0])
                g1 = int(photo1_data[yy1][xx1][1])
                b1 = int(photo1_data[yy1][xx1][2])

                r2 = int(photo2_data[y2][x2][0])
                g2 = int(photo2_data[y2][x2][1])
                b2 = int(photo2_data[y2][x2][2])

                # Convert to Python int (just in case they're bytes).
                dr = (r1 - r2)
                dg = (g1 - g2)
                db = (b1 - b2)

                sum_dr += dr
                sum_dg += dg
                sum_db += db

                sum_dr2 += dr*dr
                sum_dg2 += dg*dg
                sum_db2 += db*db

                count += 1

        if count == 0:
            return float('inf')  # No pixels were compared

        # Compute means for R/G/B diffs
        mean_dr = sum_dr / count
        mean_dg = sum_dg / count
        mean_db = sum_db / count

        # Variance = E(X^2) - [E(X)]^2
        # For R, G, B combined
        variance_dr = (sum_dr2 / count) - (mean_dr*mean_dr)
        variance_dg = (sum_dg2 / count) - (mean_dg*mean_dg)
        variance_db = (sum_db2 / count) - (mean_db*mean_db)

        score = variance_dr + variance_dg + variance_db

        # Increment total scoring calls
        self.total_invocations += 1

        return float(score)

    def get_num_comparisons(self):
        """
        Returns the total number of similarity score calculations performed.
        """
        return self.total_invocations
