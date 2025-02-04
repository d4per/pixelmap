import math
import numpy as np
import copy

class DensePhotoMap:
    """
    A dense 2D mapping between two images (photo1, photo2), each stored as NumPy arrays.
    The map is a grid of size (grid_width x grid_height), with each cell holding (x2, y2).
    Most methods replicate the Rust logic from dense_photo_map.rs.
    """

    def __init__(self, photo1: np.ndarray, photo2: np.ndarray,
                 grid_width: int, grid_height: int):
        """
        :param photo1: np.ndarray for the first image, shape = (H, W, C).
        :param photo2: np.ndarray for the second image, shape = (H, W, C).
        :param grid_width: number of columns in the mapping grid
        :param grid_height: number of rows in the mapping grid
        """
        self.photo1 = photo1   # shape (height, width, channels)
        self.photo2 = photo2
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Equivalent to: photo1.width / (grid_width - 1) in Rust
        # width is photo1.shape[1]
        if grid_width < 2:
            raise ValueError("grid_width must be >= 2")

        self.grid_cell_size = self.photo1.shape[1] // (grid_width - 1)

        # map_data: length = grid_width * grid_height * 2  (storing x2,y2)
        total_cells = grid_width * grid_height * 2
        self.map_data = [math.nan] * total_cells

    def copy(self):
        """
        Creates a 'clone' of this DensePhotoMap.
        The same photo references are used (they're NumPy arrays),
        but map_data is a new copy of the list.
        """
        new_obj = DensePhotoMap(
            self.photo1,
            self.photo2,
            self.grid_width,
            self.grid_height
        )
        new_obj.grid_cell_size = self.grid_cell_size
        new_obj.map_data = self.map_data[:]  # shallow copy of list
        return new_obj

    def get_grid_cell_size(self) -> int:
        """
        Returns the number of pixels each grid cell covers horizontally in photo1.
        """
        return self.grid_cell_size

    def set_grid_coordinates(self, x1: int, y1: int, x2: float, y2: float):
        """
        Sets the mapped coordinates (x2, y2) in this map at grid location (x1, y1).
        If out of range, does nothing.
        """
        if x1 >= self.grid_width or y1 >= self.grid_height:
            return
        index = (y1 * self.grid_width + x1) * 2
        self.map_data[index] = x2
        self.map_data[index + 1] = y2

    def get_grid_coordinates(self, x1: int, y1: int) -> (float, float):
        """
        Retrieves the mapped coordinates (x2, y2) from this map at grid location (x1, y1).
        Returns (NaN, NaN) if out of range or if the cell was never set.
        """
        if x1 >= self.grid_width or y1 >= self.grid_height:
            return (math.nan, math.nan)
        index = (y1 * self.grid_width + x1) * 2
        return (self.map_data[index], self.map_data[index + 1])

    def map_photo_pixel(self, x1: float, y1: float) -> (float, float):
        """
        Maps a pixel (x1, y1) from photo1 to the corresponding location in photo2,
        using bilinear interpolation of the grid cells.
        """
        gx = x1 / float(self.grid_cell_size)
        gy = y1 / float(self.grid_cell_size)
        return self.get_interpolated_point(gx, gy)

    def get_interpolated_point(self, xin: float, yin: float) -> (float, float):
        """
        Interpolates the mapping for a fractional grid coordinate (xin, yin) using bilinear interpolation.
        Uses the four surrounding corners in the grid. If any corner is NaN or
        the data fails distance checks, returns (NaN, NaN).
        """
        xxx = int(xin)
        yyy = int(yin)
        xr = xin - xxx
        yr = yin - yyy

        # Gather corner mappings: (xxx, yyy), (xxx+1, yyy), (xxx+1, yyy+1), (xxx, yyy+1)
        (x1, y1) = self.get_grid_coordinates(xxx, yyy)
        (x2, y2) = self.get_grid_coordinates(xxx + 1, yyy)
        (x3, y3) = self.get_grid_coordinates(xxx + 1, yyy + 1)
        (x4, y4) = self.get_grid_coordinates(xxx, yyy + 1)

        # If any corner is NaN, cannot interpolate
        if any(math.isnan(v) for v in [x1, x2, x3, x4]):
            return (math.nan, math.nan)

        # Distance check: no corner can be too far from the center
        cell_width = float(self.grid_cell_size)
        max_dist_sq = (cell_width * 3.0) ** 2
        center_x = (x1 + x2 + x3 + x4) / 4.0
        center_y = (y1 + y2 + y3 + y4) / 4.0

        for (cx, cy) in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
            dist_sq = (center_x - cx) ** 2 + (center_y - cy) ** 2
            if dist_sq > max_dist_sq:
                return (math.nan, math.nan)

        # Perform bilinear interpolation
        xx1 = x1 * (1.0 - xr) + x2 * xr
        yy1 = y1 * (1.0 - xr) + y2 * xr

        xx2 = x4 * (1.0 - xr) + x3 * xr
        yy2 = y4 * (1.0 - xr) + y3 * xr

        xt = xx1 * (1.0 - yr) + xx2 * yr
        yt = yy1 * (1.0 - yr) + yy2 * yr

        return (xt, yt)

    def remove_outliers(self, other: 'DensePhotoMap', max_dist: float):
        """
        Removes "outlier" mappings by checking forward-backward consistency with 'other'.
        - For cell (x, y), we map (x*grid_cell_size, y*grid_cell_size) forward to photo2 => mapped
        - Then use 'other' to map 'mapped' back to photo1 => mapped_back
        - If the round trip is too far (distance > max_dist in grid units), mark as invalid => set to NaN.
        """
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                mapped = self.map_photo_pixel(x * self.grid_cell_size,
                                              y * self.grid_cell_size)
                if math.isnan(mapped[0]):
                    # Already invalid
                    self.set_grid_coordinates(x, y, math.nan, math.nan)
                else:
                    mapped_back = other.map_photo_pixel(mapped[0], mapped[1])
                    if math.isnan(mapped_back[0]):
                        self.set_grid_coordinates(x, y, math.nan, math.nan)
                    else:
                        dx = x - (mapped_back[0] / float(self.grid_cell_size))
                        dy = y - (mapped_back[1] / float(self.grid_cell_size))
                        dist_sq = dx*dx + dy*dy
                        if dist_sq > max_dist:
                            self.set_grid_coordinates(x, y, math.nan, math.nan)

    def calculate_used_area(self) -> float:
        """
        Returns the fraction of valid (non-NaN) cells in this map.
        A value between 0.0 and 1.0.
        """
        count_valid = 0
        total = self.grid_width * self.grid_height
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                (x2, _) = self.get_grid_coordinates(x, y)
                if not math.isnan(x2):
                    count_valid += 1
        return count_valid / float(total)

    def interpolate_photo(self, interpolation_value: float, detail_level: int) -> np.ndarray:
        """
        Generates a new image (NumPy array) by interpolating between:
          - the original coordinate (x, y)
          - the mapped coordinate (x1, y1)
        blending them according to interpolation_value in [0,1].

        If mapped coords are NaN, paints the pixel red.
        Otherwise, it takes the color from photo1 at (floor(x), floor(y)).

        :param interpolation_value: 0.0 => all original coords, 1.0 => all mapped coords
        :param detail_level: how many sub-pixels to iterate over (like super-sampling).
        :return: a NumPy array (H, W, 4) with RGBA data, same size as photo1.
        """
        # Clamp the interpolation factor
        interpolation_value = max(0.0, min(1.0, interpolation_value))

        height, width = self.photo1.shape[0], self.photo1.shape[1]
        channels = self.photo1.shape[2] if self.photo1.ndim == 3 else 1
        # We'll build an RGBA output with shape (height, width, 4).
        interpolated_img_data = np.zeros((height, width, 4), dtype=np.uint8)

        for yi in range(height * detail_level):
            y = yi / float(detail_level)
            for xi in range(width * detail_level):
                x = xi / float(detail_level)
                (x_mapped, y_mapped) = self.map_photo_pixel(x, y)
                if math.isnan(x_mapped):
                    continue

                # Interpolate final coordinate
                x_interpolated = x*(1.0 - interpolation_value) + x_mapped*interpolation_value
                y_interpolated = y*(1.0 - interpolation_value) + y_mapped*interpolation_value

                # Round to nearest pixel in photo1
                xx1 = int(round(x_interpolated))
                yy1 = int(round(y_interpolated))

                if 0 <= xx1 < width and 0 <= yy1 < height:
                    # If invalid => paint red
                    if math.isnan(x_mapped):
                        interpolated_img_data[yy1, xx1, 0] = 255  # R
                        interpolated_img_data[yy1, xx1, 1] = 0    # G
                        interpolated_img_data[yy1, xx1, 2] = 0    # B
                        interpolated_img_data[yy1, xx1, 3] = 255  # A
                    else:
                        # Use color from photo1[y, x]
                        # Check bounds for x,y in photo1 as integers
                        orig_x = int(math.floor(x))
                        orig_y = int(math.floor(y))
                        if 0 <= orig_x < width and 0 <= orig_y < height:
                            rgb = self.photo1[orig_y, orig_x]
                            # If the original array has only 3 channels (RGB),
                            # we set alpha=255. If it has 4, we keep it.
                            r = rgb[0]
                            g = rgb[1]
                            b = rgb[2]
                            a = 255
                            if channels == 4:
                                a = rgb[3]
                            interpolated_img_data[yy1, xx1] = [r, g, b, a]

        return interpolated_img_data

    def smooth_grid_points_n_times(self, iterations: int) -> 'DensePhotoMap':
        """
        Repeatedly applies average_grid_points 'iterations' times.
        Returns a new DensePhotoMap that is smoothed.
        """
        pm = self.copy()
        for _ in range(iterations):
            pm = pm.average_grid_points()
        return pm

    def average_grid_points(self) -> 'DensePhotoMap':
        """
        Creates a new DensePhotoMap where each interior cell is replaced by
        the average of its left/right/up/down neighbors, if valid (non-NaN)
        and not too far from the center. This helps smooth the mapping.
        """
        result = self.copy()
        max_dist = float(self.grid_cell_size) * 4.0

        for y in range(1, self.grid_height - 1):
            for x in range(1, self.grid_width - 1):
                (a1x, a1y) = self.get_grid_coordinates(x - 1, y)
                (a2x, a2y) = self.get_grid_coordinates(x + 1, y)
                (b1x, b1y) = self.get_grid_coordinates(x, y - 1)
                (b2x, b2y) = self.get_grid_coordinates(x, y + 1)

                # Center average
                center_x = (a1x + a2x + b1x + b2x) / 4.0
                center_y = (a1y + a2y + b1y + b2y) / 4.0

                # Distances for neighbor validity
                def valid_pair(px, py):
                    if math.isnan(px):
                        return False
                    dx = px - center_x
                    dy = py - center_y
                    dist = math.sqrt(dx*dx + dy*dy)
                    return dist < max_dist

                aa_valid = valid_pair(a1x, a1y) and valid_pair(a2x, a2y)
                bb_valid = valid_pair(b1x, b1y) and valid_pair(b2x, b2y)

                if aa_valid and bb_valid:
                    avg_x = (a1x + a2x + b1x + b2x) / 4.0
                    avg_y = (a1y + a2y + b1y + b2y) / 4.0
                    result.set_grid_coordinates(x, y, avg_x, avg_y)
                elif aa_valid:
                    avg_x = (a1x + a2x) / 2.0
                    avg_y = (a1y + a2y) / 2.0
                    result.set_grid_coordinates(x, y, avg_x, avg_y)
                elif bb_valid:
                    avg_x = (b1x + b2x) / 2.0
                    avg_y = (b1y + b2y) / 2.0
                    result.set_grid_coordinates(x, y, avg_x, avg_y)

        return result
