import math
import numpy as np
from math import isnan

class TexturePoint:
    """
    Represents a single point's 3D position (x, y, z), texture coordinates (u, v),
    and the (grid_x, grid_y) location in the Model3D grid.
    """

    __slots__ = ['x', 'y', 'z', 'u', 'v', 'grid_x', 'grid_y']

    def __init__(self, x=math.nan, y=math.nan, z=math.nan,
                 u=0.0, v=0.0,
                 grid_x=0, grid_y=0):
        self.x = x
        self.y = y
        self.z = z
        self.u = u
        self.v = v
        self.grid_x = grid_x
        self.grid_y = grid_y

    def distance(self, other):
        """
        Euclidean distance between self and other, using (x, y, z).
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


class Model3D:
    """
    A 3D model built from a 2D grid of correspondences (DensePhotoMap).
    Each cell in the grid has a (x,y,z) in 3D plus texture coordinates (u, v).
    """

    def __init__(self, photo_mapping):
        """
        Replicates the logic of Model3D::new in Rust.
        The constructor performs two passes of SVD-based dimensionality reduction.
        """
        # 1. Clone the photo_mapping so we can invalidate cells on each iteration
        #    without mutating the original.
        self.photo_mapping = photo_mapping.clone()  # or a custom "deepcopy" if needed

        self.grid_width = self.photo_mapping.grid_width
        self.grid_height = self.photo_mapping.grid_height
        self.photo = self.photo_mapping.photo1  # reference to Photo for texturing

        # Initialize a flat list of TexturePoint with default=NaN.
        self.grid = [TexturePoint() for _ in range(self.grid_width * self.grid_height)]

        grid_cell_size = self.photo_mapping.get_grid_cell_size()

        # We'll do 2 cleanup iterations:
        for cleanup_iteration in range(2):
            # 2A. Collect valid points from the photo_mapping (x, y) => (data_x, data_y).
            valid_points = []
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    data_x, data_y = self.photo_mapping.get_grid_coordinates(x, y)
                    if not math.isnan(data_x):
                        # We store [X, Y, dataX, dataY]
                        # where X = x * grid_cell_size, Y = y * grid_cell_size
                        valid_points.append((
                            float(x*grid_cell_size),
                            float(y*grid_cell_size),
                            float(data_x),
                            float(data_y)
                        ))

            print(f"valid points size: {len(valid_points)} iteration {cleanup_iteration}")

            if len(valid_points) == 0:
                # If no valid points remain, we can break early
                break

            # 2B. Build an Nx4 matrix from valid_points.
            #     Each row is [X, Y, dataX, dataY].
            matrix = np.zeros((len(valid_points), 4), dtype=np.float64)
            for i, (X, Y, dataX, dataY) in enumerate(valid_points):
                matrix[i, 0] = X
                matrix[i, 1] = Y
                matrix[i, 2] = dataX
                matrix[i, 3] = dataY

            # 2C. Compute column means, then center the data around 0.
            col_means = self._compute_column_means(matrix)
            self._center_data(matrix, col_means)

            # 2D. Perform SVD on the centered data, keep first 3 principal components.
            #     np.linalg.svd returns U, s, Vt
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            # Vt has shape (4, 4). We want the first 3 rows of Vt => shape (3,4).
            # In Rust, it took `v_t.rows(0,3)`. We'll do the same slicing in Python.
            V_reduced = Vt[0:3, :]  # shape (3,4)

            # 2E. Project original Nx4 data onto the new 3D basis => Nx3
            #     i.e. reduced_data = matrix * V_reduced^T in Rust.
            # In Python: shape(matrix) is (N,4), shape(V_reduced.T) is (4,3).
            # So shape(reduced_data) => (N,3)
            reduced_data = matrix @ V_reduced.T

            # 2F. Determine min/max using the 10th and 90th percentile in each of the 3 new dims.
            min_values, max_values = self._compute_min_max(reduced_data, 0.1, 0.9)

            # 2G. Now update each cell in the DensePhotoMap. For each valid (x, y),
            #     build a row [X, Y, dataX, dataY], center, project, rescale,
            #     then store or invalidate.
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    data_x, data_y = self.photo_mapping.get_grid_coordinates(x, y)
                    if math.isnan(data_x):
                        continue

                    # Build 1x4 row
                    point_row = np.array([
                        float(x*grid_cell_size),
                        float(y*grid_cell_size),
                        float(data_x),
                        float(data_y)
                    ], dtype=np.float64)

                    # Apply centering
                    for col in range(4):
                        point_row[col] -= col_means[col]

                    # Now point_row shape is (4,). We want a (1,4) for multiplication.
                    point_row = point_row.reshape(1, 4)
                    # data_3d => shape (1, 3)
                    data_3d = point_row @ V_reduced.T

                    # data_3d[0,0..2] are the three coords.
                    xx = self._rescale_value(data_3d[0,0], min_values[0], max_values[0])
                    yy = self._rescale_value(data_3d[0,1], min_values[1], max_values[1])
                    zz = self._rescale_value(data_3d[0,2], min_values[2], max_values[2])

                    # If valid, store in self.grid. Otherwise, set cell invalid => NaN
                    if (not math.isnan(xx) and not math.isnan(yy) and not math.isnan(zz)
                        and abs(xx) < 3.0 and abs(yy) < 3.0 and abs(zz) < 3.0):
                        # fill a TexturePoint
                        self.grid[y*self.grid_width + x] = TexturePoint(
                            x=xx,
                            y=yy,
                            z=zz,
                            u=float(x)/self.grid_width,
                            v=1.0 - (float(y)/self.grid_height),
                            grid_x=x,
                            grid_y=y
                        )
                    else:
                        # Invalidate for next iteration
                        self.photo_mapping.set_grid_coordinates(x, y, float('nan'), float('nan'))

    def get_texture_point(self, x, y):
        """
        Retrieves the TexturePoint at (x, y).
        Raises IndexError if out of range.
        """
        if x >= self.grid_width or y >= self.grid_height:
            raise IndexError("Out-of-bounds in Model3D grid.")
        return self.grid[y*self.grid_width + x]

    def get_X3D(self):
        """
        Creates and returns an X3D string representing the 3D mesh of points.
        Each cell is a node in the mesh. We form quads for adjacent cells, then
        output them as two triangles in an <IndexedFaceSet>.
        """
        result = []
        result.append(
            """<X3D width="1000px" height="1000px">
    <head>
        <meta name='title' content='3D Model'/>
        <meta name='description' content='3D Model with texture'/>
    </head>
    <Scene>
        <Shape>
            <Appearance>
                <ImageTexture id="imagetexture" url='[photo_placeholder]'></ImageTexture>
            </Appearance>
            <IndexedFaceSet solid="false" ccw="true" colorPerVertex="false" coordIndex='"""
        )

        # Build triangles for each quad (x, y)-(x+1, y)-(x, y+1)-(x+1, y+1)
        for y in range(self.grid_height - 1):
            for x in range(self.grid_width - 1):
                p1 = self.get_texture_point(x, y)
                p2 = self.get_texture_point(x+1, y)
                p3 = self.get_texture_point(x, y+1)
                p4 = self.get_texture_point(x+1, y+1)

                # Skip if any point is invalid
                if (math.isnan(p1.x) or math.isnan(p2.x) or
                    math.isnan(p3.x) or math.isnan(p4.x)):
                    continue

                # Also skip if points are too far from each other (>0.5 distance)
                if (p1.distance(p2) > 0.5 or p1.distance(p3) > 0.5 or
                    p1.distance(p4) > 0.5):
                    continue

                # Build two triangles: (p1->p2->p4) and (p1->p4->p3)
                # X3D uses -1 to separate faces.
                # The index of p in the flattened array is (y*grid_width + x).
                idx_p1 = y*self.grid_width + x
                idx_p2 = y*self.grid_width + (x+1)
                idx_p3 = (y+1)*self.grid_width + x
                idx_p4 = (y+1)*self.grid_width + (x+1)

                # Tri 1: p1, p2, p4
                result.append(f"{idx_p1} {idx_p2} {idx_p4} -1 ")
                # Tri 2: p1, p4, p3
                result.append(f"{idx_p1} {idx_p4} {idx_p3} -1 ")

        result.append("'>\n<Coordinate point='")

        # Next, output the 3D coordinates for each point in row-major order
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                p = self.get_texture_point(x, y)
                if math.isnan(p.x):
                    # If invalid, place them at 0,0,0
                    result.append("0 0 0 ")
                else:
                    result.append(f"{p.x} {p.y} {p.z} ")

        result.append("'></Coordinate>\n<TextureCoordinate point='")

        # Output (u, v) for each point
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                p = self.get_texture_point(x, y)
                if math.isnan(p.x):
                    # invalid => 0,0
                    result.append("0 0 ")
                else:
                    result.append(f"{p.u} {p.v} ")

        result.append(
            """'></TextureCoordinate>
            </IndexedFaceSet>
        </Shape>
    </Scene>
</X3D>"""
        )

        return "".join(result)

    # ----------------------------------------------------------------
    # Helper methods to replicate the Rust 'compute_column_means',
    # 'center_data', 'compute_min_max', 'rescale_value', etc.
    # ----------------------------------------------------------------

    def _compute_column_means(self, matrix):
        """
        Computes column-wise means for a NumPy 2D array. Returns a list of length matrix.shape[1].
        """
        return np.mean(matrix, axis=0)

    def _center_data(self, matrix, col_means):
        """
        Subtracts col_means from each column in 'matrix' to center data around zero.
        """
        matrix -= col_means  # NumPy broadcasting

    def _compute_min_max(self, data, lower_percentile, upper_percentile):
        """
        Finds the lower_percentile and upper_percentile for each column in 'data'.
        Returns two lists: (min_values, max_values), each of length data.shape[1].
        """
        ncols = data.shape[1]
        min_values = []
        max_values = []

        for c in range(ncols):
            col = data[:, c]
            sorted_col = np.sort(col)
            size = len(sorted_col)
            lower_idx = int(size * lower_percentile)
            upper_idx = int(size * upper_percentile)
            # clamp indices
            lower_idx = max(0, min(lower_idx, size-1))
            upper_idx = max(0, min(upper_idx, size-1))

            min_values.append(sorted_col[lower_idx])
            max_values.append(sorted_col[upper_idx])

        return min_values, max_values

    def _rescale_value(self, value, min_val, max_val):
        """
        Maps 'value' from the range [min_val, max_val] to [-1, 1].
        If max_val == min_val, this may produce NaN.
        """
        if max_val == min_val:
            return float('nan')
        return ((value - min_val)/(max_val - min_val))*2.0 - 1.0
