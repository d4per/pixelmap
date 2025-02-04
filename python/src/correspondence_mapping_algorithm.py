import math
import random

from src.ac_grid import ACGrid
from src.affine_transform import AffineTransform
from src.correspondence_scoring import CorrespondenceScoring
from src.dense_photo_map import DensePhotoMap
from src.image_util import get_scaled_proportional


class CorrespondenceMappingAlgorithm:
    """
    Manages an iterative process for matching two images (photo1 and photo2) by
    assigning an AffineTransform to each cell in a grid. The algorithm refines
    these transforms using a scoring function until no further improvements occur.
    """

    def __init__(self, photo_width, photo1, photo2, grid_cell_size, neighborhood_radius):
        """
        Creates a new mapping algorithm for the given images and parameters.

        :param photo_width:        The width to which both photos are scaled (preserves aspect ratio).
        :param photo1, photo2:     The original Photo objects (with .width/.height, etc.).
        :param grid_cell_size:     Size of each cell in the grid (in pixels).
        :param neighborhood_radius: Radius (in pixels) for the circular neighborhood scoring.
        """

        # Scale the original photos to the specified width.
        self.photo1 = get_scaled_proportional(photo1, photo_width)
        self.photo2 = get_scaled_proportional(photo2, photo_width)

        self.grid_cell_size = grid_cell_size
        self.queue = []

        # Create a scorer for local comparisons.
        # The constructor of CorrespondenceScoring presumably needs (photo1, photo2, radius).
        self.scorer = CorrespondenceScoring(
            self.photo1,
            self.photo2,
            neighborhood_radius
        )

        # Figure out how many grid cells we have in each dimension.
        grid_width = self.photo1.shape[1] // grid_cell_size + 1
        grid_height = self.photo1.shape[0] // grid_cell_size + 1

        # Create an ACGrid to store the best-known transform for each cell.
        self.ac_grid = ACGrid(grid_width, grid_height)

    def get_ac_grid(self):
        """
        Returns a reference to the internal ACGrid, which stores the best-known transforms for each cell.
        """
        return self.ac_grid

    def get_total_comparisons(self):
        """
        Returns the total number of scoring function invocations performed so far.
        (For debugging/diagnostic use.)
        """
        return self.scorer.get_num_comparisons()

    def run_until_done(self):
        """
        Repeatedly processes (and shuffles) the queue of transforms until no further improvements
        can be made (i.e. the queue is empty at the end of a cycle).
        """
        while True:
            # Shuffle transforms to avoid bias in processing order.
            random.shuffle(self.queue)
            is_done = self._run_queue()  # Returns True if queue is empty after processing
            if is_done:
                break

    def add_init_point(self, x1, y1, x2, y2, angle):
        """
        Adds a new transform to the queue, deriving rotation from 'angle'.
        The transform is then snapped to a corresponding grid coordinate.

        :param x1, y1: Coordinates in photo1
        :param x2, y2: Target coordinates in photo2
        :param angle: Rotation angle (radians) around (x1, y1)
        """
        s = math.sin(angle)
        c = math.cos(angle)

        # Create an affine transform from (x1, y1) to (x2, y2) with rotation.
        cm = AffineTransform(
            origin_x=int(x1),
            origin_y=int(y1),
            translate_x=x2,
            translate_y=y2,
            a11=c,
            a12=-s,
            a21=s,
            a22=c
        )

        # Snap transform's origin to the nearest grid cell boundary.
        snapped_origin_x = (int(x1) // self.grid_cell_size) * self.grid_cell_size
        snapped_origin_y = (int(y1) // self.grid_cell_size) * self.grid_cell_size
        snap_to_grid_cm = cm.extrapolate_mapping(
            snapped_origin_x,
            snapped_origin_y
        )

        self.queue.append(snap_to_grid_cm)

    def init_from_photomapping(self, pm):
        """
        Initializes the queue with transforms derived from an existing DensePhotoMap.
        This allows continuing refinement of a previously computed mapping.

        :param pm: A DensePhotoMap describing approximate mappings from photo1->photo2.
        """
        pm_grid_cell_size = pm.get_grid_cell_size()

        for y in range(pm.grid_height):
            for x in range(pm.grid_width):
                x2a, y2a = pm.get_grid_coordinates(x, y)
                if math.isnan(x2a):
                    # Skip if the cell is invalid (NaN).
                    continue

                # Compute the origin in the scaled photo1.
                #   x1 ~ (x * pm_grid_cell_size * self.photo1.width / pm.photo1.width)
                # Note that in Rust, 'photo1.width' is the scaled photo1's width
                # and 'pm.photo1.width' is the DensePhotoMap's original 'photo1.width'.
                x1 = round(
                    (x * pm_grid_cell_size * self.photo1.shape[1]) / pm.photo1.shape[1]
                )
                y1 = round(
                    (y * pm_grid_cell_size * self.photo1.shape[1]) / pm.photo1.shape[1]
                )

                # Compute mapped position in scaled photo2.
                x2 = x2a * (self.photo1.shape[1] / pm.photo1.shape[1])
                y2 = y2a * (self.photo1.shape[1] / pm.photo1.shape[1])

                cm = AffineTransform(
                    origin_x=x1,
                    origin_y=y1,
                    translate_x=x2,
                    translate_y=y2,
                    a11=1.0,
                    a12=0.0,
                    a21=0.0,
                    a22=1.0
                )

                # Approximate local scaling from neighbors (left/up).
                # If x > 0, check the left neighbor's mapping, etc.
                if x > 0:
                    left_x, left_y = pm.get_grid_coordinates(x - 1, y)
                    if not math.isnan(left_x):
                        cm.a11 = (x2a - left_x) / pm_grid_cell_size
                        cm.a21 = (y2a - left_y) / pm_grid_cell_size

                if y > 0:
                    up_x, up_y = pm.get_grid_coordinates(x, y - 1)
                    if not math.isnan(up_x):
                        cm.a22 = (y2a - up_y) / pm_grid_cell_size
                        cm.a12 = (x2a - up_x) / pm_grid_cell_size

                # Snap the transform's origin to the nearest grid coordinates.
                snapped_grid_x = int(round(x1 / self.grid_cell_size)) * self.grid_cell_size
                snapped_grid_y = int(round(y1 / self.grid_cell_size)) * self.grid_cell_size
                snap_to_grid_cm = cm.extrapolate_mapping(snapped_grid_x, snapped_grid_y)

                self.queue.append(snap_to_grid_cm)

    def init_identity(self):
        """
        Initializes the queue with the "identity" transform for each cell,
        i.e. (x,y)->(x,y) with no rotation/scale.
        """
        # Step over photo1 in increments of grid_cell_size.
        for y in range(0, self.photo1.height, self.grid_cell_size):
            for x in range(0, self.photo1.width, self.grid_cell_size):
                cm = AffineTransform(
                    origin_x=x,
                    origin_y=y,
                    translate_x=float(x),
                    translate_y=float(y),
                    a11=1.0,
                    a12=0.0,
                    a21=0.0,
                    a22=1.0
                )
                self.queue.append(cm)

    def _run_queue(self):
        """
        Processes the current queue of transforms. For each transform:
          1. Validates its scale and position (no out-of-bounds).
          2. Uses optimize_position() to adjust it.
          3. Checks whether it improves upon the stored transform in the grid cell.
          4. If improved, updates that cell and extrapolates to neighboring cells,
             pushing new transforms back into a temporary queue.

        Returns True if the queue is empty afterward (no improvements),
        or False if more transforms remain.
        """
        out_queue = []
        while self.queue:
            cm = self.queue.pop()

            # Skip if invalid scale or out-of-bounds translation.
            if (not cm.is_scale_valid(4.0)) or (not self._is_valid(cm)):
                continue

            # Attempt a small local search to optimize translation.
            score, cm_out = self.optimize_position(cm)

            # Determine which cell (cm_out) belongs to in the ACGrid.
            grid_x = cm_out.origin_x // self.grid_cell_size
            grid_y = cm_out.origin_y // self.grid_cell_size

            if grid_x >= self.ac_grid.get_grid_width() or grid_y >= self.ac_grid.get_grid_height():
                # Out of ACGrid range, skip.
                continue

            # Compare to the current best transform stored in that cell.
            cell = self.ac_grid.get_grid_square(grid_x, grid_y)
            if cell.get_score() > score:
                # Found an improvement: update the cell, and extrapolate to neighbors.
                cell.set(cm_out, score)

                # Generate neighbor transforms for out_queue.
                # Left neighbor
                if grid_x > 0:
                    out_queue.append(
                        cm.extrapolate_mapping(
                            (grid_x - 1)*self.grid_cell_size,
                            grid_y*self.grid_cell_size
                        )
                    )
                # Right neighbor
                if grid_x < self.ac_grid.get_grid_width()-1:
                    out_queue.append(
                        cm.extrapolate_mapping(
                            (grid_x + 1)*self.grid_cell_size,
                            grid_y*self.grid_cell_size
                        )
                    )
                # Up neighbor
                if grid_y > 0:
                    out_queue.append(
                        cm.extrapolate_mapping(
                            grid_x*self.grid_cell_size,
                            (grid_y - 1)*self.grid_cell_size
                        )
                    )
                # Down neighbor
                if grid_y < self.ac_grid.get_grid_height()-1:
                    out_queue.append(
                        cm.extrapolate_mapping(
                            grid_x*self.grid_cell_size,
                            (grid_y + 1)*self.grid_cell_size
                        )
                    )

        # Replace queue with out_queue for the next iteration.
        self.queue = out_queue
        # If self.queue is empty, no further improvements can be made.
        return (len(self.queue) == 0)

    def queue_length(self):
        """
        Returns the current length of the queue (useful for diagnostics).
        """
        return len(self.queue)

    def optimize_position(self, cm):
        """
        Performs a small local search by shifting the transform's translation
        by ±1 in X and ±1 in Y, checking if the score improves.

        :param cm: An AffineTransform
        :return: (best_score, best_cm)
        """
        best_cm = cm
        best_score = self.scorer.calculate_similarity_score(cm)

        # Shift translate_x by -1
        test_cmx1 = AffineTransform(
            origin_x=cm.origin_x,
            origin_y=cm.origin_y,
            translate_x=cm.translate_x - 1.0,
            translate_y=cm.translate_y,
            a11=cm.a11, a12=cm.a12,
            a21=cm.a21, a22=cm.a22
        )
        score_x1 = self.scorer.calculate_similarity_score(test_cmx1)

        # Shift translate_x by +1
        test_cmx2 = AffineTransform(
            origin_x=cm.origin_x,
            origin_y=cm.origin_y,
            translate_x=cm.translate_x + 1.0,
            translate_y=cm.translate_y,
            a11=cm.a11, a12=cm.a12,
            a21=cm.a21, a22=cm.a22
        )
        score_x2 = self.scorer.calculate_similarity_score(test_cmx2)

        # Update best if needed
        if score_x1 < best_score and score_x1 < score_x2:
            best_cm = test_cmx1
            best_score = score_x1
        elif score_x2 < best_score:
            best_cm = test_cmx2
            best_score = score_x2

        # Now shift translate_y by ±1 on the "best so far".
        test_cmy1 = AffineTransform(
            origin_x=best_cm.origin_x,
            origin_y=best_cm.origin_y,
            translate_x=best_cm.translate_x,
            translate_y=best_cm.translate_y - 1.0,
            a11=best_cm.a11, a12=best_cm.a12,
            a21=best_cm.a21, a22=best_cm.a22
        )
        score_y1 = self.scorer.calculate_similarity_score(test_cmy1)

        test_cmy2 = AffineTransform(
            origin_x=best_cm.origin_x,
            origin_y=best_cm.origin_y,
            translate_x=best_cm.translate_x,
            translate_y=best_cm.translate_y + 1.0,
            a11=best_cm.a11, a12=best_cm.a12,
            a21=best_cm.a21, a22=best_cm.a22
        )
        score_y2 = self.scorer.calculate_similarity_score(test_cmy2)

        if score_y1 < best_score and score_y1 < score_y2:
            best_cm = test_cmy1
            best_score = score_y1
        elif score_y2 < best_score:
            best_cm = test_cmy2
            best_score = score_y2

        return (best_score, best_cm)

    def _is_valid(self, cm):
        """
        Checks if a given AffineTransform has valid translation coordinates
        (within [0, photo1.width] and [0, photo1.height]).
        """
        return (
            cm.translate_x >= 0.0 and
            cm.translate_y >= 0.0 and
            cm.translate_x <= self.photo1.shape[1] and
            cm.translate_y <= self.photo1.shape[0]
        )

    def get_photo_mapping(self):
        """
        Builds a DensePhotoMap from the best transforms in the ACGrid.
        Each grid cell transforms (x, y) -> (translate_x, translate_y).
        """
        grid_w = self.ac_grid.get_grid_width()
        grid_h = self.ac_grid.get_grid_height()

        # Construct a DensePhotoMap with the same grid dimensions.
        # The DensePhotoMap presumably expects references to photo1, photo2, etc.
        pm = DensePhotoMap(self.photo1, self.photo2, grid_w, grid_h)

        for y in range(grid_h):
            for x in range(grid_w):
                cell = self.ac_grid.get_grid_square(x, y)
                cm_opt = cell.get_affine_transform()
                if cm_opt is None:
                    continue
                # For each transform in that cell (in Rust, it's just 0 or 1),
                # we set the corresponding coordinate in pm.
                # The code does "pm.set_grid_coordinates(...)"
                # dividing origin_x by grid_cell_size, etc.
                origin_cell_x = cm_opt.origin_x // self.grid_cell_size
                origin_cell_y = cm_opt.origin_y // self.grid_cell_size
                pm.set_grid_coordinates(
                    origin_cell_x,
                    origin_cell_y,
                    cm_opt.translate_x,
                    cm_opt.translate_y
                )

        return pm
