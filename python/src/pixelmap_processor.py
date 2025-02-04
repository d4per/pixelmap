import math

from src.circlular_feature_descriptor_matcher import CircularFeatureDescriptorMatcher
from src.circular_feature_grid import CircularFeatureGrid
from src.correspondence_mapping_algorithm import CorrespondenceMappingAlgorithm


class PixelMapProcessor:
    """
    Manages a pipeline for finding and refining a mapping between two images (photo1, photo2).

    Steps:
      1. Circular feature extraction + matching on scaled versions.
      2. Initialize a CorrespondenceMappingAlgorithm with matched points (both directions).
      3. Iterative refinement: remove outliers, smooth, re-initialize.
      4. Produce final DensePhotoMaps for forward/backward transformations.
    """

    def __init__(self, photo1, photo2, photo_width):
        """
        Constructs a new PixelMapProcessor with the given photo1, photo2, and an initial width.

        :param photo1: The first Photo object.
        :param photo2: The second Photo object.
        :param photo_width: The width to which images may be scaled before feature extraction.
        """
        # We'll initialize the internal structure similarly to Rust:
        self.photo1 = photo1
        self.photo2 = photo2
        self.initial_photo_width = photo_width

        self.ocm_manager1 = CorrespondenceMappingAlgorithm(
            photo_width,
            None,  # references to dummy
            None,
            5,
            5
        )
        self.ocm_manager2 = CorrespondenceMappingAlgorithm(
            photo_width,
            None,
            None,
            5,
            5
        )

        self.total_comparisons = 0

        # Re-initialize with the actual photos.
        self.ocm_manager1 = CorrespondenceMappingAlgorithm(
            photo_width,
            self.photo1,
            self.photo2,
            5,
            5
        )
        self.ocm_manager2 = CorrespondenceMappingAlgorithm(
            photo_width,
            self.photo2,
            self.photo1,
            5,
            5
        )

    def init(self):
        """
        Performs the initial matching step:

        1. Scales both photo1 and photo2 to self.initial_photo_width.
        2. Uses CircularFeatureGrid to extract descriptors.
        3. Matches descriptors with CircularFeatureDescriptorMatcher.
        4. Initializes new CorrespondenceMappingAlgorithm with matched points.
        5. Runs both managers until done.

        Updates self.total_comparisons accordingly.
        """
        # 1. Scale down images if needed
        width = min(self.initial_photo_width, self.photo1.width)
        photo1scaled = self.photo1.get_scaled_proportional(width)
        photo2scaled = self.photo2.get_scaled_proportional(width)

        # 2. Create circular feature grids
        image1 = CircularFeatureGrid(
            photo1scaled.img_data,
            photo1scaled.width,
            photo1scaled.height,
            10
        )
        image2 = CircularFeatureGrid(
            photo2scaled.img_data,
            photo2scaled.width,
            photo2scaled.height,
            10
        )

        # 3. Match features
        circle_area_info_matcher = CircularFeatureDescriptorMatcher()
        pairs = circle_area_info_matcher.match_areas(image1, image2)

        # 4. Create new managers for the scaled images
        ocm_manager1 = CorrespondenceMappingAlgorithm(
            photo1scaled.width,
            self.photo1,
            self.photo2,
            5,
            5
        )
        ocm_manager2 = CorrespondenceMappingAlgorithm(
            photo1scaled.width,
            self.photo2,
            self.photo1,
            5,
            5
        )

        # 5. Add the matched points
        for (p1, p2) in pairs:
            s = 1.0
            # The Rust code does "v = p1.total_angle - p2.total_angle"
            v = p1.total_angle - p2.total_angle
            vv = -v

            # Forward mapping (photo1->photo2)
            ocm_manager1.add_init_point(
                p1.center_x*s, p1.center_y*s,
                p2.center_x*s, p2.center_y*s,
                vv
            )
            # Reverse mapping (photo2->photo1)
            ocm_manager2.add_init_point(
                p2.center_x*s, p2.center_y*s,
                p1.center_x*s, p1.center_y*s,
                -vv
            )

        print("init points added")

        # Run both managers until done
        ocm_manager1.run_until_done()
        ocm_manager2.run_until_done()

        # Update total comparisons
        self.total_comparisons = (ocm_manager1.get_total_comparisons() +
                                  self.ocm_manager2.get_total_comparisons())

        # Store them
        self.ocm_manager1 = ocm_manager1
        self.ocm_manager2 = ocm_manager2

    def get_total_comparisons(self):
        """
        Returns how many comparisons have been made so far by the two managers.
        """
        return self.total_comparisons

    def get_ac_grids(self):
        """
        Retrieves a pair of ACGrid from ocm_manager1 and ocm_manager2.

        :return: (ACGrid, ACGrid)
        """
        grid1 = self.ocm_manager1.get_ac_grid()
        grid2 = self.ocm_manager2.get_ac_grid()
        return (grid1, grid2)

    def iterate(self, photo_width, grid_cell_size, neighborhood_radius, smooth_iterations, clean_max_dist):
        """
        Performs one iteration of mapping refinement:

          1. Get the current dense maps from each manager.
          2. Remove outliers by forward-backward consistency check.
          3. Smooth them.
          4. Re-init the managers with the smoothed maps.
          5. Run until done, accumulate total comparisons.

        :param photo_width: used to re-scale the images for the new managers
        :param grid_cell_size: how large each grid cell is
        :param neighborhood_radius: how far each manager looks for matches
        :param smooth_iterations: how many times to call smooth
        :param clean_max_dist: threshold for outlier removal
        """
        print(photo_width)

        # 1. Get the current DensePhotoMaps
        pm1 = self.ocm_manager1.get_photo_mapping()
        pm2 = self.ocm_manager2.get_photo_mapping()

        # 2. Remove outliers in forward/backward consistency
        pm1.remove_outliers(pm2, clean_max_dist)
        pm2.remove_outliers(pm1, clean_max_dist)

        # 3. Smooth
        pm1_smooth = pm1.smooth_grid_points_n_times(smooth_iterations)
        pm2_smooth = pm2.smooth_grid_points_n_times(smooth_iterations)

        # 4. Re-init managers with the smoothed maps
        ocm_manager1 = CorrespondenceMappingAlgorithm(
            photo_width,
            self.photo1,
            self.photo2,
            grid_cell_size,
            neighborhood_radius
        )
        ocm_manager2 = CorrespondenceMappingAlgorithm(
            photo_width,
            self.photo2,
            self.photo1,
            grid_cell_size,
            neighborhood_radius
        )

        ocm_manager1.init_from_photomapping(pm1_smooth)
        ocm_manager2.init_from_photomapping(pm2_smooth)

        # 5. Run until done, accumulate comparisons
        ocm_manager1.run_until_done()
        ocm_manager2.run_until_done()

        self.total_comparisons += (ocm_manager1.get_total_comparisons() +
                                   ocm_manager2.get_total_comparisons())

        # Store them
        self.ocm_manager1 = ocm_manager1
        self.ocm_manager2 = ocm_manager2

    def get_result(self, clean_max_dist):
        """
        Retrieves final forward/backward mappings, removing any remaining outliers with
        a given distance threshold.

        :param clean_max_dist: The max distance for outlier removal.
        :return: (DensePhotoMap, DensePhotoMap)
                 - First is from photo1->photo2
                 - Second is from photo2->photo1
        Prints total comparisons as well.
        """
        pm1 = self.ocm_manager1.get_photo_mapping()
        pm2 = self.ocm_manager2.get_photo_mapping()

        pm1.remove_outliers(pm2, clean_max_dist)
        pm2.remove_outliers(pm1, clean_max_dist)

        print(f"tot comparisons : {self.total_comparisons}")

        return (pm1, pm2)

    def get_matched_area(self):
        """
        Computes how much of the mapping is valid (non-outlier) in ocm_manager1
        by removing outliers with threshold=2.0, then computing the fraction
        of valid cells.

        :return: fraction in [0.0..1.0]
        """
        pm1 = self.ocm_manager1.get_photo_mapping()
        pm2 = self.ocm_manager2.get_photo_mapping()

        # clone to avoid mutating the originals
        pm1_clone = pm1.clone()
        pm2_clone = pm2.clone()

        pm1_clone.remove_outliers(pm2_clone, 2.0)
        pm2_clone.remove_outliers(pm1_clone, 2.0)

        return pm1_clone.calculate_used_area()
