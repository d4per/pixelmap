import argparse
import sys
import math
import numpy as np
from PIL import Image

from src.circlular_feature_descriptor_matcher import CircularFeatureDescriptorMatcher
from src.circular_feature_grid import CircularFeatureGrid
from src.correspondence_mapping_algorithm import CorrespondenceMappingAlgorithm


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def read_photo_np(filename):
    """
    Loads an image from disk as an RGBA NumPy array of shape (H, W, 4).
    """
    img = Image.open(filename).convert("RGBA")
    arr = np.array(img)  # shape: (H, W, 4), dtype=uint8
    return arr

def save_photo_np(np_array, filename):
    """
    Saves a NumPy array (H, W, 4) back to disk (e.g., as PNG).
    """
    # np_array should be shape (H, W, 4) with dtype=uint8
    img = Image.fromarray(np_array, "RGBA")
    img.save(filename)

def get_width(np_array):
    """
    Convenience: returns the width of the image (second dimension).
    """
    return np_array.shape[1]

def get_height(np_array):
    """
    Convenience: returns the height of the image (first dimension).
    """
    return np_array.shape[0]

def scale_photo_np(np_array, new_width):
    """
    Scales a NumPy image (H, W, 4) proportionally to new_width,
    returns a new (H2, W2, 4) array.

    Equivalent to 'get_scaled_proportional' in your Rust Photo code.
    """
    from PIL import Image

    h, w, c = np_array.shape
    if w == 0:
        return np_array

    # Compute new height to keep aspect ratio
    scale_factor = new_width / float(w)
    new_height = int(round(h * scale_factor))

    # Use Pillow to do the resizing
    img = Image.fromarray(np_array, "RGBA")
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    # Return as np array again
    return np.array(img_resized)

# -------------------------------------------------------------------
# The main CLI logic, adapted from your Rust code but using NumPy arrays
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI for pixelmap-based photo matching and interpolation (NumPy-based)."
    )

    parser.add_argument("photo1", help="First photo filename")
    parser.add_argument("photo2", help="Second photo filename")

    parser.add_argument("--output-dense-map", help="Optionally output DensePhotoMap as a text file")
    parser.add_argument("--processing-mode", default="low",
                        help="Processing mode: low, medium, or high. Default is 'low'.")
    parser.add_argument("--num-interpolations", type=int, default=10,
                        help="How many interpolations to produce. Default=10.")
    parser.add_argument("--base-filename", default="interpolation",
                        help="Base filename for interpolated outputs. Default='interpolation'.")

    return parser.parse_args()

def process_low(processor):
    processor.iterate(400, 5, 5, 2, 3.0)
    processor.iterate(400, 5, 5, 2, 2.0)
    processor.iterate(400, 5, 5, 2, 1.0)
    processor.iterate(400, 5, 5, 2, 1.0)
    return processor

def process_medium(processor):
    processor.iterate(400, 5, 5, 2, 3.0)
    processor.iterate(400, 5, 5, 2, 2.0)
    processor.iterate(400, 5, 5, 2, 1.0)
    processor.iterate(400, 5, 5, 2, 0.5)
    processor.iterate(400, 5, 5, 2, 1.0)
    processor.iterate(400, 5, 5, 2, 2.0)
    processor.iterate(400, 5, 5, 2, 2.0)
    processor.iterate(800, 5, 5, 2, 3.0)
    processor.iterate(800, 5, 5, 2, 1.0)
    processor.iterate(800, 5, 5, 2, 2.0)
    return processor

def process_high(processor):
    processor.iterate(400, 5, 5, 2, 3.0)
    processor.iterate(400, 5, 5, 2, 2.0)
    processor.iterate(400, 5, 5, 2, 1.0)
    processor.iterate(400, 5, 5, 2, 0.5)
    processor.iterate(400, 5, 5, 2, 1.0)
    processor.iterate(400, 5, 5, 2, 2.0)
    processor.iterate(400, 5, 5, 2, 2.0)
    processor.iterate(800, 5, 5, 2, 3.0)
    processor.iterate(800, 5, 5, 2, 1.0)
    processor.iterate(800, 5, 5, 2, 2.0)
    processor.iterate(1600, 5, 5, 2, 1.0)
    processor.iterate(1600, 5, 5, 2, 2.0)
    processor.iterate(1600, 5, 5, 2, 2.0)
    return processor


def to_dense_map(dense_photo_map):
    """
    Equivalent to the Rust function that iterates over each pixel (x, y),
    calls map_photo_pixel, etc.
    We'll assume dense_photo_map.photo1 is a NumPy array and that
    map_photo_pixel can still handle float x, y.
    """
    lines = []
    h = get_height(dense_photo_map.photo1)
    w = get_width(dense_photo_map.photo1)
    # The Rust snippet has y in [0..width) and x in [0..height).
    # We'll replicate that literally:
    for y in range(w):
        for x in range(h):
            x2, y2 = dense_photo_map.map_photo_pixel(float(x), float(y))
            if math.isnan(x2):
                continue
            lines.append(f"{x} {y} {x2:.3f} {y2:.3f}")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------
# Example PixelMapProcessor with NumPy arrays
# -------------------------------------------------------------------

class PixelMapProcessor:
    """
    A pythonic version of your Rust PixelMapProcessor, but photo1/photo2
    are NumPy arrays rather than Photo structs.
    """
    def __init__(self, photo1_np, photo2_np, photo_width):
        """
        :param photo1_np, photo2_np: NumPy arrays for the two images.
        :param photo_width: the initial scaling width for feature extraction, etc.
        """
        self.photo1 = photo1_np
        self.photo2 = photo2_np
        self.initial_photo_width = photo_width

        # Create dummy images to init the managers, as in Rust
        dummy = np.zeros((1,1,4), dtype=np.uint8)  # a 1x1 RGBA black image

        self.ocm_manager1 = CorrespondenceMappingAlgorithm(
            photo_width, dummy, dummy, 5, 5
        )
        self.ocm_manager2 = CorrespondenceMappingAlgorithm(
            photo_width, dummy, dummy, 5, 5
        )
        self.total_comparisons = 0

        # Re-init with actual photos
        self.ocm_manager1 = CorrespondenceMappingAlgorithm(
            photo_width, self.photo1, self.photo2, 5, 5
        )
        self.ocm_manager2 = CorrespondenceMappingAlgorithm(
            photo_width, self.photo2, self.photo1, 5, 5
        )

    def init(self):
        """
        The same as Rust's init() method: scale, build feature grids, match, etc.
        We'll show the same logic, but you must adapt your
        CircularFeatureGrid + matcher to NumPy.
        """
        w1 = get_width(self.photo1)
        w = min(self.initial_photo_width, w1)

        photo1scaled = scale_photo_np(self.photo1, w)
        photo2scaled = scale_photo_np(self.photo2, w)

        # Build circular feature grids
        image1 = CircularFeatureGrid(photo1scaled, get_width(photo1scaled), get_height(photo1scaled), 10)
        image2 = CircularFeatureGrid(photo2scaled, get_width(photo2scaled), get_height(photo2scaled), 10)

        # Match features
        matcher = CircularFeatureDescriptorMatcher()
        pairs = matcher.match_areas(image1, image2)

        # Create new managers
        ocm_manager1 = CorrespondenceMappingAlgorithm(
            get_width(photo1scaled), self.photo1, self.photo2, 5, 5
        )
        ocm_manager2 = CorrespondenceMappingAlgorithm(
            get_width(photo1scaled), self.photo2, self.photo1, 5, 5
        )

        # Add matched points
        for (p1, p2) in pairs:
            s = 1.0
            v = p1.total_angle - p2.total_angle
            vv = -v

            ocm_manager1.add_init_point(
                float(p1.center_x)*s, float(p1.center_y)*s,
                float(p2.center_x)*s, float(p2.center_y)*s,
                vv
            )
            ocm_manager2.add_init_point(
                float(p2.center_x)*s, float(p2.center_y)*s,
                float(p1.center_x)*s, float(p1.center_y)*s,
                -vv
            )

        print("init points added")

        # run both managers
        ocm_manager1.run_until_done()
        print("manager1 done")
        ocm_manager2.run_until_done()
        print("manager2 done")

        self.total_comparisons = ocm_manager1.get_total_comparisons() + ocm_manager2.get_total_comparisons()

        self.ocm_manager1 = ocm_manager1
        self.ocm_manager2 = ocm_manager2

    def get_result(self, clean_max_dist):
        pm1 = self.ocm_manager1.get_photo_mapping()
        pm2 = self.ocm_manager2.get_photo_mapping()

        pm1.remove_outliers(pm2, clean_max_dist)
        pm2.remove_outliers(pm1, clean_max_dist)

        print(f"tot comparisons : {self.total_comparisons}")

        return (pm1, pm2)

    def iterate(self, photo_width, grid_cell_size, neighborhood_radius, smooth_iterations, clean_max_dist):
        print("debug1")
        pm1 = self.ocm_manager1.get_photo_mapping()
        pm2 = self.ocm_manager2.get_photo_mapping()

        pm1.remove_outliers(pm2, clean_max_dist)
        pm2.remove_outliers(pm1, clean_max_dist)

        pm1_smooth = pm1.smooth_grid_points_n_times(smooth_iterations)
        pm2_smooth = pm2.smooth_grid_points_n_times(smooth_iterations)

        # create new managers
        ocm_manager1 = CorrespondenceMappingAlgorithm(
            photo_width, self.photo1, self.photo2, grid_cell_size, neighborhood_radius
        )
        ocm_manager2 = CorrespondenceMappingAlgorithm(
            photo_width, self.photo2, self.photo1, grid_cell_size, neighborhood_radius
        )

        ocm_manager1.init_from_photomapping(pm1_smooth)
        ocm_manager2.init_from_photomapping(pm2_smooth)

        ocm_manager1.run_until_done()
        ocm_manager2.run_until_done()

        self.total_comparisons += ocm_manager1.get_total_comparisons() + ocm_manager2.get_total_comparisons()

        self.ocm_manager1 = ocm_manager1
        self.ocm_manager2 = ocm_manager2
        print("debug2")

    def get_total_comparisons(self):
        return self.total_comparisons


# -------------------------------------------------------------------
# Example main() using NumPy-based PixelMapProcessor
# -------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        photo1_np = read_photo_np(args.photo1)  # shape (H,W,4)
        photo2_np = read_photo_np(args.photo2)  # shape (H,W,4)
    except Exception as e:
        print(f"Error reading images: {e}")
        sys.exit(1)

    processing_mode = args.processing_mode.lower()
    if processing_mode not in ["low", "medium", "high"]:
        print(f"Invalid processing mode: {processing_mode}")
        sys.exit(1)

    # Choose initial photo_width based on mode
    if processing_mode == "low":
        processor = PixelMapProcessor(photo1_np, photo2_np, 400)
    elif processing_mode == "medium":
        processor = PixelMapProcessor(photo1_np, photo2_np, 800)
    else:
        processor = PixelMapProcessor(photo1_np, photo2_np, 1600)

    processor.init()

    if processing_mode == "low":
        process_low(processor)
    elif processing_mode == "medium":
        process_medium(processor)
    elif processing_mode == "high":
        process_high(processor)

    # final result
    (map1, map2) = processor.get_result(clean_max_dist=2.0)
    final_map = map1  # pick one for interpolation or whatever

    # Optionally save the dense map text
    if args.output_dense_map:
        data_text = to_dense_map(final_map)
        with open(args.output_dense_map, "w", encoding="utf-8") as f:
            f.write(data_text)
        print(f"DensePhotoMap written to {args.output_dense_map}")

    # Produce interpolations
    if args.num_interpolations > 0:
        base_fname = args.base_filename
        for i in range(1, args.num_interpolations+1):
            alpha = float(i)/(args.num_interpolations+1)
            detail_level = 4
            # final_map.interpolate_photo(...) should produce a new np array or similar
            interp_np = final_map.interpolate_photo(alpha, detail_level)
            out_filename = f"{base_fname}_{i}.png"
            save_photo_np(interp_np, out_filename)

    print("Done.")

if __name__ == "__main__":
    main()
