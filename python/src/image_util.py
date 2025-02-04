from PIL import Image
import numpy as np

def get_scaled_proportional(np_img: np.ndarray, new_width: int) -> np.ndarray:
    """
    Scales a NumPy image (H, W, C) so that its width becomes `new_width`,
    preserving the aspect ratio. Uses PIL/Pillow for resizing.
    """
    if new_width <= 0:
        raise ValueError("new_width must be > 0")

    original_height, original_width = np_img.shape[0], np_img.shape[1]
    if original_width == 0:
        return np_img

    # Compute new dimensions
    scale_factor = new_width / float(original_width)
    new_height = int(round(original_height * scale_factor))

    # Convert np array -> PIL Image
    mode = "RGBA" if np_img.shape[2] == 4 else "RGB"  # or handle other channel counts
    pil_img = Image.fromarray(np_img, mode=mode)

    # Use LANCZOS filter for high-quality down/up sampling
    pil_resized = pil_img.resize((new_width, new_height), Image.LANCZOS)

    # Convert back to NumPy
    resized_np = np.array(pil_resized)
    return resized_np
