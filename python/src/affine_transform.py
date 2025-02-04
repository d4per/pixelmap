import math
import numpy as np

class AffineTransform:
    """
    Represents a 2D affine transformation:
      X = (x - origin_x)*a11 + (y - origin_y)*a12 + translate_x
      Y = (x - origin_x)*a21 + (y - origin_y)*a22 + translate_y

    Attributes:
        origin_x  (int):   integer reference x-coordinate (origin in input).
        origin_y  (int):   integer reference y-coordinate (origin in input).
        translate_x (float): translation offset (x-direction).
        translate_y (float): translation offset (y-direction).
        a11, a12, a21, a22 (float): 2×2 matrix entries for rotation/scale/shear.
    """
    def __init__(self, origin_x, origin_y, translate_x, translate_y, a11, a12, a21, a22):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.a11 = a11
        self.a12 = a12
        self.a21 = a21
        self.a22 = a22

    def is_scale_valid(self, scale_bound: float) -> bool:
        """
        Checks if the scale factors (length of each row in the 2x2 matrix)
        lie within [1/scale_bound, scale_bound].
        """
        if scale_bound <= 0.0:
            return False

        min_scale = 1.0 / scale_bound
        # row 1 scale
        scale_x = math.sqrt(self.a11**2 + self.a12**2)
        # row 2 scale
        scale_y = math.sqrt(self.a21**2 + self.a22**2)

        return (scale_x > min_scale and scale_x < scale_bound and
                scale_y > min_scale and scale_y < scale_bound)

    def extrapolate_point(self, x: int, y: int):
        """
        Applies the affine transform to an integer coordinate (x, y),
        returning a float coordinate (X, Y).
        """
        dx = float(x - self.origin_x)
        dy = float(y - self.origin_y)
        X = dx*self.a11 + dy*self.a12 + self.translate_x
        Y = dx*self.a21 + dy*self.a22 + self.translate_y
        return (X, Y)

    def extrapolate_mapping(self, x: int, y: int):
        """
        Produces a *new* AffineTransform with a new origin (x, y).
        The translation is adjusted so that (x, y) in the old transform
        is the new transform's origin.
        """
        dx = float(x - self.origin_x)
        dy = float(y - self.origin_y)
        new_translate_x = dx*self.a11 + dy*self.a12 + self.translate_x
        new_translate_y = dx*self.a21 + dy*self.a22 + self.translate_y

        return AffineTransform(
            origin_x = x,
            origin_y = y,
            translate_x = new_translate_x,
            translate_y = new_translate_y,
            a11 = self.a11,
            a12 = self.a12,
            a21 = self.a21,
            a22 = self.a22
        )
