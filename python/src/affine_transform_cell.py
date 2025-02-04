from src.affine_transform import AffineTransform


class AffineTransformCell:
    """
    Wraps an optional AffineTransform plus a floating-point 'best score.'
    In Rust, interior mutability was needed, but in Python we can store it directly.
    """
    def __init__(self):
        # No transform initially
        self.cm = None          # type: AffineTransform or None
        self.best_score = float('inf')

    def set(self, transform: AffineTransform, new_score: float):
        self.cm = transform
        self.best_score = new_score

    def get_score(self) -> float:
        return self.best_score

    def get_affine_transform(self):
        return self.cm