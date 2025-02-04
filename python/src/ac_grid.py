from src.affine_transform_cell import AffineTransformCell


class ACGrid:
    """
    A 2D grid of AffineTransformCell. The grid is stored in row-major order:
      index = x + y * grid_width
    """
    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = [AffineTransformCell() for _ in range(grid_width * grid_height)]

    def get_grid_square(self, x: int, y: int) -> AffineTransformCell:
        # Panics or errors if out of bounds in the original code
        index = x + y*self.grid_width
        return self.grid[index]

    def get_grid_width(self) -> int:
        return self.grid_width

    def get_grid_height(self) -> int:
        return self.grid_height
