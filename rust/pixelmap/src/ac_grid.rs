use crate::affine_transform_cell::AffineTransformCell;

/// A 2D grid of [`AffineTransformCell`] elements, indexed by (x, y).
/// Each cell can hold:
/// - An optional [`AffineTransform`]
/// - A "best score" (`f32`) associated with that transform
///
/// This struct allows easy creation, access, and storage of `AffineTransformCell`s in a
/// row-major layout (1D `Vec`, logically treated as 2D).
#[derive(Debug, Clone)]
pub struct ACGrid {
    /// The underlying storage for all grid cells. It is laid out row by row in a single vector.
    grid: Vec<AffineTransformCell>,

    /// The number of cells in each row of the grid.
    grid_width: usize,

    /// The number of rows in the grid (the vertical dimension).
    grid_height: usize,
}

impl ACGrid {
    /// Creates a new `ACGrid` with the given width and height,
    /// initializing each cell with an empty [`AffineTransformCell`].
    ///
    /// # Parameters
    /// - `grid_width`: The width (number of columns) of the grid.
    /// - `grid_height`: The height (number of rows) of the grid.
    ///
    /// # Returns
    /// A fully allocated `ACGrid` where each cell contains
    /// a default/empty `AffineTransformCell`.
    ///
    /// # Examples
    /// ```
    /// use pixelmap::ac_grid::ACGrid;
    /// let my_grid = ACGrid::new(10, 5);
    /// assert_eq!(my_grid.get_grid_width(), 10);
    /// assert_eq!(my_grid.get_grid_height(), 5);
    /// ```
    pub fn new(grid_width: usize, grid_height: usize) -> ACGrid {
        let default_grid_square = AffineTransformCell::new_empty();
        Self {
            grid: vec![default_grid_square; grid_width * grid_height],
            grid_width,
            grid_height,
        }
    }

    /// Returns a reference to the [`AffineTransformCell`] at the specified (x, y) coordinate.
    ///
    /// # Parameters
    /// - `x`: The x-coordinate (column index).
    /// - `y`: The y-coordinate (row index).
    ///
    /// # Panics
    /// Panics if `(x, y)` is out of bounds (i.e., if `x >= grid_width` or `y >= grid_height`).
    ///
    /// # Examples
    /// ```
    /// # use pixelmap::ac_grid::ACGrid;
    /// # use pixelmap::ac_grid::AffineTransformCell;
    /// let my_grid = ACGrid::new(10, 5);
    /// let cell_ref = my_grid.get_grid_square(2, 3);
    /// // `cell_ref` is an &AffineTransformCell; you can call its methods, e.g.:
    /// let score = cell_ref.get_score();
    /// ```
    pub fn get_grid_square(&self, x: usize, y: usize) -> &AffineTransformCell {
        &self.grid[x + y * self.grid_width]
    }

    /// Returns the width of the grid (the number of columns).
    ///
    /// # Examples
    /// ```
    /// # use pixelmap::ac_grid::ACGrid;
    /// let my_grid = ACGrid::new(10, 5);
    /// assert_eq!(my_grid.get_grid_width(), 10);
    /// ```
    pub fn get_grid_width(&self) -> usize {
        self.grid_width
    }

    /// Returns the height of the grid (the number of rows).
    ///
    /// # Examples
    /// ```
    /// # use pixelmap::ac_grid::ACGrid;
    /// let my_grid = ACGrid::new(10, 5);
    /// assert_eq!(my_grid.get_grid_height(), 5);
    /// ```
    pub fn get_grid_height(&self) -> usize {
        self.grid_height
    }
}
