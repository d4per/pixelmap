# pixelmap_model_3d

3D reconstruction from a dense correspondence map.

Two views of the same static scene constrain the shape of that scene. Given a finished
mapping from [`pixelmap`](../pixelmap), `Model3D::new` lifts the correspondence field
into a grid of 3D points — each carrying the texture coordinates it came from — and
`Model3D::to_x3d` writes that grid out as a textured X3D mesh.

The photos are treated as two perspective cameras. A fundamental matrix is fitted to the
correspondences with RANSAC. With an assumed focal length it becomes an essential
matrix, which gives the relative camera pose. Every consistent correspondence is then
triangulated. The depth therefore comes out the right way round for convex and concave
scenes alike, in its true proportions. The mesh opens in a viewer at the first camera's
viewpoint.

Some pairs don't determine the geometry well: a flat scene, a camera that only rotated,
or too little parallax. With `Projection::Auto` (the default), those fall back to an
affine camera model. `Model3D::projection` and `Model3D::two_view_geometry` report
which model was used and why.

This lives outside the `pixelmap` crate on purpose: it is a consumer of the algorithm
rather than part of it, and it is the only thing here that needs `nalgebra`.
It is not published to crates.io.

## Example

    cargo run --release -p pixelmap_model_3d --example x3d -- photo1.jpg photo2.jpg [f1 f2]

The two photos must have the same dimensions. `f1` and `f2` are the 35 mm equivalent
focal lengths of the two shots, as recorded in their EXIF data. Without them, a typical
phone lens (about 28 mm) is assumed. A wrong focal length mostly stretches or squashes
the model along the viewing direction.

The example writes `model.x3d` and the `model_texture.png` it references; open the
`.x3d` in any X3D viewer.
