# pixelmap_model_3d

3D reconstruction from a dense correspondence map.

Two views of the same static scene constrain the shape of that scene. Given a finished
mapping from [`pixelmap`](../pixelmap), `Model3D::new` lifts the correspondence field
into a grid of 3D points — each carrying the texture coordinates it came from — and
`Model3D::to_x3d` writes that grid out as a textured X3D mesh.

This lives outside the `pixelmap` crate on purpose: it is a consumer of the algorithm
rather than part of it, and it is the only thing here that needs `nalgebra` for its SVD.
It is not published to crates.io.

## Example

    cargo run --release -p pixelmap_model_3d --example x3d -- photo1.jpg photo2.jpg

The two photos must have the same dimensions. Writes `model.x3d` and the
`model_texture.png` it references; open the `.x3d` in any X3D viewer.
