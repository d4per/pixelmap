//! Writing results out for inspection in other tools.
//!
//! These write to any [`std::io::Write`], so the crate itself still never touches a
//! filesystem.

use std::io::{self, Write};

use nalgebra::Point3;
use pixelmap::Photo;

use crate::calib::Intrinsics;
use crate::mesh::Mesh;
use crate::pose::Pose;
use crate::texture::Texture;
use crate::types::{PhotoPx, World};

/// The colour of the pixel nearest `p`, or magenta outside the photo.
pub fn sample_colour(photo: &Photo, p: PhotoPx) -> [u8; 3] {
    let (x, y) = (p.x().round(), p.y().round());
    if x < 0.0 || y < 0.0 {
        return [255, 0, 255];
    }
    photo
        .pixel(x as usize, y as usize)
        .map_or([255, 0, 255], |[r, g, b, _]| [r, g, b])
}

/// Writes coloured points, plus a small pyramid per camera pointing along its view, as
/// ASCII PLY. Opens in MeshLab, Blender and CloudCompare.
///
/// The reconstruction's frame is the seed camera's, with y pointing down. The file is
/// turned half a revolution about x so that viewers show the scene upright. Each pyramid
/// is `frustum_depth` deep, in reconstruction units (the seed pair's baseline is 1).
pub fn write_ply(
    out: &mut impl Write,
    points: &[(World, [u8; 3])],
    cameras: &[Pose],
    intrinsics: &Intrinsics,
    (width, height): (usize, usize),
    frustum_depth: f64,
) -> io::Result<()> {
    const CAMERA_COLOUR: [u8; 3] = [230, 40, 40];
    let vertex = |out: &mut dyn Write, p: &Point3<f64>, [r, g, b]: [u8; 3]| {
        writeln!(out, "{:.6} {:.6} {:.6} {r} {g} {b}", p.x, -p.y, -p.z)
    };

    writeln!(out, "ply")?;
    writeln!(out, "format ascii 1.0")?;
    writeln!(out, "comment pixelmap_multiview: points and cameras, y up")?;
    writeln!(out, "element vertex {}", points.len() + cameras.len() * 5)?;
    for axis in ["x", "y", "z"] {
        writeln!(out, "property float {axis}")?;
    }
    for channel in ["red", "green", "blue"] {
        writeln!(out, "property uchar {channel}")?;
    }
    writeln!(out, "element face {}", cameras.len() * 4)?;
    writeln!(out, "property list uchar int vertex_indices")?;
    writeln!(out, "end_header")?;

    for (point, colour) in points {
        vertex(out, &point.0, *colour)?;
    }

    let corners = [
        (0.0, 0.0),
        (width as f32, 0.0),
        (width as f32, height as f32),
        (0.0, height as f32),
    ];
    for pose in cameras {
        let to_world = pose.inverse();
        vertex(out, &pose.centre(), CAMERA_COLOUR)?;
        for (x, y) in corners {
            let n = intrinsics.normalize(PhotoPx::new(x, y));
            let corner = Point3::new(n.x() * frustum_depth, n.y() * frustum_depth, frustum_depth);
            vertex(out, &to_world.to_camera(&corner), CAMERA_COLOUR)?;
        }
    }

    for camera in 0..cameras.len() {
        let apex = points.len() + camera * 5;
        for side in 0..4 {
            writeln!(
                out,
                "3 {apex} {} {}",
                apex + 1 + side,
                apex + 1 + (side + 1) % 4
            )?;
        }
    }
    Ok(())
}

/// Writes a mesh as Wavefront OBJ, with normals when it has them.
///
/// Turned half a revolution about x, like [`write_ply`], so that viewers show it upright.
pub fn write_obj(out: &mut impl Write, mesh: &Mesh) -> io::Result<()> {
    writeln!(out, "# pixelmap_multiview mesh, y up")?;
    for p in &mesh.positions {
        writeln!(out, "v {:.6} {:.6} {:.6}", p.x, -p.y, -p.z)?;
    }
    let with_normals = mesh.normals.len() == mesh.positions.len();
    if with_normals {
        for n in &mesh.normals {
            writeln!(out, "vn {:.5} {:.5} {:.5}", n.x, -n.y, -n.z)?;
        }
    }
    for t in &mesh.triangles {
        let [a, b, c] = t.map(|i| i + 1);
        if with_normals {
            writeln!(out, "f {a}//{a} {b}//{b} {c}//{c}")?;
        } else {
            writeln!(out, "f {a} {b} {c}")?;
        }
    }
    Ok(())
}

/// Writes a textured mesh as Wavefront OBJ. `material_library` is the file name
/// [`write_mtl`]'s output is saved under, next to the OBJ.
pub fn write_textured_obj(
    out: &mut impl Write,
    mesh: &Mesh,
    texture: &Texture,
    material_library: &str,
) -> io::Result<()> {
    writeln!(out, "# pixelmap_multiview textured mesh, y up")?;
    writeln!(out, "mtllib {material_library}")?;
    for p in &mesh.positions {
        writeln!(out, "v {:.6} {:.6} {:.6}", p.x, -p.y, -p.z)?;
    }
    for [u, v] in &texture.texcoords {
        writeln!(out, "vt {u:.6} {v:.6}")?;
    }
    let with_normals = mesh.normals.len() == mesh.positions.len();
    if with_normals {
        for n in &mesh.normals {
            writeln!(out, "vn {:.5} {:.5} {:.5}", n.x, -n.y, -n.z)?;
        }
    }
    writeln!(out, "usemtl surface")?;
    for (t, uv) in mesh.triangles.iter().zip(&texture.face_texcoords) {
        let [a, b, c] = t.map(|i| i + 1);
        let [ta, tb, tc] = uv.map(|i| i + 1);
        if with_normals {
            writeln!(out, "f {a}/{ta}/{a} {b}/{tb}/{b} {c}/{tc}/{c}")?;
        } else {
            writeln!(out, "f {a}/{ta} {b}/{tb} {c}/{tc}")?;
        }
    }
    Ok(())
}

/// Writes the material library a textured OBJ refers to. `texture_file` is the file name
/// the atlas is saved under, next to it.
pub fn write_mtl(out: &mut impl Write, texture_file: &str) -> io::Result<()> {
    writeln!(out, "newmtl surface")?;
    writeln!(out, "Ka 1.0 1.0 1.0")?;
    writeln!(out, "Kd 1.0 1.0 1.0")?;
    writeln!(out, "Ks 0.0 0.0 0.0")?;
    writeln!(out, "d 1.0")?;
    writeln!(out, "illum 1")?;
    writeln!(out, "map_Kd {texture_file}")
}

/// Writes a mesh with one colour per vertex as ASCII PLY, for viewers that do not load
/// textures.
pub fn write_ply_mesh(out: &mut impl Write, mesh: &Mesh, colours: &[[u8; 3]]) -> io::Result<()> {
    writeln!(out, "ply")?;
    writeln!(out, "format ascii 1.0")?;
    writeln!(
        out,
        "comment pixelmap_multiview mesh with vertex colours, y up"
    )?;
    writeln!(out, "element vertex {}", mesh.positions.len())?;
    for axis in ["x", "y", "z"] {
        writeln!(out, "property float {axis}")?;
    }
    for channel in ["red", "green", "blue"] {
        writeln!(out, "property uchar {channel}")?;
    }
    writeln!(out, "element face {}", mesh.triangles.len())?;
    writeln!(out, "property list uchar int vertex_indices")?;
    writeln!(out, "end_header")?;
    for (i, p) in mesh.positions.iter().enumerate() {
        let [r, g, b] = colours.get(i).copied().unwrap_or([200, 200, 200]);
        writeln!(out, "{:.6} {:.6} {:.6} {r} {g} {b}", p.x, -p.y, -p.z)?;
    }
    for [a, b, c] in &mesh.triangles {
        writeln!(out, "3 {a} {b} {c}")?;
    }
    Ok(())
}

/// Writes a textured mesh as an X3D scene in the XML encoding. `texture_file` is the file
/// name the atlas is saved under, next to it.
///
/// One `IndexedFaceSet` with texture coordinates per triangle corner, since a vertex on a
/// chart border has a different place in the atlas for each chart. Turned half a
/// revolution about x, like the other exporters, so that viewers show it upright.
pub fn write_textured_x3d(
    out: &mut impl Write,
    mesh: &Mesh,
    texture: &Texture,
    texture_file: &str,
) -> io::Result<()> {
    writeln!(out, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(
        out,
        r#"<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" "http://www.web3d.org/specifications/x3d-3.3.dtd">"#
    )?;
    writeln!(
        out,
        r#"<X3D profile="Interchange" version="3.3" xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.3.xsd">"#
    )?;
    writeln!(out, "  <head>")?;
    writeln!(
        out,
        r#"    <meta name="generator" content="pixelmap_multiview"/>"#
    )?;
    writeln!(out, "  </head>")?;
    writeln!(out, "  <Scene>")?;
    writeln!(out, "    <Shape>")?;
    writeln!(out, "      <Appearance>")?;
    writeln!(out, r#"        <Material diffuseColor="1 1 1"/>"#)?;
    // An MFString inside an XML attribute: escape for the string, then for XML.
    let url = texture_file.replace('\\', "\\\\").replace('"', "\\\"");
    writeln!(
        out,
        r#"        <ImageTexture url='"{}"'/>"#,
        xml_escape(&url)
    )?;
    writeln!(out, "      </Appearance>")?;

    let with_normals = mesh.normals.len() == mesh.positions.len();
    writeln!(
        out,
        r#"      <IndexedFaceSet solid="false" ccw="true" normalPerVertex="true""#
    )?;
    write!(out, r#"        coordIndex=""#)?;
    for (i, [a, b, c]) in mesh.triangles.iter().enumerate() {
        let separator = if i == 0 { "" } else { " " };
        write!(out, "{separator}{a} {b} {c} -1")?;
    }
    writeln!(out, "\"")?;
    write!(out, r#"        texCoordIndex=""#)?;
    for (i, [a, b, c]) in texture.face_texcoords.iter().enumerate() {
        let separator = if i == 0 { "" } else { " " };
        write!(out, "{separator}{a} {b} {c} -1")?;
    }
    writeln!(out, "\">")?;

    write!(out, r#"        <Coordinate point=""#)?;
    for (i, p) in mesh.positions.iter().enumerate() {
        let separator = if i == 0 { "" } else { ", " };
        write!(out, "{separator}{:.6} {:.6} {:.6}", p.x, -p.y, -p.z)?;
    }
    writeln!(out, "\"/>")?;
    write!(out, r#"        <TextureCoordinate point=""#)?;
    for (i, [u, v]) in texture.texcoords.iter().enumerate() {
        let separator = if i == 0 { "" } else { ", " };
        write!(out, "{separator}{u:.6} {v:.6}")?;
    }
    writeln!(out, "\"/>")?;
    if with_normals {
        write!(out, r#"        <Normal vector=""#)?;
        for (i, n) in mesh.normals.iter().enumerate() {
            let separator = if i == 0 { "" } else { ", " };
            write!(out, "{separator}{:.5} {:.5} {:.5}", n.x, -n.y, -n.z)?;
        }
        writeln!(out, "\"/>")?;
    }

    writeln!(out, "      </IndexedFaceSet>")?;
    writeln!(out, "    </Shape>")?;
    writeln!(out, "  </Scene>")?;
    writeln!(out, "</X3D>")
}

/// `text` with the five characters XML reserves replaced by entities.
fn xml_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(c),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn writes_a_textured_x3d_scene() {
        let mut mesh = Mesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: Vec::new(),
            triangles: vec![[0, 1, 2]],
        };
        mesh.compute_normals();
        let texture = Texture {
            vertex_colours: vec![[1, 2, 3]; 3],
            face_views: vec![crate::types::ViewId(0)],
            texcoords: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            face_texcoords: vec![[2, 1, 0]],
            atlas: Photo::from_rgba(1, 1, vec![0, 0, 0, 255]).unwrap(),
            charts: 1,
            atlas_scale: 1.0,
        };
        let mut buffer = Vec::new();
        write_textured_x3d(&mut buffer, &mesh, &texture, "a & 'b'.png").unwrap();
        let text = String::from_utf8(buffer).unwrap();

        assert!(text.starts_with("<?xml"));
        assert!(text.trim_end().ends_with("</X3D>"));
        assert!(text.contains(r#"coordIndex="0 1 2 -1""#));
        assert!(text.contains(r#"texCoordIndex="2 1 0 -1">"#));
        assert!(text.contains(r#"<ImageTexture url='"a &amp; &apos;b&apos;.png"'/>"#));
        assert!(text.contains(r#"<Coordinate point="0.000000 -0.000000 -0.000000, 1.000000"#));
        assert!(text.contains(r#"<TextureCoordinate point="0.000000 0.000000, 1.000000 0.000000, 0.000000 1.000000"/>"#));
        assert!(text.contains(r#"<Normal vector="0.00000 -0.00000 -1.00000"#));
    }

    #[test]
    fn writes_an_obj_with_one_based_faces() {
        let mut mesh = Mesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: Vec::new(),
            triangles: vec![[0, 1, 2]],
        };
        mesh.compute_normals();
        let mut buffer = Vec::new();
        write_obj(&mut buffer, &mesh).unwrap();
        let text = String::from_utf8(buffer).unwrap();
        assert_eq!(text.lines().filter(|l| l.starts_with("v ")).count(), 3);
        assert!(text.contains("vn 0.00000 -0.00000 -1.00000"));
        assert!(text.contains("f 1//1 2//2 3//3"));
    }

    #[test]
    fn writes_consistent_counts() {
        let points = [
            (World::new(0.0, 0.0, 1.0), [10, 20, 30]),
            (World::new(1.0, -1.0, 2.0), [40, 50, 60]),
        ];
        let cameras = [
            Pose::identity(),
            Pose::look_at(
                &Point3::new(1.0, 0.0, 0.0),
                &Point3::new(0.0, 0.0, 2.0),
                &Vector3::y(),
            ),
        ];
        let k = Intrinsics::estimated(64, 48);
        let mut buffer = Vec::new();
        write_ply(&mut buffer, &points, &cameras, &k, (64, 48), 0.2).unwrap();
        let text = String::from_utf8(buffer).unwrap();

        let (header, body) = text.split_once("end_header\n").unwrap();
        assert!(header.contains("element vertex 12"));
        assert!(header.contains("element face 8"));
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 12 + 8);
        assert_eq!(lines[0], "0.000000 -0.000000 -1.000000 10 20 30");
        assert_eq!(lines[12], "3 2 3 4");
    }
}
