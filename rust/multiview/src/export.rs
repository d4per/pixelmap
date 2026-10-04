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

/// Writes a textured mesh as binary glTF 2.0 (GLB): one self-contained file with the atlas
/// embedded, which opens in Blender, the Windows 3D Viewer, three.js and most engines.
/// `texture_png` is the atlas already encoded as PNG, so that this crate needs no codec.
///
/// glTF has one texture coordinate per vertex, so a vertex on a chart border is written
/// once for each place it has in the atlas. The material is marked unlit, since the
/// atlas is photographed colour with the lighting already in it. Turned half a revolution
/// about x, like the other exporters, which makes it y up as glTF expects.
pub fn write_textured_glb(
    out: &mut impl Write,
    mesh: &Mesh,
    texture: &Texture,
    texture_png: &[u8],
) -> io::Result<()> {
    if mesh.triangles.is_empty() {
        // glTF accessors must hold at least one element.
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "glTF cannot hold an empty mesh",
        ));
    }
    let with_normals = mesh.normals.len() == mesh.positions.len();

    // One glTF vertex per distinct (position, texture coordinate) corner.
    let mut corners = std::collections::HashMap::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(mesh.triangles.len() * 3);
    for (triangle, uv) in mesh.triangles.iter().zip(&texture.face_texcoords) {
        for (&v, &t) in triangle.iter().zip(uv) {
            let index = *corners.entry((v, t)).or_insert_with(|| {
                let p = mesh.positions[v as usize];
                positions.push([p.x as f32, -p.y as f32, -p.z as f32]);
                if with_normals {
                    let n = mesh.normals[v as usize];
                    normals.push([n.x as f32, -n.y as f32, -n.z as f32]);
                }
                // glTF puts v = 0 at the top of the image; the atlas coordinates do not.
                let [s, t] = texture.texcoords[t as usize];
                texcoords.push([s, 1.0 - t]);
                positions.len() as u32 - 1
            });
            indices.push(index);
        }
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
        }
    }

    // The binary chunk: every vertex attribute, the indices and the PNG, each a buffer view.
    let mut bin: Vec<u8> = Vec::new();
    let mut views: Vec<String> = Vec::new();
    let mut add_view = |bin: &mut Vec<u8>, bytes: &[u8], target: Option<u32>| {
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let target = target.map_or(String::new(), |t| format!(r#","target":{t}"#));
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{offset},"byteLength":{}{target}}}"#,
            bytes.len()
        ));
        views.len() - 1
    };
    const ARRAY_BUFFER: Option<u32> = Some(34962);
    const ELEMENT_ARRAY_BUFFER: Option<u32> = Some(34963);
    let position_view = add_view(
        &mut bin,
        &le_bytes(positions.iter().flatten()),
        ARRAY_BUFFER,
    );
    let normal_view =
        with_normals.then(|| add_view(&mut bin, &le_bytes(normals.iter().flatten()), ARRAY_BUFFER));
    let texcoord_view = add_view(
        &mut bin,
        &le_bytes(texcoords.iter().flatten()),
        ARRAY_BUFFER,
    );
    let index_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let index_view = add_view(&mut bin, &index_bytes, ELEMENT_ARRAY_BUFFER);
    let image_view = add_view(&mut bin, texture_png, None);

    const FLOAT: u32 = 5126;
    const UNSIGNED_INT: u32 = 5125;
    let vertices = positions.len();
    let mut accessors = vec![format!(
        r#"{{"bufferView":{position_view},"componentType":{FLOAT},"count":{vertices},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}}"#,
        min[0], min[1], min[2], max[0], max[1], max[2]
    )];
    let mut attributes = vec![r#""POSITION":0"#.to_string()];
    if let Some(view) = normal_view {
        attributes.push(format!(r#""NORMAL":{}"#, accessors.len()));
        accessors.push(format!(
            r#"{{"bufferView":{view},"componentType":{FLOAT},"count":{vertices},"type":"VEC3"}}"#
        ));
    }
    attributes.push(format!(r#""TEXCOORD_0":{}"#, accessors.len()));
    accessors.push(format!(
        r#"{{"bufferView":{texcoord_view},"componentType":{FLOAT},"count":{vertices},"type":"VEC2"}}"#
    ));
    let index_accessor = accessors.len();
    accessors.push(format!(
        r#"{{"bufferView":{index_view},"componentType":{UNSIGNED_INT},"count":{},"type":"SCALAR"}}"#,
        indices.len()
    ));

    let mut json = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"pixelmap_multiview"}},"#,
            r#""extensionsUsed":["KHR_materials_unlit"],"#,
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"#,
            r#""meshes":[{{"primitives":[{{"attributes":{{{attributes}}},"indices":{index_accessor},"material":0,"mode":4}}]}}],"#,
            r#""materials":[{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}},"metallicFactor":0,"roughnessFactor":1}},"doubleSided":true,"extensions":{{"KHR_materials_unlit":{{}}}}}}],"#,
            r#""textures":[{{"sampler":0,"source":0}}],"#,
            r#""samplers":[{{"magFilter":9729,"minFilter":9729,"wrapS":33071,"wrapT":33071}}],"#,
            r#""images":[{{"bufferView":{image_view},"mimeType":"image/png"}}],"#,
            r#""accessors":[{accessors}],"bufferViews":[{views}],"buffers":[{{"byteLength":{bin_length}}}]}}"#,
        ),
        attributes = attributes.join(","),
        index_accessor = index_accessor,
        image_view = image_view,
        accessors = accessors.join(","),
        views = views.join(","),
        bin_length = bin.len(),
    )
    .into_bytes();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }

    let total = 12 + 8 + json.len() + 8 + bin.len();
    let total = u32::try_from(total).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "mesh too large for one GLB file",
        )
    })?;
    out.write_all(b"glTF")?;
    out.write_all(&2u32.to_le_bytes())?;
    out.write_all(&total.to_le_bytes())?;
    out.write_all(&(json.len() as u32).to_le_bytes())?;
    out.write_all(b"JSON")?;
    out.write_all(&json)?;
    out.write_all(&(bin.len() as u32).to_le_bytes())?;
    out.write_all(b"BIN\0")?;
    out.write_all(&bin)
}

/// The `<model-viewer>` web component [`write_textured_html`] loads, pinned so that a page
/// written today keeps working the same way.
pub const MODEL_VIEWER_URL: &str =
    "https://cdn.jsdelivr.net/npm/@google/model-viewer@4.3.1/dist/model-viewer.min.js";

/// Writes a textured mesh as one self-contained HTML page that shows it in 3D, turned with
/// the mouse or a finger. `texture_png` is the atlas encoded as PNG, as for
/// [`write_textured_glb`], and `title` names the page.
///
/// The model is the GLB from [`write_textured_glb`], embedded as a base64 `data:` URI, so
/// the page is about four thirds the size of the GLB. Only the viewer itself,
/// [`MODEL_VIEWER_URL`], is fetched from the network; tone mapping is off so that the
/// unlit atlas is shown in the photos' own colours.
pub fn write_textured_html(
    out: &mut impl Write,
    mesh: &Mesh,
    texture: &Texture,
    texture_png: &[u8],
    title: &str,
) -> io::Result<()> {
    let mut glb = Vec::new();
    write_textured_glb(&mut glb, mesh, texture, texture_png)?;
    let title = xml_escape(title);

    writeln!(out, "<!doctype html>")?;
    writeln!(out, r#"<html lang="en">"#)?;
    writeln!(out, "<head>")?;
    writeln!(out, r#"<meta charset="utf-8">"#)?;
    writeln!(
        out,
        r#"<meta name="viewport" content="width=device-width, initial-scale=1">"#
    )?;
    writeln!(
        out,
        r#"<meta name="generator" content="pixelmap_multiview">"#
    )?;
    writeln!(out, "<title>{title}</title>")?;
    writeln!(
        out,
        r#"<script type="module" src="{MODEL_VIEWER_URL}"></script>"#
    )?;
    writeln!(out, "<style>")?;
    writeln!(
        out,
        "  html, body {{ margin: 0; height: 100%; background: #202124; color: #e8eaed; font: 14px system-ui, sans-serif; }}"
    )?;
    writeln!(
        out,
        "  model-viewer {{ width: 100%; height: 100%; --poster-color: transparent; }}"
    )?;
    writeln!(
        out,
        "  .caption {{ position: fixed; left: 12px; bottom: 10px; opacity: 0.7; pointer-events: none; }}"
    )?;
    writeln!(
        out,
        "  .fallback {{ position: fixed; inset: 0; display: grid; place-items: center; padding: 16px; text-align: center; }}"
    )?;
    writeln!(out, "</style>")?;
    writeln!(out, "</head>")?;
    writeln!(out, "<body>")?;
    write!(
        out,
        r#"<model-viewer alt="{title}" camera-controls touch-action="pan-y" tone-mapping="none" shadow-intensity="0" src="data:model/gltf-binary;base64,"#
    )?;
    write_base64(out, &glb)?;
    writeln!(out, "\">")?;
    writeln!(
        out,
        r#"  <div class="fallback" slot="poster">Loading the 3D viewer… It is fetched from cdn.jsdelivr.net, so this page needs a network connection.</div>"#
    )?;
    writeln!(out, "</model-viewer>")?;
    writeln!(
        out,
        r#"<noscript><div class="fallback">This page needs JavaScript to show the 3D model.</div></noscript>"#
    )?;
    writeln!(
        out,
        r#"<div class="caption">{title} · made with pixelmap_multiview</div>"#
    )?;
    writeln!(out, "</body>")?;
    writeln!(out, "</html>")
}

/// Writes `bytes` as standard base64 with padding (RFC 4648 §4).
fn write_base64(out: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut line = Vec::with_capacity(4096);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let group = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                line.push(ALPHABET[(group >> (18 - 6 * i) & 63) as usize]);
            } else {
                line.push(b'=');
            }
        }
        if line.len() >= 4096 {
            out.write_all(&line)?;
            line.clear();
        }
    }
    out.write_all(&line)
}

/// The little-endian bytes of a run of floats.
fn le_bytes<'a>(values: impl Iterator<Item = &'a f32>) -> Vec<u8> {
    values.flat_map(|v| v.to_le_bytes()).collect()
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

    /// Two triangles sharing an edge. Vertex 2 sits on a chart border: it has a different
    /// place in the atlas in each triangle.
    fn two_triangles() -> (Mesh, Texture) {
        let mut mesh = Mesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            normals: Vec::new(),
            triangles: vec![[0, 1, 2], [2, 1, 3]],
        };
        mesh.compute_normals();
        let texture = Texture {
            vertex_colours: vec![[1, 2, 3]; 4],
            face_views: vec![crate::types::ViewId(0); 2],
            texcoords: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]],
            face_texcoords: vec![[0, 1, 2], [3, 1, 4]],
            atlas: Photo::from_rgba(1, 1, vec![0, 0, 0, 255]).unwrap(),
            charts: 2,
            atlas_scale: 1.0,
        };
        (mesh, texture)
    }

    #[test]
    fn writes_a_well_formed_glb() {
        let (mesh, texture) = two_triangles();
        let png = b"not really a png";
        let mut glb = Vec::new();
        write_textured_glb(&mut glb, &mesh, &texture, png).unwrap();

        let word = |at: usize| u32::from_le_bytes(glb[at..at + 4].try_into().unwrap()) as usize;
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(word(4), 2);
        assert_eq!(word(8), glb.len());
        let json_length = word(12);
        assert_eq!(&glb[16..20], b"JSON");
        assert_eq!(json_length % 4, 0);
        let json: serde_json::Value = serde_json::from_slice(&glb[20..20 + json_length]).unwrap();
        let bin_start = 20 + json_length;
        let bin_length = word(bin_start);
        assert_eq!(&glb[bin_start + 4..bin_start + 8], b"BIN\0");
        assert_eq!(bin_length % 4, 0);
        assert_eq!(bin_start + 8 + bin_length, glb.len());
        let bin = &glb[bin_start + 8..];
        assert_eq!(json["buffers"][0]["byteLength"], bin_length);

        // Four positions, five corners: vertex 2 is split in two.
        let primitive = &json["meshes"][0]["primitives"][0];
        let accessor = |name: &str| {
            &json["accessors"][primitive["attributes"][name].as_u64().unwrap() as usize]
        };
        assert_eq!(accessor("POSITION")["count"], 5);
        assert_eq!(accessor("NORMAL")["count"], 5);
        assert_eq!(accessor("TEXCOORD_0")["count"], 5);
        let bound = |key: &str| -> Vec<f64> {
            let values = accessor("POSITION")[key].as_array().unwrap();
            values.iter().map(|v| v.as_f64().unwrap()).collect()
        };
        assert_eq!(bound("min"), [0.0, -1.0, 0.0]);
        assert_eq!(bound("max"), [1.0, 0.0, 0.0]);
        let indices = &json["accessors"][primitive["indices"].as_u64().unwrap() as usize];
        assert_eq!(indices["count"], 6);

        let view_bytes = |view: &serde_json::Value| {
            let offset = view["byteOffset"].as_u64().unwrap() as usize;
            &bin[offset..offset + view["byteLength"].as_u64().unwrap() as usize]
        };
        let index_view = &json["bufferViews"][indices["bufferView"].as_u64().unwrap() as usize];
        let index_values: Vec<u32> = view_bytes(index_view)
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(index_values, [0, 1, 2, 3, 1, 4]);
        let uv_view =
            &json["bufferViews"][accessor("TEXCOORD_0")["bufferView"].as_u64().unwrap() as usize];
        let uvs: Vec<f32> = view_bytes(uv_view)
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // Flipped to glTF's v-down convention.
        assert_eq!(uvs, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0]);

        let image = &json["images"][0];
        assert_eq!(image["mimeType"], "image/png");
        let image_view = &json["bufferViews"][image["bufferView"].as_u64().unwrap() as usize];
        assert_eq!(view_bytes(image_view), png);
    }

    #[test]
    fn writes_rfc_4648_base64() {
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            let mut buffer = Vec::new();
            write_base64(&mut buffer, plain.as_bytes()).unwrap();
            assert_eq!(String::from_utf8(buffer).unwrap(), encoded);
        }
        // Long enough to flush mid-way.
        let bytes: Vec<u8> = (0..10_000u32).map(|i| (i * 7) as u8).collect();
        let mut buffer = Vec::new();
        write_base64(&mut buffer, &bytes).unwrap();
        assert_eq!(decode_base64(&buffer), bytes);
    }

    #[test]
    fn writes_an_html_page_with_the_glb_inside() {
        let (mesh, texture) = two_triangles();
        let png = b"not really a png";
        let mut glb = Vec::new();
        write_textured_glb(&mut glb, &mesh, &texture, png).unwrap();
        let mut buffer = Vec::new();
        write_textured_html(&mut buffer, &mesh, &texture, png, "Steps & <stones>").unwrap();
        let html = String::from_utf8(buffer).unwrap();

        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<title>Steps &amp; &lt;stones&gt;</title>"));
        assert!(html.contains(&format!(
            r#"<script type="module" src="{MODEL_VIEWER_URL}">"#
        )));
        assert!(html.contains(r#"tone-mapping="none""#));
        let prefix = "data:model/gltf-binary;base64,";
        let start = html.find(prefix).unwrap() + prefix.len();
        let end = start + html[start..].find('"').unwrap();
        assert_eq!(decode_base64(&html.as_bytes()[start..end]), glb);
    }

    fn decode_base64(text: &[u8]) -> Vec<u8> {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut bytes = Vec::new();
        for group in text.chunks(4) {
            let padding = group.iter().filter(|&&c| c == b'=').count();
            let value = group.iter().fold(0u32, |acc, &c| {
                let digit = ALPHABET.iter().position(|&a| a == c).unwrap_or(0) as u32;
                acc << 6 | digit
            });
            bytes.extend_from_slice(&value.to_be_bytes()[1..4 - padding]);
        }
        bytes
    }

    #[test]
    fn refuses_an_empty_glb() {
        let mesh = Mesh {
            positions: Vec::new(),
            normals: Vec::new(),
            triangles: Vec::new(),
        };
        let texture = Texture {
            vertex_colours: Vec::new(),
            face_views: Vec::new(),
            texcoords: Vec::new(),
            face_texcoords: Vec::new(),
            atlas: Photo::from_rgba(1, 1, vec![0, 0, 0, 255]).unwrap(),
            charts: 0,
            atlas_scale: 1.0,
        };
        let error = write_textured_glb(&mut Vec::new(), &mesh, &texture, &[]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
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
