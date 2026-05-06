//! Build the icon atlas from Phosphor-style duotone SVGs.
//!
//! Each Phosphor duotone SVG ships with two `<path>` elements: a
//! "fill" path tagged `opacity="0.2"` (the icon's body region — the
//! shallow tier in the carve) and the main outline path with no
//! opacity (the strokes — the deeper tier). We classify by opacity,
//! flatten each path's bezier data to polylines via kurbo, then
//! apply the SVG nonzero fill rule by treating positive-area
//! sub-contours as additive regions and negative-area sub-contours
//! as holes (subtracted from the additive union via geo's boolean
//! ops). The result is a polygonal atlas in the same JSON shape
//! that `extract_icons` produces from PNG sheets, so the runtime
//! loader doesn't need to care which extractor authored it.
//!
//! Run:    cargo run --bin extract_svg_icons
//! Reads:  assets/icons/phosphor/*.svg
//! Writes: assets/icons/icons.json

use geo::{Contains, Coord, LineString, MultiPolygon, Point as GeoPoint, Polygon};
use kurbo::{BezPath, PathEl, Point as KurboPoint};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
struct IconAtlas {
    legend: Vec<[u8; 3]>,
    icons: Vec<IconEntry>,
}

#[derive(Serialize)]
struct IconEntry {
    row: usize,
    col: usize,
    name: String,
    aspect: f32,
    layers: Vec<DepthLayer>,
}

#[derive(Serialize)]
struct DepthLayer {
    depth: u8,
    polygons: Vec<PolygonWithHoles>,
}

#[derive(Serialize)]
struct PolygonWithHoles {
    exterior: Vec<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    holes: Vec<Vec<[f32; 2]>>,
}

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let svg_dir = manifest.join("assets/icons/phosphor");
    let output = manifest.join("assets/icons/icons.json");

    let mut entries: Vec<_> = std::fs::read_dir(&svg_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", svg_dir.display()))
        .filter_map(|r| r.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "svg"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut icons = Vec::new();
    for entry in entries {
        let path = entry.path();
        let svg = std::fs::read_to_string(&path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        match parse_svg_icon(&svg, name.clone()) {
            Some(icon) => {
                let n_polys: usize = icon.layers.iter().map(|l| l.polygons.len()).sum();
                eprintln!("{}: {} layers, {} polys", name, icon.layers.len(), n_polys);
                icons.push(icon);
            }
            None => eprintln!("skipped {} (no recognizable paths)", name),
        }
    }
    eprintln!("extracted {} icons", icons.len());

    let atlas = IconAtlas {
        legend: vec![],
        icons,
    };
    let json = serde_json::to_string_pretty(&atlas).unwrap();
    std::fs::write(&output, json).unwrap_or_else(|e| panic!("write {}: {e}", output.display()));
    eprintln!("wrote {}", output.display());
}

fn parse_svg_icon(svg: &str, name: String) -> Option<IconEntry> {
    let doc = roxmltree::Document::parse(svg).ok()?;
    let root = doc.root_element();
    let viewbox = parse_viewbox(root.attribute("viewBox")?)?;

    // Two depth bins matching Phosphor's duotone convention.
    let mut depth_polys: [MultiPolygon<f64>; 2] =
        [MultiPolygon(vec![]), MultiPolygon(vec![])];

    for node in root.descendants() {
        if !node.has_tag_name("path") {
            continue;
        }
        let Some(d) = node.attribute("d") else {
            continue;
        };
        let opacity: f32 = node
            .attribute("opacity")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        // opacity ≈ 0.2 → faint duotone fill (icon body, depth 0)
        // opacity ≈ 1.0 → solid outline (icon strokes, depth 1)
        let depth = if opacity < 0.5 { 0 } else { 1 };
        let mp = path_to_multipolygon(d);
        if !mp.0.is_empty() {
            depth_polys[depth].0.extend(mp.0);
        }
    }

    let (vx, vy, vw, vh) = viewbox;
    let layers: Vec<DepthLayer> = depth_polys
        .iter()
        .enumerate()
        .filter_map(|(d, mp)| {
            if mp.0.is_empty() {
                return None;
            }
            // Emit each polygon's exterior ring. Holes inside
            // composite polygons (e.g., the inside of a ring shape)
            // get dropped here — for the icons in our set this is
            // fine because the runtime carve subtracts polygons
            // (no nesting needed), and the duotone source data has
            // already had its fill rule resolved by the boolean
            // ops above so each emitted exterior is a real solid
            // region.
            let polygons: Vec<PolygonWithHoles> = mp
                .0
                .iter()
                .filter_map(|p| {
                    let exterior = ring_to_local(p.exterior(), vx, vy, vw, vh);
                    if exterior.len() < 3 {
                        return None;
                    }
                    let holes: Vec<Vec<[f32; 2]>> = p
                        .interiors()
                        .iter()
                        .map(|h| ring_to_local(h, vx, vy, vw, vh))
                        .filter(|r| r.len() >= 3)
                        .collect();
                    Some(PolygonWithHoles { exterior, holes })
                })
                .collect();
            if polygons.is_empty() {
                None
            } else {
                Some(DepthLayer {
                    depth: d as u8,
                    polygons,
                })
            }
        })
        .collect();

    if layers.is_empty() {
        return None;
    }

    Some(IconEntry {
        row: 0,
        col: 0,
        name,
        aspect: (vw / vh) as f32,
        layers,
    })
}

fn parse_viewbox(s: &str) -> Option<(f64, f64, f64, f64)> {
    let parts: Vec<f64> = s
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    if parts.len() == 4 {
        Some((parts[0], parts[1], parts[2], parts[3]))
    } else {
        None
    }
}

/// Parse an SVG path's `d` attribute, flatten its curves to a
/// polyline at sub-pixel tolerance, and resolve the nonzero fill
/// rule by computing `union(positive_area_subcontours) - union(
/// negative_area_subcontours)`.
fn path_to_multipolygon(d: &str) -> MultiPolygon<f64> {
    let bez = match BezPath::from_svg(d) {
        Ok(b) => b,
        Err(_) => return MultiPolygon(vec![]),
    };

    let mut subcontours: Vec<Vec<Coord<f64>>> = Vec::new();
    let mut current: Vec<Coord<f64>> = Vec::new();

    let tolerance = 0.5;
    kurbo::flatten(bez.iter(), tolerance, |el| match el {
        PathEl::MoveTo(p) => {
            if current.len() >= 3 {
                subcontours.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
            current.push(point_to_coord(p));
        }
        PathEl::LineTo(p) => {
            current.push(point_to_coord(p));
        }
        PathEl::ClosePath => {
            if current.len() >= 3 {
                if current[0] != *current.last().unwrap() {
                    let first = current[0];
                    current.push(first);
                }
                subcontours.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
        // After flatten we only see Move/Line/Close — Quad/Cubic
        // get expanded into LineTo segments.
        _ => {}
    });
    if current.len() >= 3 {
        subcontours.push(current);
    }

    // Phosphor (and most icon SVGs) draws ring shapes by combining
    // an outer contour with inner contours of opposite winding.
    // Sign of the signed area is *not* a reliable indicator of
    // "exterior vs hole" — we've seen Phosphor write outers CW and
    // inners CCW, but other authors do the opposite. The robust
    // structural test is geometric containment: sort contours
    // largest-first by absolute area, normalize them all to CCW,
    // and nest each contour as a hole of the first existing
    // exterior that contains its first point.
    let mut contours: Vec<Vec<Coord<f64>>> = subcontours
        .into_iter()
        .filter(|c| c.len() >= 3 && signed_area(c).abs() > 0.01)
        .map(|mut c| {
            if signed_area(&c) < 0.0 {
                c.reverse();
            }
            c
        })
        .collect();
    contours.sort_by(|a, b| {
        signed_area(b)
            .abs()
            .partial_cmp(&signed_area(a).abs())
            .unwrap()
    });

    // Each entry: (exterior ring, holes).
    let mut shells: Vec<(Vec<Coord<f64>>, Vec<Vec<Coord<f64>>>)> = Vec::new();
    for contour in contours {
        let test_pt = GeoPoint::new(contour[0].x, contour[0].y);
        let mut parent_idx = None;
        for (i, (ext, _)) in shells.iter().enumerate() {
            let parent = Polygon::new(LineString::from(ext.clone()), vec![]);
            if parent.contains(&test_pt) {
                parent_idx = Some(i);
                break;
            }
        }
        if let Some(i) = parent_idx {
            shells[i].1.push(contour);
        } else {
            shells.push((contour, vec![]));
        }
    }

    let polys: Vec<Polygon<f64>> = shells
        .into_iter()
        .map(|(ext, holes)| {
            let hole_ls: Vec<LineString<f64>> = holes.into_iter().map(LineString::from).collect();
            Polygon::new(LineString::from(ext), hole_ls)
        })
        .collect();
    MultiPolygon(polys)
}

fn point_to_coord(p: KurboPoint) -> Coord<f64> {
    Coord { x: p.x, y: p.y }
}

fn signed_area(pts: &[Coord<f64>]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        s += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    s * 0.5
}

fn ring_to_local(ring: &LineString<f64>, vx: f64, vy: f64, vw: f64, vh: f64) -> Vec<[f32; 2]> {
    ring.0
        .iter()
        .map(|c| [((c.x - vx) / vw) as f32, ((c.y - vy) / vh) as f32])
        .collect()
}
