//! SVG path tessellation using lyon
//!
//! Converts SVG path commands (M, L, A, C, Z) to triangles for WebGL rendering.

use lyon::math::{point, Angle};
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};

/// Tessellated path ready for WebGL
#[derive(Debug, Clone)]
pub struct TessellatedPath {
    /// Vertex positions [x, y, x, y, ...]
    pub vertices: Vec<f32>,
    /// Triangle indices
    pub indices: Vec<u32>,
}

impl TessellatedPath {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

impl Default for TessellatedPath {
    fn default() -> Self {
        Self::new()
    }
}

/// SVG path commands we need to support
#[derive(Debug, Clone)]
pub enum SvgCommand {
    /// M x y - move to absolute position
    MoveTo(f32, f32),
    /// L x y - line to absolute position
    LineTo(f32, f32),
    /// A rx ry x-rotation large-arc sweep x y - elliptical arc
    Arc {
        rx: f32,
        ry: f32,
        x_rotation: f32,
        large_arc: bool,
        sweep: bool,
        x: f32,
        y: f32,
    },
    /// C x1 y1 x2 y2 x y - cubic bezier
    CubicTo {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        x: f32,
        y: f32,
    },
    /// Q x1 y1 x y - quadratic bezier
    QuadTo {
        x1: f32,
        y1: f32,
        x: f32,
        y: f32,
    },
    /// Z - close path
    Close,
}

/// Parse SVG path d attribute into commands
pub fn parse_svg_path(d: &str) -> Vec<SvgCommand> {
    let mut commands = Vec::new();
    let mut chars = d.chars().peekable();
    let mut current_cmd = ' ';

    while chars.peek().is_some() {
        // Skip whitespace and commas
        while chars.peek().map_or(false, |c| c.is_whitespace() || *c == ',') {
            chars.next();
        }

        if chars.peek().is_none() {
            break;
        }

        // Check for command letter
        if chars.peek().map_or(false, |c| c.is_alphabetic()) {
            current_cmd = chars.next().unwrap();
        }

        // Skip whitespace after command
        while chars.peek().map_or(false, |c| c.is_whitespace() || *c == ',') {
            chars.next();
        }

        match current_cmd.to_ascii_uppercase() {
            'M' => {
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                commands.push(SvgCommand::MoveTo(x, y));
                // Subsequent coordinates are LineTo
                current_cmd = if current_cmd.is_uppercase() { 'L' } else { 'l' };
            }
            'L' => {
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                commands.push(SvgCommand::LineTo(x, y));
            }
            'A' => {
                let rx = parse_number(&mut chars);
                let ry = parse_number(&mut chars);
                let x_rotation = parse_number(&mut chars);
                let large_arc = parse_number(&mut chars) != 0.0;
                let sweep = parse_number(&mut chars) != 0.0;
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                commands.push(SvgCommand::Arc {
                    rx,
                    ry,
                    x_rotation,
                    large_arc,
                    sweep,
                    x,
                    y,
                });
            }
            'C' => {
                let x1 = parse_number(&mut chars);
                let y1 = parse_number(&mut chars);
                let x2 = parse_number(&mut chars);
                let y2 = parse_number(&mut chars);
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                commands.push(SvgCommand::CubicTo {
                    x1,
                    y1,
                    x2,
                    y2,
                    x,
                    y,
                });
            }
            'Q' => {
                let x1 = parse_number(&mut chars);
                let y1 = parse_number(&mut chars);
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                commands.push(SvgCommand::QuadTo { x1, y1, x, y });
            }
            'Z' | 'z' => {
                commands.push(SvgCommand::Close);
            }
            _ => {
                // Unknown command, skip to next letter
                while chars.peek().map_or(false, |c| !c.is_alphabetic()) {
                    chars.next();
                }
            }
        }
    }

    commands
}

/// Parse a number from the character stream
fn parse_number<I: Iterator<Item = char>>(chars: &mut std::iter::Peekable<I>) -> f32 {
    // Skip whitespace and commas
    while chars.peek().map_or(false, |c| c.is_whitespace() || *c == ',') {
        chars.next();
    }

    let mut s = String::new();

    // Handle negative sign
    if chars.peek() == Some(&'-') {
        s.push(chars.next().unwrap());
    } else if chars.peek() == Some(&'+') {
        chars.next();
    }

    // Collect digits and decimal point
    while chars
        .peek()
        .map_or(false, |c| c.is_ascii_digit() || *c == '.')
    {
        s.push(chars.next().unwrap());
    }

    // Handle exponent
    if chars.peek() == Some(&'e') || chars.peek() == Some(&'E') {
        s.push(chars.next().unwrap());
        if chars.peek() == Some(&'-') || chars.peek() == Some(&'+') {
            s.push(chars.next().unwrap());
        }
        while chars.peek().map_or(false, |c| c.is_ascii_digit()) {
            s.push(chars.next().unwrap());
        }
    }

    s.parse().unwrap_or(0.0)
}

/// Convert parsed commands to a lyon Path
fn commands_to_path(commands: &[SvgCommand]) -> Path {
    let mut builder = Path::builder();
    let mut current = point(0.0, 0.0);
    let mut path_start = current;
    let mut first_move = true;

    for cmd in commands {
        match cmd {
            SvgCommand::MoveTo(x, y) => {
                if !first_move {
                    builder.end(false);
                }
                first_move = false;
                current = point(*x, *y);
                path_start = current;
                builder.begin(current);
            }
            SvgCommand::LineTo(x, y) => {
                current = point(*x, *y);
                builder.line_to(current);
            }
            SvgCommand::Arc {
                rx,
                ry,
                x_rotation,
                large_arc,
                sweep,
                x,
                y,
            } => {
                let end = point(*x, *y);
                // Convert SVG arc to lyon arc using the svg_arc helper
                let arc = lyon::path::geom::SvgArc {
                    from: current,
                    to: end,
                    radii: lyon::math::vector(*rx, *ry),
                    x_rotation: Angle::degrees(*x_rotation),
                    flags: lyon::path::geom::ArcFlags {
                        large_arc: *large_arc,
                        sweep: *sweep,
                    },
                };

                // Convert arc to quadratic beziers
                arc.for_each_quadratic_bezier(&mut |curve| {
                    builder.quadratic_bezier_to(curve.ctrl, curve.to);
                });

                current = end;
            }
            SvgCommand::CubicTo {
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            } => {
                let ctrl1 = point(*x1, *y1);
                let ctrl2 = point(*x2, *y2);
                current = point(*x, *y);
                builder.cubic_bezier_to(ctrl1, ctrl2, current);
            }
            SvgCommand::QuadTo { x1, y1, x, y } => {
                let ctrl = point(*x1, *y1);
                current = point(*x, *y);
                builder.quadratic_bezier_to(ctrl, current);
            }
            SvgCommand::Close => {
                builder.end(true);
                current = path_start;
                first_move = true;
            }
        }
    }

    if !first_move {
        builder.end(false);
    }

    builder.build()
}

/// Tessellate an SVG path string into triangles for stroke rendering
pub fn tessellate_stroke(d: &str, stroke_width: f32) -> TessellatedPath {
    let commands = parse_svg_path(d);
    if commands.is_empty() {
        return TessellatedPath::new();
    }

    let path = commands_to_path(&commands);
    let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut tessellator = StrokeTessellator::new();

    let result = tessellator.tessellate_path(
        &path,
        &StrokeOptions::default()
            .with_line_width(stroke_width)
            .with_line_cap(lyon::tessellation::LineCap::Round)
            .with_line_join(lyon::tessellation::LineJoin::Round)
            .with_tolerance(0.5),
        &mut BuffersBuilder::new(&mut geometry, |vertex: lyon::tessellation::StrokeVertex| {
            [vertex.position().x, vertex.position().y]
        }),
    );

    if result.is_err() {
        return TessellatedPath::new();
    }

    // Flatten vertex array
    let vertices: Vec<f32> = geometry
        .vertices
        .iter()
        .flat_map(|v| [v[0], v[1]])
        .collect();

    TessellatedPath {
        vertices,
        indices: geometry.indices,
    }
}

/// Tessellate an SVG path string into triangles for fill rendering
pub fn tessellate_fill(d: &str) -> TessellatedPath {
    let commands = parse_svg_path(d);
    if commands.is_empty() {
        return TessellatedPath::new();
    }

    let path = commands_to_path(&commands);
    let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();

    let result = tessellator.tessellate_path(
        &path,
        &FillOptions::default().with_tolerance(0.5),
        &mut BuffersBuilder::new(&mut geometry, |vertex: lyon::tessellation::FillVertex| {
            [vertex.position().x, vertex.position().y]
        }),
    );

    if result.is_err() {
        return TessellatedPath::new();
    }

    // Flatten vertex array
    let vertices: Vec<f32> = geometry
        .vertices
        .iter()
        .flat_map(|v| [v[0], v[1]])
        .collect();

    TessellatedPath {
        vertices,
        indices: geometry.indices,
    }
}

/// Create a simple triangle (for arrowheads)
pub fn create_triangle(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> TessellatedPath {
    TessellatedPath {
        vertices: vec![p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]],
        indices: vec![0, 1, 2],
    }
}

/// Create a rectangle
pub fn create_rect(x: f32, y: f32, width: f32, height: f32) -> TessellatedPath {
    TessellatedPath {
        vertices: vec![
            x,
            y, // top-left
            x + width,
            y, // top-right
            x + width,
            y + height, // bottom-right
            x,
            y + height, // bottom-left
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// Create a stroked rectangle (4 quads for the border)
pub fn create_stroked_rect(x: f32, y: f32, width: f32, height: f32, stroke: f32) -> TessellatedPath {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let hs = stroke / 2.0; // half stroke

    // Top edge
    add_quad(
        &mut vertices,
        &mut indices,
        x - hs,
        y - hs,
        width + stroke,
        stroke,
    );
    // Bottom edge
    add_quad(
        &mut vertices,
        &mut indices,
        x - hs,
        y + height - hs,
        width + stroke,
        stroke,
    );
    // Left edge (excluding corners already covered)
    add_quad(
        &mut vertices,
        &mut indices,
        x - hs,
        y + hs,
        stroke,
        height - stroke,
    );
    // Right edge
    add_quad(
        &mut vertices,
        &mut indices,
        x + width - hs,
        y + hs,
        stroke,
        height - stroke,
    );

    TessellatedPath { vertices, indices }
}

fn add_quad(vertices: &mut Vec<f32>, indices: &mut Vec<u32>, x: f32, y: f32, w: f32, h: f32) {
    let base = (vertices.len() / 2) as u32;
    vertices.extend_from_slice(&[x, y, x + w, y, x + w, y + h, x, y + h]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_path() {
        let commands = parse_svg_path("M 10 20 L 30 40");
        assert_eq!(commands.len(), 2);
        assert!(matches!(commands[0], SvgCommand::MoveTo(10.0, 20.0)));
        assert!(matches!(commands[1], SvgCommand::LineTo(30.0, 40.0)));
    }

    #[test]
    fn test_parse_arc() {
        let commands = parse_svg_path("M 0 0 A 10 10 0 0 1 20 20");
        assert_eq!(commands.len(), 2);
        if let SvgCommand::Arc {
            rx,
            ry,
            large_arc,
            sweep,
            x,
            y,
            ..
        } = &commands[1]
        {
            assert_eq!(*rx, 10.0);
            assert_eq!(*ry, 10.0);
            assert!(!*large_arc);
            assert!(*sweep);
            assert_eq!(*x, 20.0);
            assert_eq!(*y, 20.0);
        } else {
            panic!("Expected Arc command");
        }
    }

    #[test]
    fn test_tessellate_line() {
        let path = tessellate_stroke("M 0 0 L 100 0", 2.0);
        assert!(!path.is_empty());
        assert!(!path.vertices.is_empty());
        assert!(!path.indices.is_empty());
    }

    #[test]
    fn test_create_rect() {
        let rect = create_rect(10.0, 20.0, 100.0, 50.0);
        assert_eq!(rect.vertices.len(), 8); // 4 vertices * 2 components
        assert_eq!(rect.indices.len(), 6); // 2 triangles * 3 indices
    }
}
