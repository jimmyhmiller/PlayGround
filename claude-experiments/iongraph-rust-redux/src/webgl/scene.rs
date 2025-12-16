//! Scene graph and draw call structures
//!
//! The scene represents the complete set of primitives to render,
//! organized as a flat list of draw calls with z-ordering and ownership metadata.

use super::element::Rect;

/// Font weight for text rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontWeight {
    Normal,
    Bold,
}

/// Text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

/// RGBA color as f32 components (0.0 - 1.0)
pub type Color = [f32; 4];

/// Convert hex color string to Color
pub fn hex_to_color(hex: &str) -> Color {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0) as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0) as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0) as f32 / 255.0;
    let a = if hex.len() == 8 {
        u8::from_str_radix(&hex[6..8], 16).unwrap_or(255) as f32 / 255.0
    } else {
        1.0
    };
    [r, g, b, a]
}

/// Standard colors from iongraph.css
pub mod colors {
    use super::Color;

    pub const WHITE: Color = [1.0, 1.0, 1.0, 1.0];
    pub const BLACK: Color = [0.0, 0.0, 0.0, 1.0];
    pub const BLOCK_BACKGROUND: Color = [0.976, 0.976, 0.976, 1.0]; // #f9f9f9
    pub const BLOCK_HEADER: Color = [0.047, 0.047, 0.051, 1.0]; // #0c0c0d
    pub const LOOP_HEADER: Color = [0.122, 0.643, 0.067, 1.0]; // #1fa411
    pub const BACKEDGE_BLOCK: Color = [0.078, 0.275, 0.800, 1.0]; // #1446cc
    pub const ARROW_STROKE: Color = [0.0, 0.0, 0.0, 1.0];
    pub const SELECTION_HIGHLIGHT: Color = [1.0, 0.71, 0.31, 0.8]; // Orange
    pub const HOVER_HIGHLIGHT: Color = [1.0, 0.8, 0.5, 0.4]; // Light orange, semi-transparent
}

/// Types of drawable primitives
#[derive(Debug, Clone)]
pub enum Primitive {
    /// Filled rectangle (block background, header)
    Rect {
        rect: Rect,
        color: Color,
        corner_radius: f32,
    },

    /// Stroked rectangle (block border)
    StrokedRect {
        rect: Rect,
        color: Color,
        stroke_width: f32,
    },

    /// Text label
    Text {
        x: f32,
        y: f32,
        text: String,
        color: Color,
        font_size: f32,
        font_weight: FontWeight,
        align: TextAlign,
    },

    /// Tessellated path (arrows) - vertices and indices from lyon
    /// vertices is a flat array: [x0, y0, x1, y1, ...]
    Path {
        vertices: Vec<f32>,
        indices: Vec<u32>,
        color: Color,
    },

    /// Triangle (arrowhead)
    Triangle {
        points: [[f32; 2]; 3],
        color: Color,
    },

    /// Line segment (simple arrows without curves)
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        color: Color,
        width: f32,
    },
}

/// Who owns this draw call (for hit testing and highlighting)
#[derive(Debug, Clone)]
pub enum DrawCallOwner {
    /// Draw call belongs to a block
    Block { block_id: String },

    /// Draw call belongs to an arrow between blocks
    Arrow { from_block: String, to_block: String },

    /// Background element (not interactive)
    Background,
}

/// A draw call with hit-testing metadata
#[derive(Debug, Clone)]
pub struct DrawCall {
    /// The primitive to render
    pub primitive: Primitive,

    /// Z-order for sorting (higher = drawn later = on top)
    pub z_order: i32,

    /// Owner for hit testing and highlighting
    pub owner: DrawCallOwner,
}

impl DrawCall {
    pub fn new(primitive: Primitive, z_order: i32, owner: DrawCallOwner) -> Self {
        Self {
            primitive,
            z_order,
            owner,
        }
    }

    /// Create a background primitive
    pub fn background(primitive: Primitive) -> Self {
        Self::new(primitive, 0, DrawCallOwner::Background)
    }

    /// Create a block primitive
    pub fn block(primitive: Primitive, z_order: i32, block_id: &str) -> Self {
        Self::new(
            primitive,
            z_order,
            DrawCallOwner::Block {
                block_id: block_id.to_string(),
            },
        )
    }

    /// Create an arrow primitive
    pub fn arrow(primitive: Primitive, z_order: i32, from: &str, to: &str) -> Self {
        Self::new(
            primitive,
            z_order,
            DrawCallOwner::Arrow {
                from_block: from.to_string(),
                to_block: to.to_string(),
            },
        )
    }
}

/// Z-order constants for consistent layering
pub mod z_order {
    pub const BACKGROUND: i32 = 0;
    pub const BLOCK_BACKGROUND: i32 = 10;
    pub const BLOCK_BORDER: i32 = 20;
    pub const ARROW_PATH: i32 = 30;
    pub const ARROWHEAD: i32 = 40;
    pub const BLOCK_HEADER: i32 = 50;
    pub const HEADER_TEXT: i32 = 60;
    pub const INSTRUCTION_TEXT: i32 = 70;
    pub const EDGE_LABEL: i32 = 80;
    pub const SELECTION_HIGHLIGHT: i32 = 90;
    pub const HOVER_HIGHLIGHT: i32 = 100;
}

/// The complete scene to render
#[derive(Debug)]
pub struct Scene {
    /// All draw calls in the scene
    pub draw_calls: Vec<DrawCall>,

    /// Bounding box of the entire scene
    pub bounds: Rect,

    /// Whether the scene needs to be re-rendered
    pub dirty: bool,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            draw_calls: Vec::new(),
            bounds: Rect::default(),
            dirty: true,
        }
    }

    /// Clear all draw calls
    pub fn clear(&mut self) {
        self.draw_calls.clear();
        self.bounds = Rect::default();
        self.dirty = true;
    }

    /// Add a draw call to the scene
    pub fn add(&mut self, call: DrawCall) {
        // Update bounds based on primitive
        match &call.primitive {
            Primitive::Rect { rect, .. } | Primitive::StrokedRect { rect, .. } => {
                self.expand_bounds(rect);
            }
            Primitive::Text { x, y, .. } => {
                // Approximate text bounds (will be refined during rendering)
                self.expand_bounds(&Rect::new(*x, *y, 100.0, 20.0));
            }
            Primitive::Triangle { points, .. } => {
                for [px, py] in points {
                    self.expand_bounds(&Rect::new(*px, *py, 1.0, 1.0));
                }
            }
            Primitive::Path { vertices, .. } => {
                // vertices is a flat array [x0, y0, x1, y1, ...]
                for i in (0..vertices.len()).step_by(2) {
                    if i + 1 < vertices.len() {
                        self.expand_bounds(&Rect::new(vertices[i], vertices[i + 1], 1.0, 1.0));
                    }
                }
            }
            Primitive::Line { x1, y1, x2, y2, .. } => {
                self.expand_bounds(&Rect::new(*x1, *y1, 1.0, 1.0));
                self.expand_bounds(&Rect::new(*x2, *y2, 1.0, 1.0));
            }
        }

        self.draw_calls.push(call);
        self.dirty = true;
    }

    /// Expand scene bounds to include a rect
    fn expand_bounds(&mut self, rect: &Rect) {
        if self.bounds.width == 0.0 && self.bounds.height == 0.0 {
            self.bounds = *rect;
        } else {
            let min_x = self.bounds.x.min(rect.x);
            let min_y = self.bounds.y.min(rect.y);
            let max_x = self.bounds.right().max(rect.right());
            let max_y = self.bounds.bottom().max(rect.bottom());
            self.bounds = Rect::new(min_x, min_y, max_x - min_x, max_y - min_y);
        }
    }

    /// Sort draw calls by z-order for proper layering
    pub fn sort_for_rendering(&mut self) {
        self.draw_calls.sort_by_key(|c| c.z_order);
    }

    /// Get draw calls for a specific block (for highlighting)
    pub fn get_block_calls(&self, block_id: &str) -> Vec<&DrawCall> {
        self.draw_calls
            .iter()
            .filter(|c| {
                matches!(&c.owner, DrawCallOwner::Block { block_id: id } if id == block_id)
            })
            .collect()
    }

    /// Get draw calls for arrows connected to a block
    pub fn get_connected_arrow_calls(&self, block_id: &str) -> Vec<&DrawCall> {
        self.draw_calls
            .iter()
            .filter(|c| {
                matches!(&c.owner, DrawCallOwner::Arrow { from_block, to_block }
                    if from_block == block_id || to_block == block_id)
            })
            .collect()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_to_color() {
        let color = hex_to_color("#ff0000");
        assert_eq!(color, [1.0, 0.0, 0.0, 1.0]);

        let color = hex_to_color("#00ff00");
        assert_eq!(color, [0.0, 1.0, 0.0, 1.0]);

        let color = hex_to_color("0000ff80");
        assert!((color[0] - 0.0).abs() < 0.01);
        assert!((color[1] - 0.0).abs() < 0.01);
        assert!((color[2] - 1.0).abs() < 0.01);
        assert!((color[3] - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_scene_bounds() {
        let mut scene = Scene::new();

        scene.add(DrawCall::background(Primitive::Rect {
            rect: Rect::new(10.0, 20.0, 100.0, 50.0),
            color: colors::WHITE,
            corner_radius: 0.0,
        }));

        scene.add(DrawCall::background(Primitive::Rect {
            rect: Rect::new(50.0, 60.0, 200.0, 100.0),
            color: colors::WHITE,
            corner_radius: 0.0,
        }));

        assert_eq!(scene.bounds.x, 10.0);
        assert_eq!(scene.bounds.y, 20.0);
        assert_eq!(scene.bounds.right(), 250.0);
        assert_eq!(scene.bounds.bottom(), 160.0);
    }
}
