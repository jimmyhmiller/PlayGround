//! Element-type-specific payloads. Reimplemented from Excalidraw's per-type
//! element interfaces.

use super::ElementId;
use crate::geometry::Point;
use crate::text::{FontFamily, TextAlign, VerticalAlign};
use serde::{Deserialize, Serialize};

/// The element-type-specific data carried by an [`super::Element`].
///
/// The discriminant corresponds to Excalidraw's `type` field. Generic shapes
/// (rectangle/ellipse/diamond/frame) need no extra data; the rest carry a
/// payload struct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ElementKind {
    Rectangle,
    Ellipse,
    Diamond,
    Line(LinearData),
    Arrow(LinearData),
    Freedraw(FreedrawData),
    Text(TextData),
    Image(ImageData),
    /// A frame: a named container that clips and groups the elements inside it.
    Frame(FrameData),
    /// A selection marquee (transient; not persisted as a real element).
    Selection,
}

impl ElementKind {
    pub fn type_name(&self) -> &'static str {
        match self {
            ElementKind::Rectangle => "rectangle",
            ElementKind::Ellipse => "ellipse",
            ElementKind::Diamond => "diamond",
            ElementKind::Line(_) => "line",
            ElementKind::Arrow(_) => "arrow",
            ElementKind::Freedraw(_) => "freedraw",
            ElementKind::Text(_) => "text",
            ElementKind::Image(_) => "image",
            ElementKind::Frame(_) => "frame",
            ElementKind::Selection => "selection",
        }
    }

    /// Whether the interior can be filled (closed generic shapes + closed lines).
    pub fn is_fillable(&self) -> bool {
        match self {
            ElementKind::Rectangle | ElementKind::Ellipse | ElementKind::Diamond => true,
            ElementKind::Line(l) => l.polygon,
            _ => false,
        }
    }
}

/// Whether a linear element is an open line or an arrow (affects arrowheads).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearKind {
    Line,
    Arrow,
}

/// Shared payload for `line` and `arrow` elements.
///
/// `points` are relative to the element's `(x, y)` origin (matching Excalidraw):
/// the first point is conventionally `(0, 0)` and the box `width`/`height`
/// derive from the point extents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearData {
    /// Vertices relative to the element origin.
    pub points: Vec<Point>,
    /// Whether the linear path is closed and fillable (Excalidraw `polygon`).
    #[serde(default)]
    pub polygon: bool,
    /// Whether the arrow routes as orthogonal "elbow" segments.
    #[serde(default)]
    pub elbowed: bool,
    pub start_binding: Option<PointBinding>,
    pub end_binding: Option<PointBinding>,
    #[serde(default)]
    pub start_arrowhead: Option<Arrowhead>,
    #[serde(default)]
    pub end_arrowhead: Option<Arrowhead>,
}

impl LinearData {
    pub fn line(points: Vec<Point>) -> Self {
        LinearData {
            points,
            polygon: false,
            elbowed: false,
            start_binding: None,
            end_binding: None,
            start_arrowhead: None,
            end_arrowhead: None,
        }
    }

    pub fn arrow(points: Vec<Point>) -> Self {
        LinearData {
            points,
            polygon: false,
            elbowed: false,
            start_binding: None,
            end_binding: None,
            start_arrowhead: None,
            end_arrowhead: Some(Arrowhead::Triangle),
        }
    }
}

/// Binding from an arrow/line endpoint to another element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointBinding {
    pub element_id: ElementId,
    /// How far along the bound element's edge the focus point sits (-1..=1).
    pub focus: f64,
    /// Gap kept between the arrow tip and the bound element's edge.
    pub gap: f64,
}

/// Arrowhead shapes. Mirrors Excalidraw's arrowhead set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arrowhead {
    Arrow,
    Triangle,
    TriangleOutline,
    Bar,
    Dot,
    Circle,
    CircleOutline,
    Diamond,
    DiamondOutline,
    Crowfoot,
}

/// Payload for `freedraw` (pen / pencil) elements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreedrawData {
    /// Path points relative to the element origin.
    pub points: Vec<Point>,
    /// Per-point pen pressure (0..=1); empty if pressure was not captured.
    #[serde(default)]
    pub pressures: Vec<f64>,
    /// Whether pressure-based variable width is applied.
    #[serde(default)]
    pub simulate_pressure: bool,
}

impl FreedrawData {
    pub fn new(points: Vec<Point>) -> Self {
        FreedrawData {
            points,
            pressures: Vec::new(),
            simulate_pressure: true,
        }
    }
}

/// Payload for `text` elements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextData {
    pub text: String,
    pub font_family: FontFamily,
    pub font_size: f64,
    pub text_align: TextAlign,
    pub vertical_align: VerticalAlign,
    pub line_height: f64,
    /// If this text is bound inside another element (label), its container id.
    pub container_id: Option<ElementId>,
    /// Cached original text before wrapping (Excalidraw keeps this for re-wrap).
    #[serde(default)]
    pub original_text: Option<String>,
}

impl TextData {
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        TextData {
            original_text: Some(text.clone()),
            text,
            font_family: FontFamily::HandDrawn,
            font_size: 20.0,
            text_align: TextAlign::Left,
            vertical_align: VerticalAlign::Top,
            line_height: 1.25,
            container_id: None,
        }
    }
}

/// Load/decoding status of an image's bytes (held by the backend, not core).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageStatus {
    Pending,
    Saved,
    Error,
}

/// Payload for `image` elements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageData {
    /// Identifier the backend uses to resolve the actual pixels.
    pub file_id: String,
    pub status: ImageStatus,
    /// Horizontal/vertical flip scale (1.0 or -1.0 per axis).
    pub scale: (f64, f64),
}

impl ImageData {
    pub fn new(file_id: impl Into<String>) -> Self {
        ImageData {
            file_id: file_id.into(),
            status: ImageStatus::Pending,
            scale: (1.0, 1.0),
        }
    }
}

/// Payload for `frame` elements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameData {
    pub name: Option<String>,
}

/// What kind of element a [`BoundElement`] reference points to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoundElementKind {
    Text,
    Arrow,
}

/// A reference from a container element to an element bound to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundElement {
    pub id: ElementId,
    #[serde(rename = "type")]
    pub kind: BoundElementKind,
}

/// Corner-rounding style for generic shapes and lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundnessKind {
    /// Legacy proportional radius.
    Legacy,
    /// Radius proportional to the smaller side.
    ProportionalRadius,
    /// Fixed radius in scene units.
    AdaptiveRadius,
}

/// Optional roundness applied to a shape's corners.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Roundness {
    #[serde(rename = "type")]
    pub kind: RoundnessKind,
    /// Radius value; interpretation depends on `kind`.
    pub value: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_tags_match_excalidraw() {
        assert_eq!(ElementKind::Rectangle.type_name(), "rectangle");
        assert_eq!(
            ElementKind::Arrow(LinearData::arrow(vec![])).type_name(),
            "arrow"
        );
        assert_eq!(
            ElementKind::Freedraw(FreedrawData::new(vec![])).type_name(),
            "freedraw"
        );
    }

    #[test]
    fn fillable_rules() {
        assert!(ElementKind::Rectangle.is_fillable());
        assert!(!ElementKind::Line(LinearData::line(vec![])).is_fillable());
        let mut poly = LinearData::line(vec![]);
        poly.polygon = true;
        assert!(ElementKind::Line(poly).is_fillable());
    }

    #[test]
    fn kind_serializes_with_type_tag() {
        let k = ElementKind::Text(TextData::new("hi"));
        let json = serde_json::to_value(&k).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hi");
    }
}
