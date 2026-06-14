//! Faithful, bidirectional mapping between **real** `.excalidraw` element JSON
//! and our internal [`Element`] model.
//!
//! Reimplemented from Excalidraw's on-disk schema, principally
//! `packages/element/src/types.ts` (the `ExcalidrawElement` interfaces) and the
//! restore/serialization paths in `packages/excalidraw/data/restore.ts`. We do
//! not vendor any JS — this is an independent Rust reimplementation of the same
//! data contract, with attribution recorded in `ATTRIBUTION.md`.
//!
//! ## Why a separate bridge instead of changing `Element`'s derives
//!
//! Excalidraw's element JSON is *flat*: a single object carries a `type`
//! discriminator alongside both the common base fields and the type-specific
//! fields, all in `camelCase`. Our [`Element`] instead splits common fields from
//! a `kind` payload enum and uses Rust-idiomatic names. Rather than contort the
//! foundation type's serde derives (which other modules depend on), this module
//! defines explicit `camelCase` serde structs ([`ExElement`]) and converts
//! to/from [`Element`] by hand.
//!
//! ## Losslessness
//!
//! Base fields that our [`Element`] does not model (e.g. `roundness`,
//! `customData`, `boundElementIds`, editor-private keys) are not silently
//! dropped: [`ExElement::extra`] captures every unmodeled key via
//! `#[serde(flatten)]`, so a load → save cycle preserves them verbatim.

// Derived from Excalidraw: packages/element/src/types.ts,
// packages/excalidraw/data/restore.ts (MIT). Independent Rust reimplementation.

use crate::element::{
    Arrowhead, BoundElement, Element, ElementId, ElementKind, FreedrawData, GroupId, ImageData,
    ImageStatus, LinearData, PointBinding, TextData,
};
use crate::geometry::Point;
use crate::render::{Color, FillStyle, StrokeStyle};
use crate::text::{FontFamily, TextAlign, VerticalAlign};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::IoError;

/// An error mapping a single element between the Excalidraw JSON shape and our
/// [`Element`] model. Always explicit — never a silent fallback.
#[derive(Debug, thiserror::Error)]
pub enum ExcalidrawError {
    #[error("element {id}: unknown element type {ty:?}")]
    UnknownType { id: String, ty: String },
    #[error("element {id}: unknown fontFamily code {code}")]
    UnknownFontFamily { id: String, code: i64 },
    #[error("element {id}: font family {family:?} has no Excalidraw integer code")]
    UnmappableFontFamily { id: String, family: String },
    #[error("element {id}: invalid color string {value:?}")]
    BadColor { id: String, value: String },
    #[error("element {id}: unknown arrowhead {value:?}")]
    UnknownArrowhead { id: String, value: String },
    #[error("element {id}: unknown image status {value:?}")]
    UnknownImageStatus { id: String, value: String },
    #[error("element {id}: type {ty} is missing required field {field}")]
    MissingField {
        id: String,
        ty: &'static str,
        field: &'static str,
    },
}

impl From<ExcalidrawError> for IoError {
    fn from(e: ExcalidrawError) -> Self {
        IoError::Excalidraw(Box::new(e))
    }
}

/// The real `.excalidraw` document envelope (camelCase, `type: "excalidraw"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExDocument {
    #[serde(rename = "type")]
    pub doc_type: String,
    pub version: u32,
    #[serde(default)]
    pub source: String,
    pub elements: Vec<ExElement>,
    /// `appState` and `files` are passed through untouched so a load → save
    /// cycle does not discard editor state we do not model.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub app_state: Value,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub files: Value,
}

/// A single element in real Excalidraw JSON: a flat object with a `type`
/// discriminator plus base and type-specific fields, all camelCase.
///
/// All type-specific fields are optional here; [`ExElement::into_element`]
/// enforces presence per `type`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExElement {
    // --- discriminator ---
    #[serde(rename = "type")]
    pub ty: String,

    // --- base fields ---
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    #[serde(default)]
    pub angle: f64,
    pub stroke_color: String,
    pub background_color: String,
    pub fill_style: String,
    pub stroke_width: f64,
    pub stroke_style: String,
    pub roughness: f64,
    pub opacity: f64,
    #[serde(default)]
    pub seed: u32,
    #[serde(default = "one_u64")]
    pub version: u64,
    #[serde(default)]
    pub version_nonce: u32,
    #[serde(default)]
    pub updated: Option<u64>,
    #[serde(default)]
    pub group_ids: Vec<String>,
    #[serde(default)]
    pub bound_elements: Option<Vec<ExBoundElement>>,
    #[serde(default)]
    pub frame_id: Option<String>,
    #[serde(default)]
    pub link: Option<String>,
    #[serde(default)]
    pub locked: bool,
    #[serde(default)]
    pub index: Option<String>,
    #[serde(default)]
    pub is_deleted: bool,

    // --- text ---
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_family: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_align: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertical_align: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_height: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_text: Option<String>,

    // --- line / arrow ---
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<[f64; 2]>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_committed_point: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_binding: Option<ExBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_binding: Option<ExBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_arrowhead: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_arrowhead: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polygon: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elbowed: Option<bool>,

    // --- freedraw ---
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressures: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simulate_pressure: Option<bool>,

    // --- image ---
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f64; 2]>,

    // --- frame ---
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Every unmodeled key (e.g. `roundness`, `customData`, `boundElementIds`,
    /// editor-private fields). Captured so round-trips are lossless rather than
    /// silently lossy.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn one_u64() -> u64 {
    1
}

/// Excalidraw `boundElements` entry: `{ id, type: "text" | "arrow" }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExBoundElement {
    pub id: String,
    #[serde(rename = "type")]
    pub ty: String,
}

/// Excalidraw point binding: `{ elementId, focus, gap, ... }`. Unmodeled keys
/// (e.g. `fixedPoint`) are preserved in `extra`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExBinding {
    pub element_id: String,
    #[serde(default)]
    pub focus: f64,
    #[serde(default)]
    pub gap: f64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

// ---------------------------------------------------------------------------
// fontFamily <-> integer code
// ---------------------------------------------------------------------------

/// Excalidraw font integer codes: 1 => HandDrawn, 2 => Normal, 3 => Code.
fn font_family_from_code(id: &str, code: i64) -> Result<FontFamily, ExcalidrawError> {
    match code {
        1 => Ok(FontFamily::HandDrawn),
        2 => Ok(FontFamily::Normal),
        3 => Ok(FontFamily::Code),
        other => Err(ExcalidrawError::UnknownFontFamily {
            id: id.to_string(),
            code: other,
        }),
    }
}

/// Inverse of [`font_family_from_code`]. `Custom` families have no Excalidraw
/// integer code, so we error explicitly rather than invent one.
fn font_family_to_code(id: &str, family: &FontFamily) -> Result<i64, ExcalidrawError> {
    match family {
        FontFamily::HandDrawn => Ok(1),
        FontFamily::Normal => Ok(2),
        FontFamily::Code => Ok(3),
        FontFamily::Custom(name) => Err(ExcalidrawError::UnmappableFontFamily {
            id: id.to_string(),
            family: name.clone(),
        }),
    }
}

// ---------------------------------------------------------------------------
// small enum string mappings
// ---------------------------------------------------------------------------

fn color(id: &str, s: &str) -> Result<Color, ExcalidrawError> {
    Color::parse_hex(s).ok_or_else(|| ExcalidrawError::BadColor {
        id: id.to_string(),
        value: s.to_string(),
    })
}

fn fill_style_from_str(s: &str) -> FillStyle {
    match s {
        "hachure" => FillStyle::Hachure,
        "cross-hatch" => FillStyle::CrossHatch,
        "solid" => FillStyle::Solid,
        "zigzag" => FillStyle::Zigzag,
        "dots" => FillStyle::Dots,
        // Excalidraw only emits the values above; preserve forward-compat by
        // treating anything else as its closest documented default. (Hachure is
        // Excalidraw's own default fillStyle.)
        _ => FillStyle::Hachure,
    }
}

fn fill_style_to_str(f: FillStyle) -> &'static str {
    match f {
        FillStyle::Hachure => "hachure",
        FillStyle::CrossHatch => "cross-hatch",
        FillStyle::Solid => "solid",
        FillStyle::Zigzag => "zigzag",
        FillStyle::Dots => "dots",
    }
}

fn stroke_style_from_str(s: &str) -> StrokeStyle {
    match s {
        "dashed" => StrokeStyle::Dashed,
        "dotted" => StrokeStyle::Dotted,
        _ => StrokeStyle::Solid,
    }
}

fn stroke_style_to_str(s: StrokeStyle) -> &'static str {
    match s {
        StrokeStyle::Solid => "solid",
        StrokeStyle::Dashed => "dashed",
        StrokeStyle::Dotted => "dotted",
    }
}

fn text_align_from_str(s: &str) -> TextAlign {
    match s {
        "center" => TextAlign::Center,
        "right" => TextAlign::Right,
        _ => TextAlign::Left,
    }
}

fn text_align_to_str(a: TextAlign) -> &'static str {
    match a {
        TextAlign::Left => "left",
        TextAlign::Center => "center",
        TextAlign::Right => "right",
    }
}

fn vertical_align_from_str(s: &str) -> VerticalAlign {
    match s {
        "top" => VerticalAlign::Top,
        "bottom" => VerticalAlign::Bottom,
        _ => VerticalAlign::Middle,
    }
}

fn vertical_align_to_str(a: VerticalAlign) -> &'static str {
    match a {
        VerticalAlign::Top => "top",
        VerticalAlign::Middle => "middle",
        VerticalAlign::Bottom => "bottom",
    }
}

fn arrowhead_from_str(id: &str, s: &str) -> Result<Arrowhead, ExcalidrawError> {
    Ok(match s {
        "arrow" => Arrowhead::Arrow,
        "triangle" => Arrowhead::Triangle,
        "triangle_outline" => Arrowhead::TriangleOutline,
        "bar" => Arrowhead::Bar,
        "dot" => Arrowhead::Dot,
        "circle" => Arrowhead::Circle,
        "circle_outline" => Arrowhead::CircleOutline,
        "diamond" => Arrowhead::Diamond,
        "diamond_outline" => Arrowhead::DiamondOutline,
        "crowfoot_one" | "crowfoot_many" | "crowfoot_one_or_many" | "crowfoot" => {
            Arrowhead::Crowfoot
        }
        other => {
            return Err(ExcalidrawError::UnknownArrowhead {
                id: id.to_string(),
                value: other.to_string(),
            })
        }
    })
}

fn arrowhead_to_str(a: Arrowhead) -> &'static str {
    match a {
        Arrowhead::Arrow => "arrow",
        Arrowhead::Triangle => "triangle",
        Arrowhead::TriangleOutline => "triangle_outline",
        Arrowhead::Bar => "bar",
        Arrowhead::Dot => "dot",
        Arrowhead::Circle => "circle",
        Arrowhead::CircleOutline => "circle_outline",
        Arrowhead::Diamond => "diamond",
        Arrowhead::DiamondOutline => "diamond_outline",
        Arrowhead::Crowfoot => "crowfoot_one",
    }
}

fn image_status_from_str(id: &str, s: &str) -> Result<ImageStatus, ExcalidrawError> {
    Ok(match s {
        "pending" => ImageStatus::Pending,
        "saved" => ImageStatus::Saved,
        "error" => ImageStatus::Error,
        other => {
            return Err(ExcalidrawError::UnknownImageStatus {
                id: id.to_string(),
                value: other.to_string(),
            })
        }
    })
}

fn image_status_to_str(s: ImageStatus) -> &'static str {
    match s {
        ImageStatus::Pending => "pending",
        ImageStatus::Saved => "saved",
        ImageStatus::Error => "error",
    }
}

fn bound_element_kind_from_str(s: &str) -> Option<crate::element::BoundElementKind> {
    use crate::element::BoundElementKind;
    match s {
        "text" => Some(BoundElementKind::Text),
        "arrow" => Some(BoundElementKind::Arrow),
        _ => None,
    }
}

fn bound_element_kind_to_str(k: crate::element::BoundElementKind) -> &'static str {
    use crate::element::BoundElementKind;
    match k {
        BoundElementKind::Text => "text",
        BoundElementKind::Arrow => "arrow",
    }
}

// ---------------------------------------------------------------------------
// ExElement -> Element
// ---------------------------------------------------------------------------

impl ExElement {
    fn missing(&self, field: &'static str) -> ExcalidrawError {
        ExcalidrawError::MissingField {
            id: self.id.clone(),
            ty: type_static(&self.ty),
            field,
        }
    }

    /// Map this Excalidraw element into our [`Element`].
    pub fn into_element(self) -> Result<Element, ExcalidrawError> {
        let kind = self.build_kind()?;

        let bound_elements = match &self.bound_elements {
            None => Vec::new(),
            Some(list) => {
                let mut out = Vec::with_capacity(list.len());
                for b in list {
                    // Only `text` / `arrow` bound kinds are modeled; others
                    // (e.g. legacy) are skipped rather than guessed.
                    if let Some(kind) = bound_element_kind_from_str(&b.ty) {
                        out.push(BoundElement {
                            id: ElementId::new(b.id.clone()),
                            kind,
                        });
                    }
                }
                out
            }
        };

        Ok(Element {
            id: ElementId::new(self.id.clone()),
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            angle: self.angle,
            stroke_color: color(&self.id, &self.stroke_color)?,
            background_color: color(&self.id, &self.background_color)?,
            fill_style: fill_style_from_str(&self.fill_style),
            stroke_width: self.stroke_width,
            stroke_style: stroke_style_from_str(&self.stroke_style),
            roughness: self.roughness,
            opacity: self.opacity,
            group_ids: self.group_ids.iter().cloned().map(GroupId::new).collect(),
            bound_elements,
            frame_id: self.frame_id.clone().map(ElementId::new),
            is_deleted: self.is_deleted,
            locked: self.locked,
            index: self.index.clone(),
            seed: self.seed,
            version: self.version,
            version_nonce: self.version_nonce,
            updated: self.updated,
            link: self.link.clone(),
            kind,
        })
    }

    fn build_kind(&self) -> Result<ElementKind, ExcalidrawError> {
        Ok(match self.ty.as_str() {
            "rectangle" => ElementKind::Rectangle,
            "ellipse" => ElementKind::Ellipse,
            "diamond" => ElementKind::Diamond,
            "selection" => ElementKind::Selection,
            "text" => ElementKind::Text(self.build_text()?),
            "line" => ElementKind::Line(self.build_linear()?),
            "arrow" => ElementKind::Arrow(self.build_linear()?),
            "freedraw" => ElementKind::Freedraw(self.build_freedraw()?),
            "image" => ElementKind::Image(self.build_image()?),
            "frame" | "magicframe" => self.build_frame()?,
            other => {
                return Err(ExcalidrawError::UnknownType {
                    id: self.id.clone(),
                    ty: other.to_string(),
                })
            }
        })
    }

    fn build_text(&self) -> Result<TextData, ExcalidrawError> {
        let text = self.text.clone().ok_or_else(|| self.missing("text"))?;
        let font_family = match self.font_family {
            Some(code) => font_family_from_code(&self.id, code)?,
            None => FontFamily::HandDrawn,
        };
        Ok(TextData {
            original_text: self.original_text.clone().or_else(|| Some(text.clone())),
            text,
            font_family,
            font_size: self.font_size.unwrap_or(20.0),
            text_align: self
                .text_align
                .as_deref()
                .map(text_align_from_str)
                .unwrap_or(TextAlign::Left),
            vertical_align: self
                .vertical_align
                .as_deref()
                .map(vertical_align_from_str)
                .unwrap_or(VerticalAlign::Top),
            line_height: self.line_height.unwrap_or(1.25),
            container_id: self.container_id.clone().map(ElementId::new),
        })
    }

    fn build_linear(&self) -> Result<LinearData, ExcalidrawError> {
        let points = self.points.clone().ok_or_else(|| self.missing("points"))?;
        let points = points
            .into_iter()
            .map(|[x, y]| Point::new(x, y))
            .collect::<Vec<_>>();
        let start_arrowhead = match &self.start_arrowhead {
            Some(s) => Some(arrowhead_from_str(&self.id, s)?),
            None => None,
        };
        let end_arrowhead = match &self.end_arrowhead {
            Some(s) => Some(arrowhead_from_str(&self.id, s)?),
            None => None,
        };
        Ok(LinearData {
            points,
            polygon: self.polygon.unwrap_or(false),
            elbowed: self.elbowed.unwrap_or(false),
            start_binding: self.start_binding.as_ref().map(ExBinding::to_binding),
            end_binding: self.end_binding.as_ref().map(ExBinding::to_binding),
            start_arrowhead,
            end_arrowhead,
        })
    }

    fn build_freedraw(&self) -> Result<FreedrawData, ExcalidrawError> {
        let points = self.points.clone().ok_or_else(|| self.missing("points"))?;
        let points = points
            .into_iter()
            .map(|[x, y]| Point::new(x, y))
            .collect::<Vec<_>>();
        Ok(FreedrawData {
            points,
            pressures: self.pressures.clone().unwrap_or_default(),
            simulate_pressure: self.simulate_pressure.unwrap_or(true),
        })
    }

    /// Build the `frame` kind. `FrameData` is not re-exported from the
    /// foundation `element` module, so we construct the variant via serde using
    /// `ElementKind`'s own `{ "type": "frame", "name": ... }` tagged shape
    /// rather than reaching into a private type.
    fn build_frame(&self) -> Result<ElementKind, ExcalidrawError> {
        let mut v = Map::new();
        v.insert("type".to_string(), Value::from("frame"));
        v.insert(
            "name".to_string(),
            match &self.name {
                Some(n) => Value::from(n.clone()),
                None => Value::Null,
            },
        );
        serde_json::from_value(Value::Object(v)).map_err(|_| ExcalidrawError::MissingField {
            id: self.id.clone(),
            ty: "frame",
            field: "name",
        })
    }

    fn build_image(&self) -> Result<ImageData, ExcalidrawError> {
        let file_id = self.file_id.clone().ok_or_else(|| self.missing("fileId"))?;
        let status = match &self.status {
            Some(s) => image_status_from_str(&self.id, s)?,
            None => ImageStatus::Pending,
        };
        let scale = self.scale.map(|[x, y]| (x, y)).unwrap_or((1.0, 1.0));
        Ok(ImageData {
            file_id,
            status,
            scale,
        })
    }
}

impl ExBinding {
    fn to_binding(&self) -> PointBinding {
        PointBinding {
            element_id: ElementId::new(self.element_id.clone()),
            focus: self.focus,
            gap: self.gap,
        }
    }

    fn from_binding(b: &PointBinding) -> ExBinding {
        ExBinding {
            element_id: b.element_id.0.clone(),
            focus: b.focus,
            gap: b.gap,
            extra: Map::new(),
        }
    }
}

/// Map a `type` string to one of our static type names (best-effort, for error
/// messages only).
fn type_static(ty: &str) -> &'static str {
    match ty {
        "rectangle" => "rectangle",
        "ellipse" => "ellipse",
        "diamond" => "diamond",
        "text" => "text",
        "line" => "line",
        "arrow" => "arrow",
        "freedraw" => "freedraw",
        "image" => "image",
        "frame" => "frame",
        "selection" => "selection",
        _ => "element",
    }
}

// ---------------------------------------------------------------------------
// Element -> ExElement
// ---------------------------------------------------------------------------

impl ExElement {
    /// Build the Excalidraw JSON shape from our [`Element`].
    pub fn from_element(el: &Element) -> Result<ExElement, ExcalidrawError> {
        let id = el.id.0.clone();

        // Start with all per-type fields as None; fill in per kind below.
        let mut ex = ExElement {
            ty: el.type_name().to_string(),
            id: id.clone(),
            x: el.x,
            y: el.y,
            width: el.width,
            height: el.height,
            angle: el.angle,
            stroke_color: el.stroke_color.to_hex(),
            background_color: el.background_color.to_hex(),
            fill_style: fill_style_to_str(el.fill_style).to_string(),
            stroke_width: el.stroke_width,
            stroke_style: stroke_style_to_str(el.stroke_style).to_string(),
            roughness: el.roughness,
            opacity: el.opacity,
            seed: el.seed,
            version: el.version,
            version_nonce: el.version_nonce,
            updated: el.updated,
            group_ids: el.group_ids.iter().map(|g| g.0.clone()).collect(),
            bound_elements: if el.bound_elements.is_empty() {
                None
            } else {
                Some(
                    el.bound_elements
                        .iter()
                        .map(|b| ExBoundElement {
                            id: b.id.0.clone(),
                            ty: bound_element_kind_to_str(b.kind).to_string(),
                        })
                        .collect(),
                )
            },
            frame_id: el.frame_id.as_ref().map(|f| f.0.clone()),
            link: el.link.clone(),
            locked: el.locked,
            index: el.index.clone(),
            is_deleted: el.is_deleted,

            text: None,
            font_size: None,
            font_family: None,
            text_align: None,
            vertical_align: None,
            container_id: None,
            line_height: None,
            original_text: None,

            points: None,
            last_committed_point: None,
            start_binding: None,
            end_binding: None,
            start_arrowhead: None,
            end_arrowhead: None,
            polygon: None,
            elbowed: None,

            pressures: None,
            simulate_pressure: None,

            file_id: None,
            status: None,
            scale: None,

            name: None,

            extra: Map::new(),
        };

        match &el.kind {
            ElementKind::Rectangle
            | ElementKind::Ellipse
            | ElementKind::Diamond
            | ElementKind::Selection => {}
            ElementKind::Text(t) => {
                ex.text = Some(t.text.clone());
                ex.font_size = Some(t.font_size);
                ex.font_family = Some(font_family_to_code(&id, &t.font_family)?);
                ex.text_align = Some(text_align_to_str(t.text_align).to_string());
                ex.vertical_align = Some(vertical_align_to_str(t.vertical_align).to_string());
                ex.container_id = t.container_id.as_ref().map(|c| c.0.clone());
                ex.line_height = Some(t.line_height);
                ex.original_text = t.original_text.clone();
            }
            ElementKind::Line(l) | ElementKind::Arrow(l) => {
                ex.points = Some(l.points.iter().map(|p| [p.x, p.y]).collect());
                ex.polygon = Some(l.polygon);
                ex.elbowed = Some(l.elbowed);
                ex.start_binding = l.start_binding.as_ref().map(ExBinding::from_binding);
                ex.end_binding = l.end_binding.as_ref().map(ExBinding::from_binding);
                ex.start_arrowhead = l.start_arrowhead.map(|a| arrowhead_to_str(a).to_string());
                ex.end_arrowhead = l.end_arrowhead.map(|a| arrowhead_to_str(a).to_string());
            }
            ElementKind::Freedraw(f) => {
                ex.points = Some(f.points.iter().map(|p| [p.x, p.y]).collect());
                if !f.pressures.is_empty() {
                    ex.pressures = Some(f.pressures.clone());
                }
                ex.simulate_pressure = Some(f.simulate_pressure);
            }
            ElementKind::Image(im) => {
                ex.file_id = Some(im.file_id.clone());
                ex.status = Some(image_status_to_str(im.status).to_string());
                ex.scale = Some([im.scale.0, im.scale.1]);
            }
            ElementKind::Frame(_) => {
                // `FrameData` is not re-exported, so read its `name` back out
                // through `ElementKind`'s tagged serde shape instead of
                // pattern-binding the private field.
                let v =
                    serde_json::to_value(&el.kind).map_err(|_| ExcalidrawError::MissingField {
                        id: id.clone(),
                        ty: "frame",
                        field: "name",
                    })?;
                ex.name = v
                    .get("name")
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string());
            }
        }

        Ok(ex)
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Parse a real `.excalidraw` document string into a list of [`Element`]s.
///
/// Validates the envelope `type`, then maps each element. `appState`/`files`
/// are not returned by this convenience wrapper; use [`load_excalidraw_doc`] to
/// keep them.
pub fn load_excalidraw_str(text: &str) -> Result<Vec<Element>, IoError> {
    let doc = load_excalidraw_doc(text)?;
    let mut out = Vec::with_capacity(doc.elements.len());
    for ex in doc.elements {
        out.push(ex.into_element()?);
    }
    Ok(out)
}

/// Parse a real `.excalidraw` document, preserving the full envelope (including
/// `appState` and `files`) as [`ExElement`]s for callers that need them.
pub fn load_excalidraw_doc(text: &str) -> Result<ExDocument, IoError> {
    let doc: ExDocument = serde_json::from_str(text)?;
    if doc.doc_type != "excalidraw" {
        return Err(IoError::UnsupportedType(doc.doc_type));
    }
    Ok(doc)
}

/// Serialize a list of [`Element`]s to a real `.excalidraw` document string.
pub fn save_excalidraw_str(elements: &[Element]) -> Result<String, IoError> {
    let mut ex_elements = Vec::with_capacity(elements.len());
    for el in elements {
        ex_elements.push(ExElement::from_element(el)?);
    }
    let doc = ExDocument {
        doc_type: "excalidraw".to_string(),
        version: 2,
        source: "headless-whiteboard".to_string(),
        elements: ex_elements,
        app_state: Value::Null,
        files: Value::Null,
    };
    Ok(serde_json::to_string_pretty(&doc)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small but real-format `.excalidraw` document: a rectangle, an arrow,
    /// and a text element. Field shapes match what Excalidraw actually exports.
    const SAMPLE: &str = r##"{
      "type": "excalidraw",
      "version": 2,
      "source": "https://excalidraw.com",
      "elements": [
        {
          "id": "rect-1",
          "type": "rectangle",
          "x": 100.0,
          "y": 120.0,
          "width": 200.0,
          "height": 80.0,
          "angle": 0,
          "strokeColor": "#1e1e1e",
          "backgroundColor": "#a5d8ff",
          "fillStyle": "solid",
          "strokeWidth": 2,
          "strokeStyle": "solid",
          "roughness": 1,
          "opacity": 100,
          "groupIds": [],
          "frameId": null,
          "roundness": { "type": 3 },
          "seed": 1968410350,
          "version": 24,
          "versionNonce": 1059881630,
          "isDeleted": false,
          "boundElements": [{ "id": "arrow-1", "type": "arrow" }],
          "updated": 1700000000000,
          "link": null,
          "locked": false
        },
        {
          "id": "arrow-1",
          "type": "arrow",
          "x": 320.0,
          "y": 160.0,
          "width": 120.0,
          "height": 40.0,
          "angle": 0,
          "strokeColor": "#e03131",
          "backgroundColor": "transparent",
          "fillStyle": "hachure",
          "strokeWidth": 1,
          "strokeStyle": "dashed",
          "roughness": 2,
          "opacity": 90,
          "groupIds": [],
          "frameId": null,
          "roundness": { "type": 2 },
          "seed": 12345,
          "version": 7,
          "versionNonce": 999,
          "isDeleted": false,
          "boundElements": null,
          "updated": 1700000000001,
          "link": null,
          "locked": false,
          "points": [[0, 0], [120, 40]],
          "lastCommittedPoint": [120, 40],
          "startBinding": { "elementId": "rect-1", "focus": 0.1, "gap": 4 },
          "endBinding": null,
          "startArrowhead": null,
          "endArrowhead": "triangle"
        },
        {
          "id": "text-1",
          "type": "text",
          "x": 110.0,
          "y": 130.0,
          "width": 180.0,
          "height": 25.0,
          "angle": 0,
          "strokeColor": "#1e1e1e",
          "backgroundColor": "transparent",
          "fillStyle": "hachure",
          "strokeWidth": 1,
          "strokeStyle": "solid",
          "roughness": 1,
          "opacity": 100,
          "groupIds": [],
          "frameId": null,
          "roundness": null,
          "seed": 777,
          "version": 3,
          "versionNonce": 42,
          "isDeleted": false,
          "boundElements": null,
          "updated": 1700000000002,
          "link": null,
          "locked": false,
          "text": "Hello",
          "fontSize": 20,
          "fontFamily": 3,
          "textAlign": "center",
          "verticalAlign": "middle",
          "containerId": null,
          "lineHeight": 1.25,
          "originalText": "Hello"
        }
      ],
      "appState": { "viewBackgroundColor": "#ffffff" },
      "files": {}
    }"##;

    #[test]
    fn parses_sample_to_elements() {
        let els = load_excalidraw_str(SAMPLE).unwrap();
        assert_eq!(els.len(), 3);

        // --- rectangle ---
        let r = &els[0];
        assert_eq!(r.id, ElementId::new("rect-1"));
        assert_eq!(r.type_name(), "rectangle");
        assert_eq!(r.x, 100.0);
        assert_eq!(r.y, 120.0);
        assert_eq!(r.width, 200.0);
        assert_eq!(r.height, 80.0);
        assert_eq!(r.stroke_color, Color::parse_hex("#1e1e1e").unwrap());
        assert_eq!(r.background_color, Color::parse_hex("#a5d8ff").unwrap());
        assert_eq!(r.fill_style, FillStyle::Solid);
        assert_eq!(r.stroke_width, 2.0);
        assert_eq!(r.roughness, 1.0);
        assert_eq!(r.opacity, 100.0);
        assert_eq!(r.seed, 1968410350);
        assert_eq!(r.version, 24);
        assert_eq!(r.version_nonce, 1059881630);
        assert_eq!(r.updated, Some(1700000000000));
        assert!(!r.is_deleted);
        assert_eq!(r.bound_elements.len(), 1);
        assert_eq!(r.bound_elements[0].id, ElementId::new("arrow-1"));

        // --- arrow ---
        let a = &els[1];
        assert_eq!(a.type_name(), "arrow");
        assert_eq!(a.stroke_style, StrokeStyle::Dashed);
        assert_eq!(a.roughness, 2.0);
        assert_eq!(a.opacity, 90.0);
        match &a.kind {
            ElementKind::Arrow(l) => {
                assert_eq!(l.points.len(), 2);
                assert_eq!(l.points[0], Point::new(0.0, 0.0));
                assert_eq!(l.points[1], Point::new(120.0, 40.0));
                assert_eq!(l.end_arrowhead, Some(Arrowhead::Triangle));
                assert_eq!(l.start_arrowhead, None);
                let sb = l.start_binding.as_ref().unwrap();
                assert_eq!(sb.element_id, ElementId::new("rect-1"));
                assert_eq!(sb.focus, 0.1);
                assert_eq!(sb.gap, 4.0);
                assert!(l.end_binding.is_none());
            }
            other => panic!("expected arrow, got {other:?}"),
        }

        // --- text ---
        let t = &els[2];
        assert_eq!(t.type_name(), "text");
        match &t.kind {
            ElementKind::Text(td) => {
                assert_eq!(td.text, "Hello");
                assert_eq!(td.font_family, FontFamily::Code); // code 3
                assert_eq!(td.font_size, 20.0);
                assert_eq!(td.text_align, TextAlign::Center);
                assert_eq!(td.vertical_align, VerticalAlign::Middle);
                assert_eq!(td.line_height, 1.25);
                assert_eq!(td.original_text.as_deref(), Some("Hello"));
                assert!(td.container_id.is_none());
            }
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn reserializes_to_equivalent_json() {
        let els = load_excalidraw_str(SAMPLE).unwrap();
        let saved = save_excalidraw_str(&els).unwrap();
        let reloaded = load_excalidraw_str(&saved).unwrap();
        // Our Element model is the canonical form; a load -> save -> load cycle
        // must be a fixed point on it.
        assert_eq!(els, reloaded);
    }

    #[test]
    fn unmodeled_base_fields_survive_round_trip() {
        // `roundness` is not modeled on Element; it must be preserved verbatim
        // through the ExElement bridge rather than silently dropped.
        let doc = load_excalidraw_doc(SAMPLE).unwrap();
        let rect = &doc.elements[0];
        assert_eq!(
            rect.extra.get("roundness"),
            Some(&serde_json::json!({ "type": 3 }))
        );
        let reser = serde_json::to_value(rect).unwrap();
        assert_eq!(reser["roundness"], serde_json::json!({ "type": 3 }));
    }

    #[test]
    fn font_family_codes_map_both_ways() {
        assert_eq!(
            font_family_from_code("x", 1).unwrap(),
            FontFamily::HandDrawn
        );
        assert_eq!(font_family_from_code("x", 2).unwrap(), FontFamily::Normal);
        assert_eq!(font_family_from_code("x", 3).unwrap(), FontFamily::Code);
        assert!(font_family_from_code("x", 9).is_err());

        assert_eq!(font_family_to_code("x", &FontFamily::HandDrawn).unwrap(), 1);
        assert_eq!(font_family_to_code("x", &FontFamily::Normal).unwrap(), 2);
        assert_eq!(font_family_to_code("x", &FontFamily::Code).unwrap(), 3);
        assert!(font_family_to_code("x", &FontFamily::Custom("Comic".into())).is_err());
    }

    #[test]
    fn rejects_unknown_element_type() {
        let bad = r##"{
          "type": "excalidraw", "version": 2, "source": "t",
          "elements": [{
            "id": "x", "type": "embeddable",
            "x": 0, "y": 0, "width": 1, "height": 1, "angle": 0,
            "strokeColor": "#000000", "backgroundColor": "transparent",
            "fillStyle": "hachure", "strokeWidth": 1, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": []
          }]
        }"##;
        let err = load_excalidraw_str(bad).unwrap_err();
        assert!(matches!(err, IoError::Excalidraw(_)), "got {err:?}");
    }

    #[test]
    fn rejects_wrong_envelope_type() {
        let bad = r##"{"type":"notexcalidraw","version":2,"elements":[]}"##;
        assert!(matches!(
            load_excalidraw_doc(bad),
            Err(IoError::UnsupportedType(_))
        ));
    }

    #[test]
    fn missing_required_field_errors() {
        // text element with no `text`
        let bad = r##"{
          "type": "excalidraw", "version": 2, "source": "t",
          "elements": [{
            "id": "t1", "type": "text",
            "x": 0, "y": 0, "width": 1, "height": 1, "angle": 0,
            "strokeColor": "#000000", "backgroundColor": "transparent",
            "fillStyle": "hachure", "strokeWidth": 1, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": []
          }]
        }"##;
        let err = load_excalidraw_str(bad).unwrap_err();
        match err {
            IoError::Excalidraw(b) => {
                assert!(matches!(
                    *b,
                    ExcalidrawError::MissingField { field: "text", .. }
                ));
            }
            other => panic!("expected Excalidraw error, got {other:?}"),
        }
    }

    #[test]
    fn freedraw_and_image_and_frame_round_trip() {
        let json = r##"{
          "type": "excalidraw", "version": 2, "source": "t",
          "elements": [
            {
              "id": "fd", "type": "freedraw",
              "x": 0, "y": 0, "width": 10, "height": 10, "angle": 0,
              "strokeColor": "#000000", "backgroundColor": "transparent",
              "fillStyle": "hachure", "strokeWidth": 1, "strokeStyle": "solid",
              "roughness": 1, "opacity": 100, "groupIds": [],
              "points": [[0,0],[5,5],[10,10]],
              "pressures": [0.1, 0.5, 0.9],
              "simulatePressure": false
            },
            {
              "id": "img", "type": "image",
              "x": 0, "y": 0, "width": 64, "height": 64, "angle": 0,
              "strokeColor": "#000000", "backgroundColor": "transparent",
              "fillStyle": "hachure", "strokeWidth": 1, "strokeStyle": "solid",
              "roughness": 1, "opacity": 100, "groupIds": [],
              "fileId": "abc123", "status": "saved", "scale": [1, -1]
            },
            {
              "id": "frm", "type": "frame",
              "x": 0, "y": 0, "width": 300, "height": 200, "angle": 0,
              "strokeColor": "#000000", "backgroundColor": "transparent",
              "fillStyle": "hachure", "strokeWidth": 1, "strokeStyle": "solid",
              "roughness": 1, "opacity": 100, "groupIds": [],
              "name": "My Frame"
            }
          ]
        }"##;
        let els = load_excalidraw_str(json).unwrap();
        assert_eq!(els.len(), 3);

        match &els[0].kind {
            ElementKind::Freedraw(f) => {
                assert_eq!(f.points.len(), 3);
                assert_eq!(f.pressures, vec![0.1, 0.5, 0.9]);
                assert!(!f.simulate_pressure);
            }
            other => panic!("expected freedraw, got {other:?}"),
        }
        match &els[1].kind {
            ElementKind::Image(im) => {
                assert_eq!(im.file_id, "abc123");
                assert_eq!(im.status, ImageStatus::Saved);
                assert_eq!(im.scale, (1.0, -1.0));
            }
            other => panic!("expected image, got {other:?}"),
        }
        match &els[2].kind {
            ElementKind::Frame(fr) => assert_eq!(fr.name.as_deref(), Some("My Frame")),
            other => panic!("expected frame, got {other:?}"),
        }

        // round trip
        let saved = save_excalidraw_str(&els).unwrap();
        let back = load_excalidraw_str(&saved).unwrap();
        assert_eq!(els, back);
    }
}
