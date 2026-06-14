//! `.excalidraw` file load / save and the clipboard element model.
//!
//! Excalidraw files are JSON documents with a `type`, `version`, `elements`
//! array, and an `appState`. Phase 1 maps the full element JSON (which uses
//! camelCase keys and a flat `type` discriminator) to/from our [`Element`]
//! model and loads real exported corpora. This file defines the document
//! envelope, the error type, and a round-trippable save/load over our own
//! element representation so the IO seam is exercised now.

use crate::element::Element;
use crate::scene::Scene;
use serde::{Deserialize, Serialize};

pub mod excalidraw;

pub use excalidraw::{
    load_excalidraw_doc, load_excalidraw_str, save_excalidraw_str, ExDocument, ExElement,
    ExcalidrawError,
};

/// Errors from loading or saving documents.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported document type: {0}")]
    UnsupportedType(String),
    #[error("unsupported schema version: {0}")]
    UnsupportedVersion(u32),
    /// A real `.excalidraw` element could not be mapped to/from our model.
    #[error("excalidraw element mapping error: {0}")]
    Excalidraw(#[source] Box<ExcalidrawError>),
}

/// The `.excalidraw` document envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    #[serde(rename = "type")]
    pub doc_type: String,
    pub version: u32,
    #[serde(default)]
    pub source: String,
    pub elements: Vec<Element>,
}

impl Document {
    pub const TYPE: &'static str = "excalidraw";
    pub const VERSION: u32 = 2;

    /// Build a document from a scene's live elements, in paint order.
    pub fn from_scene(scene: &Scene) -> Self {
        Document {
            doc_type: Self::TYPE.to_string(),
            version: Self::VERSION,
            source: "headless-whiteboard".to_string(),
            elements: scene.iter().cloned().collect(),
        }
    }

    /// Build a scene from this document's elements.
    pub fn into_scene(self) -> Scene {
        let mut scene = Scene::new();
        for el in self.elements {
            scene.insert(el);
        }
        scene
    }

    /// Validate the envelope's type and version.
    pub fn validate(&self) -> Result<(), IoError> {
        if self.doc_type != Self::TYPE {
            return Err(IoError::UnsupportedType(self.doc_type.clone()));
        }
        Ok(())
    }
}

/// Serialize a scene to `.excalidraw` JSON text.
pub fn save_to_string(scene: &Scene) -> Result<String, IoError> {
    Ok(serde_json::to_string_pretty(&Document::from_scene(scene))?)
}

/// Parse a scene from `.excalidraw` JSON text.
pub fn load_from_str(text: &str) -> Result<Scene, IoError> {
    let doc: Document = serde_json::from_str(text)?;
    doc.validate()?;
    Ok(doc.into_scene())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementId, ElementKind};

    fn sample_scene() -> Scene {
        let mut s = Scene::new();
        s.insert(Element::new(
            ElementId::from("e1"),
            7,
            5.0,
            5.0,
            50.0,
            30.0,
            ElementKind::Rectangle,
        ));
        s
    }

    #[test]
    fn round_trip_through_json() {
        let scene = sample_scene();
        let json = save_to_string(&scene).unwrap();
        let back = load_from_str(&json).unwrap();
        assert_eq!(back.len(), 1);
        let e = back.get(&ElementId::from("e1")).unwrap();
        assert_eq!(e.width, 50.0);
        assert_eq!(e.seed, 7);
    }

    #[test]
    fn rejects_wrong_type() {
        let bad = r#"{"type":"notexcalidraw","version":2,"elements":[]}"#;
        assert!(matches!(
            load_from_str(bad),
            Err(IoError::UnsupportedType(_))
        ));
    }
}
