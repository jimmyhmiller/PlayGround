use serde::{Deserialize, Serialize};

/// Stable identifier for an element. Opaque string to stay compatible with
/// Excalidraw's nanoid-style ids when loading `.excalidraw` files.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ElementId(pub String);

impl ElementId {
    pub fn new(s: impl Into<String>) -> Self {
        ElementId(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ElementId {
    fn from(s: &str) -> Self {
        ElementId(s.to_string())
    }
}

impl From<String> for ElementId {
    fn from(s: String) -> Self {
        ElementId(s)
    }
}

impl std::fmt::Display for ElementId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identifier for a group of elements that select/move together.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GroupId(pub String);

impl GroupId {
    pub fn new(s: impl Into<String>) -> Self {
        GroupId(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for GroupId {
    fn from(s: &str) -> Self {
        GroupId(s.to_string())
    }
}
