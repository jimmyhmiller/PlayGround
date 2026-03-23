use serde::{Deserialize, Serialize};

/// Atomic types in the system.
/// `Del` is a tombstone marking a deleted position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomicType {
    Num,
    Str,
    Bool,
    Del,
}

/// A value conforming to an atomic type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
    Null,
    Error,
}

// Manual Eq/Hash for Value because f64 doesn't implement them.
// We use the bit representation of f64 for Hash (treats NaN as equal to itself).
impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Num(n) => n.to_bits().hash(state),
            Value::Str(s) => s.hash(state),
            Value::Bool(b) => b.hash(state),
            Value::Null | Value::Error => {}
        }
    }
}

/// A single field in a document: a value paired with its type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub value: Value,
    pub ty: AtomicType,
}

/// A document is a typed tuple of fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub fields: Vec<Field>,
}

/// A unique identifier for Ins operations (to distinguish inserts).
pub type InsId = u64;

/// Edit operations on documents.
///
/// The paper defines 4 operations (Id, Ins, Conv, Move). We extend with 2 more:
/// - Rename: changes the name of a field (paper Section 3, point 5)
/// - Set: changes the value of a field (orthogonal to Conv which changes type)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Edit {
    /// Do nothing.
    Id,
    /// Insert type `ty` at index `idx`, shifting indexes >= idx right.
    /// `id` is a unique identifier for this insert.
    Ins { idx: usize, ty: AtomicType, id: InsId },
    /// Convert index `idx` to type `ty`. Value is retained (paper p. 8).
    Conv { idx: usize, ty: AtomicType },
    /// Set index `i` to have the type and value of index `j`, then delete `j`.
    Move { i: usize, j: usize },
    /// Rename field at `idx` to `name`. Schema metadata only — the document
    /// structure is unchanged. Follows the same OT patterns as Conv (shifts
    /// with Ins, follows Move, conflicts at same index).
    Rename { idx: usize, name: String },
    /// Set the value at `idx`. Type is NOT changed (orthogonal to Conv).
    /// Follows the same OT patterns as Conv.
    Set { idx: usize, value: Value },
}

impl Document {
    pub fn new(fields: Vec<Field>) -> Self {
        Document { fields }
    }

    pub fn empty() -> Self {
        Document { fields: vec![] }
    }

    pub fn from_types(types: &[AtomicType]) -> Self {
        Document {
            fields: types
                .iter()
                .map(|ty| Field {
                    value: Value::Null,
                    ty: *ty,
                })
                .collect(),
        }
    }

    pub fn types(&self) -> Vec<AtomicType> {
        self.fields.iter().map(|f| f.ty).collect()
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl Field {
    pub fn new(value: Value, ty: AtomicType) -> Self {
        Field { value, ty }
    }

    pub fn null(ty: AtomicType) -> Self {
        Field {
            value: Value::Null,
            ty,
        }
    }
}

impl Edit {
    pub fn is_id(&self) -> bool {
        matches!(self, Edit::Id)
    }
}
