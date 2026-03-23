use crate::types::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A named field in a schema. Position in the Vec = index in the edit algebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedField {
    pub name: String,
    pub ty: AtomicType,
}

/// Error type for schema operations.
#[derive(Debug)]
pub enum SchemaError {
    FieldNotFound(String),
    FieldAlreadyExists(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::FieldNotFound(name) => write!(f, "field not found: {}", name),
            SchemaError::FieldAlreadyExists(name) => write!(f, "field already exists: {}", name),
        }
    }
}

impl std::error::Error for SchemaError {}

/// A schema defines named, typed fields over the positional edit algebra.
///
/// Internally, fields are a Vec where position = index in the core Document.
/// Deleted fields (AtomicType::Del) keep their slot (tombstone) but are hidden
/// from the "active" named view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub fields: Vec<NamedField>,
    next_ins_id: u64,
}

impl Schema {
    /// Create a schema from ordered (name, type) pairs.
    pub fn new(fields: Vec<(&str, AtomicType)>, ins_id_start: u64) -> Self {
        Schema {
            fields: fields
                .into_iter()
                .map(|(name, ty)| NamedField {
                    name: name.to_string(),
                    ty,
                })
                .collect(),
            next_ins_id: ins_id_start,
        }
    }

    /// Look up a field's positional index by name.
    /// Returns None if the field doesn't exist or is deleted.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields
            .iter()
            .position(|f| f.name == name && f.ty != AtomicType::Del)
    }

    /// Return all active (non-deleted) fields with their positional indexes.
    pub fn active_fields(&self) -> Vec<(usize, &NamedField)> {
        self.fields
            .iter()
            .enumerate()
            .filter(|(_, f)| f.ty != AtomicType::Del)
            .collect()
    }

    /// Add a new field at the end. Returns the edits to apply:
    /// an Ins (structural) and a Rename (sets the field name).
    /// Both must be recorded in diff tracking for proper merge.
    pub fn add_field(&mut self, name: &str, ty: AtomicType) -> Result<Vec<Edit>, SchemaError> {
        if self.index_of(name).is_some() {
            return Err(SchemaError::FieldAlreadyExists(name.to_string()));
        }
        let idx = self.fields.len();
        let id = self.next_ins_id;
        self.next_ins_id += 1;
        self.fields.push(NamedField {
            name: name.to_string(),
            ty,
        });
        Ok(vec![
            Edit::Ins { idx, ty, id },
            Edit::Rename { idx, name: name.to_string() },
        ])
    }

    /// Remove a field by name (tombstones it). Returns the Edit.
    pub fn remove_field(&mut self, name: &str) -> Result<Edit, SchemaError> {
        let idx = self
            .index_of(name)
            .ok_or_else(|| SchemaError::FieldNotFound(name.to_string()))?;
        self.fields[idx].ty = AtomicType::Del;
        Ok(Edit::Conv {
            idx,
            ty: AtomicType::Del,
        })
    }

    /// Convert a field's type. Returns the Edit.
    pub fn convert_field(&mut self, name: &str, to: AtomicType) -> Result<Edit, SchemaError> {
        let idx = self
            .index_of(name)
            .ok_or_else(|| SchemaError::FieldNotFound(name.to_string()))?;
        self.fields[idx].ty = to;
        Ok(Edit::Conv { idx, ty: to })
    }

    /// Rename a field. Returns a Rename edit for OT tracking.
    pub fn rename_field(&mut self, old: &str, new: &str) -> Result<Edit, SchemaError> {
        if self.index_of(new).is_some() {
            return Err(SchemaError::FieldAlreadyExists(new.to_string()));
        }
        let idx = self
            .index_of(old)
            .ok_or_else(|| SchemaError::FieldNotFound(old.to_string()))?;
        self.fields[idx].name = new.to_string();
        Ok(Edit::Rename { idx, name: new.to_string() })
    }

    /// Create a Document with null values matching this schema's types.
    pub fn empty_document(&self) -> Document {
        Document::from_types(
            &self.fields.iter().map(|f| f.ty).collect::<Vec<_>>(),
        )
    }

    /// Get the field name at a positional index.
    pub fn name_at(&self, idx: usize) -> Option<&str> {
        self.fields.get(idx).map(|f| f.name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_field() {
        let mut schema = Schema::new(vec![("name", AtomicType::Str)], 100);
        let edits = schema.add_field("age", AtomicType::Num).unwrap();
        assert_eq!(edits.len(), 2);
        assert_eq!(edits[0], Edit::Ins { idx: 1, ty: AtomicType::Num, id: 100 });
        assert_eq!(edits[1], Edit::Rename { idx: 1, name: "age".into() });
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.index_of("age"), Some(1));
    }

    #[test]
    fn test_add_field_duplicate() {
        let mut schema = Schema::new(vec![("name", AtomicType::Str)], 100);
        assert!(schema.add_field("name", AtomicType::Str).is_err());
    }

    #[test]
    fn test_remove_field() {
        let mut schema = Schema::new(
            vec![("name", AtomicType::Str), ("age", AtomicType::Num)],
            100,
        );
        let edit = schema.remove_field("age").unwrap();
        assert_eq!(
            edit,
            Edit::Conv {
                idx: 1,
                ty: AtomicType::Del
            }
        );
        // Field is tombstoned, not removed
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.index_of("age"), None);
        assert_eq!(schema.active_fields().len(), 1);
    }

    #[test]
    fn test_convert_field() {
        let mut schema = Schema::new(
            vec![("name", AtomicType::Str), ("age", AtomicType::Num)],
            100,
        );
        let edit = schema.convert_field("age", AtomicType::Str).unwrap();
        assert_eq!(
            edit,
            Edit::Conv {
                idx: 1,
                ty: AtomicType::Str
            }
        );
        assert_eq!(schema.fields[1].ty, AtomicType::Str);
    }

    #[test]
    fn test_rename_field() {
        let mut schema = Schema::new(
            vec![("name", AtomicType::Str), ("age", AtomicType::Num)],
            100,
        );
        schema.rename_field("name", "full_name").unwrap();
        assert_eq!(schema.index_of("name"), None);
        assert_eq!(schema.index_of("full_name"), Some(0));
    }

    #[test]
    fn test_add_after_remove_reuses_nothing() {
        let mut schema = Schema::new(
            vec![("name", AtomicType::Str), ("age", AtomicType::Num)],
            100,
        );
        schema.remove_field("age").unwrap();
        let edits = schema.add_field("email", AtomicType::Str).unwrap();
        // New field goes at the end (index 2), not in the tombstoned slot
        assert_eq!(edits[0], Edit::Ins { idx: 2, ty: AtomicType::Str, id: 100 });
        assert_eq!(schema.fields.len(), 3);
    }

    #[test]
    fn test_schema_applies_to_document() {
        use crate::apply::apply;

        let mut schema = Schema::new(
            vec![("name", AtomicType::Str), ("age", AtomicType::Num)],
            100,
        );
        let mut doc = schema.empty_document();
        assert_eq!(doc.types(), vec![AtomicType::Str, AtomicType::Num]);

        // Add a field to the schema and apply structural edit to document
        let edits = schema.add_field("active", AtomicType::Bool).unwrap();
        doc = apply(&doc, &edits[0]).unwrap(); // apply only the Ins, not the Rename
        assert_eq!(
            doc.types(),
            vec![AtomicType::Str, AtomicType::Num, AtomicType::Bool]
        );

        // Convert age from Num to Str
        let edit = schema.convert_field("age", AtomicType::Str).unwrap();
        doc = apply(&doc, &edit).unwrap();
        assert_eq!(
            doc.types(),
            vec![AtomicType::Str, AtomicType::Str, AtomicType::Bool]
        );

        // Remove the active field
        let edit = schema.remove_field("active").unwrap();
        doc = apply(&doc, &edit).unwrap();
        assert_eq!(
            doc.types(),
            vec![AtomicType::Str, AtomicType::Str, AtomicType::Del]
        );
    }
}
