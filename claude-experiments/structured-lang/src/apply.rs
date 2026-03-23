use crate::types::*;

/// Default value for a type, per the paper (p. 8):
/// "as we do in our implementation, assign a default value based on the type."
pub fn default_value(ty: AtomicType) -> Value {
    match ty {
        AtomicType::Num => Value::Num(0.0),
        AtomicType::Str => Value::Str(String::new()),
        AtomicType::Bool => Value::Bool(false),
        AtomicType::Del => Value::Null,
    }
}

/// Attempt to convert a value to conform to a new type.
/// Returns the converted value if possible, or Error if not.
///
/// For Null (uninitialized) values, assigns a default per the paper:
/// "The conform function can treat an uninitialized value as an error,
/// or as we do in our implementation, assign a default value based on
/// the type."
pub fn conform_value(value: &Value, from: AtomicType, to: AtomicType) -> Value {
    if from == to {
        return value.clone();
    }
    match (value, from, to) {
        // Del target: tombstoned, value is null
        (_, _, AtomicType::Del) => Value::Null,
        // Null (uninitialized): assign default for the target type
        (Value::Null, _, _) => default_value(to),
        // Error propagates
        (Value::Error, _, _) => Value::Error,

        // Num -> Str (lossless)
        (Value::Num(n), AtomicType::Num, AtomicType::Str) => Value::Str(n.to_string()),
        // Str -> Num (lossy, may fail)
        (Value::Str(s), AtomicType::Str, AtomicType::Num) => s
            .parse::<f64>()
            .map(Value::Num)
            .unwrap_or(Value::Error),
        // Bool -> Str (lossless)
        (Value::Bool(b), AtomicType::Bool, AtomicType::Str) => {
            Value::Str(b.to_string())
        }
        // Bool -> Num (lossless)
        (Value::Bool(b), AtomicType::Bool, AtomicType::Num) => {
            Value::Num(if *b { 1.0 } else { 0.0 })
        }
        // Num -> Bool (lossy)
        (Value::Num(n), AtomicType::Num, AtomicType::Bool) => Value::Bool(*n != 0.0),
        // Str -> Bool (lossy)
        (Value::Str(s), AtomicType::Str, AtomicType::Bool) => {
            Value::Bool(!s.is_empty() && s != "false")
        }

        _ => Value::Error,
    }
}

/// Apply an edit to a document, producing a new document.
/// Returns None if the edit is invalid for this document.
pub fn apply(doc: &Document, edit: &Edit) -> Option<Document> {
    match edit {
        Edit::Id => Some(doc.clone()),

        Edit::Ins { idx, ty, id: _ } => {
            if *idx > doc.len() {
                return None;
            }
            let mut fields = doc.fields.clone();
            fields.insert(*idx, Field::null(*ty));
            Some(Document::new(fields))
        }

        Edit::Conv { idx, ty } => {
            if *idx >= doc.len() {
                return None;
            }
            let mut fields = doc.fields.clone();
            // Per the paper (Section 2.3, p. 8): "The Conv edit just sets the
            // desired type, leaving the existing value in place." The value is
            // NOT converted here — that's the job of the separate `conform`
            // function. "By retaining the unconverted value we also allow
            // conversions to compose losslessly."
            fields[*idx] = Field::new(doc.fields[*idx].value.clone(), *ty);
            Some(Document::new(fields))
        }

        Edit::Move { i, j } => {
            if *i == *j || *i >= doc.len() || *j >= doc.len() {
                return None;
            }
            let mut fields = doc.fields.clone();
            fields[*i] = doc.fields[*j].clone();
            fields[*j] = Field::null(AtomicType::Del);
            Some(Document::new(fields))
        }

        Edit::Rename { idx, .. } => {
            // Rename is schema metadata — no effect on the document structure.
            // The edit is tracked for OT/merge purposes but apply is a no-op.
            if *idx >= doc.len() {
                return None;
            }
            Some(doc.clone())
        }

        Edit::Set { idx, value } => {
            if *idx >= doc.len() {
                return None;
            }
            let mut fields = doc.fields.clone();
            // Set changes value only, type is unchanged (orthogonal to Conv).
            fields[*idx] = Field::new(value.clone(), doc.fields[*idx].ty);
            Some(Document::new(fields))
        }
    }
}

/// Produce a type-conforming version of a document.
///
/// Per the paper (p. 8): "We define a function conform that takes a document
/// and produces a type-conforming version of it, in which all values either
/// conform to their type or are the special value error. Programs can only
/// see conforming values."
///
/// The raw document retains original values for lossless composition.
/// This function produces the view that programs/agents see.
pub fn conform(doc: &Document) -> Document {
    let fields = doc
        .fields
        .iter()
        .map(|field| {
            let conformed = conform_value(&field.value, value_type(&field.value), field.ty);
            Field::new(conformed, field.ty)
        })
        .collect();
    Document::new(fields)
}

/// Infer the "natural" type of a value (what type it was originally).
fn value_type(value: &Value) -> AtomicType {
    match value {
        Value::Num(_) => AtomicType::Num,
        Value::Str(_) => AtomicType::Str,
        Value::Bool(_) => AtomicType::Bool,
        Value::Null => AtomicType::Del, // Null has no natural type
        Value::Error => AtomicType::Del,
    }
}

/// Apply a sequence of edits to a document.
pub fn apply_seq(doc: &Document, edits: &[Edit]) -> Option<Document> {
    let mut current = doc.clone();
    for edit in edits {
        current = apply(&current, edit)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_id() {
        let doc = Document::from_types(&[AtomicType::Num, AtomicType::Str]);
        assert_eq!(apply(&doc, &Edit::Id), Some(doc));
    }

    #[test]
    fn test_apply_ins() {
        let doc = Document::from_types(&[AtomicType::Num]);
        let result = apply(
            &doc,
            &Edit::Ins {
                idx: 1,
                ty: AtomicType::Bool,
                id: 1,
            },
        )
        .unwrap();
        assert_eq!(result.types(), vec![AtomicType::Num, AtomicType::Bool]);
    }

    #[test]
    fn test_apply_conv() {
        let doc = Document::from_types(&[AtomicType::Num, AtomicType::Str]);
        let result = apply(&doc, &Edit::Conv { idx: 0, ty: AtomicType::Bool }).unwrap();
        assert_eq!(result.types(), vec![AtomicType::Bool, AtomicType::Str]);
    }

    #[test]
    fn test_apply_move() {
        let doc = Document::from_types(&[AtomicType::Num, AtomicType::Str, AtomicType::Bool]);
        let result = apply(&doc, &Edit::Move { i: 0, j: 2 }).unwrap();
        assert_eq!(
            result.types(),
            vec![AtomicType::Bool, AtomicType::Str, AtomicType::Del]
        );
    }
}
