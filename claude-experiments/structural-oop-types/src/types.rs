//! Abstract type representation (like OCaml's `tree` type)
//!
//! These types are used for display and as input to the type checker.
//! During unification, we use the Node representation instead.

/// A type in the system
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// Type variable with name and unique id
    Var(String, u32),
    /// Base/constant type (int, bool, or user-defined)
    Const(String),
    /// Function type: domain -> codomain
    Arrow(Box<Type>, Box<Type>),
    /// Record/object type containing a row
    Record(Box<Row>),
    /// Recursive type μα.τ (used for display of equi-recursive types)
    Mu(String, Box<Type>),
}

/// A row type for records
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Row {
    /// Empty row (closed record)
    Empty,
    /// Row extension: { field: presence | rest }
    Extend {
        field: String,
        presence: FieldPresence,
        rest: Box<Row>,
    },
    /// Row variable (open record)
    Var(String, u32),
}

/// Whether a field is present or absent in a row
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldPresence {
    /// Field is present with the given type
    Present(Type),
    /// Field is explicitly absent
    Absent,
}

impl Type {
    /// Create a type variable
    pub fn var(name: impl Into<String>, id: u32) -> Self {
        Type::Var(name.into(), id)
    }

    /// Create a constant/base type
    pub fn constant(name: impl Into<String>) -> Self {
        Type::Const(name.into())
    }

    /// Create a function type
    pub fn arrow(domain: Type, codomain: Type) -> Self {
        Type::Arrow(Box::new(domain), Box::new(codomain))
    }

    /// Create a record type
    pub fn record(row: Row) -> Self {
        Type::Record(Box::new(row))
    }

    /// Create a recursive type
    pub fn mu(var: impl Into<String>, body: Type) -> Self {
        Type::Mu(var.into(), Box::new(body))
    }

    /// Convenience: bool type
    pub fn bool() -> Self {
        Type::Const("bool".into())
    }

    /// Convenience: int type
    pub fn int() -> Self {
        Type::Const("int".into())
    }
}

impl Row {
    /// Create an empty row
    pub fn empty() -> Self {
        Row::Empty
    }

    /// Create a row variable
    pub fn var(name: impl Into<String>, id: u32) -> Self {
        Row::Var(name.into(), id)
    }

    /// Extend a row with a present field
    pub fn extend(field: impl Into<String>, ty: Type, rest: Row) -> Self {
        Row::Extend {
            field: field.into(),
            presence: FieldPresence::Present(ty),
            rest: Box::new(rest),
        }
    }

    /// Extend a row with an absent field
    pub fn extend_absent(field: impl Into<String>, rest: Row) -> Self {
        Row::Extend {
            field: field.into(),
            presence: FieldPresence::Absent,
            rest: Box::new(rest),
        }
    }

    /// Build a closed record row from a list of (field, type) pairs
    pub fn from_fields(fields: impl IntoIterator<Item = (String, Type)>) -> Self {
        let fields: Vec<_> = fields.into_iter().collect();
        let mut row = Row::Empty;
        for (field, ty) in fields.into_iter().rev() {
            row = Row::extend(field, ty, row);
        }
        row
    }
}
