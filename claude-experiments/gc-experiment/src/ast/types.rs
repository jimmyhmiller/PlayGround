/// Types in our simple language
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// Void type (for functions that don't return a value)
    Void,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// Boolean
    Bool,
    /// A GC-managed pointer to a struct
    Struct(String),
    /// A GC-managed array of a given element type
    Array(Box<Type>),
}

impl Type {
    /// Check if this type is a GC-managed reference
    pub fn is_gc_ref(&self) -> bool {
        matches!(self, Type::Struct(_) | Type::Array(_))
    }

    /// Check if this type is a primitive (not GC-managed)
    pub fn is_primitive(&self) -> bool {
        matches!(self, Type::Void | Type::I32 | Type::I64 | Type::Bool)
    }
}

/// A field in a struct definition
#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub typ: Type,
}

/// A struct type definition
#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<StructField>,
}

impl StructDef {
    pub fn new(name: impl Into<String>, fields: Vec<StructField>) -> Self {
        Self {
            name: name.into(),
            fields,
        }
    }

    /// Get indices of fields that are GC references (need scanning)
    pub fn gc_field_indices(&self) -> Vec<usize> {
        self.fields
            .iter()
            .enumerate()
            .filter(|(_, f)| f.typ.is_gc_ref())
            .map(|(i, _)| i)
            .collect()
    }

    /// Find a field by name and return its index
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }
}

/// A function parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub typ: Type,
}

impl Parameter {
    pub fn new(name: impl Into<String>, typ: Type) -> Self {
        Self {
            name: name.into(),
            typ,
        }
    }
}

/// A function signature
#[derive(Debug, Clone)]
pub struct FunctionSig {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Type,
}
