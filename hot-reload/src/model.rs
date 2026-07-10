use std::collections::BTreeMap;

pub type DefId = u64;
pub type FieldId = u64;
pub type ObjectId = u64;
pub type ActorId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Unit,
    I64,
    Ref(DefId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Unit,
    I64(i64),
    Ref(ObjectId),
}

impl Value {
    pub fn shallow_type(&self, heap: &BTreeMap<ObjectId, Object>) -> Option<Type> {
        match self {
            Self::Unit => Some(Type::Unit),
            Self::I64(_) => Some(Type::I64),
            Self::Ref(id) => Some(Type::Ref(heap.get(id)?.type_id)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub id: FieldId,
    pub name: String,
    pub ty: Type,
    pub default: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schema {
    pub type_id: DefId,
    pub version: Version,
    pub name: String,
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn field(&self, id: FieldId) -> Option<&Field> {
        self.fields.iter().find(|field| field.id == id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Object {
    pub id: ObjectId,
    pub type_id: DefId,
    pub schema: Version,
    pub fields: BTreeMap<FieldId, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MigrationSource {
    Copy(FieldId),
    Value(Value),
    /// Construct a new object around a migrated scalar/reference. This remains
    /// declarative, so the runtime can validate and stage it before publication.
    Wrap {
        type_id: DefId,
        field: FieldId,
        source: FieldId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Migration {
    pub type_id: DefId,
    pub from: Version,
    pub to: Version,
    pub fields: BTreeMap<FieldId, MigrationSource>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instruction {
    Const {
        dst: usize,
        value: Value,
    },
    New {
        dst: usize,
        type_id: DefId,
        fields: Vec<(FieldId, usize)>,
    },
    GetField {
        dst: usize,
        object: usize,
        field: FieldId,
    },
    SubI64 {
        dst: usize,
        left: usize,
        right: usize,
    },
    Call {
        dst: usize,
        function: DefId,
        args: Vec<usize>,
    },
    Emit {
        value: usize,
    },
    Return {
        value: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub id: DefId,
    pub version: Version,
    pub name: String,
    pub params: Vec<Type>,
    pub result: Type,
    pub registers: usize,
    pub code: Vec<Instruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionState {
    Ready(Function),
    Broken {
        id: DefId,
        version: Version,
        name: String,
        diagnostics: Vec<String>,
    },
}

impl FunctionState {
    pub fn id(&self) -> DefId {
        match self {
            Self::Ready(f) => f.id,
            Self::Broken { id, .. } => *id,
        }
    }
    pub fn version(&self) -> Version {
        match self {
            Self::Ready(f) => f.version,
            Self::Broken { version, .. } => *version,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct World {
    pub epoch: u64,
    pub schemas: BTreeMap<(DefId, Version), Schema>,
    pub current_schemas: BTreeMap<DefId, Version>,
    pub functions: BTreeMap<(DefId, Version), FunctionState>,
    pub current_functions: BTreeMap<DefId, Version>,
    pub migrations: BTreeMap<(DefId, Version), Migration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReturnTo {
    pub register: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub function: (DefId, Version),
    pub pc: usize,
    pub registers: Vec<Option<Value>>,
    pub return_to: Option<ReturnTo>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Actor {
    pub id: ActorId,
    pub frames: Vec<Frame>,
    pub status: ActorStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActorStatus {
    Runnable,
    Paused(Condition),
    Complete(Value),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Condition {
    BrokenFunction {
        function: DefId,
        diagnostics: Vec<String>,
    },
    MissingMigration {
        object: ObjectId,
        type_id: DefId,
        from: Version,
        to: Version,
    },
    RuntimeTypeError {
        function: DefId,
        pc: usize,
        message: String,
    },
}
