use std::collections::{BTreeMap, BTreeSet};

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
    Bool,
    Ref(DefId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Unit,
    I64(i64),
    Bool(bool),
    Ref(ObjectId),
}

impl Value {
    pub fn shallow_type(&self, heap: &BTreeMap<ObjectId, Object>) -> Option<Type> {
        match self {
            Self::Unit => Some(Type::Unit),
            Self::I64(_) => Some(Type::I64),
            Self::Bool(_) => Some(Type::Bool),
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

/// The mutable-by-migration part of an object: its schema-versioned layout and
/// field values. A migration builds a *new* `Body` and swaps it in; the body
/// itself is never mutated in place. Held behind an [`Arc`] so a swap is a
/// cheap pointer replacement and readers keep the body they loaded alive — the
/// reclamation half of "moving bodies behind non-moving handles."
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Body {
    pub type_id: DefId,
    pub schema: Version,
    pub fields: BTreeMap<FieldId, Value>,
}

/// A heap object: a stable handle ([`ObjectId`]) plus its current [`Body`].
/// References name the handle; the body behind it can change version under a
/// migration without the handle moving. `Deref` exposes the body's fields
/// directly, so `object.schema` / `object.fields` read through to the body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Object {
    pub id: ObjectId,
    pub body: std::sync::Arc<Body>,
}

impl std::ops::Deref for Object {
    type Target = Body;
    fn deref(&self) -> &Body {
        &self.body
    }
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
    /// Copy a value from one register to another (any type). The one op that
    /// isn't produced by a single source construct — the frontend emits it to
    /// bind a local to another local's value and to land branch results in a
    /// common register (this IR has no phi nodes).
    Copy {
        dst: usize,
        src: usize,
    },
    AddI64 {
        dst: usize,
        left: usize,
        right: usize,
    },
    SubI64 {
        dst: usize,
        left: usize,
        right: usize,
    },
    /// `dst := left < right` (signed). Produces a `Bool`, the only condition
    /// source for `Branch`. Pure, so it never ends a basic block.
    LtI64 {
        dst: usize,
        left: usize,
        right: usize,
    },
    /// Unconditional transfer to another program counter. A back-edge (a target
    /// at or before this pc) forms a loop.
    Jump {
        target: usize,
    },
    /// Transfer to `then_pc` when `cond` is `true`, else `else_pc`.
    Branch {
        cond: usize,
        then_pc: usize,
        else_pc: usize,
    },
    /// A recurring safe point (DESIGN.md T5). The native `step` hands control
    /// back here so a pending hot update can land between iterations; the
    /// interpreter treats it as a no-op advance. It computes nothing, so both
    /// executors stay observationally identical.
    Yield,
    Call {
        dst: usize,
        function: DefId,
        args: Vec<usize>,
    },
    Emit {
        value: usize,
    },
    /// Send a value to another actor's mailbox. `target` holds the recipient
    /// actor id (an `I64`); `value` is any value (including a `Ref` into the
    /// shared heap). Concurrent tier only.
    Send {
        target: usize,
        value: usize,
    },
    /// Block until a message arrives in this actor's mailbox, then bind it to
    /// `dst`. The message is checked to have type `ty` — a mismatch traps like
    /// any other con-freeness violation, so message contracts stay sound.
    /// Concurrent tier only.
    Recv {
        dst: usize,
        ty: Type,
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
    /// The nominal types each function's current Ready version references (its
    /// dependency set, D7). A schema change re-verifies only the functions whose
    /// set contains the changed type instead of every current function.
    pub function_deps: BTreeMap<DefId, BTreeSet<DefId>>,
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
