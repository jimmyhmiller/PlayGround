use std::collections::{BTreeMap, BTreeSet};

pub type DefId = u64;
pub type FieldId = u64;
pub type ObjectId = u64;
pub type ActorId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(pub u64);

/// A foreign resource kind — the nominal identity of a `foreign type`
/// (a `Window`, a socket, a GL context). It is opaque: the runtime never
/// inspects its layout, so it has no schema version and can never go stale,
/// which is exactly why a foreign handle never traps at a use-boundary.
pub type ForeignKind = u32;
/// The id of a `foreign fn` — a native function registered on the [`Runtime`].
pub type ForeignFnId = u32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Unit,
    I64,
    Bool,
    Ref(DefId),
    /// An opaque handle to a native resource. Matched nominally by kind; the GC
    /// never traces through it (native code owns the pointee's lifetime).
    Foreign(ForeignKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Unit,
    I64(i64),
    Bool(bool),
    Ref(ObjectId),
    /// A native pointer behind a kind tag. Passed to and returned from foreign
    /// calls; stored in globals and object fields like any other value. Opaque
    /// to the GC — `ptr` is never dereferenced or traced by the runtime.
    Foreign { kind: ForeignKind, ptr: u64 },
}

impl Value {
    /// The nominal type of a *scalar* value — everything except a reference,
    /// whose type depends on its object's current schema and is resolved by the
    /// [`crate::Heap`]. Used where no heap is in scope (e.g. checking a code
    /// constant, which is never a reference).
    pub fn scalar_type(&self) -> Option<Type> {
        match self {
            Self::Unit => Some(Type::Unit),
            Self::I64(_) => Some(Type::I64),
            Self::Bool(_) => Some(Type::Bool),
            Self::Foreign { kind, .. } => Some(Type::Foreign(*kind)),
            Self::Ref(_) => None,
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
    MulI64 {
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
    /// `dst := left == right` (on i64). Produces a `Bool`.
    EqI64 {
        dst: usize,
        left: usize,
        right: usize,
    },
    /// `dst := !src` (on a Bool). Produces a `Bool`; lets the frontend build
    /// `!=`, `<=`, `>=`, and unary `!` from `<`/`==`.
    Not {
        dst: usize,
        src: usize,
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
    /// Call a registered native function (managed → native). Atomic and
    /// uninterruptible from the runtime's view: it runs to completion with no
    /// safepoint, GC, trap, or hot-update landing inside it. All reference
    /// arguments are pinned in frame slots across the call (non-moving GC), so
    /// a raw pointer handed to native code stays valid for its duration.
    CallForeign {
        dst: usize,
        foreign: ForeignFnId,
        args: Vec<usize>,
    },
    /// Read a top-level `letonce` binding. Globals are persistent runtime state
    /// that survives hot edits — where native resources (a window, a context)
    /// live so a reload changes code, not the running world.
    LoadGlobal {
        dst: usize,
        global: DefId,
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
    /// Declared signatures of `foreign fn`s (the ABI contract with native code).
    /// The verifier checks `CallForeign` arguments against these; the native
    /// implementations themselves live on the [`Runtime`], not in the `World`.
    pub foreign_sigs: BTreeMap<ForeignFnId, (Vec<Type>, Type)>,
    /// Declared types of top-level `letonce` globals, so a `LoadGlobal` types.
    pub global_types: BTreeMap<DefId, Type>,
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
