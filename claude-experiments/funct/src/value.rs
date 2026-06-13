//! Runtime values. All non-Atom values are immutable with structural sharing
//! (shared-pointer clone-on-write). `Atom` is the only user-visible mutable
//! cell; `Cell` is the internal slot used for captured `let mut` locals.
//!
//! Values are Arc/Mutex-backed: `Value`, `Funct` and `VmState` are
//! `Send + Sync`, so engines and paused states can move across threads and
//! live in thread-safe containers. The flip side: registered natives must be
//! `Send + Sync` and host types `Send`. (Benchmarked: the atomics cost is
//! noise next to interpretation overhead.)

use std::collections::BTreeMap;
use std::fmt;

/// Shared-pointer and interior-mutability primitives. Everything else in the
/// crate uses these aliases — they are the single seam to change if a
/// non-thread-safe (Rc/RefCell) build is ever needed again.
pub mod shared {
    pub use std::sync::{Arc as Sh, Weak as ShWeak};

    use std::any::Any;

    /// Interior-mutable slot (a Mutex; poisoning is ignored — a panicking
    /// host is already broken, the value itself is still consistent).
    pub struct Lock<T>(std::sync::Mutex<T>);

    impl<T> Lock<T> {
        pub fn new(v: T) -> Self {
            Lock(std::sync::Mutex::new(v))
        }
        pub fn read(&self) -> impl std::ops::Deref<Target = T> + '_ {
            self.0.lock().unwrap_or_else(|e| e.into_inner())
        }
        pub fn write(&self) -> impl std::ops::DerefMut<Target = T> + '_ {
            self.0.lock().unwrap_or_else(|e| e.into_inner())
        }
    }

    /// Bound on data stored in `Value::Native` host objects.
    pub trait NativeBound: 'static + Send {}
    impl<T: 'static + Send> NativeBound for T {}

    /// Bound on host callbacks (registered fns, getters).
    pub trait HostBound: 'static + Send + Sync {}
    impl<T: 'static + Send + Sync> HostBound for T {}

    /// Shared type-erased cell holding a host (Rust) value.
    pub type AnyLock = Sh<std::sync::Mutex<dyn Any + Send>>;

    pub fn new_any<T: NativeBound>(v: T) -> AnyLock {
        Sh::new(std::sync::Mutex::new(v))
    }

    /// Run `f` with mutable access to the type-erased host value.
    pub fn with_any_mut<R>(a: &AnyLock, f: impl FnOnce(&mut dyn Any) -> R) -> R {
        f(&mut *a.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub fn any_ptr(a: &AnyLock) -> usize {
        Sh::as_ptr(a) as *const () as usize
    }

    pub fn any_ptr_eq(a: &AnyLock, b: &AnyLock) -> bool {
        any_ptr(a) == any_ptr(b)
    }
}

use shared::{AnyLock, Lock, NativeBound, Sh};

/// Persistent (structurally-shared) sequence: O(1) clone, ~O(log n) push/index,
/// amortized O(1) push when uniquely owned. Backs both lists and tuples.
pub type FList = imbl::Vector<Value>;
/// Persistent sorted map (structural sharing) backing records; keeps keys
/// ordered, so `keys()`/iteration stay deterministic.
pub type FMap = imbl::OrdMap<String, Value>;

#[derive(Clone)]
pub enum Value {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Sh<str>),
    // List/Tuple hold the persistent vector behind an `Sh` (Arc) so the `Value`
    // enum stays small (~32 B not ~72 B): a bare `imbl::Vector` is 64 bytes
    // inline, and `Value` is moved/cloned/dropped on every stack operation.
    List(Sh<FList>),
    Tuple(Sh<FList>),
    Record(FMap),
    Variant(Sh<Variant>),
    Closure(Sh<Closure>),
    /// Index into the engine's native function table.
    NativeFn(u32),
    Atom(Sh<AtomCell>),
    /// Internal: shared slot for a captured `let mut` local. Never user-visible.
    Cell(Sh<Lock<Value>>),
    /// Opaque host value registered via interop.
    Native(NativeValue),
    /// start, end, inclusive
    Range(i64, i64, bool),
}

pub struct Variant {
    pub tag: String,
    pub payload: VariantPayload,
}

#[derive(Clone)]
pub enum VariantPayload {
    Unit,
    Positional(Vec<Value>),
    Named(BTreeMap<String, Value>),
}

pub struct Closure {
    pub fn_id: u32,
    pub upvals: Vec<Value>,
}

pub struct AtomCell {
    /// Stable id, used for serialization identity.
    pub id: u64,
    pub value: Lock<Value>,
    /// (key, watcher-fn) pairs, fired after swap!/reset!.
    pub watchers: Lock<Vec<(String, Value)>>,
}

#[derive(Clone)]
pub struct NativeValue {
    pub type_name: Sh<str>,
    pub data: AnyLock,
}

impl NativeValue {
    /// Run `f` with a shared borrow of the host value, if it is a `T`.
    pub fn with_ref<T: 'static, R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        shared::with_any_mut(&self.data, |any| any.downcast_ref::<T>().map(f))
    }

    /// Run `f` with a mutable borrow of the host value, if it is a `T`.
    pub fn with_mut<T: 'static, R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        shared::with_any_mut(&self.data, |any| any.downcast_mut::<T>().map(f))
    }
}

impl Value {
    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(Sh::from(s.into().as_str()))
    }

    pub fn list(items: Vec<Value>) -> Value {
        Value::List(Sh::new(items.into_iter().collect()))
    }

    pub fn tuple(items: Vec<Value>) -> Value {
        Value::Tuple(Sh::new(items.into_iter().collect()))
    }

    /// Build a list directly from an already-constructed persistent vector.
    pub fn list_v(items: FList) -> Value {
        Value::List(Sh::new(items))
    }

    pub fn record(fields: BTreeMap<String, Value>) -> Value {
        Value::Record(fields.into_iter().collect())
    }

    pub fn variant(tag: &str, payload: VariantPayload) -> Value {
        Value::Variant(Sh::new(Variant {
            tag: tag.to_string(),
            payload,
        }))
    }

    pub fn ok(v: Value) -> Value {
        Value::variant("Ok", VariantPayload::Positional(vec![v]))
    }

    pub fn err(v: Value) -> Value {
        Value::variant("Err", VariantPayload::Positional(vec![v]))
    }

    pub fn some(v: Value) -> Value {
        Value::variant("Some", VariantPayload::Positional(vec![v]))
    }

    pub fn none() -> Value {
        Value::variant("None", VariantPayload::Unit)
    }

    pub fn native<T: NativeBound>(type_name: &str, v: T) -> Value {
        Value::Native(NativeValue {
            type_name: Sh::from(type_name),
            data: shared::new_any(v),
        })
    }

    pub fn type_name(&self) -> String {
        match self {
            Value::Unit => "Unit".into(),
            Value::Bool(_) => "Bool".into(),
            Value::Int(_) => "Int".into(),
            Value::Float(_) => "Float".into(),
            Value::Str(_) => "Str".into(),
            Value::List(_) => "List".into(),
            Value::Tuple(_) => "Tuple".into(),
            Value::Record(_) => "Record".into(),
            Value::Variant(v) => format!("Variant({})", v.tag),
            Value::Closure(_) => "Fn".into(),
            Value::NativeFn(_) => "NativeFn".into(),
            Value::Atom(_) => "Atom".into(),
            Value::Cell(_) => "Cell".into(),
            Value::Native(n) => n.type_name.to_string(),
            Value::Range(..) => "Range".into(),
        }
    }

    pub fn is_truthy_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;
        match (self, other) {
            (Unit, Unit) => true,
            (Bool(a), Bool(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (Float(a), Float(b)) => a == b,
            (Int(a), Float(b)) | (Float(b), Int(a)) => (*a as f64) == *b,
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) | (Tuple(a), Tuple(b)) => a == b,
            (Record(a), Record(b)) => a == b,
            (Variant(a), Variant(b)) => {
                a.tag == b.tag
                    && match (&a.payload, &b.payload) {
                        (VariantPayload::Unit, VariantPayload::Unit) => true,
                        (VariantPayload::Positional(x), VariantPayload::Positional(y)) => x == y,
                        (VariantPayload::Named(x), VariantPayload::Named(y)) => x == y,
                        _ => false,
                    }
            }
            (Closure(a), Closure(b)) => Sh::ptr_eq(a, b),
            (NativeFn(a), NativeFn(b)) => a == b,
            (Atom(a), Atom(b)) => Sh::ptr_eq(a, b),
            (Cell(a), Cell(b)) => Sh::ptr_eq(a, b),
            (Native(a), Native(b)) => shared::any_ptr_eq(&a.data, &b.data),
            (Range(a, b, c), Range(d, e, f)) => a == d && b == e && c == f,
            _ => false,
        }
    }
}

/// Display = how values print in string interpolation / `str()`.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(x) => {
                if x.fract() == 0.0 && x.is_finite() {
                    write!(f, "{:.1}", x)
                } else {
                    write!(f, "{}", x)
                }
            }
            Value::Str(s) => write!(f, "{}", s),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", DebugStr(v))?;
                }
                write!(f, "]")
            }
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", DebugStr(v))?;
                }
                write!(f, ")")
            }
            Value::Record(fields) => {
                write!(f, "{{ ")?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, DebugStr(v))?;
                }
                write!(f, " }}")
            }
            Value::Variant(v) => match &v.payload {
                VariantPayload::Unit => write!(f, "{}", v.tag),
                VariantPayload::Positional(items) => {
                    write!(f, "{}(", v.tag)?;
                    for (i, x) in items.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", DebugStr(x))?;
                    }
                    write!(f, ")")
                }
                VariantPayload::Named(fields) => {
                    write!(f, "{} {{ ", v.tag)?;
                    for (i, (k, x)) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", k, DebugStr(x))?;
                    }
                    write!(f, " }}")
                }
            },
            Value::Closure(c) => write!(f, "<fn #{}>", c.fn_id),
            Value::NativeFn(id) => write!(f, "<native fn #{}>", id),
            Value::Atom(a) => write!(f, "atom({})", DebugStr(&a.value.read())),
            Value::Cell(c) => write!(f, "{}", &*c.read()),
            Value::Native(n) => write!(f, "<{}>", n.type_name),
            Value::Range(a, b, inc) => write!(f, "{}{}{}", a, if *inc { "..=" } else { ".." }, b),
        }
    }
}

/// Like Display, but quotes strings (used for nesting inside containers).
struct DebugStr<'a>(&'a Value);
impl fmt::Display for DebugStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Value::Str(s) => write!(f, "\"{}\"", s),
            v => write!(f, "{}", v),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", DebugStr(self))
    }
}
