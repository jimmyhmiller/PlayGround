//! Hash-cached typechecker.
//!
//! This is a CONTENT-ADDRESSED language. A `Def`'s hash IS its identity,
//! and once a hash has been type-checked the result is known forever.
//! Re-checking a module hits zero cache misses; adding one def to an
//! n-def module costs exactly one new typecheck.
//!
//! The typechecker is therefore not a whole-program pass; it's a function
//! `Def -> TypeScheme` whose result is memoised by hash in `TypeCache`.
//!
//! v1 caveats:
//!
//! - No generics yet (`Type::TypeVar` always typechecks as non-Wire).
//! - All v1 builtin types are Wire (Int / Bool / String / Float / Bytes).
//! - `Wire` checking on `core/net.at` is reserved but not yet enforced
//!   (every program in v1 is Wire by construction).
//! - The typechecker does not yet gate compilation; it is purely additive
//!   for now.

use crate::ast::{Def, Expr, Pattern, Type};
use crate::hash::Hash;
use crate::resolve::{ResolvedDef, ResolvedModule};

use std::collections::HashMap;

// =============================================================================
// TypeScheme — the cached fact about a hash
// =============================================================================

/// A typed view of a `Def`, indexed by its hash in the `TypeCache`.
///
/// `wire` indicates whether values of this type may cross a `core/net.at`
/// boundary: all v1 builtins are Wire, structs / enums are Wire if all
/// their components are Wire, fn types are Wire if all params and the
/// return type are Wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeScheme {
    Fn {
        params: Vec<Type>,
        ret: Type,
        wire: bool,
    },
    Struct {
        /// Declared generic arity. Stored explicitly (rather than
        /// derived from the fields) so a phantom type param — one that
        /// appears in no field, like `Atom<T>` — is still reported as
        /// arity 1 by `scheme_type_params`.
        type_params: u32,
        fields: Vec<(String, Type)>,
        wire: bool,
    },
    Enum {
        type_params: u32,
        variants: Vec<(String, Option<Type>)>,
        wire: bool,
    },
    /// A node-resident `state` binding. `ty` is its declared type
    /// (typically `Atom<T>`). A `StateRef(hash)` has type `ty`.
    State {
        ty: Type,
    },
}

impl TypeScheme {
    pub fn as_fn(&self) -> Option<(&[Type], &Type)> {
        match self {
            TypeScheme::Fn { params, ret, .. } => Some((params.as_slice(), ret)),
            TypeScheme::Struct { .. } | TypeScheme::Enum { .. } | TypeScheme::State { .. } => None,
        }
    }

    pub fn as_struct(&self) -> Option<&[(String, Type)]> {
        match self {
            TypeScheme::Struct { fields, .. } => Some(fields.as_slice()),
            TypeScheme::Fn { .. } | TypeScheme::Enum { .. } | TypeScheme::State { .. } => None,
        }
    }

    pub fn as_enum(&self) -> Option<&[(String, Option<Type>)]> {
        match self {
            TypeScheme::Enum { variants, .. } => Some(variants.as_slice()),
            TypeScheme::Fn { .. } | TypeScheme::Struct { .. } | TypeScheme::State { .. } => None,
        }
    }

    pub fn is_wire(&self) -> bool {
        match self {
            TypeScheme::Fn { wire, .. }
            | TypeScheme::Struct { wire, .. }
            | TypeScheme::Enum { wire, .. } => *wire,
            // A node-resident state cell never crosses the wire by value.
            TypeScheme::State { .. } => false,
        }
    }
}

// =============================================================================
// TypeCache — the whole point
// =============================================================================

/// Memoisation table from content hash to type scheme.
///
/// `typecheck_module` consults this cache before re-checking any def;
/// an already-typed hash is never re-checked. This is what makes the
/// typechecker scale: cost is proportional to *new* code, not to total
/// code in the codebase.
#[derive(Debug, Clone, Default)]
pub struct TypeCache {
    schemes: HashMap<Hash, TypeScheme>,
    /// `extern fn` signatures registered for this typecheck session,
    /// keyed by surface name (e.g. `"print_int"`). Calls to
    /// `BuiltinRef("ext/<name>")` look here. Externs are NOT
    /// content-addressed; the cache holds the live set declared by
    /// the module being typechecked.
    /// Value is `(fixed_params, ret, variadic)`. A variadic extern
    /// (e.g. `curl_easy_setopt`) accepts the fixed params followed by
    /// any number of trailing C-scalar (Int/Ptr) args.
    externs: HashMap<String, (Vec<Type>, Type, bool)>,
    /// The declared return type of the function whose body is currently
    /// being checked. Set by `typecheck_def` before walking a fn body so
    /// the `?` operator can verify the enclosing function returns a
    /// `Result<_, E>` with the same error type `E`. Interior mutability
    /// keeps the rest of the `&TypeCache` API immutable.
    current_fn_ret: std::cell::RefCell<Option<Type>>,
    /// Hashes of defs that forward a thunk straight to `core/thread.spawn`
    /// (the stdlib `spawn` wrapper, and any alias of it). Calls to these
    /// are spawn sites whose thunk must be checked for mobility — see
    /// `typecheck_call`. Populated by `register_spawn_wrappers`.
    spawn_wrappers: std::collections::HashSet<Hash>,
}

impl TypeCache {
    pub fn new() -> Self {
        Self {
            schemes: HashMap::new(),
            externs: HashMap::new(),
            current_fn_ret: std::cell::RefCell::new(None),
            spawn_wrappers: std::collections::HashSet::new(),
        }
    }

    /// Record which defs forward to `core/thread.spawn` (spawn wrappers),
    /// so `typecheck_call` can require a mobile thunk at their call sites.
    pub fn register_spawn_wrappers(&mut self, hs: std::collections::HashSet<Hash>) {
        self.spawn_wrappers = hs;
    }

    fn is_spawn_wrapper(&self, h: &Hash) -> bool {
        self.spawn_wrappers.contains(h)
    }

    pub fn get(&self, h: &Hash) -> Option<&TypeScheme> {
        self.schemes.get(h)
    }

    pub fn insert(&mut self, h: Hash, scheme: TypeScheme) {
        self.schemes.insert(h, scheme);
    }

    pub fn contains(&self, h: &Hash) -> bool {
        self.schemes.contains_key(h)
    }

    pub fn len(&self) -> usize {
        self.schemes.len()
    }

    /// Iterate (hash, scheme) pairs. Used by the disk-backed cache to
    /// flush newly-typed entries.
    pub fn iter(&self) -> impl Iterator<Item = (&Hash, &TypeScheme)> {
        self.schemes.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.schemes.is_empty()
    }

    /// Register the externs the resolver discovered. Must be called
    /// before `typecheck_module` so bodies that call externs typecheck
    /// correctly.
    pub fn register_externs(
        &mut self,
        externs: &HashMap<String, crate::resolve::ExternSig>,
    ) {
        for (name, sig) in externs {
            self.externs.insert(
                name.clone(),
                (sig.params.clone(), sig.ret.clone(), sig.variadic),
            );
        }
    }

    pub fn extern_signature(&self, name: &str) -> Option<&(Vec<Type>, Type, bool)> {
        self.externs.get(name)
    }
}

// =============================================================================
// Errors — exhaustive, no Other(String) fallback
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    /// A `TopRef(h)` references a hash that hasn't been typed yet.
    UnknownTopRef(Hash),
    /// A struct hash referenced by `StructNew`/`Field` isn't a struct in cache.
    UnknownStruct(Hash),
    /// An enum hash referenced by `EnumNew`/`Match` isn't an enum in cache.
    UnknownEnum(Hash),
    /// `BuiltinRef(name)` doesn't match any known builtin signature.
    UnknownBuiltin(String),
    /// Callee in a `Call` expression isn't a function type.
    ExpectedFn { got: Type },
    /// Number of args / fields / variants didn't match the expected count.
    ArityMismatch {
        what: String,
        expected: usize,
        got: usize,
    },
    /// Encountered an expression of the wrong type.
    TypeMismatch {
        context: String,
        expected: Type,
        got: Type,
    },
    /// `Field` access on a non-struct value.
    ExpectedStruct { got: Type },
    /// `Match` scrutinee that isn't an enum value.
    ExpectedEnum { got: Type },
    /// `Field.index` past the end of the struct's fields.
    FieldIndexOutOfRange {
        struct_ref: Hash,
        index: u32,
        field_count: usize,
    },
    /// Variant index out of range for the enum.
    VariantIndexOutOfRange {
        enum_ref: Hash,
        index: u32,
        variant_count: usize,
    },
    /// `Match` arms produced different result types.
    MatchArmsDisagree { first: Type, found: Type },
    /// Reserved for future `at()` checking; v1 is always Wire so this won't fire.
    NotWire { context: String, ty: Type },
    /// `LocalVar(i)` out of range for the current lexical environment.
    LocalVarOutOfRange { index: u32, env_size: usize },
    /// `SelfRef` shouldn't appear after resolution — it's a stored-only artefact.
    SelfRefInTypecheck { component_index: u32 },
    /// Encountered a construct the typechecker doesn't yet support.
    Unsupported(String),
    /// A variant pattern said "nullary" but the variant has a payload (or vice versa).
    VariantPayloadShapeMismatch {
        enum_ref: Hash,
        variant_index: u32,
        pattern_has_payload: bool,
        variant_has_payload: bool,
    },
    /// An `EnumNew` payload didn't match the variant's declared shape.
    EnumNewPayloadShapeMismatch {
        enum_ref: Hash,
        variant_index: u32,
        new_has_payload: bool,
        variant_has_payload: bool,
    },
    /// A `Type::TypeRef(h)` used in a position where `h` must already be in cache.
    UnknownTypeRef(Hash),
}

impl core::fmt::Display for TypeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TypeError::UnknownTopRef(h) => write!(f, "unknown top-level ref {}", h),
            TypeError::UnknownStruct(h) => write!(f, "unknown struct {}", h),
            TypeError::UnknownEnum(h) => write!(f, "unknown enum {}", h),
            TypeError::UnknownBuiltin(name) => write!(f, "unknown builtin `{}`", name),
            TypeError::ExpectedFn { got } => {
                write!(f, "expected function type, got {:?}", got)
            }
            TypeError::ArityMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "arity mismatch in {}: expected {} but got {}",
                what, expected, got
            ),
            TypeError::TypeMismatch {
                context,
                expected,
                got,
            } => write!(
                f,
                "type mismatch in {}: expected {:?}, got {:?}",
                context, expected, got
            ),
            TypeError::ExpectedStruct { got } => {
                write!(f, "expected struct value, got {:?}", got)
            }
            TypeError::ExpectedEnum { got } => {
                write!(f, "expected enum value, got {:?}", got)
            }
            TypeError::FieldIndexOutOfRange {
                struct_ref,
                index,
                field_count,
            } => write!(
                f,
                "field index {} out of range (struct {} has {} fields)",
                index, struct_ref, field_count
            ),
            TypeError::VariantIndexOutOfRange {
                enum_ref,
                index,
                variant_count,
            } => write!(
                f,
                "variant index {} out of range (enum {} has {} variants)",
                index, enum_ref, variant_count
            ),
            TypeError::MatchArmsDisagree { first, found } => write!(
                f,
                "match arms produced different types: first arm {:?}, later arm {:?}",
                first, found
            ),
            TypeError::NotWire { context, ty } => write!(
                f,
                "type {:?} is not Wire (required in {})",
                ty, context
            ),
            TypeError::LocalVarOutOfRange { index, env_size } => write!(
                f,
                "local var index {} out of range (environment has {} bindings)",
                index, env_size
            ),
            TypeError::SelfRefInTypecheck { component_index } => write!(
                f,
                "SelfRef({}) reached the typechecker; resolver should have replaced it",
                component_index
            ),
            TypeError::Unsupported(msg) => {
                write!(f, "typechecker does not yet support: {}", msg)
            }
            TypeError::VariantPayloadShapeMismatch {
                enum_ref,
                variant_index,
                pattern_has_payload,
                variant_has_payload,
            } => write!(
                f,
                "variant {}/{} payload shape mismatch (pattern has payload: {}, variant has payload: {})",
                enum_ref, variant_index, pattern_has_payload, variant_has_payload
            ),
            TypeError::EnumNewPayloadShapeMismatch {
                enum_ref,
                variant_index,
                new_has_payload,
                variant_has_payload,
            } => write!(
                f,
                "EnumNew on variant {}/{} payload shape mismatch (constructor supplied: {}, variant declares: {})",
                enum_ref, variant_index, new_has_payload, variant_has_payload
            ),
            TypeError::UnknownTypeRef(h) => write!(f, "unknown type ref {}", h),
        }
    }
}

impl std::error::Error for TypeError {}

// =============================================================================
// Builtins
// =============================================================================

fn int_t() -> Type {
    Type::Builtin("Int".to_owned())
}

fn bool_t() -> Type {
    Type::Builtin("Bool".to_owned())
}

fn float_t() -> Type {
    Type::Builtin("Float".to_owned())
}

/// `Ptr` — a raw, i64-represented machine address (non-GC). The scalar
/// the C FFI passes and returns for all pointer-typed C arguments.
fn ptr_t() -> Type {
    Type::Builtin("Ptr".to_owned())
}

/// `Array<elem>` as `Apply(Builtin("Array"), [elem])`.
fn array_t(elem: Type) -> Type {
    Type::Apply(Box::new(Type::Builtin("Array".to_owned())), vec![elem])
}

/// `Atom<elem>` as `Apply(Builtin("Atom"), [elem])` — the dedicated
/// single-cell mutable shape, distinct from `Array`.
fn atom_t(elem: Type) -> Type {
    Type::Apply(Box::new(Type::Builtin("Atom".to_owned())), vec![elem])
}

/// `ThreadHandle<elem>` as `Apply(Builtin("ThreadHandle"), [elem])` — the
/// handle returned by `spawn` and consumed by `join`.
fn thread_handle_t(elem: Type) -> Type {
    Type::Apply(
        Box::new(Type::Builtin("ThreadHandle".to_owned())),
        vec![elem],
    )
}

/// Does `ty` contain a `Ptr` anywhere in its structure? A `Ptr` is a raw
/// local machine address; it is meaningless in any other process, so it
/// must never cross the `at(...)` wire boundary (as a thunk return type
/// or as a captured value). This walks Apply/FnType so an enclosing
/// type like `Array<Ptr>` or `fn() -> Ptr` is caught too.
fn type_contains_ptr(ty: &Type) -> bool {
    match ty {
        Type::Builtin(n) => n == "Ptr",
        Type::TypeRef(_) | Type::TypeVar(_) | Type::SelfRef(_) => false,
        Type::FnType { params, ret } => {
            type_contains_ptr(ret) || params.iter().any(type_contains_ptr)
        }
        Type::Apply(head, args) => {
            type_contains_ptr(head) || args.iter().any(type_contains_ptr)
        }
    }
}

/// Does `ty` contain an `Atom<_>` anywhere in its structure? An `Atom` is a
/// node-resident mutable cell with node identity; shipping one by value
/// across `at(...)` would silently fork it (the remote would mutate a dead
/// copy). Shared mutable state belongs to a `state` binding, reached by
/// reference on the owning node, never captured or returned by value. This
/// walks `Apply`/`FnType` so `Array<Atom<Int>>` or `fn() -> Atom<Int>` is
/// caught too.
fn type_contains_atom(ty: &Type) -> bool {
    match ty {
        Type::Builtin(_) | Type::TypeRef(_) | Type::TypeVar(_) | Type::SelfRef(_) => false,
        Type::FnType { params, ret } => {
            type_contains_atom(ret) || params.iter().any(type_contains_atom)
        }
        Type::Apply(head, args) => {
            matches!(head.as_ref(), Type::Builtin(n) if n == "Atom")
                || type_contains_atom(head)
                || args.iter().any(type_contains_atom)
        }
    }
}

/// Number of binders a pattern introduces (mirror of codegen's
/// `count_pattern_vars`). Used to track de Bruijn depth when walking a
/// thunk body for captured-variable indices.
fn pattern_binder_count(p: &Pattern) -> u32 {
    match p {
        Pattern::Wildcard => 0,
        Pattern::Var => 1,
        Pattern::Enum { payload, .. } => {
            payload.as_deref().map(pattern_binder_count).unwrap_or(0)
        }
    }
}

/// Collect the *outer* de Bruijn indices that `e` captures, given that
/// `depth` binders have been introduced inside the capturing lambda so
/// far. A `LocalVar(i)` with `i >= depth` references the enclosing
/// environment at outer index `i - depth`.
/// Mobility check for a `spawn`ed thunk: it may not return or capture an
/// `Atom` or `Ptr`, so a thread shares no mutable state with its parent.
/// (Same constraint `at` enforces for remote thunks; threads are just
/// in-process "nodes" under the share-nothing model.)
fn check_spawn_thunk_mobile(
    thunk: &Expr,
    thunk_ret: &Type,
    locals: &[Type],
) -> Result<(), TypeError> {
    if type_contains_ptr(thunk_ret) {
        return Err(TypeError::Unsupported(
            "spawn thunk may not return a `Ptr`: a raw machine address must not \
             cross a thread boundary in the share-nothing model. Return a value \
             (Int/String/Bytes/struct/enum) instead."
                .to_owned(),
        ));
    }
    if type_contains_atom(thunk_ret) {
        return Err(TypeError::Unsupported(
            "spawn thunk may not return an `Atom`: returning a mutable cell would \
             share mutable state across threads. Return a snapshot (e.g. `deref`)."
                .to_owned(),
        ));
    }
    if let Expr::Lambda { params, body } = thunk {
        let mut outer: Vec<u32> = Vec::new();
        collect_outer_captures(body, params.len() as u32, &mut outer);
        for k in outer {
            if (k as usize) < locals.len() {
                let cap_ty = &locals[locals.len() - 1 - k as usize];
                if type_contains_ptr(cap_ty) {
                    return Err(TypeError::Unsupported(
                        "spawn thunk captures a `Ptr`: a raw machine address must not \
                         be shared across threads. Capture only values."
                            .to_owned(),
                    ));
                }
                if type_contains_atom(cap_ty) {
                    return Err(TypeError::Unsupported(
                        "spawn thunk captures an `Atom`: threads are share-nothing, so \
                         a shared mutable cell can't be captured (this is what makes \
                         data races impossible). Capture a snapshot via `deref`, or \
                         model shared state as a node `state` reached through a \
                         message instead."
                            .to_owned(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn collect_outer_captures(e: &Expr, depth: u32, out: &mut Vec<u32>) {
    match e {
        Expr::LocalVar(i) => {
            if *i >= depth {
                out.push(*i - depth);
            }
        }
        Expr::Call(callee, args) => {
            collect_outer_captures(callee, depth, out);
            for a in args {
                collect_outer_captures(a, depth, out);
            }
        }
        Expr::Lambda { params, body } => {
            collect_outer_captures(body, depth + params.len() as u32, out);
        }
        Expr::Let { value, body } => {
            collect_outer_captures(value, depth, out);
            collect_outer_captures(body, depth + 1, out);
        }
        Expr::Defer { cleanup, body } => {
            collect_outer_captures(cleanup, depth, out);
            collect_outer_captures(body, depth, out);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                collect_outer_captures(f, depth, out);
            }
        }
        Expr::Field { base, .. } => collect_outer_captures(base, depth, out),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                collect_outer_captures(p, depth, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_outer_captures(scrutinee, depth, out);
            for arm in arms {
                let binds = pattern_binder_count(&arm.pattern);
                collect_outer_captures(&arm.body, depth + binds, out);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_outer_captures(cond, depth, out);
            collect_outer_captures(then_branch, depth, out);
            collect_outer_captures(else_branch, depth, out);
        }
        Expr::Try { expr, .. } => collect_outer_captures(expr, depth, out),
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

/// Signature of a builtin by stable name. `None` indicates the builtin is
/// either unknown OR special-cased at the call site (e.g. `core/net.at`,
/// which is polymorphic and handled directly in `typecheck_expr`).
pub fn builtin_signature(name: &str) -> Option<(Vec<Type>, Type)> {
    match name {
        // Binary i64 arithmetic + comparisons. All return Int in v1
        // (we widen i1 to i64 in codegen; keep the type story consistent here).
        "core/i64.add"
        | "core/i64.sub"
        | "core/i64.mul"
        | "core/i64.div"
        | "core/i64.rem"
        | "core/i64.eq"
        | "core/i64.ne"
        | "core/i64.lt"
        | "core/i64.le"
        | "core/i64.gt"
        | "core/i64.ge" => Some((vec![int_t(), int_t()], int_t())),

        // Unary i64 negation, unary bool not.
        "core/i64.neg" | "core/bool.not" => Some((vec![int_t()], int_t())),

        // `gc_collect()` — forces a stop-the-world collection. Returns
        // Int (always 0) so it can be used as a statement-like side
        // effect from let bindings: `let _ = gc_collect();`.
        "core/gc.collect" => Some((vec![], int_t())),

        // `panic(msg)` — prints `msg` to stderr and aborts. Diverges, so
        // its result type is the bottom type `Never`, which is compatible
        // with every expected type (it can appear in any match arm / if
        // branch / function body).
        "core/panic" => Some((
            vec![Type::Builtin("String".to_owned())],
            Type::Builtin("Never".to_owned()),
        )),

        // String ops. Strings are pointer-typed (TypeRef-like) heap
        // values; we represent the type as Type::Builtin("String").
        "core/string.len" => Some((
            vec![Type::Builtin("String".to_owned())],
            int_t(),
        )),
        "core/string.eq" => Some((
            vec![
                Type::Builtin("String".to_owned()),
                Type::Builtin("String".to_owned()),
            ],
            int_t(),
        )),
        "core/string.concat" => Some((
            vec![
                Type::Builtin("String".to_owned()),
                Type::Builtin("String".to_owned()),
            ],
            Type::Builtin("String".to_owned()),
        )),

        // Bytes ops. `Bytes` is a mutable, indexable byte buffer sharing
        // the heap layout of String but kept distinct by the type system.
        "core/bytes.new" => Some((
            vec![int_t()],
            Type::Builtin("Bytes".to_owned()),
        )),
        // Decode a Call-payload frame (an encoded zero-arg closure), invoke
        // the closure on this node, and return the encoded result frame.
        // The handler-agnostic core of an ail `serve` loop. No memoization.
        "core/wire.invoke" => Some((
            vec![Type::Builtin("Bytes".to_owned())],
            Type::Builtin("Bytes".to_owned()),
        )),
        // Encode ANY value to a wire frame; decode an Int / shipped fn back.
        // `wire_encode(value: T) -> Bytes`.
        "core/wire.encode" => Some((
            vec![Type::TypeVar(0)],
            Type::Builtin("Bytes".to_owned()),
        )),
        "core/wire.decode_int" => Some((
            vec![Type::Builtin("Bytes".to_owned())],
            int_t(),
        )),
        "core/wire.decode_fn1" => Some((
            vec![Type::Builtin("Bytes".to_owned())],
            Type::FnType {
                params: vec![int_t()],
                ret: Box::new(int_t()),
            },
        )),
        // Structural hash / equality of ANY value (generic key support for
        // HashMap). `hash_value(k) -> Int`, `value_eq(a, b) -> Int` (0/1).
        "core/hash.value" => Some((vec![Type::TypeVar(0)], int_t())),
        "core/value.eq" => Some((vec![Type::TypeVar(0), Type::TypeVar(0)], int_t())),
        "core/bytes.len" => Some((
            vec![Type::Builtin("Bytes".to_owned())],
            int_t(),
        )),
        "core/bytes.get" => Some((
            vec![Type::Builtin("Bytes".to_owned()), int_t()],
            int_t(),
        )),
        "core/bytes.set" => Some((
            vec![Type::Builtin("Bytes".to_owned()), int_t(), int_t()],
            int_t(),
        )),
        "core/bytes.slice" => Some((
            vec![Type::Builtin("Bytes".to_owned()), int_t(), int_t()],
            Type::Builtin("Bytes".to_owned()),
        )),
        "core/bytes.concat" => Some((
            vec![
                Type::Builtin("Bytes".to_owned()),
                Type::Builtin("Bytes".to_owned()),
            ],
            Type::Builtin("Bytes".to_owned()),
        )),
        "core/bytes.from_string" => Some((
            vec![Type::Builtin("String".to_owned())],
            Type::Builtin("Bytes".to_owned()),
        )),
        "core/string.from_bytes" => Some((
            vec![Type::Builtin("Bytes".to_owned())],
            Type::Builtin("String".to_owned()),
        )),

        // Array ops. `Array<T>` is a fixed-size, O(1)-indexable vector of
        // GC-traced pointer slots. Generic over T via TypeVar(0); the
        // general call path unifies the array arg's `Array<Concrete>`
        // against `Array<T>` to recover T. Array type = Apply(Builtin
        // "Array", [T]).
        "core/array.new" => Some((
            vec![int_t()],
            array_t(Type::TypeVar(0)),
        )),
        "core/array.len" => Some((
            vec![array_t(Type::TypeVar(0))],
            int_t(),
        )),
        "core/array.get" => Some((
            vec![array_t(Type::TypeVar(0)), int_t()],
            Type::TypeVar(0),
        )),
        "core/array.set" => Some((
            vec![array_t(Type::TypeVar(0)), int_t(), Type::TypeVar(0)],
            int_t(),
        )),
        // Atom primitives over the dedicated `Atom<T>` cell shape.
        //   atom_new(init: T) -> Atom<T>
        "core/atom.new" => Some((
            vec![Type::TypeVar(0)],
            atom_t(Type::TypeVar(0)),
        )),
        //   atom_load(a: Atom<T>) -> T
        "core/atom.load" => Some((
            vec![atom_t(Type::TypeVar(0))],
            Type::TypeVar(0),
        )),
        //   atom_swap(a: Atom<T>, f: fn(T) -> T) -> T  (lock-free CAS loop)
        "core/atom.swap" => Some((
            vec![
                atom_t(Type::TypeVar(0)),
                Type::FnType {
                    params: vec![Type::TypeVar(0)],
                    ret: Box::new(Type::TypeVar(0)),
                },
            ],
            Type::TypeVar(0),
        )),
        // Thread primitives over the `ThreadHandle<T>` shape.
        //   thread_spawn(thunk: fn() -> T) -> ThreadHandle<T>
        //   thread_spawn_shared: same signature, zero-copy opt-out.
        "core/thread.spawn" | "core/thread.spawn_shared" => Some((
            vec![Type::FnType {
                params: vec![],
                ret: Box::new(Type::TypeVar(0)),
            }],
            thread_handle_t(Type::TypeVar(0)),
        )),
        //   thread_join(h: ThreadHandle<T>) -> T
        "core/thread.join" => Some((
            vec![thread_handle_t(Type::TypeVar(0))],
            Type::TypeVar(0),
        )),

        // Float arithmetic — operands and result are Float.
        "core/f64.add" | "core/f64.sub" | "core/f64.mul" | "core/f64.div"
        | "core/f64.rem" => Some((vec![float_t(), float_t()], float_t())),
        // Float comparisons — Float operands, Int (0/1) result.
        "core/f64.eq" | "core/f64.ne" | "core/f64.lt" | "core/f64.le"
        | "core/f64.gt" | "core/f64.ge" => Some((vec![float_t(), float_t()], int_t())),
        // Int <-> Float conversions.
        "core/f64.of_int" => Some((vec![int_t()], float_t())),
        "core/f64.to_int" => Some((vec![float_t()], int_t())),

        // Raw-pointer / memory intrinsics. `Ptr` is an i64-represented
        // raw address (non-GC). Built on by the C FFI: allocate with
        // libc `malloc`, read/write bytes with these.
        // ptr_null() -> Ptr
        "core/ptr.null" => Some((vec![], ptr_t())),
        // ptr_is_null(Ptr) -> Int (1 if null, else 0)
        "core/ptr.is_null" => Some((vec![ptr_t()], int_t())),
        // ptr_add(Ptr, Int) -> Ptr (byte offset)
        "core/ptr.add" => Some((vec![ptr_t(), int_t()], ptr_t())),
        // ptr_read_u8(Ptr, offset: Int) -> Int (zero-extended byte at base+offset)
        "core/ptr.read_u8" => Some((vec![ptr_t(), int_t()], int_t())),
        // ptr_write_u8(Ptr, offset: Int, value: Int) -> Int (returns 0)
        "core/ptr.write_u8" => Some((vec![ptr_t(), int_t(), int_t()], int_t())),
        // ptr_read_i64(Ptr, offset: Int) -> Int
        "core/ptr.read_i64" => Some((vec![ptr_t(), int_t()], int_t())),
        // ptr_write_i64(Ptr, offset: Int, value: Int) -> Int (returns 0)
        "core/ptr.write_i64" => Some((vec![ptr_t(), int_t(), int_t()], int_t())),
        // ptr_read_ptr(Ptr, offset: Int) -> Ptr
        "core/ptr.read_ptr" => Some((vec![ptr_t(), int_t()], ptr_t())),
        // ptr_write_ptr(Ptr, offset: Int, value: Ptr) -> Int (returns 0)
        "core/ptr.write_ptr" => Some((vec![ptr_t(), int_t(), ptr_t()], int_t())),
        // Explicit reinterpret casts: ptr_to_int(Ptr) -> Int (the raw
        // address as data), int_to_ptr(Int) -> Ptr (fabricate a pointer).
        "core/ptr.to_int" => Some((vec![ptr_t()], int_t())),
        "core/ptr.from_int" => Some((vec![int_t()], ptr_t())),
        // Bitwise ops on Int (a, b) -> Int.
        "core/i64.and" | "core/i64.or" | "core/i64.xor" | "core/i64.shl"
        | "core/i64.shr" => Some((vec![int_t(), int_t()], int_t())),

        // Bool and / or — operands are still Int in v1 (Bool widens to Int).
        "core/bool.and" | "core/bool.or" => Some((vec![int_t(), int_t()], int_t())),

        // core/net.at is polymorphic; handled at the call site.
        "core/net.at" => None,

        _ => None,
    }
}

// =============================================================================
// Wire computation
// =============================================================================

/// Recursively determine whether a type is Wire (transmittable across an
/// `at()` boundary). Looks up named types via the cache.
pub fn is_wire(ty: &Type, cache: &TypeCache) -> bool {
    match ty {
        // All v1 builtin types are Wire.
        Type::Builtin(_) => true,
        Type::FnType { params, ret } => {
            params.iter().all(|p| is_wire(p, cache)) && is_wire(ret, cache)
        }
        Type::TypeRef(h) => cache.get(h).map(|s| s.is_wire()).unwrap_or(false),
        // Generics aren't supported in v1; conservatively non-Wire.
        Type::TypeVar(_) => false,
        Type::Apply(head, args) => {
            is_wire(head, cache) && args.iter().all(|a| is_wire(a, cache))
        }
        // SelfRef only appears in canonical bytes during hashing;
        // it should never reach a stored typecheck. Conservatively
        // non-Wire if it ever does.
        Type::SelfRef(_) => false,
    }
}

/// Verify that a `Type` is well-formed against the cache (every `TypeRef`
/// is known). Returns the first unknown ref encountered, if any.
fn check_type_known(ty: &Type, cache: &TypeCache) -> Result<(), TypeError> {
    match ty {
        Type::Builtin(_) | Type::TypeVar(_) => Ok(()),
        Type::TypeRef(h) => {
            if cache.contains(h) {
                Ok(())
            } else {
                Err(TypeError::UnknownTypeRef(*h))
            }
        }
        Type::FnType { params, ret } => {
            for p in params {
                check_type_known(p, cache)?;
            }
            check_type_known(ret, cache)
        }
        Type::Apply(head, args) => {
            check_type_known(head, cache)?;
            for a in args {
                check_type_known(a, cache)?;
            }
            Ok(())
        }
        Type::SelfRef(_) => Err(TypeError::Unsupported(
            "Type::SelfRef should not appear in stored ASTs (resolver bug?)".to_owned(),
        )),
    }
}

// =============================================================================
// Def-level typechecking
// =============================================================================

/// Compute the `TypeScheme` for a single def. Pure function of `(def, cache)`:
/// idempotent and cache-friendly. The caller is responsible for inserting
/// the returned scheme into the cache.
pub fn typecheck_def(def: &Def, cache: &TypeCache) -> Result<TypeScheme, TypeError> {
    match def {
        Def::Struct { type_params, fields } => {
            // Each field type must be well-formed against the cache,
            // taking declared type-vars into account (any `TypeVar(i)`
            // with i < type_params is in scope and well-formed by
            // construction).
            for (_, ty) in fields {
                check_type_well_formed(ty, cache, *type_params)?;
            }
            // Generic structs and enums are conservatively non-Wire for
            // now — Wire-ness depends on the concrete instantiation.
            let wire = *type_params == 0 && fields.iter().all(|(_, t)| is_wire(t, cache));
            Ok(TypeScheme::Struct {
                type_params: *type_params,
                fields: fields.clone(),
                wire,
            })
        }
        Def::Enum { type_params, variants } => {
            for (_, payload) in variants {
                if let Some(ty) = payload {
                    check_type_well_formed(ty, cache, *type_params)?;
                }
            }
            let wire = *type_params == 0
                && variants
                    .iter()
                    .all(|(_, p)| p.as_ref().map(|t| is_wire(t, cache)).unwrap_or(true));
            Ok(TypeScheme::Enum {
                type_params: *type_params,
                variants: variants.clone(),
                wire,
            })
        }
        Def::Fn {
            is_local: _,
            type_params,
            params,
            ret,
            body,
        } => {
            for p in params {
                check_type_well_formed(p, cache, *type_params)?;
            }
            check_type_well_formed(ret, cache, *type_params)?;

            // Build the initial lexical env. Params are pushed in source
            // order: with N params, source-order param 0 is at env[0]
            // (the outermost binder), source-order param N-1 is at
            // env[N-1] (the innermost binder, i.e. LocalVar(0)).
            let mut locals: Vec<Type> = params.clone();
            // Publish this fn's return type so any `?` in the body can
            // check the enclosing function returns a matching Result.
            *cache.current_fn_ret.borrow_mut() = Some(ret.clone());
            let body_ty = typecheck_expr(body, &mut locals, cache)?;
            *cache.current_fn_ret.borrow_mut() = None;
            if !types_equiv(&body_ty, ret) {
                return Err(TypeError::TypeMismatch {
                    context: "fn body".to_owned(),
                    expected: ret.clone(),
                    got: body_ty,
                });
            }
            let wire = *type_params == 0
                && params.iter().all(|p| is_wire(p, cache))
                && is_wire(ret, cache);
            Ok(TypeScheme::Fn {
                params: params.clone(),
                ret: ret.clone(),
                wire,
            })
        }
        Def::State { ty, init } => {
            check_type_well_formed(ty, cache, 0)?;
            // The initializer is a zero-parameter expression; it must
            // produce a value of the declared type. No enclosing fn return
            // type is published, so `?` in a state initializer is rejected.
            let mut locals: Vec<Type> = Vec::new();
            let init_ty = typecheck_expr(init, &mut locals, cache)?;
            if !types_equiv(&init_ty, ty) {
                return Err(TypeError::TypeMismatch {
                    context: "state initializer".to_owned(),
                    expected: ty.clone(),
                    got: init_ty,
                });
            }
            Ok(TypeScheme::State { ty: ty.clone() })
        }
    }
}

/// Like `check_type_known` but additionally permits `TypeVar(i)` for
/// `i < bound` (declared type-params in the enclosing def).
fn check_type_well_formed(
    ty: &Type,
    cache: &TypeCache,
    bound: u32,
) -> Result<(), TypeError> {
    match ty {
        Type::Builtin(_) => Ok(()),
        Type::TypeVar(i) => {
            if *i < bound {
                Ok(())
            } else {
                Err(TypeError::Unsupported(format!(
                    "TypeVar({}) out of scope (only {} type params declared)",
                    i, bound
                )))
            }
        }
        Type::TypeRef(h) => {
            if cache.contains(h) {
                Ok(())
            } else {
                Err(TypeError::UnknownTopRef(*h))
            }
        }
        Type::FnType { params, ret } => {
            for p in params {
                check_type_well_formed(p, cache, bound)?;
            }
            check_type_well_formed(ret, cache, bound)
        }
        Type::Apply(head, args) => {
            check_type_well_formed(head, cache, bound)?;
            for a in args {
                check_type_well_formed(a, cache, bound)?;
            }
            // Arity check: head must be a TypeRef to a scheme whose
            // declared type_params count matches args.len().
            if let Type::TypeRef(h) = head.as_ref() {
                if let Some(scheme) = cache.get(h) {
                    let declared = scheme_type_params(scheme);
                    if declared as usize != args.len() {
                        return Err(TypeError::TypeMismatch {
                            context: format!(
                                "type application of {} expects {} args, got {}",
                                h,
                                declared,
                                args.len()
                            ),
                            expected: Type::TypeVar(declared),
                            got: Type::TypeVar(args.len() as u32),
                        });
                    }
                }
            }
            Ok(())
        }
        Type::SelfRef(_) => Err(TypeError::Unsupported(
            "Type::SelfRef should not appear in stored ASTs".to_owned(),
        )),
    }
}

/// How many type params the given scheme declared. We need this for
/// arity checking of `Type::Apply`.
fn scheme_type_params(scheme: &TypeScheme) -> u32 {
    match scheme {
        TypeScheme::Fn { params, ret, .. } => max_type_var(params, ret),
        // Use the declared arity, not the field-derived one, so a
        // phantom type param (appears in no field) still counts.
        TypeScheme::Struct {
            type_params,
            fields,
            ..
        } => {
            let mut m = *type_params;
            for (_, t) in fields {
                m = m.max(max_typevar_in(t));
            }
            m
        }
        TypeScheme::Enum {
            type_params,
            variants,
            ..
        } => {
            let mut m = *type_params;
            for (_, p) in variants {
                if let Some(t) = p {
                    m = m.max(max_typevar_in(t));
                }
            }
            m
        }
        // A `state` binding is not a type constructor; arity 0.
        TypeScheme::State { .. } => 0,
    }
}

fn max_type_var(params: &[Type], ret: &Type) -> u32 {
    let mut m = max_typevar_in(ret);
    for p in params {
        m = m.max(max_typevar_in(p));
    }
    m
}

/// Highest `TypeVar(i)` index + 1 that occurs in `ty`, i.e. the
/// number of distinct type params it would need to scope.
fn max_typevar_in(ty: &Type) -> u32 {
    match ty {
        Type::TypeVar(i) => i + 1,
        Type::Builtin(_) | Type::TypeRef(_) | Type::SelfRef(_) => 0,
        Type::FnType { params, ret } => max_type_var(params, ret),
        Type::Apply(head, args) => {
            let mut m = max_typevar_in(head);
            for a in args {
                m = m.max(max_typevar_in(a));
            }
            m
        }
    }
}

/// Substitute the type-var entries in `subst` (positional, indexed by
/// TypeVar index) throughout `ty`. Unbound vars (i >= subst.len()) are
/// left alone.
fn substitute(ty: &Type, subst: &[Type]) -> Type {
    match ty {
        Type::TypeVar(i) => subst
            .get(*i as usize)
            .cloned()
            .unwrap_or_else(|| Type::TypeVar(*i)),
        Type::Builtin(_) | Type::TypeRef(_) | Type::SelfRef(_) => ty.clone(),
        Type::FnType { params, ret } => Type::FnType {
            params: params.iter().map(|p| substitute(p, subst)).collect(),
            ret: Box::new(substitute(ret, subst)),
        },
        Type::Apply(head, args) => Type::Apply(
            Box::new(substitute(head, subst)),
            args.iter().map(|a| substitute(a, subst)).collect(),
        ),
    }
}

/// Try to unify a declared type (possibly containing TypeVars) with an
/// actual type. On success, `subst[i]` either is `None` (var still
/// unbound) or contains the type the var must be. On conflict, returns
/// the offending pair.
fn unify(
    declared: &Type,
    actual: &Type,
    subst: &mut Vec<Option<Type>>,
) -> Result<(), (Type, Type)> {
    // A `TypeVar(_)` on the actual side is treated as "instantiation
    // unknown" — never constrains the substitution and is compatible
    // with any declared shape. Bare constructors like `Nil` flow into
    // concrete-typed positions this way. Handled symmetrically below.
    if matches!(actual, Type::TypeVar(_)) {
        return Ok(());
    }
    match (declared, actual) {
        (Type::TypeVar(i), other) => {
            let idx = *i as usize;
            if idx >= subst.len() {
                subst.resize(idx + 1, None);
            }
            match subst[idx].clone() {
                None => {
                    subst[idx] = Some(other.clone());
                    Ok(())
                }
                Some(t) if types_equiv(&t, other) => Ok(()),
                Some(Type::TypeVar(_)) => {
                    // Stored a placeholder from an earlier TypeVar-vs-
                    // TypeVar unification; refine to concrete.
                    subst[idx] = Some(other.clone());
                    Ok(())
                }
                Some(t) => Err((t, other.clone())),
            }
        }
        (Type::Builtin(a), Type::Builtin(b)) if a == b => Ok(()),
        (Type::TypeRef(a), Type::TypeRef(b)) if a == b => Ok(()),
        (
            Type::FnType { params: pa, ret: ra },
            Type::FnType { params: pb, ret: rb },
        ) if pa.len() == pb.len() => {
            for (a, b) in pa.iter().zip(pb.iter()) {
                unify(a, b, subst)?;
            }
            unify(ra, rb, subst)
        }
        (Type::Apply(ha, aa), Type::Apply(hb, ab)) if aa.len() == ab.len() => {
            unify(ha, hb, subst)?;
            for (a, b) in aa.iter().zip(ab.iter()) {
                unify(a, b, subst)?;
            }
            Ok(())
        }
        _ => Err((declared.clone(), actual.clone())),
    }
}

/// Whether two types are equivalent for our generics-aware checks.
///
/// This is symmetric structural equality, EXCEPT inside type arguments
/// of an `Apply` (and the params/ret of a `FnType` payload), where a
/// `TypeVar(_)` on either side is treated as "instantiation unknown" —
/// a wildcard that matches any concrete type. This lets bare nullary
/// constructors like `Nil : Apply(List, [TypeVar(0)])` flow into
/// concrete-typed positions like `acc: List<Int>` without forcing the
/// user to annotate or requiring full bidirectional inference.
///
/// At the *top level* (i.e. comparing whole types, not nested in an
/// Apply or FnType payload), TypeVars must still match exactly — that
/// preserves checking inside generic fn bodies (where `TypeVar(0)`
/// means a specific declared type param).
/// The bottom type for diverging expressions (e.g. `panic`). Represented
/// as `Builtin("Never")` so it needs no new `Type` variant (and never
/// reaches serialization — users can't write it; the resolver only
/// produces it for `panic` calls).
fn is_never(t: &Type) -> bool {
    matches!(t, Type::Builtin(n) if n == "Never")
}

fn types_equiv(a: &Type, b: &Type) -> bool {
    types_equiv_inner(a, b, false)
}

fn types_equiv_inner(a: &Type, b: &Type, in_apply_args: bool) -> bool {
    match (a, b) {
        // `Never` (a diverging expression's type) is compatible with any
        // type in both directions.
        (Type::Builtin(n), _) | (_, Type::Builtin(n)) if n == "Never" => true,
        (Type::TypeVar(_), _) | (_, Type::TypeVar(_)) if in_apply_args => true,
        (Type::TypeVar(i), Type::TypeVar(j)) => i == j,
        (Type::Builtin(x), Type::Builtin(y)) => x == y,
        (Type::TypeRef(x), Type::TypeRef(y)) => x == y,
        (Type::SelfRef(x), Type::SelfRef(y)) => x == y,
        (Type::FnType { params: pa, ret: ra }, Type::FnType { params: pb, ret: rb }) => {
            pa.len() == pb.len()
                && pa.iter().zip(pb).all(|(x, y)| types_equiv_inner(x, y, true))
                && types_equiv_inner(ra, rb, true)
        }
        (Type::Apply(ha, aa), Type::Apply(hb, ab)) => {
            types_equiv_inner(ha, hb, false)
                && aa.len() == ab.len()
                && aa.iter().zip(ab).all(|(x, y)| types_equiv_inner(x, y, true))
        }
        _ => false,
    }
}

// =============================================================================
// Expression typechecking
// =============================================================================

/// Infer the type of `expr` evaluated in a lexical environment whose binders
/// (outermost first) have the types in `env` — exactly `Def::Fn`'s `params`
/// ordering: with N binders, `env[0]` is the outermost binder and `env[N-1]`
/// is the innermost (`LocalVar(0)`).
///
/// This is the edit layer's only supported way to recover the type of a
/// sub-expression (e.g. a `let` value lifted by `extract`) without a full
/// `Def`, since `Def::Fn` requires a declared return type up front. Returns a
/// clean `TypeError` on ill-typed input — never a silent guess.
pub fn infer_expr_type(
    expr: &Expr,
    env: &[Type],
    cache: &TypeCache,
) -> Result<Type, TypeError> {
    let mut locals: Vec<Type> = env.to_vec();
    typecheck_expr(expr, &mut locals, cache)
}

fn typecheck_expr(
    expr: &Expr,
    locals: &mut Vec<Type>,
    cache: &TypeCache,
) -> Result<Type, TypeError> {
    match expr {
        Expr::IntLit(_) => Ok(int_t()),
        Expr::FloatLit(_) => Ok(Type::Builtin("Float".to_owned())),
        Expr::BoolLit(_) => Ok(bool_t()),
        Expr::StringLit(_) => Ok(Type::Builtin("String".to_owned())),

        Expr::LocalVar(idx) => {
            // De Bruijn: 0 is the innermost (last-pushed) binder.
            let n = locals.len();
            let i = *idx as usize;
            if i >= n {
                return Err(TypeError::LocalVarOutOfRange {
                    index: *idx,
                    env_size: n,
                });
            }
            Ok(locals[n - 1 - i].clone())
        }

        Expr::TopRef(h) => match cache.get(h) {
            None => Err(TypeError::UnknownTopRef(*h)),
            Some(TypeScheme::Fn { params, ret, .. }) => Ok(Type::FnType {
                params: params.clone(),
                ret: Box::new(ret.clone()),
            }),
            Some(TypeScheme::Struct { .. }) | Some(TypeScheme::Enum { .. }) => {
                Ok(Type::TypeRef(*h))
            }
            Some(TypeScheme::State { .. }) => Err(TypeError::UnknownTopRef(*h)),
        },

        // A reference to a node `state` binding: its declared type.
        Expr::StateRef(h) => match cache.get(h) {
            Some(TypeScheme::State { ty }) => Ok(ty.clone()),
            _ => Err(TypeError::UnknownTopRef(*h)),
        },

        Expr::SelfRef(idx) | Expr::StateSelfRef(idx) => Err(TypeError::SelfRefInTypecheck {
            component_index: *idx,
        }),

        Expr::BuiltinRef(name) => match builtin_signature(name) {
            Some((params, ret)) => Ok(Type::FnType {
                params,
                ret: Box::new(ret),
            }),
            // `core/net.at#<hex>` and the legacy `core/net.at` are
            // polymorphic — using them as bare refs (not in call
            // position) isn't supported in v1.
            None if name == "core/net.at"
                || crate::resolve::parse_at_builtin_name(name).is_some() =>
            {
                Err(TypeError::Unsupported(
                    "core/net.at must appear in call position".to_owned(),
                ))
            }
            // `ext/<name>` extern call — look up signature from the
            // type cache's registered externs.
            None if name.starts_with("ext/") => {
                let ext_name = &name["ext/".len()..];
                match cache.extern_signature(ext_name) {
                    Some((params, ret, _variadic)) => Ok(Type::FnType {
                        params: params.clone(),
                        ret: Box::new(ret.clone()),
                    }),
                    None => Err(TypeError::UnknownBuiltin(name.clone())),
                }
            }
            None => Err(TypeError::UnknownBuiltin(name.clone())),
        },

        Expr::Call(callee, args) => typecheck_call(callee, args, locals, cache),

        Expr::Lambda { params, body } => {
            for p in params {
                check_type_known(p, cache)?;
            }
            // Push params in source order — same convention as fn defs.
            let pushed = params.len();
            for p in params {
                locals.push(p.clone());
            }
            let body_ty = typecheck_expr(body, locals, cache)?;
            for _ in 0..pushed {
                locals.pop();
            }
            Ok(Type::FnType {
                params: params.clone(),
                ret: Box::new(body_ty),
            })
        }

        Expr::Let { value, body } => {
            let value_ty = typecheck_expr(value, locals, cache)?;
            locals.push(value_ty);
            let body_ty = typecheck_expr(body, locals, cache)?;
            locals.pop();
            Ok(body_ty)
        }

        // `defer cleanup; body`: the cleanup is checked in the SAME
        // environment as the body (it adds no binder), and its type is
        // discarded. The whole expression has the body's type.
        Expr::Defer { cleanup, body } => {
            let _ = typecheck_expr(cleanup, locals, cache)?;
            typecheck_expr(body, locals, cache)
        }

        Expr::StructNew { struct_ref, fields } => {
            let scheme = cache
                .get(struct_ref)
                .ok_or(TypeError::UnknownStruct(*struct_ref))?;
            // Declared generic arity — includes phantom params (those
            // that appear in no field, like `Atom<T>`), so the result is
            // `Apply(head, [..])` and can unify against an annotation
            // that pins the phantom.
            let declared_params = scheme_type_params(scheme);
            let decl_fields = scheme
                .as_struct()
                .ok_or(TypeError::UnknownStruct(*struct_ref))?;
            if decl_fields.len() != fields.len() {
                return Err(TypeError::ArityMismatch {
                    what: format!("struct {}", struct_ref),
                    expected: decl_fields.len(),
                    got: fields.len(),
                });
            }
            // Clone the expected types so we can release the borrow.
            let expected_tys: Vec<Type> =
                decl_fields.iter().map(|(_, t)| t.clone()).collect();
            // Bottom-up generic inference: if the declared field types
            // contain `TypeVar(i)` (i.e. this struct is generic), unify
            // each declared field type against the actual arg's type
            // to recover the instantiation. The result type is
            // `Apply(TypeRef(h), [...])` with inferred type-args; if
            // the struct is monomorphic, this stays bare `TypeRef`.
            let n_params: u32 = declared_params.max(
                expected_tys
                    .iter()
                    .map(max_typevar_in)
                    .max()
                    .unwrap_or(0),
            );
            let mut subst: Vec<Option<Type>> = vec![None; n_params as usize];
            for (i, (e, expected)) in fields.iter().zip(expected_tys.iter()).enumerate() {
                let got = typecheck_expr(e, locals, cache)?;
                if n_params > 0 {
                    if let Err((d, a)) = unify(expected, &got, &mut subst) {
                        return Err(TypeError::TypeMismatch {
                            context: format!("struct {} field {}", struct_ref, i),
                            expected: d,
                            got: a,
                        });
                    }
                } else if !types_equiv(&got, expected) {
                    return Err(TypeError::TypeMismatch {
                        context: format!("struct {} field {}", struct_ref, i),
                        expected: expected.clone(),
                        got,
                    });
                }
            }
            if n_params == 0 {
                Ok(Type::TypeRef(*struct_ref))
            } else {
                let args: Vec<Type> = (0..n_params as usize)
                    .map(|i| {
                        subst.get(i)
                            .and_then(|o| o.clone())
                            .unwrap_or(Type::TypeVar(i as u32))
                    })
                    .collect();
                Ok(Type::Apply(Box::new(Type::TypeRef(*struct_ref)), args))
            }
        }

        Expr::Field {
            base,
            struct_ref,
            index,
        } => {
            let base_ty = typecheck_expr(base, locals, cache)?;
            // Base must reference the named struct, either as bare
            // `TypeRef(h)` (non-generic / inferred-nothing) or as
            // `Apply(TypeRef(h), [args])` (instantiated generic). The
            // instantiation gets substituted into the declared field
            // type so `(cell : ListCell<Int>).head` projects as `Int`
            // not `TypeVar(0)`.
            let (got_ref, instantiation) = match &base_ty {
                Type::TypeRef(h) => (*h, Vec::new()),
                Type::Apply(head, args) => match head.as_ref() {
                    Type::TypeRef(h) => (*h, args.clone()),
                    _ => return Err(TypeError::ExpectedStruct { got: base_ty }),
                },
                _ => return Err(TypeError::ExpectedStruct { got: base_ty }),
            };
            if got_ref != *struct_ref {
                return Err(TypeError::ExpectedStruct { got: base_ty });
            }
            let scheme = cache
                .get(struct_ref)
                .ok_or(TypeError::UnknownStruct(*struct_ref))?;
            let decl_fields = scheme
                .as_struct()
                .ok_or(TypeError::UnknownStruct(*struct_ref))?;
            let i = *index as usize;
            if i >= decl_fields.len() {
                return Err(TypeError::FieldIndexOutOfRange {
                    struct_ref: *struct_ref,
                    index: *index,
                    field_count: decl_fields.len(),
                });
            }
            let field_ty = &decl_fields[i].1;
            if instantiation.is_empty() {
                Ok(field_ty.clone())
            } else {
                Ok(substitute(field_ty, &instantiation))
            }
        }

        Expr::EnumNew {
            enum_ref,
            variant_index,
            payload,
        } => {
            let scheme = cache
                .get(enum_ref)
                .ok_or(TypeError::UnknownEnum(*enum_ref))?;
            let variants = scheme.as_enum().ok_or(TypeError::UnknownEnum(*enum_ref))?;
            let i = *variant_index as usize;
            if i >= variants.len() {
                return Err(TypeError::VariantIndexOutOfRange {
                    enum_ref: *enum_ref,
                    index: *variant_index,
                    variant_count: variants.len(),
                });
            }
            let expected_payload = variants[i].1.clone();
            // Compute the enum's type-param count from the scheme so
            // we can build an `Apply(...)` result with inferred args.
            let n_params: u32 = variants
                .iter()
                .filter_map(|(_, p)| p.as_ref())
                .map(max_typevar_in)
                .max()
                .unwrap_or(0);
            let mut subst: Vec<Option<Type>> = vec![None; n_params as usize];
            match (expected_payload.as_ref(), payload.as_ref()) {
                (None, None) => {}
                (Some(expected_ty), Some(p)) => {
                    let expected_ty = expected_ty.clone();
                    let got = typecheck_expr(p, locals, cache)?;
                    if n_params > 0 {
                        if let Err((d, a)) = unify(&expected_ty, &got, &mut subst) {
                            return Err(TypeError::TypeMismatch {
                                context: format!(
                                    "EnumNew payload of variant {} of {}",
                                    variant_index, enum_ref
                                ),
                                expected: d,
                                got: a,
                            });
                        }
                    } else if !types_equiv(&got, &expected_ty) {
                        return Err(TypeError::TypeMismatch {
                            context: format!(
                                "EnumNew payload of variant {} of {}",
                                variant_index, enum_ref
                            ),
                            expected: expected_ty,
                            got,
                        });
                    }
                }
                (None, Some(_)) => {
                    return Err(TypeError::EnumNewPayloadShapeMismatch {
                        enum_ref: *enum_ref,
                        variant_index: *variant_index,
                        new_has_payload: true,
                        variant_has_payload: false,
                    });
                }
                (Some(_), None) => {
                    return Err(TypeError::EnumNewPayloadShapeMismatch {
                        enum_ref: *enum_ref,
                        variant_index: *variant_index,
                        new_has_payload: false,
                        variant_has_payload: true,
                    });
                }
            }
            if n_params == 0 {
                Ok(Type::TypeRef(*enum_ref))
            } else {
                let args: Vec<Type> = (0..n_params as usize)
                    .map(|i| {
                        subst.get(i)
                            .and_then(|o| o.clone())
                            .unwrap_or(Type::TypeVar(i as u32))
                    })
                    .collect();
                Ok(Type::Apply(Box::new(Type::TypeRef(*enum_ref)), args))
            }
        }

        Expr::Match { scrutinee, arms } => {
            let scrut_ty = typecheck_expr(scrutinee, locals, cache)?;
            // Scrutinee may be a bare `TypeRef` (monomorphic enum) or
            // an `Apply(TypeRef, args)` (instantiated generic enum).
            let (enum_hash, instantiation) = match &scrut_ty {
                Type::TypeRef(h) => (*h, Vec::new()),
                Type::Apply(head, args) => match head.as_ref() {
                    Type::TypeRef(h) => (*h, args.clone()),
                    _ => return Err(TypeError::ExpectedEnum { got: scrut_ty }),
                },
                _ => return Err(TypeError::ExpectedEnum { got: scrut_ty }),
            };
            let scheme = cache.get(&enum_hash).ok_or(TypeError::UnknownEnum(enum_hash))?;
            if scheme.as_enum().is_none() {
                return Err(TypeError::ExpectedEnum { got: scrut_ty });
            }

            if arms.is_empty() {
                return Err(TypeError::ArityMismatch {
                    what: "match arms".to_owned(),
                    expected: 1,
                    got: 0,
                });
            }

            let mut result_ty: Option<Type> = None;
            for arm in arms {
                let pushed = typecheck_pattern(
                    &arm.pattern,
                    &scrut_ty,
                    cache,
                    locals,
                    &instantiation,
                )?;
                let arm_ty = typecheck_expr(&arm.body, locals, cache)?;
                for _ in 0..pushed {
                    locals.pop();
                }
                result_ty = Some(match result_ty {
                    None => arm_ty,
                    Some(first) => {
                        if !types_equiv(&first, &arm_ty) {
                            return Err(TypeError::MatchArmsDisagree {
                                first,
                                found: arm_ty,
                            });
                        }
                        // Prefer a concrete type over `Never` so the match
                        // result isn't bottom just because an earlier arm
                        // diverged (e.g. `panic`).
                        if is_never(&first) { arm_ty } else { first }
                    }
                });
            }
            // Safe: arms.is_empty() was checked above.
            Ok(result_ty.expect("at least one arm checked"))
        }

        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let cond_ty = typecheck_expr(cond, locals, cache)?;
            // Condition is Int (0 = false). Bool widens to Int in v1.
            if !types_equiv(&cond_ty, &int_t()) {
                return Err(TypeError::TypeMismatch {
                    context: "if condition".to_owned(),
                    expected: int_t(),
                    got: cond_ty,
                });
            }
            let then_ty = typecheck_expr(then_branch, locals, cache)?;
            let else_ty = typecheck_expr(else_branch, locals, cache)?;
            if !types_equiv(&then_ty, &else_ty) {
                return Err(TypeError::MatchArmsDisagree {
                    first: then_ty,
                    found: else_ty,
                });
            }
            // Prefer the concrete branch type when the other diverges
            // (e.g. `if c { compute() } else { panic("...") }`).
            Ok(if is_never(&then_ty) { else_ty } else { then_ty })
        }

        Expr::Try {
            expr,
            enum_ref,
            ok_index,
            err_index,
        } => {
            let inner = typecheck_expr(expr, locals, cache)?;
            // The operand must be the `Result` enum named by `enum_ref`,
            // either bare (`TypeRef`) or instantiated (`Apply`).
            let (head_hash, args): (Hash, Vec<Type>) = match &inner {
                Type::TypeRef(h) => (*h, Vec::new()),
                Type::Apply(head, a) => match head.as_ref() {
                    Type::TypeRef(h) => (*h, a.clone()),
                    _ => {
                        return Err(TypeError::Unsupported(format!(
                            "`?` operand is not a Result: {:?}",
                            inner
                        )));
                    }
                },
                _ => {
                    return Err(TypeError::Unsupported(format!(
                        "`?` operand is not a Result: {:?}",
                        inner
                    )));
                }
            };
            if head_hash != *enum_ref {
                return Err(TypeError::Unsupported(
                    "`?` operand enum does not match the resolved Result type".to_owned(),
                ));
            }
            let variants = cache
                .get(enum_ref)
                .and_then(|s| s.as_enum())
                .ok_or(TypeError::UnknownTypeRef(*enum_ref))?
                .to_vec();
            let ok_payload = variants[*ok_index as usize].1.clone().ok_or_else(|| {
                TypeError::Unsupported("Result::Ok must carry a payload for `?`".to_owned())
            })?;
            let err_payload = variants[*err_index as usize].1.clone().ok_or_else(|| {
                TypeError::Unsupported("Result::Err must carry a payload for `?`".to_owned())
            })?;
            let ok_ty = substitute(&ok_payload, &args);
            let err_ty = substitute(&err_payload, &args);

            // The enclosing function must return a `Result<_, E>` with the
            // same error type `E`, since `?` early-returns the `Err` value.
            if let Some(ret) = cache.current_fn_ret.borrow().as_ref() {
                let ret_err: Option<Type> = match ret {
                    Type::Apply(head, a) => match head.as_ref() {
                        Type::TypeRef(h) if *h == *enum_ref => a.get(1).cloned(),
                        _ => None,
                    },
                    Type::TypeRef(h) if *h == *enum_ref => Some(err_ty.clone()),
                    _ => None,
                };
                match ret_err {
                    Some(re) if types_equiv(&re, &err_ty) => {}
                    _ => {
                        return Err(TypeError::Unsupported(format!(
                            "`?` requires the enclosing function to return a Result with error \
                             type {:?}, but it returns {:?}",
                            err_ty, ret
                        )));
                    }
                }
            }
            Ok(ok_ty)
        }
    }
}

/// Specialised handling of `Call` nodes — includes the `core/net.at`
/// polymorphic special case.
fn typecheck_call(
    callee: &Expr,
    args: &[Expr],
    locals: &mut Vec<Type>,
    cache: &TypeCache,
) -> Result<Type, TypeError> {
    // ---- spawn mobility check ----
    // A closure run on another thread must be MOBILE: like a closure
    // shipped to a remote node, it may not capture (or return) an `Atom`
    // or `Ptr`. This keeps threads share-nothing for mutable state, so
    // data races are impossible by construction. Pure/immutable captures
    // are shared safely and freely. The check runs at the user's spawn
    // call site (where the thunk is a literal closure); inside the
    // forwarding wrapper the thunk is just a parameter, so nothing fires.
    // We only check here, then fall through to normal call typing.
    let spawn_thunk = match callee {
        Expr::BuiltinRef(n) if n == "core/thread.spawn" => args.first(),
        Expr::TopRef(h) if cache.is_spawn_wrapper(h) => args.first(),
        _ => None,
    };
    if let Some(thunk) = spawn_thunk {
        let thunk_ty = typecheck_expr(thunk, locals, cache)?;
        if let Type::FnType { params, ret } = &thunk_ty {
            if params.is_empty() {
                check_spawn_thunk_mobile(thunk, ret, locals)?;
            }
        }
    }

    // ---- core/net.at#<hex>[#<hex>] special case ----
    if let Expr::BuiltinRef(name) = callee {
        let parsed = if name == "core/net.at" {
            None
        } else {
            crate::resolve::parse_at_builtin_name(name)
        };
        let is_at = parsed.is_some() || name == "core/net.at";
        if is_at {
            if args.len() != 2 {
                return Err(TypeError::ArityMismatch {
                    what: "call to core/net.at".to_owned(),
                    expected: 2,
                    got: args.len(),
                });
            }
            // Arg 0: any TypeRef (a pointer to a struct or enum).
            let arg0_ty = typecheck_expr(&args[0], locals, cache)?;
            match &arg0_ty {
                Type::TypeRef(_) => {}
                _ => {
                    return Err(TypeError::TypeMismatch {
                        context: "core/net.at first argument".to_owned(),
                        // Expected "any TypeRef" — there's no concrete
                        // expected type to name, so use a sentinel that
                        // reads well in error messages.
                        expected: Type::TypeRef(Hash([0u8; 32])),
                        got: arg0_ty.clone(),
                    });
                }
            }
            // Arg 1: a zero-arg thunk `fn() -> T`. T can be any
            // type; the wire format ships the thunk's return as a
            // heap pointer (uniform closure ABI) so the receiver
            // doesn't care whether it's an Int (boxed), a struct, or
            // an enum.
            let arg1_ty = typecheck_expr(&args[1], locals, cache)?;
            let thunk_ret = match &arg1_ty {
                Type::FnType { params, ret } if params.is_empty() => (**ret).clone(),
                _ => {
                    return Err(TypeError::TypeMismatch {
                        context: "core/net.at second argument (must be a zero-arg thunk)"
                            .to_owned(),
                        expected: Type::FnType {
                            params: vec![],
                            ret: Box::new(int_t()),
                        },
                        got: arg1_ty,
                    });
                }
            };
            // A `Ptr` is a raw local machine address; shipping the thunk's
            // result back as a `Ptr` would hand the caller a garbage
            // address into another process's memory. Reject it. (A real
            // "remote pointer" abstraction would be the only sound way to
            // return a foreign address, and that is not what `Ptr` is.)
            if type_contains_ptr(&thunk_ret) {
                return Err(TypeError::Unsupported(
                    "core/net.at thunk may not return a `Ptr`: it is a raw local \
                     machine address, meaningless on another node. Marshal the data \
                     into a wire-portable value (Int/String/Bytes/struct/enum) before \
                     returning."
                        .to_owned(),
                ));
            }
            if type_contains_atom(&thunk_ret) {
                return Err(TypeError::Unsupported(
                    "core/net.at thunk may not return an `Atom`: a node-resident \
                     mutable cell has node identity and cannot be shipped by value \
                     (it would fork). Return a snapshot via `deref`, or keep the cell \
                     as a node `state` binding and have the remote call a handler that \
                     mutates it in place."
                        .to_owned(),
                ));
            }
            // The thunk is shipped with its captures; a captured `Ptr`
            // would travel as a plain Int and deref to garbage on the
            // remote node. Reject any capture whose type contains a `Ptr`.
            if let Expr::Lambda { params, body } = &args[1] {
                let mut outer: Vec<u32> = Vec::new();
                collect_outer_captures(body, params.len() as u32, &mut outer);
                for k in outer {
                    // Outer index `k` addresses `locals` from the top.
                    if (k as usize) < locals.len() {
                        let cap_ty = &locals[locals.len() - 1 - k as usize];
                        if type_contains_ptr(cap_ty) {
                            return Err(TypeError::Unsupported(
                                "core/net.at thunk captures a `Ptr`: a raw local machine \
                                 address cannot be shipped to another node. Capture only \
                                 wire-portable values (Int/String/Bytes/struct/enum)."
                                    .to_owned(),
                            ));
                        }
                        if type_contains_atom(cap_ty) {
                            return Err(TypeError::Unsupported(
                                "core/net.at thunk captures an `Atom`: a node-resident \
                                 mutable cell cannot be shipped by value (it would fork). \
                                 To share node state, make it a `state` binding and call a \
                                 handler that reads/swaps it on the owning node; to ship a \
                                 value, capture `deref(atom)` instead."
                                    .to_owned(),
                            ));
                        }
                    }
                }
            }
            // Return type: `Apply(Result, [thunk_ret, Failure])`.
            // The builtin name always carries both hashes after the
            // resolver enforced the generic `Result<T, E>` form.
            let (result_hash, failure_hash) = match parsed {
                Some((r, Some(f))) => (r, f),
                _ => {
                    return Err(TypeError::TypeMismatch {
                        context: "core/net.at builtin name missing failure hash"
                            .to_owned(),
                        expected: Type::TypeRef(Hash([0u8; 32])),
                        got: Type::Builtin(name.clone()),
                    });
                }
            };
            return Ok(Type::Apply(
                Box::new(Type::TypeRef(result_hash)),
                vec![thunk_ret, Type::TypeRef(failure_hash)],
            ));
        }
    }

    // ---- core/wire.decode#<...> special case ----
    // `decode::<T>(bytes) -> Result<T, DecodeError>`. The runtime does the
    // shape check on `T`; the typechecker only needs the enum shape, so `T`
    // is left as a wildcard TypeVar (the call site's match recovers it).
    if let Expr::BuiltinRef(name) = callee {
        if let Some((result_hash, decode_error_hash)) =
            crate::resolve::parse_decode_builtin_name(name)
        {
            if args.len() != 1 {
                return Err(TypeError::ArityMismatch {
                    what: "call to decode::<T>".to_owned(),
                    expected: 1,
                    got: args.len(),
                });
            }
            let arg0 = typecheck_expr(&args[0], locals, cache)?;
            match &arg0 {
                Type::Builtin(b) if b == "Bytes" => {}
                _ => {
                    return Err(TypeError::TypeMismatch {
                        context: "decode::<T> argument (must be Bytes)".to_owned(),
                        expected: Type::Builtin("Bytes".to_owned()),
                        got: arg0,
                    });
                }
            }
            return Ok(Type::Apply(
                Box::new(Type::TypeRef(result_hash)),
                vec![Type::TypeVar(0), Type::TypeRef(decode_error_hash)],
            ));
        }
    }

    // ---- Variadic C extern ----
    // A variadic extern (`curl_easy_setopt(h: Ptr, opt: Int, ...)`)
    // accepts its fixed params followed by any number of trailing C
    // scalars. Check the fixed prefix, then require each extra arg to
    // be an Int or Ptr (the only types passable over the C ABI here).
    if let Expr::BuiltinRef(name) = callee {
        if let Some(ext_name) = name.strip_prefix("ext/") {
            if let Some((params, ret, true)) = cache.extern_signature(ext_name) {
                let params = params.clone();
                let ret = ret.clone();
                if args.len() < params.len() {
                    return Err(TypeError::ArityMismatch {
                        what: format!("call to variadic extern `{}`", ext_name),
                        expected: params.len(),
                        got: args.len(),
                    });
                }
                for (i, (arg, expected)) in args.iter().zip(params.iter()).enumerate() {
                    let got = typecheck_expr(arg, locals, cache)?;
                    if !types_equiv(&got, expected) {
                        return Err(TypeError::TypeMismatch {
                            context: format!("variadic extern `{}` fixed arg {}", ext_name, i),
                            expected: expected.clone(),
                            got,
                        });
                    }
                }
                for (i, arg) in args.iter().enumerate().skip(params.len()) {
                    let got = typecheck_expr(arg, locals, cache)?;
                    let ok = matches!(&got, Type::Builtin(b) if b == "Int" || b == "Ptr");
                    if !ok {
                        return Err(TypeError::TypeMismatch {
                            context: format!(
                                "variadic extern `{}` variadic arg {} (must be Int or Ptr)",
                                ext_name, i
                            ),
                            expected: Type::Builtin("Ptr".to_owned()),
                            got,
                        });
                    }
                }
                return Ok(ret);
            }
        }
    }

    // ---- General path ----
    let callee_ty = typecheck_expr(callee, locals, cache)?;
    let (param_tys, ret_ty) = match &callee_ty {
        Type::FnType { params, ret } => (params.clone(), (**ret).clone()),
        _ => return Err(TypeError::ExpectedFn { got: callee_ty }),
    };
    if param_tys.len() != args.len() {
        let what = describe_callee(callee);
        return Err(TypeError::ArityMismatch {
            what,
            expected: param_tys.len(),
            got: args.len(),
        });
    }

    // If any declared param/ret contains a TypeVar, the callee is
    // generic. Unify the actual arg types against the declared params
    // to recover the instantiation, then substitute ret.
    let n_typevars =
        max_type_var(&param_tys, &ret_ty);
    let is_generic_callee = n_typevars > 0;
    let mut subst: Vec<Option<Type>> = vec![None; n_typevars as usize];

    for (i, (arg, expected)) in args.iter().zip(param_tys.iter()).enumerate() {
        let got = typecheck_expr(arg, locals, cache)?;
        if is_generic_callee {
            if let Err((d, a)) = unify(expected, &got, &mut subst) {
                return Err(TypeError::TypeMismatch {
                    context: format!(
                        "{} arg {} (could not unify generic param)",
                        describe_callee(callee),
                        i
                    ),
                    expected: d,
                    got: a,
                });
            }
        } else if !types_equiv(&got, expected) {
            return Err(TypeError::TypeMismatch {
                context: format!("{} arg {}", describe_callee(callee), i),
                expected: expected.clone(),
                got,
            });
        }
    }

    if is_generic_callee {
        // Bake in the substitution; any unbound type-vars remain.
        let concrete_subst: Vec<Type> = subst
            .into_iter()
            .enumerate()
            .map(|(i, o)| o.unwrap_or(Type::TypeVar(i as u32)))
            .collect();
        Ok(substitute(&ret_ty, &concrete_subst))
    } else {
        Ok(ret_ty)
    }
}

fn describe_callee(callee: &Expr) -> String {
    match callee {
        Expr::TopRef(h) => format!("call to {}", h),
        Expr::BuiltinRef(name) => format!("call to builtin {}", name),
        _ => "call".to_owned(),
    }
}

/// Check a pattern against a scrutinee type, pushing any bindings the
/// pattern introduces onto `locals`. Returns the number of bindings
/// pushed so the caller can pop them after type-checking the arm body.
///
/// `instantiation` is the type arguments applied to a generic enum
/// scrutinee, in positional `TypeVar` order. Empty for monomorphic.
fn typecheck_pattern(
    pat: &Pattern,
    scrutinee_ty: &Type,
    cache: &TypeCache,
    locals: &mut Vec<Type>,
    instantiation: &[Type],
) -> Result<usize, TypeError> {
    match pat {
        Pattern::Wildcard => Ok(0),
        Pattern::Var => {
            locals.push(scrutinee_ty.clone());
            Ok(1)
        }
        Pattern::Enum {
            enum_ref,
            variant_index,
            payload,
        } => {
            // Scrutinee must reference the same enum (either as a bare
            // TypeRef or as Apply(TypeRef, ...)).
            let scrut_enum = match scrutinee_ty {
                Type::TypeRef(h) => *h,
                Type::Apply(head, _) => match head.as_ref() {
                    Type::TypeRef(h) => *h,
                    _ => {
                        return Err(TypeError::ExpectedEnum {
                            got: scrutinee_ty.clone(),
                        });
                    }
                },
                _ => {
                    return Err(TypeError::ExpectedEnum {
                        got: scrutinee_ty.clone(),
                    });
                }
            };
            if scrut_enum != *enum_ref {
                return Err(TypeError::ExpectedEnum {
                    got: scrutinee_ty.clone(),
                });
            }
            let scheme = cache.get(enum_ref).ok_or(TypeError::UnknownEnum(*enum_ref))?;
            let variants = scheme.as_enum().ok_or(TypeError::UnknownEnum(*enum_ref))?;
            let i = *variant_index as usize;
            if i >= variants.len() {
                return Err(TypeError::VariantIndexOutOfRange {
                    enum_ref: *enum_ref,
                    index: *variant_index,
                    variant_count: variants.len(),
                });
            }
            let variant_payload = variants[i].1.clone();
            match (variant_payload, payload) {
                (None, None) => Ok(0),
                (Some(payload_ty), Some(sub)) => {
                    // Substitute the variant's declared payload type
                    // using the scrutinee's instantiation, so a
                    // `Result<Int, Failure>::Ok(x)` binds x: Int rather
                    // than x: TypeVar(0).
                    let inst_payload = substitute(&payload_ty, instantiation);
                    // Nested pattern continues with no further
                    // instantiation; we don't support generic structs
                    // / enums nested inside patterns yet.
                    typecheck_pattern(sub, &inst_payload, cache, locals, &[])
                }
                (None, Some(_)) => Err(TypeError::VariantPayloadShapeMismatch {
                    enum_ref: *enum_ref,
                    variant_index: *variant_index,
                    pattern_has_payload: true,
                    variant_has_payload: false,
                }),
                (Some(_), None) => Err(TypeError::VariantPayloadShapeMismatch {
                    enum_ref: *enum_ref,
                    variant_index: *variant_index,
                    pattern_has_payload: false,
                    variant_has_payload: true,
                }),
            }
        }
    }
}

// =============================================================================
// Module-level entry point
// =============================================================================

/// Reports the cache behaviour of a `typecheck_module` call. The key
/// invariant: re-running on the same module yields `newly_typed == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeCheckReport {
    pub newly_typed: usize,
    pub cache_hits: usize,
    pub total: usize,
}

/// Typecheck all defs in a resolved module, consulting and updating the
/// cache. The cost of this call is proportional to the number of defs
/// whose hashes are NOT yet in the cache; everything else is a cache hit.
pub fn typecheck_module(
    rm: &ResolvedModule,
    cache: &mut TypeCache,
) -> Result<TypeCheckReport, TypeError> {
    // Make extern signatures visible before body-typechecking — bodies
    // may call externs and need their declared signature.
    cache.register_externs(&rm.externs);

    // Identify spawn wrappers (defs whose body forwards directly to
    // `core/thread.spawn`) so calls to them require a mobile thunk.
    {
        let mut wrappers = std::collections::HashSet::new();
        for rd in &rm.defs {
            if let Def::Fn { body, .. } = &rd.def {
                if let Expr::Call(callee, _) = body {
                    if matches!(callee.as_ref(), Expr::BuiltinRef(n) if n == "core/thread.spawn")
                    {
                        wrappers.insert(rd.hash);
                    }
                }
            }
        }
        cache.register_spawn_wrappers(wrappers);
    }

    let mut newly_typed = 0usize;
    let mut cache_hits = 0usize;

    // Pre-pass: insert provisional schemes for EVERY def whose hash
    // isn't yet in the cache. This makes:
    //   - recursive + mutually-recursive fn calls work (a body
    //     referencing `TopRef(my_own_hash)` or `TopRef(scc_peer_hash)`
    //     finds the scheme during the body check).
    //   - recursive type defs work (struct `IntListCell { tail: IntList }`
    //     references `IntList`'s hash; both need to be in the cache
    //     before either's well-formedness check runs).
    //
    // Struct/Enum provisional schemes carry their final field/variant
    // types — `typecheck_def` will re-insert the same data (with
    // updated `wire`); idempotent. Fn provisional schemes carry their
    // declared param/ret; body check refines `wire`.
    let mut needs_check: Vec<&ResolvedDef> = Vec::new();
    for rdef in &rm.defs {
        if cache.contains(&rdef.hash) {
            cache_hits += 1;
            continue;
        }
        match &rdef.def {
            crate::ast::Def::Fn { params, ret, .. } => {
                cache.insert(
                    rdef.hash,
                    TypeScheme::Fn {
                        params: params.clone(),
                        ret: ret.clone(),
                        wire: false,
                    },
                );
            }
            crate::ast::Def::Struct {
                type_params,
                fields,
            } => {
                cache.insert(
                    rdef.hash,
                    TypeScheme::Struct {
                        type_params: *type_params,
                        fields: fields.clone(),
                        wire: false,
                    },
                );
            }
            crate::ast::Def::Enum {
                type_params,
                variants,
            } => {
                cache.insert(
                    rdef.hash,
                    TypeScheme::Enum {
                        type_params: *type_params,
                        variants: variants.clone(),
                        wire: false,
                    },
                );
            }
            crate::ast::Def::State { ty, .. } => {
                cache.insert(rdef.hash, TypeScheme::State { ty: ty.clone() });
            }
        }
        needs_check.push(rdef);
    }

    for rdef in needs_check {
        let scheme = typecheck_def(&rdef.def, cache)?;
        cache.insert(rdef.hash, scheme);
        newly_typed += 1;
    }

    Ok(TypeCheckReport {
        newly_typed,
        cache_hits,
        total: rm.defs.len(),
    })
}

// =============================================================================
// Canonical encoding of TypeScheme (for on-disk persistence).
//
// Mirrors the conventions of `codec.rs`:
//
//   - All integers big-endian, fixed-width.
//   - Sums are tagged with a length-prefixed ASCII name so adding new
//     variants later doesn't invalidate stored bytes.
//   - Optionals: u8 0 (None) or 1 (Some) + payload.
//   - Strings are length-prefixed UTF-8.
//   - Each `Type` field is stored via codec::encode_type as a
//     length-prefixed sub-blob; this avoids reaching into codec's
//     internal Reader/Writer and keeps the type-cache codec
//     self-contained.
//
// Layout:
//
//   TypeScheme:
//     tag: str        ("Fn" | "Struct" | "Enum")
//     wire: u8        (0 or 1)
//     ...payload by variant...
//
//   Fn payload:
//     n_params: u32
//     params[i]: length-prefixed encoded Type   × n_params
//     ret:       length-prefixed encoded Type
//
//   Struct payload:
//     n_fields: u32
//     fields[i]: (name: str, ty: length-prefixed Type)   × n_fields
//
//   Enum payload:
//     n_variants: u32
//     variants[i]: (name: str, payload: option<length-prefixed Type>)   × n_variants
// =============================================================================

#[derive(Debug)]
pub enum SchemeCodecError {
    UnexpectedEof,
    BadUtf8,
    BadBool(u8),
    BadOption(u8),
    UnknownTag { kind: &'static str, tag: String },
    TrailingBytes(usize),
    TypeDecode(crate::codec::DecodeError),
}

impl core::fmt::Display for SchemeCodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SchemeCodecError::UnexpectedEof => f.write_str("unexpected end of input"),
            SchemeCodecError::BadUtf8 => f.write_str("invalid utf-8 in string field"),
            SchemeCodecError::BadBool(b) => write!(f, "bool must be 0 or 1, got {}", b),
            SchemeCodecError::BadOption(b) => write!(f, "option must be 0 or 1, got {}", b),
            SchemeCodecError::UnknownTag { kind, tag } => {
                write!(f, "unknown {} tag {:?}", kind, tag)
            }
            SchemeCodecError::TrailingBytes(n) => {
                write!(f, "{} trailing bytes after decode", n)
            }
            SchemeCodecError::TypeDecode(e) => write!(f, "embedded Type decode: {}", e),
        }
    }
}

impl std::error::Error for SchemeCodecError {}

impl From<crate::codec::DecodeError> for SchemeCodecError {
    fn from(e: crate::codec::DecodeError) -> Self {
        SchemeCodecError::TypeDecode(e)
    }
}

pub fn encode_scheme(s: &TypeScheme) -> Vec<u8> {
    let mut buf = Vec::new();
    match s {
        TypeScheme::Fn { params, ret, wire } => {
            write_str(&mut buf, "Fn");
            buf.push(*wire as u8);
            let n: u32 = params.len().try_into().expect("too many params");
            buf.extend_from_slice(&n.to_be_bytes());
            for p in params {
                write_blob(&mut buf, &crate::codec::encode_type(p));
            }
            write_blob(&mut buf, &crate::codec::encode_type(ret));
        }
        TypeScheme::Struct {
            type_params,
            fields,
            wire,
        } => {
            write_str(&mut buf, "Struct");
            buf.push(*wire as u8);
            buf.extend_from_slice(&type_params.to_be_bytes());
            let n: u32 = fields.len().try_into().expect("too many fields");
            buf.extend_from_slice(&n.to_be_bytes());
            for (name, ty) in fields {
                write_str(&mut buf, name);
                write_blob(&mut buf, &crate::codec::encode_type(ty));
            }
        }
        TypeScheme::Enum {
            type_params,
            variants,
            wire,
        } => {
            write_str(&mut buf, "Enum");
            buf.push(*wire as u8);
            buf.extend_from_slice(&type_params.to_be_bytes());
            let n: u32 = variants.len().try_into().expect("too many variants");
            buf.extend_from_slice(&n.to_be_bytes());
            for (name, payload) in variants {
                write_str(&mut buf, name);
                match payload {
                    None => buf.push(0),
                    Some(t) => {
                        buf.push(1);
                        write_blob(&mut buf, &crate::codec::encode_type(t));
                    }
                }
            }
        }
        TypeScheme::State { ty } => {
            write_str(&mut buf, "State");
            // Uniform layout: a wire byte follows every tag. A state cell
            // never crosses the wire by value, so it is always 0.
            buf.push(0);
            write_blob(&mut buf, &crate::codec::encode_type(ty));
        }
    }
    buf
}

pub fn decode_scheme(bytes: &[u8]) -> Result<TypeScheme, SchemeCodecError> {
    let mut r = SchemeReader::new(bytes);
    let tag = r.read_str()?;
    let wire_byte = r.read_u8()?;
    let wire = match wire_byte {
        0 => false,
        1 => true,
        other => return Err(SchemeCodecError::BadBool(other)),
    };
    let scheme = match tag.as_str() {
        "Fn" => {
            let n = r.read_u32()? as usize;
            let mut params = Vec::with_capacity(n);
            for _ in 0..n {
                let blob = r.read_blob()?;
                params.push(crate::codec::decode_type(blob)?);
            }
            let ret_blob = r.read_blob()?;
            let ret = crate::codec::decode_type(ret_blob)?;
            TypeScheme::Fn { params, ret, wire }
        }
        "Struct" => {
            let type_params = r.read_u32()?;
            let n = r.read_u32()? as usize;
            let mut fields = Vec::with_capacity(n);
            for _ in 0..n {
                let name = r.read_str()?;
                let blob = r.read_blob()?;
                fields.push((name, crate::codec::decode_type(blob)?));
            }
            TypeScheme::Struct {
                type_params,
                fields,
                wire,
            }
        }
        "Enum" => {
            let type_params = r.read_u32()?;
            let n = r.read_u32()? as usize;
            let mut variants = Vec::with_capacity(n);
            for _ in 0..n {
                let name = r.read_str()?;
                let kind = r.read_u8()?;
                let payload = match kind {
                    0 => None,
                    1 => {
                        let blob = r.read_blob()?;
                        Some(crate::codec::decode_type(blob)?)
                    }
                    other => return Err(SchemeCodecError::BadOption(other)),
                };
                variants.push((name, payload));
            }
            TypeScheme::Enum {
                type_params,
                variants,
                wire,
            }
        }
        "State" => {
            // The wire byte was already consumed above (and ignored).
            let blob = r.read_blob()?;
            TypeScheme::State {
                ty: crate::codec::decode_type(blob)?,
            }
        }
        _ => {
            return Err(SchemeCodecError::UnknownTag {
                kind: "TypeScheme",
                tag,
            });
        }
    };
    r.finish()?;
    Ok(scheme)
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len: u32 = bytes.len().try_into().expect("string too long");
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(bytes);
}

fn write_blob(buf: &mut Vec<u8>, bs: &[u8]) {
    let len: u32 = bs.len().try_into().expect("blob too long");
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(bs);
}

struct SchemeReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SchemeReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        SchemeReader { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], SchemeCodecError> {
        if self.pos + n > self.bytes.len() {
            return Err(SchemeCodecError::UnexpectedEof);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_u8(&mut self) -> Result<u8, SchemeCodecError> {
        Ok(self.take(1)?[0])
    }
    fn read_u32(&mut self) -> Result<u32, SchemeCodecError> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn read_str(&mut self) -> Result<String, SchemeCodecError> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| SchemeCodecError::BadUtf8)
    }
    fn read_blob(&mut self) -> Result<&'a [u8], SchemeCodecError> {
        let len = self.read_u32()? as usize;
        self.take(len)
    }
    fn finish(self) -> Result<(), SchemeCodecError> {
        let trailing = self.bytes.len() - self.pos;
        if trailing == 0 {
            Ok(())
        } else {
            Err(SchemeCodecError::TrailingBytes(trailing))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

// =============================================================================
// typecheck_cone — recheck only the changed hashes (edit-layer support)
// =============================================================================

/// A single unresolved item produced by an `update` whose type changed: a
/// dependent definition that no longer typechecks against the new signature.
/// Returned in `EditResult.todos` so an agent (or human) can resolve it with
/// further `update`s, mirroring Unison's `todo` worklist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Todo {
    /// The (new) hash of the dependent that failed to typecheck.
    pub hash: Hash,
    /// The name the failing hash resolves through, if any.
    pub name: Option<String>,
    /// Human-readable description of the type error.
    pub message: String,
}

/// Typecheck only the given `changed` hashes against the cached schemes of
/// everything else. Used by the edit layer after an `update` re-points a
/// dependency cone: each rewritten dependent is rechecked in topological
/// order (dependencies-before-dependents), so by the time a def is checked,
/// the new schemes of everything it depends on are already in the cache.
///
/// Unlike `typecheck_module`, this never aborts the whole batch on the first
/// error: a def that fails to typecheck is recorded as a [`Todo`] and the
/// remaining defs are still checked (so the worklist is complete). A def that
/// *does* typecheck has its fresh scheme inserted into `cache`, which is what
/// makes the topological ordering correct for later defs in the same call.
///
/// `name_of` recovers a display name for a hash (e.g. from the codebase
/// namespace); pass a closure that returns `None` for unnamed intermediates.
pub fn typecheck_cone<F>(
    cb: &crate::codebase::Codebase,
    changed: &[Hash],
    cache: &mut TypeCache,
    mut name_of: F,
) -> Result<Vec<Todo>, crate::codebase::CodebaseError>
where
    F: FnMut(&Hash) -> Option<String>,
{
    let mut todos: Vec<Todo> = Vec::new();
    for h in changed {
        let def = cb.load_def(h)?;
        match typecheck_def(&def, cache) {
            Ok(scheme) => {
                // Insert so dependents later in `changed` see the new type.
                cache.insert(*h, scheme);
            }
            Err(e) => {
                todos.push(Todo {
                    hash: *h,
                    name: name_of(h),
                    message: e.to_string(),
                });
            }
        }
    }
    Ok(todos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;

    fn tc(src: &str) -> (ResolvedModule, TypeCache, TypeCheckReport) {
        let m = parse_module(src).unwrap();
        let rm = resolve_module(&m).unwrap();
        let mut cache = TypeCache::new();
        let report = typecheck_module(&rm, &mut cache).expect("typecheck succeeds");
        (rm, cache, report)
    }

    fn tc_err(src: &str) -> TypeError {
        let m = parse_module(src).unwrap();
        let rm = resolve_module(&m).unwrap();
        let mut cache = TypeCache::new();
        typecheck_module(&rm, &mut cache).expect_err("typecheck must fail")
    }

    #[test]
    fn spawn_rejects_atom_capture() {
        let with_std = |p: &str| format!("{}\n{}", crate::stdlib::SOURCE, p);
        // Capturing an Atom in a spawned thunk shares mutable state across
        // threads — rejected by the share-nothing mobility check.
        let err = tc_err(&with_std(
            "def bad() -> Int = { let a = atom(0); join(spawn(|| deref(a))) }",
        ));
        match err {
            TypeError::Unsupported(m) => {
                assert!(m.contains("Atom"), "expected an Atom-capture error, got: {m}")
            }
            other => panic!("expected Unsupported(Atom capture), got {other:?}"),
        }
        // A spawn capturing only a value (or nothing) is fine.
        let _ = tc(&with_std(
            "def good() -> Int = { let n = 41; join(spawn(|| n + 1)) }",
        ));
        // `spawn_shared` is the opt-out: capturing an Atom is allowed
        // (you deliberately share the lock-free cell across threads).
        let _ = tc(&with_std(
            "def ok_shared() -> Int = { let a = atom(0); join(spawn_shared(|| deref(a))) }",
        ));
    }

    #[test]
    fn int_arith_typechecks() {
        let (rm, cache, report) = tc("def f(x: Int) -> Int = x * 2 + 1");
        assert_eq!(report.newly_typed, 1);
        assert_eq!(report.cache_hits, 0);
        assert_eq!(report.total, 1);
        let h = rm.defs[0].hash;
        let scheme = cache.get(&h).unwrap();
        let (params, ret) = scheme.as_fn().expect("Fn scheme");
        assert_eq!(params.len(), 1);
        assert!(matches!(&params[0], Type::Builtin(n) if n == "Int"));
        assert!(matches!(ret, Type::Builtin(n) if n == "Int"));
        assert!(scheme.is_wire());
    }

    #[test]
    fn arity_mismatch_caught() {
        // `double` takes one Int; calling with two args must fail.
        let err = tc_err(
            "def double(x: Int) -> Int = x * 2
             def bad(x: Int) -> Int = double(x, x)",
        );
        match err {
            TypeError::ArityMismatch { expected, got, .. } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("expected ArityMismatch, got {:?}", other),
        }
    }

    #[test]
    fn type_mismatch_caught() {
        // Field access on an Int should fail with ExpectedStruct (via Field
        // node); but the resolver doesn't synthesise that. Instead, build
        // a StructNew where the first field type is mismatched.
        let err = tc_err(
            "struct Point { x: Int, y: Int }
             def origin() -> Point = Point { x: 1, y: 2 }
             def bad(p: Point) -> Int = origin",
        );
        // `origin` is a fn (the TopRef returns FnType), so the body of
        // `bad` (which must be Int) has type FnType — that's a type
        // mismatch.
        match err {
            TypeError::TypeMismatch { .. } => {}
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn unknown_field_caught() {
        // Use a hand-built AST: the parser/resolver won't synthesise an
        // out-of-range field index since they reject bad names. Patch the
        // AST after resolution.
        let src = "struct Point { x: Int, y: Int }
                   def get(p: Point) -> Int = p.x";
        let m = parse_module(src).unwrap();
        let mut rm = resolve_module(&m).unwrap();
        // Find `get`, mutate its body's `index` to be out-of-range.
        let get_def = rm
            .defs
            .iter_mut()
            .find(|d| d.name == "get")
            .expect("get exists");
        if let Def::Fn { body, .. } = &mut get_def.def {
            if let Expr::Field { index, .. } = body {
                *index = 99; // out of range
            } else {
                panic!("expected Field body");
            }
        }
        let mut cache = TypeCache::new();
        let err = typecheck_module(&rm, &mut cache).expect_err("should fail");
        match err {
            TypeError::FieldIndexOutOfRange {
                index, field_count, ..
            } => {
                assert_eq!(index, 99);
                assert_eq!(field_count, 2);
            }
            other => panic!("expected FieldIndexOutOfRange, got {:?}", other),
        }
    }

    #[test]
    fn match_arms_must_agree() {
        // Build an enum with two variants, then make a match where the
        // arms return different types. We need a hand-edit because the
        // resolver picks the result type from the first arm.
        let src = "enum Shape { Circle(Int), Square(Int) }
                   def area(s: Shape) -> Int = match s {
                       Shape::Circle(r) => r * r,
                       Shape::Square(s) => s * s,
                   }";
        let (rm, cache, _) = tc(src);
        // First, ensure it typechecks fine.
        assert!(cache.get(&rm.get("area").unwrap().hash).is_some());

        // Now mutate the second arm body to a Bool literal.
        let m = parse_module(src).unwrap();
        let mut rm = resolve_module(&m).unwrap();
        let area = rm
            .defs
            .iter_mut()
            .find(|d| d.name == "area")
            .expect("area");
        if let Def::Fn { body, .. } = &mut area.def {
            if let Expr::Match { arms, .. } = body {
                arms[1].body = Expr::BoolLit(true);
            } else {
                panic!("expected Match");
            }
        }
        let mut cache = TypeCache::new();
        let err = typecheck_module(&rm, &mut cache).expect_err("arms disagree");
        match err {
            TypeError::MatchArmsDisagree { .. } => {}
            other => panic!("expected MatchArmsDisagree, got {:?}", other),
        }
    }

    #[test]
    fn at_special_case_typechecks() {
        // `at(node, || 42)` should typecheck — node is a struct, thunk
        // is fn() -> Int. The result type is `Result<Int, Failure>`.
        let src = "struct Node { x: Int }
                   enum Failure {
                       Unreachable(Node),
                       Crashed(Node),
                       CodeMissing(Node),
                       Cancelled(Node),
                   }
                   enum Result<T, E> { Ok(T), Err(E) }
                   def root() -> Node = Node { x: 0 }
                   def run(n: Node) -> Result<Int, Failure> = at(n, || 42)";
        let (rm, cache, report) = tc(src);
        assert_eq!(report.newly_typed, 5);
        let run = rm.get("run").unwrap();
        assert!(cache.get(&run.hash).is_some());

        // `at(n, || node)` returns `Result<Node, Failure>` — non-Int
        // thunks are allowed; the wire ships the heap pointer.
        let non_int = "struct Node { x: Int }
                       enum Failure {
                           Unreachable(Node),
                           Crashed(Node),
                           CodeMissing(Node),
                           Cancelled(Node),
                       }
                       enum Result<T, E> { Ok(T), Err(E) }
                       def root() -> Node = Node { x: 0 }
                       def run(n: Node) -> Result<Node, Failure> = at(n, || n)";
        let (_rm, _cache, _report) = tc(non_int);

        // `at(n, foo)` where `foo` is an Int (not a thunk) must fail.
        let bad = "struct Node { x: Int }
                   enum Failure {
                       Unreachable(Node),
                       Crashed(Node),
                       CodeMissing(Node),
                       Cancelled(Node),
                   }
                   enum Result<T, E> { Ok(T), Err(E) }
                   def run(n: Node) -> Result<Int, Failure> = at(n, 42)";
        let m = parse_module(bad).unwrap();
        let rm = resolve_module(&m).unwrap();
        let mut cache = TypeCache::new();
        let err = typecheck_module(&rm, &mut cache).expect_err("non-thunk 2nd arg rejected");
        match err {
            TypeError::TypeMismatch { context, .. } => {
                assert!(
                    context.contains("core/net.at second argument"),
                    "{}",
                    context
                );
            }
            other => panic!("expected TypeMismatch on at thunk, got {:?}", other),
        }
    }

    #[test]
    fn at_requires_result_and_failure_in_scope() {
        // Missing both Result and Failure (and Node) — resolve errors.
        let bad = "def run(n: Int) -> Int = at(n, || 0)";
        let m = parse_module(bad).unwrap();
        let err = resolve_module(&m).expect_err("at requires bindings");
        match err {
            crate::resolve::ResolveError::AtRequiresBinding { missing, .. } => {
                assert!(missing.contains("Node"), "{}", missing);
            }
            other => panic!("expected AtRequiresBinding, got {:?}", other),
        }
    }

    #[test]
    fn cache_hits_on_rerun() {
        // THE HEADLINE: re-typechecking a module hits the cache for every def.
        let src = "def double(x: Int) -> Int = x * 2
                   def triple(x: Int) -> Int = x * 3
                   def six(x: Int) -> Int = double(triple(x))";
        let m = parse_module(src).unwrap();
        let rm = resolve_module(&m).unwrap();
        let mut cache = TypeCache::new();

        let r1 = typecheck_module(&rm, &mut cache).unwrap();
        assert_eq!(r1.newly_typed, 3);
        assert_eq!(r1.cache_hits, 0);
        assert_eq!(r1.total, 3);

        let r2 = typecheck_module(&rm, &mut cache).unwrap();
        assert_eq!(r2.newly_typed, 0);
        assert_eq!(r2.cache_hits, 3);
        assert_eq!(r2.total, 3);

        eprintln!(
            "cache_hits_on_rerun: run1={:?}, run2={:?}",
            r1, r2
        );
    }

    #[test]
    fn adding_one_def_typechecks_only_that_one() {
        let src_a = "def double(x: Int) -> Int = x * 2
                     def triple(x: Int) -> Int = x * 3";
        let src_b = "def double(x: Int) -> Int = x * 2
                     def triple(x: Int) -> Int = x * 3
                     def quad(x: Int) -> Int = double(double(x))";
        let m_a = parse_module(src_a).unwrap();
        let m_b = parse_module(src_b).unwrap();
        let rm_a = resolve_module(&m_a).unwrap();
        let rm_b = resolve_module(&m_b).unwrap();

        let mut cache = TypeCache::new();
        let r_a = typecheck_module(&rm_a, &mut cache).unwrap();
        assert_eq!(r_a.newly_typed, 2);
        assert_eq!(r_a.cache_hits, 0);

        let r_b = typecheck_module(&rm_b, &mut cache).unwrap();
        assert_eq!(r_b.newly_typed, 1, "only `quad` is new");
        assert_eq!(r_b.cache_hits, 2, "double + triple are cached");
        assert_eq!(r_b.total, 3);

        eprintln!(
            "adding_one_def_typechecks_only_that_one: rm_a={:?}, rm_b={:?}",
            r_a, r_b
        );
    }

    #[test]
    fn wire_is_true_for_all_v1_types() {
        let src = "struct Point { x: Int, y: Int }
                   enum Shape { Circle(Int), Square(Int) }
                   def make() -> Point = Point { x: 1, y: 2 }
                   def kind(s: Shape) -> Int = match s {
                       Shape::Circle(r) => r,
                       Shape::Square(s) => s,
                   }";
        let (rm, cache, _) = tc(src);
        for d in &rm.defs {
            let s = cache.get(&d.hash).expect("scheme");
            assert!(s.is_wire(), "def {} should be Wire", d.name);
        }
    }

    #[test]
    fn struct_with_struct_field() {
        let src = "struct Inner { v: Int }
                   struct Outer { inner: Inner, tag: Int }
                   def make(i: Inner) -> Outer = Outer { inner: i, tag: 0 }
                   def peek(o: Outer) -> Int = o.tag";
        let (rm, cache, _) = tc(src);
        let outer = rm.get("Outer").unwrap();
        let scheme = cache.get(&outer.hash).unwrap();
        let fields = scheme.as_struct().unwrap();
        assert_eq!(fields.len(), 2);
        // First field is a TypeRef to Inner.
        let inner = rm.get("Inner").unwrap();
        assert_eq!(fields[0].1, Type::TypeRef(inner.hash));
    }

    #[test]
    fn unknown_builtin_caught() {
        // Hand-build a def that references a non-existent builtin.
        let bogus = Def::Fn {
            is_local: false,
            type_params: 0,
            params: vec![int_t()],
            ret: int_t(),
            body: Expr::Call(
                Box::new(Expr::BuiltinRef("core/totally.fake".to_owned())),
                vec![Expr::LocalVar(0), Expr::IntLit(1)],
            ),
        };
        let cache = TypeCache::new();
        let err = typecheck_def(&bogus, &cache).expect_err("bogus builtin");
        match err {
            TypeError::UnknownBuiltin(name) => {
                assert_eq!(name, "core/totally.fake");
            }
            other => panic!("expected UnknownBuiltin, got {:?}", other),
        }
    }

    #[test]
    fn unknown_top_ref_caught() {
        // A TopRef to a hash that isn't in cache.
        let phantom = Hash([0xab; 32]);
        let bogus = Def::Fn {
            is_local: false,
            type_params: 0,
            params: vec![],
            ret: int_t(),
            body: Expr::Call(Box::new(Expr::TopRef(phantom)), vec![]),
        };
        let cache = TypeCache::new();
        let err = typecheck_def(&bogus, &cache).expect_err("phantom topref");
        match err {
            TypeError::UnknownTopRef(h) => assert_eq!(h, phantom),
            other => panic!("expected UnknownTopRef, got {:?}", other),
        }
    }

    #[test]
    fn lambda_typechecks_and_is_wire() {
        let src = "def make() -> fn(Int) -> Int = |x: Int| x + 1";
        let (rm, cache, _) = tc(src);
        let scheme = cache.get(&rm.defs[0].hash).unwrap();
        let (params, ret) = scheme.as_fn().unwrap();
        assert!(params.is_empty());
        assert!(matches!(ret, Type::FnType { .. }));
        assert!(scheme.is_wire());
    }

    #[test]
    fn let_binding_typechecks() {
        let src = "def f(x: Int) -> Int = {
                       let y = x * 2;
                       y + 1
                   }";
        let (_rm, _cache, report) = tc(src);
        assert_eq!(report.newly_typed, 1);
    }

    #[test]
    fn self_ref_is_explicit_error() {
        let phantom_body = Expr::SelfRef(7);
        let bogus = Def::Fn {
            is_local: false,
            type_params: 0,
            params: vec![],
            ret: int_t(),
            body: phantom_body,
        };
        let cache = TypeCache::new();
        let err = typecheck_def(&bogus, &cache).expect_err("self-ref must error");
        match err {
            TypeError::SelfRefInTypecheck { component_index } => {
                assert_eq!(component_index, 7);
            }
            other => panic!("expected SelfRefInTypecheck, got {:?}", other),
        }
    }

    #[test]
    fn generic_enum_constructor_inference() {
        // `Some(42)` should typecheck as `Apply(Option, [Int])` —
        // bottom-up inference from the payload arg.
        let (rm, cache, _) = tc(
            "enum Option<T> { Some(T), None }
             def t() -> Int = match Option::Some(42) { Option::Some(v) => v, Option::None => 0 }",
        );
        let h = rm.defs.iter().find(|d| d.name == "t").unwrap().hash;
        let scheme = cache.get(&h).unwrap();
        let (_, ret) = scheme.as_fn().unwrap();
        assert!(matches!(ret, Type::Builtin(n) if n == "Int"));
    }

    #[test]
    fn generic_struct_field_substitutes_typevars() {
        // `cell.head` for `cell : Pair<Int, Int>` projects as Int, not
        // TypeVar(0). If substitution were broken, the return type
        // wouldn't match `Int` and the def body would error.
        let (_, _, _) = tc(
            "struct Pair<A, B> { fst: A, snd: B }
             def first(p: Pair<Int, Int>) -> Int = p.fst
             def second(p: Pair<Int, Int>) -> Int = p.snd",
        );
    }

    #[test]
    fn recursive_struct_enum_typecheck() {
        // Mutually recursive `IntListCell` ↔ `IntList`. Pre-pass must
        // register both before well-formedness checks run.
        let (_, _, report) = tc(
            "struct Cell { head: Int, tail: LL }
             enum LL { Cons(Cell), Nil }
             def len(xs: LL) -> Int =
                 match xs { LL::Cons(c) => 1 + len(c.tail), LL::Nil => 0 }",
        );
        assert_eq!(report.newly_typed, 3);
    }

    #[test]
    fn generic_arm_typecheck_with_unknown_payload_compatible() {
        // `Some(v)` arm body produces Apply(Option, [TypeVar(0)]) (we
        // learned the payload type from `v` which IS TypeVar(0)). None
        // arm body is `b: Option<T>` also Apply(Option, [TypeVar(0)]).
        // Both arms agree.
        let _ = tc(
            "enum Option<T> { Some(T), None }
             def opt_or<T>(a: Option<T>, b: Option<T>) -> Option<T> =
                 match a { Option::Some(v) => Option::Some(v), Option::None => b }",
        );
    }

    #[test]
    fn nullary_variant_flows_into_concrete_position() {
        // `Nil` typechecks to Apply(List, [TypeVar(0)]) with no
        // instantiation. Passing it where `List<Int>` is expected
        // must be accepted via the wildcard rule in `unify`.
        let _ = tc(
            "struct LCell<T> { head: T, tail: List<T> }
             enum List<T> { LCons(LCell<T>), LNil }
             def make_one(x: Int) -> List<Int> =
                 List::LCons(LCell { head: x, tail: List::LNil })",
        );
    }

    #[test]
    fn struct_field_type_mismatch_caught() {
        // Pair { fst: 1, snd: \"x\" } should fail — snd is declared Int
        // but the arg is a String.
        let err = tc_err(
            "struct Pair { fst: Int, snd: Int }
             def t() -> Pair = Pair { fst: 1, snd: \"oops\" }",
        );
        match err {
            TypeError::TypeMismatch { .. } => {}
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn enum_payload_type_mismatch_caught() {
        // `Some(\"x\")` typed as `Option<Int>` should fail because the
        // payload is String, not Int.
        let err = tc_err(
            "enum Option<T> { Some(T), None }
             def t() -> Option<Int> = Option::Some(\"oops\")",
        );
        match err {
            TypeError::TypeMismatch { .. } => {}
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }
}
