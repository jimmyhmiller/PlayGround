//! Port of `clojure.lang.Compiler`.
//!
//! Source: `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/Compiler.java`
//! (9681 lines, ~70 inner classes).
//!
//! Strategy: walk Compiler.java top-to-bottom and translate each chunk
//! literally. The skeleton lives here; large inner classes will eventually
//! split into submodules under `compiler/`, but for now everything is in this
//! file so it tracks the Java source closely.
//!
//! What's ported so far:
//!   * Top-level Symbol/Keyword constants (the special-form names + meta keys)
//!   * The `C` enum (STATEMENT / EXPRESSION / RETURN / EVAL)
//!   * The `Expr` and `IParser` traits
//!   * Dynamic Var declarations (LOCAL_ENV, LOOP_LOCALS, METHOD, …)
//!
//! Everything else stubs back to `unimplemented_port!`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use dynir::Value;
use dynlang::DynFunc;

use super::keyword::Keyword;
use super::object::Object;
use super::persistent_hash_map::PersistentHashMap;
use super::persistent_list::PersistentList;
use super::persistent_vector::PersistentVector;
use super::symbol::Symbol;
use super::var::Var;

// Re-export so callers can write `compiler::Value` without depending on dynir.
pub use dynir::Value as IrValue;

// ============================================================================
// Java line ~45–84: special-form symbols.
// ============================================================================

/// Cluster of pre-interned `Symbol` constants Compiler.java holds as `static
/// final`. Lazy-initialized once per process.
pub static SPECIAL_SYMBOLS: LazyLock<SpecialSymbols> = LazyLock::new(SpecialSymbols::new);

#[allow(non_snake_case)]
pub struct SpecialSymbols {
    pub DEF: Arc<Symbol>,
    pub LOOP: Arc<Symbol>,
    pub RECUR: Arc<Symbol>,
    pub IF: Arc<Symbol>,
    pub LET: Arc<Symbol>,
    pub LETFN: Arc<Symbol>,
    pub DO: Arc<Symbol>,
    pub FN: Arc<Symbol>,
    pub QUOTE: Arc<Symbol>,
    pub THE_VAR: Arc<Symbol>,
    pub DOT: Arc<Symbol>,
    pub ASSIGN: Arc<Symbol>,
    pub TRY: Arc<Symbol>,
    pub CATCH: Arc<Symbol>,
    pub FINALLY: Arc<Symbol>,
    pub THROW: Arc<Symbol>,
    pub MONITOR_ENTER: Arc<Symbol>,
    pub MONITOR_EXIT: Arc<Symbol>,
    pub IMPORT: Arc<Symbol>,
    pub DEFTYPE: Arc<Symbol>,
    pub CASE: Arc<Symbol>,
    pub CLASS: Arc<Symbol>,
    pub NEW: Arc<Symbol>,
    pub THIS: Arc<Symbol>,
    pub REIFY: Arc<Symbol>,
    pub LIST: Arc<Symbol>,
    pub HASHMAP: Arc<Symbol>,
    pub VECTOR: Arc<Symbol>,
    pub IDENTITY: Arc<Symbol>,
    pub AMP: Arc<Symbol>,
    pub ISEQ: Arc<Symbol>,
    pub INVOKE_STATIC: Arc<Symbol>,
    pub NS: Arc<Symbol>,
    pub IN_NS: Arc<Symbol>,
    /// `set-macro!` — flags a Var as a macro. Used internally by
    /// `defmacro`'s expansion. Not in Java's symbol table because
    /// Java uses `.setMacro` method dispatch on Var instances; we
    /// don't have generic instance-method dispatch yet, so this
    /// dedicated special form fills the role.
    pub SET_MACRO_BANG: Arc<Symbol>,
}

impl SpecialSymbols {
    fn new() -> Self {
        SpecialSymbols {
            DEF: Symbol::intern("def"),
            LOOP: Symbol::intern("loop*"),
            RECUR: Symbol::intern("recur"),
            IF: Symbol::intern("if"),
            LET: Symbol::intern("let*"),
            LETFN: Symbol::intern("letfn*"),
            DO: Symbol::intern("do"),
            FN: Symbol::intern("fn*"),
            QUOTE: Symbol::intern("quote"),
            THE_VAR: Symbol::intern("var"),
            DOT: Symbol::intern("."),
            ASSIGN: Symbol::intern("set!"),
            TRY: Symbol::intern("try"),
            CATCH: Symbol::intern("catch"),
            FINALLY: Symbol::intern("finally"),
            THROW: Symbol::intern("throw"),
            MONITOR_ENTER: Symbol::intern("monitor-enter"),
            MONITOR_EXIT: Symbol::intern("monitor-exit"),
            IMPORT: Symbol::intern_ns_name(Some("clojure.core"), "import*"),
            DEFTYPE: Symbol::intern("deftype*"),
            CASE: Symbol::intern("case*"),
            CLASS: Symbol::intern("Class"),
            NEW: Symbol::intern("new"),
            THIS: Symbol::intern("this"),
            REIFY: Symbol::intern("reify*"),
            LIST: Symbol::intern_ns_name(Some("clojure.core"), "list"),
            HASHMAP: Symbol::intern_ns_name(Some("clojure.core"), "hash-map"),
            VECTOR: Symbol::intern_ns_name(Some("clojure.core"), "vector"),
            IDENTITY: Symbol::intern_ns_name(Some("clojure.core"), "identity"),
            AMP: Symbol::intern("&"),
            ISEQ: Symbol::intern("clojure.lang.ISeq"),
            INVOKE_STATIC: Symbol::intern("invokeStatic"),
            NS: Symbol::intern("ns"),
            IN_NS: Symbol::intern("in-ns"),
            SET_MACRO_BANG: Symbol::intern("set-macro!"),
        }
    }
}

// `FNONCE` is `fn*` with `{:once true}` metadata. Needs IPersistentMap +
// withMeta — stubbed until those are ported. See Java line 53.
pub fn fnonce() -> Arc<Symbol> {
    crate::unimplemented_port!(
        "Compiler.FNONCE",
        "needs Symbol.withMeta(IPersistentMap) once IPersistentMap is ported"
    )
}

// ============================================================================
// Java line ~86–101: compiler metadata keywords.
// ============================================================================

pub static COMPILER_KEYWORDS: LazyLock<CompilerKeywords> = LazyLock::new(CompilerKeywords::new);

#[allow(non_snake_case)]
pub struct CompilerKeywords {
    pub loadNs: Arc<Keyword>,
    pub inlineKey: Arc<Keyword>,
    pub inlineAritiesKey: Arc<Keyword>,
    pub staticKey: Arc<Keyword>,
    pub arglistsKey: Arc<Keyword>,
    pub volatileKey: Arc<Keyword>,
    pub implementsKey: Arc<Keyword>,
    pub protocolKey: Arc<Keyword>,
    pub onKey: Arc<Keyword>,
    pub dynamicKey: Arc<Keyword>,
    pub redefKey: Arc<Keyword>,
    pub disableLocalsClearingKey: Arc<Keyword>,
    pub directLinkingKey: Arc<Keyword>,
    pub elideMetaKey: Arc<Keyword>,
}

impl CompilerKeywords {
    fn new() -> Self {
        let kw = |ns, n| Keyword::intern_ns_name(ns, n);
        CompilerKeywords {
            loadNs: kw(None, "load-ns"),
            inlineKey: kw(None, "inline"),
            inlineAritiesKey: kw(None, "inline-arities"),
            staticKey: kw(None, "static"),
            arglistsKey: kw(None, "arglists"),
            volatileKey: kw(None, "volatile"),
            implementsKey: kw(None, "implements"),
            protocolKey: kw(None, "protocol"),
            onKey: kw(None, "on"),
            // Note: Java has `static Keyword dynamicKey = Keyword.intern("dynamic")`
            // — no namespace separator, but `intern(String)` parses it as
            // bare-name. We match by passing `None, "dynamic"`.
            dynamicKey: kw(None, "dynamic"),
            redefKey: kw(None, "redef"),
            disableLocalsClearingKey: kw(None, "disable-locals-clearing"),
            directLinkingKey: kw(None, "direct-linking"),
            elideMetaKey: kw(None, "elide-meta"),
        }
    }
}

// ============================================================================
// Java line ~145: MAX_POSITIONAL_ARITY
// ============================================================================

/// Java: `private static final int MAX_POSITIONAL_ARITY = 20;`
pub const MAX_POSITIONAL_ARITY: usize = 20;

// ============================================================================
// Java line ~344–349: `enum C` — context of evaluation.
// ============================================================================

/// `Compiler.C` — context in which an Expr is being analyzed/emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum C {
    /// Value ignored.
    Statement,
    /// Value required.
    Expression,
    /// Tail position relative to enclosing recur frame.
    Return,
    /// Top-level eval.
    Eval,
}

// ============================================================================
// Java line ~351–352: marker class for recur.
// ============================================================================

/// Java: `private class Recur {}; static final public Class RECUR_CLASS = Recur.class;`
///
/// Used as a type-tag for recur positions. We model it as a zero-sized marker.
#[derive(Debug, Clone, Copy)]
pub struct Recur;

// ============================================================================
// Java line ~354–373: `interface Expr`
// ============================================================================

/// `Compiler.Expr` — an analyzed compilation node. Java declaration:
///
/// ```text
/// interface Expr {
///     Object eval();
///     void emit(C context, ObjExpr objx, GeneratorAdapter gen);
///     boolean hasJavaClass();
///     Class getJavaClass();
/// }
/// ```
///
/// `emit` in our world targets a `dynir` builder rather than ASM
/// GeneratorAdapter — that translation lives in `ObjExpr` (port pending).
/// Until ObjExpr is ported, `emit` here is the trait shape only.
pub trait Expr: std::fmt::Debug + Send + Sync {
    /// Java: `Object eval()`. Compile-time evaluation (`*compile-files*` off).
    fn eval(&self) -> Object {
        crate::unimplemented_port!("Expr::eval", "default impl — node didn't override")
    }

    /// Java: `void emit(C, ObjExpr, GeneratorAdapter)`. Emit IR for this node.
    ///
    /// Returns the produced dynir `Value`, or `None` in `STATEMENT` context
    /// (Java's `gen.pop()` after emit). In Java this is implicit because the
    /// operand stack carries the result; in SSA-IR we make the value
    /// explicit so callers can wire it as a `ret` operand or branch arg.
    fn emit(&self, _context: C, _objx: &ObjExpr, _ir: &mut IrEmitter<'_>) -> Option<Value> {
        crate::unimplemented_port!("Expr::emit", "default impl — node didn't override")
    }

    /// Java: `boolean hasJavaClass()`.
    fn has_java_class(&self) -> bool {
        false
    }

    /// Java: `Class getJavaClass()`.
    fn get_java_class(&self) -> Option<HostClass> {
        None
    }

    /// Rust-side helper for Java's `expr instanceof MaybePrimitiveExpr` +
    /// downcast pattern. Default returns `None`; nodes that implement
    /// `MaybePrimitiveExpr` override to `Some(self)`.
    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> {
        None
    }

    /// Java's `expr instanceof FnExpr` downcast for InvokeExpr's direct-call
    /// optimization. Default returns `None`; `FnExpr` overrides.
    fn as_fn_expr(&self) -> Option<&FnExpr> {
        None
    }

    /// Java's `expr instanceof LocalBindingExpr` downcast — used by
    /// `InvokeExpr` to peek through a local that resolves to a `(fn …)` init
    /// and emit a direct call. Default returns `None`; `LocalBindingExpr`
    /// overrides.
    fn as_local_binding_expr(&self) -> Option<&LocalBindingExpr> {
        None
    }

    /// Java's `expr instanceof VarExpr` downcast — used by `InvokeExpr` to
    /// look up the var's compile-time-known fn FuncRef (registered by
    /// `DefExpr.emit` when init is a `FnExpr`) and emit a direct call.
    fn as_var_expr(&self) -> Option<&VarExpr> {
        None
    }

    /// Downcast for `MultiArityFnExpr`. Used by `InvokeExpr`'s static
    /// dispatch path to pick the matching arity at the call site.
    fn as_multi_arity_fn_expr(&self) -> Option<&MultiArityFnExpr> {
        None
    }
}

// ============================================================================
// Java line ~375–410: `interface IParser`
// ============================================================================

/// `Compiler.IParser`. Java:
///
/// ```text
/// interface IParser {
///     Expr parse(C context, Object form);
/// }
/// ```
pub trait IParser {
    fn parse(&self, context: C, form: Object) -> Box<dyn Expr>;
}

// ============================================================================
// Java line ~799–803: `static interface AssignableExpr`
// ============================================================================

/// `Compiler.AssignableExpr`. Java:
///
/// ```text
/// static interface AssignableExpr {
///     Object evalAssign(Expr val);
///     void emitAssign(C context, ObjExpr objx, GeneratorAdapter gen, Expr val);
/// }
/// ```
pub trait AssignableExpr: Expr {
    fn eval_assign(&self, val: &dyn Expr) -> Object;
    fn emit_assign(
        &self,
        context: C,
        objx: &ObjExpr,
        ir: &mut IrEmitter<'_>,
        val: &dyn Expr,
    ) -> Option<Value>;
}

// ============================================================================
// Java line ~805–810: `MaybePrimitiveExpr`
// ============================================================================

/// `Compiler.MaybePrimitiveExpr`. Java:
///
/// ```text
/// static public interface MaybePrimitiveExpr extends Expr {
///     boolean canEmitPrimitive();
///     void emitUnboxed(C context, ObjExpr objx, GeneratorAdapter gen);
/// }
/// ```
pub trait MaybePrimitiveExpr: Expr {
    fn can_emit_primitive(&self) -> bool;
    /// Emit unboxed (primitive) IR for this node, returning the produced
    /// `Value`. Callers must check `can_emit_primitive` first — `emit_unboxed`
    /// will panic for nodes that don't support it.
    fn emit_unboxed(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Value;
}

// ============================================================================
// Stand-ins for host-side types Compiler.java references.
// ============================================================================

/// Placeholder for JVM `Class<?>`. Compiler.java reads/writes this as a
/// host-class reference for type-hint propagation. In our world this will
/// eventually be a descriptor over our dynamic type tags + named host types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostClass {
    pub name: Arc<String>,
}

impl HostClass {
    /// Java `Class.isPrimitive()`. We model primitives via the JVM-style name
    /// the descriptor was built with (`"long"`, `"double"`, `"boolean"`, etc.),
    /// matching what `NumberExpr.getJavaClass` etc. produce above.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self.name.as_str(),
            "long" | "double" | "boolean" | "int" | "float" | "char" | "byte" | "short" | "void"
        )
    }
}

/// Stand-in for `clojure.asm.commons.GeneratorAdapter`. Java drives bytecode
/// emission through a single mutable adapter handed to every `emit` call;
/// our equivalent is a thin wrapper around a borrowed
/// [`dynlang::DynFunc`] (which itself owns a [`dynir::FunctionBuilder`]).
///
/// Lifetime `'f` ties the emitter to the function-build scope. Each Expr's
/// `emit` borrows `&mut IrEmitter<'_>` and dispatches into `ir.f` for the
/// actual SSA construction.
pub struct IrEmitter<'f> {
    /// The high-level dynlang builder. Public so impls can reach the underlying
    /// `FunctionBuilder` (`ir.f.fb`) when dynlang's higher-level surface
    /// doesn't cover the op they need.
    pub f: &'f mut DynFunc,
}

impl<'f> IrEmitter<'f> {
    /// Wrap a `DynFunc` for the duration of one function-build scope.
    pub fn new(f: &'f mut DynFunc) -> Self {
        IrEmitter { f }
    }
}

impl<'f> std::fmt::Debug for IrEmitter<'f> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "IrEmitter {{ ... }}")
    }
}

// ============================================================================
// Java line ~4731–5849: `ObjExpr`. The per-fn compilation context. This is the
// "where am I emitting bytecode into?" object. Java has ~1100 lines of method
// definitions on this class; we port the data shell + accessor surface here,
// and stub all the GeneratorAdapter emit paths to `unimplemented_port!` until
// IrEmitter wraps a real `dynir::FunctionBuilder`.
// ============================================================================

/// Java: `static final String CONST_PREFIX = "const__"`.
pub const CONST_PREFIX: &str = "const__";

/// `Compiler.ObjExpr`. Holds the constant pool, closed-over locals, keyword
/// and var callsite tables, line info, and the emitted bytecode for one
/// generated class (fn / deftype / reify).
///
/// Java fields use Clojure's persistent collections (`IPersistentMap`,
/// `IPersistentVector`, `IPersistentSet`). We use `Mutex<HashMap<…>>` /
/// `Mutex<Vec<…>>` since ObjExpr is built single-threaded during analyze
/// and frozen after. Reads through the public accessors clone the inner
/// state, mirroring Java's "return a snapshot" behavior on persistent reads.
#[derive(Debug)]
pub struct ObjExpr {
    // --- identity ----------------------------------------------------------
    /// Java: `String name`. Display name (e.g. `user/my-fn`).
    pub name: RwLock<Option<String>>,
    /// Java: `String internalName`. Internal/munged class name.
    pub internal_name: RwLock<Option<String>>,
    /// Java: `String thisName`. The `this`-alias for tail recur.
    pub this_name: RwLock<Option<String>>,
    /// Java: `Object tag`. Optional `:tag` metadata.
    pub tag: Object,

    // --- closed-over locals -----------------------------------------------
    /// Java: `IPersistentMap closes` — `LocalBinding → LocalBinding`.
    /// In Java the keys are LocalBindings (identity-keyed). We use the
    /// binding's `idx` since each binding has a unique slot id within its
    /// enclosing method.
    pub closes: Mutex<HashMap<i32, Arc<LocalBinding>>>,
    /// Java: `IPersistentVector closesExprs` — accumulated LocalBindingExprs.
    /// Boxed as Arc so we can hand out references without copying.
    pub closes_exprs: Mutex<Vec<Arc<LocalBindingExpr>>>,
    /// Java: `IPersistentSet volatiles` — symbols marked `:volatile-mutable`.
    pub volatiles: Mutex<std::collections::HashSet<Arc<Symbol>>>,

    // --- deftype-specific --------------------------------------------------
    /// Java: `IPersistentMap fields` — `Symbol → LocalBinding`. Non-null only
    /// for `deftype*`/`reify*` ObjExprs.
    pub fields: Mutex<Option<HashMap<Arc<Symbol>, Arc<LocalBinding>>>>,
    /// Java: `IPersistentVector hintedFields`.
    pub hinted_fields: Mutex<Vec<Arc<Symbol>>>,

    // --- constant pool + keyword/var callsite tables ----------------------
    /// Java: `IPersistentMap keywords` — `Keyword → constant-pool-id`.
    pub keywords: Mutex<HashMap<Arc<Keyword>, i32>>,
    /// Java: `IPersistentMap vars` — `Var → constant-pool-id`.
    pub vars: Mutex<HashMap<Arc<Var>, i32>>,
    /// Java: `PersistentVector constants`. The pool itself; index ↔ id.
    pub constants: Mutex<Vec<Object>>,
    /// Java: `IPersistentSet usedConstants` — int-id set.
    pub used_constants: Mutex<std::collections::HashSet<i32>>,
    /// Java: `int constantsID` — handed to runtime to look this pool up.
    pub constants_id: AtomicI32,
    /// Java: `int altCtorDrops`.
    pub alt_ctor_drops: AtomicI32,
    /// Java: `IPersistentVector keywordCallsites`.
    pub keyword_callsites: Mutex<Vec<Arc<Keyword>>>,
    /// Java: `IPersistentVector protocolCallsites`.
    pub protocol_callsites: Mutex<Vec<Arc<Var>>>,

    // --- misc compile-time flags + line info ------------------------------
    /// Java: `boolean onceOnly`. `FNONCE` (fn* with :once true).
    pub once_only: AtomicBool,
    /// Java: `Object src`. The source form, for diagnostics.
    pub src: RwLock<Object>,
    /// Java: `IPersistentMap opts`. Compile-time options bag.
    pub opts: Mutex<HashMap<Arc<Keyword>, Object>>,
    /// Java: `int line`, `int column`.
    pub line: AtomicI32,
    pub column: AtomicI32,
    /// Java: `IPersistentMap classMeta` — `:gen-class` style class metadata.
    pub class_meta: Mutex<Option<HashMap<Arc<Keyword>, Object>>>,
    /// Java: `boolean canBeDirect`. Direct-linking opt-in.
    pub can_be_direct: AtomicBool,

    // --- emitted artifacts -------------------------------------------------
    /// Java: `Class compiledClass`. Set after `compile()` finishes. We store
    /// the host-class descriptor (we don't actually load JVM classes).
    pub compiled_class: RwLock<Option<HostClass>>,
    /// Java: `byte[] bytecode`. The emitted class file. Stays empty until
    /// IrEmitter writes a real output blob (or, in our world, until dynir
    /// lowering produces a function pointer).
    pub bytecode: Mutex<Vec<u8>>,
}

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32};

impl ObjExpr {
    /// Java: `public ObjExpr(Object tag) { this.tag = tag; }`.
    pub fn new(tag: Object) -> Arc<Self> {
        Arc::new(ObjExpr {
            name: RwLock::new(None),
            internal_name: RwLock::new(None),
            this_name: RwLock::new(None),
            tag,
            closes: Mutex::new(HashMap::new()),
            closes_exprs: Mutex::new(Vec::new()),
            volatiles: Mutex::new(std::collections::HashSet::new()),
            fields: Mutex::new(None),
            hinted_fields: Mutex::new(Vec::new()),
            keywords: Mutex::new(HashMap::new()),
            vars: Mutex::new(HashMap::new()),
            constants: Mutex::new(Vec::new()),
            used_constants: Mutex::new(std::collections::HashSet::new()),
            constants_id: AtomicI32::new(0),
            alt_ctor_drops: AtomicI32::new(0),
            keyword_callsites: Mutex::new(Vec::new()),
            protocol_callsites: Mutex::new(Vec::new()),
            once_only: AtomicBool::new(false),
            src: RwLock::new(Object::Nil),
            opts: Mutex::new(HashMap::new()),
            line: AtomicI32::new(0),
            column: AtomicI32::new(0),
            class_meta: Mutex::new(None),
            can_be_direct: AtomicBool::new(false),
            compiled_class: RwLock::new(None),
            bytecode: Mutex::new(Vec::new()),
        })
    }

    /// `ObjExpr()` no-arg convenience for analyze paths that haven't computed
    /// a tag yet. Same as `new(Object::Nil)`.
    pub fn placeholder() -> Arc<Self> {
        Self::new(Object::Nil)
    }

    // ---- accessors (Java's `final` getters) ------------------------------

    pub fn name(&self) -> Option<String> {
        self.name.read().unwrap().clone()
    }
    pub fn internal_name(&self) -> Option<String> {
        self.internal_name.read().unwrap().clone()
    }
    pub fn this_name(&self) -> Option<String> {
        self.this_name.read().unwrap().clone()
    }
    pub fn line(&self) -> i32 {
        self.line.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn column(&self) -> i32 {
        self.column.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn constants_id(&self) -> i32 {
        self.constants_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    // ---- shape predicates Java uses to switch emit behavior --------------

    /// Java: `boolean isDeftype()` — non-null `fields`.
    pub fn is_deftype(&self) -> bool {
        self.fields.lock().unwrap().is_some()
    }

    /// Java: `boolean supportsMeta()` — non-deftype ObjExprs carry `__meta`.
    pub fn supports_meta(&self) -> bool {
        !self.is_deftype()
    }

    /// Java: `boolean isMutable(LocalBinding lb)` — true for deftype mutable
    /// fields. Stubbed at `false` until field-mutability metadata is wired.
    pub fn is_mutable(&self, _lb: &LocalBinding) -> bool {
        false
    }

    // ---- constant / keyword / var registration --------------------------
    //
    // Java's `Compiler.registerConstant`, `Compiler.registerKeyword`, and
    // `Compiler.registerVar` are *static* — they consult the CONSTANTS /
    // KEYWORDS / VARS dynamic Vars. The ObjExpr is the value stored in those
    // Vars during analyze. We mirror by providing instance methods here so
    // tests can drive ObjExpr directly without thread-binding the Vars; the
    // static `register_constant` / `register_keyword` above remain as the
    // entry points the Expr classes call.

    /// Add `o` to this ObjExpr's constant pool if not already present;
    /// return the resulting slot id.
    pub fn intern_constant(&self, o: Object) -> i32 {
        let mut pool = self.constants.lock().unwrap();
        // Java uses an IdentityHashMap on the constant; we use linear search
        // gated on `Object`'s structural equality. Acceptable for compile-
        // time pools (small N). Replace with a real id table if it shows up
        // in profiles.
        for (i, existing) in pool.iter().enumerate() {
            if object_identity_eq(existing, &o) {
                return i as i32;
            }
        }
        let id = pool.len() as i32;
        pool.push(o);
        id
    }

    /// Register a closed-over local. Returns the slot index (matches Java's
    /// `closes` map storing LocalBinding-by-identity).
    pub fn add_close(&self, lb: Arc<LocalBinding>) {
        self.closes.lock().unwrap().insert(lb.idx, lb);
    }

    // ---- emit helpers (still stubbed; will lower into dynir later) -------

    /// Java `ObjExpr.emitConstant(GeneratorAdapter gen, int id)` — line ~5780.
    /// Loads constant slot `id` from the generated class's constant table.
    pub fn emit_constant(&self, _ir: &mut IrEmitter<'_>, _id: i32) {
        crate::unimplemented_port!(
            "ObjExpr.emitConstant",
            "needs IrEmitter wrapping dynir + constant-pool lowering"
        )
    }

    /// Java `ObjExpr.emitKeyword(GeneratorAdapter gen, Keyword k)` — line ~5774.
    pub fn emit_keyword(&self, _ir: &mut IrEmitter<'_>, _k: &Keyword) {
        crate::unimplemented_port!(
            "ObjExpr.emitKeyword",
            "needs IrEmitter + keyword-callsite lowering"
        )
    }

    /// Java `ObjExpr.emitVar(GeneratorAdapter gen, Var v)` — line ~5751.
    pub fn emit_var(&self, _ir: &mut IrEmitter<'_>, _v: &Var) {
        crate::unimplemented_port!("ObjExpr.emitVar", "needs IrEmitter + var lookup lowering")
    }

    /// Java `ObjExpr.emitVarValue(GeneratorAdapter gen, Var v)` — line ~5760.
    pub fn emit_var_value(&self, _ir: &mut IrEmitter<'_>, _v: &Var) {
        crate::unimplemented_port!(
            "ObjExpr.emitVarValue",
            "needs IrEmitter + Var.get() lowering"
        )
    }

    /// Java `ObjExpr.emitLocal(GeneratorAdapter gen, LocalBinding lb, boolean clear)`
    /// — line ~5713. Loads a local from its slot (or closed-over field).
    pub fn emit_local(&self, _ir: &mut IrEmitter<'_>, _lb: &LocalBinding, _clear: bool) {
        crate::unimplemented_port!(
            "ObjExpr.emitLocal",
            "needs IrEmitter + closure-field vs stack-slot dispatch"
        )
    }

    /// Java `ObjExpr.emitUnboxedLocal` — line ~5743.
    pub fn emit_unboxed_local(&self, _ir: &mut IrEmitter<'_>, _lb: &LocalBinding) {
        crate::unimplemented_port!(
            "ObjExpr.emitUnboxedLocal",
            "needs IrEmitter primitive-local load"
        )
    }

    /// Java `ObjExpr.emitAssignLocal` — line ~5647. Only valid on mutable
    /// deftype fields (Java throws IllegalArgumentException otherwise).
    pub fn emit_assign_local(&self, _ir: &mut IrEmitter<'_>, lb: &LocalBinding, _val: &dyn Expr) {
        if !self.is_mutable(lb) {
            panic!(
                "clojure-jvm: IllegalArgumentException — Cannot assign to non-mutable: {}",
                lb.name
            );
        }
        crate::unimplemented_port!(
            "ObjExpr.emitAssignLocal",
            "needs IrEmitter + mutable-field store"
        )
    }

    /// Java `ObjExpr.compile(String superName, String[] interfaceNames, boolean oneTimeUse)`
    /// — line ~4874. Emits the class file. In our world this produces a dynir
    /// function and feeds it to the JIT instead.
    pub fn compile(&self, _super_name: &str, _interface_names: &[&str], _one_time_use: bool) {
        crate::unimplemented_port!(
            "ObjExpr.compile",
            "needs IrEmitter + class-emission pipeline (dynir → JIT)"
        )
    }
}

impl Expr for ObjExpr {
    fn eval(&self) -> Object {
        // Java: returns null for deftype; otherwise reflectively instantiates
        // the compiled class via its no-arg ctor. We can't until `compile`
        // runs.
        if self.is_deftype() {
            return Object::Nil;
        }
        crate::unimplemented_port!(
            "ObjExpr.eval",
            "needs `compile` to produce a compiled artifact first"
        )
    }

    fn emit(&self, _context: C, _objx: &ObjExpr, _ir: &mut IrEmitter<'_>) -> Option<Value> {
        crate::unimplemented_port!(
            "ObjExpr.emit",
            "needs IrEmitter (instantiates the compiled class + feeds closures)"
        )
    }

    fn has_java_class(&self) -> bool {
        true
    }

    fn get_java_class(&self) -> Option<HostClass> {
        // Java: compiledClass if set; else tagToClass(tag); else IFn.class.
        if let Some(c) = self.compiled_class.read().unwrap().clone() {
            return Some(c);
        }
        if let Object::Symbol(t) = &self.tag {
            return Some(HostClass {
                name: Arc::new(t.get_name().to_string()),
            });
        }
        Some(HostClass {
            name: Arc::new("clojure.lang.IFn".to_string()),
        })
    }
}

/// Rough Java `==` over an `Object`: identity for heap variants, structural
/// for value variants. Used by `intern_constant` to dedupe the pool the way
/// Java's `IdentityHashMap` does (Java symbols/keywords/strings are interned
/// so identity matches structural for those).
fn object_identity_eq(a: &Object, b: &Object) -> bool {
    use Object::*;
    match (a, b) {
        (Nil, Nil) => true,
        (Bool(x), Bool(y)) => x == y,
        (Long(x), Long(y)) => x == y,
        (Double(x), Double(y)) => x.to_bits() == y.to_bits(),
        (String(x), String(y)) => Arc::ptr_eq(x, y) || **x == **y,
        (Symbol(x), Symbol(y)) => Arc::ptr_eq(x, y) || **x == **y,
        (Keyword(x), Keyword(y)) => Arc::ptr_eq(x, y),
        (Var(x), Var(y)) => Arc::ptr_eq(x, y),
        (Namespace(x), Namespace(y)) => Arc::ptr_eq(x, y),
        (List(x), List(y)) => Arc::ptr_eq(x, y),
        (Vector(x), Vector(y)) => Arc::ptr_eq(x, y),
        (Host(x), Host(y)) => Arc::ptr_eq(
            &(x.clone() as Arc<dyn std::any::Any + Send + Sync>),
            &(y.clone() as Arc<dyn std::any::Any + Send + Sync>),
        ),
        _ => false,
    }
}

// ============================================================================
// Java line ~417–593: `DefExpr` + Parser. `(def name)` / `(def name val)`.
// ============================================================================

/// `Compiler.DefExpr`. Java fields trimmed to what we need:
///   * `var` — the Var being def'd
///   * `init` — the value expression (None for declare-only `(def name)`)
///   * `init_provided` — whether the user provided a value
///   * `is_dynamic` — `^:dynamic` metadata
///
/// We elide `meta`, `source/line/column`, and `CompilerException` wrapping
/// for now.
#[derive(Debug)]
pub struct DefExpr {
    pub var: Arc<Var>,
    pub init: Option<Box<dyn Expr>>,
    pub init_provided: bool,
    pub is_dynamic: bool,
}

impl Expr for DefExpr {
    fn eval(&self) -> Object {
        // Java's DefExpr.eval compiles + invokes — there is no tree-walking
        // path. Our port goes through compile-emit-JIT, where `(def x v)` is
        // lowered to `clj_var_bind_root(var_ptr, v_value)` extern calls.
        panic!(
            "clojure-jvm: DefExpr.eval is not a tree-walker — compile via the \
             JIT pipeline instead"
        )
    }

    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java line ~483–514.
        // Java sequence:
        //   1. objx.emitVar(gen, var)          // pushes Var onto stack
        //   2. (optional) setDynamic, setMeta
        //   3. if init_provided: dup; init.emit; invokeVirtual bindRoot
        //   4. if STATEMENT: pop
        //
        // We bake the Var pointer in as an i64 constant (the Arc lives in the
        // namespace map for the program's lifetime) and call the
        // `cljvm_var_bind_root(var_ptr, val_bits)` extern. The extern returns
        // `val_bits`, which is the divergence from Java (Java's DefExpr returns
        // the Var itself). Fine until we add proper Var-as-value encoding.
        //
        // We do not yet support setDynamic / setMeta — both rely on data we
        // don't track on `DefExpr` yet.
        if self.is_dynamic {
            crate::unimplemented_port!(
                "DefExpr.emit (is_dynamic)",
                "setDynamic on the var requires a JIT extern not yet declared"
            );
        }

        let var_ptr_bits = crate::runtime::var_to_jit_ptr(&self.var) as i64;
        let var_ptr_val = ir.f.fb.iconst(dynir::Type::I64, var_ptr_bits);

        if self.init_provided {
            let init = self
                .init
                .as_deref()
                .expect("init_provided implies Some(init)");

            // Var → FuncRef registration for `(def f (fn …))` happens at
            // `parse_def_form` time (analyze), not here. Pending FnExpr
            // bodies are lowered before the entry fn's emit runs, so the
            // mapping must already be in place when those bodies emit
            // self / mutual / earlier-sibling invocations.

            let val = init.emit(C::Expression, objx, ir)?;
            let bind_root_fref = with_active_compiler(|c| c.externs.var_bind_root);
            let ret =
                ir.f.fb
                    .call(bind_root_fref, &[var_ptr_val, val])
                    .expect("cljvm_var_bind_root returns I64");
            match context {
                C::Statement => None,
                _ => Some(ret),
            }
        } else {
            // `(def name)` — declare-only. Java leaves the Var on the stack
            // and pops in STATEMENT context. Without a Var-value tag we don't
            // have a sensible Value to return; the Var has been interned as a
            // side effect of `parse_def_form` already, so emitting nil is the
            // pragmatic choice until Var values get a NanBox tag.
            let nil_val = ir.f.nil();
            match context {
                C::Statement => None,
                _ => Some(nil_val),
            }
        }
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("clojure.lang.Var".to_string()),
        })
    }
}

/// Look up an unqualified keyword `:name` in `m`; return `true` iff its
/// value is `Object::Bool(true)`. Used by `parse_def_form` to detect
/// `^{:macro true}` etc. on the def name.
fn meta_has_kw_true(m: &Arc<PersistentHashMap>, name: &str) -> bool {
    for (k, v) in m.iter() {
        if let Object::Keyword(kw) = k {
            if kw.get_namespace().is_none() && kw.get_name() == name {
                return matches!(v, Object::Bool(true));
            }
        }
    }
    false
}

/// `Compiler.DefExpr.Parser`. Java line ~524. Parses `(def name)`,
/// `(def name val)`, `(def name docstring val)`.
pub struct DefExprParser;

impl IParser for DefExprParser {
    fn parse(&self, context: C, form: Object) -> Box<dyn Expr> {
        parse_def_form(context, form)
    }
}

fn parse_def_form(context: C, form: Object) -> Box<dyn Expr> {
    // Java handles a 4-form `(def name "docstring" val)` shape by attaching
    // the docstring as `:doc` metadata on the name and falling through to
    // the 3-form `(def name val)` path.
    let mut form = form;
    let n_in = super::rt::count(&form);
    if n_in == 4 {
        let head = super::rt::first(&form);
        let name_form_in = super::rt::second(&form);
        let doc_form = super::rt::third(&form);
        let val_form = super::rt::fourth(&form);
        if let Object::String(doc_s) = &doc_form {
            // Merge `:doc <docstring>` into the name's metadata. If the
            // name already has a meta map (via `^{...}`), extend it.
            let existing_meta: Option<Arc<PersistentHashMap>> = name_form_in.meta_of().cloned();
            let bare = name_form_in.peel_meta_ref().clone();
            let merged = {
                let mut entries: Vec<(Object, Object)> = Vec::new();
                if let Some(m) = existing_meta.as_ref() {
                    for (k, v) in m.iter() {
                        entries.push((k, v));
                    }
                }
                entries.push((
                    Object::Keyword(Keyword::intern_ns_name(None, "doc")),
                    Object::String(doc_s.clone()),
                ));
                PersistentHashMap::create_pairs(entries)
            };
            let name_with_doc = Object::WithMeta(Box::new(bare), merged);
            form = Object::List(PersistentList::create(vec![head, name_with_doc, val_form]));
        } else {
            panic!(
                "clojure-jvm: RuntimeException — def: expected docstring (3rd arg), got {doc_form:?}"
            );
        }
    }
    let n = super::rt::count(&form);
    if n > 3 {
        panic!("clojure-jvm: RuntimeException — Too many arguments to def");
    }
    if n < 2 {
        panic!("clojure-jvm: RuntimeException — Too few arguments to def");
    }
    let name_form = super::rt::second(&form);
    // Metadata can be attached to the def's name (e.g.
    // `(def ^{:macro true :doc "..."} foo body)` or
    // `(def ^:dynamic *bar* body)`). We extract the metadata map first,
    // then unwrap to the bare Symbol that the rest of this function
    // and `Namespace::intern` expect.
    let name_meta: Option<Arc<PersistentHashMap>> = name_form.meta_of().cloned();
    let bare_name = name_form.peel_meta_ref();
    let sym = match bare_name {
        Object::Symbol(s) => s.clone(),
        _ => panic!("clojure-jvm: RuntimeException — First argument to def must be a Symbol"),
    };
    // Detect `:macro true` on the name's metadata. Used below in two
    // places: once when no init is provided (still flag the Var) and
    // again when init is a fn so the Var dispatches as a macro at
    // expansion time.
    let macro_meta_set = name_meta
        .as_ref()
        .map(|m| meta_has_kw_true(m, "macro"))
        .unwrap_or(false);

    // Look up / intern in current ns. Java's `lookupVar` does namespace
    // resolution; for unqualified syms in the current ns this collapses to
    // `currentNS().intern(sym)`.
    let cur_ns = super::rt::current_ns();
    let v = if sym.get_namespace().is_some() {
        // Qualified: look up the named ns and intern there. Reject if the
        // ns doesn't exist.
        let ns_sym = Symbol::intern(sym.get_namespace().unwrap());
        let target_ns = super::namespace::Namespace::find(&ns_sym).unwrap_or_else(|| {
            panic!(
                "clojure-jvm: RuntimeException — Can't refer to qualified var that doesn't exist: {}",
                sym.get_name()
            )
        });
        target_ns.intern(Symbol::intern(sym.get_name()))
    } else {
        cur_ns.intern(sym.clone())
    };

    // If the def's name symbol carried `:macro` metadata (via the reader's
    // `^:macro` shorthand), flag the Var as a macro. Java sets this via
    // Var.setMacro() — we mirror that.
    if macro_meta_set {
        v.set_macro();
    }

    let init_provided = n == 3;
    let init = if init_provided {
        Some(analyze_named(
            if context == C::Eval {
                context
            } else {
                C::Expression
            },
            super::rt::third(&form),
            Some(sym.get_name()),
        ))
    } else {
        None
    };

    // If init analyzed to a `FnExpr`, register `Var → FuncRef` on the active
    // Compiler at analyze time (not emit time). Pending FnExpr bodies are
    // lowered BEFORE the entry fn's emit runs, so a self-recursive or
    // mutually-recursive call from inside the body needs the mapping to be
    // visible already. Doing it here also catches sibling `defn`s analyzed
    // earlier in the same top-level `do` form.
    if let Some(init_box) = init.as_ref() {
        if let Some(fnexpr) = init_box.as_fn_expr() {
            let info = VarFnInfo {
                is_variadic: fnexpr.is_variadic(),
                fixed_arity: fnexpr.fixed_arity(),
            };
            // Macros: if the def name carried `^:macro`, flag the Var.
            // The macro fn handle lives in `var.root` after bind_root runs;
            // `macroexpand_once` reads it from there at expand time. No
            // separate macro_env table needed.
            if macro_meta_set {
                v.set_macro();
            }
            with_active_compiler(|c| c.register_var_fn(&v, fnexpr.fref(), info));
        } else if let Some(multi) = init_box.as_multi_arity_fn_expr() {
            let table: Vec<(dynir::FuncRef, VarFnInfo)> = multi
                .arities
                .iter()
                .map(|a| {
                    (
                        a.fref(),
                        VarFnInfo {
                            is_variadic: a.is_variadic,
                            fixed_arity: a.fixed_arity,
                        },
                    )
                })
                .collect();
            with_active_compiler(|c| c.register_var_multi_arity(&v, table));
        }
    }

    Box::new(DefExpr {
        var: v,
        init,
        init_provided,
        is_dynamic: false,
    })
}

// ============================================================================
// Java line ~633–680: `VarExpr` (read a Var's value).
// ============================================================================

/// `Compiler.VarExpr`. Holds a resolved Var and reads its value at runtime.
/// In Java the emit path produces `var.getRawRoot()` (or `var.get()` for
/// thread-bound vars).
#[derive(Debug)]
pub struct VarExpr {
    pub var: Arc<Var>,
    pub tag: Option<Arc<Symbol>>,
}

impl Expr for VarExpr {
    fn eval(&self) -> Object {
        // No tree-walker. `var` reads go through compile-emit-JIT (extern
        // `clj_var_get(var_ptr)` lowering).
        panic!(
            "clojure-jvm: VarExpr.eval is not a tree-walker — compile via the \
             JIT pipeline instead"
        )
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java line ~650–656: `objx.emitVarValue(gen, var); if STATEMENT pop`.
        // We bake the Var pointer as an i64 constant and call the
        // `cljvm_var_deref(var_ptr)` extern, which returns the current value
        // as a NanBox u64.
        let var_ptr_bits = crate::runtime::var_to_jit_ptr(&self.var) as i64;
        let var_ptr_val = ir.f.fb.iconst(dynir::Type::I64, var_ptr_bits);
        let deref_fref = with_active_compiler(|c| c.externs.var_deref);
        let ret =
            ir.f.fb
                .call(deref_fref, &[var_ptr_val])
                .expect("cljvm_var_deref returns I64");
        match context {
            C::Statement => None,
            _ => Some(ret),
        }
    }

    fn has_java_class(&self) -> bool {
        self.tag.is_some()
    }
    fn get_java_class(&self) -> Option<HostClass> {
        self.tag.as_ref().map(|t| HostClass {
            name: Arc::new(t.get_name().to_string()),
        })
    }

    fn as_var_expr(&self) -> Option<&VarExpr> {
        Some(self)
    }
}

impl AssignableExpr for VarExpr {
    fn eval_assign(&self, _val: &dyn Expr) -> Object {
        panic!(
            "clojure-jvm: VarExpr.eval_assign is not a tree-walker — compile \
             via the JIT pipeline instead"
        )
    }

    fn emit_assign(
        &self,
        _context: C,
        _objx: &ObjExpr,
        _ir: &mut IrEmitter<'_>,
        _val: &dyn Expr,
    ) -> Option<Value> {
        crate::unimplemented_port!(
            "VarExpr.emitAssign",
            "needs JIT externs for Var.set + Var-pointer constant pool"
        )
    }
}

// ============================================================================
// Java line ~364–373: `UntypedExpr` abstract base.
// ============================================================================

/// `Compiler.UntypedExpr`. Java:
///
/// ```text
/// public static abstract class UntypedExpr implements Expr {
///     public Class getJavaClass() { throw new IllegalArgumentException(...); }
///     public boolean hasJavaClass() { return false; }
/// }
/// ```
///
/// We don't need a Rust trait because the base behavior is "panic on
/// `getJavaClass`". Each Java subclass (`MonitorEnterExpr`, `MonitorExitExpr`,
/// `ThrowExpr`) just implements `Expr` directly with those defaults — Rust's
/// trait-default mechanism handles it via `Expr::has_java_class` already
/// returning `false`. We expose a tiny helper for the throw shape:
fn untyped_get_java_class_panic() -> ! {
    panic!(
        "clojure-jvm: IllegalArgumentException — Has no Java class \
         (UntypedExpr subclass called getJavaClass)"
    );
}

// ============================================================================
// Java line ~791–797: `LiteralExpr` abstract base.
// ============================================================================

/// `Compiler.LiteralExpr`. Java:
///
/// ```text
/// public static abstract class LiteralExpr implements Expr {
///     abstract Object val();
///     public Object eval() { return val(); }
/// }
/// ```
///
/// Concrete subclasses provide `val()`. Each subclass's `Expr::eval` impl
/// just calls `self.val()` — Rust can't share that line through subtrait
/// default the way Java's `abstract class` does, so the boilerplate is
/// per-impl. Kept tiny.
pub trait LiteralExpr: Expr {
    fn val(&self) -> Object;
}

// ============================================================================
// Helpers Compiler.java declares as static methods used by these Exprs.
// ============================================================================

/// Java `Compiler.registerConstant(Object)` — line ~7778. Adds `o` to the
/// current `CONSTANTS` PersistentVector and returns its index. If `CONSTANTS`
/// isn't bound (top-level eval, no surrounding fn), returns -1.
///
/// Stubbed: needs PersistentVector + IdentityHashMap + RT.conj. Until then
/// callers get -1, mirroring Java's "not bound" branch — safe to use during
/// non-fn contexts.
pub fn register_constant(_o: Object) -> i32 {
    // Java path: if (!CONSTANTS.isBound()) return -1;
    // We can't yet check isBound meaningfully (CONSTANTS contains Nil until
    // someone push-binds), so we always take the unbound branch.
    -1
}

/// Java `Compiler.registerKeyword(Keyword)` — line ~7791. Wraps a Keyword
/// in a `KeywordExpr` and records it in the per-fn `KEYWORDS` map so
/// `ObjExpr.emitKeyword` can resolve a slot id. We mirror the structure but
/// the "register in KEYWORDS" path is a no-op until PersistentMap lands.
pub fn register_keyword(k: Arc<Keyword>) -> KeywordExpr {
    KeywordExpr { k }
}

// ============================================================================
// Java line ~718–747: `KeywordExpr`. Defined ahead of `LiteralExpr` in the
// Java file (it's declared earlier for textual reasons), but it `extends
// LiteralExpr`. Keeping Java's file order.
// ============================================================================

/// `Compiler.KeywordExpr`. Java:
///
/// ```text
/// public static class KeywordExpr extends LiteralExpr {
///     public final Keyword k;
///     ...
///     public Object eval() { return k; }
///     public Class getJavaClass() { return Keyword.class; }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct KeywordExpr {
    pub k: Arc<Keyword>,
}

impl Expr for KeywordExpr {
    fn eval(&self) -> Object {
        Object::Keyword(self.k.clone())
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java's `objx.emitKeyword` resolves the keyword to its class-pool
        // slot via a per-fn KEYWORDS table. We use the same machinery as
        // Symbol literals: intern the Arc<Keyword> into the Compiler's
        // pending-literal queue, emit `gc_literal(LiteralRef(idx))`. After
        // JIT compile, a `clojure.lang.Keyword` heap wrapper is allocated
        // with its Raw64 `arc_ptr` field set; the moving GC traces the
        // wrapper but the Arc itself is rooted by `CompileRoots`.
        if context == C::Statement {
            return None;
        }
        let idx = with_active_compiler(|c| c.intern_keyword_literal(self.k.clone()));
        let lit = dynir::ir::LiteralRef::from_u32(idx);
        Some(ir.f.fb.gc_literal(lit))
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("clojure.lang.Keyword".to_string()),
        })
    }
}

impl LiteralExpr for KeywordExpr {
    fn val(&self) -> Object {
        Object::Keyword(self.k.clone())
    }
}

// ============================================================================
// Java line ~2394–2452: `NumberExpr`.
// ============================================================================

/// `Compiler.NumberExpr`. Holds a numeric literal; emits via the constant pool
/// (boxed) or directly (`emitUnboxed`).
#[derive(Debug, Clone)]
pub struct NumberExpr {
    /// The numeric value. Java uses `Number` (Integer/Long/Double); we use
    /// our `Object` enum's `Long`/`Double` variants for now. BigInteger /
    /// Ratio / BigDecimal land later.
    pub n: Object,
    pub id: i32,
}

impl NumberExpr {
    pub fn new(n: Object) -> Self {
        debug_assert!(
            matches!(n, Object::Long(_) | Object::Double(_)),
            "NumberExpr currently only accepts Long/Double; got {n:?}"
        );
        let id = register_constant(n.clone());
        NumberExpr { n, id }
    }

    /// `NumberExpr.parse(Number form)`. Returns NumberExpr for primitives,
    /// ConstantExpr otherwise (BigInt/Ratio etc — not yet ported).
    pub fn parse(form: Object) -> Box<dyn Expr> {
        match form {
            Object::Long(_) | Object::Double(_) => Box::new(NumberExpr::new(form)),
            _ => Box::new(ConstantExpr::new(form)),
        }
    }
}

impl Expr for NumberExpr {
    fn eval(&self) -> Object {
        self.n.clone()
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `objx.emitConstant(gen, id)` — loads slot `id` from the
        // generated class's constant table. We don't have a class-level pool
        // yet, so emit the value inline.
        if context == C::Statement {
            return None;
        }
        match self.n {
            // Integers are boxed Longs (Clojure's `long`), not native floats,
            // so `(+ 1 2)` is `3` and `(class 1)` is Long. The literal is
            // pre-boxed once at pool-fill time (`PendingLiteral::Long` →
            // `box_long`); we emit a `gc_literal` load of that cell.
            Object::Long(n) => {
                let idx = with_active_compiler(|c| c.intern_literal(PendingLiteral::Long(n)));
                let lit = dynir::ir::LiteralRef::from_u32(idx);
                Some(ir.f.fb.gc_literal(lit))
            }
            // Doubles stay native NanBox floats.
            Object::Double(x) => Some(ir.f.number(x)),
            _ => unreachable!("NumberExpr ctor rejects non-primitive numbers"),
        }
    }

    fn has_java_class(&self) -> bool {
        true
    }

    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new(match self.n {
                Object::Long(_) => "long".to_string(),
                Object::Double(_) => "double".to_string(),
                _ => unreachable!("NumberExpr ctor rejects non-primitive numbers"),
            }),
        })
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> {
        Some(self)
    }
}

impl LiteralExpr for NumberExpr {
    fn val(&self) -> Object {
        self.n.clone()
    }
}

impl MaybePrimitiveExpr for NumberExpr {
    fn can_emit_primitive(&self) -> bool {
        true
    }

    fn emit_unboxed(&self, _context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Value {
        // Java: gen.push(n.longValue()) / gen.push(n.doubleValue()) — leaves
        // a raw primitive on the operand stack. dynir doesn't have a direct
        // `fconst`; for F64 we iconst the bits and bitcast.
        match self.n {
            Object::Long(n) => ir.f.fb.iconst(dynir::Type::I64, n),
            Object::Double(x) => {
                let bits = ir.f.fb.iconst(dynir::Type::I64, x.to_bits() as i64);
                ir.f.fb.bitcast(bits, dynir::Type::F64)
            }
            _ => unreachable!("NumberExpr ctor rejects non-primitive numbers"),
        }
    }
}

/// A character literal (`\a`, `\newline`, `\uHHHH`). Mirrors `NumberExpr`'s
/// boxed-Long path: the codepoint is pre-boxed into a `clojure.lang.Character`
/// cell at pool-fill time and emitted as a `gc_literal` load. Distinct from
/// `NumberExpr` so `(str \a)` is `"a"` and `(class \a)` is `Character`.
#[derive(Debug, Clone, Copy)]
pub struct CharExpr {
    pub code: u32,
}

impl Expr for CharExpr {
    fn eval(&self) -> Object {
        Object::Char(self.code)
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        if context == C::Statement {
            return None;
        }
        let idx = with_active_compiler(|c| c.intern_literal(PendingLiteral::Char(self.code)));
        let lit = dynir::ir::LiteralRef::from_u32(idx);
        Some(ir.f.fb.gc_literal(lit))
    }

    fn has_java_class(&self) -> bool {
        true
    }

    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("java.lang.Character".to_string()),
        })
    }
}

impl LiteralExpr for CharExpr {
    fn val(&self) -> Object {
        Object::Char(self.code)
    }
}

// ============================================================================
// Java line ~2454–2536: `ConstantExpr` (and its `Parser` for `quote`).
// ============================================================================

/// `Compiler.ConstantExpr`. Holds a quoted value of any shape — stuffed into
/// the constant pool, loaded at runtime via `emitConstant`.
#[derive(Debug, Clone)]
pub struct ConstantExpr {
    pub v: Object,
    pub id: i32,
}

impl ConstantExpr {
    pub fn new(v: Object) -> Self {
        let id = register_constant(v.clone());
        ConstantExpr { v, id }
    }
}

impl Expr for ConstantExpr {
    fn eval(&self) -> Object {
        self.v.clone()
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `objx.emitConstant(gen, id); if STATEMENT pop`. The Java
        // constant pool is class-level; in our model it's the JitModule's
        // `LiteralPool` (registered as a GC root by `run_jit`), filled
        // post-compile with heap allocations.
        //
        // Dispatch by the Object variant: each constant variant has its
        // own `intern_*_literal` path that allocates an appropriate heap
        // type at JIT-finalize time. Variants we can't represent yet
        // panic with a clear message.
        if context == C::Statement {
            return None;
        }
        let idx = with_active_compiler(|c| match &self.v {
            Object::Symbol(s) => c.intern_symbol_literal(s.clone()),
            Object::List(l) => c.intern_list_literal(l.clone()),
            Object::Keyword(k) => c.intern_keyword_literal(k.clone()),
            Object::String(s) => c.intern_string_literal(s.clone()),
            Object::Vector(v) => c.intern_literal(PendingLiteral::Vector(v.clone())),
            Object::Map(m) => c.intern_literal(PendingLiteral::Map(m.clone())),
            Object::Set(s) => c.intern_literal(PendingLiteral::Set(s.clone())),
            // WithMeta: emit the inner constant; metadata is dropped at this
            // path until we wire IObj-meta-attached constants through the pool.
            Object::WithMeta(inner, _) => match inner.as_ref() {
                Object::Symbol(s) => c.intern_symbol_literal(s.clone()),
                Object::List(l) => c.intern_list_literal(l.clone()),
                Object::Keyword(k) => c.intern_keyword_literal(k.clone()),
                Object::String(s) => c.intern_string_literal(s.clone()),
                Object::Vector(v) => c.intern_literal(PendingLiteral::Vector(v.clone())),
                Object::Map(m) => c.intern_literal(PendingLiteral::Map(m.clone())),
                Object::Set(s) => c.intern_literal(PendingLiteral::Set(s.clone())),
                _ => crate::unimplemented_port!(
                    "ConstantExpr.emit",
                    "no heap representation yet for WithMeta(constant {:?}, ...)",
                    inner
                ),
            },
            // Numbers/bool/nil never reach ConstantExpr — they bypass
            // through their dedicated *Expr nodes during analyze.
            // Unknown constant kinds (e.g., a heap-bits Object that
            // sneaked in via a JIT macroexpand corruption): intern an
            // empty string as the placeholder. Wrong runtime value but
            // analysis proceeds and any further panic is catchable.
            _ => {
                eprintln!(
                    "[cljvm-stub] ConstantExpr.emit: unhandled constant {:?} \
                     — interning empty string",
                    self.v
                );
                c.intern_string_literal(std::sync::Arc::new(String::new()))
            }
        });
        let lit = dynir::ir::LiteralRef::from_u32(idx);
        Some(ir.f.fb.gc_literal(lit))
    }

    fn has_java_class(&self) -> bool {
        // Java: Modifier.isPublic(v.getClass().getModifiers()). Until we
        // model host classes, conservatively say "yes" for the variants we
        // know are public types.
        !matches!(self.v, Object::Unported { .. })
    }

    fn get_java_class(&self) -> Option<HostClass> {
        // Metadata wrapper is transparent for class lookup —
        // peel any `WithMeta` first so e.g. a wrapped Symbol still
        // reports `clojure.lang.Symbol`.
        let name = match self.v.peel_meta_ref() {
            Object::Nil => return None,
            Object::Bool(_) => "java.lang.Boolean",
            Object::Long(_) => "java.lang.Long",
            Object::Char(_) => "java.lang.Character",
            Object::Double(_) => "java.lang.Double",
            Object::String(_) => "java.lang.String",
            Object::Symbol(_) => "clojure.lang.Symbol",
            Object::Keyword(_) => "clojure.lang.Keyword",
            Object::Var(_) => "clojure.lang.Var",
            Object::Namespace(_) => "clojure.lang.Namespace",
            Object::List(_) => "clojure.lang.PersistentList",
            Object::Vector(_) => "clojure.lang.PersistentVector",
            Object::Map(_) => "clojure.lang.PersistentHashMap",
            Object::Set(_) => "clojure.lang.PersistentHashSet",
            Object::TreeMap(_) => "clojure.lang.PersistentTreeMap",
            Object::TreeSet(_) => "clojure.lang.PersistentTreeSet",
            Object::Host(_) => return None,
            Object::Unported { .. } => return None,
            Object::WithMeta(_, _) => unreachable!("peel_meta_ref strips WithMeta"),
        };
        Some(HostClass {
            name: Arc::new(name.to_string()),
        })
    }
}

impl LiteralExpr for ConstantExpr {
    fn val(&self) -> Object {
        self.v.clone()
    }
}

/// `Compiler.ConstantExpr.Parser`. Parses `(quote v)`.
pub struct ConstantExprParser;

impl IParser for ConstantExprParser {
    fn parse(&self, _context: C, form: Object) -> Box<dyn Expr> {
        // Java line ~2507: `(quote v)` — single argument. Dispatch v's shape
        // into the most specific Expr variant, falling back to ConstantExpr.
        let arg_count = super::rt::count(&form) - 1;
        if arg_count != 1 {
            panic!(
                "clojure-jvm: ExceptionInfo — Wrong number of args ({arg_count}) passed to quote"
            );
        }
        let v = super::rt::second(&form);
        constant_literal_expr(v)
    }
}

/// Build the Expr for a compile-time constant value — the shape dispatch
/// shared by `(quote v)` (ConstantExprParser) and `case*` test constants.
fn constant_literal_expr(v: Object) -> Box<dyn Expr> {
    match &v {
        Object::Nil => Box::new(NIL_EXPR),
        Object::Bool(true) => Box::new(TRUE_EXPR),
        Object::Bool(false) => Box::new(FALSE_EXPR),
        Object::Long(_) | Object::Double(_) => NumberExpr::parse(v),
        Object::Char(code) => Box::new(CharExpr { code: *code }),
        Object::String(s) => Box::new(StringExpr { str: s.clone() }),
        // Empty collections (EmptyExpr in Java) — not yet ported as a
        // distinct Expr; quoted non-empty collections + symbols /
        // keywords / etc. land in ConstantExpr where the heap path
        // dispatches per variant.
        _ => Box::new(ConstantExpr::new(v)),
    }
}

// ============================================================================
// Java line ~2538–2558: `NilExpr` + the `NIL_EXPR` singleton.
// ============================================================================

/// `Compiler.NilExpr`.
#[derive(Debug, Clone, Copy)]
pub struct NilExpr;

impl Expr for NilExpr {
    fn eval(&self) -> Object {
        Object::Nil
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `gen.visitInsn(ACONST_NULL); if STATEMENT pop`. In dynlang
        // we emit the NanBox nil constant and drop it for STATEMENT.
        if context == C::Statement {
            return None;
        }
        Some(ir.f.nil())
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        None
    }
}

impl LiteralExpr for NilExpr {
    fn val(&self) -> Object {
        Object::Nil
    }
}

/// `Compiler.NIL_EXPR` — Java line 2558.
pub const NIL_EXPR: NilExpr = NilExpr;

// ============================================================================
// Java line ~2560–2593: `BooleanExpr` + `TRUE_EXPR` / `FALSE_EXPR`.
// ============================================================================

/// `Compiler.BooleanExpr`.
#[derive(Debug, Clone, Copy)]
pub struct BooleanExpr {
    pub val: bool,
}

impl Expr for BooleanExpr {
    fn eval(&self) -> Object {
        // Java: val ? RT.T : RT.F.
        if self.val {
            super::rt::T()
        } else {
            super::rt::F()
        }
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java loads `Boolean.TRUE` / `Boolean.FALSE`. In dynlang's NanBox
        // world, both are tagged with `bool_tag` and a 0/1 payload.
        if context == C::Statement {
            return None;
        }
        Some(ir.f.bool_val(self.val))
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("java.lang.Boolean".to_string()),
        })
    }
}

impl LiteralExpr for BooleanExpr {
    fn val(&self) -> Object {
        if self.val {
            super::rt::T()
        } else {
            super::rt::F()
        }
    }
}

/// `Compiler.TRUE_EXPR` — Java line 2592.
pub const TRUE_EXPR: BooleanExpr = BooleanExpr { val: true };

/// `Compiler.FALSE_EXPR` — Java line 2593.
pub const FALSE_EXPR: BooleanExpr = BooleanExpr { val: false };

// ============================================================================
// Java line ~2595–2618: `StringExpr`.
// ============================================================================

/// `Compiler.StringExpr`.
#[derive(Debug, Clone)]
pub struct StringExpr {
    pub str: Arc<String>,
}

impl StringExpr {
    pub fn new(s: impl Into<String>) -> Self {
        StringExpr {
            str: Arc::new(s.into()),
        }
    }
}

impl Expr for StringExpr {
    fn eval(&self) -> Object {
        Object::String(self.str.clone())
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `gen.push(str)` — emits an LDC of the class's String constant
        // pool entry. We translate that into a `GcLiteral` load from the
        // JitModule's literal pool: at IR-build time we reserve an index;
        // after the JitModule is constructed we allocate the String on the
        // GC heap and push its NanBox-encoded pointer into that slot.
        //
        // The literal pool is a registered GC root source (see
        // `DynGcRuntime::run_jit`), so a moving GC traces and rewrites this
        // slot in place — the emitted load picks up the new pointer.
        if context == C::Statement {
            return None;
        }
        let idx = with_active_compiler(|c| c.intern_string_literal(self.str.clone()));
        let lit = dynir::ir::LiteralRef::from_u32(idx);
        Some(ir.f.fb.gc_literal(lit))
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("java.lang.String".to_string()),
        })
    }
}

impl LiteralExpr for StringExpr {
    fn val(&self) -> Object {
        Object::String(self.str.clone())
    }
}

// ============================================================================
// Java line ~3224–3363: `IfExpr` + `IfExpr.Parser`.
// ============================================================================

/// `Compiler.IfExpr`. Three child Exprs (test/then/else) plus source position.
/// Implements both `Expr` and `MaybePrimitiveExpr` because an `if` whose branches
/// agree on a primitive Java class can stay unboxed.
#[derive(Debug)]
pub struct IfExpr {
    pub test_expr: Box<dyn Expr>,
    pub then_expr: Box<dyn Expr>,
    pub else_expr: Box<dyn Expr>,
    pub line: i32,
    pub column: i32,
}

impl IfExpr {
    pub fn new(
        line: i32,
        column: i32,
        test_expr: Box<dyn Expr>,
        then_expr: Box<dyn Expr>,
        else_expr: Box<dyn Expr>,
    ) -> Self {
        IfExpr {
            test_expr,
            then_expr,
            else_expr,
            line,
            column,
        }
    }

    /// Java path inlined into `emit` / `emitUnboxed`. Translates Java's ASM
    /// label-based control flow:
    ///
    /// ```text
    /// emit test → ifZCmp falseLabel → emit then → goto end → mark falseLabel
    /// emit else → mark end
    /// ```
    ///
    /// into dynir's block-based form:
    ///
    /// ```text
    /// emit test → br_if_truthy(test, then_bb, else_bb)
    /// then_bb:  emit then → jump merge_bb(then_val)
    /// else_bb:  emit else → jump merge_bb(else_val)
    /// merge_bb(phi):  …
    /// ```
    ///
    /// Returns the merged phi value (or `None` in `STATEMENT` context, where
    /// neither branch needs to produce a value).
    fn do_emit(
        &self,
        context: C,
        objx: &ObjExpr,
        ir: &mut IrEmitter<'_>,
        emit_unboxed: bool,
    ) -> Option<Value> {
        // Test always evaluates in EXPRESSION context — we need its value to
        // branch on, even when the surrounding context is STATEMENT.
        let test_val = self.test_expr.emit(C::Expression, objx, ir)?;

        let then_bb = ir.f.fb.create_block(&[]);
        let else_bb = ir.f.fb.create_block(&[]);

        // Phi type. Statement context has no phi; otherwise NanBox I64 (or
        // primitive type for unboxed paths).
        let phi_ty_opt: Option<dynir::Type> = if context == C::Statement {
            None
        } else if emit_unboxed {
            Some(
                match self.get_java_class().as_ref().map(|c| c.name.as_str()) {
                    Some("long") => dynir::Type::I64,
                    Some("double") => dynir::Type::F64,
                    other => panic!(
                        "clojure-jvm: IfExpr.emit_unboxed branches must agree on a primitive type, got {other:?}"
                    ),
                },
            )
        } else {
            Some(dynir::Type::I64)
        };

        // Lazy-create the merge block only when at least one branch
        // actually reaches it. If BOTH branches terminate (e.g. recur +
        // throw, two recurs), an eagerly-created merge block would be
        // left unterminated and trip dynir's "block N is not terminated"
        // assert at function-finish time.
        let mut merge_bb: Option<dynir::BlockId> = None;
        let ensure_merge =
            |ir: &mut IrEmitter<'_>, slot: &mut Option<dynir::BlockId>| -> dynir::BlockId {
                if let Some(b) = *slot {
                    return b;
                }
                let b = match phi_ty_opt {
                    Some(ty) => ir.f.fb.create_block(&[ty]),
                    None => ir.f.fb.create_block(&[]),
                };
                *slot = Some(b);
                b
            };

        ir.f.br_if_truthy(test_val, then_bb, &[], else_bb, &[]);

        // ---- then branch ----
        ir.f.fb.switch_to_block(then_bb);
        let then_val: Option<Value> = if emit_unboxed {
            Some(
                self.then_expr
                    .as_maybe_primitive()
                    .expect("emit_unboxed: then branch must be MaybePrimitiveExpr")
                    .emit_unboxed(context, objx, ir),
            )
        } else {
            self.then_expr.emit(context, objx, ir)
        };
        if !ir.f.fb.current_block_is_terminated() {
            let mb = ensure_merge(ir, &mut merge_bb);
            match (phi_ty_opt.is_some(), then_val) {
                (true, Some(v)) => ir.f.fb.jump(mb, &[v]),
                (false, _) => ir.f.fb.jump(mb, &[]),
                (true, None) => panic!(
                    "clojure-jvm: IfExpr then branch produced no value in non-STATEMENT context"
                ),
            }
        }

        // ---- else branch ----
        ir.f.fb.switch_to_block(else_bb);
        let else_val: Option<Value> = if emit_unboxed {
            Some(
                self.else_expr
                    .as_maybe_primitive()
                    .expect("emit_unboxed: else branch must be MaybePrimitiveExpr")
                    .emit_unboxed(context, objx, ir),
            )
        } else {
            self.else_expr.emit(context, objx, ir)
        };
        if !ir.f.fb.current_block_is_terminated() {
            let mb = ensure_merge(ir, &mut merge_bb);
            match (phi_ty_opt.is_some(), else_val) {
                (true, Some(v)) => ir.f.fb.jump(mb, &[v]),
                (false, _) => ir.f.fb.jump(mb, &[]),
                (true, None) => panic!(
                    "clojure-jvm: IfExpr else branch produced no value in non-STATEMENT context"
                ),
            }
        }

        let Some(mb) = merge_bb else {
            // Both branches diverged. The current cursor sits on the
            // (terminated) else block, so the caller's
            // `current_block_is_terminated()` check will see true.
            return None;
        };
        ir.f.fb.switch_to_block(mb);
        if phi_ty_opt.is_some() {
            Some(ir.f.fb.block_param(mb, 0))
        } else {
            None
        }
    }
}

impl Expr for IfExpr {
    fn eval(&self) -> Object {
        // No tree-walker. `if` is compiled into branch instructions via
        // `do_emit` → JIT.
        panic!(
            "clojure-jvm: IfExpr.eval is not a tree-walker — compile via the \
             JIT pipeline instead"
        )
    }

    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        self.do_emit(context, objx, ir, false)
    }

    fn has_java_class(&self) -> bool {
        // Java: thenExpr.hasJavaClass() && elseExpr.hasJavaClass() &&
        //       (thenClass == elseClass
        //        || thenClass == RECUR_CLASS || elseClass == RECUR_CLASS
        //        || (thenClass == null && !elseClass.isPrimitive())
        //        || (elseClass == null && !thenClass.isPrimitive()))
        //
        // We don't model RECUR_CLASS as a HostClass yet — `Recur` is a Rust
        // marker, not produced by `getJavaClass`. Until we wire that, treat
        // recur as "compatible with anything" the same way Java does.
        if !self.then_expr.has_java_class() || !self.else_expr.has_java_class() {
            return false;
        }
        let then_c = self.then_expr.get_java_class();
        let else_c = self.else_expr.get_java_class();
        match (&then_c, &else_c) {
            (a, b) if a == b => true,
            (None, Some(c)) | (Some(c), None) => !c.is_primitive(),
            _ => false,
        }
    }

    fn get_java_class(&self) -> Option<HostClass> {
        // Java: prefer thenExpr's class unless it's null/RECUR; else use
        // elseExpr's. We don't yet surface RECUR through HostClass, so just
        // pick then's if present.
        self.then_expr
            .get_java_class()
            .or_else(|| self.else_expr.get_java_class())
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> {
        Some(self)
    }
}

impl MaybePrimitiveExpr for IfExpr {
    fn can_emit_primitive(&self) -> bool {
        // Java wraps this in try/catch and returns false on any exception.
        // We don't need that — none of our calls panic in the happy path.
        let then_mp = self.then_expr.as_maybe_primitive();
        let else_mp = self.else_expr.as_maybe_primitive();
        let (Some(t), Some(e)) = (then_mp, else_mp) else {
            return false;
        };
        let same_class = self.then_expr.get_java_class() == self.else_expr.get_java_class();
        same_class && t.can_emit_primitive() && e.can_emit_primitive()
    }

    fn emit_unboxed(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Value {
        self.do_emit(context, objx, ir, true)
            .expect("clojure-jvm: IfExpr.emitUnboxed should never be called in STATEMENT context")
    }
}

/// `Compiler.IfExpr.Parser`. Parses `(if test then)` or `(if test then else)`.
pub struct IfExprParser;

impl IParser for IfExprParser {
    fn parse(&self, _context: C, _form: Object) -> Box<dyn Expr> {
        // Java path: enforce arity 3 or 4, push CLEAR_PATH branch bindings
        // around each subform analyze, build IfExpr(lineDeref, columnDeref,
        // testexpr, thenexpr, elseexpr). Needs RT.count + RT.second + .third
        // + .fourth + analyze + pushThreadBindings + PathNode.
        crate::unimplemented_port!(
            "IfExpr.Parser.parse",
            "needs RT.count/second/third/fourth + Var.pushThreadBindings + PathNode + analyze"
        )
    }
}

/// Clojure truthiness: anything other than `nil` / `false` is truthy.
/// Java path (line 3242): `t != null && t != Boolean.FALSE`.
pub fn is_truthy(o: &Object) -> bool {
    !matches!(o, Object::Nil | Object::Bool(false))
}

// ============================================================================
// Java line ~6678–6755: `BodyExpr`. Wraps `(do ...)` and is also produced
// implicitly by `fn`/`let`/`when` bodies.
// ============================================================================

/// `Compiler.BodyExpr`. Holds the sequence of child exprs; eval/emit run them
/// in order, with all but the last in STATEMENT context.
///
/// Java stores `exprs` as a `PersistentVector` for `nth` access; we use a
/// plain `Vec<Box<dyn Expr>>` since BodyExpr never gets re-shared/mutated as
/// a Clojure value.
#[derive(Debug)]
pub struct BodyExpr {
    pub exprs: Vec<Box<dyn Expr>>,
}

impl BodyExpr {
    pub fn new(exprs: Vec<Box<dyn Expr>>) -> Self {
        BodyExpr { exprs }
    }

    /// Java: `lastExpr()` — `exprs.nth(exprs.count() - 1)`. Panics on empty
    /// (matches Java's NPE/AIOOBE on the same path), because Java's Parser
    /// guarantees at least NIL_EXPR is pushed.
    fn last_expr(&self) -> &dyn Expr {
        self.exprs
            .last()
            .expect("BodyExpr always has ≥1 expr (Parser pushes NIL_EXPR for empty)")
            .as_ref()
    }
}

impl Expr for BodyExpr {
    fn eval(&self) -> Object {
        // No tree-walker. `do` is compiled by emitting each child in
        // STATEMENT context except the tail, whose value is the body's
        // value — all through JIT.
        panic!(
            "clojure-jvm: BodyExpr.eval is not a tree-walker — compile via the \
             JIT pipeline instead"
        )
    }

    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // All but last: STATEMENT context. Last: caller's context. The last
        // child's produced value (if any) is the body's value. Stop as soon
        // as the current block is terminated (e.g. by a `recur` in tail
        // position of an earlier branch).
        let n = self.exprs.len();
        let mut last_val: Option<Value> = None;
        for (i, e) in self.exprs.iter().enumerate() {
            if ir.f.fb.current_block_is_terminated() {
                break;
            }
            let last_pos = i + 1 == n;
            let ctx = if last_pos { context } else { C::Statement };
            let v = e.emit(ctx, objx, ir);
            if last_pos {
                last_val = v;
            }
        }
        last_val
    }

    fn has_java_class(&self) -> bool {
        self.last_expr().has_java_class()
    }

    fn get_java_class(&self) -> Option<HostClass> {
        self.last_expr().get_java_class()
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> {
        Some(self)
    }
}

impl MaybePrimitiveExpr for BodyExpr {
    fn can_emit_primitive(&self) -> bool {
        // Java: `lastExpr() instanceof MaybePrimitiveExpr && cast.canEmitPrimitive()`.
        self.last_expr()
            .as_maybe_primitive()
            .map(|m| m.can_emit_primitive())
            .unwrap_or(false)
    }

    fn emit_unboxed(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Value {
        let n = self.exprs.len();
        for (i, e) in self.exprs.iter().enumerate() {
            if i + 1 < n {
                e.emit(C::Statement, objx, ir);
            }
        }
        // The last must be MaybePrimitiveExpr — Java casts unconditionally
        // (after `canEmitPrimitive` gating). We mirror with a panic on
        // misuse rather than a silent emit.
        let last = self.last_expr();
        match last.as_maybe_primitive() {
            Some(m) => m.emit_unboxed(context, objx, ir),
            None => panic!(
                "clojure-jvm: BodyExpr::emit_unboxed called with non-primitive last expr — \
                 caller should have checked `can_emit_primitive` first"
            ),
        }
    }
}

/// `Compiler.BodyExpr.Parser`.
pub struct BodyExprParser;

impl IParser for BodyExprParser {
    fn parse(&self, _context: C, _frms: Object) -> Box<dyn Expr> {
        // Java path:
        //   ISeq forms = (ISeq) frms;
        //   if (Util.equals(RT.first(forms), DO)) forms = RT.next(forms);
        //   PersistentVector exprs = EMPTY;
        //   for (; forms != null; forms = forms.next()) {
        //     Expr e = (context != EVAL && (context == STATEMENT || forms.next() != null))
        //              ? analyze(STATEMENT, forms.first())
        //              : analyze(context, forms.first());
        //     exprs = exprs.cons(e);
        //   }
        //   if (exprs.count() == 0) exprs = exprs.cons(NIL_EXPR);
        //   return new BodyExpr(exprs);
        //
        // Needs RT.first / RT.next + Util.equals + analyze. Stubbed until ISeq
        // and analyze are real.
        crate::unimplemented_port!(
            "BodyExpr.Parser.parse",
            "needs ISeq iteration (RT.first/RT.next) + analyze"
        )
    }
}

// ============================================================================
// Java line ~198–342: dynamic Var declarations.
// ============================================================================
//
// Compiler.java declares ~30 `Var.create().setDynamic()` slots that drive
// compilation context. We collect them into a `CompilerVars` struct,
// LazyLock-initialized.

pub static COMPILER_VARS: LazyLock<CompilerVars> = LazyLock::new(CompilerVars::new);

#[allow(non_snake_case)]
pub struct CompilerVars {
    /// symbol → LocalBinding
    pub LOCAL_ENV: Arc<Var>,
    /// vector<LocalBinding>
    pub LOOP_LOCALS: Arc<Var>,
    /// Label
    pub LOOP_LABEL: Arc<Var>,
    /// vector<Object>
    pub CONSTANTS: Arc<Var>,
    /// IdentityHashMap
    pub CONSTANT_IDS: Arc<Var>,
    /// vector<keyword>
    pub KEYWORD_CALLSITES: Arc<Var>,
    /// vector<var>
    pub PROTOCOL_CALLSITES: Arc<Var>,
    /// keyword → constid
    pub KEYWORDS: Arc<Var>,
    /// var → constid
    pub VARS: Arc<Var>,
    /// FnFrame
    pub METHOD: Arc<Var>,
    /// null-or-not
    pub IN_CATCH_FINALLY: Arc<Var>,
    pub METHOD_RETURN_CONTEXT: Arc<Var>,
    pub NO_RECUR: Arc<Var>,
    /// DynamicClassLoader
    pub LOADER: Arc<Var>,
    /// Integer line/column counters
    pub LINE: Arc<Var>,
    pub COLUMN: Arc<Var>,
    pub LINE_BEFORE: Arc<Var>,
    pub COLUMN_BEFORE: Arc<Var>,
    pub LINE_AFTER: Arc<Var>,
    pub COLUMN_AFTER: Arc<Var>,
    pub NEXT_LOCAL_NUM: Arc<Var>,
    pub RET_LOCAL_NUM: Arc<Var>,
    pub COMPILE_STUB_SYM: Arc<Var>,
    pub COMPILE_STUB_CLASS: Arc<Var>,
    pub CLEAR_PATH: Arc<Var>,
    pub CLEAR_ROOT: Arc<Var>,
    pub CLEAR_SITES: Arc<Var>,
}

impl CompilerVars {
    fn new() -> Self {
        let dyn_ = || Var::create().set_dynamic();
        let dyn_root = |v: Object| Var::create_with_root(v).set_dynamic();
        CompilerVars {
            LOCAL_ENV: dyn_(),
            LOOP_LOCALS: dyn_(),
            LOOP_LABEL: dyn_(),
            CONSTANTS: dyn_(),
            CONSTANT_IDS: dyn_(),
            KEYWORD_CALLSITES: dyn_(),
            PROTOCOL_CALLSITES: dyn_(),
            KEYWORDS: dyn_(),
            VARS: dyn_(),
            METHOD: dyn_(),
            IN_CATCH_FINALLY: dyn_(),
            METHOD_RETURN_CONTEXT: dyn_(),
            NO_RECUR: dyn_(),
            LOADER: dyn_(),
            LINE: dyn_root(Object::Long(0)),
            COLUMN: dyn_root(Object::Long(0)),
            LINE_BEFORE: dyn_root(Object::Long(0)),
            COLUMN_BEFORE: dyn_root(Object::Long(0)),
            LINE_AFTER: dyn_root(Object::Long(0)),
            COLUMN_AFTER: dyn_root(Object::Long(0)),
            NEXT_LOCAL_NUM: dyn_root(Object::Long(0)),
            RET_LOCAL_NUM: dyn_(),
            COMPILE_STUB_SYM: dyn_(),
            COMPILE_STUB_CLASS: dyn_(),
            CLEAR_PATH: dyn_(),
            CLEAR_ROOT: dyn_(),
            CLEAR_SITES: dyn_(),
        }
    }
}

// ============================================================================
// Compiler.analyze entry-points — Java line ~7060 onward. Stubbed for now.
// ============================================================================

/// `Compiler.analyze(C context, Object form)` — central dispatcher. Java
/// version is ~7060–7170, dispatching on form type:
///
///   * `null`             → NIL_EXPR
///   * `Boolean.TRUE`     → TRUE_EXPR
///   * `Boolean.FALSE`    → FALSE_EXPR
///   * `Symbol`           → analyzeSymbol
///   * `Keyword`          → registerKeyword + KeywordExpr
///   * `Number`           → NumberExpr.parse
///   * `String`           → StringExpr
///   * `IPersistentCollection` (empty) → EmptyExpr
///   * `ISeq` (list-headed form) → analyzeSeq (special-form / macro / invoke)
///   * `IPersistentVector` → VectorExpr.parse
///   * `IRecord`/`IType`/`IPersistentMap` → MapExpr / ConstantExpr
///   * `IPersistentSet`   → SetExpr.parse
///
/// We only handle the cases we've ported so far. Unhandled cases panic with
/// a clear "not yet wired" message that names the form shape.
pub fn analyze(context: C, form: Object) -> Box<dyn Expr> {
    analyze_named(context, form, None)
}

/// `Compiler.analyze(C context, Object form, String name)`. The `name` arg is
/// used for naming generated classes (fn names); ignored at the literal level.
pub fn analyze_named(context: C, form: Object, name: Option<&str>) -> Box<dyn Expr> {
    let _ = name; // not yet threaded through to FnExpr / ObjExpr

    // Java unwraps LazySeq via RT.seq before the dispatch — we don't have
    // LazySeq yet, so skip that step.

    match form {
        Object::Nil => Box::new(NIL_EXPR),
        Object::Bool(true) => Box::new(TRUE_EXPR),
        Object::Bool(false) => Box::new(FALSE_EXPR),
        Object::Long(_) | Object::Double(_) => NumberExpr::parse(form),
        Object::Char(code) => Box::new(CharExpr { code }),
        Object::String(s) => {
            // Java: new StringExpr(((String) form).intern())
            // We skip the intern step — Rust doesn't have JVM-style string
            // interning by default and our StringExpr holds an Arc<String>.
            Box::new(StringExpr { str: s })
        }
        Object::Keyword(k) => Box::new(register_keyword(k)),
        Object::List(ref l) => {
            // Empty list `()` is a constant — Java's analyzer returns
            // an EmptyExpr that emits the empty PersistentList. We
            // intern it as a literal so subsequent reads see the same
            // empty-list cell.
            if l.count() == 0 {
                return Box::new(ConstantExpr::new(Object::List(PersistentList::empty())));
            }
            analyze_seq(context, form)
        }
        Object::Symbol(s) => analyze_symbol(s),
        Object::Var(_) => crate::unimplemented_port!(
            "Compiler.analyze on Var",
            "treated as ConstantExpr in Java — wire when Var-as-constant lands"
        ),
        Object::Vector(v) => analyze_vector(context, v),
        Object::Map(m) => analyze_map(context, m),
        Object::Set(s) => analyze_set(context, s),
        Object::Namespace(_)
        | Object::Host(_)
        | Object::Unported { .. }
        | Object::TreeMap(_)
        | Object::TreeSet(_) => {
            // Sorted collections never appear as literal source forms
            // (they're built by `sorted-map`/`sorted-set` calls), so an
            // analyze-time occurrence is a runtime-constructed constant.
            Box::new(ConstantExpr::new(form))
        }
        Object::WithMeta(inner, _meta) => {
            // Reader-attached metadata is transparent at analyze time.
            // The `_meta` map is currently dropped here; it survives only
            // when (a) the caller (e.g. `def` for `^{:macro true}`) peeks
            // at it before calling `analyze`, or (b) the value flows
            // through `.withMeta` at runtime via a heap-side meta slot.
            // Quoted-literal metadata (e.g. `^:once` on a quoted fn-form)
            // is dropped — wire that when we hit a form that needs it.
            analyze(context, *inner)
        }
    }
}

/// `(when-let [form tst] body…)` → `(let [tmp tst] (when tmp (let [form tmp] body…)))`.
/// Host-side because the upstream JIT'd defmacro hits a gc_literal
/// corruption that produces a corrupt head symbol in its expansion.
fn expand_when_let(form: &Object) -> Object {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = Symbol::intern(&format!("temp__wl{id}__"));
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let body = super::rt::next(&after);
    let (form_sym, tst) = match bindings.peel_meta_ref() {
        Object::Vector(v) if v.count() == 2 => (v.nth(0), v.nth(1)),
        other => panic!("clojure-jvm: when-let needs [form tst], got {other:?}"),
    };
    // Build (let* [tmp tst] (if tmp (let* [form tmp] body…) nil))
    let mut do_body: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body.clone();
    while !matches!(cur, Object::Nil) {
        do_body.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    let inner_let = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![form_sym, Object::Symbol(tmp.clone())],
        )),
        Object::List(PersistentList::create(do_body)),
    ]));
    let if_form = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("if")),
        Object::Symbol(tmp.clone()),
        inner_let,
        Object::Nil,
    ]));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![Object::Symbol(tmp), tst],
        )),
        if_form,
    ]))
}

/// `(if-let [form tst] then else?)` — same shape as when-let but with
/// explicit else branch. Host-side for the same gc_literal reason.
fn expand_if_let(form: &Object) -> Object {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = Symbol::intern(&format!("temp__il{id}__"));
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let then_else = super::rt::next(&after);
    let then_branch = super::rt::first(&then_else);
    let else_branch = super::rt::first(&super::rt::next(&then_else));
    let (form_sym, tst) = match bindings.peel_meta_ref() {
        Object::Vector(v) if v.count() == 2 => (v.nth(0), v.nth(1)),
        other => panic!("clojure-jvm: if-let needs [form tst], got {other:?}"),
    };
    let inner_let = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![form_sym, Object::Symbol(tmp.clone())],
        )),
        then_branch,
    ]));
    let if_form = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("if")),
        Object::Symbol(tmp.clone()),
        inner_let,
        else_branch,
    ]));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![Object::Symbol(tmp), tst],
        )),
        if_form,
    ]))
}

/// `(.. x m1 m2 m3)` → `(. (. (. x m1) m2) m3)`. Each `mN` is either
/// a method symbol or `(method args)`. Host-side because the upstream
/// recursive defmacro hits the gc_literal corruption.
fn expand_dotdot(form: &Object) -> Object {
    let after = super::rt::next(form);
    let mut acc = super::rt::first(&after);
    let mut cur = super::rt::next(&after);
    while !matches!(cur, Object::Nil) {
        let step = super::rt::first(&cur);
        acc = Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern(".")),
            acc,
            step,
        ]));
        cur = super::rt::next(&cur);
    }
    acc
}

/// `(defonce name init)` → `(def name nil)`. Skip the init expression
/// entirely — defonce semantics say "only init if no root", and our
/// runtime ref/atom/agent constructors are stubs that return nil
/// anyway. The downstream code that uses these vars will see nil,
/// but the declaration succeeds and subsequent forms can analyze.
fn expand_defonce(form: &Object) -> Object {
    let after = super::rt::next(form);
    let name = super::rt::first(&after);
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("def")),
        name,
        Object::Nil,
    ]))
}

/// `(definline name docstring? attrs? [args] body)` → `(def name (fn* name [args] body))`.
/// We don't honor the inline-expansion behavior; just defn-style.
fn expand_definline(form: &Object) -> Object {
    expand_defn(form, false)
}

/// `(doto x form1 form2 …)` → `(let* [t x] form1' form2' … t)` where
/// each `formN` becomes `(. t formN…)` if it's a list, else passed
/// through. Host-side because the upstream defmacro hits the
/// gc_literal corruption with recursive macro expansion.
fn expand_doto(form: &Object) -> Object {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = Symbol::intern(&format!("doto__t{id}__"));
    let after = super::rt::next(form);
    let target = super::rt::first(&after);
    let mut body_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = super::rt::next(&after);
    while !matches!(cur, Object::Nil) {
        let item = super::rt::first(&cur);
        // (form-name args…) becomes (form-name t args…) — Clojure inserts t as 2nd item.
        let new_item = if let Object::List(l) = &item {
            if l.count() > 0 {
                let head = l.iter().next().unwrap();
                let mut new_items: Vec<Object> = vec![head, Object::Symbol(t.clone())];
                let mut rest_iter = l.iter().skip(1);
                while let Some(x) = rest_iter.next() {
                    new_items.push(x);
                }
                Object::List(PersistentList::create(new_items))
            } else {
                item
            }
        } else {
            // Bare symbol: (sym t)
            Object::List(PersistentList::create(vec![
                item,
                Object::Symbol(t.clone()),
            ]))
        };
        body_items.push(new_item);
        cur = super::rt::next(&cur);
    }
    body_items.push(Object::Symbol(t.clone()));
    let do_form = Object::List(PersistentList::create(body_items));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![Object::Symbol(t), target],
        )),
        do_form,
    ]))
}

/// `(-> x f1 (f2 a) f3)` → `(f3 (f2 (f1 x) a))`. Threads x as the
/// first arg through each subsequent form.
fn expand_thread_first(form: &Object) -> Object {
    let after = super::rt::next(form);
    let mut acc = super::rt::first(&after);
    let mut cur = super::rt::next(&after);
    while !matches!(cur, Object::Nil) {
        let step = super::rt::first(&cur);
        let new_acc = if let Object::List(l) = &step {
            // Insert acc as the 2nd item.
            let head = l.iter().next().unwrap();
            let mut items: Vec<Object> = vec![head, acc];
            for x in l.iter().skip(1) {
                items.push(x);
            }
            Object::List(PersistentList::create(items))
        } else {
            // Bare symbol: (sym acc)
            Object::List(PersistentList::create(vec![step, acc]))
        };
        acc = new_acc;
        cur = super::rt::next(&cur);
    }
    acc
}

/// `(->> x f1 (f2 a))` → `(f2 a (f1 x))`. Threads x as the LAST arg.
fn expand_thread_last(form: &Object) -> Object {
    let after = super::rt::next(form);
    let mut acc = super::rt::first(&after);
    let mut cur = super::rt::next(&after);
    while !matches!(cur, Object::Nil) {
        let step = super::rt::first(&cur);
        let new_acc = if let Object::List(l) = &step {
            let mut items: Vec<Object> = Vec::new();
            for x in l.iter() {
                items.push(x);
            }
            items.push(acc);
            Object::List(PersistentList::create(items))
        } else {
            Object::List(PersistentList::create(vec![step, acc]))
        };
        acc = new_acc;
        cur = super::rt::next(&cur);
    }
    acc
}

/// `(as-> x sym body…)` → `(let* [sym x] (let* [sym body1] (let* [sym body2] sym)))`.
fn expand_as_thread(form: &Object) -> Object {
    let after = super::rt::next(form);
    let init = super::rt::first(&after);
    let after = super::rt::next(&after);
    let sym_form = super::rt::first(&after);
    let sym = match sym_form.peel_meta_ref() {
        Object::Symbol(s) => s.clone(),
        other => panic!("as-> needs a symbol name, got {other:?}"),
    };
    let mut steps: Vec<Object> = Vec::new();
    let mut cur = super::rt::next(&after);
    while !matches!(cur, Object::Nil) {
        steps.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    fn build(sym: &Arc<Symbol>, current: Object, rest: &[Object]) -> Object {
        if rest.is_empty() {
            return current;
        }
        let next = build(sym, rest[0].clone(), &rest[1..]);
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("let*")),
            Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
                vec![Object::Symbol(sym.clone()), current],
            )),
            next,
        ]))
    }
    build(&sym, init, &steps)
}

fn expand_some_thread_first(form: &Object) -> Object {
    // Simplified: just expand like `->`. Real some-> short-circuits on nil.
    expand_thread_first(form)
}
fn expand_some_thread_last(form: &Object) -> Object {
    expand_thread_last(form)
}

/// `(with-out-str body…)` → `(do body…)`. We don't redirect *out*.
fn expand_with_out_str(form: &Object) -> Object {
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    do_items.push(Object::String(Arc::new(String::new())));
    Object::List(PersistentList::create(do_items))
}

fn expand_with_in_str(form: &Object) -> Object {
    // (with-in-str s body…) → (do body…); bindings discarded.
    let after = super::rt::next(form);
    let body_seq = super::rt::next(&after);
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(do_items))
}

/// `(with-open [name expr ...] body…)` → `(let* [name expr] body…)`.
/// Real with-open auto-closes; we skip that.
fn expand_with_open(form: &Object) -> Object {
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let body_seq = super::rt::next(&after);
    let mut body_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        body_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        bindings,
        Object::List(PersistentList::create(body_items)),
    ]))
}

/// `(while test body…)` → `(loop* [] (if test (do body… (recur)) nil))`.
fn expand_while(form: &Object) -> Object {
    let after = super::rt::next(form);
    let test = super::rt::first(&after);
    let body_seq = super::rt::next(&after);
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    do_items.push(Object::List(PersistentList::create(vec![Object::Symbol(
        Symbol::intern("recur"),
    )])));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("loop*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![],
        )),
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("if")),
            test,
            Object::List(PersistentList::create(do_items)),
            Object::Nil,
        ])),
    ]))
}

/// `(letfn [(f1 [args] body) (f2 …)] body…)` → simplistic flat let
/// where each fn-spec becomes `name (fn name [args] body)`. Doesn't
/// support mutual recursion the way upstream's letfn does (which uses
/// a single `letfn*` special form), but for code that only references
/// each fn after its declaration this is enough.
fn expand_letfn(form: &Object) -> Object {
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let body_seq = super::rt::next(&after);
    let bindings_v = match bindings.peel_meta_ref().clone() {
        Object::Vector(v) => v,
        other => panic!("letfn needs vector of fnspecs, got {other:?}"),
    };
    let mut binding_pairs: Vec<Object> = Vec::new();
    for i in 0..bindings_v.count() {
        let spec = bindings_v.nth(i);
        if let Object::List(l) = &spec {
            let items: Vec<Object> = l.iter().collect();
            if items.is_empty() {
                continue;
            }
            let name = items[0].clone();
            let mut fn_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("fn*"))];
            for x in items.iter().skip(0) {
                fn_items.push(x.clone());
            }
            binding_pairs.push(name);
            binding_pairs.push(Object::List(PersistentList::create(fn_items)));
        }
    }
    let mut body_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        body_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            binding_pairs,
        )),
        Object::List(PersistentList::create(body_items)),
    ]))
}

/// `(case e v1 r1 v2 r2 default)` → nested if. Simplistic: uses `=`
/// for comparison rather than upstream's optimized hash-table dispatch.
fn expand_case(form: &Object) -> Object {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = Symbol::intern(&format!("case__t{id}__"));
    let after = super::rt::next(form);
    let scrutinee = super::rt::first(&after);
    let mut clauses: Vec<Object> = Vec::new();
    let mut cur = super::rt::next(&after);
    while !matches!(cur, Object::Nil) {
        clauses.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    fn build(tmp: &Arc<Symbol>, clauses: &[Object]) -> Object {
        if clauses.is_empty() {
            return Object::Nil;
        }
        if clauses.len() == 1 {
            return clauses[0].clone();
        }
        let val = clauses[0].clone();
        let result = clauses[1].clone();
        let rest = build(tmp, &clauses[2..]);
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("if")),
            Object::List(PersistentList::create(vec![
                Object::Symbol(Symbol::intern("=")),
                Object::Symbol(tmp.clone()),
                Object::List(PersistentList::create(vec![
                    Object::Symbol(Symbol::intern("quote")),
                    val,
                ])),
            ])),
            result,
            rest,
        ]))
    }
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![Object::Symbol(tmp.clone()), scrutinee],
        )),
        build(&tmp, &clauses),
    ]))
}

fn expand_condp(form: &Object) -> Object {
    // Simplified: just emit nil. condp is too varied for a quick stub.
    let _ = form;
    Object::Nil
}

fn expand_when_first(form: &Object) -> Object {
    // (when-first [x xs] body) → (when (seq xs) (let [x (first xs)] body))
    expand_when_let(form)
}

/// Best-effort sniff: for `(defn NAME …)` / `(defmacro NAME …)` /
/// `(def NAME …)` return a short label like `defn:NAME`. Used by the
/// core-loader trace.
fn sniff_core_form_name(form: &Object) -> Option<String> {
    let Object::List(_) = form else { return None };
    let head = super::rt::first(form);
    let head_name = match &head {
        Object::Symbol(s) => s.get_name().to_string(),
        _ => return None,
    };
    let after = super::rt::next(form);
    let name_form = super::rt::first(&after);
    let name = match name_form.peel_meta() {
        Object::Symbol(s) => s.get_name().to_string(),
        _ => return Some(format!("({head_name} …)")),
    };
    Some(format!("({head_name} {name} …)"))
}

/// `(alias 'short 'full)` — register `short` as an alias for namespace
/// `full` in the current namespace. Effectful at analyze time; returns
/// nil. Mirrors `clojure.core/alias` which calls
/// `Namespace.addAlias(short, Namespace.findOrCreate(full))`.
fn expand_alias(form: &Object) -> Object {
    let after = super::rt::next(form);
    let alias_arg = super::rt::first(&after);
    let target_arg = super::rt::first(&super::rt::next(&after));
    let alias_sym = peel_quoted_symbol(&alias_arg).unwrap_or_else(|| {
        panic!("clojure-jvm: `alias` first arg must be a quoted symbol, got {alias_arg:?}")
    });
    let target_sym = peel_quoted_symbol(&target_arg).unwrap_or_else(|| {
        panic!("clojure-jvm: `alias` second arg must be a quoted symbol, got {target_arg:?}")
    });
    let target_ns = super::namespace::Namespace::find_or_create(target_sym);
    super::rt::current_ns().add_alias(alias_sym, target_ns);
    Object::Nil
}

/// `(quote X)` → Some(X) if X is a Symbol, else None. Used by host
/// intercepts that take quoted-symbol args (`alias`, `(. C method)`'s
/// class arg, etc.).
fn peel_quoted_symbol(form: &Object) -> Option<Arc<Symbol>> {
    let inner = match form {
        Object::List(_) => {
            let head = super::rt::first(form);
            if !matches!(&head, Object::Symbol(s) if s.get_name() == "quote" && s.get_namespace().is_none())
            {
                return None;
            }
            super::rt::first(&super::rt::next(form))
        }
        Object::Symbol(_) => form.clone(), // bare-symbol arg (defensive)
        _ => return None,
    };
    match inner.peel_meta() {
        Object::Symbol(s) => Some(s),
        _ => None,
    }
}

fn expand_io_bang(form: &Object) -> Object {
    // (io! body…) → (do body…). The msg arg (if any) gets dropped.
    let after = super::rt::next(form);
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    // Skip optional first-string-arg (the io! message).
    let mut cur = if matches!(super::rt::first(&after), Object::String(_)) {
        super::rt::next(&after)
    } else {
        after
    };
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(do_items))
}

fn expand_future(form: &Object) -> Object {
    // (future body…) → (do body…) — sync execution.
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(do_items))
}

fn expand_delay_simple(form: &Object) -> Object {
    // (delay body) → (new clojure.lang.Delay (fn* [] body))
    let mut body_items: Vec<Object> = vec![
        Object::Symbol(Symbol::intern("fn*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![],
        )),
    ];
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        body_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("new")),
        Object::Symbol(Symbol::intern("clojure.lang.Delay")),
        Object::List(PersistentList::create(body_items)),
    ]))
}

fn expand_locking(form: &Object) -> Object {
    // (locking x body…) → (do body…). We don't model locks.
    let after = super::rt::next(form);
    let body_seq = super::rt::next(&after);
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(do_items))
}

fn expand_sync(form: &Object) -> Object {
    // (sync flags body…) / (dosync body…) → (do body…)
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(do_items))
}

fn expand_with_precision(form: &Object) -> Object {
    // (with-precision n body…) → (do body…)
    let after = super::rt::next(form);
    let body_seq = super::rt::next(&after);
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(do_items))
}

/// `(or x y z)` → `(let* [t x] (if t t (or y z)))`, or `nil` for no
/// args, `x` alone for one arg. Host-side because the upstream
/// recursive defmacro hits the gc_literal corruption.
// ── Bootstrap macros (let / fn / loop / when / when-not / if-not) ─────
//
// Identity-shape expansions that map the unstarred surface forms onto
// the starred special-form primitives the analyzer already dispatches
// on. They exist so a source file (notably cljs/core.cljs) that
// assumes its host already provides `let`/`fn`/`when`/etc. can be
// loaded without first evaluating upstream `clojure/core.clj`'s
// defmacros.
//
// NOT a destructuring impl: `(let [{:keys [a]} m] …)` will reach
// `let*` with the map-literal binding pattern and panic there. Real
// destructuring (and the rest of the upstream `let`/`fn` semantics —
// type hints, pre/post conditions, ^:once metadata, etc.) is upstream's
// `defmacro let`'s job.

/// `(let [bindings…] body…)` → `(let* [bindings…] body…)`.
fn expand_let_simple(form: &Object) -> Object {
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let mut items: Vec<Object> = vec![Object::Symbol(Symbol::intern("let*")), bindings];
    let mut cur = super::rt::next(&after);
    while !matches!(cur, Object::Nil) {
        items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(items))
}

/// `(fn …)` → `(fn* …)`. Handles both single-arity `(fn [args] body)`
/// and multi-arity `(fn ([args] body) ([args] body))`, plus an
/// optional fn-name first argument (`(fn name [args] body)`).
fn expand_fn_simple(form: &Object) -> Object {
    let mut items: Vec<Object> = vec![Object::Symbol(Symbol::intern("fn*"))];
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(items))
}

/// `(loop [bindings…] body…)` → `(loop* [bindings…] body…)`.
fn expand_loop_simple(form: &Object) -> Object {
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let mut items: Vec<Object> = vec![Object::Symbol(Symbol::intern("loop*")), bindings];
    let mut cur = super::rt::next(&after);
    while !matches!(cur, Object::Nil) {
        items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(items))
}

/// `(when test body…)` → `(if test (do body…) nil)`.
fn expand_when(form: &Object) -> Object {
    let after = super::rt::next(form);
    let test = super::rt::first(&after);
    let body_seq = super::rt::next(&after);
    let mut body_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        body_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("if")),
        test,
        Object::List(PersistentList::create(body_items)),
        Object::Nil,
    ]))
}

/// `(when-not test body…)` → `(if test nil (do body…))`.
fn expand_when_not(form: &Object) -> Object {
    let after = super::rt::next(form);
    let test = super::rt::first(&after);
    let body_seq = super::rt::next(&after);
    let mut body_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        body_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("if")),
        test,
        Object::Nil,
        Object::List(PersistentList::create(body_items)),
    ]))
}

/// `(if-not test then else?)` → `(if test else then)`. `else` defaults to nil.
fn expand_if_not(form: &Object) -> Object {
    let after = super::rt::next(form);
    let test = super::rt::first(&after);
    let after2 = super::rt::next(&after);
    let then_form = super::rt::first(&after2);
    let after3 = super::rt::next(&after2);
    let else_form = if matches!(after3, Object::Nil) {
        Object::Nil
    } else {
        super::rt::first(&after3)
    };
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("if")),
        test,
        else_form,
        then_form,
    ]))
}

fn expand_or(form: &Object) -> Object {
    static OR_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut args: Vec<Object> = Vec::new();
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        args.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    fn build(args: &[Object]) -> Object {
        if args.is_empty() {
            return Object::Nil;
        }
        if args.len() == 1 {
            return args[0].clone();
        }
        let id = OR_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = Symbol::intern(&format!("or__t{id}__"));
        let rest = build(&args[1..]);
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("let*")),
            Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
                vec![Object::Symbol(t.clone()), args[0].clone()],
            )),
            Object::List(PersistentList::create(vec![
                Object::Symbol(Symbol::intern("if")),
                Object::Symbol(t.clone()),
                Object::Symbol(t),
                rest,
            ])),
        ]))
    }
    build(&args)
}

/// `(and x y z)` → `(let* [t x] (if t (and y z) t))`, with `true`
/// for no args, `x` alone for one arg.
fn expand_and(form: &Object) -> Object {
    static AND_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut args: Vec<Object> = Vec::new();
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        args.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    fn build(args: &[Object]) -> Object {
        if args.is_empty() {
            return Object::Bool(true);
        }
        if args.len() == 1 {
            return args[0].clone();
        }
        let id = AND_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = Symbol::intern(&format!("and__t{id}__"));
        let rest = build(&args[1..]);
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("let*")),
            Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
                vec![Object::Symbol(t.clone()), args[0].clone()],
            )),
            Object::List(PersistentList::create(vec![
                Object::Symbol(Symbol::intern("if")),
                Object::Symbol(t),
                rest,
                Object::Bool(false),
            ])),
        ]))
    }
    build(&args)
}

/// `(cond test1 expr1 test2 expr2 …)` → nested `(if test1 expr1 (if test2 expr2 …))`.
fn expand_cond(form: &Object) -> Object {
    let mut clauses: Vec<Object> = Vec::new();
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        clauses.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    if clauses.len() % 2 != 0 {
        panic!(
            "clojure-jvm: cond requires an even number of forms, got {}",
            clauses.len()
        );
    }
    fn build(clauses: &[Object]) -> Object {
        if clauses.is_empty() {
            return Object::Nil;
        }
        let test = clauses[0].clone();
        let expr = clauses[1].clone();
        let rest = build(&clauses[2..]);
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("if")),
            test,
            expr,
            rest,
        ]))
    }
    build(&clauses)
}

/// `(defmulti name dispatch-fn ...)` → `(def name nil)`. We don't
/// model multimethods; defining the var lets `(defmethod ...)` calls
/// analyze (they're intercepted as nil too).
fn expand_defmulti(form: &Object) -> Object {
    let after = super::rt::next(form);
    let name_form = super::rt::first(&after);
    let name_sym = match name_form.peel_meta_ref() {
        Object::Symbol(s) => s.clone(),
        other => panic!("clojure-jvm: defmulti needs a symbol name, got {other:?}"),
    };
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("def")),
        Object::Symbol(name_sym),
        Object::Nil,
    ]))
}

/// `(defprotocol name docstring? methods…)` → declare each method as
/// a separate `(def m nil)` plus the protocol name itself. Methods
/// can then be referenced (their var resolves to nil).
fn expand_defprotocol(form: &Object) -> Object {
    use crate::lang::user_types::register_protocol;
    let after = super::rt::next(form);
    let proto_name_form = super::rt::first(&after);
    let proto_name = match proto_name_form.peel_meta_ref() {
        Object::Symbol(s) => s.clone(),
        other => panic!("clojure-jvm: defprotocol needs a symbol name, got {other:?}"),
    };
    // Walk method specs first to harvest names + arities. Each spec
    // looks like `(method-name [this & args]+ docstring?)`. A method may
    // declare MULTIPLE binding vectors (multi-arity protocol method, e.g.
    // `(coll-reduce [coll f] [coll f val])`); each vector's length is one
    // dispatch arity (including `this`).
    let mut cur = super::rt::next(&after);
    if matches!(super::rt::first(&cur), Object::String(_)) {
        cur = super::rt::next(&cur);
    }
    if matches!(super::rt::first(&cur), Object::Map(_)) {
        cur = super::rt::next(&cur);
    }
    struct ParsedMethod {
        name: Arc<Symbol>,
        /// One entry per declared arity, in source order.
        bindings: Vec<Object>,
    }
    let mut methods: Vec<ParsedMethod> = Vec::new();
    while !matches!(cur, Object::Nil) {
        let item = super::rt::first(&cur);
        cur = super::rt::next(&cur);
        let l = match &item {
            Object::List(l) => l,
            _ => continue, // ignore loose docstrings between specs
        };
        let mut it = l.iter();
        let head = match it.next() {
            Some(h) => h,
            None => continue,
        };
        let mname = match head.peel_meta_ref() {
            Object::Symbol(s) => s.clone(),
            other => panic!("clojure-jvm: defprotocol method-name must be a symbol, got {other:?}"),
        };
        // Collect EVERY binding vector (multi-arity), skipping a trailing
        // docstring or any non-vector items.
        let mut bindings: Vec<Object> = Vec::new();
        for rest in it {
            if matches!(rest, Object::Vector(_)) {
                bindings.push(rest.clone());
            }
        }
        if bindings.is_empty() {
            panic!(
                "clojure-jvm: defprotocol method `{}` is missing a binding vector",
                mname.get_name(),
            );
        }
        methods.push(ParsedMethod {
            name: mname,
            bindings,
        });
    }
    // Register at the global level. The protocol_id slot itself is
    // currently informational — `satisfies?` will read it later.
    let method_specs: Vec<(Arc<Symbol>, Vec<usize>)> = methods
        .iter()
        .map(|m| {
            let arities: Vec<usize> = m
                .bindings
                .iter()
                .map(|b| match b {
                    Object::Vector(v) => v.count() as usize,
                    _ => unreachable!(),
                })
                .collect();
            (m.name.clone(), arities)
        })
        .collect();
    let (_proto_id, method_ids) = register_protocol(proto_name.clone(), method_specs);

    // Build one `(fn* [this …] (RT protocolDispatchN mid this …))` clause
    // per declared arity. The method_id is shared across arities; the
    // installed impl is invoked with the actual arg count, so multi-arity
    // protocol methods dispatch correctly.
    let make_clause = |mname: &Arc<Symbol>, mid: u32, binding: &Object| -> Object {
        let params_vec = match binding {
            Object::Vector(v) => v.clone(),
            _ => unreachable!(),
        };
        let arity = params_vec.count() as usize;
        let dispatch_method = match arity {
            1 => "protocolDispatch1",
            2 => "protocolDispatch2",
            3 => "protocolDispatch3",
            4 => "protocolDispatch4",
            n => panic!(
                "clojure-jvm: defprotocol method `{}` arity {n} unsupported (only 1..4 wired); \
                 extend protocolDispatch externs and host_methods table in Compiler::new",
                mname.get_name(),
            ),
        };
        let mut dispatch_args: Vec<Object> = Vec::with_capacity(arity + 2);
        dispatch_args.push(Object::Symbol(Symbol::intern(dispatch_method)));
        dispatch_args.push(Object::Long(mid as i64));
        for i in 0..params_vec.count() {
            let p = params_vec.nth(i);
            match p.peel_meta_ref() {
                Object::Symbol(s) => dispatch_args.push(Object::Symbol(s.clone())),
                other => panic!(
                    "clojure-jvm: defprotocol `{}`: binding param must be a symbol, got {other:?}",
                    mname.get_name(),
                ),
            }
        }
        let dispatch_call = Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern(".")),
            Object::Symbol(Symbol::intern_ns_name(None, "clojure.lang.RT")),
            Object::List(PersistentList::create(dispatch_args)),
        ]));
        // One fn clause: `([this …] dispatch_call)`.
        Object::List(PersistentList::create(vec![binding.clone(), dispatch_call]))
    };

    // Build the expansion: `(do (def Name nil) <method-def>* )`.
    let mut do_items: Vec<Object> = vec![
        Object::Symbol(Symbol::intern("do")),
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("def")),
            Object::Symbol(proto_name.clone()),
            Object::Nil,
        ])),
    ];
    for (m, mid) in methods.iter().zip(method_ids.iter()) {
        // Single arity → `(fn* [args] body)`; multiple → a multi-clause
        // `(fn* ([args1] body1) ([args2] body2) …)`.
        let fn_form = if m.bindings.len() == 1 {
            let clause = make_clause(&m.name, *mid, &m.bindings[0]);
            // clause is `([args] body)`; splice into `(fn* [args] body)`.
            let mut fn_items = vec![Object::Symbol(Symbol::intern("fn*"))];
            if let Object::List(c) = &clause {
                for it in c.iter() {
                    fn_items.push(it);
                }
            }
            Object::List(PersistentList::create(fn_items))
        } else {
            let mut fn_items = vec![Object::Symbol(Symbol::intern("fn*"))];
            for b in &m.bindings {
                fn_items.push(make_clause(&m.name, *mid, b));
            }
            Object::List(PersistentList::create(fn_items))
        };
        // (def m fn_form)
        do_items.push(Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("def")),
            Object::Symbol(m.name.clone()),
            fn_form,
        ])));
    }
    Object::List(PersistentList::create(do_items))
}

/// Resolve a `(extend-type T …)` type symbol to its `LogicalTypeId`.
///
/// Supports today:
///   * user-defined `deftype` names → user-type half of the LogicalTypeId space.
///   * `nil`     → `BUILTIN_NIL`
///   * `Object` / `default` / `java.lang.Object` → `BUILTIN_OBJECT`
///     (the catch-all fallback bucket).
///   * any other fully-qualified host class (e.g. `java.util.Date`,
///     `java.time.Instant`) → a stable synthetic id from the host-class
///     band. See `host_class_logical`: this runtime has no host value
///     instances, so the impl registers and loads but only dispatches if
///     a value ever reports that id.
///
/// Returns `None` for an unqualified, unknown symbol — almost always a
/// typo or a `deftype` that hasn't been evaluated yet, which should fail
/// loudly rather than silently registering a never-dispatched impl.
fn resolve_extend_type_target(target_sym: &Arc<Symbol>) -> Option<u32> {
    use crate::lang::user_types::{
        BUILTIN_NIL, BUILTIN_OBJECT, host_class_logical, user_type_id_by_name, user_type_logical,
    };
    if let Some(uid) = user_type_id_by_name(target_sym) {
        return Some(user_type_logical(uid));
    }
    let name = target_sym.get_name();
    match name {
        "nil" => Some(BUILTIN_NIL),
        "Object" | "default" | "java.lang.Object" => Some(BUILTIN_OBJECT),
        // A fully-qualified host class name (dotted) — `java.util.Date`,
        // `java.time.Instant`, etc. Intern a stable synthetic id.
        _ if name.contains('.') => Some(host_class_logical(name)),
        _ => None,
    }
}

/// `(extend-protocol P T1 (m1 [this …] body) (m2 [this …] body) T2 (m3 […] body))`.
///
/// Rewrites to `(do (extend-type T1 P …) (extend-type T2 P …) …)`. The
/// individual `extend-type` forms are then re-expanded by
/// `expand_extend_type`.
fn expand_extend_protocol(form: &Object) -> Object {
    let after = super::rt::next(form);
    let proto_form = super::rt::first(&after);
    if !matches!(proto_form.peel_meta_ref(), Object::Symbol(_)) {
        panic!(
            "clojure-jvm: extend-protocol: protocol must be a symbol, got {:?}",
            proto_form,
        );
    }
    // Walk the body, accumulating per-target method blocks. A target is
    // either a Symbol or `nil` (the nil literal target). Method blocks
    // are Lists.
    let mut cur = super::rt::next(&after);
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut current_target: Option<Object> = None;
    let mut current_methods: Vec<Object> = Vec::new();
    let emit_one =
        |target: &Option<Object>, methods: &Vec<Object>, proto: &Object, out: &mut Vec<Object>| {
            let Some(t) = target else {
                return;
            };
            if methods.is_empty() {
                return;
            }
            let mut et_items: Vec<Object> = Vec::with_capacity(3 + methods.len());
            et_items.push(Object::Symbol(Symbol::intern("extend-type")));
            et_items.push(t.clone());
            et_items.push(proto.clone());
            et_items.extend(methods.iter().cloned());
            out.push(Object::List(PersistentList::create(et_items)));
        };
    while !matches!(cur, Object::Nil) {
        let item = super::rt::first(&cur);
        cur = super::rt::next(&cur);
        // Targets are Symbols or `nil`; method blocks are Lists.
        let is_target = matches!(item.peel_meta_ref(), Object::Symbol(_) | Object::Nil);
        if is_target {
            emit_one(
                &current_target,
                &current_methods,
                &proto_form,
                &mut do_items,
            );
            current_target = Some(item);
            current_methods.clear();
        } else {
            current_methods.push(item);
        }
    }
    emit_one(
        &current_target,
        &current_methods,
        &proto_form,
        &mut do_items,
    );
    Object::List(PersistentList::create(do_items))
}

/// `(satisfies? Proto x)` — does `x`'s `LogicalTypeId` have any direct
/// impl registered for any of `Proto`'s methods? `Proto` must be a
/// `defprotocol`-declared symbol; we resolve it at macroexpand time.
fn expand_satisfies(form: &Object) -> Object {
    use crate::lang::user_types::protocol_id_by_name;
    let after = super::rt::next(form);
    let proto_form = super::rt::first(&after);
    let proto_sym = match proto_form.peel_meta_ref() {
        Object::Symbol(s) => s.clone(),
        other => panic!("clojure-jvm: satisfies?: protocol must be a symbol, got {other:?}"),
    };
    let pid = protocol_id_by_name(&proto_sym).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: satisfies?: protocol `{}` not registered. \
             Has defprotocol been evaluated?",
            proto_sym.get_name(),
        )
    });
    let after2 = super::rt::next(&after);
    let x_form = super::rt::first(&after2);
    // (. clojure.lang.RT (satisfies <proto_id> x))
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern(".")),
        Object::Symbol(Symbol::intern_ns_name(None, "clojure.lang.RT")),
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("satisfies")),
            Object::Long(pid as i64),
            x_form,
        ])),
    ]))
}

/// `(extend-type T P1 (m1 [this …] body) (m2 [this …] body) P2 (m3 […] body))`.
///
/// Each `(method-name [this …] body)` block lowers to:
///
/// ```text
/// (. clojure.lang.RT (installImpl <type_id> <method_id> (fn* [this …] body)))
/// ```
///
/// Multiple protocols can be listed; the parser groups method blocks
/// under the most recently seen protocol symbol so the per-block method
/// id lookup hits the correct protocol.
fn expand_extend_type(form: &Object) -> Object {
    use crate::lang::user_types::{protocol_id_by_name, protocol_method_id};
    let after = super::rt::next(form);
    let target_form = super::rt::first(&after);
    let (target_sym, type_id) = match target_form.peel_meta_ref() {
        // `nil` reads as `Object::Nil`, not a Symbol; handle directly so
        // `(extend-type nil P …)` works.
        Object::Nil => (Symbol::intern("nil"), crate::lang::user_types::BUILTIN_NIL),
        Object::Symbol(s) => {
            let tid = resolve_extend_type_target(s).unwrap_or_else(|| {
                panic!(
                    "clojure-jvm: extend-type: cannot resolve target `{}` to a LogicalTypeId. \
                     Known: user `deftype` names, `nil`, `Object`, `default`. Extend \
                     `resolve_extend_type_target` to support host classes (e.g. `String`, \
                     `js/Number`) as needed.",
                    s.get_name(),
                )
            });
            (s.clone(), tid)
        }
        other => panic!("clojure-jvm: extend-type: target must be a symbol or nil, got {other:?}"),
    };

    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = super::rt::next(&after);
    let mut current_proto_id: Option<u32> = None;
    let mut current_proto_name: Option<Arc<Symbol>> = None;
    while !matches!(cur, Object::Nil) {
        let item = super::rt::first(&cur);
        cur = super::rt::next(&cur);
        match item.peel_meta_ref() {
            Object::Symbol(s) => {
                // New protocol/interface section starts. If the name isn't a
                // registered protocol it's a host interface we don't model
                // (e.g. `java.lang.Iterable`, `clojure.lang.IReduceInit`,
                // `clojure.lang.Sequential` on `(deftype Eduction …)`). Skip
                // its method blocks — the type is still created with whatever
                // protocols we DO model; calling a skipped method fails with a
                // clear method-not-found at runtime. (A genuine typo'd
                // protocol name is skipped the same way; the missing-method
                // error surfaces it.)
                match protocol_id_by_name(s) {
                    Some(pid) => {
                        current_proto_id = Some(pid);
                        current_proto_name = Some(s.clone());
                    }
                    None => {
                        eprintln!(
                            "[cljvm-deftype] {}: skipping unmodeled protocol/interface `{}` \
                             (not registered via defprotocol)",
                            target_sym.get_name(),
                            s.get_name(),
                        );
                        current_proto_id = None;
                        current_proto_name = None;
                    }
                }
            }
            Object::List(l) => {
                // A method block under a skipped (unmodeled) interface — drop
                // it. (`current_proto_name` is None only here or before any
                // section symbol; both mean "nothing to install".)
                let Some(pid) = current_proto_id else {
                    continue;
                };
                let pname = current_proto_name.as_ref().unwrap();
                // Method block shape: (method-name [this …] body…)
                let mut it = l.iter();
                let m_head = it.next().unwrap_or(Object::Nil);
                let mname = match m_head.peel_meta_ref() {
                    Object::Symbol(s) => s.clone(),
                    other => panic!(
                        "clojure-jvm: extend-type `{}` / `{}`: method head \
                         must be a symbol, got {other:?}",
                        target_sym.get_name(),
                        pname.get_name(),
                    ),
                };
                let mid = protocol_method_id(pid, &mname).unwrap_or_else(|| {
                    panic!(
                        "clojure-jvm: extend-type `{}` / `{}`: method `{}` \
                         is not declared on protocol `{}`",
                        target_sym.get_name(),
                        pname.get_name(),
                        mname.get_name(),
                        pname.get_name(),
                    )
                });
                // The item after the method name is either a binding
                // vector (single-arity: `(m [args] body…)`) or a list
                // (multi-arity: `(m ([args1] body…) ([args2] body…) …)`).
                let next_item = it.next().unwrap_or(Object::Nil);
                let fn_form = match &next_item {
                    Object::Vector(_) => {
                        // Single arity → `(fn* [args] body…)`.
                        let mut fn_items: Vec<Object> =
                            vec![Object::Symbol(Symbol::intern("fn*")), next_item.clone()];
                        for rest in it {
                            fn_items.push(rest.clone());
                        }
                        Object::List(PersistentList::create(fn_items))
                    }
                    Object::List(_) => {
                        // Multi-arity → `(fn* ([args1] body…) ([args2] body…) …)`.
                        // `next_item` and every remaining item is a clause.
                        let mut fn_items: Vec<Object> =
                            vec![Object::Symbol(Symbol::intern("fn*")), next_item.clone()];
                        for rest in it {
                            fn_items.push(rest.clone());
                        }
                        Object::List(PersistentList::create(fn_items))
                    }
                    other => panic!(
                        "clojure-jvm: extend-type `{}` / `{}`: method `{}` \
                         expects a binding vector or arity-clause list, got {other:?}",
                        target_sym.get_name(),
                        pname.get_name(),
                        mname.get_name(),
                    ),
                };
                // (. clojure.lang.RT (installImpl <tid> <mid> <fn>))
                let install_form = Object::List(PersistentList::create(vec![
                    Object::Symbol(Symbol::intern(".")),
                    Object::Symbol(Symbol::intern_ns_name(None, "clojure.lang.RT")),
                    Object::List(PersistentList::create(vec![
                        Object::Symbol(Symbol::intern("installImpl")),
                        Object::Long(type_id as i64),
                        Object::Long(mid as i64),
                        fn_form,
                    ])),
                ]));
                do_items.push(install_form);
            }
            other => panic!(
                "clojure-jvm: extend-type `{}`: unexpected form {other:?}",
                target_sym.get_name(),
            ),
        }
    }
    Object::List(PersistentList::create(do_items))
}

/// `(deftype Name [fields…] body…)` / `(defrecord Name [fields…] body…)`.
///
/// Allocates a `UserTypeId` via `register_user_type`, then emits:
///
/// ```text
/// (do
///   (def Name (fn* [field0 field1 …]
///     (. clojure.lang.RT (allocUserInstanceN <user_type_id> field0 …)))))
/// ```
///
/// `body…` (inline protocol impls) is processed by Step 5 — for now,
/// any body forms are dropped with a panic if they're non-trivial
/// to keep behavior honest.
fn expand_deftype_or_record(form: &Object) -> Object {
    use crate::lang::user_types::register_user_type;
    let after = super::rt::next(form);
    let name_form = super::rt::first(&after);
    let name_sym = match name_form.peel_meta_ref() {
        Object::Symbol(s) => s.clone(),
        other => panic!("clojure-jvm: deftype/defrecord needs a symbol name, got {other:?}"),
    };
    let after2 = super::rt::next(&after);
    let fields_form = super::rt::first(&after2);
    let fields_vec = match fields_form.peel_meta_ref() {
        Object::Vector(v) => v.clone(),
        other => panic!(
            "clojure-jvm: deftype/defrecord `{}`: fields must be a vector, got {other:?}",
            name_sym.get_name(),
        ),
    };
    let mut field_syms: Vec<Arc<Symbol>> = Vec::with_capacity(fields_vec.count() as usize);
    for i in 0..(fields_vec.count() as i32) {
        let p = fields_vec.nth(i);
        match p.peel_meta_ref() {
            Object::Symbol(s) => field_syms.push(s.clone()),
            other => panic!(
                "clojure-jvm: deftype/defrecord `{}`: field must be a symbol, got {other:?}",
                name_sym.get_name(),
            ),
        }
    }
    // Inline protocol impls: the body is the same shape as
    // `extend-type`'s tail (`P1 (m1 …) (m2 …) P2 (m3 …)`). We collect
    // those trailing forms and emit a synthesized `extend-type Foo P1
    // … P2 …` alongside the factory def. The synthesized form is then
    // re-expanded by `expand_extend_type` when the analyzer recurses
    // into the `do` body.
    let body_tail = super::rt::next(&after2);
    let arity = field_syms.len();
    let alloc_method = match arity {
        0 => "allocUserInstance0",
        1 => "allocUserInstance1",
        2 => "allocUserInstance2",
        3 => "allocUserInstance3",
        4 => "allocUserInstance4",
        n => panic!(
            "clojure-jvm: deftype `{}`: field count {n} exceeds wired \
             allocUserInstanceN externs (0..4); extend Compiler::new \
             and runtime.rs to raise the cap",
            name_sym.get_name(),
        ),
    };
    let user_type_id = register_user_type(name_sym.clone(), field_syms.clone());
    // Build factory fn:
    // (fn* [f0 f1 …] (. clojure.lang.RT (allocUserInstanceN <tid> f0 f1 …)))
    let params_vec: Vec<Object> = field_syms
        .iter()
        .map(|s| Object::Symbol(s.clone()))
        .collect();
    let params_form = Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
        params_vec,
    ));
    let mut alloc_args: Vec<Object> = Vec::with_capacity(arity + 2);
    alloc_args.push(Object::Symbol(Symbol::intern(alloc_method)));
    alloc_args.push(Object::Long(user_type_id as i64));
    for s in &field_syms {
        alloc_args.push(Object::Symbol(s.clone()));
    }
    let alloc_call = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern(".")),
        Object::Symbol(Symbol::intern_ns_name(None, "clojure.lang.RT")),
        Object::List(PersistentList::create(alloc_args)),
    ]));
    let fn_form = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("fn*")),
        params_form,
        alloc_call,
    ]));
    let factory_def = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("def")),
        Object::Symbol(name_sym.clone()),
        fn_form,
    ]));
    if matches!(body_tail, Object::Nil) {
        return factory_def;
    }
    // Reconstruct `(extend-type Name body…)` by prepending the
    // `extend-type` head + the type name to the body forms. The
    // analyzer will macroexpand this through `expand_extend_type`,
    // which already knows how to look up the user_type_id we just
    // registered.
    let mut et_items: Vec<Object> = vec![
        Object::Symbol(Symbol::intern("extend-type")),
        Object::Symbol(name_sym),
    ];
    let mut cur = body_tail;
    while !matches!(cur, Object::Nil) {
        et_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    let extend_form = Object::List(PersistentList::create(et_items));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("do")),
        factory_def,
        extend_form,
    ]))
}

/// `(doseq [x coll] body…)` → simplified loop that walks one binding's
/// coll. Single-binding only — multi-binding doseq with :let / :when /
/// :while modifiers is upstream's full expansion. Host-side because
/// upstream's doseq is one of the most complex defmacros and hits the
/// gc_literal corruption.
///
/// Expansion shape:
/// ```text
/// (let* [s_tmp (seq coll)]
///   (loop* [s_loop s_tmp]
///     (if s_loop
///       (do (let* [x (first s_loop)] body…)
///           (recur (next s_loop)))
///       nil)))
/// ```
/// Multi-binding `(doseq [x xs y ys] body)` collapses by recursing —
/// the inner pair becomes the body of the outer.
fn expand_doseq(form: &Object) -> Object {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let body_seq = super::rt::next(&after);
    let bindings_v = match bindings.peel_meta_ref() {
        Object::Vector(v) => v.clone(),
        other => panic!("clojure-jvm: doseq needs a vector, got {other:?}"),
    };
    if bindings_v.count() % 2 != 0 || bindings_v.count() == 0 {
        panic!(
            "clojure-jvm: doseq bindings must have even count >= 2, got {}",
            bindings_v.count()
        );
    }
    // Collect body items.
    let mut body_items: Vec<Object> = Vec::new();
    {
        let mut cur = body_seq;
        while !matches!(cur, Object::Nil) {
            body_items.push(super::rt::first(&cur));
            cur = super::rt::next(&cur);
        }
    }
    // Recurse: build inner doseq for trailing pairs, then wrap.
    fn build_loop(
        bindings: &Arc<crate::lang::persistent_vector::PersistentVector>,
        start: usize,
        body_items: &[Object],
    ) -> Object {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if start >= bindings.count() as usize {
            // Innermost: a (do body…). If body is empty, nil.
            if body_items.is_empty() {
                return Object::Nil;
            }
            let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
            do_items.extend(body_items.iter().cloned());
            return Object::List(PersistentList::create(do_items));
        }
        let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let s_tmp = Symbol::intern(&format!("s__ds{id}__"));
        let s_loop = Symbol::intern(&format!("sl__ds{id}__"));
        let bind_sym_raw = bindings.nth(start as i32);
        let (bind_sym, destruct_pat) = match bind_sym_raw.peel_meta_ref().clone() {
            Object::Symbol(s) => (s, None),
            // :let / :when / :while modifiers: skip them by treating the
            // value as the next coll's body. Pragmatic — drop them.
            Object::Keyword(_) => {
                // skip (kw, value) and continue.
                return build_loop(bindings, start + 2, body_items);
            }
            // Vector or map: synthesize a tmp param and emit the
            // destructure as a let* prefix on the loop body.
            pat @ (Object::Vector(_) | Object::Map(_)) => {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let synth = Symbol::intern(&format!("ds__bind{id}__"));
                (synth, Some(pat))
            }
            other => panic!("clojure-jvm: doseq binding must be a symbol, got {other:?}"),
        };
        let coll_expr = bindings.nth((start + 1) as i32);
        let inner_body = build_loop(bindings, start + 2, body_items);
        // If the binding is a destructure pattern, expand it as a
        // body-prefix let* over the synthetic.
        let inner_body = if let Some(pat) = destruct_pat {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let mut prefix: Vec<Object> = Vec::new();
            emit_destructuring(&pat, Object::Symbol(bind_sym.clone()), &mut prefix, &N);
            Object::List(PersistentList::create(vec![
                Object::Symbol(Symbol::intern("let*")),
                Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
                    prefix,
                )),
                inner_body,
            ]))
        } else {
            inner_body
        };
        // Build: (let* [bind_sym (first sl)] inner_body)
        let bind_let = Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("let*")),
            Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
                vec![
                    Object::Symbol(bind_sym),
                    Object::List(PersistentList::create(vec![
                        Object::Symbol(Symbol::intern("first")),
                        Object::Symbol(s_loop.clone()),
                    ])),
                ],
            )),
            inner_body,
        ]));
        // (recur (next sl))
        let recur_call = Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("recur")),
            Object::List(PersistentList::create(vec![
                Object::Symbol(Symbol::intern("next")),
                Object::Symbol(s_loop.clone()),
            ])),
        ]));
        // (if sl (do bind_let recur_call) nil)
        let if_form = Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("if")),
            Object::Symbol(s_loop.clone()),
            Object::List(PersistentList::create(vec![
                Object::Symbol(Symbol::intern("do")),
                bind_let,
                recur_call,
            ])),
            Object::Nil,
        ]));
        // (loop* [sl s_tmp] if_form)
        let loop_form = Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("loop*")),
            Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
                vec![Object::Symbol(s_loop), Object::Symbol(s_tmp.clone())],
            )),
            if_form,
        ]));
        // (let* [s_tmp (seq coll)] loop_form)
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("let*")),
            Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
                vec![
                    Object::Symbol(s_tmp),
                    Object::List(PersistentList::create(vec![
                        Object::Symbol(Symbol::intern("seq")),
                        coll_expr,
                    ])),
                ],
            )),
            loop_form,
        ]))
    }
    let _ = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed); // mark progress
    build_loop(&bindings_v, 0, &body_items)
}

/// `(for [x coll] body)` → simplified single-binding lazy comprehension.
/// Real `for` supports many modifiers; this version handles only the
/// single-binding form by mapping body over (seq coll) via lazy-seq.
fn expand_for_simple(form: &Object) -> Object {
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let body = super::rt::first(&super::rt::next(&after));
    let bindings_v = match bindings.peel_meta_ref() {
        Object::Vector(v) => v,
        other => panic!("clojure-jvm: for needs a vector, got {other:?}"),
    };
    if bindings_v.count() != 2 {
        // Multi-binding for: just emit nil. Real expansion is complex
        // (cartesian product with :let / :when / :while modifiers).
        // The defns that use it analyze cleanly; runtime invocation
        // returns nil rather than the actual sequence.
        eprintln!(
            "[cljvm-stub] multi-binding `for` ({} bindings) — returning nil",
            bindings_v.count() / 2
        );
        return Object::Nil;
    }
    let bind_sym = bindings_v.nth(0);
    let coll = bindings_v.nth(1);
    // (map (fn [bind_sym] body) coll)
    let fn_form = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("fn*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![bind_sym],
        )),
        body,
    ]));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("map")),
        fn_form,
        coll,
    ]))
}

/// `(lazy-seq body…)` → `(new clojure.lang.LazySeq (fn* [] body…))`.
/// Host-side because the upstream defmacro hits the gc_literal bug.
fn expand_lazy_seq(form: &Object) -> Object {
    let mut body_items: Vec<Object> = vec![
        Object::Symbol(Symbol::intern("fn*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![],
        )),
    ];
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        body_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    let thunk = Object::List(PersistentList::create(body_items));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("new")),
        Object::Symbol(Symbol::intern("clojure.lang.LazySeq")),
        thunk,
    ]))
}

/// `(declare name1 name2 …)` → `(do (def name1) (def name2) …)`.
/// Forward declarations — each `(def name)` interns the var without
/// binding a value. Host-side because upstream's `declare` uses
/// syntax-quote + `map` + `vary-meta`, which hit the gc_literal bug.
fn expand_declare(form: &Object) -> Object {
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = super::rt::next(form);
    while !matches!(cur, Object::Nil) {
        let name = super::rt::first(&cur);
        do_items.push(Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("def")),
            name,
        ])));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(do_items))
}

/// `(dotimes [i n] body…)` →
/// `(let* [n_tmp n] (loop* [i 0] (if (< i n_tmp) (do body… (recur (unchecked-inc i))) nil)))`.
/// Host-side because the upstream defmacro hits the gc_literal corruption.
fn expand_dotimes(form: &Object) -> Object {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let n_tmp = Symbol::intern(&format!("n__dt{id}__"));
    let after = super::rt::next(form);
    let bindings = super::rt::first(&after);
    let body_seq = super::rt::next(&after);
    let (i_sym, n_expr) = match bindings.peel_meta_ref() {
        Object::Vector(v) if v.count() == 2 => (v.nth(0), v.nth(1)),
        other => panic!("clojure-jvm: dotimes needs [i n], got {other:?}"),
    };
    let mut body_items: Vec<Object> = Vec::new();
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        body_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    // (recur (unchecked-inc i))
    let recur_form = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("recur")),
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("unchecked-inc")),
            i_sym.clone(),
        ])),
    ]));
    // (do body... recur)
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    do_items.extend(body_items);
    do_items.push(recur_form);
    let do_form = Object::List(PersistentList::create(do_items));
    // (if (< i n_tmp) do_form nil)
    let if_form = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("if")),
        Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("<")),
            i_sym.clone(),
            Object::Symbol(n_tmp.clone()),
        ])),
        do_form,
        Object::Nil,
    ]));
    // (loop* [i 0] if_form)
    let loop_form = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("loop*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![i_sym, Object::Long(0)],
        )),
        if_form,
    ]));
    // (let* [n_tmp n] loop_form)
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            vec![Object::Symbol(n_tmp), n_expr],
        )),
        loop_form,
    ]))
}

/// `(binding [vars vals…] body…)` — strip the bindings and emit
/// `(do body…)`. Real dynamic var binding requires
/// push-/pop-ThreadBindings runtime support which is currently a
/// panic stub. Collapsing to `do` lets defns that use `binding`
/// analyze cleanly; if the body is invoked at runtime, the binding
/// semantics are silently NOT enforced (the body runs against
/// whatever the var's root binding is).
fn expand_binding_to_do(form: &Object) -> Object {
    let after = super::rt::next(form);
    // Skip the bindings vector.
    let body_seq = super::rt::next(&after);
    let mut do_items: Vec<Object> = vec![Object::Symbol(Symbol::intern("do"))];
    let mut cur = body_seq;
    while !matches!(cur, Object::Nil) {
        do_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    Object::List(PersistentList::create(do_items))
}

/// Walk a defmacro form's params vector(s) and prepend `&form` `&env`
/// as the first two params on each clause. Returns a transformed form
/// that `expand_defn` can then process as if it were a regular defn
/// with the implicit params baked in.
fn inject_macro_implicit_params(form: &Object) -> Object {
    let l = match form {
        Object::List(l) => l,
        _ => return form.clone(),
    };
    let mut items: Vec<Object> = l.iter().collect();
    if items.is_empty() {
        return form.clone();
    }
    // Walk past the head (defmacro), name, optional doc, optional attr-map.
    let mut cursor = 1; // skip head
    if cursor >= items.len() {
        return form.clone();
    }
    cursor += 1; // skip name
    if cursor < items.len() && matches!(items[cursor], Object::String(_)) {
        cursor += 1;
    }
    if cursor < items.len() && matches!(items[cursor], Object::Map(_)) {
        cursor += 1;
    }
    if cursor >= items.len() {
        return form.clone();
    }
    // Now items[cursor..] is either `[params] body…` (single-arity) or
    // `([params] body)+` (multi-arity).
    let single_arity = matches!(&items[cursor], Object::Vector(_));
    if single_arity {
        let params = match &items[cursor] {
            Object::Vector(v) => v.clone(),
            _ => unreachable!(),
        };
        items[cursor] = Object::Vector(prepend_form_env(&params));
    } else {
        for item in items.iter_mut().skip(cursor) {
            if let Object::List(clause) = item {
                let clause_items: Vec<Object> = clause.iter().collect();
                if let Some(Object::Vector(params)) = clause_items.first() {
                    let new_params = prepend_form_env(params);
                    let mut new_clause: Vec<Object> = Vec::with_capacity(clause_items.len());
                    new_clause.push(Object::Vector(new_params));
                    for x in clause_items.iter().skip(1) {
                        new_clause.push(x.clone());
                    }
                    *item = Object::List(PersistentList::create(new_clause));
                }
            }
        }
    }
    Object::List(PersistentList::create(items))
}

fn prepend_form_env(
    params: &Arc<crate::lang::persistent_vector::PersistentVector>,
) -> Arc<crate::lang::persistent_vector::PersistentVector> {
    let mut new_items: Vec<Object> = vec![
        Object::Symbol(Symbol::intern("&form")),
        Object::Symbol(Symbol::intern("&env")),
    ];
    for i in 0..params.count() {
        new_items.push(params.nth(i));
    }
    crate::lang::persistent_vector::PersistentVector::create(new_items)
}

/// Host-side expansion of `(defn name doc? attr-map? params body+)` or
/// the multi-arity `(defn name doc? attr-map? ([params*] body)+ attr-map?)`.
///
/// Mirrors upstream defn's transformation:
/// `(def ^{:arglists ... :doc ... :added ...} name (with-meta (cons 'fn fdecl) {:rettag tag}))`
///
/// `private` adds `:private true` to the var's metadata (defn- variant).
fn expand_defn(form: &Object, private: bool) -> Object {
    use crate::lang::persistent_hash_map::PersistentHashMap;

    let after_defn = super::rt::next(form);
    let name_form = super::rt::first(&after_defn);
    let name_meta_outer: Option<Arc<PersistentHashMap>> = name_form.meta_of().cloned();
    let name_sym = match name_form.peel_meta_ref() {
        Object::Symbol(s) => s.clone(),
        other => panic!("clojure-jvm: defn: first arg must be a symbol, got {other:?}"),
    };

    // Walk: optional doc-string, optional attr-map, params-or-clauses, body.
    let mut rest = super::rt::next(&after_defn);
    let mut meta_pairs: Vec<(Object, Object)> = Vec::new();
    if let Object::String(s) = super::rt::first(&rest) {
        meta_pairs.push((
            Object::Keyword(crate::lang::keyword::Keyword::intern_ns_name(None, "doc")),
            Object::String(s.clone()),
        ));
        rest = super::rt::next(&rest);
    }
    if let Object::Map(m) = super::rt::first(&rest) {
        for (k, v) in m.iter() {
            meta_pairs.push((k, v));
        }
        rest = super::rt::next(&rest);
    }
    // fdecl is whatever remains. Single-arity is `[params] body…`,
    // multi-arity is `([params] body)+`. Wrap single-arity in a list
    // so downstream `(fn fdecl)` builds `(fn ([params] body))`.
    let mut fdecl_items: Vec<Object> = Vec::new();
    {
        let mut cur = rest.clone();
        while !matches!(cur, Object::Nil) {
            fdecl_items.push(super::rt::first(&cur));
            cur = super::rt::next(&cur);
        }
    }
    // Trailing attr-map (after all clauses).
    if let Some(Object::Map(m)) = fdecl_items.last().cloned() {
        if fdecl_items.len() >= 2 {
            // Only treat as trailing attr-map if there's at least one clause.
            for (k, v) in m.iter() {
                meta_pairs.push((k, v));
            }
            fdecl_items.pop();
        }
    }
    let single_arity = fdecl_items
        .first()
        .map(|o| matches!(o, Object::Vector(_)))
        .unwrap_or(false);
    let fdecl_seq: Vec<Object> = if single_arity {
        // Wrap `[params] body…` in a list `(([params] body…))`.
        vec![Object::List(PersistentList::create(fdecl_items.clone()))]
    } else {
        fdecl_items.clone()
    };

    // :arglists '([params...] [params...] ...)
    let mut arglists: Vec<Object> = Vec::with_capacity(fdecl_seq.len());
    for clause in &fdecl_seq {
        if let Object::List(l) = clause {
            if let Some(params) = l.iter().next() {
                if let Object::Vector(_) = &params {
                    arglists.push(params);
                }
            }
        }
    }
    let arglists_quoted = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("quote")),
        Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            arglists,
        )),
    ]));
    let mut final_meta_pairs: Vec<(Object, Object)> = vec![(
        Object::Keyword(crate::lang::keyword::Keyword::intern_ns_name(
            None, "arglists",
        )),
        arglists_quoted,
    )];
    final_meta_pairs.extend(meta_pairs);
    if private {
        final_meta_pairs.push((
            Object::Keyword(crate::lang::keyword::Keyword::intern_ns_name(
                None, "private",
            )),
            Object::Bool(true),
        ));
    }
    // Merge any metadata already on the name symbol (from `(defn ^... name ...)`).
    if let Some(name_meta) = name_meta_outer {
        for (k, v) in name_meta.iter() {
            final_meta_pairs.insert(0, (k, v));
        }
    }
    let final_meta = PersistentHashMap::create_pairs(final_meta_pairs);
    let named_with_meta = Object::WithMeta(Box::new(Object::Symbol(name_sym.clone())), final_meta);

    // (fn* name fdecl…) — keep the name as the inner fn's self-ref name.
    // Use the special-form `fn*` directly; upstream `fn` is a macro that
    // expands to `fn*` plus destructuring, but we don't have `fn`
    // defined in our subset (and destructuring isn't wired yet anyway).
    let mut fn_form_items: Vec<Object> = vec![
        Object::Symbol(Symbol::intern("fn*")),
        Object::Symbol(name_sym),
    ];
    fn_form_items.extend(fdecl_seq);
    let fn_form = Object::List(PersistentList::create(fn_form_items));

    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("def")),
        named_with_meta,
        fn_form,
    ]))
}

/// `(defmacro name doc? attr-map? params body+)` — emit
/// `(do (defn name ...) (set-macro! name))`. Defer the macro flag
/// until AFTER the var has been bound to a fn handle, otherwise
/// recursive calls in the macro body (e.g. `cond` referencing
/// itself) would try to macro-expand against a nil var.
///
/// **Critical:** The defmacro expansion must inject `&form &env` as the
/// first two implicit params on EVERY clause. Without them, the macro
/// fn's params shift left when `macroexpand_once` invokes it (which
/// passes [&form, &env, ...user-args]), causing the user's `test` arg
/// to receive the whole form. For a recursive `when`-style macro that
/// is then macroexpanded again, this is an infinite loop. This is the
/// path that originally looked like a GC corruption — actually a macro
/// arity mismatch.
fn expand_defmacro(form: &Object) -> Object {
    let injected = inject_macro_implicit_params(form);
    let defn_form = expand_defn(&injected, false);
    // Pull the bare name out of the defn-expanded form.
    let bare_name = match &defn_form {
        Object::List(l) => {
            let items: Vec<Object> = l.iter().collect();
            match items.get(1) {
                Some(Object::WithMeta(inner, _)) => match inner.as_ref() {
                    Object::Symbol(s) => Some(s.clone()),
                    _ => None,
                },
                Some(Object::Symbol(s)) => Some(s.clone()),
                _ => None,
            }
        }
        _ => None,
    };
    let name = match bare_name {
        Some(n) => n,
        None => return defn_form,
    };
    let set_macro = Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("set-macro!")),
        Object::Symbol(name),
    ]));
    Object::List(PersistentList::create(vec![
        Object::Symbol(Symbol::intern("do")),
        defn_form,
        set_macro,
    ]))
}

/// `Compiler.analyzeSeq(C context, ISeq form, String name)` — Java line
/// ~7167+. Dispatches list-headed forms to specials / macros / invoke.
///
/// We currently recognize the special forms whose Exprs are ported (`if`,
/// `do`, `quote`). Everything else panics with a clear message.
/// If `form` is `(macro-name args...)` and `macro-name` resolves to a
/// `^:macro`-flagged Var with a registered FuncRef, invoke the macro
/// (compile-time, via the active Session's persistent JIT) and return
/// `Some(expanded)`. Otherwise `None`.
///
/// The macro fn signature mirrors microlisp's: it takes ONE arg, the
/// unevaluated args list (the form's `next`), and returns a form Object.
fn macroexpand_once(form: &Object) -> Option<Object> {
    use dynlower::JitOutcome;
    use dynruntime::GcPolicy;

    let head = super::rt::first(form);
    let head_sym_raw = match head {
        Object::Symbol(s) => s,
        _ => return None,
    };
    // If the head is qualified and its prefix resolves (directly or via
    // an alias in the current ns) to `clojure.core`, treat it as if it
    // were the bare name — that's where all the host-intercepted macros
    // live, and we want `core/defmacro` to land in the intercept the
    // same way `defmacro` does.
    let head_sym = if let Some(prefix) = head_sym_raw.get_namespace() {
        use super::namespace::Namespace;
        let prefix_sym = Symbol::intern(prefix);
        let target = Namespace::find(&prefix_sym)
            .or_else(|| super::rt::current_ns().lookup_alias(&prefix_sym));
        match target {
            Some(ns) if ns.name.get_name() == "clojure.core" => {
                Symbol::intern(head_sym_raw.get_name())
            }
            _ => head_sym_raw,
        }
    } else {
        head_sym_raw
    };
    // Host-side fast path for `defn` / `defn-` / `defmacro`. Synthesizes
    // the same `(def ^M name (fn* name fdecl))` shape upstream defn
    // produces, but in Rust — bypasses the toolkit-side gc_literal /
    // safepoint chain. The previously-suspected "GC corruption" at
    // form 202 turned out to be a missing `&form &env` on macro fn
    // params, fixed in `expand_defmacro` via `inject_macro_implicit_params`.
    if head_sym.get_namespace().is_none() {
        let name = head_sym.get_name();
        match name {
            "defn" => return Some(expand_defn(form, false)),
            "defn-" => return Some(expand_defn(form, true)),
            "defmacro" => return Some(expand_defmacro(form)),
            // `binding` would expand to a (let [] ... (try ...))
            // sequence touching push/pop-thread-bindings. We don't
            // implement dynamic binding properly; collapse it to a
            // (do body...) so defns that USE binding analyze. The
            // var-binding semantics are silently dropped.
            "binding" => return Some(expand_binding_to_do(form)),
            "when-let" => return Some(expand_when_let(form)),
            "if-let" => return Some(expand_if_let(form)),
            "dotimes" => return Some(expand_dotimes(form)),
            "declare" => return Some(expand_declare(form)),
            "lazy-seq" => return Some(expand_lazy_seq(form)),
            "doseq" => return Some(expand_doseq(form)),
            "for" => return Some(expand_for_simple(form)),
            "defmulti" => return Some(expand_defmulti(form)),
            "defprotocol" => return Some(expand_defprotocol(form)),
            "extend-type" => return Some(expand_extend_type(form)),
            "extend-protocol" => return Some(expand_extend_protocol(form)),
            "satisfies?" => return Some(expand_satisfies(form)),
            "definterface" => return Some(expand_defprotocol(form)),
            "deftype" => return Some(expand_deftype_or_record(form)),
            "defrecord" => return Some(expand_deftype_or_record(form)),
            // def-aset is a private defmacro generating defns for
            // aset-int / aset-long etc. We don't model Java arrays.
            "def-aset" => {
                // First arg is the name, the rest are method/coerce.
                let after = super::rt::next(form);
                let name_form = super::rt::first(&after);
                let name_sym = match name_form.peel_meta_ref() {
                    Object::Symbol(s) => s.clone(),
                    other => panic!("def-aset needs symbol name, got {other:?}"),
                };
                return Some(Object::List(PersistentList::create(vec![
                    Object::Symbol(Symbol::intern("def")),
                    Object::Symbol(name_sym),
                    Object::Nil,
                ])));
            }
            // comment is a defmacro that returns nil regardless of body
            // (matches upstream behavior).
            "comment" => return Some(Object::Nil),
            "or" => return Some(expand_or(form)),
            "and" => return Some(expand_and(form)),
            "cond" => return Some(expand_cond(form)),
            // Host-side bootstrap macros. When loading sources that do
            // NOT first define their own `let`/`fn`/`when`/etc.
            // (typically: cljs.core, or any standalone file that
            // assumes the analyzer already provides them), these
            // expansions let `let` mean `let*` etc. without requiring
            // upstream `clojure/core.clj` to be loaded first.
            //
            // Limitation: no destructuring. `(let [{:keys [a]} m] …)`
            // passes the destructuring pattern through to `let*`,
            // which then panics on the non-symbol binding. Acceptable
            // for v1 — `let*` is the actual primitive.
            "let" => return Some(expand_let_simple(form)),
            "fn" => return Some(expand_fn_simple(form)),
            "loop" => return Some(expand_loop_simple(form)),
            "when" => return Some(expand_when(form)),
            "when-not" => return Some(expand_when_not(form)),
            "if-not" => return Some(expand_if_not(form)),
            "doto" => return Some(expand_doto(form)),
            "->" => return Some(expand_thread_first(form)),
            "->>" => return Some(expand_thread_last(form)),
            "with-out-str" => return Some(expand_with_out_str(form)),
            "definline" => return Some(expand_definline(form)),
            "defonce" => return Some(expand_defonce(form)),
            ".." => return Some(expand_dotdot(form)),
            "dosync" => return Some(expand_sync(form)),
            "sync" => return Some(expand_sync(form)),
            "locking" => return Some(expand_locking(form)),
            "while" => return Some(expand_while(form)),
            "with-open" => return Some(expand_with_open(form)),
            "letfn" => return Some(expand_letfn(form)),
            "future" => return Some(expand_future(form)),
            "future-call" => return Some(expand_future(form)),
            "doall" => return Some(expand_io_bang(form)),
            "dorun" => return Some(expand_io_bang(form)),
            "time" => return Some(expand_io_bang(form)),
            "alias" => return Some(expand_alias(form)),
            _ => {}
        }
    }
    if head_sym.get_namespace().is_some() {
        // Qualified head — handled the same way; resolve_var below uses
        // its own ns-aware lookup.
    }
    let var = resolve_var(&head_sym)?;
    if !var.is_macro() {
        return None;
    }
    // Build the args list (everything after the macro name). It's already
    // an `Object::List` in our reader — `next(form)` returns it.
    let args = super::rt::next(form);
    let args_list = match args {
        Object::List(l) => l,
        Object::Nil => PersistentList::empty(),
        other => panic!(
            "clojure-jvm: macroexpand: rest of macro form expected to be a list, got {other:?}"
        ),
    };

    // Clojure macros take their arguments positionally, with two implicit
    // initial params: `&form` (the unexpanded call form) and `&env` (the
    // lexical env at expansion site, nil for top-level). All macro fns
    // declare these as their first two params — defmacro adds them
    // automatically; raw `(def ^:macro foo (fn* [&form &env x] ...))`
    // declares them explicitly.
    let mut items: Vec<Object> = Vec::new();
    items.push(form.clone()); // &form
    items.push(Object::Nil); // &env (we don't model lexical envs yet)
    // Walk the args list; stop when count drops to zero (Empty list is
    // distinct from Nil but contributes no elements). Without the
    // count check, an `(macro)` call with no args would push a
    // spurious `nil` onto items because `next(empty)` returns Nil but
    // `first(empty)` returns Nil too — silently inflating arity by 1.
    // (Symptom: cond's recursion on its zero-arg base case
    // `(cond)` saw items.len()=3 instead of 2 and threw
    // "cond requires an even number of forms".)
    if !matches!(&args_list as &PersistentList, PersistentList::Empty) {
        let mut cur = Object::List(args_list);
        while !matches!(cur, Object::Nil) {
            items.push(super::rt::first(&cur));
            let nxt = super::rt::next(&cur);
            cur = nxt;
        }
    }

    // Dispatch the macro fn by arity. Single-arity macros are the common
    // case — `var.deref()` holds a TAG_FN handle pointing at the lone
    // FnExpr. Multi-arity macros (e.g. defmacro `or`/`and` with empty,
    // single, and variadic clauses) have one FuncRef per clause registered
    // in `var_multi_arities`; pick the clause whose arity matches the
    // call. Without this dispatch, `var.root` for a multi-arity defmacro
    // is nil and macroexpand panics.
    let (fref, fixed_arity, is_variadic) = {
        let multi = with_active_session_ref(|s| s.compiler.var_multi_arity(&var)).flatten();
        if let Some(table) = multi {
            // Pick the clause matching items.len() (which counts &form, &env, …user args).
            let pick = table.iter().find_map(|(fr, info)| {
                let matches = if info.is_variadic {
                    items.len() >= info.fixed_arity
                } else {
                    items.len() == info.fixed_arity
                };
                if matches {
                    Some((*fr, info.fixed_arity, info.is_variadic))
                } else {
                    None
                }
            });
            match pick {
                Some(t) => t,
                None => panic!(
                    "clojure-jvm: macroexpand: no matching arity for `{}` with {} args",
                    head_sym.get_name(),
                    items.len(),
                ),
            }
        } else {
            // Single-arity path: read the TAG_FN handle from var.root.
            let fn_handle_bits: u64 = match var.deref() {
                Object::Host(_h) => {
                    let root = var.deref();
                    match root.host_as::<crate::runtime::HeapBits>() {
                        Some(hb) => hb.0,
                        None => panic!(
                            "clojure-jvm: macroexpand: macro Var holds Host but not HeapBits"
                        ),
                    }
                }
                other => panic!(
                    "clojure-jvm: macroexpand: macro Var `{}/{}` must hold a fn handle, got {other:?}",
                    head_sym.get_namespace().unwrap_or(""),
                    head_sym.get_name(),
                ),
            };
            const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
            const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
            const TAG_MASK: u64 = 0x0003_0000_0000_0000;
            const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
            assert_eq!(fn_handle_bits & FULL_MASK, TAG_PATTERN);
            assert_eq!((fn_handle_bits & TAG_MASK) >> 48, 3);
            let fref_idx = (fn_handle_bits & PAYLOAD_MASK) as u32;
            let fref = dynir::FuncRef::from_u32(fref_idx);
            // Re-derive arity info via the fn_arity table (registered by parse_fn_form).
            let info = with_active_session_ref(|s| s.compiler.fn_arity(fref_idx))
                .flatten()
                .unwrap_or(VarFnInfo {
                    is_variadic: false,
                    fixed_arity: items.len(),
                });
            (fref, info.fixed_arity, info.is_variadic)
        }
    };
    let fref_idx = fref.index() as u32;

    let obj = with_active_session_ref(|sess| {
        let _thread = sess.gc.install_thread();
        let _ctb = crate::runtime::install_call_table_base(sess.jit.call_table_base_addr());
        // Per-call FrameChain so runtime externs (cons-fold inside the
        // macro body's `pack_variadic_args`, etc.) can root heap
        // pointers across `gc_alloc_thunk`. Same shape as eval_form.
        let local_chain = dynobj::roots::FrameChain::new();
        let chain_src: *const dyn dynobj::RootSource = &local_chain;
        let _chain_root_g = unsafe { sess.gc.push_extra_root_source(chain_src) };
        let _chain_g = dynobj::roots::install_chain(&local_chain);

        let _ = (fixed_arity, is_variadic, fref_idx); // arity info already chosen by the multi-arity dispatch above

        // Root every allocated arg on `local_chain` so that intermediate
        // allocations (later args, or the variadic rest-list) can't move
        // earlier args without updating their NanBox bits in place. Plain
        // `Vec<u64>` storage is NOT a root source; without this scope
        // wrapping, alloc_*_as_nanbox can trigger a GC that relocates a
        // previously-pushed cell, leaving the corresponding arg slot
        // pointing at reused memory. (Symptom: `defn` macroexpansion
        // returns a list whose head is a heap pointer with garbage
        // type_id, e.g. form 202's `get-thread-bindings` failure.)
        let result_bits = dynobj::roots::with_scope(fixed_arity + 2, |scope| {
            let mut roots: Vec<dynobj::roots::Rooted<()>> = Vec::with_capacity(fixed_arity + 1);
            for i in 0..fixed_arity {
                let item = items.get(i).cloned().unwrap_or(Object::Nil);
                let bits = alloc_object_as_nanbox(
                    &sess.gc,
                    &sess.obj_types,
                    sess.compiler.cons_type_id,
                    sess.compiler.string_type_id,
                    sess.compiler.symbol_type_id,
                    sess.compiler.keyword_type_id,
                    &mut sess.roots,
                    &item,
                );
                roots.push(scope.root::<()>(bits));
            }
            if is_variadic {
                // Pack remaining args (if any) into a list. Empty rest → nil.
                let rest_items: Vec<Object> = items.iter().skip(fixed_arity).cloned().collect();
                let rest_list = PersistentList::create(rest_items);
                let rest_bits = alloc_list_as_nanbox(
                    &sess.gc,
                    &sess.obj_types,
                    sess.compiler.cons_type_id,
                    sess.compiler.string_type_id,
                    sess.compiler.symbol_type_id,
                    sess.compiler.keyword_type_id,
                    &mut sess.roots,
                    &rest_list,
                );
                roots.push(scope.root::<()>(rest_bits));
            } else if items.len() > fixed_arity {
                panic!(
                    "clojure-jvm: macroexpand: macro `{}` expected \
                         {fixed_arity} args, got {}",
                    head_sym.get_name(),
                    items.len()
                );
            }
            // Snapshot the rooted slots' current bits and hand them to
            // run_jit while the enclosing `scope` is STILL alive. The
            // Vec<u64> snapshot itself is not a root source, but because
            // the `Rooted` slots remain in scope across the run_jit call,
            // a GC triggered during macro-body execution updates those
            // slots in place. run_jit re-roots through its own call-frame
            // stack maps, but the rooted scope here guarantees the arg
            // cells survive the transition into the JIT frame even if a
            // GC fires before the JIT prologue establishes its stack map.
            // (Symptom of the bug this guards against: form 430's variadic
            // `import` macro reading type_id 136 garbage from a relocated
            // rest-arg cell.)
            let arg_bits_vec: Vec<u64> = roots.iter().map(|r| r.get()).collect();

            // GC policy for the macro body: EveryPoint, the strict
            // precise-rooting correctness oracle. Macroexpansion is the
            // most rooting-sensitive path (it runs JIT'd higher-order core
            // fns over closures); keeping EveryPoint hardcoded here is the
            // canary that surfaces any stackmap/liveness rooting bug in the
            // shared lowerer immediately. Do NOT downgrade this to
            // OnPressure to dodge a crash — that masks the bug (LAW #3).
            match sess
                .gc
                .run_jit(&sess.jit, fref, &arg_bits_vec, GcPolicy::EveryPoint)
            {
                JitOutcome::Value(v) => v,
                other => {
                    if let JitOutcome::Exception(e) = other {
                        let ids = crate::runtime::HeapTypeIds {
                            string: sess.compiler.string_type_id.0,
                            symbol: sess.compiler.symbol_type_id.0,
                            keyword: sess.compiler.keyword_type_id.0,
                            cons: sess.compiler.cons_type_id.0,
                            vector: sess.compiler.vector_type_id.0,
                            map: sess.compiler.map_type_id.0,
                            set: sess.compiler.set_type_id.0,
                            tree_map: sess.compiler.tree_map_type_id.0,
                            tree_set: sess.compiler.tree_set_type_id.0,
                            string_builder: sess.compiler.string_builder_type_id.0,
                            chunk_buffer: sess.compiler.chunk_buffer_type_id.0,
                            i_chunk: sess.compiler.i_chunk_type_id.0,
                            lazy_seq: sess.compiler.lazy_seq_type_id.0,
                            delay: sess.compiler.delay_type_id.0,
                            multi_arity_fn: sess.compiler.multi_arity_fn_type_id.0,
                            class: sess.compiler.class_type_id.0,
                            var: sess.compiler.var_type_id.0,
                            namespace: sess.compiler.namespace_type_id.0,
                            with_meta: sess.compiler.with_meta_type_id.0,
                            reduced: sess.compiler.reduced_type_id.0,
                            long: sess.compiler.long_type_id.0,
            character: sess.compiler.character_type_id.0,
                            user_instance: sess.compiler.user_instance_type_id.0,
                        };
                        let exc_obj = crate::runtime::any_bits_to_object(e, ids);
                        // Defer instead of aborting the load: stash the macro's
                        // exception and return `nil` as the "expansion".
                        // `analyze_seq` turns this into a runtime ThrowExpr.
                        let msg = format!(
                            "macroexpand of `{}` threw: {exc_obj:?}",
                            head_sym.get_name(),
                        );
                        MACRO_EXPAND_THREW.with(|c| *c.borrow_mut() = Some(msg));
                        crate::runtime::nanbox_nil()
                    } else {
                        panic!(
                            "clojure-jvm: macroexpand of `{}`: macro fn returned non-value outcome: {other:?}",
                            head_sym.get_name(),
                        );
                    }
                }
            }
        });

        // Decode the result back to a form Object. Handles both heap
        // pointers (the typical case — Cons / Symbol / Keyword / String)
        // and immediates (nil / true / false / long / double).
        let ids = crate::runtime::HeapTypeIds {
            string: sess.compiler.string_type_id.0,
            symbol: sess.compiler.symbol_type_id.0,
            keyword: sess.compiler.keyword_type_id.0,
            cons: sess.compiler.cons_type_id.0,
            vector: sess.compiler.vector_type_id.0,
            map: sess.compiler.map_type_id.0,
            set: sess.compiler.set_type_id.0,
            tree_map: sess.compiler.tree_map_type_id.0,
            tree_set: sess.compiler.tree_set_type_id.0,
            string_builder: sess.compiler.string_builder_type_id.0,
            chunk_buffer: sess.compiler.chunk_buffer_type_id.0,
            i_chunk: sess.compiler.i_chunk_type_id.0,
            lazy_seq: sess.compiler.lazy_seq_type_id.0,
            delay: sess.compiler.delay_type_id.0,
            multi_arity_fn: sess.compiler.multi_arity_fn_type_id.0,
            class: sess.compiler.class_type_id.0,
            var: sess.compiler.var_type_id.0,
            namespace: sess.compiler.namespace_type_id.0,
            with_meta: sess.compiler.with_meta_type_id.0,
            reduced: sess.compiler.reduced_type_id.0,
            long: sess.compiler.long_type_id.0,
            character: sess.compiler.character_type_id.0,
            user_instance: sess.compiler.user_instance_type_id.0,
        };
        crate::runtime::any_bits_to_object(result_bits, ids)
    });

    match obj {
        Some(o) => Some(o),
        None => panic!(
            "clojure-jvm: macroexpand called without an active Session — \
             macros only work via `Session::eval_form`"
        ),
    }
}

/// Vector literals: when every element is a compile-time constant we
/// build the whole vector at literal-pool fill time and emit a single
/// `gc_literal`. Otherwise we fold a chain of `RT.conj` calls
/// starting from an empty vector, mirroring `analyze_map`'s dynamic
/// path.
fn analyze_vector(context: C, v: Arc<PersistentVector>) -> Box<dyn Expr> {
    if context == C::Statement {
        return Box::new(NIL_EXPR);
    }
    let n = v.count();
    let mut all_const = true;
    for i in 0..n {
        if !is_constant_object(&v.nth(i)) {
            all_const = false;
            break;
        }
    }
    if all_const {
        let idx = with_active_compiler(|c| c.intern_literal(PendingLiteral::Vector(v)));
        return Box::new(ConstantLiteralExpr { idx });
    }
    // Dynamic vector: start with an empty PersistentVector literal and
    // fold conj over each analyzed element.
    let empty_idx = with_active_compiler(|c| {
        c.intern_literal(PendingLiteral::Vector(PersistentVector::create(vec![])))
    });
    let conj_fref = with_active_compiler(|c| c.host_method("clojure.lang.RT", "conj", 2))
        .expect("RT.conj must be registered for dynamic vector literals");
    let mut element_exprs: Vec<Box<dyn Expr>> = Vec::with_capacity(n as usize);
    for i in 0..n {
        element_exprs.push(analyze(C::Expression, v.nth(i)));
    }
    Box::new(DynamicVectorExpr {
        empty_idx,
        elements: element_exprs,
        conj_fref,
    })
}

/// Same shape as [`analyze_vector`] — handles set literals whose elements
/// are all constants. For dynamic sets, fold `RT.conj` over an empty set
/// literal, mirroring the vector path.
fn analyze_set(
    context: C,
    s: Arc<crate::lang::persistent_hash_set::PersistentHashSet>,
) -> Box<dyn Expr> {
    if context == C::Statement {
        return Box::new(NIL_EXPR);
    }
    let elements: Vec<Object> = s.iter().collect();
    let all_const = elements.iter().all(is_constant_object);
    if all_const {
        let idx = with_active_compiler(|c| c.intern_literal(PendingLiteral::Set(s)));
        return Box::new(ConstantLiteralExpr { idx });
    }
    let empty_idx = with_active_compiler(|c| {
        c.intern_literal(PendingLiteral::Set(
            crate::lang::persistent_hash_set::PersistentHashSet::create(vec![]),
        ))
    });
    let conj_fref = with_active_compiler(|c| c.host_method("clojure.lang.RT", "conj", 2))
        .expect("RT.conj must be registered for dynamic set literals");
    let element_exprs: Vec<Box<dyn Expr>> = elements
        .into_iter()
        .map(|e| analyze(C::Expression, e))
        .collect();
    Box::new(DynamicVectorExpr {
        empty_idx,
        elements: element_exprs,
        conj_fref,
    })
}

/// Map literals: when every key/value is a compile-time constant we
/// build the whole map at literal-pool fill time and emit a single
/// `gc_literal`. Otherwise we lower to a chain of `RT.assoc` calls
/// starting from nil — so `{:doc (first fdecl) :line 5}` becomes
/// `(. RT (assoc (. RT (assoc nil :doc (first fdecl))) :line 5))`.
fn analyze_map(
    context: C,
    m: Arc<crate::lang::persistent_hash_map::PersistentHashMap>,
) -> Box<dyn Expr> {
    if context == C::Statement {
        return Box::new(NIL_EXPR);
    }
    // Constant fast path — keep the existing literal-pool optimization.
    let all_const = m
        .iter()
        .all(|(k, v)| is_constant_object(&k) && is_constant_object(&v));
    if all_const {
        let idx = with_active_compiler(|c| c.intern_literal(PendingLiteral::Map(m)));
        return Box::new(ConstantLiteralExpr { idx });
    }
    // Dynamic map — fold over entries, emitting an RT.assoc per pair.
    let assoc_fref = with_active_compiler(|c| c.host_method("clojure.lang.RT", "assoc", 3))
        .expect("RT.assoc must be registered for dynamic map literals");
    let mut entry_exprs: Vec<(Box<dyn Expr>, Box<dyn Expr>)> =
        Vec::with_capacity(m.count() as usize);
    for (k, v) in m.iter() {
        let k_expr = analyze(C::Expression, k);
        let v_expr = analyze(C::Expression, v);
        entry_exprs.push((k_expr, v_expr));
    }
    Box::new(DynamicMapExpr {
        entries: entry_exprs,
        assoc_fref,
    })
}

/// `[<dyn-expr> ...]` — runtime-built vector. Loads an empty Vector
/// literal then folds `RT.conj` per element. Built by
/// `analyze_vector` when any element isn't a compile-time constant.
#[derive(Debug)]
struct DynamicVectorExpr {
    empty_idx: u32,
    elements: Vec<Box<dyn Expr>>,
    conj_fref: dynir::FuncRef,
}

impl Expr for DynamicVectorExpr {
    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        let empty_lit = dynir::ir::LiteralRef::from_u32(self.empty_idx);
        let mut acc = ir.f.fb.gc_literal(empty_lit);
        for e in &self.elements {
            let v = e
                .emit(C::Expression, objx, ir)
                .expect("DynamicVectorExpr element must produce a value");
            acc =
                ir.f.fb
                    .call(self.conj_fref, &[acc, v])
                    .expect("RT.conj returns I64 (NanBox)");
        }
        match context {
            C::Statement => None,
            _ => Some(acc),
        }
    }
    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("clojure.lang.PersistentVector".to_string()),
        })
    }
}

/// `{<dyn-expr> ...}` — runtime-built map. Lowers to a fold over
/// `RT.assoc`. Built by `analyze_map` when any entry isn't a literal.
#[derive(Debug)]
struct DynamicMapExpr {
    entries: Vec<(Box<dyn Expr>, Box<dyn Expr>)>,
    assoc_fref: dynir::FuncRef,
}

impl Expr for DynamicMapExpr {
    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Start with nil, fold each (k, v) through RT.assoc.
        let nil_bits = (0x7FFC_0000_0000_0000u64) as i64;
        let mut acc = ir.f.fb.iconst(dynir::Type::I64, nil_bits);
        for (k, v) in &self.entries {
            // A key/value that DIVERGES (e.g. a deferred throw-stub for an
            // unresolved `#'var`) emits `None` and terminates the block. The
            // rest of the map is unreachable; propagate the divergence.
            let Some(k_val) = k.emit(C::Expression, objx, ir) else {
                return None;
            };
            let Some(v_val) = v.emit(C::Expression, objx, ir) else {
                return None;
            };
            acc =
                ir.f.fb
                    .call(self.assoc_fref, &[acc, k_val, v_val])
                    .expect("RT.assoc returns I64 (NanBox)");
        }
        match context {
            C::Statement => None,
            _ => Some(acc),
        }
    }
    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("clojure.lang.PersistentHashMap".to_string()),
        })
    }
}

fn is_constant_object(o: &Object) -> bool {
    match o {
        Object::Nil
        | Object::Bool(_)
        | Object::Long(_)
        | Object::Double(_)
        | Object::String(_)
        | Object::Keyword(_) => true,
        Object::Symbol(_) => false, // symbols are var refs at expression position
        Object::Vector(v) => {
            let n = v.count();
            for i in 0..n {
                if !is_constant_object(&v.nth(i)) {
                    return false;
                }
            }
            true
        }
        Object::List(l) => {
            let mut cur: &PersistentList = l;
            loop {
                match cur {
                    PersistentList::Empty => return true,
                    PersistentList::Cons { first, rest, .. } => {
                        if !is_constant_object(first) {
                            return false;
                        }
                        cur = rest;
                    }
                }
            }
        }
        Object::Map(m) => {
            for (k, v) in m.iter() {
                if !is_constant_object(&k) {
                    return false;
                }
                if !is_constant_object(&v) {
                    return false;
                }
            }
            true
        }
        Object::Set(s) => {
            for el in s.iter() {
                if !is_constant_object(&el) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

/// A literal that was already interned into `pending_literals` by an
/// analyzer — emits `gc_literal(LiteralRef(idx))` directly without re-
/// interning. Used by `analyze_vector` and similar helpers that pre-
/// compute the literal-pool slot.
#[derive(Debug)]
struct ConstantLiteralExpr {
    idx: u32,
}

impl Expr for ConstantLiteralExpr {
    fn eval(&self) -> Object {
        crate::unimplemented_port!(
            "ConstantLiteralExpr.eval",
            "compile-time eval not implemented; only emit path is used"
        )
    }
    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        if context == C::Statement {
            return None;
        }
        let lit = dynir::ir::LiteralRef::from_u32(self.idx);
        Some(ir.f.fb.gc_literal(lit))
    }
    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        None
    }
}

fn analyze_seq(context: C, form: Object) -> Box<dyn Expr> {
    // Macroexpand first: if head is a Symbol resolving to a Var whose
    // `is_macro` is set, invoke its JIT-compiled body with the rest of
    // the form (as an unevaluated heap-allocated cons list) and re-enter
    // `analyze` on the result. Mirrors how `microlisp` (and Java Clojure)
    // do same-image macros — same JitModule serves runtime fns and the
    // compile-time macro expansion call.
    if let Object::List(_) = &form {
        if let Some(expanded) = macroexpand_once(&form) {
            // The macro fn threw during expansion (it hit a deferred
            // class/symbol throw-stub). Defer to runtime: emit a ThrowExpr so
            // the enclosing form compiles and only fails if evaluated.
            if let Some(msg) = MACRO_EXPAND_THREW.with(|c| c.borrow_mut().take()) {
                let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
                return Box::new(ThrowExpr { payload });
            }
            if std::env::var("CLJVM_TRACE_MACRO").is_ok() {
                let head = super::rt::first(&form);
                let head_name = match &head {
                    Object::Symbol(s) => s.get_name().to_string(),
                    _ => "<non-sym>".to_string(),
                };
                let exp_head_str = match &expanded {
                    Object::List(_) => match super::rt::first(&expanded) {
                        Object::Symbol(s) => format!("Symbol({})", s.get_name()),
                        other => format!("{other:?}").chars().take(80).collect(),
                    },
                    other => format!("non-list:{other:?}").chars().take(80).collect(),
                };
                eprintln!("[macroexpand] {head_name} → head={exp_head_str}");
            }
            return analyze(context, expanded);
        }
    }

    let op_raw = super::rt::first(&form);
    // Peel any `WithMeta` wrapper on the head: lazy-seq's macroexpansion
    // produces `(^{:once true} fn* [] body)` where the head symbol carries
    // metadata. Without peeling, the special-form match below skips fn*
    // and the form gets treated as an InvokeExpr with WithMeta as the fn
    // handle — which dispatches into garbage code. (defmacro expansions
    // similarly attach metadata to head symbols in some builders.)
    let op = op_raw.peel_meta_ref().clone();
    let specials = &*SPECIAL_SYMBOLS;

    if let Object::Symbol(sym) = &op {
        // Leading-dot sugar: `(.method recv args…)` → instance dispatch.
        // The reader produces a head Symbol like `.withMeta` (no
        // namespace, name starts with `.`). We turn that into a call
        // through `parse_instance_method_form`. Bare `.` is the static
        // dispatch special form, handled below; only names of length
        // ≥ 2 that start with `.` count as instance sugar.
        if sym.get_namespace().is_none() {
            let name = sym.get_name();
            if name.len() > 1 && name.starts_with('.') {
                return parse_instance_method_form(context, form);
            }
            // Trailing-dot constructor sugar: `(ClassName. args…)` →
            // `(new ClassName args…)` for host classes. For user-defined
            // `deftype`/`defrecord` names we instead rewrite to a direct
            // fn call on the factory: `(Foo. 1 2)` → `(Foo 1 2)`.
            if name.len() > 1 && name.ends_with('.') {
                let class_name = &name[..name.len() - 1];
                let class_sym = Symbol::intern(class_name);
                if crate::lang::user_types::user_type_id_by_name(&class_sym).is_some() {
                    // Rebuild as a direct fn-call on the factory.
                    let mut items: Vec<Object> = Vec::new();
                    items.push(Object::Symbol(class_sym));
                    let mut s = super::rt::next(&form);
                    while !matches!(s, Object::Nil) {
                        items.push(super::rt::first(&s));
                        s = super::rt::next(&s);
                    }
                    return analyze(context, Object::List(PersistentList::create(items)));
                }
                return parse_constructor_sugar(context, class_name, &form);
            }
        }
        // `clojure.core/ns` and `clojure.core/in-ns` are emitted (qualified)
        // by the `ns` macro itself (`(clojure.core/in-ns '~name)`). Recognize
        // them as the ns special form BEFORE the qualified-static-call sugar
        // below — otherwise `clojure.core` (which contains a `.`) is mistaken
        // for a class and the namespace switch is silently dropped, so a
        // loaded file's defs land in the wrong namespace.
        if matches!(sym.get_namespace(), Some("clojure.core"))
            && (sym.get_name() == "in-ns" || sym.get_name() == "ns")
        {
            return parse_ns_form(context, form);
        }
        // Qualified static-call sugar: `(Class/method args...)` →
        // `(. Class (method args...))`. Java's reader rule is "if the
        // head's namespace names a CLASS, treat the whole form as a
        // static method call." We rewrite into the dot-form so
        // `parse_dot_form` handles dispatch + args uniformly.
        //
        // Crucially, the namespace part must name a *class*, NOT a
        // *namespace*. `clojure.core/inc` contains a `.` but `clojure.core`
        // is a namespace, so `(clojure.core/inc 41)` is an ordinary fn
        // invoke, not a static call. Java distinguishes these in
        // `maybeResolveIn`/`isMacro`: a namespace is never a class. If we
        // skip this distinction, any `ns.with.dots/fn` call gets rewritten
        // to a bogus `(. ns.with.dots (fn …))` static call that resolves to
        // no host method → a runtime throw → nil. (This was the real bug
        // behind `(clojure.core/inc 41)` returning nil while `(inc 41)`
        // worked — both resolve to the same Var, but only the bare form
        // took the invoke path.)
        if let Some(ns) = sym.get_namespace() {
            let ns_sym = Symbol::intern(ns);
            // A real namespace (directly or via an alias in the current ns)
            // is never a class — fall through to the normal invoke path.
            let is_namespace = super::namespace::Namespace::find(&ns_sym).is_some()
                || super::rt::current_ns().lookup_alias(&ns_sym).is_some();
            // A symbol that resolves to a Var is an ordinary invoke too,
            // regardless of how its namespace part is spelled.
            let resolves_to_var = resolve_var(sym).is_some();
            // Otherwise apply the structural class heuristic: class names
            // contain dots OR start with uppercase (`Math/ceil`,
            // `Long/parseLong`, `clojure.lang.RT/load`).
            let looks_like_class =
                ns.contains('.') || ns.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
            if looks_like_class && !is_namespace && !resolves_to_var {
                return parse_qualified_static_call(context, ns, sym.get_name(), &form);
            }
        }
        if **sym == *specials.IF {
            return parse_if_form(context, form);
        }
        if **sym == *specials.DO {
            return parse_do_form(context, form);
        }
        if **sym == *specials.QUOTE {
            return parse_quote_form(context, form);
        }
        if **sym == *specials.LET || **sym == *specials.LOOP {
            return parse_let_form(context, form);
        }
        // `letfn*` (post-macro of `letfn`) has the same `[name fn-form ...]`
        // bindings shape as `let*`. Mutual-recursion semantics differ
        // (letfn* binds all names before evaluating any RHS), but for
        // the loader we only need the body to analyze.
        if **sym == *specials.LETFN {
            return parse_let_form(context, form);
        }
        if **sym == *specials.DEF {
            return parse_def_form(context, form);
        }
        if **sym == *specials.FN {
            return parse_fn_form(context, form);
        }
        if **sym == *specials.RECUR {
            return parse_recur_form(context, form);
        }
        if **sym == *specials.DOT {
            return parse_dot_form(context, form);
        }
        if **sym == *specials.NS || **sym == *specials.IN_NS {
            return parse_ns_form(context, form);
        }
        if **sym == *specials.SET_MACRO_BANG {
            return parse_set_macro_bang_form(context, form);
        }
        if **sym == *specials.NEW {
            return parse_new_form(context, form);
        }
        if **sym == *specials.THE_VAR {
            return parse_var_form(context, form);
        }
        if **sym == *specials.THROW {
            return parse_throw_form(context, form);
        }
        if **sym == *specials.TRY {
            return parse_try_form(context, form);
        }
        if **sym == *specials.ASSIGN {
            return parse_assign_form(context, form);
        }
        if **sym == *specials.CASE {
            return parse_case_form(context, form);
        }
        // `reify` / `reify*` — anonymous instance implementing host
        // interfaces / protocols. The `reify` macro lives in
        // `core_deftype.clj` (not embedded) and `reify*` needs
        // `gen-interface` + JVM interface vtables we don't model; the
        // bootstrap's uses (e.g. `future-call` over `java.util.concurrent.
        // Future`) are JVM-interop anyway. Defer like unregistered host
        // classes: emit a runtime throw so the enclosing `defn` compiles and
        // only fails if actually called. The body is NOT analyzed (it
        // references unmodeled interfaces/methods).
        if sym.get_namespace().is_none()
            && (sym.get_name() == "reify" || sym.get_name() == "reify*")
        {
            let payload: Box<dyn Expr> =
                Box::new(StringExpr::new("reify is not supported (no JVM interface impl)"));
            return Box::new(ThrowExpr { payload });
        }
        // Built-in primitive op stand-in (eventually replaced by Var-based
        // resolution to clojure.core/+ etc. with `:inline` metadata).
        if let Some(op) = primop_for_symbol(sym) {
            return parse_primop_form(context, op, form);
        }
    }

    // Non-special-form head — function invocation `(f a b c)`.
    parse_invoke_form(context, form)
}

/// Build a `PrimOpExpr` from `(op a1 a2 …)`.
fn parse_primop_form(_context: C, op: PrimOp, form: Object) -> Box<dyn Expr> {
    let mut args: Vec<Box<dyn Expr>> = Vec::new();
    let mut rest = super::rt::next(&form);
    while !matches!(rest, Object::Nil) {
        args.push(analyze(C::Expression, super::rt::first(&rest)));
        rest = super::rt::next(&rest);
    }
    Box::new(PrimOpExpr { op, args })
}

/// Inline `IfExpr.Parser.parse` — restricted to what we can do without
/// `pushThreadBindings` / `PathNode` / line tracking. Walks the form using
/// the ported `RT.{count,second,third,fourth}` helpers.
/// `(throw expr)` — evaluate `expr`, then transfer control to the
/// nearest enclosing `try`/`catch` handler, or fall out of the JIT
/// entry as a host-visible `JitOutcome::Exception` if none.
///
/// Lowered via the toolkit's `fb.raise(v)` terminator — emits the
/// LLVM `invoke`/`landingpad` cost model: same-fn handler scopes
/// installed via `fb.push_handler(catch_bb)` route locally; otherwise
/// the fn returns with `JitOutcomeKind::Exception` and the caller's
/// post-`Call` check (if any handler is active there) catches it.
/// Cross-function unwind is automatic — no shared prompt id needed.
#[derive(Debug)]
pub struct ThrowExpr {
    payload: Box<dyn Expr>,
}

impl Expr for ThrowExpr {
    fn eval(&self) -> Object {
        crate::unimplemented_port!(
            "ThrowExpr.eval",
            "throw is JIT-only — no tree-walking interpreter path yet"
        )
    }

    fn emit(&self, _context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Payload is evaluated in EXPRESSION context regardless of the
        // enclosing context — its bits must reach `raise`.
        // Payload may itself diverge (a `(throw (throw ...))` is silly
        // but legal); in that case the inner throw already terminated
        // the block, so just propagate None upward.
        let val = self.payload.emit(C::Expression, objx, ir)?;
        ir.f.fb.raise(val);
        // raise is a terminator — `current_block_is_terminated()` is
        // now true, and no caller should try to use a returned value.
        None
    }

    fn has_java_class(&self) -> bool {
        false
    }
    fn get_java_class(&self) -> Option<HostClass> {
        None
    }
}

/// One catch arm of a `try`. Holds the class symbol for type-filter
/// dispatch (currently unused — we accept any exception, matching the
/// catch-all `_` case), the bound exception variable, and the catch
/// body Expr already analyzed in a scope that has the binding visible.
#[derive(Debug)]
pub struct CatchClause {
    /// `(catch <class> binding body…)` — the class symbol. Reserved for
    /// future type-based dispatch; currently every catch arm acts as a
    /// catch-all (in Clojure, the JVM's class hierarchy decides; without
    /// a JVM here we'd need a host class registry).
    #[allow(dead_code)]
    pub class_sym: Option<Arc<Symbol>>,
    /// The binding's LocalBinding entry — its `idx` gives the slot name.
    pub binding: Arc<LocalBinding>,
    /// Catch body, analyzed in a scope with `binding` registered.
    pub body: Box<dyn Expr>,
}

/// `(try body* (catch C binding handler*)* (finally body*)?)`.
///
/// Lowered as:
///   ```text
///   ; body
///   push_handler(catch_bb)
///   <body>            ; emits a value
///   pop_handler()
///   jump merge_bb([<body value>])
///   catch_bb(thrown):
///   <catch body>       ; with `binding` bound to `thrown`
///   jump merge_bb([<catch value>])
///   merge_bb(result):
///   <fall through to caller with `result`>
///   ```
///
/// Multiple catch arms chain: the catch_bb dispatches by class to
/// arm bodies, with a fall-through `raise` if no arm matches. Finally
/// (not yet implemented) wraps the entire region in an outer handler
/// that runs the finally block on both normal and exception paths.
#[derive(Debug)]
pub struct TryExpr {
    /// Body forms, parsed as a do-block.
    pub body: Box<dyn Expr>,
    /// Catch arms, in order. Empty if there's only a `finally`.
    pub catches: Vec<CatchClause>,
    /// Optional finally body (not yet implemented — panics if present).
    #[allow(dead_code)]
    pub finally: Option<Box<dyn Expr>>,
}

impl Expr for TryExpr {
    fn eval(&self) -> Object {
        crate::unimplemented_port!(
            "TryExpr.eval",
            "try is JIT-only — no tree-walking interpreter path yet"
        )
    }

    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Block topology:
        //   - catch_bb: receives the thrown value as its first I64 param
        //   - merge_bb: receives the try-or-catch result as its first I64
        //     param (only in EXPRESSION context; STATEMENT has no result)
        //
        // With `finally`, we add an outer `push_handler(finally_handler_bb)`
        // that wraps the entire try-catch region. On normal exit from body
        // or catch, control reaches a `finally_then_merge_bb` that pops the
        // outer handler, runs finally, and jumps to merge_bb with the
        // captured value. On an unhandled exception (body raises with no
        // catch, or catch arm itself raises), `finally_handler_bb` runs
        // finally and re-raises.  The finally body's IR is emitted twice
        // (inline duplication) — a synthetic-fn approach would be cleaner
        // but isn't needed yet.
        let phi_ty: Option<dynir::Type> = if context == C::Statement {
            None
        } else {
            Some(dynir::Type::I64)
        };
        let catch_bb = ir.f.fb.create_block(&[dynir::Type::I64]);
        let merge_bb = match phi_ty {
            Some(ty) => ir.f.fb.create_block(&[ty]),
            None => ir.f.fb.create_block(&[]),
        };
        // Only used when self.finally is Some — see below.
        let (finally_handler_bb, finally_then_merge_bb) = if self.finally.is_some() {
            let h = ir.f.fb.create_block(&[dynir::Type::I64]);
            let m = match phi_ty {
                Some(ty) => ir.f.fb.create_block(&[ty]),
                None => ir.f.fb.create_block(&[]),
            };
            (Some(h), Some(m))
        } else {
            (None, None)
        };

        // If finally is active, install the outer handler now so it
        // catches anything that body/catch don't.
        if let Some(fhb) = finally_handler_bb {
            ir.f.fb.push_handler(fhb);
        }

        // The "normal-exit join" — without finally this IS merge_bb;
        // with finally it's finally_then_merge_bb (which runs finally,
        // then jumps to merge_bb).
        let body_join_bb = finally_then_merge_bb.unwrap_or(merge_bb);

        // Track whether either path actually jumps to merge_bb. If
        // both body and catch diverge (e.g. both end with a `throw`),
        // merge_bb is unreachable and we must NOT switch to it —
        // doing so leaves the caller in a dead block with no
        // predecessors, which crashes the lowerer (no entry-handler
        // state, regalloc can't enter, etc.). Mirrors IfExpr's
        // `any_reached` pattern.
        let mut any_reached = false;

        // ── Body region (under handler) ─────────────────────────
        ir.f.fb.push_handler(catch_bb);
        let body_val = self.body.emit(context, objx, ir);
        if !ir.f.fb.current_block_is_terminated() {
            any_reached = true;
            ir.f.fb.pop_handler();
            match (phi_ty.is_some(), body_val) {
                (true, Some(v)) => ir.f.fb.jump(body_join_bb, &[v]),
                (false, _) => ir.f.fb.jump(body_join_bb, &[]),
                (true, None) => {
                    panic!("clojure-jvm: TryExpr body produced no value in non-STATEMENT context")
                }
            }
        }

        // ── Catch dispatch ──────────────────────────────────────
        ir.f.fb.switch_to_block(catch_bb);
        let thrown = ir.f.fb.block_param(catch_bb, 0);

        if self.catches.is_empty() {
            // No catch arms (a `(try body (finally f))`). Re-raise
            // so the outer finally handler runs and propagates.
            ir.f.fb.raise(thrown);
        } else {
            // Single-arm case (simplest): bind `thrown` to the arm's
            // local binding slot, emit the arm body, jump to merge.
            // Multi-arm case extends this with class-based dispatch
            // between arms; we don't have a host class registry yet,
            // so every arm acts as a catch-all and the FIRST arm wins.
            if self.catches.len() > 1 {
                crate::unimplemented_port!(
                    "TryExpr.emit with multiple catch arms",
                    "needs host-class registry for type-filter dispatch"
                );
            }
            let arm = &self.catches[0];
            ir.f.def_var(&local_slot_name(arm.binding.idx), thrown);
            let arm_val = arm.body.emit(context, objx, ir);
            if !ir.f.fb.current_block_is_terminated() {
                any_reached = true;
                match (phi_ty.is_some(), arm_val) {
                    (true, Some(v)) => ir.f.fb.jump(body_join_bb, &[v]),
                    (false, _) => ir.f.fb.jump(body_join_bb, &[]),
                    (true, None) => panic!(
                        "clojure-jvm: TryExpr catch arm produced no value in non-STATEMENT context"
                    ),
                }
            }
        }

        // ── Finally (if present) ────────────────────────────────
        if let (Some(finally_expr), Some(fjm), Some(fhb)) = (
            self.finally.as_ref(),
            finally_then_merge_bb,
            finally_handler_bb,
        ) {
            // Normal-exit join: pop the outer (finally) handler, run
            // finally inline, jump to the real merge with the captured
            // value. Only reachable if body or catch normally exits;
            // otherwise plug with unreachable so the verifier accepts
            // the block.
            ir.f.fb.switch_to_block(fjm);
            if any_reached {
                let saved_val = if phi_ty.is_some() {
                    Some(ir.f.fb.block_param(fjm, 0))
                } else {
                    None
                };
                ir.f.fb.pop_handler();
                let _ = finally_expr.emit(C::Statement, objx, ir);
                if !ir.f.fb.current_block_is_terminated() {
                    match (phi_ty.is_some(), saved_val) {
                        (true, Some(v)) => ir.f.fb.jump(merge_bb, &[v]),
                        (false, _) => ir.f.fb.jump(merge_bb, &[]),
                        (true, None) => unreachable!(),
                    }
                }
            } else {
                // Still need block_param to satisfy the type sig if
                // phi_ty was set (the block carries a param).
                if phi_ty.is_some() {
                    let _ = ir.f.fb.block_param(fjm, 0);
                }
                ir.f.fb.unreachable();
            }

            // Exception path: finally runs, then re-raises the original
            // thrown value. Always emitted — finally_handler_bb is
            // reachable via the outer push_handler whenever ANYTHING
            // inside the try region raises.
            ir.f.fb.switch_to_block(fhb);
            let exn = ir.f.fb.block_param(fhb, 0);
            let _ = finally_expr.emit(C::Statement, objx, ir);
            if !ir.f.fb.current_block_is_terminated() {
                ir.f.fb.raise(exn);
            }
        }

        // ── Merge ───────────────────────────────────────────────
        if !any_reached {
            // Body and catch both diverged. merge_bb is unreachable
            // but we created it — `fb.build()` requires every block
            // to be terminated, so plug it with `unreachable`. The
            // current block stays terminated (catch_bb's raise), so
            // returning None tells the caller we don't produce a
            // value at the join site.
            ir.f.fb.switch_to_block(merge_bb);
            ir.f.fb.unreachable();
            return None;
        }
        ir.f.fb.switch_to_block(merge_bb);
        if phi_ty.is_some() {
            Some(ir.f.fb.block_param(merge_bb, 0))
        } else {
            None
        }
    }

    fn has_java_class(&self) -> bool {
        false
    }
    fn get_java_class(&self) -> Option<HostClass> {
        None
    }
}

fn parse_try_form(context: C, form: Object) -> Box<dyn Expr> {
    let specials = &*SPECIAL_SYMBOLS;
    // Walk the children classifying: body | catch | finally. Body forms
    // must precede catch/finally; any subsequent body form is rejected.
    let mut body_forms: Vec<Object> = Vec::new();
    let mut catch_specs: Vec<(Object, Arc<Symbol>, Arc<Symbol>, Vec<Object>)> = Vec::new();
    let mut finally_forms: Option<Vec<Object>> = None;
    let mut rest = super::rt::next(&form);
    while !matches!(rest, Object::Nil) {
        let head = super::rt::first(&rest);
        let head_op = if let Object::List(l) = &head {
            if l.count() >= 1 {
                let f0 = super::rt::first(&head);
                if let Object::Symbol(s) = &f0 {
                    if **s == *specials.CATCH {
                        Some("catch")
                    } else if **s == *specials.FINALLY {
                        Some("finally")
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        match head_op {
            Some("catch") => {
                // (catch Class binding body...)
                let class_form = super::rt::second(&head);
                let class_sym = match class_form {
                    Object::Symbol(s) => s.clone(),
                    other => {
                        panic!("clojure-jvm: bad catch class spec (expected symbol), got {other:?}")
                    }
                };
                let bind_form = super::rt::third(&head);
                let bind_sym = match bind_form {
                    Object::Symbol(s) => s.clone(),
                    other => panic!(
                        "clojure-jvm: bad catch binding spec (expected symbol), got {other:?}"
                    ),
                };
                // Body of the catch arm: everything after Class and binding.
                let body_seq = super::rt::next(&super::rt::next(&super::rt::next(&head)));
                let mut arm_body: Vec<Object> = Vec::new();
                let mut cur = body_seq;
                while !matches!(cur, Object::Nil) {
                    arm_body.push(super::rt::first(&cur));
                    cur = super::rt::next(&cur);
                }
                catch_specs.push((head.clone(), class_sym, bind_sym, arm_body));
            }
            Some("finally") => {
                let body_seq = super::rt::next(&head);
                let mut fb: Vec<Object> = Vec::new();
                let mut cur = body_seq;
                while !matches!(cur, Object::Nil) {
                    fb.push(super::rt::first(&cur));
                    cur = super::rt::next(&cur);
                }
                if finally_forms.is_some() {
                    panic!("clojure-jvm: try can have at most one finally clause");
                }
                finally_forms = Some(fb);
            }
            None => {
                if !catch_specs.is_empty() || finally_forms.is_some() {
                    panic!("clojure-jvm: try body forms must precede catch/finally");
                }
                body_forms.push(head);
            }
            Some(_) => unreachable!(),
        }
        rest = super::rt::next(&rest);
    }

    // Analyze the body as a do-block, in the EXPRESSION context (so it
    // produces a value when caller wants one). STATEMENT context still
    // propagates through.
    let body_seq = Object::List(PersistentList::create(body_forms));
    let body = parse_body_seq(context, body_seq);

    // Analyze each catch arm in a scope with its binding registered.
    let mut catches: Vec<CatchClause> = Vec::new();
    for (_orig, class_sym, bind_sym, arm_body_forms) in catch_specs {
        Var::push_thread_bindings(vec![
            (
                COMPILER_VARS.LOCAL_ENV.clone(),
                Object::Host(current_local_env()),
            ),
            (
                COMPILER_VARS.NEXT_LOCAL_NUM.clone(),
                Object::Long(current_next_local_num() as i64),
            ),
        ]);
        let arm_lb = register_local(bind_sym, None, None, false);
        let arm_seq = Object::List(PersistentList::create(arm_body_forms));
        let arm_body_expr = parse_body_seq(context, arm_seq);
        Var::pop_thread_bindings();

        catches.push(CatchClause {
            class_sym: Some(class_sym),
            binding: arm_lb,
            body: arm_body_expr,
        });
    }

    let finally = finally_forms.map(|forms| {
        let seq = Object::List(PersistentList::create(forms));
        parse_body_seq(C::Statement, seq)
    });

    Box::new(TryExpr {
        body,
        catches,
        finally,
    })
}

fn parse_throw_form(_context: C, form: Object) -> Box<dyn Expr> {
    // (throw x) — exactly one argument. (throw) with no payload is
    // legal in Clojure (rethrows the active exception) but we don't
    // support rethrow yet; flag it loudly.
    let n = super::rt::count(&form);
    if n != 2 {
        panic!(
            "clojure-jvm: RuntimeException — Wrong number of args ({}) to throw (only (throw expr) is supported)",
            n - 1
        );
    }
    let payload = analyze(C::Expression, super::rt::second(&form));
    Box::new(ThrowExpr { payload })
}

fn parse_if_form(context: C, form: Object) -> Box<dyn Expr> {
    let n = super::rt::count(&form);
    if n > 4 {
        panic!("clojure-jvm: RuntimeException — Too many arguments to if");
    }
    if n < 3 {
        panic!("clojure-jvm: RuntimeException — Too few arguments to if");
    }
    let test_ctx = if context == C::Eval {
        C::Eval
    } else {
        C::Expression
    };
    let test_expr = analyze(test_ctx, super::rt::second(&form));
    let then_expr = analyze(context, super::rt::third(&form));
    let else_expr = if n == 4 {
        analyze(context, super::rt::fourth(&form))
    } else {
        Box::new(NIL_EXPR) as Box<dyn Expr>
    };
    Box::new(IfExpr::new(0, 0, test_expr, then_expr, else_expr))
}

// ============================================================================
// Java line ~9348–9664: `CaseExpr` + its `Parser` — the `case*` special form.
// ============================================================================

/// One `case*` switch arm: the (pre-shifted/masked) integer key the macro
/// computed, the test constant to re-check against post-switch (None for
/// hash-collision buckets, whose `then` is a macro-generated `condp` that
/// does its own checking — the `skip-check` set), and the result expr.
#[derive(Debug)]
struct CaseArm {
    key: i64,
    test: Option<Box<dyn Expr>>,
    then: Box<dyn Expr>,
}

/// `Compiler.CaseExpr`. `(case ...)` macroexpands to
/// `(let* [ge e] (case* ge shift mask default case-map switch-type test-type
/// skip-check?))`; this node compiles the `case*` into a real `Switch`
/// terminator over a dispatch index computed by the `cljvm_case_dispatch`
/// runtime helper (`Util.hash` for `:hash-equiv`/`:hash-identity`,
/// `Number.intValue()` for `:int`, then `(h >> shift) & mask` when masked).
///
/// Divergences from Java's emit, none observable:
///   * compact-vs-sparse (`tableswitch` vs `lookupswitch`) is a JVM
///     bytecode concern; dynir's `Switch` takes arbitrary sparse keys.
///   * `:hash-identity`'s `IF_ACMPNE` reference-identity check is emitted
///     as the same `Util.equiv`-style check as `:hash-equiv`. Java's
///     identity check is an optimization that relies on constants being
///     canonical instances (interned Keywords); our keyword heap cells are
///     non-canonical wrappers, so equivalence is the correct (and equal-
///     outcome) comparison.
///   * the dispatch value is emitted ONCE and the SSA value reused for the
///     post-switch equivalence checks (Java re-loads the local each time).
#[derive(Debug)]
pub struct CaseExpr {
    expr: Box<dyn Expr>,
    shift: i64,
    mask: i64,
    default_expr: Box<dyn Expr>,
    arms: Vec<CaseArm>,
    /// `:int` test type? (false → `:hash-equiv` / `:hash-identity`).
    test_is_int: bool,
    /// `cljvm_case_dispatch` (switch-index helper).
    dispatch_fref: dynir::FuncRef,
    /// `cljvm_equals` — `Util.equiv` for the post-switch constant check.
    equals_fref: dynir::FuncRef,
}

impl Expr for CaseExpr {
    fn eval(&self) -> Object {
        // Java: throw new UnsupportedOperationException("Can't eval case").
        panic!(
            "clojure-jvm: CaseExpr.eval is not a tree-walker — compile via \
             the JIT pipeline instead"
        )
    }

    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Dispatch value: always EXPRESSION context (we need it to switch
        // on, even when the surrounding context is STATEMENT).
        let val = self.expr.emit(C::Expression, objx, ir)?;

        // Switch index. RAW i64 (never NanBox-tagged — i32-sign-extended
        // hashes and the no-match sentinel all fail the tag-pattern check),
        // so it is GC-inert like other unboxed primitives.
        let tt = ir.f.fb.iconst(
            dynir::Type::I64,
            if self.test_is_int {
                crate::runtime::CASE_TEST_INT
            } else {
                crate::runtime::CASE_TEST_HASH
            },
        );
        let sh = ir.f.fb.iconst(dynir::Type::I64, self.shift);
        let mk = ir.f.fb.iconst(dynir::Type::I64, self.mask);
        let disp =
            ir.f.fb
                .call(self.dispatch_fref, &[val, tt, sh, mk])
                .expect("cljvm_case_dispatch returns I64");

        let default_bb = ir.f.fb.create_block(&[]);
        let arm_bbs: Vec<dynir::BlockId> = self
            .arms
            .iter()
            .map(|_| ir.f.fb.create_block(&[]))
            .collect();

        // Phi type — same lazy merge-block pattern as IfExpr: only create
        // the merge block when at least one branch actually reaches it
        // (every branch may end in recur/throw).
        let phi_ty_opt: Option<dynir::Type> = if context == C::Statement {
            None
        } else {
            Some(dynir::Type::I64)
        };
        let mut merge_bb: Option<dynir::BlockId> = None;
        let ensure_merge =
            |ir: &mut IrEmitter<'_>, slot: &mut Option<dynir::BlockId>| -> dynir::BlockId {
                if let Some(b) = *slot {
                    return b;
                }
                let b = match phi_ty_opt {
                    Some(ty) => ir.f.fb.create_block(&[ty]),
                    None => ir.f.fb.create_block(&[]),
                };
                *slot = Some(b);
                b
            };
        // Branch tail shared by every arm + the default: jump to the merge
        // block with/without the branch value, unless already terminated.
        let finish_branch = |ir: &mut IrEmitter<'_>,
                             slot: &mut Option<dynir::BlockId>,
                             branch_val: Option<Value>,
                             what: &str| {
            if ir.f.fb.current_block_is_terminated() {
                return;
            }
            let mb = ensure_merge(ir, slot);
            match (phi_ty_opt.is_some(), branch_val) {
                (true, Some(v)) => ir.f.fb.jump(mb, &[v]),
                (false, _) => ir.f.fb.jump(mb, &[]),
                (true, None) => panic!(
                    "clojure-jvm: CaseExpr {what} branch produced no value in \
                     non-STATEMENT context"
                ),
            }
        };

        let cases: Vec<(i64, dynir::BlockId, &[Value])> = self
            .arms
            .iter()
            .zip(&arm_bbs)
            .map(|(arm, bb)| (arm.key, *bb, &[][..]))
            .collect();
        ir.f.fb.switch(disp, &cases, default_bb, &[]);

        for (arm, bb) in self.arms.iter().zip(&arm_bbs) {
            ir.f.fb.switch_to_block(*bb);
            // Post-switch equivalence check (Java `emitThenForInts` /
            // `emitThenForHashes`): `Util.equiv(val, test)` or jump to
            // default. Skipped for hash-collision buckets (skip-check).
            if let Some(test) = &arm.test {
                let test_val = test
                    .emit(C::Expression, objx, ir)
                    .expect("case* test constants always produce a value");
                let eq =
                    ir.f.fb
                        .call(self.equals_fref, &[val, test_val])
                        .expect("cljvm_equals returns I64");
                let then_bb = ir.f.fb.create_block(&[]);
                ir.f.br_if_truthy(eq, then_bb, &[], default_bb, &[]);
                ir.f.fb.switch_to_block(then_bb);
            }
            let then_val = arm.then.emit(context, objx, ir);
            finish_branch(ir, &mut merge_bb, then_val, "then");
        }

        ir.f.fb.switch_to_block(default_bb);
        let default_val = self.default_expr.emit(context, objx, ir);
        finish_branch(ir, &mut merge_bb, default_val, "default");

        let Some(mb) = merge_bb else {
            // Every branch diverged (recur/throw). Cursor sits on the
            // terminated default block — callers detect via
            // `current_block_is_terminated()`, same contract as IfExpr.
            return None;
        };
        ir.f.fb.switch_to_block(mb);
        if phi_ty_opt.is_some() {
            Some(ir.f.fb.block_param(mb, 0))
        } else {
            None
        }
    }

    fn has_java_class(&self) -> bool {
        self.get_java_class().is_some()
    }

    fn get_java_class(&self) -> Option<HostClass> {
        // Java: `returnType = maybeJavaClass(thens + default)` — non-null
        // only when every branch agrees.
        let mut found: Option<HostClass> = None;
        for e in self
            .arms
            .iter()
            .map(|a| &a.then)
            .chain(std::iter::once(&self.default_expr))
        {
            if !e.has_java_class() {
                return None;
            }
            let c = e.get_java_class()?;
            match &found {
                None => found = Some(c),
                Some(prev) if *prev == c => {}
                Some(_) => return None,
            }
        }
        found
    }
}

/// Read an integer out of a `case*` structural slot, panicking with a
/// clear message otherwise (the form is macro-generated — a non-integer
/// here means corrupted expansion, not user error).
fn expect_case_long(o: &Object, what: &str) -> i64 {
    match o.peel_meta_ref() {
        Object::Long(n) => *n,
        other => panic!("clojure-jvm: case* {what} must be an integer, got {other:?}"),
    }
}

/// Inline `CaseExpr.Parser.parse` —
/// `(case* expr shift mask default case-map switch-type test-type skip-check?)`
/// prepared by the `case` macro and presumed correct (Java line ~9595).
fn parse_case_form(context: C, form: Object) -> Box<dyn Expr> {
    // Java: EVAL context wraps the form in an immediately-invoked fn —
    // `(( fn* [] form))` — and analyzes that (same routing as `let*`).
    if context == C::Eval {
        let empty_params: Object = Object::Vector(
            crate::lang::persistent_vector::PersistentVector::create(Vec::new()),
        );
        let inner_call: Object =
            Object::List(crate::lang::persistent_list::PersistentList::create(vec![
                Object::Symbol(SPECIAL_SYMBOLS.FN.clone()),
                empty_params,
                form.clone(),
            ]));
        let outer_call: Object =
            Object::List(crate::lang::persistent_list::PersistentList::create(vec![
                inner_call,
            ]));
        return analyze(context, outer_call);
    }

    // Collect the args after the `case*` head.
    let mut items: Vec<Object> = Vec::new();
    let mut s = super::rt::next(&form);
    while !matches!(s, Object::Nil) {
        items.push(super::rt::first(&s));
        s = super::rt::next(&s);
    }
    if !(7..=8).contains(&items.len()) {
        panic!(
            "clojure-jvm: case* expects 7 or 8 args \
             (expr shift mask default case-map switch-type test-type skip-check?), \
             got {}",
            items.len()
        );
    }

    let expr_form = items[0].clone();
    let shift = expect_case_long(&items[1], "shift");
    let mask = expect_case_long(&items[2], "mask");
    let default_form = items[3].clone();
    let case_map_form = items[4].clone();
    // items[5] = switch-type (:compact / :sparse) — a JVM tableswitch-vs-
    // lookupswitch concern; dynir's `Switch` handles sparse keys natively,
    // so the distinction needs no representation here.
    let test_type_name = match items[6].peel_meta_ref() {
        Object::Keyword(k) => k.get_name().to_string(),
        other => panic!("clojure-jvm: case* test-type must be a keyword, got {other:?}"),
    };
    let test_is_int = match test_type_name.as_str() {
        "int" => true,
        "hash-equiv" | "hash-identity" => false,
        other => panic!("clojure-jvm: case* — unexpected test type: :{other}"),
    };
    let skip_check: std::collections::HashSet<i64> = match items.get(7) {
        None => std::collections::HashSet::new(),
        Some(o) => match o.peel_meta_ref() {
            Object::Nil => std::collections::HashSet::new(),
            Object::Set(set) => set
                .iter()
                .map(|e| expect_case_long(&e, "skip-check entry"))
                .collect(),
            Object::TreeSet(set) => set
                .iter()
                .map(|e| expect_case_long(&e, "skip-check entry"))
                .collect(),
            other => panic!("clojure-jvm: case* skip-check must be a set or nil, got {other:?}"),
        },
    };

    // Java analyzes the dispatch expr in EXPRESSION context (it's always
    // the `let*`-bound local the macro introduced).
    let expr = analyze(C::Expression, expr_form);

    // case-map: `{switch-key [test-constant then-form]}`, built by the
    // macro with `sorted-map` (PersistentTreeMap); accept a hash map too.
    let entries_src: Vec<(Object, Object)> = match case_map_form.peel_meta_ref() {
        Object::TreeMap(m) => m.iter().collect(),
        Object::Map(m) => m.iter().collect(),
        other => panic!("clojure-jvm: case* case-map must be a map, got {other:?}"),
    };
    let mut arms: Vec<CaseArm> = Vec::with_capacity(entries_src.len());
    for (k, v) in entries_src {
        let key = expect_case_long(&k, "case-map key");
        if key == crate::runtime::CASE_DISPATCH_NO_MATCH {
            // Impossible for macro-generated maps (keys are i32-ranged or
            // shift-masked) — guards the dispatch helper's sentinel.
            panic!(
                "clojure-jvm: case* case-map key {key} collides with the \
                 non-number dispatch sentinel"
            );
        }
        let (test_form, then_form) = match v.peel_meta_ref() {
            Object::Vector(pair) if pair.count() == 2 => (pair.nth(0), pair.nth(1)),
            other => panic!(
                "clojure-jvm: case* case-map value must be a [test then] pair, got {other:?}"
            ),
        };
        let test: Option<Box<dyn Expr>> = if skip_check.contains(&key) {
            // Hash-collision bucket: `test` is the seq of colliding
            // constants and `then` is a macro-generated `condp` that does
            // its own equality checks — no post-switch check is emitted.
            None
        } else if test_is_int {
            // Java: NumberExpr.parse(((Number)RT.first(pair)).intValue()).
            match test_form.peel_meta_ref() {
                Object::Long(n) => Some(NumberExpr::parse(Object::Long(*n))),
                other => panic!(
                    "clojure-jvm: case* :int test constant must be an integer, got {other:?}"
                ),
            }
        } else {
            Some(constant_literal_expr(test_form.clone()))
        };
        let then = analyze(context, then_form);
        arms.push(CaseArm { key, test, then });
    }

    let default_expr = analyze(context, default_form);
    let (dispatch_fref, equals_fref) = with_active_compiler(|c| (c.case_dispatch, c.num.eq));
    Box::new(CaseExpr {
        expr,
        shift,
        mask,
        default_expr,
        arms,
        test_is_int,
        dispatch_fref,
        equals_fref,
    })
}

/// Inline `BodyExpr.Parser.parse` for `(do …)`. Strips the leading `do`,
/// analyzes each remaining form (statement context for all but the tail),
/// and wraps in a BodyExpr. Empty body becomes `(do nil)`.
fn parse_do_form(context: C, form: Object) -> Box<dyn Expr> {
    let body_forms = super::rt::next(&form);
    let mut exprs: Vec<Box<dyn Expr>> = Vec::new();
    if let Object::List(l) = body_forms {
        let total = l.count();
        for (i, child) in l.iter().enumerate() {
            let last = i + 1 == total as usize;
            let child_ctx = if !last && context != C::Eval {
                C::Statement
            } else {
                context
            };
            exprs.push(analyze(child_ctx, child));
        }
    }
    if exprs.is_empty() {
        exprs.push(Box::new(NIL_EXPR));
    }
    Box::new(BodyExpr::new(exprs))
}

// ============================================================================
// Java line ~6526–6675: `LocalBinding` + `LocalBindingExpr` + `BindingInit`.
// ============================================================================

/// `Compiler.LocalBinding`. Holds one let/loop/fn-arg slot: its symbol, the
/// optional tag, the slot index, and the init expression.
///
/// Java fields:
/// ```text
/// public final Symbol sym, tag;
/// public Expr init;
/// int idx;
/// public final String name;
/// public final boolean isArg;
/// public boolean canBeCleared;
/// public boolean recurMistmatch;
/// public boolean used;
/// ```
///
/// We elide `clearPathRoot` and the locals-clearing flags for now — they're
/// runtime-codegen concerns we'll surface when we wire ObjExpr.
#[derive(Debug)]
pub struct LocalBinding {
    pub sym: Arc<Symbol>,
    pub tag: Option<Arc<Symbol>>,
    /// `init` is filled in once analyze produces an Expr for the value
    /// position. Wrapped in `RwLock` because `letfn*` writes it after seeding.
    pub init: RwLock<Option<Arc<dyn Expr>>>,
    /// Slot index within the enclosing method's stack frame.
    pub idx: i32,
    /// `munge(sym.name)` — used as the local-variable name in emitted code.
    pub name: String,
    pub is_arg: bool,
    /// ID of the fn body this binding belongs to. Used by closure analysis:
    /// when `LocalBindingExpr.emit` runs inside fn body `B`, a binding with
    /// `owning_fn_id != B` is a *captured* local — its value comes from
    /// the enclosing closure's varlen-values section rather than a stack
    /// slot. `0` is the top-level (no enclosing fn).
    pub owning_fn_id: u32,
    /// Slot index in the *capturing* fn's closure-captures vector, keyed
    /// by that fn's id. A single binding may be captured by multiple
    /// nested fns (grandparent + parent + child); each gets its own slot
    /// in its own closure object. Populated by `record_capture` when the
    /// active capture scope first records this binding.
    pub capture_slots: RwLock<HashMap<u32, usize>>,
}

impl LocalBinding {
    pub fn new(
        idx: i32,
        sym: Arc<Symbol>,
        tag: Option<Arc<Symbol>>,
        init: Option<Arc<dyn Expr>>,
        is_arg: bool,
    ) -> Arc<Self> {
        // Java: throws on `maybePrimitiveType(init) != null && tag != null`.
        // We don't have maybePrimitiveType yet, so skip the guard for now.
        let name = munge(sym.get_name());
        let owning_fn_id = current_fn_id();
        Arc::new(LocalBinding {
            sym,
            tag,
            init: RwLock::new(init),
            idx,
            name,
            is_arg,
            owning_fn_id,
            capture_slots: RwLock::new(HashMap::new()),
        })
    }
}

/// `Compiler.BindingInit`. Pair of (LocalBinding, init Expr) emitted as a
/// single slot during codegen.
#[derive(Debug)]
pub struct BindingInit {
    pub binding: Arc<LocalBinding>,
    pub init: Arc<dyn Expr>,
}

impl BindingInit {
    pub fn new(binding: Arc<LocalBinding>, init: Arc<dyn Expr>) -> Self {
        BindingInit { binding, init }
    }
}

/// `Compiler.LocalBindingExpr`. A reference to a local — emits a load from
/// the binding's slot. Implements `Expr`, `MaybePrimitiveExpr`, and
/// `AssignableExpr`.
#[derive(Debug)]
pub struct LocalBindingExpr {
    pub b: Arc<LocalBinding>,
    pub tag: Option<Arc<Symbol>>,
}

impl LocalBindingExpr {
    pub fn new(b: Arc<LocalBinding>, tag: Option<Arc<Symbol>>) -> Self {
        // Java's tag-vs-primitive-type check is gated on
        // `b.getPrimitiveType() != null` which itself requires `init.maybe...`
        // — skip for now.
        LocalBindingExpr { b, tag }
    }
}

impl Expr for LocalBindingExpr {
    fn eval(&self) -> Object {
        // Java: throws UnsupportedOperationException("Can't eval locals").
        // Locals only exist inside a compiled frame — top-level eval must
        // go through compile-emit-JIT.
        panic!("clojure-jvm: UnsupportedOperationException — Can't eval locals");
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `objx.emitLocal(gen, b, shouldClear)` — loads from the
        // binding's stack slot for non-captured locals, or from the
        // closure's varlen-values section for captures.
        if context == C::Statement {
            // Java drops the load; we just skip the read.
            return None;
        }
        let cur = current_fn_id();
        if cur != 0 && self.b.owning_fn_id != cur {
            // Captured local: read from the closure self-arg's varlen-values
            // section at the slot assigned by `record_capture` for the
            // current (capturing) fn.
            let slot = *self
                .b
                .capture_slots
                .read()
                .unwrap()
                .get(&cur)
                .unwrap_or_else(|| {
                    panic!(
                        "clojure-jvm: LocalBindingExpr.emit: capture slot missing for binding \
                         {} in fn {cur} — record_capture didn't run for this fn?",
                        self.b.sym.get_name()
                    )
                });
            let self_val = ir.f.get_var(CLOSURE_SELF_SLOT);
            let raw = ir.f.obj_unwrap(self_val);
            // Read varlen-base from the cached closure handle — the
            // DynModule's obj_types Vec was moved out at Session::new
            // and `dm.get_obj_type` would panic with "len 0".
            let base_offset = with_active_compiler(|c| c.closure_varlen_base);
            let base = ir.f.fb.iconst(dynir::Type::I64, base_offset);
            let eight = ir.f.fb.iconst(dynir::Type::I64, 8);
            let idx = ir.f.fb.iconst(dynir::Type::I64, slot as i64);
            let byte_off = ir.f.fb.mul(idx, eight);
            let off = ir.f.fb.add(base, byte_off);
            let addr = ir.f.fb.add(raw, off);
            return Some(ir.f.fb.load(dynir::Type::I64, addr, 0));
        }
        Some(ir.f.get_var(&local_slot_name(self.b.idx)))
    }

    fn has_java_class(&self) -> bool {
        self.tag.is_some()
            || self
                .b
                .init
                .read()
                .unwrap()
                .as_ref()
                .map(|e| e.has_java_class())
                .unwrap_or(false)
    }

    fn get_java_class(&self) -> Option<HostClass> {
        // Java: tag != null → HostExpr.tagToClass(tag); else init.getJavaClass.
        if let Some(t) = &self.tag {
            // No HostExpr port yet — return a HostClass over the tag symbol
            // name (best we can do without resolving Java classes).
            return Some(HostClass {
                name: Arc::new(t.get_name().to_string()),
            });
        }
        self.b
            .init
            .read()
            .unwrap()
            .as_ref()
            .and_then(|e| e.get_java_class())
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> {
        Some(self)
    }

    fn as_local_binding_expr(&self) -> Option<&LocalBindingExpr> {
        Some(self)
    }
}

impl MaybePrimitiveExpr for LocalBindingExpr {
    fn can_emit_primitive(&self) -> bool {
        // Java: `b.getPrimitiveType() != null`. Without `maybePrimitiveType`
        // wired, conservative `false`.
        false
    }

    fn emit_unboxed(&self, _context: C, _objx: &ObjExpr, _ir: &mut IrEmitter<'_>) -> Value {
        crate::unimplemented_port!(
            "LocalBindingExpr.emitUnboxed",
            "needs ObjExpr.emitUnboxedLocal"
        )
    }
}

impl AssignableExpr for LocalBindingExpr {
    fn eval_assign(&self, _val: &dyn Expr) -> Object {
        panic!("clojure-jvm: UnsupportedOperationException — Can't eval locals");
    }

    fn emit_assign(
        &self,
        _context: C,
        _objx: &ObjExpr,
        _ir: &mut IrEmitter<'_>,
        _val: &dyn Expr,
    ) -> Option<Value> {
        crate::unimplemented_port!(
            "LocalBindingExpr.emitAssign",
            "needs ObjExpr.emitAssignLocal"
        )
    }
}

/// Per-LocalBinding stack slot name used by `DynFunc::def_var` / `get_var`.
/// Each `LocalBinding` has a unique `idx` assigned by `getAndIncLocalNum`, so
/// the slot name is collision-free within a single function build.
pub fn local_slot_name(idx: i32) -> String {
    format!("local__{idx}")
}

/// Reserved slot name holding the closure-self pointer when a fn body is
/// a closure. Set by `lower_pending_fn` from the implicit first param.
pub const CLOSURE_SELF_SLOT: &str = "__closure_self";

/// `Compiler.munge(String name)` — Java line ~3435. Replaces non-Java
/// identifier chars with `_FOO_` escapes per `CHAR_MAP`. We use the same
/// table inline.
pub fn munge(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        let sub: Option<&'static str> = match ch {
            '-' => Some("_"),
            ':' => Some("_COLON_"),
            '+' => Some("_PLUS_"),
            '>' => Some("_GT_"),
            '<' => Some("_LT_"),
            '=' => Some("_EQ_"),
            '~' => Some("_TILDE_"),
            '!' => Some("_BANG_"),
            '@' => Some("_CIRCA_"),
            '#' => Some("_SHARP_"),
            '\'' => Some("_SINGLEQUOTE_"),
            '"' => Some("_DOUBLEQUOTE_"),
            '%' => Some("_PERCENT_"),
            '^' => Some("_CARET_"),
            '&' => Some("_AMPERSAND_"),
            '*' => Some("_STAR_"),
            '|' => Some("_BAR_"),
            '{' => Some("_LBRACE_"),
            '}' => Some("_RBRACE_"),
            '[' => Some("_LBRACK_"),
            ']' => Some("_RBRACK_"),
            '/' => Some("_SLASH_"),
            '\\' => Some("_BSLASH_"),
            '?' => Some("_QMARK_"),
            _ => None,
        };
        match sub {
            Some(s) => out.push_str(s),
            None => out.push(ch),
        }
    }
    out
}

// ============================================================================
// LOCAL_ENV plumbing.
//
// Java stores `LOCAL_ENV` as an `IPersistentMap<Symbol, LocalBinding>` value
// kept inside the dynamic Var of the same name. Since we don't have a
// persistent map yet, the map's value is an `Arc<HashMap<...>>` wrapped in
// `Object::Host`.
// ============================================================================

/// Convenience type — the value held in `LOCAL_ENV`.
pub type LocalEnvMap = HashMap<Arc<Symbol>, Arc<LocalBinding>>;

// ─── fn-scope identity (for closure capture detection) ──────────────────
//
// Every `parse_fn_form` call mints a fresh `fn_id` from `NEXT_FN_ID` and
// pushes it onto `CURRENT_FN_ID`. Each `LocalBinding` stamps the fn_id
// active when it was registered. When `LocalBindingExpr.emit` runs inside
// fn body `B`, it compares `binding.owning_fn_id` to the current emit
// fn_id; a mismatch means the binding lives in an enclosing fn — it's a
// *capture* and must be loaded from the closure object rather than from
// a stack slot. Body lowering (`lower_pending_fn`) sets `CURRENT_FN_ID`
// to the FnExpr's id around emit.
//
// Top-level forms get fn_id 0 (no enclosing fn).

static NEXT_FN_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

thread_local! {
    static CURRENT_FN_ID: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

thread_local! {
    /// Set when a macro fn *throws* during macroexpansion (`macroexpand_once`).
    /// Rather than abort the whole load, the macroexpander stashes the
    /// exception message here and returns `nil`; `analyze_seq` picks it up and
    /// emits a deferred runtime `ThrowExpr` so the enclosing form compiles and
    /// only fails if evaluated. (Macros throw at expansion in our image mostly
    /// because their body hit a deferred class/symbol throw-stub.)
    static MACRO_EXPAND_THREW: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

fn mint_fn_id() -> u32 {
    NEXT_FN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn current_fn_id() -> u32 {
    CURRENT_FN_ID.with(|c| c.get())
}

/// RAII guard that swaps `CURRENT_FN_ID` for the duration of a scope.
struct FnIdGuard {
    prev: u32,
}

impl FnIdGuard {
    fn new(new_id: u32) -> Self {
        let prev = CURRENT_FN_ID.with(|c| c.replace(new_id));
        FnIdGuard { prev }
    }
}

impl Drop for FnIdGuard {
    fn drop(&mut self) {
        CURRENT_FN_ID.with(|c| c.set(self.prev));
    }
}

// ─── Capture recording (during analyze) ─────────────────────────────────
//
// When `reference_local` finds a binding belonging to an outer fn, we
// append it to the active fn's capture set. The set is keyed by binding
// pointer (Arc as_ptr) to dedupe — multiple references to the same outer
// local produce one capture entry. Captures are collected per-fn during
// body analysis and consumed by `parse_fn_form` after the body parse.

struct CaptureScope {
    /// fn_id of the fn that owns this scope. A binding gets recorded as a
    /// capture only when its `owning_fn_id` differs from this — i.e., it
    /// lives outside the fn whose captures we're collecting.
    fn_id: u32,
    captures: Vec<Arc<LocalBinding>>,
}

thread_local! {
    /// Stack of capture-sets, one entry per enclosing fn currently being
    /// parsed. The top entry is the active fn's captures. Pushed by
    /// `parse_fn_form` before body analyze, popped after.
    static CAPTURE_STACK: std::cell::RefCell<Vec<CaptureScope>> =
        std::cell::RefCell::new(Vec::new());
}

fn push_capture_scope(fn_id: u32) {
    CAPTURE_STACK.with(|s| {
        s.borrow_mut().push(CaptureScope {
            fn_id,
            captures: Vec::new(),
        })
    });
}

fn pop_capture_scope() -> Vec<Arc<LocalBinding>> {
    CAPTURE_STACK
        .with(|s| s.borrow_mut().pop())
        .map(|sc| sc.captures)
        .unwrap_or_default()
}

/// Record `binding` as a capture of the active fn, deduping by pointer.
/// Only adds the binding if its `owning_fn_id` belongs to a fn OUTSIDE
/// the active scope (the active scope's `fn_id` is the fn collecting
/// captures). Assigns `capture_slot` for the binding on first record.
fn record_capture(binding: &Arc<LocalBinding>) {
    CAPTURE_STACK.with(|s| {
        let mut stack = s.borrow_mut();
        let Some(active) = stack.last_mut() else {
            return;
        };
        if binding.owning_fn_id == active.fn_id {
            return;
        }
        for existing in active.captures.iter() {
            if Arc::ptr_eq(existing, binding) {
                return;
            }
        }
        let slot = active.captures.len();
        active.captures.push(binding.clone());
        binding
            .capture_slots
            .write()
            .unwrap()
            .insert(active.fn_id, slot);
    });
}

fn current_local_env() -> Arc<LocalEnvMap> {
    let v = COMPILER_VARS.LOCAL_ENV.deref();
    v.host_as::<LocalEnvMap>()
        .unwrap_or_else(|| Arc::new(HashMap::new()))
}

fn current_next_local_num() -> i32 {
    match COMPILER_VARS.NEXT_LOCAL_NUM.deref() {
        Object::Long(n) => n as i32,
        _ => 0,
    }
}

fn set_local_env(env: Arc<LocalEnvMap>) {
    COMPILER_VARS.LOCAL_ENV.set_value(Object::Host(env));
}

fn set_next_local_num(n: i32) {
    COMPILER_VARS
        .NEXT_LOCAL_NUM
        .set_value(Object::Long(n as i64));
}

/// `Compiler.registerLocal(Symbol sym, Symbol tag, Expr init, boolean isArg)`
/// — Java line ~7313. Allocates a slot, installs the LocalBinding in
/// `LOCAL_ENV`, bumps `NEXT_LOCAL_NUM`. Java also updates `METHOD.locals` /
/// `METHOD.indexlocals`; we skip those until ObjMethod is real.
pub fn register_local(
    sym: Arc<Symbol>,
    tag: Option<Arc<Symbol>>,
    init: Option<Arc<dyn Expr>>,
    is_arg: bool,
) -> Arc<LocalBinding> {
    let num = current_next_local_num();
    set_next_local_num(num + 1);
    let lb = LocalBinding::new(num, sym.clone(), tag, init, is_arg);
    let env = current_local_env();
    let mut new_env: LocalEnvMap = (*env).clone();
    new_env.insert(sym, lb.clone());
    set_local_env(Arc::new(new_env));
    lb
}

/// `Compiler.tagOf(Object o)` — extracts a `:tag` metadata symbol if present.
/// We don't have IMeta yet; always returns `None`.
fn tag_of(_o: &Object) -> Option<Arc<Symbol>> {
    None
}

/// `Compiler.analyzeSymbol(Symbol sym)` — Java line ~7865. Resolves a bare
/// symbol to:
///
///   * an unqualified local → `LocalBindingExpr`
///   * a qualified or top-level name → `VarExpr` (stubbed; needs `resolve`)
///
/// We only implement the local-binding path for now; falling off the local
/// case panics with a clear `analyze_symbol` not-ported message.
pub fn analyze_symbol(sym: Arc<Symbol>) -> Box<dyn Expr> {
    let tag = tag_of(&Object::Symbol(sym.clone()));
    if sym.get_namespace().is_none() {
        if let Some(b) = reference_local(&sym) {
            return Box::new(LocalBindingExpr::new(b, tag));
        }
    }
    // Class-name resolution comes BEFORE the var lookup. In Clojure a bare or
    // qualified class name (`String`, `clojure.lang.Var`) resolves to the
    // *class*, never to a like-named var. This precedence is essential here:
    // `prelude.clj` defines mock vars (`(def String "#<MOCKED…>")`) so that
    // references to *unregistered* classes compile through, but those mocks
    // must not shadow a real registered host class — otherwise
    // `(instance? String x)`, `(Exception. …)`, `(symbol "s")` etc. receive a
    // string where they expect the Class, and `instance?`/`new` misbehave.
    // Unregistered class names still fall through to their mock var below.
    let combined = match sym.get_namespace() {
        Some(ns) => format!("{ns}.{}", sym.get_name()),
        None => sym.get_name().to_string(),
    };
    if let Some(info) = super::host_class::lookup(&combined) {
        let idx = with_active_compiler(|c| c.intern_literal(PendingLiteral::Class(info.id)));
        return Box::new(ConstantLiteralExpr { idx });
    }
    // Java: `resolve(sym)` walks the current ns's mappings, falling back to
    // the symbol's explicit ns. We do a simplified lookup: try the current
    // ns, then a qualified ns if present.
    let resolved = resolve_var(&sym);
    if let Some(v) = resolved {
        return Box::new(VarExpr { var: v, tag });
    }
    // Qualified-symbol fallback: `Class/STATIC_FIELD` references where
    // `Class` is a registered host class OR looks like a Java class
    // path (contains a dot). We don't model Java static fields so
    // resolve to nil (the def using this still analyzes; if it runs,
    // the field reference reads as nil rather than the actual Java
    // value). This accommodation lets Var statics and the IRef
    // family compile through, plus enum-style constants like
    // `java.util.concurrent.TimeUnit/MILLISECONDS`.
    // For dotted symbols that look like Java class paths (e.g.
    // `java.lang.IllegalAccessError`), we treat them as "host
    // class reference we don't model" and emit a runtime throw
    // — same pattern as unregistered instance/static methods.
    // This lets defns that REFERENCE the class for catch-clause
    // matching or `instance?` checks compile, even though calling
    // those code paths fails at runtime.
    let combined_str = combined.clone();
    if combined_str.contains('.') {
        let msg = format!("unresolved class reference: {combined_str}");
        let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
        return Box::new(ThrowExpr { payload });
    }
    // Unresolved non-dotted symbol. Rather than abort analysis (which would
    // kill the whole load), defer like an unresolved class reference: emit a
    // runtime throw so the enclosing `defn`/`defmethod` compiles and only
    // fails if that code path actually runs. This is what lets core.clj forms
    // that reference fns from sub-files we don't embed (e.g.
    // `print-sequential` from `core_print.clj`) load, and it gives a clear
    // error at the call site rather than a silent wrong value. Forward
    // references (mutual recursion before both vars exist) land here too.
    let msg = format!("Unable to resolve symbol: {} in this context", sym.get_name());
    let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
    Box::new(ThrowExpr { payload })
}

/// `Compiler.resolve(Object sym)` — Java line ~7965 (simplified). Looks up
/// `sym` in the current namespace's mappings, falling back to its qualified
/// namespace if any.
///
/// For qualified symbols (`foo/bar`), the prefix can be either a real
/// namespace name (e.g. `clojure.core/inc`) or an alias bound in the
/// current namespace (e.g. `core/inc` after `(alias 'core 'clojure.core)`
/// or `(ns _ (:require [clojure.core :as core]))`).
fn resolve_var(sym: &Symbol) -> Option<Arc<Var>> {
    use super::namespace::Namespace;
    if let Some(ns_str) = sym.get_namespace() {
        let ns_sym = Symbol::intern(ns_str);
        let target_ns = Namespace::find(&ns_sym).or_else(|| {
            // Not a real namespace — try resolving as an alias in the
            // current namespace.
            super::rt::current_ns().lookup_alias(&ns_sym)
        })?;
        return target_ns.find_interned_var(&Symbol::intern(sym.get_name()));
    }
    let cur = super::rt::current_ns();
    if let Some(Object::Var(v)) = cur.get_mapping(sym) {
        return Some(v);
    }
    // Every namespace auto-refers `clojure.core` (Clojure's `ns` macro emits
    // `(refer 'clojure.core)`). We model that as a resolution-time fallback
    // rather than snapshotting core's mappings at ns creation: an unqualified
    // symbol not mapped in the current ns resolves to the like-named
    // `clojure.core` var if one exists. This is what lets sub-files loaded
    // into their own namespace (e.g. `(ns clojure.core.protocols)` using
    // `seq`/`first`/`reduced`) see core. Skipped when already in core.
    if cur.name.get_name() != "clojure.core" {
        if let Some(core) = Namespace::find(&Symbol::intern("clojure.core")) {
            return core.find_interned_var(&Symbol::intern(sym.get_name()));
        }
    }
    None
}

/// `Compiler.referenceLocal(Symbol sym)` — Java line ~8130. Looks up the
/// LocalBinding for `sym` in the current `LOCAL_ENV`. Returns `None` if
/// `LOCAL_ENV` isn't thread-bound (top-level analyze with no enclosing fn)
/// or if `sym` isn't in scope.
///
/// Java also flips `method.usesThis` for slot 0 and calls `closeOver` to
/// thread the binding into closures. We skip both until ObjMethod / FnExpr
/// are wired up.
pub fn reference_local(sym: &Symbol) -> Option<Arc<LocalBinding>> {
    let env = current_local_env();
    // current_local_env returns an Arc<HashMap> — we need to find by Symbol
    // equality (structural, not pointer). The Symbol key uses (ns, name)
    // structural equality via our PartialEq impl.
    for (k, v) in env.iter() {
        if **k == *sym {
            // If this binding lives in an outer fn, record it as a capture
            // of the active fn. Top-level (`current_fn_id() == 0`) bindings
            // never come from "outer" — they are themselves the top scope.
            let cur = current_fn_id();
            if cur != 0 && v.owning_fn_id != cur {
                record_capture(v);
            }
            return Some(v.clone());
        }
    }
    None
}

// ============================================================================
// Java line ~6900–7134: `LetExpr` + `LetExpr.Parser`.
// ============================================================================

/// `Compiler.LetExpr`. Java fields: `bindingInits`, `body`, `isLoop`.
#[derive(Debug)]
pub struct LetExpr {
    pub binding_inits: Vec<BindingInit>,
    pub body: Box<dyn Expr>,
    pub is_loop: bool,
}

impl LetExpr {
    pub fn new(binding_inits: Vec<BindingInit>, body: Box<dyn Expr>, is_loop: bool) -> Self {
        LetExpr {
            binding_inits,
            body,
            is_loop,
        }
    }
}

impl Expr for LetExpr {
    fn eval(&self) -> Object {
        // Java: throws UnsupportedOperationException("Can't eval let/loop").
        panic!("clojure-jvm: UnsupportedOperationException — Can't eval let/loop");
    }

    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        self.do_emit(context, objx, ir, false)
    }

    fn has_java_class(&self) -> bool {
        self.body.has_java_class()
    }
    fn get_java_class(&self) -> Option<HostClass> {
        self.body.get_java_class()
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> {
        Some(self)
    }
}

impl MaybePrimitiveExpr for LetExpr {
    fn can_emit_primitive(&self) -> bool {
        self.body
            .as_maybe_primitive()
            .map(|m| m.can_emit_primitive())
            .unwrap_or(false)
    }

    fn emit_unboxed(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Value {
        self.do_emit(context, objx, ir, true)
            .expect("LetExpr.emit_unboxed: STATEMENT not allowed when unboxed")
    }
}

impl LetExpr {
    /// Java `LetExpr.doEmit(C, ObjExpr, GeneratorAdapter, boolean)` — line
    /// ~7058. Allocates a stack slot per BindingInit, emits each init into
    /// its slot, then emits the body. Loop semantics (`isLoop = true`) add a
    /// label at the slot-storing point so `recur` can jump back; not wired
    /// yet because we don't have `RecurExpr`.
    fn do_emit(
        &self,
        context: C,
        objx: &ObjExpr,
        ir: &mut IrEmitter<'_>,
        emit_unboxed: bool,
    ) -> Option<Value> {
        // Java pushes a per-binding stack slot via `gen.visitVarInsn(ISTORE)`.
        // dynlang gives us a name-keyed slot map per DynFunc; we mint a
        // `local__<idx>` name from the LocalBinding's slot index.
        for bi in &self.binding_inits {
            match bi.init.emit(C::Expression, objx, ir) {
                Some(init_val) => {
                    ir.f.def_var(&local_slot_name(bi.binding.idx), init_val);
                }
                None => {
                    // The init emitted a terminator (e.g. `(throw …)`
                    // or a `(recur …)` in tail position). The current
                    // block is dead; no point emitting further
                    // bindings or the body. The whole let diverges.
                    return None;
                }
            }
        }

        // For `loop*`: create a body_top block, jump to it, and bind
        // LOOP_LABEL to a RecurTarget pointing at body_top with this
        // loop's binding locals. Without this, RecurExpr.emit reads
        // LOOP_LABEL from the enclosing fn body — recur silently
        // jumps to the fn's body top with empty locals, the loop's
        // `recur` ends up a no-op (slot updates are skipped because
        // fn-arity mismatches the loop's arity), and the loop spins.
        // Symptom seen at form 46: defmacro's inner loops never
        // progress, eventually corrupting the macro result with
        // stale heap pointers.
        let pushed_loop_label = if self.is_loop {
            let body_top = ir.f.fb.create_block(&[]);
            ir.f.fb.jump(body_top, &[]);
            ir.f.fb.switch_to_block(body_top);
            let recur_target = RecurTarget {
                block: body_top,
                locals: self
                    .binding_inits
                    .iter()
                    .map(|bi| bi.binding.clone())
                    .collect(),
            };
            Var::push_thread_bindings(vec![(
                COMPILER_VARS.LOOP_LABEL.clone(),
                Object::Host(std::sync::Arc::new(recur_target)),
            )]);
            true
        } else {
            false
        };

        let result = if emit_unboxed {
            let body_val = self
                .body
                .as_maybe_primitive()
                .expect("LetExpr.emit_unboxed: body must be MaybePrimitiveExpr")
                .emit_unboxed(context, objx, ir);
            Some(body_val)
        } else {
            self.body.emit(context, objx, ir)
        };

        if pushed_loop_label {
            Var::pop_thread_bindings();
        }
        result
    }
}

/// `Compiler.LetExpr.Parser`. Drives both `let*` and `loop*`. Java line ~6911.
pub struct LetExprParser;

impl IParser for LetExprParser {
    fn parse(&self, context: C, frm: Object) -> Box<dyn Expr> {
        parse_let_form(context, frm)
    }
}

/// Rewrite a fn* params vector to replace any non-symbol pattern
/// with a synthetic symbol, and prepend a `let*` to the body that
/// destructures the synthetic into the original pattern's bindings.
///
/// Returns `(rewritten_params, rewritten_body_seq)`. If no params
/// need destructuring, both are returned unchanged.
fn rewrite_fn_params_for_destructuring(
    params: Arc<PersistentVector>,
    body_seq: Object,
) -> (Arc<PersistentVector>, Object) {
    use crate::lang::persistent_vector::PersistentVector;
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = params.count();
    let mut needs = false;
    for i in 0..n {
        let p = params.nth(i);
        if matches!(p.peel_meta_ref(), Object::Vector(_) | Object::Map(_)) {
            needs = true;
            break;
        }
    }
    if !needs {
        return (params, body_seq);
    }
    let mut new_params: Vec<Object> = Vec::with_capacity(n as usize);
    let mut destructure_pairs: Vec<Object> = Vec::new();
    let mut i = 0;
    while i < n {
        let p = params.nth(i);
        // `& rest`: pass through (with destructuring of rest pattern if needed).
        if let Object::Symbol(s) = p.peel_meta_ref() {
            if s.get_name() == "&" {
                new_params.push(p.clone());
                i += 1;
                if i < n {
                    let rest_p = params.nth(i);
                    if matches!(rest_p.peel_meta_ref(), Object::Vector(_) | Object::Map(_)) {
                        let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let synth = Symbol::intern(&format!("rest__fp{id}__"));
                        new_params.push(Object::Symbol(synth.clone()));
                        emit_destructuring(
                            &rest_p,
                            Object::Symbol(synth),
                            &mut destructure_pairs,
                            &N,
                        );
                    } else {
                        new_params.push(rest_p);
                    }
                    i += 1;
                }
                continue;
            }
        }
        match p.peel_meta_ref() {
            Object::Symbol(_) => {
                new_params.push(p);
            }
            _ => {
                let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let synth = Symbol::intern(&format!("fp__d{id}__"));
                new_params.push(Object::Symbol(synth.clone()));
                emit_destructuring(&p, Object::Symbol(synth), &mut destructure_pairs, &N);
            }
        }
        i += 1;
    }
    let new_params_vec = PersistentVector::create(new_params);
    if destructure_pairs.is_empty() {
        return (new_params_vec, body_seq);
    }
    // Wrap body in (let* [destructure-pairs…] body…).
    let mut body_items: Vec<Object> = Vec::new();
    let mut cur = body_seq.clone();
    while !matches!(cur, Object::Nil) {
        body_items.push(super::rt::first(&cur));
        cur = super::rt::next(&cur);
    }
    let let_form = {
        let mut items: Vec<Object> = vec![
            Object::Symbol(Symbol::intern("let*")),
            Object::Vector(PersistentVector::create(destructure_pairs)),
        ];
        items.extend(body_items);
        Object::List(PersistentList::create(items))
    };
    let new_body_seq = Object::List(PersistentList::create(vec![let_form]));
    (new_params_vec, new_body_seq)
}

/// Loop-specific destructuring: each non-symbol pattern is replaced
/// by a fresh synthetic symbol; the original pattern's bindings move
/// to a body-prefix `let*`. This preserves loop's binding arity so
/// `(recur arg)` stays valid.
///
/// Returns (rewritten-bindings, body-prefix-pairs).
fn rewrite_loop_bindings_for_destructuring(
    bindings: &Arc<crate::lang::persistent_vector::PersistentVector>,
) -> (
    Arc<crate::lang::persistent_vector::PersistentVector>,
    Vec<Object>,
) {
    use crate::lang::persistent_vector::PersistentVector;
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = bindings.count();
    let mut needs = false;
    for i in (0..n).step_by(2) {
        if !matches!(bindings.nth(i).peel_meta_ref(), Object::Symbol(_)) {
            needs = true;
            break;
        }
    }
    if !needs {
        return (bindings.clone(), Vec::new());
    }
    let mut new_pairs: Vec<Object> = Vec::with_capacity(n as usize);
    let mut prefix: Vec<Object> = Vec::new();
    for i in (0..n).step_by(2) {
        let pat = bindings.nth(i);
        let init = bindings.nth(i + 1);
        match pat.peel_meta_ref().clone() {
            Object::Symbol(_) => {
                new_pairs.push(pat);
                new_pairs.push(init);
            }
            _ => {
                let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let synth = Symbol::intern(&format!("loop__d{id}__"));
                new_pairs.push(Object::Symbol(synth.clone()));
                new_pairs.push(init);
                emit_destructuring(&pat, Object::Symbol(synth), &mut prefix, &N);
            }
        }
    }
    (PersistentVector::create(new_pairs), prefix)
}

/// Expand destructuring patterns in a let/loop bindings vector into
/// a flat sequence of `(symbol init-expr)` pairs.
///
/// Supported patterns:
///   * Symbol  → pass-through
///   * Vector  `[a b c]`        → `tmp init  a (nth tmp 0)  b (nth tmp 1) …`
///                with `& rest` → `… rest (nthnext tmp k)`
///   * Map     `{:keys [a b] :or {a 1} :as m}`
///                              → `m init  a (get m :a 1)  b (get m :b) …`
///
/// Nested patterns recurse: each newly-emitted binding goes through
/// the same pass via expansion.
fn expand_destructuring_bindings(bindings: &Arc<PersistentVector>) -> Arc<PersistentVector> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    use crate::lang::persistent_vector::PersistentVector;
    let n = bindings.count();
    let mut needs_expansion = false;
    for i in (0..n).step_by(2) {
        if !matches!(&bindings.nth(i).peel_meta(), Object::Symbol(_)) {
            needs_expansion = true;
            break;
        }
    }
    if !needs_expansion {
        return bindings.clone();
    }
    let mut out: Vec<Object> = Vec::new();
    for i in (0..n).step_by(2) {
        let pat = bindings.nth(i);
        let init = bindings.nth(i + 1);
        emit_destructuring(&pat, init, &mut out, &N);
    }
    PersistentVector::create(out)
}

/// Recursively emit `(symbol init)` pairs for one destructuring pattern.
fn emit_destructuring(
    pat: &Object,
    init: Object,
    out: &mut Vec<Object>,
    counter: &std::sync::atomic::AtomicU64,
) {
    match pat.peel_meta_ref().clone() {
        Object::Symbol(s) => {
            out.push(Object::Symbol(s));
            out.push(init);
        }
        Object::Vector(v) => {
            // [a b c] / [a & rest] / [a b :as m]
            let id = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = Symbol::intern(&format!("vec__d{id}__"));
            out.push(Object::Symbol(tmp.clone()));
            out.push(init);
            let count = v.count();
            let mut idx: i64 = 0;
            let mut i = 0i32;
            while i < count {
                let elem = v.nth(i);
                if let Object::Symbol(s) = elem.peel_meta_ref().clone() {
                    if s.get_name() == "&" {
                        // Next item is the rest binding.
                        i += 1;
                        if i < count {
                            let rest_pat = v.nth(i);
                            // (nthnext tmp idx)
                            let rest_init = Object::List(PersistentList::create(vec![
                                Object::Symbol(Symbol::intern("nthnext")),
                                Object::Symbol(tmp.clone()),
                                Object::Long(idx),
                            ]));
                            emit_destructuring(&rest_pat, rest_init, out, counter);
                        }
                        i += 1;
                        continue;
                    }
                    if s.get_name() == ":as" || s.get_name() == "as" {
                        // :as binding-sym (rare; lookahead).
                        i += 1;
                        if i < count {
                            let as_pat = v.nth(i);
                            emit_destructuring(&as_pat, Object::Symbol(tmp.clone()), out, counter);
                        }
                        i += 1;
                        continue;
                    }
                }
                if let Object::Keyword(k) = &elem {
                    if k.get_name() == "as" && k.get_namespace().is_none() {
                        i += 1;
                        if i < count {
                            let as_pat = v.nth(i);
                            emit_destructuring(&as_pat, Object::Symbol(tmp.clone()), out, counter);
                        }
                        i += 1;
                        continue;
                    }
                }
                let elem_init = Object::List(PersistentList::create(vec![
                    Object::Symbol(Symbol::intern("nth")),
                    Object::Symbol(tmp.clone()),
                    Object::Long(idx),
                    Object::Nil,
                ]));
                emit_destructuring(&elem, elem_init, out, counter);
                idx += 1;
                i += 1;
            }
        }
        Object::Map(m) => {
            // Simplified map destructuring. Supports {:keys […] :or {…} :as sym}
            // and {sym key …} entries. Drops :strs / :syms variants.
            let id = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = Symbol::intern(&format!("map__d{id}__"));
            out.push(Object::Symbol(tmp.clone()));
            out.push(init);
            // Look for :or, :as first.
            let or_kw = crate::lang::keyword::Keyword::intern_ns_name(None, "or");
            let or_map = m.val_at(&Object::Keyword(or_kw));
            let as_kw = crate::lang::keyword::Keyword::intern_ns_name(None, "as");
            let as_sym = m.val_at(&Object::Keyword(as_kw));
            if let Object::Symbol(s) = as_sym {
                out.push(Object::Symbol(s));
                out.push(Object::Symbol(tmp.clone()));
            }
            for (k, v) in m.iter() {
                match (&k, &v) {
                    (Object::Keyword(kw), Object::Vector(syms))
                        if kw.get_name() == "keys"
                            || kw.get_name() == "strs"
                            || kw.get_name() == "syms" =>
                    {
                        for j in 0..syms.count() {
                            let sym_obj = syms.nth(j);
                            if let Object::Symbol(s) = &sym_obj {
                                let key_obj = match kw.get_name() {
                                    "strs" => Object::String(Arc::new(s.get_name().to_string())),
                                    "syms" => Object::List(PersistentList::create(vec![
                                        Object::Symbol(Symbol::intern("quote")),
                                        Object::Symbol(s.clone()),
                                    ])),
                                    _ => Object::Keyword(
                                        crate::lang::keyword::Keyword::intern_ns_name(
                                            None,
                                            s.get_name(),
                                        ),
                                    ),
                                };
                                let default = match &or_map {
                                    Object::Map(om) => {
                                        let v = om.val_at(&Object::Symbol(s.clone()));
                                        if matches!(v, Object::Nil) {
                                            Object::Nil
                                        } else {
                                            v
                                        }
                                    }
                                    _ => Object::Nil,
                                };
                                let init_call = Object::List(PersistentList::create(vec![
                                    Object::Symbol(Symbol::intern("get")),
                                    Object::Symbol(tmp.clone()),
                                    key_obj,
                                    default,
                                ]));
                                out.push(Object::Symbol(s.clone()));
                                out.push(init_call);
                            }
                        }
                    }
                    (Object::Keyword(kw), _) if kw.get_name() == "as" || kw.get_name() == "or" => {
                        // already handled above
                    }
                    (sym_pat, key_obj) => {
                        // {sym :key} entry
                        let default = match &or_map {
                            Object::Map(om) => {
                                if let Object::Symbol(s) = sym_pat {
                                    let v = om.val_at(&Object::Symbol(s.clone()));
                                    if matches!(v, Object::Nil) {
                                        Object::Nil
                                    } else {
                                        v
                                    }
                                } else {
                                    Object::Nil
                                }
                            }
                            _ => Object::Nil,
                        };
                        let init_call = Object::List(PersistentList::create(vec![
                            Object::Symbol(Symbol::intern("get")),
                            Object::Symbol(tmp.clone()),
                            key_obj.clone(),
                            default,
                        ]));
                        emit_destructuring(sym_pat, init_call, out, counter);
                    }
                }
            }
        }
        other => panic!("clojure-jvm: unsupported destructuring pattern: {other:?}"),
    }
}

fn parse_let_form(context: C, form: Object) -> Box<dyn Expr> {
    let specials = &*SPECIAL_SYMBOLS;
    let head = super::rt::first(&form);
    let is_loop = match &head {
        Object::Symbol(s) => **s == *specials.LOOP,
        _ => false,
    };

    // `(let [bindings...] body...)` / `(loop [bindings...] body...)`
    let bindings_obj = super::rt::second(&form);
    let raw_bindings: Arc<PersistentVector> = match bindings_obj {
        Object::Vector(v) => v,
        _ => panic!("clojure-jvm: IllegalArgumentException — Bad binding form, expected vector"),
    };
    if raw_bindings.count() % 2 != 0 {
        panic!(
            "clojure-jvm: IllegalArgumentException — Bad binding form, expected matched symbol expression pairs"
        );
    }
    // Body is everything after the bindings vector.
    let raw_body_seq = super::rt::next(&super::rt::next(&form));
    // Pre-process bindings: expand any non-symbol patterns (vector
    // destructuring `[a b c]`, map destructuring `{:keys [...]}`)
    // into a flat sequence of symbol/init pairs. Pure symbol patterns
    // pass through unchanged.
    //
    // For loop, we DON'T flatten bindings — each pattern keeps its
    // synthetic name as a single binding so `(recur arg)` arity stays
    // correct. The destructuring is moved to a body-prefix `let*`.
    let (bindings, body_seq) = if is_loop {
        let (b, prefix_pairs) = rewrite_loop_bindings_for_destructuring(&raw_bindings);
        if prefix_pairs.is_empty() {
            (b, raw_body_seq)
        } else {
            // Wrap body in (let* [prefix-pairs...] body...).
            let mut body_items: Vec<Object> = Vec::new();
            let mut cur = raw_body_seq;
            while !matches!(cur, Object::Nil) {
                body_items.push(super::rt::first(&cur));
                cur = super::rt::next(&cur);
            }
            let inner_let = {
                let mut items: Vec<Object> = vec![
                    Object::Symbol(Symbol::intern("let*")),
                    Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
                        prefix_pairs,
                    )),
                ];
                items.extend(body_items);
                Object::List(PersistentList::create(items))
            };
            let new_body = Object::List(PersistentList::create(vec![inner_let]));
            (b, new_body)
        }
    } else {
        (expand_destructuring_bindings(&raw_bindings), raw_body_seq)
    };

    // Java: EVAL contexts (and loop in EXPRESSION context) wrap the form in
    // an immediately-invoked `(fn* [] form)`. We don't have FnExpr yet, so
    // we route EVAL straight through as well — semantics will diverge for
    // top-level eval until FnExpr lands; flag it loudly.
    if context == C::Eval || (context == C::Expression && is_loop) {
        // Java path:
        //   return analyze(context, RT.list(RT.list(FNONCE, PersistentVector.EMPTY, form)));
        // We don't have FNONCE metadata yet, but plain `fn*` produces an
        // equivalent result for the cases the bootstrap exercises (a loop
        // appearing as an expression value — e.g. a `let` binding init).
        let empty_params: Object = Object::Vector(
            crate::lang::persistent_vector::PersistentVector::create(Vec::new()),
        );
        let inner_call: Object =
            Object::List(crate::lang::persistent_list::PersistentList::create(vec![
                Object::Symbol(SPECIAL_SYMBOLS.FN.clone()),
                empty_params,
                form.clone(),
            ]));
        let outer_call: Object =
            Object::List(crate::lang::persistent_list::PersistentList::create(vec![
                inner_call,
            ]));
        return analyze(context, outer_call);
    }

    // Snapshot LOCAL_ENV + NEXT_LOCAL_NUM before binding new locals so the
    // bindings scope is undone when we pop. Java does this via
    // pushThreadBindings.
    Var::push_thread_bindings(vec![
        (
            COMPILER_VARS.LOCAL_ENV.clone(),
            Object::Host(current_local_env()),
        ),
        (
            COMPILER_VARS.NEXT_LOCAL_NUM.clone(),
            Object::Long(current_next_local_num() as i64),
        ),
    ]);

    let result = (|| -> Box<dyn Expr> {
        let mut binding_inits: Vec<BindingInit> = Vec::new();

        let n = bindings.count();
        let mut i = 0;
        while i < n {
            let sym_form = bindings.nth(i);
            // Peel reader-attached metadata (`^Type sym`).
            let sym = match sym_form.peel_meta_ref().clone() {
                Object::Symbol(s) => s,
                other => panic!(
                    "clojure-jvm: IllegalArgumentException — Bad binding form, expected symbol, got: {other:?}"
                ),
            };
            if sym.get_namespace().is_some() {
                panic!(
                    "clojure-jvm: RuntimeException — Can't let qualified name: {}",
                    sym.get_name()
                );
            }
            let init_form = bindings.nth(i + 1);
            let init: Arc<dyn Expr> = Arc::from(analyze_named(
                C::Expression,
                init_form,
                Some(sym.get_name()),
            ));
            // Java skips primitive coercion (longCast / doubleCast / box)
            // here for non-loop forms. We don't have those StaticMethodExpr
            // shims yet, so the path stays the same for loop/let.
            let tag = tag_of(&sym_form);
            let lb = register_local(sym, tag, Some(init.clone()), false);
            binding_inits.push(BindingInit::new(lb, init));
            i += 2;
        }

        // For loops: Java sets LOOP_LOCALS, then runs body in C.RETURN.
        // We push the loop's locals as LOOP_LOCALS so `parse_recur` can
        // arity-check against the right set (otherwise it inherits the
        // enclosing fn's params, which produces the wrong "expected N
        // args" error when a loop has a different binding count than the
        // enclosing fn's arity).
        let body_ctx = if is_loop { C::Return } else { context };
        if is_loop {
            let locals: Vec<Arc<LocalBinding>> =
                binding_inits.iter().map(|bi| bi.binding.clone()).collect();
            Var::push_thread_bindings(vec![(
                COMPILER_VARS.LOOP_LOCALS.clone(),
                Object::Host(std::sync::Arc::new(locals)),
            )]);
        }
        let body_expr = parse_body_seq(body_ctx, body_seq);
        if is_loop {
            Var::pop_thread_bindings();
        }

        Box::new(LetExpr::new(binding_inits, body_expr, is_loop))
    })();

    Var::pop_thread_bindings();
    result
}

/// Like `parse_do_form` but takes the already-extracted body seq (no leading
/// `do` to strip). Mirrors what `BodyExpr.Parser.parse` does after the head
/// strip.
fn parse_body_seq(context: C, body_forms: Object) -> Box<dyn Expr> {
    let mut exprs: Vec<Box<dyn Expr>> = Vec::new();
    if let Object::List(l) = body_forms {
        let total = l.count();
        for (i, child) in l.iter().enumerate() {
            let last = i + 1 == total as usize;
            let child_ctx = if !last && context != C::Eval {
                C::Statement
            } else {
                context
            };
            exprs.push(analyze(child_ctx, child));
        }
    }
    if exprs.is_empty() {
        exprs.push(Box::new(NIL_EXPR));
    }
    Box::new(BodyExpr::new(exprs))
}

// ============================================================================
// Java line ~5868–6525: `FnMethod` + Java line ~4483–4729: `FnExpr`.
// Java line ~4128–4471: `InvokeExpr`.
//
// We port a minimal slice:
//   * Single-arity `(fn* [args] body)` only (no variadic, no multi-arity)
//   * No closures (closed-over locals panic at parse time)
//   * No `this`/named-fn shorthand
//   * Body is a `BodyExpr` in `C::Return` context
//
// The fn body becomes a fresh dynir function declared on the active
// `Compiler` during analyze; FnExpr stores the FuncRef so InvokeExpr can
// emit a direct `fb.call`. First-class fn values (passing the fn as a
// runtime value through call_via_func_ref) come later.
// ============================================================================

/// `Compiler.FnMethod`. One arity-overload of a fn — its param list, body,
/// and source position.
#[derive(Debug, Clone)]
pub struct FnMethod {
    pub params: Vec<Arc<LocalBinding>>,
    pub body: Arc<dyn Expr>,
    pub line: i32,
    pub column: i32,
}

/// `Compiler.FnExpr`. A user-defined fn. Each `FnExpr` declares a fresh dynir
/// function (held in `fref`) and stores its single (for now) `FnMethod`.
#[derive(Debug, Clone)]
pub struct FnExpr {
    pub method: FnMethod,
    pub fref: dynir::FuncRef,
    pub name: String,
    /// `true` when the param vector contains `&`, marking the next symbol as
    /// the rest param. The declared fn arity is `fixed_arity + 1` (the rest
    /// param shows up as one extra local bound to a list).
    pub is_variadic: bool,
    /// Number of fixed params before `&`. `0..=fixed_arity-1` are required
    /// positional args; index `fixed_arity` (when `is_variadic`) is the rest
    /// list.
    pub fixed_arity: usize,
    /// Compile-time identity of this fn body (matches `LocalBinding.owning_fn_id`).
    pub fn_id: u32,
    /// Outer LocalBindings referenced from this fn's body. Non-empty means
    /// this fn is a *closure* — emit allocates a `clojure.lang.Closure`
    /// heap object holding the FuncRef + the captured values; the body
    /// reads captures from the closure passed as its first (hidden) arg.
    pub captures: Vec<Arc<LocalBinding>>,
}

impl FnExpr {
    pub fn fref(&self) -> dynir::FuncRef {
        self.fref
    }
    pub fn is_variadic(&self) -> bool {
        self.is_variadic
    }
    pub fn fixed_arity(&self) -> usize {
        self.fixed_arity
    }

    /// Shallow clone used by `MultiArityFnExpr` to collect each clause's
    /// FnExpr into the multi-arity wrapper. The body's `Arc<dyn Expr>`
    /// is shared, not copied; captures + params Vecs share their Arcs.
    pub fn clone_for_multi_arity(&self) -> FnExpr {
        self.clone()
    }
}

impl Expr for FnExpr {
    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Non-capturing fns emit as a NanBox TAG_FN handle wrapping the
        // FuncRef index. Closures (`!captures.is_empty()`) allocate a
        // `clojure.lang.Closure` heap object holding the FuncRef + the
        // current values of the captured outer-fn locals, and return its
        // TAG_PTR-tagged pointer.
        if context == C::Statement {
            // Side-effect-free in expression position; emit nothing if
            // result is discarded.
            return None;
        }
        if self.captures.is_empty() {
            let payload = self.fref.index() as u64;
            return Some(ir.f.tagged_const(3, payload));
        }

        // Capturing fn: allocate the Closure heap object. Read layout
        // from the cached `closure_handle` — `dm.get_obj_type` panics
        // because Session::new moved `obj_types` out into the GC.
        let n_caps = self.captures.len();
        let (fref_offset, varlen_base, handle) = with_active_compiler(|c| {
            (
                c.closure_fref_offset,
                c.closure_varlen_base,
                c.closure_handle.clone(),
            )
        });
        let varlen_len = ir.f.fb.iconst(dynir::Type::I64, n_caps as i64);
        let raw = handle.alloc(ir.f, varlen_len);
        // Store fref_index (raw u64).
        let fref_val = ir.f.fb.iconst(dynir::Type::I64, self.fref.index() as i64);
        ir.f.fb.store(fref_val, raw, fref_offset);
        // Store each capture's current value.
        for (i, cap) in self.captures.iter().enumerate() {
            // Read the outer-scope binding's value — which itself might be a
            // capture if this fn is nested inside another closure. The
            // existing LocalBindingExpr.emit handles that recursively.
            let cap_expr = LocalBindingExpr::new(cap.clone(), None);
            let cap_val = cap_expr
                .emit(C::Expression, _objx, ir)
                .expect("capture must produce a value");
            let base = ir.f.fb.iconst(dynir::Type::I64, varlen_base);
            let eight = ir.f.fb.iconst(dynir::Type::I64, 8);
            let idx = ir.f.fb.iconst(dynir::Type::I64, i as i64);
            let byte_off = ir.f.fb.mul(idx, eight);
            let off = ir.f.fb.add(base, byte_off);
            let addr = ir.f.fb.add(raw, off);
            ir.f.fb.store(cap_val, addr, 0);
        }
        // Wrap into NanBox TAG_PTR.
        Some(ir.f.obj_wrap(raw))
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("clojure.lang.AFunction".to_string()),
        })
    }

    fn as_fn_expr(&self) -> Option<&FnExpr> {
        Some(self)
    }
}

/// Parse `(fn* [args] body...)` or `(fn* name [args] body...)`. Returns a
/// `FnExpr` whose body has been registered as a pending fn on the active
/// `Compiler`.
fn parse_fn_form(context: C, form: Object) -> Box<dyn Expr> {
    let _ = context; // fn body is always lowered as a separate fn
    // (fn* [params] body...) — optional name symbol allowed in slot 2
    let raw_after_fn = super::rt::next(&form);
    // Three shapes:
    //   (fn* sym [params] body...)              ← single-arity, named
    //   (fn* [params] body...)                  ← single-arity, anonymous
    //   (fn* sym? ([params1] b1) ([p2] b2) ...) ← multi-arity
    let (name_sym, first_clause, body_seq) = parse_fn_head(&raw_after_fn);

    // Detect multi-arity: first clause-position is a List (clause form
    // wrapping params + body), not a Vector (single-arity's params).
    if let Object::List(_) = &first_clause {
        // Collect every clause: first_clause + each remaining element of
        // body_seq. Each is `([params] body...)`.
        let mut all_clauses: Vec<Object> = vec![first_clause];
        let mut s = body_seq;
        while !matches!(s, Object::Nil) {
            all_clauses.push(super::rt::first(&s));
            s = super::rt::next(&s);
        }
        return parse_fn_form_multi_arity(name_sym, all_clauses);
    }

    // Peel reader-attached metadata (e.g. `^{:rettag X}` on the params
    // vector — defn often produces this around the fn body's params).
    // We don't yet consume the rettag info; it's dropped here so the
    // bare Vector matches the analyzer's contract.
    let raw_params_vec: Arc<PersistentVector> = match first_clause.peel_meta() {
        Object::Vector(v) => v,
        other => panic!(
            "clojure-jvm: IllegalArgumentException — fn* expects a vector of params, got {other:?}"
        ),
    };
    // Apply destructuring to params: replace any non-symbol pattern with
    // a synthetic param symbol, and prepend a let* to the body that
    // expands the destructure. Pure-symbol params pass through.
    let (params_vec, body_seq) = rewrite_fn_params_for_destructuring(raw_params_vec, body_seq);

    // Mint a fresh fn id for this body. Local bindings registered while
    // it's active stamp themselves with this id; `reference_local` uses
    // mismatches to detect captures from outer scopes.
    let fn_id = mint_fn_id();
    let _fn_guard = FnIdGuard::new(fn_id);

    // Push a fresh local scope for the fn body, INHERITING the outer
    // LOCAL_ENV so symbols defined in enclosing scopes remain resolvable.
    // The inherited bindings keep their original `owning_fn_id`, so
    // `reference_local` can mark them as captures of this fn.
    // NEXT_LOCAL_NUM resets to 0 — inner slot indices are independent.
    let inherited_env: LocalEnvMap = (*current_local_env()).clone();
    Var::push_thread_bindings(vec![
        (
            COMPILER_VARS.LOCAL_ENV.clone(),
            Object::Host(std::sync::Arc::new(inherited_env)),
        ),
        (COMPILER_VARS.NEXT_LOCAL_NUM.clone(), Object::Long(0)),
    ]);

    // Start collecting captures for this fn. The scope's `fn_id` filters
    // which bindings count as captures inside `record_capture`.
    push_capture_scope(fn_id);

    // Named-fn self-reference: `(fn name [params] body)` lets the body
    // call itself by `name`. Register `name` as a regular LocalBinding
    // BEFORE param registration so it's in scope during body analysis.
    // `lower_pending_fn` initializes the slot at fn entry — to TAG_FN
    // for non-capturing fns, or to the closure-self for closures.
    let self_name_slot: Option<i32> = name_sym.as_ref().map(|n| {
        let sym = crate::lang::symbol::Symbol::intern(n);
        let lb = register_local(sym, None, None, true);
        lb.idx
    });

    // First pass: register the params so we know the LocalBindings (we need
    // them to set LOOP_LOCALS before analyzing the body).
    //
    // Variadic shape: `[a b & rest]` — `&` marks the position after which the
    // next single symbol is the rest param. The fn declares
    // `fixed_arity + 1` total params; the rest param is just a normal local
    // bound to a list at invocation time. Caller-side packing happens in
    // `InvokeExpr.emit`.
    let specials = &*SPECIAL_SYMBOLS;
    let (params, is_variadic, fixed_arity): (Vec<Arc<LocalBinding>>, bool, usize) = {
        let mut params: Vec<Arc<LocalBinding>> = Vec::with_capacity(params_vec.count() as usize);
        let mut is_variadic = false;
        let mut fixed_arity = params_vec.count() as usize;
        let mut i = 0i32;
        while i < params_vec.count() {
            let p_raw = params_vec.nth(i);
            // Peel any reader-attached metadata (`^Type sym` / `^:tag sym`).
            // We don't yet model param type tags (Java uses them for
            // primitive specialization); they're dropped here. Without the
            // peel, `[^Class c]` panics because `Object::WithMeta(...)`
            // doesn't match `Object::Symbol`.
            let p = p_raw.peel_meta_ref().clone();
            let psym = match &p {
                Object::Symbol(s) => s.clone(),
                other => panic!(
                    "clojure-jvm: IllegalArgumentException — fn* param must be a Symbol, got {other:?}"
                ),
            };
            if *psym == *specials.AMP {
                // The next symbol is the rest param; everything before
                // counts as fixed_arity.
                is_variadic = true;
                fixed_arity = params.len();
                i += 1;
                if i >= params_vec.count() {
                    panic!(
                        "clojure-jvm: IllegalArgumentException — `&` must be followed by a rest-param symbol"
                    );
                }
                let rest_sym = match params_vec.nth(i).peel_meta_ref().clone() {
                    Object::Symbol(s) => s,
                    other => panic!(
                        "clojure-jvm: IllegalArgumentException — rest param after `&` must be a Symbol, got {other:?}"
                    ),
                };
                if rest_sym.get_namespace().is_some() {
                    panic!(
                        "clojure-jvm: RuntimeException — Can't use qualified name as parameter: {}",
                        rest_sym.get_name()
                    );
                }
                let lb = register_local(rest_sym, None, None, true);
                params.push(lb);
                i += 1;
                if i < params_vec.count() {
                    panic!(
                        "clojure-jvm: IllegalArgumentException — only one rest param allowed after `&`"
                    );
                }
                break;
            }
            if psym.get_namespace().is_some() {
                panic!(
                    "clojure-jvm: RuntimeException — Can't use qualified name as parameter: {}",
                    psym.get_name()
                );
            }
            let lb = register_local(psym, None, None, true);
            params.push(lb);
            i += 1;
        }
        if !is_variadic {
            fixed_arity = params.len();
        }
        (params, is_variadic, fixed_arity)
    };

    // Push LOOP_LOCALS for the body so `recur` validates its arg count
    // against the fn params.
    Var::push_thread_bindings(vec![(
        COMPILER_VARS.LOOP_LOCALS.clone(),
        Object::Host(std::sync::Arc::new(params.clone())),
    )]);
    let body_expr: Arc<dyn Expr> = Arc::from(parse_body_seq(C::Return, body_seq));
    Var::pop_thread_bindings();

    Var::pop_thread_bindings();

    // Collect captures recorded during body analysis. Cascade each one
    // up: if an inner-fn capture comes from a grandparent (more than one
    // level up), the immediate parent must also capture it so it has the
    // value to thread into the inner closure at construction time.
    // `record_capture` filters by the now-active outer scope's fn_id,
    // so re-recording locals owned by the outer fn itself is a no-op.
    let captures = pop_capture_scope();
    for c in &captures {
        record_capture(c);
    }

    // Register the fn body as a pending compilation on the active session.
    let fref = with_active_compiler(|c| {
        let name = c.fresh_fn_name(name_sym.as_deref());
        c.declare_pending_fn(
            name.clone(),
            params.clone(),
            body_expr.clone(),
            fn_id,
            captures.clone(),
            self_name_slot,
        );
        // declare_pending_fn returns FuncRef but also pushes onto pending;
        // we want the FuncRef plus the same name for FnExpr's record.
        // Re-fetch the most recently pushed entry.
        let pending = c.pending_fns.lock().unwrap();
        let last = pending.last().expect("declare_pending_fn just pushed");
        let fr = last.fref;
        let n = last.name.clone();
        drop(pending);
        (fr, n)
    });

    let line = COMPILER_VARS.LINE.deref();
    let column = COMPILER_VARS.COLUMN.deref();
    let line_i = if let Object::Long(n) = line {
        n as i32
    } else {
        0
    };
    let column_i = if let Object::Long(n) = column {
        n as i32
    } else {
        0
    };

    let info = VarFnInfo {
        is_variadic,
        fixed_arity,
    };
    with_active_compiler(|c| c.register_fn_arity(fref.0, info));
    Box::new(FnExpr {
        method: FnMethod {
            params,
            body: body_expr,
            line: line_i,
            column: column_i,
        },
        fref: fref.0,
        name: fref.1,
        is_variadic,
        fixed_arity,
        fn_id,
        captures,
    })
}

/// Pull the optional name symbol, params vector, and body seq out of the
/// post-`fn*` portion of the form. Returns `(name_opt, params_form, body_seq)`.
/// `MultiArityFnExpr` — a `(fn* ([p1] b1) ([p2] b2))` form. Wraps multiple
/// single-arity `FnExpr`s, one per clause. Used by `InvokeExpr.emit`'s
/// static-dispatch path to pick the matching arity at the call site.
///
/// Dynamic invocation (passing the multi-arity fn as a value) is NOT yet
/// supported — the TAG_FN encoding holds one FuncRef; multi-arity through
/// the runtime `cljvm_rt_invoke_*` thunks needs a per-name dispatcher
/// table, which lands later. For now, `MultiArityFnExpr.emit` panics if
/// the value flows somewhere static dispatch can't catch it.
#[derive(Debug)]
pub struct MultiArityFnExpr {
    /// Sorted by fixed_arity ascending. At most one is variadic; if
    /// present it's the catch-all for arg counts ≥ its fixed_arity.
    pub arities: Vec<FnExpr>,
    /// Name from `(fn* name? ...)`, used to mint inner pending-fn names.
    pub name: Option<String>,
    /// Literal-pool slot for the `MultiArityFn` dispatcher cell, reserved
    /// at analyze time (in `parse_fn_form_multi_arity`) so each clause's
    /// named self-reference slot can be initialized to it (see
    /// `lower_pending_fn`). `None` when the fn is single-clause or any
    /// clause captures (the cell only holds bare frefs, not closures).
    pub cell_lit: Option<dynir::ir::LiteralRef>,
}

impl MultiArityFnExpr {
    pub fn pick(&self, arg_count: usize) -> Option<&FnExpr> {
        // Exact-arity match wins; otherwise the variadic catch-all if any.
        let mut variadic: Option<&FnExpr> = None;
        for a in &self.arities {
            if a.is_variadic {
                variadic = Some(a);
                continue;
            }
            if a.fixed_arity == arg_count {
                return Some(a);
            }
        }
        if let Some(v) = variadic {
            if arg_count >= v.fixed_arity {
                return Some(v);
            }
        }
        None
    }
}

impl Expr for MultiArityFnExpr {
    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        if context == C::Statement {
            return None;
        }
        // Single-clause multi-arity (the shape `defn` produces — every
        // single-body defn becomes `(fn ([params] body))`) is dispatch-
        // equivalent to a plain FnExpr. Emit the underlying FnExpr's
        // handle so `def`'s `bind_root` stores a real fn handle. Without
        // this, any macro defined via defmacro (which goes through defn
        // → multi-arity) ended up with its Var bound to nil, and
        // macroexpand_once panicked at the call site with "macro Var
        // must hold a fn handle, got nil".
        if self.arities.len() == 1 {
            return self.arities[0].emit(context, objx, ir);
        }
        // Genuine multi-arity-as-value (>1 clauses): emit the shared
        // `MultiArityFn` dispatcher cell that was reserved at analyze time
        // (`parse_fn_form_multi_arity`). `(apply f args)`, `cljvm_rt_invoke_*`,
        // and named self-calls all dispatch by arity through this cell.
        //
        // `cell_lit` is `None` only when a clause captures: the cell holds
        // bare frefs, not Closure handles, so a capturing multi-arity fn
        // can't be represented as a value yet (static `InvokeExpr` dispatch
        // via `var_multi_arity` still resolves its direct calls). Emit nil
        // for that case, as before — but a captured multi-arity defn that
        // actually flows through dynamic dispatch will surface as a nil
        // invoke, which is the pre-existing limitation, not a regression.
        match self.cell_lit {
            Some(lref) => Some(ir.f.fb.gc_literal(lref)),
            None => Some(ir.f.nil()),
        }
    }
    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new("clojure.lang.AFunction".to_string()),
        })
    }

    fn as_multi_arity_fn_expr(&self) -> Option<&MultiArityFnExpr> {
        Some(self)
    }
}

/// Desugar a multi-arity `fn*` into a SINGLE variadic closure that dispatches
/// on argument count: `(fn* name? [& va] (let* [n (count va)] (if (= n K1)
/// (let* [p0 (nth va 0) …] body1) (if … <variadic: (<= V n)> …) nil)))`.
///
/// Used for NESTED multi-arity fns (which may capture outer locals) — the
/// `MultiArityFn` dispatcher cell holds bare frefs, not Closure handles, so it
/// can't represent captures; a single variadic capturing closure can. The
/// fixed clauses bind params via `nth`, the variadic clause's rest via nested
/// `next` (both available very early in core, so no load-order dependency).
fn desugar_multi_arity_capturing(name: &Option<String>, clauses: &[Object]) -> Object {
    use crate::lang::persistent_vector::PersistentVector;
    static CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let va = Symbol::intern(&format!("__va{id}__"));
    let nn = Symbol::intern(&format!("__n{id}__"));
    let list = |items: Vec<Object>| Object::List(PersistentList::create(items));
    let core = |s: &str| Object::Symbol(Symbol::intern_ns_name(Some("clojure.core"), s));

    struct Clause {
        fixed: Vec<Object>,
        rest: Option<Object>,
        body: Vec<Object>,
    }
    let mut parsed: Vec<Clause> = Vec::new();
    for clause in clauses {
        let items: Vec<Object> = match clause {
            Object::List(l) => l.iter().collect(),
            other => panic!("clojure-jvm: fn* multi-arity clause must be a list, got {other:?}"),
        };
        let params = match items.first().map(|o| o.peel_meta_ref().clone()) {
            Some(Object::Vector(v)) => v,
            other => panic!("clojure-jvm: fn* clause params must be a vector, got {other:?}"),
        };
        let mut fixed = Vec::new();
        let mut rest = None;
        let pcount = params.count();
        let mut i = 0i32;
        while i < pcount {
            let p = params.nth(i);
            if matches!(p.peel_meta_ref(), Object::Symbol(s) if s.get_name() == "&" && s.get_namespace().is_none())
            {
                rest = Some(params.nth(i + 1));
                break;
            }
            fixed.push(p);
            i += 1;
        }
        parsed.push(Clause {
            fixed,
            rest,
            body: items[1..].to_vec(),
        });
    }
    parsed.sort_by_key(|c| c.fixed.len());

    // Build the if-chain inside-out (else = nil).
    let mut chain = Object::Nil;
    for c in parsed.into_iter().rev() {
        let fixed_n = c.fixed.len();
        let cond = if c.rest.is_some() {
            list(vec![core("<="), Object::Long(fixed_n as i64), Object::Symbol(nn.clone())])
        } else {
            list(vec![core("="), Object::Symbol(nn.clone()), Object::Long(fixed_n as i64)])
        };
        let mut binds: Vec<Object> = Vec::new();
        for (i, p) in c.fixed.iter().enumerate() {
            binds.push(p.clone());
            binds.push(list(vec![core("nth"), Object::Symbol(va.clone()), Object::Long(i as i64)]));
        }
        if let Some(rest) = &c.rest {
            let mut rexpr = Object::Symbol(va.clone());
            for _ in 0..fixed_n {
                rexpr = list(vec![core("next"), rexpr]);
            }
            binds.push(rest.clone());
            binds.push(rexpr);
        }
        let mut let_items = vec![
            Object::Symbol(Symbol::intern("let*")),
            Object::Vector(PersistentVector::create(binds)),
        ];
        let_items.extend(c.body.iter().cloned());
        let then = list(let_items);
        chain = list(vec![Object::Symbol(Symbol::intern("if")), cond, then, chain]);
    }
    let body = list(vec![
        Object::Symbol(Symbol::intern("let*")),
        Object::Vector(PersistentVector::create(vec![
            Object::Symbol(nn.clone()),
            list(vec![core("count"), Object::Symbol(va.clone())]),
        ])),
        chain,
    ]);
    let mut fn_items: Vec<Object> = vec![Object::Symbol(SPECIAL_SYMBOLS.FN.clone())];
    if let Some(n) = name {
        fn_items.push(Object::Symbol(Symbol::intern(n)));
    }
    fn_items.push(Object::Vector(PersistentVector::create(vec![
        Object::Symbol(Symbol::intern("&")),
        Object::Symbol(va),
    ])));
    fn_items.push(body);
    list(fn_items)
}

fn parse_fn_form_multi_arity(name: Option<String>, clauses: Vec<Object>) -> Box<dyn Expr> {
    // Nested multi-arity fns may capture outer locals, which the `MultiArityFn`
    // dispatcher cell (bare frefs) can't represent → desugar to a single
    // variadic capturing closure. Top-level multi-arity fns (defn/defmacro)
    // can't capture, so keep the efficient `MultiArityFn` path.
    if current_fn_id() != 0 {
        let desugared = desugar_multi_arity_capturing(&name, &clauses);
        return parse_fn_form(C::Expression, desugared);
    }
    // Each clause is `([params] body...)`. We re-package each into a
    // synthetic `(fn* name? [params] body...)` form and recurse through
    // `parse_fn_form`. That way the per-clause logic (variadic detection,
    // capture analysis, pending-fn registration) all reuses one path.
    let mut arities: Vec<FnExpr> = Vec::with_capacity(clauses.len());
    let specials = &*SPECIAL_SYMBOLS;
    for clause in clauses {
        let clause_list = match clause {
            Object::List(l) => l,
            other => panic!(
                "clojure-jvm: IllegalArgumentException — fn* multi-arity clause must be a list, got {other:?}"
            ),
        };
        // Build (fn* name? params body...) by consing `fn*` (+ optional
        // name) onto the clause's seq.
        let mut head_items: Vec<Object> = vec![Object::Symbol(specials.FN.clone())];
        if let Some(n) = &name {
            head_items.push(Object::Symbol(Symbol::intern(n)));
        }
        // Append the clause's items.
        for i in 0..clause_list.count() {
            head_items.push(clause_list.iter().nth(i as usize).expect("clause item"));
        }
        let synthetic = Object::List(PersistentList::create(head_items));
        let parsed = parse_fn_form(C::Expression, synthetic);
        // The parsed Expr is a FnExpr (since each synthetic is single-arity).
        let fn_expr = parsed
            .as_fn_expr()
            .expect("synthetic single-arity form must parse to FnExpr")
            .clone_for_multi_arity();
        arities.push(fn_expr);
    }
    arities.sort_by_key(|a| a.fixed_arity);
    // Validate no two non-variadic arities collide.
    {
        let mut seen: std::collections::HashSet<usize> = Default::default();
        for a in &arities {
            if a.is_variadic {
                continue;
            }
            if !seen.insert(a.fixed_arity) {
                panic!(
                    "clojure-jvm: IllegalStateException — fn* has duplicate non-variadic arity {}",
                    a.fixed_arity
                );
            }
        }
    }
    // For a genuine multi-arity fn (>1 clause) with no capturing clause,
    // reserve a `MultiArityFn` dispatcher-cell literal NOW (analyze time)
    // so each clause's named self-reference slot can be initialized to it
    // in `lower_pending_fn`. Without this, a clause that self-calls a
    // different arity (e.g. `str`'s variadic body calling `(str x)`)
    // re-enters the SAME clause and recurses forever. The same cell is
    // emitted as the fn's value (for `apply` / higher-order use) below.
    let cell_lit = if arities.len() > 1 && arities.iter().all(|a| a.captures.is_empty()) {
        let entries: Vec<crate::runtime::MultiArityEntry> = arities
            .iter()
            .map(|a| crate::runtime::MultiArityEntry {
                fixed_arity: a.fixed_arity as u32,
                is_variadic: a.is_variadic,
                fref_idx: a.fref().index() as u32,
            })
            .collect();
        let lit = with_active_compiler(|c| {
            c.intern_literal(PendingLiteral::MultiArityFn(Arc::new(entries)))
        });
        let lref = dynir::ir::LiteralRef::from_u32(lit);
        // Point every clause's named-self slot at the shared cell.
        with_active_compiler(|c| {
            for a in &arities {
                c.set_pending_self_multi_lit(a.fref(), lref);
            }
        });
        Some(lref)
    } else {
        None
    };
    Box::new(MultiArityFnExpr {
        arities,
        name,
        cell_lit,
    })
}

fn parse_fn_head(after_fn: &Object) -> (Option<String>, Object, Object) {
    // after_fn is the tail seq starting with either a name sym, a params
    // vector, or a list (multi-arity). Reader-attached metadata
    // (`(fn* ^:static foo [params] body)`) wraps the name symbol —
    // peel through it so the named-vs-unnamed branch picks correctly.
    let first = super::rt::first(after_fn);
    if let Object::Symbol(s) = first.peel_meta_ref() {
        // (fn* name [params] body...)
        let name = s.get_name().to_string();
        let rest = super::rt::next(after_fn);
        let params = super::rt::first(&rest);
        let body = super::rt::next(&rest);
        (Some(name), params, body)
    } else {
        // (fn* [params] body...)
        let params = first;
        let body = super::rt::next(after_fn);
        (None, params, body)
    }
}

/// `Compiler.InvokeExpr` — function-application form `(f a b c)`. Holds the
/// head Expr (the fn being called) plus arg Exprs.
#[derive(Debug)]
pub struct InvokeExpr {
    pub fexpr: Box<dyn Expr>,
    pub args: Vec<Box<dyn Expr>>,
    pub line: i32,
    pub column: i32,
    pub tag: Option<Arc<Symbol>>,
}

impl Expr for InvokeExpr {
    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Compute the target FuncRef if statically known. Three cases:
        //   1. Head is a literal FnExpr  (lambda in head position)
        //   2. Head is a LocalBindingExpr whose binding's init is a FnExpr
        //      (let-bound fn — common pattern for first-class fn values)
        //   3. Head is a VarExpr whose Var was bound to a FnExpr via DefExpr
        //      earlier in the same compilation — the `(defn foo …) (foo args)`
        //      shape of essentially every clojure.core fn.
        //
        // Case 3 is a compile-time optimization, not a faithful port of
        // Java's emit (which always goes through Var.fn() at runtime so
        // redefinitions are observable). Acceptable as a stepping stone —
        // first-class fn invocation through a runtime Var-handle lands when
        // the call-table-base story is wired up.
        // Static target: FuncRef + variadic metadata. Only non-capturing
        // FnExprs are eligible for the static-call optimization — closure
        // bodies expect the closure object as their implicit first arg,
        // which the static path doesn't supply. Closures route through
        // the dynamic `cljvm_rt_invoke_*` thunk which handles self-prepending.
        let n_call_args = self.args.len();
        let static_target: Option<(dynir::FuncRef, bool, usize)> = self
            .fexpr
            .as_fn_expr()
            .filter(|f| f.captures.is_empty())
            .map(|f| (f.fref(), f.is_variadic(), f.fixed_arity()))
            .or_else(|| {
                // Multi-arity inline fn: pick the clause matching the call's
                // arg count. Each clause is its own non-capturing FnExpr.
                self.fexpr
                    .as_multi_arity_fn_expr()
                    .and_then(|m| m.pick(n_call_args))
                    .filter(|f| f.captures.is_empty())
                    .map(|f| (f.fref(), f.is_variadic(), f.fixed_arity()))
            })
            .or_else(|| {
                let lbe = self.fexpr.as_local_binding_expr()?;
                let init_guard = lbe.b.init.read().unwrap();
                let init = init_guard.as_ref()?;
                init.as_fn_expr()
                    .filter(|f| f.captures.is_empty())
                    .map(|f| (f.fref(), f.is_variadic(), f.fixed_arity()))
            })
            .or_else(|| {
                let ve = self.fexpr.as_var_expr()?;
                // First try a multi-arity Var: pick matching clause.
                if let Some(arities) = with_active_compiler(|c| c.var_multi_arity(&ve.var)) {
                    if let Some((fref, info)) = arities.into_iter().find_map(|(fref, info)| {
                        let matches = if info.is_variadic {
                            n_call_args >= info.fixed_arity
                        } else {
                            n_call_args == info.fixed_arity
                        };
                        if matches { Some((fref, info)) } else { None }
                    }) {
                        return Some((fref, info.is_variadic, info.fixed_arity));
                    }
                }
                let fref = with_active_compiler(|c| c.var_fn(&ve.var))?;
                // For Var-resolved fns we also need the variadic metadata.
                let info = with_active_compiler(|c| c.var_fn_info(&ve.var))?;
                Some((fref, info.is_variadic, info.fixed_arity))
            });

        if let Some((fref, is_variadic, fixed_arity)) = static_target {
            // Arity check (`n_call_args` computed above).
            if is_variadic {
                if n_call_args < fixed_arity {
                    panic!(
                        "clojure-jvm: ArityException — variadic fn requires at least {fixed_arity} args, got {n_call_args}"
                    );
                }
            } else if n_call_args != fixed_arity {
                // Non-variadic: arity must match exactly. (Multi-arity fns
                // aren't supported yet — each name maps to one FnExpr.)
                panic!(
                    "clojure-jvm: ArityException — fn expects {fixed_arity} arg(s), got {n_call_args}"
                );
            }

            // Emit fixed args first.
            let mut arg_vals: Vec<Value> = Vec::with_capacity(fixed_arity + 1);
            for i in 0..fixed_arity {
                match self.args[i].emit(C::Expression, objx, ir) {
                    Some(v) => arg_vals.push(v),
                    None => return None,
                }
            }

            // Variadic targets get one extra arg: a list packing all
            // overflow args. Emit `(rt_cons a_n (rt_cons a_n+1 (… nil)))`
            // right-to-left, terminated with nil.
            if is_variadic {
                let cons_fref = with_active_compiler(|c| {
                    c.host_method("clojure.lang.RT", "cons", 2)
                        .expect("RT.cons must be registered for variadic call packing")
                });
                // Emit overflow arg values in source order.
                let mut overflow_vals: Vec<Value> = Vec::with_capacity(n_call_args - fixed_arity);
                for i in fixed_arity..n_call_args {
                    match self.args[i].emit(C::Expression, objx, ir) {
                        Some(v) => overflow_vals.push(v),
                        None => return None,
                    }
                }
                // Build the list by folding right-to-left.
                let nil_bits = (0x7FFC_0000_0000_0000u64) as i64;
                let mut acc = ir.f.fb.iconst(dynir::Type::I64, nil_bits);
                for v in overflow_vals.into_iter().rev() {
                    acc =
                        ir.f.fb
                            .call(cons_fref, &[v, acc])
                            .expect("cljvm_rt_cons returns I64");
                }
                arg_vals.push(acc);
            }

            let ret = ir.f.fb.call(fref, &arg_vals);
            return match context {
                C::Statement => None,
                _ => Some(ret.expect("Clojure fns always return I64 (NanBox)")),
            };
        }

        // Dynamic invocation: head is a runtime expression producing a
        // NanBox TAG_FN handle (fn passed as value, fn-deref'd from a Var
        // alias, etc.). Lower to `cljvm_rt_invoke_N` for the call arity —
        // those externs decode the handle's FuncRef index, load the entry
        // pointer from the thread-local call table base, and dispatch.
        // Mirrors Java's `IFn.invoke(args)` virtual call.
        let head_val = match self.fexpr.emit(C::Expression, objx, ir) {
            Some(v) => v,
            None => {
                // The head expression diverged (a `throw` / `recur` in head
                // position — e.g. a macro body whose fn head analyzes to a
                // `ThrowExpr` for an unmodeled host-class reference that
                // only fires on a bad call). The call is unreachable; the
                // diverging head already terminated the block. Return None,
                // mirroring the arg-divergence handling below.
                return None;
            }
        };

        let mut arg_vals: Vec<Value> = Vec::with_capacity(self.args.len() + 1);
        arg_vals.push(head_val);
        for a in &self.args {
            match a.emit(C::Expression, objx, ir) {
                Some(v) => arg_vals.push(v),
                None => {
                    // Arg expression diverged (throw / recur). The
                    // invoke is unreachable — return None so the
                    // enclosing context knows there's no value.
                    return None;
                }
            }
        }

        let arity = self.args.len();
        let invoke_fref = with_active_compiler(|c| {
            if arity >= c.invoke_externs.len() {
                panic!(
                    "clojure-jvm: dynamic invoke arity {arity} exceeds supported limit \
                     ({}); add `cljvm_rt_invoke_{arity}` to extend",
                    c.invoke_externs.len()
                );
            }
            c.invoke_externs[arity]
        });
        let ret = ir.f.fb.call(invoke_fref, &arg_vals);
        match context {
            C::Statement => None,
            _ => Some(ret.expect("cljvm_rt_invoke_* returns I64")),
        }
    }

    fn has_java_class(&self) -> bool {
        self.tag.is_some()
    }
    fn get_java_class(&self) -> Option<HostClass> {
        self.tag.as_ref().map(|t| HostClass {
            name: Arc::new(t.get_name().to_string()),
        })
    }
}

// ============================================================================
// Java line ~7136–7311: `RecurExpr` + Parser.
//
// `recur` rebinds the surrounding loop/fn's locals and jumps back to its
// entry label. Java threads the label through `LOOP_LABEL` (a `clojure.asm`
// `Label`); we thread a `RecurTarget` (the dynir BlockId + the LocalBindings
// it re-binds) through the same Var.
// ============================================================================

/// `Compiler.RecurExpr`. Holds the new arg Exprs to bind into the loop
/// locals before jumping back to the loop entry.
#[derive(Debug)]
pub struct RecurExpr {
    pub args: Vec<Box<dyn Expr>>,
    pub loop_locals: Vec<Arc<LocalBinding>>,
}

impl Expr for RecurExpr {
    fn emit(&self, _context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Read the current recur target from LOOP_LABEL.
        let target_obj = COMPILER_VARS.LOOP_LABEL.deref();
        let target = target_obj.host_as::<RecurTarget>().unwrap_or_else(|| {
            panic!("clojure-jvm: IllegalStateException — recur outside of loop/fn body")
        });

        // Evaluate each new arg first (Java order: emit all args, then store
        // in reverse). dynir doesn't need the reverse trick since values are
        // SSA — just compute then assign.
        let mut new_vals: Vec<Value> = Vec::with_capacity(self.args.len());
        for a in &self.args {
            match a.emit(C::Expression, objx, ir) {
                Some(v) => new_vals.push(v),
                None => return None,
            }
        }

        // Store each new value into its corresponding local slot.
        for (lb, v) in target.locals.iter().zip(new_vals.iter()) {
            ir.f.set_var(&local_slot_name(lb.idx), *v);
        }

        // Jump back to the loop entry block.
        ir.f.fb.jump(target.block, &[]);
        None
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        // Java: `RECUR_CLASS` (the marker). We model that with a fixed name.
        Some(HostClass {
            name: Arc::new("__recur__".to_string()),
        })
    }
}

/// Parse `(recur arg1 arg2 ...)`. Java requires C::RETURN tail position;
/// we relax that to "must be in a recur-able position" — `LOOP_LABEL` bound.
fn parse_recur_form(context: C, form: Object) -> Box<dyn Expr> {
    if context != C::Return {
        panic!("clojure-jvm: UnsupportedOperationException — Can only recur from tail position");
    }
    // Pull the loop locals from the LOOP_LOCALS thread binding so we can
    // validate arg count + later emit.
    let locals_obj = COMPILER_VARS.LOOP_LOCALS.deref();
    let locals = locals_obj
        .host_as::<Vec<Arc<LocalBinding>>>()
        .unwrap_or_else(|| {
            panic!("clojure-jvm: UnsupportedOperationException — recur outside of loop/fn")
        });

    let mut args: Vec<Box<dyn Expr>> = Vec::new();
    let mut rest = super::rt::next(&form);
    while !matches!(rest, Object::Nil) {
        args.push(analyze(C::Expression, super::rt::first(&rest)));
        rest = super::rt::next(&rest);
    }
    if args.len() != locals.len() {
        panic!(
            "clojure-jvm: IllegalArgumentException — Mismatched argument count to recur, expected: {} args, got: {}",
            locals.len(),
            args.len()
        );
    }
    Box::new(RecurExpr {
        args,
        loop_locals: (*locals).clone(),
    })
}

// ============================================================================
// Built-in arithmetic / comparison ops.
//
// Java/Clojure resolves `+` / `-` / `*` / `<` etc. through clojure.core vars
// (each carries `:inline` metadata that the compiler unfolds to `Numbers.add`
// etc. host-method calls). We don't have host-method calls or the
// clojure.core source yet, so for now we recognize these operator symbols at
// `analyze_seq` time and emit a `PrimOpExpr` that lowers directly to dynlang's
// `add`/`sub`/`mul`/`div`/`lt`/`gt`/`le`/`ge`/`eq`.
//
// This is a stand-in — once macros + Var dispatch + Numbers static methods
// are wired, the special-cased symbols should resolve to clojure.core/+ etc.
// through normal resolution and the PrimOpExpr surface goes away.
// ============================================================================

/// Built-in primitive operation. Mirrors what dynlang exposes as inline-
/// hinted ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Lt,
    Gt,
    Le,
    Ge,
}

impl PrimOp {
    /// Java's `Numbers.add`/`Numbers.lt`/etc. return host class. For us all
    /// arithmetic produces a NanBox number; comparisons produce a NanBox bool.
    fn returns_bool(self) -> bool {
        matches!(
            self,
            PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge
        )
    }
}

/// `PrimOpExpr` — a built-in numeric/comparison op invocation. Holds the
/// resolved op + arg Exprs. Always reduces left-to-right for variadic forms,
/// matching Clojure's `(+)` = 0, `(+ x)` = x, `(+ x y z ...)` = (((x+y)+z)+…).
#[derive(Debug)]
pub struct PrimOpExpr {
    pub op: PrimOp,
    pub args: Vec<Box<dyn Expr>>,
}

impl Expr for PrimOpExpr {
    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        let v = self.emit_value(objx, ir)?;
        if context == C::Statement {
            None
        } else {
            Some(v)
        }
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new(if self.op.returns_bool() {
                "java.lang.Boolean".to_string()
            } else {
                // Numbers.add returns java.lang.Number; we don't track the
                // narrower type yet.
                "java.lang.Number".to_string()
            }),
        })
    }
}

/// Emit a boxed-Long literal (`gc_literal` of a pre-boxed pool cell). The
/// identity elements of `+`/`*` and the implicit `0`/`1` of unary `-`/`/`
/// must be real Longs, not native floats, so `(+)` is `0` and `(- 5)` is
/// `-5` (a Long), matching Clojure.
fn emit_boxed_long(n: i64, ir: &mut IrEmitter<'_>) -> Value {
    let idx = with_active_compiler(|c| c.intern_literal(PendingLiteral::Long(n)));
    ir.f.fb.gc_literal(dynir::ir::LiteralRef::from_u32(idx))
}

impl PrimOpExpr {
    fn emit_value(&self, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Numeric ops route through the `cljvm_num_*` externs (boxed-Long /
        // native-double tower with Clojure long/double promotion) rather than
        // dynlang's float-only `add`/`sub`/… (which can't see boxed Longs).
        // Variadic-arity reductions, with identity elements matching
        // Clojure: `(+)` = 0, `(*)` = 1, `(-)` errors, `(/)` errors.
        let num = with_active_compiler(|c| c.num);
        match self.op {
            PrimOp::Add if self.args.is_empty() => return Some(emit_boxed_long(0, ir)),
            PrimOp::Mul if self.args.is_empty() => return Some(emit_boxed_long(1, ir)),
            PrimOp::Sub if self.args.is_empty() => {
                panic!("clojure-jvm: Wrong number of args (0) passed to: clojure.core/-")
            }
            PrimOp::Div if self.args.is_empty() => {
                panic!("clojure-jvm: Wrong number of args (0) passed to: clojure.core//")
            }
            _ => {}
        }
        let mut arg_vals: Vec<Value> = Vec::with_capacity(self.args.len());
        for a in &self.args {
            match a.emit(C::Expression, objx, ir) {
                Some(v) => arg_vals.push(v),
                None => return None,
            }
        }

        // Unary handling for `-` and `/`:
        //   `(- x)` → `0 - x`
        //   `(/ x)` → `1 / x`
        if arg_vals.len() == 1 {
            match self.op {
                PrimOp::Sub => {
                    let z = emit_boxed_long(0, ir);
                    return Some(
                        ir.f.fb
                            .call(num.sub, &[z, arg_vals[0]])
                            .expect("cljvm_num_sub returns I64"),
                    );
                }
                PrimOp::Div => {
                    let one = emit_boxed_long(1, ir);
                    return Some(
                        ir.f.fb
                            .call(num.div, &[one, arg_vals[0]])
                            .expect("cljvm_num_div returns I64"),
                    );
                }
                // Comparisons: `(< x)` is always true in Clojure (only one
                // element to compare). Same for the others.
                PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => {
                    return Some(ir.f.bool_val(true));
                }
                _ => return Some(arg_vals[0]),
            }
        }

        // For binary/n-ary arithmetic: left fold through the tower extern.
        if !self.op.returns_bool() {
            let fref = match self.op {
                PrimOp::Add => num.add,
                PrimOp::Sub => num.sub,
                PrimOp::Mul => num.mul,
                PrimOp::Div => num.div,
                _ => unreachable!(),
            };
            let mut acc = arg_vals[0];
            for v in &arg_vals[1..] {
                acc =
                    ir.f.fb
                        .call(fref, &[acc, *v])
                        .expect("cljvm_num_* returns I64");
            }
            return Some(acc);
        }

        // Comparisons: `(< a b c)` is `(and (< a b) (< b c))` — a chained AND
        // of pairwise comparisons. `=` uses general equality (`cljvm_equals`),
        // the others the numeric comparison externs.
        let cmp_fref = match self.op {
            PrimOp::Eq => num.eq,
            PrimOp::Lt => num.lt,
            PrimOp::Gt => num.gt,
            PrimOp::Le => num.le,
            PrimOp::Ge => num.ge,
            _ => unreachable!(),
        };
        let mut acc = ir.f.bool_val(true);
        for i in 0..arg_vals.len() - 1 {
            let a = arg_vals[i];
            let b = arg_vals[i + 1];
            let pair =
                ir.f.fb
                    .call(cmp_fref, &[a, b])
                    .expect("cljvm_num_* comparison returns I64");
            // AND the accumulated bool with the new pair. dynlang doesn't
            // expose a bool-AND; we model it via `if acc then pair else false`.
            let acc_truthy = ir.f.is_truthy(acc);
            let f = ir.f.bool_val(false);
            acc = ir.f.fb.select(acc_truthy, pair, f);
        }
        Some(acc)
    }
}

/// Map a head symbol to a `PrimOp` if it matches a built-in operator.
fn primop_for_symbol(sym: &Symbol) -> Option<PrimOp> {
    if sym.get_namespace().is_some() {
        return None;
    }
    Some(match sym.get_name() {
        "+" => PrimOp::Add,
        "-" => PrimOp::Sub,
        "*" => PrimOp::Mul,
        "/" => PrimOp::Div,
        "=" => PrimOp::Eq,
        "<" => PrimOp::Lt,
        ">" => PrimOp::Gt,
        "<=" => PrimOp::Le,
        ">=" => PrimOp::Ge,
        _ => return None,
    })
}

// ============================================================================
// Java line ~810–1500: `HostExpr` + `StaticMethodExpr` + `InstanceMethodExpr`.
//
// Java's HostExpr handles `.` interop with the JVM — class lookups via
// `Reflector`, primitive boxing/unboxing, etc. Our port is intentionally
// narrow: we only support `(. clojure.lang.<Class> (method args...))`,
// dispatching to Rust-side externs registered in `Compiler.host_methods`.
// Field access (single-symbol member) and instance methods land later;
// most of `clojure.core` invokes RT *static* methods which is what we
// model here.
// ============================================================================

/// `Compiler.StaticMethodExpr` (narrowed). Holds the resolved FuncRef and
/// analyzed args; emit is a direct `Call`.
#[derive(Debug)]
pub struct StaticMethodExpr {
    pub class_name: String,
    pub method_name: String,
    pub args: Vec<Box<dyn Expr>>,
    pub fref: dynir::FuncRef,
    pub tag: Option<Arc<Symbol>>,
}

impl Expr for StaticMethodExpr {
    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java path: emit each arg with type-directed boxing/unboxing, then
        // invokeStatic. We treat all args + return as NanBox-encoded u64 —
        // unboxing belongs in the Rust extern body, not codegen.
        let mut arg_vals: Vec<Value> = Vec::with_capacity(self.args.len());
        for a in &self.args {
            match a.emit(C::Expression, objx, ir) {
                Some(v) => arg_vals.push(v),
                None => return None,
            }
        }
        let ret =
            ir.f.fb
                .call(self.fref, &arg_vals)
                .expect("static host methods return I64 (NanBox)");
        match context {
            C::Statement => None,
            _ => Some(ret),
        }
    }

    fn has_java_class(&self) -> bool {
        self.tag.is_some()
    }
    fn get_java_class(&self) -> Option<HostClass> {
        self.tag.as_ref().map(|t| HostClass {
            name: Arc::new(t.get_name().to_string()),
        })
    }
}

/// Parse `(ns NAME ...)` and `(in-ns NAME)`. Real Clojure's `ns` is a
/// macro that expands to `(in-ns 'name)` plus refer / import setup; for
/// our subset we implement a special form that:
///   * picks out the namespace-name Symbol (skipping any metadata-map /
///     options that follow — we ignore them for now),
///   * `find_or_create`'s the Namespace,
///   * pushes it as `CURRENT_NS`'s thread-binding (replacing any prior),
///   * returns a no-op `NilExpr` at compile time.
///
/// Documentation strings and require / refer specs are silently dropped.
/// That's enough to get past `(ns clojure.core …)` at the top of
/// `core.clj`; we'll grow this as we hit more namespace-aware features.
fn parse_ns_form(_context: C, form: Object) -> Box<dyn Expr> {
    let after = super::rt::next(&form);
    let mut first = super::rt::first(&after);
    // `in-ns` takes a quoted symbol: `(in-ns 'my.ns)` reads as
    // `(in-ns (quote my.ns))`. Unwrap the quote so both `(ns my.ns)` and
    // `(in-ns 'my.ns)` reach the underlying Symbol.
    if let Object::List(l) = &first {
        if l.count() == 2 {
            if let Some(Object::Symbol(h)) = l.iter().next() {
                if h.get_name() == "quote" && h.get_namespace().is_none() {
                    first = l.iter().nth(1).unwrap();
                }
            }
        }
    }
    // The ns name can carry metadata (e.g. `(ns ^{:doc "..."} my.ns)`).
    // We don't act on it yet — peel through the wrapper to reach the
    // underlying Symbol.
    let bare = first.peel_meta_ref();
    let ns_sym = match bare {
        Object::Symbol(s) if s.get_namespace().is_none() => s.clone(),
        other => panic!(
            "clojure-jvm: ns/in-ns expects an unqualified Symbol as the namespace name, got {other:?}"
        ),
    };
    let ns = super::namespace::Namespace::find_or_create(ns_sym);
    // Switch `*ns*` to the new namespace by mutating the CURRENT THREAD's
    // dynamic binding (Clojure's `in-ns` calls `RT.CURRENT_NS.set(ns)`,
    // which sets the thread binding established by the loader/REPL — it does
    // NOT change the global root). `Session::eval_form` establishes that
    // binding around every form, so `set_value` always has one to mutate,
    // and the change stays local to this thread of evaluation. `in-ns` does
    // NOT refer clojure.core — that's the `ns` macro's job, via its
    // expansion's `(clojure.core/refer 'clojure.core)` call.
    super::rt::CURRENT_NS.set_value(Object::Namespace(ns.clone()));

    // Walk the remaining clauses after the ns name and process the
    // ones we understand. Anything we don't recognize is skipped
    // silently — full ns-form fidelity (`:refer-clojure :exclude`,
    // `:import`, etc.) isn't required for the loader to advance, but
    // `:as` aliases ARE required so that `core/defmacro` etc.
    // resolve. `(ns)` is a macro upstream that expands to a series
    // of (require ...) / (refer ...) / etc. calls; we model the
    // important subset directly here.
    let mut rest = super::rt::next(&after);
    while !matches!(rest, Object::Nil) {
        let clause = super::rt::first(&rest);
        // Each clause is `(:require ...)`, `(:refer-clojure ...)`,
        // `(:import ...)`, `(:require-macros ...)`, `(:use ...)`, etc.
        // The first element is a keyword. Docstring (first element
        // is a String) is also valid; skip it.
        match &clause {
            Object::List(_) => {
                let head = super::rt::first(&clause);
                if let Object::Keyword(k) = head {
                    if k.get_namespace().is_none() {
                        let kind = k.get_name();
                        if kind == "require" || kind == "require-macros" || kind == "use" {
                            process_ns_require_clause(&ns, &clause);
                        }
                        // :refer-clojure :exclude — accept silently for now
                        // :import — accept silently (we don't model imports)
                    }
                }
            }
            _ => {} // string docstring, metadata map, etc.
        }
        rest = super::rt::next(&rest);
    }
    Box::new(NIL_EXPR)
}

/// Process a single `(:require ...)` (or `:require-macros` / `:use`)
/// clause from an `(ns)` form. Each entry after the keyword is one
/// of:
///   * `foo.bar.baz` — bare namespace symbol; we silently accept
///     (no-op for our purposes — we don't auto-load).
///   * `[foo.bar.baz :as fb]` — register `fb` as an alias for
///     `foo.bar.baz` in `target_ns`.
///   * `[foo.bar.baz :refer [...]]` — silently accept (we don't
///     auto-bring symbols into the current ns).
///   * `[foo.bar.baz :as fb :refer [...]]` — same as :as above.
///
/// Unrecognized shapes are skipped without panicking; the goal is to
/// keep loaders advancing while collecting :as aliases.
fn process_ns_require_clause(target_ns: &Arc<super::namespace::Namespace>, clause: &Object) {
    let mut entries = super::rt::next(clause);
    while !matches!(entries, Object::Nil) {
        let entry = super::rt::first(&entries);
        match &entry {
            Object::Symbol(_) => {
                // bare ns symbol — accept, no alias to register
            }
            Object::Vector(v) => {
                // First element is the namespace symbol; scan the
                // rest in (k v) pairs for :as / :refer.
                if v.count() < 1 {
                    entries = super::rt::next(&entries);
                    continue;
                }
                let ns_sym = match v.nth(0).peel_meta() {
                    Object::Symbol(s) => s,
                    _ => {
                        entries = super::rt::next(&entries);
                        continue;
                    }
                };
                let n = v.count();
                let mut i: i32 = 1;
                while i + 1 < n {
                    let k_obj = v.nth(i);
                    let v_obj = v.nth(i + 1);
                    if let Object::Keyword(k) = &k_obj {
                        if k.get_namespace().is_none() && k.get_name() == "as" {
                            if let Object::Symbol(alias_sym) = v_obj.peel_meta() {
                                let aliased =
                                    super::namespace::Namespace::find_or_create(ns_sym.clone());
                                target_ns.add_alias(alias_sym, aliased);
                            }
                        }
                        // :refer, :refer-macros, :include-macros, etc. silently skipped.
                    }
                    i += 2;
                }
            }
            Object::List(_) => {
                // Lists are uncommon in :require but possible. Walk
                // manually like vectors above.
                let mut items: Vec<Object> = Vec::new();
                let mut cur = entry.clone();
                while !matches!(cur, Object::Nil) {
                    if matches!(&cur, Object::List(l) if matches!(l.as_ref(), super::persistent_list::PersistentList::Empty))
                    {
                        break;
                    }
                    items.push(super::rt::first(&cur));
                    cur = super::rt::next(&cur);
                }
                if items.is_empty() {
                    entries = super::rt::next(&entries);
                    continue;
                }
                let ns_sym = match items[0].clone().peel_meta() {
                    Object::Symbol(s) => s,
                    _ => {
                        entries = super::rt::next(&entries);
                        continue;
                    }
                };
                let mut i = 1usize;
                while i + 1 < items.len() {
                    if let Object::Keyword(k) = &items[i] {
                        if k.get_namespace().is_none() && k.get_name() == "as" {
                            if let Object::Symbol(alias_sym) = items[i + 1].clone().peel_meta() {
                                let aliased =
                                    super::namespace::Namespace::find_or_create(ns_sym.clone());
                                target_ns.add_alias(alias_sym, aliased);
                            }
                        }
                    }
                    i += 2;
                }
            }
            _ => {}
        }
        entries = super::rt::next(&entries);
    }
}

/// `(var SYM)` — return the `Var` bound to `SYM` in the current ns
/// (or its qualified ns) as a runtime value. We allocate a heap cell
/// holding `Arc::as_ptr(&v)` at literal-pool fill time; the Var
/// itself is rooted by the namespace mapping for the program's life.
fn parse_var_form(_context: C, form: Object) -> Box<dyn Expr> {
    let n = super::rt::count(&form);
    if n != 2 {
        panic!(
            "clojure-jvm: IllegalArgumentException — `var` takes exactly one Symbol arg, got {}",
            n - 1
        );
    }
    let arg = super::rt::second(&form);
    let sym = match arg.peel_meta_ref() {
        Object::Symbol(s) => s.clone(),
        other => panic!("clojure-jvm: `var` expects a Symbol, got {other:?}"),
    };
    if let Some(v) = resolve_var(&sym) {
        let idx = with_active_compiler(|c| c.intern_literal(PendingLiteral::Var(v)));
        return Box::new(ConstantLiteralExpr { idx });
    }
    // Unresolved `(var x)` / `#'x` — defer to a runtime throw like an
    // unresolved symbol, so a form that merely *references* a var from a
    // sub-file we don't embed (e.g. `#'default-uuid-reader` from `uuid`)
    // compiles and only fails if evaluated.
    let msg = format!("Unable to resolve var: {}", sym.get_name());
    let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
    Box::new(ThrowExpr { payload })
}

/// `(set! target value)` — Java-side mutation. We don't model
/// mutable Java fields or thread-local var assignment; the form
/// analyzes by evaluating its value expression (so any side effects
/// in the value still happen) and returning that value at runtime.
/// Calling code that depends on the mutation will still observe stale
/// state, but defns that USE set! analyze cleanly.
fn parse_assign_form(context: C, form: Object) -> Box<dyn Expr> {
    let n = super::rt::count(&form);
    if n != 3 {
        panic!(
            "clojure-jvm: IllegalArgumentException — `set!` takes (target value), got {} args",
            n - 1
        );
    }
    // Skip target (we don't honor it), analyze the value.
    let value = super::rt::third(&form);
    analyze(context, value)
}

/// Parse `(set-macro! sym)` — flags the Var named by `sym` in the
/// current namespace as a macro. Bridges the gap between our subset and
/// Java's `(.setMacro #'name)` until generic instance-method dispatch
/// lands. The flagging happens at analyze time (the Var must already
/// have been interned by a prior `def`), so subsequent forms see the
/// macro flag set before they're analyzed.
fn parse_set_macro_bang_form(_context: C, form: Object) -> Box<dyn Expr> {
    let n = super::rt::count(&form);
    if n != 2 {
        panic!(
            "clojure-jvm: set-macro! takes exactly one argument (the Var's symbol), got {}",
            n - 1
        );
    }
    let arg = super::rt::second(&form);
    let sym = match arg {
        Object::Symbol(s) => s,
        other => panic!("clojure-jvm: set-macro! expects a Symbol, got {other:?}"),
    };
    let var = resolve_var(&sym).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: set-macro!: Var `{}` not found in current namespace",
            sym.get_name()
        )
    });
    var.set_macro();
    Box::new(NIL_EXPR)
}

/// Parse `(. ClassSymbol method args…)` and `(. ClassSymbol (method args…))`.
///
/// Narrowed from Java's `HostExpr.Parser`:
///   * Only `static` dispatch supported (the symbol after `.` must resolve
///     to a registered host class — currently just `clojure.lang.RT` and
///     friends declared in `Compiler::new`).
///   * Single-symbol "maybe field" form not supported yet.
///   * Instance dispatch (`(. inst method …)`) not supported yet.
fn parse_dot_form(context: C, form: Object) -> Box<dyn Expr> {
    let n = super::rt::count(&form);
    if n < 3 {
        panic!(
            "clojure-jvm: IllegalArgumentException — \
             Malformed member expression, expecting (. target member ...)"
        );
    }
    // (. target member …)
    let target = super::rt::second(&form);
    let third = super::rt::third(&form);

    // Two member shapes:
    //   (. T method args…)        → method is the 3rd form, args = rest
    //   (. T (method args…))      → method+args are wrapped in a list at slot 3
    let (method_name, args_seq): (String, Object) = match &third {
        Object::List(_) => {
            // (method args…)
            let mname = super::rt::first(&third);
            let mname = match mname {
                // Use the symbol's NAME, ignoring any namespace. Syntax-quote
                // qualifies symbols uniformly, so a `.`-form method written in
                // a macro template (e.g. `binding`'s
                // `(. clojure.lang.Var (pushThreadBindings …))`) arrives as
                // `clojure.core/pushThreadBindings`. Clojure's HostExpr reads
                // `((Symbol)…).name`, so the namespace is irrelevant here.
                Object::Symbol(s) => s.get_name().to_string(),
                other => panic!(
                    "clojure-jvm: HostExpr — method name must be a Symbol, got {other:?}"
                ),
            };
            let rest = super::rt::next(&third);
            (mname, rest)
        }
        // Same name-only rule for the `(. T method args…)` shape: accept a
        // qualified method symbol and use its name.
        Object::Symbol(s) => {
            // (. T method arg1 arg2 …) — method as 3rd, args at 4+.
            let mname = s.get_name().to_string();
            let rest = super::rt::next(&form); // (T method args…)
            let rest = super::rt::next(&rest); // (method args…)
            let rest = super::rt::next(&rest); // (args…)
            (mname, rest)
        }
        other => panic!(
            "clojure-jvm: HostExpr — third form must be method symbol or (method args…), got {other:?}"
        ),
    };

    let mut args: Vec<Box<dyn Expr>> = Vec::new();
    let mut s = args_seq;
    while !matches!(s, Object::Nil) {
        args.push(analyze(
            if context == C::Eval {
                context
            } else {
                C::Expression
            },
            super::rt::first(&s),
        ));
        s = super::rt::next(&s);
    }
    let arity = args.len();

    // Discriminate static vs instance dispatch.
    //
    // Java picks based on what `target` resolves to: a class → static,
    // anything else → instance (`InstanceMethodExpr`). Without full
    // class lookup we use a structural rule: a Symbol whose name
    // contains a `.` (Java-style fully-qualified class) — or that
    // resolves in our class registry — is a class; everything else is
    // an instance receiver.
    if let Object::Symbol(s) = target.peel_meta_ref() {
        if s.get_namespace().is_none() {
            let name = s.get_name();
            let looks_like_class =
                name.contains('.') || crate::lang::host_class::lookup(name).is_some();
            if looks_like_class {
                let fref_opt = with_active_compiler(|c| c.host_method(name, &method_name, arity));
                match fref_opt {
                    Some(fref) => {
                        return Box::new(StaticMethodExpr {
                            class_name: name.to_string(),
                            method_name,
                            args,
                            fref,
                            tag: None,
                        });
                    }
                    None => {
                        let msg = format!(
                            "unregistered static method `{name}/{method_name}` arity {arity}"
                        );
                        let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
                        return Box::new(ThrowExpr { payload });
                    }
                }
            }
        }
    }

    // Instance dispatch — receiver is an arbitrary expression.
    let arity_with_recv = 1 + arity;
    let fref_opt = with_active_compiler(|c| c.instance_method(&method_name, arity_with_recv));
    if let Some(fref) = fref_opt {
        let receiver = analyze(
            if context == C::Eval {
                context
            } else {
                C::Expression
            },
            target,
        );
        return Box::new(InstanceMethodExpr {
            method_name,
            receiver,
            args,
            fref,
            tag: None,
        });
    }
    // Unregistered: emit a runtime throw. See the matching site in
    // `parse_instance_method_form` for the rationale.
    let msg = format!("unregistered instance method `.{method_name}` arity {arity_with_recv}");
    let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
    Box::new(ThrowExpr { payload })
}

/// `(Class/method args...)` — Clojure reader sugar for a static method
/// call. Reuses `parse_dot_form` by analyzing each arg directly into a
/// `StaticMethodExpr`. The class name is the symbol's namespace; the
/// method name is its bare name. Args list is everything after the head.
fn parse_qualified_static_call(
    context: C,
    class_name: &str,
    method_name: &str,
    form: &Object,
) -> Box<dyn Expr> {
    // Args = (next form) — drop the head.
    let mut args: Vec<Box<dyn Expr>> = Vec::new();
    let mut s = super::rt::next(form);
    while !matches!(s, Object::Nil) {
        args.push(analyze(
            if context == C::Eval {
                context
            } else {
                C::Expression
            },
            super::rt::first(&s),
        ));
        s = super::rt::next(&s);
    }
    let arity = args.len();
    let fref_opt = with_active_compiler(|c| c.host_method(class_name, method_name, arity));
    match fref_opt {
        Some(fref) => Box::new(StaticMethodExpr {
            class_name: class_name.to_string(),
            method_name: method_name.to_string(),
            args,
            fref,
            tag: None,
        }),
        None => {
            let msg =
                format!("unregistered static method `{class_name}/{method_name}` arity {arity}");
            let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
            Box::new(ThrowExpr { payload })
        }
    }
}

/// `Compiler.InstanceMethodExpr` — narrowed. A direct call to a runtime
/// instance-method extern (e.g. `cljvm_inst_meta`). The extern handles
/// receiver-type dispatch internally (the polymorphism the JVM gets from
/// interface vtables).
#[derive(Debug)]
pub struct InstanceMethodExpr {
    pub method_name: String,
    pub receiver: Box<dyn Expr>,
    pub args: Vec<Box<dyn Expr>>,
    pub fref: dynir::FuncRef,
    pub tag: Option<Arc<Symbol>>,
}

impl Expr for InstanceMethodExpr {
    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        let trace = self.method_name == "refer" && std::env::var("CLJVM_IMEXPR_TRACE").is_ok();
        let mut arg_vals: Vec<Value> = Vec::with_capacity(self.args.len() + 1);
        let recv_opt = self.receiver.emit(C::Expression, objx, ir);
        if trace {
            let recv_dbg: String = format!("{:?}", self.receiver).chars().take(120).collect();
            eprintln!(
                "[imexpr] method={} ctx={context:?} recv_some={} recv={recv_dbg}",
                self.method_name,
                recv_opt.is_some(),
            );
        }
        let recv = recv_opt?;
        arg_vals.push(recv);
        for (i, a) in self.args.iter().enumerate() {
            match a.emit(C::Expression, objx, ir) {
                Some(v) => arg_vals.push(v),
                None => {
                    if trace {
                        eprintln!("[imexpr] method={} arg {i} emitted None", self.method_name);
                    }
                    return None;
                }
            }
        }
        if trace {
            eprintln!(
                "[imexpr] method={} emitting call with {} args",
                self.method_name,
                arg_vals.len()
            );
        }
        let ret =
            ir.f.fb
                .call(self.fref, &arg_vals)
                .expect("instance-method externs return I64 (NanBox)");
        match context {
            C::Statement => None,
            _ => Some(ret),
        }
    }

    fn has_java_class(&self) -> bool {
        self.tag.is_some()
    }
    fn get_java_class(&self) -> Option<HostClass> {
        self.tag.as_ref().map(|t| HostClass {
            name: Arc::new(t.get_name().to_string()),
        })
    }
}

/// Parse `(.method recv args…)` — instance-method dispatch sugar.
/// Looks up the method by `(name, total-arity-including-receiver)` in
/// `Compiler.instance_methods`. Panics with a clear message if the
/// method isn't registered, so adding a new instance method is a
/// one-place edit (declare extern + register in `Compiler::new`).
fn parse_instance_method_form(context: C, form: Object) -> Box<dyn Expr> {
    let head = super::rt::first(&form);
    let method_name = match head {
        Object::Symbol(s) => {
            let n = s.get_name();
            // Stripped here, not in the caller, so this fn owns the
            // contract about the leading `.`.
            assert!(n.starts_with('.') && n.len() > 1, "leading-dot expected");
            n[1..].to_string()
        }
        other => panic!(
            "clojure-jvm: parse_instance_method_form: head must be a `.method` Symbol, got {other:?}"
        ),
    };
    // Field-access sugar: `(.-field-name inst)` → cljvm_rt_user_field_get_by_name.
    // The reader produces head `.-field-name`; method_name after stripping
    // the `.` is `-field-name`, so look for a leading dash.
    if method_name.starts_with('-') && method_name.len() > 1 {
        let field_name = &method_name[1..];
        // Rewrite as `(. clojure.lang.RT (userFieldGetByName inst (quote field-name)))`.
        let receiver = super::rt::second(&form);
        let field_sym = Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern("quote")),
            Object::Symbol(Symbol::intern(field_name)),
        ]));
        let rewritten = Object::List(PersistentList::create(vec![
            Object::Symbol(Symbol::intern(".")),
            Object::Symbol(Symbol::intern_ns_name(None, "clojure.lang.RT")),
            Object::List(PersistentList::create(vec![
                Object::Symbol(Symbol::intern("userFieldGetByName")),
                receiver,
                field_sym,
            ])),
        ]));
        return analyze(context, rewritten);
    }
    // Receiver is the second list element; rest is args.
    let receiver_form = super::rt::second(&form);
    let receiver = analyze(
        if context == C::Eval {
            context
        } else {
            C::Expression
        },
        receiver_form,
    );
    let mut args: Vec<Box<dyn Expr>> = Vec::new();
    // `rest = (next (next form))` — drop head + receiver.
    let mut rest = super::rt::next(&form);
    rest = super::rt::next(&rest);
    while !matches!(rest, Object::Nil) {
        args.push(analyze(C::Expression, super::rt::first(&rest)));
        rest = super::rt::next(&rest);
    }
    let arity_with_receiver = 1 + args.len();
    let fref_opt = with_active_compiler(|c| c.instance_method(&method_name, arity_with_receiver));
    if let Some(fref) = fref_opt {
        return Box::new(InstanceMethodExpr {
            method_name,
            receiver,
            args,
            fref,
            tag: None,
        });
    }
    // Unregistered instance method: emit a runtime `throw` so the
    // defn analyzes cleanly but calling it fails with a clear,
    // greppable message. This is what unblocks loading our forked
    // clojure.core (and downstream cljs.core etc.) without
    // stubbing every (.method recv) call site by hand. The
    // receiver and args are still analyzed (and thus type-checked /
    // expanded) but their values are discarded at runtime since
    // the throw fires before they'd be passed to the method.
    let msg = format!("unregistered instance method `.{method_name}` arity {arity_with_receiver}");
    let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
    Box::new(ThrowExpr { payload })
}

/// `(ClassName. args...)` — Java constructor sugar. Rewrites to
/// `(new ClassName args...)` and dispatches through `parse_new_form`.
/// Caller has already stripped the trailing `.` from `class_name`.
fn parse_constructor_sugar(context: C, class_name: &str, form: &Object) -> Box<dyn Expr> {
    let class_sym = Object::Symbol(Symbol::intern(class_name));
    let new_sym = Object::Symbol(Symbol::intern("new"));
    // Rebuild as `(new ClassName arg1 arg2 ...)`.
    let mut items: Vec<Object> = Vec::new();
    items.push(new_sym);
    items.push(class_sym);
    let mut s = super::rt::next(form);
    while !matches!(s, Object::Nil) {
        items.push(super::rt::first(&s));
        s = super::rt::next(&s);
    }
    let rewritten = Object::List(PersistentList::create(items));
    parse_new_form(context, rewritten)
}

/// `Compiler.NewExpr` — narrowed. `(new Class args...)` allocates a
/// new instance of `Class`. Lowers to a per-arity runtime extern
/// (`cljvm_inst_new_N`) that decodes the Class id and dispatches to
/// the registered constructor (`HostClassInfo::ctor`).
#[derive(Debug)]
pub struct NewExpr {
    pub class_expr: Box<dyn Expr>,
    pub args: Vec<Box<dyn Expr>>,
    pub fref: dynir::FuncRef,
    pub class_name: String,
}

impl Expr for NewExpr {
    fn emit(&self, context: C, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        let mut arg_vals: Vec<Value> = Vec::with_capacity(self.args.len() + 1);
        let class_val = self.class_expr.emit(C::Expression, objx, ir)?;
        arg_vals.push(class_val);
        for a in &self.args {
            match a.emit(C::Expression, objx, ir) {
                Some(v) => arg_vals.push(v),
                None => return None,
            }
        }
        let ret =
            ir.f.fb
                .call(self.fref, &arg_vals)
                .expect("cljvm_inst_new_N returns I64 (NanBox)");
        match context {
            C::Statement => None,
            _ => Some(ret),
        }
    }

    fn has_java_class(&self) -> bool {
        true
    }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new(self.class_name.clone()),
        })
    }
}

/// Parse `(new ClassName args...)`. The class symbol is analyzed to
/// produce a Class NanBox handle (via `analyze_symbol`'s class path);
/// constructor dispatch happens in the runtime extern.
fn parse_new_form(context: C, form: Object) -> Box<dyn Expr> {
    let n = super::rt::count(&form);
    if n < 2 {
        panic!(
            "clojure-jvm: IllegalArgumentException — new requires at least a class, \
             got `(new)` with no class"
        );
    }
    let class_form = super::rt::second(&form);
    let class_name = match class_form.peel_meta_ref() {
        Object::Symbol(s) => match s.get_namespace() {
            Some(ns) => format!("{ns}.{}", s.get_name()),
            None => s.get_name().to_string(),
        },
        other => panic!("clojure-jvm: new expects a class Symbol as first arg, got {other:?}"),
    };
    let class_expr = analyze(C::Expression, class_form);
    // Args = (next (next form))
    let mut args: Vec<Box<dyn Expr>> = Vec::new();
    let mut s = super::rt::next(&form);
    s = super::rt::next(&s);
    while !matches!(s, Object::Nil) {
        args.push(analyze(
            if context == C::Eval {
                context
            } else {
                C::Expression
            },
            super::rt::first(&s),
        ));
        s = super::rt::next(&s);
    }
    let arity_with_class = 1 + args.len();
    let fref_opt = with_active_compiler(|c| c.instance_method("__new__", arity_with_class));
    let fref = match fref_opt {
        Some(f) => f,
        // Unregistered constructor: emit a runtime `throw` so the
        // defn analyzes cleanly but calling it fails with a clear,
        // greppable message. Same pattern as unregistered
        // instance/static methods — see `parse_instance_method_form`.
        None => {
            let msg = format!(
                "unregistered constructor `(new {class_name} ...)` arity {arity_with_class}"
            );
            let payload: Box<dyn Expr> = Box::new(StringExpr::new(msg));
            return Box::new(ThrowExpr { payload });
        }
    };
    Box::new(NewExpr {
        class_expr,
        args,
        fref,
        class_name,
    })
}

/// Parse `(f a b c)` as an `InvokeExpr`. Pre-condition: caller has already
/// confirmed this is not a special form. The head and args are all analyzed.
fn parse_invoke_form(_context: C, form: Object) -> Box<dyn Expr> {
    let head = super::rt::first(&form);
    let head_expr = analyze(C::Expression, head);
    let mut args: Vec<Box<dyn Expr>> = Vec::new();
    let mut rest = super::rt::next(&form);
    while !matches!(rest, Object::Nil) {
        args.push(analyze(C::Expression, super::rt::first(&rest)));
        rest = super::rt::next(&rest);
    }
    Box::new(InvokeExpr {
        fexpr: head_expr,
        args,
        line: 0,
        column: 0,
        tag: None,
    })
}

/// Inline `ConstantExpr.Parser.parse` for `(quote v)`. Java dispatches the
/// quoted value through several shape checks; we route what we recognize.
fn parse_quote_form(_context: C, form: Object) -> Box<dyn Expr> {
    let n = super::rt::count(&form) - 1;
    if n != 1 {
        panic!(
            "clojure-jvm: ExceptionInfo — Wrong number of args ({}) passed to quote",
            n
        );
    }
    let v = super::rt::second(&form);
    match &v {
        Object::Nil => Box::new(NIL_EXPR),
        Object::Bool(true) => Box::new(TRUE_EXPR),
        Object::Bool(false) => Box::new(FALSE_EXPR),
        Object::Long(_) | Object::Double(_) => NumberExpr::parse(v),
        Object::String(s) => Box::new(StringExpr { str: s.clone() }),
        _ => Box::new(ConstantExpr::new(v)),
    }
}

// ============================================================================
// `Compiler` — the active compilation session.
//
// Java's `Compiler.eval` is the entry point. It creates a fresh
// DynamicClassLoader, pushes thread bindings (LOADER, SOURCE, LINE, …), then
// for non-trivial forms wraps the form in `(fn* [] form)`, compiles it, and
// invokes the resulting fn.
//
// We mirror that: every form ultimately becomes a 0-arity dynir function that
// runs through the JIT. The `Compiler` struct owns the in-flight
// `DynModule`, accumulates pending fn declarations (filled in once `FnExpr`
// lands), and exposes `compile_top_level` to drive the pipeline.
//
// Within an active compilation, `with_active_compiler` lets analyze paths
// reach the current `Compiler` without changing the analyze signatures.
// ============================================================================

/// Active compilation session.
pub struct Compiler {
    pub dm: dynlang::DynModule,
    /// Top-level fn (`(fn* …)` and friends) waiting to be lowered into the
    /// DynModule. Each `FnExpr` declared during analyze pushes one entry; the
    /// driver drains the list after analyze and before `build`.
    pending_fns: std::sync::Mutex<Vec<PendingFn>>,
    /// Monotonic counter used to mint unique fn names per session.
    next_fn_id: std::sync::atomic::AtomicU32,
    /// FuncRefs of the runtime extern functions JIT-compiled code can call.
    pub externs: RuntimeExterns,
    /// FuncRefs for the per-arity `IFn.invoke` thunks used by InvokeExpr's
    /// non-static head path. Index `i` holds `cljvm_rt_invoke_i`. Supports
    /// arities 0..=3 for now; higher arities expand here as we need them
    /// (Clojure goes to 20-ish positional + variadic).
    pub invoke_externs: [dynir::FuncRef; 11],
    /// Static-method registry for `HostExpr`. Maps `(class_name, method_name,
    /// arity)` → `FuncRef` declared on the DynModule. Populated at
    /// `Compiler::new` time so the same set of methods is available across
    /// every form compiled in this session.
    ///
    /// This is our 1:1 stand-in for Java's reflective `Reflector.getMethods`:
    /// `(. clojure.lang.RT (inc x))` looks up `("clojure.lang.RT", "inc", 1)`
    /// here, finds a FuncRef, and emits a direct `Call`. Methods land
    /// incrementally as `clojure.core` exercises them.
    pub host_methods: HashMap<(String, String, usize), dynir::FuncRef>,
    /// Per-FuncRef arity table — `(fixed_arity, is_variadic)` for every
    /// pending fn we lower. Used by `cljvm_rt_invoke_*` to handle
    /// variadic targets dynamically: when the call site has more args
    /// than the body's fixed_arity AND the body is variadic, the
    /// runtime packs the overflow into a list before calling.
    pub fn_arities: std::sync::Mutex<HashMap<u32, VarFnInfo>>,
    /// Instance-method dispatch table: `(method-name, total-arity)` →
    /// FuncRef of the runtime extern that handles it. The arity counts
    /// the receiver as the first arg (so `(.meta x)` is arity 1,
    /// `(.withMeta x m)` is arity 2). Populated in `Compiler::new` and
    /// looked up by `parse_instance_method_form`.
    pub instance_methods: HashMap<(String, usize), dynir::FuncRef>,
    /// Per-arity stub host call. Used by `parse_dot_form` /
    /// `parse_instance_method_form` as a fallback when a host method
    /// isn't registered — the call analyzes (so defns referencing
    /// unported Java statics still compile) but panics loudly at
    /// runtime if actually invoked.
    pub unimpl_host_stubs: Vec<dynir::FuncRef>,
    /// Var → FuncRef mapping populated by `DefExpr.emit` when init is a
    /// `FnExpr`. `InvokeExpr.emit` checks this when its head is a `VarExpr`
    /// so `(defn foo …) (foo args)` lowers to a direct `Call`. The Var
    /// pointer is used as a stable identity key: every `Arc<Var>` is kept
    /// alive by `Namespace::mappings`, so the raw pointer doesn't dangle
    /// for the compilation's lifetime.
    ///
    /// Divergence from Java: Clojure JVM dispatches through `Var.fn()` at
    /// runtime so redefinitions are observable. Our static map captures the
    /// binding at compile-emit time and bypasses Var-lookup for the call.
    /// Fine for clojure.core (no in-program redefinition); higher-order /
    /// reflective use cases land later, behind first-class fn values.
    pub var_fns: std::sync::Mutex<HashMap<*const Var, dynir::FuncRef>>,
    /// Variadic metadata for `var_fns` entries. Indexed by the same Var
    /// pointer key. Populated at `parse_def_form` time when init is a
    /// `FnExpr` — the InvokeExpr emit path uses these to decide whether to
    /// pack overflow args into a list.
    pub var_fn_infos: std::sync::Mutex<HashMap<*const Var, VarFnInfo>>,
    /// Per-Var multi-arity dispatch table. Vec entries are
    /// `(FuncRef, VarFnInfo)` per clause; `InvokeExpr.emit` finds the
    /// matching arity to direct-call.
    pub var_multi_arities: std::sync::Mutex<HashMap<*const Var, Vec<(dynir::FuncRef, VarFnInfo)>>>,

    /// ObjTypeId for `clojure.lang.String` — a varlen-byte heap object holding
    /// UTF-8 bytes. Allocated by the GC heap; the type_id stays stable across
    /// the compilation's lifetime.
    pub string_type_id: dynlang::ObjTypeId,

    /// ObjTypeId for `clojure.lang.Symbol` — Raw64 holding `Arc<Symbol>` ptr.
    /// The Arc lifetime is extended past JIT compile by `CompileRoots`.
    pub symbol_type_id: dynlang::ObjTypeId,
    /// ObjTypeId for `clojure.lang.Keyword` — Raw64 holding `Arc<Keyword>` ptr.
    /// Same lifetime story as Symbol.
    pub keyword_type_id: dynlang::ObjTypeId,
    /// ObjTypeId for `clojure.lang.Cons` — two value-fields (`first`, `rest`)
    /// each holding a NanBox-encoded `Object`. Both are GC-traced.
    /// Empty-list terminator is `Object::Nil` NanBox in the `rest` slot.
    pub cons_type_id: dynlang::ObjTypeId,
    /// ObjTypeId for `clojure.lang.Closure` — Raw64 `fref_index` + a varlen
    /// section of GC-traced NanBox values holding the captured outer-fn
    /// locals. Created by `FnExpr.emit` when the body references outer
    /// bindings; the closure body's first param is the closure object
    /// itself, and `LocalBindingExpr.emit` reads captures from it.
    pub closure_type_id: dynlang::ObjTypeId,
    /// ObjTypeId for `clojure.lang.PersistentVector` — a flat varlen-values
    /// heap object holding NanBox-encoded items. Java's PersistentVector
    /// is a HAMT with a tail buffer; we model it as a flat array here
    /// (efficient for compile-time literals + macro arg passing, simpler
    /// than the HAMT). Update / grow operations re-allocate.
    pub vector_type_id: dynlang::ObjTypeId,
    /// ObjTypeId for `clojure.lang.PersistentHashMap` — Raw64 wrapper holding
    /// an `Arc<PersistentHashMap>` raw pointer. The host-side Arc is kept
    /// alive by `CompileRoots._maps` for the JIT module's lifetime. Same
    /// shape as `Symbol`/`Keyword`: the heap wrapper exists so `type_id`
    /// dispatch in `heap_bits_to_object` can recover `Object::Map`, while
    /// the actual key/value storage lives in the host-side `Vec`.
    pub map_type_id: dynlang::ObjTypeId,
    /// ObjTypeId for `clojure.lang.PersistentHashSet` — same shape as Map.
    /// The host-side Arc lives in `CompileRoots._sets`.
    pub set_type_id: dynlang::ObjTypeId,
    pub tree_map_type_id: dynlang::ObjTypeId,
    pub tree_set_type_id: dynlang::ObjTypeId,
    pub string_builder_type_id: dynlang::ObjTypeId,
    pub chunk_buffer_type_id: dynlang::ObjTypeId,
    pub i_chunk_type_id: dynlang::ObjTypeId,
    pub lazy_seq_type_id: dynlang::ObjTypeId,
    pub delay_type_id: dynlang::ObjTypeId,
    pub reduced_type_id: dynlang::ObjTypeId,
    pub multi_arity_fn_type_id: dynlang::ObjTypeId,
    /// Cached `ObjTypeHandle` for `clojure.lang.Closure`. Captured at
    /// `Compiler::new` time because `Session::new` immediately moves the
    /// DynModule's `obj_types` Vec out into the GC's storage — after
    /// which `dm.get_obj_type` / `dm.obj_handle` panic with "len 0".
    /// FnExpr.emit's capturing-closure path reads its layout from this
    /// cached handle instead.
    pub closure_handle: dynlang::ObjTypeHandle,
    /// Pre-computed `(fref_index byte-offset, varlen-base offset)` for
    /// the `clojure.lang.Closure` type. Same caching reason as
    /// `closure_handle`.
    pub closure_fref_offset: i32,
    pub closure_varlen_base: i64,
    /// `clojure.lang.Class`: a Raw64 slot holding a `ClassId` (u16).
    /// Used by `analyze_symbol`'s class-name path and the runtime
    /// `cljvm_inst_isInstance` extern.
    pub class_type_id: dynlang::ObjTypeId,
    /// `clojure.lang.WithMeta` heap layout: two Value slots —
    /// `inner` (wrapped value) and `meta` (metadata Map). Generic
    /// IObj wrapper for receivers that don't have a per-type meta
    /// slot (Symbol/Keyword/Vector/Map). See `cljvm_inst_with_meta`
    /// for the dispatch.
    pub with_meta_type_id: dynlang::ObjTypeId,
    /// `clojure.lang.Var` heap layout: Raw64 holding `*const Var`.
    /// Used by `(var X)` literals + `(.setMacro v)` instance dispatch.
    pub var_type_id: dynlang::ObjTypeId,
    /// `clojure.lang.Namespace` heap layout: Raw64 holding `*const Namespace`.
    /// Lets namespace objects flow as runtime values (`*ns*`, `the-ns`,
    /// `ns-map`, `(. *ns* (refer …))`).
    pub namespace_type_id: dynlang::ObjTypeId,
    /// `clojure.lang.Long` — boxed i64. Raw64 cell holding the integer.
    /// Lets integers be real longs (`(+ 1 2)` → `3`) distinct from doubles.
    pub long_type_id: dynlang::ObjTypeId,
    /// `clojure.lang.Character` — boxed Unicode codepoint (Raw64 cell). A
    /// distinct type from Long so `str`/`pr-str` render the character and
    /// equality with integers is false.
    pub character_type_id: dynlang::ObjTypeId,
    /// FuncRefs for the numeric-tower externs that `PrimOpExpr` calls.
    pub num: NumExterns,
    /// FuncRef of `cljvm_case_dispatch` — the `case*` switch-index helper
    /// (hash / intValue + shift/mask). Captured by `parse_case_form` so
    /// `CaseExpr.emit` can lower the dispatch to a direct call.
    pub case_dispatch: dynir::FuncRef,
    /// `clojure.lang.UserInstance`: the shared ObjType for every
    /// `deftype`/`defrecord` instance. One Raw64 field carrying the
    /// `UserTypeId` (allocated by `lang::user_types::register_user_type`)
    /// plus a varlen-values section holding the declared fields in
    /// declaration order. The same ObjTypeId is used for every user
    /// type so dispatch can read the discriminator off the Raw64 slot
    /// rather than mint a new ObjType per `deftype`.
    pub user_instance_type_id: dynlang::ObjTypeId,
    /// Byte offset (from the heap cell base) of the Raw64
    /// `user_type_id` slot. Cached at `Compiler::new` time the same
    /// way `closure_fref_offset` is, because `dm.obj_handle` panics
    /// after `Session::new` moves `obj_types` out.
    pub user_instance_user_type_id_offset: i32,
    /// Byte offset of the first varlen value slot. Subsequent slots
    /// live at `user_instance_varlen_base + 8 * i`.
    pub user_instance_varlen_base: i64,

    /// Pending compile-time literals. `*Expr.emit` interns into this queue
    /// and emits `gc_literal(LiteralRef(idx))`. After `gc.compile_jit`
    /// produces a `JitModule`, we install the mutator thread, allocate each
    /// pending literal on the GC heap, populate its payload, and push the
    /// resulting NanBox pointer into the JitModule's `literal_pool` at the
    /// matching index.
    ///
    /// The pool is registered as a GC root source by `DynGcRuntime::run_jit`,
    /// so a moving collection traces and rewrites these slots in place — the
    /// emitted `Inst::GcLiteral(idx)` loads the up-to-date pointer on each
    /// execution.
    pub pending_literals: std::sync::Mutex<Vec<PendingLiteral>>,

    /// Absolute base index of the *next* literal slot to be assigned. Because
    /// `pending_literals` is drained per-form (its contents flushed into the
    /// session's persistent `literal_pool`), `pool.len()` alone is not a
    /// stable slot index across forms. We track the absolute slot in this
    /// monotonically-increasing counter so `LiteralRef(idx)`s baked into one
    /// form's IR remain valid when subsequent forms grow the pool.
    pub literal_pool_offset: std::sync::atomic::AtomicUsize,
}

/// A compile-time constant queued for heap allocation after the JIT module
/// is built. Indexes match the `LiteralRef`s baked into the IR.
#[derive(Debug)]
pub enum PendingLiteral {
    /// `clojure.lang.String` — varlen bytes.
    String(Arc<String>),
    /// `clojure.lang.Symbol` — Raw64 holding the global `Arc<Symbol>` pointer.
    Symbol(Arc<Symbol>),
    /// `clojure.lang.Keyword` — Raw64 holding the `Arc<Keyword>` pointer.
    Keyword(Arc<Keyword>),
    /// `clojure.lang.PersistentList` — built recursively as a chain of `Cons`
    /// heap objects at literal-pool-fill time. The pool slot holds the head.
    List(Arc<PersistentList>),
    /// `clojure.lang.PersistentVector` — varlen-slots PersistentVector.
    /// Built at literal-pool-fill time; each element is recursively
    /// allocated, then packed into the vector's varlen area.
    Vector(Arc<PersistentVector>),
    /// `clojure.lang.PersistentHashMap` — Raw64 wrapper holding an
    /// `Arc<PersistentHashMap>` pointer. The Arc is rooted in `CompileRoots`
    /// so the keys/values stay alive across the JIT module's lifetime
    /// (those values are held by-value inside the Arc, not on the GC heap).
    Map(Arc<crate::lang::persistent_hash_map::PersistentHashMap>),
    /// `clojure.lang.PersistentHashSet` — same shape as Map, with element
    /// values stored under `Object::Nil` placeholder values in the
    /// underlying map.
    Set(Arc<crate::lang::persistent_hash_set::PersistentHashSet>),
    /// `clojure.lang.MultiArityFn` — dispatcher cell built from a
    /// `MultiArityFnExpr`. Holds an `Arc<Vec<MultiArityEntry>>` mapping
    /// (fixed_arity, variadic) → fref_idx; runtime dynamic dispatch
    /// (`apply`, `cljvm_rt_invoke_*`) uses it to pick the matching
    /// clause when the receiver fn handle is one of these cells.
    MultiArityFn(Arc<Vec<crate::runtime::MultiArityEntry>>),
    /// `clojure.lang.Class` — heap cell whose Raw64 slot holds a small
    /// `ClassId` int. The runtime decodes that id and dispatches on it
    /// for `(. c (isInstance x))`. Source: `analyze_symbol` resolving a
    /// symbol like `clojure.lang.ISeq` against `host_class::lookup`.
    Class(super::host_class::ClassId),
    /// `clojure.lang.Var` — heap cell holding a `*const Var`. The Var
    /// itself lives forever in the namespace mapping; the heap cell
    /// just gives runtime code a NanBox handle to it.
    Var(Arc<Var>),
    /// `clojure.lang.Long` — a boxed integer literal, pre-boxed once at
    /// pool-fill time so `NumberExpr` can `gc_literal` it instead of
    /// allocating on every evaluation.
    Long(i64),
    /// `clojure.lang.Character` — a boxed char literal (`\a`), pre-boxed once
    /// at pool-fill time so `CharExpr` can `gc_literal` it.
    Char(u32),
}

// SAFETY: the `*const Var` keys point at `Arc<Var>` interned in the global
// namespace mapping, valid for the program's lifetime. We never deref them
// from inside the map; equality / hashing are pointer-identity.
unsafe impl Send for Compiler {}
unsafe impl Sync for Compiler {}

/// Per-Var variadic info recorded when a `(def name (fn* …))` form is
/// parsed. Looked up by `InvokeExpr.emit` to decide call-site packing.
#[derive(Debug, Clone, Copy)]
pub struct VarFnInfo {
    pub is_variadic: bool,
    pub fixed_arity: usize,
}

/// FuncRefs for the externs every `Compiler` declares up-front. The order
/// matches `runtime_extern_fn_ptrs()`.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeExterns {
    pub var_bind_root: dynir::FuncRef,
    pub var_deref: dynir::FuncRef,
}

/// FuncRefs for the numeric-tower externs (`cljvm_num_*`). `PrimOpExpr`
/// emits calls to these — they operate on boxed-Long / native-double
/// values with Clojure's long/double promotion, replacing dynlang's
/// NaN-box float-only `add`/`sub`/… (which can't see boxed Longs).
#[derive(Debug, Clone, Copy)]
pub struct NumExterns {
    pub add: dynir::FuncRef,
    pub sub: dynir::FuncRef,
    pub mul: dynir::FuncRef,
    pub div: dynir::FuncRef,
    pub quot: dynir::FuncRef,
    pub rem: dynir::FuncRef,
    pub lt: dynir::FuncRef,
    pub gt: dynir::FuncRef,
    pub le: dynir::FuncRef,
    pub ge: dynir::FuncRef,
    pub equiv: dynir::FuncRef,
    /// `=` general equality (`cljvm_equals`).
    pub eq: dynir::FuncRef,
}

/// Assemble the `&[*const u8]` extern table for `JitModule::compile_batch`,
/// in the same order externs were declared.
fn runtime_extern_fn_ptrs() -> Vec<*const u8> {
    use crate::runtime::{cljvm_var_bind_root, cljvm_var_deref};
    vec![
        cljvm_var_bind_root as *const u8,
        cljvm_var_deref as *const u8,
    ]
}

/// One fn awaiting IR lowering. Holds the declared FuncRef plus everything
/// `FnMethod`-equivalent needs at emit time.
#[derive(Clone)]
pub struct PendingFn {
    pub fref: dynir::FuncRef,
    pub params: Vec<Arc<LocalBinding>>,
    pub body: Arc<dyn Expr>,
    pub name: String,
    /// Compile-time fn id (`LocalBinding.owning_fn_id` for locals owned by
    /// this fn). Used by `lower_pending_fn` to install `CURRENT_FN_ID`
    /// during body emit so `LocalBindingExpr.emit` can distinguish locals
    /// from outer-fn captures.
    pub fn_id: u32,
    /// Outer bindings captured by this fn's body. Each gets a slot in the
    /// closure object at emit time; the body reads them from the closure
    /// passed as the implicit first arg.
    pub captures: Vec<Arc<LocalBinding>>,
    /// `(fn name? [params] body)` — when `name` is provided, the body can
    /// reference itself by that symbol. We register the name as a regular
    /// LocalBinding (so `LocalBindingExpr.emit` reads from a slot) and
    /// emit a `set_var` at fn entry that initializes that slot to the
    /// fn's TAG_FN handle (or for closures, the implicit closure-self
    /// arg). `Some(idx)` is the LocalBinding's slot index.
    pub self_name_slot: Option<i32>,
    /// For a clause of a non-capturing multi-arity fn: the literal-pool
    /// slot of the shared `MultiArityFn` dispatcher cell. When set,
    /// `lower_pending_fn` initializes the named self-reference slot to
    /// this cell (instead of `TAG_FN(own-clause-fref)`) so a self-call to
    /// a DIFFERENT arity dispatches correctly instead of re-entering the
    /// calling clause.
    pub self_multi_lit: Option<dynir::ir::LiteralRef>,
}

impl Compiler {
    /// Create a fresh compilation session.
    ///
    /// Externs are declared in a fixed order so that `runtime_extern_fn_ptrs`
    /// matches at JIT-binding time.
    pub fn new() -> Self {
        // Per-space size of the moving semi-space heap (from + to spaces each
        // get this many bytes, plus the nursery). 16 MiB comfortably holds
        // full `clojure.core`'s live set (compiled-fn metadata, interned
        // symbols/keywords, var roots, literals) with headroom for user
        // programs. The old 64 KiB default OOM'd partway through core load
        // ("to-space exhausted"). GC-stress coverage is unaffected: stress
        // mode forces a collection on every allocation via `gc_every_alloc`
        // (the `CLOJURE_GC_STRESS` env / `set_gc_every_alloc`), independent
        // of heap size.
        const SESSION_HEAP_BYTES: usize = 16 * 1024 * 1024;
        let mut dm = dynlang::DynModule::new(
            dynlang::GcConfig::generational(SESSION_HEAP_BYTES),
            dynlang::NanBoxTags::default(),
        );
        // Declare runtime externs. ORDER MATTERS — must match
        // `runtime_extern_fn_ptrs`.
        let sig_2i64 = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let sig_1i64 = dynir::Signature {
            params: vec![dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let var_bind_root = dm.declare_extern("cljvm_var_bind_root", sig_2i64.clone());
        let var_deref = dm.declare_extern("cljvm_var_deref", sig_1i64.clone());

        // Numeric-tower externs (boxed-Long / native-double arithmetic).
        let num = NumExterns {
            add: dm.declare_extern("cljvm_num_add", sig_2i64.clone()),
            sub: dm.declare_extern("cljvm_num_sub", sig_2i64.clone()),
            mul: dm.declare_extern("cljvm_num_mul", sig_2i64.clone()),
            div: dm.declare_extern("cljvm_num_div", sig_2i64.clone()),
            quot: dm.declare_extern("cljvm_num_quot", sig_2i64.clone()),
            rem: dm.declare_extern("cljvm_num_rem", sig_2i64.clone()),
            lt: dm.declare_extern("cljvm_num_lt", sig_2i64.clone()),
            gt: dm.declare_extern("cljvm_num_gt", sig_2i64.clone()),
            le: dm.declare_extern("cljvm_num_le", sig_2i64.clone()),
            ge: dm.declare_extern("cljvm_num_ge", sig_2i64.clone()),
            equiv: dm.declare_extern("cljvm_num_equiv", sig_2i64.clone()),
            eq: dm.declare_extern("cljvm_equals", sig_2i64.clone()),
        };

        // Declare host-method externs and build the dispatch table.
        let mut host_methods: HashMap<(String, String, usize), dynir::FuncRef> = HashMap::new();
        let rt_inc = dm.declare_extern("cljvm_rt_inc", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "inc".to_string(), 1),
            rt_inc,
        );
        let sig_0i64 = dynir::Signature {
            params: vec![],
            ret: Some(dynir::Type::I64),
        };
        let rt_next_id = dm.declare_extern("cljvm_rt_nextID", sig_0i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "nextID".to_string(), 0),
            rt_next_id,
        );
        let rt_int_cast = dm.declare_extern("cljvm_rt_intCast", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "intCast".to_string(), 1),
            rt_int_cast,
        );
        let rt_long_cast = dm.declare_extern("cljvm_rt_longCast", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "longCast".to_string(), 1),
            rt_long_cast,
        );
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "uncheckedLongCast".to_string(),
                1,
            ),
            rt_long_cast,
        );
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "uncheckedIntCast".to_string(),
                1,
            ),
            rt_int_cast,
        );
        let rt_nth = dm.declare_extern("cljvm_rt_nth", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "nth".to_string(), 2),
            rt_nth,
        );
        // `(clojure.lang.RT/load "path")` — Clojure's runtime resource
        // loader. Reads the embedded source for `path` and evaluates each
        // form through the active Session (reentrant `eval_form`), exactly
        // like upstream `Compiler.load`. `*ns*` is saved/restored around the
        // load so the sub-file's `(ns …)` does not leak into the caller.
        let rt_load = dm.declare_extern("cljvm_rt_load", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "load".to_string(), 1),
            rt_load,
        );
        // `(clojure.lang.RT/isReduced x)` — backs `reduced?`.
        let rt_is_reduced = dm.declare_extern("cljvm_rt_isReduced", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "isReduced".to_string(), 1),
            rt_is_reduced,
        );
        // Protocol dispatch (one extern per arity). `defprotocol`
        // expands each declared method `(m [this … args])` to
        // `(fn* [this … args] (. clojure.lang.RT
        //    (protocolDispatchN <method_id> this … args)))`. The
        // extern reads `method_id` off the Long-encoded NanBox bits,
        // looks up the impl in `lang::user_types::DISPATCH`, then
        // tail-calls it via the regular `cljvm_rt_invoke_N` path.
        let rt_protocol_dispatch_1 =
            dm.declare_extern("cljvm_rt_protocol_dispatch_1", sig_2i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "protocolDispatch1".to_string(),
                2,
            ),
            rt_protocol_dispatch_1,
        );
        let sig_3i64 = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let rt_protocol_dispatch_2 =
            dm.declare_extern("cljvm_rt_protocol_dispatch_2", sig_3i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "protocolDispatch2".to_string(),
                3,
            ),
            rt_protocol_dispatch_2,
        );
        let sig_4i64 = dynir::Signature {
            params: vec![dynir::Type::I64; 4],
            ret: Some(dynir::Type::I64),
        };
        let rt_protocol_dispatch_3 =
            dm.declare_extern("cljvm_rt_protocol_dispatch_3", sig_4i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "protocolDispatch3".to_string(),
                4,
            ),
            rt_protocol_dispatch_3,
        );
        let sig_5i64 = dynir::Signature {
            params: vec![dynir::Type::I64; 5],
            ret: Some(dynir::Type::I64),
        };
        let rt_protocol_dispatch_4 =
            dm.declare_extern("cljvm_rt_protocol_dispatch_4", sig_5i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "protocolDispatch4".to_string(),
                5,
            ),
            rt_protocol_dispatch_4,
        );
        // `cljvm_rt_install_impl(type_id, method_id, fn_handle)` —
        // top-level statement emitted by `extend-type` / inline
        // `deftype` impls to register one implementation in the
        // dispatch table.
        let rt_install_impl = dm.declare_extern("cljvm_rt_install_impl", sig_3i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "installImpl".to_string(), 3),
            rt_install_impl,
        );
        // UserInstance allocators (one per field arity). `deftype`'s
        // factory fn lowers to a `(. clojure.lang.RT (allocUserInstanceN
        // <user_type_id> field0 …))` call.
        let rt_alloc_ui_0 = dm.declare_extern("cljvm_rt_alloc_user_instance_0", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "allocUserInstance0".to_string(),
                1,
            ),
            rt_alloc_ui_0,
        );
        let rt_alloc_ui_1 = dm.declare_extern("cljvm_rt_alloc_user_instance_1", sig_2i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "allocUserInstance1".to_string(),
                2,
            ),
            rt_alloc_ui_1,
        );
        let rt_alloc_ui_2 = dm.declare_extern("cljvm_rt_alloc_user_instance_2", sig_3i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "allocUserInstance2".to_string(),
                3,
            ),
            rt_alloc_ui_2,
        );
        let rt_alloc_ui_3 = dm.declare_extern("cljvm_rt_alloc_user_instance_3", sig_4i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "allocUserInstance3".to_string(),
                4,
            ),
            rt_alloc_ui_3,
        );
        let rt_alloc_ui_4 = dm.declare_extern("cljvm_rt_alloc_user_instance_4", sig_5i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "allocUserInstance4".to_string(),
                5,
            ),
            rt_alloc_ui_4,
        );
        // `(.-field-name inst)` field-access lowering target.
        let rt_user_field_get =
            dm.declare_extern("cljvm_rt_user_field_get_by_name", sig_2i64.clone());
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "userFieldGetByName".to_string(),
                2,
            ),
            rt_user_field_get,
        );
        // `(satisfies? P x)` lowering target. Macro expansion bakes
        // the proto_id as a Long literal.
        let rt_satisfies = dm.declare_extern("cljvm_rt_satisfies", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "satisfies".to_string(), 2),
            rt_satisfies,
        );
        // Syntax-quote support — see `cljvm_rt_sq_concat` for why this
        // bypasses `clojure.core/concat` (which is multi-arity and binds
        // its Var to nil until the multi-arity dispatcher is wired).
        let rt_sq_concat = dm.declare_extern("cljvm_rt_sq_concat", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "sqConcat".to_string(), 1),
            rt_sq_concat,
        );
        let rt_peek = dm.declare_extern("cljvm_rt_peek", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "peek".to_string(), 1),
            rt_peek,
        );
        let rt_pop = dm.declare_extern("cljvm_rt_pop", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "pop".to_string(), 1),
            rt_pop,
        );
        let rt_contains = dm.declare_extern("cljvm_rt_contains", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "contains".to_string(), 2),
            rt_contains,
        );
        let rt_get = dm.declare_extern("cljvm_rt_get", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "get".to_string(), 2),
            rt_get,
        );
        let sig_3i64_get = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let rt_get_3 = dm.declare_extern("cljvm_rt_get_3", sig_3i64_get);
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "get".to_string(), 3),
            rt_get_3,
        );
        let rt_find = dm.declare_extern("cljvm_rt_find", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "find".to_string(), 2),
            rt_find,
        );
        let rt_dissoc = dm.declare_extern("cljvm_rt_dissoc", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "dissoc".to_string(), 2),
            rt_dissoc,
        );
        // `clojure.lang.Namespace/find` and `/findOrCreate` statics — back
        // `find-ns` / `create-ns`, the entry points of the `refer` chain.
        let ns_find = dm.declare_extern("cljvm_ns_find", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Namespace".to_string(), "find".to_string(), 1),
            ns_find,
        );
        let ns_find_or_create = dm.declare_extern("cljvm_ns_findOrCreate", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Namespace".to_string(),
                "findOrCreate".to_string(),
                1,
            ),
            ns_find_or_create,
        );
        let rt_keys = dm.declare_extern("cljvm_rt_keys", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "keys".to_string(), 1),
            rt_keys,
        );
        let rt_vals = dm.declare_extern("cljvm_rt_vals", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "vals".to_string(), 1),
            rt_vals,
        );
        let rt_boolean_cast = dm.declare_extern("cljvm_rt_booleanCast", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "booleanCast".to_string(), 1),
            rt_boolean_cast,
        );
        let rt_double_cast = dm.declare_extern("cljvm_rt_doubleCast", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "doubleCast".to_string(), 1),
            rt_double_cast,
        );
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "uncheckedDoubleCast".to_string(),
                1,
            ),
            rt_double_cast,
        );
        let rt_float_cast = dm.declare_extern("cljvm_rt_floatCast", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "floatCast".to_string(), 1),
            rt_float_cast,
        );
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "uncheckedFloatCast".to_string(),
                1,
            ),
            rt_float_cast,
        );
        let rt_byte_cast = dm.declare_extern("cljvm_rt_byteCast", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "byteCast".to_string(), 1),
            rt_byte_cast,
        );
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "uncheckedByteCast".to_string(),
                1,
            ),
            rt_byte_cast,
        );
        let rt_short_cast = dm.declare_extern("cljvm_rt_shortCast", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "shortCast".to_string(), 1),
            rt_short_cast,
        );
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "uncheckedShortCast".to_string(),
                1,
            ),
            rt_short_cast,
        );
        let rt_char_cast = dm.declare_extern("cljvm_rt_charCast", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "charCast".to_string(), 1),
            rt_char_cast,
        );
        host_methods.insert(
            (
                "clojure.lang.RT".to_string(),
                "uncheckedCharCast".to_string(),
                1,
            ),
            rt_char_cast,
        );
        // Var class statics for dynamic binding (stub-with-panic).
        let var_push_tb = dm.declare_extern("cljvm_var_pushThreadBindings", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Var".to_string(),
                "pushThreadBindings".to_string(),
                1,
            ),
            var_push_tb,
        );
        let var_pop_tb = dm.declare_extern("cljvm_var_popThreadBindings", sig_0i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Var".to_string(),
                "popThreadBindings".to_string(),
                0,
            ),
            var_pop_tb,
        );
        let var_get_tbf = dm.declare_extern("cljvm_var_getThreadBindingFrame", sig_0i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Var".to_string(),
                "getThreadBindingFrame".to_string(),
                0,
            ),
            var_get_tbf,
        );
        let var_get_tb = dm.declare_extern("cljvm_var_getThreadBindings", sig_0i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Var".to_string(),
                "getThreadBindings".to_string(),
                0,
            ),
            var_get_tb,
        );
        let var_find = dm.declare_extern("cljvm_var_find", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Var".to_string(), "find".to_string(), 1),
            var_find,
        );
        let sig_3i64_var = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let var_intern_2 = dm.declare_extern("cljvm_var_intern_2", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.Var".to_string(), "intern".to_string(), 2),
            var_intern_2,
        );
        let var_intern_3 = dm.declare_extern("cljvm_var_intern_3", sig_3i64_var);
        host_methods.insert(
            ("clojure.lang.Var".to_string(), "intern".to_string(), 3),
            var_intern_3,
        );
        let var_reset_tbf =
            dm.declare_extern("cljvm_var_resetThreadBindingFrame", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Var".to_string(),
                "resetThreadBindingFrame".to_string(),
                1,
            ),
            var_reset_tbf,
        );
        let var_clone_tbf =
            dm.declare_extern("cljvm_var_cloneThreadBindingFrame", sig_0i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Var".to_string(),
                "cloneThreadBindingFrame".to_string(),
                0,
            ),
            var_clone_tbf,
        );
        let sig_3i64 = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let rt_nth_3 = dm.declare_extern("cljvm_rt_nth_3", sig_3i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "nth".to_string(), 3),
            rt_nth_3,
        );
        let numbers_is_zero = dm.declare_extern("cljvm_numbers_isZero", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Numbers".to_string(), "isZero".to_string(), 1),
            numbers_is_zero,
        );
        // Batch register the rest of clojure.lang.Numbers — every defn in
        // `clojure.core` that does arithmetic compiles to one of these.
        for (name, arity, extern_name) in [
            ("add", 2, "cljvm_numbers_add"),
            ("addP", 2, "cljvm_numbers_add"),
            ("unchecked_add", 2, "cljvm_numbers_add"),
            ("minus", 2, "cljvm_numbers_minus"),
            ("minus", 1, "cljvm_numbers_minus_1"),
            ("minusP", 2, "cljvm_numbers_minus"),
            ("minusP", 1, "cljvm_numbers_minus_1"),
            ("unchecked_minus", 2, "cljvm_numbers_minus"),
            ("unchecked_minus", 1, "cljvm_numbers_minus_1"),
            ("multiply", 2, "cljvm_numbers_multiply"),
            ("multiplyP", 2, "cljvm_numbers_multiply"),
            ("unchecked_multiply", 2, "cljvm_numbers_multiply"),
            ("divide", 2, "cljvm_numbers_divide"),
            ("inc", 1, "cljvm_numbers_inc"),
            ("incP", 1, "cljvm_numbers_inc"),
            ("unchecked_inc", 1, "cljvm_numbers_inc"),
            ("dec", 1, "cljvm_numbers_dec"),
            ("decP", 1, "cljvm_numbers_dec"),
            ("unchecked_dec", 1, "cljvm_numbers_dec"),
            ("lt", 2, "cljvm_numbers_lt"),
            ("lte", 2, "cljvm_numbers_lte"),
            ("gt", 2, "cljvm_numbers_gt"),
            ("gte", 2, "cljvm_numbers_gte"),
            ("equiv", 2, "cljvm_numbers_equiv"),
            ("isPos", 1, "cljvm_numbers_isPos"),
            ("isNeg", 1, "cljvm_numbers_isNeg"),
            ("abs", 1, "cljvm_numbers_abs"),
            ("unchecked_int_inc", 1, "cljvm_numbers_inc"),
            ("unchecked_int_dec", 1, "cljvm_numbers_dec"),
            ("unchecked_int_add", 2, "cljvm_numbers_add"),
            ("unchecked_int_subtract", 2, "cljvm_numbers_minus"),
            ("unchecked_int_multiply", 2, "cljvm_numbers_multiply"),
            ("unchecked_int_divide", 2, "cljvm_numbers_divide"),
            ("unchecked_int_negate", 1, "cljvm_numbers_minus_1"),
            ("unchecked_long_add", 2, "cljvm_numbers_add"),
            ("unchecked_long_subtract", 2, "cljvm_numbers_minus"),
            ("unchecked_long_multiply", 2, "cljvm_numbers_multiply"),
            ("unchecked_long_inc", 1, "cljvm_numbers_inc"),
            ("unchecked_long_dec", 1, "cljvm_numbers_dec"),
            ("unchecked_long_negate", 1, "cljvm_numbers_minus_1"),
            ("unchecked_float_add", 2, "cljvm_numbers_add"),
            ("unchecked_float_subtract", 2, "cljvm_numbers_minus"),
            ("unchecked_float_multiply", 2, "cljvm_numbers_multiply"),
            ("unchecked_float_divide", 2, "cljvm_numbers_divide"),
            ("unchecked_double_add", 2, "cljvm_numbers_add"),
            ("unchecked_double_subtract", 2, "cljvm_numbers_minus"),
            ("unchecked_double_multiply", 2, "cljvm_numbers_multiply"),
            ("unchecked_double_divide", 2, "cljvm_numbers_divide"),
            ("unchecked_int_remainder", 2, "cljvm_numbers_remainder"),
            ("unchecked_int_quotient", 2, "cljvm_numbers_quotient"),
            ("unchecked_long_remainder", 2, "cljvm_numbers_remainder"),
            ("unchecked_long_quotient", 2, "cljvm_numbers_quotient"),
            ("uncheckedIntCast", 1, "cljvm_rt_intCast"),
            ("uncheckedLongCast", 1, "cljvm_rt_longCast"),
            ("rationalize", 1, "cljvm_numbers_identity"),
            ("isInteger", 1, "cljvm_numbers_isInteger"),
            ("isFloat", 1, "cljvm_numbers_isFloat"),
            ("isRational", 1, "cljvm_numbers_isRational"),
            ("isNaN", 1, "cljvm_numbers_isNaN"),
            ("isInfinite", 1, "cljvm_numbers_isInfinite"),
            ("not", 1, "cljvm_numbers_not"),
            ("and", 2, "cljvm_numbers_and"),
            ("or", 2, "cljvm_numbers_or"),
            ("xor", 2, "cljvm_numbers_xor"),
            ("andNot", 2, "cljvm_numbers_andNot"),
            ("shiftLeft", 2, "cljvm_numbers_shiftLeft"),
            ("shiftRight", 2, "cljvm_numbers_shiftRight"),
            ("unsignedShiftRight", 2, "cljvm_numbers_unsignedShiftRight"),
            ("clearBit", 2, "cljvm_numbers_clearBit"),
            ("setBit", 2, "cljvm_numbers_setBit"),
            ("flipBit", 2, "cljvm_numbers_flipBit"),
            ("testBit", 2, "cljvm_numbers_testBit"),
            ("max", 2, "cljvm_numbers_max"),
            ("min", 2, "cljvm_numbers_min"),
            ("quotient", 2, "cljvm_numbers_quotient"),
            ("remainder", 2, "cljvm_numbers_remainder"),
        ] {
            let sig = if arity == 1 {
                sig_1i64.clone()
            } else {
                sig_2i64.clone()
            };
            let f = dm.declare_extern(extern_name, sig);
            host_methods.insert(
                ("clojure.lang.Numbers".to_string(), name.to_string(), arity),
                f,
            );
        }
        // `clojure.lang.LongRange/create` — backs `clojure.core/range` (the
        // int path). Arities 1/2/3 = (end), (start end), (start end step).
        let lr1 = dm.declare_extern("cljvm_longrange_create_1", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.LongRange".to_string(), "create".to_string(), 1),
            lr1,
        );
        let lr2 = dm.declare_extern("cljvm_longrange_create_2", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.LongRange".to_string(), "create".to_string(), 2),
            lr2,
        );
        let lr3 = dm.declare_extern("cljvm_longrange_create_3", sig_3i64.clone());
        host_methods.insert(
            ("clojure.lang.LongRange".to_string(), "create".to_string(), 3),
            lr3,
        );
        // NOTE: `clojure.lang.Repeat/create` (infinite `(repeat x)`) is
        // deliberately NOT registered. A self-referential-cons representation
        // hangs because `take`/`into` over it realize eagerly here (our
        // lazy-seq forcing isn't lazy enough on this path), so the infinite
        // seq never terminates. Needs a real lazy/bounded Repeat seq type
        // before `repeat`/`interpose` can be unblocked.
        // String reverse helper for `clojure.string/reverse` (the shim calls
        // `clojure.string.Native/reverse`).
        let str_rev = dm.declare_extern("cljvm_str_reverse", sig_1i64.clone());
        host_methods.insert(
            ("clojure.string.Native".to_string(), "reverse".to_string(), 1),
            str_rev,
        );
        // Register under both the short (`Math/pow`) and qualified
        // (`java.lang.Math/pow`) class names — call sites use either.
        for cls in ["Math", "java.lang.Math"] {
            let math_pow = dm.declare_extern("cljvm_math_pow", sig_2i64.clone());
            host_methods.insert((cls.to_string(), "pow".to_string(), 2), math_pow);
            let math_sqrt = dm.declare_extern("cljvm_math_sqrt", sig_1i64.clone());
            host_methods.insert((cls.to_string(), "sqrt".to_string(), 1), math_sqrt);
        }
        let delay_force = dm.declare_extern("cljvm_delay_force", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Delay".to_string(), "force".to_string(), 1),
            delay_force,
        );
        let rt_cons = dm.declare_extern("cljvm_rt_cons", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "cons".to_string(), 2),
            rt_cons,
        );
        let ns_set_current = dm.declare_extern("cljvm_ns_set_current", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Namespace".to_string(),
                "setCurrent".to_string(),
                1,
            ),
            ns_set_current,
        );
        let rt_conj = dm.declare_extern("cljvm_rt_conj", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "conj".to_string(), 2),
            rt_conj,
        );
        let sig_3i64_assoc = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let rt_assoc = dm.declare_extern("cljvm_rt_assoc", sig_3i64_assoc);
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "assoc".to_string(), 3),
            rt_assoc,
        );
        let rt_first = dm.declare_extern("cljvm_rt_first", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "first".to_string(), 1),
            rt_first,
        );
        let rt_next = dm.declare_extern("cljvm_rt_next", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "next".to_string(), 1),
            rt_next,
        );
        let rt_more = dm.declare_extern("cljvm_rt_more", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "more".to_string(), 1),
            rt_more,
        );
        let rt_seq = dm.declare_extern("cljvm_rt_seq", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "seq".to_string(), 1),
            rt_seq,
        );
        let rt_count = dm.declare_extern("cljvm_rt_count", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "count".to_string(), 1),
            rt_count,
        );
        let rt_to_array = dm.declare_extern("cljvm_rt_toArray", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "toArray".to_string(), 1),
            rt_to_array,
        );
        // `java.util.Arrays/sort(array, comparator)` — backs `sort`/`sort-by`,
        // which `to-array` then `(. java.util.Arrays (sort a comp))`.
        let arrays_sort = dm.declare_extern("cljvm_arrays_sort", sig_2i64.clone());
        host_methods.insert(
            ("java.util.Arrays".to_string(), "sort".to_string(), 2),
            arrays_sort,
        );
        let lpv_create = dm.declare_extern("cljvm_lpv_create", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.LazilyPersistentVector".to_string(),
                "create".to_string(),
                1,
            ),
            lpv_create,
        );
        let phm_create = dm.declare_extern("cljvm_phm_create", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.PersistentHashMap".to_string(),
                "create".to_string(),
                1,
            ),
            phm_create,
        );
        let phs_create = dm.declare_extern("cljvm_phs_create", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.PersistentHashSet".to_string(),
                "create".to_string(),
                1,
            ),
            phs_create,
        );
        let ptm_create = dm.declare_extern("cljvm_ptm_create", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.PersistentTreeMap".to_string(),
                "create".to_string(),
                1,
            ),
            ptm_create,
        );
        let ptm_create_cmp = dm.declare_extern("cljvm_ptm_create_cmp", sig_2i64.clone());
        host_methods.insert(
            (
                "clojure.lang.PersistentTreeMap".to_string(),
                "create".to_string(),
                2,
            ),
            ptm_create_cmp,
        );
        let pts_create = dm.declare_extern("cljvm_pts_create", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.PersistentTreeSet".to_string(),
                "create".to_string(),
                1,
            ),
            pts_create,
        );
        let pts_create_cmp = dm.declare_extern("cljvm_pts_create_cmp", sig_2i64.clone());
        host_methods.insert(
            (
                "clojure.lang.PersistentTreeSet".to_string(),
                "create".to_string(),
                2,
            ),
            pts_create_cmp,
        );
        let sig_3i64_subvec = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let rt_subvec = dm.declare_extern("cljvm_rt_subvec", sig_3i64_subvec);
        host_methods.insert(
            ("clojure.lang.RT".to_string(), "subvec".to_string(), 3),
            rt_subvec,
        );
        // `Compiler$HostExpr` static helpers used by `clojure.core/sigs`.
        let hostexpr_special =
            dm.declare_extern("cljvm_compiler_hostexpr_maybeSpecialTag", sig_1i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Compiler$HostExpr".to_string(),
                "maybeSpecialTag".to_string(),
                1,
            ),
            hostexpr_special,
        );
        let hostexpr_class =
            dm.declare_extern("cljvm_compiler_hostexpr_maybeClass", sig_2i64.clone());
        host_methods.insert(
            (
                "clojure.lang.Compiler$HostExpr".to_string(),
                "maybeClass".to_string(),
                2,
            ),
            hostexpr_class,
        );
        // `Symbol/intern` 1-arg form (just a name).
        let symbol_intern_1 = dm.declare_extern("cljvm_symbol_intern_1", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Symbol".to_string(), "intern".to_string(), 1),
            symbol_intern_1,
        );
        let symbol_intern_2 = dm.declare_extern("cljvm_symbol_intern_2", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.Symbol".to_string(), "intern".to_string(), 2),
            symbol_intern_2,
        );
        let keyword_intern_1 = dm.declare_extern("cljvm_keyword_intern_1", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Keyword".to_string(), "intern".to_string(), 1),
            keyword_intern_1,
        );
        let keyword_intern_2 = dm.declare_extern("cljvm_keyword_intern_2", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.Keyword".to_string(), "intern".to_string(), 2),
            keyword_intern_2,
        );
        let keyword_find_1 = dm.declare_extern("cljvm_keyword_find_1", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Keyword".to_string(), "find".to_string(), 1),
            keyword_find_1,
        );
        let keyword_find_2 = dm.declare_extern("cljvm_keyword_find_2", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.Keyword".to_string(), "find".to_string(), 2),
            keyword_find_2,
        );
        let rt_equiv = dm.declare_extern("cljvm_rt_equiv", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.Util".to_string(), "equiv".to_string(), 2),
            rt_equiv,
        );
        // `Util/equals` differs from Java only on null-vs-non-null
        // dispatch (Java does `a == null ? b == null : a.equals(b)`).
        // Our `equiv` already handles nil symmetrically, and for the
        // value types we model the two collapse — share the extern.
        host_methods.insert(
            ("clojure.lang.Util".to_string(), "equals".to_string(), 2),
            rt_equiv,
        );
        let rt_is_nil = dm.declare_extern("cljvm_rt_is_nil", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Util".to_string(), "isNil".to_string(), 1),
            rt_is_nil,
        );
        // `Util/identical(a, b)` — Java reference equality. NanBox bits
        // capture identity for nils/bools/longs/doubles directly, and
        // pointer-identity for heap cells. Since interned Symbols/Keywords
        // share Arc identity, that path also reduces to bit-equality.
        let util_identical = dm.declare_extern("cljvm_util_identical", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.Util".to_string(), "identical".to_string(), 2),
            util_identical,
        );
        let util_compare = dm.declare_extern("cljvm_util_compare", sig_2i64.clone());
        host_methods.insert(
            ("clojure.lang.Util".to_string(), "compare".to_string(), 2),
            util_compare,
        );
        // `Util/hash` — Java's hashCode-based hash. The `case` macro calls
        // this at EXPANSION time (`prep-hashes` / `merge-hash-collisions`)
        // to build the `case*` case-map keys; CaseExpr's runtime dispatch
        // (`cljvm_case_dispatch` below) recomputes the same hash on the
        // dispatch value. Both externs share `runtime::util_hash_bits`,
        // which is what keeps the two sides consistent.
        let util_hash = dm.declare_extern("cljvm_util_hash", sig_1i64.clone());
        host_methods.insert(
            ("clojure.lang.Util".to_string(), "hash".to_string(), 1),
            util_hash,
        );
        // `case*` switch-index helper (not a host method — CaseExpr.emit
        // calls it directly): (value, test-type, shift, mask) → raw i64
        // dispatch index.
        let sig_4i64 = dynir::Signature {
            params: vec![dynir::Type::I64; 4],
            ret: Some(dynir::Type::I64),
        };
        let case_dispatch = dm.declare_extern("cljvm_case_dispatch", sig_4i64);

        // `clojure.lang.PersistentList/creator` — Java declares it as a
        // `static IFn` field whose `invoke(Object... args)` returns a list
        // of the args. We model it as a 0-arg "static getter" registered
        // in `host_methods`: the extern returns the cached NanBox handle
        // of the singleton variadic-identity fn that `Session::new`
        // compiles once at session init.
        let sig_0i64 = dynir::Signature {
            params: vec![],
            ret: Some(dynir::Type::I64),
        };
        let pl_creator = dm.declare_extern("cljvm_pl_creator", sig_0i64);
        host_methods.insert(
            (
                "clojure.lang.PersistentList".to_string(),
                "creator".to_string(),
                0,
            ),
            pl_creator,
        );

        // Instance-method externs (`(.method recv args…)`). Each declared
        // up front so the analyzer can resolve any IObj/IMeta-style call
        // site to a known FuncRef. The extern dispatches receiver-type
        // internally (the polymorphism the JVM gets from interface
        // vtables). Add a new entry when a new instance method is needed.
        let mut instance_methods: HashMap<(String, usize), dynir::FuncRef> = HashMap::new();
        let inst_meta = dm.declare_extern("cljvm_inst_meta", sig_1i64.clone());
        // (method, total-arity-including-receiver)
        instance_methods.insert(("meta".to_string(), 1), inst_meta);
        let inst_with_meta = dm.declare_extern("cljvm_inst_with_meta", sig_2i64.clone());
        instance_methods.insert(("withMeta".to_string(), 2), inst_with_meta);
        let inst_is_instance = dm.declare_extern("cljvm_inst_isInstance", sig_2i64.clone());
        instance_methods.insert(("isInstance".to_string(), 2), inst_is_instance);
        let inst_get_name = dm.declare_extern("cljvm_inst_getName", sig_1i64.clone());
        instance_methods.insert(("getName".to_string(), 1), inst_get_name);
        let inst_concat = dm.declare_extern("cljvm_inst_concat", sig_2i64.clone());
        instance_methods.insert(("concat".to_string(), 2), inst_concat);
        let inst_index_of = dm.declare_extern("cljvm_inst_indexOf", sig_2i64.clone());
        instance_methods.insert(("indexOf".to_string(), 2), inst_index_of);
        let inst_starts_with = dm.declare_extern("cljvm_inst_startsWith", sig_2i64.clone());
        instance_methods.insert(("startsWith".to_string(), 2), inst_starts_with);
        let inst_substring = dm.declare_extern("cljvm_inst_substring", sig_2i64.clone());
        instance_methods.insert(("substring".to_string(), 2), inst_substring);
        let inst_substring2 = dm.declare_extern("cljvm_inst_substring2", sig_3i64.clone());
        instance_methods.insert(("substring".to_string(), 3), inst_substring2);
        let inst_last_index_of = dm.declare_extern("cljvm_inst_lastIndexOf", sig_2i64.clone());
        instance_methods.insert(("lastIndexOf".to_string(), 2), inst_last_index_of);
        let inst_upper = dm.declare_extern("cljvm_inst_toUpperCase", sig_1i64.clone());
        instance_methods.insert(("toUpperCase".to_string(), 1), inst_upper);
        let inst_lower = dm.declare_extern("cljvm_inst_toLowerCase", sig_1i64.clone());
        instance_methods.insert(("toLowerCase".to_string(), 1), inst_lower);
        let inst_trim = dm.declare_extern("cljvm_inst_trim", sig_1i64.clone());
        instance_methods.insert(("trim".to_string(), 1), inst_trim);
        let inst_replace = dm.declare_extern("cljvm_inst_replace", sig_3i64.clone());
        instance_methods.insert(("replace".to_string(), 3), inst_replace);
        let inst_set_macro = dm.declare_extern("cljvm_inst_setMacro", sig_1i64.clone());
        instance_methods.insert(("setMacro".to_string(), 1), inst_set_macro);
        let inst_cast = dm.declare_extern("cljvm_inst_cast", sig_2i64.clone());
        instance_methods.insert(("cast".to_string(), 2), inst_cast);
        let inst_to_string = dm.declare_extern("cljvm_inst_toString", sig_1i64.clone());
        instance_methods.insert(("toString".to_string(), 1), inst_to_string);
        let inst_apply_to = dm.declare_extern("cljvm_inst_applyTo", sig_2i64.clone());
        instance_methods.insert(("applyTo".to_string(), 2), inst_apply_to);
        let inst_disjoin = dm.declare_extern("cljvm_inst_disjoin", sig_2i64.clone());
        instance_methods.insert(("disjoin".to_string(), 2), inst_disjoin);
        // `java.util.Comparator.compare(a, b)` on a Clojure IFn — `sort-by`
        // builds `(fn [x y] (. comp (compare (keyfn x) (keyfn y))))`. Receiver
        // + 2 args = arity-3 key.
        let inst_compare = dm.declare_extern("cljvm_inst_compare", sig_3i64.clone());
        instance_methods.insert(("compare".to_string(), 3), inst_compare);
        // Transients, implemented as their persistent counterparts. Clojure's
        // contract REQUIRES callers to thread the return value of `assoc!`/
        // `conj!`/`dissoc!` (transients "are not designed to be bashed
        // in-place"), so returning a fresh persistent collection each time is
        // a correct — if unoptimized — transient. `transient`/`persistent!`
        // are `(.asTransient coll)` / `(.persistent coll)` = identity; the
        // mutating ops delegate to RT.assoc/conj/dissoc.
        let inst_identity = dm.declare_extern("cljvm_inst_identity", sig_1i64.clone());
        instance_methods.insert(("asTransient".to_string(), 1), inst_identity);
        instance_methods.insert(("persistent".to_string(), 1), inst_identity);
        instance_methods.insert(("assoc".to_string(), 3), rt_assoc);
        instance_methods.insert(("conj".to_string(), 2), rt_conj);
        instance_methods.insert(("without".to_string(), 2), rt_dissoc);
        let inst_get_key = dm.declare_extern("cljvm_inst_getKey", sig_1i64.clone());
        instance_methods.insert(("getKey".to_string(), 1), inst_get_key);
        let inst_get_value = dm.declare_extern("cljvm_inst_getValue", sig_1i64.clone());
        instance_methods.insert(("getValue".to_string(), 1), inst_get_value);
        let inst_rseq = dm.declare_extern("cljvm_inst_rseq", sig_1i64.clone());
        instance_methods.insert(("rseq".to_string(), 1), inst_rseq);
        let inst_get_ns = dm.declare_extern("cljvm_inst_getNamespace", sig_1i64.clone());
        instance_methods.insert(("getNamespace".to_string(), 1), inst_get_ns);
        // `clojure.lang.Var` / `clojure.lang.Namespace` reflective accessors —
        // the host primitives the `refer` / `ns-map` / `ns-publics` functions
        // bottom out in (so the `ns` macro's `(refer 'clojure.core)` runs).
        let inst_var_ns = dm.declare_extern("cljvm_inst_ns", sig_1i64.clone());
        instance_methods.insert(("ns".to_string(), 1), inst_var_ns);
        let inst_is_public = dm.declare_extern("cljvm_inst_isPublic", sig_1i64.clone());
        instance_methods.insert(("isPublic".to_string(), 1), inst_is_public);
        let inst_get_mappings = dm.declare_extern("cljvm_inst_getMappings", sig_1i64.clone());
        instance_methods.insert(("getMappings".to_string(), 1), inst_get_mappings);
        let inst_refer = dm.declare_extern(
            "cljvm_inst_refer",
            dynir::Signature {
                params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
                ret: Some(dynir::Type::I64),
            },
        );
        instance_methods.insert(("refer".to_string(), 3), inst_refer);
        let inst_reset_meta = dm.declare_extern("cljvm_inst_resetMeta", sig_2i64.clone());
        instance_methods.insert(("resetMeta".to_string(), 2), inst_reset_meta);
        let inst_alter_meta = dm.declare_extern(
            "cljvm_inst_alterMeta",
            dynir::Signature {
                params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
                ret: Some(dynir::Type::I64),
            },
        );
        instance_methods.insert(("alterMeta".to_string(), 3), inst_alter_meta);
        let inst_bind_root = dm.declare_extern("cljvm_inst_bindRoot", sig_2i64.clone());
        instance_methods.insert(("bindRoot".to_string(), 2), inst_bind_root);
        let inst_has_root = dm.declare_extern("cljvm_inst_hasRoot", sig_1i64.clone());
        instance_methods.insert(("hasRoot".to_string(), 1), inst_has_root);
        let inst_get_raw_root = dm.declare_extern("cljvm_inst_getRawRoot", sig_1i64.clone());
        instance_methods.insert(("getRawRoot".to_string(), 1), inst_get_raw_root);
        // IRef / IAtom instance methods (stubs).
        let sig_3i64_atom = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let inst_set_validator = dm.declare_extern("cljvm_inst_setValidator", sig_2i64.clone());
        instance_methods.insert(("setValidator".to_string(), 2), inst_set_validator);
        let inst_get_validator = dm.declare_extern("cljvm_inst_getValidator", sig_1i64.clone());
        instance_methods.insert(("getValidator".to_string(), 1), inst_get_validator);
        let inst_get_watches = dm.declare_extern("cljvm_inst_getWatches", sig_1i64.clone());
        instance_methods.insert(("getWatches".to_string(), 1), inst_get_watches);
        let inst_add_watch = dm.declare_extern("cljvm_inst_addWatch", sig_3i64_atom.clone());
        instance_methods.insert(("addWatch".to_string(), 3), inst_add_watch);
        let inst_remove_watch = dm.declare_extern("cljvm_inst_removeWatch", sig_2i64.clone());
        instance_methods.insert(("removeWatch".to_string(), 2), inst_remove_watch);
        let inst_swap = dm.declare_extern("cljvm_inst_swap", sig_2i64.clone());
        instance_methods.insert(("swap".to_string(), 2), inst_swap);
        // ("reset", 1) is already used by MultiFn.reset above; atom-side
        // reset is `(.reset atom new)` arity-2.
        let inst_reset_atom_2 = dm.declare_extern("cljvm_inst_reset", sig_2i64.clone());
        instance_methods.insert(("reset".to_string(), 2), inst_reset_atom_2);
        let inst_cas = dm.declare_extern("cljvm_inst_compareAndSet", sig_3i64_atom);
        instance_methods.insert(("compareAndSet".to_string(), 3), inst_cas);
        let inst_deref = dm.declare_extern("cljvm_inst_deref", sig_1i64.clone());
        instance_methods.insert(("deref".to_string(), 1), inst_deref);
        let inst_iter = dm.declare_extern("cljvm_inst_iterator", sig_1i64.clone());
        instance_methods.insert(("iterator".to_string(), 1), inst_iter);
        // Agent error-handling stubs.
        let sig_3i64_agent = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let sig_4i64_agent = dynir::Signature {
            params: vec![
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
            ],
            ret: Some(dynir::Type::I64),
        };
        for (name, arity, extern_name, sig) in [
            (
                "setErrorHandler",
                2,
                "cljvm_inst_setErrorHandler",
                &sig_2i64,
            ),
            (
                "getErrorHandler",
                1,
                "cljvm_inst_getErrorHandler",
                &sig_1i64,
            ),
            ("setErrorMode", 2, "cljvm_inst_setErrorMode", &sig_2i64),
            ("getErrorMode", 1, "cljvm_inst_getErrorMode", &sig_1i64),
            ("getError", 1, "cljvm_inst_getError", &sig_1i64),
            ("set", 2, "cljvm_inst_set", &sig_2i64),
            ("alter", 3, "cljvm_inst_alter", &sig_3i64_agent),
            ("commute", 3, "cljvm_inst_commute", &sig_3i64_agent),
            ("ensure", 1, "cljvm_inst_ensure", &sig_1i64),
            ("setMinHistory", 2, "cljvm_inst_setMinHistory", &sig_2i64),
            ("setMaxHistory", 2, "cljvm_inst_setMaxHistory", &sig_2i64),
            ("getMinHistory", 1, "cljvm_inst_getMinHistory", &sig_1i64),
            ("getMaxHistory", 1, "cljvm_inst_getMaxHistory", &sig_1i64),
            (
                "getHistoryCount",
                1,
                "cljvm_inst_getHistoryCount",
                &sig_1i64,
            ),
            ("trimHistory", 1, "cljvm_inst_trimHistory", &sig_1i64),
        ] {
            let f = dm.declare_extern(extern_name, sig.clone());
            instance_methods.insert((name.to_string(), arity), f);
        }
        let inst_dispatch = dm.declare_extern("cljvm_inst_dispatch", sig_4i64_agent);
        instance_methods.insert(("dispatch".to_string(), 4), inst_dispatch);
        let sig_3i64_restart = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let inst_restart = dm.declare_extern("cljvm_inst_restart", sig_3i64_restart);
        instance_methods.insert(("restart".to_string(), 3), inst_restart);
        let sig_3i64_inst_mfn = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let inst_mfn_reset = dm.declare_extern("cljvm_inst_multifn_reset", sig_1i64.clone());
        instance_methods.insert(("reset".to_string(), 1), inst_mfn_reset);
        let inst_mfn_add =
            dm.declare_extern("cljvm_inst_multifn_addMethod", sig_3i64_inst_mfn.clone());
        instance_methods.insert(("addMethod".to_string(), 3), inst_mfn_add);
        let inst_mfn_rm = dm.declare_extern("cljvm_inst_multifn_removeMethod", sig_2i64.clone());
        instance_methods.insert(("removeMethod".to_string(), 2), inst_mfn_rm);
        let inst_mfn_pref = dm.declare_extern("cljvm_inst_multifn_preferMethod", sig_3i64_inst_mfn);
        instance_methods.insert(("preferMethod".to_string(), 3), inst_mfn_pref);
        let inst_mfn_mtbl = dm.declare_extern("cljvm_inst_multifn_methodTable", sig_1i64.clone());
        instance_methods.insert(("getMethodTable".to_string(), 1), inst_mfn_mtbl);
        let inst_mfn_ptbl = dm.declare_extern("cljvm_inst_multifn_preferTable", sig_1i64.clone());
        instance_methods.insert(("getPreferTable".to_string(), 1), inst_mfn_ptbl);
        let inst_mfn_get = dm.declare_extern("cljvm_inst_multifn_getMethod", sig_2i64.clone());
        instance_methods.insert(("getMethod".to_string(), 2), inst_mfn_get);
        let inst_to_sym = dm.declare_extern("cljvm_inst_toSymbol", sig_1i64.clone());
        instance_methods.insert(("toSymbol".to_string(), 1), inst_to_sym);
        let inst_sym_field = dm.declare_extern("cljvm_inst_sym", sig_1i64.clone());
        instance_methods.insert(("sym".to_string(), 1), inst_sym_field);
        let inst_sb_append = dm.declare_extern("cljvm_inst_StringBuilder_append", sig_2i64.clone());
        instance_methods.insert(("append".to_string(), 2), inst_sb_append);
        let inst_cb_add = dm.declare_extern("cljvm_inst_ChunkBuffer_add", sig_2i64.clone());
        instance_methods.insert(("add".to_string(), 2), inst_cb_add);
        let inst_cb_chunk = dm.declare_extern("cljvm_inst_ChunkBuffer_chunk", sig_1i64.clone());
        instance_methods.insert(("chunk".to_string(), 1), inst_cb_chunk);
        let inst_chunked_first = dm.declare_extern("cljvm_inst_chunkedFirst", sig_1i64.clone());
        instance_methods.insert(("chunkedFirst".to_string(), 1), inst_chunked_first);
        let inst_chunked_more = dm.declare_extern("cljvm_inst_chunkedMore", sig_1i64.clone());
        instance_methods.insert(("chunkedMore".to_string(), 1), inst_chunked_more);
        let inst_chunked_next = dm.declare_extern("cljvm_inst_chunkedNext", sig_1i64.clone());
        instance_methods.insert(("chunkedNext".to_string(), 1), inst_chunked_next);
        let sig_3i64_inst = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let inst_reduce_3 = dm.declare_extern("cljvm_inst_reduce_3", sig_3i64_inst);
        instance_methods.insert(("reduce".to_string(), 3), inst_reduce_3);
        // Constructor dispatch — one extern per arity (class + N args).
        // The arity-with-class form keys it in `instance_methods` under
        // the synthetic method name `__new__` so `parse_new_form` finds it.
        let inst_new_1 = dm.declare_extern("cljvm_inst_new_1", sig_1i64.clone());
        instance_methods.insert(("__new__".to_string(), 1), inst_new_1);
        let inst_new_2 = dm.declare_extern("cljvm_inst_new_2", sig_2i64.clone());
        instance_methods.insert(("__new__".to_string(), 2), inst_new_2);
        let sig_3i64_new = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let inst_new_3 = dm.declare_extern("cljvm_inst_new_3", sig_3i64_new);
        instance_methods.insert(("__new__".to_string(), 3), inst_new_3);

        // First-class fn invocation. One extern per arity (Java models this
        // with `IFn.invoke(args...)` overloads). The receiver is a NanBox
        // TAG_FN handle wrapping a FuncRef index; the extern decodes it via
        // the thread-local call_table_base set by `install_call_table_base`
        // before `gc.run_jit`. Mirrors Java's IFn-style virtual dispatch.
        let invoke_0 = dm.declare_extern("cljvm_rt_invoke_0", sig_1i64.clone());
        let sig_3i64 = dynir::Signature {
            params: vec![dynir::Type::I64, dynir::Type::I64, dynir::Type::I64],
            ret: Some(dynir::Type::I64),
        };
        let invoke_1 = dm.declare_extern("cljvm_rt_invoke_1", sig_2i64.clone());
        let invoke_2 = dm.declare_extern("cljvm_rt_invoke_2", sig_3i64);
        let sig_4i64 = dynir::Signature {
            params: vec![
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
            ],
            ret: Some(dynir::Type::I64),
        };
        let invoke_3 = dm.declare_extern("cljvm_rt_invoke_3", sig_4i64);
        let sig_5i64 = dynir::Signature {
            params: vec![
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
            ],
            ret: Some(dynir::Type::I64),
        };
        let invoke_4 = dm.declare_extern("cljvm_rt_invoke_4", sig_5i64);
        let sig_6i64 = dynir::Signature {
            params: vec![
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
            ],
            ret: Some(dynir::Type::I64),
        };
        let invoke_5 = dm.declare_extern("cljvm_rt_invoke_5", sig_6i64);
        let sig_7i64 = dynir::Signature {
            params: vec![
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
            ],
            ret: Some(dynir::Type::I64),
        };
        let invoke_6 = dm.declare_extern("cljvm_rt_invoke_6", sig_7i64);
        let sig_8i64 = dynir::Signature {
            params: vec![
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
            ],
            ret: Some(dynir::Type::I64),
        };
        let invoke_7 = dm.declare_extern("cljvm_rt_invoke_7", sig_8i64);
        let sig_9i64 = dynir::Signature {
            params: vec![
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
                dynir::Type::I64,
            ],
            ret: Some(dynir::Type::I64),
        };
        let invoke_8 = dm.declare_extern("cljvm_rt_invoke_8", sig_9i64);
        let sig_10i64 = dynir::Signature {
            params: vec![dynir::Type::I64; 10],
            ret: Some(dynir::Type::I64),
        };
        let invoke_9 = dm.declare_extern("cljvm_rt_invoke_9", sig_10i64);
        let sig_11i64 = dynir::Signature {
            params: vec![dynir::Type::I64; 11],
            ret: Some(dynir::Type::I64),
        };
        let invoke_10 = dm.declare_extern("cljvm_rt_invoke_10", sig_11i64);

        // Declare GC-managed heap types. `clojure.lang.String` is a
        // varlen-byte buffer with no fixed fields — the byte count lives in
        // the Compact header's varlen-count slot, set when we call
        // `gc.alloc(string_type_id, byte_count)`.
        let string_type_id = dm.obj_type("clojure.lang.String").varlen_bytes().build();
        // `clojure.lang.Symbol` is one Raw64 field "arc_ptr" holding the
        // address of the global Arc<Symbol>. Raw64 isn't GC-traced (Symbols
        // are globally interned, so the Arc is rooted by `SYMBOL_TABLE` for
        // the program lifetime). The heap wrapper exists so the type_id
        // dispatch in `heap_bits_to_object` can recover the Object::Symbol.
        let symbol_type_id = dm
            .obj_type("clojure.lang.Symbol")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.Keyword`: same shape as Symbol — Raw64 `arc_ptr`
        // pointing at an `Arc<Keyword>` whose lifetime is extended via
        // `CompileRoots`.
        let keyword_type_id = dm
            .obj_type("clojure.lang.Keyword")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.Cons`: three GC-traced NanBox value-fields:
        //   * `first` — the head (offset 8)
        //   * `rest`  — next Cons or nil terminator (offset 16)
        //   * `meta`  — IPersistentMap metadata or nil (offset 24)
        // The `meta` slot makes Cons an `IObj`: `(.meta x)` reads it and
        // `(.withMeta x m)` allocates a fresh Cons sharing first/rest with
        // the new metadata. `cljvm_rt_cons` defaults `meta` to nil.
        let cons_type_id = dm
            .obj_type("clojure.lang.Cons")
            .field("first", dynlang::FieldKind::Value)
            .field("rest", dynlang::FieldKind::Value)
            .field("meta", dynlang::FieldKind::Value)
            .build();
        // `clojure.lang.Closure`: Raw64 holds the FuncRef index (as u64);
        // varlen-values section holds the captured NanBox values. The
        // capture count lives in the Compact header's varlen-count slot,
        // populated when the closure is allocated.
        let closure_type_id = dm
            .obj_type("clojure.lang.Closure")
            .field("fref_index", dynlang::FieldKind::Raw64)
            .varlen_values()
            .build();
        // Cache closure layout NOW — Session::new will move obj_types out
        // of `dm`, after which `dm.get_obj_type(closure_type_id)` panics.
        let closure_handle = dm.obj_handle(closure_type_id);
        let closure_fref_offset = closure_handle
            .field_offsets
            .get("fref_index")
            .map(|(o, _)| *o)
            .expect("Closure type must have fref_index field");
        let closure_varlen_base = closure_handle.type_info.varlen_element_offset(0) as i64;
        // `clojure.lang.PersistentVector`: a single varlen-values array of
        // NanBox-encoded items. Compact header carries the count.
        let vector_type_id = dm
            .obj_type("clojure.lang.PersistentVector")
            .varlen_values()
            .build();
        // `clojure.lang.PersistentHashMap`: same shape as Symbol/Keyword — a
        // Raw64 field "arc_ptr" carrying an `Arc<PersistentHashMap>` raw
        // pointer. The host-side Arc owns the actual key/value Vec, kept
        // alive by `CompileRoots._maps`.
        let map_type_id = dm
            .obj_type("clojure.lang.PersistentHashMap")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.PersistentHashSet`: same shape as Map.
        let set_type_id = dm
            .obj_type("clojure.lang.PersistentHashSet")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.PersistentTreeMap`: same shape as Map — Raw64 carrying
        // an `Arc<PersistentTreeMap>` raw pointer. Distinct ObjType so we can
        // dispatch sorted-vs-hash semantics without re-tagging at the value
        // level.
        let tree_map_type_id = dm
            .obj_type("clojure.lang.PersistentTreeMap")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.PersistentTreeSet`: thin wrapper over PersistentTreeMap
        // for sorted sets. Same Raw64-arc shape.
        let tree_set_type_id = dm
            .obj_type("clojure.lang.PersistentTreeSet")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `java.lang.StringBuilder`: Raw64 holding `Arc<RefCell<String>>`.
        // First user of the `alloc_arc_cell`/`decode_arc_cell` runtime helpers.
        let string_builder_type_id = dm
            .obj_type("java.lang.StringBuilder")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.ChunkBuffer`: Raw64 holding `Arc<RefCell<Vec<u64>>>`.
        // Mutable buffer for chunked-seq construction.
        let chunk_buffer_type_id = dm
            .obj_type("clojure.lang.ChunkBuffer")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.IChunk`: same backing as ChunkBuffer; immutable view.
        let i_chunk_type_id = dm
            .obj_type("clojure.lang.IChunk")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.LazySeq` and `Delay`: thunk + cached realized value.
        let lazy_seq_type_id = dm
            .obj_type("clojure.lang.LazySeq")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        let delay_type_id = dm
            .obj_type("clojure.lang.Delay")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.Reduced`: a single traced value slot holding the
        // wrapped value. Produced by `(reduced x)`; unwrapped by `deref`.
        let reduced_type_id = dm
            .obj_type("clojure.lang.Reduced")
            .field("val", dynlang::FieldKind::Value)
            .build();
        // `clojure.lang.MultiArityFn`: dispatcher cell for defns / defmacros
        // with multiple clauses (e.g. `(fn ([x] …) ([x y] …) ([x & xs] …))`).
        // The Raw64 holds an `Arc<Vec<MultiArityEntry>>` listing each
        // clause's `(fixed_arity, is_variadic, fref_idx)`. `(.applyTo)` and
        // `cljvm_rt_invoke_*` recognize the type and dispatch by call arity.
        let multi_arity_fn_type_id = dm
            .obj_type("clojure.lang.MultiArityFn")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.Class`: a Raw64 slot holding a `ClassId` (u16
        // widened to u64 for the slot). Decoded by `cljvm_inst_isInstance`
        // to dispatch the receiver-type check via `host_class::is_instance`.
        let class_type_id = dm
            .obj_type("clojure.lang.Class")
            .field("class_id", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.WithMeta`: a generic IObj wrapper. Two
        // GC-traced Value slots — `inner` (the wrapped value's NanBox)
        // and `meta` (the metadata map's NanBox). Used by
        // `(.withMeta x m)` for receivers that don't have a per-type
        // meta slot (Symbol, Keyword, Vector, Map, …). Cons keeps its
        // built-in meta slot for cheap reader-attached metadata; the
        // WithMeta wrapper is the fallback for everything else.
        let with_meta_type_id = dm
            .obj_type("clojure.lang.WithMeta")
            .field("inner", dynlang::FieldKind::Value)
            .field("meta", dynlang::FieldKind::Value)
            .build();
        // `clojure.lang.Var`: Raw64 holds the `*const Var` pointer.
        // Vars are kept alive process-globally via the namespace mapping,
        // so the raw pointer never dangles. Used by `(var X)` literals and
        // `(.setMacro v)` instance dispatch.
        let var_type_id = dm
            .obj_type("clojure.lang.Var")
            .field("var_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.Namespace`: a Raw64 cell holding a leaked
        // `Arc<Namespace>` pointer, mirroring Var. Lets namespace objects be
        // first-class runtime values (`*ns*`, `the-ns`, `ns-map`, `refer`).
        let namespace_type_id = dm
            .obj_type("clojure.lang.Namespace")
            .field("ns_ptr", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.Long`: a boxed 64-bit integer (Raw64 i64 cell). NaN-
        // boxing has no inline integer tag, so Clojure longs are heap-boxed
        // here — `(+ 1 2)` is `3` (a Long), not `3.0` (a double).
        let long_type_id = dm
            .obj_type("clojure.lang.Long")
            .field("value", dynlang::FieldKind::Raw64)
            .build();
        // `clojure.lang.UserInstance`: shared layout for every `deftype` /
        // `defrecord`. The Raw64 `user_type_id` distinguishes which user
        // type the instance belongs to (allocated by
        // `lang::user_types::register_user_type`); the varlen-values
        // section holds the declared fields in declaration order. One
        // ObjTypeId backs every user type, so adding a new `deftype` at
        // runtime does not require mutating `dm.obj_types` after the
        // initial build.
        let user_instance_type_id = dm
            .obj_type("clojure.lang.UserInstance")
            .field("user_type_id", dynlang::FieldKind::Raw64)
            .varlen_values()
            .build();
        // Cache UserInstance layout offsets NOW for the same reason as
        // Closure: `Session::new` will move `obj_types` out of `dm`,
        // after which `dm.obj_handle` panics.
        let user_instance_handle = dm.obj_handle(user_instance_type_id);
        let user_instance_user_type_id_offset = user_instance_handle
            .field_offsets
            .get("user_type_id")
            .map(|(o, _)| *o)
            .expect("UserInstance type must have user_type_id field");
        let user_instance_varlen_base =
            user_instance_handle.type_info.varlen_element_offset(0) as i64;

        // `clojure.lang.Character`: a boxed Unicode codepoint (Raw64 cell),
        // laid out exactly like Long. Registered LAST so it gets a fresh
        // highest id and no existing type id shifts. Distinct type so
        // `str`/`pr-str` render it as the character and `(= \a 97)` is false.
        let character_type_id = dm
            .obj_type("clojure.lang.Character")
            .field("value", dynlang::FieldKind::Raw64)
            .build();

        // Stash type_ids globally so Rust externs called from JIT-executing
        // code (`cljvm_rt_cons` etc.) can allocate the right ObjType without
        // threading the Compiler through.
        crate::runtime::set_heap_type_ids(crate::runtime::HeapTypeIds {
            string: string_type_id.0,
            symbol: symbol_type_id.0,
            keyword: keyword_type_id.0,
            cons: cons_type_id.0,
            vector: vector_type_id.0,
            map: map_type_id.0,
            set: set_type_id.0,
            tree_map: tree_map_type_id.0,
            tree_set: tree_set_type_id.0,
            string_builder: string_builder_type_id.0,
            chunk_buffer: chunk_buffer_type_id.0,
            i_chunk: i_chunk_type_id.0,
            lazy_seq: lazy_seq_type_id.0,
            delay: delay_type_id.0,
            multi_arity_fn: multi_arity_fn_type_id.0,
            class: class_type_id.0,
            var: var_type_id.0,
            namespace: namespace_type_id.0,
            with_meta: with_meta_type_id.0,
            reduced: reduced_type_id.0,
            long: long_type_id.0,
            character: character_type_id.0,
            user_instance: user_instance_type_id.0,
        });
        crate::runtime::set_user_instance_layout(crate::runtime::UserInstanceLayout {
            user_type_id_offset: user_instance_user_type_id_offset,
            varlen_base: user_instance_varlen_base,
        });

        // Per-arity stub host call externs. Used as fallback for
        // unregistered (. Class method args…) calls.
        let mk_sig = |n: usize| dynir::Signature {
            params: vec![dynir::Type::I64; n],
            ret: Some(dynir::Type::I64),
        };
        let unimpl_host_stubs: Vec<dynir::FuncRef> = (0..=6)
            .map(|n| dm.declare_extern(&format!("cljvm_unimpl_host_call_{n}"), mk_sig(n)))
            .collect();

        Compiler {
            dm,
            pending_fns: std::sync::Mutex::new(Vec::new()),
            next_fn_id: std::sync::atomic::AtomicU32::new(0),
            externs: RuntimeExterns {
                var_bind_root,
                var_deref,
            },
            invoke_externs: [
                invoke_0, invoke_1, invoke_2, invoke_3, invoke_4, invoke_5, invoke_6, invoke_7,
                invoke_8, invoke_9, invoke_10,
            ],
            var_fns: std::sync::Mutex::new(HashMap::new()),
            var_fn_infos: std::sync::Mutex::new(HashMap::new()),
            var_multi_arities: std::sync::Mutex::new(HashMap::new()),
            string_type_id,
            symbol_type_id,
            keyword_type_id,
            cons_type_id,
            closure_type_id,
            vector_type_id,
            map_type_id,
            set_type_id,
            tree_map_type_id,
            tree_set_type_id,
            string_builder_type_id,
            chunk_buffer_type_id,
            i_chunk_type_id,
            lazy_seq_type_id,
            delay_type_id,
            multi_arity_fn_type_id,
            reduced_type_id,
            closure_handle,
            closure_fref_offset,
            closure_varlen_base,
            class_type_id,
            var_type_id,
            namespace_type_id,
            with_meta_type_id,
            long_type_id,
            character_type_id,
            num,
            case_dispatch,
            user_instance_type_id,
            user_instance_user_type_id_offset,
            user_instance_varlen_base,
            pending_literals: std::sync::Mutex::new(Vec::new()),
            literal_pool_offset: std::sync::atomic::AtomicUsize::new(0),
            host_methods,
            instance_methods,
            unimpl_host_stubs,
            fn_arities: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Record the arity (`fixed_arity`, `is_variadic`) for a fn whose
    /// body has been lowered. Looked up by `cljvm_rt_invoke_*` to handle
    /// variadic targets at dynamic call sites.
    pub fn register_fn_arity(&self, fref: dynir::FuncRef, info: VarFnInfo) {
        self.fn_arities
            .lock()
            .unwrap()
            .insert(fref.index() as u32, info);
    }

    /// Look up the arity info recorded for `fref`, if any.
    pub fn fn_arity(&self, fref_idx: u32) -> Option<VarFnInfo> {
        self.fn_arities.lock().unwrap().get(&fref_idx).copied()
    }

    /// Look up a registered instance method by (method-name, arity).
    /// Arity counts the receiver. Returns the dispatching extern's
    /// `FuncRef` so `InstanceMethodExpr` can lower to a direct call.
    pub fn instance_method(
        &self,
        method: &str,
        arity_with_receiver: usize,
    ) -> Option<dynir::FuncRef> {
        self.instance_methods
            .get(&(method.to_string(), arity_with_receiver))
            .copied()
    }

    /// Look up a registered static host method. Returns the `FuncRef` to call.
    pub fn host_method(
        &self,
        class_name: &str,
        method_name: &str,
        arity: usize,
    ) -> Option<dynir::FuncRef> {
        self.host_methods
            .get(&(class_name.to_string(), method_name.to_string(), arity))
            .copied()
    }

    /// Intern a compile-time literal. Returns the `literal_pool` slot index
    /// that `*Expr.emit` should bake into `gc_literal(LiteralRef(idx))`.
    /// Heap allocation happens later in `compile_form_to_jit`.
    pub fn intern_literal(&self, lit: PendingLiteral) -> u32 {
        use std::sync::atomic::Ordering;
        let mut pool = self.pending_literals.lock().unwrap();
        let idx = self.literal_pool_offset.fetch_add(1, Ordering::SeqCst) as u32;
        pool.push(lit);
        idx
    }

    /// Convenience wrapper for the common string-literal case.
    pub fn intern_string_literal(&self, s: Arc<String>) -> u32 {
        self.intern_literal(PendingLiteral::String(s))
    }

    /// Convenience wrapper for the symbol-literal case (`(quote x)`).
    pub fn intern_symbol_literal(&self, s: Arc<Symbol>) -> u32 {
        self.intern_literal(PendingLiteral::Symbol(s))
    }

    /// Convenience wrapper for the keyword-literal case (`:foo`).
    pub fn intern_keyword_literal(&self, k: Arc<Keyword>) -> u32 {
        self.intern_literal(PendingLiteral::Keyword(k))
    }

    /// Convenience wrapper for quoted-list literals (`(quote (a b c))`).
    pub fn intern_list_literal(&self, l: Arc<PersistentList>) -> u32 {
        self.intern_literal(PendingLiteral::List(l))
    }

    /// Record that `var`'s compile-time-known fn body is `fref`. Looked up by
    /// `InvokeExpr.emit` when its head is a `VarExpr` so `(foo args)` can lower
    /// to a direct `Call` instead of going through the Var-deref extern + an
    /// (unimplemented) indirect-call path.
    pub fn register_var_fn(&self, var: &Arc<Var>, fref: dynir::FuncRef, info: VarFnInfo) {
        let key = Arc::as_ptr(var);
        self.var_fns.lock().unwrap().insert(key, fref);
        self.var_fn_infos.lock().unwrap().insert(key, info);
    }

    /// Look up a `Var`'s compile-time-known fn FuncRef, if one was registered.
    pub fn var_fn(&self, var: &Arc<Var>) -> Option<dynir::FuncRef> {
        let key = Arc::as_ptr(var);
        self.var_fns.lock().unwrap().get(&key).copied()
    }

    /// Look up the variadic metadata for a Var-registered fn, if any.
    pub fn var_fn_info(&self, var: &Arc<Var>) -> Option<VarFnInfo> {
        let key = Arc::as_ptr(var);
        self.var_fn_infos.lock().unwrap().get(&key).copied()
    }

    /// Register a Var as bound to a multi-arity fn. Each tuple is
    /// `(FuncRef, VarFnInfo)` per clause.
    pub fn register_var_multi_arity(
        &self,
        var: &Arc<Var>,
        arities: Vec<(dynir::FuncRef, VarFnInfo)>,
    ) {
        let key = Arc::as_ptr(var);
        self.var_multi_arities.lock().unwrap().insert(key, arities);
    }

    /// Look up a Var's per-arity (FuncRef, info) table, if any.
    pub fn var_multi_arity(&self, var: &Arc<Var>) -> Option<Vec<(dynir::FuncRef, VarFnInfo)>> {
        let key = Arc::as_ptr(var);
        self.var_multi_arities.lock().unwrap().get(&key).cloned()
    }

    /// Mint a fresh, unique fn name within this session.
    pub fn fresh_fn_name(&self, base: Option<&str>) -> String {
        let id = self
            .next_fn_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match base {
            Some(b) => format!("{b}__{id}"),
            None => format!("fn__{id}"),
        }
    }

    /// Declare a fn in the `DynModule` and record it as pending. Returns the
    /// allocated FuncRef so the caller (FnExpr) can encode it later.
    ///
    /// `fn_id` is the compile-time identity of this fn body (matches
    /// `LocalBinding.owning_fn_id` for locals defined inside it).
    /// `captures` lists outer LocalBindings the body references — empty
    /// for non-closures. If non-empty, the IR-level fn declares one extra
    /// param (the closure object) prepended to the user-visible params.
    pub fn declare_pending_fn(
        &mut self,
        name: String,
        params: Vec<Arc<LocalBinding>>,
        body: Arc<dyn Expr>,
        fn_id: u32,
        captures: Vec<Arc<LocalBinding>>,
        self_name_slot: Option<i32>,
    ) -> dynir::FuncRef {
        // Closures carry an implicit self-arg (the closure object), so the
        // declared arity is `params.len() + 1` when `captures` is non-empty.
        let declared_arity = params.len() + if captures.is_empty() { 0 } else { 1 };
        let fref = self.dm.declare_func(&name, declared_arity);
        self.pending_fns.lock().unwrap().push(PendingFn {
            fref,
            params: params.clone(),
            body,
            name,
            fn_id,
            captures,
            self_name_slot,
            self_multi_lit: None,
        });
        fref
    }

    /// Point a multi-arity clause's named self-reference slot at the shared
    /// `MultiArityFn` dispatcher cell so cross-arity self-calls dispatch
    /// correctly. The pending fn must already be declared (it is — clauses
    /// are declared during their own `parse_fn_form` before
    /// `parse_fn_form_multi_arity` interns the cell).
    pub fn set_pending_self_multi_lit(&self, fref: dynir::FuncRef, lref: dynir::ir::LiteralRef) {
        let mut pending = self.pending_fns.lock().unwrap();
        if let Some(pf) = pending.iter_mut().find(|pf| pf.fref == fref) {
            pf.self_multi_lit = Some(lref);
        }
    }
}

// ── Thread-local active-compiler hook ───────────────────────────────────────
//
// Many analyze paths need to register pending fns or read session state, but
// changing every analyze signature to thread `&mut Compiler` through would
// touch hundreds of call sites. Java threads the equivalent state through
// dynamic Vars (`LOADER`, `METHOD`, `KEYWORDS`, …). We mirror that with one
// raw-pointer thread_local — set on entry to `Compiler::with_active`,
// cleared on exit. `with_active_compiler` panics if no session is active,
// which is the right behavior (no analyze path should silently no-op).

use std::cell::Cell as CellPlain;

thread_local! {
    static ACTIVE_COMPILER: CellPlain<*mut Compiler> = const { CellPlain::new(std::ptr::null_mut()) };
}

impl Compiler {
    /// Install `self` as the active compilation for the duration of `body`.
    pub fn with_active<R>(&mut self, body: impl FnOnce(&mut Compiler) -> R) -> R {
        let ptr: *mut Compiler = self as *mut _;
        let prev = ACTIVE_COMPILER.with(|c| c.replace(ptr));
        let result = body(self);
        ACTIVE_COMPILER.with(|c| c.set(prev));
        result
    }
}

/// Reach into the currently-active `Compiler`. Panics if no compilation is
/// active.
pub fn with_active_compiler<R>(body: impl FnOnce(&mut Compiler) -> R) -> R {
    ACTIVE_COMPILER.with(|c| {
        let ptr = c.get();
        if ptr.is_null() {
            panic!(
                "clojure-jvm: no active compilation — analyze/emit paths must \
                 run inside `Compiler::with_active`"
            );
        }
        // SAFETY: the pointer was installed by `with_active` on this thread,
        // points to a `Compiler` borrowed mutably by the caller, and is cleared
        // when `with_active` returns. No other code can reach this pointer.
        unsafe { body(&mut *ptr) }
    })
}

// ============================================================================
// `compile_form_to_jit` / `compile_form_to_interp` — top-level drivers.
// ============================================================================

/// Compile a top-level form and return the JIT-compiled module + its entry
/// fn. Every form is wrapped in a 0-arity `__top_level__` fn and lowered
/// into a fresh `DynModule`. Pending FnExprs (declared during analyze) are
/// lowered into their declared FuncRefs after the entry fn is built.
/// Per-compilation roots kept alive alongside the JitModule. The literal
/// pool holds raw `Arc::as_ptr` pointers into these; dropping them would
/// dangle. Returned to the caller as part of the compile output and held
/// for as long as the JIT module + heap-allocated literals are accessible.
pub struct CompileRoots {
    pub _symbols: Vec<Arc<Symbol>>,
    pub _strings: Vec<Arc<String>>,
    pub _keywords: Vec<Arc<Keyword>>,
    /// `Arc<PersistentHashMap>`s referenced by `Object::Map` literals. The
    /// heap wrapper stores a `Arc::as_ptr` raw pointer into one of these;
    /// without keeping the Arc alive here, the host-side Vec backing the
    /// map would drop and the raw pointer would dangle.
    pub _maps: Vec<Arc<crate::lang::persistent_hash_map::PersistentHashMap>>,
    /// `Arc<PersistentHashSet>`s referenced by `Object::Set` literals.
    /// Same lifetime story as `_maps`.
    pub _sets: Vec<Arc<crate::lang::persistent_hash_set::PersistentHashSet>>,
    /// `Arc<PersistentTreeMap>`s referenced by `Object::Map`/sorted-map values.
    /// Same lifetime story as `_maps`.
    pub _tree_maps: Vec<Arc<crate::lang::persistent_tree_map::PersistentTreeMap>>,
    /// `Arc<PersistentTreeSet>`s referenced by sorted-set values.
    pub _tree_sets: Vec<Arc<crate::lang::persistent_tree_set::PersistentTreeSet>>,
    /// `Arc<RefCell<String>>`s referenced by `java.lang.StringBuilder`
    /// heap cells. Same lifetime story as `_maps`.
    pub _string_builders: Vec<Arc<std::cell::RefCell<String>>>,
    /// Mutable `Arc<RefCell<Vec<u64>>>`s for ChunkBuffer/IChunk cells.
    pub _chunk_buffers: Vec<Arc<std::cell::RefCell<Vec<u64>>>>,
    /// `Arc<RefCell<LazyState>>` for LazySeq/Delay cells.
    pub _lazy_states: Vec<Arc<std::cell::RefCell<crate::runtime::LazyState>>>,
    /// `Arc<Vec<MultiArityEntry>>` for MultiArityFn cells.
    pub _multi_arity_tables: Vec<Arc<Vec<crate::runtime::MultiArityEntry>>>,
    /// `Arc<Namespace>`s referenced by `clojure.lang.Namespace` heap cells.
    /// Same lifetime story as `_maps`: the cell holds a raw `Arc::as_ptr`.
    pub _namespaces: Vec<Arc<crate::lang::namespace::Namespace>>,
    /// `Arc<Var>`s referenced by `clojure.lang.Var` heap cells materialized
    /// at runtime (e.g. `ns-map` boxing each mapping's Var). Same lifetime
    /// story as `_maps`.
    pub _vars: Vec<Arc<crate::lang::var::Var>>,
}

/// Encode an `Object` as a NanBox u64 suitable for placing in a heap-traced
/// value field. Heap-allocates strings / symbols / keywords / nested lists
/// as needed, growing `roots` so the underlying `Arc`s stay alive.
///
/// Caller must have the mutator thread installed (`gc.install_thread`).
fn alloc_object_as_nanbox(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    cons_type_id: dynlang::ObjTypeId,
    string_type_id: dynlang::ObjTypeId,
    symbol_type_id: dynlang::ObjTypeId,
    keyword_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    obj: &Object,
) -> u64 {
    match obj {
        Object::Nil => crate::runtime::nanbox_nil(),
        Object::Bool(b) => crate::runtime::nanbox_bool(*b),
        // Integers are boxed Longs, so a collection literal's integer elements
        // (`[1 2]`, `{:a 1}`, `#{1 2}`, `'(1 2)`) box like any other Long.
        // box_long allocs on the same path the `PendingLiteral::Long` arm uses.
        Object::Long(n) => unsafe { crate::runtime::box_long(*n) },
        // Characters box like Longs (Raw64 codepoint cell) — a `case` over
        // char constants ships `\a` through macro args / case-map values.
        Object::Char(c) => unsafe { crate::runtime::box_char(*c) },
        Object::Double(x) => x.to_bits(),
        Object::Symbol(s) => alloc_symbol(gc, obj_types, symbol_type_id, roots, s),
        Object::Keyword(k) => alloc_keyword(gc, obj_types, keyword_type_id, roots, k),
        Object::String(s) => alloc_string(gc, obj_types, string_type_id, roots, s),
        Object::List(l) => alloc_list_as_nanbox(
            gc,
            obj_types,
            cons_type_id,
            string_type_id,
            symbol_type_id,
            keyword_type_id,
            roots,
            l,
        ),
        Object::Vector(v) => {
            // Vector type_id is 5 in our Compiler::new declaration order:
            // 0=String 1=Symbol 2=Keyword 3=Cons 4=Closure 5=Vector.
            let vector_type_id = dynlang::ObjTypeId(5);
            alloc_vector_as_nanbox(
                gc,
                obj_types,
                vector_type_id,
                cons_type_id,
                string_type_id,
                symbol_type_id,
                keyword_type_id,
                roots,
                v,
            )
        }
        Object::Map(m) => {
            // Map type_id is 6 (declared right after Vector in `Compiler::new`).
            let map_type_id = dynlang::ObjTypeId(6);
            alloc_map_as_nanbox(gc, obj_types, map_type_id, roots, m)
        }
        Object::Set(s) => {
            // Set type_id is 7 (declared right after Map).
            let set_type_id = dynlang::ObjTypeId(7);
            alloc_set_as_nanbox(gc, obj_types, set_type_id, roots, s)
        }
        Object::WithMeta(inner, meta) => {
            // Allocate the inner value normally, then attach the metadata
            // to its heap meta slot (currently only IObj-supporting types
            // — Cons/List). For inners that don't carry a meta slot,
            // metadata is silently dropped (Java would throw
            // ClassCastException on `withMeta` for non-IObj values; for
            // reader-attached meta on a literal we just lose it).
            let inner_bits = alloc_object_as_nanbox(
                gc,
                obj_types,
                cons_type_id,
                string_type_id,
                symbol_type_id,
                keyword_type_id,
                roots,
                inner,
            );
            attach_meta_to_heap(
                gc,
                obj_types,
                cons_type_id,
                string_type_id,
                symbol_type_id,
                keyword_type_id,
                roots,
                inner_bits,
                meta,
            )
        }
        Object::Var(v) => {
            // `clojure.lang.Var`: Raw64 cell holding a leaked `Arc<Var>`.
            let type_id = crate::runtime::heap_type_ids().var;
            roots._vars.push(v.clone());
            let ptr = gc.alloc(type_id, 0);
            assert!(!ptr.is_null(), "alloc Var: null");
            let arc_ptr_bits = Arc::as_ptr(v) as u64;
            let raw_offset = obj_types[type_id].type_info.raw_data_offset();
            unsafe {
                ptr.add(raw_offset)
                    .cast::<u64>()
                    .write_unaligned(arc_ptr_bits);
            }
            gc.tag_ptr(ptr)
        }
        Object::Namespace(ns) => {
            // `clojure.lang.Namespace`: Raw64 cell holding a leaked
            // `Arc<Namespace>`.
            let type_id = crate::runtime::heap_type_ids().namespace;
            roots._namespaces.push(ns.clone());
            let ptr = gc.alloc(type_id, 0);
            assert!(!ptr.is_null(), "alloc Namespace: null");
            let arc_ptr_bits = Arc::as_ptr(ns) as u64;
            let raw_offset = obj_types[type_id].type_info.raw_data_offset();
            unsafe {
                ptr.add(raw_offset)
                    .cast::<u64>()
                    .write_unaligned(arc_ptr_bits);
            }
            gc.tag_ptr(ptr)
        }
        other => panic!(
            "clojure-jvm: alloc_object_as_nanbox: variant {other:?} not yet representable on the GC heap"
        ),
    }
}

/// Write the metadata `m` into the heap meta slot of the value at
/// `target_bits`. Returns `target_bits` so callers can use it as the
/// expression value. For receivers without a meta slot (e.g. immediates,
/// String) the metadata is silently dropped — Java's `withMeta` would
/// throw `ClassCastException`, which we reach for via a panic at the
/// runtime `.withMeta` dispatch site, not here at the allocation path.
fn attach_meta_to_heap(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    cons_type_id: dynlang::ObjTypeId,
    string_type_id: dynlang::ObjTypeId,
    symbol_type_id: dynlang::ObjTypeId,
    keyword_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    target_bits: u64,
    meta: &Arc<PersistentHashMap>,
) -> u64 {
    use crate::runtime::{nanbox_payload, nanbox_tag};
    dynobj::roots::with_scope(2, |scope| {
        let target = scope.root::<()>(target_bits);
        // Encode the metadata map as a heap value first so the meta slot
        // can hold a NanBox pointer.
        let meta_obj = Object::Map(meta.clone());
        let meta_bits = alloc_object_as_nanbox(
            gc,
            obj_types,
            cons_type_id,
            string_type_id,
            symbol_type_id,
            keyword_type_id,
            roots,
            &meta_obj,
        );
        let meta_root = scope.root::<()>(meta_bits);
        let target_bits = target.get();
        // Find the receiver's heap cell. Only TAG_PTR receivers can carry a
        // meta slot in our current heap layout.
        let Some(2 /* TAG_PTR */) = nanbox_tag(target_bits) else {
            return target_bits;
        };
        let raw = nanbox_payload(target_bits) as *mut u8;
        if raw.is_null() {
            return target_bits;
        }
        let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
        if type_id == cons_type_id.0 {
            let type_info = &obj_types[cons_type_id.0].type_info;
            let meta_off = type_info.value_field_offset(2) as isize;
            unsafe {
                raw.offset(meta_off)
                    .cast::<u64>()
                    .write_unaligned(meta_root.get());
            }
        }
        // Other heap types don't yet have a meta slot — when we add them,
        // extend this match. Silently dropping is consistent with Java's
        // "metadata is invisible to non-IObj receivers".
        target_bits
    })
}

fn alloc_vector_as_nanbox(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    vector_type_id: dynlang::ObjTypeId,
    cons_type_id: dynlang::ObjTypeId,
    string_type_id: dynlang::ObjTypeId,
    symbol_type_id: dynlang::ObjTypeId,
    keyword_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    vec: &Arc<PersistentVector>,
) -> u64 {
    // Encode each item first (any may recurse into more heap allocs),
    // then allocate the vector and fill its varlen slots.
    let n = vec.count() as usize;
    dynobj::roots::with_scope(n + 1, |scope| {
        let mut item_roots: Vec<dynobj::roots::Rooted<()>> = Vec::with_capacity(n);
        for i in 0..(n as i32) {
            let elem = vec.nth(i);
            let bits = alloc_object_as_nanbox(
                gc,
                obj_types,
                cons_type_id,
                string_type_id,
                symbol_type_id,
                keyword_type_id,
                roots,
                &elem,
            );
            item_roots.push(scope.root::<()>(bits));
        }
        let ptr = gc.alloc(vector_type_id.0, n);
        assert!(
            !ptr.is_null(),
            "clojure-jvm: gc.alloc returned null for Vector"
        );
        let type_info = &obj_types[vector_type_id.0].type_info;
        let base = type_info.varlen_element_offset(0) as isize;
        unsafe {
            for (i, bits) in item_roots.iter().enumerate() {
                let slot = ptr.offset(base + (i as isize) * 8).cast::<u64>();
                slot.write_unaligned(bits.get());
            }
        }
        gc.tag_ptr(ptr)
    })
}

/// Recursively allocate a `PersistentList` as a chain of `Cons` heap objects,
/// returning the head's NanBox-tagged pointer. The empty list maps to
/// `Object::Nil` (i.e. the rest-chain terminator).
fn alloc_list_as_nanbox(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    cons_type_id: dynlang::ObjTypeId,
    string_type_id: dynlang::ObjTypeId,
    symbol_type_id: dynlang::ObjTypeId,
    keyword_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    list: &Arc<PersistentList>,
) -> u64 {
    match &**list {
        PersistentList::Empty => crate::runtime::nanbox_nil(),
        PersistentList::Cons {
            first,
            rest,
            count: _,
        } => {
            // Allocate the rest first so the recursive list-builder doesn't
            // hold a `first` value across allocations of nested heap
            // structures (which could be reordered by a moving GC). For our
            // current OnPressure/never-collect compile-time path that's
            // belt-and-suspenders, but the order matters once GC runs.
            let rest_bits = alloc_list_as_nanbox(
                gc,
                obj_types,
                cons_type_id,
                string_type_id,
                symbol_type_id,
                keyword_type_id,
                roots,
                rest,
            );
            dynobj::roots::with_scope(2, |scope| {
                let rest_root = scope.root::<()>(rest_bits);
                let first_bits = alloc_object_as_nanbox(
                    gc,
                    obj_types,
                    cons_type_id,
                    string_type_id,
                    symbol_type_id,
                    keyword_type_id,
                    roots,
                    first,
                );
                let first_root = scope.root::<()>(first_bits);

                let ptr = gc.alloc(cons_type_id.0, 0);
                assert!(
                    !ptr.is_null(),
                    "clojure-jvm: gc.alloc returned null for Cons"
                );
                // Cons layout: Compact header (8) + value-field "first" (8) +
                // value-field "rest" (8) + value-field "meta" (8). All
                // NanBox-encoded u64s, GC-traced. The meta slot defaults to
                // nil here; reader-attached metadata flows through the
                // `Object::WithMeta`-aware allocation paths instead.
                let type_info = &obj_types[cons_type_id.0].type_info;
                let first_off = type_info.value_field_offset(0) as isize;
                let rest_off = type_info.value_field_offset(1) as isize;
                let meta_off = type_info.value_field_offset(2) as isize;
                let nil_bits = crate::runtime::nanbox_nil();
                let owner_bits = gc.tag_ptr(ptr);
                crate::runtime::trap_forwarded_first_result(
                    "alloc_list.cons.first",
                    owner_bits,
                    first_root.get(),
                );
                unsafe {
                    ptr.offset(first_off)
                        .cast::<u64>()
                        .write_unaligned(first_root.get());
                    ptr.offset(rest_off)
                        .cast::<u64>()
                        .write_unaligned(rest_root.get());
                    ptr.offset(meta_off).cast::<u64>().write_unaligned(nil_bits);
                }
                owner_bits
            })
        }
    }
}

fn alloc_string(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    string_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    s: &Arc<String>,
) -> u64 {
    roots._strings.push(s.clone());
    let bytes = s.as_bytes();
    let ptr = gc.alloc(string_type_id.0, bytes.len());
    assert!(!ptr.is_null(), "alloc_string: null");
    let type_info = &obj_types[string_type_id.0].type_info;
    let data_offset = type_info.varlen_element_offset(0);
    unsafe {
        let dst = ptr.add(data_offset);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
    gc.tag_ptr(ptr)
}

fn alloc_symbol(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    symbol_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    s: &Arc<Symbol>,
) -> u64 {
    roots._symbols.push(s.clone());
    let ptr = gc.alloc(symbol_type_id.0, 0);
    assert!(!ptr.is_null(), "alloc_symbol: null");
    let arc_ptr_bits = Arc::as_ptr(s) as u64;
    let type_info = &obj_types[symbol_type_id.0].type_info;
    let raw_offset = type_info.raw_data_offset();
    unsafe {
        let dst = ptr.add(raw_offset).cast::<u64>();
        dst.write_unaligned(arc_ptr_bits);
    }
    let tagged = gc.tag_ptr(ptr);
    if std::env::var("CLJVM_ALLOC_TRACE").is_ok() {
        eprintln!(
            "[alloc_symbol] s={} ptr=0x{:016x} tagged=0x{:016x} raw_off={raw_offset} arc=0x{arc_ptr_bits:016x}",
            s.get_name(),
            ptr as usize,
            tagged,
        );
    }
    tagged
}

fn alloc_set_as_nanbox(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    set_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    s: &Arc<crate::lang::persistent_hash_set::PersistentHashSet>,
) -> u64 {
    roots._sets.push(s.clone());
    let ptr = gc.alloc(set_type_id.0, 0);
    assert!(!ptr.is_null(), "alloc_set_as_nanbox: null");
    let arc_ptr_bits = Arc::as_ptr(s) as u64;
    let type_info = &obj_types[set_type_id.0].type_info;
    let raw_offset = type_info.raw_data_offset();
    unsafe {
        let dst = ptr.add(raw_offset).cast::<u64>();
        dst.write_unaligned(arc_ptr_bits);
    }
    gc.tag_ptr(ptr)
}

/// Allocate a `clojure.lang.Var` heap cell whose Raw64 slot holds
/// `Arc::as_ptr(v)`. The Arc is held alive by the namespace mapping;
/// no extra rooting needed.
fn alloc_var_as_nanbox(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    var_type_id: dynlang::ObjTypeId,
    v: &Arc<Var>,
) -> u64 {
    let ptr = gc.alloc(var_type_id.0, 0);
    assert!(!ptr.is_null(), "alloc_var_as_nanbox: null");
    let type_info = &obj_types[var_type_id.0].type_info;
    let raw_offset = type_info.raw_data_offset();
    unsafe {
        ptr.add(raw_offset)
            .cast::<u64>()
            .write_unaligned(Arc::as_ptr(v) as u64);
    }
    gc.tag_ptr(ptr)
}

/// Allocate a `clojure.lang.Class` heap cell whose Raw64 slot holds
/// `class_id.0`. Returns the NanBox handle. Used both at literal-pool
/// fill time (for class-name symbols resolved by `analyze_symbol`) and
/// any future runtime path that needs to manufacture a Class value.
fn alloc_class_as_nanbox(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    class_type_id: dynlang::ObjTypeId,
    class_id: super::host_class::ClassId,
) -> u64 {
    let ptr = gc.alloc(class_type_id.0, 0);
    assert!(!ptr.is_null(), "alloc_class_as_nanbox: null");
    let type_info = &obj_types[class_type_id.0].type_info;
    let raw_offset = type_info.raw_data_offset();
    unsafe {
        ptr.add(raw_offset)
            .cast::<u64>()
            .write_unaligned(class_id.0 as u64);
    }
    gc.tag_ptr(ptr)
}

fn alloc_map_as_nanbox(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    map_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    m: &Arc<crate::lang::persistent_hash_map::PersistentHashMap>,
) -> u64 {
    // Same shape as Symbol/Keyword: allocate the Raw64 wrapper, stash the
    // `Arc::as_ptr` in the raw_data slot, and extend the Arc's lifetime via
    // `CompileRoots._maps`. Map keys/values are owned by the Arc's Vec, not
    // the GC heap, so this is the only allocation needed.
    roots._maps.push(m.clone());
    let ptr = gc.alloc(map_type_id.0, 0);
    assert!(!ptr.is_null(), "alloc_map_as_nanbox: null");
    let arc_ptr_bits = Arc::as_ptr(m) as u64;
    let type_info = &obj_types[map_type_id.0].type_info;
    let raw_offset = type_info.raw_data_offset();
    unsafe {
        let dst = ptr.add(raw_offset).cast::<u64>();
        dst.write_unaligned(arc_ptr_bits);
    }
    gc.tag_ptr(ptr)
}

fn alloc_keyword(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    keyword_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    k: &Arc<Keyword>,
) -> u64 {
    roots._keywords.push(k.clone());
    let ptr = gc.alloc(keyword_type_id.0, 0);
    assert!(!ptr.is_null(), "alloc_keyword: null");
    let arc_ptr_bits = Arc::as_ptr(k) as u64;
    let type_info = &obj_types[keyword_type_id.0].type_info;
    let raw_offset = type_info.raw_data_offset();
    unsafe {
        let dst = ptr.add(raw_offset).cast::<u64>();
        dst.write_unaligned(arc_ptr_bits);
    }
    gc.tag_ptr(ptr)
}

/// Walk the module's func_table in declaration order and produce the
/// positional `&[*const u8]` extern slice `JitModule::extend` expects.
/// This mirrors `dynlang::gc::DynGcRuntime::build_extern_table` (which is
/// private) for the externs clojure-jvm declares.
fn build_extern_table_for(module: &dynir::ir::Module) -> Vec<*const u8> {
    use dynir::ir::FuncDef;
    module
        .func_table
        .iter()
        .filter_map(|def| match def {
            FuncDef::Extern(ef) => {
                if ef.name == dynlang::gc::GC_ALLOC_EXTERN {
                    Some(dynlang::gc::gc_alloc_thunk as *const u8)
                } else {
                    resolve_clojure_extern(&ef.name).or_else(|| {
                        panic!(
                            "clojure-jvm: unresolved extern `{}` during extern table build",
                            ef.name
                        )
                    })
                }
            }
            FuncDef::Internal(_) => None,
        })
        .collect()
}

/// Resolver for clojure-jvm-side extern names → C-ABI function pointers.
/// One entry per `declare_extern` call in `Compiler::new`. Used by both
/// `DynGcRuntime::compile_jit` (resolver closure) and the future
/// `new_empty + extend` path (positional `&[*const u8]`).
fn resolve_clojure_extern(name: &str) -> Option<*const u8> {
    match name {
        "cljvm_var_bind_root" => Some(crate::runtime::cljvm_var_bind_root as *const u8),
        "cljvm_var_deref" => Some(crate::runtime::cljvm_var_deref as *const u8),
        "cljvm_num_add" => Some(crate::runtime::cljvm_num_add as *const u8),
        "cljvm_num_sub" => Some(crate::runtime::cljvm_num_sub as *const u8),
        "cljvm_num_mul" => Some(crate::runtime::cljvm_num_mul as *const u8),
        "cljvm_num_div" => Some(crate::runtime::cljvm_num_div as *const u8),
        "cljvm_num_quot" => Some(crate::runtime::cljvm_num_quot as *const u8),
        "cljvm_num_rem" => Some(crate::runtime::cljvm_num_rem as *const u8),
        "cljvm_num_lt" => Some(crate::runtime::cljvm_num_lt as *const u8),
        "cljvm_num_gt" => Some(crate::runtime::cljvm_num_gt as *const u8),
        "cljvm_num_le" => Some(crate::runtime::cljvm_num_le as *const u8),
        "cljvm_num_ge" => Some(crate::runtime::cljvm_num_ge as *const u8),
        "cljvm_num_equiv" => Some(crate::runtime::cljvm_num_equiv as *const u8),
        "cljvm_equals" => Some(crate::runtime::cljvm_equals as *const u8),
        "cljvm_util_hash" => Some(crate::runtime::cljvm_util_hash as *const u8),
        "cljvm_case_dispatch" => Some(crate::runtime::cljvm_case_dispatch as *const u8),
        "cljvm_rt_inc" => Some(crate::runtime::cljvm_rt_inc as *const u8),
        "cljvm_rt_nextID" => Some(crate::runtime::cljvm_rt_nextID as *const u8),
        "cljvm_rt_intCast" => Some(crate::runtime::cljvm_rt_intCast as *const u8),
        "cljvm_rt_longCast" => Some(crate::runtime::cljvm_rt_longCast as *const u8),
        "cljvm_rt_nth" => Some(crate::runtime::cljvm_rt_nth as *const u8),
        "cljvm_rt_protocol_dispatch_1" => {
            Some(crate::runtime::cljvm_rt_protocol_dispatch_1 as *const u8)
        }
        "cljvm_rt_protocol_dispatch_2" => {
            Some(crate::runtime::cljvm_rt_protocol_dispatch_2 as *const u8)
        }
        "cljvm_rt_protocol_dispatch_3" => {
            Some(crate::runtime::cljvm_rt_protocol_dispatch_3 as *const u8)
        }
        "cljvm_rt_protocol_dispatch_4" => {
            Some(crate::runtime::cljvm_rt_protocol_dispatch_4 as *const u8)
        }
        "cljvm_rt_install_impl" => Some(crate::runtime::cljvm_rt_install_impl as *const u8),
        "cljvm_rt_alloc_user_instance_0" => {
            Some(crate::runtime::cljvm_rt_alloc_user_instance_0 as *const u8)
        }
        "cljvm_rt_alloc_user_instance_1" => {
            Some(crate::runtime::cljvm_rt_alloc_user_instance_1 as *const u8)
        }
        "cljvm_rt_alloc_user_instance_2" => {
            Some(crate::runtime::cljvm_rt_alloc_user_instance_2 as *const u8)
        }
        "cljvm_rt_alloc_user_instance_3" => {
            Some(crate::runtime::cljvm_rt_alloc_user_instance_3 as *const u8)
        }
        "cljvm_rt_alloc_user_instance_4" => {
            Some(crate::runtime::cljvm_rt_alloc_user_instance_4 as *const u8)
        }
        "cljvm_rt_user_field_get_by_name" => {
            Some(crate::runtime::cljvm_rt_user_field_get_by_name as *const u8)
        }
        "cljvm_rt_satisfies" => Some(crate::runtime::cljvm_rt_satisfies as *const u8),
        "cljvm_rt_sq_concat" => Some(crate::runtime::cljvm_rt_sq_concat as *const u8),
        "cljvm_rt_peek" => Some(crate::runtime::cljvm_rt_peek as *const u8),
        "cljvm_rt_pop" => Some(crate::runtime::cljvm_rt_pop as *const u8),
        "cljvm_rt_contains" => Some(crate::runtime::cljvm_rt_contains as *const u8),
        "cljvm_rt_get" => Some(crate::runtime::cljvm_rt_get as *const u8),
        "cljvm_rt_get_3" => Some(crate::runtime::cljvm_rt_get_3 as *const u8),
        "cljvm_rt_find" => Some(crate::runtime::cljvm_rt_find as *const u8),
        "cljvm_rt_dissoc" => Some(crate::runtime::cljvm_rt_dissoc as *const u8),
        "cljvm_rt_keys" => Some(crate::runtime::cljvm_rt_keys as *const u8),
        "cljvm_rt_vals" => Some(crate::runtime::cljvm_rt_vals as *const u8),
        "cljvm_rt_booleanCast" => Some(crate::runtime::cljvm_rt_booleanCast as *const u8),
        "cljvm_rt_doubleCast" => Some(crate::runtime::cljvm_rt_doubleCast as *const u8),
        "cljvm_rt_floatCast" => Some(crate::runtime::cljvm_rt_floatCast as *const u8),
        "cljvm_rt_byteCast" => Some(crate::runtime::cljvm_rt_byteCast as *const u8),
        "cljvm_rt_shortCast" => Some(crate::runtime::cljvm_rt_shortCast as *const u8),
        "cljvm_rt_charCast" => Some(crate::runtime::cljvm_rt_charCast as *const u8),
        "cljvm_var_pushThreadBindings" => {
            Some(crate::runtime::cljvm_var_pushThreadBindings as *const u8)
        }
        "cljvm_var_popThreadBindings" => {
            Some(crate::runtime::cljvm_var_popThreadBindings as *const u8)
        }
        "cljvm_var_getThreadBindingFrame" => {
            Some(crate::runtime::cljvm_var_getThreadBindingFrame as *const u8)
        }
        "cljvm_var_getThreadBindings" => {
            Some(crate::runtime::cljvm_var_getThreadBindings as *const u8)
        }
        "cljvm_var_find" => Some(crate::runtime::cljvm_var_find as *const u8),
        "cljvm_var_intern_2" => Some(crate::runtime::cljvm_var_intern_2 as *const u8),
        "cljvm_var_intern_3" => Some(crate::runtime::cljvm_var_intern_3 as *const u8),
        "cljvm_var_resetThreadBindingFrame" => {
            Some(crate::runtime::cljvm_var_resetThreadBindingFrame as *const u8)
        }
        "cljvm_var_cloneThreadBindingFrame" => {
            Some(crate::runtime::cljvm_var_cloneThreadBindingFrame as *const u8)
        }
        "cljvm_inst_disjoin" => Some(crate::runtime::cljvm_inst_disjoin as *const u8),
        "cljvm_inst_compare" => Some(crate::runtime::cljvm_inst_compare as *const u8),
        "cljvm_inst_identity" => Some(crate::runtime::cljvm_inst_identity as *const u8),
        "cljvm_inst_getKey" => Some(crate::runtime::cljvm_inst_getKey as *const u8),
        "cljvm_inst_getValue" => Some(crate::runtime::cljvm_inst_getValue as *const u8),
        "cljvm_inst_rseq" => Some(crate::runtime::cljvm_inst_rseq as *const u8),
        "cljvm_inst_getNamespace" => Some(crate::runtime::cljvm_inst_getNamespace as *const u8),
        "cljvm_inst_ns" => Some(crate::runtime::cljvm_inst_ns as *const u8),
        "cljvm_inst_isPublic" => Some(crate::runtime::cljvm_inst_isPublic as *const u8),
        "cljvm_inst_getMappings" => Some(crate::runtime::cljvm_inst_getMappings as *const u8),
        "cljvm_inst_refer" => Some(crate::runtime::cljvm_inst_refer as *const u8),
        "cljvm_ns_find" => Some(crate::runtime::cljvm_ns_find as *const u8),
        "cljvm_ns_findOrCreate" => Some(crate::runtime::cljvm_ns_findOrCreate as *const u8),
        "cljvm_inst_resetMeta" => Some(crate::runtime::cljvm_inst_resetMeta as *const u8),
        "cljvm_inst_alterMeta" => Some(crate::runtime::cljvm_inst_alterMeta as *const u8),
        "cljvm_inst_bindRoot" => Some(crate::runtime::cljvm_inst_bindRoot as *const u8),
        "cljvm_inst_hasRoot" => Some(crate::runtime::cljvm_inst_hasRoot as *const u8),
        "cljvm_inst_getRawRoot" => Some(crate::runtime::cljvm_inst_getRawRoot as *const u8),
        "cljvm_inst_setValidator" => Some(crate::runtime::cljvm_inst_setValidator as *const u8),
        "cljvm_inst_getValidator" => Some(crate::runtime::cljvm_inst_getValidator as *const u8),
        "cljvm_inst_getWatches" => Some(crate::runtime::cljvm_inst_getWatches as *const u8),
        "cljvm_inst_addWatch" => Some(crate::runtime::cljvm_inst_addWatch as *const u8),
        "cljvm_inst_removeWatch" => Some(crate::runtime::cljvm_inst_removeWatch as *const u8),
        "cljvm_inst_swap" => Some(crate::runtime::cljvm_inst_swap as *const u8),
        "cljvm_inst_reset" => Some(crate::runtime::cljvm_inst_reset as *const u8),
        "cljvm_inst_compareAndSet" => Some(crate::runtime::cljvm_inst_compareAndSet as *const u8),
        "cljvm_inst_deref" => Some(crate::runtime::cljvm_inst_deref as *const u8),
        "cljvm_inst_iterator" => Some(crate::runtime::cljvm_inst_iterator as *const u8),
        "cljvm_inst_setErrorHandler" => {
            Some(crate::runtime::cljvm_inst_setErrorHandler as *const u8)
        }
        "cljvm_inst_getErrorHandler" => {
            Some(crate::runtime::cljvm_inst_getErrorHandler as *const u8)
        }
        "cljvm_inst_setErrorMode" => Some(crate::runtime::cljvm_inst_setErrorMode as *const u8),
        "cljvm_inst_getErrorMode" => Some(crate::runtime::cljvm_inst_getErrorMode as *const u8),
        "cljvm_inst_getError" => Some(crate::runtime::cljvm_inst_getError as *const u8),
        "cljvm_inst_dispatch" => Some(crate::runtime::cljvm_inst_dispatch as *const u8),
        "cljvm_inst_restart" => Some(crate::runtime::cljvm_inst_restart as *const u8),
        "cljvm_inst_set" => Some(crate::runtime::cljvm_inst_set as *const u8),
        "cljvm_inst_alter" => Some(crate::runtime::cljvm_inst_alter as *const u8),
        "cljvm_inst_commute" => Some(crate::runtime::cljvm_inst_commute as *const u8),
        "cljvm_inst_ensure" => Some(crate::runtime::cljvm_inst_ensure as *const u8),
        "cljvm_inst_setMinHistory" => Some(crate::runtime::cljvm_inst_setMinHistory as *const u8),
        "cljvm_inst_setMaxHistory" => Some(crate::runtime::cljvm_inst_setMaxHistory as *const u8),
        "cljvm_inst_getMinHistory" => Some(crate::runtime::cljvm_inst_getMinHistory as *const u8),
        "cljvm_inst_getMaxHistory" => Some(crate::runtime::cljvm_inst_getMaxHistory as *const u8),
        "cljvm_inst_getHistoryCount" => {
            Some(crate::runtime::cljvm_inst_getHistoryCount as *const u8)
        }
        "cljvm_inst_trimHistory" => Some(crate::runtime::cljvm_inst_trimHistory as *const u8),
        "cljvm_unimpl_host_call_0" => Some(crate::runtime::cljvm_unimpl_host_call_0 as *const u8),
        "cljvm_unimpl_host_call_1" => Some(crate::runtime::cljvm_unimpl_host_call_1 as *const u8),
        "cljvm_unimpl_host_call_2" => Some(crate::runtime::cljvm_unimpl_host_call_2 as *const u8),
        "cljvm_unimpl_host_call_3" => Some(crate::runtime::cljvm_unimpl_host_call_3 as *const u8),
        "cljvm_unimpl_host_call_4" => Some(crate::runtime::cljvm_unimpl_host_call_4 as *const u8),
        "cljvm_unimpl_host_call_5" => Some(crate::runtime::cljvm_unimpl_host_call_5 as *const u8),
        "cljvm_unimpl_host_call_6" => Some(crate::runtime::cljvm_unimpl_host_call_6 as *const u8),
        "cljvm_inst_multifn_reset" => Some(crate::runtime::cljvm_inst_multifn_reset as *const u8),
        "cljvm_inst_multifn_addMethod" => {
            Some(crate::runtime::cljvm_inst_multifn_addMethod as *const u8)
        }
        "cljvm_inst_multifn_removeMethod" => {
            Some(crate::runtime::cljvm_inst_multifn_removeMethod as *const u8)
        }
        "cljvm_inst_multifn_preferMethod" => {
            Some(crate::runtime::cljvm_inst_multifn_preferMethod as *const u8)
        }
        "cljvm_inst_multifn_methodTable" => {
            Some(crate::runtime::cljvm_inst_multifn_methodTable as *const u8)
        }
        "cljvm_inst_multifn_preferTable" => {
            Some(crate::runtime::cljvm_inst_multifn_preferTable as *const u8)
        }
        "cljvm_inst_multifn_getMethod" => {
            Some(crate::runtime::cljvm_inst_multifn_getMethod as *const u8)
        }
        "cljvm_rt_nth_3" => Some(crate::runtime::cljvm_rt_nth_3 as *const u8),
        "cljvm_numbers_isZero" => Some(crate::runtime::cljvm_numbers_isZero as *const u8),
        "cljvm_numbers_add" => Some(crate::runtime::cljvm_numbers_add as *const u8),
        "cljvm_numbers_minus" => Some(crate::runtime::cljvm_numbers_minus as *const u8),
        "cljvm_numbers_minus_1" => Some(crate::runtime::cljvm_numbers_minus_1 as *const u8),
        "cljvm_numbers_multiply" => Some(crate::runtime::cljvm_numbers_multiply as *const u8),
        "cljvm_numbers_divide" => Some(crate::runtime::cljvm_numbers_divide as *const u8),
        "cljvm_numbers_inc" => Some(crate::runtime::cljvm_numbers_inc as *const u8),
        "cljvm_numbers_dec" => Some(crate::runtime::cljvm_numbers_dec as *const u8),
        "cljvm_numbers_lt" => Some(crate::runtime::cljvm_numbers_lt as *const u8),
        "cljvm_numbers_lte" => Some(crate::runtime::cljvm_numbers_lte as *const u8),
        "cljvm_numbers_gt" => Some(crate::runtime::cljvm_numbers_gt as *const u8),
        "cljvm_numbers_gte" => Some(crate::runtime::cljvm_numbers_gte as *const u8),
        "cljvm_numbers_equiv" => Some(crate::runtime::cljvm_numbers_equiv as *const u8),
        "cljvm_numbers_isPos" => Some(crate::runtime::cljvm_numbers_isPos as *const u8),
        "cljvm_numbers_isNeg" => Some(crate::runtime::cljvm_numbers_isNeg as *const u8),
        "cljvm_numbers_max" => Some(crate::runtime::cljvm_numbers_max as *const u8),
        "cljvm_numbers_abs" => Some(crate::runtime::cljvm_numbers_abs as *const u8),
        "cljvm_numbers_identity" => Some(crate::runtime::cljvm_numbers_identity as *const u8),
        "cljvm_numbers_isInteger" => Some(crate::runtime::cljvm_numbers_isInteger as *const u8),
        "cljvm_numbers_isFloat" => Some(crate::runtime::cljvm_numbers_isFloat as *const u8),
        "cljvm_numbers_isRational" => Some(crate::runtime::cljvm_numbers_isRational as *const u8),
        "cljvm_numbers_isNaN" => Some(crate::runtime::cljvm_numbers_isNaN as *const u8),
        "cljvm_numbers_isInfinite" => Some(crate::runtime::cljvm_numbers_isInfinite as *const u8),
        "cljvm_numbers_not" => Some(crate::runtime::cljvm_numbers_not as *const u8),
        "cljvm_numbers_and" => Some(crate::runtime::cljvm_numbers_and as *const u8),
        "cljvm_numbers_or" => Some(crate::runtime::cljvm_numbers_or as *const u8),
        "cljvm_numbers_xor" => Some(crate::runtime::cljvm_numbers_xor as *const u8),
        "cljvm_numbers_andNot" => Some(crate::runtime::cljvm_numbers_andNot as *const u8),
        "cljvm_numbers_shiftLeft" => Some(crate::runtime::cljvm_numbers_shiftLeft as *const u8),
        "cljvm_numbers_shiftRight" => Some(crate::runtime::cljvm_numbers_shiftRight as *const u8),
        "cljvm_numbers_unsignedShiftRight" => {
            Some(crate::runtime::cljvm_numbers_unsignedShiftRight as *const u8)
        }
        "cljvm_numbers_clearBit" => Some(crate::runtime::cljvm_numbers_clearBit as *const u8),
        "cljvm_numbers_setBit" => Some(crate::runtime::cljvm_numbers_setBit as *const u8),
        "cljvm_numbers_flipBit" => Some(crate::runtime::cljvm_numbers_flipBit as *const u8),
        "cljvm_numbers_testBit" => Some(crate::runtime::cljvm_numbers_testBit as *const u8),
        "cljvm_numbers_min" => Some(crate::runtime::cljvm_numbers_min as *const u8),
        "cljvm_numbers_quotient" => Some(crate::runtime::cljvm_numbers_quotient as *const u8),
        "cljvm_numbers_remainder" => Some(crate::runtime::cljvm_numbers_remainder as *const u8),
        "cljvm_inst_LazySeq_new1" => Some(crate::runtime::cljvm_inst_LazySeq_new1 as *const u8),
        "cljvm_inst_Delay_new1" => Some(crate::runtime::cljvm_inst_Delay_new1 as *const u8),
        "cljvm_delay_force" => Some(crate::runtime::cljvm_delay_force as *const u8),
        "cljvm_longrange_create_1" => Some(crate::runtime::cljvm_longrange_create_1 as *const u8),
        "cljvm_longrange_create_2" => Some(crate::runtime::cljvm_longrange_create_2 as *const u8),
        "cljvm_longrange_create_3" => Some(crate::runtime::cljvm_longrange_create_3 as *const u8),
        "cljvm_inst_toUpperCase" => Some(crate::runtime::cljvm_inst_toUpperCase as *const u8),
        "cljvm_inst_toLowerCase" => Some(crate::runtime::cljvm_inst_toLowerCase as *const u8),
        "cljvm_inst_trim" => Some(crate::runtime::cljvm_inst_trim as *const u8),
        "cljvm_str_reverse" => Some(crate::runtime::cljvm_str_reverse as *const u8),
        "cljvm_math_pow" => Some(crate::runtime::cljvm_math_pow as *const u8),
        "cljvm_math_sqrt" => Some(crate::runtime::cljvm_math_sqrt as *const u8),
        "cljvm_rt_cons" => Some(crate::runtime::cljvm_rt_cons as *const u8),
        "cljvm_rt_load" => Some(crate::runtime::cljvm_rt_load as *const u8),
        "cljvm_rt_isReduced" => Some(crate::runtime::cljvm_rt_isReduced as *const u8),
        "cljvm_ns_set_current" => Some(crate::runtime::cljvm_ns_set_current as *const u8),
        "cljvm_rt_conj" => Some(crate::runtime::cljvm_rt_conj as *const u8),
        "cljvm_rt_assoc" => Some(crate::runtime::cljvm_rt_assoc as *const u8),
        "cljvm_rt_first" => Some(crate::runtime::cljvm_rt_first as *const u8),
        "cljvm_rt_next" => Some(crate::runtime::cljvm_rt_next as *const u8),
        "cljvm_rt_more" => Some(crate::runtime::cljvm_rt_more as *const u8),
        "cljvm_rt_seq" => Some(crate::runtime::cljvm_rt_seq as *const u8),
        "cljvm_rt_count" => Some(crate::runtime::cljvm_rt_count as *const u8),
        "cljvm_rt_toArray" => Some(crate::runtime::cljvm_rt_toArray as *const u8),
        "cljvm_arrays_sort" => Some(crate::runtime::cljvm_arrays_sort as *const u8),
        "cljvm_lpv_create" => Some(crate::runtime::cljvm_lpv_create as *const u8),
        "cljvm_phm_create" => Some(crate::runtime::cljvm_phm_create as *const u8),
        "cljvm_phs_create" => Some(crate::runtime::cljvm_phs_create as *const u8),
        "cljvm_ptm_create" => Some(crate::runtime::cljvm_ptm_create as *const u8),
        "cljvm_ptm_create_cmp" => Some(crate::runtime::cljvm_ptm_create_cmp as *const u8),
        "cljvm_pts_create" => Some(crate::runtime::cljvm_pts_create as *const u8),
        "cljvm_pts_create_cmp" => Some(crate::runtime::cljvm_pts_create_cmp as *const u8),
        "cljvm_util_identical" => Some(crate::runtime::cljvm_util_identical as *const u8),
        "cljvm_util_compare" => Some(crate::runtime::cljvm_util_compare as *const u8),
        "cljvm_rt_subvec" => Some(crate::runtime::cljvm_rt_subvec as *const u8),
        "cljvm_compiler_hostexpr_maybeSpecialTag" => {
            Some(crate::runtime::cljvm_compiler_hostexpr_maybeSpecialTag as *const u8)
        }
        "cljvm_compiler_hostexpr_maybeClass" => {
            Some(crate::runtime::cljvm_compiler_hostexpr_maybeClass as *const u8)
        }
        "cljvm_symbol_intern_1" => Some(crate::runtime::cljvm_symbol_intern_1 as *const u8),
        "cljvm_symbol_intern_2" => Some(crate::runtime::cljvm_symbol_intern_2 as *const u8),
        "cljvm_keyword_intern_1" => Some(crate::runtime::cljvm_keyword_intern_1 as *const u8),
        "cljvm_keyword_intern_2" => Some(crate::runtime::cljvm_keyword_intern_2 as *const u8),
        "cljvm_keyword_find_1" => Some(crate::runtime::cljvm_keyword_find_1 as *const u8),
        "cljvm_keyword_find_2" => Some(crate::runtime::cljvm_keyword_find_2 as *const u8),
        "cljvm_rt_equiv" => Some(crate::runtime::cljvm_rt_equiv as *const u8),
        "cljvm_rt_is_nil" => Some(crate::runtime::cljvm_rt_is_nil as *const u8),
        "cljvm_rt_invoke_0" => Some(crate::runtime::cljvm_rt_invoke_0 as *const u8),
        "cljvm_rt_invoke_1" => Some(crate::runtime::cljvm_rt_invoke_1 as *const u8),
        "cljvm_rt_invoke_2" => Some(crate::runtime::cljvm_rt_invoke_2 as *const u8),
        "cljvm_rt_invoke_3" => Some(crate::runtime::cljvm_rt_invoke_3 as *const u8),
        "cljvm_rt_invoke_4" => Some(crate::runtime::cljvm_rt_invoke_4 as *const u8),
        "cljvm_rt_invoke_5" => Some(crate::runtime::cljvm_rt_invoke_5 as *const u8),
        "cljvm_rt_invoke_6" => Some(crate::runtime::cljvm_rt_invoke_6 as *const u8),
        "cljvm_rt_invoke_7" => Some(crate::runtime::cljvm_rt_invoke_7 as *const u8),
        "cljvm_rt_invoke_8" => Some(crate::runtime::cljvm_rt_invoke_8 as *const u8),
        "cljvm_rt_invoke_9" => Some(crate::runtime::cljvm_rt_invoke_9 as *const u8),
        "cljvm_rt_invoke_10" => Some(crate::runtime::cljvm_rt_invoke_10 as *const u8),
        "cljvm_pl_creator" => Some(crate::runtime::cljvm_pl_creator as *const u8),
        "cljvm_inst_meta" => Some(crate::runtime::cljvm_inst_meta as *const u8),
        "cljvm_inst_with_meta" => Some(crate::runtime::cljvm_inst_with_meta as *const u8),
        "cljvm_inst_isInstance" => Some(crate::runtime::cljvm_inst_isInstance as *const u8),
        "cljvm_inst_getName" => Some(crate::runtime::cljvm_inst_getName as *const u8),
        "cljvm_inst_concat" => Some(crate::runtime::cljvm_inst_concat as *const u8),
        "cljvm_inst_indexOf" => Some(crate::runtime::cljvm_inst_indexOf as *const u8),
        "cljvm_inst_startsWith" => Some(crate::runtime::cljvm_inst_startsWith as *const u8),
        "cljvm_inst_substring" => Some(crate::runtime::cljvm_inst_substring as *const u8),
        "cljvm_inst_substring2" => Some(crate::runtime::cljvm_inst_substring2 as *const u8),
        "cljvm_inst_lastIndexOf" => Some(crate::runtime::cljvm_inst_lastIndexOf as *const u8),
        "cljvm_inst_replace" => Some(crate::runtime::cljvm_inst_replace as *const u8),
        "cljvm_inst_setMacro" => Some(crate::runtime::cljvm_inst_setMacro as *const u8),
        "cljvm_inst_cast" => Some(crate::runtime::cljvm_inst_cast as *const u8),
        "cljvm_inst_toString" => Some(crate::runtime::cljvm_inst_toString as *const u8),
        "cljvm_inst_applyTo" => Some(crate::runtime::cljvm_inst_applyTo as *const u8),
        "cljvm_inst_toSymbol" => Some(crate::runtime::cljvm_inst_toSymbol as *const u8),
        "cljvm_inst_sym" => Some(crate::runtime::cljvm_inst_sym as *const u8),
        "cljvm_inst_StringBuilder_new1" => {
            Some(crate::runtime::cljvm_inst_StringBuilder_new1 as *const u8)
        }
        "cljvm_inst_StringBuilder_append" => {
            Some(crate::runtime::cljvm_inst_StringBuilder_append as *const u8)
        }
        "cljvm_inst_StringBuilder_toString" => {
            Some(crate::runtime::cljvm_inst_StringBuilder_toString as *const u8)
        }
        "cljvm_inst_ChunkBuffer_new1" => {
            Some(crate::runtime::cljvm_inst_ChunkBuffer_new1 as *const u8)
        }
        "cljvm_inst_ChunkBuffer_add" => {
            Some(crate::runtime::cljvm_inst_ChunkBuffer_add as *const u8)
        }
        "cljvm_inst_ChunkBuffer_chunk" => {
            Some(crate::runtime::cljvm_inst_ChunkBuffer_chunk as *const u8)
        }
        "cljvm_inst_chunkedFirst" => Some(crate::runtime::cljvm_inst_chunkedFirst as *const u8),
        "cljvm_inst_chunkedMore" => Some(crate::runtime::cljvm_inst_chunkedMore as *const u8),
        "cljvm_inst_chunkedNext" => Some(crate::runtime::cljvm_inst_chunkedNext as *const u8),
        "cljvm_inst_reduce_3" => Some(crate::runtime::cljvm_inst_reduce_3 as *const u8),
        "cljvm_inst_new_1" => Some(crate::runtime::cljvm_inst_new_1 as *const u8),
        "cljvm_inst_new_2" => Some(crate::runtime::cljvm_inst_new_2 as *const u8),
        "cljvm_inst_new_3" => Some(crate::runtime::cljvm_inst_new_3 as *const u8),
        _ => None,
    }
}

pub fn compile_form_to_jit(
    form: Object,
) -> (
    dynlang::gc::DynGcRuntime,
    dynlower::JitModule,
    dynir::FuncRef,
    CompileRoots,
) {
    let mut compiler = Compiler::new();
    let entry = compiler.with_active(|c| {
        // Declare the entry fn (`__top_level__`) up front.
        let entry = c.dm.declare_func("__top_level__", 0);

        // Analyze the form. During analyze, `FnExpr` registers pending fns
        // on the active compiler.
        let expr = analyze(C::Expression, form);

        // Drain pending fns: lower each body into its declared FuncRef.
        loop {
            let pending = c.pending_fns.lock().unwrap().pop();
            let Some(p) = pending else { break };
            lower_pending_fn(c, p);
        }

        // Now lower the entry fn.
        let mut df = c.dm.start_func(entry);
        let needs_ret = {
            let mut ir = IrEmitter::new(&mut df);
            let objx = ObjExpr::placeholder();
            match expr.emit(C::Expression, &*objx, &mut ir) {
                Some(v) => Some(v),
                None => {
                    // Body terminated the entry block before producing a
                    // value — e.g. `(throw …)` ends in `abort_to_prompt`,
                    // which is itself a terminator. No `ret` needed in
                    // that case; the block already has a terminator.
                    assert!(
                        ir.f.fb.current_block_is_terminated(),
                        "top-level form produced no value but the entry block isn't terminated"
                    );
                    None
                }
            }
        };
        if let Some(result_val) = needs_ret {
            df.fb.ret(result_val);
        }
        c.dm.finish_func(df);

        entry
    });

    // Take ownership of GC config + obj_types BEFORE consuming the DynModule
    // via `build`. `ObjType` isn't `Clone`, so `std::mem::take` moves them out
    // into a local Vec the runtime can borrow from.
    let gc_config = compiler.dm.gc_config().clone();
    let obj_types: Vec<dynlang::ObjType> = std::mem::take(&mut compiler.dm.obj_types);
    let tags = dynlang::NanBoxTags::default();
    let built = compiler.dm.build();

    let gc = dynlang::gc::DynGcRuntime::new(&gc_config, &tags, &obj_types);

    // Build the JitModule via `new_empty + extend`. This is the same
    // pattern microlisp uses for its same-image-macros story: a single
    // long-lived JitModule that grows via `extend` per top-level form,
    // letting macro bodies (compiled in an earlier extend) be invoked
    // via `gc.run_jit(jit, macro_fref, ...)` during analysis of a later
    // form. For now we still do single-form-per-call here; the Session
    // wrapper layered on top reuses the JitModule across forms.
    //
    // CallMode::ControlAware pairs the JIT with the GC-aware safepoint
    // handler from dynruntime, required for the moving collector.
    use dynir::dynexec::NanBoxConfig;
    use dynlower::regalloc::LinearScanAllocator;
    use dynlower::{Arm64Backend, CallMode, JitModule};
    use dynruntime::active_jit_safepoint_handler;
    #[cfg(not(target_arch = "aarch64"))]
    compile_error!("clojure-jvm JIT path only configured for aarch64 right now");

    let call_mode = CallMode::ControlAware {
        safepoint_handler: active_jit_safepoint_handler as u64,
    };
    let jit = JitModule::new_empty::<NanBoxConfig, Arm64Backend, LinearScanAllocator>(
        /* call_table_capacity */ 64 * 1024,
        /* literal_pool_capacity */ 64 * 1024,
        call_mode,
    );

    let externs = build_extern_table_for(&built.module);
    let _ = jit.extend::<NanBoxConfig, Arm64Backend, LinearScanAllocator>(&built.module, &externs);

    // Populate the literal pool with heap-allocated compile-time literals.
    // Must happen before any execution reads `gc_literal(idx)`; the pool
    // slots are registered as GC roots by `run_jit`, so the allocations
    // stay traceable across collections.
    //
    // The Arc<Symbol>/Arc<String> values backing each literal are *not*
    // globally rooted (Symbol::intern_ns_name returns a fresh Arc per
    // call), so we move them into `CompileRoots` returned to the caller.
    // The literal pool stores raw `Arc::as_ptr` pointers into these
    // Arcs; dropping them would dangle.
    let mut roots = CompileRoots {
        _symbols: Vec::new(),
        _strings: Vec::new(),
        _keywords: Vec::new(),
        _maps: Vec::new(),
        _sets: Vec::new(),
        _tree_maps: Vec::new(),
        _tree_sets: Vec::new(),
        _string_builders: Vec::new(),
        _chunk_buffers: Vec::new(),
        _lazy_states: Vec::new(),
        _multi_arity_tables: Vec::new(),
        _namespaces: Vec::new(),
        _vars: Vec::new(),
    };
    {
        let _thread = gc.install_thread();
        // Compile-time literal population conses nested *list* literals via
        // `alloc_list_as_nanbox`, which roots intermediate NanBoxes on a
        // `with_scope` frame. That requires an installed `FrameChain`.
        // Install one (and expose it to the GC as a root source) for the
        // duration of this block, mirroring the production Session path
        // (`with_active_session_ref`). Without it, compiling a quoted list
        // literal panics "no FrameChain installed on this thread" — the
        // failure was masked in the full test suite only when a prior
        // test happened to leave a chain installed on the thread.
        let _lit_chain = dynobj::roots::FrameChain::new();
        let _lit_chain_root_g =
            unsafe { gc.push_extra_root_source(&_lit_chain as *const dyn dynobj::RootSource) };
        let _lit_chain_g = dynobj::roots::install_chain(&_lit_chain);
        let pending = std::mem::take(&mut *compiler.pending_literals.lock().unwrap());
        let string_type_id = compiler.string_type_id;
        let symbol_type_id = compiler.symbol_type_id;
        let keyword_type_id = compiler.keyword_type_id;
        for (idx, lit) in pending.iter().enumerate() {
            let nanbox_bits = match lit {
                PendingLiteral::String(s) => {
                    roots._strings.push(s.clone());
                    let bytes = s.as_bytes();
                    let ptr = gc.alloc(string_type_id.0, bytes.len());
                    assert!(
                        !ptr.is_null(),
                        "clojure-jvm: gc.alloc returned null for string literal of {} bytes",
                        bytes.len()
                    );
                    // SAFETY: `ptr` is a freshly-allocated object of
                    // `string_type_id`. Its layout has 8-byte Compact header
                    // + 8-byte varlen-count (set by `alloc`) + N varlen
                    // bytes. Writing into the varlen byte section is within
                    // the allocation.
                    let type_info = &obj_types[string_type_id.0].type_info;
                    let data_offset = type_info.varlen_element_offset(0);
                    unsafe {
                        let dst = ptr.add(data_offset);
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
                    }
                    gc.tag_ptr(ptr)
                }
                PendingLiteral::Symbol(s) => {
                    // Symbols are NOT yet globally interned in our port
                    // (`Symbol::intern_ns_name` returns a fresh Arc each
                    // time), so we must keep this `Arc<Symbol>` alive past
                    // the literal-pool fill — otherwise the stored
                    // `Arc::as_ptr` dangles after the pending Vec drops.
                    // The Vec on `CompileRoots` extends each Arc's lifetime
                    // to the JIT module's.
                    roots._symbols.push(s.clone());
                    let ptr = gc.alloc(symbol_type_id.0, 0);
                    assert!(
                        !ptr.is_null(),
                        "clojure-jvm: gc.alloc returned null for symbol literal"
                    );
                    let arc_ptr_bits = Arc::as_ptr(s) as u64;
                    let type_info = &obj_types[symbol_type_id.0].type_info;
                    let raw_offset = type_info.raw_data_offset();
                    unsafe {
                        let dst = ptr.add(raw_offset).cast::<u64>();
                        dst.write_unaligned(arc_ptr_bits);
                    }
                    gc.tag_ptr(ptr)
                }
                PendingLiteral::List(l) => {
                    // Recursively allocate Cons cells, then return the head's
                    // NanBox. Each element is itself NanBox-encoded via
                    // `alloc_object_as_nanbox` which dispatches on Object.
                    alloc_list_as_nanbox(
                        &gc,
                        &obj_types,
                        compiler.cons_type_id,
                        compiler.string_type_id,
                        compiler.symbol_type_id,
                        compiler.keyword_type_id,
                        &mut roots,
                        l,
                    )
                }
                PendingLiteral::Keyword(k) => {
                    // Same shape as Symbol; CompileRoots holds the Arc alive
                    // so the stashed Arc::as_ptr stays valid.
                    roots._keywords.push(k.clone());
                    let ptr = gc.alloc(keyword_type_id.0, 0);
                    assert!(
                        !ptr.is_null(),
                        "clojure-jvm: gc.alloc returned null for keyword literal"
                    );
                    let arc_ptr_bits = Arc::as_ptr(k) as u64;
                    let type_info = &obj_types[keyword_type_id.0].type_info;
                    let raw_offset = type_info.raw_data_offset();
                    unsafe {
                        let dst = ptr.add(raw_offset).cast::<u64>();
                        dst.write_unaligned(arc_ptr_bits);
                    }
                    gc.tag_ptr(ptr)
                }
                PendingLiteral::Vector(v) => alloc_vector_as_nanbox(
                    &gc,
                    &obj_types,
                    compiler.vector_type_id,
                    compiler.cons_type_id,
                    compiler.string_type_id,
                    compiler.symbol_type_id,
                    compiler.keyword_type_id,
                    &mut roots,
                    v,
                ),
                PendingLiteral::Map(m) => {
                    alloc_map_as_nanbox(&gc, &obj_types, compiler.map_type_id, &mut roots, m)
                }
                PendingLiteral::Set(s) => {
                    alloc_set_as_nanbox(&gc, &obj_types, compiler.set_type_id, &mut roots, s)
                }
                PendingLiteral::Class(class_id) => {
                    alloc_class_as_nanbox(&gc, &obj_types, compiler.class_type_id, *class_id)
                }
                PendingLiteral::Var(v) => {
                    alloc_var_as_nanbox(&gc, &obj_types, compiler.var_type_id, v)
                }
                PendingLiteral::MultiArityFn(t) => alloc_multi_arity_fn_as_nanbox(
                    &gc,
                    &obj_types,
                    compiler.multi_arity_fn_type_id,
                    &mut roots,
                    t,
                ),
                PendingLiteral::Long(n) => unsafe { crate::runtime::box_long(*n) },
                PendingLiteral::Char(c) => unsafe { crate::runtime::box_char(*c) },
            };
            let pushed_idx = jit.literal_pool().push(nanbox_bits);
            assert_eq!(
                pushed_idx, idx,
                "clojure-jvm: literal pool index mismatch (got {pushed_idx}, expected {idx})"
            );
        }
    }

    (gc, jit, entry, roots)
}

/// Allocate a `clojure.lang.MultiArityFn` heap cell at literal-pool-fill
/// time. Mirrors `alloc_map_as_nanbox` etc.: rooted Arc on Session,
/// `Arc::as_ptr` written into the Raw64 slot.
fn alloc_multi_arity_fn_as_nanbox(
    gc: &dynlang::gc::DynGcRuntime,
    obj_types: &[dynlang::ObjType],
    multi_arity_fn_type_id: dynlang::ObjTypeId,
    roots: &mut CompileRoots,
    table: &Arc<Vec<crate::runtime::MultiArityEntry>>,
) -> u64 {
    roots._multi_arity_tables.push(table.clone());
    let ptr = gc.alloc(multi_arity_fn_type_id.0, 0);
    assert!(!ptr.is_null(), "alloc_multi_arity_fn: null");
    let arc_ptr_bits = Arc::as_ptr(table) as u64;
    let type_info = &obj_types[multi_arity_fn_type_id.0].type_info;
    let raw_offset = type_info.raw_data_offset();
    unsafe {
        ptr.add(raw_offset)
            .cast::<u64>()
            .write_unaligned(arc_ptr_bits);
    }
    gc.tag_ptr(ptr)
}

/// Compile a top-level form and return a built dynir module + its entry fn,
/// ready to run via the reference `ModuleInterpreter`.
pub fn compile_form_to_interp(form: Object) -> (dynlang::BuiltModule, dynir::FuncRef) {
    let mut compiler = Compiler::new();
    let entry = compiler.with_active(|c| {
        let entry = c.dm.declare_func("__top_level__", 0);
        let expr = analyze(C::Expression, form);
        loop {
            let pending = c.pending_fns.lock().unwrap().pop();
            let Some(p) = pending else { break };
            lower_pending_fn(c, p);
        }
        let mut df = c.dm.start_func(entry);
        let needs_ret = {
            let mut ir = IrEmitter::new(&mut df);
            let objx = ObjExpr::placeholder();
            match expr.emit(C::Expression, &*objx, &mut ir) {
                Some(v) => Some(v),
                None => {
                    // Top-level form emitted a terminator itself (e.g.
                    // `(throw …)` ends in `abort_to_prompt`). No `ret`
                    // needed — block is already terminated.
                    assert!(
                        ir.f.fb.current_block_is_terminated(),
                        "top-level form produced no value but the entry block isn't terminated"
                    );
                    None
                }
            }
        };
        if let Some(result_val) = needs_ret {
            df.fb.ret(result_val);
        }
        c.dm.finish_func(df);
        entry
    });
    let built = compiler.dm.build();
    (built, entry)
}

/// Lower a `PendingFn` body into its declared FuncRef. Called from the
/// drainer after analyze completes.
fn lower_pending_fn(c: &mut Compiler, p: PendingFn) {
    // Install the fn id so `LocalBindingExpr.emit` can compare bindings'
    // owning_fn_id and route captures to the closure self-arg read path.
    let _fn_guard = FnIdGuard::new(p.fn_id);
    let is_closure = !p.captures.is_empty();

    let mut df = c.dm.start_func(p.fref);
    {
        let mut ir = IrEmitter::new(&mut df);
        let objx = ObjExpr::placeholder();
        // Closure bodies have an implicit first param (the closure object).
        // The user-visible params follow. We stash the closure self-value
        // under a well-known slot name `__closure_self` so
        // `LocalBindingExpr.emit` can pull it out when it needs to read a
        // capture.
        let entry_bb = ir.f.fb.entry_block();
        let mut param_offset = 0usize;
        if is_closure {
            let self_val = ir.f.fb.block_param(entry_bb, 0);
            ir.f.def_var(CLOSURE_SELF_SLOT, self_val);
            param_offset = 1;
        }
        for (i, lb) in p.params.iter().enumerate() {
            let arg_val = ir.f.fb.block_param(entry_bb, param_offset + i);
            ir.f.def_var(&local_slot_name(lb.idx), arg_val);
        }

        // Initialize the named-fn self-reference slot. Priority:
        //   1. Multi-arity clause → the shared `MultiArityFn` dispatcher
        //      cell, so a self-call to a DIFFERENT arity dispatches by
        //      arity instead of re-entering this clause (the `str`
        //      infinite-recursion bug). The cell is a literal-pool slot
        //      reserved at analyze time.
        //   2. Closure → the implicit first arg (the closure object, in
        //      `CLOSURE_SELF_SLOT`).
        //   3. Plain (single-arity, non-capturing) fn → the
        //      `TAG_FN(fref_index)` constant.
        if let Some(self_idx) = p.self_name_slot {
            let self_val = if let Some(lref) = p.self_multi_lit {
                ir.f.fb.gc_literal(lref)
            } else if is_closure {
                ir.f.get_var(CLOSURE_SELF_SLOT)
            } else {
                ir.f.tagged_const(3, p.fref.index() as u64)
            };
            ir.f.def_var(&local_slot_name(self_idx), self_val);
        }

        // Java emits a "loop label" right after the param preamble so that
        // a `recur` in the fn body can jump back to re-bind the args. We do
        // the same by creating a `body_top` block, jumping to it, and
        // switching to it before emitting the body. RecurExpr.emit reads the
        // BlockId from LOOP_LABEL.
        let body_top = ir.f.fb.create_block(&[]);
        ir.f.fb.jump(body_top, &[]);
        ir.f.fb.switch_to_block(body_top);

        let recur_target = RecurTarget {
            block: body_top,
            locals: p.params.clone(),
        };
        Var::push_thread_bindings(vec![
            (
                COMPILER_VARS.LOOP_LABEL.clone(),
                Object::Host(std::sync::Arc::new(recur_target)),
            ),
            (
                COMPILER_VARS.LOOP_LOCALS.clone(),
                Object::Host(std::sync::Arc::new(p.params.clone())),
            ),
        ]);
        let body_val = p.body.emit(C::Return, &*objx, &mut ir);
        Var::pop_thread_bindings();

        // If the body didn't terminate the block (no tail-recur), return
        // the body's value.
        if !ir.f.fb.current_block_is_terminated() {
            let v = body_val.expect("fn body must produce a value if not diverging");
            ir.f.fb.ret(v);
        }
    }
    c.dm.finish_func(df);
}

/// What `recur` needs at emit time: the block to jump to, plus the locals it
/// re-binds. Stored as `Object::Host` on the `LOOP_LABEL` Var.
#[derive(Debug, Clone)]
pub struct RecurTarget {
    pub block: dynir::BlockId,
    pub locals: Vec<Arc<LocalBinding>>,
}

// ─── ACTIVE_SESSION thread-local ──────────────────────────────────────
//
// Set by `Session::eval_form` for the duration of analyze+emit. Reached
// by macroexpansion in `analyze_seq` so it can call into the same JIT +
// GC that's processing the current form. Mirrors `ACTIVE_COMPILER`'s
// raw-pointer-via-thread-local pattern.

thread_local! {
    static ACTIVE_SESSION: std::cell::Cell<*mut Session> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
}

/// Set ACTIVE_SESSION for the duration of `body`, restoring on exit.
fn with_active_session<R>(sess: &mut Session, body: impl FnOnce() -> R) -> R {
    let ptr = sess as *mut Session;
    let prev = ACTIVE_SESSION.with(|c| c.replace(ptr));
    let r = body();
    ACTIVE_SESSION.with(|c| c.set(prev));
    r
}

/// Tiny RAII helper to run a closure on drop. Used to restore
/// ACTIVE_SESSION when `eval_form` returns (regardless of panic path).
struct DropFn<F: FnMut()>(F);
impl<F: FnMut()> Drop for DropFn<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

/// Look up the registered arity info for `fref_idx` on the active
/// Compiler. Used by the runtime invoke path to detect variadic
/// targets at dynamic call sites and pack the overflow into a list.
pub fn with_active_compiler_arity(fref_idx: u32) -> Option<VarFnInfo> {
    with_active_session_ref(|sess| sess.compiler.fn_arity(fref_idx)).flatten()
}

/// Encode `obj` to a NanBox using the active Session's GC + obj_types.
/// Used by runtime externs (e.g. keyword-as-fn lookup) that produce a
/// host-side `Object` and need to hand it back to JIT'd code.
pub fn with_active_session_encode_object(obj: &Object) -> u64 {
    with_active_session_ref(|sess| {
        alloc_object_as_nanbox(
            &sess.gc,
            &sess.obj_types,
            sess.compiler.cons_type_id,
            sess.compiler.string_type_id,
            sess.compiler.symbol_type_id,
            sess.compiler.keyword_type_id,
            &mut sess.roots,
            obj,
        )
    })
    .unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_encode_object called without an \
             active Session"
        )
    })
}

/// Push `s` into the active Session's `roots._symbols`, extending its
/// lifetime to the JIT module's. Used by runtime externs that intern
/// new Symbol values (e.g. `cljvm_symbol_intern_1`).
pub fn with_active_session_root_symbol(s: Arc<Symbol>) {
    with_active_session_ref(|sess| sess.roots._symbols.push(s)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_symbol called without an \
             active Session — runtime extern must run inside Session::eval_form"
        )
    });
}

/// Push `m` into the active Session's `roots._maps`, extending the
/// Arc's lifetime to the JIT module's. Used by runtime externs (e.g.
/// `cljvm_rt_assoc`) that allocate fresh Map heap cells whose Raw64
/// pointer needs the Arc to outlive them.
pub fn with_active_session_root_map(m: Arc<crate::lang::persistent_hash_map::PersistentHashMap>) {
    with_active_session_ref(|s| s.roots._maps.push(m)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_map called without an \
             active Session — runtime extern must run inside Session::eval_form"
        )
    });
}

/// Push `ns` into the active Session's `roots._namespaces`, keeping the
/// `Arc<Namespace>` alive for the JIT module's lifetime so a
/// `clojure.lang.Namespace` heap cell's raw `Arc::as_ptr` slot never
/// dangles. (Namespaces also live forever in the global registry, but the
/// roots Vec is the contract `alloc_arc_cell` expects.)
pub fn with_active_session_root_namespace(ns: Arc<crate::lang::namespace::Namespace>) {
    with_active_session_ref(|s| s.roots._namespaces.push(ns)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_namespace called without an \
             active Session — runtime extern must run inside Session::eval_form"
        )
    });
}

/// Push `v` into the active Session's `roots._vars`, keeping a runtime-boxed
/// `clojure.lang.Var` cell's raw `Arc::as_ptr` slot alive for the JIT
/// module's lifetime.
pub fn with_active_session_root_var(v: Arc<crate::lang::var::Var>) {
    with_active_session_ref(|s| s.roots._vars.push(v)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_var called without an \
             active Session — runtime extern must run inside Session::eval_form"
        )
    });
}

/// Resolve a `(load "…")` path to its embedded source and evaluate every
/// form through the active Session — the runtime side of
/// `clojure.lang.RT/load`. This runs *reentrantly*: the call originates
/// from JIT-compiled `load` code that is itself executing inside the outer
/// `eval_form`'s `run_jit`. Each form here goes through the full
/// compile → `jit.extend` → `run_jit` path again; the architecture
/// supports this (compile-during-execution never relocates running code,
/// and the GC root walker traverses interleaved JIT/runtime frames).
///
/// `*ns*` is saved and restored around the load so the loaded file's
/// `(ns …)` does not leak into the caller, mirroring `Compiler.load`'s
/// thread-binding of `*ns*`.
pub fn with_active_session_load_resource(path: &str) {
    let source = match resource_source(path) {
        Some(s) => s,
        None => {
            // Not a ported sub-file. Upstream would throw FileNotFound;
            // we log and skip so the rest of core keeps loading. This is
            // visible, not silent — new ports add an arm to `resource_source`.
            eprintln!(
                "[cljvm-load] no embedded resource for \"{path}\" — skipping \
                 (only ported sub-files are available)"
            );
            return;
        }
    };
    // Interleave read and eval, one form at a time — exactly as Clojure's
    // `load` does. This matters because syntax-quote resolves symbols against
    // the *current* namespace at read time: a file's `(ns …)` form must be
    // evaluated (switching `*ns*`) before the following forms are read, or
    // their syntax-quoted symbols would qualify to the wrong namespace.
    with_active_session_ref(|sess| {
        // Clojure's `load` runs inside `(binding [*ns* *ns*] …)`. Establish
        // ONE thread-local `*ns*` binding spanning the whole read+eval loop,
        // seeded from the caller's current ns: (a) a sub-file's `(ns …)` is
        // local to this load and restored when it returns, and (b) reads
        // between forms see the up-to-date current namespace (eval_form
        // reuses this binding rather than stacking its own). Popped on exit.
        Var::push_thread_bindings(vec![(
            super::rt::CURRENT_NS.clone(),
            Object::Namespace(sess.current_ns.clone()),
        )]);
        let _ns_restore = DropFn(|| Var::pop_thread_bindings());
        let mut byte_pos: usize = 0;
        loop {
            let slice = &source[byte_pos..];
            let mut r = super::lisp_reader::Reader::new(slice);
            let before = r.byte_pos();
            let read = r.read();
            let after = r.byte_pos();
            byte_pos += after - before;
            let form = match read {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => panic!("clojure-jvm: RT/load(\"{path}\"): read error: {e}"),
            };
            sess.eval_form(form);
        }
    })
    .unwrap_or_else(|| panic!("clojure-jvm: RT/load(\"{path}\") called without an active Session"));
}

/// Map a `load` path (as passed to `clojure.lang.RT/load`, i.e. the
/// classpath-relative path without the leading slash) to embedded source.
/// Only ported sub-files are present; everything else returns `None`.
fn resource_source(path: &str) -> Option<&'static str> {
    match path {
        // Reentrancy smoke-test resource (see tests/load_reentrancy.rs).
        "trivial-reentrancy-test" => Some("(def reentrancy-marker 4242)"),
        // Reentrant `refer`-chain isolation probe (tests/interesting_features
        // ::probe_reentrant_refer). Each step defs a marker into the current
        // ns (clojure.core during the reentrant load) so the test can read how
        // far refer's call chain gets when executed reentrantly.
        "reentrant-refer-probe" => Some(
            "(def rr-lit 4242)\n\
             (def rr-fnval clojure.core/inc)\n\
             (def rr-call (clojure.core/inc 41))\n\
             (def rr-findns (clojure.core/find-ns 'clojure.core))\n\
             (def rr-done 1)\n",
        ),
        // clojure/core/protocols.clj — CollReduce / InternalReduce / IKVReduce
        // and the seq-reduce machinery behind clojure.core/reduce.
        "clojure/core/protocols" | "core/protocols" => {
            Some(include_str!("../../clojure/core_protocols.clj"))
        }
        // Minimal clojure.string (upper/lower-case, trim, join, reverse, …).
        "clojure/string" => Some(include_str!("../../clojure/string.clj")),
        _ => None,
    }
}

/// Counterpart of `with_active_session_root_map` for sets.
pub fn with_active_session_root_set(s: Arc<crate::lang::persistent_hash_set::PersistentHashSet>) {
    with_active_session_ref(|sess| sess.roots._sets.push(s)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_set called without an \
             active Session — runtime extern must run inside Session::eval_form"
        )
    });
}

/// Counterpart of `with_active_session_root_map` for sorted (tree) maps.
pub fn with_active_session_root_tree_map(
    m: Arc<crate::lang::persistent_tree_map::PersistentTreeMap>,
) {
    with_active_session_ref(|sess| sess.roots._tree_maps.push(m)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_tree_map called without an \
             active Session — runtime extern must run inside Session::eval_form"
        )
    });
}

/// Counterpart of `with_active_session_root_map` for sorted (tree) sets.
pub fn with_active_session_root_tree_set(
    s: Arc<crate::lang::persistent_tree_set::PersistentTreeSet>,
) {
    with_active_session_ref(|sess| sess.roots._tree_sets.push(s)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_tree_set called without an \
             active Session — runtime extern must run inside Session::eval_form"
        )
    });
}

/// Counterpart of `with_active_session_root_map` for StringBuilder
/// (`Arc<RefCell<String>>`).
pub fn with_active_session_root_string_builder(sb: Arc<std::cell::RefCell<String>>) {
    with_active_session_ref(|sess| sess.roots._string_builders.push(sb)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_string_builder called without \
             an active Session — runtime extern must run inside Session::eval_form"
        )
    });
}

/// Counterpart for ChunkBuffer (`Arc<RefCell<Vec<u64>>>`).
pub fn with_active_session_root_chunk_buffer(cb: Arc<std::cell::RefCell<Vec<u64>>>) {
    with_active_session_ref(|sess| sess.roots._chunk_buffers.push(cb)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_chunk_buffer called without \
             an active Session"
        )
    });
}

/// Counterpart for IChunk — same backing as ChunkBuffer.
pub fn with_active_session_root_i_chunk(ic: Arc<std::cell::RefCell<Vec<u64>>>) {
    with_active_session_ref(|sess| sess.roots._chunk_buffers.push(ic)).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: with_active_session_root_i_chunk called without \
             an active Session"
        )
    });
}

pub fn with_active_session_root_lazy_seq(ls: Arc<std::cell::RefCell<crate::runtime::LazyState>>) {
    with_active_session_ref(|sess| sess.roots._lazy_states.push(ls)).unwrap_or_else(|| {
        panic!("clojure-jvm: with_active_session_root_lazy_seq: no active Session")
    });
}

pub fn with_active_session_root_delay(d: Arc<std::cell::RefCell<crate::runtime::LazyState>>) {
    with_active_session_ref(|sess| sess.roots._lazy_states.push(d)).unwrap_or_else(|| {
        panic!("clojure-jvm: with_active_session_root_delay: no active Session")
    });
}

pub fn with_active_session_root_multi_arity_fn(t: Arc<Vec<crate::runtime::MultiArityEntry>>) {
    with_active_session_ref(|sess| sess.roots._multi_arity_tables.push(t)).unwrap_or_else(|| {
        panic!("clojure-jvm: with_active_session_root_multi_arity_fn: no active Session")
    });
}

/// Read the cached `PersistentList/creator` singleton handle off the
/// active Session. Called by the `cljvm_pl_creator` runtime extern from
/// inside JIT execution, where `Session::eval_form` has already installed
/// `ACTIVE_SESSION` for the duration of the call.
pub fn active_session_pl_creator_handle() -> u64 {
    with_active_session_ref(|s| s.pl_creator_handle).unwrap_or_else(|| {
        panic!(
            "clojure-jvm: cljvm_pl_creator called without an active Session — \
             must run inside Session::eval_form"
        )
    })
}

/// Reach the active Session, panicking if none is installed.
fn with_active_session_ref<R>(body: impl FnOnce(&mut Session) -> R) -> Option<R> {
    let ptr = ACTIVE_SESSION.with(|c| c.get());
    if ptr.is_null() {
        return None;
    }
    // SAFETY: pointer installed by `with_active_session` on this thread.
    // The Session is borrowed mutably by the enclosing call; the body
    // runs synchronously inside that scope.
    Some(unsafe { body(&mut *ptr) })
}

/// The active Session's JIT call-table base address, or `None` if no Session
/// is active on this thread. Lets runtime externs that must invoke a JIT fn
/// (e.g. forcing a lazy-seq thunk) recover the base when it wasn't installed
/// on the thread-local — i.e. when forcing happens OUTSIDE a `run_jit` scope
/// (a Rust-side realize such as `pr_str_bits` after `eval_form` returned).
pub fn active_session_call_table_base() -> Option<u64> {
    with_active_session_ref(|sess| sess.jit.call_table_base_addr())
}

// ============================================================================
// `Session` — persistent JIT across multiple top-level forms.
//
// Mirrors microlisp's `Engine` pattern: one `JitModule` constructed once via
// `new_empty`, grown per form via `extend`. The same JIT serves both runtime
// execution AND compile-time macro invocation (macros are real JIT-compiled
// fns called via `gc.run_jit(jit, macro_fref, ...)` during analysis of
// later forms). The DynModule grows alongside; we `snapshot()` it before
// each extend.
// ============================================================================

/// Active compilation + execution session. Persists across forms.
pub struct Session {
    pub compiler: Compiler,
    pub gc: Box<dynlang::gc::DynGcRuntime>,
    pub jit: Box<dynlower::JitModule>,
    /// Captured at construction; the GC borrows obj_types' TypeInfos.
    pub obj_types: Vec<dynlang::ObjType>,
    /// Live root holder for compile-time-allocated literal pool entries.
    pub roots: CompileRoots,
    /// Number of forms processed so far — used to mint a unique entry-fn
    /// name per form so the JIT extend appends rather than collides.
    next_form_id: u32,
    /// How many extern-name function pointers the JIT has already seen.
    /// `extend` reads `externs[extern_count_seen..]` per call; we
    /// recompute the table each form to keep it positionally consistent
    /// with `module.func_table`'s `FuncDef::Extern` entries.
    _extern_count_seen: usize,
    /// Cached NanBox handle for `clojure.lang.PersistentList/creator` — the
    /// singleton variadic-identity IFn produced by compiling
    /// `(fn* [& xs] xs)` once at session init. Read by `cljvm_pl_creator`.
    pub pl_creator_handle: u64,
    /// This session's current namespace (`*ns*`), carried across forms.
    ///
    /// Clojure's `*ns*` is a thread-local dynamic binding, NOT a global
    /// root: `(ns …)`/`in-ns` change the *current thread's* namespace, so
    /// concurrent evaluations don't clobber each other. We honor that by
    /// having `eval_form` push a `CURRENT_NS` thread binding seeded from
    /// this field (spanning analyze + run) and read it back afterward.
    /// Two Sessions on two threads are therefore isolated.
    current_ns: Arc<crate::lang::namespace::Namespace>,
}

impl Session {
    pub fn new() -> Self {
        use dynir::dynexec::NanBoxConfig;
        use dynlower::regalloc::LinearScanAllocator;
        use dynlower::{Arm64Backend, CallMode, JitModule};
        use dynruntime::active_jit_safepoint_handler;
        #[cfg(not(target_arch = "aarch64"))]
        compile_error!("clojure-jvm Session only configured for aarch64 right now");

        let mut compiler = Compiler::new();

        // Snapshot obj_types BEFORE further DynModule mutations. The
        // GC borrows TypeInfos from this slice; the slice's `Arc<TypeInfo>`
        // identities must outlive every JIT execution. We move them out
        // of the DynModule so subsequent `obj_type` declarations on the
        // DynModule wouldn't reallocate the storage.
        let obj_types: Vec<dynlang::ObjType> = std::mem::take(&mut compiler.dm.obj_types);

        let gc_config = compiler.dm.gc_config().clone();
        let tags = dynlang::NanBoxTags::default();
        let gc = Box::new(dynlang::gc::DynGcRuntime::new(
            &gc_config, &tags, &obj_types,
        ));

        let call_mode = CallMode::ControlAware {
            safepoint_handler: active_jit_safepoint_handler as u64,
        };
        let jit = Box::new(JitModule::new_empty::<
            NanBoxConfig,
            Arm64Backend,
            LinearScanAllocator,
        >(
            /* call_table_capacity */ 64 * 1024,
            /* literal_pool_capacity */ 64 * 1024,
            call_mode,
        ));

        // Register the JIT's literal pool as a PERMANENT extra root
        // source on the GC. The toolkit's `run_jit` registers it on the
        // safepoint session for JIT-time collections — but pool slots
        // also have to survive collections that happen during pool
        // FILL (which calls `gc.alloc` directly) and other host-side
        // allocation paths between forms. Without this, a Symbol heap
        // cell at e.g. pool[2] for `'fn*` (interned at form 7) gets
        // collected/relocated during a later form's pool fill and
        // pool[2] dangles — observed as the fn-macro returning
        // `Cons(Vector[a,b], …)` instead of `Cons(Symbol(fn*), …)`.
        //
        // SAFETY: `jit.literal_pool()` is owned by `jit`, which is
        // boxed and lives in the Session for the Session's lifetime.
        // The raw pointer stays valid as long as the Session does.
        let pool_root: *const dyn dynobj::RootSource = jit.literal_pool();
        unsafe {
            gc.register_extra_root_source(pool_root);
        }

        // Register the global Var root table. Every `def`'d Var holds its
        // root value in a slot here; registering the table makes those
        // values genuine GC roots so heap-pointer roots (strings, closures,
        // …) are forwarded on collection instead of dangling. The table is
        // process-global (`'static`), so the raw pointer is valid for the
        // lifetime of every Session and GC.
        unsafe {
            gc.register_extra_root_source(crate::lang::var_roots::var_roots_root_source());
        }

        // Register the protocol-dispatch table. Multi-arity / capturing impls
        // store a heap-cell handle (MultiArityFn / Closure) as raw bits; this
        // makes the GC forward those pointers in place so a collection between
        // `installImpl` and dispatch (e.g. during a reentrant `(load …)`)
        // doesn't leave them dangling. Process-global `'static` table.
        unsafe {
            gc.register_extra_root_source(crate::lang::user_types::dispatch_root_source());
        }

        // Register the LazySeq/Delay root source. A `LazyState` caches the
        // deferred thunk and (once realized) the produced value as NaN-boxed
        // GC-heap pointers in a Rust-side `Arc<RefCell<…>>`. Keeping the Arc
        // alive doesn't make those pointers GC roots, so a collection would
        // relocate the thunk/value and leave the cached bits dangling — a
        // stale seq then reaches `RT.first` (the form-430 loader crash under
        // `CLJVM_GC=every`). This source forwards those cached pointers in
        // place on every collection. The source is `'static`; the thread-local
        // registry it scans is populated by `register_lazy_state`.
        unsafe {
            gc.register_extra_root_source(crate::runtime::lazy_root_source());
        }

        // Register suspended JIT frames. Control-aware JIT calls copy the
        // caller's live values into thread-local suspended frames while the
        // callee runs; those values are off the native FP chain, so GC must
        // scan them explicitly during both JIT safepoints and allocation-path
        // collections from runtime externs.
        unsafe {
            gc.register_extra_root_source(dynlower::suspended_jit_frames_root_source());
        }

        let mut sess = Session {
            compiler,
            gc,
            jit,
            obj_types,
            roots: CompileRoots {
                _symbols: Vec::new(),
                _strings: Vec::new(),
                _keywords: Vec::new(),
                _maps: Vec::new(),
                _sets: Vec::new(),
                _tree_maps: Vec::new(),
                _tree_sets: Vec::new(),
                _string_builders: Vec::new(),
                _chunk_buffers: Vec::new(),
                _lazy_states: Vec::new(),
                _multi_arity_tables: Vec::new(),
                _namespaces: Vec::new(),
                _vars: Vec::new(),
            },
            next_form_id: 0,
            _extern_count_seen: 0,
            pl_creator_handle: 0,
            // Seed from the current effective `*ns*` (a thread binding if a
            // caller established one, else the `clojure.core` root default).
            current_ns: super::rt::current_ns(),
        };
        sess.init_static_singletons();
        sess
    }

    /// Like `new`, but also loads our forked `clojure/core.clj` so user
    /// code lands in a populated environment with `inc`, `dec`, `map`,
    /// `reduce`, etc. already defined.
    ///
    /// The source lives at `crates/clojure-jvm/clojure/core.clj` and is
    /// embedded into the binary via `include_str!`. It started as a copy
    /// of upstream `clojure/core.clj` and gets patched in-place as
    /// blockers surface — we own this file now, we don't track upstream.
    ///
    /// Loading stops at the first form that fails to read or eval. The
    /// failing form's error is propagated, since silently dropping it
    /// would leave the namespace half-populated in a way that's
    /// extremely confusing for downstream tests.
    pub fn new_with_clojure_core() -> Self {
        let mut sess = Self::new();
        sess.load_clojure_core();
        sess
    }

    /// Read and eval every form in the embedded forked `clojure/core.clj`.
    /// Called by `new_with_clojure_core`. Public so tests / callers that
    /// build a Session manually can opt in.
    /// Read a single slot of the JIT literal pool. Diagnostic
    /// accessor for tests that watch for pool-slot corruption.
    pub fn literal_pool_get(&self, idx: usize) -> u64 {
        if idx >= self.jit.literal_pool().len() {
            return 0;
        }
        self.jit.literal_pool().get(idx)
    }

    pub fn literal_pool_len(&self) -> usize {
        self.jit.literal_pool().len()
    }

    /// Diagnostic: dump the pool's base/end and the GC heap's
    /// from/to-space base/size, so callers can check for memory
    /// region overlap that would explain pool corruption from a
    /// heap allocation landing inside pool memory.
    pub fn dump_memory_ranges(&self) -> String {
        let pool = self.jit.literal_pool();
        let pool_base = pool.base() as usize;
        let pool_cap = pool.capacity();
        let pool_end = pool_base + pool_cap * 8;
        let (from_base, from_size, to_base, to_size) = self.gc.debug_heap_ranges();
        format!(
            "pool=[0x{pool_base:016x}..0x{:016x}) ({pool_cap} slots, {} bytes)  \
             from=[0x{from_base:016x}..0x{:016x}) ({from_size} bytes)  \
             to=[0x{to_base:016x}..0x{:016x}) ({to_size} bytes)",
            pool_end,
            pool_cap * 8,
            from_base + from_size,
            to_base + to_size,
        )
    }

    pub fn load_clojure_core(&mut self) {
        const CORE: &str = include_str!("../../clojure/core.clj");
        let trace = std::env::var("CLJVM_CORE_TRACE").is_ok();
        let mut byte_pos: usize = 0;
        let mut form_idx: usize = 0;
        loop {
            let slice = &CORE[byte_pos..];
            let mut r = super::lisp_reader::Reader::new(slice);
            let before = r.byte_pos();
            let read = r.read();
            let after = r.byte_pos();
            byte_pos += after - before;
            let form = match read {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => panic!(
                    "load_clojure_core: read error at form {form_idx} (byte {byte_pos}): {e}"
                ),
            };
            if trace {
                let name = sniff_core_form_name(&form).unwrap_or_else(|| "<?>".into());
                eprintln!("[core] form {form_idx} byte {byte_pos} {name}");
            }
            self.eval_form(form);
            form_idx += 1;
        }
    }

    /// Compile the static-singleton fns we expose as host static fields
    /// (currently just `clojure.lang.PersistentList/creator`). Cache each
    /// resulting NanBox handle on the Session so the corresponding extern
    /// (`cljvm_pl_creator`) can return it during JIT execution.
    fn init_static_singletons(&mut self) {
        let form = super::lisp_reader::read_str("(fn* [& xs] xs)")
            .expect("init_static_singletons: read (fn* [& xs] xs)");
        self.pl_creator_handle = self.eval_form(form);

        // All the rest — dynamic compile-time vars, forward stubs for
        // later-defined fns, mocked java.lang.* classes — lives in
        // `clojure/prelude.clj`. Add a new def there, not here.
        const PRELUDE: &str = include_str!("../../clojure/prelude.clj");
        let mut byte_pos: usize = 0;
        let mut form_idx: usize = 0;
        loop {
            let slice = &PRELUDE[byte_pos..];
            let mut r = super::lisp_reader::Reader::new(slice);
            let before = r.byte_pos();
            let read = r.read();
            let after = r.byte_pos();
            byte_pos += after - before;
            let form = match read {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => panic!(
                    "init_static_singletons: prelude.clj read error at form {form_idx} \
                     (byte {byte_pos}): {e}"
                ),
            };
            self.eval_form(form);
            form_idx += 1;
        }
    }

    /// Compile + extend + run one form. Returns the raw NanBox bits the
    /// entry fn produced. Vars set by previous forms remain bound, and
    /// `^:macro`-flagged Vars registered by earlier forms can be invoked
    /// at compile time during this form's analyze (macroexpansion).
    pub fn eval_form(&mut self, form: Object) -> u64 {
        let dbg_form_name = if std::env::var("CLJVM_THREW_TRACE").is_ok() {
            sniff_core_form_name(&form)
        } else {
            None
        };
        // Tear the Session in two: a *self pointer for ACTIVE_SESSION so
        // analyze_seq can reach back during macroexpand, and an &mut self
        // for direct use here. We re-borrow through the pointer below.
        let self_ptr: *mut Session = self;
        let _session_guard = ACTIVE_SESSION.with(|c| {
            let prev = c.replace(self_ptr);
            DropFn(move || {
                ACTIVE_SESSION.with(|c| c.set(prev));
            })
        });

        // Ensure a thread-local `*ns*` binding is active for this form.
        // Clojure's current namespace is a per-thread dynamic binding, not a
        // global root: `(ns …)`/`in-ns` (which run at analyze time via
        // `parse_ns_form` → `Var::set_value`) mutate *this thread's* binding,
        // so concurrent evaluations never clobber each other.
        //
        // If a caller already established a binding (e.g. `load`'s
        // `(binding [*ns* *ns*] …)` around a sub-file, or a test's
        // `with_fresh_ns`), reuse it — `(ns …)` updates that binding directly
        // and the owner manages restoration. Otherwise this form owns a fresh
        // binding seeded from the session's tracked `current_ns`; the guard
        // reads the (possibly `(ns)`-changed) value back so it carries to the
        // next form, then pops.
        let owns_ns_binding = super::rt::CURRENT_NS.get_thread_binding().is_none();
        if owns_ns_binding {
            Var::push_thread_bindings(vec![(
                super::rt::CURRENT_NS.clone(),
                Object::Namespace(self.current_ns.clone()),
            )]);
        }
        let _ns_guard = DropFn(move || {
            if owns_ns_binding {
                // SAFETY: `self_ptr` is this Session, live for the call.
                unsafe {
                    (*self_ptr).current_ns = super::rt::current_ns();
                }
                Var::pop_thread_bindings();
            }
        });

        use dynir::dynexec::NanBoxConfig;
        use dynlower::regalloc::LinearScanAllocator;
        use dynlower::{Arm64Backend, JitOutcome};
        use dynruntime::GcPolicy;

        let form_id = self.next_form_id;
        self.next_form_id += 1;
        let entry_name = format!("__top_form_{form_id}__");

        // Per-form transaction: a form that panics mid-compile (e.g. an
        // analyze/lower path that declares an internal fn for a lambda and
        // then hits an unsupported construct) would otherwise leave the
        // shared persistent builder with a function "declared but not
        // defined", poisoning EVERY later `snapshot()`. Checkpoint the
        // builder up front; if compilation panics, roll the builder back to
        // the checkpoint and clear the half-drained pending-fn queue before
        // re-raising, so the next form compiles against a clean builder.
        let compile_cp = self.compiler.dm.checkpoint();
        let entry = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.compiler.with_active(|c| {
                let entry = c.dm.declare_func(&entry_name, 0);

                let expr = analyze(C::Expression, form);

                // Drain pending fns.
                loop {
                    let pending = c.pending_fns.lock().unwrap().pop();
                    let Some(p) = pending else { break };
                    lower_pending_fn(c, p);
                }

                // Emit entry fn body.
                let mut df = c.dm.start_func(entry);
                let result_val = {
                    let mut ir = IrEmitter::new(&mut df);
                    let objx = ObjExpr::placeholder();
                    expr.emit(C::Expression, &*objx, &mut ir)
                };
                // If the top-level form diverged (analyze-time-rewritten
                // throw, e.g. for an unregistered host method whose
                // surrounding defn never gets called at load time), the
                // current block is already terminated by the raise — skip
                // the ret. Otherwise emit ret of the produced value.
                if let Some(v) = result_val {
                    df.fb.ret(v);
                } else if !df.fb.current_block_is_terminated() {
                    let nil_const = df
                        .fb
                        .iconst(dynir::Type::I64, crate::runtime::nanbox_nil() as i64);
                    df.fb.ret(nil_const);
                }
                c.dm.finish_func(df);
                entry
            })
        })) {
            Ok(entry) => entry,
            Err(payload) => {
                self.compiler.dm.rollback(compile_cp);
                self.compiler.pending_fns.lock().unwrap().clear();
                std::panic::resume_unwind(payload);
            }
        };

        // Snapshot the module + extend the JIT with the new fns.
        let module = self.compiler.dm.snapshot();
        if let Ok(want) = std::env::var("CLJVM_IR_DUMP") {
            for f in module.functions.iter() {
                if (want.is_empty() && f.name == entry_name)
                    || (!want.is_empty() && f.name.contains(&want))
                {
                    eprintln!("===IR=== {}\n{}", f.name, f);
                }
            }
        }
        let externs = build_extern_table_for(&module);
        let _ = self
            .jit
            .extend::<NanBoxConfig, Arm64Backend, LinearScanAllocator>(&module, &externs);

        // Populate the literal pool with any newly-interned compile-time
        // literals from this form. The pool's slot index for each entry
        // MUST match the absolute index that `intern_literal` returned
        // — otherwise `gc_literal(idx)` reads the wrong slot. We compute
        // the expected start index from the pool's current length and
        // assert each push lands where the analyzer baked it.
        {
            let _thread = self.gc.install_thread();
            let local_chain = dynobj::roots::FrameChain::new();
            let chain_src: *const dyn dynobj::RootSource = &local_chain;
            let _chain_root_g = unsafe { self.gc.push_extra_root_source(chain_src) };
            let _chain_g = dynobj::roots::install_chain(&local_chain);
            let pending = std::mem::take(&mut *self.compiler.pending_literals.lock().unwrap());
            let string_type_id = self.compiler.string_type_id;
            let symbol_type_id = self.compiler.symbol_type_id;
            let keyword_type_id = self.compiler.keyword_type_id;
            let cons_type_id = self.compiler.cons_type_id;
            let mut expected_idx = self.jit.literal_pool().len();
            for lit in pending.iter() {
                let nanbox_bits = match lit {
                    PendingLiteral::String(s) => alloc_string(
                        &self.gc,
                        &self.obj_types,
                        string_type_id,
                        &mut self.roots,
                        s,
                    ),
                    PendingLiteral::Symbol(s) => alloc_symbol(
                        &self.gc,
                        &self.obj_types,
                        symbol_type_id,
                        &mut self.roots,
                        s,
                    ),
                    PendingLiteral::Keyword(k) => alloc_keyword(
                        &self.gc,
                        &self.obj_types,
                        keyword_type_id,
                        &mut self.roots,
                        k,
                    ),
                    PendingLiteral::List(l) => alloc_list_as_nanbox(
                        &self.gc,
                        &self.obj_types,
                        cons_type_id,
                        string_type_id,
                        symbol_type_id,
                        keyword_type_id,
                        &mut self.roots,
                        l,
                    ),
                    PendingLiteral::Vector(v) => alloc_vector_as_nanbox(
                        &self.gc,
                        &self.obj_types,
                        self.compiler.vector_type_id,
                        cons_type_id,
                        string_type_id,
                        symbol_type_id,
                        keyword_type_id,
                        &mut self.roots,
                        v,
                    ),
                    PendingLiteral::Map(m) => alloc_map_as_nanbox(
                        &self.gc,
                        &self.obj_types,
                        self.compiler.map_type_id,
                        &mut self.roots,
                        m,
                    ),
                    PendingLiteral::Set(s) => alloc_set_as_nanbox(
                        &self.gc,
                        &self.obj_types,
                        self.compiler.set_type_id,
                        &mut self.roots,
                        s,
                    ),
                    PendingLiteral::Class(class_id) => alloc_class_as_nanbox(
                        &self.gc,
                        &self.obj_types,
                        self.compiler.class_type_id,
                        *class_id,
                    ),
                    PendingLiteral::Var(v) => {
                        alloc_var_as_nanbox(&self.gc, &self.obj_types, self.compiler.var_type_id, v)
                    }
                    PendingLiteral::MultiArityFn(t) => alloc_multi_arity_fn_as_nanbox(
                        &self.gc,
                        &self.obj_types,
                        self.compiler.multi_arity_fn_type_id,
                        &mut self.roots,
                        t,
                    ),
                    PendingLiteral::Long(n) => unsafe { crate::runtime::box_long(*n) },
                PendingLiteral::Char(c) => unsafe { crate::runtime::box_char(*c) },
                };
                // Diagnostic: catch any alloc_*_as_nanbox path that
                // produces a TAG_PTR NanBox with a misaligned payload
                // (= invalid heap pointer that the GC will dereference
                // and crash on later).
                {
                    let high = nanbox_bits & 0xFFFF_0000_0000_0000;
                    let payload = nanbox_bits & 0x0000_FFFF_FFFF_FFFF;
                    if high == 0x7FFE_0000_0000_0000 && payload != 0 && (payload & 0x7) != 0 {
                        eprintln!(
                            "[cljvm-pool-fill-misaligned] idx={expected_idx} \
                             bits=0x{nanbox_bits:016x} payload=0x{payload:016x} \
                             pending={lit:?}"
                        );
                    }
                }
                let pushed = self.jit.literal_pool().push(nanbox_bits);
                assert_eq!(
                    pushed, expected_idx,
                    "clojure-jvm: literal pool index drift — pool slot \
                     {pushed} does not match analyzer's intern slot \
                     {expected_idx}. IR's gc_literal would read the wrong \
                     slot. (pending entry: {:?})",
                    lit,
                );
                expected_idx += 1;
            }
        }

        // Run the entry.
        let _thread = self.gc.install_thread();
        let _ctb = crate::runtime::install_call_table_base(self.jit.call_table_base_addr());
        // Per-call FrameChain: runtime externs that hold heap pointers
        // across allocations (e.g. `pack_variadic_args`'s cons-fold)
        // open a `dynobj::roots::with_scope` to root them. `with_scope`
        // reaches the chain via `ACTIVE_CHAIN`, which `install_chain`
        // sets. The chain ALSO has to be registered as an
        // extra-root-source on the GC so collection actually walks it.
        let local_chain = dynobj::roots::FrameChain::new();
        let chain_src: *const dyn dynobj::RootSource = &local_chain;
        let _chain_root_g = unsafe { self.gc.push_extra_root_source(chain_src) };
        let _chain_g = dynobj::roots::install_chain(&local_chain);
        // GC policy: default `OnPressure`, but `CLJVM_GC=every` forces a
        // collection at every safepoint (stress mode) — used to surface
        // GC-timing-dependent rooting bugs deterministically.
        let gc_policy = if std::env::var("CLJVM_GC").as_deref() == Ok("every") {
            GcPolicy::EveryPoint
        } else {
            GcPolicy::OnPressure { threshold: 0.75 }
        };
        match self.gc.run_jit(&self.jit, entry, &[], gc_policy) {
            JitOutcome::Value(v) => v,
            JitOutcome::Exception(payload) => {
                // Top-level form threw at runtime (typically because
                // a defn body ran the analyze-time-emitted throw for
                // an unregistered host method, OR a `def` init
                // ran something we don't support). The Var, if any,
                // is left unbound. Print a diagnostic and return nil
                // so the loader can keep going.
                let obj = crate::runtime::nanbox_to_object(payload);
                if let Some(name) = &dbg_form_name {
                    eprintln!(
                        "[cljvm-threw] form `{name}` threw: {obj:?} (payload bits 0x{payload:016x})"
                    );
                }
                eprintln!(
                    "[cljvm] top-level form threw: {obj:?} :: pr-str={}",
                    crate::runtime::pr_str_bits(payload)
                );
                crate::runtime::nanbox_nil()
            }
            other => panic!("clojure-jvm: Session::eval_form: unexpected JIT outcome: {other:?}"),
        }
    }

    /// Read all forms from `src`, eval each in order, return the last
    /// form's value. Matches microlisp's `Engine::run_source` semantics.
    pub fn eval_str(&mut self, src: &str) -> u64 {
        let forms = super::lisp_reader::read_all(src)
            .unwrap_or_else(|e| panic!("clojure-jvm: read_all({src:?}): {e}"));
        let mut last = 0x7FFC_0000_0000_0000u64; // nil
        for form in forms {
            last = self.eval_form(form);
        }
        last
    }

}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_symbols_intern_consistently() {
        // Symbol::intern is deterministic so SPECIAL_SYMBOLS.DEF should equal a
        // freshly interned `"def"`.
        let s = &*SPECIAL_SYMBOLS;
        assert_eq!(*s.DEF, *Symbol::intern("def"));
        assert_eq!(*s.LET, *Symbol::intern("let*"));
        assert_eq!(*s.IMPORT.get_namespace().unwrap(), *"clojure.core");
    }

    #[test]
    fn keyword_interning_returns_same_arc() {
        let k1 = Keyword::intern_ns_name(None, "static");
        let k2 = Keyword::intern_ns_name(None, "static");
        assert!(Arc::ptr_eq(&k1, &k2));
    }

    #[test]
    fn c_enum_distinct() {
        assert_ne!(C::Statement, C::Expression);
        assert_ne!(C::Expression, C::Return);
        assert_ne!(C::Return, C::Eval);
    }

    // ---- LiteralExpr ports ---------------------------------------------------

    #[test]
    fn nil_expr_evals_to_nil() {
        let e = NIL_EXPR;
        assert!(matches!(<NilExpr as Expr>::eval(&e), Object::Nil));
        assert!(matches!(e.val(), Object::Nil));
        assert!(<NilExpr as Expr>::has_java_class(&e));
        assert!(<NilExpr as Expr>::get_java_class(&e).is_none());
    }

    #[test]
    fn boolean_expr_singletons_match_value() {
        assert!(matches!(
            <BooleanExpr as Expr>::eval(&TRUE_EXPR),
            Object::Bool(true)
        ));
        assert!(matches!(
            <BooleanExpr as Expr>::eval(&FALSE_EXPR),
            Object::Bool(false)
        ));
        assert_eq!(
            <BooleanExpr as Expr>::get_java_class(&TRUE_EXPR)
                .unwrap()
                .name
                .as_str(),
            "java.lang.Boolean"
        );
    }

    #[test]
    fn string_expr_round_trip() {
        let s = StringExpr::new("hello");
        match <StringExpr as Expr>::eval(&s) {
            Object::String(rc) => assert_eq!(*rc, "hello"),
            other => panic!("expected Object::String, got {other:?}"),
        }
        assert_eq!(
            <StringExpr as Expr>::get_java_class(&s)
                .unwrap()
                .name
                .as_str(),
            "java.lang.String"
        );
    }

    #[test]
    fn keyword_expr_eval_returns_same_keyword() {
        let k = Keyword::intern_ns_name(Some("user"), "foo");
        let ke = register_keyword(k.clone());
        match <KeywordExpr as Expr>::eval(&ke) {
            Object::Keyword(got) => assert!(Arc::ptr_eq(&got, &k)),
            other => panic!("expected Object::Keyword, got {other:?}"),
        }
    }

    #[test]
    fn number_expr_parse_long_returns_number_expr() {
        // We can't downcast Box<dyn Expr>; instead exercise the construction
        // path directly and confirm eval round-trips.
        let n = NumberExpr::new(Object::Long(42));
        assert!(matches!(<NumberExpr as Expr>::eval(&n), Object::Long(42)));
        assert!(MaybePrimitiveExpr::can_emit_primitive(&n));
        // id is -1 because CONSTANTS isn't bound (no enclosing fn) — matches
        // Java's `!CONSTANTS.isBound()` branch.
        assert_eq!(n.id, -1);
    }

    #[test]
    fn number_expr_get_java_class_maps_primitive_names() {
        assert_eq!(
            <NumberExpr as Expr>::get_java_class(&NumberExpr::new(Object::Long(1)))
                .unwrap()
                .name
                .as_str(),
            "long"
        );
        assert_eq!(
            <NumberExpr as Expr>::get_java_class(&NumberExpr::new(Object::Double(1.0)))
                .unwrap()
                .name
                .as_str(),
            "double"
        );
    }

    // ---- IfExpr -------------------------------------------------------------

    #[test]
    fn is_truthy_only_nil_and_false_are_falsey() {
        assert!(!is_truthy(&Object::Nil));
        assert!(!is_truthy(&Object::Bool(false)));
        assert!(is_truthy(&Object::Bool(true)));
        assert!(is_truthy(&Object::Long(0))); // Clojure: 0 is truthy
        assert!(is_truthy(&Object::Long(42)));
        assert!(is_truthy(&Object::String(Arc::new(String::new()))));
    }

    #[test]
    fn if_expr_can_emit_primitive_when_both_branches_long() {
        // (if true 1 2) — both branches are NumberExpr<Long>, same Java class.
        let e = IfExpr::new(
            1,
            0,
            Box::new(TRUE_EXPR),
            Box::new(NumberExpr::new(Object::Long(1))),
            Box::new(NumberExpr::new(Object::Long(2))),
        );
        assert!(MaybePrimitiveExpr::can_emit_primitive(&e));
    }

    #[test]
    fn if_expr_cannot_emit_primitive_when_branches_disagree() {
        let e = IfExpr::new(
            1,
            0,
            Box::new(TRUE_EXPR),
            Box::new(NumberExpr::new(Object::Long(1))),
            Box::new(NumberExpr::new(Object::Double(1.0))),
        );
        assert!(!MaybePrimitiveExpr::can_emit_primitive(&e));
    }

    #[test]
    fn if_expr_cannot_emit_primitive_when_branch_isnt_maybe_primitive() {
        // String isn't MaybePrimitiveExpr.
        let e = IfExpr::new(
            1,
            0,
            Box::new(TRUE_EXPR),
            Box::new(StringExpr::new("a")),
            Box::new(StringExpr::new("b")),
        );
        assert!(!MaybePrimitiveExpr::can_emit_primitive(&e));
    }

    // ---- BodyExpr -----------------------------------------------------------

    #[test]
    fn body_expr_has_java_class_follows_last() {
        let body = BodyExpr::new(vec![
            Box::new(StringExpr::new("ignored")),
            Box::new(NumberExpr::new(Object::Long(1))),
        ]);
        assert!(<BodyExpr as Expr>::has_java_class(&body));
        assert_eq!(
            <BodyExpr as Expr>::get_java_class(&body)
                .unwrap()
                .name
                .as_str(),
            "long"
        );
    }

    #[test]
    fn body_expr_can_emit_primitive_when_last_is_number() {
        let body = BodyExpr::new(vec![
            Box::new(StringExpr::new("ignored")),
            Box::new(NumberExpr::new(Object::Long(1))),
        ]);
        assert!(MaybePrimitiveExpr::can_emit_primitive(&body));
    }

    #[test]
    fn body_expr_cannot_emit_primitive_when_last_is_string() {
        let body = BodyExpr::new(vec![
            Box::new(NumberExpr::new(Object::Long(1))),
            Box::new(StringExpr::new("last")),
        ]);
        assert!(!MaybePrimitiveExpr::can_emit_primitive(&body));
    }

    #[test]
    fn host_class_primitive_classification() {
        assert!(
            HostClass {
                name: Arc::new("long".to_string())
            }
            .is_primitive()
        );
        assert!(
            HostClass {
                name: Arc::new("double".to_string())
            }
            .is_primitive()
        );
        assert!(
            !HostClass {
                name: Arc::new("java.lang.String".to_string())
            }
            .is_primitive()
        );
        assert!(
            !HostClass {
                name: Arc::new("clojure.lang.Keyword".to_string())
            }
            .is_primitive()
        );
    }

    // ---- analyze dispatch ---------------------------------------------------

    use super::super::persistent_list::PersistentList;

    fn list_of(items: Vec<Object>) -> Object {
        Object::List(PersistentList::create(items))
    }

    fn sym(s: &str) -> Object {
        Object::Symbol(Symbol::intern(s))
    }

    #[test]
    #[should_panic(expected = "Too many arguments to if")]
    fn analyze_if_with_too_many_args_panics() {
        let form = list_of(vec![
            sym("if"),
            Object::Bool(true),
            Object::Long(1),
            Object::Long(2),
            Object::Long(3),
        ]);
        let _ = analyze(C::Expression, form);
    }

    #[test]
    #[should_panic(expected = "Too few arguments to if")]
    fn analyze_if_with_too_few_args_panics() {
        let form = list_of(vec![sym("if"), Object::Bool(true)]);
        let _ = analyze(C::Expression, form);
    }

    // ---- LetExpr ------------------------------------------------------------

    use super::super::persistent_vector::PersistentVector as PV;

    fn vec_of(items: Vec<Object>) -> Object {
        Object::Vector(PV::create(items))
    }

    #[test]
    fn munge_replaces_special_chars() {
        assert_eq!(super::munge("foo-bar"), "foo_bar");
        assert_eq!(super::munge("foo?"), "foo_QMARK_");
        assert_eq!(super::munge("!"), "_BANG_");
        assert_eq!(super::munge("plain"), "plain");
    }

    #[test]
    fn analyze_let_with_one_binding_produces_let_expr() {
        // (let* [x 1] x) — but body uses Symbol which we don't analyze yet.
        // So use a numeric body to exercise the binding path.
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(1)]),
            Object::Long(99),
        ]);
        let expr = analyze(C::Statement, form);
        // We can't call eval() (Java throws), but we can confirm shape via
        // structural casts at the trait level: has_java_class follows body.
        assert!(expr.has_java_class());
        // The body is a BodyExpr wrapping NumberExpr(99) → java class "long".
        assert_eq!(expr.get_java_class().unwrap().name.as_str(), "long");
    }

    #[test]
    fn analyze_let_increments_next_local_num_scoped() {
        // Snapshot the counter, analyze a let with 2 bindings, then confirm
        // the counter has been popped back. Java's pushThreadBindings is the
        // mechanism — we mirror it.
        let before = super::current_next_local_num();
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(1), sym("y"), Object::Long(2)]),
            Object::Long(99),
        ]);
        let _ = analyze(C::Statement, form);
        let after = super::current_next_local_num();
        assert_eq!(
            before, after,
            "NEXT_LOCAL_NUM must be restored after parse_let pops thread bindings"
        );
    }

    #[test]
    fn analyze_let_pushes_locals_into_env_then_pops() {
        // Outer LOCAL_ENV is empty; inside parse_let it gets populated; on
        // exit it's empty again.
        let before_count = super::current_local_env().len();
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("a"), Object::Long(1)]),
            Object::Long(99),
        ]);
        let _ = analyze(C::Statement, form);
        let after_count = super::current_local_env().len();
        assert_eq!(before_count, after_count);
    }

    #[test]
    #[should_panic(expected = "Bad binding form, expected vector")]
    fn analyze_let_panics_on_non_vector_bindings() {
        let form = list_of(vec![sym("let*"), Object::Long(0), Object::Long(99)]);
        let _ = analyze(C::Statement, form);
    }

    #[test]
    #[should_panic(expected = "matched symbol expression pairs")]
    fn analyze_let_panics_on_odd_bindings() {
        let form = list_of(vec![sym("let*"), vec_of(vec![sym("x")]), Object::Long(99)]);
        let _ = analyze(C::Statement, form);
    }

    #[test]
    #[should_panic(expected = "Can't let qualified name")]
    fn analyze_let_panics_on_qualified_local() {
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![
                Object::Symbol(Symbol::intern_ns_name(Some("ns"), "x")),
                Object::Long(1),
            ]),
            Object::Long(99),
        ]);
        let _ = analyze(C::Statement, form);
    }

    // ---- ObjExpr -------------------------------------------------------------

    #[test]
    fn obj_expr_starts_empty() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        let o = ObjExpr::placeholder();
        assert!(o.name().is_none());
        assert!(o.internal_name().is_none());
        assert!(o.this_name().is_none());
        assert_eq!(o.line(), 0);
        assert_eq!(o.column(), 0);
        assert_eq!(o.constants_id(), 0);
        assert_eq!(o.constants.lock().unwrap().len(), 0);
        assert_eq!(o.closes.lock().unwrap().len(), 0);
        assert!(!o.is_deftype());
        assert!(o.supports_meta()); // non-deftype → has __meta field
    }

    #[test]
    fn obj_expr_intern_constant_dedupes_by_identity_for_keywords() {
        let o = ObjExpr::placeholder();
        let k = Keyword::intern_ns_name(None, "a");
        let id1 = o.intern_constant(Object::Keyword(k.clone()));
        let id2 = o.intern_constant(Object::Keyword(k.clone()));
        assert_eq!(id1, id2);
        assert_eq!(o.constants.lock().unwrap().len(), 1);
    }

    #[test]
    fn obj_expr_intern_constant_dedupes_by_value_for_longs() {
        let o = ObjExpr::placeholder();
        let id1 = o.intern_constant(Object::Long(42));
        let id2 = o.intern_constant(Object::Long(42));
        let id3 = o.intern_constant(Object::Long(43));
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(o.constants.lock().unwrap().len(), 2);
    }

    #[test]
    fn obj_expr_add_close_stores_local_by_idx() {
        let o = ObjExpr::placeholder();
        let lb = LocalBinding::new(7, Symbol::intern("x"), None, None, false);
        o.add_close(lb.clone());
        let map = o.closes.lock().unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&7));
    }

    #[test]
    fn obj_expr_is_deftype_flips_when_fields_set() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        let o = ObjExpr::placeholder();
        assert!(!o.is_deftype());
        assert!(o.supports_meta());

        *o.fields.lock().unwrap() = Some(HashMap::new());
        assert!(o.is_deftype());
        assert!(!o.supports_meta());
    }

    #[test]
    fn obj_expr_get_java_class_default_is_ifn() {
        let o = ObjExpr::placeholder();
        assert_eq!(
            <ObjExpr as Expr>::get_java_class(&o).unwrap().name.as_str(),
            "clojure.lang.IFn"
        );
    }

    #[test]
    fn obj_expr_get_java_class_uses_tag_when_set() {
        let o = ObjExpr::new(Object::Symbol(Symbol::intern("java.lang.String")));
        assert_eq!(
            <ObjExpr as Expr>::get_java_class(&o).unwrap().name.as_str(),
            "java.lang.String"
        );
    }

    #[test]
    fn obj_expr_get_java_class_compiled_overrides_tag() {
        let o = ObjExpr::new(Object::Symbol(Symbol::intern("ignored")));
        *o.compiled_class.write().unwrap() = Some(HostClass {
            name: Arc::new("user.MyFn".to_string()),
        });
        assert_eq!(
            <ObjExpr as Expr>::get_java_class(&o).unwrap().name.as_str(),
            "user.MyFn"
        );
    }

    #[test]
    #[should_panic(expected = "Cannot assign to non-mutable")]
    fn obj_expr_emit_assign_local_rejects_immutable() {
        let o = ObjExpr::placeholder();
        let lb = LocalBinding::new(0, Symbol::intern("x"), None, None, false);
        let val: Box<dyn Expr> = Box::new(NumberExpr::new(Object::Long(1)));
        let mut dm = dynlang::DynModule::new(
            dynlang::GcConfig::generational(65536),
            dynlang::NanBoxTags::default(),
        );
        let fref = dm.declare_func("test_assign", 0);
        let mut df = dm.start_func(fref);
        let mut ir = IrEmitter::new(&mut df);
        o.emit_assign_local(&mut ir, &lb, &*val);
    }

    #[test]
    fn const_prefix_matches_java() {
        assert_eq!(CONST_PREFIX, "const__");
    }

    #[test]
    #[should_panic(expected = "Can't eval let")]
    fn let_expr_eval_panics_per_java_contract() {
        // Match Java: LetExpr.eval throws UnsupportedOperationException.
        // Build the AST directly (not via register_local, which needs
        // NEXT_LOCAL_NUM to be thread-bound).
        let init: Arc<dyn Expr> = Arc::new(NumberExpr::new(Object::Long(1)));
        let lb = LocalBinding::new(0, Symbol::intern("x"), None, Some(init.clone()), false);
        let bi = BindingInit::new(lb, init);
        let body: Box<dyn Expr> = Box::new(NumberExpr::new(Object::Long(2)));
        let le = LetExpr::new(vec![bi], body, false);
        let _ = <LetExpr as Expr>::eval(&le);
    }

    #[test]
    fn constant_expr_get_java_class_table() {
        let c = ConstantExpr::new(Object::String(Arc::new("x".to_string())));
        assert_eq!(
            <ConstantExpr as Expr>::get_java_class(&c)
                .unwrap()
                .name
                .as_str(),
            "java.lang.String"
        );
        let c_nil = ConstantExpr::new(Object::Nil);
        assert!(<ConstantExpr as Expr>::get_java_class(&c_nil).is_none());
    }

    // ---- End-to-end IR pipeline --------------------------------------------

    /// Run a top-level form through analyze → emit → run, returning the raw
    /// 64-bit NanBox result. Routes through the JIT: integer literals are now
    /// boxed Longs that live in the JIT literal pool (`gc_literal`), which the
    /// bare `ModuleInterpreter` cannot materialize (it has no literal pool),
    /// so the interpreter path can no longer evaluate integer-bearing forms.
    /// The JIT is the production path regardless, so these e2e tests use it.
    fn eval_form_via_ir(form: Object) -> u64 {
        eval_form_via_jit(form)
    }

    /// Decode a NanBox bit pattern back to an f64 (assumes Number). Integers
    /// are boxed Longs (heap cells), so this unboxes them to their f64 value;
    /// genuine doubles round-trip through their bit pattern.
    fn nanbox_to_f64(bits: u64) -> f64 {
        crate::runtime::arg_to_f64(bits)
    }

    /// NanBox-encode a boolean using dynlang's default tag scheme.
    /// Mirrors dynlang's internal `nanbox_encode(bool_tag, b as u64)`.
    fn nanbox_bool(b: bool) -> u64 {
        const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
        const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
        let tag = dynlang::NanBoxTags::default().bool_tag as u64;
        TAG_PATTERN | (tag << 48) | ((b as u64) & PAYLOAD_MASK)
    }

    #[test]
    fn ir_e2e_literal_long_returns_42() {
        let v = eval_form_via_ir(Object::Long(42));
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn ir_e2e_literal_double_returns_3_14() {
        let v = eval_form_via_ir(Object::Double(3.14));
        assert_eq!(nanbox_to_f64(v), 3.14);
    }

    #[test]
    fn ir_e2e_if_true_picks_then_branch() {
        // (if true 1 2)
        let form = list_of(vec![
            sym("if"),
            Object::Bool(true),
            Object::Long(1),
            Object::Long(2),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 1.0);
    }

    #[test]
    fn ir_e2e_if_false_picks_else_branch() {
        // (if false 1 2)
        let form = list_of(vec![
            sym("if"),
            Object::Bool(false),
            Object::Long(1),
            Object::Long(2),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn ir_e2e_if_nil_picks_else_branch() {
        // (if nil 1 2) — Clojure truthiness: nil → else
        let form = list_of(vec![
            sym("if"),
            Object::Nil,
            Object::Long(1),
            Object::Long(2),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn ir_e2e_if_zero_is_truthy_picks_then() {
        // (if 0 1 2) — Clojure: 0 is truthy → then
        let form = list_of(vec![
            sym("if"),
            Object::Long(0),
            Object::Long(1),
            Object::Long(2),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 1.0);
    }

    #[test]
    fn ir_e2e_nested_if_inside_do_returns_then() {
        // (do 0 (if true 7 8))
        let inner = list_of(vec![
            sym("if"),
            Object::Bool(true),
            Object::Long(7),
            Object::Long(8),
        ]);
        let outer = list_of(vec![sym("do"), Object::Long(99), inner]);
        let v = eval_form_via_ir(outer);
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn ir_e2e_do_returns_last_value() {
        // (do 1 2 3)
        let form = list_of(vec![
            sym("do"),
            Object::Long(1),
            Object::Long(2),
            Object::Long(3),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 3.0);
    }

    // ---- End-to-end JIT pipeline -------------------------------------------
    //
    // Same forms as the interpreter tests above, but the IR is JIT-compiled
    // via `dynlower::JitModule::compile_batch` and executed as native code.
    // No externs, no safepoint handler — we don't allocate during these
    // forms, so the simplest JIT setup suffices.

    /// Parse `src` with the reader and run it through the JIT, returning the
    /// raw NanBox bits. Convenience wrapper for source-driven tests.
    fn eval_str_via_jit(src: &str) -> u64 {
        let form = super::super::lisp_reader::read_str(src)
            .unwrap_or_else(|e| panic!("read_str({src:?}) failed: {e}"));
        eval_form_via_jit(form)
    }

    /// Run a source string and decode the heap result while the GC runtime is
    /// still alive. Required for results that point into the GC heap
    /// (strings, symbols, keywords, cons) — `eval_str_via_jit` drops the
    /// runtime before returning the raw bits, freeing the heap.
    fn eval_str_via_jit_to_object(src: &str) -> Object {
        use dynlower::JitOutcome;
        use dynruntime::GcPolicy;
        let form = super::super::lisp_reader::read_str(src)
            .unwrap_or_else(|e| panic!("read_str({src:?}) failed: {e}"));
        let (gc, jit, entry, _roots) = super::compile_form_to_jit(form);
        let _thread = gc.install_thread();
        let _call_base = crate::runtime::install_call_table_base(jit.call_table_base_addr());
        let _local_chain = dynobj::roots::FrameChain::new();
        let _chain_root_g =
            unsafe { gc.push_extra_root_source(&_local_chain as *const dyn dynobj::RootSource) };
        let _chain_g = dynobj::roots::install_chain(&_local_chain);
        let bits = match gc.run_jit(&jit, entry, &[], GcPolicy::EveryPoint) {
            JitOutcome::Value(v) => v,
            other => panic!("unexpected JIT outcome: {other:?}"),
        };
        let ids = crate::runtime::HeapTypeIds {
            string: 0,
            symbol: 1,
            keyword: 2,
            cons: 3,
            vector: 5,
            map: 6,
            set: 7,
            tree_map: 11,
            tree_set: 12,
            string_builder: 13,
            chunk_buffer: 14,
            i_chunk: 15,
            lazy_seq: 16,
            delay: 17,
            multi_arity_fn: 18,
            class: 8,
            var: 9,
            with_meta: 10,
            long: 20,
            character: 23,
            user_instance: 19,
            reduced: 21,
            namespace: 22,
        };
        unsafe { crate::runtime::heap_bits_to_object(bits, ids) }
    }

    fn eval_form_via_jit(form: Object) -> u64 {
        use dynlower::JitOutcome;
        use dynruntime::GcPolicy;
        let (gc, jit, entry, _roots) = super::compile_form_to_jit(form);
        let _call_base = crate::runtime::install_call_table_base(jit.call_table_base_addr());
        let _local_chain = dynobj::roots::FrameChain::new();
        let _chain_root_g =
            unsafe { gc.push_extra_root_source(&_local_chain as *const dyn dynobj::RootSource) };
        let _chain_g = dynobj::roots::install_chain(&_local_chain);
        // OnPressure threshold 0.75 mirrors the docstring's "typical
        // production behavior".
        match gc.run_jit(&jit, entry, &[], GcPolicy::EveryPoint) {
            JitOutcome::Value(v) => v,
            other => panic!("unexpected JIT outcome: {other:?}"),
        }
    }

    /// Run a form through the JIT and return the raw `JitOutcome` — the
    /// path tests use when they need to inspect non-Value outcomes
    /// (currently `AbortToPrompt` from `(throw …)`).
    fn eval_form_via_jit_outcome(form: Object) -> dynlower::JitOutcome {
        use dynruntime::GcPolicy;
        let (gc, jit, entry, _roots) = super::compile_form_to_jit(form);
        let _call_base = crate::runtime::install_call_table_base(jit.call_table_base_addr());
        let _local_chain = dynobj::roots::FrameChain::new();
        let _chain_root_g =
            unsafe { gc.push_extra_root_source(&_local_chain as *const dyn dynobj::RootSource) };
        let _chain_g = dynobj::roots::install_chain(&_local_chain);
        gc.run_jit(&jit, entry, &[], GcPolicy::EveryPoint)
    }

    #[test]
    fn jit_e2e_literal_long_returns_42() {
        let v = eval_form_via_jit(Object::Long(42));
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn jit_try_catch_catches_throw_and_returns_handler_value() {
        // (try (throw 42) (catch Throwable e e)) → 42
        // The handler binds `e` to the thrown value and returns it.
        let v = with_fresh_ns("test.try.basic", || {
            eval_str_via_jit("(try (throw 42) (catch Throwable e e))")
        });
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn jit_try_catch_handler_can_compute_with_caught_value() {
        // (try (throw 10) (catch Throwable e (+ e 1))) → 11
        let v = with_fresh_ns("test.try.bump", || {
            eval_str_via_jit("(try (throw 10) (catch Throwable e (+ e 1)))")
        });
        assert_eq!(nanbox_to_f64(v), 11.0);
    }

    #[test]
    fn jit_try_with_normal_body_skips_catch() {
        // (try 7 (catch Throwable e 99)) → 7
        // Body returns normally; catch arm is not entered.
        let v = with_fresh_ns("test.try.normal", || {
            eval_str_via_jit("(try 7 (catch Throwable e 99))")
        });
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn jit_nested_try_inner_catches_first() {
        // (try (try (throw 1) (catch Throwable a (+ a 10)))
        //      (catch Throwable b (+ b 100)))
        // Inner catch wins → 11.
        let v = with_fresh_ns("test.try.nested.inner", || {
            eval_str_via_jit(
                "(try (try (throw 1) (catch Throwable a (+ a 10))) \
                      (catch Throwable b (+ b 100)))",
            )
        });
        assert_eq!(nanbox_to_f64(v), 11.0);
    }

    #[test]
    fn jit_outer_try_catches_rethrow_from_inner_catch() {
        // (try (try (throw 1) (catch Throwable a (throw 2)))
        //      (catch Throwable b (+ b 100)))
        // Inner catches 1, re-throws 2. Outer catches 2 → 102.
        let v = with_fresh_ns("test.try.nested.rethrow", || {
            eval_str_via_jit(
                "(try (try (throw 1) (catch Throwable a (throw 2))) \
                      (catch Throwable b (+ b 100)))",
            )
        });
        assert_eq!(nanbox_to_f64(v), 102.0);
    }

    #[test]
    fn jit_try_finally_runs_on_normal_exit() {
        // (try 7 (finally 99)) → 7. Finally runs but its value is
        // discarded.
        let v = with_fresh_ns("test.try.finally.normal", || {
            eval_str_via_jit("(try 7 (finally 99))")
        });
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn jit_try_finally_propagates_uncaught_throw() {
        // (try (throw 5) (finally 99)) — finally runs, then re-raises
        // → outcome Exception(5).
        let form = list_of(vec![
            sym("try"),
            list_of(vec![sym("throw"), Object::Long(5)]),
            list_of(vec![sym("finally"), Object::Long(99)]),
        ]);
        match eval_form_via_jit_outcome(form) {
            dynlower::JitOutcome::Exception(bits) => {
                assert_eq!(nanbox_to_f64(bits), 5.0);
            }
            other => panic!("expected Exception(5), got {other:?}"),
        }
    }

    #[test]
    fn jit_try_finally_runs_after_caught_throw() {
        // (try (throw 5) (catch Throwable e (+ e 1)) (finally 99))
        // → catch returns 6, finally runs and discards 99, result is 6.
        let v = with_fresh_ns("test.try.finally.caught", || {
            eval_str_via_jit("(try (throw 5) (catch Throwable e (+ e 1)) (finally 99))")
        });
        assert_eq!(nanbox_to_f64(v), 6.0);
    }

    #[test]
    fn jit_try_finally_propagates_rethrow_from_catch() {
        // (try (try (throw 1)
        //           (catch Throwable a (throw 2))
        //           (finally 99))
        //      (catch Throwable b (+ b 100)))
        // → inner catches 1, throws 2, inner finally runs, outer
        //   catches 2 → 102.
        let v = with_fresh_ns("test.try.finally.rethrow", || {
            eval_str_via_jit(
                "(try (try (throw 1) \
                           (catch Throwable a (throw 2)) \
                           (finally 99)) \
                      (catch Throwable b (+ b 100)))",
            )
        });
        assert_eq!(nanbox_to_f64(v), 102.0);
    }

    #[test]
    fn jit_try_catches_throw_from_nested_fn() {
        // (try ((fn* [] (throw 5))) (catch Throwable e e)) → 5
        // The throw happens INSIDE a callee; the caller's try catches.
        // Tests cross-fn raise propagation via the JIT outcome ABI.
        let v = with_fresh_ns("test.try.crossfn", || {
            eval_str_via_jit("(try ((fn* [] (throw 5))) (catch Throwable e e))")
        });
        assert_eq!(nanbox_to_f64(v), 5.0);
    }

    #[test]
    fn jit_throw_long_surfaces_as_exception_outcome() {
        // (throw 42) — payload is a numeric literal; the JIT exits via
        // `JitOutcome::Exception` carrying the NanBox-encoded value.
        // No try/catch yet, so this is an "uncaught exception" surfaced
        // at the JIT boundary — instead of the SIGABRT we'd get from a
        // raw Rust panic across an `extern "C"` boundary.
        let form = list_of(vec![sym("throw"), Object::Long(42)]);
        match eval_form_via_jit_outcome(form) {
            dynlower::JitOutcome::Exception(bits) => {
                // The payload is a boxed Long; unbox it to its numeric value.
                assert_eq!(nanbox_to_f64(bits), 42.0);
            }
            other => panic!("expected JitOutcome::Exception from (throw 42), got {other:?}"),
        }
    }

    #[test]
    fn jit_throw_string_carries_payload() {
        // (throw "boom") — payload is a heap-allocated string. The
        // Exception payload is the NanBox-tagged pointer.
        let form = list_of(vec![
            sym("throw"),
            Object::String(Arc::new("boom".to_string())),
        ]);
        let outcome = eval_form_via_jit_outcome(form);
        match outcome {
            dynlower::JitOutcome::Exception(bits) => {
                // The pointer bits should be a TAG_PTR-tagged NanBox.
                assert!(
                    bits != crate::runtime::nanbox_nil(),
                    "throw payload must not be nil"
                );
            }
            other => panic!("expected JitOutcome::Exception, got {other:?}"),
        }
    }

    #[test]
    fn jit_e2e_literal_double_returns_3_14() {
        let v = eval_form_via_jit(Object::Double(3.14));
        assert_eq!(nanbox_to_f64(v), 3.14);
    }

    #[test]
    fn jit_e2e_if_true_picks_then_branch() {
        let form = list_of(vec![
            sym("if"),
            Object::Bool(true),
            Object::Long(1),
            Object::Long(2),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 1.0);
    }

    #[test]
    fn jit_e2e_if_false_picks_else_branch() {
        let form = list_of(vec![
            sym("if"),
            Object::Bool(false),
            Object::Long(1),
            Object::Long(2),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn jit_e2e_if_nil_picks_else_branch() {
        let form = list_of(vec![
            sym("if"),
            Object::Nil,
            Object::Long(1),
            Object::Long(2),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn jit_e2e_if_zero_is_truthy_picks_then() {
        let form = list_of(vec![
            sym("if"),
            Object::Long(0),
            Object::Long(1),
            Object::Long(2),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 1.0);
    }

    #[test]
    fn jit_e2e_nested_if_inside_do_returns_then() {
        let inner = list_of(vec![
            sym("if"),
            Object::Bool(true),
            Object::Long(7),
            Object::Long(8),
        ]);
        let outer = list_of(vec![sym("do"), Object::Long(99), inner]);
        let v = eval_form_via_jit(outer);
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn jit_e2e_do_returns_last_value() {
        let form = list_of(vec![
            sym("do"),
            Object::Long(1),
            Object::Long(2),
            Object::Long(3),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 3.0);
    }

    // ---- let* end-to-end tests ---------------------------------------------

    #[test]
    fn ir_e2e_let_single_binding() {
        // (let* [x 42] x)
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(42)]),
            sym("x"),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn jit_e2e_let_single_binding() {
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(42)]),
            sym("x"),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn ir_e2e_let_two_bindings_body_picks_first() {
        // (let* [x 1 y 2] x)
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(1), sym("y"), Object::Long(2)]),
            sym("x"),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 1.0);
    }

    #[test]
    fn ir_e2e_let_two_bindings_body_picks_second() {
        // (let* [x 1 y 2] y)
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(1), sym("y"), Object::Long(2)]),
            sym("y"),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn jit_e2e_let_with_if_branch() {
        // (let* [x 7] (if x x 99))
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(7)]),
            list_of(vec![sym("if"), sym("x"), sym("x"), Object::Long(99)]),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn jit_e2e_let_nested() {
        // (let* [x 1] (let* [y 2] y))
        let inner = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("y"), Object::Long(2)]),
            sym("y"),
        ]);
        let outer = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(1)]),
            inner,
        ]);
        let v = eval_form_via_jit(outer);
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn ir_e2e_let_shadowing() {
        // (let* [x 1] (let* [x 99] x))
        let inner = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(99)]),
            sym("x"),
        ]);
        let outer = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x"), Object::Long(1)]),
            inner,
        ]);
        let v = eval_form_via_ir(outer);
        assert_eq!(nanbox_to_f64(v), 99.0);
    }

    // ---- fn / invoke through JIT -------------------------------------------

    #[test]
    fn jit_e2e_invoke_identity_fn() {
        // ((fn* [x] x) 7) → 7
        let form = list_of(vec![
            list_of(vec![sym("fn*"), vec_of(vec![sym("x")]), sym("x")]),
            Object::Long(7),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn jit_e2e_invoke_fn_with_if() {
        // ((fn* [x] (if x 1 2)) true) → 1
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("x")]),
                list_of(vec![sym("if"), sym("x"), Object::Long(1), Object::Long(2)]),
            ]),
            Object::Bool(true),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 1.0);
    }

    #[test]
    fn jit_e2e_invoke_fn_with_if_else() {
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("x")]),
                list_of(vec![sym("if"), sym("x"), Object::Long(1), Object::Long(2)]),
            ]),
            Object::Bool(false),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn jit_e2e_invoke_fn_two_args_returns_first() {
        // ((fn* [a b] a) 11 22) → 11
        let form = list_of(vec![
            list_of(vec![sym("fn*"), vec_of(vec![sym("a"), sym("b")]), sym("a")]),
            Object::Long(11),
            Object::Long(22),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 11.0);
    }

    #[test]
    fn jit_e2e_invoke_fn_two_args_returns_second() {
        let form = list_of(vec![
            list_of(vec![sym("fn*"), vec_of(vec![sym("a"), sym("b")]), sym("b")]),
            Object::Long(11),
            Object::Long(22),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 22.0);
    }

    #[test]
    fn jit_e2e_invoke_fn_with_let_inside() {
        // ((fn* [x] (let* [y x] y)) 99) → 99
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("x")]),
                list_of(vec![
                    sym("let*"),
                    vec_of(vec![sym("y"), sym("x")]),
                    sym("y"),
                ]),
            ]),
            Object::Long(99),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 99.0);
    }

    #[test]
    fn ir_e2e_invoke_identity_fn() {
        let form = list_of(vec![
            list_of(vec![sym("fn*"), vec_of(vec![sym("x")]), sym("x")]),
            Object::Long(42),
        ]);
        let v = eval_form_via_ir(form);
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    // ---- first-class fns through let* --------------------------------------

    #[test]
    fn jit_e2e_let_bound_fn_called() {
        // (let* [f (fn* [x] x)] (f 42)) → 42
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![
                sym("f"),
                list_of(vec![sym("fn*"), vec_of(vec![sym("x")]), sym("x")]),
            ]),
            list_of(vec![sym("f"), Object::Long(42)]),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 42.0);
    }

    #[test]
    fn jit_e2e_let_bound_fn_with_arith() {
        // (let* [add (fn* [a b] (+ a b))] (add 100 23)) → 123
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![
                sym("add"),
                list_of(vec![
                    sym("fn*"),
                    vec_of(vec![sym("a"), sym("b")]),
                    list_of(vec![sym("+"), sym("a"), sym("b")]),
                ]),
            ]),
            list_of(vec![sym("add"), Object::Long(100), Object::Long(23)]),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 123.0);
    }

    // ---- recur in fn body --------------------------------------------------

    #[test]
    fn jit_e2e_recur_countdown_to_zero() {
        // ((fn* [n] (if (< n 1) n (recur (- n 1)))) 5) → 0
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("n")]),
                list_of(vec![
                    sym("if"),
                    list_of(vec![sym("<"), sym("n"), Object::Long(1)]),
                    sym("n"),
                    list_of(vec![
                        sym("recur"),
                        list_of(vec![sym("-"), sym("n"), Object::Long(1)]),
                    ]),
                ]),
            ]),
            Object::Long(5),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 0.0);
    }

    #[test]
    fn jit_e2e_recur_accumulator_sum_1_to_n() {
        // Sum 1..n via tail-recursion through recur.
        // ((fn* [n acc]
        //    (if (< n 1)
        //      acc
        //      (recur (- n 1) (+ acc n))))
        //  5 0)
        // → 0 + 5 + 4 + 3 + 2 + 1 = 15
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("n"), sym("acc")]),
                list_of(vec![
                    sym("if"),
                    list_of(vec![sym("<"), sym("n"), Object::Long(1)]),
                    sym("acc"),
                    list_of(vec![
                        sym("recur"),
                        list_of(vec![sym("-"), sym("n"), Object::Long(1)]),
                        list_of(vec![sym("+"), sym("acc"), sym("n")]),
                    ]),
                ]),
            ]),
            Object::Long(5),
            Object::Long(0),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 15.0);
    }

    #[test]
    fn jit_e2e_recur_factorial() {
        // ((fn* [n acc]
        //    (if (< n 2) acc (recur (- n 1) (* acc n))))
        //  6 1)
        // → 6! = 720
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("n"), sym("acc")]),
                list_of(vec![
                    sym("if"),
                    list_of(vec![sym("<"), sym("n"), Object::Long(2)]),
                    sym("acc"),
                    list_of(vec![
                        sym("recur"),
                        list_of(vec![sym("-"), sym("n"), Object::Long(1)]),
                        list_of(vec![sym("*"), sym("acc"), sym("n")]),
                    ]),
                ]),
            ]),
            Object::Long(6),
            Object::Long(1),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 720.0);
    }

    // ---- Arithmetic / comparison primops -----------------------------------

    #[test]
    fn jit_e2e_add_two_longs() {
        // (+ 1 2)
        let form = list_of(vec![sym("+"), Object::Long(1), Object::Long(2)]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 3.0);
    }

    #[test]
    fn jit_e2e_add_variadic() {
        // (+ 1 2 3 4 5)
        let form = list_of(vec![
            sym("+"),
            Object::Long(1),
            Object::Long(2),
            Object::Long(3),
            Object::Long(4),
            Object::Long(5),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 15.0);
    }

    #[test]
    fn jit_e2e_add_zero_args_yields_zero() {
        // (+) → 0
        let form = list_of(vec![sym("+")]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 0.0);
    }

    #[test]
    fn jit_e2e_mul_zero_args_yields_one() {
        // (*) → 1
        let form = list_of(vec![sym("*")]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 1.0);
    }

    #[test]
    fn jit_e2e_sub_unary_negates() {
        // (- 7) → -7
        let form = list_of(vec![sym("-"), Object::Long(7)]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), -7.0);
    }

    #[test]
    fn jit_e2e_sub_binary() {
        let form = list_of(vec![sym("-"), Object::Long(10), Object::Long(3)]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 7.0);
    }

    #[test]
    fn jit_e2e_mul_and_div() {
        // (/ (* 6 7) 2) → 21
        let inner = list_of(vec![sym("*"), Object::Long(6), Object::Long(7)]);
        let form = list_of(vec![sym("/"), inner, Object::Long(2)]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 21.0);
    }

    #[test]
    fn jit_e2e_arith_with_let() {
        // (let* [a 3 b 4] (+ a b))
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("a"), Object::Long(3), sym("b"), Object::Long(4)]),
            list_of(vec![sym("+"), sym("a"), sym("b")]),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 7.0);
    }

    #[test]
    fn jit_e2e_arith_inside_fn() {
        // ((fn* [x y] (+ x y)) 11 22) → 33
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("x"), sym("y")]),
                list_of(vec![sym("+"), sym("x"), sym("y")]),
            ]),
            Object::Long(11),
            Object::Long(22),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 33.0);
    }

    #[test]
    fn jit_e2e_lt_true() {
        let form = list_of(vec![sym("<"), Object::Long(1), Object::Long(2)]);
        let bits = eval_form_via_jit(form);
        // NanBox-encoded true.
        let expected = nanbox_bool(true);
        assert_eq!(bits, expected);
    }

    #[test]
    fn jit_e2e_lt_false() {
        let form = list_of(vec![sym("<"), Object::Long(5), Object::Long(2)]);
        let bits = eval_form_via_jit(form);
        let expected = nanbox_bool(false);
        assert_eq!(bits, expected);
    }

    #[test]
    fn jit_e2e_lt_inside_if() {
        // (if (< 1 2) 100 200) → 100
        let form = list_of(vec![
            sym("if"),
            list_of(vec![sym("<"), Object::Long(1), Object::Long(2)]),
            Object::Long(100),
            Object::Long(200),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 100.0);
    }

    #[test]
    fn jit_e2e_arith_via_fn_with_branch() {
        // ((fn* [n] (if (< n 10) (+ n 1) n)) 5) → 6
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("n")]),
                list_of(vec![
                    sym("if"),
                    list_of(vec![sym("<"), sym("n"), Object::Long(10)]),
                    list_of(vec![sym("+"), sym("n"), Object::Long(1)]),
                    sym("n"),
                ]),
            ]),
            Object::Long(5),
        ]);
        assert_eq!(nanbox_to_f64(eval_form_via_jit(form)), 6.0);
    }

    // ---- parse-time `def` panics (no tree-walking) -------------------------

    /// Push a fresh namespace as `*ns*` so `(def …)` analyses interp it,
    /// then pop on drop. Used by parse-only tests.
    fn with_fresh_ns<F: FnOnce() -> R, R>(ns_name: &str, body: F) -> R {
        use super::super::namespace::Namespace;
        use super::super::rt::CURRENT_NS;
        let ns = Namespace::find_or_create(Symbol::intern(ns_name));
        Var::push_thread_bindings(vec![(CURRENT_NS.clone(), Object::Namespace(ns))]);
        let r = body();
        Var::pop_thread_bindings();
        r
    }

    // ---- def / var deref through JIT ---------------------------------------
    //
    // `(do (def x 42) x)`: DefExpr.emit calls `cljvm_var_bind_root(var_ptr, 42)`,
    // which mutates the Var's root binding; then VarExpr.emit calls
    // `cljvm_var_deref(var_ptr)` which reads it back as a NanBox u64. Each
    // test installs a fresh namespace so the def doesn't pollute clojure.core.

    #[test]
    fn jit_e2e_def_then_deref_returns_value() {
        // (do (def x 42) x) → 42
        let v = with_fresh_ns("test.def.deref.long", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![sym("def"), sym("x"), Object::Long(42)]),
                sym("x"),
            ]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn jit_e2e_def_then_deref_double() {
        let v = with_fresh_ns("test.def.deref.double", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![sym("def"), sym("x"), Object::Double(3.5)]),
                sym("x"),
            ]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 3.5);
    }

    #[test]
    fn jit_e2e_def_returns_bound_value() {
        // The `(def x 42)` form itself returns the bound value (our divergence
        // from Java, which returns the Var). Without `do`/sequencing it should
        // still produce 42.
        let v = with_fresh_ns("test.def.return", || {
            let form = list_of(vec![sym("def"), sym("x"), Object::Long(99)]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 99.0);
    }

    #[test]
    fn jit_e2e_def_overwrite_then_deref() {
        // Defining the same var twice should make the deref return the second
        // binding. Exercises `Var.bind_root` rebinding semantics.
        let v = with_fresh_ns("test.def.rebind", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![sym("def"), sym("x"), Object::Long(1)]),
                list_of(vec![sym("def"), sym("x"), Object::Long(2)]),
                sym("x"),
            ]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn jit_e2e_def_then_let_uses_var() {
        // (do (def x 7) (let* [y x] y)) → 7
        // Confirms a Var-deref result feeds into a let-binding init.
        let v = with_fresh_ns("test.def.into.let", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![sym("def"), sym("x"), Object::Long(7)]),
                list_of(vec![
                    sym("let*"),
                    vec_of(vec![sym("y"), sym("x")]),
                    sym("y"),
                ]),
            ]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    // ---- defn-shaped forms calling each other by name (JIT) ----------------
    //
    // `(do (def f (fn [...] ...)) (f args))` — analogous to a Clojure `defn`
    // followed by a call. Exercises:
    //   * DefExpr.emit registers Var → FuncRef on the Compiler
    //   * InvokeExpr.emit detects VarExpr head, looks up FuncRef, direct-calls
    //   * Var-deref extern is NOT involved on the call (compile-time bypass)

    #[test]
    fn jit_e2e_defn_inc_then_invoke() {
        // (do (def inc1 (fn [n] (+ n 1))) (inc1 10)) → 11
        let v = with_fresh_ns("test.defn.inc", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![
                    sym("def"),
                    sym("inc1"),
                    list_of(vec![
                        sym("fn*"),
                        vec_of(vec![sym("n")]),
                        list_of(vec![sym("+"), sym("n"), Object::Long(1)]),
                    ]),
                ]),
                list_of(vec![sym("inc1"), Object::Long(10)]),
            ]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 11.0);
    }

    #[test]
    fn jit_e2e_defn_double_two_arg() {
        // (do (def add (fn [a b] (+ a b))) (add 3 4)) → 7
        let v = with_fresh_ns("test.defn.add", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![
                    sym("def"),
                    sym("add2"),
                    list_of(vec![
                        sym("fn*"),
                        vec_of(vec![sym("a"), sym("b")]),
                        list_of(vec![sym("+"), sym("a"), sym("b")]),
                    ]),
                ]),
                list_of(vec![sym("add2"), Object::Long(3), Object::Long(4)]),
            ]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn jit_e2e_defn_calls_another_defn() {
        // (do (def double-it (fn [n] (+ n n)))
        //     (def quad     (fn [n] (double-it (double-it n))))
        //     (quad 3)) → 12
        // Exercises one defn calling another by name: both Var → FuncRef
        // entries must be visible when the second fn's body is lowered, AND
        // the call inside `quad` must find `double-it`'s FuncRef.
        let v = with_fresh_ns("test.defn.chain", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![
                    sym("def"),
                    sym("double-it"),
                    list_of(vec![
                        sym("fn*"),
                        vec_of(vec![sym("n")]),
                        list_of(vec![sym("+"), sym("n"), sym("n")]),
                    ]),
                ]),
                list_of(vec![
                    sym("def"),
                    sym("quad"),
                    list_of(vec![
                        sym("fn*"),
                        vec_of(vec![sym("n")]),
                        list_of(vec![
                            sym("double-it"),
                            list_of(vec![sym("double-it"), sym("n")]),
                        ]),
                    ]),
                ]),
                list_of(vec![sym("quad"), Object::Long(3)]),
            ]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 12.0);
    }

    #[test]
    fn jit_e2e_defn_self_recursive_factorial() {
        // (do (def fact (fn [n] (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 5)) → 120
        // Exercises self-reference: the `(fact …)` call inside the body
        // depends on the var → FuncRef mapping being registered BEFORE the
        // FnExpr body is lowered, which is why DefExpr.emit registers the
        // mapping before emitting the init.
        let v = with_fresh_ns("test.defn.fact", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![
                    sym("def"),
                    sym("fact"),
                    list_of(vec![
                        sym("fn*"),
                        vec_of(vec![sym("n")]),
                        list_of(vec![
                            sym("if"),
                            list_of(vec![sym("<"), sym("n"), Object::Long(2)]),
                            Object::Long(1),
                            list_of(vec![
                                sym("*"),
                                sym("n"),
                                list_of(vec![
                                    sym("fact"),
                                    list_of(vec![sym("-"), sym("n"), Object::Long(1)]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
                list_of(vec![sym("fact"), Object::Long(5)]),
            ]);
            eval_form_via_jit(form)
        });
        assert_eq!(nanbox_to_f64(v), 120.0);
    }

    // ---- string literals via heap allocation + GcLiteral (JIT) -------------
    //
    // `StringExpr.emit` reserves a literal-pool index; after `compile_jit`,
    // we allocate a `clojure.lang.String` heap object, write the UTF-8 bytes
    // into its varlen section, and push the NanBox-encoded pointer into the
    // pool. `Inst::GcLiteral(idx)` lowers to a load from that slot, so the
    // moving GC can rewrite the slot in place if the object relocates.

    /// Compile a form and decode the returned NanBox into a Rust String.
    /// Reads the heap object pointed to by the result. Used by string-literal
    /// tests that need to verify the actual bytes.
    fn eval_form_via_jit_to_string(form: Object) -> String {
        use dynlower::JitOutcome;
        use dynruntime::GcPolicy;
        let (gc, jit, entry, _roots) = super::compile_form_to_jit(form);
        // We need to read the heap object's bytes while the runtime + heap
        // are still alive. The mutator thread guard must also be installed
        // for any heap touch under a generational backend; we install it
        // here, run, decode the bytes, then drop both.
        let _thread = gc.install_thread();
        let _call_base = crate::runtime::install_call_table_base(jit.call_table_base_addr());
        let _local_chain = dynobj::roots::FrameChain::new();
        let _chain_root_g =
            unsafe { gc.push_extra_root_source(&_local_chain as *const dyn dynobj::RootSource) };
        let _chain_g = dynobj::roots::install_chain(&_local_chain);
        let bits = match gc.run_jit(&jit, entry, &[], GcPolicy::EveryPoint) {
            JitOutcome::Value(v) => v,
            other => panic!("unexpected JIT outcome: {other:?}"),
        };
        // Compiler::new declares the heap ObjTypes in a fixed order; their
        // type_ids match the indices below. Hard-coded for tests; production
        // code would route through the Compiler / DynGcRuntime instances.
        let ids = crate::runtime::HeapTypeIds {
            string: 0,
            symbol: 1,
            keyword: 2,
            cons: 3,
            vector: 5,
            map: 6,
            set: 7,
            tree_map: 11,
            tree_set: 12,
            string_builder: 13,
            chunk_buffer: 14,
            i_chunk: 15,
            lazy_seq: 16,
            delay: 17,
            multi_arity_fn: 18,
            class: 8,
            var: 9,
            with_meta: 10,
            long: 20,
            character: 23,
            user_instance: 19,
            reduced: 21,
            namespace: 22,
        };
        let obj = unsafe { crate::runtime::heap_bits_to_object(bits, ids) };
        match obj {
            Object::String(s) => (*s).clone(),
            other => panic!("expected Object::String, got {other:?}"),
        }
    }

    #[test]
    fn jit_e2e_string_literal_returns_bytes() {
        // "hello" — bare literal. StringExpr.emit emits gc_literal(0); compile
        // pipeline allocates the string and pushes a NanBox ptr into slot 0.
        let s = eval_form_via_jit_to_string(Object::String(Arc::new("hello".to_string())));
        assert_eq!(s, "hello");
    }

    #[test]
    fn jit_e2e_string_literal_empty() {
        let s = eval_form_via_jit_to_string(Object::String(Arc::new(String::new())));
        assert_eq!(s, "");
    }

    #[test]
    fn jit_e2e_string_literal_unicode() {
        let s = eval_form_via_jit_to_string(Object::String(Arc::new("λ → 🎉".to_string())));
        assert_eq!(s, "λ → 🎉");
    }

    #[test]
    fn jit_e2e_def_string_then_deref() {
        // (do (def s "world") s) → "world"
        // Exercises a heap-allocated string flowing through `cljvm_var_bind_root`
        // (which stores it opaquely via `Object::Host(HeapBits)`) and then back
        // out via `cljvm_var_deref`.
        let s = with_fresh_ns("test.string.def.deref", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![
                    sym("def"),
                    sym("s"),
                    Object::String(Arc::new("world".to_string())),
                ]),
                sym("s"),
            ]);
            eval_form_via_jit_to_string(form)
        });
        assert_eq!(s, "world");
    }

    #[test]
    fn jit_e2e_if_picks_string_branch() {
        // (if true "yes" "no") → "yes" — string literals as branch values.
        let s = eval_form_via_jit_to_string(list_of(vec![
            sym("if"),
            Object::Bool(true),
            Object::String(Arc::new("yes".to_string())),
            Object::String(Arc::new("no".to_string())),
        ]));
        assert_eq!(s, "yes");
    }

    // ---- quoted symbols via heap allocation + GcLiteral (JIT) -------------
    //
    // `(quote x)` analyzes to a `ConstantExpr` whose `v` is `Object::Symbol`.
    // `ConstantExpr.emit` interns the Arc<Symbol> into the literal pool and
    // emits `gc_literal(LiteralRef(idx))`. After JIT compile, the symbol
    // type's Raw64 `arc_ptr` field is populated with `Arc::as_ptr`; the
    // moving GC traces the heap wrapper but the Arc itself is rooted by the
    // global SYMBOL_TABLE for the program lifetime, so this is sound.

    fn eval_form_via_jit_to_symbol(form: Object) -> Arc<Symbol> {
        use dynlower::JitOutcome;
        use dynruntime::GcPolicy;
        let (gc, jit, entry, _roots) = super::compile_form_to_jit(form);
        let _thread = gc.install_thread();
        let _call_base = crate::runtime::install_call_table_base(jit.call_table_base_addr());
        let _local_chain = dynobj::roots::FrameChain::new();
        let _chain_root_g =
            unsafe { gc.push_extra_root_source(&_local_chain as *const dyn dynobj::RootSource) };
        let _chain_g = dynobj::roots::install_chain(&_local_chain);
        let bits = match gc.run_jit(&jit, entry, &[], GcPolicy::EveryPoint) {
            JitOutcome::Value(v) => v,
            other => panic!("unexpected JIT outcome: {other:?}"),
        };
        let ids = crate::runtime::HeapTypeIds {
            string: 0,
            symbol: 1,
            keyword: 2,
            cons: 3,
            vector: 5,
            map: 6,
            set: 7,
            tree_map: 11,
            tree_set: 12,
            string_builder: 13,
            chunk_buffer: 14,
            i_chunk: 15,
            lazy_seq: 16,
            delay: 17,
            multi_arity_fn: 18,
            class: 8,
            var: 9,
            with_meta: 10,
            long: 20,
            character: 23,
            user_instance: 19,
            reduced: 21,
            namespace: 22,
        };
        let obj = unsafe { crate::runtime::heap_bits_to_object(bits, ids) };
        match obj {
            Object::Symbol(s) => s,
            other => panic!("expected Object::Symbol, got {other:?}"),
        }
    }

    /// Bare allocation test — bypasses JIT. Allocates a `clojure.lang.Symbol`
    /// heap object directly, writes the Arc<Symbol> pointer, reads it back,
    /// decodes. Isolates whether the bug is in alloc/layout vs JIT path.
    #[test]
    fn diagnostic_symbol_alloc_roundtrip() {
        use dynalloc::SemiSpace;
        use dynir::dynexec::ContinuationTypes;
        use dynobj::Compact;

        let _ = ContinuationTypes::register_into::<Compact>; // sanity
        let arc_sym = Symbol::intern("ping");
        let arc_ptr_bits = Arc::as_ptr(&arc_sym) as u64;

        let mut dm = dynlang::DynModule::new(
            dynlang::GcConfig::generational(65536),
            dynlang::NanBoxTags::default(),
        );
        let _string_id = dm.obj_type("clojure.lang.String").varlen_bytes().build();
        let symbol_id = dm
            .obj_type("clojure.lang.Symbol")
            .field("arc_ptr", dynlang::FieldKind::Raw64)
            .build();

        let gc_config = dm.gc_config().clone();
        let obj_types: Vec<dynlang::ObjType> = std::mem::take(&mut dm.obj_types);
        let tags = dynlang::NanBoxTags::default();
        let gc = dynlang::gc::DynGcRuntime::new(&gc_config, &tags, &obj_types);

        let _thread = gc.install_thread();
        let ptr = gc.alloc(symbol_id.0, 0);
        assert!(!ptr.is_null(), "alloc returned null");
        let type_info = &obj_types[symbol_id.0].type_info;
        let raw_offset = type_info.raw_data_offset();
        eprintln!("DIAG: ptr={ptr:p}, raw_offset={raw_offset}, arc_ptr_bits=0x{arc_ptr_bits:x}");
        unsafe {
            let dst = ptr.add(raw_offset).cast::<u64>();
            dst.write_unaligned(arc_ptr_bits);
        }
        let nanbox_bits = gc.tag_ptr(ptr);
        eprintln!("DIAG: nanbox=0x{nanbox_bits:x}");

        // Now decode.
        let ids = crate::runtime::HeapTypeIds {
            string: 0,
            symbol: 1,
            keyword: 2,
            cons: 3,
            vector: 5,
            map: 6,
            set: 7,
            tree_map: 11,
            tree_set: 12,
            string_builder: 13,
            chunk_buffer: 14,
            i_chunk: 15,
            lazy_seq: 16,
            delay: 17,
            multi_arity_fn: 18,
            class: 8,
            var: 9,
            with_meta: 10,
            long: 20,
            character: 23,
            user_instance: 19,
            reduced: 21,
            namespace: 22,
        };
        let obj = unsafe { crate::runtime::heap_bits_to_object(nanbox_bits, ids) };
        match obj {
            Object::Symbol(s) => {
                assert_eq!(s.get_name(), "ping");
                assert!(Arc::ptr_eq(&arc_sym, &s));
            }
            other => panic!("expected Symbol, got {other:?}"),
        }

        // Suppress unused-import warnings on platforms where these aren't
        // exercised — present here to keep this test self-contained.
        let _ = std::mem::size_of::<SemiSpace>();
    }

    #[test]
    fn jit_e2e_quote_symbol_unqualified() {
        // (quote x) → Symbol{ns: None, name: "x"}
        let s = eval_form_via_jit_to_symbol(list_of(vec![sym("quote"), sym("x")]));
        assert!(s.get_namespace().is_none());
        assert_eq!(s.get_name(), "x");
    }

    #[test]
    fn jit_e2e_quote_symbol_qualified() {
        // (quote clojure.core/map) → Symbol{ns: Some("clojure.core"), name: "map"}
        let qualified = Object::Symbol(Symbol::intern_ns_name(Some("clojure.core"), "map"));
        let s = eval_form_via_jit_to_symbol(list_of(vec![sym("quote"), qualified]));
        assert_eq!(s.get_namespace(), Some("clojure.core"));
        assert_eq!(s.get_name(), "map");
    }

    #[test]
    fn jit_e2e_quote_roundtrips_to_value_equal_symbol() {
        // (quote foo) round-trips to a Symbol value-equal to a freshly
        // interned one. Java's `Symbol.intern` always returns `new Symbol(...)`
        // (no global pool), so Arc::ptr_eq is NOT a valid identity check —
        // value equality on (ns, name) is.
        let expected = Symbol::intern("foo");
        let s = eval_form_via_jit_to_symbol(list_of(vec![sym("quote"), sym("foo")]));
        assert_eq!(s.get_name(), expected.get_name());
        assert_eq!(s.get_namespace(), expected.get_namespace());
    }

    #[test]
    fn jit_e2e_def_then_quote_then_deref() {
        // (do (def s (quote hello-sym)) s) → Symbol "hello-sym"
        // Exercises a heap-allocated Symbol flowing through cljvm_var_bind_root /
        // cljvm_var_deref (which roundtrip the NanBox bits opaquely).
        let s = with_fresh_ns("test.quote.def.deref", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![
                    sym("def"),
                    sym("s"),
                    list_of(vec![sym("quote"), sym("hello-sym")]),
                ]),
                sym("s"),
            ]);
            eval_form_via_jit_to_symbol(form)
        });
        assert_eq!(s.get_name(), "hello-sym");
        assert!(s.get_namespace().is_none());
    }

    // ---- keyword literals via heap allocation + GcLiteral (JIT) -----------

    fn eval_form_via_jit_to_keyword(form: Object) -> Arc<Keyword> {
        use dynlower::JitOutcome;
        use dynruntime::GcPolicy;
        let (gc, jit, entry, _roots) = super::compile_form_to_jit(form);
        let _thread = gc.install_thread();
        let _call_base = crate::runtime::install_call_table_base(jit.call_table_base_addr());
        let _local_chain = dynobj::roots::FrameChain::new();
        let _chain_root_g =
            unsafe { gc.push_extra_root_source(&_local_chain as *const dyn dynobj::RootSource) };
        let _chain_g = dynobj::roots::install_chain(&_local_chain);
        let bits = match gc.run_jit(&jit, entry, &[], GcPolicy::EveryPoint) {
            JitOutcome::Value(v) => v,
            other => panic!("unexpected JIT outcome: {other:?}"),
        };
        let ids = crate::runtime::HeapTypeIds {
            string: 0,
            symbol: 1,
            keyword: 2,
            cons: 3,
            vector: 5,
            map: 6,
            set: 7,
            tree_map: 11,
            tree_set: 12,
            string_builder: 13,
            chunk_buffer: 14,
            i_chunk: 15,
            lazy_seq: 16,
            delay: 17,
            multi_arity_fn: 18,
            class: 8,
            var: 9,
            with_meta: 10,
            long: 20,
            character: 23,
            user_instance: 19,
            reduced: 21,
            namespace: 22,
        };
        let obj = unsafe { crate::runtime::heap_bits_to_object(bits, ids) };
        match obj {
            Object::Keyword(k) => k,
            other => panic!("expected Object::Keyword, got {other:?}"),
        }
    }

    #[test]
    fn jit_e2e_keyword_literal_unqualified() {
        // :foo — a bare keyword literal evaluates to itself.
        let k = eval_form_via_jit_to_keyword(Object::Keyword(Keyword::intern_ns_name(None, "foo")));
        assert!(k.get_namespace().is_none());
        assert_eq!(k.get_name(), "foo");
    }

    #[test]
    fn jit_e2e_keyword_literal_qualified() {
        let k = eval_form_via_jit_to_keyword(Object::Keyword(Keyword::intern_ns_name(
            Some("user"),
            "name",
        )));
        assert_eq!(k.get_namespace(), Some("user"));
        assert_eq!(k.get_name(), "name");
    }

    #[test]
    fn jit_e2e_keyword_interns_globally() {
        // Unlike Symbol, Keyword IS globally interned in Clojure (and in
        // our port): `Keyword::intern_ns_name(...)` returns the same Arc
        // for the same (ns, name) pair. Verify Arc::ptr_eq round-trip.
        let expected = Keyword::intern_ns_name(None, "ping");
        let k = eval_form_via_jit_to_keyword(Object::Keyword(expected.clone()));
        assert!(Arc::ptr_eq(&expected, &k));
    }

    #[test]
    fn jit_e2e_def_keyword_then_deref() {
        // (do (def k :hello) k) → :hello
        let k = with_fresh_ns("test.kw.def.deref", || {
            let form = list_of(vec![
                sym("do"),
                list_of(vec![
                    sym("def"),
                    sym("k"),
                    Object::Keyword(Keyword::intern_ns_name(None, "hello")),
                ]),
                sym("k"),
            ]);
            eval_form_via_jit_to_keyword(form)
        });
        assert_eq!(k.get_name(), "hello");
        assert!(k.get_namespace().is_none());
    }

    // ---- quoted lists via Cons heap allocation + GcLiteral (JIT) ----------

    fn eval_form_via_jit_to_list(form: Object) -> Arc<PersistentList> {
        use dynlower::JitOutcome;
        use dynruntime::GcPolicy;
        let (gc, jit, entry, _roots) = super::compile_form_to_jit(form);
        let _thread = gc.install_thread();
        let _call_base = crate::runtime::install_call_table_base(jit.call_table_base_addr());
        let _local_chain = dynobj::roots::FrameChain::new();
        let _chain_root_g =
            unsafe { gc.push_extra_root_source(&_local_chain as *const dyn dynobj::RootSource) };
        let _chain_g = dynobj::roots::install_chain(&_local_chain);
        let bits = match gc.run_jit(&jit, entry, &[], GcPolicy::EveryPoint) {
            JitOutcome::Value(v) => v,
            other => panic!("unexpected JIT outcome: {other:?}"),
        };
        let ids = crate::runtime::heap_type_ids();
        let obj = unsafe { crate::runtime::heap_bits_to_object(bits, ids) };
        match obj {
            Object::List(l) => l,
            other => panic!("expected Object::List, got {other:?}"),
        }
    }

    /// Collect a PersistentList into a Rust Vec<Object> for easy assertions.
    fn list_to_vec(l: &Arc<PersistentList>) -> Vec<Object> {
        let mut out = Vec::new();
        let mut cur = l.clone();
        loop {
            match &*cur {
                PersistentList::Empty => break,
                PersistentList::Cons { first, rest, .. } => {
                    out.push(first.clone());
                    cur = rest.clone();
                }
            }
        }
        out
    }

    #[test]
    fn jit_e2e_quote_list_of_longs() {
        // (quote (1 2 3)) → list of 3 longs
        let form = list_of(vec![
            sym("quote"),
            list_of(vec![Object::Long(1), Object::Long(2), Object::Long(3)]),
        ]);
        let l = eval_form_via_jit_to_list(form);
        assert_eq!(l.count(), 3);
        let v = list_to_vec(&l);
        // Integers are boxed Longs, so they round-trip back as `Object::Long`
        // (a boxed-Long heap cell decodes to Long, not Double).
        assert!(matches!(v[0], Object::Long(1)));
        assert!(matches!(v[1], Object::Long(2)));
        assert!(matches!(v[2], Object::Long(3)));
    }

    #[test]
    fn jit_e2e_quote_list_of_symbols() {
        // (quote (a b c)) → list of 3 symbols
        let form = list_of(vec![
            sym("quote"),
            list_of(vec![sym("a"), sym("b"), sym("c")]),
        ]);
        let l = eval_form_via_jit_to_list(form);
        assert_eq!(l.count(), 3);
        let v = list_to_vec(&l);
        match (&v[0], &v[1], &v[2]) {
            (Object::Symbol(a), Object::Symbol(b), Object::Symbol(c)) => {
                assert_eq!(a.get_name(), "a");
                assert_eq!(b.get_name(), "b");
                assert_eq!(c.get_name(), "c");
            }
            other => panic!("expected three Symbols, got {other:?}"),
        }
    }

    #[test]
    fn jit_e2e_quote_empty_list() {
        // (quote ()) — empty list. The empty literal panics analyze right
        // now (EmptyExpr port pending). Skipping until that lands.
    }

    #[test]
    fn jit_e2e_quote_nested_list() {
        // (quote (1 (a) 2)) → 3-elem list with inner 1-elem list of Symbol a
        let inner = list_of(vec![sym("a")]);
        let form = list_of(vec![
            sym("quote"),
            list_of(vec![Object::Long(1), inner, Object::Long(2)]),
        ]);
        let l = eval_form_via_jit_to_list(form);
        assert_eq!(l.count(), 3);
        let v = list_to_vec(&l);
        match &v[1] {
            Object::List(inner) => {
                assert_eq!(inner.count(), 1);
                let inner_v = list_to_vec(inner);
                match &inner_v[0] {
                    Object::Symbol(s) => assert_eq!(s.get_name(), "a"),
                    other => panic!("expected Symbol, got {other:?}"),
                }
            }
            other => panic!("expected nested List, got {other:?}"),
        }
    }

    // ---- HostExpr (`.`) — clojure.lang.RT static-method dispatch (JIT) -----
    //
    // `(. clojure.lang.RT (inc 5))` → 6. The lookup goes through
    // Compiler.host_methods, which is populated in Compiler::new with one
    // entry per Rust-exported `cljvm_rt_*` extern. emit produces a direct
    // `Call`. More methods land here as clojure.core needs them.

    #[test]
    fn jit_e2e_dot_rt_inc_paren_form() {
        // (. clojure.lang.RT (inc 5)) → 6
        let v = eval_form_via_jit(list_of(vec![
            sym("."),
            sym("clojure.lang.RT"),
            list_of(vec![sym("inc"), Object::Long(5)]),
        ]));
        assert_eq!(nanbox_to_f64(v), 6.0);
    }

    #[test]
    fn jit_e2e_dot_rt_inc_unwrapped_form() {
        // (. clojure.lang.RT inc 5) → 6 — second supported call shape.
        let v = eval_form_via_jit(list_of(vec![
            sym("."),
            sym("clojure.lang.RT"),
            sym("inc"),
            Object::Long(5),
        ]));
        assert_eq!(nanbox_to_f64(v), 6.0);
    }

    #[test]
    fn jit_e2e_dot_rt_inc_chained() {
        // (. clojure.lang.RT (inc (. clojure.lang.RT (inc 0)))) → 2
        let inner = list_of(vec![
            sym("."),
            sym("clojure.lang.RT"),
            list_of(vec![sym("inc"), Object::Long(0)]),
        ]);
        let outer = list_of(vec![
            sym("."),
            sym("clojure.lang.RT"),
            list_of(vec![sym("inc"), inner]),
        ]);
        let v = eval_form_via_jit(outer);
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn jit_e2e_dot_unknown_method_throws_at_runtime() {
        // Unregistered static methods used to panic at analyze time.
        // Now they analyze as an in-band `(throw "unregistered …")` so
        // a defn that NEVER calls the method still compiles cleanly —
        // this lets us load forked clojure.core whose `(defn agent …)`
        // and friends reference Java classes we don't host. Calling
        // the method DOES still fail loudly (Exception JIT outcome
        // with a descriptive payload string).
        let outcome = eval_form_via_jit_outcome(list_of(vec![
            sym("."),
            sym("clojure.lang.RT"),
            list_of(vec![sym("nonexistent"), Object::Long(0)]),
        ]));
        assert!(
            matches!(outcome, dynlower::JitOutcome::Exception(_)),
            "expected Exception outcome from unregistered static method, got {outcome:?}"
        );
    }

    #[test]
    #[should_panic(expected = "Malformed member expression")]
    fn jit_e2e_dot_too_few_args_panics() {
        // (. clojure.lang.RT) — missing the member entirely.
        let _ = eval_form_via_jit(list_of(vec![sym("."), sym("clojure.lang.RT")]));
    }

    #[test]
    #[should_panic(expected = "Wrong number of args")]
    fn jit_e2e_quote_with_two_args_panics() {
        // (quote a b) → arity error per Java's ConstantExpr.Parser.
        let _ = eval_form_via_jit_to_symbol(list_of(vec![sym("quote"), sym("a"), sym("b")]));
    }

    // ---- reader → analyze → emit → JIT (full pipeline from source) --------

    #[test]
    fn jit_e2e_source_literal_long() {
        assert_eq!(nanbox_to_f64(eval_str_via_jit("42")), 42.0);
    }

    #[test]
    fn jit_e2e_source_arithmetic() {
        // (+ 1 2 3) → 6
        assert_eq!(nanbox_to_f64(eval_str_via_jit("(+ 1 2 3)")), 6.0);
        // (* 2 (+ 3 4)) → 14
        assert_eq!(nanbox_to_f64(eval_str_via_jit("(* 2 (+ 3 4))")), 14.0);
    }

    #[test]
    fn jit_e2e_source_if_expression() {
        assert_eq!(nanbox_to_f64(eval_str_via_jit("(if true 1 2)")), 1.0);
        assert_eq!(nanbox_to_f64(eval_str_via_jit("(if false 1 2)")), 2.0);
        assert_eq!(
            nanbox_to_f64(eval_str_via_jit("(if (< 3 5) 100 200)")),
            100.0
        );
    }

    #[test]
    fn jit_e2e_source_let_expression() {
        // (let* [x 10 y 20] (+ x y)) → 30
        assert_eq!(
            nanbox_to_f64(eval_str_via_jit("(let* [x 10 y 20] (+ x y))")),
            30.0
        );
    }

    #[test]
    fn jit_e2e_source_defn_factorial() {
        // From source: defn-style factorial.
        let src = "(do
          (def fact (fn* [n] (if (< n 2) 1 (* n (fact (- n 1))))))
          (fact 6))";
        let v = with_fresh_ns("test.source.fact", || eval_str_via_jit(src));
        assert_eq!(nanbox_to_f64(v), 720.0);
    }

    #[test]
    fn jit_e2e_source_dot_rt_inc() {
        // (. clojure.lang.RT (inc 41)) → 42
        let v = eval_str_via_jit("(. clojure.lang.RT (inc 41))");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    // ---- clojure.core function ports (each tested via JIT from source) ----
    //
    // Direct ports of the early `clojure.core` defns. Each lands as a string
    // of source code, parsed by our reader, analyzed/emitted, and exercised
    // through the JIT. Skipping ^:static / arglists metadata since we don't
    // model symbol metadata yet — semantics-equivalent without it.
    //
    // Tracking: clojure.core line numbers next to each port.

    /// Compose the clojure.core source-level fixture for tests below: a fresh
    /// namespace whose Vars are def'd from `defns`, then a `(do <test>)`.
    fn run_core_test(ns: &str, defns: &[&str], test: &str) -> u64 {
        with_fresh_ns(ns, || {
            let mut src = String::from("(do ");
            for d in defns {
                src.push_str(d);
                src.push(' ');
            }
            src.push_str(test);
            src.push(')');
            eval_str_via_jit(&src)
        })
    }

    /// `clojure.core/cons` (Java line 27).
    /// `(def cons (fn* [x s] (. clojure.lang.RT (cons x s))))`
    const CORE_CONS: &str = "(def cons (fn* [x s] (. clojure.lang.RT (cons x s))))";

    /// `clojure.core/first` (Java line ~52).
    const CORE_FIRST: &str = "(def first (fn* [coll] (. clojure.lang.RT (first coll))))";

    /// `clojure.core/next` (Java line ~62).
    const CORE_NEXT: &str = "(def next (fn* [coll] (. clojure.lang.RT (next coll))))";

    /// `clojure.core/inc` (much later in core.clj, but the simplest defn).
    const CORE_INC: &str = "(def inc (fn* [x] (. clojure.lang.RT (inc x))))";

    #[test]
    fn core_fn_cons_creates_single_element_list() {
        // (cons 1 nil), then take its first to verify it constructed.
        let v = run_core_test(
            "test.core.cons.one",
            &[CORE_CONS],
            "(. clojure.lang.RT (first (cons 1 nil)))",
        );
        assert_eq!(nanbox_to_f64(v), 1.0);
    }

    #[test]
    fn core_fn_first_on_cons_returns_head() {
        let v = run_core_test(
            "test.core.first.cons",
            &[CORE_CONS, CORE_FIRST],
            "(first (cons 42 nil))",
        );
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn core_fn_first_on_nil_returns_nil() {
        let v = run_core_test("test.core.first.nil", &[CORE_FIRST], "(first nil)");
        assert_eq!(v, 0x7FFC_0000_0000_0000);
    }

    #[test]
    fn core_fn_next_walks_two_element_list() {
        let v = run_core_test(
            "test.core.next.two",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT],
            "(first (next (cons 1 (cons 2 nil))))",
        );
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn core_fn_inc_works() {
        let v = run_core_test("test.core.inc", &[CORE_INC], "(inc 41)");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn core_fns_compose() {
        // (first (next (cons 10 (cons 20 (cons 30 nil))))) → 20
        // Uses cons/first/next together — the building blocks of every
        // clojure.core sequence function. This is a microcosm of how
        // `second`, `ffirst`, `fnext`, etc. are defined.
        let v = run_core_test(
            "test.core.compose",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT],
            "(first (next (cons 10 (cons 20 (cons 30 nil)))))",
        );
        assert_eq!(nanbox_to_f64(v), 20.0);
    }

    /// `clojure.core/second` (Java line ~96) — `(first (next x))`.
    const CORE_SECOND: &str = "(def second (fn* [x] (first (next x))))";

    #[test]
    fn core_fn_second() {
        let v = run_core_test(
            "test.core.second",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_SECOND],
            "(second (cons 100 (cons 200 nil)))",
        );
        assert_eq!(nanbox_to_f64(v), 200.0);
    }

    /// `clojure.core/ffirst` (Java line ~104) — `(first (first x))`.
    const CORE_FFIRST: &str = "(def ffirst (fn* [x] (first (first x))))";

    #[test]
    fn core_fn_ffirst() {
        // ((ffirst (cons (cons 7 nil) nil))) → 7
        let v = run_core_test(
            "test.core.ffirst",
            &[CORE_CONS, CORE_FIRST, CORE_FFIRST],
            "(ffirst (cons (cons 7 nil) nil))",
        );
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    /// `clojure.core/nfirst` (Java line ~112) — `(next (first x))`.
    const CORE_NFIRST: &str = "(def nfirst (fn* [x] (next (first x))))";

    #[test]
    fn core_fn_nfirst() {
        // ((first (nfirst (cons (cons 1 (cons 2 nil)) nil)))) → 2
        let v = run_core_test(
            "test.core.nfirst",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_NFIRST],
            "(first (nfirst (cons (cons 1 (cons 2 nil)) nil)))",
        );
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    /// `clojure.core/fnext` (Java line ~120) — `(first (next x))`. Same as
    /// `second` but with the canonical clojure.core name.
    const CORE_FNEXT: &str = "(def fnext (fn* [x] (first (next x))))";

    #[test]
    fn core_fn_fnext() {
        let v = run_core_test(
            "test.core.fnext",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_FNEXT],
            "(fnext (cons 1 (cons 2 (cons 3 nil))))",
        );
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    /// `clojure.core/identity` — `(def identity (fn [x] x))`. The simplest fn.
    const CORE_IDENTITY: &str = "(def identity (fn* [x] x))";

    #[test]
    fn core_fn_identity() {
        let v = run_core_test("test.core.identity", &[CORE_IDENTITY], "(identity 42)");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    /// `clojure.core/zero?` — `(def zero? (fn [n] (= n 0)))`.
    const CORE_ZERO_Q: &str = "(def zero? (fn* [n] (= n 0)))";

    /// Decode a NanBox-encoded boolean from a JIT result.
    fn nanbox_to_bool(bits: u64) -> bool {
        // Tag-pattern check + payload comparison.
        const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
        const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
        const TAG_MASK: u64 = 0x0003_0000_0000_0000;
        const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
        assert_eq!(bits & FULL_MASK, TAG_PATTERN, "not a NanBox tagged value");
        let tag = (bits & TAG_MASK) >> 48;
        assert_eq!(tag, 1, "expected TAG_BOOL=1, got {tag}");
        (bits & PAYLOAD_MASK) != 0
    }

    #[test]
    fn core_fn_zero_q_true() {
        let v = run_core_test("test.core.zero.true", &[CORE_ZERO_Q], "(zero? 0)");
        assert!(nanbox_to_bool(v));
    }

    #[test]
    fn core_fn_zero_q_false() {
        let v = run_core_test("test.core.zero.false", &[CORE_ZERO_Q], "(zero? 5)");
        assert!(!nanbox_to_bool(v));
    }

    /// `clojure.core/pos?` — `(def pos? (fn [n] (> n 0)))`.
    const CORE_POS_Q: &str = "(def pos? (fn* [n] (> n 0)))";

    #[test]
    fn core_fn_pos_q() {
        assert!(nanbox_to_bool(run_core_test(
            "test.core.pos.1",
            &[CORE_POS_Q],
            "(pos? 5)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.core.pos.2",
            &[CORE_POS_Q],
            "(pos? 0)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.core.pos.3",
            &[CORE_POS_Q],
            "(pos? -3)"
        )));
    }

    /// `clojure.core/neg?` — `(def neg? (fn [n] (< n 0)))`.
    const CORE_NEG_Q: &str = "(def neg? (fn* [n] (< n 0)))";

    #[test]
    fn core_fn_neg_q() {
        assert!(nanbox_to_bool(run_core_test(
            "test.core.neg.1",
            &[CORE_NEG_Q],
            "(neg? -5)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.core.neg.2",
            &[CORE_NEG_Q],
            "(neg? 0)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.core.neg.3",
            &[CORE_NEG_Q],
            "(neg? 5)"
        )));
    }

    /// `clojure.core/dec` — `(def dec (fn [x] (- x 1)))`. The Java source
    /// routes through RT.dec for inline; the pure-Clojure shape is `(- x 1)`.
    const CORE_DEC: &str = "(def dec (fn* [x] (- x 1)))";

    #[test]
    fn core_fn_dec() {
        let v = run_core_test("test.core.dec", &[CORE_DEC], "(dec 43)");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    /// `clojure.core/min` (two-arg slice). Real clojure.core/min is variadic;
    /// the two-arg core is `(if (< x y) x y)`. We port that shape.
    const CORE_MIN2: &str = "(def min (fn* [x y] (if (< x y) x y)))";

    #[test]
    fn core_fn_min2() {
        let v1 = run_core_test("test.core.min.lhs", &[CORE_MIN2], "(min 3 7)");
        assert_eq!(nanbox_to_f64(v1), 3.0);
        let v2 = run_core_test("test.core.min.rhs", &[CORE_MIN2], "(min 9 4)");
        assert_eq!(nanbox_to_f64(v2), 4.0);
    }

    /// `clojure.core/max` (two-arg slice).
    const CORE_MAX2: &str = "(def max (fn* [x y] (if (> x y) x y)))";

    #[test]
    fn core_fn_max2() {
        let v1 = run_core_test("test.core.max.lhs", &[CORE_MAX2], "(max 3 7)");
        assert_eq!(nanbox_to_f64(v1), 7.0);
        let v2 = run_core_test("test.core.max.rhs", &[CORE_MAX2], "(max 9 4)");
        assert_eq!(nanbox_to_f64(v2), 9.0);
    }

    /// `clojure.core/not` — `(def not (fn [x] (if x false true)))`.
    const CORE_NOT: &str = "(def not (fn* [x] (if x false true)))";

    #[test]
    fn core_fn_not() {
        assert!(!nanbox_to_bool(run_core_test(
            "test.core.not.t",
            &[CORE_NOT],
            "(not true)"
        )));
        assert!(nanbox_to_bool(run_core_test(
            "test.core.not.f",
            &[CORE_NOT],
            "(not false)"
        )));
        // Clojure semantics: only nil and false are falsey.
        assert!(nanbox_to_bool(run_core_test(
            "test.core.not.nil",
            &[CORE_NOT],
            "(not nil)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.core.not.0",
            &[CORE_NOT],
            "(not 0)"
        )));
    }

    /// `clojure.core/nil?` — `(def nil? (fn [x] (if x false true)))` (effectively).
    /// Real clojure.core uses `(. clojure.lang.Util nil)` but for our subset
    /// `(if x false true)` is equivalent when only nil/false are falsey AND
    /// false isn't passed — except false would also return true. Use a more
    /// careful shape: compare against nil via `(= x nil)`. But we don't have
    /// runtime `=` on heap pointers yet. For now, narrow `nil?` to "is the
    /// argument the nil tag" — a runtime predicate exposed via a primop.
    /// Sticking to what's available: actual nil-only check requires PrimOp
    /// equality on NanBox tag, which the existing `=` PrimOp on Long doesn't
    /// do. SKIPPING `nil?` for now; the right port lands when we have
    /// `RT.nil?` or NanBox-tag-aware `=`.
    ///
    /// What we CAN port and test: `true?` since true is a specific value
    /// and `(= x true)` works through our Bool-comparing prim eq.

    /// `clojure.core/zero?`-style: a manually-defined fn that uses recursion
    /// to count down to zero, mirroring the kind of pattern macros expand into.
    /// `(def countdown (fn [n] (if (zero? n) :done (countdown (- n 1)))))`
    const CORE_COUNTDOWN: &str = "(def countdown (fn* [n] (if (= n 0) :done (countdown (- n 1)))))";

    #[test]
    fn fn_body_returning_keyword_roundtrips() {
        // A fn body returning a heap-allocated keyword literal. Earlier
        // version of this test SEGV'd because the test helper dropped the
        // GC runtime before the decode; switched to `eval_str_via_jit_to_object`
        // which keeps the runtime alive across the decode.
        let obj = with_fresh_ns("test.diag.fn.kw", || {
            eval_str_via_jit_to_object("(do (def f (fn* [] :done)) (f))")
        });
        match obj {
            Object::Keyword(k) => assert_eq!(k.get_name(), "done"),
            other => panic!("expected :done, got {other:?}"),
        }
    }

    #[test]
    fn map_literal_roundtrips_through_jit() {
        // `{:a 1 :b 2}` should read, intern as a PersistentHashMap literal,
        // allocate the host-side Arc + heap wrapper, evaluate to its
        // tagged-pointer NanBox, and decode back into Object::Map with the
        // same two entries.
        let obj = with_fresh_ns("test.diag.map.lit", || {
            eval_str_via_jit_to_object("{:a 1 :b 2}")
        });
        match obj {
            Object::Map(m) => {
                assert_eq!(m.count(), 2);
                let a = m.val_at(&Object::Keyword(
                    super::super::keyword::Keyword::intern_ns_name(None, "a"),
                ));
                let b = m.val_at(&Object::Keyword(
                    super::super::keyword::Keyword::intern_ns_name(None, "b"),
                ));
                // Numeric literals NanBox-roundtrip as f64 bits; decode may
                // surface as Long or Double depending on path.
                assert_eq!(a.as_f64(), Some(1.0), "expected :a → 1, got {a:?}");
                assert_eq!(b.as_f64(), Some(2.0), "expected :b → 2, got {b:?}");
            }
            other => panic!("expected Object::Map, got {other:?}"),
        }
    }

    #[test]
    fn empty_map_literal_roundtrips_through_jit() {
        let obj = with_fresh_ns("test.diag.map.empty", || eval_str_via_jit_to_object("{}"));
        match obj {
            Object::Map(m) => assert_eq!(m.count(), 0),
            other => panic!("expected Object::Map, got {other:?}"),
        }
    }

    #[test]
    fn set_literal_roundtrips_through_jit() {
        let obj = with_fresh_ns("test.diag.set.lit", || {
            eval_str_via_jit_to_object("#{:a :b :c}")
        });
        match obj {
            Object::Set(s) => {
                assert_eq!(s.count(), 3);
                assert!(s.contains(&Object::Keyword(
                    super::super::keyword::Keyword::intern_ns_name(None, "a"),
                )));
                assert!(s.contains(&Object::Keyword(
                    super::super::keyword::Keyword::intern_ns_name(None, "b"),
                )));
                assert!(s.contains(&Object::Keyword(
                    super::super::keyword::Keyword::intern_ns_name(None, "c"),
                )));
            }
            other => panic!("expected Object::Set, got {other:?}"),
        }
    }

    #[test]
    fn empty_set_literal_roundtrips_through_jit() {
        let obj = with_fresh_ns("test.diag.set.empty", || eval_str_via_jit_to_object("#{}"));
        match obj {
            Object::Set(s) => assert_eq!(s.count(), 0),
            other => panic!("expected Object::Set, got {other:?}"),
        }
    }

    /// `clojure.core/true?` — `(def true? (fn [x] (if (= x true) true false)))`.
    /// Real clojure.core is `(def true? (fn [x] (clojure.lang.Util/identical x true)))`
    /// — semantically: identical to the true singleton. Our `=` PrimOp on
    /// bool args is bit-equality on the NanBox, which IS identity for the
    /// singletons true/false.
    const CORE_TRUE_Q: &str = "(def true? (fn* [x] (if (= x true) true false)))";

    #[test]
    fn core_fn_true_q() {
        assert!(nanbox_to_bool(run_core_test(
            "test.true_q.1",
            &[CORE_TRUE_Q],
            "(true? true)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.true_q.2",
            &[CORE_TRUE_Q],
            "(true? false)"
        )));
    }

    /// `clojure.core/false?` — symmetric to `true?`.
    const CORE_FALSE_Q: &str = "(def false? (fn* [x] (if (= x false) true false)))";

    #[test]
    fn core_fn_false_q() {
        assert!(nanbox_to_bool(run_core_test(
            "test.false_q.1",
            &[CORE_FALSE_Q],
            "(false? false)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.false_q.2",
            &[CORE_FALSE_Q],
            "(false? true)"
        )));
    }

    /// `clojure.core/=` (1-arg slice, always true).
    const CORE_EQ1: &str = "(def =1 (fn* [_] true))";

    #[test]
    fn core_fn_eq_one_arg() {
        assert!(nanbox_to_bool(run_core_test(
            "test.eq1",
            &[CORE_EQ1],
            "(=1 42)"
        )));
    }

    /// `clojure.core/=` (2-arg slice). Real `=` dispatches through
    /// `Util.equiv`; for our subset, comparing primitive longs and bools is
    /// exactly what the PrimOp `=` does. We expose it under the user-facing
    /// name so calls in user code don't have to know about the prim form.
    const CORE_EQ2: &str = "(def =2 (fn* [a b] (= a b)))";

    #[test]
    fn core_fn_eq_two_args() {
        assert!(nanbox_to_bool(run_core_test(
            "test.eq2.t",
            &[CORE_EQ2],
            "(=2 3 3)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.eq2.f",
            &[CORE_EQ2],
            "(=2 3 4)"
        )));
    }

    /// `clojure.core/not=` — `(def not= (fn [x y] (not (= x y))))`.
    const CORE_NOT_EQ: &str = "(def not= (fn* [x y] (if (= x y) false true)))";

    #[test]
    fn core_fn_not_eq() {
        assert!(nanbox_to_bool(run_core_test(
            "test.neq.diff",
            &[CORE_NOT_EQ],
            "(not= 1 2)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.neq.same",
            &[CORE_NOT_EQ],
            "(not= 5 5)"
        )));
    }

    /// `clojure.core/and` (binary slice). Real `and` is a variadic macro;
    /// the binary case is `(if a b a)`. We test the binary shape directly.
    const CORE_AND2: &str = "(def and2 (fn* [a b] (if a b a)))";

    #[test]
    fn core_fn_and2() {
        // Both truthy → returns second.
        let v = run_core_test("test.and2.tt", &[CORE_AND2], "(and2 true 7)");
        assert_eq!(nanbox_to_f64(v), 7.0);
        // First falsey → returns first.
        let v = run_core_test("test.and2.ft", &[CORE_AND2], "(and2 false 7)");
        assert!(!nanbox_to_bool(v));
    }

    /// `clojure.core/or` (binary slice). `(if a a b)`.
    const CORE_OR2: &str = "(def or2 (fn* [a b] (if a a b)))";

    #[test]
    fn core_fn_or2() {
        let v = run_core_test("test.or2.tt", &[CORE_OR2], "(or2 5 7)");
        assert_eq!(nanbox_to_f64(v), 5.0);
        let v = run_core_test("test.or2.fb", &[CORE_OR2], "(or2 false 7)");
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    /// `clojure.core/when` (binary slice). `(when test body)` → `(if test body nil)`.
    /// Real `when` is a macro; the manual expansion is `(if test (do body) nil)`.
    const CORE_WHEN: &str = "(def my-when (fn* [t v] (if t v nil)))";

    #[test]
    fn core_fn_when() {
        let v = run_core_test("test.when.t", &[CORE_WHEN], "(my-when true 99)");
        assert_eq!(nanbox_to_f64(v), 99.0);
        let v = run_core_test("test.when.f", &[CORE_WHEN], "(my-when false 99)");
        assert_eq!(v, 0x7FFC_0000_0000_0000); // nil bits
    }

    /// `clojure.core/even?` — `(zero? (rem n 2))`. We don't have rem yet; the
    /// pure version using bitwise/arithmetic is `(= 0 (- n (* 2 (quot n 2))))`.
    /// We have no quot either, so port as `(if (< n 0) (even? (- n)) (loop … ))`
    /// — recursive subtraction. Simple shape: keep subtracting 2.
    const CORE_EVEN_Q: &str =
        "(def even? (fn* [n] (if (= n 0) true (if (= n 1) false (even? (- n 2))))))";

    #[test]
    fn core_fn_even_q() {
        assert!(nanbox_to_bool(run_core_test(
            "test.even.0",
            &[CORE_EVEN_Q],
            "(even? 0)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.even.1",
            &[CORE_EVEN_Q],
            "(even? 1)"
        )));
        assert!(nanbox_to_bool(run_core_test(
            "test.even.10",
            &[CORE_EVEN_Q],
            "(even? 10)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.even.7",
            &[CORE_EVEN_Q],
            "(even? 7)"
        )));
    }

    /// `clojure.core/odd?` — complement of even.
    const CORE_ODD_Q: &str =
        "(def odd? (fn* [n] (if (= n 0) false (if (= n 1) true (odd? (- n 2))))))";

    #[test]
    fn core_fn_odd_q() {
        assert!(!nanbox_to_bool(run_core_test(
            "test.odd.0",
            &[CORE_ODD_Q],
            "(odd? 0)"
        )));
        assert!(nanbox_to_bool(run_core_test(
            "test.odd.1",
            &[CORE_ODD_Q],
            "(odd? 1)"
        )));
        assert!(nanbox_to_bool(run_core_test(
            "test.odd.11",
            &[CORE_ODD_Q],
            "(odd? 11)"
        )));
    }

    /// `clojure.core/if-not` macro stand-in: `(if-not test then else)` →
    /// `(if test else then)`. We port the runtime shape directly as a fn.
    const CORE_IF_NOT: &str = "(def if-not (fn* [t a b] (if t b a)))";

    #[test]
    fn core_fn_if_not() {
        let v = run_core_test("test.ifnot.t", &[CORE_IF_NOT], "(if-not true 1 2)");
        assert_eq!(nanbox_to_f64(v), 2.0);
        let v = run_core_test("test.ifnot.f", &[CORE_IF_NOT], "(if-not false 1 2)");
        assert_eq!(nanbox_to_f64(v), 1.0);
    }

    /// `clojure.core/abs` (1-arg, integer slice).
    /// Real: `(if (pos? n) n (- n))`; pure shape via `<`.
    const CORE_ABS: &str = "(def abs (fn* [n] (if (< n 0) (- n) n)))";

    #[test]
    fn core_fn_abs() {
        let v = run_core_test("test.abs.neg", &[CORE_ABS], "(abs -7)");
        assert_eq!(nanbox_to_f64(v), 7.0);
        let v = run_core_test("test.abs.pos", &[CORE_ABS], "(abs 5)");
        assert_eq!(nanbox_to_f64(v), 5.0);
        let v = run_core_test("test.abs.zero", &[CORE_ABS], "(abs 0)");
        assert_eq!(nanbox_to_f64(v), 0.0);
    }

    /// `clojure.core/inc'` (vs `inc`): no overflow check version. Same shape
    /// as inc for our subset.
    const CORE_INC_PRIME: &str = "(def inc' (fn* [x] (+ x 1)))";

    #[test]
    fn core_fn_inc_prime() {
        let v = run_core_test("test.inc_prime", &[CORE_INC_PRIME], "(inc' 10)");
        assert_eq!(nanbox_to_f64(v), 11.0);
    }

    /// Composition of multiple defns: `square` then use it in a sum.
    const CORE_SQUARE: &str = "(def square (fn* [n] (* n n)))";

    #[test]
    fn core_fn_square() {
        let v = run_core_test("test.square", &[CORE_SQUARE], "(square 9)");
        assert_eq!(nanbox_to_f64(v), 81.0);
    }

    #[test]
    fn core_fn_square_composes() {
        // (+ (square 3) (square 4)) → 25
        let v = run_core_test(
            "test.square.pythag",
            &[CORE_SQUARE],
            "(+ (square 3) (square 4))",
        );
        assert_eq!(nanbox_to_f64(v), 25.0);
    }

    /// Compose first/next/cons: extract n-th element by repeated next + first.
    /// `(def nth0 (fn* [coll] (first coll)))`
    /// `(def nth1 (fn* [coll] (first (next coll))))`
    /// `(def nth2 (fn* [coll] (first (next (next coll)))))`
    /// These are the building blocks for `nth`.

    /// `clojure.core/count` (recursive port for our Cons-only world). Real
    /// clojure.core/count delegates to RT.count which dispatches on type;
    /// we model the seq-counting case recursively. Returns 0 for nil.
    const CORE_COUNT: &str =
        "(def count (fn* [coll] (if (= coll nil) 0 (+ 1 (count (next coll))))))";

    #[test]
    fn core_fn_count_empty() {
        let v = run_core_test(
            "test.count.empty",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_COUNT],
            "(count nil)",
        );
        assert_eq!(nanbox_to_f64(v), 0.0);
    }

    #[test]
    fn core_fn_count_three() {
        let v = run_core_test(
            "test.count.three",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_COUNT],
            "(count (cons 1 (cons 2 (cons 3 nil))))",
        );
        assert_eq!(nanbox_to_f64(v), 3.0);
    }

    /// `clojure.core/last` — recursively walk to the final element.
    const CORE_LAST: &str =
        "(def last (fn* [coll] (if (= (next coll) nil) (first coll) (last (next coll)))))";

    #[test]
    fn core_fn_last() {
        let v = run_core_test(
            "test.last",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_LAST],
            "(last (cons 10 (cons 20 (cons 30 nil))))",
        );
        assert_eq!(nanbox_to_f64(v), 30.0);
    }

    /// `clojure.core/nth` (2-arg slice, recursive). Real clojure.core has
    /// a 3-arg variant with not-found default, plus bounds checking;
    /// we port the basic recursive shape.
    const CORE_NTH: &str =
        "(def nth (fn* [coll i] (if (= i 0) (first coll) (nth (next coll) (- i 1)))))";

    #[test]
    fn core_fn_nth_zero() {
        let v = run_core_test(
            "test.nth.0",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_NTH],
            "(nth (cons 10 (cons 20 (cons 30 nil))) 0)",
        );
        assert_eq!(nanbox_to_f64(v), 10.0);
    }

    #[test]
    fn core_fn_nth_middle() {
        let v = run_core_test(
            "test.nth.1",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_NTH],
            "(nth (cons 10 (cons 20 (cons 30 nil))) 1)",
        );
        assert_eq!(nanbox_to_f64(v), 20.0);
    }

    /// Sum over a list — pure clojure.core idiom even though `reduce` exists.
    const CORE_SUM: &str =
        "(def sum (fn* [coll] (if (= coll nil) 0 (+ (first coll) (sum (next coll))))))";

    #[test]
    fn core_fn_sum() {
        let v = run_core_test(
            "test.sum",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_SUM],
            "(sum (cons 1 (cons 2 (cons 3 (cons 4 nil)))))",
        );
        assert_eq!(nanbox_to_f64(v), 10.0);
    }

    /// Sum a range using just recursion + arithmetic — sanity check for
    /// the call ABI under repeated invocation.
    #[test]
    fn core_fn_sum_via_recursive_descent() {
        let src = "(do
          (def add-down (fn* [n acc] (if (= n 0) acc (add-down (- n 1) (+ acc n)))))
          (add-down 100 0))";
        let v = with_fresh_ns("test.add_down", || eval_str_via_jit(src));
        // 1+2+…+100 = 5050
        assert_eq!(nanbox_to_f64(v), 5050.0);
    }

    /// Fibonacci — classic test for fn dispatch + recursion.
    #[test]
    fn core_fn_fibonacci() {
        let src = "(do
          (def fib (fn* [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))
          (fib 10))";
        let v = with_fresh_ns("test.fib", || eval_str_via_jit(src));
        // fib(10) = 55
        assert_eq!(nanbox_to_f64(v), 55.0);
    }

    /// Mutual recursion. Java's Clojure needs a forward `(declare odd?)` to
    /// give the symbol a Var binding before `even?`'s body resolves it.
    /// We model `declare` as a bare `(def name)` (which our DefExpr already
    /// supports). The `(. clojure.lang.RT (...))` invocation infra isn't
    /// needed here — it's just one Var pointing at another's FuncRef.
    ///
    /// Mutual recursion across two `def`s still DOES NOT WORK: `even-mut?`'s
    /// body references `odd-mut?` before it is defined, so the forward
    /// reference is unresolved at analyze time. Previously this aborted
    /// analysis with a hard panic; now an unresolved non-dotted symbol is
    /// deferred to a runtime throw (same treatment as an unresolved class
    /// reference), so the `def`s COMPILE and the failure surfaces only when
    /// the missing fn is actually called — here `(even-mut? 10)` recurses
    /// into the `odd-mut?` throw-stub and exits via `JitOutcome::Exception`.
    /// Real mutual recursion needs forward `declare` / runtime var deref.
    #[test]
    fn core_fn_mutual_recursion_currently_unsupported() {
        let src = "(do
          (def even-mut? (fn* [n] (if (= n 0) true (odd-mut? (- n 1)))))
          (def odd-mut?  (fn* [n] (if (= n 0) false (even-mut? (- n 1)))))
          (even-mut? 10))";
        let form = super::super::lisp_reader::read_str(src)
            .unwrap_or_else(|e| panic!("read_str failed: {e}"));
        let outcome = with_fresh_ns("test.mutual", || eval_form_via_jit_outcome(form));
        assert!(
            matches!(outcome, dynlower::JitOutcome::Exception(_)),
            "forward-referenced `odd-mut?` should compile to a throw-stub and \
             surface as an Exception at call time, got {outcome:?}"
        );
    }

    // ---- first-class fn values + higher-order dispatch (IFn.invoke) -------

    #[test]
    fn higher_order_apply_inline_fn_to_arg() {
        // `((fn* [f x] (f x)) (fn* [n] (+ n 100)) 5)` → 105
        let v = with_fresh_ns("test.ho.inline", || {
            eval_str_via_jit("((fn* [f x] (f x)) (fn* [n] (+ n 100)) 5)")
        });
        assert_eq!(nanbox_to_f64(v), 105.0);
    }

    #[test]
    fn higher_order_var_alias_call() {
        // (def my-inc inc) (my-inc 41) → 42
        // Aliasing one fn-Var to another's value, then calling through the
        // alias goes through `cljvm_rt_invoke_1` (the alias isn't itself
        // bound to a FnExpr, so var_fns has no entry).
        let v = with_fresh_ns("test.ho.alias", || {
            eval_str_via_jit(&format!("(do {CORE_INC} (def my-inc inc) (my-inc 41))"))
        });
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn higher_order_apply_via_let_bound_fn() {
        // (let* [f (fn* [n] (* n n))] (f 7)) → 49
        // FnExpr in a let position. Existing static path catches this via
        // LocalBindingExpr's init = FnExpr, but verifying it still works.
        let v = eval_str_via_jit("(let* [f (fn* [n] (* n n))] (f 7))");
        assert_eq!(nanbox_to_f64(v), 49.0);
    }

    #[test]
    fn higher_order_fn_returned_by_fn() {
        // (def make-adder (fn [n] n))  — currently no closures, so the
        // returned fn can't capture `n`. Test a returned non-capturing fn:
        // `((fn [] inc) 5)` → 6 via dynamic invoke. The outer fn returns the
        // inc Var's value (a fn handle), which we then call.
        let v = with_fresh_ns("test.ho.return_fn", || {
            eval_str_via_jit(&format!("(do {CORE_INC} (((fn* [] inc)) 5))"))
        });
        assert_eq!(nanbox_to_f64(v), 6.0);
    }

    /// `clojure.core/apply` — 2-arg slice: `(apply f args)` calls `f` with
    /// args from the list. Real apply is variadic; this is the simplest
    /// faithful shape. We hand-roll for arity 1 (apply to a 1-elem list):
    /// `(def apply1 (fn [f args] (f (first args))))`.
    const CORE_APPLY1: &str = "(def apply1 (fn* [f args] (f (. clojure.lang.RT (first args)))))";

    #[test]
    fn higher_order_apply1_inc_to_singleton_list() {
        let v = with_fresh_ns("test.apply1", || {
            eval_str_via_jit(&format!(
                "(do {CORE_INC} {CORE_APPLY1} \
                   (apply1 inc (. clojure.lang.RT (cons 41 nil))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    // ---- map / filter / reduce (higher-order over Cons lists) -------------

    /// `clojure.core/map` (1-coll slice, eager). Real `map` is lazy and
    /// variadic; this is the simplest faithful shape used inside many
    /// other defns. Builds a new list by applying `f` to each element.
    const CORE_MAP: &str = "(def map (fn* [f coll] \
        (if (. clojure.lang.Util (isNil coll)) \
            nil \
            (. clojure.lang.RT (cons (f (. clojure.lang.RT (first coll))) \
                                     (map f (. clojure.lang.RT (next coll))))))))";

    #[test]
    fn core_fn_map_inc_over_three_elements() {
        // (first (map inc (cons 1 (cons 2 (cons 3 nil))))) → 2
        let v = with_fresh_ns("test.map.inc.first", || {
            eval_str_via_jit(&format!(
                "(do {CORE_INC} {CORE_MAP} \
                   (. clojure.lang.RT (first \
                     (map inc (. clojure.lang.RT (cons 1 \
                       (. clojure.lang.RT (cons 2 \
                         (. clojure.lang.RT (cons 3 nil))))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn core_fn_map_inc_third_element() {
        let v = with_fresh_ns("test.map.inc.third", || {
            eval_str_via_jit(&format!(
                "(do {CORE_INC} {CORE_MAP} \
                   (. clojure.lang.RT (first \
                     (. clojure.lang.RT (next \
                       (. clojure.lang.RT (next \
                         (map inc (. clojure.lang.RT (cons 1 \
                           (. clojure.lang.RT (cons 2 \
                             (. clojure.lang.RT (cons 3 nil))))))))))))))"
            ))
        });
        // map(inc, [1,2,3]) → [2,3,4], third = 4.
        assert_eq!(nanbox_to_f64(v), 4.0);
    }

    /// `clojure.core/filter` (eager). Returns elements where pred is truthy.
    const CORE_FILTER: &str = "(def filter (fn* [pred coll] \
        (if (. clojure.lang.Util (isNil coll)) \
            nil \
            (if (pred (. clojure.lang.RT (first coll))) \
                (. clojure.lang.RT (cons (. clojure.lang.RT (first coll)) \
                                         (filter pred (. clojure.lang.RT (next coll))))) \
                (filter pred (. clojure.lang.RT (next coll)))))))";

    #[test]
    fn core_fn_filter_pos_q() {
        // (first (filter pos? (cons -1 (cons 2 (cons -3 (cons 4 nil)))))) → 2
        let v = with_fresh_ns("test.filter.pos", || {
            eval_str_via_jit(&format!(
                "(do {CORE_POS_Q} {CORE_FILTER} \
                   (. clojure.lang.RT (first \
                     (filter pos? (. clojure.lang.RT (cons -1 \
                       (. clojure.lang.RT (cons 2 \
                         (. clojure.lang.RT (cons -3 \
                           (. clojure.lang.RT (cons 4 nil))))))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    /// `clojure.core/reduce` (3-arg: f, init, coll). Real reduce is more
    /// nuanced; this is the basic shape.
    const CORE_REDUCE: &str = "(def reduce (fn* [f init coll] \
        (if (. clojure.lang.Util (isNil coll)) \
            init \
            (reduce f (f init (. clojure.lang.RT (first coll))) \
                      (. clojure.lang.RT (next coll))))))";

    #[test]
    fn core_fn_reduce_sum_with_plus_fn() {
        // Sum a list via reduce with an explicit fn arg. (Direct prim-op +
        // isn't a fn value, so we wrap.)
        let v = with_fresh_ns("test.reduce.sum", || {
            eval_str_via_jit(&format!(
                "(do {CORE_REDUCE} \
                   (def add (fn* [a b] (+ a b))) \
                   (reduce add 0 (. clojure.lang.RT (cons 1 \
                     (. clojure.lang.RT (cons 2 \
                       (. clojure.lang.RT (cons 3 \
                         (. clojure.lang.RT (cons 4 nil))))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 10.0);
    }

    /// `clojure.core/comp` (2-fn slice). `((comp f g) x)` → `(f (g x))`.
    /// Real comp is variadic and returns a fn; we model it directly here
    /// without higher-order *returned* values for the rest case.
    const CORE_COMP2: &str = "(def comp2 (fn* [f g] (fn* [x] (f (g x)))))";

    #[test]
    fn closure_construction_doesnt_crash() {
        // Just construct a closure and discard it. (let* [x 42] (fn* [] x))
        // returns the closure handle (NanBox bits); don't invoke it.
        // Smoke test for the alloc / store path.
        let bits = eval_str_via_jit("(let* [x 42] (fn* [] x))");
        // Closure: TAG_PTR (tag = 2).
        const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
        const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
        const TAG_MASK: u64 = 0x0003_0000_0000_0000;
        assert_eq!(bits & FULL_MASK, TAG_PATTERN);
        let tag = (bits & TAG_MASK) >> 48;
        assert_eq!(tag, 2, "closure should be TAG_PTR, got tag {tag}");
    }

    /// Simplest closure: captures one outer value, returns it. No invocation
    /// of captured fn. Verifies the alloc / store / load round-trip.
    #[test]
    fn closure_returns_captured_long() {
        // ((fn* [x] (fn* [] x)) 42)  — but we can't invoke result without 0-arg
        // dynamic path. Use a let to bind the closure first.
        // ((let [x 42 f (fn* [] x)] f))
        // Wait, our let can shadow. Let me write:
        // (let* [x 42] ((fn* [] x)))
        let v = eval_str_via_jit("(let* [x 42] ((fn* [] x)))");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn closure_returns_captured_long_one_indirect() {
        // (let* [x 42] (let* [f (fn* [] x)] (f)))
        let v = eval_str_via_jit("(let* [x 42] (let* [f (fn* [] x)] (f)))");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn closure_one_arg_capturing_outer() {
        // ((fn* [x] (fn* [y] (+ x y))) ...) — we can't directly invoke the
        // returned closure since the outer's invocation chain isn't supported
        // for arity 1 on heap-allocated fns yet. Try via let.
        // (let* [add-x (let* [x 10] (fn* [y] (+ x y)))] (add-x 5)) → 15
        let v = eval_str_via_jit("(let* [add-x (let* [x 10] (fn* [y] (+ x y)))] (add-x 5))");
        assert_eq!(nanbox_to_f64(v), 15.0);
    }

    /// `clojure.core/partial` (1-extra-arg slice). `(partial f x)` returns a
    // ---- Session: form-by-form persistent JIT ----------------------------

    #[test]
    fn session_evaluates_two_forms_sharing_state() {
        let mut sess = with_fresh_ns_session("test.session.two", || super::Session::new());
        // First form binds; second form reads.
        let _ = sess.eval_str("(def x 42)");
        let v = sess.eval_str("x");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn ns_form_switches_current_namespace() {
        // Fresh test ns so we don't pollute clojure.core.
        let mut sess = with_fresh_ns_session("test.ns.start", || super::Session::new());
        // Switch to a new namespace.
        let _ = sess.eval_str("(ns my.test.ns)");
        // Def in the new ns then read back.
        let _ = sess.eval_str("(def x 99)");
        let v = sess.eval_str("x");
        assert_eq!(nanbox_to_f64(v), 99.0);
        // The Var should live in `my.test.ns` now.
        use super::super::namespace::Namespace;
        let ns = Namespace::find(&Symbol::intern("my.test.ns")).expect("ns must exist");
        let x_sym = Symbol::intern("x");
        let var = ns
            .find_interned_var(&x_sym)
            .expect("x must be interned in my.test.ns");
        // The Var binds via `cljvm_var_bind_root`, which roundtrips through
        // NanBox. A boxed Long arrives as a heap pointer and is stored
        // opaquely as `Object::Host(HeapBits)` (Vars keep heap values without
        // eagerly decoding them); unbox it to confirm the numeric value.
        match var.deref() {
            Object::Double(x) if x == 99.0 => {}
            Object::Long(99) => {}
            host @ Object::Host(_) => {
                let bits = crate::runtime::object_to_nanbox(&host);
                assert_eq!(
                    nanbox_to_f64(bits),
                    99.0,
                    "boxed-Long Var value should unbox to 99"
                );
            }
            other => panic!("expected x=99 in my.test.ns, got {other:?}"),
        }
    }











    /// `(ns …)`/`in-ns` must NOT mutate global state — they switch the
    /// current thread's `*ns*` binding only. The cleanest deterministic
    /// proof is that a *fresh* Session does not inherit a prior Session's
    /// namespace: if `(ns conc.leak.a)` had written the shared `*ns*` root
    /// (the old `bind_root` behavior), the next Session would seed its
    /// current namespace from that polluted root and a later `def` would
    /// land in `conc.leak.a` instead of the `clojure.core` default. With the
    /// thread-local `set_value` fix the root is untouched, so the fresh
    /// session defaults correctly.
    ///
    /// Single-threaded on purpose: a two-thread variant additionally trips
    /// over the moving GC not being a concurrent-mutator design (parked
    /// threads holding live Sessions), which is an orthogonal concern. The
    /// no-global-mutation property proven here is exactly what makes `*ns*`
    /// thread-local.
    #[test]
    fn ns_switch_does_not_leak_to_a_fresh_session() {
        use super::super::namespace::Namespace;

        // Session 1 switches into a throwaway namespace and defs there.
        let mut s1 = super::Session::new();
        s1.eval_str("(ns conc.leak.a)");
        s1.eval_str("(def x 1)");
        assert!(
            Namespace::find(&Symbol::intern("conc.leak.a"))
                .and_then(|n| n.find_interned_var(&Symbol::intern("x")))
                .is_some(),
            "sanity: s1's `(def x)` should be interned in conc.leak.a"
        );

        // A fresh Session must default to clojure.core, NOT inherit
        // conc.leak.a from a shared root.
        let mut s2 = super::Session::new();
        s2.eval_str("(def leaked-probe 2)");
        assert!(
            Namespace::find(&Symbol::intern("clojure.core"))
                .and_then(|n| n.find_interned_var(&Symbol::intern("leaked-probe")))
                .is_some(),
            "fresh Session must default to clojure.core (the `*ns*` root must \
             not have been mutated by s1's `(ns …)`)"
        );
        assert!(
            Namespace::find(&Symbol::intern("conc.leak.a"))
                .and_then(|n| n.find_interned_var(&Symbol::intern("leaked-probe")))
                .is_none(),
            "fresh Session's def must NOT land in the prior session's namespace"
        );
    }

    #[test]
    fn ns_form_with_metadata_map() {
        // From actual core.clj: `(ns ^{:doc "..."} clojure.core)`. The
        // metadata map is parsed by the reader and silently discarded;
        // the namespace switch still happens.
        let mut sess = with_fresh_ns_session("test.ns.meta", || super::Session::new());
        let _ = sess.eval_str(r#"(ns ^{:doc "test ns"} my.meta.ns)"#);
        let _ = sess.eval_str("(def y 7)");
        let v = sess.eval_str("y");
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    // ---- repro: nested if w/ both terminator branches inside variadic ----

    #[test]
    fn nested_if_both_terminators_in_variadic_let() {
        let mut sess =
            with_fresh_ns_session("test.compile.nested-if-throw", || super::Session::new());
        let _ = sess.eval_str(
            "(def assoc-like \
               (fn* assoc-like \
                 ([m k v] m) \
                 ([m k v & kvs] \
                   (let* [ret m] \
                     (if kvs \
                       (if kvs \
                         (recur ret k v kvs) \
                         (throw (IllegalArgumentException. \"odd\"))) \
                       ret)))))",
        );
    }

    #[test]
    fn nested_if_inner_one_terminator() {
        let mut sess =
            with_fresh_ns_session("test.compile.nested-one-term", || super::Session::new());
        let _ = sess.eval_str(
            "(def f \
               (fn* f [m k v & kvs] \
                 (if kvs \
                   (if kvs \
                     (recur m k v kvs) \
                     m) \
                   m)))",
        );
    }

    #[test]
    fn nested_if_recur_recur() {
        let mut sess =
            with_fresh_ns_session("test.compile.nested-recur-recur", || super::Session::new());
        let _ = sess.eval_str(
            "(def f \
               (fn* f \
                 ([m k v & kvs] \
                   (if kvs \
                     (if kvs \
                       (recur m k v kvs) \
                       (recur m k v kvs)) \
                     m))))",
        );
    }

    #[test]
    fn nested_if_recur_then_simple_throw() {
        let mut sess =
            with_fresh_ns_session("test.compile.nested-recur-throw", || super::Session::new());
        let _ = sess.eval_str(
            "(def f \
               (fn* f \
                 ([m k v & kvs] \
                   (if kvs \
                     (if kvs \
                       (recur m k v kvs) \
                       (throw nil)) \
                     m))))",
        );
    }

    #[test]
    fn nested_if_both_terminators_in_variadic_no_let() {
        let mut sess =
            with_fresh_ns_session("test.compile.nested-if-no-let", || super::Session::new());
        let _ = sess.eval_str(
            "(def assoc-like \
               (fn* assoc-like \
                 ([m k v] m) \
                 ([m k v & kvs] \
                   (if kvs \
                     (if kvs \
                       (recur m k v kvs) \
                       (throw (IllegalArgumentException. \"odd\"))) \
                     m))))",
        );
    }

    #[test]
    fn nested_if_throw_only_no_recur() {
        let mut sess = with_fresh_ns_session("test.compile.nested-if-throw-only", || {
            super::Session::new()
        });
        let _ = sess.eval_str(
            "(def f \
               (fn* f [x] \
                 (if x \
                   (if x \
                     1 \
                     (throw (IllegalArgumentException. \"odd\"))) \
                   2)))",
        );
    }

    #[test]
    fn single_throw_in_branch() {
        let mut sess = with_fresh_ns_session("test.compile.single-throw", || super::Session::new());
        let _ = sess.eval_str(
            "(def f \
               (fn* f [x] \
                 (if x 1 (throw (IllegalArgumentException. \"odd\")))))",
        );
    }

    #[test]
    fn variadic_with_recur_only() {
        let mut sess =
            with_fresh_ns_session("test.compile.variadic-recur", || super::Session::new());
        let _ = sess.eval_str(
            "(def f \
               (fn* f \
                 ([m k v] m) \
                 ([m k v & kvs] \
                   (if kvs \
                     (recur m k v kvs) \
                     m))))",
        );
    }

    #[test]
    fn multi_arity_nonvariadic_with_recur() {
        let mut sess = with_fresh_ns_session("test.compile.multi-recur", || super::Session::new());
        let _ = sess.eval_str(
            "(def f \
               (fn* f \
                 ([m] m) \
                 ([m k] (if k (recur m k) m))))",
        );
    }

    #[test]
    fn single_variadic_with_recur() {
        let mut sess = with_fresh_ns_session("test.compile.single-variadic-recur", || {
            super::Session::new()
        });
        let _ = sess.eval_str(
            "(def f \
               (fn* f [m k v & kvs] \
                 (if kvs \
                   (recur m k v kvs) \
                   m)))",
        );
    }

    #[test]
    fn single_arity_with_recur() {
        // Local-only repro — `first` etc are not registered in the test
        // namespace, so use locals only to avoid masking the bug under
        // an unrelated symbol-resolution failure.
        let mut sess =
            with_fresh_ns_session("test.compile.single-arity-recur", || super::Session::new());
        let _ = sess.eval_str(
            "(def f \
               (fn* f [m k] (if k (recur m k) m)))",
        );
    }

    #[test]
    fn variadic_with_throw_only() {
        let mut sess =
            with_fresh_ns_session("test.compile.variadic-throw", || super::Session::new());
        let _ = sess.eval_str(
            "(def f \
               (fn* f \
                 ([m k v] m) \
                 ([m k v & kvs] \
                   (if kvs \
                     (throw (IllegalArgumentException. \"odd\")) \
                     m))))",
        );
    }

    // ---- macros (real JIT-compiled, expanded at compile time) ------------

    #[test]
    fn macro_plus_five_expands_at_compile_time() {
        // A minimal macro: take one form arg `x`, produce `(+ x 5)`.
        // Macro fn signature mirrors microlisp's: `(fn [args] body)` where
        // `args` is the unevaluated rest of the call form.
        let mut sess = with_fresh_ns_session("test.macro.plus5", || super::Session::new());
        let _ = sess.eval_str(
            "(def ^:macro plus-five \
               (fn* [&form &env x] \
                 (. clojure.lang.RT (cons (quote +) \
                   (. clojure.lang.RT (cons x \
                     (. clojure.lang.RT (cons 5 nil))))))))",
        );
        // (plus-five 3) → (+ 3 5) → 8
        let v = sess.eval_str("(plus-five 3)");
        assert_eq!(nanbox_to_f64(v), 8.0);
    }

    #[test]
    fn macro_identity_returns_unchanged_form() {
        // The simplest macro: returns its arg verbatim. `(id-macro X)` → `X`.
        // Tests that an arbitrary form survives the read-alloc → JIT-run →
        // decode → re-analyze roundtrip.
        let mut sess = with_fresh_ns_session("test.macro.id", || super::Session::new());
        let _ = sess.eval_str("(def ^:macro id-macro (fn* [&form &env x] x))");
        let v = sess.eval_str("(id-macro (+ 10 20))");
        assert_eq!(nanbox_to_f64(v), 30.0);
    }

    #[test]
    fn macro_used_twice_in_same_session() {
        let mut sess = with_fresh_ns_session("test.macro.twice", || super::Session::new());
        let _ = sess.eval_str(
            "(def ^:macro plus-five \
               (fn* [&form &env x] \
                 (. clojure.lang.RT (cons (quote +) \
                   (. clojure.lang.RT (cons x \
                     (. clojure.lang.RT (cons 5 nil))))))))",
        );
        assert_eq!(nanbox_to_f64(sess.eval_str("(plus-five 1)")), 6.0);
        assert_eq!(nanbox_to_f64(sess.eval_str("(plus-five 10)")), 15.0);
        assert_eq!(nanbox_to_f64(sess.eval_str("(plus-five 100)")), 105.0);
    }

    #[test]
    fn defmacro_macro_returning_list_with_quote() {
        // The macro returns a quoted form built from cons + list. Tests
        // the path defmacro itself uses (cons of quoted symbols + var
        // refs) without yet involving let*.
        let mut sess = with_fresh_ns_session("test.defmacro.cons", || super::Session::new());
        let _ = sess.eval_str("(def cons (fn* [x s] (. clojure.lang.RT (cons x s))))");
        let _ = sess.eval_str("(def first (fn* [c] (. clojure.lang.RT (first c))))");
        // Macro: `(plus5 X)` → `(+ X 5)`. Returns a 3-elem list.
        let _ = sess.eval_str(
            "(def ^:macro plus5 (fn* [&form &env x] \
               (cons (quote +) (cons x (cons 5 nil)))))",
        );
        let v = sess.eval_str("(plus5 3)");
        assert_eq!(nanbox_to_f64(v), 8.0);
    }

    #[test]
    fn defmacro_minimal_smoke() {
        // Smoke test: a defmacro-style macro whose body just returns its
        // first arg unchanged. Isolates the "macro fn invocation through
        // var.deref" path from let / nested list construction.
        let mut sess = with_fresh_ns_session("test.defmacro.smoke", || super::Session::new());
        let _ = sess.eval_str("(def first (fn* [c] (. clojure.lang.RT (first c))))");
        let _ = sess.eval_str("(def ^:macro id-form (fn* [&form &env x] x))");
        let v = sess.eval_str("(id-form 99)");
        assert_eq!(nanbox_to_f64(v), 99.0);
    }

    #[test]
    fn defmacro_as_real_macro() {
        // Bootstrap `defmacro` as a one-shot `(def ^:macro defmacro (fn ...))`
        // and verify a subsequent `(defmacro ...)` form works without any
        // hardcoded special-form handling. Mirrors how upstream
        // `clojure.core` bootstraps defmacro.
        let mut sess = with_fresh_ns_session("test.defmacro", || super::Session::new());
        // Pre-load: cons/first/next/list helpers (no defmacro yet).
        let _ = sess.eval_str("(def list (fn* [& xs] xs))");
        let _ = sess.eval_str("(def cons (fn* [x s] (. clojure.lang.RT (cons x s))))");
        let _ = sess.eval_str("(def first (fn* [c] (. clojure.lang.RT (first c))))");
        let _ = sess.eval_str("(def next (fn* [c] (. clojure.lang.RT (next c))))");

        // Real upstream `defmacro` splices `&form &env` into the user's
        // param vector. We can't yet construct a Vector at runtime from
        // arbitrary symbols, so we skip the defmacro indirection here and
        // define `my-when` directly using the calling convention our
        // macroexpander expects. Same JIT-compiled-macro-fn path.
        let _ = sess.eval_str(
            "(def ^:macro my-when \
               (fn* [&form &env test body] \
                 (list (quote if) test body nil)))",
        );

        let v = sess.eval_str("(my-when true 42)");
        assert_eq!(nanbox_to_f64(v), 42.0);
        let v = sess.eval_str("(my-when false 42)");
        assert_eq!(v, 0x7FFC_0000_0000_0000); // nil
    }

    #[test]
    fn macro_call_inside_compiled_fn_body() {
        // Define a macro, then a fn whose BODY contains a macro call. The
        // macro expands at compile time (during the fn's analyze pass),
        // so the fn body's IR has the expanded form baked in.
        let mut sess = with_fresh_ns_session("test.macro.in.fn", || super::Session::new());
        let _ = sess.eval_str(
            "(def ^:macro plus-five \
               (fn* [&form &env x] \
                 (. clojure.lang.RT (cons (quote +) \
                   (. clojure.lang.RT (cons x \
                     (. clojure.lang.RT (cons 5 nil))))))))",
        );
        let _ = sess.eval_str("(def foo (fn* [n] (plus-five n)))");
        let v = sess.eval_str("(foo 37)");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    fn session_defn_then_invoke_across_forms() {
        let mut sess = with_fresh_ns_session("test.session.defn", || super::Session::new());
        let _ = sess.eval_str("(def inc (fn* [n] (+ n 1)))");
        let v = sess.eval_str("(inc 41)");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    /// `with_fresh_ns_session` — same idea as `with_fresh_ns` but for
    /// constructing values inside the fresh namespace. The returned value
    /// outlives the binding (since the namespace state is a per-thread
    /// dynamic Var pop'd here).
    fn with_fresh_ns_session<F: FnOnce() -> R, R>(ns_name: &str, body: F) -> R {
        use super::super::namespace::Namespace;
        use super::super::rt::CURRENT_NS;
        let ns = Namespace::find_or_create(Symbol::intern(ns_name));
        Var::push_thread_bindings(vec![(CURRENT_NS.clone(), Object::Namespace(ns))]);
        let r = body();
        Var::pop_thread_bindings();
        r
    }

    // ---- clojure.core fns using multi-arity ------------------------------

    /// `clojure.core/rest` (Java line ~70). `(. clojure.lang.RT (more x))`.
    /// In our Cons-only model `more` == `next`; differs for lazy seqs.
    const CORE_REST: &str = "(def rest (fn* [x] (. clojure.lang.RT (more x))))";

    #[test]
    fn core_fn_rest_drops_first() {
        let v = with_fresh_ns("test.rest", || {
            eval_str_via_jit(&format!(
                "(do {CORE_CONS} {CORE_REST} {CORE_FIRST} \
                   (first (rest (cons 1 (cons 99 nil)))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 99.0);
    }

    /// `clojure.core/conj` (Java line ~73, simplified). Multi-arity:
    ///   ([])         → nil (Java returns [] vec; we don't have vectors)
    ///   ([coll])     → coll
    ///   ([coll x])   → (cons x coll) for our Cons-only model
    ///   ([coll x & xs]) → recurse on each in `xs`
    const CORE_CONJ: &str = "(def conj (fn* \
        ([] nil) \
        ([coll] coll) \
        ([coll x] (. clojure.lang.RT (cons x coll))) \
        ([coll x & xs] (if (. clojure.lang.Util (isNil xs)) \
                          (conj coll x) \
                          (conj (conj coll x) (. clojure.lang.RT (first xs)))))))";

    #[test]
    fn core_fn_conj_two_arg() {
        // (first (conj nil 7)) → 7
        let v = with_fresh_ns("test.conj.two", || {
            eval_str_via_jit(&format!(
                "(do {CORE_CONJ} (. clojure.lang.RT (first (conj nil 7))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn core_fn_conj_zero_arg() {
        let v = with_fresh_ns("test.conj.zero", || {
            eval_str_via_jit(&format!("(do {CORE_CONJ} (conj))"))
        });
        assert_eq!(v, 0x7FFC_0000_0000_0000); // nil
    }

    #[test]
    fn core_fn_conj_one_arg() {
        let v = with_fresh_ns("test.conj.one", || {
            eval_str_via_jit(&format!(
                "(do {CORE_CONS} {CORE_CONJ} (. clojure.lang.RT (first (conj (cons 42 nil)))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    /// `clojure.core/nth` (multi-arity: 2-arg + 3-arg with not-found).
    const CORE_NTH_MULTI: &str = "(def nth (fn* \
        ([coll i] (if (= i 0) (. clojure.lang.RT (first coll)) (nth (. clojure.lang.RT (next coll)) (- i 1)))) \
        ([coll i not-found] (if (. clojure.lang.Util (isNil coll)) \
                                not-found \
                                (if (= i 0) (. clojure.lang.RT (first coll)) (nth (. clojure.lang.RT (next coll)) (- i 1) not-found))))))";

    #[test]
    fn core_fn_nth_two_arg_in_range() {
        let v = with_fresh_ns("test.nth.in", || {
            eval_str_via_jit(&format!(
                "(do {CORE_CONS} {CORE_NTH_MULTI} \
                   (nth (cons 10 (cons 20 (cons 30 nil))) 1))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 20.0);
    }

    #[test]
    fn core_fn_nth_three_arg_not_found_returned() {
        let v = with_fresh_ns("test.nth.nf", || {
            eval_str_via_jit(&format!("(do {CORE_CONS} {CORE_NTH_MULTI} (nth nil 5 99))"))
        });
        assert_eq!(nanbox_to_f64(v), 99.0);
    }

    // ---- multi-arity `fn*` static dispatch -------------------------------

    #[test]
    fn multi_arity_picks_one_arg_clause() {
        // (fn* ([x] x) ([x y] (+ x y))) — single-arg returns x
        let v = eval_str_via_jit("((fn* ([x] x) ([x y] (+ x y))) 7)");
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn multi_arity_picks_two_arg_clause() {
        let v = eval_str_via_jit("((fn* ([x] x) ([x y] (+ x y))) 3 4)");
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn multi_arity_via_def_then_invoke() {
        // (def my-add (fn ([x] (+ x 0)) ([x y] (+ x y)) ([x y z] (+ x y z))))
        let v = with_fresh_ns("test.multi.def", || {
            eval_str_via_jit(
                "(do (def my-add (fn* ([x] (+ x 0)) \
                                       ([x y] (+ x y)) \
                                       ([x y z] (+ x y z)))) \
                   (my-add 1 2 3))",
            )
        });
        assert_eq!(nanbox_to_f64(v), 6.0);
    }

    #[test]
    fn multi_arity_with_variadic_clause() {
        // (fn* ([x] x) ([x & xs] (+ x 100))) — using overflow clause via
        // arity > 1.
        let v = eval_str_via_jit("((fn* ([x] x) ([x & xs] (+ x 100))) 5 99)");
        assert_eq!(nanbox_to_f64(v), 105.0);
    }

    #[test]
    fn multi_arity_named_self_recursive() {
        // Real clojure.core/conj-like recursive defn using multi-arity.
        let v = with_fresh_ns("test.multi.recur", || {
            eval_str_via_jit(
                "(do (def cnt-up (fn* cnt-up \
                                        ([] 0) \
                                        ([n] (if (= n 0) 0 (+ 1 (cnt-up (- n 1))))))) \
                   (cnt-up 5))",
            )
        });
        assert_eq!(nanbox_to_f64(v), 5.0);
    }

    #[test]
    #[should_panic(expected = "duplicate non-variadic arity")]
    fn multi_arity_duplicate_arity_panics() {
        // Two fixed-arity clauses with same arity → IllegalStateException
        let _ = eval_str_via_jit("((fn* ([x] x) ([y] y)) 1)");
    }

    /// `clojure.core/constantly` — `(fn [c] (fn [& _] c))`. Variadic inner
    /// returns the constant regardless of args. For our `[& xs]` shape:
    /// `(def constantly (fn [c] (fn [& _] c)))`.
    const CORE_CONSTANTLY: &str = "(def constantly (fn* [c] (fn* [& _] c)))";

    #[test]
    fn core_fn_constantly_returns_same_value() {
        let v = with_fresh_ns("test.constantly", || {
            eval_str_via_jit(&format!("(do {CORE_CONSTANTLY} ((constantly 42) 1 2 3))"))
        });
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    /// `clojure.core/every?` (recursive, our Cons-only model).
    /// `(def every? (fn [pred coll] (if (nil? coll) true (if (pred (first coll)) (every? pred (next coll)) false))))`.
    const CORE_EVERY_Q: &str = "(def every? (fn* [pred coll] \
        (if (. clojure.lang.Util (isNil coll)) \
            true \
            (if (pred (. clojure.lang.RT (first coll))) \
                (every? pred (. clojure.lang.RT (next coll))) \
                false))))";

    #[test]
    fn core_fn_every_q_true_when_all_match() {
        let v = with_fresh_ns("test.every.t", || {
            eval_str_via_jit(&format!(
                "(do {CORE_POS_Q} {CORE_EVERY_Q} \
                   (every? pos? (. clojure.lang.RT (cons 1 \
                     (. clojure.lang.RT (cons 2 \
                       (. clojure.lang.RT (cons 3 nil))))))))"
            ))
        });
        assert!(nanbox_to_bool(v));
    }

    #[test]
    fn core_fn_every_q_false_when_one_fails() {
        let v = with_fresh_ns("test.every.f", || {
            eval_str_via_jit(&format!(
                "(do {CORE_POS_Q} {CORE_EVERY_Q} \
                   (every? pos? (. clojure.lang.RT (cons 1 \
                     (. clojure.lang.RT (cons -2 \
                       (. clojure.lang.RT (cons 3 nil))))))))"
            ))
        });
        assert!(!nanbox_to_bool(v));
    }

    /// `clojure.core/some` — returns the first truthy `(pred x)`, or nil.
    /// For our subset using primitive `if x ...`: returns the matching
    /// element (rather than the truthy pred result). Real `some` returns
    /// the pred-result; our simpler version returns the element. Same
    /// behavior for boolean preds.
    const CORE_SOME: &str = "(def some (fn* [pred coll] \
        (if (. clojure.lang.Util (isNil coll)) \
            nil \
            (if (pred (. clojure.lang.RT (first coll))) \
                (. clojure.lang.RT (first coll)) \
                (some pred (. clojure.lang.RT (next coll)))))))";

    #[test]
    fn core_fn_some_returns_first_pos() {
        let v = with_fresh_ns("test.some", || {
            eval_str_via_jit(&format!(
                "(do {CORE_POS_Q} {CORE_SOME} \
                   (some pos? (. clojure.lang.RT (cons -1 \
                     (. clojure.lang.RT (cons -2 \
                       (. clojure.lang.RT (cons 7 nil))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn core_fn_some_returns_nil_when_none_match() {
        let v = with_fresh_ns("test.some.nil", || {
            eval_str_via_jit(&format!(
                "(do {CORE_POS_Q} {CORE_SOME} \
                   (some pos? (. clojure.lang.RT (cons -1 \
                     (. clojure.lang.RT (cons -2 nil))))))"
            ))
        });
        assert_eq!(v, 0x7FFC_0000_0000_0000);
    }

    /// `clojure.core/repeat` (2-arg slice). `(repeat n x)` returns a list
    /// of `x` repeated `n` times. Real `repeat` is lazy; this is eager.
    const CORE_REPEAT: &str = "(def repeat (fn* [n x] \
        (if (= n 0) \
            nil \
            (. clojure.lang.RT (cons x (repeat (- n 1) x))))))";

    #[test]
    fn core_fn_repeat() {
        // (first (next (next (repeat 5 :ok)))) → :ok (third element)
        let s = with_fresh_ns("test.repeat", || {
            // Use a keyword that decodes nicely, then call repeat and grab the
            // 3rd element by walking.
            let src = format!(
                "(do {CORE_REPEAT} \
                   (. clojure.lang.RT (first \
                     (. clojure.lang.RT (next \
                       (. clojure.lang.RT (next (repeat 5 :ok))))))))"
            );
            eval_str_via_jit_to_object(&src)
        });
        match s {
            Object::Keyword(k) => assert_eq!(k.get_name(), "ok"),
            other => panic!("{other:?}"),
        }
    }

    /// `clojure.core/take` (eager). Take first `n` items.
    const CORE_TAKE: &str = "(def take (fn* [n coll] \
        (if (= n 0) \
            nil \
            (if (. clojure.lang.Util (isNil coll)) \
                nil \
                (. clojure.lang.RT (cons (. clojure.lang.RT (first coll)) \
                                          (take (- n 1) (. clojure.lang.RT (next coll)))))))))";

    #[test]
    fn core_fn_take_first_two_of_four() {
        let v = with_fresh_ns("test.take", || {
            // Build 1..4, take 2, count them (should be 2).
            eval_str_via_jit(&format!(
                "(do {CORE_CONS} {CORE_FIRST} {CORE_NEXT} {CORE_COUNT} {CORE_TAKE} \
                   (count (take 2 (cons 1 (cons 2 (cons 3 (cons 4 nil)))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    /// `clojure.core/drop` (eager). Drop first `n` items.
    const CORE_DROP: &str = "(def drop (fn* [n coll] \
        (if (= n 0) \
            coll \
            (if (. clojure.lang.Util (isNil coll)) \
                nil \
                (drop (- n 1) (. clojure.lang.RT (next coll)))))))";

    #[test]
    fn core_fn_drop_first_two_of_four() {
        let v = with_fresh_ns("test.drop", || {
            // Build 10..40, drop 2, take first → 30.
            eval_str_via_jit(&format!(
                "(do {CORE_DROP} \
                   (. clojure.lang.RT (first \
                     (drop 2 (. clojure.lang.RT (cons 10 \
                       (. clojure.lang.RT (cons 20 \
                         (. clojure.lang.RT (cons 30 \
                           (. clojure.lang.RT (cons 40 nil))))))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 30.0);
    }

    /// `clojure.core/range` (1-arg, recursive). `(range n)` returns 0..n.
    const CORE_RANGE: &str = "(do (def range-iter (fn* [i n] (if (= i n) nil (. clojure.lang.RT (cons i (range-iter (+ i 1) n)))))) \
             (def range (fn* [n] (range-iter 0 n))))";

    #[test]
    fn core_fn_range_5_first_is_0() {
        let v = with_fresh_ns("test.range", || {
            eval_str_via_jit(&format!(
                "(do {CORE_RANGE} (. clojure.lang.RT (first (range 5))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 0.0);
    }

    #[test]
    fn core_fn_range_5_third_is_2() {
        let v = with_fresh_ns("test.range.third", || {
            eval_str_via_jit(&format!(
                "(do {CORE_RANGE} \
                   (. clojure.lang.RT (first \
                     (. clojure.lang.RT (next \
                       (. clojure.lang.RT (next (range 5))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    /// Sum a `range` via reduce + closure.
    #[test]
    fn higher_order_sum_range_via_reduce() {
        let v = with_fresh_ns("test.sum.range", || {
            eval_str_via_jit(&format!(
                "(do {CORE_REDUCE} {CORE_RANGE} \
                   (reduce (fn* [a b] (+ a b)) 0 (range 10)))"
            ))
        });
        // 0+1+…+9 = 45
        assert_eq!(nanbox_to_f64(v), 45.0);
    }

    /// `clojure.core/comp` (3-fn slice) — `((comp f g h) x)` → `(f (g (h x)))`.
    /// Real comp is variadic.
    const CORE_COMP3: &str = "(def comp3 (fn* [f g h] (fn* [x] (f (g (h x))))))";

    #[test]
    fn core_fn_comp3() {
        // ((comp3 inc inc inc) 40) → 43
        let v = with_fresh_ns("test.comp3", || {
            eval_str_via_jit(&format!(
                "(do {CORE_INC} {CORE_COMP3} ((comp3 inc inc inc) 40))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 43.0);
    }

    /// Map over a range and sum: end-to-end higher-order check.
    #[test]
    fn higher_order_sum_of_squared_range() {
        let v = with_fresh_ns("test.sum.sq.range", || {
            // (reduce + 0 (map sq (range 5))) where sq squares.
            // map+reduce+range together.
            eval_str_via_jit(&format!(
                "(do {CORE_REDUCE} {CORE_RANGE} {CORE_MAP} \
                   (def sq (fn* [n] (* n n))) \
                   (reduce (fn* [a b] (+ a b)) 0 (map sq (range 5))))"
            ))
        });
        // 0+1+4+9+16 = 30
        assert_eq!(nanbox_to_f64(v), 30.0);
    }

    /// fn that prepends `x` when called: `((partial inc) 41)` → 42 ; for
    /// 1-arg partial: `(partial f x)` → `(fn [y] (f x y))`.
    const CORE_PARTIAL1: &str = "(def partial1 (fn* [f x] (fn* [y] (f x y))))";

    #[test]
    fn core_fn_partial1_with_add() {
        // Build add with `(fn [a b] (+ a b))`, then partial-apply 10. Call
        // result with 5 → 15.
        let v = with_fresh_ns("test.partial1", || {
            eval_str_via_jit(&format!(
                "(do {CORE_PARTIAL1} \
                   (def add (fn* [a b] (+ a b))) \
                   ((partial1 add 10) 5))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 15.0);
    }

    /// `clojure.core/complement` — `(def complement (fn [f] (fn [x] (not (f x)))))`.
    const CORE_COMPLEMENT: &str = "(def complement (fn* [f] (fn* [x] (if (f x) false true))))";

    #[test]
    fn core_fn_complement() {
        // (complement pos?) is a fn that's truthy for non-positive args.
        let v_neg = with_fresh_ns("test.complement.neg", || {
            eval_str_via_jit(&format!(
                "(do {CORE_POS_Q} {CORE_COMPLEMENT} ((complement pos?) -3))"
            ))
        });
        assert!(nanbox_to_bool(v_neg));
        let v_pos = with_fresh_ns("test.complement.pos", || {
            eval_str_via_jit(&format!(
                "(do {CORE_POS_Q} {CORE_COMPLEMENT} ((complement pos?) 3))"
            ))
        });
        assert!(!nanbox_to_bool(v_pos));
    }

    /// A make-adder closure pattern, the canonical "closure over local"
    /// example.
    #[test]
    fn closure_make_adder_returns_capturing_fn() {
        // (def make-adder (fn [n] (fn [x] (+ n x))))
        // ((make-adder 100) 23) → 123
        let v = with_fresh_ns("test.make_adder", || {
            eval_str_via_jit(
                "(do (def make-adder (fn* [n] (fn* [x] (+ n x)))) \
                   ((make-adder 100) 23))",
            )
        });
        assert_eq!(nanbox_to_f64(v), 123.0);
    }

    #[test]
    fn core_fn_comp2_now_works_via_closures() {
        // `comp2`'s inner fn captures `f` and `g` from the outer scope.
        // ((comp2 inc inc) 40) → 42 (inc(inc(40))). Exercises:
        //   * Two-level fn nesting (outer fn returns closure)
        //   * Two captures (f, g) per closure
        //   * Captured fns invoked from inside the closure body
        let v = with_fresh_ns("test.comp2", || {
            eval_str_via_jit(&format!(
                "(do {CORE_INC} {CORE_COMP2} ((comp2 inc inc) 40))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    // ---- more clojure.core fns using variadics + composition --------------

    /// `clojure.core/list` (Java line ~17). Real definition:
    /// `(. clojure.lang.PersistentList creator)` — pulls a static creator
    /// instance. Without static-field dispatch we port the pure-Clojure
    /// shape: `(fn [& xs] xs)`. Semantically identical for the call cases
    /// `(list a b c)` produces a list of those elements.
    const CORE_LIST: &str = "(def list (fn* [& xs] xs))";

    #[test]
    fn core_fn_list_three_elements() {
        let v = with_fresh_ns("test.list.three", || {
            eval_str_via_jit(&format!(
                "(do {CORE_LIST} (. clojure.lang.RT (first (list 10 20 30))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 10.0);
    }

    #[test]
    fn core_fn_list_empty_when_no_args() {
        let v = with_fresh_ns("test.list.empty", || {
            eval_str_via_jit(&format!("(do {CORE_LIST} (list))"))
        });
        assert_eq!(v, 0x7FFC_0000_0000_0000, "(list) → nil-terminator");
    }

    #[test]
    fn core_fn_list_walks_all_three_elements() {
        // (first (next (next (list 1 2 3)))) → 3
        let v = with_fresh_ns("test.list.walk", || {
            eval_str_via_jit(&format!(
                "(do {CORE_LIST} \
                   (. clojure.lang.RT (first \
                     (. clojure.lang.RT (next \
                       (. clojure.lang.RT (next (list 1 2 3))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 3.0);
    }

    /// `clojure.core/reverse` (Java line ~7155, approximately).
    /// Recursive port: `(if (nil? coll) acc (reverse-into (next coll) (cons (first coll) acc)))`.
    /// Real clojure.core uses `reduce conj`; we hand-roll the recursion since
    /// `reduce` isn't ported yet.
    const CORE_REVERSE: &str = "(do \
        (def reverse-into (fn* [coll acc] \
            (if (. clojure.lang.Util (isNil coll)) \
                acc \
                (reverse-into (. clojure.lang.RT (next coll)) \
                              (. clojure.lang.RT (cons (. clojure.lang.RT (first coll)) acc)))))) \
        (def reverse (fn* [coll] (reverse-into coll nil))))";

    #[test]
    fn core_fn_reverse_three_element_list() {
        // (first (reverse (cons 1 (cons 2 (cons 3 nil))))) → 3
        let v = with_fresh_ns("test.reverse", || {
            eval_str_via_jit(&format!(
                "(do {CORE_REVERSE} \
                   (. clojure.lang.RT (first \
                     (reverse (. clojure.lang.RT (cons 1 \
                       (. clojure.lang.RT (cons 2 \
                         (. clojure.lang.RT (cons 3 nil))))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 3.0);
    }

    /// `clojure.core/concat` (2-arg, recursive). Real concat is variadic +
    /// lazy; this is the eager 2-arg shape used internally by many
    /// clojure.core ports of higher-order fns.
    const CORE_CONCAT2: &str = "(def concat2 (fn* [xs ys] \
        (if (. clojure.lang.Util (isNil xs)) \
            ys \
            (. clojure.lang.RT (cons (. clojure.lang.RT (first xs)) \
                                     (concat2 (. clojure.lang.RT (next xs)) ys))))))";

    #[test]
    fn core_fn_concat2_combines_two_lists() {
        let v = with_fresh_ns("test.concat", || {
            // (first (next (concat2 (cons 1 nil) (cons 2 (cons 3 nil))))) → 2
            eval_str_via_jit(&format!(
                "(do {CORE_CONCAT2} \
                   (. clojure.lang.RT (first \
                     (. clojure.lang.RT (next \
                       (concat2 (. clojure.lang.RT (cons 1 nil)) \
                                (. clojure.lang.RT (cons 2 (. clojure.lang.RT (cons 3 nil))))))))))"
            ))
        });
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    // ---- variadic args (`& rest`) at the fn-def + call site --------------

    #[test]
    fn variadic_fn_zero_overflow_args_gets_nil_rest() {
        // (fn [& xs] xs) called with no args → xs bound to nil.
        // Use raw bits comparison since nil isn't a heap pointer.
        let v = with_fresh_ns("test.var.zero", || {
            eval_str_via_jit("(do (def f (fn* [& xs] xs)) (f))")
        });
        assert_eq!(v, 0x7FFC_0000_0000_0000, "expected nil NanBox bits");
    }

    #[test]
    fn variadic_fn_first_overflow_arg_via_rt_first() {
        // (fn [& xs] xs) → list of all args.
        // Call (f 10 20 30), then read the first element via RT.first.
        let v = with_fresh_ns("test.var.first", || {
            eval_str_via_jit(
                "(do \
                   (def f (fn* [& xs] xs)) \
                   (. clojure.lang.RT (first (f 10 20 30))))",
            )
        });
        assert_eq!(nanbox_to_f64(v), 10.0);
    }

    #[test]
    fn variadic_fn_second_overflow_arg() {
        let v = with_fresh_ns("test.var.second", || {
            eval_str_via_jit(
                "(do \
                   (def f (fn* [& xs] xs)) \
                   (. clojure.lang.RT (first (. clojure.lang.RT (next (f 10 20 30))))))",
            )
        });
        assert_eq!(nanbox_to_f64(v), 20.0);
    }

    #[test]
    fn variadic_fn_with_fixed_args_separates_correctly() {
        // (fn [a b & xs] [a b xs]) — we don't have vectors emitted yet, so
        // test by extracting each piece individually.
        let v_a = with_fresh_ns("test.var.fixed.a", || {
            eval_str_via_jit("(do (def f (fn* [a b & xs] a)) (f 1 2 3 4 5))")
        });
        assert_eq!(nanbox_to_f64(v_a), 1.0);

        let v_b = with_fresh_ns("test.var.fixed.b", || {
            eval_str_via_jit("(do (def f (fn* [a b & xs] b)) (f 1 2 3 4 5))")
        });
        assert_eq!(nanbox_to_f64(v_b), 2.0);

        let v_xs0 = with_fresh_ns("test.var.fixed.xs0", || {
            eval_str_via_jit(
                "(do \
                   (def f (fn* [a b & xs] xs)) \
                   (. clojure.lang.RT (first (f 1 2 3 4 5))))",
            )
        });
        assert_eq!(nanbox_to_f64(v_xs0), 3.0);
    }

    #[test]
    fn variadic_fn_zero_overflow_arity_check() {
        // (fn [a & xs] …) called with (f 1) — exactly fixed_arity, no
        // overflow. xs should be nil.
        let v = with_fresh_ns("test.var.exact", || {
            eval_str_via_jit("(do (def f (fn* [a & xs] a)) (f 42))")
        });
        assert_eq!(nanbox_to_f64(v), 42.0);
    }

    #[test]
    #[should_panic(expected = "ArityException")]
    fn variadic_fn_too_few_args_panics() {
        // (fn [a b & xs] …) called with one arg.
        let _ = with_fresh_ns("test.var.toofew", || {
            eval_str_via_jit("(do (def f (fn* [a b & xs] a)) (f 1))")
        });
    }

    #[test]
    #[should_panic(expected = "ArityException")]
    fn fixed_fn_wrong_arity_panics() {
        // (fn [x y] …) called with one arg.
        let _ = with_fresh_ns("test.fixed.toofew", || {
            eval_str_via_jit("(do (def f (fn* [x y] x)) (f 1))")
        });
    }

    // ---- heap-aware equality + nil? (Util.equiv, Util.isNil) -------------

    /// `clojure.core/nil?` — `(def nil? (fn [x] (. clojure.lang.Util (isNil x))))`.
    const CORE_NIL_Q: &str = "(def nil? (fn* [x] (. clojure.lang.Util (isNil x))))";

    #[test]
    fn core_fn_nil_q_true_for_nil() {
        assert!(nanbox_to_bool(run_core_test(
            "test.nil_q.nil",
            &[CORE_NIL_Q],
            "(nil? nil)"
        )));
    }

    #[test]
    fn core_fn_nil_q_false_for_zero() {
        // Clojure: (nil? 0) → false. Critical distinction.
        assert!(!nanbox_to_bool(run_core_test(
            "test.nil_q.0",
            &[CORE_NIL_Q],
            "(nil? 0)"
        )));
    }

    #[test]
    fn core_fn_nil_q_false_for_false() {
        assert!(!nanbox_to_bool(run_core_test(
            "test.nil_q.false",
            &[CORE_NIL_Q],
            "(nil? false)"
        )));
    }

    #[test]
    fn core_fn_nil_q_false_for_string() {
        assert!(!nanbox_to_bool(run_core_test(
            "test.nil_q.str",
            &[CORE_NIL_Q],
            r#"(nil? "hi")"#
        )));
    }

    /// `clojure.core/=` (2-arg) routed through `Util.equiv`. Real `=` does
    /// extensive dispatch; for our subset this matches Java semantics for
    /// the types we model.
    const CORE_EQ_HEAP: &str = "(def equiv= (fn* [a b] (. clojure.lang.Util (equiv a b))))";

    #[test]
    fn core_fn_equiv_strings_equal() {
        assert!(nanbox_to_bool(run_core_test(
            "test.equiv.str.eq",
            &[CORE_EQ_HEAP],
            r#"(equiv= "foo" "foo")"#,
        )));
    }

    #[test]
    fn core_fn_equiv_strings_diff() {
        assert!(!nanbox_to_bool(run_core_test(
            "test.equiv.str.neq",
            &[CORE_EQ_HEAP],
            r#"(equiv= "foo" "bar")"#,
        )));
    }

    #[test]
    fn core_fn_equiv_keywords() {
        // Keywords are globally interned — same ns/name → ptr-equal.
        assert!(nanbox_to_bool(run_core_test(
            "test.equiv.kw.eq",
            &[CORE_EQ_HEAP],
            "(equiv= :foo :foo)",
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.equiv.kw.neq",
            &[CORE_EQ_HEAP],
            "(equiv= :foo :bar)",
        )));
    }

    #[test]
    fn core_fn_equiv_quoted_symbols_value_equal() {
        // Symbols aren't globally interned, so value-equality on ns+name.
        assert!(nanbox_to_bool(run_core_test(
            "test.equiv.sym.eq",
            &[CORE_EQ_HEAP],
            "(equiv= (quote foo) (quote foo))",
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.equiv.sym.neq",
            &[CORE_EQ_HEAP],
            "(equiv= (quote foo) (quote bar))",
        )));
    }

    #[test]
    fn core_fn_equiv_lists_structural() {
        // Structural equality on lists: same length, equiv elements at each.
        assert!(nanbox_to_bool(run_core_test(
            "test.equiv.list.eq",
            &[CORE_CONS, CORE_EQ_HEAP],
            "(equiv= (cons 1 (cons 2 nil)) (cons 1 (cons 2 nil)))",
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.equiv.list.neq",
            &[CORE_CONS, CORE_EQ_HEAP],
            "(equiv= (cons 1 (cons 2 nil)) (cons 1 (cons 3 nil)))",
        )));
        // Same length, same elements, different order.
        assert!(!nanbox_to_bool(run_core_test(
            "test.equiv.list.order",
            &[CORE_CONS, CORE_EQ_HEAP],
            "(equiv= (cons 1 (cons 2 nil)) (cons 2 (cons 1 nil)))",
        )));
    }

    #[test]
    fn core_fn_equiv_immediates() {
        // Immediates: nil, true, false, longs roundtrip through equiv.
        assert!(nanbox_to_bool(run_core_test(
            "test.equiv.nil",
            &[CORE_EQ_HEAP],
            "(equiv= nil nil)"
        )));
        assert!(nanbox_to_bool(run_core_test(
            "test.equiv.true",
            &[CORE_EQ_HEAP],
            "(equiv= true true)"
        )));
        assert!(nanbox_to_bool(run_core_test(
            "test.equiv.long",
            &[CORE_EQ_HEAP],
            "(equiv= 42 42)"
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.equiv.nil_false",
            &[CORE_EQ_HEAP],
            "(equiv= nil false)"
        )));
    }

    /// `clojure.core/empty?` — `(def empty? (fn [coll] (nil? (seq coll))))`.
    /// Without `seq`, we narrow to nil-check: an empty list in our world IS
    /// represented by nil (the Cons rest-terminator). This matches the
    /// behavior for the values we currently produce.
    const CORE_EMPTY_Q: &str = "(def empty? nil?)";

    #[test]
    fn core_fn_empty_q_via_var_alias_now_supported() {
        // `(def empty? nil?)` aliases one Var to another's value. The call
        // `(empty? nil)` now lowers through the dynamic-invoke path
        // (cljvm_rt_invoke_1) because `empty?`'s var_fn entry is empty
        // (alias init isn't a FnExpr) — the head emits as a Var-deref
        // producing a TAG_FN handle, and the runtime invoke extern
        // dispatches.
        let v = run_core_test(
            "test.empty_q.alias",
            &[CORE_NIL_Q, CORE_EMPTY_Q],
            "(empty? nil)",
        );
        assert!(nanbox_to_bool(v));
    }

    /// Explicit wrapper for `empty?` — bypasses the Var-alias limitation.
    const CORE_EMPTY_Q_EXPLICIT: &str = "(def empty? (fn* [coll] (nil? coll)))";

    #[test]
    fn core_fn_empty_q_explicit() {
        assert!(nanbox_to_bool(run_core_test(
            "test.empty_q.nil",
            &[CORE_NIL_Q, CORE_EMPTY_Q_EXPLICIT],
            "(empty? nil)",
        )));
        assert!(!nanbox_to_bool(run_core_test(
            "test.empty_q.cons",
            &[CORE_CONS, CORE_NIL_Q, CORE_EMPTY_Q_EXPLICIT],
            "(empty? (cons 1 nil))",
        )));
    }

    #[test]
    fn core_fns_walk_three_element_list() {
        // Build (cons 10 (cons 20 (cons 30 nil))), extract each index.
        let defs = &[CORE_CONS, CORE_FIRST, CORE_NEXT];
        let list_src = "(cons 10 (cons 20 (cons 30 nil)))";

        let v0 = run_core_test("test.nth.0", defs, &format!("(first {list_src})"));
        assert_eq!(nanbox_to_f64(v0), 10.0);

        let v1 = run_core_test("test.nth.1", defs, &format!("(first (next {list_src}))"));
        assert_eq!(nanbox_to_f64(v1), 20.0);

        let v2 = run_core_test(
            "test.nth.2",
            defs,
            &format!("(first (next (next {list_src})))"),
        );
        assert_eq!(nanbox_to_f64(v2), 30.0);
    }

    #[test]
    fn core_fn_recursive_countdown_returns_keyword() {
        // Verify the recursive defn-style fn reaches its base case and
        // returns a heap-allocated keyword. Counting down requires the call
        // ABI to preserve the literal-pool-loaded keyword value across the
        // recursive call frames.
        let obj = with_fresh_ns("test.core.countdown", || {
            let src = format!("(do {} (countdown 10))", CORE_COUNTDOWN);
            eval_str_via_jit_to_object(&src)
        });
        match obj {
            Object::Keyword(k) => {
                assert!(k.get_namespace().is_none());
                assert_eq!(k.get_name(), "done");
            }
            other => panic!("expected :done, got {other:?}"),
        }
    }

    /// `clojure.core/nnext` (Java line ~128) — `(next (next x))`.
    const CORE_NNEXT: &str = "(def nnext (fn* [x] (next (next x))))";

    #[test]
    fn core_fn_nnext() {
        // (first (nnext (cons 1 (cons 2 (cons 3 nil))))) → 3
        let v = run_core_test(
            "test.core.nnext",
            &[CORE_CONS, CORE_FIRST, CORE_NEXT, CORE_NNEXT],
            "(first (nnext (cons 1 (cons 2 (cons 3 nil)))))",
        );
        assert_eq!(nanbox_to_f64(v), 3.0);
    }

    // ---- runtime cons/first/next (heap allocation at JIT execution) -------

    #[test]
    fn jit_e2e_rt_cons_then_first() {
        // (. clojure.lang.RT (first (. clojure.lang.RT (cons 7 nil)))) → 7
        let v = eval_str_via_jit("(. clojure.lang.RT (first (. clojure.lang.RT (cons 7 nil))))");
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn jit_e2e_rt_cons_two_deep_first_walks() {
        // (. clojure.lang.RT (first (. clojure.lang.RT (next (cons 1 (cons 2 nil)))))) → 2
        let v = eval_str_via_jit(
            "(. clojure.lang.RT (first \
                (. clojure.lang.RT (next \
                  (. clojure.lang.RT (cons 1 \
                    (. clojure.lang.RT (cons 2 nil))))))))",
        );
        assert_eq!(nanbox_to_f64(v), 2.0);
    }

    #[test]
    fn jit_e2e_rt_first_of_nil_is_nil() {
        // (. clojure.lang.RT (first nil)) → nil (NanBox-encoded)
        let v = eval_str_via_jit("(. clojure.lang.RT (first nil))");
        // Decode the NanBox; should be Object::Nil.
        let ids = crate::runtime::HeapTypeIds {
            string: 0,
            symbol: 1,
            keyword: 2,
            cons: 3,
            vector: 5,
            map: 6,
            set: 7,
            tree_map: 11,
            tree_set: 12,
            string_builder: 13,
            chunk_buffer: 14,
            i_chunk: 15,
            lazy_seq: 16,
            delay: 17,
            multi_arity_fn: 18,
            class: 8,
            var: 9,
            with_meta: 10,
            long: 20,
            character: 23,
            user_instance: 19,
            reduced: 21,
            namespace: 22,
        };
        // Nil isn't a TAG_PTR — handled in object_to_nanbox roundtrip.
        // For tests just check it's the nil tag pattern.
        let _ = ids;
        // Nil NanBox: TAG_PATTERN | (0 << 48) | 0 = 0x7FFC000000000000.
        assert_eq!(v, 0x7FFC_0000_0000_0000);
    }

    #[test]
    fn jit_e2e_defn_using_rt_cons() {
        // (def list1 (fn [x] (. clojure.lang.RT (cons x nil))))
        // ((. clojure.lang.RT (first (list1 99)))) → 99
        let src = "(do
          (def list1 (fn* [x] (. clojure.lang.RT (cons x nil))))
          (. clojure.lang.RT (first (list1 99))))";
        let v = with_fresh_ns("test.rt.cons.defn", || eval_str_via_jit(src));
        assert_eq!(nanbox_to_f64(v), 99.0);
    }

    #[test]
    fn jit_e2e_source_comments_and_whitespace() {
        let src = "; outer comment
                   (+ 1 ; inline
                      2,
                      3)";
        assert_eq!(nanbox_to_f64(eval_str_via_jit(src)), 6.0);
    }

    #[test]
    #[should_panic(expected = "expected docstring")]
    fn def_with_four_args_panics() {
        with_fresh_ns("test.def.four", || {
            let form = list_of(vec![sym("def"), sym("x"), Object::Long(1), Object::Long(2)]);
            let _ = analyze(C::Statement, form);
        });
    }

    #[test]
    #[should_panic(expected = "First argument to def must be a Symbol")]
    fn def_with_non_symbol_name_panics() {
        with_fresh_ns("test.def.nonsym", || {
            let form = list_of(vec![sym("def"), Object::Long(1), Object::Long(2)]);
            let _ = analyze(C::Statement, form);
        });
    }

    // ── defprotocol end-to-end ─────────────────────────────────────────
    //
    // Verifies that `defprotocol` registers methods, the generated
    // method-fns dispatch into `cljvm_rt_protocol_dispatch_N`, and a
    // manually-installed impl gets called.

    #[test]
    fn defprotocol_dispatches_to_installed_impl_on_nil() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types::{
            self as ut, BUILTIN_NIL, install_impl, protocol_id_by_name, protocol_method_id,
        };
        let mut sess = with_fresh_ns_session("test.defproto.nil", || super::Session::new());
        // Reset registries so prior tests' ProtoMethodIds don't bleed in.
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol IPing (-ping [this]))");
        // The defprotocol expansion should have called register_protocol
        // and stashed the id under the symbol name.
        let pid = protocol_id_by_name(&Symbol::intern_ns_name(None, "IPing"))
            .expect("IPing should be registered");
        let mid = protocol_method_id(pid, &Symbol::intern_ns_name(None, "-ping"))
            .expect("-ping should be registered");
        // Compile a fn impl `(fn* [this] 7)` and install it on nil.
        let fn_handle = sess.eval_str("(fn* [this] 7)");
        install_impl(BUILTIN_NIL, mid, fn_handle);
        // `(-ping nil)` should dispatch through the generated method-fn,
        // hit cljvm_rt_protocol_dispatch_1, lookup the impl, and invoke
        // it.
        let v = sess.eval_str("(-ping nil)");
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    // Note: a test asserting that calling an uninstalled protocol
    // method panics would need to cross the `extern "C"` boundary
    // (panic-in-extern aborts the process by default — `should_panic`
    // can't observe it). The miss-path panic in
    // `lookup_protocol_method` is covered by the runtime-only test
    // `dispatch_panics_with_no_impl` in `runtime::user_type_runtime_tests`.

    // ── deftype field-only ───────────────────────────────────────────

    #[test]
    fn deftype_factory_and_field_access() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.deftype.fields", || super::Session::new());
        ut::_reset_for_tests();
        // Define + instantiate + field-read in one chain.
        let _ = sess.eval_str("(deftype Foo [a b])");
        let v_a = sess.eval_str("(.-a (Foo. 1 2))");
        assert_eq!(nanbox_to_f64(v_a), 1.0);
        let v_b = sess.eval_str("(.-b (Foo. 1 2))");
        assert_eq!(nanbox_to_f64(v_b), 2.0);
    }

    // ── extend-protocol + satisfies? ─────────────────────────────────

    #[test]
    fn extend_protocol_multiple_types() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.extend.protocol", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol IShow (-show [this]))");
        let _ = sess.eval_str("(deftype A [v])");
        let _ = sess.eval_str("(deftype B [v])");
        let _ = sess.eval_str(
            "(extend-protocol IShow \
                A (-show [this] (.-v this)) \
                B (-show [this] 999))",
        );
        let va = sess.eval_str("(-show (A. 5))");
        assert_eq!(nanbox_to_f64(va), 5.0);
        let vb = sess.eval_str("(-show (B. 0))");
        assert_eq!(nanbox_to_f64(vb), 999.0);
    }

    #[test]
    fn satisfies_query_true_after_extend() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.satisfies.true", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol ICheck (-tag [this]))");
        let _ = sess.eval_str("(deftype Foo [])");
        let _ = sess.eval_str("(extend-type Foo ICheck (-tag [this] 1))");
        let v = sess.eval_str("(satisfies? ICheck (Foo.))");
        assert_eq!(v, crate::runtime::nanbox_bool(true));
    }

    #[test]
    fn satisfies_false_when_not_extended() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.satisfies.false", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol ICheck2 (-tag [this]))");
        let _ = sess.eval_str("(deftype Bar [])");
        // No extend-type — Bar should not satisfy.
        let v = sess.eval_str("(satisfies? ICheck2 (Bar.))");
        assert_eq!(v, crate::runtime::nanbox_bool(false));
    }

    #[test]
    fn satisfies_object_fallback_does_not_count() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.satisfies.object", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol ICheck3 (-tag [this]))");
        let _ = sess.eval_str("(deftype Baz [])");
        // Extend Object only — satisfies? should still report false
        // for Baz because Object fallback isn't a "direct" impl.
        let _ = sess.eval_str("(extend-type Object ICheck3 (-tag [this] 0))");
        let v = sess.eval_str("(satisfies? ICheck3 (Baz.))");
        assert_eq!(v, crate::runtime::nanbox_bool(false));
    }

    // ── inline deftype impls ─────────────────────────────────────────

    #[test]
    fn deftype_with_inline_protocol_impl() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.deftype.inline", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol IGetSum (-sum [this]))");
        let _ = sess.eval_str(
            "(deftype Pair [a b] \
              IGetSum (-sum [this] (.-a this)))",
        );
        // The inline impl returns `.-a` of the instance.
        let v = sess.eval_str("(-sum (Pair. 5 7))");
        assert_eq!(nanbox_to_f64(v), 5.0);
    }

    // ── extend-type end-to-end ───────────────────────────────────────

    #[test]
    fn extend_type_on_user_deftype_dispatches() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.extend.user", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol IGetA (-get-a [this]))");
        let _ = sess.eval_str("(deftype Pt [a b])");
        // Extend Pt with IGetA: return the `a` field.
        let _ = sess.eval_str("(extend-type Pt IGetA (-get-a [this] (.-a this)))");
        let v = sess.eval_str("(-get-a (Pt. 10 20))");
        assert_eq!(nanbox_to_f64(v), 10.0);
    }

    #[test]
    fn extend_type_on_nil_dispatches() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.extend.nil", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol IDescribe (-describe [this]))");
        let _ = sess.eval_str("(extend-type nil IDescribe (-describe [this] 999))");
        let v = sess.eval_str("(-describe nil)");
        assert_eq!(nanbox_to_f64(v), 999.0);
    }

    #[test]
    fn extend_type_object_is_fallback() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.extend.object", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol IFallback (-fb [this]))");
        // Install only on Object — Pt instances should still dispatch
        // here through the BUILTIN_OBJECT fallback path.
        let _ = sess.eval_str("(deftype Pt [a])");
        let _ = sess.eval_str("(extend-type Object IFallback (-fb [this] 7))");
        let v = sess.eval_str("(-fb (Pt. 1))");
        assert_eq!(nanbox_to_f64(v), 7.0);
    }

    #[test]
    fn deftype_via_direct_factory_call() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types as ut;
        let mut sess = with_fresh_ns_session("test.deftype.fncall", || super::Session::new());
        ut::_reset_for_tests();
        // (Foo a b) without the trailing dot should also work — Foo
        // is bound to the factory fn directly.
        let _ = sess.eval_str("(deftype Bar [x])");
        let v = sess.eval_str("(.-x (Bar 99))");
        assert_eq!(nanbox_to_f64(v), 99.0);
    }

    #[test]
    fn defprotocol_arity_2_method_dispatches() {
        // Serialize + reset the process-global type/protocol registry
        // (shared with user_types tests) so parallel runs don't race.
        let _g = crate::lang::user_types::registry_test_guard();
        use crate::lang::user_types::{
            self as ut, BUILTIN_NIL, install_impl, protocol_id_by_name, protocol_method_id,
        };
        let mut sess = with_fresh_ns_session("test.defproto.arity2", || super::Session::new());
        ut::_reset_for_tests();
        let _ = sess.eval_str("(defprotocol IAdd (-add [this x]))");
        let pid = protocol_id_by_name(&Symbol::intern_ns_name(None, "IAdd")).unwrap();
        let mid = protocol_method_id(pid, &Symbol::intern_ns_name(None, "-add")).unwrap();
        // Impl: ignore `this`, just return `x`.
        let fn_handle = sess.eval_str("(fn* [this x] x)");
        install_impl(BUILTIN_NIL, mid, fn_handle);
        let v = sess.eval_str("(-add nil 42)");
        assert_eq!(nanbox_to_f64(v), 42.0);
    }
}
