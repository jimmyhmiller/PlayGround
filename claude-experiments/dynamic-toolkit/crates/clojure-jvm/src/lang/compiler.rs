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

pub static COMPILER_KEYWORDS: LazyLock<CompilerKeywords> =
    LazyLock::new(CompilerKeywords::new);

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
    fn has_java_class(&self) -> bool { false }

    /// Java: `Class getJavaClass()`.
    fn get_java_class(&self) -> Option<HostClass> { None }

    /// Rust-side helper for Java's `expr instanceof MaybePrimitiveExpr` +
    /// downcast pattern. Default returns `None`; nodes that implement
    /// `MaybePrimitiveExpr` override to `Some(self)`.
    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> { None }

    /// Java's `expr instanceof FnExpr` downcast for InvokeExpr's direct-call
    /// optimization. Default returns `None`; `FnExpr` overrides.
    fn as_fn_expr(&self) -> Option<&FnExpr> { None }

    /// Java's `expr instanceof LocalBindingExpr` downcast — used by
    /// `InvokeExpr` to peek through a local that resolves to a `(fn …)` init
    /// and emit a direct call. Default returns `None`; `LocalBindingExpr`
    /// overrides.
    fn as_local_binding_expr(&self) -> Option<&LocalBindingExpr> { None }
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
    pub fn new(f: &'f mut DynFunc) -> Self { IrEmitter { f } }
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

use std::sync::atomic::{AtomicBool, AtomicI32};
use std::sync::Mutex;

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
    pub fn placeholder() -> Arc<Self> { Self::new(Object::Nil) }

    // ---- accessors (Java's `final` getters) ------------------------------

    pub fn name(&self) -> Option<String> { self.name.read().unwrap().clone() }
    pub fn internal_name(&self) -> Option<String> {
        self.internal_name.read().unwrap().clone()
    }
    pub fn this_name(&self) -> Option<String> { self.this_name.read().unwrap().clone() }
    pub fn line(&self) -> i32 { self.line.load(std::sync::atomic::Ordering::Relaxed) }
    pub fn column(&self) -> i32 { self.column.load(std::sync::atomic::Ordering::Relaxed) }
    pub fn constants_id(&self) -> i32 {
        self.constants_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    // ---- shape predicates Java uses to switch emit behavior --------------

    /// Java: `boolean isDeftype()` — non-null `fields`.
    pub fn is_deftype(&self) -> bool { self.fields.lock().unwrap().is_some() }

    /// Java: `boolean supportsMeta()` — non-deftype ObjExprs carry `__meta`.
    pub fn supports_meta(&self) -> bool { !self.is_deftype() }

    /// Java: `boolean isMutable(LocalBinding lb)` — true for deftype mutable
    /// fields. Stubbed at `false` until field-mutability metadata is wired.
    pub fn is_mutable(&self, _lb: &LocalBinding) -> bool { false }

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
        crate::unimplemented_port!(
            "ObjExpr.emitVar",
            "needs IrEmitter + var lookup lowering"
        )
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
    pub fn emit_assign_local(
        &self,
        _ir: &mut IrEmitter<'_>,
        lb: &LocalBinding,
        _val: &dyn Expr,
    ) {
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
    pub fn compile(
        &self,
        _super_name: &str,
        _interface_names: &[&str],
        _one_time_use: bool,
    ) {
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

    fn has_java_class(&self) -> bool { true }

    fn get_java_class(&self) -> Option<HostClass> {
        // Java: compiledClass if set; else tagToClass(tag); else IFn.class.
        if let Some(c) = self.compiled_class.read().unwrap().clone() {
            return Some(c);
        }
        if let Object::Symbol(t) = &self.tag {
            return Some(HostClass { name: Arc::new(t.get_name().to_string()) });
        }
        Some(HostClass { name: Arc::new("clojure.lang.IFn".to_string()) })
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

    fn emit(&self, _context: C, _objx: &ObjExpr, _ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java emits `objx.emitVar(gen, var); var.bindRoot(init); …`. Through
        // dynir that would require a Var-pointer extern and a `clj_bind_root`
        // extern. Until those are registered, `(def …)` goes through
        // tree-walking `eval` only.
        crate::unimplemented_port!(
            "DefExpr.emit",
            "needs JIT externs for Var.bindRoot + Var-pointer constant pool"
        )
    }

    fn has_java_class(&self) -> bool { true }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass { name: Arc::new("clojure.lang.Var".to_string()) })
    }
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
    // Java handles a 4-form `(def name "docstring" val)` shape by rewriting
    // to `(def name val)`. We accept 2-form / 3-form for now.
    let n = super::rt::count(&form);
    if n > 3 {
        panic!("clojure-jvm: RuntimeException — Too many arguments to def");
    }
    if n < 2 {
        panic!("clojure-jvm: RuntimeException — Too few arguments to def");
    }
    let name_form = super::rt::second(&form);
    let sym = match &name_form {
        Object::Symbol(s) => s.clone(),
        _ => panic!(
            "clojure-jvm: RuntimeException — First argument to def must be a Symbol"
        ),
    };

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

    let init_provided = n == 3;
    let init = if init_provided {
        Some(analyze_named(
            if context == C::Eval { context } else { C::Expression },
            super::rt::third(&form),
            Some(sym.get_name()),
        ))
    } else {
        None
    };

    Box::new(DefExpr { var: v, init, init_provided, is_dynamic: false })
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

    fn emit(&self, _context: C, _objx: &ObjExpr, _ir: &mut IrEmitter<'_>) -> Option<Value> {
        crate::unimplemented_port!(
            "VarExpr.emit",
            "needs JIT externs for Var.get + Var-pointer constant pool"
        )
    }

    fn has_java_class(&self) -> bool { self.tag.is_some() }
    fn get_java_class(&self) -> Option<HostClass> {
        self.tag.as_ref().map(|t| HostClass {
            name: Arc::new(t.get_name().to_string()),
        })
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
    fn eval(&self) -> Object { Object::Keyword(self.k.clone()) }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Real keyword emission needs an ObjExpr-backed keyword-callsite table
        // (Java's `objx.emitKeyword` looks up the keyword's pool slot). Until
        // ObjExpr is wired into the constant pool, emit the keyword inline as
        // a string-pool ID tagged with the NanBox string tag — a stand-in that
        // round-trips through eval but isn't faithful to Java's runtime layout.
        let payload = self.k.get_name().len() as u64;
        let v = ir.f.tagged_const(0, payload);
        if context == C::Statement { None } else { Some(v) }
    }

    fn has_java_class(&self) -> bool { true }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass { name: Arc::new("clojure.lang.Keyword".to_string()) })
    }
}

impl LiteralExpr for KeywordExpr {
    fn val(&self) -> Object { Object::Keyword(self.k.clone()) }
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
    fn eval(&self) -> Object { self.n.clone() }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `objx.emitConstant(gen, id)` — loads slot `id` from the
        // generated class's constant table. We don't have a class-level pool
        // yet, so emit the boxed primitive inline (NanBox float).
        if context == C::Statement {
            return None;
        }
        let f = self.n.as_f64().expect("NumberExpr value must be Long/Double");
        Some(ir.f.number(f))
    }

    fn has_java_class(&self) -> bool { true }

    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass {
            name: Arc::new(match self.n {
                Object::Long(_) => "long".to_string(),
                Object::Double(_) => "double".to_string(),
                _ => unreachable!("NumberExpr ctor rejects non-primitive numbers"),
            }),
        })
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> { Some(self) }
}

impl LiteralExpr for NumberExpr {
    fn val(&self) -> Object { self.n.clone() }
}

impl MaybePrimitiveExpr for NumberExpr {
    fn can_emit_primitive(&self) -> bool { true }

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
    fn eval(&self) -> Object { self.v.clone() }

    fn emit(&self, _context: C, _objx: &ObjExpr, _ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `objx.emitConstant(gen, id)` then drop in STATEMENT. We don't
        // yet have a class-level constant pool wired through ObjExpr, so the
        // path stub-panics; literals (Long/Double/String/Bool/nil) bypass
        // ConstantExpr through `analyze` and emit directly.
        crate::unimplemented_port!(
            "ConstantExpr.emit",
            "needs ObjExpr-backed constant-pool lowering"
        )
    }

    fn has_java_class(&self) -> bool {
        // Java: Modifier.isPublic(v.getClass().getModifiers()). Until we
        // model host classes, conservatively say "yes" for the variants we
        // know are public types.
        !matches!(self.v, Object::Unported { .. })
    }

    fn get_java_class(&self) -> Option<HostClass> {
        let name = match &self.v {
            Object::Nil => return None,
            Object::Bool(_) => "java.lang.Boolean",
            Object::Long(_) => "java.lang.Long",
            Object::Double(_) => "java.lang.Double",
            Object::String(_) => "java.lang.String",
            Object::Symbol(_) => "clojure.lang.Symbol",
            Object::Keyword(_) => "clojure.lang.Keyword",
            Object::Var(_) => "clojure.lang.Var",
            Object::Namespace(_) => "clojure.lang.Namespace",
            Object::List(_) => "clojure.lang.PersistentList",
            Object::Vector(_) => "clojure.lang.PersistentVector",
            Object::Host(_) => return None,
            Object::Unported { .. } => return None,
        };
        Some(HostClass { name: Arc::new(name.to_string()) })
    }
}

impl LiteralExpr for ConstantExpr {
    fn val(&self) -> Object { self.v.clone() }
}

/// `Compiler.ConstantExpr.Parser`. Parses `(quote v)`.
pub struct ConstantExprParser;

impl IParser for ConstantExprParser {
    fn parse(&self, _context: C, _form: Object) -> Box<dyn Expr> {
        // Java path: enforce argCount==1, then dispatch on `v`'s shape into
        // NIL_EXPR / TRUE_EXPR / FALSE_EXPR / NumberExpr / StringExpr /
        // EmptyExpr / ConstantExpr. We can't fully implement it until RT.count
        // and ISeq are real; mark the path clearly.
        crate::unimplemented_port!(
            "ConstantExpr.Parser.parse",
            "needs RT.count + RT.second + EmptyExpr + IPersistentCollection"
        )
    }
}

// ============================================================================
// Java line ~2538–2558: `NilExpr` + the `NIL_EXPR` singleton.
// ============================================================================

/// `Compiler.NilExpr`.
#[derive(Debug, Clone, Copy)]
pub struct NilExpr;

impl Expr for NilExpr {
    fn eval(&self) -> Object { Object::Nil }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `gen.visitInsn(ACONST_NULL); if STATEMENT pop`. In dynlang
        // we emit the NanBox nil constant and drop it for STATEMENT.
        if context == C::Statement { return None; }
        Some(ir.f.nil())
    }

    fn has_java_class(&self) -> bool { true }
    fn get_java_class(&self) -> Option<HostClass> { None }
}

impl LiteralExpr for NilExpr {
    fn val(&self) -> Object { Object::Nil }
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
        if self.val { super::rt::T() } else { super::rt::F() }
    }

    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java loads `Boolean.TRUE` / `Boolean.FALSE`. In dynlang's NanBox
        // world, both are tagged with `bool_tag` and a 0/1 payload.
        if context == C::Statement { return None; }
        Some(ir.f.bool_val(self.val))
    }

    fn has_java_class(&self) -> bool { true }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass { name: Arc::new("java.lang.Boolean".to_string()) })
    }
}

impl LiteralExpr for BooleanExpr {
    fn val(&self) -> Object {
        if self.val { super::rt::T() } else { super::rt::F() }
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
        StringExpr { str: Arc::new(s.into()) }
    }
}

impl Expr for StringExpr {
    fn eval(&self) -> Object { Object::String(self.str.clone()) }

    fn emit(&self, _context: C, _objx: &ObjExpr, _ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Java: `gen.push(str)` — pushes a string ref. We don't yet have a
        // string-pool path through ObjExpr (it'd resolve to a NanBox pointer
        // tag for a heap-allocated string). Stub for now; literal Long/Bool/
        // nil are enough to cover our first end-to-end test.
        crate::unimplemented_port!(
            "StringExpr.emit",
            "needs string-pool lowering (NanBox heap-string tag)"
        )
    }

    fn has_java_class(&self) -> bool { true }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass { name: Arc::new("java.lang.String".to_string()) })
    }
}

impl LiteralExpr for StringExpr {
    fn val(&self) -> Object { Object::String(self.str.clone()) }
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
        IfExpr { test_expr, then_expr, else_expr, line, column }
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
        let test_val = self.test_expr.emit(C::Expression, objx, ir)
            .expect("clojure-jvm: IfExpr test must produce a value");

        let then_bb = ir.f.fb.create_block(&[]);
        let else_bb = ir.f.fb.create_block(&[]);

        // Phi type. Statement context has no phi; otherwise NanBox I64 (or
        // primitive type for unboxed paths).
        let phi_ty_opt: Option<dynir::Type> = if context == C::Statement {
            None
        } else if emit_unboxed {
            Some(match self.get_java_class().as_ref().map(|c| c.name.as_str()) {
                Some("long") => dynir::Type::I64,
                Some("double") => dynir::Type::F64,
                other => panic!(
                    "clojure-jvm: IfExpr.emit_unboxed branches must agree on a primitive type, got {other:?}"
                ),
            })
        } else {
            Some(dynir::Type::I64)
        };

        let merge_bb = match phi_ty_opt {
            Some(ty) => ir.f.fb.create_block(&[ty]),
            None => ir.f.fb.create_block(&[]),
        };
        ir.f.br_if_truthy(test_val, then_bb, &[], else_bb, &[]);

        // Emit each branch. Track whether it reached the merge block.
        let mut any_reached = false;

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
            any_reached = true;
            match (phi_ty_opt.is_some(), then_val) {
                (true, Some(v)) => ir.f.fb.jump(merge_bb, &[v]),
                (false, _) => ir.f.fb.jump(merge_bb, &[]),
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
            any_reached = true;
            match (phi_ty_opt.is_some(), else_val) {
                (true, Some(v)) => ir.f.fb.jump(merge_bb, &[v]),
                (false, _) => ir.f.fb.jump(merge_bb, &[]),
                (true, None) => panic!(
                    "clojure-jvm: IfExpr else branch produced no value in non-STATEMENT context"
                ),
            }
        }

        if !any_reached {
            // Both branches diverged (e.g. both recur). The IfExpr itself
            // terminates — nothing reaches merge_bb. Caller checks
            // `current_block_is_terminated`; for that to be true, we need
            // the current block to be terminated. Branch chose between then
            // and else, but the predecessor was branched-from. We're not in
            // a "terminated" state per dynir though — the else block we just
            // emitted is the current block, and it IS terminated. So we
            // return None and dynir's terminator state reflects that.
            return None;
        }

        ir.f.fb.switch_to_block(merge_bb);
        if phi_ty_opt.is_some() {
            Some(ir.f.fb.block_param(merge_bb, 0))
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
        self.then_expr.get_java_class().or_else(|| self.else_expr.get_java_class())
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> { Some(self) }
}

impl MaybePrimitiveExpr for IfExpr {
    fn can_emit_primitive(&self) -> bool {
        // Java wraps this in try/catch and returns false on any exception.
        // We don't need that — none of our calls panic in the happy path.
        let then_mp = self.then_expr.as_maybe_primitive();
        let else_mp = self.else_expr.as_maybe_primitive();
        let (Some(t), Some(e)) = (then_mp, else_mp) else { return false };
        let same_class = self.then_expr.get_java_class()
            == self.else_expr.get_java_class();
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
    pub fn new(exprs: Vec<Box<dyn Expr>>) -> Self { BodyExpr { exprs } }

    /// Java: `lastExpr()` — `exprs.nth(exprs.count() - 1)`. Panics on empty
    /// (matches Java's NPE/AIOOBE on the same path), because Java's Parser
    /// guarantees at least NIL_EXPR is pushed.
    fn last_expr(&self) -> &dyn Expr {
        self.exprs.last().expect("BodyExpr always has ≥1 expr (Parser pushes NIL_EXPR for empty)").as_ref()
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
            if ir.f.fb.current_block_is_terminated() { break; }
            let last_pos = i + 1 == n;
            let ctx = if last_pos { context } else { C::Statement };
            let v = e.emit(ctx, objx, ir);
            if last_pos {
                last_val = v;
            }
        }
        last_val
    }

    fn has_java_class(&self) -> bool { self.last_expr().has_java_class() }

    fn get_java_class(&self) -> Option<HostClass> {
        self.last_expr().get_java_class()
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> { Some(self) }
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
pub fn analyze_named(
    context: C,
    form: Object,
    name: Option<&str>,
) -> Box<dyn Expr> {
    let _ = name; // not yet threaded through to FnExpr / ObjExpr

    // Java unwraps LazySeq via RT.seq before the dispatch — we don't have
    // LazySeq yet, so skip that step.

    match form {
        Object::Nil => Box::new(NIL_EXPR),
        Object::Bool(true) => Box::new(TRUE_EXPR),
        Object::Bool(false) => Box::new(FALSE_EXPR),
        Object::Long(_) | Object::Double(_) => NumberExpr::parse(form),
        Object::String(s) => {
            // Java: new StringExpr(((String) form).intern())
            // We skip the intern step — Rust doesn't have JVM-style string
            // interning by default and our StringExpr holds an Arc<String>.
            Box::new(StringExpr { str: s })
        }
        Object::Keyword(k) => Box::new(register_keyword(k)),
        Object::List(ref l) => {
            // Empty list → EmptyExpr (not yet ported). Non-empty: analyzeSeq.
            if l.count() == 0 {
                crate::unimplemented_port!(
                    "Compiler.analyze on empty collection",
                    "needs EmptyExpr port"
                )
            }
            analyze_seq(context, form)
        }
        Object::Symbol(s) => analyze_symbol(s),
        Object::Var(_) => crate::unimplemented_port!(
            "Compiler.analyze on Var",
            "treated as ConstantExpr in Java — wire when Var-as-constant lands"
        ),
        Object::Vector(_) => crate::unimplemented_port!(
            "Compiler.analyze on Vector",
            "needs VectorExpr port"
        ),
        Object::Namespace(_) | Object::Host(_) | Object::Unported { .. } => {
            Box::new(ConstantExpr::new(form))
        }
    }
}

/// `Compiler.analyzeSeq(C context, ISeq form, String name)` — Java line
/// ~7167+. Dispatches list-headed forms to specials / macros / invoke.
///
/// We currently recognize the special forms whose Exprs are ported (`if`,
/// `do`, `quote`). Everything else panics with a clear message.
fn analyze_seq(context: C, form: Object) -> Box<dyn Expr> {
    let op = super::rt::first(&form);
    let specials = &*SPECIAL_SYMBOLS;

    // Java would also expand macros + resolve `op` through namespaces here.
    // For now we only match on raw Symbol head.
    if let Object::Symbol(sym) = &op {
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
        if **sym == *specials.DEF {
            return parse_def_form(context, form);
        }
        if **sym == *specials.FN {
            return parse_fn_form(context, form);
        }
        if **sym == *specials.RECUR {
            return parse_recur_form(context, form);
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
fn parse_if_form(context: C, form: Object) -> Box<dyn Expr> {
    let n = super::rt::count(&form);
    if n > 4 {
        panic!("clojure-jvm: RuntimeException — Too many arguments to if");
    }
    if n < 3 {
        panic!("clojure-jvm: RuntimeException — Too few arguments to if");
    }
    let test_ctx = if context == C::Eval { C::Eval } else { C::Expression };
    let test_expr = analyze(test_ctx, super::rt::second(&form));
    let then_expr = analyze(context, super::rt::third(&form));
    let else_expr = if n == 4 {
        analyze(context, super::rt::fourth(&form))
    } else {
        Box::new(NIL_EXPR) as Box<dyn Expr>
    };
    Box::new(IfExpr::new(0, 0, test_expr, then_expr, else_expr))
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
        Arc::new(LocalBinding {
            sym,
            tag,
            init: RwLock::new(init),
            idx,
            name,
            is_arg,
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
        // binding's stack slot. In dynlang the slot is a `DynFunc::def_var`
        // entry keyed by `local_slot_name(idx)`.
        if context == C::Statement {
            // Java drops the load; we just skip the read.
            return None;
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
            return Some(HostClass { name: Arc::new(t.get_name().to_string()) });
        }
        self.b.init.read().unwrap().as_ref().and_then(|e| e.get_java_class())
    }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> { Some(self) }

    fn as_local_binding_expr(&self) -> Option<&LocalBindingExpr> { Some(self) }
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
pub fn local_slot_name(idx: i32) -> String { format!("local__{idx}") }

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

fn current_local_env() -> Arc<LocalEnvMap> {
    let v = COMPILER_VARS.LOCAL_ENV.deref();
    v.host_as::<LocalEnvMap>().unwrap_or_else(|| Arc::new(HashMap::new()))
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
    COMPILER_VARS.NEXT_LOCAL_NUM.set_value(Object::Long(n as i64));
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
fn tag_of(_o: &Object) -> Option<Arc<Symbol>> { None }

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
    // Java: `resolve(sym)` walks the current ns's mappings, falling back to
    // the symbol's explicit ns. We do a simplified lookup: try the current
    // ns, then a qualified ns if present.
    let resolved = resolve_var(&sym);
    if let Some(v) = resolved {
        return Box::new(VarExpr { var: v, tag });
    }
    panic!(
        "clojure-jvm: RuntimeException — Unable to resolve symbol: {} in this context",
        sym.get_name()
    );
}

/// `Compiler.resolve(Object sym)` — Java line ~7965 (simplified). Looks up
/// `sym` in the current namespace's mappings, falling back to its qualified
/// namespace if any.
fn resolve_var(sym: &Symbol) -> Option<Arc<Var>> {
    use super::namespace::Namespace;
    if let Some(ns_str) = sym.get_namespace() {
        let ns_sym = Symbol::intern(ns_str);
        let ns = Namespace::find(&ns_sym)?;
        return ns.find_interned_var(&Symbol::intern(sym.get_name()));
    }
    let cur = super::rt::current_ns();
    match cur.get_mapping(sym) {
        Some(Object::Var(v)) => Some(v),
        _ => None,
    }
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
        LetExpr { binding_inits, body, is_loop }
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

    fn has_java_class(&self) -> bool { self.body.has_java_class() }
    fn get_java_class(&self) -> Option<HostClass> { self.body.get_java_class() }

    fn as_maybe_primitive(&self) -> Option<&dyn MaybePrimitiveExpr> { Some(self) }
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
            let init_val = bi.init.emit(C::Expression, objx, ir)
                .expect("clojure-jvm: LetExpr init must produce a value");
            ir.f.def_var(&local_slot_name(bi.binding.idx), init_val);
        }
        // Java emits a `loopLabel` here for recur; skip until RecurExpr lands.
        if self.is_loop {
            // Best-effort warning baked into the panic site rather than
            // silently dropping the loop semantics.
            // (RecurExpr / LOOP_LABEL plumbing is the next piece.)
        }
        if emit_unboxed {
            let body_val = self
                .body
                .as_maybe_primitive()
                .expect("LetExpr.emit_unboxed: body must be MaybePrimitiveExpr")
                .emit_unboxed(context, objx, ir);
            Some(body_val)
        } else {
            self.body.emit(context, objx, ir)
        }
    }
}

/// `Compiler.LetExpr.Parser`. Drives both `let*` and `loop*`. Java line ~6911.
pub struct LetExprParser;

impl IParser for LetExprParser {
    fn parse(&self, context: C, frm: Object) -> Box<dyn Expr> {
        parse_let_form(context, frm)
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
    let bindings: Arc<PersistentVector> = match bindings_obj {
        Object::Vector(v) => v,
        _ => panic!(
            "clojure-jvm: IllegalArgumentException — Bad binding form, expected vector"
        ),
    };
    if bindings.count() % 2 != 0 {
        panic!(
            "clojure-jvm: IllegalArgumentException — Bad binding form, expected matched symbol expression pairs"
        );
    }
    // Body is everything after the bindings vector.
    let body_seq = super::rt::next(&super::rt::next(&form));

    // Java: EVAL contexts (and loop in EXPRESSION context) wrap the form in
    // an immediately-invoked `(fn* [] form)`. We don't have FnExpr yet, so
    // we route EVAL straight through as well — semantics will diverge for
    // top-level eval until FnExpr lands; flag it loudly.
    if context == C::Eval || (context == C::Expression && is_loop) {
        // Java path:
        //   return analyze(context, RT.list(RT.list(FNONCE, PersistentVector.EMPTY, form)));
        crate::unimplemented_port!(
            "LetExpr.Parser eval-wrap-in-fn",
            "needs FnExpr port (Java wraps let-at-eval in `(fn* [] ...)`)"
        );
    }

    // Snapshot LOCAL_ENV + NEXT_LOCAL_NUM before binding new locals so the
    // bindings scope is undone when we pop. Java does this via
    // pushThreadBindings.
    Var::push_thread_bindings(vec![
        (COMPILER_VARS.LOCAL_ENV.clone(), Object::Host(current_local_env())),
        (COMPILER_VARS.NEXT_LOCAL_NUM.clone(), Object::Long(current_next_local_num() as i64)),
    ]);

    let result = (|| -> Box<dyn Expr> {
        let mut binding_inits: Vec<BindingInit> = Vec::new();

        let n = bindings.count();
        let mut i = 0;
        while i < n {
            let sym_form = bindings.nth(i);
            let sym = match &sym_form {
                Object::Symbol(s) => s.clone(),
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

        // For loops: Java sets LOOP_LOCALS, then runs body in C.RETURN. We
        // don't have RecurExpr yet, so we run the body in `context` either
        // way. LOOP_LOCALS stays nil.
        let body_ctx = if is_loop { C::Return } else { context };
        let body_expr = parse_body_seq(body_ctx, body_seq);

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
#[derive(Debug)]
pub struct FnMethod {
    pub params: Vec<Arc<LocalBinding>>,
    pub body: Arc<dyn Expr>,
    pub line: i32,
    pub column: i32,
}

/// `Compiler.FnExpr`. A user-defined fn. Each `FnExpr` declares a fresh dynir
/// function (held in `fref`) and stores its single (for now) `FnMethod`.
#[derive(Debug)]
pub struct FnExpr {
    pub method: FnMethod,
    pub fref: dynir::FuncRef,
    pub name: String,
}

impl FnExpr {
    pub fn fref(&self) -> dynir::FuncRef { self.fref }
}

impl Expr for FnExpr {
    fn emit(&self, context: C, _objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Option<Value> {
        // Produce a NanBox-tagged FuncRef handle. Tag 3 is the only free tag
        // in dynlang's default NanBoxTags scheme (0=nil, 1=bool, 2=ptr).
        // Decoders extract the payload (`bits & PAYLOAD_MASK`) and treat it
        // as a `FuncRef` index for indirect dispatch.
        //
        // For the common case where `InvokeExpr.fexpr` is a `FnExpr` (or a
        // `LocalBindingExpr` whose init is a `FnExpr`), the call site
        // bypasses this value and uses `FnExpr::fref` directly. The handle
        // exists so the value can flow through `let` and other expression
        // positions without panicking — fully dynamic dispatch (calling
        // through it) lands when we wire `call_via_func_ref` + the call
        // table base extern.
        if context == C::Statement { return None; }
        let payload = self.fref.index() as u64;
        Some(ir.f.tagged_const(3, payload))
    }

    fn has_java_class(&self) -> bool { true }
    fn get_java_class(&self) -> Option<HostClass> {
        Some(HostClass { name: Arc::new("clojure.lang.AFunction".to_string()) })
    }

    fn as_fn_expr(&self) -> Option<&FnExpr> { Some(self) }
}

/// Parse `(fn* [args] body...)` or `(fn* name [args] body...)`. Returns a
/// `FnExpr` whose body has been registered as a pending fn on the active
/// `Compiler`.
fn parse_fn_form(context: C, form: Object) -> Box<dyn Expr> {
    let _ = context; // fn body is always lowered as a separate fn
    // (fn* [params] body...) — optional name symbol allowed in slot 2
    let raw_after_fn = super::rt::next(&form);
    // Two shapes:
    //   (fn* sym [params] body...)
    //   (fn* [params] body...)
    //   (fn* ([params] body...) ([params2] body2...) ...)   — multi-arity NYI
    let (name_sym, params_form, body_seq) = parse_fn_head(&raw_after_fn);
    let params_vec: Arc<PersistentVector> = match params_form {
        Object::Vector(v) => v,
        other => panic!(
            "clojure-jvm: IllegalArgumentException — fn* expects a vector of params, got {other:?}"
        ),
    };

    // Push a fresh local scope for the fn body. NEXT_LOCAL_NUM resets to 0;
    // LOCAL_ENV starts empty (no closures yet).
    Var::push_thread_bindings(vec![
        (
            COMPILER_VARS.LOCAL_ENV.clone(),
            Object::Host(std::sync::Arc::new(LocalEnvMap::new())),
        ),
        (COMPILER_VARS.NEXT_LOCAL_NUM.clone(), Object::Long(0)),
    ]);

    // First pass: register the params so we know the LocalBindings (we need
    // them to set LOOP_LOCALS before analyzing the body).
    let params: Vec<Arc<LocalBinding>> = {
        let mut params: Vec<Arc<LocalBinding>> = Vec::with_capacity(params_vec.count() as usize);
        for i in 0..params_vec.count() {
            let p = params_vec.nth(i);
            let psym = match &p {
                Object::Symbol(s) => s.clone(),
                other => panic!(
                    "clojure-jvm: IllegalArgumentException — fn* param must be a Symbol, got {other:?}"
                ),
            };
            if psym.get_namespace().is_some() {
                panic!(
                    "clojure-jvm: RuntimeException — Can't use qualified name as parameter: {}",
                    psym.get_name()
                );
            }
            let lb = register_local(psym, None, None, true);
            params.push(lb);
        }
        params
    };

    // Push LOOP_LOCALS for the body so `recur` validates its arg count
    // against the fn params.
    Var::push_thread_bindings(vec![
        (
            COMPILER_VARS.LOOP_LOCALS.clone(),
            Object::Host(std::sync::Arc::new(params.clone())),
        ),
    ]);
    let body_expr: Arc<dyn Expr> = Arc::from(parse_body_seq(C::Return, body_seq));
    Var::pop_thread_bindings();

    Var::pop_thread_bindings();

    // Register the fn body as a pending compilation on the active session.
    let fref = with_active_compiler(|c| {
        let name = c.fresh_fn_name(name_sym.as_deref());
        c.declare_pending_fn(name.clone(), params.clone(), body_expr.clone());
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
    let line_i = if let Object::Long(n) = line { n as i32 } else { 0 };
    let column_i = if let Object::Long(n) = column { n as i32 } else { 0 };

    Box::new(FnExpr {
        method: FnMethod {
            params,
            body: body_expr,
            line: line_i,
            column: column_i,
        },
        fref: fref.0,
        name: fref.1,
    })
}

/// Pull the optional name symbol, params vector, and body seq out of the
/// post-`fn*` portion of the form. Returns `(name_opt, params_form, body_seq)`.
fn parse_fn_head(after_fn: &Object) -> (Option<String>, Object, Object) {
    // after_fn is the tail seq starting with either a name sym, a params
    // vector, or a list (multi-arity).
    let first = super::rt::first(after_fn);
    if let Object::Symbol(s) = &first {
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
        // Compute the target FuncRef if statically known. Two cases:
        //   1. Head is a literal FnExpr  (lambda in head position)
        //   2. Head is a LocalBindingExpr whose binding's init is a FnExpr
        //      (let-bound fn — common pattern for first-class fn values)
        let static_fref: Option<dynir::FuncRef> = self
            .fexpr
            .as_fn_expr()
            .map(|f| f.fref())
            .or_else(|| {
                let lbe = self.fexpr.as_local_binding_expr()?;
                let init_guard = lbe.b.init.read().unwrap();
                let init = init_guard.as_ref()?;
                init.as_fn_expr().map(|f| f.fref())
            });

        if let Some(fref) = static_fref {
            let mut arg_vals: Vec<Value> = Vec::with_capacity(self.args.len());
            for a in &self.args {
                let v = a.emit(C::Expression, objx, ir)
                    .expect("InvokeExpr arg must produce a value");
                arg_vals.push(v);
            }
            let ret = ir.f.fb.call(fref, &arg_vals);
            return match context {
                C::Statement => None,
                _ => Some(ret.expect("Clojure fns always return I64 (NanBox)")),
            };
        }

        // Fully dynamic invocation (e.g., calling through a Var or a
        // non-constant value) needs first-class fn values + a call-table
        // base extern. Not yet wired.
        crate::unimplemented_port!(
            "InvokeExpr.emit (non-FnExpr head)",
            "needs first-class fn values via NanBox FuncRef + call_via_func_ref"
        )
    }

    fn has_java_class(&self) -> bool { self.tag.is_some() }
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
            panic!(
                "clojure-jvm: IllegalStateException — recur outside of loop/fn body"
            )
        });

        // Evaluate each new arg first (Java order: emit all args, then store
        // in reverse). dynir doesn't need the reverse trick since values are
        // SSA — just compute then assign.
        let mut new_vals: Vec<Value> = Vec::with_capacity(self.args.len());
        for a in &self.args {
            let v = a
                .emit(C::Expression, objx, ir)
                .expect("recur arg must produce a value");
            new_vals.push(v);
        }

        // Store each new value into its corresponding local slot.
        for (lb, v) in target.locals.iter().zip(new_vals.iter()) {
            ir.f.set_var(&local_slot_name(lb.idx), *v);
        }

        // Jump back to the loop entry block.
        ir.f.fb.jump(target.block, &[]);
        None
    }

    fn has_java_class(&self) -> bool { true }
    fn get_java_class(&self) -> Option<HostClass> {
        // Java: `RECUR_CLASS` (the marker). We model that with a fixed name.
        Some(HostClass { name: Arc::new("__recur__".to_string()) })
    }
}

/// Parse `(recur arg1 arg2 ...)`. Java requires C::RETURN tail position;
/// we relax that to "must be in a recur-able position" — `LOOP_LABEL` bound.
fn parse_recur_form(context: C, form: Object) -> Box<dyn Expr> {
    if context != C::Return {
        panic!(
            "clojure-jvm: UnsupportedOperationException — Can only recur from tail position"
        );
    }
    // Pull the loop locals from the LOOP_LOCALS thread binding so we can
    // validate arg count + later emit.
    let locals_obj = COMPILER_VARS.LOOP_LOCALS.deref();
    let locals = locals_obj
        .host_as::<Vec<Arc<LocalBinding>>>()
        .unwrap_or_else(|| {
            panic!(
                "clojure-jvm: UnsupportedOperationException — recur outside of loop/fn"
            )
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
    Box::new(RecurExpr { args, loop_locals: (*locals).clone() })
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
    Add, Sub, Mul, Div,
    Eq, Lt, Gt, Le, Ge,
}

impl PrimOp {
    /// Java's `Numbers.add`/`Numbers.lt`/etc. return host class. For us all
    /// arithmetic produces a NanBox number; comparisons produce a NanBox bool.
    fn returns_bool(self) -> bool {
        matches!(self, PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge)
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
        let v = self.emit_value(objx, ir);
        if context == C::Statement { None } else { Some(v) }
    }

    fn has_java_class(&self) -> bool { true }
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

impl PrimOpExpr {
    fn emit_value(&self, objx: &ObjExpr, ir: &mut IrEmitter<'_>) -> Value {
        // Variadic-arity reductions, with identity elements matching
        // Clojure: `(+)` = 0, `(*)` = 1, `(-)` errors (we'll match), `(/)` errors.
        match self.op {
            PrimOp::Add if self.args.is_empty() => return ir.f.number(0.0),
            PrimOp::Mul if self.args.is_empty() => return ir.f.number(1.0),
            PrimOp::Sub if self.args.is_empty() => panic!(
                "clojure-jvm: Wrong number of args (0) passed to: clojure.core/-"
            ),
            PrimOp::Div if self.args.is_empty() => panic!(
                "clojure-jvm: Wrong number of args (0) passed to: clojure.core//"
            ),
            _ => {}
        }
        let arg_vals: Vec<Value> = self
            .args
            .iter()
            .map(|a| {
                a.emit(C::Expression, objx, ir)
                    .expect("PrimOpExpr arg must produce a value")
            })
            .collect();

        // Unary handling for `-` and `/`:
        //   `(- x)` → `0 - x`
        //   `(/ x)` → `1 / x`
        if arg_vals.len() == 1 {
            match self.op {
                PrimOp::Sub => {
                    let z = ir.f.number(0.0);
                    return ir.f.sub(z, dynlang::TypeHint::Number, arg_vals[0], dynlang::TypeHint::Number);
                }
                PrimOp::Div => {
                    let one = ir.f.number(1.0);
                    return ir.f.div(one, dynlang::TypeHint::Number, arg_vals[0], dynlang::TypeHint::Number);
                }
                // Comparisons: `(< x)` is always true in Clojure (only one
                // element to compare). Same for the others.
                PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => {
                    return ir.f.bool_val(true);
                }
                _ => return arg_vals[0],
            }
        }

        // For binary/n-ary arithmetic: left fold.
        if !self.op.returns_bool() {
            let mut acc = arg_vals[0];
            for v in &arg_vals[1..] {
                acc = match self.op {
                    PrimOp::Add => ir.f.add(acc, dynlang::TypeHint::Number, *v, dynlang::TypeHint::Number),
                    PrimOp::Sub => ir.f.sub(acc, dynlang::TypeHint::Number, *v, dynlang::TypeHint::Number),
                    PrimOp::Mul => ir.f.mul(acc, dynlang::TypeHint::Number, *v, dynlang::TypeHint::Number),
                    PrimOp::Div => ir.f.div(acc, dynlang::TypeHint::Number, *v, dynlang::TypeHint::Number),
                    _ => unreachable!(),
                };
            }
            return acc;
        }

        // Comparisons: `(< a b c)` is `(and (< a b) (< b c))`. For now we
        // only implement the binary case correctly; n-ary comparisons emit
        // a chained AND.
        // We start with a `true` accumulator and AND each pairwise comparison.
        let mut acc = ir.f.bool_val(true);
        for i in 0..arg_vals.len() - 1 {
            let a = arg_vals[i];
            let b = arg_vals[i + 1];
            let pair = match self.op {
                PrimOp::Eq => ir.f.bit_eq(a, b),
                PrimOp::Lt => ir.f.lt(a, dynlang::TypeHint::Number, b, dynlang::TypeHint::Number),
                PrimOp::Gt => ir.f.gt(a, dynlang::TypeHint::Number, b, dynlang::TypeHint::Number),
                PrimOp::Le => ir.f.le(a, dynlang::TypeHint::Number, b, dynlang::TypeHint::Number),
                PrimOp::Ge => ir.f.ge(a, dynlang::TypeHint::Number, b, dynlang::TypeHint::Number),
                _ => unreachable!(),
            };
            // AND the accumulated bool with the new pair. dynlang doesn't
            // expose a bool-AND; we model it via `if acc then pair else false`.
            // Since both are NanBox bools, we can use is_truthy + select.
            let acc_truthy = ir.f.is_truthy(acc);
            let f = ir.f.bool_val(false);
            acc = ir.f.fb.select(acc_truthy, pair, f);
        }
        acc
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
}

/// FuncRefs for the externs every `Compiler` declares up-front. The order
/// matches `runtime_extern_fn_ptrs()`.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeExterns {
    pub var_bind_root: dynir::FuncRef,
    pub var_deref: dynir::FuncRef,
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
}

impl Compiler {
    /// Create a fresh compilation session.
    ///
    /// Externs are declared in a fixed order so that `runtime_extern_fn_ptrs`
    /// matches at JIT-binding time.
    pub fn new() -> Self {
        let mut dm = dynlang::DynModule::new(
            dynlang::GcConfig::leak(),
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
        let var_bind_root = dm.declare_extern("cljvm_var_bind_root", sig_2i64);
        let var_deref = dm.declare_extern("cljvm_var_deref", sig_1i64);
        Compiler {
            dm,
            pending_fns: std::sync::Mutex::new(Vec::new()),
            next_fn_id: std::sync::atomic::AtomicU32::new(0),
            externs: RuntimeExterns { var_bind_root, var_deref },
        }
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
    pub fn declare_pending_fn(
        &mut self,
        name: String,
        params: Vec<Arc<LocalBinding>>,
        body: Arc<dyn Expr>,
    ) -> dynir::FuncRef {
        let fref = self.dm.declare_func(&name, params.len());
        self.pending_fns
            .lock()
            .unwrap()
            .push(PendingFn { fref, params: params.clone(), body, name });
        fref
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
pub fn compile_form_to_jit(form: Object) -> (dynlower::JitModule, dynir::FuncRef) {
    use dynvalue::NanBox;

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
        let result_val = {
            let mut ir = IrEmitter::new(&mut df);
            let objx = ObjExpr::placeholder();
            expr.emit(C::Expression, &*objx, &mut ir)
                .expect("top-level form must produce a value in EXPRESSION context")
        };
        df.fb.ret(result_val);
        c.dm.finish_func(df);

        entry
    });

    let built = compiler.dm.build();
    let jit = dynlower::JitModule::compile_batch::<NanBox>(&built.module, &[], None);
    (jit, entry)
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
        let result_val = {
            let mut ir = IrEmitter::new(&mut df);
            let objx = ObjExpr::placeholder();
            expr.emit(C::Expression, &*objx, &mut ir)
                .expect("top-level form must produce a value in EXPRESSION context")
        };
        df.fb.ret(result_val);
        c.dm.finish_func(df);
        entry
    });
    let built = compiler.dm.build();
    (built, entry)
}

/// Lower a `PendingFn` body into its declared FuncRef. Called from the
/// drainer after analyze completes.
fn lower_pending_fn(c: &mut Compiler, p: PendingFn) {
    let mut df = c.dm.start_func(p.fref);
    {
        let mut ir = IrEmitter::new(&mut df);
        let objx = ObjExpr::placeholder();
        // Bind each param's slot to its incoming function argument. dynir
        // exposes params via `FunctionBuilder::block_param`; the entry
        // block's params are positional in declaration order.
        let entry_bb = ir.f.fb.entry_block();
        for (i, lb) in p.params.iter().enumerate() {
            let arg_val = ir.f.fb.block_param(entry_bb, i);
            ir.f.def_var(&local_slot_name(lb.idx), arg_val);
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
            <BooleanExpr as Expr>::get_java_class(&TRUE_EXPR).unwrap().name.as_str(),
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
            <StringExpr as Expr>::get_java_class(&s).unwrap().name.as_str(),
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
        assert!(is_truthy(&Object::Long(0)));        // Clojure: 0 is truthy
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
            <BodyExpr as Expr>::get_java_class(&body).unwrap().name.as_str(),
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
        assert!(HostClass { name: Arc::new("long".to_string()) }.is_primitive());
        assert!(HostClass { name: Arc::new("double".to_string()) }.is_primitive());
        assert!(!HostClass { name: Arc::new("java.lang.String".to_string()) }.is_primitive());
        assert!(!HostClass { name: Arc::new("clojure.lang.Keyword".to_string()) }.is_primitive());
    }

    // ---- analyze dispatch ---------------------------------------------------

    use super::super::persistent_list::PersistentList;

    fn list_of(items: Vec<Object>) -> Object {
        Object::List(PersistentList::create(items))
    }

    fn sym(s: &str) -> Object { Object::Symbol(Symbol::intern(s)) }

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
            vec_of(vec![
                sym("x"),
                Object::Long(1),
                sym("y"),
                Object::Long(2),
            ]),
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
        let form = list_of(vec![
            sym("let*"),
            vec_of(vec![sym("x")]),
            Object::Long(99),
        ]);
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
        *o.compiled_class.write().unwrap() =
            Some(HostClass { name: Arc::new("user.MyFn".to_string()) });
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
            dynlang::GcConfig::leak(),
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
        let lb = LocalBinding::new(
            0,
            Symbol::intern("x"),
            None,
            Some(init.clone()),
            false,
        );
        let bi = BindingInit::new(lb, init);
        let body: Box<dyn Expr> = Box::new(NumberExpr::new(Object::Long(2)));
        let le = LetExpr::new(vec![bi], body, false);
        let _ = <LetExpr as Expr>::eval(&le);
    }

    #[test]
    fn constant_expr_get_java_class_table() {
        let c = ConstantExpr::new(Object::String(Arc::new("x".to_string())));
        assert_eq!(
            <ConstantExpr as Expr>::get_java_class(&c).unwrap().name.as_str(),
            "java.lang.String"
        );
        let c_nil = ConstantExpr::new(Object::Nil);
        assert!(<ConstantExpr as Expr>::get_java_class(&c_nil).is_none());
    }

    // ---- End-to-end IR pipeline --------------------------------------------

    /// Run a top-level form through analyze → emit → ModuleInterpreter.
    /// Returns the raw 64-bit NanBox the interpreter reports.
    fn eval_form_via_ir(form: Object) -> u64 {
        use dynir::interp::{InterpResult, ModuleInterpreter, NoGcRoots};
        use dynvalue::NanBox;
        let (built, entry) = super::compile_form_to_interp(form);
        let roots = NoGcRoots;
        let interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
        match interp.run(entry, &[]) {
            Ok(InterpResult::Value(v)) => v,
            other => panic!("unexpected interp result: {other:?}"),
        }
    }

    /// Decode a NanBox bit pattern back to an f64 (assumes Number).
    fn nanbox_to_f64(bits: u64) -> f64 { f64::from_bits(bits) }

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

    fn eval_form_via_jit(form: Object) -> u64 {
        let (jit, entry) = super::compile_form_to_jit(form);
        jit.call(entry, &[])
    }

    #[test]
    fn jit_e2e_literal_long_returns_42() {
        let v = eval_form_via_jit(Object::Long(42));
        assert_eq!(nanbox_to_f64(v), 42.0);
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
            vec_of(vec![
                sym("x"),
                Object::Long(1),
                sym("y"),
                Object::Long(2),
            ]),
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
            vec_of(vec![
                sym("x"),
                Object::Long(1),
                sym("y"),
                Object::Long(2),
            ]),
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
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("a"), sym("b")]),
                sym("a"),
            ]),
            Object::Long(11),
            Object::Long(22),
        ]);
        let v = eval_form_via_jit(form);
        assert_eq!(nanbox_to_f64(v), 11.0);
    }

    #[test]
    fn jit_e2e_invoke_fn_two_args_returns_second() {
        let form = list_of(vec![
            list_of(vec![
                sym("fn*"),
                vec_of(vec![sym("a"), sym("b")]),
                sym("b"),
            ]),
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
        Var::push_thread_bindings(vec![
            (CURRENT_NS.clone(), Object::Namespace(ns)),
        ]);
        let r = body();
        Var::pop_thread_bindings();
        r
    }

    #[test]
    #[should_panic(expected = "Too many arguments to def")]
    fn def_with_four_args_panics() {
        with_fresh_ns("test.def.four", || {
            let form = list_of(vec![
                sym("def"),
                sym("x"),
                Object::Long(1),
                Object::Long(2),
            ]);
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
}
