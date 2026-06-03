//! Runtime Class objects — the values that satisfy `(instance? c x)`.
//!
//! Java Clojure leans on `java.lang.Class` as the runtime witness for
//! `instance?` and `^Type` hints. We model the analogue as a small
//! enum (`ClassId`) over the foundational types our heap knows how to
//! recognize. Each id carries:
//!   * a fully-qualified Clojure-class name (`clojure.lang.ISeq`,
//!     `java.lang.String`, …) plus any short alias the reader accepts
//!     (`String`, `Number`, …)
//!   * a predicate that takes a NanBox-encoded value + the runtime
//!     `HeapTypeIds` and returns whether the value is an instance.
//!
//! The compile-time entry point is [`lookup`] — `analyze_symbol`
//! checks unresolved symbols against the registered names and emits a
//! Class literal when one matches. The runtime entry point is
//! [`is_instance`], called by `cljvm_inst_isInstance`.
//!
//! Adding a new class is a one-place edit to `register_classes` below.

use std::sync::OnceLock;

use crate::runtime::HeapTypeIds;

/// Tag identifying a registered runtime Class. Assigned by registration
/// order in [`register_classes`]; the `u16` is what gets baked into a
/// Class heap cell's Raw64 slot so the runtime can dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClassId(pub u16);

/// Static metadata about one registered class — the JVM-style name(s)
/// and the predicate used by `instance?`. Lives in a process-global
/// table; never mutated after `register_classes` runs.
#[derive(Debug)]
pub struct HostClassInfo {
    pub id: ClassId,
    /// Canonical name (`clojure.lang.ISeq`, `java.lang.String`).
    pub name: &'static str,
    /// Optional short aliases the reader can spell (`String`,
    /// `Number`). `None` for classes that have no shorthand.
    pub aliases: &'static [&'static str],
    /// Run at `(. c (isInstance x))` time. Returns whether `x_bits` is
    /// an instance of this class. The predicate gets `HeapTypeIds`
    /// (the per-Session type-id mapping from `Compiler::new`) so it
    /// can compare against the heap's runtime layout.
    pub predicate: fn(x_bits: u64, ids: HeapTypeIds) -> bool,
    /// Optional constructor — called by `(new ClassName args)`. Gets
    /// the args' NanBox bits (excluding the Class itself) and the
    /// runtime type ids; returns the new instance as a NanBox handle.
    /// Classes without a constructor (interfaces, etc.) leave this
    /// `None` — invoking `new` on them panics with a clear message.
    pub ctor: Option<fn(args: &[u64], ids: HeapTypeIds) -> u64>,
}

/// Registry of all classes the runtime knows about. Indexed by
/// `ClassId.0` — first registration is id 0, second is id 1, etc.
static CLASSES: OnceLock<Vec<HostClassInfo>> = OnceLock::new();

fn classes() -> &'static [HostClassInfo] {
    CLASSES.get_or_init(register_classes)
}

/// Look up a class by name (or alias). Returns the `HostClassInfo`
/// with the matching ClassId. Used by `analyze_symbol` to recognize
/// class-name symbols and by tests.
pub fn lookup(name: &str) -> Option<&'static HostClassInfo> {
    classes()
        .iter()
        .find(|c| c.name == name || c.aliases.contains(&name))
}

/// Look up by id — used at runtime by `cljvm_inst_isInstance` after
/// decoding the Class heap cell's Raw64 slot.
pub fn by_id(id: ClassId) -> &'static HostClassInfo {
    classes()
        .get(id.0 as usize)
        .unwrap_or_else(|| panic!("clojure-jvm: HostClass: no class with id {}", id.0))
}

/// Top-level dispatch for `(instance? c x)` — caller has already
/// decoded the Class object's id off its heap cell.
pub fn is_instance(class_id: ClassId, x_bits: u64, ids: HeapTypeIds) -> bool {
    (by_id(class_id).predicate)(x_bits, ids)
}

// ── Predicates ─────────────────────────────────────────────────────────
//
// Each predicate inspects the NanBox tag (and where appropriate, the
// type_id of a TAG_PTR heap cell) to decide membership. Kept tiny and
// self-contained so adding a new class is mechanical.

const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const TAG_MASK: u64 = 0x0003_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

const TAG_NIL: u32 = 0;
const TAG_BOOL: u32 = 1;
const TAG_PTR: u32 = 2;
const TAG_FN: u32 = 3;

fn nanbox_tag(bits: u64) -> Option<u32> {
    if (bits & FULL_MASK) != TAG_PATTERN {
        return None;
    }
    Some(((bits & TAG_MASK) >> 48) as u32)
}

fn nanbox_payload(bits: u64) -> u64 {
    bits & PAYLOAD_MASK
}

fn heap_type_id(bits: u64) -> Option<usize> {
    if !matches!(nanbox_tag(bits), Some(TAG_PTR)) {
        return None;
    }
    let raw = nanbox_payload(bits) as *const u8;
    if raw.is_null() {
        return None;
    }
    Some(unsafe { raw.cast::<u16>().read_unaligned() } as usize)
}

fn is_seq(bits: u64, ids: HeapTypeIds) -> bool {
    // Java's `RT.seq` returns nil for empty/nil, so `(instance? ISeq nil)`
    // is false. A non-empty seq today means a Cons heap cell.
    heap_type_id(bits) == Some(ids.cons)
}

fn is_persistent_map(bits: u64, ids: HeapTypeIds) -> bool {
    heap_type_id(bits) == Some(ids.map)
}

fn is_persistent_vector(bits: u64, ids: HeapTypeIds) -> bool {
    heap_type_id(bits) == Some(ids.vector)
}

/// `(instance? clojure.lang.IDeref x)` — the deref-able heap cells we
/// support: `Reduced` (via `@`) and `Delay`.
fn is_ideref(bits: u64, ids: HeapTypeIds) -> bool {
    matches!(heap_type_id(bits), Some(t) if t == ids.reduced || t == ids.delay)
}

fn is_persistent_set(bits: u64, ids: HeapTypeIds) -> bool {
    heap_type_id(bits) == Some(ids.set)
}

fn is_string(bits: u64, ids: HeapTypeIds) -> bool {
    heap_type_id(bits) == Some(ids.string)
}

fn is_symbol(bits: u64, ids: HeapTypeIds) -> bool {
    heap_type_id(bits) == Some(ids.symbol)
}

fn is_keyword(bits: u64, ids: HeapTypeIds) -> bool {
    heap_type_id(bits) == Some(ids.keyword)
}

fn is_ifn(bits: u64, _ids: HeapTypeIds) -> bool {
    // TAG_FN values are direct fn handles. Closures are TAG_PTR with a
    // dedicated type_id, but we don't have that id in HeapTypeIds yet
    // — when closures grow IObj support add it here.
    matches!(nanbox_tag(bits), Some(TAG_FN))
}

fn is_iobj(bits: u64, ids: HeapTypeIds) -> bool {
    // Anything that can carry metadata. Today only Cons does (per the
    // heap-meta-slot work in compiler.rs). Extend as Symbol/Vector/Map
    // etc. grow meta slots.
    heap_type_id(bits) == Some(ids.cons)
}

fn is_imeta(bits: u64, ids: HeapTypeIds) -> bool {
    // IMeta is the read side; same set of receivers as IObj for now.
    is_iobj(bits, ids)
}

fn is_number(bits: u64, ids: HeapTypeIds) -> bool {
    // Double: NanBox-encoded as a non-tagged f64 (`nanbox_tag` → None).
    // Long: a boxed heap cell (TAG_PTR with the `long` type_id). (Until the
    // boxed-Long flip lands, integer-valued floats also satisfy this via the
    // None branch.)
    nanbox_tag(bits).is_none() || heap_type_id(bits) == Some(ids.long)
}

/// Integer-typed predicate: any number whose NanBox round-trips as an
/// integer-valued double. Used as the `instance?` impl for the various
/// JVM integer types (Integer / Long / Short / Byte / BigInteger /
/// clojure.lang.BigInt) since we represent all of them as Long-shaped
/// f64 NanBoxes.
fn is_integer(bits: u64, ids: HeapTypeIds) -> bool {
    // A boxed Long is always an integer.
    if heap_type_id(bits) == Some(ids.long) {
        return true;
    }
    // PRE-FLIP: integer-valued native floats still count (ints are floats).
    // POST-FLIP (Task A Step 2): delete this branch — a native float is a
    // genuine `double` and `(integer? 3.0)` must be false.
    if nanbox_tag(bits).is_some() {
        return false;
    }
    let f = f64::from_bits(bits);
    f.is_finite() && f == (f as i64) as f64
}

fn is_double(bits: u64, _ids: HeapTypeIds) -> bool {
    if nanbox_tag(bits).is_some() {
        return false;
    }
    let f = f64::from_bits(bits);
    f.is_finite() && f != (f as i64) as f64
}

fn is_boolean(bits: u64, _ids: HeapTypeIds) -> bool {
    matches!(nanbox_tag(bits), Some(crate::runtime::TAG_BOOL))
}

fn is_character(_bits: u64, _ids: HeapTypeIds) -> bool {
    // We don't model java.lang.Character yet — no character literals,
    // no Char value. Always false until we add them.
    false
}

fn is_var(bits: u64, ids: HeapTypeIds) -> bool {
    heap_type_id(bits) == Some(ids.var)
}

fn is_namespace(bits: u64, ids: HeapTypeIds) -> bool {
    heap_type_id(bits) == Some(ids.namespace)
}

// ── Registration ───────────────────────────────────────────────────────
//
// Order matters: a class's `ClassId` is its index in this Vec. Once a
// class is registered, never reorder — the id is baked into compiled
// code and saved Class heap cells.

fn register_classes() -> Vec<HostClassInfo> {
    let mut v = Vec::new();
    let mut next_id: u16 = 0;
    let mut push = |v: &mut Vec<HostClassInfo>,
                    name: &'static str,
                    aliases: &'static [&'static str],
                    predicate: fn(u64, HeapTypeIds) -> bool,
                    ctor: Option<fn(&[u64], HeapTypeIds) -> u64>| {
        v.push(HostClassInfo {
            id: ClassId(next_id),
            name,
            aliases,
            predicate,
            ctor,
        });
        next_id += 1;
    };

    push(&mut v, "clojure.lang.ISeq", &[], is_seq, None);
    push(
        &mut v,
        "clojure.lang.IPersistentMap",
        &[],
        is_persistent_map,
        None,
    );
    push(
        &mut v,
        "clojure.lang.IPersistentVector",
        &[],
        is_persistent_vector,
        None,
    );
    push(
        &mut v,
        "clojure.lang.IPersistentSet",
        &[],
        is_persistent_set,
        None,
    );
    push(&mut v, "clojure.lang.IFn", &[], is_ifn, None);
    push(&mut v, "clojure.lang.IObj", &[], is_iobj, None);
    push(&mut v, "clojure.lang.IMeta", &[], is_imeta, None);
    push(&mut v, "clojure.lang.Symbol", &[], is_symbol, None);
    push(&mut v, "clojure.lang.Keyword", &[], is_keyword, None);
    push(&mut v, "java.lang.String", &["String"], is_string, None);
    push(&mut v, "java.lang.Number", &["Number"], is_number, None);
    push(&mut v, "java.lang.Boolean", &["Boolean"], is_boolean, None);
    // JVM integer family — all represented as our Long-shaped f64.
    push(&mut v, "java.lang.Integer", &["Integer"], is_integer, None);
    push(&mut v, "java.lang.Long", &["Long"], is_integer, None);
    push(&mut v, "java.lang.Short", &["Short"], is_integer, None);
    push(&mut v, "java.lang.Byte", &["Byte"], is_integer, None);
    push(
        &mut v,
        "java.math.BigInteger",
        &["BigInteger"],
        is_integer,
        None,
    );
    push(&mut v, "clojure.lang.BigInt", &[], is_integer, None);
    push(&mut v, "java.lang.Double", &["Double"], is_double, None);
    push(&mut v, "java.lang.Float", &["Float"], is_double, None);
    push(
        &mut v,
        "java.math.BigDecimal",
        &["BigDecimal"],
        is_double,
        None,
    );
    push(&mut v, "clojure.lang.Ratio", &[], always_false, None);
    push(&mut v, "java.util.Map$Entry", &[], always_false, None);
    push(
        &mut v,
        "clojure.lang.MapEntry",
        &["MapEntry"],
        always_false,
        None,
    );
    push(&mut v, "clojure.lang.IRecord", &[], always_false, None);
    push(&mut v, "clojure.lang.Sequential", &[], is_seq, None);
    push(&mut v, "clojure.lang.Reversible", &[], always_false, None);
    push(&mut v, "clojure.lang.Counted", &[], always_false, None);
    push(&mut v, "clojure.lang.Indexed", &[], always_false, None);
    push(&mut v, "clojure.lang.Associative", &[], always_false, None);
    push(&mut v, "clojure.lang.IPending", &[], always_false, None);
    push(
        &mut v,
        "clojure.lang.IBlockingDeref",
        &[],
        always_false,
        None,
    );
    push(&mut v, "clojure.lang.IDeref", &[], is_ideref, None);
    push(&mut v, "clojure.lang.IRef", &[], always_false, None);
    push(&mut v, "clojure.lang.Ref", &[], always_false, None);
    push(&mut v, "clojure.lang.Atom", &[], always_false, None);
    push(&mut v, "clojure.lang.Agent", &[], always_false, None);
    push(&mut v, "clojure.lang.Volatile", &[], always_false, None);
    push(&mut v, "clojure.lang.Namespace", &[], is_namespace, None);
    push(&mut v, "clojure.lang.IRecord", &[], always_false, None);
    push(&mut v, "clojure.lang.IReduce", &[], always_false, None);
    push(&mut v, "clojure.lang.IReduceInit", &[], always_false, None);
    push(&mut v, "clojure.lang.IKVReduce", &[], always_false, None);
    push(
        &mut v,
        "clojure.lang.ITransientCollection",
        &[],
        always_false,
        None,
    );
    push(&mut v, "clojure.lang.MultiFn", &[], always_false, None);
    push(&mut v, "clojure.lang.Named", &[], always_false, None);
    push(
        &mut v,
        "clojure.lang.PersistentList",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentVector",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentHashMap",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentArrayMap",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentHashSet",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentTreeMap",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentTreeSet",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentQueue",
        &[],
        always_false,
        None,
    );
    push(&mut v, "clojure.lang.IteratorSeq", &[], always_false, None);
    push(&mut v, "clojure.lang.ArraySeq", &[], always_false, None);
    push(
        &mut v,
        "java.lang.Iterable",
        &["Iterable"],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.Iterator",
        &["Iterator"],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.Collection",
        &["Collection"],
        always_false,
        None,
    );
    push(&mut v, "java.util.List", &["List"], always_false, None);
    push(&mut v, "java.util.Map", &["Map"], always_false, None);
    push(&mut v, "java.util.Set", &["Set"], always_false, None);
    push(
        &mut v,
        "java.lang.CharSequence",
        &["CharSequence"],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.Comparable",
        &["Comparable"],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.Throwable",
        &["Throwable"],
        always_false,
        None,
    );
    push(&mut v, "java.lang.Class", &["Class"], always_false, None);
    push(&mut v, "java.lang.Object", &["Object"], always_false, None);
    push(
        &mut v,
        "clojure.lang.Reduced",
        &[],
        is_reduced_class,
        Some(reduced_ctor),
    );
    push(&mut v, "clojure.lang.Reductions", &[], always_false, None);
    push(&mut v, "clojure.lang.AFn", &[], always_false, None);
    push(&mut v, "clojure.lang.AFunction", &[], always_false, None);
    push(&mut v, "clojure.lang.RestFn", &[], always_false, None);
    push(
        &mut v,
        "clojure.lang.LineNumberingPushbackReader",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PushbackReader",
        &[],
        always_false,
        None,
    );
    push(&mut v, "clojure.lang.Compiler", &[], always_false, None);
    push(&mut v, "clojure.lang.LispReader", &[], always_false, None);
    push(&mut v, "clojure.lang.Util", &[], always_false, None);
    push(&mut v, "clojure.lang.Var$Frame", &[], always_false, None);
    push(
        &mut v,
        "clojure.lang.PersistentArrayMap$TransientArrayMap",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentHashMap$TransientHashMap",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentHashSet$TransientHashSet",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentVector$TransientVector",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.ITransientVector",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.ITransientMap",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.ITransientSet",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.ITransientAssociative",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.IEditableCollection",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.IPersistentStack",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.IExceptionInfo",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.ExceptionInfo",
        &[],
        always_false,
        None,
    );
    push(&mut v, "clojure.lang.IHashEq", &[], always_false, None);
    push(
        &mut v,
        "clojure.lang.MapEquivalence",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.PersistentArrayMap",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.LazilyPersistentVector",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.LockingTransaction",
        &[],
        always_false,
        None,
    );
    push(&mut v, "clojure.lang.Numbers$Ops", &[], always_false, None);
    push(
        &mut v,
        "clojure.lang.Numbers$LongOps",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.Numbers$DoubleOps",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.Numbers$RatioOps",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.Numbers$BigIntOps",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.Numbers$BigDecimalOps",
        &[],
        always_false,
        None,
    );
    push(&mut v, "clojure.lang.IDrop", &[], always_false, None);
    push(&mut v, "java.io.Reader", &[], always_false, None);
    push(&mut v, "java.io.Writer", &[], always_false, None);
    push(&mut v, "java.io.PrintWriter", &[], always_false, None);
    push(&mut v, "java.io.StringReader", &[], always_false, None);
    push(&mut v, "java.io.StringWriter", &[], always_false, None);
    push(&mut v, "java.io.BufferedReader", &[], always_false, None);
    push(&mut v, "java.io.IOException", &[], always_false, None);
    push(&mut v, "java.io.File", &[], always_false, None);
    push(&mut v, "java.lang.System", &[], always_false, None);
    push(&mut v, "java.lang.ClassLoader", &[], always_false, None);
    push(&mut v, "java.lang.Thread", &[], always_false, None);
    push(&mut v, "java.lang.Runtime", &[], always_false, None);
    push(
        &mut v,
        "java.lang.UnsupportedOperationException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.NullPointerException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.IndexOutOfBoundsException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.ArrayIndexOutOfBoundsException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.StringIndexOutOfBoundsException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.ClassCastException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.ArithmeticException",
        &[],
        always_false,
        None,
    );
    push(&mut v, "java.lang.AssertionError", &[], always_false, None);
    push(
        &mut v,
        "java.lang.NumberFormatException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.SecurityException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.InterruptedException",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.OutOfMemoryError",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.lang.StackOverflowError",
        &[],
        always_false,
        None,
    );
    push(&mut v, "java.lang.Error", &[], always_false, None);
    push(
        &mut v,
        "java.util.concurrent.TimeUnit",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.Executor",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.ExecutorService",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.Executors",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.Future",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.CountDownLatch",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.Callable",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.atomic.AtomicReference",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.atomic.AtomicLong",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.atomic.AtomicInteger",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.locks.Lock",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.locks.ReentrantLock",
        &[],
        always_false,
        None,
    );
    push(
        &mut v,
        "java.util.concurrent.locks.ReentrantReadWriteLock",
        &[],
        always_false,
        None,
    );
    push(&mut v, "java.util.HashMap", &[], always_false, None);
    push(&mut v, "java.util.HashSet", &[], always_false, None);
    push(&mut v, "java.util.LinkedList", &[], always_false, None);
    push(&mut v, "java.util.ArrayList", &[], always_false, None);
    push(&mut v, "java.util.TreeMap", &[], always_false, None);
    push(&mut v, "java.util.TreeSet", &[], always_false, None);
    push(&mut v, "java.util.UUID", &[], always_false, None);
    push(&mut v, "java.util.regex.Pattern", &[], always_false, None);
    push(&mut v, "java.util.regex.Matcher", &[], always_false, None);
    push(&mut v, "java.util.Random", &[], always_false, None);
    push(&mut v, "java.util.Properties", &[], always_false, None);
    push(&mut v, "java.lang.reflect.Method", &[], always_false, None);
    push(
        &mut v,
        "java.lang.reflect.Constructor",
        &[],
        always_false,
        None,
    );
    push(&mut v, "java.lang.reflect.Field", &[], always_false, None);
    push(
        &mut v,
        "java.lang.reflect.Modifier",
        &[],
        always_false,
        None,
    );
    push(&mut v, "java.lang.reflect.Array", &[], always_false, None);
    push(
        &mut v,
        "java.lang.StringBuilder",
        &["StringBuilder"],
        is_string_builder,
        Some(string_builder_ctor),
    );
    push(
        &mut v,
        "clojure.lang.ChunkBuffer",
        &["ChunkBuffer"],
        is_chunk_buffer,
        Some(chunk_buffer_ctor),
    );
    push(&mut v, "clojure.lang.IChunk", &["IChunk"], is_i_chunk, None);
    push(
        &mut v,
        "clojure.lang.IChunkedSeq",
        &["IChunkedSeq"],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.ChunkedCons",
        &["ChunkedCons"],
        always_false,
        None,
    );
    push(
        &mut v,
        "clojure.lang.LazySeq",
        &["LazySeq"],
        is_lazy_seq,
        Some(lazy_seq_ctor),
    );
    push(
        &mut v,
        "clojure.lang.Delay",
        &["Delay"],
        is_delay,
        Some(delay_ctor),
    );
    push(
        &mut v,
        "java.lang.Character",
        &["Character"],
        is_character,
        None,
    );
    push(&mut v, "clojure.lang.Var", &[], is_var, None);
    // Exception classes — bootstrap of `clojure.core` constructs these
    // via `(IllegalArgumentException. "msg")` etc. inside guard
    // branches that throw on bad input. Predicates always return false
    // for now (we don't model Exception instances as a distinct
    // runtime type — the constructor returns a Map representation).
    push(
        &mut v,
        "java.lang.IllegalArgumentException",
        &["IllegalArgumentException"],
        always_false,
        Some(make_exception_ctor("java.lang.IllegalArgumentException")),
    );
    push(
        &mut v,
        "java.lang.RuntimeException",
        &["RuntimeException"],
        always_false,
        Some(make_exception_ctor("java.lang.RuntimeException")),
    );
    push(
        &mut v,
        "java.lang.Exception",
        &["Exception"],
        always_false,
        Some(make_exception_ctor("java.lang.Exception")),
    );
    push(
        &mut v,
        "java.lang.IllegalStateException",
        &["IllegalStateException"],
        always_false,
        Some(make_exception_ctor("java.lang.IllegalStateException")),
    );

    v
}

fn always_false(_bits: u64, _ids: HeapTypeIds) -> bool {
    false
}

fn is_string_builder(bits: u64, ids: HeapTypeIds) -> bool {
    use crate::runtime::{nanbox_payload, nanbox_tag};
    match nanbox_tag(bits) {
        Some(2 /* TAG_PTR */) => {
            let p = nanbox_payload(bits) as *const u8;
            if p.is_null() {
                return false;
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            tid == ids.string_builder
        }
        _ => false,
    }
}

/// `(new StringBuilder s)` ctor. Forwards to the runtime extern, which
/// handles Arc allocation + Session rooting + heap-cell write.
fn string_builder_ctor(args: &[u64], _ids: HeapTypeIds) -> u64 {
    if args.len() != 1 {
        panic!(
            "clojure-jvm: StringBuilder ctor expects 1 String arg, got {}",
            args.len()
        );
    }
    unsafe { crate::runtime::cljvm_inst_StringBuilder_new1(args[0]) }
}

fn is_chunk_buffer(bits: u64, ids: HeapTypeIds) -> bool {
    use crate::runtime::{nanbox_payload, nanbox_tag};
    match nanbox_tag(bits) {
        Some(2) => {
            let p = nanbox_payload(bits) as *const u8;
            if p.is_null() {
                return false;
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            tid == ids.chunk_buffer
        }
        _ => false,
    }
}

fn is_i_chunk(bits: u64, ids: HeapTypeIds) -> bool {
    use crate::runtime::{nanbox_payload, nanbox_tag};
    match nanbox_tag(bits) {
        Some(2) => {
            let p = nanbox_payload(bits) as *const u8;
            if p.is_null() {
                return false;
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            tid == ids.i_chunk
        }
        _ => false,
    }
}

fn chunk_buffer_ctor(args: &[u64], _ids: HeapTypeIds) -> u64 {
    if args.len() != 1 {
        panic!(
            "clojure-jvm: ChunkBuffer ctor expects 1 capacity arg, got {}",
            args.len()
        );
    }
    unsafe { crate::runtime::cljvm_inst_ChunkBuffer_new1(args[0]) }
}

fn is_lazy_seq(bits: u64, ids: HeapTypeIds) -> bool {
    use crate::runtime::{nanbox_payload, nanbox_tag};
    matches!(nanbox_tag(bits), Some(2)) && {
        let p = nanbox_payload(bits) as *const u8;
        !p.is_null() && unsafe { p.cast::<u16>().read_unaligned() } as usize == ids.lazy_seq
    }
}

fn is_delay(bits: u64, ids: HeapTypeIds) -> bool {
    use crate::runtime::{nanbox_payload, nanbox_tag};
    matches!(nanbox_tag(bits), Some(2)) && {
        let p = nanbox_payload(bits) as *const u8;
        !p.is_null() && unsafe { p.cast::<u16>().read_unaligned() } as usize == ids.delay
    }
}

fn lazy_seq_ctor(args: &[u64], _ids: HeapTypeIds) -> u64 {
    if args.len() != 1 {
        panic!(
            "clojure-jvm: LazySeq ctor expects 1 fn arg, got {}",
            args.len()
        );
    }
    unsafe { crate::runtime::cljvm_inst_LazySeq_new1(args[0]) }
}

/// `(instance? clojure.lang.Reduced x)` predicate.
fn is_reduced_class(bits: u64, _ids: HeapTypeIds) -> bool {
    crate::runtime::is_reduced(bits)
}

/// `(clojure.lang.Reduced. x)` — wrap `x` in a Reduced cell.
fn reduced_ctor(args: &[u64], _ids: HeapTypeIds) -> u64 {
    if args.len() != 1 {
        panic!(
            "clojure-jvm: Reduced ctor expects 1 arg, got {}",
            args.len()
        );
    }
    unsafe { crate::runtime::cljvm_inst_Reduced_new1(args[0]) }
}

fn delay_ctor(args: &[u64], _ids: HeapTypeIds) -> u64 {
    if args.len() != 1 {
        panic!(
            "clojure-jvm: Delay ctor expects 1 fn arg, got {}",
            args.len()
        );
    }
    unsafe { crate::runtime::cljvm_inst_Delay_new1(args[0]) }
}

/// Build a constructor closure for an exception class. The resulting
/// fn allocates a `PersistentHashMap` carrying `:class <name>` plus
/// `:message <msg>` (when one was supplied), and returns the
/// NanBox-encoded heap handle. Roots the new Arc on the active
/// Session so the heap pointer stays valid.
fn make_exception_ctor(class_name: &'static str) -> fn(&[u64], HeapTypeIds) -> u64 {
    // We can't capture `class_name` in a fn pointer, so generate one
    // per-class via a small dispatch table.
    match class_name {
        "java.lang.IllegalArgumentException" => exc_ctor_iae,
        "java.lang.RuntimeException" => exc_ctor_re,
        "java.lang.Exception" => exc_ctor_e,
        "java.lang.IllegalStateException" => exc_ctor_ise,
        _ => unreachable!("make_exception_ctor: unhandled {class_name}"),
    }
}

fn exc_ctor_iae(args: &[u64], ids: HeapTypeIds) -> u64 {
    build_exception("java.lang.IllegalArgumentException", args, ids)
}
fn exc_ctor_re(args: &[u64], ids: HeapTypeIds) -> u64 {
    build_exception("java.lang.RuntimeException", args, ids)
}
fn exc_ctor_e(args: &[u64], ids: HeapTypeIds) -> u64 {
    build_exception("java.lang.Exception", args, ids)
}
fn exc_ctor_ise(args: &[u64], ids: HeapTypeIds) -> u64 {
    build_exception("java.lang.IllegalStateException", args, ids)
}

/// Allocate `{:class <class_name> :message <args[0]>}` and return its
/// NanBox handle. Used by every exception constructor.
fn build_exception(class_name: &str, args: &[u64], _ids: HeapTypeIds) -> u64 {
    use crate::lang::keyword::Keyword;
    use crate::lang::object::Object;
    use crate::lang::persistent_hash_map::PersistentHashMap;
    use crate::lang::symbol::Symbol;

    let class_kw = Object::Keyword(Keyword::intern(Symbol::intern("class")));
    let msg_kw = Object::Keyword(Keyword::intern(Symbol::intern("message")));
    let class_obj = Object::String(std::sync::Arc::new(class_name.to_string()));
    let msg_obj = match args.first() {
        Some(bits) => crate::runtime::any_bits_to_object(*bits, crate::runtime::heap_type_ids()),
        None => Object::Nil,
    };
    let m = PersistentHashMap::create_pairs(vec![(class_kw, class_obj), (msg_kw, msg_obj)]);
    // Allocate a Map heap cell pointing at this Arc. Mirrors
    // `cljvm_rt_assoc`'s allocation+root path.
    let runtime_ids = crate::runtime::heap_type_ids();
    let raw_arc = std::sync::Arc::as_ptr(&m) as u64;
    crate::lang::compiler::with_active_session_root_map(m);
    let new_raw = dynlang::gc::gc_alloc_thunk(runtime_ids.map as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: build_exception: gc_alloc returned null");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    crate::runtime::nanbox_ptr(new_raw)
}
