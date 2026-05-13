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

fn is_number(bits: u64, _ids: HeapTypeIds) -> bool {
    // Long: NanBox-encoded as a non-tagged f64 (round-trips integer).
    // Double: ditto. Both have `nanbox_tag` returning None.
    nanbox_tag(bits).is_none()
}

fn is_boolean(bits: u64, _ids: HeapTypeIds) -> bool {
    matches!(nanbox_tag(bits), Some(crate::runtime::TAG_BOOL))
}

fn is_character(_bits: u64, _ids: HeapTypeIds) -> bool {
    // We don't model java.lang.Character yet — no character literals,
    // no Char value. Always false until we add them.
    false
}

fn is_var(_bits: u64, _ids: HeapTypeIds) -> bool {
    // Vars don't currently flow through the runtime as values (deref
    // happens at the `cljvm_var_deref` extern; the resulting value is
    // whatever the Var holds). Add a Var heap type to make this true.
    false
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
    push(&mut v, "clojure.lang.IPersistentMap", &[], is_persistent_map, None);
    push(&mut v, "clojure.lang.IPersistentVector", &[], is_persistent_vector, None);
    push(&mut v, "clojure.lang.IPersistentSet", &[], is_persistent_set, None);
    push(&mut v, "clojure.lang.IFn", &[], is_ifn, None);
    push(&mut v, "clojure.lang.IObj", &[], is_iobj, None);
    push(&mut v, "clojure.lang.IMeta", &[], is_imeta, None);
    push(&mut v, "clojure.lang.Symbol", &[], is_symbol, None);
    push(&mut v, "clojure.lang.Keyword", &[], is_keyword, None);
    push(&mut v, "java.lang.String", &["String"], is_string, None);
    push(&mut v, "java.lang.Number", &["Number"], is_number, None);
    push(&mut v, "java.lang.Boolean", &["Boolean"], is_boolean, None);
    push(&mut v, "java.lang.Character", &["Character"], is_character, None);
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

fn always_false(_bits: u64, _ids: HeapTypeIds) -> bool { false }

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
