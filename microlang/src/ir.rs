//! The High IR: the compiler-internal form a `CodeSpace` executes.
//!
//! Crucially, this is *not* the surface syntax. The surface is `Val` (code is
//! data). `analyze` lowers a `Val` into `Ir`. The toolkit has no macro system of
//! its own — a frontend that wants macros expands the `Val` tree before handing
//! it to `analyze` (see the `mclj` frontend). Arithmetic primitives get their own node
//! (`Prim`) so the value-model fast path is a first-class lowering, the way
//! `dyn_add` is in the real toolkit — not a call into an opaque builtin.

use std::sync::Arc;

use crate::value::Sym;

/// Index into the runtime constant pool. Literals are indirected through the
/// pool (which is a GC root) so `Ir` itself holds NO heap pointers — a moving
/// collector could not rewrite pointers embedded in an immutable `Arc<Ir>`.
pub type ConstId = u32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Prim {
    Add,
    Sub,
    Mul,
    Lt,
    Eq,
    /// `(%quot a b)` -> integer quotient, truncated toward zero (Clojure `quot`).
    Quot,
    /// `(%rem a b)` -> integer remainder; sign follows the dividend (`rem`).
    Rem,
    /// `(%mod a b)` -> integer modulo; sign follows the divisor (`mod`).
    Mod,
    /// `(%div a b)` -> division. Exact integer result when it divides evenly;
    /// otherwise float division (there is no Ratio type). Integer `/0` errors.
    Div,
    /// `(%str-cat a b)` -> concatenation of two strings. Neutral string op (no
    /// frontend-specific formatting); the value-to-string logic lives in the
    /// frontend library.
    StrCat,
    /// `(%str-of x)` -> a value's string form for `str`: a string is returned
    /// RAW (unquoted), any other atom via the neutral printer. Records/lists are
    /// formatted by the frontend, so this is only called on leaf values.
    StrOf,
    /// `(%make-array n)` -> a fresh mutable array of `n` nil slots. The substrate
    /// for in-language persistent data structures (ClojureScript-style), just as
    /// `%cell` is but sized rather than element-listed.
    MakeArray,
    /// `(%aclone a)` -> a shallow copy of a mutable array (for trie path-copying).
    AClone,
    /// `(%apush a x)` -> push `x` onto the end of a mutable array, GROWING it in
    /// place; returns the array. Backs cljs `(array)` + `.push`/`.add`.
    ArrPush,
    /// `(%ashift a)` -> remove and return the FIRST element of a mutable array,
    /// shrinking it in place (cljs `.shift`); nil if empty.
    ArrShift,
    /// `(%aclear a)` -> truncate a mutable array to length 0 in place; returns it.
    ArrClear,
    /// Bitwise ops on integers, for trie index math and HAMT bitmaps.
    BitAnd,
    BitOr,
    BitXor,
    /// `(%bit-shl x n)` / `(%bit-shr x n)` -> logical shifts (operands are the
    /// non-negative indices/hashes the persistent structures use).
    BitShl,
    BitShr,
    /// `(%bit-count x)` -> population count (number of set bits); HAMT indexing.
    BitCount,
    /// `(%register-fields type-sym (f0 f1 …))` -> record the field-name order of a
    /// record type, so `(.-field x)` can resolve a name to a slot index. Generic
    /// record reflection the frontend uses to support ClojureScript-style field
    /// access on `deftype` instances. Returns nil.
    RegisterFields,
    /// `(%field-by-name x 'field)` -> the value of `x`'s field named `field`,
    /// resolved through the registry keyed by `x`'s type tag.
    FieldByName,
    /// `(%field-names x)` -> list of the registered field-name symbols of x's type.
    FieldNames,
    /// `(%make-record 'Type (v0 v1 …))` -> a record of that type with those fields.
    MakeRecord,
    /// `(%hash x)` -> a 32-bit content hash of any value (ints by value, strings/
    /// symbols/keywords/chars by content, collections structurally, nil=0). The
    /// hashing a HAMT needs; deterministic but not tied to any host's exact hash.
    Hash,
    List,
    Cons,
    First,
    Rest,
    IsNil,
    Println,
    /// `(%print x)` -> write `x`'s str-form to stdout with NO trailing newline.
    /// The no-newline sibling of `Println`; the in-language `print`/`pr` families
    /// build on it.
    Print,
    /// Force a garbage collection. A safepoint you can place explicitly, so the
    /// GC-during-macro hazard is deterministic to demonstrate.
    Gc,
    /// `(type-of x)` -> a symbol naming the runtime type of ANY value: a
    /// record's own `type_id`, or a built-in tag (`List`/`Vector`/`String`/
    /// `Char`/`Long`/`Double`/`Boolean`/`Symbol`/`Fn`/`nil`). The general
    /// reflection hook a dynamic frontend needs to dispatch outside records.
    TypeOf,
    /// `(record 'Type f0 f1 ...)` -> a record value.
    Record,
    /// `(nfields r)` -> the number of fields of a record (0 for non-records). A
    /// reflection hook a frontend uses to store optional trailing data (e.g.
    /// metadata) on a record without changing how its leading fields read.
    NFields,
    /// `(throw x)` -> abort with `x`'s printed form. (A catchable `try` is a
    /// separate, larger feature; this is the unconditional-abort primitive.)
    Throw,
    /// `(%spawn f)` -> run the thunk `f` (a 0-arg closure) on a fresh OS thread
    /// that SHARES this runtime's heap/globals/interner, returning a `Future`.
    /// Backend-handled (needs to invoke a closure on the child), like `Gc`.
    Spawn,
    /// `(%await fut)` -> block until the future's thread finishes; its value.
    Await,
    /// `(%atom-new x)` -> a fresh atom holding `x`.
    AtomNew,
    /// `(%atom-get a)` -> the atom's current value (atomic load).
    AtomGet,
    /// `(%atom-set a v)` -> store `v` unconditionally; returns `v`.
    AtomSet,
    /// `(%atom-cas a old new)` -> atomically set to `new` iff current == `old`;
    /// returns true on success. The primitive `swap!` retries on.
    AtomCas,
    /// `(field r i)` -> the i-th field of a record.
    Field,
    /// `(%callec f)` — call `f` with a fresh escape continuation. Backend-handled
    /// (needs to invoke a closure and catch a non-local exit), like `Gc`.
    CallEc,
    /// `(string-length s)` -> the character count of a string.
    StrLen,
    /// `(char->integer c)` -> the Unicode scalar value of a char.
    CharToInt,
    /// `(integer->char n)` -> the char with Unicode scalar value `n`.
    IntToChar,
    /// `(vector e ...)` -> a fresh vector of the arguments.
    Vector,
    /// `(vector-ref v i)` -> the i-th element.
    VectorRef,
    /// `(vector-set! v i x)` -> set the i-th element to `x`; returns nil.
    VectorSet,
    /// `(vector-length v)` -> the element count.
    VectorLen,
    /// `(values e ...)` -> a multiple-values packet.
    Values,
    /// `(%values->list v)` -> the list of values in a packet (a lone non-packet
    /// value becomes a one-element list). Bridges `values` to `apply`.
    ValuesToList,
    /// `(apply f a ... lst)` -> apply `f` to the leading args followed by the
    /// elements of the final list. Requires a backend that can invoke closures
    /// (intercepted by the `CekMachine`), like `%callcc`.
    Apply,
    /// `(%eq a b)` — object identity (`eq?`/`eqv?`): equal iff the encoded bits
    /// are equal. For immediates that is value equality; for heap values it is
    /// pointer identity. (Contrast `Eq`, which is structural `equal?`.)
    Identical,
    /// `(%keyword name-sym)` -> THE canonical keyword object for that name.
    /// Keywords are INTERNED (as in Clojure and ClojureScript), so every
    /// construction of `:foo` — reader literal, `::foo` resolution, or a
    /// runtime `(keyword "foo")` — must come through here or identity breaks.
    Keyword,
    /// `(%callcc f)` — full call-with-current-continuation. Only the stackless
    /// `CekMachine` supports it (the continuation is a first-class, multi-shot,
    /// re-installable value); host-stack tiers cannot.
    CallCc,
    /// `(%reset body)` — install a continuation delimiter (prompt) and evaluate
    /// `body` under it. A NATIVE delimited-control primitive; only the stackless
    /// `CekMachine` supports it.
    Reset,
    /// `(%shift f)` — capture the continuation from here up to the nearest
    /// enclosing `%reset`, reify it as a COMPOSABLE (re-delimited, multi-shot)
    /// procedure, and apply `f` to it under a re-established prompt. Native
    /// delimited control; `CekMachine` only.
    Shift,

    // ── dynamic vars (`^:dynamic` + `binding`) ───────────────────────────────
    // A per-thread stack of (sym, value) bindings on the runtime, so a dynamic
    // var reads its innermost thread-local binding (or its root global). All are
    // ordinary `rt.prim` ops, so every tier — including the JIT — runs them.
    /// `(%dyn-get 'v)` -> the innermost thread-local binding of `v`, or its root
    /// global value if unbound. Emitted for a reference to a `^:dynamic` var.
    DynGet,
    /// `(%dyn-set 'v x)` -> mutate the innermost thread-local binding of `v` to
    /// `x` (`set!` on a dynamic var); errors if `v` has no active binding.
    DynSet,
    /// `(%dyn-mark)` -> push a delimiter onto the dynamic stack; returns nil. The
    /// `binding` desugar places one before its binds so `%dyn-unwind` pops exactly
    /// this scope's bindings (however many were installed before a throw).
    DynMark,
    /// `(%dyn-bind 'v x)` -> push a thread-local binding `v = x`; returns nil.
    DynBind,
    /// `(%dyn-unwind)` -> pop the dynamic stack back through the last `%dyn-mark`
    /// (inclusive). Run in a `finally`, so bindings unwind even on a throw.
    DynUnwind,

    // ── first-class vars (reflective global access by symbol) ────────────────
    // A frontend Var is a thin handle over a global's SYMBOL; these read/write the
    // global table by that sym, so `#'x`/deref/`alter-var-root` all work without
    // adding indirection to the ordinary (compiled) global-reference path.
    /// `(%global-get 'sym)` -> the global's value; THROWS (catchable) if unbound.
    GlobalGet,
    /// `(%global-set 'sym x)` -> set the global to `x` (creating/rebinding its
    /// root); returns `x`.
    GlobalSet,
    /// `(%global-bound? 'sym)` -> whether the global currently has a value.
    GlobalBound,
    /// `(%sym-name 'a/b)` -> the NAME part as a string (`"b"`; whole sym if no `/`).
    SymName,
    /// `(%sym-ns 'a/b)` -> the NAMESPACE part as a string (`"a"`), or nil if none.
    SymNs,
    /// `(%var-flags 'sym)` -> the var's flag bits as an int (dynamic|private|macro).
    VarFlags,
    /// `(%ns-interns 'ns)` -> a list of the fully-qualified syms interned in `ns`.
    NsInterns,
    /// `(%all-ns)` -> a list of all registered namespace-name symbols.
    AllNs,
    /// `(%method-types 'method)` -> a list of the type-name symbols that have a
    /// concrete impl registered for `method` in the dispatch registry (excluding
    /// the `-protocol-default` sentinel). Reflection over protocol extensions,
    /// used by `satisfies?`/`extends?`/`extenders`.
    MethodTypes,
    /// `(%method-has-type? method-sym ty-sym)` -> is `ty` registered for that
    /// protocol method? ONE lookup in the dispatch table.
    ///
    /// `MethodTypes` answers the same question by locking the table, scanning
    /// EVERY (method, type) pair in the whole registry, and allocating a list —
    /// per call. `satisfies?` did that once per protocol method and then walked
    /// the result with lazy-seq closures, which cost ~2.4µs and made every
    /// `instance?` against a host interface (what core.match emits, several per
    /// match) pathological. The table is keyed on exactly this pair.
    MethodHasType,
    /// `(%read-string s)` -> read the FIRST datum from string `s` (the reader as a
    /// runtime op). Routes through the frontend eval-bridge.
    ReadString,
    /// `(%eval form)` -> compile & run `form` (a datum) in the current namespace,
    /// returning its value. Routes through the frontend eval-bridge.
    Eval,
    /// `(%macroexpand-1 form)` -> expand `form` by one macro step (or return it
    /// unchanged). Routes through the frontend eval-bridge.
    MacroExpand1,
    /// `(%numerator r)` / `(%denominator r)` -> the numerator / denominator of a
    /// Ratio (an integer `x` is `x/1`).
    Numerator,
    Denominator,
    /// `(%bigint? x)` -> is `x` a heap-boxed integer (beyond the immediate fixnum
    /// range)? Best-effort `bigint?` (this tower auto-promotes, so there is no
    /// distinct BigInt type — a "big" integer is simply a boxed one).
    BigIntP,
    /// `(%to-long x)` -> truncate any number toward zero to an integer (backs
    /// `long`/`int`; float→int, ratio→quotient, int→itself).
    ToLong,
    /// `(%symbol "a/b")` -> the interned symbol for a string (reverse of `%str-of`).
    SymbolOf,
    /// `(%var-arglists 'sym)` -> the var's captured `:arglists` datum, or nil.
    VarArglists,
    /// `(%str->chars s)` -> a list of the string's characters. THE string-
    /// introspection primitive: with it (plus `str`/`str-cat` to rebuild), all of
    /// `clojure.string`, regex, and a reader can be written in the language.
    StrChars,
    /// `(%str->bytes s)` -> a fresh mutable array of the string's UTF-8 bytes as
    /// SIGNED ints (-128..=127, the JVM's byte), so byte-level protocol code
    /// (bencode, wire formats) can be written in the language.
    StrToBytes,
    /// `(%bytes->str arr)` -> the string for an array of signed byte ints
    /// (UTF-8, invalid sequences replaced — java.lang.String's behavior).
    BytesToStr,
    /// `(%tcp-listen port)` -> a listener handle (int). Binds 127.0.0.1:port
    /// (port 0 picks a free port; read it back with `%tcp-local-port`).
    TcpListen,
    /// `(%tcp-accept lh)` -> a connected-stream handle. BLOCKS; safe from any
    /// spawned thread (I/O happens outside the handle-registry lock).
    TcpAccept,
    /// `(%tcp-read h)` -> the next byte (unsigned 0..=255), or -1 at EOF /
    /// on a closed or errored connection. BLOCKS.
    TcpRead,
    /// `(%tcp-write h arr)` -> write an array of signed byte ints, flushing.
    /// Returns nil; throws (catchably, via panic) on a broken connection.
    TcpWrite,
    /// `(%tcp-close h)` -> close + drop a listener or stream handle. nil.
    TcpClose,
    /// `(%tcp-local-port h)` -> the local port a listener is bound to.
    TcpLocalPort,
    /// `(%err-print x)` -> write `x`'s str-form to STDERR, no newline. The
    /// stderr sibling of `Print`; `*err*`'s default writer.
    ErrPrint,
    /// `(%current-ns)` -> the current namespace's name symbol, via the eval
    /// bridge (namespace state is frontend policy; this is the reflection hook).
    CurrentNs,
    /// `(%nanos)` -> monotonic nanoseconds since an arbitrary process-local
    /// origin (the shape of `System/nanoTime`): elapsed-time measurement, not
    /// wall-clock time.
    Nanos,
    /// `(%pow base exp)` -> `base ** exp` as an f64, IEEE-754 exactly as
    /// `Math/pow` (and Rust's `powf`) define it: overflow gives ##Inf, and
    /// `(%pow 9 0.5)` is 3.0. Real libraries need genuine floating-point pow —
    /// meander compares an overflowing `(Math/pow m n)` against ##Inf to decide
    /// whether a search space is finite — and it cannot be faked with repeated
    /// multiplication (that has no answer for a fractional exponent).
    Pow,

    // ── optimizer-introduced fixnum specializations ──────────────────────────
    // These are produced ONLY by the `optimize` nanopass (never by `analyze`).
    // `FxAdd/FxSub/FxMul/FxLt/FxEq` mean "same as `Add/Sub/Mul/Lt/Eq`, but the
    // operands have been PROVEN to be immediate fixnums" (by a dominating
    // `AllFixnum` guard the pass inserts). Semantically identical to the checked
    // op — every interpreter tier lowers them to the very same `prim` — so a
    // backend that ignores the distinction is still correct. Only the JIT reads
    // it, to skip the per-op tag check (it still keeps the overflow → promotion
    // path, since two fixnums can still overflow to a bignum).
    FxAdd,
    FxSub,
    FxMul,
    FxLt,
    FxEq,
    /// `(%pv-conj pv x)` / `(%pv-nth pv i)` / `(%pv-assoc pv i x)` — the persistent
    /// vector's tail+trie operations, implemented natively over the `'PVec` record
    /// (`cnt shift root tail`, arrays are `Obj::Vector`). One native call replaces
    /// the ~20 interpreted helper calls the in-language trie code compiled to.
    PvConj,
    PvNth,
    PvAssoc,
    /// `(%lazy-realize! ls v)` — cache a LazySeq's forced value IN PLACE: set the
    /// `'LazySeq` record's field 0 = `v` and field 1 = `true`, then return `v`. Lets
    /// a LazySeq be one record (mutated on realize) instead of a record + two mutable
    /// `%cell` arrays — 1 allocation per lazy step instead of 3. GC-safe (the heap is
    /// non-relocating; the mutation happens between safepoints).
    LazyRealize,
    /// `(%range-fill start end step)` -> a fresh array (`Obj::Vector`) holding up to
    /// 32 fixnums `start, start+step, start+2*step, ...`, stopping before `end`
    /// (`step > 0`: while `< end`; `step < 0`: while `> end`) or after 32 elements,
    /// whichever comes first. One native call fills a whole range chunk instead of
    /// 32 interpreted `%cell-set!` calls — the chunk producer for `range`.
    RangeFill,
    /// `(%hamt-assoc root key val)` -> `[new-root, added?]` (a 2-elem array; see
    /// `Runtime::hamt_map_assoc`). Native port of PersistentHashMap's
    /// `-inode-assoc` trie descent (BitmapIndexedNode/ArrayNode/HashCollisionNode).
    HamtAssoc,
    /// `(%hamt-lookup root key not-found)` -> the value, or `not-found`.
    HamtLookup,
    /// `(%hamt-without root key)` -> the new root (or `nil` if now empty), or
    /// `root` itself (bit-identical) if `key` was not present.
    HamtWithout,
    /// `(%str-join-arr arr sep)` -> ONE fresh string joining every element of
    /// `arr` (already-stringified, via a Rust `String::push_str` loop) with
    /// `sep` between them. Replaces the O(N^2) `%str-cat`-chain join every
    /// `str`/`apply str`/`clojure.string/join`/collection-printing path used
    /// (each intermediate concat touched a string as long as everything
    /// built so far) with one O(total length) native pass.
    StrJoinArr,
    /// `(%str-cmp a b)` -> -1 / 0 / 1 comparing two strings. Rust byte-wise
    /// `str::cmp` equals code-point order for all valid UTF-8, which is exactly
    /// what the in-language char-by-char `-str<` computed — one native call
    /// instead of building two char lists and walking them per comparison
    /// (`sort`/`sort-by` on strings called that O(len) helper per comparison).
    StrCmp,
    /// `(%pv-conj-chunk pv arr off end)` -> the PersistentVector `pv` with
    /// `arr[off..end]` conj'd on, the whole run done in ONE native call (a Rust
    /// `pv_conj` loop) instead of a `%pv-conj` FFI per element. The vec/mapv/
    /// into-[]/filterv build path (chunk-scanned) uses it — one call per 32-elem
    /// chunk. Vectors are ORDERED, so there is no print-order/type ambiguity to
    /// preserve (unlike a native map batch-build).
    PvConjChunk,
    /// `(%pv-from-array arr)` -> a PersistentVector holding `arr`'s elements,
    /// built BOTTOM-UP (leaves then parent levels) in one native pass — O(n),
    /// with NO per-element tail-array clone (unlike repeated conj, which is
    /// O(32n)). The trie is internally consistent (shift == depth); exact shape
    /// need not match incremental conj since nth/seq/count/pop all key off shift.
    PvFromArray,
    /// `(%apush-chunk arr src off end)` -> append `src[off..end]` to the growable
    /// array `arr` in one native call (a Rust extend), returning `arr`. Lets a
    /// chunked seq be collected into a flat array a whole chunk at a time.
    ApushChunk,
    /// Stage F3 TRANSIENTS — ownership-stamped in-place builders (see
    /// `Runtime::fresh_session` for the design). `(%tv-new pv)` -> a
    /// `'TransientVector [session cnt shift root tail meta]` whose tail is an
    /// owned 32-capacity array; `(%tv-conj! tv x)` appends IN PLACE (tail
    /// push, or an editable trie spill 1/32 of the time); `(%tv-assoc! tv n
    /// x)` writes in place through editable nodes; `(%tv-nth tv n)` reads;
    /// `(%tv-persistent! tv)` invalidates the session (O(1)) and returns a
    /// `'PersistentVector` sharing the structure.
    TvNew,
    TvConj,
    TvAssoc,
    TvNth,
    TvPop,
    TvPersistent,
    /// `(%tam-new pam)` -> `'TransientArrayMap [session arr cnt meta]` (owned
    /// flat kv array); `(%tam-assoc! tam k v)` updates/appends in place —
    /// past the 8-pair threshold it PROMOTES to (and returns) a
    /// `'TransientHashMap` carrying the same session, exactly like cljs'
    /// TransientArrayMap (callers must use the return value, the transient
    /// contract); `(%tam-dissoc! tam k)` removes in place (order-preserving);
    /// `(%tam-persistent! tam)` -> `'PersistentArrayMap` (insertion order —
    /// the small-map conformance point — preserved).
    TamNew,
    TamAssoc,
    TamDissoc,
    TamPersistent,
    /// `(%thm-new phm)` -> `'TransientHashMap [session root cnt has-nil?
    /// nil-val meta]`; `(%thm-assoc! thm k v)` edits session-owned trie nodes
    /// in place (copy-on-first-touch otherwise); `(%thm-dissoc! thm k)`
    /// removes (persistent trie op; root swapped in place); `(%thm-persistent!
    /// thm)` -> `'PersistentHashMap`.
    ThmNew,
    ThmAssoc,
    ThmDissoc,
    ThmPersistent,
    /// `(%sort-arr arr)` -> a FRESH array of `arr`'s elements sorted by the
    /// DEFAULT total order, natively — but only when the elements are
    /// homogeneous fixnums or homogeneous strings (the overwhelmingly common
    /// `sort` inputs; code-point string order matches `%str-cmp`). Returns
    /// nil for anything else — the caller falls back to the in-language
    /// comparator merge sort. Zero comparator callbacks.
    SortArr,
    /// `(%multifn f g ...)` — build a MULTI-ARITY function value from closure
    /// arguments: each fixed-arity closure is selectable by its own param
    /// count; at most one variadic closure serves every higher/unmatched count.
    /// The frontend's multi-arity `fn` desugars to per-clause closures + this.
    MultiFnNew,
    /// `(%all-fixnum? a b ...)` — true iff EVERY argument is an immediate fixnum.
    /// The guard the specializer places at a lambda's entry; when it holds, the
    /// body's `Fx*` ops are valid. On the JIT it lowers to a single combined
    /// tag test; on the interpreter tiers it is a normal predicate (and the
    /// result only picks between two equivalent bodies, so it can't be wrong).
    AllFixnum,
}

impl Prim {
    /// The checked arithmetic op a fixnum-specialized `Fx*` op stands for
    /// (identity for everything else). Interpreter tiers use this to give `Fx*`
    /// exactly the semantics of the base op.
    pub fn dechecked(self) -> Prim {
        match self {
            Prim::FxAdd => Prim::Add,
            Prim::FxSub => Prim::Sub,
            Prim::FxMul => Prim::Mul,
            Prim::FxLt => Prim::Lt,
            Prim::FxEq => Prim::Eq,
            other => other,
        }
    }
}

/// Where a closure's captured value comes from, in the terms of the ENCLOSING
/// activation at closure-creation time: a slot of the enclosing frame, or a
/// capture of the enclosing closure (for transitive capture). Produced by the
/// `flatten` closure-conversion pass; frontends never build these directly.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CapSrc {
    Slot(u16),
    Cap(u16),
}

#[derive(Clone, PartialEq)]
pub enum Ir {
    /// A literal, held in the constant pool (no embedded heap pointer).
    Const(ConstId),
    /// A quoted datum, held in the constant pool.
    Quote(ConstId),
    /// Lexical variable, resolved to a slot at analyze time. Frontends produce
    /// chain-scoped references (`up` frames out, slot `idx`); the `flatten`
    /// pass rewrites every reference to the FLAT form (`up == 0`, idx = a slot
    /// of the single per-call activation frame), which is the only shape the
    /// execution tiers accept.
    Local { up: u16, idx: u16 },
    /// Read slot `idx` of the CURRENT closure's capture array (a value copied
    /// at closure-creation time). Only the `flatten` pass produces these.
    Capture(u16),
    /// Global variable: resolved through the Var table at RUN time, so a
    /// reference can precede the definition (late binding).
    Global(Sym),
    /// Assign an existing lexical slot (`set!` on a local). Returns the value.
    SetLocal { up: u16, idx: u16, val: Box<Ir> },
    /// Assign an existing global Var. Returns the value.
    SetGlobal { name: Sym, val: Box<Ir> },
    If(Box<Ir>, Box<Ir>, Box<Ir>),
    Do(Vec<Ir>),
    /// `def`: bind a global to the value of `init`.
    Def {
        name: Sym,
        init: Box<Ir>,
    },
    /// `let*`: binding inits in order (each occupies the next slot of a single
    /// frame and can see earlier ones), then the body. FRONTEND-ONLY: the
    /// `flatten` pass rewrites every `Let` into `SetLocal` stores against the
    /// function-level activation frame; tiers reject a surviving `Let` loudly.
    Let(Vec<Ir>, Box<Ir>),
    /// A closure. Frontends emit `nslots: 0, captures: []` placeholders; the
    /// `flatten` pass fills them in: `nslots` is the size of the ONE flat
    /// activation frame a call allocates (params, rest arg, every let/catch
    /// slot), and `captures` says which enclosing values to COPY into the
    /// closure at creation time (flat closures — no environment chain).
    Lambda {
        nparams: usize,
        variadic: bool,
        nslots: u16,
        captures: Vec<CapSrc>,
        body: Arc<Ir>,
    },
    Call(Box<Ir>, Vec<Ir>),
    Prim(Prim, Vec<Ir>),
    /// `(defmethod name Type impl)`: register an impl for `(name, Type)`.
    DefMethod {
        name: Sym,
        ty: Sym,
        imp: Box<Ir>,
    },
    /// A polymorphic call site: `(method recv args...)`. `site` is a stable id
    /// used as the inline-cache key; the dispatch strategy resolves it.
    Dispatch {
        site: usize,
        method: Sym,
        args: Vec<Ir>,
    },
    /// `(.-field obj)` — read a record field BY NAME, with a per-site inline
    /// cache. `site` keys a `(type, index)` cache: a monomorphic access resolves
    /// the field name to a slot index once (a scan), then reuses it. A field
    /// layout is fixed per type, so the cache only needs a type-tag check.
    FieldGet {
        site: usize,
        field: Sym,
        obj: Box<Ir>,
    },
    /// Structured exception handling. Evaluate `body`; if it unwinds with a
    /// `Throw`, bind the thrown value in a fresh one-slot frame and evaluate
    /// `catch` (when present). `finally` always runs, for its effect, on both the
    /// normal and the unwinding path. A general control construct (like `If`), not
    /// tied to any one frontend; only stack-based tiers (TreeWalk) implement it.
    Try {
        body: Box<Ir>,
        /// The catch handler. Frontends compile it so the thrown value reads as
        /// `Local{up:0, idx:0}` in a fresh 1-slot frame; the `flatten` pass
        /// re-homes that binding to activation slot `cslot` of the SAME frame.
        catch: Option<Box<Ir>>,
        finally: Option<Box<Ir>>,
        /// The activation slot the thrown value is stored into before the catch
        /// body runs. Assigned by `flatten` (0 until then).
        cslot: u16,
    },
}
