//! `clojure.lang` package — one Rust module per Java file in
//! `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/`.
//!
//! Modules are added incrementally as Compiler.java pulls them in. Everything
//! that has a Java analog stays named after its Java source file (e.g.
//! `Symbol.java` → `symbol.rs`, `IFn.java` → `i_fn.rs`).

// java.lang.Object stand-in — Compiler.java references `Object` everywhere.
// Lives at the package root because it's not from clojure.lang/.
pub mod object;

// Low-bit tagged value representation (replacing NaN-boxing). See the module
// docs; the migration off NaN-boxing is in progress.
pub mod value_repr;

// Identity types.
pub mod keyword;
pub mod named;
pub mod symbol;

// Var / Namespace.
pub mod namespace;
pub mod var;
pub mod var_roots;

// Metadata / meta protocols (Symbol/Var/etc. implement these).
pub mod a_reference;
pub mod i_meta;
pub mod i_obj;
pub mod i_reference;

// Function abstractions.
pub mod a_fn;
pub mod a_function;
pub mod i_fn;

// Collection protocols & impls actually touched by Compiler.java.
pub mod associative;
pub mod counted;
pub mod i_lookup;
pub mod i_map_entry;
pub mod i_persistent_collection;
pub mod i_persistent_list;
pub mod i_persistent_map;
pub mod i_persistent_stack;
pub mod i_persistent_vector;
pub mod i_seq;
pub mod seqable;
pub mod sequential;

pub mod a_seq;
pub mod cons;
pub mod persistent_hash_map;
pub mod persistent_hash_set;
pub mod persistent_list;
pub mod persistent_tree_map;
pub mod persistent_tree_set;
pub mod persistent_vector;

// Runtime utilities.
pub mod numbers;
pub mod reflector;
pub mod rt;
pub mod tuple;
pub mod util;

// Reader — Compiler.java calls into LispReader and LineNumberingPushbackReader.
pub mod line_numbering_pushback_reader;
pub mod lisp_reader;

// Exception types raised by the compiler.
pub mod arity_exception;
pub mod exception_info;
pub mod i_exception_info;

// JVM-specific support (mostly stubs in our world — we don't emit class files).
pub mod dynamic_class_loader;

// The main event.
pub mod compiler;

// Runtime Class object — name + isInstance predicate. Stand-in for
// `java.lang.Class` as a Clojure value (registered foundational types
// like `clojure.lang.ISeq`, `String`, etc.). Used at compile time by
// `analyze_symbol` to resolve class-name symbols, and at runtime by
// `cljvm_inst_isInstance` to dispatch `instance?` checks.
pub mod host_class;

// User-defined protocols / deftypes / extend-type. The registry of
// user-allocated TypeIds (deftype) and ProtocolIds + the global
// (LogicalTypeId, ProtoMethodId) → FnHandle dispatch table that
// `cljvm_rt_protocol_dispatch` reads.
pub mod user_types;
