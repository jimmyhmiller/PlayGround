//! `clojure.lang` package — one Rust module per Java file in
//! `~/Documents/Code/open-source/clojure/src/jvm/clojure/lang/`.
//!
//! Modules are added incrementally as Compiler.java pulls them in. Everything
//! that has a Java analog stays named after its Java source file (e.g.
//! `Symbol.java` → `symbol.rs`, `IFn.java` → `i_fn.rs`).

// java.lang.Object stand-in — Compiler.java references `Object` everywhere.
// Lives at the package root because it's not from clojure.lang/.
pub mod object;

// Identity types.
pub mod named;
pub mod symbol;
pub mod keyword;

// Var / Namespace.
pub mod var;
pub mod namespace;

// Metadata / meta protocols (Symbol/Var/etc. implement these).
pub mod i_meta;
pub mod i_obj;
pub mod i_reference;
pub mod a_reference;

// Function abstractions.
pub mod i_fn;
pub mod a_fn;
pub mod a_function;

// Collection protocols & impls actually touched by Compiler.java.
pub mod seqable;
pub mod sequential;
pub mod counted;
pub mod i_seq;
pub mod i_persistent_collection;
pub mod i_persistent_list;
pub mod i_persistent_vector;
pub mod i_persistent_map;
pub mod i_persistent_stack;
pub mod i_lookup;
pub mod i_map_entry;
pub mod associative;

pub mod a_seq;
pub mod cons;
pub mod persistent_list;
pub mod persistent_vector;
pub mod persistent_hash_map;
pub mod persistent_hash_set;
pub mod persistent_tree_map;
pub mod persistent_tree_set;

// Runtime utilities.
pub mod rt;
pub mod util;
pub mod numbers;
pub mod reflector;
pub mod tuple;

// Reader — Compiler.java calls into LispReader and LineNumberingPushbackReader.
pub mod lisp_reader;
pub mod line_numbering_pushback_reader;

// Exception types raised by the compiler.
pub mod arity_exception;
pub mod i_exception_info;
pub mod exception_info;

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
