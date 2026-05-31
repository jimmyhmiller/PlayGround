;; clojure-jvm prelude. Loaded by Session::new BEFORE any user code or
;; forked core.clj.  This file owns the small set of `def`s that real
;; Clojure resolves via Java statics (dynamic compile-time vars, agent,
;; auto-imported java.lang.* classes) plus a handful of forward stubs
;; for functions defined later in core.clj that earlier forms reference.
;;
;; Every form here is loaded via the same `eval_form` pipeline as user
;; code — adding a new var means editing this file, not Rust source.

;; ── Compile-time control vars ─────────────────────────────────────
;; Each is a Var in clojure.core initialized to a sane default. Without
;; them, metadata analysis (e.g. :inline forms in upstream core) fails
;; with "Unable to resolve symbol".
(def ^:dynamic *unchecked-math* false)
(def ^:dynamic *warn-on-reflection* false)
(def ^:dynamic *compile-files* false)
(def ^:dynamic *assert* true)

;; *agent* is a Java-defined var (clojure.lang.Agent.LOCAL). We don't
;; have agent semantics; declaring the var is enough for
;; `(binding [*agent* a] ...)` to analyze.
(def ^:dynamic *agent* nil)

;; Other Java-defined dynamic vars referenced from clojure.core.
;; *ns* is intentionally NOT defined here — it's already managed via
;; `super::rt::CURRENT_NS` in the Rust runtime; redeclaring it would bind
;; it to nil and break namespace resolution.
(def ^:dynamic *out* nil)
(def ^:dynamic *in* nil)
(def ^:dynamic *err* nil)
(def ^:dynamic *print-readably* true)
(def ^:dynamic *flush-on-newline* true)
(def ^:dynamic *print-meta* false)
(def ^:dynamic *print-dup* false)
(def ^:dynamic *print-length* nil)
(def ^:dynamic *print-level* nil)
(def ^:dynamic *print-namespace-maps* false)
(def ^:dynamic *read-eval* true)
(def ^:dynamic *file* "NO_SOURCE_PATH")
(def ^:dynamic *source-path* "NO_SOURCE_PATH")
(def ^:dynamic *command-line-args* nil)
(def ^:dynamic *math-context* nil)
(def ^:dynamic *data-readers* nil)
(def ^:dynamic *default-data-reader-fn* nil)
(def ^:dynamic *suppress-read* nil)
(def ^:dynamic *fn-loader* nil)
(def ^:dynamic *use-context-classloader* true)
(def ^:dynamic *allow-unresolved-vars* false)
(def ^:dynamic *compile-path* nil)

;; Vars normally bound by upstream's (let [props ...] (def ...)) top-level
;; bootstrap forms that we skip via SKIP_BYTE_POS. Predefining as nil so
;; subsequent forms that reference them analyze cleanly.
(def ^:dynamic *clojure-version* nil)
(def ^:dynamic *loaded-libs* nil)
(def ^:dynamic *pending-paths* nil)
(def ^:dynamic *compiler-options* nil)

;; ── Bootstrap fns normally provided by clojure.lang.RT ────────────
;; Real Clojure provides these as Java statics; we wire each through a
;; runtime extern (declared in compiler.rs `host_methods`).
;;
;; `in-ns` switches the current namespace. Upstream returns the new
;; Namespace; we return nil — the loader only needs the side effect.
(def in-ns (fn* [ns-sym] (. clojure.lang.Namespace (setCurrent ns-sym))))

;; `load-file` reads and evaluates a file. We don't have a filesystem
;; loader; the Var must exist (core.clj attaches doc meta via
;; `#'load-file`) but calling it errors.
(def load-file
  (fn* [_path]
    (throw "clojure-jvm: load-file is not supported (no filesystem loader)")))

;; ── Forward stubs ─────────────────────────────────────────────────
;; Protocol/class predicates that are defined later in core.clj as defns
;; but referenced earlier. Stub each as a fn returning nil/false so
;; analyze succeeds at the first reference; the real definitions later
;; in core.clj will overwrite these Vars.
(def satisfies? (fn* [_ _] false))
(def find-protocol-impl (fn* [_ _] nil))
(def find-protocol-method (fn* [_ _ _] nil))
(def extends? (fn* [_ _] false))
(def extenders (fn* [_] nil))
(def supers (fn* [_] nil))
(def bases (fn* [_] nil))
(def ancestors (fn* [_] nil))
(def descendants (fn* [_] nil))

;; ── Mocked Java classes ───────────────────────────────────────────
;; Defined so `^Foo` type hints and `(into-array Foo ...)` /
;; `(instance? Foo x)` references in forked core.clj resolve at analyze
;; time. Each is a string sentinel, NOT a real Class object. Code paths
;; that actually depend on Class identity (reflection, typed arrays)
;; will misbehave or throw at runtime — these are placeholders, not
;; implementations.
(def Array "#<MOCKED java.lang.reflect.Array>")
(def AssertionError "#<MOCKED java.lang.AssertionError>")
(def BigDecimal "#<MOCKED java.math.BigDecimal>")
(def BigInteger "#<MOCKED java.math.BigInteger>")
(def BlockingQueue "#<MOCKED java.util.concurrent.BlockingQueue>")
(def Boolean "#<MOCKED java.lang.Boolean>")
(def Byte "#<MOCKED java.lang.Byte>")
(def Callable "#<MOCKED java.util.concurrent.Callable>")
(def Character "#<MOCKED java.lang.Character>")
(def Class "#<MOCKED java.lang.Class>")
(def ClassCastException "#<MOCKED java.lang.ClassCastException>")
(def ClassNotFoundException "#<MOCKED java.lang.ClassNotFoundException>")
(def Double "#<MOCKED java.lang.Double>")
(def Eduction "#<MOCKED clojure.core.Eduction>")
(def Exception "#<MOCKED java.lang.Exception>")
(def ExceptionInfo "#<MOCKED clojure.lang.ExceptionInfo>")
(def Float "#<MOCKED java.lang.Float>")
(def IExceptionInfo "#<MOCKED clojure.lang.IExceptionInfo>")
(def IllegalAccessError "#<MOCKED java.lang.IllegalAccessError>")
(def IllegalArgumentException "#<MOCKED java.lang.IllegalArgumentException>")
(def IllegalStateException "#<MOCKED java.lang.IllegalStateException>")
(def Inst "#<MOCKED clojure.core.Inst>")
(def Integer "#<MOCKED java.lang.Integer>")
(def LinkedBlockingQueue "#<MOCKED java.util.concurrent.LinkedBlockingQueue>")
(def Long "#<MOCKED java.lang.Long>")
(def Math "#<MOCKED java.lang.Math>")
(def Number "#<MOCKED java.lang.Number>")
(def NumberFormatException "#<MOCKED java.lang.NumberFormatException>")
(def Object "#<MOCKED java.lang.Object>")
(def Runtime "#<MOCKED java.lang.Runtime>")
(def RuntimeException "#<MOCKED java.lang.RuntimeException>")
(def Short "#<MOCKED java.lang.Short>")
(def StackTraceElement "#<MOCKED java.lang.StackTraceElement>")
(def String "#<MOCKED java.lang.String>")
;; StringBuilder is a fully-supported host class (ctor + append + toString),
;; so it must resolve to the real Class (via the host_class registry), not a
;; mock Var — otherwise `(new StringBuilder …)` decodes the receiver as this
;; string and fails. No mock def here on purpose.
(def System "#<MOCKED java.lang.System>")
(def Thread "#<MOCKED java.lang.Thread>")
(def Throwable "#<MOCKED java.lang.Throwable>")
(def UnsupportedOperationException "#<MOCKED java.lang.UnsupportedOperationException>")
(def Writer "#<MOCKED java.io.Writer>")
