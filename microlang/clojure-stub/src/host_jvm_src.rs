//! host.jvm — the JVM, modeled ENTIRELY in the language. The Rust expander
//! lowers interop *syntax* only (`(.m x)` → a `%dispatch` site on the
//! dot-munged method name; `(C/m …)`/`(C. …)`/`Class` values → calls into the
//! fns here); every class name, method body, static, and inheritance edge is
//! data in `-jvm-registry`, declared with `defclass`. Adding a host class is a
//! library edit, not a compiler edit.
//!
//! Two axes:
//!  * FAST — instance methods are protocol-method entries under dot names
//!    (`.charAt` on tag `String`), emitted as ordinary `-proto-method` forms,
//!    so calls ride the same inline-cached dispatch as protocols. `Object`
//!    registrations are universal fallbacks (the dispatch registry's root).
//!  * REFLECTIVE — `-jvm-registry` maps FQN → a `JvmClass` descriptor record;
//!    Class VALUES are `(record 'Class fqn)`, so `class?` is finally honest
//!    and `Class/forName` misses throw catchably.
//!
//! Everything on the LOAD path (defclass expansion, registration) is
//! prim-style eager code, like core's own bootstrap macros: this file runs at
//! every prelude load, and lazy-seq/hash-map machinery would cost thousands of
//! interpreted calls per class. Descriptors are flat records; the registry and
//! per-class static tables are flat `(k v k v …)` plists scanned with `%num-eq`
//! (reflective ops are cold; loads are O(1) conses).
//!
//! Loaded into `clojure.core` right after the cljs persistent types.

pub const HOST_JVM: &str = r##"
;; ─────────────── registry ───────────────
;; descriptor: (record 'JvmClass simple kind tag protocol pred extends implements ctor statics static-fns)
(defn -jvm-c-simple     [d] (field d 0))  ;; 'String — the simple name symbol
(defn -jvm-c-kind       [d] (field d 1))  ;; :class | :interface | :static
(defn -jvm-c-tag        [d] (field d 2))  ;; runtime type tag its instances carry (or nil)
(defn -jvm-c-protocol   [d] (field d 3))  ;; interface satisfaction protocol (or nil)
(defn -jvm-c-pred       [d] (field d 4))  ;; interface satisfaction predicate (or nil)
(defn -jvm-c-extends    [d] (field d 5))  ;; superclass FQN sym (or nil)
(defn -jvm-c-implements [d] (field d 6))  ;; list of interface FQN syms
(defn -jvm-c-ctor       [d] (field d 7))  ;; constructor fn (or nil = tagged record)
(defn -jvm-c-statics    [d] (field d 8))  ;; ('NAME value …) plist
(defn -jvm-c-static-fns [d] (field d 9))  ;; ('name fn …) plist

(def -jvm-registry (%atom-new nil))   ;; (fqn desc fqn desc …) plist
(def -jvm-tag-index (%atom-new nil))  ;; (tag fqn tag fqn …) plist
(def -jvm-none (record 'JvmNone 0))

;; plist scan; -jvm-none marks ABSENT (a stored value may be nil).
(defn -jvm-plist-get* [l k]
  (loop [l l]
    (if (nil? l)
      -jvm-none
      (if (%num-eq (%first l) k) (%first (%rest l)) (recur (%rest (%rest l)))))))
(defn -jvm-plist-get [l k]
  (let [v (-jvm-plist-get* l k)] (if (identical? v -jvm-none) nil v)))

(defn -jvm-kw? [x k]
  (if (%num-eq (type-of x) 'Keyword) (%num-eq (field x 0) k) false))

(defn -jvm-register! [fqn desc]
  (%atom-set -jvm-registry (%cons fqn (%cons desc (%atom-get -jvm-registry))))
  ;; The tag index answers `(class x)` — only a CONCRETE class may claim a
  ;; runtime tag (interfaces map many-to-one onto representative tags).
  (if (nil? (-jvm-c-tag desc))
    nil
    (if (-jvm-kw? (-jvm-c-kind desc) 'interface)
      nil
      (if (nil? (-jvm-plist-get (%atom-get -jvm-tag-index) (-jvm-c-tag desc)))
        (%atom-set -jvm-tag-index
                   (%cons (-jvm-c-tag desc) (%cons fqn (%atom-get -jvm-tag-index))))
        nil)))
  nil)

(defn -jvm-descriptor [fqn] (-jvm-plist-get (%atom-get -jvm-registry) fqn))

;; "java.lang.String" -> "String" (prim loop — the string library isn't loaded yet).
(defn -jvm-simple-str [s]
  (loop [cs (%str->chars s) acc ""]
    (if (nil? cs)
      acc
      (recur (%rest cs)
             (if (%num-eq (%first cs) \.) "" (%str-cat acc (%str-of (%first cs))))))))

;; ─────────────── defclass ───────────────
;; (defclass java.lang.String
;;   (:tag String) (:extends java.lang.Object) (:implements java.lang.CharSequence)
;;   (:ctor ([] "") ([x] (str x)))
;;   (:method charAt [s i] (nth s i))
;;   (:static-fn valueOf [x] (str x))
;;   (:static CASE_INSENSITIVE_ORDER cmp))
;; Expands to a `-jvm-register!` call (the reflective descriptor) plus one
;; `-proto-method` per :method under the DOT name (the fast dispatch axis).
;; Not a primitive and not deftype: it describes behavior for a type that
;; already exists (deftype creates one). :pred / :protocol let an interface
;; delegate instance? to a predicate (`map?`) or protocol satisfaction.
(defn -jvm-clause [clauses k]
  (loop [cs clauses]
    (if (nil? cs)
      nil
      (if (-jvm-kw? (%first (%first cs)) k) (%first cs) (recur (%rest cs))))))
(defn -jvm-clauses [clauses k]
  (-rev (loop [cs clauses acc nil]
          (if (nil? cs)
            acc
            (recur (%rest cs)
                   (if (-jvm-kw? (%first (%first cs)) k) (%cons (%first cs) acc) acc))))))
;; ((:static PI 3.14) …) -> ('PI 3.14 …); ((:static-fn abs [x] …) …) -> ('abs (fn [x] …) …)
(defn -jvm-static-kvs [ss fn?]
  (-rev (loop [ss ss acc nil]
          (if (nil? ss)
            acc
            (let [c (%first ss)
                  v (if fn? (%cons 'fn (%rest (%rest c))) (%first (%rest (%rest c))))]
              (recur (%rest ss)
                     (%cons v (%cons (list 'quote (%first (%rest c))) acc))))))))
(defn -jvm-method-forms [ms tag]
  (loop [ms ms acc nil]
    (if (nil? ms)
      acc
      (let [c (%first ms)]
        (recur (%rest ms)
               (%cons (list '-proto-method
                            (symbol (%str-cat "." (name (%first (%rest c)))))
                            tag
                            (%cons 'fn (%rest (%rest c))))
                      acc))))))
(defmacro defclass [fqn & clauses]
  (let [second* (fn [c] (%first (%rest c)))
        tagc    (-jvm-clause clauses 'tag)
        tag     (if tagc (second* tagc) nil)
        kindc   (-jvm-clause clauses 'kind)
        protoc  (-jvm-clause clauses 'protocol)
        predc   (-jvm-clause clauses 'pred)
        extc    (-jvm-clause clauses 'extends)
        implc   (-jvm-clause clauses 'implements)
        ctorc   (-jvm-clause clauses 'ctor)
        statics (-jvm-clauses clauses 'static)
        sfns    (-jvm-clauses clauses 'static-fn)
        methods (-jvm-clauses clauses 'method)
        simple  (symbol (-jvm-simple-str (name fqn)))]
    (if (if methods (nil? tag) false)
      (throw (%str-cat "defclass: :method requires a :tag to dispatch on — " (name fqn))))
    (%cons 'do
      (%cons
        (list '-jvm-register! (list 'quote fqn)
          (list 'record (list 'quote 'JvmClass)
                (list 'quote simple)
                (if kindc (second* kindc) :class)
                (if tag (list 'quote tag) nil)
                (if protoc (second* protoc) nil)
                (if predc (second* predc) nil)
                (if extc (list 'quote (second* extc)) nil)
                (if implc (list 'quote (%rest implc)) nil)
                (if ctorc (%cons 'fn (%rest ctorc)) nil)
                (if statics (%cons 'list (-jvm-static-kvs statics false)) nil)
                (if sfns (%cons 'list (-jvm-static-kvs sfns true)) nil)))
        (-jvm-method-forms methods tag)))))

;; ─────────────── the reflective operations ───────────────
(defn -jvm-invoke-static [fqn m & args]
  (let [d (-jvm-descriptor fqn)]
    (if (nil? d)
      (throw (str "No such class: " fqn))
      (let [f (-jvm-plist-get* (-jvm-c-static-fns d) m)]
        (if (identical? f -jvm-none)
          (throw (str "No such static method: " fqn "/" m))
          (apply f args))))))

(defn -jvm-static-member [fqn m]
  (let [d (-jvm-descriptor fqn)]
    (if (nil? d)
      (throw (str "No such class: " fqn))
      (let [v (-jvm-plist-get* (-jvm-c-statics d) m)]
        (if (identical? v -jvm-none)
          (let [f (-jvm-plist-get* (-jvm-c-static-fns d) m)]
            (if (identical? f -jvm-none)
              (throw (str "No such static member: " fqn "/" m))
              (f)))
          v)))))

(defn -jvm-construct [fqn & args]
  (let [d (-jvm-descriptor fqn)]
    (if (nil? d)
      (throw (str "No such class: " fqn))
      (let [c (-jvm-c-ctor d)]
        (if (nil? c)
          ;; default constructor: a record tagged with the class's runtime tag
          (%make-record (or (-jvm-c-tag d) (-jvm-c-simple d)) args)
          (apply c args))))))

(defn -jvm-class-named [fqn]
  (if (-jvm-descriptor fqn) (record 'Class fqn) nil))

(defn -jvm-for-name [n]
  (let [c (-jvm-class-named (symbol n))]
    (if (nil? c) (throw (str "ClassNotFoundException: " n)) c)))

;; A dotted name in VALUE position: a registered class -> its Class record; a
;; js-style `pkg.Class.MEMBER` static ref -> the member; unknown -> the bare
;; simple symbol (the legacy tag-symbol behavior for unregistered names).
(defn -jvm-class-value [fqn simple parent]
  (if (-jvm-descriptor fqn)
    (record 'Class fqn)
    (let [d (-jvm-descriptor parent)]
      (if (nil? d)
        simple
        (let [v (-jvm-plist-get* (-jvm-c-statics d) simple)]
          (if (identical? v -jvm-none)
            (let [f (-jvm-plist-get* (-jvm-c-static-fns d) simple)]
              (if (identical? f -jvm-none) simple f))
            v))))))

(defn -jvm-in-list? [l x]
  (loop [l l]
    (if (nil? l) false (if (%num-eq (%first l) x) true (recur (%rest l))))))

;; Does runtime tag `ty`'s class chain (extends + implements) reach `target`?
(defn -jvm-extends? [ty target]
  (loop [f (-jvm-plist-get (%atom-get -jvm-tag-index) ty)]
    (if (nil? f)
      false
      (if (%num-eq f target)
        true
        (let [d (-jvm-descriptor f)]
          (if (nil? d)
            false
            (if (-jvm-in-list? (-jvm-c-implements d) target)
              true
              (recur (-jvm-c-extends d)))))))))

;; `(instance? pkg.Class x)` — interfaces by :pred / :protocol, classes by tag
;; match or inheritance walk; an UNREGISTERED name is a deftype/dialect record
;; tag: plain tag equality.
(defn -jvm-instance-of? [fqn simple x]
  (let [d (-jvm-descriptor fqn)
        ty (type-of x)]
    (cond
      (nil? d) (%num-eq ty simple)
      (-jvm-c-pred d) (boolean ((-jvm-c-pred d) x))
      (-jvm-c-protocol d) (satisfies? (-jvm-c-protocol d) x)
      :else (if (%num-eq ty (-jvm-c-tag d)) true (-jvm-extends? ty fqn)))))

;; Fully-dynamic instance?: the class is a runtime VALUE — a Class record, a
;; protocol, or a type-tag symbol.
(defn -jvm-instance? [c x]
  (cond
    (class? c) (let [f (field c 0)
                     d (-jvm-descriptor f)]
                 (-jvm-instance-of? f (if d (-jvm-c-simple d) f) x))
    (-protocol? c) (satisfies? c x)
    (symbol? c) (%num-eq (type-of x) c)
    :else (throw (str "instance?: not a class value: " (pr-str c)))))

;; One typed catch clause's test (the roots — Throwable/Exception/Error/Object
;; — are compile-time catch-alls in the expander; this handles specific classes).
(defn -jvm-catch-match? [fqn simple e]
  (-jvm-instance-of? fqn simple e))

;; `(class x)` — the Class record for x's registered class, or a Class wrapping
;; the bare runtime tag for dialect/deftype values (so .getName always answers).
(defn class [x]
  (if (nil? x)
    nil
    (let [f (-jvm-plist-get (%atom-get -jvm-tag-index) (type-of x))]
      (record 'Class (if (nil? f) (type-of x) f)))))

;; ─────────────── java.lang.Class is itself a class ───────────────
(defclass java.lang.Class
  (:tag Class)
  (:method getName [c] (name (field c 0)))
  (:method getSimpleName [c] (-jvm-simple-str (name (field c 0))))
  (:method isArray [c] false)
  (:method isInterface [c] (-jvm-kw? (-jvm-c-kind (-jvm-descriptor (field c 0))) 'interface))
  (:method toString [c] (str "class " (name (field c 0))))
  (:static-fn forName [n] (-jvm-for-name n)))

;; ─────────────── universal defaults (dispatch root = Object) ───────────────
;; `nfields` answers 0 for any non-record, so message extraction is total.
(defn -jvm-message-of [e]
  (cond (string? e) e
        (%lt 0 (nfields e)) (field e 0)
        :else nil))

(defclass java.lang.Object
  (:tag Object)
  (:method withMeta [o m] (-with-meta o m))
  (:method meta [o] (-meta o))
  (:method indexOf [o x] (-index-of o x))
  (:method toString [o] (str o))
  (:method equals [a b] (= a b))
  (:method hashCode [o] (hash o))
  (:method getMessage [e] (-jvm-message-of e)))

;; mutable raw-array ("array-list") methods — cljs host-array interop
(defclass cljs.core.ArrayList
  (:tag Vector)
  (:method isEmpty [a] (-al-empty? a))
  (:method toArray [a] (-array->vec a))
  (:method slice [a] (-array->vec a))
  (:method size [a] (%alength a))
  (:method count [a] (%alength a))
  (:method length [a] (%alength a))
  (:method add [a x] (-al-add! a x))
  (:method push [a x] (-al-add! a x))
  (:method clear [a] (-al-clear! a))
  (:method shift [a] (-al-shift! a)))

;; ─────────────── java.lang ───────────────
(defclass java.lang.String
  (:tag String)
  (:extends java.lang.Object)
  (:implements java.lang.CharSequence)
  (:ctor ([] "") ([x] (str x)))
  (:method length [s] (count s))
  (:method charAt [s i] (nth s i))
  (:method substring ([s b] (subs s b)) ([s b e] (subs s b e)))
  (:method replace [s t r] (clojure.string/replace s t r))
  (:method toString [s] s)
  (:method isEmpty [s] (= 0 (count s)))
  (:static-fn valueOf [x] (str x)))

(defclass java.lang.Math
  (:kind :static)
  (:static PI 3.141592653589793)
  (:static E 2.718281828459045)
  (:static-fn abs [x] (if (neg? x) (- x) x))
  (:static-fn max [a b] (if (< a b) b a))
  (:static-fn min [a b] (if (< a b) a b)))

;; value wrappers / interfaces mapping onto the dialect's native types
(defclass java.lang.CharSequence (:kind :interface) (:tag String) (:pred string?))
(defclass java.lang.Character (:tag Char))
(defclass java.lang.Number (:kind :interface) (:tag Long) (:pred number?))
(defclass java.lang.Long (:tag Long) (:extends java.lang.Number))
(defclass java.lang.Integer (:tag Long) (:extends java.lang.Number))
(defclass java.lang.Short (:tag Long) (:extends java.lang.Number))
(defclass java.lang.Byte (:tag Long) (:extends java.lang.Number))
(defclass java.math.BigInteger (:tag Long) (:extends java.lang.Number))
(defclass clojure.lang.BigInt (:tag Long) (:extends java.lang.Number))
(defclass java.lang.Double (:tag Double) (:extends java.lang.Number))
(defclass java.lang.Float (:tag Double) (:extends java.lang.Number))
(defclass java.math.BigDecimal (:tag Double) (:extends java.lang.Number))
(defclass java.lang.Boolean (:tag Boolean))
(defclass java.util.regex.Pattern (:tag Regex))
(defclass js.RegExp (:tag Regex))

;; ─────────────── throwables ───────────────
(defclass java.lang.Throwable (:tag Throwable))
(defclass java.lang.Exception (:tag Exception) (:extends java.lang.Throwable))
(defclass java.lang.Error (:tag Error) (:extends java.lang.Throwable))
(defclass java.lang.RuntimeException (:tag RuntimeException) (:extends java.lang.Exception))
(defclass java.lang.IllegalArgumentException
  (:tag IllegalArgumentException) (:extends java.lang.RuntimeException))
(defclass java.lang.IllegalStateException
  (:tag IllegalStateException) (:extends java.lang.RuntimeException))
(defclass java.lang.UnsupportedOperationException
  (:tag UnsupportedOperationException) (:extends java.lang.RuntimeException))
(defclass java.lang.IndexOutOfBoundsException
  (:tag IndexOutOfBoundsException) (:extends java.lang.RuntimeException))
(defclass java.lang.ArithmeticException
  (:tag ArithmeticException) (:extends java.lang.RuntimeException))
(defclass java.lang.NullPointerException
  (:tag NullPointerException) (:extends java.lang.RuntimeException))
(defclass java.lang.ClassCastException
  (:tag ClassCastException) (:extends java.lang.RuntimeException))
(defclass java.lang.NumberFormatException
  (:tag NumberFormatException) (:extends java.lang.RuntimeException))
(defclass java.lang.AssertionError (:tag AssertionError) (:extends java.lang.Error))
(defclass java.lang.StackOverflowError (:tag StackOverflowError) (:extends java.lang.Error))
;; cljs host errors (`(js/Error. m)` in .cljc files) — same records as java.lang.Error
(defclass js.Error (:tag Error) (:extends java.lang.Throwable))
(defclass js.TypeError (:tag TypeError) (:extends js.Error))
(defclass js.RangeError (:tag RangeError) (:extends js.Error))

;; ─────────────── clojure.lang ───────────────
(defclass clojure.lang.RT
  (:kind :static)
  (:static-fn cons [x s] (%cons x s))
  (:static-fn first [s] (-rt-first s))
  (:static-fn next [s] (-rt-next s))
  (:static-fn more [s] (-rt-rest s))
  (:static-fn seq [s] (-rt-seq s))
  (:static-fn conj [c x] (-rt-conj c x))
  (:static-fn assoc [m k v] (-rt-assoc m k v)))

(defclass clojure.lang.Symbol (:tag Symbol))
(defclass clojure.lang.Keyword (:tag Keyword))
(defclass clojure.lang.PersistentVector (:tag PersistentVector))
(defclass clojure.lang.IPersistentVector
  (:kind :interface) (:tag PersistentVector) (:pred vector?))
(defclass clojure.lang.APersistentVector
  (:kind :interface) (:tag PersistentVector) (:pred vector?))
(defclass clojure.lang.PersistentArrayMap (:tag PersistentArrayMap))
(defclass clojure.lang.PersistentHashMap (:tag PersistentHashMap))
(defclass clojure.lang.IPersistentMap
  (:kind :interface) (:tag PersistentArrayMap) (:pred map?))
(defclass clojure.lang.APersistentMap
  (:kind :interface) (:tag PersistentArrayMap) (:pred map?))
(defclass clojure.lang.PersistentHashSet (:tag PersistentHashSet))
(defclass clojure.lang.IPersistentSet
  (:kind :interface) (:tag PersistentHashSet) (:pred set?))
(defclass clojure.lang.ISeq (:kind :interface) (:tag List) (:pred seq?))
(defclass clojure.lang.Seqable (:kind :interface) (:tag List) (:pred -seqable?))
(defclass clojure.lang.Cons (:tag List))
(defclass clojure.lang.IPersistentList (:kind :interface) (:tag List) (:pred list?))
(defclass clojure.lang.PersistentList
  (:tag List)
  (:static creator -list))
(defclass clojure.lang.IFn (:kind :interface) (:tag Fn) (:pred ifn?))
(defclass clojure.lang.ILookup (:kind :interface) (:protocol ILookup))

;; ─────────────── dialect-native host types (cljs heritage) ───────────────
;; a map entry IS a `[k v]` vector in this dialect
(defclass cljs.core.MapEntry
  (:ctor ([k v] (vector k v)) ([k v h] (vector k v))))
(defclass cljs.core.PersistentQueue
  (:tag PersistentQueue)
  (:static EMPTY -empty-queue))
"##;
