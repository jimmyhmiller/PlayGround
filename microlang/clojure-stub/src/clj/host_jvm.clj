;; host.jvm — the JVM, modeled ENTIRELY in the language. The Rust expander
;; lowers interop *syntax* only (`(.m x)` → a `%dispatch` site on the
;; dot-munged method name; `(C/m …)`/`(C. …)`/`Class` values → calls into the
;; fns here); every class name, method body, static, and inheritance edge is
;; data in `-jvm-registry`, declared with `defclass`. Adding a host class is a
;; library edit, not a compiler edit.
;; 
;; Two axes:
;;  * FAST — instance methods are protocol-method entries under dot names
;;    (`.charAt` on tag `String`), emitted as ordinary `-proto-method` forms,
;;    so calls ride the same inline-cached dispatch as protocols. `Object`
;;    registrations are universal fallbacks (the dispatch registry's root).
;;  * REFLECTIVE — `-jvm-registry` maps FQN → a `JvmClass` descriptor record;
;;    Class VALUES are `(record 'Class fqn)`, so `class?` is finally honest
;;    and `Class/forName` misses throw catchably.
;; 
;; Everything on the LOAD path (defclass expansion, registration) is
;; prim-style eager code, like core's own bootstrap macros: this file runs at
;; every prelude load, and lazy-seq/hash-map machinery would cost thousands of
;; interpreted calls per class. Descriptors are flat records; the registry and
;; per-class static tables are flat `(k v k v …)` plists scanned with `%num-eq`
;; (reflective ops are cold; loads are O(1) conses).
;; 
;; Loaded into `clojure.core` right after the cljs persistent types.

;; ─────────────── registry ───────────────
;; descriptor: (record 'JvmClass simple kind tag protocol pred extends
;;                     implements ctor statics static-fns component extend-tags)
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
(defn -jvm-c-component  [d] (field d 10)) ;; array component class FQN (or nil)
;; field 11 = :extend-tags — the CONCRETE tags an `extend-type` on this
;; interface registers against (read at compile time by the Rust expander:
;; jvm_registry_tags). An interface spans several runtime types here (Named =
;; Symbol + Keyword), and this list IS the dispatch-specificity policy.
(defn -jvm-c-extend-tags [d] (field d 11))

(def -jvm-registry (%atom-new nil))   ;; (fqn desc fqn desc …) plist
;; …and an O(1) INDEX over the same data: fqn -> desc.
;;
;; The plist above is the load-path representation (see the header: registration
;; must stay prim-style eager). Scanning it is O(registry), which the header
;; called fine because "reflective ops are cold". core.match proved that false —
;; its expansion tests `(instance? clojure.lang.ILookup x)` several times per
;; `match`, and every one of those scanned all ~80 classes (~340ns each).
;;
;; Maintained INCREMENTALLY — one `assoc` per class at load, not a rebuild per
;; lookup — so the load path pays ~80 assocs total and reads are a single map
;; lookup. `cljs_types` (which defines the map types) loads before this file.
(def -jvm-index (%atom-new nil))
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
  ;; Keep the index in step. A re-registration must WIN, exactly as the plist's
  ;; cons-to-the-front makes it win for a scan.
  (%atom-set -jvm-index (assoc (%atom-get -jvm-index) fqn desc))
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

;; One map lookup, not a scan of the whole registry. (`-jvm-index` is nil until
;; the first class registers; `get` on nil is nil, which is the same "no such
;; class" answer the plist scan gave.)
(defn -jvm-descriptor [fqn] (get (%atom-get -jvm-index) fqn))

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
        compc   (-jvm-clause clauses 'component)
        etagsc  (-jvm-clause clauses 'extend-tags)
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
                (if sfns (%cons 'list (-jvm-static-kvs sfns true)) nil)
                (if compc (list 'quote (second* compc)) nil)
                (if etagsc (list 'quote (%rest etagsc)) nil)))
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

;; Throws a REAL ClassNotFoundException object, not a string: the whole point of
;; `Class/forName` for a library is that the miss is CATCHABLE BY TYPE.
(defn -jvm-for-name [n]
  (let [c (-jvm-class-named (symbol n))]
    (if (nil? c) (throw (record 'ClassNotFoundException (str n))) c)))

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

;; ─────────────── namespaces as values (*ns*) ───────────────
;; `*ns*` compiles to `(-ns-object)` (compile.rs global_ref): a `Namespace`
;; record wrapping the CURRENT ns name from the compiler (via the eval
;; bridge). Read-only v1 — `(binding [*ns* …])` is not supported; switch with
;; `ns`/`in-ns`, which eval'd code handles at the top level.
(defn -ns-object [] (record 'Namespace (%current-ns)))
(defn ns-name [n] (if (%num-eq (type-of n) 'Namespace) (field n 0) n))
(defn all-ns []
  (loop [ns (%all-ns) acc nil]
    (if (nil? ns) (-rev acc) (recur (%rest ns) (%cons (record 'Namespace (%first ns)) acc)))))
(defn find-ns [s]
  (loop [all (%all-ns)]
    (if (nil? all)
      nil
      (if (= (%first all) s) (record 'Namespace s) (recur (%rest all))))))
(defn the-ns [x]
  (cond (%num-eq (type-of x) 'Namespace) x
        (symbol? x) (or (find-ns x) (throw (str "No namespace: " x " found")))
        :else (throw (str "the-ns: not a namespace: " (pr-str x)))))
(extend-type Namespace
  IEquiv (-equiv [a b] (and (%num-eq (type-of b) 'Namespace) (= (field a 0) (field b 0)))))

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
  (:method isArray [c]
    (let [d (-jvm-descriptor (field c 0))]
      (if d (if (nil? (-jvm-c-component d)) false true) false)))
  (:method getComponentType [c]
    (let [d (-jvm-descriptor (field c 0))]
      (if (nil? d)
        nil
        (let [comp (-jvm-c-component d)]
          (if (nil? comp) nil (-jvm-class-named comp))))))
  (:method isInterface [c] (-jvm-kw? (-jvm-c-kind (-jvm-descriptor (field c 0))) 'interface))
  (:method toString [c] (str "class " (name (field c 0))))
  (:static-fn forName [n] (-jvm-for-name n)))

;; Class VALUES compare by the class they name (Byte/TYPE = the component
;; class of a byte array, wherever each record was made).
(extend-type Class
  IEquiv (-equiv [a b] (and (class? b) (= (field a 0) (field b 0)))))

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
  (:method count [o] (count o))
  (:method nth ([o i] (nth o i)) ([o i nf] (nth o i nf)))
  (:method getMessage [e] (-jvm-message-of e)))

;; mutable raw-array ("array-list") methods — cljs host-array interop.
;; `:component java.lang.Byte`: raw arrays answer as byte[] to reflective
;; array checks (`(-> o class .getComponentType (= Byte/TYPE))`, the bencode
;; Object branch) — this dialect's wire code keeps bytes in raw arrays.
(defclass cljs.core.ArrayList
  (:tag Vector)
  (:component java.lang.Byte)
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
  ;; `(String. byte-array charset)` / `(String. byte-array)` decode UTF-8, as
  ;; wire code expects; anything else stringifies.
  (:ctor ([] "")
         ([x] (if (%num-eq (type-of x) 'Vector) (%bytes->str x) (str x)))
         ([b cs] (%bytes->str b)))
  (:method length [s] (count s))
  (:method charAt [s i] (nth s i))
  (:method substring ([s b] (subs s b)) ([s b e] (subs s b e)))
  (:method replace [s t r] (clojure.string/replace s t r))
  (:method getBytes ([s] (%str->bytes s)) ([s charset] (%str->bytes s)))
  (:method toString [s] s)
  (:method isEmpty [s] (= 0 (count s)))
  (:static-fn valueOf [x] (str x)))

(defn -jvm-array-sort! [arr cmp]
  ;; a Clojure comparator may return an int (-1/0/1) or a boolean less?
  (let [cmp (fn [a b] (let [r (cmp a b)] (if (number? r) (neg? r) r)))
        sorted (sort cmp (-array->vec arr))]
    (loop [i 0 s (seq sorted)]
      (if (nil? s)
        arr
        (do (%cell-set! arr i (first s)) (recur (%add i 1) (next s)))))))

(defclass java.nio.charset.StandardCharsets
  (:kind :static)
  ;; charset markers — every string<->bytes conversion here IS UTF-8
  (:static UTF_8 :charset/utf-8)
  (:static ISO_8859_1 :charset/iso-8859-1))

(defclass java.util.Arrays
  (:kind :static)
  ;; in-place sort of a raw array, optionally by comparator
  (:static-fn sort
    ([arr] (-jvm-array-sort! arr compare))
    ([arr cmp] (-jvm-array-sort! arr cmp)))
  (:static-fn equals [a b]
    (if (%num-eq (%alength a) (%alength b))
      (loop [i 0]
        (if (%lt i (%alength a))
          (if (= (%aget a i) (%aget b i)) (recur (%add i 1)) false)
          true))
      false)))

(defclass java.lang.Math
  (:kind :static)
  (:static PI 3.141592653589793)
  (:static E 2.718281828459045)
  (:static-fn abs [x] (if (neg? x) (- x) x))
  ;; REAL IEEE-754 pow (the `%pow` prim), not repeated multiplication: it must
  ;; answer for fractional exponents and overflow to ##Inf, both of which real
  ;; libraries rely on.
  (:static-fn pow [b e] (%pow b e))
  (:static-fn sqrt [x] (%pow x 0.5))
  (:static-fn max [a b] (if (< a b) b a))
  (:static-fn min [a b] (if (< a b) a b)))

;; Only the members backed by a real primitive are registered: `nanoTime` is
;; `%nanos` (monotonic, arbitrary origin — nanoTime's contract exactly). There
;; is no wall-clock prim, so `currentTimeMillis` is deliberately ABSENT and a
;; call to it throws "No such static method" rather than returning a lie.
(defclass java.lang.System
  (:kind :static)
  (:static-fn nanoTime [] (%nanos))
  (:static-fn gc [] (gc)))

;; value wrappers / interfaces mapping onto the dialect's native types
(defclass java.lang.CharSequence (:kind :interface) (:tag String) (:pred string?))
(defclass java.lang.Character (:tag Char))
(defclass java.lang.Number (:kind :interface) (:tag Long) (:pred number?)
  (:extend-tags Long Double Ratio))
(defclass java.lang.Long (:tag Long) (:extends java.lang.Number)
  (:static MAX_VALUE 9223372036854775807)
  (:static MIN_VALUE -9223372036854775808))
(defclass java.lang.Integer (:tag Long) (:extends java.lang.Number)
  (:static MAX_VALUE 2147483647)
  (:static MIN_VALUE -2147483648)
  (:static-fn parseInt [s] (read-string (str s)))
  (:static-fn valueOf [s] (read-string (str s)))
  (:static-fn toString [n] (str n)))
;; the cljs host number type (`js/Number.MAX_SAFE_INTEGER` in .cljc branches)
(defclass js.Number (:kind :static)
  (:static MAX_SAFE_INTEGER 9007199254740991)
  (:static MIN_SAFE_INTEGER -9007199254740991)
  (:static MAX_VALUE 1.7976931348623157e308)
  (:static MIN_VALUE 5e-324))
(defclass java.lang.Short (:tag Long) (:extends java.lang.Number))
(defclass java.lang.Byte (:tag Long) (:extends java.lang.Number)
  ;; Byte/TYPE — the byte primitive's class object (what a byte[]'s
  ;; getComponentType answers). Built directly: statics evaluate DURING this
  ;; class's own registration, so -jvm-class-named would still answer nil.
  (:static TYPE (record 'Class 'java.lang.Byte)))
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
;; A CHECKED exception on the JVM — it extends Exception, not RuntimeException.
;; Real libraries catch it BY NAME to probe for optional classes: meander does
;; `(try (Class/forName "cljs.tagged_literals.JSValue") … (catch
;; ClassNotFoundException _ …))` to decide whether it is running on
;; ClojureScript. Without the class registered, that catch cannot match and the
;; probe escapes as an uncaught throw.
(defclass java.lang.ClassNotFoundException
  (:tag ClassNotFoundException) (:extends java.lang.Exception))
(defclass java.lang.NumberFormatException
  (:tag NumberFormatException) (:extends java.lang.RuntimeException))
(defclass java.lang.AssertionError (:tag AssertionError) (:extends java.lang.Error))
(defclass java.lang.StackOverflowError (:tag StackOverflowError) (:extends java.lang.Error))
;; cljs host errors (`(js/Error. m)` in .cljc files) — same records as java.lang.Error
(defclass js.Error (:tag Error) (:extends java.lang.Throwable))
(defclass js.TypeError (:tag TypeError) (:extends js.Error))
(defclass js.RangeError (:tag RangeError) (:extends js.Error))
;; java.io exceptions (wire/stream code throws + catches these)
(defclass java.io.IOException (:tag IOException) (:extends java.lang.Exception))
(defclass java.io.EOFException (:tag EOFException) (:extends java.io.IOException))
(defclass java.net.SocketException (:tag SocketException) (:extends java.io.IOException))
(defclass java.nio.channels.ClosedChannelException
  (:tag ClosedChannelException) (:extends java.io.IOException))
(defclass java.lang.InterruptedException
  (:tag InterruptedException) (:extends java.lang.Exception))

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
(defclass clojure.lang.Keyword (:tag Keyword)
  ;; `(.sym kw)` — the keyword's underlying (possibly ns-qualified) symbol.
  (:method sym [k] (field k 0)))
;; Named spans TWO concrete types: `extend-type Named` registers on both.
(defclass clojure.lang.Named
  (:kind :interface)
  (:extend-tags Symbol Keyword)
  (:pred (fn [x] (or (symbol? x) (keyword? x)))))
(defclass clojure.lang.PersistentVector (:tag PersistentVector))
(defclass clojure.lang.IPersistentVector
  (:kind :interface) (:tag PersistentVector) (:pred vector?))
(defclass clojure.lang.APersistentVector
  (:kind :interface) (:tag PersistentVector) (:pred vector?))
(defclass clojure.lang.PersistentArrayMap (:tag PersistentArrayMap))
(defclass clojure.lang.PersistentHashMap (:tag PersistentHashMap))
(defclass clojure.lang.IPersistentMap
  (:kind :interface) (:tag PersistentArrayMap) (:pred map?)
  (:extend-tags PersistentArrayMap PersistentHashMap Map SortedMap))
;; IPersistentCollection = every non-map collection tag. Maps are collections
;; too on the JVM, but protocol dispatch there picks the MOST SPECIFIC
;; interface — here the split of extend-tags between IPersistentMap and this
;; IS that specificity policy (extend-protocol registers each on its own tags).
(defclass clojure.lang.IPersistentCollection
  (:kind :interface) (:tag PersistentVector) (:pred coll?)
  (:extend-tags PersistentVector PVec Vector List EmptyList LazySeq
                PersistentHashSet Set SortedSet PersistentQueue))
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
