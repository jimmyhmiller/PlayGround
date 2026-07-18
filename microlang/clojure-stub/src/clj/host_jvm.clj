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

;; Class-to-class assignability — `Class.isAssignableFrom`, which is what `isa?`
;; asks (`(isa? (type []) clojure.lang.IPersistentVector)`) with NO instance in
;; hand. So it cannot consult the interfaces' `:pred`s, which need a value.
;;
;; It must not use `:extend-tags` either: that list is the protocol-dispatch
;; SPECIFICITY policy — IPersistentCollection deliberately omits the map tags so
;; that maps dispatch to IPersistentMap — which is a different question from "is a
;; map an IPersistentCollection". On the JVM it is. So this walks DECLARED edges
;; only: `:implements` (transitively — interfaces extend interfaces) and
;; `:extends`. An undeclared edge reads as false, so the edges below are the
;; contract; a class that gains an interface must say so.
(defn -jvm-assignable? [parent child]
  (if (%num-eq parent child)
    true
    (let [d (-jvm-descriptor child)]
      (if (nil? d)
        false
        (if (-jvm-any-assignable? parent (-jvm-c-implements d))
          true
          (let [e (-jvm-c-extends d)]
            (if (nil? e) false (-jvm-assignable? parent e))))))))
(defn -jvm-any-assignable? [parent l]
  (loop [l l]
    (if (nil? l)
      false
      (if (-jvm-assignable? parent (%first l)) true (recur (%rest l))))))

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

;; {alias-symbol -> Namespace} for a namespace's `:as` aliases. The prim hands
;; back a flat (alias real alias real …) list from the COMPILER's alias table —
;; aliases are a compile-time notion, so this crosses the eval bridge. Values are
;; Namespace objects, not names: callers do `(ns-name (get (ns-aliases *ns*) 'a))`.
(defn ns-aliases [n]
  (loop [kvs (%ns-aliases (ns-name (the-ns n))) m (hash-map)]
    (if (nil? kvs)
      m
      (recur (%rest (%rest kvs))
             (assoc m (%first kvs) (record 'Namespace (%first (%rest kvs))))))))

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
  ;; The collection interfaces' host entry points. GENERATED code calls these by
  ;; name rather than through clojure.core — meander's map patterns compile to
  ;; `(.valAt target k)`, and its substitute layer uses .without/.disjoin/.val.
  ;; Each is the method Clojure's corresponding fn is defined in terms of.
  (:method valAt ([o k] (get o k)) ([o k nf] (get o k nf)))   ;; ILookup
  (:method entryAt [o k] (find o k))                          ;; Associative
  (:method containsKey [o k] (contains? o k))
  (:method assoc [o k v] (assoc o k v))
  (:method without [o k] (dissoc o k))                        ;; IPersistentMap
  (:method disjoin [o x] (disj o x))                          ;; IPersistentSet
  (:method cons [o x] (conj o x))                             ;; IPersistentCollection
  (:method empty [o] (empty o))
  (:method equiv [a b] (= a b))
  (:method seq [o] (seq o))                                   ;; Seqable
  (:method key [e] (key e))                                   ;; IMapEntry
  (:method val [e] (val e))
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
         ([b cs] (%bytes->str b))
         ;; (String. char-array offset count) — the char[] slice constructor
         ;; data.json's read-quoted-string uses to finalize a parsed string.
         ;; ONE bulk prim: the old per-char loop paid a `(str c)` alloc plus an
         ;; O(acc) `%str-cat` copy per char — O(len^2) per constructed string.
         ([arr off len] (%chars->str arr off len)))
  (:method length [s] (count s))
  ;; O(i) index without materializing the char seq (`(nth s i)` would alloc a
  ;; whole seq PER call — quadratic + heavy garbage in data.json's char-by-char
  ;; writer). `%str-char-at` is the same primitive `nth` on a string now uses.
  (:method charAt [s i] (%str-char-at s i))
  (:method substring ([s b] (subs s b)) ([s b e] (subs s b e)))
  (:method subSequence [s b e] (subs s b e))
  (:method replace [s t r] (clojure.string/replace s t r))
  (:method getBytes ([s] (%str->bytes s)) ([s charset] (%str->bytes s)))
  ;; copy chars [srcBegin,srcEnd) of s into dst starting at dstBegin (the
  ;; String.getChars contract; data.json's StringPBR bulk-reads through it).
  ;; ONE bulk prim, not a per-char loop: a `%str-char-at` + `%cell-set!` pair
  ;; per character is two prim round-trips per char copied, and (before that)
  ;; `(nth s i)` bounds-checked with an O(n) `%str-len` of the WHOLE source per
  ;; char — O(n^2) for any block reader. Range errors still hard-error inside
  ;; the prim (Java throws IndexOutOfBounds here too).
  (:method getChars [s src-begin src-end dst dst-begin]
    (%str-get-chars s src-begin src-end dst dst-begin))
  (:method toString [s] s)
  (:method isEmpty [s] (= 0 (count s)))
  (:static-fn valueOf [x] (str x)))

;; java.lang.StringBuilder — a genuinely MUTABLE character accumulator (Java's,
;; and what data.json's reader/number parser build strings with). Field 0 is a
;; growable array of already-stringified chunks, so each `.append` is O(1) and
;; `.toString`/`(str sb)` is one O(total) join — never the O(n^2) concat chain a
;; single mutable string would force. `-str1` (clojure.core) knows this tag, so
;; `(str sb)` yields the built string too.
(defclass java.lang.StringBuilder
  (:tag StringBuilder)
  (:ctor ([] (%make-record 'StringBuilder (list (%make-array 0))))
         ([x] (let [a (%make-array 0)]
                (if (%num-eq (type-of x) 'Long) nil (%apush a (str x)))
                (%make-record 'StringBuilder (list a)))))
  ;; `(if (string? x) x (str x))` — `str` is variadic (rest-seq alloc + walk
  ;; per call), and a text accumulator appends per character/token.
  (:method append [sb x] (%apush (field sb 0) (if (string? x) x (str x))) sb)
  (:method toString [sb] (%str-join-arr (field sb 0) ""))
  (:method length [sb] (count (%str-join-arr (field sb 0) "")))
  (:method charAt [sb i] (nth (%str-join-arr (field sb 0) "") i)))

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

;; java.time — this runtime has no temporal VALUES (no Instant/Date is ever
;; constructed), so the only member reachable is the ISO_INSTANT formatter that
;; data.json names in its default write options map at load. It is a distinct
;; tagged marker; the date/instant writers that would consume it are dead code
;; here (nothing in the language produces a java.time.Instant to hand them).
(defclass java.time.format.DateTimeFormatter
  (:kind :static)
  (:static ISO_INSTANT (record 'DateTimeFormatter :iso-instant)))

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
  (:static-fn floor [x] (%floor x))
  (:static-fn ceil [x] (%ceil x))
  (:static-fn log [x] (%log x))
  (:static-fn exp [x] (%exp x))
  ;; Math/round: floor(x + 0.5) as a long (Java's spec, incl. the .5-rounds-up rule)
  (:static-fn round [x] (long (%floor (+ x 0.5))))
  ;; Math/scalb: x * 2^n. %pow of 2.0 by an integer exponent is exact through
  ;; the whole double range, so the multiply IS Java's scalb for these uses.
  (:static-fn scalb [x n] (* x (%pow 2.0 n)))
  ;; Math/getExponent: the UNBIASED exponent field, read from the actual IEEE
  ;; bits — exact, including the Java answers for zero/subnormal (-1023) and
  ;; inf/NaN (1024).
  (:static-fn getExponent [x]
    (let [e (%bit-and (unsigned-bit-shift-right (%double-bits x) 52) 2047)]
      (if (%num-eq e 0) -1023 (if (%num-eq e 2047) 1024 (- e 1023)))))
  (:static-fn max [a b] (if (< a b) b a))
  (:static-fn min [a b] (if (< a b) a b)))

;; Only the members backed by a real primitive are registered: `nanoTime` is
;; `%nanos` (monotonic, arbitrary origin — nanoTime's contract exactly),
;; `currentTimeMillis` is `%wall-millis` (real wall clock, Unix epoch —
;; test.check seeds its default RNG from it).
;; java.lang.Runtime — a SINGLETON, so `(= (Runtime/getRuntime) (Runtime/getRuntime))`
;; is true as on the JVM. core.async sizes its dispatch thread pool from
;; `(.availableProcessors (Runtime/getRuntime))`.
(def -jvm-runtime (record 'Runtime 0))
(defclass java.lang.Runtime
  (:tag Runtime)
  (:method availableProcessors [_] (%cpu-count))
  (:static-fn getRuntime [] -jvm-runtime))

(defclass java.lang.System
  (:kind :static)
  (:static-fn nanoTime [] (%nanos))
  (:static-fn currentTimeMillis [] (%wall-millis))
  (:static-fn gc [] (gc)))

;; ─────────────── java.lang.ThreadLocal + proxy ───────────────
;; REAL per-thread storage: values key on `%thread-id`, so distinct OS threads
;; see independent slots — the contract test.check's seedless-RNG path leans
;; on. `initialValue` is the template-method hook: the base returns nil, a
;; `proxy` override (carried in field 1) wins.
(defclass java.lang.ThreadLocal
  (:tag ThreadLocal)
  (:ctor ([] (record 'ThreadLocal (atom {}) nil)))
  (:static-fn -proxy [overrides]
    (record 'ThreadLocal (atom {}) (-jvm-plist-get overrides 'initialValue)))
  (:method initialValue [this]
    (let [f (field this 1)] (if (nil? f) nil (f this))))
  (:method get [this]
    (let [tid (%thread-id)
          m (%atom-get (field this 0))]
      (if (contains? m tid)
        (get m tid)
        (let [v (.initialValue this)]
          (swap! (field this 0) assoc tid v)
          v))))
  (:method set [this v] (do (swap! (field this 0) assoc (%thread-id) v) nil))
  (:method remove [this] (do (swap! (field this 0) dissoc (%thread-id)) nil)))

;; Real `proxy` compiles a JVM subclass; here a base class OPTS IN by
;; registering a `-proxy` static fn (its instances carry the override table).
;; Method bodies bind `this`, exactly like Clojure's proxy. A base without
;; `-proxy` throws a clear error rather than pretending to subclass.
(defn -jvm-proxy [base overrides]
  (let [n (name base)
        ;; the registry keys descriptors by fqn SYMBOL
        fqn (cond (-jvm-descriptor base) base
                  (-jvm-descriptor (symbol (%str-cat "java.lang." n))) (symbol (%str-cat "java.lang." n))
                  (-jvm-descriptor (symbol (%str-cat "java.util." n))) (symbol (%str-cat "java.util." n))
                  :else (throw (str "proxy: unknown class " n)))
        d (-jvm-descriptor fqn)
        f (-jvm-plist-get* (-jvm-c-static-fns d) '-proxy)]
    (if (identical? f -jvm-none)
      (throw (str "proxy: unsupported base class " fqn " (no -proxy ctor registered)"))
      (f overrides))))
(defmacro proxy [class-and-ifaces ctor-args & methods]
  (let [base (first class-and-ifaces)
        entries (mapcat (fn [m]
                          (let [mname (first m)
                                params (first (rest m))
                                body (rest (rest m))]
                            (list (list 'quote mname)
                                  (%cons 'fn (%cons (vec (cons 'this (seq params))) body)))))
                        methods)]
    (list '-jvm-proxy (list 'quote base) (%cons 'list entries))))

;; ─────────────── java.lang.Iterable / java.util.Iterator ───────────────
;; Java's iteration protocol, reached for directly by real libraries: meander's
;; substitute runtime walks a collection with .iterator/.hasNext/.next and wraps
;; the cursor with clojure.core/iterator-seq. An Iterator is STATEFUL — it holds
;; the not-yet-consumed seq in a cell and advances it on every .next.
(deftype -SeqIterator [cell])
(defn -seq-iterator [coll] (-SeqIterator. (%atom-new (seq coll))))

(defclass java.util.Iterator
  (:kind :interface) (:tag -SeqIterator)
  (:method hasNext [it] (not (nil? (%atom-get (.-cell it)))))
  (:method next [it]
    (let [s (%atom-get (.-cell it))]
      (if (nil? s)
        (throw (record 'NoSuchElementException "iterator exhausted"))
        (do (%atom-set (.-cell it) (next s)) (first s))))))
(defclass java.util.NoSuchElementException
  (:tag NoSuchElementException) (:extends java.lang.RuntimeException))

;; `.iterator` lives on Object so it reaches every collection through the same
;; fallback the other host methods use.
(-proto-method .iterator Object (fn [o] (-seq-iterator o)))

;; Iterable spans every native collection, so it answers by predicate rather
;; than by tag. `coll?` is exactly Java's answer here: true for vectors, lists,
;; maps, sets and seqs; false for strings (CharSequence is not Iterable) and nil.
(defclass java.lang.Iterable (:kind :interface) (:tag PersistentVector) (:pred coll?))

;; value wrappers / interfaces mapping onto the dialect's native types
(defclass java.lang.CharSequence (:kind :interface) (:tag String) (:pred string?))
(defclass java.lang.Character (:tag Char)
  ;; ASCII range only would be a lie for astral input, but Character/isDigit on
  ;; a char is defined over the BMP code point — %char-code gives exactly that.
  ;; (Java's full Unicode digit classes include non-ASCII digits; the reader's
  ;; symbol/keyword alphabet — what test.check generates against — is ASCII.)
  (:static-fn isDigit [c] (let [n (%char-code c)] (if (%lt n 48) false (%lt n 58)))))
;; clojure.lang.ITransientSet — the transient set's host face. Real library
;; code reaches for the interface method directly: test.check's distinctness
;; machinery calls `(.contains ^clojure.lang.ITransientSet s k)` in its
;; coll-distinct-by loop (gen/map, gen/set, vector-distinct).
(defclass clojure.lang.ITransientSet
  (:kind :interface) (:tag TransientHashSet)
  (:method contains [s k]
    (not (identical? (get s k -jvm-none) -jvm-none))))
(defclass java.lang.Number (:kind :interface) (:tag Long) (:pred number?)
  (:extend-tags Long Double Ratio))
;; Long/Double parse: `%str->long`/`%str->double` are the native fast path
;; (nil on malformed/overflow); the read-string fallback preserves the exact
;; old behavior for anything the prim declines (bigint overflow, hex, exotica).
(defn -jvm-parse-long [s]
  (let [s (if (string? s) s (str s))
        v (%str->long s)]
    (if (nil? v) (read-string s) v)))
(defn -jvm-parse-double [s]
  (let [s (if (string? s) s (str s))
        v (%str->double s)]
    (if (nil? v) (read-string s) v)))
(defclass java.lang.Long (:tag Long) (:extends java.lang.Number)
  (:static MAX_VALUE 9223372036854775807)
  (:static MIN_VALUE -9223372036854775808)
  (:static-fn valueOf [s] (-jvm-parse-long s))
  (:static-fn parseLong [s] (-jvm-parse-long s))
  ;; population count of the 64-bit two's-complement pattern — %bit-count is
  ;; Java-long semantic ((Long/bitCount -1) is 64). test.check's mix-gamma.
  (:static-fn bitCount [n] (%bit-count n))
  ;; 64-bit bit-reversal — test.check's fifty-two-bit-reverse runs it per
  ;; generated double.
  (:static-fn reverse [n] (%bit-reverse n))
  (:static-fn toString [n] (str n)))
;; Parse an integer written in `radix` (2..16), signed. Digits 0-9 then a-f/A-F,
;; matching java.lang.Integer/Long.parseInt with a radix — data.json reads a
;; `\uXXXX` escape with `(Integer/parseInt hex 16)`.
(defn -parse-int-radix [s radix]
  (let [cs (%str->chars s)
        neg (%num-eq (%first cs) \-)
        cs (if neg (%rest cs) cs)]
    (loop [cs (seq cs) acc 0]
      (if (nil? cs)
        (if neg (- acc) acc)
        (let [c (%char-code (first cs))
              d (cond (if (%lt c 48) false (not (%lt 57 c))) (- c 48)   ; 0-9
                      (if (%lt c 97) false (not (%lt 102 c))) (- c 87)  ; a-f
                      (if (%lt c 65) false (not (%lt 70 c))) (- c 55)   ; A-F
                      true (throw (record 'NumberFormatException (str "For input string: \"" s "\""))))]
          (if (not (%lt d radix))
            (throw (record 'NumberFormatException (str "For input string: \"" s "\"")))
            (recur (next cs) (%add (%mul acc radix) d))))))))
(defclass java.lang.Integer (:tag Long) (:extends java.lang.Number)
  (:static MAX_VALUE 2147483647)
  (:static MIN_VALUE -2147483648)
  (:static-fn parseInt
    ([s] (-jvm-parse-long s))
    ([s radix] (-parse-int-radix (str s) radix)))
  (:static-fn valueOf [s] (-jvm-parse-long s))
  ;; lowercase hex, as java.lang.Integer.toHexString (data.json's \uXXXX escapes)
  (:static-fn toHexString [n] (-int->radix n 16))
  (:static-fn toString [n] (str n)))
;; the cljs host number type (`js/Number.MAX_SAFE_INTEGER` in .cljc branches)
(defclass js.Number (:kind :static)
  (:static MAX_SAFE_INTEGER 9007199254740991)
  (:static MIN_SAFE_INTEGER -9007199254740991)
  (:static MAX_VALUE 1.7976931348623157e308)
  (:static MIN_VALUE 5e-324))
;; The narrower integer boxes share Long's runtime tag (this dialect has ONE
;; integer type); the MIN/MAX statics are what generator/serialization code
;; reads (test.check's gen/byte chooses in [Byte/MIN_VALUE, Byte/MAX_VALUE]).
(defclass java.lang.Short (:tag Long) (:extends java.lang.Number)
  (:static MIN_VALUE -32768)
  (:static MAX_VALUE 32767))
(defclass java.lang.Byte (:tag Long) (:extends java.lang.Number)
  ;; Byte/TYPE — the byte primitive's class object (what a byte[]'s
  ;; getComponentType answers). Built directly: statics evaluate DURING this
  ;; class's own registration, so -jvm-class-named would still answer nil.
  (:static TYPE (record 'Class 'java.lang.Byte))
  (:static MIN_VALUE -128)
  (:static MAX_VALUE 127))
(defclass java.math.BigInteger (:tag Long) (:extends java.lang.Number))
(defclass clojure.lang.BigInt (:tag Long) (:extends java.lang.Number))
(defclass java.lang.Double (:tag Double) (:extends java.lang.Number)
  (:static POSITIVE_INFINITY ##Inf)
  (:static NEGATIVE_INFINITY ##-Inf)
  (:static NaN ##NaN)
  (:static MAX_VALUE 1.7976931348623157e308)
  (:static MIN_VALUE 4.9e-324)
  (:static-fn valueOf [s] (-jvm-parse-double s))
  (:static-fn parseDouble [s] (-jvm-parse-double s))
  (:static-fn doubleToLongBits [x] (%double-bits x))
  ;; NaN is the only value not equal to itself; ±Inf are the only values whose
  ;; magnitude exceeds the largest finite double. Both answers match the JVM
  ;; without needing an Inf/NaN literal or a divide-by-zero.
  (:method isNaN [x] (not (%num-eq x x)))
  (:method isInfinite [x]
    (let [m 1.7976931348623157e308] (or (%lt m x) (%lt x (- m))))))
(defclass java.lang.Float (:tag Double) (:extends java.lang.Number)
  (:method isNaN [x] (not (%num-eq x x)))
  (:method isInfinite [x]
    (let [m 1.7976931348623157e308] (or (%lt m x) (%lt x (- m))))))
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
(defclass clojure.lang.PersistentVector (:tag PersistentVector)
  (:implements clojure.lang.IPersistentVector clojure.lang.APersistentVector
               clojure.lang.IPersistentCollection clojure.lang.Seqable
               clojure.lang.Associative clojure.lang.ILookup clojure.lang.Indexed
               clojure.lang.Counted clojure.lang.Sequential clojure.lang.IFn))
(defclass clojure.lang.IPersistentVector
  (:kind :interface) (:tag PersistentVector) (:pred vector?))
(defclass clojure.lang.APersistentVector
  (:kind :interface) (:tag PersistentVector) (:pred vector?))
(defclass clojure.lang.PersistentArrayMap (:tag PersistentArrayMap)
  (:implements clojure.lang.IPersistentMap clojure.lang.APersistentMap
               clojure.lang.IPersistentCollection clojure.lang.Seqable
               clojure.lang.Associative clojure.lang.ILookup
               clojure.lang.Counted clojure.lang.IFn))
(defclass clojure.lang.PersistentHashMap (:tag PersistentHashMap)
  (:implements clojure.lang.IPersistentMap clojure.lang.APersistentMap
               clojure.lang.IPersistentCollection clojure.lang.Seqable
               clojure.lang.Associative clojure.lang.ILookup
               clojure.lang.Counted clojure.lang.IFn))
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
(defclass clojure.lang.PersistentHashSet (:tag PersistentHashSet)
  (:implements clojure.lang.IPersistentSet clojure.lang.IPersistentCollection
               clojure.lang.Seqable clojure.lang.Counted clojure.lang.IFn))
(defclass clojure.lang.IPersistentSet
  (:kind :interface) (:tag PersistentHashSet) (:pred set?))
(defclass clojure.lang.ISeq (:kind :interface) (:tag List) (:pred seq?))
(defclass clojure.lang.Seqable (:kind :interface) (:tag List) (:pred -seqable?))
(defclass clojure.lang.Cons (:tag List)
  (:implements clojure.lang.ISeq clojure.lang.IPersistentCollection
               clojure.lang.Seqable clojure.lang.Sequential))
(defclass clojure.lang.IPersistentList (:kind :interface) (:tag List) (:pred list?))
(defclass clojure.lang.PersistentList
  (:tag List)
  (:implements clojure.lang.ISeq clojure.lang.IPersistentList
               clojure.lang.IPersistentCollection clojure.lang.Seqable
               clojure.lang.Sequential clojure.lang.Counted)
  (:static creator -list))
(defclass clojure.lang.IFn (:kind :interface) (:tag Fn) (:pred ifn?))
(defclass clojure.lang.ILookup (:kind :interface) (:protocol ILookup))

;; The java.util collection interfaces, as the JVM's own hierarchy: Map is NOT a
;; Collection, and both span every corresponding persistent tag. Real libraries
;; extend protocols to these (data.json's JSONWriter dispatches a map to
;; java.util.Map -> write-object and a vector/seq/set to java.util.Collection ->
;; write-array), so the extend targets must resolve to the same tag sets our own
;; clojure.lang.IPersistent* interfaces do.
(defclass java.util.Map
  (:kind :interface) (:tag PersistentArrayMap) (:pred map?)
  (:extend-tags PersistentArrayMap PersistentHashMap Map SortedMap))
(defclass java.util.Collection
  (:kind :interface) (:tag PersistentVector) (:pred coll?)
  (:extend-tags PersistentVector PVec Vector List EmptyList LazySeq
                PersistentHashSet Set SortedSet PersistentQueue))
(defclass java.util.List
  (:kind :interface) (:tag PersistentVector) (:pred sequential?)
  (:extend-tags PersistentVector PVec Vector List EmptyList LazySeq PersistentQueue))
(defclass java.util.Set
  (:kind :interface) (:tag PersistentHashSet) (:pred set?)
  (:extend-tags PersistentHashSet Set SortedSet))

;; ─────────────── dialect-native host types (cljs heritage) ───────────────
;; a map entry IS a `[k v]` vector in this dialect
(defclass cljs.core.MapEntry
  (:ctor ([k v] (vector k v)) ([k v h] (vector k v))))
(defclass cljs.core.PersistentQueue
  (:tag PersistentQueue)
  (:static EMPTY -empty-queue))
