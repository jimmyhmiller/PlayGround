//! `clojure.core`: the mini-Clojure standard library, written in the language on
//! top of the toolkit's primitives. Collections are LIST-BACKED tagged records
//! (`(record 'Vector elem-list)` etc.); `type-of` drives dispatch. Internals use
//! the un-shadowable `%`-prims (`%add`/`%first`/…); user-facing `+`/`=`/`first`
//! are first-class fns wrapping them (so they can be passed to `map`/`reduce`).

pub const CORE: &str = r##"
;; ─────────────── control macros ───────────────
;; `(defn f [args] body…)` or `(defn f "docstring" [args] body…)`. Only PRIMS are
;; used in the macro body (it must expand before the seq library exists); the
;; docstring path records :doc in the var registry (via `-set-var-doc!`, defined
;; once atoms are available — which is before any documented user fn).
;; (defn name docstring? attr-map? fntail) — skip an optional leading
;; docstring and an optional leading attr-map (e.g. {:arglists '([& xs])})
;; before the fn body/arity-clauses, matching Clojure's defn grammar.
(defmacro defn (name p2 & more)
  (if (%num-eq (type-of p2) 'String)
    (list 'do
          (list 'def name
                (%cons 'fn (if (%num-eq (type-of (%first more)) 'Map)
                             (%rest more) more)))
          (list '-set-var-doc! (list 'var name) p2))
    (if (%num-eq (type-of p2) 'Map)
      (list 'def name (%cons 'fn more))
      (list 'def name (%cons 'fn (%cons p2 more))))))
(defmacro when (c & body)     (list 'if c (%cons 'do body) nil))
(defmacro when-not (c & body) (list 'if c nil (%cons 'do body)))
(defmacro cond (& cs)
  (if (nil? cs) nil
      (list 'if (%first cs) (%first (%rest cs)) (%cons 'cond (%rest (%rest cs))))))
(defmacro and (& xs)
  (cond (nil? xs) true (nil? (%rest xs)) (%first xs)
        true (list 'if (%first xs) (%cons 'and (%rest xs)) false)))
(defmacro or (& xs)
  (cond (nil? xs) nil (nil? (%rest xs)) (%first xs)
        true (list 'if (%first xs) (%first xs) (%cons 'or (%rest xs)))))

;; ─────────────── constructors ───────────────
(defn hash-map [& kvs] (record 'Map kvs))
(defn hash-set [& es] (record 'Set es))

;; ─────────────── PersistentVector (ClojureScript-style, in-language) ───────────
;; A 32-way bit-partitioned trie with a tail buffer, built on the mutable-array
;; primitives (%make-array / %aget / %cell-set! / %aclone) exactly as cljs builds
;; its PersistentVector on JS arrays. Fields: (PVec cnt shift root tail) where
;; root/tail are arrays; conj/assoc/pop path-copy (aclone) so nodes are shared
;; immutably. Indexing uses 5-bit shifts (branch factor 32, mask 31).
(defn -make-array [n] (%make-array n))
(defn -aget [a i] (%aget a i))
(defn -aset [a i v] (do (%cell-set! a i v) a))   ; mutate then return the array
(defn -aclone [a] (%aclone a))
(defn -alength [a] (%alength a))
(defn -arr-conj [a v]                            ; copy of a with v appended
  (let [n (-alength a) r (-make-array (%add n 1))]
    (loop [i 0] (if (%lt i n) (do (-aset r i (-aget a i)) (recur (%add i 1))) (-aset r n v)))))
(defn -arr-pop [a]                               ; copy of a without its last element
  (let [n (%sub (-alength a) 1) r (-make-array n)]
    (loop [i 0] (if (%lt i n) (do (-aset r i (-aget a i)) (recur (%add i 1))) r))))

(defn pvec? [x] (%num-eq (type-of x) 'PVec))
(defn -pv-cnt [pv] (field pv 0))
(defn -pv-shift [pv] (field pv 1))
(defn -pv-root [pv] (field pv 2))
(defn -pv-tail [pv] (field pv 3))
(defn -tail-off [pv]
  (let [cnt (-pv-cnt pv)] (if (%lt cnt 32) 0 (%bit-shl (%bit-shr (%sub cnt 1) 5) 5))))
(defn -array-for [pv i]
  (if (%lt i (-tail-off pv))
      (loop [node (-pv-root pv) level (-pv-shift pv)]
        (if (%lt 0 level)
            (recur (-aget node (%bit-and (%bit-shr i level) 31)) (%sub level 5))
            node))
      (-pv-tail pv)))
(defn -pv-nth [pv i]
  (if (if (%lt i 0) true (%lt (%sub (-pv-cnt pv) 1) i))
      (throw (str "index out of bounds: " i))
      (-aget (-array-for pv i) (%bit-and i 31))))
(defn -new-path [level node]
  (if (%num-eq level 0) node
      (-aset (-make-array 32) 0 (-new-path (%sub level 5) node))))
(defn -push-tail [pv level parent tailnode]
  (let [subidx (%bit-and (%bit-shr (%sub (-pv-cnt pv) 1) level) 31) ret (-aclone parent)]
    (if (%num-eq level 5)
        (-aset ret subidx tailnode)
        (let [child (-aget parent subidx)]
          (if (nil? child)
              (-aset ret subidx (-new-path (%sub level 5) tailnode))
              (-aset ret subidx (-push-tail pv (%sub level 5) child tailnode)))))))
(defn -pv-conj [pv val]
  (let [cnt (-pv-cnt pv) shift (-pv-shift pv)]
    (if (%lt (%sub cnt (-tail-off pv)) 32)
        (record 'PVec (%add cnt 1) shift (-pv-root pv) (-arr-conj (-pv-tail pv) val))
        (if (%lt (%bit-shl 1 shift) (%bit-shr cnt 5))
            (let [new-root (-make-array 32)]
              (-aset new-root 0 (-pv-root pv))
              (-aset new-root 1 (-new-path shift (-pv-tail pv)))
              (record 'PVec (%add cnt 1) (%add shift 5) new-root (%anew val)))
            (record 'PVec (%add cnt 1) shift (-push-tail pv shift (-pv-root pv) (-pv-tail pv)) (%anew val))))))
(defn -do-assoc [level node i val]
  (let [ret (-aclone node)]
    (if (%num-eq level 0)
        (-aset ret (%bit-and i 31) val)
        (let [subidx (%bit-and (%bit-shr i level) 31)]
          (-aset ret subidx (-do-assoc (%sub level 5) (-aget node subidx) i val))))))
(defn -pv-assoc [pv i val]
  (let [cnt (-pv-cnt pv) shift (-pv-shift pv)]
    (cond (%num-eq i cnt) (-pv-conj pv val)
          (if (%lt i 0) true (%lt (%sub cnt 1) i)) (throw (str "assoc index out of bounds: " i))
          (%lt i (-tail-off pv)) (record 'PVec cnt shift (-do-assoc shift (-pv-root pv) i val) (-pv-tail pv))
          true (record 'PVec cnt shift (-pv-root pv) (-aset (-aclone (-pv-tail pv)) (%bit-and i 31) val)))))
(defn -pop-tail [pv level node]
  (let [subidx (%bit-and (%bit-shr (%sub (-pv-cnt pv) 2) level) 31)]
    (cond (%lt 5 level)
            (let [new-child (-pop-tail pv (%sub level 5) (-aget node subidx))]
              (if (if (nil? new-child) (%num-eq subidx 0) false) nil
                  (-aset (-aclone node) subidx new-child)))
          (%num-eq subidx 0) nil
          true (-aset (-aclone node) subidx nil))))
(defn -pv-pop [pv]
  (let [cnt (-pv-cnt pv) shift (-pv-shift pv)]
    (cond (%num-eq cnt 0) (throw "can't pop empty vector")
          (%num-eq cnt 1) -empty-pvec
          (%lt 1 (%sub cnt (-tail-off pv))) (record 'PVec (%sub cnt 1) shift (-pv-root pv) (-arr-pop (-pv-tail pv)))
          true (let [new-tail (-array-for pv (%sub cnt 2))
                     rt0 (-pop-tail pv shift (-pv-root pv))
                     new-root (if (nil? rt0) (-make-array 32) rt0)]
                 (if (if (%lt 5 shift) (nil? (-aget new-root 1)) false)
                     (record 'PVec (%sub cnt 1) (%sub shift 5) (-aget new-root 0) new-tail)
                     (record 'PVec (%sub cnt 1) shift new-root new-tail))))))
;; Eager cons realization of the vector (finite), built back-to-front with a
;; tail-recursive loop so it never grows the native stack (a chunked-lazy seq
;; would avoid realizing the whole spine for partial consumption; a later opt).
(defn -pv-seq [pv]
  (loop [i (%sub (-pv-cnt pv) 1) acc nil]
    (if (%lt i 0) acc (recur (%sub i 1) (%cons (-pv-nth pv i) acc)))))
(def -empty-pvec (record 'PVec 0 5 (-make-array 32) (-make-array 0)))
;; `vector` is used by MACROS at expansion time (e.g. `if-let`, `case`, `for`),
;; so it must depend only on prims + the in-block PVec fns — never on `seq`/
;; `reduce`/`second`, which are defined later in this file.
(defn -vec-from-list [pv s] (if (nil? s) pv (-vec-from-list (-pv-conj pv (%first s)) (%rest s))))
(defn vector [& es] (-vec-from-list -empty-pvec es))
(defn vec [c] (-vec-from-list -empty-pvec (-to-list c)))
(defn vector? [x] (%num-eq (type-of x) 'PVec))

;; ─────────────── type predicates ───────────────
(defn map? [x] (let [t (type-of x)] (or (%num-eq t 'Map) (%num-eq t 'SortedMap))))
(defn set? [x] (let [t (type-of x)] (or (%num-eq t 'Set) (%num-eq t 'SortedSet))))
;; `& {:keys […]}` keyword-argument destructuring: collect trailing kwargs into a
;; map. A single trailing MAP arg is used as-is (Clojure 1.11 map/kwargs mixing).
(defn -kwargs->map [args]
  (let [s (seq args)]
    (if (and s (nil? (next s)) (map? (first s))) (first s) (apply hash-map args))))
(defn keyword? [x] (%num-eq (type-of x) 'Keyword))
(defn list? [x] (let [t (type-of x)] (or (%num-eq t 'List) (%num-eq t 'EmptyList))))
(defn string? [x] (%num-eq (type-of x) 'String))
(defn symbol? [x] (%num-eq (type-of x) 'Symbol))
;; `name`/`namespace` split a (possibly qualified) symbol or keyword on the last `/`.
(defn name [x]
  (cond (string? x) x
        (keyword? x) (%sym-name (field x 0))
        true (%sym-name x)))
(defn namespace [x]
  (cond (keyword? x) (%sym-ns (field x 0))
        true (%sym-ns x)))
;; `(symbol "x")` / `(symbol "ns" "x")` -> an interned symbol.
(defn symbol [n & more]
  (if (nil? more) (%symbol n) (%symbol (%str-cat (%str-cat n "/") (first more)))))
(defn fn? [x] (%num-eq (type-of x) 'Fn))
(defn number? [x] (%num-eq (type-of x) 'Long))
(defn not [x] (if x false true))

;; ─────────────── first-class arithmetic / comparison (variadic) ───────────────
;; Fold with the un-passable `%`-prims via a named helper (a prim can't be passed
;; to `reduce` as a value), matching Clojure's variadic +/-/*/</>/=.
(defn -add-seq [acc s] (if (nil? s) acc (-add-seq (%add acc (%first s)) (%rest s))))
(defn -mul-seq [acc s] (if (nil? s) acc (-mul-seq (%mul acc (%first s)) (%rest s))))
(defn -sub-seq [acc s] (if (nil? s) acc (-sub-seq (%sub acc (%first s)) (%rest s))))
(defn + [& xs] (-add-seq 0 (seq xs)))
(defn * [& xs] (-mul-seq 1 (seq xs)))
(defn - [x & xs] (if (nil? (seq xs)) (%sub 0 x) (-sub-seq x (seq xs))))
;; Chained comparisons: each adjacent pair must satisfy the relation.
(defn -lt-seq [prev s] (if (nil? s) true (if (%lt prev (%first s)) (-lt-seq (%first s) (%rest s)) false)))
(defn -gt-seq [prev s] (if (nil? s) true (if (%lt (%first s) prev) (-gt-seq (%first s) (%rest s)) false)))
(defn -le-seq [prev s] (if (nil? s) true (if (%lt (%first s) prev) false (-le-seq (%first s) (%rest s)))))
(defn -ge-seq [prev s] (if (nil? s) true (if (%lt prev (%first s)) false (-ge-seq (%first s) (%rest s)))))
(defn -eq-seq [a s] (if (nil? s) true (if (-eq2 a (%first s)) (-eq-seq a (%rest s)) false)))
(defn < [x & xs] (-lt-seq x (seq xs)))
(defn > [x & xs] (-gt-seq x (seq xs)))
(defn <= [x & xs] (-le-seq x (seq xs)))
(defn >= [x & xs] (-ge-seq x (seq xs)))
(defn = [x & xs] (-eq-seq x (seq xs)))
(defn not= [x & xs] (not (-eq-seq x (seq xs))))
(defn inc [n] (%add n 1))
(defn dec [n] (%sub n 1))
;; integer division family (quot truncates toward zero; rem follows the dividend's
;; sign; mod follows the divisor's).
(defn quot [a b] (%quot a b))
(defn rem [a b] (%rem a b))
(defn mod [a b] (%mod a b))
;; `/` — 1-arg is reciprocal; otherwise a left fold. Exact when integers divide
;; evenly, else float (no Ratio type).
(defn / [x & more]
  (if (nil? more) (%div 1 x) (-div-seq x (seq more))))
(defn -div-seq [acc s] (if (nil? s) acc (-div-seq (%div acc (first s)) (next s))))

;; ── ex-info: a data-carrying exception (throw/catch already take any value) ──
(defn ex-info [msg data & more]
  (record 'ExInfo msg data (if (nil? more) nil (first more))))
(defn ex-info? [x] (%num-eq (type-of x) 'ExInfo))
;; ex-info stores (msg data cause); a bare `(new Error "msg")` is `(record 'Error msg)`.
(defn ex-message [e]
  (cond (ex-info? e) (field e 0)
        (%num-eq (type-of e) 'Error) (field e 0)
        :else nil))
(defn ex-data [e] (if (ex-info? e) (field e 1) nil))
(defn ex-cause [e] (if (ex-info? e) (field e 2) nil))
(defn pos? [n] (%lt 0 n))
(defn neg? [n] (%lt n 0))
(defn zero? [n] (%num-eq n 0))
(defn identity [x] x)
(defn even? [n] (%num-eq (%mod n 2) 0))
(defn odd? [n] (%num-eq (%mod n 2) 1))

;; ─────────────── low-level helpers (prim-based; used by the protocol impls) ────
(defn elems [c] (field c 0))
(defn entries [kvs]
  (if (nil? kvs) nil
      (%cons (vector (%first kvs) (%first (%rest kvs))) (entries (%rest (%rest kvs))))))
(defn count-seq [s n] (let [s (seq s)] (if (nil? s) n (count-seq (%rest s) (%add n 1)))))
(defn nth-seq [s i] (let [s (seq s)] (if (%num-eq i 0) (%first s) (nth-seq (%rest s) (%sub i 1)))))
(defn mget [kvs k]
  (cond (nil? kvs) nil (%num-eq (%first kvs) k) (%first (%rest kvs)) true (mget (%rest (%rest kvs)) k)))
(defn kv-has [kvs k]
  (cond (nil? kvs) false (%num-eq (%first kvs) k) true true (kv-has (%rest (%rest kvs)) k)))
(defn kv-without [kvs k]
  (cond (nil? kvs) nil
        (%num-eq (%first kvs) k) (%rest (%rest kvs))
        true (%cons (%first kvs) (%cons (%first (%rest kvs)) (kv-without (%rest (%rest kvs)) k)))))
(defn kv-keys [kvs] (if (nil? kvs) nil (%cons (%first kvs) (kv-keys (%rest (%rest kvs))))))
(defn kv-vals [kvs] (if (nil? kvs) nil (%cons (%first (%rest kvs)) (kv-vals (%rest (%rest kvs))))))
(defn append1 [lst x] (if (nil? lst) (%cons x nil) (%cons (%first lst) (append1 (%rest lst) x))))

;; A lazy seq is a record holding two mutable cells: a thunk (0-arg fn) and a
;; realized? flag. On first force we call the thunk, `seq` its result (chaining
;; through further lazy layers), and cache it — this makes infinite seqs and lazy
;; map/filter/take possible.
(defn lazy-seq? [x] (%num-eq (type-of x) 'LazySeq))
(defn -lazy-seq [thunk] (record 'LazySeq (%cell thunk) (%cell false)))
(defn -force [ls]
  (if (%cell-ref (field ls 1) 0)
      (%cell-ref (field ls 0) 0)
      (let [v (seq ((%cell-ref (field ls 0) 0)))]
        (do (%cell-set! (field ls 0) 0 v) (%cell-set! (field ls 1) 0 true) v))))
(defmacro lazy-seq (& body) (list '-lazy-seq (%cons 'fn (%cons (vector) body))))

;; ─────────────── collection protocols (ClojureScript-style) ───────────────
;; The seq/collection abstraction is polymorphic THROUGH these protocols: the
;; public fns below are thin dispatchers, and every collection type — nil, cons
;; lists, code vectors, PVec, Map, Set, LazySeq — implements them via extend-type.
;; New collection types need only implement the protocols; no core fn grows a
;; per-type branch. `Object` provides the lenient defaults (get -> not-found, etc.).
(defprotocol ISeqable (-seq [coll]))
(defprotocol ISeq (-first [coll]) (-rest [coll]))
(defprotocol ICollection (-conj [coll o]))
(defprotocol ICounted (-count [coll]))
(defprotocol IIndexed (-nth [coll n]))
(defprotocol ILookup (-lookup [coll k nf]))
(defprotocol IAssociative (-assoc [coll k v]) (-contains-key? [coll k]))
(defprotocol IStack (-peek [coll]) (-pop [coll]))
(defprotocol IEquiv (-equiv [coll other]))
(defprotocol IEmptyableCollection (-empty [coll]))
;; ClojureScript's editable-collection protocol. Nothing extends it here, so
;; `(satisfies? IEditableCollection x)` is always false and cljs library code
;; (e.g. medley) falls back to its non-transient path.
(defprotocol IEditableCollection (-as-transient [coll]))

(extend-type Object
  IEquiv (-equiv [a b] (%num-eq a b))
  ILookup (-lookup [_ k nf] nf)
  IAssociative (-contains-key? [_ k] false)
  IEmptyableCollection (-empty [_] nil))

(extend-type nil
  ISeqable (-seq [_] nil)
  ICounted (-count [_] 0)
  IIndexed (-nth [_ n] nil)
  ICollection (-conj [_ o] (%cons o nil))
  ILookup (-lookup [_ k nf] nf)
  IAssociative (-assoc [_ k v] (record 'Map (%cons k (%cons v nil)))) (-contains-key? [_ k] false)
  IStack (-peek [_] nil) (-pop [_] nil)
  IEquiv (-equiv [_ o] (nil? o))
  IEmptyableCollection (-empty [_] nil))

;; Sequential seqs (list / lazy-seq / empty-list) compare `=` element-wise with
;; each other (but NOT with vectors) — matching Clojure, where `(= '(1 2) (map …))`
;; is true but `(= '(1 2) [1 2])` is false.
;; A "sequential" collection for `=`: lists, lazy-seqs, empty-list, AND vectors all
;; compare element-wise with each other (Clojure: `(= [1 2] '(1 2))` is true), but
;; NOT with sets/maps.
(defn -seqlike? [x]
  (let [t (type-of x)]
    (or (%num-eq t 'List) (%num-eq t 'LazySeq) (%num-eq t 'EmptyList) (vector? x))))
(extend-type List
  ISeqable (-seq [l] l)
  ISeq (-first [l] (%first l)) (-rest [l] (%rest l))
  ICollection (-conj [l o] (%cons o l))
  ICounted (-count [l] (count-seq l 0))
  IIndexed (-nth [l n] (nth-seq l n))
  IStack (-peek [l] (%first l)) (-pop [l] (%rest l))
  IEquiv (-equiv [a b] (if (-seqlike? b) (-seq-eq a b) false))
  IEmptyableCollection (-empty [_] nil))
;; The empty list `()` — a distinct value from nil: `list?`/`seq?` true, prints
;; `()`, not `= nil`. Behaves as an empty seq (seq -> nil, count 0, first/rest nil).
(extend-type EmptyList
  ISeqable (-seq [_] nil)
  ISeq (-first [_] nil) (-rest [_] nil)
  ICollection (-conj [_ o] (%cons o nil))
  ICounted (-count [_] 0)
  IIndexed (-nth [_ n] nil)
  IStack (-peek [_] nil) (-pop [_] nil)
  IEquiv (-equiv [_ other] (if (-seqlike? other) (nil? (seq other)) false))
  IEmptyableCollection (-empty [e] e))
;; a String seqs as its character list (clojure treats strings as seqable).
(extend-type String
  ISeqable (-seq [s] (seq (%str->chars s)))
  ICounted (-count [s] (%str-len s)))

;; reader/code vectors (list-backed 'Vector records) — macros build & manipulate
;; binding forms like `[a 5]`, and -realize's display vectors are this shape too.
(extend-type Vector
  ISeqable (-seq [v] (field v 0))
  ICounted (-count [v] (count-seq (field v 0) 0))
  IIndexed (-nth [v n] (nth-seq (field v 0) n)))

(extend-type LazySeq
  ISeqable (-seq [l] (-force l))
  ;; forcing ISeq, so -first/-rest work on an unrealized lazy seq (the seq fns
  ;; and any consumer can treat a LazySeq like any other seq).
  ISeq (-first [l] (let [s (-force l)] (if (nil? s) nil (-first s))))
       (-rest [l] (let [s (-force l)] (if (nil? s) nil (-rest s))))
  ICounted (-count [l] (count-seq l 0))
  IIndexed (-nth [l n] (nth-seq l n))
  IEquiv (-equiv [a b] (if (-seqlike? b) (-seq-eq a b) false)))

(extend-type PVec
  ISeqable (-seq [v] (-pv-seq v))
  ICounted (-count [v] (-pv-cnt v))
  IIndexed (-nth [v n] (-pv-nth v n))
  ICollection (-conj [v o] (-pv-conj v o))
  IAssociative (-assoc [v k val] (-pv-assoc v k val))
    (-contains-key? [v k] (if (if (%lt k 0) true (%lt (%sub (-pv-cnt v) 1) k)) false true))
  ILookup (-lookup [v k nf] (if (if (%lt k 0) true (%lt (%sub (-pv-cnt v) 1) k)) nf (-pv-nth v k)))
  IStack (-peek [v] (if (%num-eq (-pv-cnt v) 0) nil (-pv-nth v (%sub (-pv-cnt v) 1))))
    (-pop [v] (-pv-pop v))
  IEquiv (-equiv [v o] (if (pvec? o) (if (%num-eq (-pv-cnt v) (-pv-cnt o)) (-seq-eq v o) false) false))
  IEmptyableCollection (-empty [_] -empty-pvec))

(extend-type Map
  ISeqable (-seq [m] (entries (field m 0)))
  ICounted (-count [m] (count-seq (entries (field m 0)) 0))
  ;; conj a [k v] pair, or MERGE another map's entries (as clojure.core does).
  ICollection (-conj [m e] (if (map? e) (reduce conj m (entries (field e 0))) (-assoc m (-nth e 0) (-nth e 1))))
  ILookup (-lookup [m k nf] (if (kv-has (field m 0) k) (mget (field m 0) k) nf))
  IAssociative (-assoc [m k v] (record 'Map (%cons k (%cons v (kv-without (field m 0) k)))))
    (-contains-key? [m k] (kv-has (field m 0) k))
  IEquiv (-equiv [m o] (if (map? o) (if (%num-eq (-count m) (-count o)) (keys-match (keys m) m o) false) false))
  IEmptyableCollection (-empty [_] (record 'Map nil)))

(extend-type Set
  ISeqable (-seq [s] (elems s))
  ICounted (-count [s] (count-seq (elems s) 0))
  ICollection (-conj [s o] (record 'Set (%cons o (field s 0))))
  ILookup (-lookup [s k nf] (if (-mem? (elems s) k) k nf))
  IAssociative (-contains-key? [s k] (-mem? (elems s) k))
  IEmptyableCollection (-empty [_] (record 'Set nil)))

;; ─────────────── public collection API (thin protocol dispatchers) ───────────
;; `seq` returns a REALIZED seq (a cons whose head is forced) or nil — never an
;; unforced lazy seq. This is Clojure's contract, and it means every downstream
;; `%first`/`%rest` (which assume a cons) is correct regardless of the source
;; (list, vector, set, map, lazy seq): a lazy `-seq` result is forced here.
(defn seq [c] (if (nil? c) nil (let [s (-seq c)] (if (lazy-seq? s) (seq s) s))))
(defn first [c] (let [s (seq c)] (if (nil? s) nil (-first s))))
(defn rest [c] (let [s (seq c)] (if (nil? s) nil (-rest s))))
(defn next [c] (seq (rest c)))
(defn cons [x c] (%cons x c))
(defn empty? [c] (nil? (seq c)))
(defn empty [c] (-empty c))
(defn count [c] (cond (nil? c) 0 (string? c) (%str-len c) true (-count c)))
;; `subs` (clojure.core) — substring via the char list; end defaults to the length.
(defn subs [s start & end]
  (let [cs (%str->chars s)
        e (if (nil? end) (%str-len s) (first end))]
    (apply str (take (- e start) (drop start cs)))))
(defn nth [c i & d]
  (cond
    (string? c) (if (and (>= i 0) (< i (%str-len c)))
                  (nth-seq (%str->chars c) i)
                  (if (seq d) (first d) (throw (str "String index out of bounds: " i))))
    (and (seq d) (or (neg? i) (not (< i (count c))))) (first d)
    :else (-nth c i)))
(defn conj
  ([] [])
  ([c] c)
  ([c x] (-conj c x))
  ;; `-conj` is a dispatch-only protocol method (no value binding), so fold with
  ;; the public `conj` wrapper instead of passing `-conj` as a value.
  ([c x & more] (reduce conj (-conj c x) more)))
(defn assoc
  ([m k v] (-assoc m k v))
  ([m k v & kvs] (reduce (fn [a p] (-assoc a (first p) (second p))) (-assoc m k v) (-pairs kvs))))
(defn contains? [c k] (-contains-key? c k))
(defn peek [c] (-peek c))
(defn pop [c] (-pop c))
(defn get [c k & r] (-lookup c k (if (nil? r) nil (%first r))))
(defn dissoc [m k] (record 'Map (kv-without (field m 0) k)))
(defn keys [m] (kv-keys (field m 0)))
(defn vals [m] (kv-vals (field m 0)))

;; tail-recursive list reverse + full realization (never grow the native stack).
(defn -rev [s] (loop [s (seq s) acc nil] (if (nil? s) acc (recur (next s) (%cons (%first s) acc)))))
(defn -to-list [s] (-rev (-rev (seq s))))
(defn doall [c] (do (-to-list c) c))
(defn dorun [c] (do (-to-list c) nil))

;; equality: maps compare order-independently, vectors element-wise, everything
;; else structurally — all via the IEquiv protocol (Object default = %num-eq).
(defn keys-match [ks a b]
  (cond (nil? ks) true
        (-eq2 (mget (field a 0) (%first ks)) (mget (field b 0) (%first ks))) (keys-match (%rest ks) a b)
        true false))
(defn -seq-eq [a b]
  (let [a (seq a) b (seq b)]
    (cond (nil? a) (nil? b) (nil? b) false
          (-eq2 (%first a) (%first b)) (-seq-eq (%rest a) (%rest b)) true false)))
(defn -eq2 [a b] (-equiv a b))

;; Callable objects: keywords, maps and vectors act as functions of one arg.
(defn -apply-obj [o & args]
  (cond (multi? o) (-multi-call o args)
        (keyword? o) (get (first args) o)
        (map? o) (get o (first args))
        (set? o) (get o (first args))
        (vector? o) (nth o (first args))
        true (throw "value is not callable")))

;; ─────────────── reduce / into ───────────────
;; honors `(reduced x)`: once the accumulator is reduced, stop and unwrap it.
(defn reduce-seq [f acc s]
  (if (reduced? acc)
    (field acc 0)
    (let [s (seq s)] (if (nil? s) acc (reduce-seq f (f acc (%first s)) (%rest s))))))
;; `reduce` is 2- or 3-arity (like clojure.core): `(reduce f coll)` seeds with the
;; first element (or `(f)` when empty); `(reduce f init coll)` seeds with `init`.
(defn reduce [f & args]
  (if (nil? (next args))
      (let [s (seq (first args))] (if (nil? s) (f) (reduce-seq f (%first s) (%rest s))))
      (reduce-seq f (first args) (seq (second args)))))
(defn into
  ([] [])
  ([to] to)
  ([to from] (reduce conj to from))
  ([to xform from] (transduce xform conj to from)))

;; ─────────────── higher-order seq fns (lazy) ───────────────
(defn -map1 [f c]
  (lazy-seq (let [s (seq c)] (if (nil? s) nil (%cons (f (%first s)) (-map1 f (%rest s)))))))
(defn -map2 [f a b]
  (lazy-seq (let [sa (seq a) sb (seq b)]
              (if (if (nil? sa) true (nil? sb)) nil
                  (%cons (f (%first sa) (%first sb)) (-map2 f (%rest sa) (%rest sb)))))))
(defn -map3 [f a b c]
  (lazy-seq (let [sa (seq a) sb (seq b) sc (seq c)]
              (if (if (nil? sa) true (if (nil? sb) true (nil? sc))) nil
                  (%cons (f (%first sa) (%first sb) (%first sc)) (-map3 f (%rest sa) (%rest sb) (%rest sc)))))))
;; N-ary map: apply f to the first item of each coll, stopping at the shortest.
(defn -some-nil? [xs]
  (let [s (seq xs)]
    (if (nil? s) false (if (nil? (first s)) true (-some-nil? (rest s))))))
(defn -mapn [f colls]
  (lazy-seq
    (let [ss (-map1 seq colls)]
      (if (-some-nil? ss) nil
        (%cons (apply f (-map1 first ss)) (-mapn f (-map1 rest ss)))))))
;; `map` is variadic over collections (like clojure.core), stopping at the shortest.
(defn map [f & colls]
  (cond (nil? colls) (fn [rf] (fn ([] (rf)) ([a] (rf a)) ([a x] (rf a (f x)))))
        (nil? (next colls)) (-map1 f (first colls))
        (nil? (next (next colls))) (-map2 f (first colls) (second colls))
        (nil? (next (next (next colls)))) (-map3 f (first colls) (second colls) (nth colls 2))
        true (-mapn f colls)))
(defn filter
  ([pred] (fn [rf] (fn ([] (rf)) ([a] (rf a)) ([a x] (if (pred x) (rf a x) a)))))
  ([f c]
   (lazy-seq (let [s (seq c)]
               (cond (nil? s) nil
                     (f (%first s)) (%cons (%first s) (filter f (%rest s)))
                     true (filter f (%rest s)))))))
(defn remove
  ([pred] (filter (fn [x] (not (pred x)))))
  ([f c] (filter (fn [x] (not (f x))) c)))
(defn keep [f c]
  (lazy-seq (let [s (seq c)]
              (if (nil? s) nil
                  (let [v (f (%first s))]
                    (if (nil? v) (keep f (%rest s)) (%cons v (keep f (%rest s)))))))))
(defn -range-inf [i] (lazy-seq (%cons i (-range-inf (%add i 1)))))
(defn -range2 [i n] (lazy-seq (if (%lt i n) (%cons i (-range2 (%add i 1) n)) nil)))
(defn -range3 [i n step] (lazy-seq (if (%lt i n) (%cons i (-range3 (%add i step) n step)) nil)))
(defn range [& args]
  (cond (nil? (seq args)) (-range-inf 0)
        (nil? (next args)) (-range2 0 (first args))
        (nil? (next (next args))) (-range2 (first args) (second args))
        true (-range3 (first args) (second args) (nth args 2))))
(defn concat2 [a b]
  (lazy-seq (let [s (seq a)] (if (nil? s) (seq b) (%cons (%first s) (concat2 (%rest s) b))))))
(defn concat-lists [lls] (if (nil? lls) nil (concat2 (%first lls) (concat-lists (%rest lls)))))
(defn concat [& lls] (concat-lists lls))
;; EAGER concat: syntax-quote builds code forms with this, since a macro must
;; return a realized (non-lazy) form the expander can splice.
(defn -concat2 [a b] (let [s (seq a)] (if (nil? s) (-to-list b) (%cons (%first s) (-concat2 (%rest s) b)))))
(defn -concat-lists [lls] (if (nil? lls) nil (-concat2 (%first lls) (-concat-lists (%rest lls)))))
(defn -concat [& lls] (-concat-lists lls))
(defn reverse-onto [s acc] (let [s (seq s)] (if (nil? s) acc (reverse-onto (%rest s) (%cons (%first s) acc)))))
(defn reverse [c] (reverse-onto c nil))
(defn drop [n c]
  (if (%lt 0 n) (let [s (seq c)] (if (nil? s) nil (drop (%sub n 1) (%rest s)))) (seq c)))
(defn take [n c]
  (lazy-seq (if (%lt 0 n)
                (let [s (seq c)] (if (nil? s) nil (%cons (%first s) (take (%sub n 1) (%rest s)))))
                nil)))
(defn take-while [pred c]
  (lazy-seq (let [s (seq c)]
              (if (nil? s) nil
                  (if (pred (%first s)) (%cons (%first s) (take-while pred (%rest s))) nil)))))
(defn drop-while [pred c]
  (lazy-seq (let [s (seq c)]
              (if (nil? s) nil (if (pred (%first s)) (drop-while pred (%rest s)) s)))))
;; ─────────────── infinite / generator seqs ───────────────
;; x is the immediate head; `(f x)` is deferred INSIDE the inner lazy-seq (so
;; realizing element n applies f exactly n times, not n+1). Exactly cljs's form.
(defn iterate [f x] (cons x (lazy-seq (iterate f (f x)))))
(defn -repeat-inf [x] (lazy-seq (%cons x (-repeat-inf x))))
(defn -repeat-n [n x] (lazy-seq (if (%lt 0 n) (%cons x (-repeat-n (%sub n 1) x)) nil)))
(defn repeat [& args] (if (nil? (next args)) (-repeat-inf (first args)) (-repeat-n (first args) (second args))))
(defn repeatedly [f] (lazy-seq (%cons (f) (repeatedly f))))
(defn -cycle [orig s]
  (lazy-seq (let [s (seq s)]
              (if (nil? s) (-cycle orig orig) (%cons (%first s) (-cycle orig (%rest s)))))))
(defn cycle [c] (if (nil? (seq c)) nil (-cycle c c)))
(defn second [c] (nth c 1))
(defn last [c] (let [s (seq c)] (if (nil? s) nil (if (nil? (next s)) (%first s) (last (%rest s))))))
(defn seq? [x] (let [t (type-of x)] (or (%num-eq t 'List) (%num-eq t 'EmptyList) (%num-eq t 'LazySeq))))
(defn butlast-seq [s] (let [s (seq s)] (if (nil? (next s)) nil (%cons (%first s) (butlast-seq (%rest s))))))
(defn butlast [c] (butlast-seq c))
;; `sigs` computes the :arglists metadata for `defn`: the parameter vector of a
;; single-arity fn, or the seq of parameter vectors of a multi-arity one. (Our
;; `def` discards symbol metadata, so this is informational — but it lets the
;; literal core.clj `defn` body load and run unchanged.)
(defn -asig [fd] (first fd))
(defn sigs [fdecl]
  (if (seq? (first fdecl))
      (map -asig fdecl)
      (list (-asig fdecl))))

;; ─────────────── more clojure.core ───────────────
(defn get-in [m ks & d]
  (let [nf (if (nil? d) nil (first d))]
    (loop [m m ks (seq ks)]
      (cond (nil? ks) m
            (and (associative? m) (contains? m (first ks))) (recur (get m (first ks)) (next ks))
            (nil? (next ks)) (get m (first ks) nf)
            true nf))))
(defn assoc-in [m ks v]
  (if (nil? (%rest (seq ks)))
      (assoc m (%first (seq ks)) v)
      (assoc m (%first (seq ks)) (assoc-in (get m (%first (seq ks))) (%rest (seq ks)) v))))
(defn update [m k f & args] (assoc m k (apply f (get m k) args)))
(defn some-seq [pred s] (let [s (seq s)] (if (nil? s) nil (let [r (pred (%first s))] (if r r (some-seq pred (%rest s)))))))
(defn some [pred c] (some-seq pred c))
(defn every-seq [pred s] (let [s (seq s)] (if (nil? s) true (if (pred (%first s)) (every-seq pred (%rest s)) false))))
(defn every? [pred c] (every-seq pred c))
(defn mapv [f c] (vec (map f c)))
(defn filterv [f c] (vec (filter f c)))
(defn -comp2 [f g] (fn [& args] (f (apply g args))))
(defn comp [& fns]
  (cond (nil? (seq fns)) identity
        (nil? (next fns)) (first fns)
        true (reduce -comp2 (first fns) (rest fns))))
(defn partial [f & pre] (fn [& args] (apply f (concat pre args))))
(defn constantly [x] (fn [& _] x))
(defn complement [f] (fn [& args] (not (apply f args))))
(defn max [a & more] (reduce (fn [x y] (if (%lt x y) y x)) a more))
(defn min [a & more] (reduce (fn [x y] (if (%lt x y) x y)) a more))
(defn map-indexed-h [f i s]
  (lazy-seq (let [s (seq s)] (if (nil? s) nil (%cons (f i (%first s)) (map-indexed-h f (%add i 1) (%rest s)))))))
(defn map-indexed [f c] (map-indexed-h f 0 c))

;; if-let / when-let are defined below, after `gensym` (they need a fresh temp so
;; a DESTRUCTURING binding form like `[k & ks]` is bound-and-tested via the temp,
;; not used as the raw `if` condition).

;; ── host runtime: our stand-in for clojure.lang.RT ──
;; The interop shim rewrites `(. clojure.lang.RT (first coll))` etc. to these, so
;; real clojure/core.clj definitions (which are written in terms of RT) execute.
(def -list (fn [& xs] xs))
;; The RT shims route straight to the PROTOCOL methods, not the public fns —
;; real core.clj REDEFINES seq/first/conj/assoc (in terms of RT), so calling the
;; public names here would recurse forever.
(defn -rt-seq [c] (if (nil? c) nil (-seq c)))
(defn -rt-first [c] (let [s (-rt-seq c)] (if (nil? s) nil (-first s))))
(defn -rt-rest [c] (let [s (-rt-seq c)] (if (nil? s) nil (-rest s))))
(defn -rt-next [c] (-rt-seq (-rt-rest c)))
(defn -rt-conj [c x] (-conj c x))
(defn -rt-assoc [m k v] (-assoc m k v))

;; metadata AS VALUES: stored as a trailing record field, so field 0 (the
;; contents) reads unchanged and `vector?`/`map?`/`seq`/… stay transparent. On a
;; symbol/other value the metadata is dropped (it is only informational there).
(defn -with-meta [x m]
  (cond (pvec? x) (record 'PVec (-pv-cnt x) (-pv-shift x) (-pv-root x) (-pv-tail x) m)
        (map? x) (record 'Map (field x 0) m)
        (set? x) (record 'Set (field x 0) m)
        true x))
(defn -meta [x]
  (cond (pvec? x) (if (%lt 4 (nfields x)) (field x 4) nil)
        (%lt 1 (nfields x)) (field x 1)
        true nil))
(def with-meta -with-meta)
(def meta -meta)

;; ─────────────── threading macros ───────────────
(defn thread-first [x form] (%cons (%first form) (%cons x (%rest form))))
(defn snoc [lst x] (if (nil? lst) (%cons x nil) (%cons (%first lst) (snoc (%rest lst) x))))
(defn thread-last [x form] (%cons (%first form) (snoc (%rest form) x)))
(defmacro -> (x & steps)
  (if (nil? steps) x
      (%cons '->
             (%cons (if (list? (%first steps))
                        (thread-first x (%first steps))
                        (list (%first steps) x))
                    (%rest steps)))))
(defmacro ->> (x & steps)
  (if (nil? steps) x
      (%cons '->>
             (%cons (if (list? (%first steps))
                        (thread-last x (%first steps))
                        (list (%first steps) x))
                    (%rest steps)))))

;; ─────────────── more control macros ───────────────
(defmacro if-not (c then & else) (list 'if c (%first else) then))
(defmacro while (test & body)
  `(loop [] (when ~test ~@body (recur))))
(defmacro dotimes (binding & body)
  (let [i (first binding) n (second binding)]
    `(loop [~i 0] (when (< ~i ~n) ~@body (recur (inc ~i))))))
;; `doseq` runs `body` for side effects over the SAME binding grammar as `for`
;; (multiple bindings, :let/:when/:while), then returns nil. `dorun` forces the
;; lazy `for` without retaining its head.
(defmacro doseq (bindings & body)
  `(dorun (for ~bindings (do ~@body))))
(defmacro when-first (binding & body)
  (let [x (first binding) coll (second binding)]
    `(let [s# (seq ~coll)]
       (when s# (let [~x (first s#)] ~@body)))))
;; `case`: constant dispatch, desugared to a `cond` of `=` tests (a trailing odd
;; clause is the default). Non-hygienic scratch symbol `-case-val`.
(defn -case-cond [g clauses]
  (cond (nil? clauses) (list true (list 'throw "no matching case clause"))
        (nil? (next clauses)) (list true (first clauses))
        true (%cons (list '= g (first clauses))
                    (%cons (second clauses)
                           (-case-cond g (next (next clauses)))))))
(defmacro case (e & clauses)
  (list 'let (vector '-case-val e)
        (%cons 'cond (-case-cond '-case-val clauses))))

;; ─────────────── atoms (the single-threaded state model) ───────────────
;; An atom is a record wrapping a mutable 1-slot cell. `@a`/`(deref a)` reads it;
;; `reset!`/`swap!` mutate it. (Threads + compare-and-set semantics are a later
;; layer; this is the identity/state box real clojure code reaches for.)
;; Atoms are a real atomic cell now (Obj::Atom), so cross-thread swap! is a
;; correct compare-and-set retry loop (not a read-modify-write race).
(defn atom [x] (%atom-new x))
(defn atom? [x] (%num-eq (type-of x) 'Atom))
(defn future? [x] (%num-eq (type-of x) 'Future))
;; First-class vars. `#'x`/`(var x)` reads as `(record 'Var 'ns/x)`; a Var is a
;; thin handle over the global's qualified SYM, so every operation dispatches
;; through the global table by that sym (matching an ordinary reference to `x`).
(defn var? [x] (%num-eq (type-of x) 'Var))
(defn -var-sym [v] (field v 0))
;; Docstrings captured by `defn` (keyed by the var's qualified sym). A plain atom
;; map, so it is GC-managed like any value (no toolkit registry change needed).
(def -var-docs (%atom-new (hash-map)))
(defn -set-var-doc! [v doc]
  (%atom-set -var-docs (assoc (%atom-get -var-docs) (field v 0) doc)))
(defn -var-doc [v] (get (%atom-get -var-docs) (field v 0)))
;; Reflection: `(find-var 'ns/x)` -> the Var for a FULLY-QUALIFIED sym (or nil).
;; `resolve`/`ns-resolve` on an unqualified literal are rewritten at compile time to
;; the qualified sym (namespace resolution is compile-time); given an already-
;; qualified sym they are just `find-var`.
(defn find-var [s] (if (%global-bound? s) (record 'Var s) nil))
(defn resolve [s] (find-var s))
(defn ns-resolve [n s] (find-var s))
;; Reflectively read/rebind the var's ROOT (throws if unbound; sets on rebind).
(defn var-get [v] (%global-get (-var-sym v)))
(defn var-set [v val] (%global-set (-var-sym v) val))
(defn bound? [v] (%global-bound? (-var-sym v)))
;; alter-var-root: apply f (plus extra args) to the current root, install result.
(defn alter-var-root [v f & args]
  (let [s (-var-sym v)
        nv (apply f (%global-get s) args)]
    (%global-set s nv)
    nv))
;; deref reads an atom (atomic load), joins a future, or reads a var's root.
(defn deref [x]
  (cond (future? x) (%await x)
        (var? x) (%global-get (-var-sym x))
        (%num-eq (type-of x) 'Volatile) (%cell-ref (field x 0) 0)
        (%num-eq (type-of x) 'Delay) (-force-delay x)
        (%num-eq (type-of x) 'Reduced) (field x 0)
        (%num-eq (type-of x) 'Promise) (-await-promise x)
        :else (%atom-get x)))
;; Real OS threads: `(future body...)` runs on a worker sharing the heap.
(defn future-call [f] (%spawn f))
(defmacro future (& body) (list 'future-call (%cons 'fn (%cons (vector) body))))
(defn pcalls-2 [f g] (let [a (%spawn f) b (%spawn g)] (list (%await a) (%await b))))
(defn reset! [a v] (%atom-set a v))
;; Apply f to the atom's current value and up to 3 extra args.
(defn -swap-apply [f old s]
  (cond (nil? s) (f old)
        (nil? (next s)) (f old (first s))
        (nil? (next (next s))) (f old (first s) (second s))
        (nil? (next (next (next s)))) (f old (first s) (second s) (nth s 2))
        true (throw "swap!: too many args (max 3 beyond the atom)")))
;; CAS retry: read, compute, compare-and-set; on contention, retry (keeping args).
(defn swap! [a f & args]
  (let [s (seq args)]
    (loop []
      (let [old (%atom-get a)
            new (-swap-apply f old s)]
        (if (%atom-cas a old new) new (recur))))))
(defn compare-and-set! [a old new] (%atom-cas a old new))
;; swap-vals!/reset-vals! return the [old new] pair.
(defn swap-vals! [a f & args]
  (let [s (seq args)]
    (loop []
      (let [old (%atom-get a) new (-swap-apply f old s)]
        (if (%atom-cas a old new) [old new] (recur))))))
(defn reset-vals! [a v]
  (loop []
    (let [old (%atom-get a)]
      (if (%atom-cas a old v) [old v] (recur)))))

;; ─────────────── misc small library fns ───────────────
(defn find-keyword
  ([nm] (keyword nm))
  ([ns nm] (keyword ns nm)))
(defn vector-of [t & elems] (vec elems))
(defn tagged-literal [tag form] (record 'TaggedLiteral tag form))
(defn tagged-literal? [x] (= (type-of x) 'TaggedLiteral))
(defn reader-conditional [form splicing?] (record 'ReaderConditional form splicing?))
(defn reader-conditional? [x] (= (type-of x) 'ReaderConditional))
;; hash mixing (matches clojure.core's small-int mixing shape closely enough for
;; use as a value-combining helper; not bit-for-bit the JVM's Murmur3).
(defn hash-combine [seed h] (%bit-xor seed (+ h 2654435769 (%bit-shl seed 6) (%bit-shr seed 2))))
(defn mix-collection-hash [hash-basis cnt] (%bit-xor hash-basis cnt))

;; ─────────────── seq/collection breadth (clojure.core) ───────────────
(defn some? [x] (not (nil? x)))
(defn true? [x] (%num-eq x true))
(defn false? [x] (%num-eq x false))
(defn ffirst [c] (first (first c)))
(defn nfirst [c] (next (first c)))
(defn fnext [c] (first (next c)))
(defn nnext [c] (next (next c)))
;; peek/pop/empty are defined above as protocol dispatchers (IStack /
;; IEmptyableCollection).
(defn not-empty [c] (if (empty? c) nil c))

;; membership / equality helpers (map-aware `-eq2`)
(defn -mem? [s x] (cond (nil? (seq s)) false (-eq2 (first s) x) true true (-mem? (rest s) x)))

(defn mapcat [f coll]
  (lazy-seq (let [s (seq coll)] (if (nil? s) nil (concat2 (f (first s)) (mapcat f (rest s)))))))
(defn interpose [sep coll] (drop 1 (mapcat (fn [x] (list sep x)) coll)))
(defn interleave [a b]
  (lazy-seq (let [sa (seq a) sb (seq b)]
              (if (if (nil? sa) true (nil? sb)) nil
                  (%cons (first sa) (%cons (first sb) (interleave (rest sa) (rest sb))))))))
(defn take-nth [n coll]
  (lazy-seq (let [s (seq coll)] (if (nil? s) nil (%cons (first s) (take-nth n (drop n s)))))))

(defn -distinct [seen coll]
  (lazy-seq (let [s (seq coll)]
              (if (nil? s) nil
                  (let [x (first s)]
                    (if (-mem? seen x) (-distinct seen (rest s))
                        (%cons x (-distinct (%cons x seen) (rest s)))))))))
(defn distinct [coll] (-distinct nil coll))
(def -none (record 'None nil))
(defn -dedupe [prev coll]
  (lazy-seq (let [s (seq coll)]
              (if (nil? s) nil
                  (let [x (first s)]
                    (if (-eq2 x prev) (-dedupe prev (rest s)) (%cons x (-dedupe x (rest s)))))))))
(defn dedupe [coll] (-dedupe -none coll))

(defn -seqable? [x] (cond (nil? x) true (list? x) true (vector? x) true (set? x) true (lazy-seq? x) true true false))
(defn flatten [coll] (mapcat (fn [x] (if (-seqable? x) (flatten x) (list x))) coll))

;; partition family (chunks are lists; short final chunk dropped unless -all)
;; Re-`seq` each iteration: `(rest s)` may hand back an unforced lazy tail that
;; forces to nil, so the `nil?` guard must see the forced value (else a phantom
;; nil element gets collected).
(defn -take-n [n s]
  (loop [n n s s acc nil]
    (let [s (seq s)]
      (cond (%num-eq n 0) (reverse acc)
            (nil? s) nil
            true (recur (dec n) (rest s) (%cons (first s) acc))))))
(defn -take-upto [n s]
  (loop [n n s s acc nil]
    (let [s (seq s)]
      (cond (%num-eq n 0) (reverse acc)
            (nil? s) (reverse acc)
            true (recur (dec n) (rest s) (%cons (first s) acc))))))
(defn -partition [n step s]
  (lazy-seq (let [s (seq s) chunk (-take-n n s)]
              (if (nil? chunk) nil (%cons chunk (-partition n step (drop step s)))))))
(defn partition [n & args]
  (if (nil? (next args)) (-partition n n (first args)) (-partition n (first args) (second args))))
(defn -partition-all [n step s]
  (lazy-seq (let [s (seq s)]
              (if (nil? s) nil (%cons (-take-upto n s) (-partition-all n step (drop step s)))))))
(defn partition-all [n & args]
  (if (nil? (next args)) (-partition-all n n (first args)) (-partition-all n (first args) (second args))))
(defn -part-by-run [f v s acc]
  (let [s (seq s)]
    (if (if (nil? s) true (not (-eq2 (f (first s)) v))) (vector (reverse acc) s)
        (-part-by-run f v (rest s) (%cons (first s) acc)))))
(defn partition-by [f coll]
  (lazy-seq (let [s (seq coll)]
              (if (nil? s) nil
                  (let [r (-part-by-run f (f (first s)) s nil)] (%cons (nth r 0) (partition-by f (nth r 1))))))))

;; indexed / scanning
(defn -keep-indexed [f i s]
  (lazy-seq (let [s (seq s)]
              (if (nil? s) nil
                  (let [v (f i (first s))]
                    (if (nil? v) (-keep-indexed f (inc i) (rest s)) (%cons v (-keep-indexed f (inc i) (rest s)))))))))
(defn keep-indexed [f coll] (-keep-indexed f 0 coll))
(defn -reductions [f init s]
  (lazy-seq (%cons init (let [s (seq s)] (if (nil? s) nil (-reductions f (f init (first s)) (rest s)))))))
(defn reductions [f & args]
  (if (nil? (next args))
      (let [s (seq (first args))] (if (nil? s) nil (-reductions f (first s) (rest s))))
      (-reductions f (first args) (second args))))

;; map building: group-by / frequencies / zipmap / merge / select-keys / update-in
(defn group-by [f coll]
  (reduce (fn [m x] (let [k (f x)] (assoc m k (conj (if (contains? m k) (get m k) (vector)) x)))) (hash-map) coll))
(defn frequencies [coll]
  (reduce (fn [m x] (assoc m x (inc (if (contains? m x) (get m x) 0)))) (hash-map) coll))
(defn zipmap [ks vs]
  (loop [ks (seq ks) vs (seq vs) m (hash-map)]
    (if (if (nil? ks) true (nil? vs)) m (recur (next ks) (next vs) (assoc m (first ks) (first vs))))))
(defn -merge2 [a b]
  (if (nil? b) a (reduce (fn [m kv] (assoc m (first kv) (second kv))) a (seq b))))
(defn merge [& maps]
  (reduce (fn [a b] (if (nil? a) b (-merge2 a b))) nil (seq maps)))
(defn -merge-with2 [f a b]
  (reduce (fn [m kv] (let [k (first kv) v (second kv)]
                       (if (contains? m k) (assoc m k (f (get m k) v)) (assoc m k v)))) a (seq b)))
(defn merge-with [f & maps]
  (reduce (fn [a b] (cond (nil? a) b (nil? b) a true (-merge-with2 f a b))) nil (seq maps)))
(defn select-keys [m ks]
  (reduce (fn [acc k] (if (contains? m k) (assoc acc k (get m k)) acc)) (hash-map) (seq ks)))
(defn update-in [m ks f & args]
  (let [k (first ks)]
    (if (nil? (next ks))
      (assoc m k (apply f (get m k) args))
      (assoc m k (apply update-in (get m k) (rest ks) f args)))))

;; sorting (tortoise/hare split + merge; numeric default via %lt)
(defn -default-less [a b] (%lt (compare a b) 0))
(defn -merge-lists [less a b]
  (cond (nil? (seq a)) (seq b)
        (nil? (seq b)) (seq a)
        (less (first b) (first a)) (%cons (first b) (-merge-lists less a (rest b)))
        true (%cons (first a) (-merge-lists less (rest a) b))))
(defn -halve [slow fast acc]
  (if (if (nil? (seq fast)) true (nil? (next fast))) (vector (reverse acc) slow)
      (-halve (rest slow) (rest (rest fast)) (%cons (first slow) acc))))
(defn -msort [less s]
  (let [s (-to-list s)]
    (if (if (nil? s) true (nil? (next s))) s
        (let [parts (-halve s s nil)]
          (-merge-lists less (-msort less (nth parts 0)) (-msort less (nth parts 1)))))))
(defn sort [& args]
  (if (nil? (next args)) (-msort -default-less (first args)) (-msort (first args) (second args))))
(defn sort-by [k & args]
  (if (nil? (next args))
      (-msort (fn [a b] (%lt (compare (k a) (k b)) 0)) (first args))
      (-msort (fn [a b] ((first args) (k a) (k b))) (second args))))

;; misc combinators
(defn juxt [& fns] (fn [& args] (vec (map (fn [f] (apply f args)) fns))))
(defn -maxk [k best s] (let [s (seq s)] (if (nil? s) best (-maxk k (if (%lt (k best) (k (first s))) (first s) best) (rest s)))))
(defn max-key [k a & more] (-maxk k a more))
(defn -mink [k best s] (let [s (seq s)] (if (nil? s) best (-mink k (if (%lt (k (first s)) (k best)) (first s) best) (rest s)))))
(defn min-key [k a & more] (-mink k a more))

;; ─────────────── for (list comprehension) ───────────────
;; Each collection binding drives `-for-drive`, which concats the sub-seqs its
;; per-element fn returns. Modifiers nest LEXICALLY inside that fn so `:let` vars
;; are in scope for a following `:when`/`:while`: `:when` false contributes nil
;; (skip, continue); `:while` false returns the `-for-stop` sentinel (halts the
;; nearest enclosing drive).
(def -for-stop (record 'ForStop nil))
(defn -for-drive [f s]
  (lazy-seq (let [s (seq s)]
              (if (nil? s) nil
                  (let [r (f (%first s))]
                    (if (%num-eq (type-of r) 'ForStop) nil (concat2 r (-for-drive f (%rest s)))))))))
(defn -for-build [pairs body]
  (if (nil? (seq pairs)) (list 'list body)
      (let [k (first pairs) v (second pairs) more (rest (rest pairs))]
        (if (keyword? k)
            (cond (= k :let)   (list 'let v (-for-build more body))
                  (= k :when)  (list 'if v (-for-build more body) nil)
                  (= k :while) (list 'if v (-for-build more body) (quote -for-stop))
                  true (throw "for: unknown modifier keyword"))
            (list (quote -for-drive) (list 'fn (vector k) (-for-build more body)) v)))))
(defmacro for (bindings body) (-for-build (seq bindings) body))

;; ─────────────── extra threading / control macros ───────────────
(defmacro cond-> (expr & clauses)
  (if (nil? (seq clauses)) expr
      (list 'let (vector (quote -ctg) expr)
            (%cons 'cond->
                   (%cons (list 'if (first clauses) (list '-> (quote -ctg) (second clauses)) (quote -ctg))
                          (rest (rest clauses)))))))
(defmacro cond->> (expr & clauses)
  (if (nil? (seq clauses)) expr
      (list 'let (vector (quote -ctg2) expr)
            (%cons 'cond->>
                   (%cons (list 'if (first clauses) (list '->> (quote -ctg2) (second clauses)) (quote -ctg2))
                          (rest (rest clauses)))))))
(defmacro some-> (expr & forms)
  (if (nil? (seq forms)) expr
      (list 'let (vector (quote -stg) expr)
            (list 'if (list 'nil? (quote -stg)) nil
                  (%cons 'some-> (%cons (list '-> (quote -stg) (first forms)) (rest forms)))))))
(defmacro some->> (expr & forms)
  (if (nil? (seq forms)) expr
      (list 'let (vector (quote -stg2) expr)
            (list 'if (list 'nil? (quote -stg2)) nil
                  (%cons 'some->> (%cons (list '->> (quote -stg2) (first forms)) (rest forms)))))))
(defn -as-binds [nm forms]
  (if (nil? (seq forms)) nil (%cons nm (%cons (first forms) (-as-binds nm (rest forms))))))
(defmacro as-> (expr nm & forms)
  (list 'let (vec (%cons nm (%cons expr (-as-binds nm forms)))) nm))
(defn -doto-forms [g forms]
  (if (nil? (seq forms)) nil
      (%cons (if (list? (first forms))
                 (%cons (first (first forms)) (%cons g (rest (first forms))))
                 (list (first forms) g))
             (-doto-forms g (rest forms)))))
(defmacro doto (expr & forms)
  (list 'let (vector (quote -dtg) expr) (%cons 'do (append1 (-doto-forms (quote -dtg) forms) (quote -dtg)))))
(defn -condp [pv ev clauses]
  (cond (nil? (seq clauses)) (list 'throw "condp: no matching clause")
        (nil? (next clauses)) (first clauses)
        (= (second clauses) :>>) (throw "condp: :>> form not supported")
        true (list 'if (list pv (first clauses) ev) (second clauses)
                   (-condp pv ev (rest (rest clauses))))))
(defmacro condp (pred expr & clauses)
  (list 'let (vector (quote -cpp) pred (quote -cpe) expr) (-condp (quote -cpp) (quote -cpe) (seq clauses))))

;; ─────────────── str (value -> string) ───────────────
;; Clojure-style `str`: strings raw, collections formatted structurally here (so
;; the toolkit stays frontend-neutral); leaf atoms go through `%str-of`. Uses
;; `%str-cat` to concatenate. `nil` stringifies to "" (as in clojure.core).
(defn -str-join [sep coll]
  (let [s (seq coll)]
    (cond (nil? s) ""
          (nil? (next s)) (-str1 (first s))
          true (%str-cat (-str1 (first s)) (%str-cat sep (-str-join sep (rest s)))))))
(defn -str-map-join [kvs]
  (cond (nil? (seq kvs)) ""
        (nil? (seq (rest (rest kvs))))
          (%str-cat (-str1 (first kvs)) (%str-cat " " (-str1 (second kvs))))
        true (%str-cat (-str1 (first kvs))
                       (%str-cat " " (%str-cat (-str1 (second kvs))
                                               (%str-cat ", " (-str-map-join (rest (rest kvs)))))))))
;; Format a seq of [k v] map entries as "k v, k v" (map keys are unordered).
(defn -str-entries [es]
  (let [es (seq es)]
    (cond (nil? es) ""
          (nil? (next es)) (let [e (first es)] (%str-cat (-str1 (first e)) (%str-cat " " (-str1 (second e)))))
          true (let [e (first es)]
                 (%str-cat (-str1 (first e))
                           (%str-cat " " (%str-cat (-str1 (second e))
                                                   (%str-cat ", " (-str-entries (rest es))))))))))
(defn -str1 [x]
  (cond (nil? x) ""
        (string? x) x
        (keyword? x) (%str-cat ":" (%str-of (field x 0)))
        (vector? x) (%str-cat "[" (%str-cat (-str-join " " x) "]"))
        (set? x) (%str-cat "#{" (%str-cat (-str-join " " (seq x)) "}"))
        (map? x) (%str-cat "{" (%str-cat (-str-entries (seq x)) "}"))
        (lazy-seq? x) (%str-cat "(" (%str-cat (-str-join " " x) ")"))
        (list? x) (%str-cat "(" (%str-cat (-str-join " " x) ")"))
        true (%str-of x)))
(defn -str-seq [acc s] (if (nil? (seq s)) acc (-str-seq (%str-cat acc (-str1 (first s))) (rest s))))
(defn str [& xs] (-str-seq "" xs))

;; ─────────────── apply ───────────────
;; `(apply f a b ... coll)` — call `f` with the leading args followed by the
;; elements of the final collection. The final collection is realized into a
;; plain cons-list (so vectors / maps / lazy seqs all spread), the leading args
;; are consed on front, and the `%apply` prim spreads that list into the call.
(defn -apply-flatten [args]
  (if (nil? (next args))
      (-to-list (first args))
      (%cons (first args) (-apply-flatten (rest args)))))
(defn apply [f & args] (%apply f (-apply-flatten args)))

;; ─────────────── multimethods (defmulti / defmethod) ───────────────
;; A multimethod is a callable record `(MultiFn dispatch-fn table-atom)` where the
;; table-atom holds a map from dispatch-value -> method fn (with a `:default` entry
;; as the fallback). Calling it computes `(dispatch-fn args...)`, looks the value
;; up, and applies the chosen method to the same args. Dispatch values compare by
;; structural equality (so keywords / numbers / vectors all work as keys).
;; MultiFn record: (record 'MultiFn dispatch-fn method-table-atom prefers-atom).
(defn -make-multi [df nm] (record 'MultiFn df (atom (hash-map)) (atom (hash-map)) nm))
(defn multi? [x] (%num-eq (type-of x) 'MultiFn))
(defn -add-method [mf dval method]
  (let [a (field mf 1)] (reset! a (assoc (deref a) dval method))))
(defn -prefers [mf] (deref (field mf 2)))
;; Does `a` win over `b` for this multimethod (transitively via the prefers graph)?
(defn -mf-prefers? [mf a b]
  (let [pm (-prefers mf) pa (get pm a)]
    (boolean (or (contains? pa b)
                 (some (fn [x] (-mf-prefers? mf x b)) pa)))))
;; Resolve the dispatch value to a method, honoring `isa?` and `prefer-method`.
(defn -multi-lookup [mf dval]
  (let [tbl (deref (field mf 1))]
    (if (contains? tbl dval)
      (get tbl dval)
      (let [matches (filter (fn [k] (and (not= k :default) (isa? dval k))) (keys tbl))]
        (cond
          (empty? matches) (if (contains? tbl :default) (get tbl :default)
                             (throw (str "No method in multimethod '" (field mf 3) "' for dispatch value: " dval)))
          (= 1 (count matches)) (get tbl (first matches))
          :else (let [best (reduce (fn [a b] (cond (nil? a) b
                                                    (-mf-prefers? mf a b) a
                                                    (-mf-prefers? mf b a) b
                                                    (isa? b a) b
                                                    (isa? a b) a
                                                    :else (throw (str "Multiple methods match dispatch value: " dval " -> " a " and " b ", and neither is preferred"))))
                            nil matches)]
                  (get tbl best)))))))
(defn -multi-call [mf args]
  (apply (-multi-lookup mf (apply (field mf 0) args)) args))
(defn -mf-prefer [mf a b]
  (let [pa (field mf 2)] (reset! pa (assoc (deref pa) a (conj (get (deref pa) a (hash-set)) b))) nil))
(defn prefer-method [mf a b] (-mf-prefer mf a b))
(defn prefers [mf] (-prefers mf))
(defn methods [mf] (deref (field mf 1)))
(defn get-method [mf dval] (-multi-lookup mf dval))
(defn remove-method [mf dval] (let [a (field mf 1)] (reset! a (dissoc (deref a) dval)) mf))
(defn remove-all-methods [mf] (let [a (field mf 1)] (reset! a (hash-map)) mf))
(defmacro defmulti (name dispatch-fn) (list 'def name (list '-make-multi dispatch-fn (list 'quote name))))
(defmacro defmethod (name dval params & body)
  (list '-add-method name dval (%cons 'fn (%cons params body))))

;; ─────────────── protocol reflection (satisfies? / extends? / extenders) ───────────────
;; A protocol is (record 'Protocol 'Name (list 'm1 'm2 …)); its methods dispatch on
;; type via the native registry, queryable with %method-types.
(defn -protocol? [p] (and (record? p) (= (type-of p) 'Protocol)))
(defn -proto-methods [p] (field p 1))
(defn satisfies? [p x]
  (let [ty (type-of x)]
    (boolean (some (fn [m] (some (fn [t] (= t ty)) (%method-types m))) (-proto-methods p)))))
(defn extends? [p ty]
  (boolean (some (fn [m] (some (fn [t] (= t ty)) (%method-types m))) (-proto-methods p))))
(defn extenders [p]
  (-to-list (reduce (fn [acc m] (reduce conj acc (%method-types m))) #{} (-proto-methods p))))

;; ─────────────── ad-hoc hierarchy (derive / isa? / parents / …) ───────────────
;; A hierarchy is {:parents {tag #{parents}} :ancestors {tag #{transitive}}
;; :descendants {tag #{transitive}}}, held in a global atom for the arity-1 forms.
;; Faithful port of clojure.core's derive/underive/isa?.
(defn make-hierarchy [] {:parents {} :ancestors {} :descendants {}})
(def -global-hierarchy (atom (make-hierarchy)))
(defn parents
  ([tag] (parents (deref -global-hierarchy) tag))
  ([h tag] (get (:parents h) tag)))
(defn ancestors
  ([tag] (ancestors (deref -global-hierarchy) tag))
  ([h tag] (get (:ancestors h) tag)))
(defn descendants
  ([tag] (descendants (deref -global-hierarchy) tag))
  ([h tag] (get (:descendants h) tag)))
(defn isa?
  ([child parent] (isa? (deref -global-hierarchy) child parent))
  ([h child parent]
   (boolean
    (or (= child parent)
        (contains? (get (:ancestors h) child) parent)
        (and (vector? child) (vector? parent)
             (= (count child) (count parent))
             (loop [i 0]
               (cond (= i (count child)) true
                     (isa? h (nth child i) (nth parent i)) (recur (inc i))
                     :else false)))))))
(defn -tf [m source sources target targets]
  (reduce (fn [ret k]
            (assoc ret k (reduce conj (get targets k (hash-set)) (cons target (get targets target)))))
          m (cons source (get sources source))))
(defn derive
  ([tag parent]
   (reset! -global-hierarchy (derive (deref -global-hierarchy) tag parent)) nil)
  ([h tag parent]
   (let [tp (:parents h) td (:descendants h) ta (:ancestors h)]
     (when (contains? (get ta tag) parent)
       (throw (str tag " already has " parent " as ancestor")))
     (when (contains? (get ta parent) tag)
       (throw (str "Cyclic derivation: " parent " has " tag " as ancestor")))
     (if (contains? (get tp tag) parent)
       h
       {:parents (assoc tp tag (conj (get tp tag (hash-set)) parent))
        :ancestors (-tf ta tag td parent ta)
        :descendants (-tf td parent ta tag td)}))))
(defn underive
  ([tag parent]
   (reset! -global-hierarchy (underive (deref -global-hierarchy) tag parent)) nil)
  ([h tag parent]
   (let [tp (:parents h)
         np (disj (get tp tag (hash-set)) parent)
         parentmap (if (seq np) (assoc tp tag np) (dissoc tp tag))
         ;; rebuild ancestor/descendant closure from scratch over the new parent map
         rebuild (reduce (fn [hh [t ps]] (reduce (fn [h2 p] (derive h2 t p)) hh ps))
                         (make-hierarchy) parentmap)]
     (if (contains? (get tp tag) parent) rebuild h))))

;; ─────────────── type / gensym / boolean ───────────────
(defn boolean [x] (if x true false))
(defn type [x] (let [m (meta x)] (if (and m (contains? m :type)) (:type m) (type-of x))))
(def -gensym-counter (atom 0))
(defn gensym
  ([] (gensym "G__"))
  ([prefix] (symbol (str prefix (swap! -gensym-counter inc)))))

;; if-let / when-let: bind the test to a fresh temp, test the TEMP, then bind the
;; (possibly destructuring) form to it. Using the raw binding form as the `if`
;; condition would break for patterns like `[k & ks]` (a vector in expression
;; position → `&` leaks as a symbol). Binding vectors are built literally.
(defmacro if-let [bv then else]
  (let [form (first bv) tst (second bv) t (gensym "if-let")]
    (list 'let (vector t tst)
          (list 'if t (list 'let (vector form t) then) else))))
(defmacro when-let [bv & body]
  (let [form (first bv) tst (second bv) t (gensym "when-let")]
    (list 'let (vector t tst)
          (list 'if t (list 'let (vector form t) (%cons 'do body)) nil))))

;; ─────────────── transients (value-level; the in-place optimization is a
;; no-op here, but the persistent result is identical, which is all a program
;; that uses the return value can observe) ───────────────
(defn transient [coll] coll)
(defn persistent! [coll] coll)
(defn conj! ([] (transient [])) ([coll] coll) ([coll x] (conj coll x)))
(defn assoc! [coll k v] (assoc coll k v))
(defn dissoc! [coll k] (dissoc coll k))
(defn disj! [coll x] (disj coll x))
(defn pop! [coll] (pop coll))

;; ─────────────── metadata combinators ───────────────
(defn vary-meta [obj f & args] (with-meta obj (apply f (meta obj) args)))

;; ─────────────── tree walking ───────────────
(defn -tree-walk [branch? children node]
  (lazy-seq
   (cons node
         (when (branch? node)
           (mapcat (fn [c] (-tree-walk branch? children c)) (children node))))))
(defn tree-seq [branch? children root] (-tree-walk branch? children root))

;; ─────────────── printing ───────────────
;; `*out*` = nil -> real stdout; an atom -> a capture buffer (used by with-out-str).
(def ^:dynamic *out* nil)
(defn -emit [s] (if (nil? *out*) (%print s) (swap! *out* str s)) nil)
(def -newline-str (%str-of (%char-of 10)))
(defn print [& xs] (-emit (apply print-str xs)))
(defn println [& xs] (-emit (str (apply print-str xs) -newline-str)))
(defn pr [& xs] (-emit (apply pr-str xs)))
(defn prn [& xs] (-emit (str (apply pr-str xs) -newline-str)))
(defn newline [] (-emit -newline-str))
(defn flush [] nil)
(defmacro with-out-str (& body)
  (list 'let (vector '-sb (list 'atom ""))
        (%cons 'binding (%cons (vector '*out* '-sb) body))
        (list 'deref '-sb)))

;; ─────────────── format / printf (subset of java.util.Formatter) ───────────────
;; Supports %[-0][width][.prec]<conv> for conv in s d x X o c b n %.
(defn -has-char? [s c] (some (fn [x] (= x c)) (%str->chars s)))
(defn -pad-str [s width left? zero?]
  (let [n (- width (count s))]
    (if (<= n 0) s
      (let [p (apply str (repeat n (if (and zero? (not left?)) "0" " ")))]
        (if left? (str s p) (str p s))))))
(defn -int->radix [n radix]
  (if (= n 0) "0"
    (let [neg? (< n 0) n (if neg? (- n) n)
          digs "0123456789abcdef"
          go (fn go [n acc] (if (= n 0) acc (recur (quot n radix) (cons (nth digs (rem n radix)) acc))))]
      (str (if neg? "-" "") (apply str (go n nil))))))
;; round the decimal string of a float to `prec` places (string-based: no
;; float->int primitive needed). Splits on the point, pads/truncates with
;; round-half-up carry propagation.
(defn -str-carry [digs]
  ;; digs: seq of digit chars for the integer part, LSB first; add 1, return chars MSB-first
  (loop [ds (seq digs) carry 1 acc nil]
    (if (nil? ds)
      (if (= carry 1) (cons \1 acc) acc)
      (let [d (+ (- (%char-code (first ds)) 48) carry)]
        (recur (next ds) (quot d 10) (cons (%char-of (+ 48 (rem d 10))) acc))))))
(defn -float->fixed [x prec]
  (let [neg? (< x 0)
        s (str (if neg? (- x) x))
        ;; strip scientific if present is out of scope; assume plain decimal repr
        dot (loop [cs (%str->chars s) i 0] (cond (nil? (seq cs)) -1 (= (first cs) \.) i :else (recur (rest cs) (inc i))))
        ipart (if (= dot -1) s (subs s 0 dot))
        fpart (if (= dot -1) "" (subs s (inc dot)))
        fpad (str fpart (apply str (repeat (max 0 (- (inc prec) (count fpart))) "0")))
        keep (subs fpad 0 prec)
        rup? (and (< prec (count fpad)) (>= (- (%char-code (nth fpad prec)) 48) 5))
        combined (str ipart keep)
        rounded (if rup? (apply str (-str-carry (reverse (%str->chars combined)))) combined)
        rlen (count rounded)
        ni (- rlen prec)
        ip2 (if (<= ni 0) "0" (subs rounded 0 ni))
        fp2 (subs rounded (max 0 ni))
        fp3 (str (apply str (repeat (max 0 (- prec (count fp2))) "0")) fp2)]
    (str (if neg? "-" "") ip2 (if (= prec 0) "" (str "." fp3)))))
;; parse one %-directive starting at index i (cs is the char vector); returns
;; [consumed-count flags width prec conv]
(defn -fmt-spec [cs i]
  (let [n (count cs)
        rd (fn rd [j flags] (if (and (< j n) (or (= (nth cs j) \-) (= (nth cs j) \0) (= (nth cs j) \+) (= (nth cs j) \space) (= (nth cs j) \#)))
                              (rd (inc j) (str flags (nth cs j))) [j flags]))
        [j flags] (rd i "")
        rn (fn rn [j acc] (if (and (< j n) (-digit? (nth cs j))) (rn (inc j) (str acc (nth cs j))) [j acc]))
        [j2 ws] (rn j "")
        [j3 ps] (if (and (< j2 n) (= (nth cs j2) \.)) (rn (inc j2) "") [j2 nil])
        conv (nth cs j3)]
    [(- (inc j3) i) flags (if (= ws "") nil (-parse-digits (%str->chars ws))) (if (nil? ps) nil (-parse-digits (%str->chars ps))) conv]))
(defn -fmt-conv [conv flags width prec arg]
  (let [left? (-has-char? flags \-)
        zero? (-has-char? flags \0)
        body (cond
               (= conv \s) (let [s (str arg)] (if (nil? prec) s (subs s 0 (min prec (count s)))))
               (= conv \d) (str arg)
               (= conv \x) (-int->radix arg 16)
               (= conv \X) (apply str (map (fn [c] (let [n (%char-code c)] (if (and (>= n 97) (<= n 122)) (%char-of (- n 32)) c))) (%str->chars (-int->radix arg 16))))
               (= conv \o) (-int->radix arg 8)
               (= conv \c) (str arg)
               (= conv \b) (str (boolean arg))
               (= conv \f) (-float->fixed arg (if (nil? prec) 6 prec))
               :else (throw (str "Unsupported format conversion: %" conv)))]
    (if (nil? width) body (-pad-str body width left? zero?))))
(defn format [fmt & args]
  (let [cs (%str->chars fmt) n (count cs)]
    (loop [i 0 args (seq args) out ""]
      (if (>= i n) out
        (let [c (nth cs i)]
          (if (= c \%)
            (if (= (nth cs (inc i)) \%)
              (recur (+ i 2) args (str out "%"))
              (if (= (nth cs (inc i)) \n)
                (recur (+ i 2) args (str out -newline-str))
                (let [[consumed flags width prec conv] (-fmt-spec cs (inc i))]
                  (recur (+ i 1 consumed) (rest args)
                         (str out (-fmt-conv conv flags width prec (first args)))))))
            (recur (inc i) args (str out (str c)))))))))
(defn printf [fmt & args] (-emit (apply format fmt args)))

;; ─────────────── realization (force a lazy result for printing) ───────────────
;; The Rust printer can't invoke thunks, so `run` calls this on the final value
;; to fully realize any lazy spine (and lazy elements) into eager collections.
;; tail-recursive so realizing/printing a large collection never overflows.
(defn -realize-list [s]
  (-rev (loop [s (seq s) acc nil] (if (nil? s) acc (recur (next s) (%cons (-realize (%first s)) acc))))))
;; Persistent structures are down-converted to the list-backed display records
;; the Rust printer understands (a PVec -> `(record 'Vector <elems>)`), so the
;; printer stays oblivious to the trie layout.
(defn -realize [x]
  (cond (lazy-seq? x) (-realize-list x)
        (%num-eq (type-of x) 'List) (-realize-list x)
        (vector? x) (record 'Vector (-realize-list (-pv-seq x)))
        (set? x) (record 'Set (-realize-list (seq x)))
        (map? x) (record 'Map (-realize-entries (seq x)))
        true x))
;; Flatten a seq of [k v] map entries into a realized (k v k v …) list for display.
(defn -realize-entries [es]
  (-rev (loop [es (seq es) acc nil]
          (if (nil? es) acc
              (let [e (first es)]
                (recur (next es) (%cons (-realize (second e)) (%cons (-realize (first e)) acc))))))))

;; ─────────────── var-defining macros ───────────────
;; `(declare a b c)` interns unbound vars (forward references); a deref before a
;; real `def` throws "Unbound".
(defmacro declare (& names)
  (%cons 'do (map (fn [n] (list 'def n)) names)))
;; `(defonce name val)` defs only if the var is not already bound.
(defmacro defonce (name val)
  (list 'if (list 'bound? (list 'var name)) nil (list 'def name val)))
;; `(defn- name [docstring] params body…)` — a PRIVATE fn (cross-namespace access
;; errors). Like `defn`, a leading docstring is skipped (so `params` is always the
;; real arg vector / multi-arity clauses).
(defmacro defn- (name p2 & more)
  (if (%num-eq (type-of p2) 'String)
    (list 'def (list '-private-meta name)
          (%cons 'fn (if (%num-eq (type-of (%first more)) 'Map)
                       (%rest more) more)))
    (if (%num-eq (type-of p2) 'Map)
      (list 'def (list '-private-meta name) (%cons 'fn more))
      (list 'def (list '-private-meta name) (%cons 'fn (%cons p2 more))))))

;; ─────────────── regex (library code over the one string primitive) ───────────────
;; A compiled pattern is `(record 'Regex pattern-string)`; `#"…"` reads as
;; `(re-pattern "…")`. A tiny backtracking engine supports literals, `.`, `*`,
;; `^`, `$` — enough for common patterns; it returns the MATCH LENGTH so re-find
;; can hand back the matched substring, not just a boolean.
(defn regexp? [x] (%num-eq (type-of x) 'Regex))
(defn re-pattern [s] (if (regexp? s) s (record 'Regex (str s))))
(defn -re-pat-chars [re] (%str->chars (if (regexp? re) (field re 0) (str re))))
;; number of leading chars of `txt` matched by `re` (both char lists), or nil.
(defn -re-len [re txt]
  (cond
    (nil? (seq re)) 0
    (and (%num-eq (first re) \$) (nil? (seq (rest re)))) (if (nil? (seq txt)) 0 nil)
    (and (seq (rest re)) (%num-eq (first (rest re)) \*))
      (-re-star-len (first re) (rest (rest re)) txt)
    (and (seq txt) (or (%num-eq (first re) \.) (%num-eq (first re) (first txt))))
      (let [n (-re-len (rest re) (rest txt))] (if (nil? n) nil (%add n 1)))
    :else nil))
;; `c* rest` against txt, greedy with backtracking; total consumed length or nil.
(defn -re-star-len [c re txt]
  (if (and (seq txt) (or (%num-eq (first txt) c) (%num-eq c \.)))
    (let [longer (-re-star-len c re (rest txt))]
      (if (nil? longer) (-re-len re txt) (%add longer 1)))
    (-re-len re txt)))
;; leftmost match of the pattern anywhere in s; returns the matched substring or nil.
(defn re-find [re s]
  (let [pc (-re-pat-chars re)
        anchored (and (seq pc) (%num-eq (first pc) \^))
        rec (if anchored (rest pc) pc)]
    (loop [t (%str->chars s)]
      (let [n (-re-len rec t)]
        (cond
          (not (nil? n)) (apply str (take n t))
          (or anchored (nil? (seq t))) nil
          :else (recur (rest t)))))))
;; whole-string match: returns s (the match) only if the pattern matches all of s.
(defn re-matches [re s]
  (let [pc (-re-pat-chars re)
        rec (if (and (seq pc) (%num-eq (first pc) \^)) (rest pc) pc)
        sc (%str->chars s)
        n (-re-len rec sc)]
    (if (and (not (nil? n)) (%num-eq n (count sc))) (apply str (take n sc)) nil)))
(defn re-seq [re s]
  (lazy-seq
    (when (seq (%str->chars s))
      (let [m (re-find re s)]
        (if (nil? m) nil
          (let [after (drop (max 1 (count m)) (%str->chars s))]
            (cons m (re-seq re (apply str after)))))))))

;; ─────────────── PersistentQueue (FIFO) ───────────────
;; `(record 'PersistentQueue items)` where items is a LIST front→back (a list, not
;; a vector — this loads before the PVec name-fields are registered). conj appends
;; to the back; first/peek read the front. `PersistentQueue/EMPTY` (or cljs's
;; `PersistentQueue.EMPTY`) resolves to `-empty-queue` in the compiler.
(def -empty-queue (record 'PersistentQueue nil))
(defn -q-append [xs x] (if (nil? (seq xs)) (%cons x nil) (%cons (first xs) (-q-append (rest xs) x))))
(extend-type PersistentQueue
  ISeqable (-seq [q] (seq (field q 0)))
  ISeq (-first [q] (first (field q 0))) (-rest [q] (record 'PersistentQueue (rest (field q 0))))
  ICounted (-count [q] (count (field q 0)))
  ICollection (-conj [q x] (record 'PersistentQueue (-q-append (field q 0) x)))
  IStack (-peek [q] (first (field q 0))) (-pop [q] (record 'PersistentQueue (rest (field q 0))))
  IEquiv (-equiv [q o] (if (or (-seqlike? o) (%num-eq (type-of o) 'PersistentQueue))
                         (-seq-eq (seq q) (seq o)) false))
  IEmptyableCollection (-empty [_] -empty-queue))

;; ─────────────── array-list (cljs's mutable growable array) ───────────────
;; A tiny mutable buffer (a record wrapping an atom holding a vector) — cljs
;; library code uses `(array-list)` + `.add`/`.isEmpty`/`.toArray`/`.clear` as a
;; local mutation optimization (e.g. medley's partition transducers).
(defn array-list [] (record 'ArrayList (%atom-new [])))
(defn -al? [al] (%num-eq (type-of al) 'ArrayList))
(defn -al-add! [al x]
  (if (-al? al) (do (%atom-set (field al 0) (conj (%atom-get (field al 0)) x)) al)
      (throw "not a mutable array-list")))
(defn -al-clear! [al] (do (%atom-set (field al 0) []) al))
;; remove and return the first element (JS array `.shift`).
(defn -al-shift! [al]
  (if (-al? al)
    (let [v (%atom-get (field al 0))]
      (%atom-set (field al 0) (vec (rest v)))
      (first v))
    (throw "not a mutable array-list")))
;; index of item in a sequential coll, or -1 (JS array/`.indexOf` semantics).
(defn -index-of [coll item]
  (loop [s (seq coll) i 0]
    (cond (nil? s) -1 (= (first s) item) i :else (recur (rest s) (%add i 1)))))
(extend-type ArrayList
  ICounted (-count [al] (count (%atom-get (field al 0))))
  ISeqable (-seq [al] (seq (%atom-get (field al 0)))))

;; ─────────────── more clojure.core (library code) ───────────────
(defn to-array [coll] (vec coll))
;; UUID: `(record 'UUID canonical-string)`; `#uuid "…"` reads as `(uuid "…")`.
(defn uuid [s] (record 'UUID (str s)))
(defn uuid? [x] (%num-eq (type-of x) 'UUID))
(def -uuid-counter (atom 0))
;; distinct (not cryptographically random) UUIDs — enough for `(not= (random-uuid) (random-uuid))`.
(defn random-uuid []
  (uuid (str "00000000-0000-4000-8000-" (swap! -uuid-counter inc))))
(extend-type UUID
  IEquiv (-equiv [u o] (if (%num-eq (type-of o) 'UUID) (= (field u 0) (field o 0)) false)))
(defn identical? [a b] (%num-eq a b))
;; keywords aren't reference-identical here (they're records), so compare by value.
(defn keyword-identical? [a b] (= a b))
(defn abs [n] (if (%lt n 0) (%sub 0 n) n))
(defn boolean? [x] (or (true? x) (false? x)))
(defn int? [x] (%num-eq (type-of x) 'Long))
(defn integer? [x] (int? x))
(defn double? [x] (%num-eq (type-of x) 'Double))
(defn float? [x] (double? x))
(defn pos-int? [x] (and (int? x) (%lt 0 x)))
(defn neg-int? [x] (and (int? x) (%lt x 0)))
(defn nat-int? [x] (and (int? x) (not (%lt x 0))))
(defn coll? [x] (or (vector? x) (map? x) (set? x) (list? x) (seq? x)))
(defn ifn? [x] (or (fn? x) (keyword? x) (map? x) (set? x) (vector? x) (symbol? x)))
(defn distinct? [& xs] (%num-eq (count xs) (count (distinct xs))))
;; comparison: numbers and strings (char-code lexicographic).
(defn -str< [a b]
  (let [as (%str->chars a) bs (%str->chars b)]
    (loop [as as bs bs]
      (cond (nil? (seq as)) (if (nil? (seq bs)) 0 -1)
            (nil? (seq bs)) 1
            (%lt (%char-code (first as)) (%char-code (first bs))) -1
            (%lt (%char-code (first bs)) (%char-code (first as))) 1
            true (recur (rest as) (rest bs))))))
(defn compare [a b]
  (cond (nil? a) (if (nil? b) 0 -1)
        (nil? b) 1
        (and (string? a) (string? b)) (-str< a b)
        (and (keyword? a) (keyword? b)) (-str< (name a) (name b))
        (and (symbol? a) (symbol? b)) (-str< (name a) (name b))
        ;; characters compare by code point (they are not numbers, so `%lt` would
        ;; error) — needed for `(sort [\c \a \b])` etc.
        (and (char? a) (char? b))
        (let [x (%char-code a) y (%char-code b)] (cond (%lt x y) -1 (%lt y x) 1 :else 0))
        (%lt a b) -1 (%lt b a) 1 :else 0))
;; replace preserves a vector input (returns a vector), else a seq.
(defn replace [smap coll]
  (let [f (fn [x] (if (contains? smap x) (get smap x) x))]
    (if (vector? coll) (mapv f coll) (map f coll))))
;; seq tail ops
(defn take-last [n coll] (loop [s (seq coll) len (count coll)] (if (%lt n (+ len 1)) (if (%num-eq len n) s (recur (rest s) (- len 1))) s)))
(defn drop-last [a & more]
  (if (nil? more) (take (- (count a) 1) a) (take (- (count (first more)) a) (first more))))
(defn split-at [n coll] [(take n coll) (drop n coll)])
(defn split-with [pred coll] [(take-while pred coll) (drop-while pred coll)])
(defn not-any? [pred coll] (not (some pred coll)))
(defn not-every? [pred coll] (not (every? pred coll)))
(defn nthnext [coll n] (loop [n n xs (seq coll)] (if (and (%lt 0 n) xs) (recur (- n 1) (next xs)) xs)))
(defn nthrest [coll n] (loop [n n xs coll] (if (and (%lt 0 n) (seq xs)) (recur (- n 1) (rest xs)) xs)))
(defn find [m k] (if (contains? m k) [k (get m k)] nil))
(defn subvec [v start & e] (vec (let [end (if (nil? e) (count v) (first e))] (take (- end start) (drop start v)))))
(defn keyword [& args]
  (if (nil? (next args))
    (let [x (first args)] (if (keyword? x) x (record 'Keyword (symbol (if (string? x) x (name x))))))
    (record 'Keyword (symbol (first args) (second args)))))
(defn fnil [f x] (fn [a & args] (apply f (if (nil? a) x a) args)))
(defn char [n] (%char-of n))
(defn char? [x] (%num-eq (type-of x) 'Char))
(defn associative? [x] (or (map? x) (vector? x) (%num-eq (type-of x) 'SortedMap)))
(defn sequential? [x] (or (vector? x) (list? x) (seq? x)))
(defn counted? [x] (or (vector? x) (map? x) (set? x) (list? x) (string? x)))
(defn indexed? [x] (vector? x))
(defn rseq [v] (reverse v))
(defn run! [f coll] (loop [s (seq coll)] (if (nil? s) nil (do (f (first s)) (recur (next s))))))
(defn -list*-seq [args] (if (nil? (next args)) (seq (first args)) (%cons (first args) (-list*-seq (next args)))))
(defn list* [& args] (-list*-seq args))
(defn reduce-kv [f init coll]
  (if (vector? coll)
    (loop [i 0 acc init s (seq coll)] (if (nil? s) acc (recur (inc i) (f acc i (first s)) (rest s))))
    (reduce (fn [acc e] (f acc (first e) (second e))) init (seq coll))))
(defn update-keys [m f] (reduce (fn [acc e] (assoc acc (f (first e)) (second e))) {} (seq m)))
(defn update-vals [m f] (reduce (fn [acc e] (assoc acc (first e) (f (second e)))) {} (seq m)))
(defn every-pred [& preds] (fn [& args] (every? (fn [p] (every? (fn [a] (p a)) args)) preds)))
(defn some-fn [& fns] (fn [& args] (some (fn [f] (some (fn [a] (f a)) args)) fns)))
;; bit ops (on integers, via the bitwise prims)
(defn bit-not [x] (%sub (%sub 0 x) 1))
(defn bit-flip [x n] (%bit-xor x (%bit-shl 1 n)))
(defn bit-set [x n] (%bit-or x (%bit-shl 1 n)))
(defn bit-clear [x n] (%bit-and x (bit-not (%bit-shl 1 n))))
(defn bit-test [x n] (not (%num-eq 0 (%bit-and x (%bit-shl 1 n)))))

;; ── SORTED collections: distinct types keeping entries in compare-order under
;; assoc/conj (so `(into (sorted-map) …)` really sorts). O(n) inserts. A SortedMap
;; stores a FLAT (k1 v1 k2 v2 …) list sorted by key — reuses the Map printer/entries.
(defn -pairs [s] (if (nil? (seq s)) nil (%cons (vector (first s) (second s)) (-pairs (rest (rest s))))))
(defn -sm-assoc [kvs k v]
  (cond (nil? (seq kvs)) (%cons k (%cons v nil))
        (= k (%first kvs)) (%cons k (%cons v (%rest (%rest kvs))))
        (%lt (compare k (%first kvs)) 0) (%cons k (%cons v kvs))
        true (%cons (%first kvs) (%cons (%first (%rest kvs)) (-sm-assoc (%rest (%rest kvs)) k v)))))
(defn -sm-get [kvs k nf]
  (cond (nil? (seq kvs)) nf (= (%first kvs) k) (%first (%rest kvs)) true (-sm-get (%rest (%rest kvs)) k nf)))
(defn -sm-has? [kvs k]
  (cond (nil? (seq kvs)) false (= (%first kvs) k) true true (-sm-has? (%rest (%rest kvs)) k)))
(defn -sm-dissoc [kvs k]
  (cond (nil? (seq kvs)) nil
        (= (%first kvs) k) (%rest (%rest kvs))
        true (%cons (%first kvs) (%cons (%first (%rest kvs)) (-sm-dissoc (%rest (%rest kvs)) k)))))
(defn sorted-map [& kvs]
  (record 'SortedMap (reduce (fn [acc p] (-sm-assoc acc (first p) (second p))) nil (-pairs kvs))))
(extend-type SortedMap
  ISeqable (-seq [m] (entries (field m 0)))
  ICounted (-count [m] (%quot (count-seq (field m 0) 0) 2))
  ILookup (-lookup [m k nf] (-sm-get (field m 0) k nf))
  IAssociative (-assoc [m k v] (record 'SortedMap (-sm-assoc (field m 0) k v)))
               (-contains-key? [m k] (-sm-has? (field m 0) k))
  IMap (-dissoc [m k] (record 'SortedMap (-sm-dissoc (field m 0) k)))
  ICollection (-conj [m e] (record 'SortedMap (-sm-assoc (field m 0) (nth e 0) (nth e 1))))
  ;; a sorted-map is = to any map (sorted or not) with the same entries.
  IEquiv (-equiv [m o] (if (map? o) (if (%num-eq (-count m) (-count o)) (keys-match (keys m) m o) false) false))
  IEmptyableCollection (-empty [_] (record 'SortedMap nil)))
(defn -ss-conj [es x]
  (cond (nil? (seq es)) (%cons x nil)
        (= x (first es)) es
        (%lt (compare x (first es)) 0) (%cons x es)
        true (%cons (first es) (-ss-conj (rest es) x))))
(defn sorted-set [& xs] (record 'SortedSet (reduce -ss-conj nil xs)))
(extend-type SortedSet
  ISeqable (-seq [s] (seq (field s 0)))
  ICounted (-count [s] (count-seq (field s 0) 0))
  ICollection (-conj [s x] (record 'SortedSet (-ss-conj (field s 0) x)))
  ILookup (-lookup [s k nf] (if (some (fn [x] (= x k)) (field s 0)) k nf))
  IEmptyableCollection (-empty [_] (record 'SortedSet nil)))
(defn sorted-map-by [cmp & kvs]
  (reduce (fn [m p] (assoc m (first p) (second p))) (record 'SortedMap nil) (-pairs kvs)))
(defn sorted-set-by [cmp & xs] (reduce conj (record 'SortedSet nil) xs))
;; lazy-cat: concatenation of the collections.
(defmacro lazy-cat (& colls) (%cons 'concat colls))

;; ─────────────── namespace reflection ───────────────
;; Backed by the runtime var registry (populated at every def). Namespaces are
;; represented by their name symbol (we have no first-class Namespace object).
(defn all-ns [] (%all-ns))
(defn find-ns [n] (if (some (fn [x] (= x n)) (%all-ns)) n nil))
(defn the-ns [n] n)
(defn ns-name [n] n)
;; {unqualified-name-symbol -> Var} for a namespace's interned vars.
(defn ns-interns [n]
  (reduce (fn [m qs] (assoc m (symbol (%sym-name qs)) (record 'Var qs)))
          (hash-map) (%ns-interns n)))
;; ns-interns minus the private vars.
(defn ns-publics [n]
  (reduce (fn [m qs]
            (if (%num-eq 2 (%bit-and (%var-flags qs) 2))
              m
              (assoc m (symbol (%sym-name qs)) (record 'Var qs))))
          (hash-map) (%ns-interns n)))
(def ns-map ns-interns)

;; ─────────────── clojure.core burn-down: predicates ───────────────
(defn any? [x] true)
(defn seqable? [x] (or (nil? x) (coll? x) (string? x)))
(defn reversible? [x] (vector? x))
(defn sorted? [x] (or (%num-eq (type-of x) 'SortedMap) (%num-eq (type-of x) 'SortedSet)))
(defn map-entry? [x] (and (vector? x) (%num-eq (count x) 2)))
(defn ident? [x] (or (keyword? x) (symbol? x)))
(defn simple-ident? [x] (and (ident? x) (nil? (namespace x))))
(defn qualified-ident? [x] (and (ident? x) (not (nil? (namespace x)))))
(defn simple-symbol? [x] (and (symbol? x) (nil? (namespace x))))
(defn qualified-symbol? [x] (and (symbol? x) (not (nil? (namespace x)))))
(defn simple-keyword? [x] (and (keyword? x) (nil? (namespace x))))
(defn qualified-keyword? [x] (and (keyword? x) (not (nil? (namespace x)))))
(defn ratio? [x] false)
(defn rational? [x] (int? x))
(defn decimal? [x] false)
(defn bytes? [x] false)
(defn class? [x] false)
(defn uri? [x] false)
(defn inst? [x] false)
(defn NaN? [x] (not (%num-eq x x)))
(defn infinite? [x] (or (%num-eq x (/ 1.0 0.0)) (%num-eq x (/ -1.0 0.0))))
(defn volatile? [x] (%num-eq (type-of x) 'Volatile))
(defn delay? [x] (%num-eq (type-of x) 'Delay))
(defn reduced? [x] (%num-eq (type-of x) 'Reduced))
(defn realized? [x] true)
(defn chunked-seq? [x] false)
(defn special-symbol? [x] (not (nil? (some (fn [s] (= x s)) '(if do let* fn* quote def loop* recur throw try catch finally var set! new .)))))
;; real `record?` is defined below with the defrecord registry (-record-types).

;; ─────────────── string->number parsing ───────────────
(defn -digit? [c] (let [n (%char-code c)] (and (%lt 47 n) (%lt n 58))))
(defn -parse-digits [cs] (reduce (fn [a c] (+ (* a 10) (- (%char-code c) 48))) 0 cs))
(defn parse-long [s]
  (let [cs (%str->chars s)
        neg (= (first cs) \-)
        ds (if (or neg (= (first cs) \+)) (rest cs) cs)]
    (if (every? -digit? (seq ds)) (let [v (-parse-digits ds)] (if neg (- v) v)) nil)))
(defn parse-boolean [s] (cond (= s "true") true (= s "false") false :else nil))
(defn parse-double [s]
  (let [cs (%str->chars s)
        neg (= (first cs) \-)
        ds (if (or neg (= (first cs) \+)) (rest cs) cs)
        dot (some (fn [c] (= c \.)) ds)]
    (if dot
      (let [ip (take-while -digit? ds)
            fp (take-while -digit? (rest (drop-while -digit? ds)))
            k (count fp)
            v (/ (+ (* (-parse-digits ip) (-pow10-p k)) (-parse-digits fp)) (-pow10-p k))]
        (if neg (- v) v))
      (let [v (-parse-digits ds)] (if neg (- v) v)))))
(defn -pow10-p [n] (if (%num-eq n 0) 1 (* 10 (-pow10-p (- n 1)))))

;; ─────────────── numeric aliases / unchecked (we auto-promote) ───────────────
(def inc' inc)
(def dec' dec)
(def +' +)
(def -' -)
(def *' *)
(defn num [x] x)
(defn rationalize [x] x)
(defn bit-and-not [x y] (%bit-and x (bit-not y)))
(defn unchecked-add [a b] (%add a b))
(defn unchecked-subtract [a b] (%sub a b))
(defn unchecked-multiply [a b] (%mul a b))
(defn unchecked-negate [a] (%sub 0 a))
(defn unchecked-inc [a] (%add a 1))
(defn unchecked-dec [a] (%sub a 1))
(defn unchecked-add-int [a b] (%add a b))
(defn unchecked-subtract-int [a b] (%sub a b))
(defn unchecked-multiply-int [a b] (%mul a b))
(defn unchecked-inc-int [a] (%add a 1))
(defn unchecked-dec-int [a] (%sub a 1))
(defn unchecked-negate-int [a] (%sub 0 a))
(defn unchecked-remainder-int [a b] (%rem a b))
(defn unchecked-divide-int [a b] (%quot a b))

;; ─────────────── dynamic vars (bindable via `binding`) ───────────────
;; Declared `^:dynamic` so references compile to a dynamic-get and `binding`
;; rebinds them. `*out*` is defined earlier (needed by the print family).
(def ^:dynamic *print-length* nil)      ;; max seq elements to print (nil = all)
(def ^:dynamic *print-level* nil)       ;; max nesting depth to print (nil = all)
(def ^:dynamic *print-readably* true)
(def ^:dynamic *print-dup* false)
(def ^:dynamic *print-meta* false)
(def ^:dynamic *print-namespace-maps* true)
(def ^:dynamic *flush-on-newline* true)
(def ^:dynamic *assert* true)
(def ^:dynamic *read-eval* true)
(def ^:dynamic *ns* 'user)
(def ^:dynamic *command-line-args* nil)
(def ^:dynamic *data-readers* {})
(def ^:dynamic *default-data-reader-fn* nil)
(def ^:dynamic *math-context* nil)
(def ^:dynamic *unchecked-math* false)
(def ^:dynamic *warn-on-reflection* false)
(def ^:dynamic *compile-files* false)
(def ^:dynamic *compile-path* "classes")
(def ^:dynamic *compiler-options* nil)
(def ^:dynamic *verbose-defrecords* false)
(def ^:dynamic *allow-unresolved-vars* false)
(def ^:dynamic *suppress-read* nil)
(def ^:dynamic *source-path* "NO_SOURCE_PATH")
(def ^:dynamic *file* "NO_SOURCE_PATH")
(def ^:dynamic *fn-loader* nil)
(def ^:dynamic *reader-resolver* nil)
(def ^:dynamic *in* nil)
(def ^:dynamic *err* nil)
(def ^:dynamic *agent* nil)
(def ^:dynamic *1 nil)
(def ^:dynamic *2 nil)
(def ^:dynamic *3 nil)
(def ^:dynamic *e nil)
(def *clojure-version* {:major 1 :minor 11 :incremental 1 :qualifier nil})
(defn clojure-version [] "1.11.1")
;; `assert` throws when its expr is falsey. Like Clojure, *assert* is consulted at
;; MACRO-EXPANSION time: when false, assert expands to nil (the check is compiled
;; out entirely — a runtime `binding` can't re-enable it).
(defmacro assert (x & msg)
  (if *assert*
    (let [detail (if (seq msg)
                   (list 'str "Assert failed: " (first msg) (%str-of (%char-of 10)) (list 'pr-str (list 'quote x)))
                   (list 'str "Assert failed: " (list 'pr-str (list 'quote x))))]
      (list 'when (list 'not x) (list 'throw detail)))
    nil))

;; ─────────────── readable printing (pr-str family, pure) ───────────────
;; escape one char for inside a readable string literal
(defn -esc-char [c]
  (let [n (%char-code c)]
    (cond (= n 34) (str (%char-of 92) (%char-of 34))
          (= n 92) (str (%char-of 92) (%char-of 92))
          (= n 10) (str (%char-of 92) "n")
          (= n 9) (str (%char-of 92) "t")
          (= n 13) (str (%char-of 92) "r")
          :else (str c))))
;; readable form of a character literal (Clojure names the special ones)
(defn -pr-char [x]
  (let [n (%char-code x) bs (%char-of 92)]
    (cond (= n 10) (str bs "newline")
          (= n 32) (str bs "space")
          (= n 9) (str bs "tab")
          (= n 13) (str bs "return")
          (= n 12) (str bs "formfeed")
          (= n 8) (str bs "backspace")
          :else (str bs x))))
;; true when the CURRENT depth exceeds *print-level* (collections beyond it -> "#")
(defn -over-level? [level] (and (not (nil? *print-level*)) (> level *print-level*)))
;; render a seq's elements honoring *print-length* (append "..." when truncated)
(defn -pr-elems [items level readable?]
  (let [lim *print-length*
        s (seq items)
        shown (if (nil? lim) s (take lim s))
        parts (map (fn [e] (-prn-el e level readable?)) shown)
        parts (if (and (not (nil? lim)) (seq (drop lim s))) (concat parts (list "...")) parts)]
    (apply str (interpose " " parts))))
(defn -pr-map-elems [es level readable?]
  (let [lim *print-length*
        s (seq es)
        shown (if (nil? lim) s (take lim s))
        parts (map (fn [e] (str (-prn-el (first e) level readable?) " " (-prn-el (second e) level readable?))) shown)
        parts (if (and (not (nil? lim)) (seq (drop lim s))) (concat parts (list "...")) parts)]
    (apply str (interpose ", " parts))))
;; the core printer. `readable?` => pr-style (quote/escape strings & chars);
;; otherwise print-style (raw). `level` starts at 1 for the top value.
(defn -prn-el [x level readable?]
  (cond (nil? x) "nil"
        (string? x) (if readable? (str (%char-of 34) (apply str (map -esc-char (%str->chars x))) (%char-of 34)) x)
        (char? x) (if readable? (-pr-char x) (str x))
        (keyword? x) (str ":" (name-with-ns x))
        (symbol? x) (str x)
        (vector? x) (if (-over-level? level) "#" (str "[" (-pr-elems x (inc level) readable?) "]"))
        (set? x) (if (-over-level? level) "#" (str "#{" (-pr-elems (seq x) (inc level) readable?) "}"))
        (map? x) (if (-over-level? level) "#" (str "{" (-pr-map-elems (seq x) (inc level) readable?) "}"))
        (or (lazy-seq? x) (list? x) (seq? x)) (if (-over-level? level) "#" (str "(" (-pr-elems x (inc level) readable?) ")"))
        true (str x)))
(defn -pr [x] (-prn-el x 1 true))
(defn -pr-print [x] (-prn-el x 1 false))
(defn name-with-ns [k] (if (nil? (namespace k)) (name k) (str (namespace k) "/" (name k))))
(defn pr-str [& xs] (apply str (interpose " " (map -pr xs))))
(defn prn-str [& xs] (str (apply pr-str xs) (%char-of 10)))
(defn print-str [& xs] (apply str (interpose " " (map -pr-print xs))))
(defn println-str [& xs] (str (apply print-str xs) (%char-of 10)))

;; ─────────────── misc combinators + control ───────────────
(defmacro when-some (bv & body)
  (list 'let [(first bv) (second bv)]
        (list 'if (list 'nil? (first bv)) nil (%cons 'do body))))
(defmacro if-some (bv then & else)
  (list 'let [(first bv) (second bv)]
        (list 'if (list 'nil? (first bv)) (if (nil? else) nil (first else)) then)))
(defmacro comment (& body) nil)
(defn comparator [pred] (fn [a b] (cond (pred a b) -1 (pred b a) 1 :else 0)))
(defn replicate [n x] (take n (repeat x)))
(defn bounded-count [n coll] (if (counted? coll) (count coll) (loop [i 0 s (seq coll)] (if (or (nil? s) (%num-eq i n)) i (recur (inc i) (next s))))))
(defn trampoline [f & args]
  (loop [r (apply f args)] (if (fn? r) (recur (r)) r)))
(defn memoize [f]
  (let [cache (atom {})]
    (fn [& args]
      (if (contains? (deref cache) args)
        (get (deref cache) args)
        (let [v (apply f args)] (swap! cache assoc args v) v)))))
;; `(letfn [(f [x] …) (g [y] …)] body)` -> a let binding each name to a
;; self-named fn. (Non-mutual / self-recursion; each fn carries its own name.)
(defmacro letfn (specs & body)
  (%cons 'let (%cons (-to-list (apply concat (map (fn [s] (list (first s) (%cons 'fn s))) specs))) body)))

;; ─────────────── transducers ───────────────
(defn reduced [x] (record 'Reduced x))
(defn unreduced [x] (if (reduced? x) (field x 0) x))
(defn ensure-reduced [x] (if (reduced? x) x (reduced x)))
(defn volatile! [x] (record 'Volatile (%cell x)))
(defn vreset! [v x] (do (%cell-set! (field v 0) 0 x) x))
(defn vswap! [v f & args] (vreset! v (apply f (%cell-ref (field v 0) 0) args)))
(defn -tr-reduce [rf init coll]
  (loop [acc init s (seq coll)]
    (if (if (nil? s) true (reduced? acc)) acc (recur (rf acc (first s)) (next s)))))
(defn transduce
  ([xform f coll] (transduce xform f (f) coll))
  ([xform f init coll] (let [rf (xform f)] (rf (unreduced (-tr-reduce rf init coll))))))
(defn sequence
  ([coll] (seq coll))
  ([xform coll] (seq (transduce xform conj [] coll))))
(defn eduction [& args] (seq (transduce (apply comp (butlast args)) conj [] (last args))))
(defn completing
  ([f] (completing f identity))
  ([f cf] (fn ([] (f)) ([x] (cf x)) ([x y] (f x y)))))
(defn cat [rf] (fn ([] (rf)) ([a] (rf a)) ([a x] (reduce rf a x))))
(defn halt-when
  ([pred] (halt-when pred nil))
  ([pred retf] (fn [rf] (fn ([] (rf)) ([a] (rf a))
                            ([a x] (if (pred x) (reduced (if retf (retf a x) x)) (rf a x)))))))
;; take/drop with a transducer arity (stateful via volatile), keeping the seq arity.
(defn take
  ([n] (fn [rf] (let [nv (volatile! n)]
                  (fn ([] (rf)) ([a] (rf a))
                      ([a x] (let [k (deref nv)] (vreset! nv (- k 1))
                               (if (%lt 0 k) (if (%lt 1 k) (rf a x) (ensure-reduced (rf a x))) (ensure-reduced a))))))))
  ([n c] (lazy-seq (if (%lt 0 n) (let [s (seq c)] (if (nil? s) nil (%cons (%first s) (take (%sub n 1) (%rest s))))) nil))))
(defn drop
  ([n] (fn [rf] (let [nv (volatile! n)]
                  (fn ([] (rf)) ([a] (rf a))
                      ([a x] (let [k (deref nv)] (vreset! nv (- k 1)) (if (%lt 0 k) a (rf a x))))))))
  ([n c] (if (%lt 0 n) (let [s (seq c)] (if (nil? s) nil (drop (%sub n 1) (%rest s)))) (seq c))))

;; ─────────────── defrecord / reify / protocols ───────────────
;; A defrecord is a deftype registered as a record; map behaviour (get/keys/seq/
;; assoc/count) is provided GENERICALLY via the field-name registry (see -rec-* and
;; the record branches in get/seq/count/assoc/contains?), so no per-record codegen.
(def -record-types (atom #{}))
(defn record? [x] (contains? (deref -record-types) (type-of x)))
(defn -rec-get [r k nf]
  (loop [fs (%field-names r) i 0]
    (cond (nil? (seq fs)) nf
          (= (keyword (first fs)) k) (field r i)
          true (recur (rest fs) (inc i)))))
(defn -rec-entries [r]
  (loop [fs (%field-names r) i 0 acc nil]
    (if (nil? (seq fs)) (reverse acc)
        (recur (rest fs) (inc i) (%cons (vector (keyword (first fs)) (field r i)) acc)))))
(defn -rec-assoc [r k v]
  (let [fs (%field-names r)
        vals (loop [fs fs i 0 acc nil]
               (if (nil? (seq fs)) (reverse acc)
                   (recur (rest fs) (inc i)
                          (%cons (if (= (keyword (first fs)) k) v (field r i)) acc))))]
    (%make-record (type-of r) vals)))
(defn -rec-has? [r k] (not (nil? (some (fn [f] (= (keyword f) k)) (%field-names r)))))
(defmacro defrecord (name fields & specs)
  (let [ctor (symbol (str "->" name))]
    (list 'do
      (%cons 'deftype (%cons name (%cons fields specs)))
      (list 'swap! '-record-types 'conj (list 'quote name))
      (list 'extend-type name
        'ILookup (list '-lookup ['r 'k 'nf] (list '-rec-get 'r 'k 'nf))
        'ICounted (list '-count ['r] (list 'nfields 'r))
        'ISeqable (list '-seq ['r] (list '-rec-entries 'r))
        'IAssociative (list '-assoc ['r 'k 'v] (list '-rec-assoc 'r 'k 'v))
                      (list '-contains-key? ['r 'k] (list '-rec-has? 'r 'k)))
      (list 'defn (symbol (str "map->" name)) ['m]
        (%cons ctor (-to-list (map (fn [f] (list 'get 'm (keyword f))) fields)))))))
;; reify: an anonymous single-instance type via a gensym'd deftype + its protocols.
(defmacro reify (& specs)
  (let [t (gensym "reify")]
    (list 'do (%cons 'deftype (%cons t (%cons [] specs))) (list (symbol (str "->" t))))))
(defmacro extend-protocol (p & body)
  ;; (extend-protocol P T1 (m [x] ..) T2 (m [x] ..)) -> (do (extend-type T1 P ..) ..)
  (%cons 'do (-extend-protocol-forms p body)))
(defn -extend-protocol-forms [p body]
  (loop [b (seq body) out nil]
    (if (nil? b) (reverse out)
      (let [ty (first b)
            impls (-to-list (take-while (fn [x] (not (symbol? x))) (rest b)))
            rest-b (drop-while (fn [x] (not (symbol? x))) (rest b))]
        (recur (seq rest-b) (%cons (%cons 'extend-type (%cons ty (%cons p impls))) out))))))

;; ─────────────── delay / promise ───────────────
(defn -force-delay [d]
  (let [done (%cell-ref (field d 1) 0)]
    (if done (%cell-ref (field d 0) 0)
        (let [v ((%cell-ref (field d 0) 0))]
          (%cell-set! (field d 0) 0 v) (%cell-set! (field d 1) 0 true) v))))
(defmacro delay (& body) (list 'record ''Delay (list '%cell (%cons 'fn (%cons (vector) body))) (list '%cell false)))
(defn force [x] (if (%num-eq (type-of x) 'Delay) (-force-delay x) x))
;; promise/deliver via an atom + spin (single-process); deref blocks until delivered.
(defn promise [] (record 'Promise (atom '-unset) (atom false)))
(defn deliver [p v] (if (deref (field p 1)) nil (do (reset! (field p 0) v) (reset! (field p 1) true) p)))
(defn -await-promise [p] (loop [] (if (deref (field p 1)) (deref (field p 0)) (recur))))
(defn realized? [x]
  (cond (%num-eq (type-of x) 'Delay) (%cell-ref (field x 1) 0)
        (%num-eq (type-of x) 'Promise) (deref (field x 1))
        :else true))
"##;
