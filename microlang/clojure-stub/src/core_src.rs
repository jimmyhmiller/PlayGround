//! `clojure.core`: the mini-Clojure standard library, written in the language on
//! top of the toolkit's primitives. Collections are LIST-BACKED tagged records
//! (`(record 'Vector elem-list)` etc.); `type-of` drives dispatch. Internals use
//! the un-shadowable `%`-prims (`%add`/`%first`/…); user-facing `+`/`=`/`first`
//! are first-class fns wrapping them (so they can be passed to `map`/`reduce`).

pub const CORE: &str = r##"
;; ─────────────── control macros ───────────────
(defmacro defn (name params & body)
  (list 'def name (%cons 'fn (%cons params body))))
(defmacro when (c & body)     (list 'if c (%cons 'do body) nil))
(defmacro when-not (c & body) (list 'if c nil (%cons 'do body)))
(defmacro cond (& cs)
  (if (nil? cs) nil
      (list 'if (%first cs) (%first (%rest cs)) (%cons 'cond (%rest (%rest cs))))))
(defmacro and (& xs)
  (cond (nil? xs) true (nil? (%rest xs)) (%first xs)
        true (list 'if (%first xs) (%cons 'and (%rest xs)) false)))
(defmacro or (& xs)
  (cond (nil? xs) false (nil? (%rest xs)) (%first xs)
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
(defn map? [x] (%num-eq (type-of x) 'Map))
(defn set? [x] (%num-eq (type-of x) 'Set))
(defn keyword? [x] (%num-eq (type-of x) 'Keyword))
(defn list? [x] (%num-eq (type-of x) 'List))
(defn string? [x] (%num-eq (type-of x) 'String))
(defn symbol? [x] (%num-eq (type-of x) 'Symbol))
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

(extend-type List
  ISeqable (-seq [l] l)
  ISeq (-first [l] (%first l)) (-rest [l] (%rest l))
  ICollection (-conj [l o] (%cons o l))
  ICounted (-count [l] (count-seq l 0))
  IIndexed (-nth [l n] (nth-seq l n))
  IEmptyableCollection (-empty [_] nil))

;; reader/code vectors (list-backed 'Vector records) — macros build & manipulate
;; binding forms like `[a 5]`, and -realize's display vectors are this shape too.
(extend-type Vector
  ISeqable (-seq [v] (field v 0))
  ICounted (-count [v] (count-seq (field v 0) 0))
  IIndexed (-nth [v n] (nth-seq (field v 0) n)))

(extend-type LazySeq
  ISeqable (-seq [l] (-force l))
  ICounted (-count [l] (count-seq l 0))
  IIndexed (-nth [l n] (nth-seq l n)))

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
(defn seq [c] (if (nil? c) nil (-seq c)))
(defn first [c] (let [s (seq c)] (if (nil? s) nil (-first s))))
(defn rest [c] (let [s (seq c)] (if (nil? s) nil (-rest s))))
(defn next [c] (seq (rest c)))
(defn cons [x c] (%cons x c))
(defn empty? [c] (nil? (seq c)))
(defn empty [c] (-empty c))
(defn count [c] (if (nil? c) 0 (-count c)))
(defn nth [c i] (-nth c i))
(defn conj [c x] (-conj c x))
(defn assoc [m k v] (-assoc m k v))
(defn contains? [c k] (-contains-key? c k))
(defn peek [c] (-peek c))
(defn pop [c] (-pop c))
(defn get [c k & r] (-lookup c k (if (nil? r) nil (%first r))))
(defn dissoc [m k] (record 'Map (kv-without (field m 0) k)))
(defn keys [m] (kv-keys (field m 0)))
(defn vals [m] (kv-vals (field m 0)))

;; tail-recursive list reverse + full realization (never grow the native stack).
(defn -rev [s] (loop [s (seq s) acc nil] (if (nil? s) acc (recur (next s) (%cons (%first s) acc)))))
(defn -to-list [s] (-rev (-rev s)))
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
        (vector? o) (nth o (first args))
        true (throw "value is not callable")))

;; ─────────────── reduce / into ───────────────
(defn reduce-seq [f acc s] (let [s (seq s)] (if (nil? s) acc (reduce-seq f (f acc (%first s)) (%rest s)))))
;; `reduce` is 2- or 3-arity (like clojure.core): `(reduce f coll)` seeds with the
;; first element (or `(f)` when empty); `(reduce f init coll)` seeds with `init`.
(defn reduce [f & args]
  (if (nil? (next args))
      (let [s (seq (first args))] (if (nil? s) (f) (reduce-seq f (%first s) (%rest s))))
      (reduce-seq f (first args) (second args))))
(defn into [to from] (reduce conj to from))

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
;; `map` is variadic over collections (like clojure.core), stopping at the
;; shortest. Beyond 3 collections is unsupported (no `apply` on the interp tiers).
(defn map [f & colls]
  (cond (nil? (next colls)) (-map1 f (first colls))
        (nil? (next (next colls))) (-map2 f (first colls) (second colls))
        (nil? (next (next (next colls)))) (-map3 f (first colls) (second colls) (nth colls 2))
        true (throw "map: only up to 3 collections supported")))
(defn filter [f c]
  (lazy-seq (let [s (seq c)]
              (cond (nil? s) nil
                    (f (%first s)) (%cons (%first s) (filter f (%rest s)))
                    true (filter f (%rest s))))))
(defn remove [f c] (filter (fn [x] (not (f x))) c))
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
(defn iterate [f x] (lazy-seq (%cons x (iterate f (f x)))))
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
(defn seq? [x] (%num-eq (type-of x) 'List))
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
(defn get-in [m ks] (if (nil? (seq ks)) m (get-in (get m (%first (seq ks))) (%rest (seq ks)))))
(defn assoc-in [m ks v]
  (if (nil? (%rest (seq ks)))
      (assoc m (%first (seq ks)) v)
      (assoc m (%first (seq ks)) (assoc-in (get m (%first (seq ks))) (%rest (seq ks)) v))))
(defn update [m k f] (assoc m k (f (get m k))))
(defn some-seq [pred s] (let [s (seq s)] (if (nil? s) nil (if (pred (%first s)) (%first s) (some-seq pred (%rest s))))))
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
(defn complement [f] (fn [x] (not (f x))))
(defn max [a & more] (reduce (fn [x y] (if (%lt x y) y x)) a more))
(defn min [a & more] (reduce (fn [x y] (if (%lt x y) x y)) a more))
(defn map-indexed-h [f i s]
  (lazy-seq (let [s (seq s)] (if (nil? s) nil (%cons (f i (%first s)) (map-indexed-h f (%add i 1) (%rest s)))))))
(defn map-indexed [f c] (map-indexed-h f 0 c))

;; if-let / when-let build a LITERAL binding vector (not syntax-quote, which
;; would produce a runtime `(vec ..)` the `let` binder can't read).
(defmacro if-let [bv then else]
  (list 'let (vector (first bv) (second bv)) (list 'if (first bv) then else)))
(defmacro when-let [bv & body]
  (list 'let (vector (first bv) (second bv)) (list 'if (first bv) (%cons 'do body) nil)))

;; ── host runtime: our stand-in for clojure.lang.RT ──
;; The interop shim rewrites `(. clojure.lang.RT (first coll))` etc. to these, so
;; real clojure/core.clj definitions (which are written in terms of RT) execute.
(def -list (fn [& xs] xs))
(defn -rt-seq [c]
  (cond (nil? c) nil
        (pvec? c) (-pv-seq c)
        (set? c) (field c 0)
        (map? c) (entries (field c 0))
        true c))
(defn -rt-first [c] (%first (-rt-seq c)))
(defn -rt-rest [c] (%rest (-rt-seq c)))
(defn -rt-next [c] (let [r (%rest (-rt-seq c))] (if (nil? r) nil r)))
(defn -rt-conj [c x]
  (cond (pvec? c) (-pv-conj c x)
        (set? c) (record 'Set (%cons x (field c 0)))
        (map? c) (assoc c (nth x 0) (nth x 1))
        (nil? c) (%cons x nil)
        true (%cons x c)))
(defn -rt-assoc [m k v]
  (cond (map? m) (record 'Map (%cons k (%cons v (kv-without (field m 0) k))))
        (nil? m) (record 'Map (%cons k (%cons v nil)))
        true m))

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
(defmacro doseq (binding & body)
  (if (> (count binding) 2)
      (throw "doseq: only a single [x coll] binding is supported")
      (let [x (first binding) coll (second binding)]
        `(loop [s# (seq ~coll)]
           (when s#
             (let [~x (first s#)] ~@body)
             (recur (next s#)))))))
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
;; deref reads an atom (atomic load), or joins a future (blocking on its worker).
(defn deref [x] (if (future? x) (%await x) (%atom-get x)))
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

(defn -seqable? [x] (cond (list? x) true (vector? x) true (set? x) true (lazy-seq? x) true true false))
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
(defn update-in [m ks f]
  (let [k (first ks)]
    (if (nil? (next ks)) (assoc m k (f (get m k))) (assoc m k (update-in (get m k) (rest ks) f)))))

;; sorting (tortoise/hare split + merge; numeric default via %lt)
(defn -default-less [a b] (%lt a b))
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
      (-msort (fn [a b] (%lt (k a) (k b))) (first args))
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
(defn -str1 [x]
  (cond (nil? x) ""
        (string? x) x
        (keyword? x) (%str-cat ":" (%str-of (field x 0)))
        (vector? x) (%str-cat "[" (%str-cat (-str-join " " x) "]"))
        (set? x) (%str-cat "#{" (%str-cat (-str-join " " (field x 0)) "}"))
        (map? x) (%str-cat "{" (%str-cat (-str-map-join (field x 0)) "}"))
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
(defn -make-multi [df] (record 'MultiFn df (atom (hash-map))))
(defn multi? [x] (%num-eq (type-of x) 'MultiFn))
(defn -add-method [mf dval method]
  (let [a (field mf 1)] (reset! a (assoc (deref a) dval method))))
(defn -multi-lookup [mf dval]
  (let [tbl (deref (field mf 1))]
    (cond (contains? tbl dval) (get tbl dval)
          (contains? tbl :default) (get tbl :default)
          true (throw (str "no method for dispatch value: " dval)))))
(defn -multi-call [mf args]
  (apply (-multi-lookup mf (apply (field mf 0) args)) args))
(defmacro defmulti (name dispatch-fn) (list 'def name (list '-make-multi dispatch-fn)))
(defmacro defmethod (name dval params & body)
  (list '-add-method name dval (%cons 'fn (%cons params body))))

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
        (pvec? x) (record 'Vector (-realize-list (-pv-seq x)))
        (set? x) (record 'Set (-realize-list (field x 0)))
        (map? x) (record 'Map (-realize-list (field x 0)))
        true x))
"##;
