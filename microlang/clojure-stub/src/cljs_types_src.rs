//! Persistent data structures ported from ClojureScript's `cljs/core.cljs`.
//!
//! The code in the `CLJS` string below is DERIVED FROM ClojureScript, which is
//! Copyright (c) Rich Hickey and the ClojureScript contributors and licensed
//! under the Eclipse Public License 1.0 (EPL-1.0). The full license and the
//! upstream source are vendored under `clojure-stub/vendor/clojurescript/`
//! (see NOTICE.md). This port therefore remains under EPL-1.0.
//!
//! The port targets microlang's host rather than JavaScript: the datatype
//! `deftype`s and their trie/HAMT algorithms are kept as close to the cljs
//! source as the host allows, with the JS host primitives swapped for microlang
//! equivalents (a small `shim` section here) and the JS-interop / perf-only
//! machinery (transients, ES6 iterators, chunked seqs, IReduce fast paths)
//! omitted in favour of the seq-based fallbacks. Every adaptation is commented
//! `;; HOST:`.

pub const CLJS: &str = r##"
;; ─────────────── HOST SHIM: JS array + bit ops the cljs datatypes use ─────────
(defn make-array [n] (%make-array n))
(defn aget [a i] (%aget a i))
(defn aset [a i v] (do (%cell-set! a i v) v))   ;; HOST: cljs aset returns the value
(defn alength [a] (%alength a))
(defn aclone [a] (%aclone a))
(defn -list->array [xs]
  (let [n (count xs) a (make-array n)]
    (loop [i 0 s (seq xs)] (if (nil? s) a (do (aset a i (first s)) (recur (inc i) (next s)))))))
(defn array [& xs] (-list->array xs))           ;; HOST: (array a b c) -> a fresh array
(defn -array-drop-last [a]                       ;; HOST: replaces (.slice arr 0 -1)
  (let [n (%sub (alength a) 1) r (make-array n)]
    (loop [i 0] (if (%lt i n) (do (aset r i (aget a i)) (recur (%add i 1))) r))))
(defn bit-and [a b] (%bit-and a b))
(defn bit-or [a b] (%bit-or a b))
(defn bit-xor [a b] (%bit-xor a b))
(defn bit-shift-left [a n] (%bit-shl a n))
(defn bit-shift-right [a n] (%bit-shr a n))
;; HOST: our ints are non-negative here (indices/counts/masked hashes), so a
;; logical shift is an arithmetic shift.
(defn bit-shift-right-zero-fill [a n] (%bit-shr a n))
(defn unsigned-bit-shift-right [a n] (%bit-shr a n))
(defn bit-count [a] (%bit-count a))
(defn hash [x] (%hash x))
(defn == [a b] (%num-eq a b))
(defn identical? [a b] (%num-eq a b))   ;; HOST: pointer identity approximated by structural =
(defn integer? [x] (%num-eq (type-of x) 'Long))
;; caching-hash degrades to recomputation (no ^:mutable field caching).
(defmacro caching-hash [coll hash-fn hash-key] (list hash-fn coll))
(defn hash-ordered-coll [coll]
  (reduce (fn [h x] (%bit-and (%bit-xor (%mul h 31) (hash x)) 2147483647)) 1 coll))
(def empty-ordered-hash nil)

;; ─────────────── protocols the vector needs beyond the core set ───────────────
(defprotocol IVector (-assoc-n [coll n val]))
(defprotocol IWithMeta (-with-meta [coll new-meta]))
(defprotocol IMeta (-meta [coll]))

;; ─────────────── PersistentVector — ported from cljs/core.cljs (EPL-1.0) ───────
;; A 32-way bit-partitioned trie with a tail. Verbatim from cljs except: field
;; access `.-x` uses microlang's registry; host array/bit ops are the shim above;
;; `.-EMPTY`/`.-EMPTY-NODE` static type fields become the globals -EMPTY-PV /
;; -EMPTY-NODE; `(js/Error. m)` -> `(throw m)`; `str_`->`str`; `js-mod`->`rem`;
;; and the transient / iterator / chunked-seq / IReduce machinery is dropped in
;; favour of seq-based fallbacks (so -seq builds a plain seq, not a ChunkedSeq).

(deftype VectorNode [edit arr])

(defn pv-fresh-node [edit] (VectorNode. edit (make-array 32)))
(defn pv-aget [node idx] (aget (.-arr node) idx))
(defn pv-aset [node idx val] (aset (.-arr node) idx val))
(defn pv-clone-node [node] (VectorNode. (.-edit node) (aclone (.-arr node))))

(defn tail-off [pv]
  (let [cnt (.-cnt pv)]
    (if (< cnt 32) 0 (bit-shift-left (bit-shift-right-zero-fill (dec cnt) 5) 5))))

(defn new-path [edit level node]
  (loop [ll level ret node]
    (if (zero? ll)
      ret
      (let [embed ret r (pv-fresh-node edit) _ (pv-aset r 0 embed)]
        (recur (- ll 5) r)))))

(defn push-tail [pv level parent tailnode]
  (let [ret (pv-clone-node parent)
        subidx (bit-and (bit-shift-right-zero-fill (dec (.-cnt pv)) level) 31)]
    (if (== 5 level)
      (do (pv-aset ret subidx tailnode) ret)
      (let [child (pv-aget parent subidx)]
        (if-not (nil? child)
          (let [node-to-insert (push-tail pv (- level 5) child tailnode)]
            (pv-aset ret subidx node-to-insert) ret)
          (let [node-to-insert (new-path nil (- level 5) tailnode)]
            (pv-aset ret subidx node-to-insert) ret))))))

(defn vector-index-out-of-bounds [i cnt]
  (throw (str "No item " i " in vector of length " cnt)))  ;; HOST: no js/Error.

(defn first-array-for-longvec [pv]
  (loop [node (.-root pv) level (.-shift pv)]
    (if (pos? level) (recur (pv-aget node 0) (- level 5)) (.-arr node))))

(defn unchecked-array-for [pv i]
  (if (>= i (tail-off pv))
    (.-tail pv)
    (loop [node (.-root pv) level (.-shift pv)]
      (if (pos? level)
        (recur (pv-aget node (bit-and (bit-shift-right-zero-fill i level) 31)) (- level 5))
        (.-arr node)))))

(defn array-for [pv i]
  (if (if (<= 0 i) (< i (.-cnt pv)) false)
    (unchecked-array-for pv i)
    (vector-index-out-of-bounds i (.-cnt pv))))

(defn do-assoc [pv level node i val]
  (let [ret (pv-clone-node node)]
    (if (zero? level)
      (do (pv-aset ret (bit-and i 31) val) ret)
      (let [subidx (bit-and (bit-shift-right-zero-fill i level) 31)]
        (pv-aset ret subidx (do-assoc pv (- level 5) (pv-aget node subidx) i val)) ret))))

(defn pop-tail [pv level node]
  (let [subidx (bit-and (bit-shift-right-zero-fill (- (.-cnt pv) 2) level) 31)]
    (cond
      (> level 5) (let [new-child (pop-tail pv (- level 5) (pv-aget node subidx))]
                    (if (if (nil? new-child) (zero? subidx) false)
                      nil
                      (let [ret (pv-clone-node node)] (pv-aset ret subidx new-child) ret)))
      (zero? subidx) nil
      :else (let [ret (pv-clone-node node)] (pv-aset ret subidx nil) ret))))

;; HOST: seq over the vector, built back-to-front with a tail loop (replaces
;; cljs's IndexedSeq/ChunkedSeq).
(defn -pv-seq [pv]
  (loop [i (%sub (.-cnt pv) 1) acc nil]
    (if (%lt i 0) acc (recur (%sub i 1) (%cons (aget (unchecked-array-for pv i) (bit-and i 31)) acc)))))

(deftype PersistentVector [meta cnt shift root tail __hash]
  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta) coll (PersistentVector. new-meta cnt shift root tail __hash)))
  IMeta
  (-meta [coll] meta)

  IStack
  (-peek [coll] (when (> cnt 0) (-nth coll (dec cnt))))
  (-pop [coll]
    (cond
      (zero? cnt) (throw "Can't pop empty vector")
      (== 1 cnt) (-with-meta -EMPTY-PV meta)
      (< 1 (- cnt (tail-off coll)))
        (PersistentVector. meta (dec cnt) shift root (-array-drop-last tail) nil)
      :else (let [new-tail (unchecked-array-for coll (- cnt 2))
                  nr (pop-tail coll shift root)
                  new-root (if (nil? nr) -EMPTY-NODE nr)
                  cnt-1 (dec cnt)]
              (if (if (< 5 shift) (nil? (pv-aget new-root 1)) false)
                (PersistentVector. meta cnt-1 (- shift 5) (pv-aget new-root 0) new-tail nil)
                (PersistentVector. meta cnt-1 shift new-root new-tail nil)))))

  ICollection
  (-conj [coll o]
    (if (< (- cnt (tail-off coll)) 32)
      (let [len (alength tail)
            new-tail (make-array (inc len))]
        (dotimes [i len] (aset new-tail i (aget tail i)))
        (aset new-tail len o)
        (PersistentVector. meta (inc cnt) shift root new-tail nil))
      (let [root-overflow? (> (bit-shift-right-zero-fill cnt 5) (bit-shift-left 1 shift))
            new-shift (if root-overflow? (+ shift 5) shift)
            new-root (if root-overflow?
                       (let [n-r (pv-fresh-node nil)]
                         (pv-aset n-r 0 root)
                         (pv-aset n-r 1 (new-path nil shift (VectorNode. nil tail)))
                         n-r)
                       (push-tail coll shift root (VectorNode. nil tail)))]
        (PersistentVector. meta (inc cnt) new-shift new-root (array o) nil))))

  IEmptyableCollection
  (-empty [coll] (-with-meta -EMPTY-PV meta))

  IEquiv
  (-equiv [coll other]
    (if (vector? other)
      (if (== cnt (count other)) (-seq-eq coll other) false)
      false))

  IHash
  (-hash [coll] (caching-hash coll hash-ordered-coll __hash))

  ISeqable
  (-seq [coll] (-pv-seq coll))

  ICounted
  (-count [coll] cnt)

  IIndexed
  (-nth [coll n] (aget (array-for coll n) (bit-and n 31)))
  (-nth [coll n not-found]
    (if (if (<= 0 n) (< n cnt) false)
      (aget (unchecked-array-for coll n) (bit-and n 31))
      not-found))

  ILookup
  (-lookup [coll k] (-lookup coll k nil))
  (-lookup [coll k not-found] (if (number? k) (-nth coll k not-found) not-found))

  IAssociative
  (-assoc [coll k v]
    (if (number? k) (-assoc-n coll k v) (throw "Vector's key for assoc must be a number.")))
  (-contains-key? [coll k] (if (integer? k) (if (<= 0 k) (< k cnt) false) false))

  IVector
  (-assoc-n [coll n val]
    (cond
      (if (<= 0 n) (< n cnt) false)
      (if (<= (tail-off coll) n)
        (let [new-tail (aclone tail)]
          (aset new-tail (bit-and n 31) val)
          (PersistentVector. meta cnt shift root new-tail nil))
        (PersistentVector. meta cnt shift (do-assoc coll shift root n val) tail nil))
      (== n cnt) (-conj coll val)
      :else (throw (str "Index " n " out of bounds  [0," cnt "]"))))

  IFn
  (-invoke [coll k] (if (number? k) (-nth coll k) (throw "Key must be integer"))))

;; HOST: static type fields (.-EMPTY / .-EMPTY-NODE, set via set!) become globals.
(def -EMPTY-NODE (VectorNode. nil (make-array 32)))
(def -EMPTY-PV (PersistentVector. nil 0 5 -EMPTY-NODE (array) empty-ordered-hash))

(defn vector? [x] (%num-eq (type-of x) 'PersistentVector))
(defn vec [coll] (reduce conj -EMPTY-PV (seq coll)))
(defn vector [& args] (vec args))

;; metadata now flows through IMeta/IWithMeta (PersistentVector carries it in its
;; `meta` field); most values drop meta (the Object default).
(extend-type Object
  IWithMeta (-with-meta [x m] x)
  IMeta (-meta [x] nil))
(defn with-meta [x m] (-with-meta x m))
(defn meta [x] (-meta x))
;; A Var's metadata: its :name and :ns, derived from the qualified sym it wraps.
;; (:doc/:arglists/:macro would need a capture pass; not modelled yet.)
(extend-type Var
  IMeta (-meta [v] (hash-map :name (%sym-name (field v 0)) :ns (%sym-ns (field v 0)))))

;; ─────────────── PersistentArrayMap — ported from cljs/core.cljs (EPL-1.0) ─────
;; A small map backed by a flat [k0 v0 k1 v1 …] array with linear scan. Verbatim
;; except: HOST: key lookup uses a single equiv scan (cljs specialises by key
;; type); map entries are `[k v]` vectors (not a MapEntry deftype) and -seq builds
;; them directly (not PersistentArrayMapSeq); promotion to PersistentHashMap at
;; the 8-entry threshold is DEFERRED until the HAMT is ported (array map just keeps
;; growing — O(n), still correct); static .-EMPTY / iterator / transient dropped.
(defprotocol IMap (-dissoc [coll k]))

(defn array-index-of [arr k]
  (let [len (alength arr)]
    (loop [i 0]
      (cond (%lt (%sub len 1) i) -1
            (-eq2 (aget arr i) k) i
            :else (recur (%add i 2))))))
(defn array-map-index-of [m k] (array-index-of (.-arr m) k))
(defn array-extend-kv [arr k v]
  (let [l (alength arr) narr (make-array (%add l 2))]
    (loop [i 0] (if (%lt i l) (do (aset narr i (aget arr i)) (recur (%add i 1))) nil))
    (aset narr l k) (aset narr (%add l 1) v) narr))

;; -seq as a seq of `[k v]` vectors, built back-to-front (tail-recursive).
(defn persistent-array-map-seq [arr]
  (loop [i (%sub (alength arr) 2) acc nil]
    (if (%lt i 0) acc (recur (%sub i 2) (%cons (vector (aget arr i) (aget arr (%add i 1))) acc)))))

(deftype PersistentArrayMap [meta cnt arr __hash]
  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta) coll (PersistentArrayMap. new-meta cnt arr __hash)))
  IMeta
  (-meta [coll] meta)

  ICollection
  (-conj [coll entry]
    (if (vector? entry)
      (-assoc coll (-nth entry 0) (-nth entry 1))
      (loop [ret coll es (seq entry)]
        (if (nil? es)
          ret
          (let [e (first es)]
            (if (vector? e)
              (recur (-assoc ret (-nth e 0) (-nth e 1)) (next es))
              (throw "conj on a map takes map entries or seqables of map entries")))))))

  IEmptyableCollection
  (-empty [coll] (-with-meta -EMPTY-PAM meta))

  IEquiv
  (-equiv [coll other]
    (if (map? other)
      (if (== cnt (-count other))
        (loop [i 0]
          (if (%lt i (alength arr))
            (let [v (-lookup other (aget arr i) -lookup-sentinel)]
              (if-not (identical? v -lookup-sentinel)
                (if (-eq2 (aget arr (inc i)) v) (recur (+ i 2)) false)
                false))
            true))
        false)
      false))

  IHash
  (-hash [coll] (caching-hash coll hash-unordered-coll __hash))

  ISeqable
  (-seq [coll] (persistent-array-map-seq arr))

  ICounted
  (-count [coll] cnt)

  ILookup
  (-lookup [coll k] (-lookup coll k nil))
  (-lookup [coll k not-found]
    (let [idx (array-map-index-of coll k)]
      (if (== idx -1) not-found (aget arr (inc idx)))))

  IAssociative
  (-assoc [coll k v]
    (let [idx (array-map-index-of coll k)]
      (cond
        (== idx -1)
        (if (%lt cnt 8)
          (let [arr (array-extend-kv arr k v)] (PersistentArrayMap. meta (inc cnt) arr nil))
          ;; promote to PersistentHashMap past the 8-entry threshold (as cljs does)
          (-assoc (reduce (fn [m e] (-conj m e)) -EMPTY-PHM (persistent-array-map-seq arr)) k v))
        (identical? v (aget arr (inc idx)))
        coll
        :else
        (let [narr (aclone arr) _ (aset narr (inc idx) v)] (PersistentArrayMap. meta cnt narr nil)))))
  (-contains-key? [coll k] (not (== (array-map-index-of coll k) -1)))

  IMap
  (-dissoc [coll k]
    (let [idx (array-map-index-of coll k)]
      (if (>= idx 0)
        (let [len (alength arr) new-len (- len 2)]
          (if (zero? new-len)
            (-empty coll)
            (let [new-arr (make-array new-len)]
              (loop [s 0 d 0]
                (cond
                  (>= s len) (PersistentArrayMap. meta (dec cnt) new-arr nil)
                  (-eq2 k (aget arr s)) (recur (+ s 2) d)
                  :else (do (aset new-arr d (aget arr s))
                            (aset new-arr (inc d) (aget arr (inc s)))
                            (recur (+ s 2) (+ d 2))))))))
        coll)))

  IFn
  (-invoke [coll k] (-lookup coll k))
  (-invoke [coll k not-found] (-lookup coll k not-found)))

;; HOST: lookup-sentinel is cljs's (js-obj); a fresh unshareable record here.
(def -lookup-sentinel (record 'LookupSentinel nil))
(defn hash-unordered-coll [coll]
  (reduce (fn [h e] (%bit-and (%add h (hash e)) 2147483647)) 0 coll))
(def -EMPTY-PAM (PersistentArrayMap. nil 0 (array) nil))
(defn map? [x] (%num-eq (type-of x) 'PersistentArrayMap))
(defn -kvs->map [kvs]
  (loop [s (seq kvs) m -EMPTY-PAM]
    (if (nil? s) m (recur (next (next s)) (-assoc m (first s) (second s))))))
(defn hash-map [& kvs] (-kvs->map kvs))
(defn array-map [& kvs] (-kvs->map kvs))
;; variadic assoc/dissoc (clojure.core takes multiple keys); the 3-arg / 2-arg
;; forms in core.clj dispatched to -assoc/-dissoc are superseded here.
(defn assoc [m k v & kvs]
  (loop [m (-assoc m k v) s (seq kvs)]
    (if (nil? s) m (recur (-assoc m (first s) (second s)) (next (next s))))))
(defn dissoc [m & ks]
  (loop [m m s (seq ks)] (if (nil? s) m (recur (-dissoc m (first s)) (next s)))))
(defn keys [m] (map first (seq m)))
(defn vals [m] (map second (seq m)))
(defn key [e] (-nth e 0))
(defn val [e] (-nth e 1))

;; ─────────────── PersistentHashMap (HAMT) — ported from cljs/core.cljs (EPL-1.0) ─
;; A hash array-mapped trie: BitmapIndexedNode / ArrayNode / HashCollisionNode +
;; PersistentHashMap. Verbatim except HOST: the nodes' cljs `Object` methods
;; (.inode-assoc/.inode-lookup/.inode-without/.kv-reduce), which cljs calls via
;; `.method`, are an `INode` protocol here (an Object method named `keys`/`get`
;; would collide with core fns if made dispatchable). The mutable-cell `Box`
;; out-param is a %cell; static .-EMPTY nodes are globals; the transient
;; assoc!/without!/ensure-editable paths and the NodeSeq/ArrayNodeSeq types are
;; dropped — -seq is built from kv-reduce; key equality via -eq2.
(defprotocol INode
  (-inode-assoc [node shift hash key val added-leaf?])
  (-inode-without [node shift hash key])
  (-inode-lookup [node shift hash key not-found])
  (-inode-kv-reduce [node f init]))

(defn -box [v] (%cell v))
(defn -box-val [b] (%cell-ref b 0))
(defn -box-set! [b v] (%cell-set! b 0 v))
(defn key-test [a b] (-eq2 a b))
(defn array-copy [src si dst di len]
  (loop [i 0] (if (%lt i len) (do (aset dst (%add di i) (aget src (%add si i))) (recur (%add i 1))) dst)))
(defn clone-and-set
  ([arr i a] (let [r (aclone arr)] (aset r i a) r))
  ([arr i a j b] (let [r (aclone arr)] (aset r i a) (aset r j b) r)))
(defn remove-pair [arr i]
  (let [new-arr (make-array (%sub (alength arr) 2))]
    (array-copy arr 0 new-arr 0 (%mul 2 i))
    (array-copy arr (%mul 2 (inc i)) new-arr (%mul 2 i) (%sub (alength new-arr) (%mul 2 i)))
    new-arr))
(defn mask [hash shift] (bit-and (bit-shift-right-zero-fill hash shift) 31))
(defn bitpos [hash shift] (bit-shift-left 1 (mask hash shift)))
(defn bitmap-indexed-node-index [bitmap bit] (bit-count (bit-and bitmap (dec bit))))
(defn inode-kv-reduce [arr f init]
  (let [len (alength arr)]
    (loop [i 0 init init]
      (if (%lt i len)
        (recur (%add i 2)
               (let [k (aget arr i)]
                 (if-not (nil? k)
                   (f init k (aget arr (inc i)))
                   (let [node (aget arr (inc i))] (if-not (nil? node) (-inode-kv-reduce node f init) init)))))
        init))))

(deftype BitmapIndexedNode [edit bitmap arr]
  INode
  (-inode-assoc [inode shift hash key val added-leaf?]
    (let [bit (bitpos hash shift) idx (bitmap-indexed-node-index bitmap bit)]
      (if (zero? (bit-and bitmap bit))
        (let [n (bit-count bitmap)]
          (if (>= n 16)
            (let [nodes (make-array 32) jdx (mask hash shift)]
              (aset nodes jdx (-inode-assoc -EMPTY-BIN (+ shift 5) hash key val added-leaf?))
              (loop [i 0 j 0]
                (if (%lt i 32)
                  (if (zero? (bit-and (bit-shift-right-zero-fill bitmap i) 1))
                    (recur (inc i) j)
                    (do (aset nodes i
                              (if-not (nil? (aget arr j))
                                ;; HOST: `hash` here is the method's hash PARAM, so call the prim.
                                (-inode-assoc -EMPTY-BIN (+ shift 5) (%hash (aget arr j)) (aget arr j) (aget arr (inc j)) added-leaf?)
                                (aget arr (inc j))))
                        (recur (inc i) (+ j 2))))
                  (ArrayNode. nil (inc n) nodes))))
            (let [new-arr (make-array (%mul 2 (inc n)))]
              (array-copy arr 0 new-arr 0 (%mul 2 idx))
              (aset new-arr (%mul 2 idx) key)
              (aset new-arr (inc (%mul 2 idx)) val)
              (array-copy arr (%mul 2 idx) new-arr (%mul 2 (inc idx)) (%mul 2 (- n idx)))
              (-box-set! added-leaf? true)
              (BitmapIndexedNode. nil (bit-or bitmap bit) new-arr))))
        (let [key-or-nil (aget arr (%mul 2 idx)) val-or-node (aget arr (inc (%mul 2 idx)))]
          (cond (nil? key-or-nil)
                (let [n (-inode-assoc val-or-node (+ shift 5) hash key val added-leaf?)]
                  (if (identical? n val-or-node) inode
                      (BitmapIndexedNode. nil bitmap (clone-and-set arr (inc (%mul 2 idx)) n))))
                (key-test key key-or-nil)
                (if (identical? val val-or-node) inode
                    (BitmapIndexedNode. nil bitmap (clone-and-set arr (inc (%mul 2 idx)) val)))
                :else
                (do (-box-set! added-leaf? true)
                    (BitmapIndexedNode. nil bitmap
                      (clone-and-set arr (%mul 2 idx) nil (inc (%mul 2 idx))
                                     (create-node (+ shift 5) key-or-nil val-or-node hash key val)))))))))
  (-inode-without [inode shift hash key]
    (let [bit (bitpos hash shift)]
      (if (zero? (bit-and bitmap bit)) inode
        (let [idx (bitmap-indexed-node-index bitmap bit)
              key-or-nil (aget arr (%mul 2 idx)) val-or-node (aget arr (inc (%mul 2 idx)))]
          (cond (nil? key-or-nil)
                (let [n (-inode-without val-or-node (+ shift 5) hash key)]
                  (cond (identical? n val-or-node) inode
                        (not (nil? n)) (BitmapIndexedNode. nil bitmap (clone-and-set arr (inc (%mul 2 idx)) n))
                        (== bitmap bit) nil
                        :else (BitmapIndexedNode. nil (bit-xor bitmap bit) (remove-pair arr idx))))
                (key-test key key-or-nil)
                (if (== bitmap bit) nil (BitmapIndexedNode. nil (bit-xor bitmap bit) (remove-pair arr idx)))
                :else inode)))))
  (-inode-lookup [inode shift hash key not-found]
    (let [bit (bitpos hash shift)]
      (if (zero? (bit-and bitmap bit)) not-found
        (let [idx (bitmap-indexed-node-index bitmap bit)
              key-or-nil (aget arr (%mul 2 idx)) val-or-node (aget arr (inc (%mul 2 idx)))]
          (cond (nil? key-or-nil) (-inode-lookup val-or-node (+ shift 5) hash key not-found)
                (key-test key key-or-nil) val-or-node
                :else not-found)))))
  (-inode-kv-reduce [inode f init] (inode-kv-reduce arr f init)))

(deftype ArrayNode [edit cnt arr]
  INode
  (-inode-assoc [inode shift hash key val added-leaf?]
    (let [idx (mask hash shift) node (aget arr idx)]
      (if (nil? node)
        (ArrayNode. nil (inc cnt) (clone-and-set arr idx (-inode-assoc -EMPTY-BIN (+ shift 5) hash key val added-leaf?)))
        (let [n (-inode-assoc node (+ shift 5) hash key val added-leaf?)]
          (if (identical? n node) inode (ArrayNode. nil cnt (clone-and-set arr idx n)))))))
  (-inode-without [inode shift hash key]
    (let [idx (mask hash shift) node (aget arr idx)]
      (if-not (nil? node)
        (let [n (-inode-without node (+ shift 5) hash key)]
          (cond (identical? n node) inode
                (nil? n) (if (<= cnt 8) (pack-array-node inode nil idx) (ArrayNode. nil (dec cnt) (clone-and-set arr idx n)))
                :else (ArrayNode. nil cnt (clone-and-set arr idx n))))
        inode)))
  (-inode-lookup [inode shift hash key not-found]
    (let [idx (mask hash shift) node (aget arr idx)]
      (if-not (nil? node) (-inode-lookup node (+ shift 5) hash key not-found) not-found)))
  (-inode-kv-reduce [inode f init]
    (let [len (alength arr)]
      (loop [i 0 init init]
        (if (%lt i len)
          (recur (inc i) (let [node (aget arr i)] (if-not (nil? node) (-inode-kv-reduce node f init) init)))
          init)))))

(defn pack-array-node [array-node edit idx]
  (let [arr (.-arr array-node) len (alength arr) new-arr (make-array (%mul 2 (dec (.-cnt array-node))))]
    (loop [i 0 j 1 bitmap 0]
      (if (%lt i len)
        (if (if (not (== i idx)) (not (nil? (aget arr i))) false)
          (do (aset new-arr j (aget arr i)) (recur (inc i) (%add j 2) (bit-or bitmap (bit-shift-left 1 i))))
          (recur (inc i) j bitmap))
        (BitmapIndexedNode. edit bitmap new-arr)))))

(defn hash-collision-node-find-index [arr cnt key]
  (let [lim (%mul 2 cnt)]
    (loop [i 0] (if (%lt i lim) (if (key-test key (aget arr i)) i (recur (%add i 2))) -1))))

(deftype HashCollisionNode [edit collision-hash cnt arr]
  INode
  (-inode-assoc [inode shift hash key val added-leaf?]
    (if (== hash collision-hash)
      (let [idx (hash-collision-node-find-index arr cnt key)]
        (if (== idx -1)
          (let [len (%mul 2 cnt) new-arr (make-array (%add len 2))]
            (array-copy arr 0 new-arr 0 len)
            (aset new-arr len key) (aset new-arr (inc len) val)
            (-box-set! added-leaf? true)
            (HashCollisionNode. nil collision-hash (inc cnt) new-arr))
          (if (-eq2 (aget arr (inc idx)) val) inode
              (HashCollisionNode. nil collision-hash cnt (clone-and-set arr (inc idx) val)))))
      (-inode-assoc (BitmapIndexedNode. nil (bitpos collision-hash shift) (array nil inode)) shift hash key val added-leaf?)))
  (-inode-without [inode shift hash key]
    (let [idx (hash-collision-node-find-index arr cnt key)]
      (cond (== idx -1) inode
            (== cnt 1) nil
            :else (HashCollisionNode. nil collision-hash (dec cnt) (remove-pair arr (%quot idx 2))))))
  (-inode-lookup [inode shift hash key not-found]
    (let [idx (hash-collision-node-find-index arr cnt key)]
      (if (%lt idx 0) not-found (aget arr (inc idx)))))
  (-inode-kv-reduce [inode f init] (inode-kv-reduce arr f init)))

(defn create-node [shift key1 val1 key2hash key2 val2]
  (let [key1hash (hash key1)]
    (if (== key1hash key2hash)
      (HashCollisionNode. nil key1hash 2 (array key1 val1 key2 val2))
      (let [added-leaf? (-box false)]
        (-inode-assoc (-inode-assoc -EMPTY-BIN shift key1hash key1 val1 added-leaf?) shift key2hash key2 val2 added-leaf?)))))

(def -EMPTY-BIN (BitmapIndexedNode. nil 0 (make-array 0)))

(defn equiv-map [m other]
  (if (map? other)
    (if (== (-count m) (-count other))
      (loop [es (seq m)]
        (if (nil? es) true
            (let [e (first es)]
              (if (-eq2 (-lookup other (key e) -lookup-sentinel) (val e)) (recur (next es)) false))))
      false)
    false))

(deftype PersistentHashMap [meta cnt root has-nil? nil-val __hash]
  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta) coll (PersistentHashMap. new-meta cnt root has-nil? nil-val __hash)))
  IMeta
  (-meta [coll] meta)

  ICollection
  (-conj [coll entry]
    (if (vector? entry)
      (-assoc coll (-nth entry 0) (-nth entry 1))
      (loop [ret coll es (seq entry)]
        (if (nil? es) ret
            (let [e (first es)]
              (if (vector? e) (recur (-assoc ret (-nth e 0) (-nth e 1)) (next es))
                  (throw "conj on a map takes map entries or seqables of map entries")))))))

  IEmptyableCollection
  (-empty [coll] (-with-meta -EMPTY-PHM meta))

  IEquiv
  (-equiv [coll other] (equiv-map coll other))

  IHash
  (-hash [coll] (caching-hash coll hash-unordered-coll __hash))

  ISeqable
  (-seq [coll]
    (if (pos? cnt)
      (let [s (if (nil? root) nil (-inode-kv-reduce root (fn [acc k v] (%cons (vector k v) acc)) nil))]
        (if has-nil? (%cons (vector nil nil-val) s) s))
      nil))

  ICounted
  (-count [coll] cnt)

  ILookup
  (-lookup [coll k] (-lookup coll k nil))
  (-lookup [coll k not-found]
    (cond (nil? k) (if has-nil? nil-val not-found)
          (nil? root) not-found
          :else (-inode-lookup root 0 (hash k) k not-found)))

  IAssociative
  (-assoc [coll k v]
    (if (nil? k)
      (if (if has-nil? (identical? v nil-val) false) coll
          (PersistentHashMap. meta (if has-nil? cnt (inc cnt)) root true v nil))
      (let [added-leaf? (-box false)
            new-root (-inode-assoc (if (nil? root) -EMPTY-BIN root) 0 (hash k) k v added-leaf?)]
        (if (identical? new-root root) coll
            (PersistentHashMap. meta (if (-box-val added-leaf?) (inc cnt) cnt) new-root has-nil? nil-val nil)))))
  (-contains-key? [coll k]
    (cond (nil? k) has-nil?
          (nil? root) false
          :else (not (identical? (-inode-lookup root 0 (hash k) k -lookup-sentinel) -lookup-sentinel))))

  IMap
  (-dissoc [coll k]
    (cond (nil? k) (if has-nil? (PersistentHashMap. meta (dec cnt) root false nil nil) coll)
          (nil? root) coll
          :else (let [new-root (-inode-without root 0 (hash k) k)]
                  (if (identical? new-root root) coll
                      (PersistentHashMap. meta (dec cnt) new-root has-nil? nil-val nil)))))

  IFn
  (-invoke [coll k] (-lookup coll k))
  (-invoke [coll k not-found] (-lookup coll k not-found)))

(def -EMPTY-PHM (PersistentHashMap. nil 0 nil false nil nil))
;; map? now recognises both persistent map types.
(defn map? [x] (let [t (type-of x)] (if (%num-eq t 'PersistentArrayMap) true (%num-eq t 'PersistentHashMap))))

;; ─────────────── PersistentHashSet — ported from cljs/core.cljs (EPL-1.0) ──────
;; A set is a wrapper over a map whose keys are the elements. Verbatim except
;; HOST: -lookup uses -contains-key? (not -find/MapEntry); -equiv scans the seq;
;; -seq/-count delegate to the backing map; iterator/transient dropped.
(defprotocol ISet (-disjoin [coll v]))
(deftype PersistentHashSet [meta hash-map __hash]
  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta) coll (PersistentHashSet. new-meta hash-map __hash)))
  IMeta
  (-meta [coll] meta)

  ICollection
  (-conj [coll o]
    (let [m (-assoc hash-map o nil)]
      (if (identical? m hash-map) coll (PersistentHashSet. meta m nil))))

  IEmptyableCollection
  (-empty [coll] (-with-meta -EMPTY-PHS meta))

  IEquiv
  (-equiv [coll other]
    (if (set? other)
      (if (== (-count coll) (-count other))
        (loop [ks (seq coll)]
          (if (nil? ks) true (if (-contains-key? (.-hash-map other) (first ks)) (recur (next ks)) false)))
        false)
      false))

  IHash
  (-hash [coll] (caching-hash coll hash-unordered-coll __hash))

  ISeqable
  (-seq [coll] (keys hash-map))

  ICounted
  (-count [coll] (-count hash-map))

  ILookup
  (-lookup [coll v] (-lookup coll v nil))
  (-lookup [coll v not-found] (if (-contains-key? hash-map v) v not-found))

  IAssociative
  (-contains-key? [coll k] (-contains-key? hash-map k))

  ISet
  (-disjoin [coll v]
    (let [m (-dissoc hash-map v)]
      (if (identical? m hash-map) coll (PersistentHashSet. meta m nil))))

  IFn
  (-invoke [coll k] (-lookup coll k)))

(def -EMPTY-PHS (PersistentHashSet. nil -EMPTY-PHM nil))
(defn set? [x] (%num-eq (type-of x) 'PersistentHashSet))
(defn set [coll] (reduce (fn [s x] (-conj s x)) -EMPTY-PHS (seq coll)))
(defn hash-set [& es] (set es))
(defn disj [s & vs] (loop [s s vs (seq vs)] (if (nil? vs) s (recur (-disjoin s (first vs)) (next vs)))))
"##;
