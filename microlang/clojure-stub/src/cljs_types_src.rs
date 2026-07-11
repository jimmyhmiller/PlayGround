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
"##;
