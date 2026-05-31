;; Port of clojure/core/protocols.clj — the reduce machinery.
;;
;; Faithful to upstream's protocol structure (CollReduce / InternalReduce /
;; IKVReduce + seq-reduce), but the impls are limited to the types this
;; runtime actually represents. The JVM-specific fast paths (IChunkedSeq,
;; StringSeq, Iterable iterators, IReduceInit) are omitted: they would
;; never dispatch here (no host values report those ids) and their bodies
;; reference host methods (chunk-first, .iterator, .reduce) we don't have.
;; Our seq types (cons, lazy-seq, vector) all reduce correctly through the
;; Object impl's seq-reduce -> internal-reduce -> naive first/next loop,
;; including `reduced`-based early termination.
(ns clojure.core.protocols)

(defprotocol CollReduce
  "Protocol for collection types that can implement reduce faster than
  first/next recursion. Called by clojure.core/reduce."
  (coll-reduce [coll f] [coll f val]))

(defprotocol InternalReduce
  "Protocol for concrete seq types that can reduce themselves
   faster than first/next recursion. Called by clojure.core/reduce."
  (internal-reduce [seq f start]))

(defn- naive-seq-reduce
  "Reduces a seq, ignoring any opportunities to switch to a more
  specialized implementation."
  [s f val]
  (loop [s (seq s)
         val val]
    (if s
      (let [ret (f val (first s))]
        (if (reduced? ret)
          @ret
          (recur (next s) ret)))
      val)))

(extend-protocol InternalReduce
  nil
  (internal-reduce
   [s f val]
   val)

  java.lang.Object
  (internal-reduce
   [s f val]
   (naive-seq-reduce s f val)))

(defn- seq-reduce
  ([coll f]
     (if-let [s (seq coll)]
       (internal-reduce (next s) f (first s))
       (f)))
  ([coll f val]
     (let [s (seq coll)]
       (internal-reduce s f val))))

(extend-protocol CollReduce
  nil
  (coll-reduce
   ([coll f] (f))
   ([coll f val] val))

  java.lang.Object
  (coll-reduce
   ([coll f] (seq-reduce coll f))
   ([coll f val] (seq-reduce coll f val))))

(defprotocol IKVReduce
  "Protocol for concrete associative types that can reduce themselves
   via a function of key and val faster than first/next recursion over map
   entries. Called by clojure.core/reduce-kv."
  (kv-reduce [amap f init]))
