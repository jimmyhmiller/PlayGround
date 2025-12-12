(ns clojure.core)

;; =============================================================================
;; Core Protocols from ClojureScript
;; Source: https://github.com/clojure/clojurescript/blob/master/src/main/cljs/cljs/core.cljs
;; License: Eclipse Public License 1.0
;; =============================================================================

;; -----------------------------------------------------------------------------
;; Marker Protocols
;; -----------------------------------------------------------------------------

(defprotocol Fn
  "Marker protocol")

(defprotocol ASeq
  "Marker protocol indicating an array sequence.")

(defprotocol ISequential
  "Marker interface indicating a persistent collection of sequential items")

(defprotocol IList
  "Marker interface indicating a persistent list")

(defprotocol IRecord
  "Marker interface indicating a record object")

(defprotocol IAtom
  "Marker protocol indicating an atom.")

;; -----------------------------------------------------------------------------
;; Function Protocol
;; -----------------------------------------------------------------------------

(defprotocol IFn
  "Protocol for adding the ability to invoke an object as a function.
  For example, a vector can also be used to look up a value:
  ([1 2 3 4] 1) => 2"
  (-invoke
    [this]
    [this a]
    [this a b]
    [this a b c]
    [this a b c d]
    [this a b c d e]
    [this a b c d e f]
    [this a b c d e f g]
    [this a b c d e f g h]
    [this a b c d e f g h i]
    [this a b c d e f g h i j]
    [this a b c d e f g h i j k]
    [this a b c d e f g h i j k l]
    [this a b c d e f g h i j k l m]
    [this a b c d e f g h i j k l m n]
    [this a b c d e f g h i j k l m n o]
    [this a b c d e f g h i j k l m n o p]
    [this a b c d e f g h i j k l m n o p q]
    [this a b c d e f g h i j k l m n o p q r]
    [this a b c d e f g h i j k l m n o p q r s]
    [this a b c d e f g h i j k l m n o p q r s t]
    [this a b c d e f g h i j k l m n o p q r s t rest]))

;; -----------------------------------------------------------------------------
;; Cloning Protocol
;; -----------------------------------------------------------------------------

(defprotocol ICloneable
  "Protocol for cloning a value."
  (-clone [value]
    "Creates a clone of value."))

;; -----------------------------------------------------------------------------
;; Collection Protocols
;; -----------------------------------------------------------------------------

(defprotocol ICounted
  "Protocol for adding the ability to count a collection in constant time."
  (-count [coll]
    "Calculates the count of coll in constant time. Used by cljs.core/count."))

(defprotocol IEmptyableCollection
  "Protocol for creating an empty collection."
  (-empty [coll]
    "Returns an empty collection of the same category as coll. Used
     by cljs.core/empty."))

(defprotocol ICollection
  "Protocol for adding to a collection."
  (-conj [coll o]
    "Returns a new collection of coll with o added to it. The new item
     should be added to the most efficient place, e.g.
     (conj [1 2 3 4] 5) => [1 2 3 4 5]
     (conj '(2 3 4 5) 1) => '(1 2 3 4 5)"))

(defprotocol IIndexed
  "Protocol for collections to provide indexed-based access to their items."
  (-nth [coll n] [coll n not-found]
    "Returns the value at the index n in the collection coll.
     Returns not-found if index n is out of bounds and not-found is supplied."))

;; -----------------------------------------------------------------------------
;; Sequence Protocols
;; -----------------------------------------------------------------------------

(defprotocol ISeq
  "Protocol for collections to provide access to their items as sequences."
  (-first [coll]
    "Returns the first item in the collection coll. Used by cljs.core/first.")
  (-rest [coll]
    "Returns a new collection of coll without the first item. It should
     always return a seq, e.g.
     (rest []) => ()
     (rest nil) => ()"))

(defprotocol INext
  "Protocol for accessing the next items of a collection."
  (-next [coll]
    "Returns a new collection of coll without the first item. In contrast to
     rest, it should return nil if there are no more items, e.g.
     (next []) => nil
     (next nil) => nil"))

(defprotocol ISeqable
  "Protocol for adding the ability to a type to be transformed into a sequence."
  (-seq [o]
    "Returns a seq of o, or nil if o is empty."))

(defprotocol IReversible
  "Protocol for reversing a seq."
  (-rseq [coll]
    "Returns a seq of the items in coll in reversed order."))

;; -----------------------------------------------------------------------------
;; Lookup Protocols
;; -----------------------------------------------------------------------------

(defprotocol ILookup
  "Protocol for looking up a value in a data structure."
  (-lookup [o k] [o k not-found]
    "Use k to look up a value in o. If not-found is supplied and k is not
     a valid value that can be used for look up, not-found is returned."))

(defprotocol IAssociative
  "Protocol for adding associativity to collections."
  (-contains-key? [coll k]
    "Returns true if k is a key in coll.")
  (-assoc [coll k v]
    "Returns a new collection of coll with a mapping from key k to
     value v added to it."))

(defprotocol IFind
  "Protocol for implementing entry finding in collections."
  (-find [coll k]
    "Returns the map entry for key, or nil if key not present."))

;; -----------------------------------------------------------------------------
;; Map Protocols
;; -----------------------------------------------------------------------------

(defprotocol IMap
  "Protocol for adding mapping functionality to collections."
  (-dissoc [coll k]
    "Returns a new collection of coll without the mapping for key k."))

(defprotocol IMapEntry
  "Protocol for examining a map entry."
  (-key [coll]
    "Returns the key of the map entry.")
  (-val [coll]
    "Returns the value of the map entry."))

;; -----------------------------------------------------------------------------
;; Set Protocol
;; -----------------------------------------------------------------------------

(defprotocol ISet
  "Protocol for adding set functionality to a collection."
  (-disjoin [coll v]
    "Returns a new collection of coll that does not contain v."))

;; -----------------------------------------------------------------------------
;; Stack Protocol
;; -----------------------------------------------------------------------------

(defprotocol IStack
  "Protocol for collections to provide access to their items as stacks. The top
  of the stack should be accessed in the most efficient way for the different
  data structures."
  (-peek [coll]
    "Returns the item from the top of the stack. Is used by cljs.core/peek.")
  (-pop [coll]
    "Returns a new stack without the item on top of the stack. Is used
     by cljs.core/pop."))

;; -----------------------------------------------------------------------------
;; Vector Protocol
;; -----------------------------------------------------------------------------

(defprotocol IVector
  "Protocol for adding vector functionality to collections."
  (-assoc-n [coll n val]
    "Returns a new vector with value val added at position n."))

;; -----------------------------------------------------------------------------
;; Deref Protocols
;; -----------------------------------------------------------------------------

(defprotocol IDeref
  "Protocol for adding dereference functionality to a reference."
  (-deref [o]
    "Returns the value of the reference o."))

(defprotocol IDerefWithTimeout
  (-deref-with-timeout [o msec timeout-val]))

;; -----------------------------------------------------------------------------
;; Metadata Protocols
;; -----------------------------------------------------------------------------

(defprotocol IMeta
  "Protocol for accessing the metadata of an object."
  (-meta [o]
    "Returns the metadata of object o."))

(defprotocol IWithMeta
  "Protocol for adding metadata to an object."
  (-with-meta [o meta]
    "Returns a new object with value of o and metadata meta added to it."))

;; -----------------------------------------------------------------------------
;; Reduce Protocols
;; -----------------------------------------------------------------------------

(defprotocol IReduce
  "Protocol for seq types that can reduce themselves.
  Called by cljs.core/reduce."
  (-reduce [coll f] [coll f start]
    "f should be a function of 2 arguments. If start is not supplied,
     returns the result of applying f to the first 2 items in coll, then
     applying f to that result and the 3rd item, etc."))

(defprotocol IKVReduce
  "Protocol for associative types that can reduce themselves
  via a function of key and val. Called by cljs.core/reduce-kv."
  (-kv-reduce [coll f init]
    "Reduces an associative collection and returns the result. f should be
     a function that takes three arguments."))

;; -----------------------------------------------------------------------------
;; Equality and Hashing Protocols
;; -----------------------------------------------------------------------------

(defprotocol IEquiv
  "Protocol for adding value comparison functionality to a type."
  (-equiv [o other]
    "Returns true if o and other are equal, false otherwise."))

(defprotocol IHash
  "Protocol for adding hashing functionality to a type."
  (-hash [o]
    "Returns the hash code of o."))

;; -----------------------------------------------------------------------------
;; Sorted Collection Protocol
;; -----------------------------------------------------------------------------

(defprotocol ISorted
  "Protocol for a collection which can represent their items
  in a sorted manner."
  (-sorted-seq [coll ascending?]
    "Returns a sorted seq from coll in either ascending or descending order.")
  (-sorted-seq-from [coll k ascending?]
    "Returns a sorted seq from coll in either ascending or descending order.
     If ascending is true, the result should contain all items which are > or >=
     than k. If ascending is false, the result should contain all items which
     are < or <= than k, e.g.
     (-sorted-seq-from (sorted-set 1 2 3 4 5) 3 true) => (3 4 5)
     (-sorted-seq-from (sorted-set 1 2 3 4 5) 3 false) => (3 2 1)")
  (-entry-key [coll entry]
    "Returns the key for entry.")
  (-comparator [coll]
    "Returns the comparator for coll."))

;; -----------------------------------------------------------------------------
;; Writer Protocols
;; -----------------------------------------------------------------------------

(defprotocol IWriter
  "Protocol for writing. Currently only implemented by StringBufferWriter."
  (-write [writer s]
    "Writes s with writer and returns the result.")
  (-flush [writer]
    "Flush writer."))

(defprotocol IPrintWithWriter
  "The old IPrintable protocol's implementation consisted of building a giant
   list of strings to concatenate.  This involved lots of concat calls,
   intermediate vectors, and lazy-seqs, and was very slow in some older JS
   engines.  IPrintWithWriter implements printing via the IWriter protocol, so it
   be implemented efficiently in terms of e.g. a StringBuffer append."
  (-pr-writer [o writer opts]))

;; -----------------------------------------------------------------------------
;; Pending Protocol
;; -----------------------------------------------------------------------------

(defprotocol IPending
  "Protocol for types which can have a deferred realization. Currently only
  implemented by Delay and LazySeq."
  (-realized? [x]
    "Returns true if a value for x has been produced, false otherwise."))

;; -----------------------------------------------------------------------------
;; Watchable Protocol
;; -----------------------------------------------------------------------------

(defprotocol IWatchable
  "Protocol for types that can be watched. Currently only implemented by Atom."
  (-notify-watches [this oldval newval]
    "Calls all watchers with this, oldval and newval.")
  (-add-watch [this key f]
    "Adds a watcher function f to this. Keys must be unique per reference,
     and can be used to remove the watch with -remove-watch.")
  (-remove-watch [this key]
    "Removes watcher that corresponds to key from this."))

;; -----------------------------------------------------------------------------
;; Transient Collection Protocols
;; -----------------------------------------------------------------------------

(defprotocol IEditableCollection
  "Protocol for collections which can transformed to transients."
  (-as-transient [coll]
    "Returns a new, transient version of the collection, in constant time."))

(defprotocol ITransientCollection
  "Protocol for adding basic functionality to transient collections."
  (-conj! [tcoll val]
    "Adds value val to tcoll and returns tcoll.")
  (-persistent! [tcoll]
    "Creates a persistent data structure from tcoll and returns it."))

(defprotocol ITransientAssociative
  "Protocol for adding associativity to transient collections."
  (-assoc! [tcoll key val]
    "Returns a new transient collection of tcoll with a mapping from key to
     val added to it."))

(defprotocol ITransientMap
  "Protocol for adding mapping functionality to transient collections."
  (-dissoc! [tcoll key]
    "Returns a new transient collection of tcoll without the mapping for key."))

(defprotocol ITransientVector
  "Protocol for adding vector functionality to transient collections."
  (-assoc-n! [tcoll n val]
    "Returns tcoll with value val added at position n.")
  (-pop! [tcoll]
    "Returns tcoll with the last item removed from it."))

(defprotocol ITransientSet
  "Protocol for adding set functionality to a transient collection."
  (-disjoin! [tcoll v]
    "Returns tcoll without v."))

;; -----------------------------------------------------------------------------
;; Comparison Protocol
;; -----------------------------------------------------------------------------

(defprotocol IComparable
  "Protocol for values that can be compared."
  (-compare [x y]
    "Returns a negative number, zero, or a positive number when x is logically
     'less than', 'equal to', or 'greater than' y."))

;; -----------------------------------------------------------------------------
;; Chunk Protocols
;; -----------------------------------------------------------------------------

(defprotocol IChunk
  "Protocol for accessing the items of a chunk."
  (-drop-first [coll]
    "Return a new chunk of coll with the first item removed."))

(defprotocol IChunkedSeq
  "Protocol for accessing a collection as sequential chunks."
  (-chunked-first [coll]
    "Returns the first chunk in coll.")
  (-chunked-rest [coll]
    "Return a new collection of coll with the first chunk removed."))

(defprotocol IChunkedNext
  "Protocol for accessing the chunks of a collection."
  (-chunked-next [coll]
    "Returns a new collection of coll without the first chunk."))

;; -----------------------------------------------------------------------------
;; Named Protocol
;; -----------------------------------------------------------------------------

(defprotocol INamed
  "Protocol for adding a name."
  (-name [x]
    "Returns the name String of x.")
  (-namespace [x]
    "Returns the namespace String of x."))

;; -----------------------------------------------------------------------------
;; Reset/Swap Protocols
;; -----------------------------------------------------------------------------

(defprotocol IReset
  "Protocol for adding resetting functionality."
  (-reset! [o new-value]
    "Sets the value of o to new-value."))

(defprotocol ISwap
  "Protocol for adding swapping functionality."
  (-swap! [o f] [o f a] [o f a b] [o f a b xs]
    "Swaps the value of o to be (apply f current-value-of-atom args)."))

;; -----------------------------------------------------------------------------
;; Volatile Protocol
;; -----------------------------------------------------------------------------

(defprotocol IVolatile
  "Protocol for adding volatile functionality."
  (-vreset! [o new-value]
    "Sets the value of volatile o to new-value without regard for the
     current value. Returns new-value."))

;; -----------------------------------------------------------------------------
;; Iterable Protocol
;; -----------------------------------------------------------------------------

(defprotocol IIterable
  "Protocol for iterating over a collection."
  (-iterator [coll]
    "Returns an iterator for coll."))

;; -----------------------------------------------------------------------------
;; Drop Protocol
;; -----------------------------------------------------------------------------

(defprotocol IDrop
  "Protocol for persistent or algorithmically defined collections to provide a
  means of dropping N items that is more efficient than sequential walking."
  (-drop [coll n]
    "Returns a collection that is ISequential, ISeq, and IReduce, or nil if past
     the end. The number of items to drop n must be > 0. It is also useful if the
     returned coll implements IDrop for subsequent use in a partition-like scenario."))

;; =============================================================================
;; Helper Predicates
;; =============================================================================

;; NOTE: satisfies? and instance? are not yet implemented in the compiler
;; For now, these always return false - we'll implement proper type checking later
(def sequential?
  (fn [x]
    false))

(def list?
  (fn [x]
    false))

;; =============================================================================
;; Reduced Type (for early termination in reduce)
;; =============================================================================

(deftype* Reduced [val])

(extend-type Reduced
  IDeref
  (-deref [this] (.-val this)))

(def reduced
  (fn [x]
    (Reduced. x)))

;; NOTE: instance? not available yet, so reduced? always returns false for now
;; This means early termination in reduce won't work
(def reduced?
  (fn [x]
    false))

;; =============================================================================
;; Polymorphic Sequence Functions
;; =============================================================================

(def seq
  (fn [coll]
    (if (nil? coll)
      nil
      (-seq coll))))

(def first
  (fn [coll]
    (if (nil? coll)
      nil
      (if (nil? (seq coll))
        nil
        (-first (seq coll))))))

(def next
  (fn [coll]
    (if (nil? coll)
      nil
      (if (nil? (seq coll))
        nil
        (-next (seq coll))))))

;; =============================================================================
;; Hashing Infrastructure
;; =============================================================================

;; Integer multiplication (32-bit, for Murmur3 hashing)
(def imul
  (fn [a b]
    (let [ah (bit-and (unsigned-bit-shift-right a 16) 0xffff)
          al (bit-and a 0xffff)
          bh (bit-and (unsigned-bit-shift-right b 16) 0xffff)
          bl (bit-and b 0xffff)]
      (bit-or
        (+ (* al bl)
           (unsigned-bit-shift-right
             (bit-shift-left (+ (* ah bl) (* al bh)) 16) 0)) 0))))

;; Murmur3 hashing constants
(def m3-seed 0)
(def m3-C1 0xcc9e2d51)
(def m3-C2 0x1b873593)

(def int-rotate-left
  (fn [x n]
    (bit-or
      (bit-shift-left x n)
      (unsigned-bit-shift-right x (- 32 n)))))

(def m3-mix-K1
  (fn [k1]
    (imul (int-rotate-left (imul k1 m3-C1) 15) m3-C2)))

(def m3-mix-H1
  (fn [h1 k1]
    (+ (imul (int-rotate-left (bit-xor h1 k1) 13) 5) 0xe6546b64)))

(def m3-fmix
  (fn [h1 len]
    (let [h1 (bit-xor h1 len)
          h1 (bit-xor h1 (unsigned-bit-shift-right h1 16))
          h1 (imul h1 0x85ebca6b)
          h1 (bit-xor h1 (unsigned-bit-shift-right h1 13))
          h1 (imul h1 0xc2b2ae35)
          h1 (bit-xor h1 (unsigned-bit-shift-right h1 16))]
      h1)))

(def empty-ordered-hash (m3-fmix m3-seed 0))

;; Polymorphic hash function - dispatches to -hash protocol method
(def hash
  (fn [x]
    (if (nil? x)
      0
      (if (number? x)
        x
        (-hash x)))))

(def hash-ordered-coll
  (fn [coll]
    (loop [n 0 hash-code 1 s (seq coll)]
      (if (nil? s)
        (m3-fmix (m3-mix-H1 m3-seed (m3-mix-K1 hash-code)) n)
        (recur (+ n 1)
               (bit-or (+ (imul 31 hash-code) (hash (first s))) 0)
               (next s))))))

;; =============================================================================
;; Sequential Equality
;; =============================================================================

(def equiv-sequential
  (fn [x y]
    (if (sequential? y)
      (loop [xs (seq x) ys (seq y)]
        (if (nil? xs)
          (nil? ys)
          (if (nil? ys)
            false
            (if (= (first xs) (first ys))
              (recur (next xs) (next ys))
              false))))
      false)))

;; =============================================================================
;; Seq Reduce
;; =============================================================================

(def seq-reduce
  (fn
    ([f coll]
     (let [s (seq coll)]
       (if (nil? s)
         (f)
         (let [fst (first s)
               rst (next s)]
           (if (nil? rst)
             fst
             (loop [acc fst s rst]
               (if (nil? s)
                 acc
                 (let [acc (f acc (first s))]
                   (if (reduced? acc)
                     (-deref acc)
                     (recur acc (next s)))))))))))
    ([f init coll]
     (loop [acc init s (seq coll)]
       (if (nil? s)
         acc
         (let [acc (f acc (first s))]
           (if (reduced? acc)
             (-deref acc)
             (recur acc (next s)))))))))

;; =============================================================================
;; List Types - Type Definitions (both types must be defined before extending)
;; =============================================================================

(deftype* EmptyList [meta])
(deftype* PList [meta first rest count ^:mutable __hash])

;; =============================================================================
;; EmptyList Protocol Implementations
;; =============================================================================

(extend-type EmptyList
  ICloneable
  (-clone [this] (EmptyList. (.-meta this)))

  IWithMeta
  (-with-meta [this new-meta]
    (if (identical? new-meta (.-meta this))
      this
      (EmptyList. new-meta)))

  IMeta
  (-meta [this] (.-meta this))

  ISeq
  (-first [this] nil)
  (-rest [this] this)

  INext
  (-next [this] nil)

  IStack
  (-peek [this] nil)
  (-pop [this] nil)

  ICollection
  (-conj [this o] (PList. (.-meta this) o nil 1 nil))

  IEmptyableCollection
  (-empty [this] this)

  IEquiv
  (-equiv [this other]
    (nil? (seq other)))

  IHash
  (-hash [this] empty-ordered-hash)

  ISeqable
  (-seq [this] nil)

  ICounted
  (-count [this] 0)

  IReduce
  (-reduce [this f] (seq-reduce f this))
  (-reduce [this f start] (seq-reduce f start this)))

(def EMPTY-LIST (EmptyList. nil))

;; =============================================================================
;; PList Protocol Implementations
;; =============================================================================

(extend-type PList
  ICloneable
  (-clone [this] (PList. (.-meta this) (.-first this) (.-rest this) (.-count this) (.-__hash this)))

  IWithMeta
  (-with-meta [this new-meta]
    (if (identical? new-meta (.-meta this))
      this
      (PList. new-meta (.-first this) (.-rest this) (.-count this) (.-__hash this))))

  IMeta
  (-meta [this] (.-meta this))

  ISeq
  (-first [this] (.-first this))
  (-rest [this]
    (if (= (.-count this) 1)
      EMPTY-LIST
      (.-rest this)))

  INext
  (-next [this]
    (if (= (.-count this) 1)
      nil
      (.-rest this)))

  IStack
  (-peek [this] (.-first this))
  (-pop [this] (-rest this))

  ICollection
  (-conj [this o] (PList. (.-meta this) o this (+ (.-count this) 1) nil))

  IEmptyableCollection
  (-empty [this] (-with-meta EMPTY-LIST (.-meta this)))

  IEquiv
  (-equiv [this other] (equiv-sequential this other))

  IHash
  (-hash [this]
    (if (nil? (.-__hash this))
      (let [h (hash-ordered-coll this)]
        (set! (.-__hash this) h)
        h)
      (.-__hash this)))

  ISeqable
  (-seq [this] this)

  ICounted
  (-count [this] (.-count this))

  IReduce
  (-reduce [this f] (seq-reduce f this))
  (-reduce [this f start] (seq-reduce f start this)))

;; =============================================================================
;; Basic Arithmetic Helpers
;; =============================================================================

(defn inc
  "Returns a number one greater than num."
  [x]
  (+ x 1))

(defn dec
  "Returns a number one less than num."
  [x]
  (- x 1))

(defn zero?
  "Returns true if num is zero."
  [x]
  (= x 0))

(defn pos?
  "Returns true if num is greater than zero."
  [x]
  (> x 0))

(defn neg?
  "Returns true if num is less than zero."
  [x]
  (< x 0))

(defn ==
  "Returns true if nums have equivalent numeric value."
  ([x] true)
  ([x y] (= x y)))

(defn not
  "Returns true if x is logical false, false otherwise."
  [x]
  (if x false true))

;; DEVIATION: Our Error type instead of js/Error
(deftype Error [message])

;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - PersistentVector
;; Source: https://github.com/clojure/clojurescript/blob/master/src/main/cljs/cljs/core.cljs
;; =============================================================================

;; DEVIATION: Forward declarations needed because our compiler doesn't support lazy lookup
(declare EMPTY-NODE EMPTY-VECTOR PersistentVector)

;; DEVIATION: instance? stub - always returns false for now
;; TODO: Implement proper type checking
(defn instance? [type obj]
  false)

;; DEVIATION: count function - calls the ICounted protocol
(defn count
  "Returns the number of items in the collection."
  [coll]
  (if (nil? coll)
    0
    (-count coll)))

;; DEVIATION: caching-hash stub - returns 0 for now
;; TODO: Implement proper hash caching
(defn caching-hash [coll hash-fn hash-key]
  0)

;; DEVIATION: integer? - uses number? since we only have integers and floats
(defn integer? [x]
  (number? x))

;; DEVIATION: vector? stub - always returns false for now
;; TODO: Implement proper type checking
(defn vector? [x]
  false)

;; DEVIATION: with-meta - calls the IWithMeta protocol
(defn with-meta [o meta]
  (-with-meta o meta))

;; DEVIATION: reduce - calls the IReduce protocol
(defn reduce
  ([f coll]
   (-reduce coll f))
  ([f init coll]
   (-reduce coll f init)))

;; DEVIATION: Simple MapEntry type for IFind protocol
(deftype MapEntry [key val __hash]
  IMapEntry
  (-key [this] key)
  (-val [this] val))

;; DEVIATION: array function - creates arrays with given elements
(defn array
  "Creates a new array with the given elements."
  ([] (make-array 0))
  ([a] (let [arr (make-array 1)]
         (aset arr 0 a)
         arr))
  ([a b] (let [arr (make-array 2)]
           (aset arr 0 a)
           (aset arr 1 b)
           arr))
  ([a b c] (let [arr (make-array 3)]
             (aset arr 0 a)
             (aset arr 1 b)
             (aset arr 2 c)
             arr)))

;; DEVIATION: array-butlast - replacement for (.slice arr 0 -1)
;; Creates a new array with all elements except the last
(defn array-butlast [arr]
  (let [len (alength arr)
        new-len (dec len)
        result (make-array new-len)]
    (loop [i 0]
      (if (< i new-len)
        (do
          (aset result i (aget arr i))
          (recur (inc i)))
        result))))

;; DEVIATION: pv-reduce - simple implementation for now
(defn pv-reduce
  ([v f start cnt]
   (if (> cnt 0)
     (loop [i start
            result (-nth v start)]
       (let [next-i (inc i)]
         (if (< next-i cnt)
           (recur next-i (f result (-nth v next-i)))
           result)))
     nil))
  ([v f init start cnt]
   (loop [i start
          result init]
     (if (< i cnt)
       (recur (inc i) (f result (-nth v i)))
       result))))

(deftype VectorNode [edit arr])

;; DEVIATION: defn- not supported, using defn
(defn pv-fresh-node [edit]
  (VectorNode. edit (make-array 32)))

;; DEVIATION: defn- not supported, using defn
(defn pv-aget [node idx]
  (aget (.-arr node) idx))

;; DEVIATION: defn- not supported, using defn
(defn pv-aset [node idx val]
  (aset (.-arr node) idx val))

;; DEVIATION: defn- not supported, using defn
(defn pv-clone-node [node]
  (VectorNode. (.-edit node) (aclone (.-arr node))))

;; DEVIATION: defn- not supported, using defn
(defn tail-off [pv]
  (let [cnt (.-cnt pv)]
    (if (< cnt 32)
      0
      (bit-shift-left (bit-shift-right-zero-fill (dec cnt) 5) 5))))

;; DEVIATION: defn- not supported, using defn
(defn new-path [edit level node]
  (loop [ll level
         ret node]
    (if (zero? ll)
      ret
      (let [embed ret
            r (pv-fresh-node edit)
            _ (pv-aset r 0 embed)]
        (recur (- ll 5) r)))))

;; DEVIATION: defn- not supported, using defn
(defn push-tail [pv level parent tailnode]
  (let [ret (pv-clone-node parent)
        subidx (bit-and (bit-shift-right-zero-fill (dec (.-cnt pv)) level) 0x01f)]
    (if (== 5 level)
      (do
        (pv-aset ret subidx tailnode)
        ret)
      (let [child (pv-aget parent subidx)]
        (if-not (nil? child)
          (let [node-to-insert (push-tail pv (- level 5) child tailnode)]
            (pv-aset ret subidx node-to-insert)
            ret)
          (let [node-to-insert (new-path nil (- level 5) tailnode)]
            (pv-aset ret subidx node-to-insert)
            ret))))))

;; DEVIATION: defn- not supported, using defn
;; DEVIATION: simplified error message (no str function yet)
(defn vector-index-out-of-bounds [i cnt]
  (throw (Error. "Vector index out of bounds")))

;; DEVIATION: defn- not supported, using defn
(defn first-array-for-longvec [pv]
  ;; invariants: (count pv) > 32.
  (loop [node (.-root pv)
         level (.-shift pv)]
    (if (pos? level)
      (recur (pv-aget node 0) (- level 5))
      (.-arr node))))

;; DEVIATION: defn- not supported, using defn
(defn unchecked-array-for [pv i]
  ;; invariant: i is a valid index of pv (use array-for if unknown).
  (if (>= i (tail-off pv))
      (.-tail pv)
      (loop [node (.-root pv)
             level (.-shift pv)]
        (if (pos? level)
          (recur (pv-aget node (bit-and (bit-shift-right-zero-fill i level) 0x01f))
                 (- level 5))
          (.-arr node)))))

;; DEVIATION: defn- not supported, using defn
(defn array-for [pv i]
  (if (and (<= 0 i) (< i (.-cnt pv)))
    (unchecked-array-for pv i)
    (vector-index-out-of-bounds i (.-cnt pv))))

;; DEVIATION: defn- not supported, using defn
(defn do-assoc [pv level node i val]
  (let [ret (pv-clone-node node)]
    (if (zero? level)
      (do
        (pv-aset ret (bit-and i 0x01f) val)
        ret)
      (let [subidx (bit-and (bit-shift-right-zero-fill i level) 0x01f)]
        (pv-aset ret subidx (do-assoc pv (- level 5) (pv-aget node subidx) i val))
        ret))))

;; DEVIATION: defn- not supported, using defn
(defn pop-tail [pv level node]
  (let [subidx (bit-and (bit-shift-right-zero-fill (- (.-cnt pv) 2) level) 0x01f)]
    (cond
     (> level 5) (let [new-child (pop-tail pv (- level 5) (pv-aget node subidx))]
                   (if (and (nil? new-child) (zero? subidx))
                     nil
                     (let [ret (pv-clone-node node)]
                       (pv-aset ret subidx new-child)
                       ret)))
     (zero? subidx) nil
     :else (let [ret (pv-clone-node node)]
             (pv-aset ret subidx nil)
             ret))))

(declare chunked-seq)

(deftype PersistentVector [meta cnt shift root tail ^:mutable __hash]
  ICloneable
  (-clone [_] (PersistentVector. meta cnt shift root tail __hash))

  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta)
      coll
      (PersistentVector. new-meta cnt shift root tail __hash)))

  IMeta
  (-meta [coll] meta)

  IStack
  (-peek [coll]
    (when (> cnt 0)
      (-nth coll (dec cnt))))
  (-pop [coll]
    (cond
     (zero? cnt) (throw (Error. "Can't pop empty vector"))
     (== 1 cnt) (-with-meta EMPTY-VECTOR meta)
     (< 1 (- cnt (tail-off coll)))
      ;; DEVIATION: array-butlast instead of (.slice tail 0 -1)
      (PersistentVector. meta (dec cnt) shift root (array-butlast tail) nil)
      :else (let [new-tail (unchecked-array-for coll (- cnt 2))
                  nr (pop-tail coll shift root)
                  new-root (if (nil? nr) EMPTY-NODE nr)
                  cnt-1 (dec cnt)]
              (if (and (< 5 shift) (nil? (pv-aget new-root 1)))
                (PersistentVector. meta cnt-1 (- shift 5) (pv-aget new-root 0) new-tail nil)
                (PersistentVector. meta cnt-1 shift new-root new-tail nil)))))

  ICollection
  (-conj [coll o]
    (if (< (- cnt (tail-off coll)) 32)
      (let [len (alength tail)
            new-tail (make-array (inc len))]
        (dotimes [i len]
          (aset new-tail i (aget tail i)))
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
  (-empty [coll] (-with-meta EMPTY-VECTOR meta))

  ISequential
  IEquiv
  ;; DEVIATION: Simplified equiv - no iterator support yet
  (-equiv [coll other]
    (if (nil? other)
      false
      (if (== cnt (count other))
        ;; Simple element-by-element comparison using nth
        (loop [i 0]
          (if (< i cnt)
            (if (= (-nth coll i) (-nth other i))
              (recur (inc i))
              false)
            true))
        false)))

  IHash
  (-hash [coll] (caching-hash coll hash-ordered-coll __hash))

  ISeqable
  ;; DEVIATION: Simplified seq - no IndexedSeq/chunked-seq yet
  (-seq [coll]
    (if (zero? cnt)
      nil
      ;; TODO: Return proper seq. For now use a simple loop approach
      coll))

  ICounted
  (-count [coll] cnt)

  IIndexed
  (-nth [coll n]
    (aget (array-for coll n) (bit-and n 0x01f)))
  (-nth [coll n not-found]
    (if (and (<= 0 n) (< n cnt))
      (aget (unchecked-array-for coll n) (bit-and n 0x01f))
      not-found))

  ILookup
  (-lookup [coll k] (-lookup coll k nil))
  (-lookup [coll k not-found] (if (number? k)
                                (-nth coll k not-found)
                                not-found))

  IAssociative
  (-assoc [coll k v]
    (if (number? k)
      (-assoc-n coll k v)
      (throw (Error. "Vector's key for assoc must be a number."))))
  (-contains-key? [coll k]
    (if (integer? k)
      (and (<= 0 k) (< k cnt))
      false))

  IFind
  (-find [coll n]
    (when (and (<= 0 n) (< n cnt))
      (MapEntry. n (aget (unchecked-array-for coll n) (bit-and n 0x01f)) nil)))

  IVector
  (-assoc-n [coll n val]
    (cond
      (and (<= 0 n) (< n cnt))
      (if (<= (tail-off coll) n)
         (let [new-tail (aclone tail)]
           (aset new-tail (bit-and n 0x01f) val)
           (PersistentVector. meta cnt shift root new-tail nil))
         (PersistentVector. meta cnt shift (do-assoc coll shift root n val) tail nil))
      (== n cnt) (-conj coll val)
      ;; DEVIATION: simplified error message (no str function yet)
      :else (throw (Error. "Index out of bounds"))))

  IReduce
  (-reduce [v f]
    (pv-reduce v f 0 cnt))
  (-reduce [v f init]
    (loop [i 0 init init]
      (if (< i cnt)
        (let [arr  (unchecked-array-for v i)
              len  (alength arr)
              init (loop [j 0 init init]
                     (if (< j len)
                       (let [init (f init (aget arr j))]
                         (if (reduced? init)
                           init
                           (recur (inc j) init)))
                       init))]
          ;; DEVIATION: no @deref support for reduced, just return init
          (if (reduced? init)
            init
            (recur (+ i len) init)))
        init)))

  IKVReduce
  (-kv-reduce [v f init]
    (loop [i 0 init init]
      (if (< i cnt)
        (let [arr  (unchecked-array-for v i)
              len  (alength arr)
              init (loop [j 0 init init]
                     (if (< j len)
                       (let [init (f init (+ j i) (aget arr j))]
                         (if (reduced? init)
                           init
                           (recur (inc j) init)))
                       init))]
          ;; DEVIATION: no @deref support for reduced
          (if (reduced? init)
            init
            (recur (+ i len) init)))
        init)))

  IFn
  (-invoke [coll k]
    (if (number? k)
      (-nth coll k)
      (throw (Error. "Key must be integer"))))

  IReversible
  ;; DEVIATION: No RSeq type yet, return nil
  (-rseq [coll] nil)

  IIterable
  ;; DEVIATION: No iterator support yet, return nil
  (-iterator [this] nil))

(def EMPTY-NODE (VectorNode. nil (make-array 32)))

(def EMPTY-VECTOR
  (PersistentVector. nil 0 5 EMPTY-NODE (array) empty-ordered-hash))

(defn vec
  "Creates a new vector containing the contents of coll."
  [coll]
  (cond
    (vector? coll)
    (with-meta coll nil)

    ;; DEVIATION: No transient support yet, use simple conj
    :else
    (reduce (fn [v x] (-conj v x)) EMPTY-VECTOR coll)))

(defn vector
  "Creates a new vector containing the args."
  [& args]
  (vec args))