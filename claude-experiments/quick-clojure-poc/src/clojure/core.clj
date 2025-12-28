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
;; Reduced Type (for early termination in reduce)
;; =============================================================================

(deftype* Reduced [val])

(extend-type Reduced
  IDeref
  (-deref [this] (.-val this)))

(def reduced
  (fn [x]
    (Reduced. x)))

(defn reduced?
  "Returns true if x is the result of a call to reduced"
  [x]
  (instance? Reduced x))

;; =============================================================================
;; Reader Type Protocol Implementations
;; These allow reader data structures (used in macros) to work with core protocols
;; =============================================================================

(extend-type __ReaderList
  IList

  ISeq
  (-first [this] (__reader_list_first this))
  (-rest [this] (__reader_list_rest this))

  INext
  (-next [this]
    (let [r (__reader_list_rest this)]
      (if (== (__reader_list_count r) 0)
        nil
        r)))

  ICounted
  (-count [this] (__reader_list_count this))

  ISeqable
  (-seq [this]
    (if (== (__reader_list_count this) 0)
      nil
      this))

  IIndexed
  (-nth [this n] (__reader_list_nth this n))
  (-nth [this n not-found]
    (if (< n (__reader_list_count this))
      (__reader_list_nth this n)
      not-found)))

(extend-type __ReaderVector
  ISeq
  (-first [this] (__reader_vector_first this))
  (-rest [this] (__reader_vector_rest this))

  INext
  (-next [this]
    (let [r (__reader_vector_rest this)]
      (if (== (__reader_vector_count r) 0)
        nil
        r)))

  ICounted
  (-count [this] (__reader_vector_count this))

  ISeqable
  (-seq [this]
    (if (== (__reader_vector_count this) 0)
      nil
      this))

  IIndexed
  (-nth [this n] (__reader_vector_nth this n))
  (-nth [this n not-found]
    (if (< n (__reader_vector_count this))
      (__reader_vector_nth this n)
      not-found)))

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
      (let [s (seq coll)]
        (if (nil? s)
          nil
          (-first s))))))

(def next
  (fn [coll]
    (if (nil? coll)
      nil
      (let [s (seq coll)]
        (if (nil? s)
          nil
          (-next s))))))

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
;; Uses hash-primitive builtin for keywords and strings
(def hash
  (fn [x]
    (if (nil? x)
      0
      (if (number? x)
        x
        (if (keyword? x)
          (hash-primitive x)
          (-hash x))))))

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
  IList

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
  IList

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
;; Cons - Simple cons cell (doesn't track count like PList)
;; =============================================================================

(deftype* Cons [meta first rest ^:mutable __hash])

(extend-type Cons
  IList

  ICloneable
  (-clone [this] (Cons. (.-meta this) (.-first this) (.-rest this) (.-__hash this)))

  IWithMeta
  (-with-meta [this new-meta]
    (if (identical? new-meta (.-meta this))
      this
      (Cons. new-meta (.-first this) (.-rest this) (.-__hash this))))

  IMeta
  (-meta [this] (.-meta this))

  ISeq
  (-first [this] (.-first this))
  (-rest [this] (if (nil? (.-rest this)) EMPTY-LIST (.-rest this)))

  INext
  (-next [this]
    (if (nil? (.-rest this)) nil (-seq (.-rest this))))

  ICollection
  (-conj [this o] (Cons. nil o this nil))

  IEmptyableCollection
  (-empty [this] EMPTY-LIST)

  ISequential

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

  IReduce
  (-reduce [this f] (seq-reduce f this))
  (-reduce [this f start] (seq-reduce f start this)))

(defn cons
  "Returns a new seq where x is the first element and coll is the rest."
  [x coll]
  (if (nil? coll)
    (PList. nil x nil 1 nil)
    (Cons. nil x (seq coll) nil)))

(defn conj
  "Returns a new collection with x 'added' to coll."
  [coll x]
  (if (nil? coll)
    (PList. nil x nil 1 nil)
    (-conj coll x)))

(defn reverse
  "Returns a seq of the items in coll in reverse order. Not lazy."
  [coll]
  (reduce conj EMPTY-LIST coll))

;; Now that list types are defined, implement the predicates
(defn list?
  "Returns true if x implements IList"
  [x]
  (satisfies? IList x))

;; NOTE: sequential? is defined later after PersistentVector and IndexedSeq exist

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

;; =============================================================================
;; Error Type (needed before IndexedSeq)
;; =============================================================================

(deftype Error [message])

;; =============================================================================
;; IndexedSeq - Wrapper for variadic arguments (like ClojureScript)
;; =============================================================================

(deftype* IndexedSeq [arr i meta])

(extend-type IndexedSeq
  ISeqable
  (-seq [this]
    (if (< (.-i this) (alength (.-arr this)))
      this
      nil))

  ISeq
  (-first [this]
    (aget (.-arr this) (.-i this)))
  (-rest [this]
    (if (< (+ (.-i this) 1) (alength (.-arr this)))
      (IndexedSeq. (.-arr this) (+ (.-i this) 1) nil)
      EMPTY-LIST))

  INext
  (-next [this]
    (if (< (+ (.-i this) 1) (alength (.-arr this)))
      (IndexedSeq. (.-arr this) (+ (.-i this) 1) nil)
      nil))

  ICounted
  (-count [this]
    (- (alength (.-arr this)) (.-i this)))

  IIndexed
  (-nth [this n]
    (let [idx (+ (.-i this) n)]
      (if (and (<= 0 idx) (< idx (alength (.-arr this))))
        (aget (.-arr this) idx)
        (throw (Error. "Index out of bounds")))))
  (-nth [this n not-found]
    (let [idx (+ (.-i this) n)]
      (if (and (<= 0 idx) (< idx (alength (.-arr this))))
        (aget (.-arr this) idx)
        not-found)))

  ICollection
  (-conj [this o]
    ;; conj on IndexedSeq creates a list
    (cons o this))

  IEmptyableCollection
  (-empty [this] EMPTY-LIST)

  IEquiv
  (-equiv [this other]
    (equiv-sequential this other))

  IHash
  (-hash [this]
    (hash-ordered-coll this))

  IReduce
  (-reduce [this f]
    (let [arr (.-arr this)
          len (alength arr)
          i (.-i this)]
      (if (< i len)
        (loop [acc (aget arr i) idx (+ i 1)]
          (if (< idx len)
            (let [acc (f acc (aget arr idx))]
              (if (reduced? acc)
                (-deref acc)
                (recur acc (+ idx 1))))
            acc))
        (f))))
  (-reduce [this f start]
    (let [arr (.-arr this)
          len (alength arr)]
      (loop [acc start idx (.-i this)]
        (if (< idx len)
          (let [acc (f acc (aget arr idx))]
            (if (reduced? acc)
              (-deref acc)
              (recur acc (+ idx 1))))
          acc))))

  IMeta
  (-meta [this] (.-meta this))

  IWithMeta
  (-with-meta [this new-meta]
    (if (identical? new-meta (.-meta this))
      this
      (IndexedSeq. (.-arr this) (.-i this) new-meta))))

;; Helper to create IndexedSeq from array (used by trampoline)
(defn indexed-seq
  "Create an IndexedSeq from an array, starting at index i."
  ([arr] (indexed-seq arr 0))
  ([arr i]
   (if (< i (alength arr))
     (IndexedSeq. arr i nil)
     nil)))

(defn list
  "Creates a new list containing the items."
  [& items]
  ;; Build a proper list by iterating from end to beginning
  (if (nil? items)
    EMPTY-LIST
    (let [c (count items)]
      (loop [i (dec c) acc EMPTY-LIST]
        (if (< i 0)
          acc
          (recur (dec i) (cons (nth items i) acc)))))))

;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - PersistentVector
;; Source: https://github.com/clojure/clojurescript/blob/master/src/main/cljs/cljs/core.cljs
;; =============================================================================

;; DEVIATION: Forward declarations needed because our compiler doesn't support lazy lookup
(declare EMPTY-NODE EMPTY-VECTOR PersistentVector)

;; NOTE: instance? is now a compiler builtin - it's resolved at compile time
;; based on the type argument and emits efficient type checking code

;; seq function - calls -seq protocol method
(defn seq
  "Returns a seq on the collection. If the collection is empty, returns nil."
  [coll]
  (if (nil? coll)
    nil
    (-seq coll)))

;; concat2 - helper for concat
(defn concat2 [x y]
  (let [s (seq x)]
    (if s
      (cons (first s) (concat2 (rest s) y))
      (seq y))))

;; concat - concatenates sequences
;; Eager implementation, variadic
(defn concat
  "Returns a seq representing the concatenation of the elements in the supplied colls."
  ([] nil)
  ([x] (seq x))
  ([x y] (concat2 x y))
  ([x y z] (concat2 x (concat2 y z)))
  ([a b c d] (concat2 a (concat2 b (concat2 c d))))
  ([a b c d e] (concat2 a (concat2 b (concat2 c (concat2 d e))))))

;; DEVIATION: count function - calls the ICounted protocol
(defn count
  "Returns the number of items in the collection."
  [coll]
  (if (nil? coll)
    0
    (-count coll)))

;; DEVIATION: rest function - calls -rest protocol method
(defn rest
  "Returns a possibly empty seq of the items after the first."
  [coll]
  (if (nil? coll)
    nil
    (let [s (-seq coll)]
      (if (nil? s)
        nil
        (-rest s)))))

;; DEVIATION: second function
(defn second
  "Returns the second item in the collection."
  [coll]
  (first (next coll)))

(defn nth
  "Returns the value at the index. get returns nil if index out of
   bounds, nth throws an exception unless not-found is supplied."
  ([coll n] (-nth coll n))
  ([coll n not-found] (-nth coll n not-found)))

;; DEVIATION: caching-hash stub - returns 0 for now
;; TODO: Implement proper hash caching
(defn caching-hash [coll hash-fn hash-key]
  0)

;; DEVIATION: integer? - uses number? since we only have integers and floats
(defn integer? [x]
  (number? x))

;; NOTE: vector? is defined after PersistentVector to use instance? properly

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
  (-val [this] val)

  ICounted
  (-count [this] 2)

  IIndexed
  (-nth [this n]
    (if (== n 0)
      key
      (if (== n 1)
        val
        (throw (Error. "Index out of bounds")))))
  (-nth [this n not-found]
    (if (== n 0)
      key
      (if (== n 1)
        val
        not-found))))

;; Factory function for MapEntry (needed for closures that can't resolve deftype constructors)
(defn make-map-entry [k v]
  "Creates a MapEntry with the given key and value."
  (MapEntry. k v nil))

;; DEVIATION: array functions - individual functions for each arity
;; Since multi-arity defn is not fully supported, we use separate functions

(defn array0 []
  "Creates an empty array."
  (make-array 0))

(defn array1 [a]
  "Creates a 1-element array."
  (let [arr (make-array 1)]
    (aset arr 0 a)
    arr))

(defn array2 [a b]
  "Creates a 2-element array."
  (let [arr (make-array 2)]
    (aset arr 0 a)
    (aset arr 1 b)
    arr))

(defn array4 [a b c d]
  "Creates a 4-element array."
  (let [arr (make-array 4)]
    (aset arr 0 a)
    (aset arr 1 b)
    (aset arr 2 c)
    (aset arr 3 d)
    arr))

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
        (PersistentVector. meta (inc cnt) new-shift new-root (array1 o) nil))))

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
  (-seq [coll]
    (if (zero? cnt)
      nil
      coll))

  ISeq
  (-first [coll] (-nth coll 0))
  (-rest [coll]
    (if (<= cnt 1)
      EMPTY-LIST
      ;; Build a list from elements 1..n
      (loop [i (dec cnt) acc EMPTY-LIST]
        (if (< i 1)
          acc
          (recur (dec i) (cons (-nth coll i) acc))))))

  INext
  (-next [coll]
    (if (<= cnt 1)
      nil
      ;; Build a list from elements 1..n
      (loop [i (dec cnt) acc EMPTY-LIST]
        (if (< i 1)
          acc
          (recur (dec i) (cons (-nth coll i) acc))))))

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
  (PersistentVector. nil 0 5 EMPTY-NODE (array0) empty-ordered-hash))

;; Now that PersistentVector is defined, we can implement vector? properly
(defn vector?
  "Return true if x is a PersistentVector"
  [x]
  (instance? PersistentVector x))

;; Now that all sequential types are defined, implement sequential?
(defn sequential?
  "Returns true if coll is a sequential collection (list, vector, cons, or indexed-seq)"
  [coll]
  (or (instance? PList coll)
      (instance? EmptyList coll)
      (instance? Cons coll)
      (instance? PersistentVector coll)
      (instance? IndexedSeq coll)))

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

;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - PersistentArrayMap and PersistentHashMap
;; Source: https://github.com/clojure/clojurescript/blob/master/src/main/cljs/cljs/core.cljs
;; =============================================================================

;; -----------------------------------------------------------------------------
;; Additional Helper Functions for Maps
;; -----------------------------------------------------------------------------

;; DEVIATION: Forward declarations
(declare PersistentArrayMap PersistentHashMap TransientArrayMap TransientHashMap)
(declare BitmapIndexedNode ArrayNode HashCollisionNode)
(declare create-inode-seq create-array-node-seq create-node)

;; Sentinel value for lookup failures
;; DEVIATION: js-obj not available, using a unique object
(deftype* LookupSentinel [])
(def ^:private lookup-sentinel (LookupSentinel.))

;; DEVIATION: Removed ^number type hint (not supported)
(defn mix-collection-hash
  "Mix final collection hash for ordered or unordered collections.
   hash-basis is the combined collection hash, count is the number
   of elements included in the basis. Note this is the hash code
   consistent with =, different from .hashCode.
   See http://clojure.org/data_structures#hash for full algorithms."
  [hash-basis count]
  (let [h1 m3-seed
        k1 (m3-mix-K1 hash-basis)
        h1 (m3-mix-H1 h1 k1)]
    (m3-fmix h1 count)))

;; DEVIATION: Removed ^number type hint (not supported)
(defn hash-unordered-coll
  "Returns the hash code, consistent with =, for an external unordered
   collection implementing Iterable. For maps, the iterator should
   return map entries whose hash is computed as
     (hash-ordered-coll [k v]).
   See http://clojure.org/data_structures#hash for full algorithms."
  [coll]
  (loop [n 0 hash-code 0 coll (seq coll)]
    (if-not (nil? coll)
      (recur (inc n) (bit-or (+ hash-code (hash (first coll))) 0) (next coll))
      (mix-collection-hash hash-code n))))

(def ^:private empty-unordered-hash
  (mix-collection-hash 0 0))

(defn bit-count
  "Counts the number of bits set in n"
  [v]
  ;; SWAR algorithm for 64-bit popcount
  (let [v (- v (bit-and (bit-shift-right v 1) 0x5555555555555555))
        v (+ (bit-and v 0x3333333333333333) (bit-and (bit-shift-right v 2) 0x3333333333333333))]
    (bit-shift-right (* (bit-and (+ v (bit-shift-right v 4)) 0x0F0F0F0F0F0F0F0F) 0x0101010101010101) 56)))

(defn unreduced
  "If x is reduced?, returns (deref x), else returns x"
  [x]
  (if (reduced? x) (-deref x) x))

;; map? is defined after PersistentHashMap below

;; DEVIATION: record? stub
(defn record?
  "Return true if x satisfies IRecord"
  [x]
  false)

;; keyword? is now a builtin - no stub needed

;; DEVIATION: symbol? stub
(defn symbol?
  "Return true if x is a Symbol"
  [x]
  false)

;; DEVIATION: string? - check if number type is a string (we might not have strings)
(defn string?
  "Returns true if x is a string"
  [x]
  false)

;; DEVIATION: keyword-identical? stub
(defn keyword-identical?
  "Efficient test to determine that two keywords are identical."
  [x y]
  (identical? x y))

;; NeverEquiv type for map equality checking
(deftype* NeverEquiv [])

(extend-type NeverEquiv
  IEquiv
  (-equiv [o other] false))

(def ^:private never-equiv (NeverEquiv.))

(defn equiv-map
  "Test map equivalence. Returns true if x equals y, otherwise returns false."
  [x y]
  (if (and (map? y) (not (record? y)))
    (if (== (count x) (count y))
      (-kv-reduce x
        (fn [_ k v]
          (if (= (-lookup y k never-equiv) v)
            true
            (reduced false)))
        true)
      false)
    false))

;; -----------------------------------------------------------------------------
;; Array Copy Utilities
;; -----------------------------------------------------------------------------

(defn array-copy
  "Copy len elements from array from starting at i to array to starting at j"
  [from i to j len]
  (loop [i i j j len len]
    (if (zero? len)
      to
      (do (aset to j (aget from i))
          (recur (inc i) (inc j) (dec len))))))

(defn array-copy-downward
  "Copy len elements from array from starting at i to array to starting at j, going downward"
  [from i to j len]
  (loop [i (+ i (dec len)) j (+ j (dec len)) len len]
    (if (zero? len)
      to
      (do (aset to j (aget from i))
          (recur (dec i) (dec j) (dec len))))))

;; -----------------------------------------------------------------------------
;; Array Index Functions
;; -----------------------------------------------------------------------------

(defn array-index-of-nil? [arr]
  (let [len (alength arr)]
    (loop [i 0]
      (cond
        (<= len i) -1
        (nil? (aget arr i)) i
        :else (recur (+ i 2))))))

;; Keywords are interned in this implementation, so identical? works
(defn array-index-of-keyword? [arr k]
  (let [len (alength arr)]
    (loop [i 0]
      (cond
        (<= len i) -1
        (and (keyword? (aget arr i))
             (identical? k (aget arr i))) i
        :else (recur (+ i 2))))))

;; Symbols are interned in this implementation, so identical? works
(defn array-index-of-symbol? [arr k]
  (let [len (alength arr)]
    (loop [i 0]
      (cond
        (<= len i) -1
        (and (symbol? (aget arr i))
             (identical? k (aget arr i))) i
        :else (recur (+ i 2))))))

(defn array-index-of-identical? [arr k]
  (let [len (alength arr)]
    (loop [i 0]
      (cond
        (<= len i) -1
        (identical? k (aget arr i)) i
        :else (recur (+ i 2))))))

(defn array-index-of-equiv? [arr k]
  (let [len (alength arr)]
    (loop [i 0]
      (cond
        (<= len i) -1
        (= k (aget arr i)) i
        :else (recur (+ i 2))))))

(defn array-index-of [arr k]
  (cond
    (keyword? k) (array-index-of-keyword? arr k)

    (or (string? k) (number? k))
    (array-index-of-identical? arr k)

    (symbol? k) (array-index-of-symbol? arr k)

    (nil? k)
    (array-index-of-nil? arr)

    :else (array-index-of-equiv? arr k)))

(defn array-map-index-of [m k]
  (array-index-of (.-arr m) k))

(defn array-extend-kv [arr k v]
  (let [l (alength arr)
        narr (make-array (+ l 2))]
    (loop [i 0]
      (when (< i l)
        (aset narr i (aget arr i))
        (recur (inc i))))
    (aset narr l k)
    (aset narr (inc l) v)
    narr))

(defn array-map-extend-kv [m k v]
  (array-extend-kv (.-arr m) k v))

;; key-test for comparing keys
(defn key-test [key other]
  (cond
    (identical? key other) true
    (keyword-identical? key other) true
    :else (= key other)))

;; -----------------------------------------------------------------------------
;; Box type for mutable flag passing
;; -----------------------------------------------------------------------------

(deftype* Box [^:mutable val])

;; -----------------------------------------------------------------------------
;; Hash Map Helper Functions
;; -----------------------------------------------------------------------------

(defn mask [hash shift]
  (bit-and (unsigned-bit-shift-right hash shift) 0x01f))

(defn clone-and-set
  ([arr i a]
   (let [clone (aclone arr)]
     (aset clone i a)
     clone))
  ([arr i a j b]
   (let [clone (aclone arr)]
     (aset clone i a)
     (aset clone j b)
     clone)))

(defn remove-pair [arr i]
  (let [new-arr (make-array (- (alength arr) 2))]
    (array-copy arr 0 new-arr 0 (* 2 i))
    (array-copy arr (* 2 (inc i)) new-arr (* 2 i) (- (alength new-arr) (* 2 i)))
    new-arr))

(defn bitmap-indexed-node-index [bitmap bit]
  (bit-count (bit-and bitmap (dec bit))))

(defn bitpos [hash shift]
  (bit-shift-left 1 (mask hash shift)))

(defn edit-and-set
  ([inode edit i a]
   (let [editable (-ensure-editable inode edit)]
     (aset (.-arr editable) i a)
     editable))
  ([inode edit i a j b]
   (let [editable (-ensure-editable inode edit)]
     (aset (.-arr editable) i a)
     (aset (.-arr editable) j b)
     editable)))

(defn inode-kv-reduce [arr f init]
  (let [len (alength arr)]
    (loop [i 0 init init]
      (if (< i len)
        (let [init (let [k (aget arr i)]
                     (if-not (nil? k)
                       (f init k (aget arr (inc i)))
                       (let [node (aget arr (inc i))]
                         (if-not (nil? node)
                           (-kv-reduce node f init)
                           init))))]
          (if (reduced? init)
            init
            (recur (+ i 2) init)))
        init))))

;; -----------------------------------------------------------------------------
;; BitmapIndexedNode
;; -----------------------------------------------------------------------------

;; DEVIATION: Removed Object from deftype (not supported)
(deftype BitmapIndexedNode [edit ^:mutable bitmap ^:mutable arr]
  IKVReduce
  (-kv-reduce [inode f init]
    (inode-kv-reduce arr f init)))

;; DEVIATION: Forward-define ArrayNode and HashCollisionNode types here
;; so they can be referenced by BitmapIndexedNode functions below
(deftype ArrayNode [edit ^:mutable cnt ^:mutable arr]
  IKVReduce
  (-kv-reduce [inode f init]
    (let [len (alength arr)]
      (loop [i 0 init init]
        (if (< i len)
          (let [node (aget arr i)]
            (if-not (nil? node)
              (let [init (-kv-reduce node f init)]
                (if (reduced? init)
                  init
                  (recur (inc i) init)))
              (recur (inc i) init)))
          init)))))

(deftype HashCollisionNode [edit
                            ^:mutable collision-hash
                            ^:mutable cnt
                            ^:mutable arr]
  IKVReduce
  (-kv-reduce [inode f init]
    (inode-kv-reduce arr f init)))

;; DEVIATION: Object methods need to be added via extend-type or we need different approach
;; For now, let's define the node operations as separate functions

(defn bitmap-indexed-node-ensure-editable [inode e]
  (if (identical? e (.-edit inode))
    inode
    (let [n (bit-count (.-bitmap inode))
          new-arr (make-array (if (neg? n) 4 (* 2 (inc n))))]
      (array-copy (.-arr inode) 0 new-arr 0 (* 2 n))
      (BitmapIndexedNode. e (.-bitmap inode) new-arr))))

(defn bitmap-indexed-node-inode-assoc [inode shift hash key val added-leaf?]
  (let [bit (bitpos hash shift)
        idx (bitmap-indexed-node-index (.-bitmap inode) bit)
        arr (.-arr inode)
        bitmap (.-bitmap inode)]
    (if (zero? (bit-and bitmap bit))
      (let [n (bit-count bitmap)]
        (if (>= n 16)
          (let [nodes (make-array 32)
                jdx (mask hash shift)]
            (aset nodes jdx (bitmap-indexed-node-inode-assoc EMPTY-BITMAP-NODE (+ shift 5) hash key val added-leaf?))
            (loop [i 0 j 0]
              (if (< i 32)
                (if (zero? (bit-and (unsigned-bit-shift-right bitmap i) 1))
                  (recur (inc i) j)
                  (do (aset nodes i
                            (if-not (nil? (aget arr j))
                              (bitmap-indexed-node-inode-assoc EMPTY-BITMAP-NODE
                                (+ shift 5) (hash (aget arr j)) (aget arr j) (aget arr (inc j)) added-leaf?)
                              (aget arr (inc j))))
                      (recur (inc i) (+ j 2))))))
            (ArrayNode. nil (inc n) nodes))
          (let [new-arr (make-array (* 2 (inc n)))]
            (array-copy arr 0 new-arr 0 (* 2 idx))
            (aset new-arr (* 2 idx) key)
            (aset new-arr (inc (* 2 idx)) val)
            (array-copy arr (* 2 idx) new-arr (* 2 (inc idx)) (* 2 (- n idx)))
            (set! (.-val added-leaf?) true)
            (BitmapIndexedNode. nil (bit-or bitmap bit) new-arr))))
      (let [key-or-nil (aget arr (* 2 idx))
            val-or-node (aget arr (inc (* 2 idx)))]
        (cond (nil? key-or-nil)
              (let [n (bitmap-indexed-node-inode-assoc val-or-node (+ shift 5) hash key val added-leaf?)]
                (if (identical? n val-or-node)
                  inode
                  (BitmapIndexedNode. nil bitmap (clone-and-set arr (inc (* 2 idx)) n))))

              (key-test key key-or-nil)
              (if (identical? val val-or-node)
                inode
                (BitmapIndexedNode. nil bitmap (clone-and-set arr (inc (* 2 idx)) val)))

              :else
              (do (set! (.-val added-leaf?) true)
                  (BitmapIndexedNode. nil bitmap
                    (clone-and-set arr (* 2 idx) nil (inc (* 2 idx))
                      (create-node5 (+ shift 5) key-or-nil val-or-node hash key val)))))))))

(defn bitmap-indexed-node-inode-without [inode shift hash key]
  (let [bit (bitpos hash shift)
        bitmap (.-bitmap inode)
        arr (.-arr inode)]
    (if (zero? (bit-and bitmap bit))
      inode
      (let [idx (bitmap-indexed-node-index bitmap bit)
            key-or-nil (aget arr (* 2 idx))
            val-or-node (aget arr (inc (* 2 idx)))]
        (cond (nil? key-or-nil)
              (let [n (bitmap-indexed-node-inode-without val-or-node (+ shift 5) hash key)]
                (cond (identical? n val-or-node) inode
                      (not (nil? n)) (BitmapIndexedNode. nil bitmap (clone-and-set arr (inc (* 2 idx)) n))
                      (== bitmap bit) nil
                      :else (BitmapIndexedNode. nil (bit-xor bitmap bit) (remove-pair arr idx))))
              (key-test key key-or-nil)
              (if (== bitmap bit)
                nil
                (BitmapIndexedNode. nil (bit-xor bitmap bit) (remove-pair arr idx)))
              :else inode)))))

(defn bitmap-indexed-node-inode-lookup [inode shift hash key not-found]
  (let [bit (bitpos hash shift)
        bitmap (.-bitmap inode)
        arr (.-arr inode)]
    (if (zero? (bit-and bitmap bit))
      not-found
      (let [idx (bitmap-indexed-node-index bitmap bit)
            key-or-nil (aget arr (* 2 idx))
            val-or-node (aget arr (inc (* 2 idx)))]
        (cond (nil? key-or-nil) (bitmap-indexed-node-inode-lookup val-or-node (+ shift 5) hash key not-found)
              (key-test key key-or-nil) val-or-node
              :else not-found)))))

(defn bitmap-indexed-node-inode-find [inode shift hash key not-found]
  (let [bit (bitpos hash shift)
        bitmap (.-bitmap inode)
        arr (.-arr inode)]
    (if (zero? (bit-and bitmap bit))
      not-found
      (let [idx (bitmap-indexed-node-index bitmap bit)
            key-or-nil (aget arr (* 2 idx))
            val-or-node (aget arr (inc (* 2 idx)))]
        (cond (nil? key-or-nil) (bitmap-indexed-node-inode-find val-or-node (+ shift 5) hash key not-found)
              (key-test key key-or-nil) (MapEntry. key-or-nil val-or-node nil)
              :else not-found)))))

(def EMPTY-BITMAP-NODE (BitmapIndexedNode. nil 0 (make-array 0)))

;; -----------------------------------------------------------------------------
;; ArrayNode Helper Functions
;; (ArrayNode deftype is defined earlier to enable forward references)
;; -----------------------------------------------------------------------------

(defn pack-array-node [array-node edit idx]
  (let [arr (.-arr array-node)
        len (alength arr)
        new-arr (make-array (* 2 (dec (.-cnt array-node))))]
    (loop [i 0 j 1 bitmap 0]
      (if (< i len)
        (if (and (not (== i idx))
                 (not (nil? (aget arr i))))
          (do (aset new-arr j (aget arr i))
              (recur (inc i) (+ j 2) (bit-or bitmap (bit-shift-left 1 i))))
          (recur (inc i) j bitmap))
        (BitmapIndexedNode. edit bitmap new-arr)))))

(defn array-node-inode-assoc [inode shift hash key val added-leaf?]
  (let [idx (mask hash shift)
        node (aget (.-arr inode) idx)]
    (if (nil? node)
      (ArrayNode. nil (inc (.-cnt inode))
        (clone-and-set (.-arr inode) idx
          (bitmap-indexed-node-inode-assoc EMPTY-BITMAP-NODE (+ shift 5) hash key val added-leaf?)))
      (let [n (bitmap-indexed-node-inode-assoc node (+ shift 5) hash key val added-leaf?)]
        (if (identical? n node)
          inode
          (ArrayNode. nil (.-cnt inode) (clone-and-set (.-arr inode) idx n)))))))

(defn array-node-inode-without [inode shift hash key]
  (let [idx (mask hash shift)
        node (aget (.-arr inode) idx)]
    (if-not (nil? node)
      (let [n (bitmap-indexed-node-inode-without node (+ shift 5) hash key)]
        (cond
          (identical? n node)
          inode

          (nil? n)
          (if (<= (.-cnt inode) 8)
            (pack-array-node inode nil idx)
            (ArrayNode. nil (dec (.-cnt inode)) (clone-and-set (.-arr inode) idx n)))

          :else
          (ArrayNode. nil (.-cnt inode) (clone-and-set (.-arr inode) idx n))))
      inode)))

(defn array-node-inode-lookup [inode shift hash key not-found]
  (let [idx (mask hash shift)
        node (aget (.-arr inode) idx)]
    (if-not (nil? node)
      (bitmap-indexed-node-inode-lookup node (+ shift 5) hash key not-found)
      not-found)))

(defn array-node-inode-find [inode shift hash key not-found]
  (let [idx (mask hash shift)
        node (aget (.-arr inode) idx)]
    (if-not (nil? node)
      (bitmap-indexed-node-inode-find node (+ shift 5) hash key not-found)
      not-found)))

;; -----------------------------------------------------------------------------
;; HashCollisionNode Helper Functions
;; (HashCollisionNode deftype is defined earlier to enable forward references)
;; -----------------------------------------------------------------------------

(defn hash-collision-node-find-index [arr cnt key]
  (let [lim (* 2 cnt)]
    (loop [i 0]
      (if (< i lim)
        (if (key-test key (aget arr i))
          i
          (recur (+ i 2)))
        -1))))

(defn hash-collision-node-inode-assoc [inode shift hash key val added-leaf?]
  (if (== hash (.-collision-hash inode))
    (let [idx (hash-collision-node-find-index (.-arr inode) (.-cnt inode) key)]
      (if (== idx -1)
        (let [len (* 2 (.-cnt inode))
              new-arr (make-array (+ len 2))]
          (array-copy (.-arr inode) 0 new-arr 0 len)
          (aset new-arr len key)
          (aset new-arr (inc len) val)
          (set! (.-val added-leaf?) true)
          (HashCollisionNode. nil (.-collision-hash inode) (inc (.-cnt inode)) new-arr))
        (if (= (aget (.-arr inode) (inc idx)) val)
          inode
          (HashCollisionNode. nil (.-collision-hash inode) (.-cnt inode)
            (clone-and-set (.-arr inode) (inc idx) val)))))
    (bitmap-indexed-node-inode-assoc
      (BitmapIndexedNode. nil (bitpos (.-collision-hash inode) shift) (array2 nil inode))
      shift hash key val added-leaf?)))

(defn hash-collision-node-inode-without [inode shift hash key]
  (let [idx (hash-collision-node-find-index (.-arr inode) (.-cnt inode) key)]
    (cond (== idx -1) inode
          (== (.-cnt inode) 1) nil
          :else (HashCollisionNode. nil (.-collision-hash inode) (dec (.-cnt inode))
                  (remove-pair (.-arr inode) (quot idx 2))))))

(defn hash-collision-node-inode-lookup [inode shift hash key not-found]
  (let [idx (hash-collision-node-find-index (.-arr inode) (.-cnt inode) key)]
    (cond (< idx 0) not-found
          :else (aget (.-arr inode) (inc idx)))))

(defn hash-collision-node-inode-find [inode shift hash key not-found]
  (let [idx (hash-collision-node-find-index (.-arr inode) (.-cnt inode) key)]
    (cond (< idx 0) not-found
          :else (MapEntry. (aget (.-arr inode) idx) (aget (.-arr inode) (inc idx)) nil))))

;; -----------------------------------------------------------------------------
;; create-node helper
;; -----------------------------------------------------------------------------

;; DEVIATION: create-node uses separate 5-arg and 6-arg functions since multi-arity not supported
(defn create-node5 [shift key1 val1 key2hash key2 val2]
  (let [key1hash (hash key1)]
    (if (== key1hash key2hash)
      (HashCollisionNode. nil key1hash 2 (array4 key1 val1 key2 val2))
      (let [added-leaf? (Box. false)
            node1 (bitmap-indexed-node-inode-assoc EMPTY-BITMAP-NODE shift key1hash key1 val1 added-leaf?)]
        (bitmap-indexed-node-inode-assoc node1 shift key2hash key2 val2 added-leaf?)))))

(defn create-node6 [edit shift key1 val1 key2hash key2 val2]
  (let [key1hash (hash key1)]
    (if (== key1hash key2hash)
      (HashCollisionNode. nil key1hash 2 (array4 key1 val1 key2 val2))
       (let [added-leaf? (Box. false)
             node1 (bitmap-indexed-node-inode-assoc EMPTY-BITMAP-NODE shift key1hash key1 val1 added-leaf?)]
         (bitmap-indexed-node-inode-assoc node1 shift key2hash key2 val2 added-leaf?)))))

;; -----------------------------------------------------------------------------
;; Map seq helper (must come before map types that use it)
;; -----------------------------------------------------------------------------

;; Helper to build a list of MapEntry from a map using kv-reduce
;; This is eager but simple - builds the full list upfront
(defn -map-to-entry-list [m]
  "Converts a map to a list of MapEntry objects using kv-reduce."
  (-kv-reduce m (fn [acc k v] (cons (make-map-entry k v) acc)) nil))

;; -----------------------------------------------------------------------------
;; PersistentArrayMap
;; -----------------------------------------------------------------------------

(deftype PersistentArrayMap [meta cnt arr ^:mutable __hash]
  ICloneable
  (-clone [_] (PersistentArrayMap. meta cnt arr __hash))

  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta)
      coll
      (PersistentArrayMap. new-meta cnt arr __hash)))

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
              (recur (-assoc ret (-nth e 0) (-nth e 1))
                     (next es))
              (throw (Error. "conj on a map takes map entries or seqables of map entries"))))))))

  IEmptyableCollection
  (-empty [coll] (-with-meta EMPTY-ARRAY-MAP meta))

  IEquiv
  (-equiv [coll other]
    (if (and (map? other) (not (record? other)))
      (let [alen (alength arr)]
        (if (== cnt (count other))
          (loop [i 0]
            (if (< i alen)
              (let [v (-lookup other (aget arr i) lookup-sentinel)]
                (if-not (identical? v lookup-sentinel)
                  (if (= (aget arr (inc i)) v)
                    (recur (+ i 2))
                    false)
                  false))
              true))
          false))
      false))

  IHash
  (-hash [coll] (caching-hash coll hash-unordered-coll __hash))

  ISeqable
  ;; DEVIATION: Builds eager list of MapEntry by walking the array directly
  (-seq [coll]
    (when (pos? cnt)
      (let [a arr
            len (alength a)]
        (loop [i (- len 2) acc nil]
          (if (>= i 0)
            (recur (- i 2) (cons (MapEntry. (aget a i) (aget a (+ i 1)) nil) acc))
            acc)))))

  ICounted
  (-count [coll] cnt)

  ILookup
  (-lookup [coll k]
    (-lookup coll k nil))

  (-lookup [coll k not-found]
    (let [idx (array-map-index-of coll k)]
      (if (== idx -1)
        not-found
        (aget arr (inc idx)))))

  IAssociative
  (-assoc [coll k v]
    (let [idx (array-map-index-of coll k)]
      (cond
        (== idx -1)
        (if (< cnt 8)  ;; HASHMAP-THRESHOLD
          (let [arr (array-map-extend-kv coll k v)]
            (PersistentArrayMap. meta (inc cnt) arr nil))
          ;; DEVIATION: Can't use into or as-> yet, use let instead
          (let [m (-with-meta (-assoc EMPTY-HASH-MAP k v) meta)]
            (loop [i 0 m m]
              (if (< i (alength arr))
                (recur (+ i 2) (-assoc m (aget arr i) (aget arr (inc i))))
                m))))

        (identical? v (aget arr (inc idx)))
        coll

        :else
        (let [new-arr (aclone arr)]
          (aset new-arr (inc idx) v)
          (PersistentArrayMap. meta cnt new-arr nil)))))

  (-contains-key? [coll k]
    (not (== (array-map-index-of coll k) -1)))

  IFind
  (-find [coll k]
    (let [idx (array-map-index-of coll k)]
      (when-not (== idx -1)
        (MapEntry. (aget arr idx) (aget arr (inc idx)) nil))))

  IMap
  (-dissoc [coll k]
    (let [idx (array-map-index-of coll k)]
      (if (>= idx 0)
        (let [len (alength arr)
              new-len (- len 2)]
          (if (zero? new-len)
            (-empty coll)
            (let [new-arr (make-array new-len)]
              (loop [s 0 d 0]
                (cond
                  (>= s len) (PersistentArrayMap. meta (dec cnt) new-arr nil)
                  (= k (aget arr s)) (recur (+ s 2) d)
                  :else (do (aset new-arr d (aget arr s))
                            (aset new-arr (inc d) (aget arr (inc s)))
                            (recur (+ s 2) (+ d 2))))))))
        coll)))

  IKVReduce
  (-kv-reduce [coll f init]
    (let [len (alength arr)]
      (loop [i 0 init init]
        (if (< i len)
          (let [init (f init (aget arr i) (aget arr (inc i)))]
            (if (reduced? init)
              (-deref init)
              (recur (+ i 2) init)))
          init))))

  IReduce
  (-reduce [coll f]
    (seq-reduce f coll))
  (-reduce [coll f start]
    (seq-reduce f start coll))

  IFn
  (-invoke [coll k]
    (-lookup coll k))

  (-invoke [coll k not-found]
    (-lookup coll k not-found)))

(def EMPTY-ARRAY-MAP (PersistentArrayMap. nil 0 (array0) empty-unordered-hash))

;; -----------------------------------------------------------------------------
;; PersistentHashMap
;; -----------------------------------------------------------------------------

(deftype PersistentHashMap [meta cnt root ^:mutable has-nil? nil-val ^:mutable __hash]
  ICloneable
  (-clone [_] (PersistentHashMap. meta cnt root has-nil? nil-val __hash))

  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta)
      coll
      (PersistentHashMap. new-meta cnt root has-nil? nil-val __hash)))

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
              (recur (-assoc ret (-nth e 0) (-nth e 1))
                     (next es))
              (throw (Error. "conj on a map takes map entries or seqables of map entries"))))))))

  IEmptyableCollection
  (-empty [coll] (-with-meta EMPTY-HASH-MAP meta))

  IEquiv
  (-equiv [coll other] (equiv-map coll other))

  IHash
  (-hash [coll] (caching-hash coll hash-unordered-coll __hash))

  ISeqable
  ;; DEVIATION: Builds eager list of MapEntry via kv-reduce (inlined)
  (-seq [coll]
    (when (pos? cnt)
      (-kv-reduce coll (fn [acc k v] (cons (MapEntry. k v nil) acc)) nil)))

  ICounted
  (-count [coll] cnt)

  ILookup
  (-lookup [coll k]
    (-lookup coll k nil))

  (-lookup [coll k not-found]
    (cond (nil? k) (if has-nil?
                     nil-val
                     not-found)
          (nil? root) not-found
          :else (bitmap-indexed-node-inode-lookup root 0 (hash k) k not-found)))

  IAssociative
  (-assoc [coll k v]
    (if (nil? k)
      (if (and has-nil? (identical? v nil-val))
        coll
        (PersistentHashMap. meta (if has-nil? cnt (inc cnt)) root true v nil))
      (let [added-leaf? (Box. false)
            base-node (if (nil? root) EMPTY-BITMAP-NODE root)
            new-root (bitmap-indexed-node-inode-assoc base-node 0 (hash k) k v added-leaf?)]
        (if (identical? new-root root)
          coll
          (PersistentHashMap. meta (if (.-val added-leaf?) (inc cnt) cnt) new-root has-nil? nil-val nil)))))

  (-contains-key? [coll k]
    (cond (nil? k) has-nil?
          (nil? root) false
          :else (not (identical? (bitmap-indexed-node-inode-lookup root 0 (hash k) k lookup-sentinel)
                                 lookup-sentinel))))

  IFind
  (-find [coll k]
    (cond
      (nil? k) (when has-nil? (MapEntry. nil nil-val nil))
      (nil? root) nil
      :else (bitmap-indexed-node-inode-find root 0 (hash k) k nil)))

  IMap
  (-dissoc [coll k]
    (cond (nil? k) (if has-nil?
                     (PersistentHashMap. meta (dec cnt) root false nil nil)
                     coll)
          (nil? root) coll
          :else
          (let [new-root (bitmap-indexed-node-inode-without root 0 (hash k) k)]
            (if (identical? new-root root)
              coll
              (PersistentHashMap. meta (dec cnt) new-root has-nil? nil-val nil)))))

  IKVReduce
  (-kv-reduce [coll f init]
    (let [init (if has-nil? (f init nil nil-val) init)]
      (cond
        (reduced? init) (-deref init)
        (not (nil? root)) (unreduced (-kv-reduce root f init))
        :else init)))

  IFn
  (-invoke [coll k]
    (-lookup coll k))

  (-invoke [coll k not-found]
    (-lookup coll k not-found)))

(def EMPTY-HASH-MAP (PersistentHashMap. nil 0 nil false nil empty-unordered-hash))

;; -----------------------------------------------------------------------------
;; Map Constructor Functions
;; -----------------------------------------------------------------------------

(defn hash-map
  "keyval => key val
  Returns a new hash map with supplied mappings."
  [& keyvals]
  (loop [in (seq keyvals) out EMPTY-HASH-MAP]
    (if in
      (let [k (first in)
            in (next in)]
        (if (nil? in)
          (throw (Error. "hash-map requires an even number of arguments"))
          (recur (next in) (-assoc out k (first in)))))
      out)))

(defn array-map
  "keyval => key val
  Returns a new array map with supplied mappings."
  [& keyvals]
  (let [arr (make-array (count keyvals))]
    (loop [i 0 s (seq keyvals)]
      (if s
        (do
          (aset arr i (first s))
          (recur (inc i) (next s)))
        (PersistentArrayMap. nil (/ (alength arr) 2) arr nil)))))

;; -----------------------------------------------------------------------------
;; Map Type Predicates
;; -----------------------------------------------------------------------------

(defn map?
  "Return true if x is a map (PersistentArrayMap or PersistentHashMap)"
  [x]
  (or (instance? PersistentArrayMap x)
      (instance? PersistentHashMap x)))

;; -----------------------------------------------------------------------------
;; Map Wrapper Functions
;; -----------------------------------------------------------------------------

(defn key
  "Returns the key of the map entry."
  [e]
  (-key e))

(defn val
  "Returns the value of the map entry."
  [e]
  (-val e))

(defn get
  "Returns the value mapped to key, not-found or nil if key not present.
   Note: Use (get m k) for lookup, (get m k default) for lookup with default."
  [m k & not-found]
  (if (nil? m)
    (first not-found)
    (if not-found
      (-lookup m k (first not-found))
      (-lookup m k))))

;; Make keywords callable as functions: (:key map) and (:key map default)
(extend-type Keyword
  IFn
  (-invoke [k coll]
    (get coll k))
  (-invoke [k coll not-found]
    (get coll k not-found)))

(defn assoc
  "assoc[iate]. When applied to a map, returns a new map of the
   same (hashed/sorted) type, that contains the mapping of key(s) to
   val(s). When applied to a vector, returns a new vector that
   contains val at index."
  [coll k v & kvs]
  (let [ret (-assoc coll k v)]
    (if kvs
      (loop [ret ret kvs kvs]
        (if kvs
          (recur (-assoc ret (first kvs) (second kvs)) (next (next kvs)))
          ret))
      ret)))

(defn dissoc
  "dissoc[iate]. Returns a new map of the same (hashed/sorted) type,
   that does not contain a mapping for key(s)."
  [coll k & ks]
  (if (nil? coll)
    nil
    (let [ret (-dissoc coll k)]
      (if ks
        (loop [ret ret ks ks]
          (if ks
            (recur (-dissoc ret (first ks)) (next ks))
            ret))
        ret))))

(defn find
  "Returns the map entry for key, or nil if key not present."
  [coll k]
  (if (nil? coll)
    nil
    (-find coll k)))

(defn contains?
  "Returns true if key is present in the given collection, otherwise
   returns false."
  [coll k]
  (if (nil? coll)
    false
    (-contains-key? coll k)))

(defn keys
  "Returns a sequence of the map's keys, in the same order as (seq map)."
  [hash-map]
  (if (nil? hash-map)
    nil
    (let [s (seq hash-map)]
      (if s
        (loop [s s acc nil]
          (if s
            (recur (next s) (cons (-key (first s)) acc))
            (reverse acc)))
        nil))))

(defn vals
  "Returns a sequence of the map's values, in the same order as (seq map)."
  [hash-map]
  (if (nil? hash-map)
    nil
    (let [s (seq hash-map)]
      (if s
        (loop [s s acc nil]
          (if s
            (recur (next s) (cons (-val (first s)) acc))
            (reverse acc)))
        nil))))

(defn merge
  "Returns a map that consists of the rest of the maps conj-ed onto
   the first. If a key occurs in more than one map, the mapping from
   the latter (left-to-right) will be the mapping in the result."
  [& maps]
  (if (nil? (first maps))
    nil
    (loop [maps maps out (first maps)]
      (if (next maps)
        (let [m (second maps)]
          (recur (next maps)
                 (if (nil? m)
                   out
                   (-kv-reduce m (fn [acc k v] (-assoc acc k v)) out))))
        out))))

(defn select-keys
  "Returns a map containing only those entries in map whose key is in keys"
  [map keyseq]
  (loop [s (seq keyseq) out (hash-map)]
    (if s
      (let [k (first s)
            entry (find map k)]
        (recur (next s)
               (if entry
                 (-assoc out k (-val entry))
                 out)))
      out)))

(defn empty
  "Returns an empty collection of the same category as coll."
  [coll]
  (if (nil? coll)
    nil
    (-empty coll)))

(defn reduce-kv
  "Reduces an associative collection. f should be a function of 3
   arguments. Returns the result of applying f to init, the first key
   and the first value in coll, then applying f to that result and the
   2nd key and value, etc."
  [f init coll]
  (if (nil? coll)
    init
    (-kv-reduce coll f init)))

(defn into
  "Returns a new coll consisting of to-coll with all of the items of
   from-coll conjoined."
  [to from]
  (if (nil? from)
    to
    (loop [s (seq from) out to]
      (if s
        (recur (next s) (conj out (first s)))
        out))))

(defn associative?
  "Returns true if coll implements Associative"
  [coll]
  (if (nil? coll)
    false
    (or (instance? PersistentArrayMap coll)
        (instance? PersistentHashMap coll)
        (instance? PersistentVector coll))))

;; =============================================================================
;; I/O Functions (like Beagle's pattern - variadic wrappers around builtins)
;; =============================================================================

(defn print
  "Prints the object(s) to stdout. print and println produce output for
   human consumption. Returns nil."
  [& args]
  (let [s (seq args)]
    (if (nil? s)
      nil
      (do
        ;; Print first item without leading space
        (_print (first s))
        ;; Print remaining items with leading space
        (loop [s (next s)]
          (if (nil? s)
            nil
            (do
              (_print-space)
              (_print (first s))
              (recur (next s)))))))))

(defn println
  "Same as print followed by (newline). Returns nil."
  [& args]
  (let [s (seq args)]
    (if (nil? s)
      (_newline)
      (do
        ;; Print first item without leading space
        (_print (first s))
        ;; Print remaining items with leading space
        (loop [s (next s)]
          (if (nil? s)
            nil
            (do
              (_print-space)
              (_print (first s))
              (recur (next s)))))
        (_newline)))))
;; =============================================================================
;; Core Macros
;; =============================================================================

;; when - evaluates body when test is true
(def ^:macro when
  (fn [form env test & body]
    (list (quote if) test (cons (quote do) body))))

;; when-not - evaluates body when test is false
(def ^:macro when-not
  (fn [form env test & body]
    (list (quote if) test nil (cons (quote do) body))))

;; if-not - inverse of if
(def ^:macro if-not
  (fn [form env test then & else]
    (if (nil? (first else))
      (list (quote if) test nil then)
      (list (quote if) test (first else) then))))

;; cond - conditional with multiple clauses
(def ^:macro cond
  (fn cond-macro [form env & clauses]
    (if (nil? (seq clauses))
      nil
      (if (nil? (next clauses))
        (throw "cond requires an even number of forms")
        (list (quote if)
              (first clauses)
              (second clauses)
              (cons (quote cond) (next (next clauses))))))))

;; and - short-circuit logical and
(def ^:macro and
  (fn and-macro [form env & exprs]
    (if (nil? (seq exprs))
      true
      (if (nil? (next exprs))
        (first exprs)
        (list (quote if) (first exprs)
              (cons (quote and) (next exprs))
              false)))))

;; or - short-circuit logical or
(def ^:macro or
  (fn or-macro [form env & exprs]
    (if (nil? (seq exprs))
      nil
      (if (nil? (next exprs))
        (first exprs)
        (list (quote let) (list (quote or__auto__) (first exprs))
              (list (quote if) (quote or__auto__)
                    (quote or__auto__)
                    (cons (quote or) (next exprs))))))))

;; -> threading macro (thread first)
(def ^:macro ->
  (fn thread-first [form env x & forms]
    (if (nil? (seq forms))
      x
      (let [f (first forms)
            threaded (if (list? f)
                       (cons (first f) (cons x (next f)))
                       (list f x))]
        (cons (quote ->) (cons threaded (next forms)))))))

;; ->> threading macro (thread last)
(def ^:macro ->>
  (fn thread-last [form env x & forms]
    (if (nil? (seq forms))
      x
      (let [f (first forms)
            threaded (if (list? f)
                       (concat (list (first f)) (next f) (list x))
                       (list f x))]
        (cons (quote ->>) (cons threaded (next forms)))))))

;; comment - ignores all forms
(def ^:macro comment
  (fn [form env & _]
    nil))
