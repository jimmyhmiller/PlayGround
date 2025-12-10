# ClojureScript Core Port

This document contains verbatim copies of ClojureScript's core collection implementations
for porting to quick-clojure-poc. All code is from:

**Source:** https://github.com/clojure/clojurescript/blob/master/src/main/cljs/cljs/core.cljs

**License:** Eclipse Public License 1.0
```
Copyright (c) Rich Hickey. All rights reserved.
The use and distribution terms for this software are covered by the
Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
which can be found in the file epl-v10.html at the root of this distribution.
By using this software in any fashion, you are agreeing to be bound by
the terms of this license.
You must not remove this notice, or any other, from this software.
```

---

## Table of Contents

1. [Compiler Changes Needed](#compiler-changes-needed)
2. [Core Protocols](#core-protocols)
3. [Hashing Infrastructure](#hashing-infrastructure)
4. [List Types](#list-types)
5. [Cons Type](#cons-type)
6. [PersistentVector](#persistentvector)
7. [PersistentHashMap](#persistenthashmap)
8. [Helper Functions](#helper-functions)

---

## Compiler Changes Needed

Before porting the ClojureScript code, the compiler needs these additions:

### 1. Bitwise Operations

```rust
// Add to is_builtin() in compiler.rs
"bit-and" | "bit-or" | "bit-xor" | "bit-not" |
"bit-shift-left" | "bit-shift-right" | "unsigned-bit-shift-right"

// IR Instructions needed:
BitAnd(dst, src1, src2)    // ARM64: AND
BitOr(dst, src1, src2)     // ARM64: ORR
BitXor(dst, src1, src2)    // ARM64: EOR
BitNot(dst, src)           // ARM64: MVN
BitShiftLeft(dst, src, amt)           // ARM64: LSL
BitShiftRight(dst, src, amt)          // ARM64: ASR (arithmetic)
UnsignedBitShiftRight(dst, src, amt)  // ARM64: LSR (logical)
```

### 2. Type Predicates

```rust
// Add builtins that check tag bits:
"nil?"      // check tag == 7 (NIL_TAG)
"number?"   // check tag == 0 (INT) or is float
"string?"   // check is string object
"fn?"       // check is closure/function
"identical?" // raw pointer/value comparison (no untagging)
```

### 3. Numeric Helpers

```rust
// These can be implemented as macros or builtins:
"inc"   // (+ x 1)
"dec"   // (- x 1)
"zero?" // (= x 0)
"pos?"  // (> x 0)
"neg?"  // (< x 0)
"=="    // numeric equality
```

### 4. Mutable Field Support

```rust
// In clojure_ast.rs - parse ^:mutable metadata on deftype fields
// In compiler.rs - track mutable fields in TypeInfo
// Support: (set! (.-field obj) value) for mutable fields
```

### 5. Native Array Type

```rust
// New heap object type for mutable arrays
// Tag value: pick unused tag (e.g., for heap object with specific type_id)
// Layout: [header][length][elem0][elem1]...

// Builtins:
"make-array"  // (make-array size) -> new array
"aget"        // (aget arr idx) -> element
"aset"        // (aset arr idx val) -> val (mutates arr)
"alength"     // (alength arr) -> length
"aclone"      // (aclone arr) -> new array copy
```

### 6. Other Needed Builtins

```rust
"int"              // coerce to integer
"instance?"        // (instance? Type x) - check if x is instance of Type
"satisfies?"       // (satisfies? Protocol x) - check if x implements protocol
"implements?"      // same as satisfies? in CLJS
"count"            // polymorphic count (calls -count protocol method)
"seq"              // polymorphic seq (calls -seq protocol method)
"first"            // polymorphic first
"rest"             // polymorphic rest
"next"             // polymorphic next
"conj"             // polymorphic conj
"get"              // polymorphic lookup
"assoc"            // polymorphic assoc
"dissoc"           // polymorphic dissoc
"nth"              // polymorphic nth
"reduce"           // polymorphic reduce
```

---

## Core Protocols

**Source: cljs/core.cljs lines 590-926**

```clojure
;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - Core Protocols
;; =============================================================================

(defprotocol Fn
  "Marker protocol")

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

(defprotocol ICloneable
  "Protocol for cloning a value."
  (-clone [value]
    "Creates a clone of value."))

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

(defprotocol ASeq
  "Marker protocol indicating an array sequence.")

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
  (-find [coll k] "Returns the map entry for key, or nil if key not present."))

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

(defprotocol ISet
  "Protocol for adding set functionality to a collection."
  (-disjoin [coll v]
    "Returns a new collection of coll that does not contain v."))

(defprotocol IStack
  "Protocol for collections to provide access to their items as stacks. The top
  of the stack should be accessed in the most efficient way for the different
  data structures."
  (-peek [coll]
    "Returns the item from the top of the stack. Is used by cljs.core/peek.")
  (-pop [coll]
    "Returns a new stack without the item on top of the stack. Is used
     by cljs.core/pop."))

(defprotocol IVector
  "Protocol for adding vector functionality to collections."
  (-assoc-n [coll n val]
    "Returns a new vector with value val added at position n."))

(defprotocol IDeref
  "Protocol for adding dereference functionality to a reference."
  (-deref [o]
    "Returns the value of the reference o."))

(defprotocol IDerefWithTimeout
  (-deref-with-timeout [o msec timeout-val]))

(defprotocol IMeta
  "Protocol for accessing the metadata of an object."
  (-meta [o]
    "Returns the metadata of object o."))

(defprotocol IWithMeta
  "Protocol for adding metadata to an object."
  (-with-meta [o meta]
    "Returns a new object with value of o and metadata meta added to it."))

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

(defprotocol IEquiv
  "Protocol for adding value comparison functionality to a type."
  (-equiv [o other]
    "Returns true if o and other are equal, false otherwise."))

(defprotocol IHash
  "Protocol for adding hashing functionality to a type."
  (-hash [o]
    "Returns the hash code of o."))

(defprotocol ISeqable
  "Protocol for adding the ability to a type to be transformed into a sequence."
  (-seq [o]
    "Returns a seq of o, or nil if o is empty."))

(defprotocol ISequential
  "Marker interface indicating a persistent collection of sequential items")

(defprotocol IList
  "Marker interface indicating a persistent list")

(defprotocol IRecord
  "Marker interface indicating a record object")

(defprotocol IReversible
  "Protocol for reversing a seq."
  (-rseq [coll]
    "Returns a seq of the items in coll in reversed order."))

(defprotocol ISorted
  "Protocol for a collection which can represent their items
  in a sorted manner. "
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

(defprotocol IPending
  "Protocol for types which can have a deferred realization. Currently only
  implemented by Delay and LazySeq."
  (-realized? [x]
    "Returns true if a value for x has been produced, false otherwise."))

(defprotocol IWatchable
  "Protocol for types that can be watched. Currently only implemented by Atom."
  (-notify-watches [this oldval newval]
    "Calls all watchers with this, oldval and newval.")
  (-add-watch [this key f]
    "Adds a watcher function f to this. Keys must be unique per reference,
     and can be used to remove the watch with -remove-watch.")
  (-remove-watch [this key]
    "Removes watcher that corresponds to key from this."))

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

(defprotocol IComparable
  "Protocol for values that can be compared."
  (-compare [x y]
    "Returns a negative number, zero, or a positive number when x is logically
     'less than', 'equal to', or 'greater than' y."))

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

(defprotocol INamed
  "Protocol for adding a name."
  (-name [x]
    "Returns the name String of x.")
  (-namespace [x]
    "Returns the namespace String of x."))

(defprotocol IAtom
  "Marker protocol indicating an atom.")

(defprotocol IReset
  "Protocol for adding resetting functionality."
  (-reset! [o new-value]
    "Sets the value of o to new-value."))

(defprotocol ISwap
  "Protocol for adding swapping functionality."
  (-swap! [o f] [o f a] [o f a b] [o f a b xs]
    "Swaps the value of o to be (apply f current-value-of-atom args)."))

(defprotocol IVolatile
  "Protocol for adding volatile functionality."
  (-vreset! [o new-value]
    "Sets the value of volatile o to new-value without regard for the
     current value. Returns new-value."))

(defprotocol IIterable
  "Protocol for iterating over a collection."
  (-iterator [coll]
    "Returns an iterator for coll."))

(defprotocol IDrop
  "Protocol for persistent or algorithmically defined collections to provide a
  means of dropping N items that is more efficient than sequential walking."
  (-drop [coll n]
    "Returns a collection that is ISequential, ISeq, and IReduce, or nil if past
     the end. The number of items to drop n must be > 0. It is also useful if the
     returned coll implements IDrop for subsequent use in a partition-like scenario."))
```

---

## Hashing Infrastructure

**Source: cljs/core.cljs lines 944-1097**

```clojure
;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - Murmur3 Hashing
;; =============================================================================

;;http://hg.openjdk.java.net/jdk7u/jdk7u6/jdk/file/8c2c5d63a17e/src/share/classes/java/lang/Integer.java
(defn int-rotate-left [x n]
  (bit-or
    (bit-shift-left x n)
    (unsigned-bit-shift-right x (- n))))

;; DEVIATION: ClojureScript uses Math/imul with a fallback for older JS engines.
;; We need to implement imul as a builtin or use the fallback version.
(defn imul [a b]
  (let [ah (bit-and (unsigned-bit-shift-right a 16) 0xffff)
        al (bit-and a 0xffff)
        bh (bit-and (unsigned-bit-shift-right b 16) 0xffff)
        bl (bit-and b 0xffff)]
    (bit-or
      (+ (* al bl)
         (unsigned-bit-shift-right
           (bit-shift-left (+ (* ah bl) (* al bh)) 16) 0)) 0)))

;; http://smhasher.googlecode.com/svn/trunk/MurmurHash3.cpp
(def m3-seed 0)
(def m3-C1 (int 0xcc9e2d51))
(def m3-C2 (int 0x1b873593))

(defn m3-mix-K1 [k1]
  (-> (int k1) (imul m3-C1) (int-rotate-left 15) (imul m3-C2)))

(defn m3-mix-H1 [h1 k1]
  (int (-> (int h1) (bit-xor (int k1)) (int-rotate-left 13) (imul 5) (+ (int 0xe6546b64)))))

(defn m3-fmix [h1 len]
  ;; DEVIATION: ClojureScript uses as-> macro. We use let instead.
  (let [h1 (int h1)
        h1 (bit-xor h1 len)
        h1 (bit-xor h1 (unsigned-bit-shift-right h1 16))
        h1 (imul h1 (int 0x85ebca6b))
        h1 (bit-xor h1 (unsigned-bit-shift-right h1 13))
        h1 (imul h1 (int 0xc2b2ae35))
        h1 (bit-xor h1 (unsigned-bit-shift-right h1 16))]
    h1))

(defn m3-hash-int [in]
  (if (zero? in)
    in
    (let [k1 (m3-mix-K1 in)
          h1 (m3-mix-H1 m3-seed k1)]
      (m3-fmix h1 4))))

(defn hash-combine [seed hash]
  ; a la boost
  (bit-xor seed
    (+ hash 0x9e3779b9
      (bit-shift-left seed 6)
      (bit-shift-right seed 2))))

;; =============================================================================
;; String Hashing (simplified from CLJS)
;; =============================================================================

;;http://hg.openjdk.java.net/jdk7u/jdk7u6/jdk/file/8c2c5d63a17e/src/share/classes/java/lang/String.java
(defn hash-string* [s]
  (if (nil? s)
    0
    (let [len (string-length s)]  ;; DEVIATION: ClojureScript uses (.-length s)
      (if (> len 0)
        (loop [i 0 hash 0]
          (if (< i len)
            (recur (+ i 1) (+ (imul 31 hash) (char-code-at s i)))  ;; DEVIATION: need char-code-at builtin
            hash))
        0))))

;; =============================================================================
;; Collection Hashing
;; =============================================================================

(def empty-ordered-hash (m3-fmix m3-seed 0))
(def empty-unordered-hash 0)

(defn hash-ordered-coll
  "Returns the hash code, consistent with =, for an external ordered
   collection implementing Iterable."
  [coll]
  (loop [n 0 hash-code 1 coll (seq coll)]
    (if coll
      (recur (+ n 1)
             (bit-or (+ (imul 31 hash-code) (hash (first coll))) 0)
             (next coll))
      (m3-fmix (m3-mix-H1 m3-seed (m3-mix-K1 hash-code)) n))))

(defn hash-unordered-coll
  "Returns the hash code, consistent with =, for an external unordered
   collection implementing Iterable. For maps, the iterator should
   return map entries whose hash is computed as
     (hash-ordered-coll [k v])."
  [coll]
  (loop [n 0 hash-code 0 coll (seq coll)]
    (if coll
      (recur (+ n 1) (bit-or (+ hash-code (hash (first coll))) 0) (next coll))
      (m3-fmix (m3-mix-H1 m3-seed (m3-mix-K1 hash-code)) n))))

;; =============================================================================
;; Caching Hash Macro (converted to function for our use)
;; =============================================================================

;; DEVIATION: ClojureScript uses a macro. We use a pattern instead.
;; In deftype implementations, use this pattern:
;;   (if (nil? __hash)
;;     (let [h (compute-hash-fn coll)]
;;       (set! __hash h)
;;       h)
;;     __hash)
```

---

## List Types

**Source: cljs/core.cljs lines 3209-3354**

```clojure
;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - List
;; =============================================================================

(deftype List [meta first rest count ^:mutable __hash]
  ;; DEVIATION: Skipping Object methods (toString, equiv, indexOf, lastIndexOf)
  ;; These require JavaScript interop that we don't have.

  IList  ;; Marker protocol

  ICloneable
  (-clone [_] (List. meta first rest count __hash))

  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta)
      coll
      (List. new-meta first rest count __hash)))

  IMeta
  (-meta [coll] meta)

  ASeq  ;; Marker protocol

  ISeq
  (-first [coll] first)
  (-rest [coll]
    (if (= count 1)  ;; DEVIATION: ClojureScript uses (== count 1)
      ()
      rest))

  INext
  (-next [coll]
    (if (= count 1)  ;; DEVIATION: ClojureScript uses (== count 1)
      nil
      rest))

  IStack
  (-peek [coll] first)
  (-pop [coll] (-rest coll))

  ICollection
  (-conj [coll o] (List. meta o coll (+ count 1) nil))  ;; DEVIATION: (inc count) -> (+ count 1)

  IEmptyableCollection
  (-empty [coll] (-with-meta (.-EMPTY List) meta))  ;; Requires static field access

  ISequential  ;; Marker protocol

  IEquiv
  (-equiv [coll other] (equiv-sequential coll other))

  IHash
  (-hash [coll]
    ;; Caching hash pattern
    (if (nil? __hash)
      (let [h (hash-ordered-coll coll)]
        (set! __hash h)
        h)
      __hash))

  ISeqable
  (-seq [coll] coll)

  ICounted
  (-count [coll] count)

  IReduce
  (-reduce [coll f] (seq-reduce f coll))
  (-reduce [coll f start] (seq-reduce f start coll)))

(defn list?
  "Returns true if x implements IList"
  [x]
  (satisfies? IList x))

;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - EmptyList
;; =============================================================================

(deftype EmptyList [meta]
  ;; DEVIATION: Skipping Object methods

  IList  ;; Marker protocol

  ICloneable
  (-clone [_] (EmptyList. meta))

  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta)
      coll
      (EmptyList. new-meta)))

  IMeta
  (-meta [coll] meta)

  ISeq
  (-first [coll] nil)
  (-rest [coll] ())  ;; Returns empty list, not nil

  INext
  (-next [coll] nil)

  IStack
  (-peek [coll] nil)
  (-pop [coll] (throw "Can't pop empty list"))  ;; DEVIATION: simplified error

  ICollection
  (-conj [coll o] (List. meta o nil 1 nil))

  IEmptyableCollection
  (-empty [coll] coll)

  ISequential  ;; Marker protocol

  IEquiv
  (-equiv [coll other]
    (if (or (list? other)
            (sequential? other))
      (nil? (seq other))
      false))

  IHash
  (-hash [coll] empty-ordered-hash)

  ISeqable
  (-seq [coll] nil)

  ICounted
  (-count [coll] 0)

  IReduce
  (-reduce [coll f] (seq-reduce f coll))
  (-reduce [coll f start] (seq-reduce f start coll)))

;; Static field initialization
;; DEVIATION: ClojureScript uses (set! (.-EMPTY List) (EmptyList. nil))
;; We need to support static field setting or use a def
(def EMPTY-LIST (EmptyList. nil))
```

---

## Cons Type

**Source: cljs/core.cljs lines 3391-3457**

```clojure
;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - Cons
;; =============================================================================

(deftype Cons [meta first rest ^:mutable __hash]
  ;; DEVIATION: Skipping Object methods

  IList  ;; Marker protocol

  ICloneable
  (-clone [_] (Cons. meta first rest __hash))

  IWithMeta
  (-with-meta [coll new-meta]
    (if (identical? new-meta meta)
      coll
      (Cons. new-meta first rest __hash)))

  IMeta
  (-meta [coll] meta)

  ASeq  ;; Marker protocol

  ISeq
  (-first [coll] first)
  (-rest [coll] (if (nil? rest) () rest))

  INext
  (-next [coll]
    (if (nil? rest) nil (seq rest)))

  ICollection
  (-conj [coll o] (Cons. nil o coll nil))

  IEmptyableCollection
  (-empty [coll] EMPTY-LIST)  ;; DEVIATION: uses our def instead of (.-EMPTY List)

  ISequential  ;; Marker protocol

  IEquiv
  (-equiv [coll other] (equiv-sequential coll other))

  IHash
  (-hash [coll]
    (if (nil? __hash)
      (let [h (hash-ordered-coll coll)]
        (set! __hash h)
        h)
      __hash))

  ISeqable
  (-seq [coll] coll)

  IReduce
  (-reduce [coll f] (seq-reduce f coll))
  (-reduce [coll f start] (seq-reduce f start coll)))

;; =============================================================================
;; cons function
;; =============================================================================

(defn cons
  "Returns a new seq where x is the first element and coll is the rest."
  [x coll]
  (if (nil? coll)
    (List. nil x nil 1 nil)
    (if (satisfies? ISeq coll)  ;; DEVIATION: ClojureScript uses (implements? ISeq coll)
      (Cons. nil x coll nil)
      (Cons. nil x (seq coll) nil))))
```

---

## Helper Functions

**Source: Various locations in cljs/core.cljs**

```clojure
;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - Sequential Equality
;; =============================================================================

(defn equiv-sequential
  "Assumes x is sequential. Returns true if x equals y, otherwise
  returns false."
  [x y]
  (if (sequential? y)  ;; DEVIATION: need sequential? predicate
    (loop [xs (seq x) ys (seq y)]
      (if (nil? xs)
        (nil? ys)
        (if (nil? ys)
          false
          (if (= (first xs) (first ys))
            (recur (next xs) (next ys))
            false))))
    false))

;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - Seq Reduce
;; =============================================================================

(defn seq-reduce
  ([f coll]
   (let [s (seq coll)]
     (if s
       (let [fst (first s)
             rst (next s)]
         (if rst
           (loop [acc fst s rst]
             (if s
               (let [acc (f acc (first s))]
                 (if (reduced? acc)  ;; DEVIATION: need reduced? predicate
                   (deref acc)       ;; DEVIATION: need deref for reduced values
                   (recur acc (next s))))
               acc))
           fst))
       (f))))  ;; No elements: call f with no args
  ([f init coll]
   (loop [acc init s (seq coll)]
     (if s
       (let [acc (f acc (first s))]
         (if (reduced? acc)
           (deref acc)
           (recur acc (next s))))
       acc))))

;; =============================================================================
;; sequential? predicate
;; =============================================================================

(defn sequential?
  "Returns true if coll implements ISequential"
  [x]
  (satisfies? ISequential x))
```

---

## PersistentVector

**Source: cljs/core.cljs lines 5575-6295**

This is a large implementation. Key components:

```clojure
;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - VectorNode
;; =============================================================================

(deftype VectorNode [edit arr])

;; =============================================================================
;; Vector Constants
;; =============================================================================

(def EMPTY_NODE (VectorNode. nil (make-array 32)))

;; =============================================================================
;; Vector Helper Functions
;; =============================================================================

(defn pv-fresh-node [edit]
  (VectorNode. edit (make-array 32)))

(defn pv-aget [node idx]
  (aget (.-arr node) idx))

(defn pv-aset [node idx val]
  (aset (.-arr node) idx val)
  node)  ;; DEVIATION: returns node for chaining, but this mutates!

(defn pv-clone-node [node]
  (VectorNode. (.-edit node) (aclone (.-arr node))))

(defn tail-off [pv]
  (let [cnt (.-cnt pv)]
    (if (< cnt 32)
      0
      (bit-shift-left (unsigned-bit-shift-right (- cnt 1) 5) 5))))

(defn new-path [edit level node]
  (loop [ll level ret node]
    (if (= ll 0)  ;; DEVIATION: ClojureScript uses (zero? ll)
      ret
      (let [embed ret
            r (pv-fresh-node edit)
            _ (pv-aset r 0 embed)]
        (recur (- ll 5) r)))))

(defn push-tail [pv level parent tailnode]
  (let [subidx (bit-and (unsigned-bit-shift-right (- (.-cnt pv) 1) level) 0x1f)
        ret (pv-clone-node parent)]
    (if (= level 5)  ;; DEVIATION: ClojureScript uses (== 5 level)
      (do
        (pv-aset ret subidx tailnode)
        ret)
      (let [child (pv-aget parent subidx)]
        (if child
          (let [node-to-insert (push-tail pv (- level 5) child tailnode)]
            (pv-aset ret subidx node-to-insert)
            ret)
          (let [node-to-insert (new-path nil (- level 5) tailnode)]
            (pv-aset ret subidx node-to-insert)
            ret))))))

(defn array-for [pv i]
  (if (and (>= i 0) (< i (.-cnt pv)))
    (if (>= i (tail-off pv))
      (.-tail pv)
      (loop [node (.-root pv) level (.-shift pv)]
        (if (> level 0)
          (recur (pv-aget node (bit-and (unsigned-bit-shift-right i level) 0x1f))
                 (- level 5))
          (.-arr node))))
    (throw "Index out of bounds")))  ;; DEVIATION: simplified error

;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - PersistentVector
;; =============================================================================

(deftype PersistentVector [meta cnt shift root tail ^:mutable __hash]
  ;; DEVIATION: Skipping Object methods and JS interop

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
    (if (> cnt 0)
      (-nth coll (- cnt 1))
      nil))
  (-pop [coll]
    (if (= cnt 0)  ;; DEVIATION: (zero? cnt)
      (throw "Can't pop empty vector")
      (if (= cnt 1)
        (-with-meta EMPTY-VECTOR meta)  ;; Need EMPTY-VECTOR defined
        (if (> (- cnt (tail-off coll)) 1)
          (let [new-tail (make-array (- (alength tail) 1))]
            (loop [i 0]
              (if (< i (alength new-tail))
                (do
                  (aset new-tail i (aget tail i))
                  (recur (+ i 1)))
                (PersistentVector. meta (- cnt 1) shift root new-tail nil))))
          (let [new-tail (array-for coll (- cnt 2))
                nr (pop-tail coll shift root)]
            (if (nil? nr)
              (PersistentVector. meta (- cnt 1) shift EMPTY_NODE new-tail nil)
              (if (and (> shift 5) (nil? (pv-aget nr 1)))
                (PersistentVector. meta (- cnt 1) (- shift 5) (pv-aget nr 0) new-tail nil)
                (PersistentVector. meta (- cnt 1) shift nr new-tail nil))))))))

  ICollection
  (-conj [coll o]
    (if (< (- cnt (tail-off coll)) 32)
      ;; Room in tail
      (let [len (alength tail)
            new-tail (make-array (+ len 1))]
        (loop [i 0]
          (if (< i len)
            (do
              (aset new-tail i (aget tail i))
              (recur (+ i 1)))
            (do
              (aset new-tail len o)
              (PersistentVector. meta (+ cnt 1) shift root new-tail nil)))))
      ;; Tail is full, push into tree
      (let [tail-node (VectorNode. (.-edit root) tail)
            new-shift shift]
        (if (> (unsigned-bit-shift-right cnt 5) (bit-shift-left 1 shift))
          ;; Overflow root
          (let [new-root (pv-fresh-node nil)]
            (pv-aset new-root 0 root)
            (pv-aset new-root 1 (new-path nil shift tail-node))
            (PersistentVector. meta (+ cnt 1) (+ shift 5) new-root (make-array 1)
              (do (aset (make-array 1) 0 o) (make-array 1))))  ;; DEVIATION: awkward
          (let [new-root (push-tail coll shift root tail-node)]
            (PersistentVector. meta (+ cnt 1) shift new-root
              (let [a (make-array 1)] (aset a 0 o) a) nil))))))

  IEmptyableCollection
  (-empty [coll] (-with-meta EMPTY-VECTOR meta))

  ISequential  ;; Marker

  IEquiv
  (-equiv [coll other]
    (if (vector? other)  ;; DEVIATION: need vector? predicate
      (if (= cnt (count other))
        (loop [i 0]
          (if (= i cnt)
            true
            (if (= (-nth coll i) (-nth other i))
              (recur (+ i 1))
              false)))
        false)
      (equiv-sequential coll other)))

  IHash
  (-hash [coll]
    (if (nil? __hash)
      (let [h (hash-ordered-coll coll)]
        (set! __hash h)
        h)
      __hash))

  ISeqable
  (-seq [coll]
    (if (= cnt 0)  ;; DEVIATION: (zero? cnt)
      nil
      ;; DEVIATION: ClojureScript returns chunked-seq, we return simple seq
      (vector-seq coll 0)))

  ICounted
  (-count [coll] cnt)

  IIndexed
  (-nth [coll n]
    (let [arr (array-for coll n)]
      (aget arr (bit-and n 0x1f))))
  (-nth [coll n not-found]
    (if (and (>= n 0) (< n cnt))
      (-nth coll n)
      not-found))

  ILookup
  (-lookup [coll k] (-lookup coll k nil))
  (-lookup [coll k not-found]
    (if (number? k)
      (-nth coll k not-found)
      not-found))

  IAssociative
  (-contains-key? [coll k]
    (if (number? k)
      (and (>= k 0) (< k cnt))
      false))
  (-assoc [coll k v]
    (if (number? k)
      (-assoc-n coll k v)
      (throw "Vector's key for assoc must be a number")))

  IFind
  (-find [coll k]
    (if (and (number? k) (>= k 0) (< k cnt))
      ;; DEVIATION: Should return MapEntry, returning vector for now
      [k (-nth coll k)]
      nil))

  IVector
  (-assoc-n [coll n val]
    (if (and (>= n 0) (< n cnt))
      (if (>= n (tail-off coll))
        ;; In tail
        (let [new-tail (aclone tail)]
          (aset new-tail (bit-and n 0x1f) val)
          (PersistentVector. meta cnt shift root new-tail nil))
        ;; In tree
        (PersistentVector. meta cnt shift (do-assoc coll shift root n val) tail nil))
      (if (= n cnt)
        (-conj coll val)
        (throw "Index out of bounds"))))

  IReduce
  (-reduce [coll f] (vector-reduce coll f))
  (-reduce [coll f init] (vector-reduce coll f init))

  IKVReduce
  (-kv-reduce [coll f init]
    (loop [i 0 acc init]
      (if (< i cnt)
        (let [acc (f acc i (-nth coll i))]
          (if (reduced? acc)
            (deref acc)
            (recur (+ i 1) acc)))
        acc)))

  IFn
  (-invoke [coll k]
    (-nth coll k))
  (-invoke [coll k not-found]
    (-nth coll k not-found))

  IReversible
  (-rseq [coll]
    (if (> cnt 0)
      (ReverseVectorSeq. coll (- cnt 1) nil)  ;; DEVIATION: need ReverseVectorSeq
      nil)))

;; =============================================================================
;; Vector Helper Types and Functions
;; =============================================================================

;; Simple vector seq (not chunked like CLJS)
(deftype VectorSeq [vec i meta]
  ISeqable
  (-seq [this] this)

  ISeq
  (-first [_] (-nth vec i))
  (-rest [_]
    (if (< (+ i 1) (-count vec))
      (VectorSeq. vec (+ i 1) nil)
      ()))

  INext
  (-next [_]
    (if (< (+ i 1) (-count vec))
      (VectorSeq. vec (+ i 1) nil)
      nil))

  ICounted
  (-count [_] (- (-count vec) i))

  ISequential

  IEquiv
  (-equiv [this other] (equiv-sequential this other)))

(defn vector-seq [v start]
  (if (< start (-count v))
    (VectorSeq. v start nil)
    nil))

(defn do-assoc [coll level node i val]
  (let [ret (pv-clone-node node)]
    (if (= level 0)  ;; DEVIATION: (zero? level)
      (do
        (pv-aset ret (bit-and i 0x1f) val)
        ret)
      (let [subidx (bit-and (unsigned-bit-shift-right i level) 0x1f)]
        (pv-aset ret subidx (do-assoc coll (- level 5) (pv-aget node subidx) i val))
        ret))))

(defn pop-tail [coll level node]
  (let [subidx (bit-and (unsigned-bit-shift-right (- (.-cnt coll) 2) level) 0x1f)]
    (if (> level 5)
      (let [new-child (pop-tail coll (- level 5) (pv-aget node subidx))]
        (if (and (nil? new-child) (= subidx 0))  ;; DEVIATION: (zero? subidx)
          nil
          (let [ret (pv-clone-node node)]
            (pv-aset ret subidx new-child)
            ret)))
      (if (= subidx 0)  ;; DEVIATION: (zero? subidx)
        nil
        (let [ret (pv-clone-node node)]
          (pv-aset ret subidx nil)
          ret)))))

(defn vector-reduce
  ([v f]
   (let [cnt (-count v)]
     (if (> cnt 0)
       (loop [i 1 acc (-nth v 0)]
         (if (< i cnt)
           (let [acc (f acc (-nth v i))]
             (if (reduced? acc)
               (deref acc)
               (recur (+ i 1) acc)))
           acc))
       (f))))
  ([v f init]
   (let [cnt (-count v)]
     (loop [i 0 acc init]
       (if (< i cnt)
         (let [acc (f acc (-nth v i))]
           (if (reduced? acc)
             (deref acc)
             (recur (+ i 1) acc)))
         acc)))))

;; Empty vector singleton
(def EMPTY-VECTOR (PersistentVector. nil 0 5 EMPTY_NODE (make-array 0) nil))

(defn vector? [x]
  (satisfies? IVector x))

(defn vec
  "Creates a new vector containing the contents of coll."
  [coll]
  (if (vector? coll)
    coll  ;; DEVIATION: CLJS does (-with-meta coll nil), we just return
    (loop [s (seq coll) v EMPTY-VECTOR]
      (if s
        (recur (next s) (-conj v (first s)))
        v))))

(defn vector
  "Creates a new vector containing the args."
  [& args]
  (vec args))
```

---

## PersistentHashMap

**Source: cljs/core.cljs - This is very large (~1500 lines)**

The HashMap implementation uses Hash Array Mapped Tries (HAMT). Key types:

```clojure
;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - HashMap Node Types
;; =============================================================================

;; BitmapIndexedNode - sparse node using bitmap
(deftype BitmapIndexedNode [edit ^:mutable bitmap ^:mutable arr]
  ;; ... implementation
)

;; ArrayNode - dense node with 32 children
(deftype ArrayNode [edit ^:mutable cnt ^:mutable arr]
  ;; ... implementation
)

;; HashCollisionNode - handles hash collisions
(deftype HashCollisionNode [edit ^:mutable collision-hash ^:mutable cnt ^:mutable arr]
  ;; ... implementation
)

;; =============================================================================
;; VERBATIM FROM CLOJURESCRIPT - PersistentHashMap
;; =============================================================================

(deftype PersistentHashMap [meta cnt root ^:mutable has-nil? ^:mutable nil-val ^:mutable __hash]
  ;; Note: has-nil? and nil-val handle the special case of nil as a key

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
              (recur (-assoc ret (-nth e 0) (-nth e 1)) (next es))
              (throw "conj on map takes map entries or vectors")))))))

  IEmptyableCollection
  (-empty [coll] (-with-meta EMPTY-HASH-MAP meta))

  IEquiv
  (-equiv [coll other]
    (equiv-map coll other))  ;; DEVIATION: need equiv-map

  IHash
  (-hash [coll]
    (if (nil? __hash)
      (let [h (hash-unordered-coll coll)]
        (set! __hash h)
        h)
      __hash))

  ISeqable
  (-seq [coll]
    (if (> cnt 0)
      (let [s (if root (inode-seq root) nil)]
        (if has-nil?
          (cons [nil nil-val] s)  ;; DEVIATION: Should be MapEntry
          s))
      nil))

  ICounted
  (-count [coll] cnt)

  ILookup
  (-lookup [coll k] (-lookup coll k nil))
  (-lookup [coll k not-found]
    (if (nil? k)
      (if has-nil?
        nil-val
        not-found)
      (if root
        (inode-lookup root 0 (hash k) k not-found)
        not-found)))

  IAssociative
  (-contains-key? [coll k]
    (if (nil? k)
      has-nil?
      (if root
        (not (identical? (inode-lookup root 0 (hash k) k lookup-sentinel) lookup-sentinel))
        false)))
  (-assoc [coll k v]
    (if (nil? k)
      (if (and has-nil? (identical? v nil-val))
        coll
        (PersistentHashMap. meta (if has-nil? cnt (+ cnt 1)) root true v nil))
      (let [added-leaf? (Box. false)  ;; DEVIATION: need Box type for mutation tracking
            new-root (inode-assoc (if root root EMPTY-BITMAP-NODE) 0 (hash k) k v added-leaf?)]
        (if (identical? new-root root)
          coll
          (PersistentHashMap. meta (if (.-val added-leaf?) (+ cnt 1) cnt) new-root has-nil? nil-val nil)))))

  IMap
  (-dissoc [coll k]
    (if (nil? k)
      (if has-nil?
        (PersistentHashMap. meta (- cnt 1) root false nil nil)
        coll)
      (if root
        (let [new-root (inode-dissoc root 0 (hash k) k)]
          (if (identical? new-root root)
            coll
            (PersistentHashMap. meta (- cnt 1) new-root has-nil? nil-val nil)))
        coll)))

  IKVReduce
  (-kv-reduce [coll f init]
    (let [init (if has-nil? (f init nil nil-val) init)]
      (if (reduced? init)
        (deref init)
        (if root
          (inode-kv-reduce root f init)
          init))))

  IFn
  (-invoke [coll k] (-lookup coll k))
  (-invoke [coll k not-found] (-lookup coll k not-found)))

;; Sentinel value for lookup misses
(def lookup-sentinel (js-obj))  ;; DEVIATION: need unique object

;; Empty hash map singleton
(def EMPTY-HASH-MAP (PersistentHashMap. nil 0 nil false nil nil))

(defn hash-map
  "Returns a new hash map with supplied mappings."
  [& keyvals]
  (loop [in (seq keyvals) out EMPTY-HASH-MAP]
    (if in
      (let [k (first in)
            in (next in)]
        (if (nil? in)
          (throw "hash-map requires even number of arguments")
          (recur (next in) (-assoc out k (first in)))))
      out)))
```

---

## Implementation Notes

### Required Additional Types

1. **Box** - Mutable container for tracking mutations in persistent operations
   ```clojure
   (deftype Box [^:mutable val])
   ```

2. **MapEntry** - Key-value pair for map iteration
   ```clojure
   (deftype MapEntry [key val]
     IMapEntry
     (-key [_] key)
     (-val [_] val)

     IVector
     ;; MapEntry acts like a 2-element vector
     )
   ```

3. **Reduced** - Wrapper for early termination in reduce
   ```clojure
   (deftype Reduced [val]
     IDeref
     (-deref [_] val))

   (defn reduced [x] (Reduced. x))
   (defn reduced? [x] (instance? Reduced x))
   ```

### Deviations Summary

| ClojureScript | Our Version | Reason |
|---------------|-------------|--------|
| `(== x y)` | `(= x y)` | No `==` builtin yet |
| `(zero? x)` | `(= x 0)` | No `zero?` builtin yet |
| `(inc x)` | `(+ x 1)` | No `inc` builtin yet |
| `(dec x)` | `(- x 1)` | No `dec` builtin yet |
| `(implements? P x)` | `(satisfies? P x)` | Same thing in CLJS |
| `(.-length s)` | `(string-length s)` | Different string access |
| `(.charCodeAt s i)` | `(char-code-at s i)` | Different method call |
| `js/Error.` | `(throw msg)` | No JS interop |
| `as->` macro | nested `let` | No `as->` macro |
| ChunkedSeq | VectorSeq | Simpler, non-chunked implementation |
| Object methods | Skipped | No JS Object protocol |
| `es6-iterable` | Skipped | No ES6 iterator support |

---

## Testing Checklist

### Protocol Tests
- [ ] Define all protocols successfully
- [ ] Protocol method dispatch works
- [ ] `satisfies?` works correctly

### List Tests
- [ ] `(list 1 2 3)` creates list
- [ ] `first`, `rest`, `next` work
- [ ] `conj` prepends to list
- [ ] `count` returns correct value
- [ ] `seq` returns self
- [ ] Empty list works correctly
- [ ] Hash is consistent

### Vector Tests
- [ ] `(vector 1 2 3)` creates vector
- [ ] `nth` returns correct element
- [ ] `conj` appends to vector
- [ ] `assoc` updates element
- [ ] `pop` removes last element
- [ ] Large vectors (> 32 elements) work
- [ ] Very large vectors (> 1024 elements) work
- [ ] Hash is consistent

### HashMap Tests
- [ ] `(hash-map :a 1 :b 2)` creates map
- [ ] `get` retrieves values
- [ ] `assoc` adds/updates
- [ ] `dissoc` removes
- [ ] `nil` key handling
- [ ] Hash collisions handled
- [ ] Large maps work
- [ ] Hash is consistent

---

## Next Steps

1. **Implement compiler primitives** (Step 1-5 from plan)
2. **Create lib/cljs/core.clj** with protocols
3. **Port List types** and test
4. **Port Vector** and test
5. **Port HashMap** and test
6. **Add remaining types** (Set, etc.)
