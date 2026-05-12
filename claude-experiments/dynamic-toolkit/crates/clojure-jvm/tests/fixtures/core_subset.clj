;; A working subset of clojure.core, with minor modifications for what
;; we can't yet load 1:1 from upstream `clojure/core.clj`. Each
;; modification is annotated with `;; PORT:` and references the original.
;;
;; Source: /Users/jimmyhmiller/Documents/Code/open-source/clojure/src/clj/clojure/core.clj
;;
;; The aim is for this file to converge on the upstream as we add
;; features (proper `defmacro`, persistent vectors, java interop bridges,
;; etc.). Until then, the modifications below are intentional.

(ns ^{:doc "The core Clojure language."
       :author "Rich Hickey"}
  clojure.core)

(def unquote)
(def unquote-splicing)

;; PORT: original is `(. clojure.lang.PersistentList creator)` (static field
;; holding an IFn). For our subset, `list` is just a variadic identity.
(def
 ^{:arglists '([& items])
   :doc "Creates a new list containing the items."
   :added "1.0"}
  list (fn* [& xs] xs))

(def
 ^{:arglists '([x seq])
    :doc "Returns a new seq where x is the first element and seq is
    the rest."
   :added "1.0"
   :static true}
 cons (fn* [x seq] (. clojure.lang.RT (cons x seq))))

(def
 ^{:arglists '([coll])
   :doc "Returns the first item in the collection. Calls seq on its
    argument. If coll is nil, returns nil."
   :added "1.0"
   :static true}
 first (fn* [coll] (. clojure.lang.RT (first coll))))

(def
 ^{:arglists '([coll])
   :doc "Returns a seq of the items after the first."
   :added "1.0"
   :static true}
 next (fn* [x] (. clojure.lang.RT (next x))))

;; PORT: original uses RT.more. We alias rest to next for our Cons-only
;; subset (lazy seqs differ from Cons here, but we don't have lazy seqs).
(def
 ^{:arglists '([coll])
   :doc "Returns a possibly empty seq of the items after the first."
   :added "1.0"
   :static true}
 rest (fn* [x] (. clojure.lang.RT (more x))))

(def
 ^{:doc "Same as (first (next x))"
   :arglists '([x])
   :added "1.0"
   :static true}
 second (fn* [x] (first (next x))))

(def
 ^{:doc "Same as (first (first x))"
   :arglists '([x])
   :added "1.0"
   :static true}
 ffirst (fn* [x] (first (first x))))

(def
 ^{:doc "Same as (next (first x))"
   :arglists '([x])
   :added "1.0"
   :static true}
 nfirst (fn* [x] (next (first x))))

(def
 ^{:doc "Same as (first (next x))"
   :arglists '([x])
   :added "1.0"
   :static true}
 fnext (fn* [x] (first (next x))))

(def
 ^{:doc "Same as (next (next x))"
   :arglists '([x])
   :added "1.0"
   :static true}
 nnext (fn* [x] (next (next x))))

;; PORT: upstream uses `clojure.lang.RT/conj` syntax we don't yet parse.
;; Substituted with the equivalent `(. clojure.lang.RT (cons x coll))`
;; — for the Cons-list world `conj` on a list IS cons.
;; Also: 0-arg returns nil (upstream returns the empty vector []).
(def
 ^{:arglists '([] [coll] [coll x] [coll x & xs])
   :doc "conj[oin]. Returns a new collection with the xs 'added'."
   :added "1.0"
   :static true}
 conj (fn*
        ([] nil)
        ([coll] coll)
        ([coll x] (. clojure.lang.RT (cons x coll)))
        ([coll x & xs]
         (if xs
           (recur (. clojure.lang.RT (cons x coll)) (first xs) (next xs))
           (. clojure.lang.RT (cons x coll))))))

;; PORT: upstream `(. clojure.lang.Util (identical x nil))`. Our
;; substitute uses the runtime nil-tag check directly.
(def
 ^{:doc "Returns true if x is nil, false otherwise."
   :added "1.0"
   :static true}
 nil? (fn* [x] (. clojure.lang.Util (isNil x))))

;; PORT: upstream uses `Util.identical x true/false` to identity-check
;; against the boolean singletons.
(def
 ^{:doc "Returns true if x is logical false, false otherwise."
   :added "1.0"
   :static true}
 false? (fn* [x] (if (= x false) true false)))

(def
 ^{:doc "Returns true if x is logical true, false otherwise."
   :added "1.0"
   :static true}
 true? (fn* [x] (if (= x true) true false)))

(def
 ^{:doc "Returns the logical negation of x."
   :added "1.0"
   :static true}
 not (fn* [x] (if x false true)))

(def
 ^{:arglists '([coll])
   :doc "Return the last item in coll, in linear time."
   :added "1.0"
   :static true}
 last (fn* [s]
        (if (next s)
          (recur (next s))
          (first s))))

;; PORT: upstream is `(. clojure.lang.Util (identical x y))`; substituted
;; with `=` which our compiler lowers to `cljvm_rt_equiv` (Util.equiv).
;; For nil/bool/number this is equivalent to identical?.
(def
 ^{:arglists '([x y])
   :doc "Tests if 2 arguments are the same object."
   :added "1.0"
   :static true}
 identical? (fn* [x y] (= x y)))

;; PORT: upstream uses Numbers/inc which throws on overflow.
;; Substituted with primop + (no overflow check).
(def
 ^{:arglists '([x])
   :doc "Returns a number one greater than num. Does not auto-promote."
   :added "1.2"
   :static true}
 inc (fn* [x] (+ x 1)))

(def
 ^{:arglists '([x])
   :doc "Returns a number one less than num. Does not auto-promote."
   :added "1.2"
   :static true}
 dec (fn* [x] (- x 1)))

;; PORT: upstream is `(clojure.lang.Numbers (isPos x))` etc. Our `>` lowers
;; to Numbers.gt; substituting the predicate to use `>` directly.
(def
 ^{:arglists '([x])
   :doc "Returns true if num is greater than zero, else false."
   :added "1.0"
   :static true}
 pos? (fn* [x] (> x 0)))

(def
 ^{:arglists '([x])
   :doc "Returns true if num is less than zero, else false."
   :added "1.0"
   :static true}
 neg? (fn* [x] (< x 0)))

(def
 ^{:arglists '([x])
   :doc "Returns true if num is zero, else false."
   :added "1.0"
   :static true}
 zero? (fn* [x] (= x 0)))

(def
 ^{:arglists '([x y] [x y & more])
   :doc "Returns the greatest of the nums."
   :added "1.0"
   :static true}
 max (fn*
       ([x y] (if (> x y) x y))
       ([x y & more]
        (if more
          (recur (if (> x y) x y) (first more) (next more))
          (if (> x y) x y)))))

(def
 ^{:arglists '([x y] [x y & more])
   :doc "Returns the least of the nums."
   :added "1.0"
   :static true}
 min (fn*
       ([x y] (if (< x y) x y))
       ([x y & more]
        (if more
          (recur (if (< x y) x y) (first more) (next more))
          (if (< x y) x y)))))

;; PORT: upstream `if-not` uses syntax-quote ``(if (not ~test) ...)``. We
;; don't have syntax-quote yet, so use a plain `(list 'if ...)` template.
;; All Clojure macros take implicit `&form &env` as first two params.
(def ^:macro if-not
  (fn* [&form &env test then else]
    (list (quote if) test else then)))

;; PORT: upstream `when` uses syntax-quote ``(if ~test (do ~@body))``. Our
;; substitution: explicit (list 'if test (cons 'do body)).
(def ^:macro when
  (fn* [&form &env test & body]
    (list (quote if) test (cons (quote do) body) nil)))

(def ^:macro when-not
  (fn* [&form &env test & body]
    (list (quote if) test nil (cons (quote do) body))))

