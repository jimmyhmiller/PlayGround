//! `clojure.core`: the mini-Clojure standard library, written in the language on
//! top of the toolkit's primitives. Collections are LIST-BACKED tagged records
//! (`(record 'Vector elem-list)` etc.); `type-of` drives dispatch. Internals use
//! the un-shadowable `%`-prims (`%add`/`%first`/…); user-facing `+`/`=`/`first`
//! are first-class fns wrapping them (so they can be passed to `map`/`reduce`).

pub const CORE: &str = r#"
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
(defn vector [& es] (record 'Vector es))
(defn hash-map [& kvs] (record 'Map kvs))
(defn hash-set [& es] (record 'Set es))

;; ─────────────── type predicates ───────────────
(defn vector? [x] (%num-eq (type-of x) 'Vector))
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
(defn pos? [n] (%lt 0 n))
(defn neg? [n] (%lt n 0))
(defn zero? [n] (%num-eq n 0))
(defn identity [x] x)
(defn even-nn [n] (cond (%num-eq n 0) true (%num-eq n 1) false true (even-nn (%sub n 2))))
(defn even? [n] (if (%lt n 0) (even-nn (%sub 0 n)) (even-nn n)))
(defn odd? [n] (not (even? n)))

;; ─────────────── seq abstraction (with lazy sequences) ───────────────
(defn elems [c] (field c 0))
(defn entries [kvs]
  (if (nil? kvs) nil
      (%cons (vector (%first kvs) (%first (%rest kvs))) (entries (%rest (%rest kvs))))))
;; A lazy seq is a record holding two mutable cells: a thunk (0-arg fn) and a
;; realized? flag. On first force we call the thunk, `seq` its result (chaining
;; through any further lazy layers), and cache it. This is what makes infinite
;; seqs (range/iterate/repeat/cycle) and lazy map/filter/take possible — every
;; seq walker below re-`seq`s its argument at entry, forcing lazy tails as it goes.
(defn lazy-seq? [x] (%num-eq (type-of x) 'LazySeq))
(defn -lazy-seq [thunk] (record 'LazySeq (%cell thunk) (%cell false)))
(defn -force [ls]
  (if (%cell-ref (field ls 1) 0)
      (%cell-ref (field ls 0) 0)
      (let [v (seq ((%cell-ref (field ls 0) 0)))]
        (do (%cell-set! (field ls 0) 0 v)
            (%cell-set! (field ls 1) 0 true)
            v))))
(defmacro lazy-seq (& body) (list '-lazy-seq (%cons 'fn (%cons (vector) body))))
(defn seq [c]
  (cond (nil? c) nil
        (lazy-seq? c) (-force c)
        (vector? c) (elems c)
        (set? c) (elems c)
        (map? c) (entries (field c 0))
        true c))
(defn first [c] (%first (seq c)))
(defn rest [c] (%rest (seq c)))
(defn next [c] (seq (%rest (seq c))))
(defn cons [x c] (%cons x c))
(defn empty? [c] (nil? (seq c)))
;; Fully realize a (possibly lazy) seq into an eager cons list.
(defn -to-list [s] (let [s (seq s)] (if (nil? s) nil (%cons (%first s) (-to-list (%rest s))))))
(defn doall [c] (do (-to-list c) c))
(defn dorun [c] (do (-to-list c) nil))

;; ─────────────── count / nth / get ───────────────
(defn count-seq [s n] (let [s (seq s)] (if (nil? s) n (count-seq (%rest s) (%add n 1)))))
(defn count [c] (count-seq c 0))
(defn nth-seq [s i] (let [s (seq s)] (if (%num-eq i 0) (%first s) (nth-seq (%rest s) (%sub i 1)))))
(defn nth [c i] (nth-seq c i))
(defn mget [kvs k]
  (cond (nil? kvs) nil
        (%num-eq (%first kvs) k) (%first (%rest kvs))
        true (mget (%rest (%rest kvs)) k)))
(defn get [c k] (cond (map? c) (mget (field c 0) k) (vector? c) (nth c k) true nil))
(defn kv-has [kvs k]
  (cond (nil? kvs) false (%num-eq (%first kvs) k) true true (kv-has (%rest (%rest kvs)) k)))
(defn contains? [c k] (if (map? c) (kv-has (field c 0) k) false))

;; Callable objects: keywords, maps and vectors act as functions of one arg.
;; The backend routes `(obj arg)` for a non-closure record here (see set_apply_fn).
(defn -apply-obj [o a]
  (cond (keyword? o) (get a o)
        (map? o) (get o a)
        (vector? o) (nth o a)
        true (throw "value is not callable")))

;; ─────────────── assoc / dissoc / keys / vals ───────────────
(defn kv-without [kvs k]
  (cond (nil? kvs) nil
        (%num-eq (%first kvs) k) (%rest (%rest kvs))
        true (%cons (%first kvs)
                    (%cons (%first (%rest kvs)) (kv-without (%rest (%rest kvs)) k)))))
(defn assoc [m k v] (record 'Map (%cons k (%cons v (kv-without (field m 0) k)))))
(defn dissoc [m k] (record 'Map (kv-without (field m 0) k)))
(defn kv-keys [kvs] (if (nil? kvs) nil (%cons (%first kvs) (kv-keys (%rest (%rest kvs))))))
(defn keys [m] (kv-keys (field m 0)))
(defn kv-vals [kvs] (if (nil? kvs) nil (%cons (%first (%rest kvs)) (kv-vals (%rest (%rest kvs))))))
(defn vals [m] (kv-vals (field m 0)))

;; map equality is ORDER-INDEPENDENT (unlike ordered record equality); override
;; the binary `=` to compare maps as key sets, structurally otherwise.
(defn keys-match [ks a b]
  (cond (nil? ks) true
        (-eq2 (mget (field a 0) (%first ks)) (mget (field b 0) (%first ks))) (keys-match (%rest ks) a b)
        true false))
;; Pairwise (map-aware) equality; the variadic `=`/`not=` above chain over it.
(defn -eq2 [a b]
  (if (and (map? a) (map? b))
      (if (%num-eq (count a) (count b)) (keys-match (keys a) a b) false)
      (%num-eq a b)))

;; ─────────────── conj / into ───────────────
(defn append1 [lst x] (if (nil? lst) (%cons x nil) (%cons (%first lst) (append1 (%rest lst) x))))
(defn conj [c x]
  (cond (vector? c) (record 'Vector (append1 (field c 0) x))
        (set? c) (record 'Set (%cons x (field c 0)))
        (map? c) (assoc c (nth x 0) (nth x 1))
        true (%cons x c)))
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
(defn comp [f g] (fn [x] (f (g x))))
(defn partial [f a] (fn [x] (f a x)))
(defn constantly [x] (fn [& _] x))
(defn complement [f] (fn [x] (not (f x))))
(defn max [a b] (if (%lt a b) b a))
(defn min [a b] (if (%lt a b) a b))
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
        (vector? c) (field c 0)
        (set? c) (field c 0)
        (map? c) (entries (field c 0))
        true c))
(defn -rt-first [c] (%first (-rt-seq c)))
(defn -rt-rest [c] (%rest (-rt-seq c)))
(defn -rt-next [c] (let [r (%rest (-rt-seq c))] (if (nil? r) nil r)))
(defn -rt-conj [c x]
  (cond (vector? c) (record 'Vector (append1 (field c 0) x))
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
  (cond (vector? x) (record 'Vector (field x 0) m)
        (map? x) (record 'Map (field x 0) m)
        (set? x) (record 'Set (field x 0) m)
        true x))
(defn -meta [x] (if (%lt 1 (nfields x)) (field x 1) nil))
(def with-meta -with-meta)
(def meta -meta)
(defn vec [c] (record 'Vector (-to-list c)))

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
(defn peek [c] (cond (vector? c) (last c) (nil? c) nil true (first c)))
(defn pop [c] (cond (vector? c) (vec (butlast c)) (nil? c) nil true (rest c)))
(defn not-empty [c] (if (empty? c) nil c))
(defn empty [c] (cond (vector? c) (vector) (map? c) (hash-map) (set? c) (hash-set) true nil))

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
(defn -juxt-apply [fns x] (map (fn [f] (f x)) fns))
(defn juxt [& fns] (fn [x] (vec (-juxt-apply fns x))))
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

;; ─────────────── realization (force a lazy result for printing) ───────────────
;; The Rust printer can't invoke thunks, so `run` calls this on the final value
;; to fully realize any lazy spine (and lazy elements) into eager collections.
(defn -realize-list [s]
  (let [s (seq s)] (if (nil? s) nil (%cons (-realize (%first s)) (-realize-list (%rest s))))))
(defn -realize [x]
  (cond (lazy-seq? x) (-realize-list x)
        (%num-eq (type-of x) 'List) (-realize-list x)
        (vector? x) (record 'Vector (-realize-list (field x 0)))
        (set? x) (record 'Set (-realize-list (field x 0)))
        true x))
"#;
