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

;; ─────────────── first-class arithmetic / comparison (binary) ───────────────
(defn + [a b] (%add a b))
(defn - [a b] (%sub a b))
(defn * [a b] (%mul a b))
(defn < [a b] (%lt a b))
(defn > [a b] (%lt b a))
(defn = [a b] (%num-eq a b))
(defn inc [n] (%add n 1))
(defn dec [n] (%sub n 1))
(defn pos? [n] (%lt 0 n))
(defn neg? [n] (%lt n 0))
(defn zero? [n] (%num-eq n 0))
(defn identity [x] x)
(defn even-nn [n] (cond (%num-eq n 0) true (%num-eq n 1) false true (even-nn (%sub n 2))))
(defn even? [n] (if (%lt n 0) (even-nn (%sub 0 n)) (even-nn n)))
(defn odd? [n] (not (even? n)))

;; ─────────────── seq abstraction ───────────────
(defn elems [c] (field c 0))
(defn entries [kvs]
  (if (nil? kvs) nil
      (%cons (vector (%first kvs) (%first (%rest kvs))) (entries (%rest (%rest kvs))))))
(defn seq [c]
  (cond (nil? c) nil
        (vector? c) (elems c)
        (set? c) (elems c)
        (map? c) (entries (field c 0))
        true c))
(defn first [c] (%first (seq c)))
(defn rest [c] (%rest (seq c)))
(defn next [c] (seq (%rest (seq c))))
(defn cons [x c] (%cons x (seq c)))
(defn empty? [c] (nil? (seq c)))

;; ─────────────── count / nth / get ───────────────
(defn count-seq [s n] (if (nil? s) n (count-seq (%rest s) (%add n 1))))
(defn count [c] (count-seq (seq c) 0))
(defn nth-seq [s i] (if (%num-eq i 0) (%first s) (nth-seq (%rest s) (%sub i 1))))
(defn nth [c i] (nth-seq (seq c) i))
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
        (= (mget (field a 0) (%first ks)) (mget (field b 0) (%first ks))) (keys-match (%rest ks) a b)
        true false))
(defn = [a b]
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
(defn reduce-seq [f acc s] (if (nil? s) acc (reduce-seq f (f acc (%first s)) (%rest s))))
(defn reduce [f init c] (reduce-seq f init (seq c)))
(defn into [to from] (reduce conj to from))

;; ─────────────── higher-order seq fns ───────────────
(defn map-seq [f s] (if (nil? s) nil (%cons (f (%first s)) (map-seq f (%rest s)))))
(defn map [f c] (map-seq f (seq c)))
(defn filter-seq [f s]
  (cond (nil? s) nil
        (f (%first s)) (%cons (%first s) (filter-seq f (%rest s)))
        true (filter-seq f (%rest s))))
(defn filter [f c] (filter-seq f (seq c)))
(defn remove [f c] (filter (fn [x] (not (f x))) c))
(defn range-from [i n] (if (%lt i n) (%cons i (range-from (%add i 1) n)) nil))
(defn range [n] (range-from 0 n))
(defn concat2 [a b] (if (nil? a) b (%cons (%first a) (concat2 (%rest a) b))))
(defn concat-lists [lls] (if (nil? lls) nil (concat2 (seq (%first lls)) (concat-lists (%rest lls)))))
(defn concat [& lls] (concat-lists lls))
(defn reverse-onto [s acc] (if (nil? s) acc (reverse-onto (%rest s) (%cons (%first s) acc))))
(defn reverse [c] (reverse-onto (seq c) nil))
(defn drop-seq [n s] (if (%num-eq n 0) s (if (nil? s) nil (drop-seq (%sub n 1) (%rest s)))))
(defn drop [n c] (drop-seq n (seq c)))
(defn take-seq [n s] (if (%num-eq n 0) nil (if (nil? s) nil (%cons (%first s) (take-seq (%sub n 1) (%rest s))))))
(defn take [n c] (take-seq n (seq c)))
(defn second [c] (nth c 1))
(defn last [c] (if (nil? (%rest (seq c))) (%first (seq c)) (last (%rest (seq c)))))

;; ─────────────── more clojure.core ───────────────
(defn get-in [m ks] (if (nil? (seq ks)) m (get-in (get m (%first (seq ks))) (%rest (seq ks)))))
(defn assoc-in [m ks v]
  (if (nil? (%rest (seq ks)))
      (assoc m (%first (seq ks)) v)
      (assoc m (%first (seq ks)) (assoc-in (get m (%first (seq ks))) (%rest (seq ks)) v))))
(defn update [m k f] (assoc m k (f (get m k))))
(defn some-seq [pred s] (if (nil? s) nil (if (pred (%first s)) (%first s) (some-seq pred (%rest s)))))
(defn some [pred c] (some-seq pred (seq c)))
(defn every-seq [pred s] (if (nil? s) true (if (pred (%first s)) (every-seq pred (%rest s)) false)))
(defn every? [pred c] (every-seq pred (seq c)))
(defn mapv [f c] (vec (map f c)))
(defn filterv [f c] (vec (filter f c)))
(defn comp [f g] (fn [x] (f (g x))))
(defn partial [f a] (fn [x] (f a x)))
(defn constantly [x] (fn [& _] x))
(defn complement [f] (fn [x] (not (f x))))
(defn max [a b] (if (%lt a b) b a))
(defn min [a b] (if (%lt a b) a b))
(defn map-indexed-h [f i s] (if (nil? s) nil (%cons (f i (%first s)) (map-indexed-h f (%add i 1) (%rest s)))))
(defn map-indexed [f c] (map-indexed-h f 0 (seq c)))

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
(defn vec [c] (record 'Vector (seq c)))

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
"#;
