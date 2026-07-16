;; Seq / rest-arg semantics probe — ONE file, run byte-identically by BOTH
;; microclj and real JVM Clojure, and diffed line-for-line:
;;
;;   ./target/release/microclj --jit clojure-stub/tests/seq_oracle/probe.clj
;;   cd /tmp && clojure -M .../clojure-stub/tests/seq_oracle/probe.clj
;;
;; Real Clojure IS the specification here, so the expected output is not
;; hand-written — it is whatever `clojure` prints. `expected.txt` is that
;; output, captured; `seq_oracle.rs` asserts microclj reproduces it exactly.
;;
;; Scope: the observable shape of rest args and seqs. `class`/`type` are
;; deliberately NOT probed — the host class names legitimately differ. What
;; must match is every PREDICATE and value a Clojure program can branch on.

(defn probe [xs]
  [(seq? xs) (list? xs) (counted? xs) (chunked-seq? xs) (sequential? xs)])

(defn show [label v] (println (str label "\t" (pr-str v))))

;; ── what a rest arg IS ────────────────────────────────────────────────
(defn v [& xs] (probe xs))
(defn v1 [a & xs] (probe xs))
(defn v2 [a b & xs] (probe xs))

(show "direct/0-extra" (v))
(show "direct/1-extra" (v 1))
(show "direct/3-extra" (v 1 2 3))
(show "direct/40-extra" (apply v (doall (map identity (range 40)))))
(show "req1/direct" (v1 :a 1 2 3))
(show "req2/direct" (v2 :a :b 1 2 3))

;; apply passes the seq THROUGH: the rest arg keeps the source's shape.
(show "apply/range-3" (apply v (range 3)))
(show "apply/range-100" (apply v (range 100)))
(show "apply/vec-100" (apply v (vec (range 100))))
(show "apply/list-100" (apply v (apply list (range 100))))
(show "apply/lazy-map" (apply v (map inc (range 100))))
(show "apply/empty-vec" (apply v []))
(show "apply/nil" (apply v nil))
(show "apply/req1-range" (apply v1 (range 100)))
(show "apply/leading+seq" (apply v 1 2 (range 10)))

;; structure sharing: apply must not copy.
(show "identical/range" (let [r (range 100)] (identical? r (apply (fn [& xs] xs) r))))
(show "identical/lazy" (let [r (map inc (range 10))] (identical? r (apply (fn [& xs] xs) r))))
(show "identical/list" (let [r (apply list (range 40))] (identical? r (apply (fn [& xs] xs) r))))

;; ── values, not just shapes ───────────────────────────────────────────
(defn vals* [& xs] [(count xs) (first xs) (nth xs 1) (last xs) (vec (take 3 xs))])
(show "vals/direct" (vals* 5 6 7 8))
(show "vals/apply-range" (apply vals* (range 50)))
(show "vals/apply-lazy" (apply vals* (map inc (range 50))))
(show "vals/leading" (apply vals* 100 (range 5)))
(show "eq/rest-vs-list" (apply (fn [& xs] (= xs '(0 1 2))) (range 3)))
(show "eq/rest-vs-vec" (apply (fn [& xs] (= xs [0 1 2])) (range 3)))
(show "rest-is-seqable" (apply (fn [& xs] (vec (rest xs))) (range 4)))
(show "empty-rest-nil" (apply (fn [& xs] [(nil? xs) (seq xs)]) []))

;; ── chunked-seq? / counted? on ordinary seqs (not just rest args) ──────
(show "chunked/range" (chunked-seq? (range 100)))
(show "chunked/seq-range" (chunked-seq? (seq (range 100))))
(show "chunked/vec-seq" (chunked-seq? (seq (vec (range 100)))))
(show "chunked/list" (chunked-seq? (apply list (range 100))))
(show "chunked/lazy-map" (chunked-seq? (map inc (range 100))))
(show "chunked/seq-lazy-map" (chunked-seq? (seq (map inc (range 100)))))
(show "chunked/cons" (chunked-seq? (cons 1 nil)))
(show "chunked/nil" (chunked-seq? nil))
(show "chunked/vector" (chunked-seq? [1 2 3]))
(show "chunked/string-seq" (chunked-seq? (seq "abc")))

(show "counted/range" (counted? (range 100)))
(show "counted/vec" (counted? [1 2 3]))
(show "counted/list" (counted? (apply list (range 40))))
(show "counted/lazy-map" (counted? (map inc (range 10))))
(show "counted/cons" (counted? (cons 1 nil)))
(show "counted/vec-seq" (counted? (seq (vec (range 100)))))

;; ── identical? is POINTER identity ────────────────────────────────────
;; Where Clojure and ClojureScript AGREE, that is the semantics and we match
;; it. Where they disagree the answer is a host artifact, and matching the JVM
;; would mean copying its accidents — so those cases are excluded on purpose:
;;   (identical? 100000 100000)  JVM false (Long cache) / CLJS true (primitive)
;;   (identical? (str "a" "b") (str "a" "b"))  JVM false / CLJS true
;; Keywords, by contrast, are interned in BOTH — so we intern them too.
(show "ident/list-vs-list" (identical? (list 1 2 3) (list 1 2 3)))
(show "ident/vec-vs-vec" (identical? [1 2] [1 2]))
(show "ident/same-obj" (let [a (list 1 2)] (identical? a a)))
(show "ident/fresh-per-call" (let [f (fn [x] (list x 2))] (identical? (f 1) (f 1))))
(show "ident/kw-literal" (identical? :a :a))
(show "ident/kw-bound" (let [k :foo] (identical? k :foo)))
(show "ident/kw-built" (identical? (keyword "ab") (keyword "ab")))
(show "ident/kw-built-vs-literal" (identical? (keyword "ab") :ab))
(show "ident/kw-ns" (identical? (keyword "n" "ab") :n/ab))
(show "ident/kw-distinct" (identical? :a :b))
(show "ident/str-built" (identical? (str "a" "b") (str "a" "b")))
(show "ident/small-int" (identical? 1 1))
(show "ident/nil" (identical? nil nil))
(show "ident/empty-list" (identical? () ()))
;; NB: no `keyword-identical?` probe — it is a ClojureScript-only fn, absent
;; from JVM clojure.core, so it cannot be oracled here. (It crashed the run and
;; silently truncated expected.txt until refresh.sh learned to fail loudly.)

;; ── laziness must survive apply ───────────────────────────────────────
;; If apply forced/copied the seq, the counter would read 100, not 0/3.
(def side (atom 0))
(defn tracked [] (map (fn [x] (swap! side inc) x) (range 100)))
(reset! side 0)
(def held (apply (fn [& xs] xs) (tracked)))
(show "lazy/unforced-after-apply" (< @side 100))
(show "lazy/forces-on-demand" (do (first held) (> @side 0)))
