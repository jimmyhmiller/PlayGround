;; Seq / rest-arg / identity probe — ONE file, run byte-identically by THREE
;; runtimes and diffed line-for-line:
;;
;;   ./target/release/microclj --jit clojure-stub/tests/seq_oracle/probe.clj
;;   cd /tmp && clojure -M .../probe.clj                             -> expected.txt
;;   cd /tmp && clojure -Sdeps '{:deps {org.clojure/clojurescript
;;                {:mvn/version "1.11.132"}}}' -M -m cljs.main -re node -O none
;;                -i .../probe.clj                                   -> expected_cljs.txt
;;
;; (`seq_oracle/refresh.sh` does the latter two.) The specs are NOT hand-written
;; — they are what those implementations actually print.
;;
;; TWO specs because the rule is "match what Clojure and ClojureScript AGREE on;
;; where they disagree it is a host artifact, not the language". See
;; ../seq_oracle.rs. This is why nothing here probes `class`/`type`: host class
;; names legitimately differ. What must match is every PREDICATE and value a
;; program can branch on.

(defn show [label v] (println (str label "\t" (pr-str v))))

;; ONE PREDICATE PER LINE, deliberately. These get classified against BOTH
;; Clojure and ClojureScript, and the two hosts can legitimately disagree about
;; one predicate of a value while agreeing about another: on `(apply f 1 2
;; coll)` they disagree about `list?` (a host artifact) but AGREE that
;; `counted?` is false. Bundling the five into one vector made that line
;; unclassifiable — and hid a real microclj bug behind the artifact.
(defn probe [label xs]
  (show (str label "/seq?") (seq? xs))
  (show (str label "/list?") (list? xs))
  (show (str label "/counted?") (counted? xs))
  (show (str label "/chunked?") (chunked-seq? xs))
  (show (str label "/sequential?") (sequential? xs)))

;; ── what a rest arg IS ────────────────────────────────────────────────
(defn v [lbl & xs] (probe lbl xs))
(defn v1 [lbl a & xs] (probe lbl xs))
(defn v2 [lbl a b & xs] (probe lbl xs))

(v "direct/0-extra")
(v "direct/1-extra" 1)
(v "direct/3-extra" 1 2 3)
(apply v "direct/40-extra" (doall (map identity (range 40))))
(v1 "req1/direct" :a 1 2 3)
(v2 "req2/direct" :a :b 1 2 3)

;; apply passes the seq THROUGH: the rest arg keeps the source's shape.
(apply v "apply/range-3" (range 3))
(apply v "apply/range-100" (range 100))
(apply v "apply/vec-100" (vec (range 100)))
(apply v "apply/list-100" (apply list (range 100)))
(apply v "apply/lazy-map" (map inc (range 100)))
(apply v "apply/empty-vec" [])
(apply v "apply/nil" nil)
(apply v1 "apply/req1-range" (range 100))
(apply v "apply/leading+seq" 1 2 (range 10))

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
