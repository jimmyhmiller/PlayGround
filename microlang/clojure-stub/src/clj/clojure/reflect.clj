;; `clojure.reflect` — the SUBSET tools.analyzer.jvm consumes (its jvm/utils
;; is the only real-library consumer we model): `type-reflect` answering
;; member queries, and `->JavaReflector` as an opaque :reflector option.
;;
;; The answers come from the SAME place all host semantics live: the
;; `-jvm-registry` descriptors. What the registry can enumerate today is the
;; STATIC surface (`:static` values and `:static-fn`s per class) and the
;; inheritance chain; instance methods register straight into the dispatch
;; registry under dot-names and are not yet enumerable per class, so
;; `:members` lists no instance members. That gap is LOUD at the right place:
;; the analyzer's validate pass throws its own "no matching method" when a
;; member query comes back empty for an interop form — analysis of pure
;; Clojure (every go-block body in the corpus) never asks.
;;
;; Member maps follow real clojure.reflect's shape: `:name` (symbol),
;; `:flags` (set), `:parameter-types`/`:return-type` for methods. A
;; `:static-fn` cannot report its arity (fns don't expose one), so static fns
;; carry `:parameter-types nil` — the FIELD shape — which answers
;; `static-field` probes (how the analyzer reads `Long/MAX_VALUE`-style
;; members) and honestly declines arity-filtered method matches.

(ns clojure.reflect)

(defn ->JavaReflector [_classloader] ::java-reflector)

(defn- plist-names [pl]
  (loop [pl pl acc ()]
    (if (nil? pl)
      acc
      (recur (rest (rest pl)) (cons (first pl) acc)))))

(defn- typeref->fqn
  "A typeref is a Class value or a (possibly simple) symbol."
  [typeref]
  (cond
    (class? typeref) (symbol (.getName typeref))
    (symbol? typeref) (if-let [c (clojure.core/-jvm-class-named typeref)]
                        (symbol (.getName c))
                        typeref)
    :else (throw (str "type-reflect: not a class or symbol: " (pr-str typeref)))))

(defn type-reflect
  "The registry-backed reflection answer: {:bases #{superclass-fqn} :flags
  #{:public} :members #{static members…}}. Options (:ancestors, :reflector)
  are accepted; :ancestors folds the whole `:extends` chain's statics in."
  [typeref & opts]
  (let [fqn (typeref->fqn typeref)
        member (fn [cls n ptypes]
                 {:name n
                  :declaring-class cls
                  :parameter-types ptypes
                  :return-type nil
                  :exception-types ()
                  :flags #{:public :static}})
        one (fn [fqn]
              (let [d (clojure.core/-jvm-descriptor fqn)]
                (if (nil? d)
                  {:bases #{} :members #{}}
                  {:bases (if-let [s (clojure.core/-jvm-c-extends d)] #{s} #{})
                   :members
                   (set (concat
                         (map #(member fqn % nil)
                              (plist-names (clojure.core/-jvm-c-statics d)))
                         (map #(member fqn % nil)
                              (plist-names (clojure.core/-jvm-c-static-fns d)))))})))]
    (loop [cur fqn acc {:bases #{} :flags #{:public} :members #{}} seen #{}]
      (if (or (nil? cur) (contains? seen cur))
        acc
        (let [r (one cur)]
          (recur (first (:bases r))
                 (-> acc
                     (update :bases into (:bases r))
                     (update :members into (:members r)))
                 (conj seen cur)))))))
