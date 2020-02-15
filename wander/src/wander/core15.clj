(ns wander.core15
  (:require [meander.epsilon :as m]
            [dorothy.core :as dot]
            [dorothy.jvm :as dot-jvm]
            [clojure.string :as string]
            [clojure.walk :as walk]))



(defn value? [m]
  (or (number? m)
      (boolean? m)
      (string? m)
      (symbol? m)
      (keyword? m)
      (vector? m)
      (map? m)))


(def normalize)
(def normalize-name)
(def normalize-name*)

(defn normalize-name [m k]
  (normalize
   m
   (fn p [n]
     (if (value? n)
       (k n)
       (let [t (gensym)]
         `(~'let [~t ~n] ~(k t)))))))


(defn normalize-name* [m* k]
  (if (empty? m*)
    (k '())
    (normalize-name
     (first m*)
     (fn [t]
       (normalize-name*
        (rest m*)
        (fn [t*]
          (k `(~t ~@t*))))))))


(defn normalize
  ([m]
   (normalize m identity))
  ([m k]
   (m/match m
     (fn ?args ?body)
     (k `(~'fn ~?args ~(normalize ?body)))

     (let [?x ?y] ?body)
     (normalize
      ?y
      (fn [n]
        `(~'let [~?x ~n]
           ~(normalize ?body k))))

     (let [?x ?y & ?more] ?body)
     (normalize `(~'let [~?x ~?y]
                   (~'let [~@?more]
                     ~?body))
                k)

     (if ?pred ?t ?f)
     (normalize-name
      ?pred
      (fn [t] (k `(if ~t
                    ~(normalize ?t)
                    ~(normalize ?f)))))

     (?f & ?args)
     (normalize-name
      ?f
      (fn [t]
        (normalize-name*
         ?args
         (fn [t*]
           (k `(~t ~@t*))))))
     (m/pred value?) (k m)

     _ (throw (ex-info "" {:m m})))))

(normalize

 '(let
      [x [1 2 3]]
    (let
        [ret__14025__auto__
         (if (vector? x)
           (if (= (count x) 3)
             (let
                 [x_nth_0__
                  (nth x 0)
                  x_nth_1__
                  (nth x 1)
                  x_nth_2__
                  (nth x 2)]
               (let
                   [?x x_nth_0__]
                 (let [?y x_nth_1__] (let [?z x_nth_2__] [?x ?y ?z]))))
             meander.match.runtime.epsilon/FAIL)
           meander.match.runtime.epsilon/FAIL)]
      (if (meander.match.runtime.epsilon/fail? ret__14025__auto__)
        (throw (ex-info "non exhaustive pattern match" '{}))
        ret__14025__auto__))))




;; http://matt.might.net/articles/a-normalization/




(def code '(let
               [x [1 2 3]]
             (let
                 [ret__14025__auto__
                  (if (vector? x)
                    (if (= (count x) 3)
                      (let
                          [x_nth_0__
                           (nth x 0)
                           x_nth_1__
                           (nth x 1)
                           x_nth_2__
                           (nth x 2)]
                        (let
                            [?x x_nth_0__]
                          (let [?y x_nth_1__] (let [?z x_nth_2__] [?x ?y ?z]))))
                      meander.match.runtime.epsilon/FAIL)
                    meander.match.runtime.epsilon/FAIL)]
               (if (meander.match.runtime.epsilon/fail? ret__14025__auto__)
                 (throw (ex-info "non exhaustive pattern match" '{}))
                 ret__14025__auto__))))


(defn a-normal [code]
  (m/rewrite code

    (defn ?name ?args ?body)
    (def ?name (fn ?name ?args (m/cata ?body)))

    (if (m/pred value? ?pred) ?t ?f)
    (if ?pred
      (m/cata ?t)
      (m/cata ?f))

    (m/let [?pred-sym (gensym)]
      (if ?pred ?t ?f))
    (let [?pred-sym (m/cata ?pred)]
      (if ?pred-sym
        (m/cata ?t)
        (m/cata ?f)))

    (let [?x (let [?y ?z] ?body1)] ?body2)
    (m/cata (let [?y (m/cata ?z)]
              (m/cata (let [?x (m/cata ?body1)]
                        (m/cata ?body2)))))

    ;; This ensures we eliminate lets in the value position, while not infinite looping
    (let [?x (m/and ?y (m/cata (m/or (m/pred value?) ((m/not let) & _)))) ] ?body)
    (let [?x ?y] (m/cata ?body))

    (let [?x ?y] ?body)
    (m/cata (let [?x (m/cata ?y)] (m/cata ?body)))

    (let [?x ?y . !rest ...] ?body)
    (m/cata (let [?x ?y]
              (m/cata (let [!rest ...]
                        (m/cata ?body)))))

    ((m/pred value? !args) ...)
    (!args ...)

    (m/let [?sym (gensym)]
      ((m/pred value? !values) ..1 . (m/pred (complement value?) ?not) . !rest ...))
    (m/cata (let [?sym (m/cata ?not)]
              (m/cata (!values ... ?sym . !rest ...))))

    (m/let [?sym (gensym)]
      ((m/pred (complement value?) ?not-value) . !rest ..1))
    (m/cata (let [?sym (m/cata ?not-value)]
              (m/cata (?sym . !rest ...))))

    (m/pred value? ?x) ?x

    ?x (:fall ?x)
    ))


(def normal-code  (a-normal code))


(def new-normal (a-normal ))



(def deflated-code
  (m/rewrite new-normal

    (let [?x (if ?pred ?t ?f)] ?body)
    (m/cata [?x (if ?pred) :new-edge :placeholder :new-edge
             :pop [:nested (m/cata ?t)] :duplicate :pop-edge :swap
             :pop [:nested (m/cata ?f)] :duplicate :pop-edge
             :pop [:nested (m/cata ?body)]])

    (let [?x ?arg] (m/and ?body (m/seqable (m/not (m/or 'let 'if)) & _ )))
    (m/cata [(m/app str ?x " = " (m/cata ?arg)) ?body])

    (let [?x ?arg] ?body)
    (m/cata [(m/app str ?x " = " (m/cata ?arg)) & (m/cata ?body)])

    
    (if ?pred ?t ?f)
    (m/cata [(if ?pred) :new-edge :new-edge
             :pop [:nested (m/cata ?t)]
             :pop [:nested (m/cata ?f)]])

    [!start ... . [:nested [!xs ...]] & ?rest]
    (m/cata [!start ... . !xs ... & (m/cata ?rest)])

    [!start ... . [:nested ?x] & ?rest]
    (m/cata [!start ... . ?x & (m/cata ?rest)])


    ?x  ?x
    ))

(def control-flow-graph
  (last
   (reduce (fn [[label stack labels] code]
             ;; this is super ugly, fix
             (let [result (m/match code
                            :swap [label (cons (second stack) (cons (first stack) (rest (rest stack)))) labels]
                            :placeholder (let [s (gensym)] [label (cons s stack) labels])
                            :new-edge (let [s (gensym)] [label (cons s stack) (update-in labels [label :edges] conj s)])
                            :push-current [label (cons label stack) labels]
                            :pop-edge  [label (rest stack) (update-in labels [label :edges] conj (first stack))]
                            :duplicate (let [s (first stack)] [label (cons s stack) labels])
                            :pop [(first stack) (rest stack) labels]
                            ?x [label stack (update-in labels [label :text] #(concat % (list ?x)))])]
               (if (keyword? code) (println code (second result)))
               result))
           (let [s (gensym)] [s '() {}])
           deflated-code)))


;; Need to find all duplicate exits/nodes and replace them.
;; Then I can simplify things a lot.
;; Also, need to go backwards from cfg to anf


;; (solve '(1 2 3 4 5) (?x ?y (meander.zeta/or ?x ?z) ?q ?r))

(defn duplicates [control-flow-graph]
  (map second
       (filter (fn [x] (> (count (second x)) 1))
               (group-by second control-flow-graph))))


(defn update-edges [edge-to-find new-edge control-flow-graph]
  (walk/postwalk (fn [x]
                  (if (= x edge-to-find)
                    new-edge
                    x))
                 control-flow-graph))

(defn remove-duplicates [control-flow-graph duplicate-collection]
  (let [new-state (gensym)
        text (second (first duplicate-collection))]
    (reduce (fn [cfg [id _]]
              (update-edges id new-state (dissoc cfg id))) 
            (assoc control-flow-graph new-state text) 
            duplicate-collection)))



(def dups-removed
  (reduce remove-duplicates control-flow-graph  (duplicates control-flow-graph)))

(defn str-even-lazy [x]
  (if (seq? x)
    (str (into [] x))
    (str x)))

(do
  (dot-jvm/save!
   
   (dot/dot
    (dot/digraph (concat [{:rankdir "LR"}]
                         (mapcat identity
                                 (m/rewrites dups-removed
                                   {(m/app str ?edge)
                                    {:text (m/app
                                            #(string/join "\\l" (map str-even-lazy %)) 
                                            ?label) 
                                     :edges (m/or (!edges ...) nil)}}
                                   [[?edge {:label ?label :shape "rectangle" :margin "0.3 0.3"}]
                                    . [?edge (m/app str !edges)] ...])))))
   
   "test.png" {:format :png})
  nil)

(dot-jvm/show!
 (dot/dot (dot/digraph) 
          [
           [:a :b :c]]))






