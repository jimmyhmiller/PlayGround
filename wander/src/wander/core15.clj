(ns wander.core15
  (:require [meander.epsilon :as m]))


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




(def deflated-code
  (m/rewrite normal-code
    (let [?x (if ?pred ?t ?f)] ?body)
    ;; What I need to do is to make this turn into a diamond shape
    ;; Where t and f both exit to body.
    ;; I can't think about how to do that with a stack for some reason.
    (m/cata [?x (if ?pred) :new-edge :new-edge
             :pop [:nested (m/cata ?t)] :new-edge
             :pop [:nested (m/cata ?f)] :pop-edge
             :pop [:nested (m/cata ?body)]])

    (let [?x ?arg] ?body)
    (m/cata [?x (m/cata ?arg) & (m/cata ?body)])

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

(last
 (reduce (fn [[label stack labels] code]
           (m/match code
             :new-edge (let [s (gensym)] [label (cons s stack) (update-in labels [label :edges] conj s)])
             :pop-edge  [(first stack) (rest stack) (update-in labels [label :edges] conj (first stack))]
             :duplicate (let [s (first stack)] [label (cons s stack) labels])
             :pop [(first stack) (rest stack) labels]
             ?x [label stack (update-in labels [label :text] #(concat % (list ?x)))]))
         (let [s (gensym)] [s '() {}])
         deflated-code))




(def ordered-code (tree-seq (fn [x] (and (seq? x) (#{'let 'if} (first x))))
                            (fn [x] (m/match x
                                      (let [_ (m/and ?arg (if & _) )] ?body)
                                      [?arg [:double-push] ?body [:let]]
                                      (let [_ _] ?body)
                                      [ ?body]
                                      (if _ ?t ?f)
                                      [?t [:else] ?f]))
                           normal-code))



(let [label (atom (gensym "label_"))
      stack (atom ())]
  (reduce (fn [acc x]
            (m/match x

              ;; Ummm, not sure what to do with this one?
              ;; This is the value of the if statement,
              ;; but we aren't using that?
              ;; but we need it in other places
              (let [?x (if & _)] _)
              (update-in acc [@label :text] #(concat % (list ?x)))

              (let [?x ?v] _)
              (update-in acc [@label :text] #(concat % (list [?x ?v])))

              (if ?pred & _)
              (do                (let [current @label
                      next (gensym "label_")
                      else (gensym "label_")]
                  (reset! label next)
                  (swap! stack conj else)
                  (-> acc
                      (update-in [current :text] #(concat % (list [:if ?pred])))
                      (assoc-in [current :edges] [next else]))))

              [:let]
              (let [current @label]
                (swap! stack conj current)
                acc)


              ;; What I really want is if it is a complex let, meaning
              ;; it has an if statement as a value, then I want to
              ;; ensure I set the exit condition of the split in the
              ;; if. Basically, It will make a diamond shape. This
              ;; currently does not do that.
              [:body]
              (do
                (let [prev (first @stack)]
                  (reset! label (gensym "label_"))
                  (swap! stack rest)
                  acc))


              [:else]
              (do
                (reset! label (first @stack))
                (swap! stack rest)
                acc)

              ?x (update-in acc [@label :text] #(concat % (list ?x)))))
          {}
          ordered-code))









(let
 [x [1 2 3]]
 (let
  [G__7447503 (vector? x)]
  (let
   [ret__14025__auto__
    (if
     G__7447503
     (let
      [G__7447507 (count x)]
      (let
       [G__7447505 (= G__7447507 3)]
       (if
        G__7447505
        (let
         [x_nth_0__ (nth x 0)]
         (let
          [x_nth_1__ (nth x 1)]
          (let
           [x_nth_2__ (nth x 2)]
           (let
            [?x x_nth_0__]
            (let [?y x_nth_1__] (let [?z x_nth_2__] [?x ?y ?z]))))))
        meander.match.runtime.epsilon/FAIL)))
     meander.match.runtime.epsilon/FAIL)]
   (let
    [G__7447586
     (meander.match.runtime.epsilon/fail? ret__14025__auto__)]
    (if
     G__7447586
     (let
      [G__7447591 '{}]
      (let
       [G__7447589
        (ex-info "non exhaustive pattern match" G__7447591)]
       (throw G__7447589)))
     ret__14025__auto__)))))
