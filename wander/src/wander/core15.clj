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


(def ordered-code (tree-seq (fn [x] (and (seq? x) (#{'let 'if} (first x))))
                            (fn [x] (m/match x 
                                      (let [_ (m/and ?arg (if & _) )] ?body)
                                      [?arg [:body] ?body]
                                      (let [_ _] ?body)
                                      [ ?body]
                                      (if _ ?t ?f)
                                      [?t [:else] ?f]))
                            (a-normal code)))


(let [label (atom (gensym "label_"))
      stack (atom ())]
  ;; The text in all of these is backwards
  ;; Should failure be a single node?
  (reduce (fn [acc x]
            (m/match x

              ;; Ummm, not sure what to do with this one?
              ;; This is the value of the if statement,
              ;; but we aren't using that?
              ;; but we need it in other places
              (let [?x (if & _)] _)
              (update-in acc [@label :text] conj ?x) 

              (let [?x ?v] _)
              (update-in acc [@label :text] conj [?x ?v])

              (if ?pred & _)
              (do
                (let [current @label
                      next (gensym "label_")
                      else (gensym "label_")]
                  (reset! label next)
                  (swap! stack conj else)
                  (-> acc
                      (update-in [current :text] conj [:if ?pred])
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
              
              ?x (update-in acc [@label :text] conj ?x)))
          {}
          ordered-code))








