(ns wander.core15
  (:require [meander.epsilon :as m]
            [dorothy.core :as dot]
            [dorothy.jvm :as dot-jvm]
            [clojure.string :as string]
            [clojure.walk :as walk]
            [clojure.set :as set]))



(defn value? [m]
  (or (number? m)
      (boolean? m)
      (string? m)
      (symbol? m)
      (keyword? m)
      (nil? m)
      (vector? m)
      (map? m)
      (and (seq? m) (= (first m) 'quote))))


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



(defn hash-to-sym [code]
  (let [value (hash code)
        prefix_neg (if (neg? value) "neg" "")
        abs-value (Math/abs value)]
    (symbol (str "sym_" prefix_neg "_" abs-value))))


(defn a-normal [code]
  (m/rewrite code

    (defn ?name ?args ?body)
    (def ?name (fn ?name ?args (m/cata ?body)))

    (if (m/pred value? ?pred) ?t ?f)
    (if ?pred
      (m/cata ?t)
      (m/cata ?f))

    (if ?pred ?t ?f)
    (let (m/app (juxt hash-to-sym identity) (m/cata ?pred))
      (if (m/app hash-to-sym (m/cata ?pred))
        (m/cata ?t)
        (m/cata ?f)))

    (let [?x (let [?y ?z] ?body1)] ?body2)
    (m/cata (let [?y (m/cata ?z)]
              (m/cata (let [?x (m/cata ?body1)]
                        (m/cata ?body2)))))

    ;; This ensures we eliminate lets in the value position, while not infinite looping
    (let [?x (m/and ?y (m/cata (m/or (m/pred value?) ((m/not let) & _)))) ] ?body)
    (let [?x ?y] (m/cata ?body))

    (let [?x (m/pred value? ?y)] ?body)
    (let [?x ?y]
      (m/cata ?body))

    (let [?x ?y] ?body)
    (m/cata (let [?x (m/cata ?y)] (m/cata ?body)))

    (let [?x ?y . !rest ...] ?body)
    (m/cata (let [?x ?y]
              (m/cata (let [!rest ...]
                        (m/cata ?body)))))

    ((m/pred value? !args) ...)
    (!args ...)

    ((m/pred value? !values) ..1 . (m/pred (complement value?) ?not) . !rest ...)
    (m/cata (let (m/app (juxt hash-to-sym identity) (m/cata ?not))
              (m/cata (!values ... (m/app hash-to-sym (m/cata ?not)) . !rest ...))))

    ((m/pred (complement value?) ?not-value) . !rest ..1)
    (m/cata (let (m/app (juxt hash-to-sym identity) (m/cata ?not-value))
              (m/cata ((m/app hash-to-sym (m/cata ?not-value)) . !rest ...))))

    (m/pred value? ?x) ?x

    ?x (:fall ?x)
    ))

;; I need to do alpha conversion on all of these, but the variable
;; name needs to be determined by the content of the binding. But the
;; content of the binding can only be applied after we substitute
;; variables that already exist.

(m/rewrite [1 2 3]
  [1 2 3]
  (m/let ['?x 2]
    3))



(hash-to-sym '(find S__21583 '?z))

(def normal-code  (a-normal code))


(def new-normal (a-normal (read-string (slurp "src/wander/big-code-example.txt"))))


(def new-code '(nth
  (let
    [?__35238
     (m.runtime/logic-variable '?__35238)
     ?__35237
     (m.runtime/logic-variable '?__35237)
     ?__35239
     (m.runtime/logic-variable '?__35239)
     ?__35240
     (m.runtime/logic-variable '?__35240)
     ?__35241
     (m.runtime/logic-variable '?__35241)]
    ((fn*
       C__35234
       ([T__35236]
         (m.runtime/knit
           [(let
              [G__35252
               (m.runtime/call
                 vec
                 (m.runtime/call
                   (fn* ([xs] (cons (nth xs 0) (nth xs 1))))
                   (m.runtime/pair
                     ?__35237
                     (m.runtime/call
                       (fn* ([xs] (cons (nth xs 0) (nth xs 1))))
                       (m.runtime/pair
                         ?__35238
                         (m.runtime/call
                           (fn* ([xs] (cons (nth xs 0) (nth xs 1))))
                           (m.runtime/pair
                             ?__35239
                             (m.runtime/const []))))))))]
              (map
                (fn*
                  ([B__35251] (m.runtime/run-gen G__35252 B__35251)))
                (let
                  [S__35235 {:max-length 32}]
                  (if (vector? T__35236)
                    (let
                      [result__26084__auto__
                       (m.runtime/-split-at T__35236 3)]
                      (if (identical?
                            result__26084__auto__
                            m.runtime/FAIL)
                        result__26084__auto__
                        (let
                          [G__35242 result__26084__auto__]
                          (let
                            [G__35243
                             (nth G__35242 0)
                             G__35244
                             (nth G__35242 1)]
                            (let
                              [G__35245 (nth G__35243 0)]
                              (let
                                [G__35246 (nth G__35243 1)]
                                (let
                                  [G__35247 (nth G__35243 2)]
                                  (if
                                    (seq G__35244)
                                    m.runtime/FAIL
                                    (let
                                      [result__26084__auto__
                                       (m.runtime/bind-variable
                                         S__35235
                                         ?__35237
                                         G__35245)]
                                      (if
                                        (identical?
                                          result__26084__auto__
                                          m.runtime/FAIL)
                                        result__26084__auto__
                                        (let
                                          [S__35235
                                           result__26084__auto__]
                                          (let
                                            [result__26084__auto__
                                             (m.runtime/bind-variable
                                               S__35235
                                               ?__35238
                                               G__35246)]
                                            (if
                                              (identical?
                                                result__26084__auto__
                                                m.runtime/FAIL)
                                              result__26084__auto__
                                              (let
                                                [S__35235
                                                 result__26084__auto__]
                                                (let
                                                  [result__26084__auto__
                                                   (m.runtime/bind-variable
                                                     S__35235
                                                     ?__35239
                                                     G__35247)]
                                                  (if
                                                    (identical?
                                                      result__26084__auto__
                                                      m.runtime/FAIL)
                                                    result__26084__auto__
                                                    (let
                                                      [S__35235
                                                       result__26084__auto__]
                                                      (list
                                                        S__35235))))))))))))))))))
                    m.runtime/FAIL))))
            (let
              [G__35261
               (m.runtime/call
                 vec
                 (m.runtime/call
                   (fn* ([xs] (cons (nth xs 0) (nth xs 1))))
                   (m.runtime/pair
                     ?__35240
                     (m.runtime/call
                       (fn* ([xs] (cons (nth xs 0) (nth xs 1))))
                       (m.runtime/pair
                         ?__35241
                         (m.runtime/const []))))))]
              (map
                (fn*
                  ([B__35260] (m.runtime/run-gen G__35261 B__35260)))
                (let
                  [S__35235 {:max-length 32}]
                  (if (vector? T__35236)
                    (let
                      [result__26084__auto__
                       (m.runtime/-split-at T__35236 2)]
                      (if (identical?
                            result__26084__auto__
                            m.runtime/FAIL)
                        result__26084__auto__
                        (let
                          [G__35253 result__26084__auto__]
                          (let
                            [G__35254
                             (nth G__35253 0)
                             G__35255
                             (nth G__35253 1)]
                            (let
                              [G__35256 (nth G__35254 0)]
                              (let
                                [G__35257 (nth G__35254 1)]
                                (if
                                  (seq G__35255)
                                  m.runtime/FAIL
                                  (let
                                    [result__26084__auto__
                                     (m.runtime/bind-variable
                                       S__35235
                                       ?__35240
                                       G__35256)]
                                    (if
                                      (identical?
                                        result__26084__auto__
                                        m.runtime/FAIL)
                                      result__26084__auto__
                                      (let
                                        [S__35235
                                         result__26084__auto__]
                                        (let
                                          [result__26084__auto__
                                           (m.runtime/bind-variable
                                             S__35235
                                             ?__35241
                                             G__35257)]
                                          (if
                                            (identical?
                                              result__26084__auto__
                                              m.runtime/FAIL)
                                            result__26084__auto__
                                            (let
                                              [S__35235
                                               result__26084__auto__]
                                              (list
                                                S__35235))))))))))))))
                    m.runtime/FAIL))))])))
      [1 2 3]))
  0
  nil)




  )

(def new-normal (a-normal new-code))


(def deflated-code
  (m/rewrite new-normal

    (let [?x (if ?pred ?t ?f)] ?body)
    (m/cata [[:phi ?x] (if ?pred) :new-edge :placeholder :new-edge
             :pop [:nested (m/cata ?t)] :duplicate :pop-edge :swap
             :pop [:nested (m/cata ?f)] :duplicate :pop-edge
             :pop [:nested (m/cata ?body)]])

    (let [?x ?arg] (m/and ?body (m/seqable (m/not (m/or 'let 'if)) & _ )))
    (m/cata [{:left ?x :right (m/cata ?arg)} ?body])

    (let [?x ?arg] ?body)
    (m/cata [{:left ?x :right (m/cata ?arg)} & (m/cata ?body)])

    
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
                            :placeholder (let [s (gensym)] [label (cons s stack) (assoc-in labels [label :phi] s)])
                            :new-edge (let [s (gensym)] [label (cons s stack) (update-in labels [label :edges] conj s)])
                            :push-current [label (cons label stack) labels]
                            :pop-edge  [label (rest stack) (update-in labels [label :edges] conj (first stack))]
                            :duplicate (let [s (first stack)] [label (cons s stack) labels])
                            :pop [(first stack) (rest stack) labels]
                            ?x [label stack (update-in labels [label :text] #(concat % (list ?x)))])]
               #_(if (keyword? code) (println code (second result)))
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


(defn keep-removing [control-flow-graph]
  (let [dups (duplicates control-flow-graph)]
    (if (empty? dups)
      control-flow-graph
      (keep-removing (reduce remove-duplicates control-flow-graph dups)))))



(def dups-removed
  (keep-removing control-flow-graph))


(do

  (defn find-entry [control-flow-graph]
    (let [edges (set (keys control-flow-graph))
          edges-visited (set (filter identity (mapcat :edges (vals control-flow-graph))))
          entry  (set/difference edges edges-visited)]
      (assert (= 1 (count entry)) "Multiple entries, maybe dead code?")
      (first entry)))

  (defn reassemble [control-flow-graph]
    (let [entry (find-entry control-flow-graph)]
      (m/rewrite (get control-flow-graph entry)
        {:text ({:left (m/some !left) :right (m/some !right)} ... . (if ?sym)) 
         :edges (?true ?false)}
        (let [!left !right ...]
          (if ?sym
            (m/cata (m/app #(get control-flow-graph % :failed-find1) ?true))
            (m/cata (m/app #(get control-flow-graph % :failed-find2) ?false))))


        {:text ({:left (m/some !left) :right (m/some !right)} ..1) 
         :edges (?edge)}
        (let [!left !right ...]
          (m/cata (m/app #(get control-flow-graph % :failed-find3) ?edge)))

        {:text ({:left (m/some !left) :right (m/some !right)} ..1 . ?result)}
        (let [!left !right ...]
          ?result)



        ;; Bug in meander without and
        {:text (m/and ({:left (m/some !left) :right (m/some !right)} ... . (m/not (m/pred map?)) ...)
                      (_ ... . [:phi ?phi] (if ?sym))) 
         :edges (?true ?false)
         :phi (m/some ?phi-edge)}
        (let [!left !right ... .
              ?phi (if ?sym 
                     (m/cata (m/app #(get control-flow-graph % :failed-find4) ?true))
                     (m/cata (m/app #(get control-flow-graph % :failed-find5) ?false)))]
          (m/cata (m/app #(get control-flow-graph % :failed-find6) ?phi-edge)))


        {:text (?x)}
        ?x

        ?x ?x)))



  (defn reassemble2 [control-flow-graph]
    (let [entry (find-entry control-flow-graph)]
      
      (m/rewrite (get control-flow-graph entry)
        {:text ({:left (m/some !left) :right (m/some !right)} ... . (if ?sym)) 
         :edges (?true ?false)}
        (m/cata [[!left !right] ... .
                 
                 [:nested (m/cata (m/app #(get control-flow-graph % :failed-find1) ?true))]
                 [:nested (m/cata (m/app #(get control-flow-graph % :failed-find2) ?false))]])


        {:text ({:left (m/some !left) :right (m/some !right)} ..1) 
         :edges (?edge)}
        (m/cata [[!left !right] ... .
                 [:nested (m/cata (m/app #(get control-flow-graph % :failed-find3) ?edge))]])

        {:text ({:left (m/some !left) :right (m/some !right)} ..1 . ?result)}
        [[!left !right] ...]

        ;; Bug in meander without and
        {:text (m/and ({:left (m/some !left) :right (m/some !right)} ... . (m/not (m/pred map?)) ...)
                      (_ ... . [:phi ?phi] (if ?sym))) 
         :edges (?true ?false)
         :phi (m/some ?phi-edge)}
        (m/cata [[!left !right] ... .
                 [:nested
                  (m/cata (m/app #(get control-flow-graph % :failed-find4) ?true))]
                 [:nested
                  (m/cata (m/app #(get control-flow-graph % :failed-find5) ?false))]
                 [:nested
                  (m/cata (m/app #(get control-flow-graph % :failed-find6) ?phi-edge))]])


        [!start ... . [:nested [!xs ...]] & ?rest]
        (m/cata [!start ... . !xs ... & (m/cata ?rest)])

        [!start ... . [:nested ?x] & ?rest]
        (m/cata [!start ... . ?x & (m/cata ?rest)])



        {:text (?x)}
        []

        ?x ?x)))


  (reassemble2 dups-removed))


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
                                     :edges (m/and ?edges (m/or (!edges ...) nil))}}
                                   [[?edge {:label ?label 
                                            :shape ~(if ?edges "rectangle" "doublecircle") 
                                            :margin "0.3 0.3"}]
                                    . [?edge (m/app str !edges)] ...])))))
   
   "test.png" {:format :png})
  nil)



(time
 (dotimes [n 10000]
   (m/match [1 [2 1]]
     [?x [?y ?x]]
     [?x ?y])))

(time
 (dotimes [n 10000]
   (m/match [[2 1] 1]
     [[?y ?x] ?x]
     [?x ?y])))


(let
  [ret__14626__auto__
   (let
     [TARGET__118096 [1 [2 1]]]
     (if (= (count TARGET__118096) 2)
       (let
         [TARGET__118096_nth_0__
          (TARGET__118096 0)
          TARGET__118096_nth_1__
          (TARGET__118096 1)]
         (let
           [?x TARGET__118096_nth_0__]
           (if (vector? TARGET__118096_nth_1__)
             (if (= (count TARGET__118096_nth_1__) 2)
               (let
                 [TARGET__118096_nth_1___nth_0__
                  (nth TARGET__118096_nth_1__ 0)
                  TARGET__118096_nth_1___nth_1__
                  (nth TARGET__118096_nth_1__ 1)]
                 (let
                   [?y TARGET__118096_nth_1___nth_0__]
                   (if (= ?x TARGET__118096_nth_1___nth_1__)
                     [?x ?y]
                     meander.match.runtime.epsilon/FAIL)))
               meander.match.runtime.epsilon/FAIL)
             meander.match.runtime.epsilon/FAIL)))
       meander.match.runtime.epsilon/FAIL))]
  (if (meander.match.runtime.epsilon/fail? ret__14626__auto__)
    (throw (ex-info "non exhaustive pattern match" '{}))
    ret__14626__auto__))


















