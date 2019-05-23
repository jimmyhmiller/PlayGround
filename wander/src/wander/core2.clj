(ns wander.core2
  (:require [meander.match.delta :as m]
            [meander.substitute.delta :refer [substitute]]
            [meander.syntax.delta :as syntax]
            [clojure.walk :as walk]
            [clojure.core.match :as clj-match]
            [meander.strategy.delta :as r]
            [meander.match.ir.delta :as ir]
            [clojure.walk :as walk]))




(walk/macroexpand-all
 (quote
  (let [x true
        y true
        z true]
    (clj-match/match [x y z]
                     [_ false true] 1
                     [false true _ ] 2
                     [_ _ false] 3
                     [_ _ true] 4
                     :else 5))))


(walk/macroexpand-all
 (quote
  (let [x true
        y true
        z true]
    (m/match [x y z]
      [_ false true] 1
      [false true _ ] 2
      [_ _ false] 3
      [_ _ true] 4
      _ 5))))


(time
 (doseq [n (range 1000000)]
   (let [x [1 2 2]]
     (clj-match/match [x]
                      [[_ _ 2]] :a0
                      [[1 1 3]] :a1
                      [[1 2 3]] :a2
                      :else :a3))))

(walk/macroexpand-all
 (quote

  (let [x [1 2 3]]
    (clj-match/match [x]
                     [[_ _ 2]] :a0
                     [[1 1 3]] :a1
                     [[1 2 3]] :a2
                     :else :a3))))

(time
 (doseq [n (range 1000000)]
   (let [x [1 2 3]]
     (m/match [x]
       [[_ _ 2]] :a0
       [[1 1 3]] :a1
       [[1 2 3]] :a2
       _ :a3))))





(defmacro bup [& body]
  `(r/until = (r/bottom-up (r/trace (r/rewrite ~@body)))))

(defmacro td [& body]
  `(r/until = (r/top-down (r/trace (r/rewrite ~@body)))))



((bup
  (+ 0 . !xs ...)
  (+ . !xs ...))
 '(+ 0 1 2))

(def nn
  (bup
    ('not ('not ?x))
    ?x

    2
    (not (not 3))
    
    ?x ?x))

(def cond-elim
  (bup
   (cond ?pred ?result
         :else ?else)
   (if ?pred ?result ?else)

   (cond ?pred ?result
         . !preds !results ...)
   (if ?pred ?result
       (cond . !preds !results ...))
   
   ?x ?x))


(def thread-first
  (r/pipe
   (r/rewrite 
    (with [%inner ((and ?f (not ->)) ?x . !xs ...)
           %recur (or (!outer (pred seq? %recur)) %inner)]
          %recur)
    (-> ?x (?f . !xs ...) . !outer ...))
   (r/bottom-up
    (r/attempt
     (r/rewrite 
      (?f)
      ?f)))))


(thread-first '(h (g (f x))))

(def thread-last
  (r/pipe
   (r/rewrite 
    (with [%inner ((and ?f (not ->)) . !xs ... . ?x)
           %recur (or (!outer ... . (pred seq? %recur)) %inner)]
          %recur)
    (->> (?f . !xs ... ?x) . !outer ...))
   (r/bottom-up
    (r/attempt
     (r/rewrite 
      (?f)
      ?f)))))


(thread-last '(map (partial + 2) (filter even? (range 10))))


(do
  (def thread-last
    (r/until =
      (r/rewrite

       ((and ?f (not ->>)) . !xs ... . ?x)
       (->> ?x (?f . !xs ..1))

       (->> (?f . !xs ..1 . ?x) . !ys ...)
       (->> ?x (?f . !xs ...) . !ys ...))))
  
  (thread-last '(map (partial + 2) (filter even? (range 10)))))




((r/until =
   (r/trace
    (r/rewrite

     ((and (not ->) ?f) ?x)
     (-> ?x ?f )
     
     ((and (not ->) ?f) ?x . !xs ...)
     (-> ?x (?f . !xs ...))
     
     (-> (?f ?x) . !ys ...)
     (-> ?x ?f . !ys ...)

     (-> (?f ?x . !xs ...) . !ys ...)
     (-> ?x (?f . !xs ...) . !ys ...))))
  '(h (g (f x y z))))

(-> x (f y z) g h)

(defmacro bup [& body]
  `(r/until = (r/bottom-up (r/trace (r/rewrite ~@body)))))

(defn repeat-n [n s]
  (apply r/pipe 
         (clojure.core/repeat n s)))

(def unpipe-first
  (repeat-n
   10
   (r/bottom-up 
    (r/rewrite

     (-> ?x (?f . !args ...))
     (?f ?x . !args ...)
     
     (-> ?x ?f)
     (?f ?x)

     (-> ?x ?f . !fs ...)
     (-> (-> ?x ?f) . !fs ...)
     
     ?x ?x))))

(unpipe-first '(-> x f g h))


()


(do
  (println "\n\n\n\n\n\n")
  (def pipe-first
    (r/pipe
    
     (repeat-n
      2
      (r/trace
       (r/top-down
        
        (r/rewrite

        
         
         ((and ?f (not ->)) ?x)
         (-> ?x ?f)
         
         (-> (-> ?x ?f) ?y)
         (-> ?x ?y ?f)
         
         (-> (?f ?x . !xs ..1) . !ys ...)
         (-> ?x (?f . !xs ...) . !ys ...)
         
         ?x ?x))))

     (r/bottom-up
      (r/rewrite
       #_(-> ?x ?f)
       #_(?f ?x)

       #_(-> ?f)
       #_?f

       ?x
       ?x))))

  {:first-example (pipe-first '(h (g (f x))))
   ;; :second-example (pipe-first '(h (g (f x y))))
  ;; :third-example (pipe-first '(h (g a (f x y z))))
   })

(println "test")

(-> x (f y) g h)

(def cond-elim
  (r/until =
    (r/bottom-up
     (r/rewrite
      (cond ?pred ?result
            :else ?else)
      (if ?pred ?result ?else)

      (cond ?pred ?result
            . !preds !results ...)
      (if ?pred ?result
          (cond . !preds !results ...))
      
      ?x ?x))))

(cond-elim 
 '(cond true true
        1 1
        3 3
        4 4
        :else false))








((r/rewrite
  (+ 0 . !xs ...)
  (+ . !xs ...))
 '(+ 0 1 2))



(m/match '(+ 1 (+ 1 2))
  (with [%const (pred number? !xs)
         %expr (or (+ %expr %expr) %const)]
        %expr)
  !xs)


(defn hiccup [{:keys [title body items]}]
  [:div {:class "card"}
   [:div {:class "card-title"} title]
   [:div {:class "card-body"} body]
   [:ul {:class "card-list"}
    (for [item items]
      [:li {:key item} item])]
   [:div {:class "card-footer"}
    [:div {:class "card-actions"}
     [:button "ok"]
     [:button "cancel"]]]])


(defn create-element [& args] 
  [:create-element args])


(require '[clojure.walk :as walk])

(def parse
  (walk/macroexpand-all
   (quote
    
    (r/rewrite
     [?tag (or {:as ?attrs}) . !body ...]
     (create-element ~(name ?tag) ?attrs . !body ...)

     [?tag . !body ...]
     (create-element  ?tag {} . !body ...)

     ?x ?x
     )
    ))

)




(walk/macroexpand-all
 (quote
  (m/match (into [] (concat (into [] (mapcat identity (repeat 100 [3 8])))
                                  [1 2]))
                 [?x1 ?x2 . ?x1 ?x2 ... 1 2]
                 (and (= 3 ?x1)
                      (= 8 ?x2))

                 _
                 false)

  ))

(ir/compile
 (syntax/parse
  (quote
   (r/rewrite
    [?tag (or {:as ?attrs}) . !body ...]
    (create-element ~(name ?tag) ?attrs . !body ...)

    [?tag . !body ...]
    (create-element  ?tag {} . !body ...)

    ?x ?x
    )))
 'fail 'rewrite)

(let [test-data {:title "hello world"
                 :body "body"
                 :items (shuffle (range 10))}]
  (time
   (doseq [x (range 1000)]
     (parse (hiccup test-data)))))


(syntax/parse 
 '(with [%h1 [!tags {:as !attrs} . (and !xs %hiccup)]
         %h2 (and (let !attrs {}) [!tags . %hiccup ...])
         %h3 !xs
         %hiccup (or %h1 %h2 %h3)]
        %hiccup))

(macroexpand
 (quote
  (m/find hiccup
    (with [%h1 [!tags {:as !attrs}  . (and !xs %hiccup)]
           %h2 (and (let !attrs {}) [!tags . %hiccup ...])
           %h3 !xs
           %hiccup (or %h1 %h2 %h3)]
          %hiccup)
    (substitute [[!tags !attrs !xs] ...]))
  ))
  
