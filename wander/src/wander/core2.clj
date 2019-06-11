(ns wander.core2
  (:require [meander.match.delta :as m]
            [meander.substitute.delta :refer [substitute]]
            [meander.syntax.delta :as syntax]
            [clojure.walk :as walk]
            [clojure.core.match :as clj-match]
            [meander.strategy.delta :as r]
            [meander.match.ir.delta :as ir]
            [clojure.walk :as walk]))



(def x 2)

(defn match-my-map [m]
  (m/match m
    {:x ~x :y ?y}
    [:okay ?y]
    
    _
    [:fail]))


(match-my-map {:x 1 :y 3})
;; =>
[:fail]

(match-my-map {:x 2 :y 3})
;;=>
[:okay 3]


(let [f (fn [x]
          (fn [z]
            (m/match z
              {:x ~x, :y ?y}
              [:okay ?y]
              _
              [:fail])))
      g (f 1)]
  [(g {:x 1 :y 2})
   (g {:x 2 :y 2})])
;; =>
[[:okay 2] [:fail]]


(defmacro bup [& body]
  `(r/until = (r/bottom-up (r/rewrite ~@body))))


(def nn
  (bup
   ('not ('not ?x))
   ?x
   
   ?x ?x))

(nn )



((r/until =
    (r/bottom-up)))
 


(def nn 
  (r/until =
    (r/bottom-up 
     (r/rewrite 
      
      ('not ('not ?x))
      ?x

      ?x ?x))))

(
 
 '(and (not (not (or 1 2)))
       (or 3 (not (not 4)))))



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


(defn match-it [x]
  (clj-match/match x
                   [[_ _ 2]] :a0
                   [[1 1 3]] :a1
                   [[1 2 3]] :a2
                   :else :a3))

(match-it [[1 2 3]])

(time
 (doseq [n (range 1000000)]
   (match-it [[1 2 3]])))

(walk/macroexpand-all
 (quote

  (clj-match/match x
                   [[_ _ 2]] :a0
                   [[1 1 3]] :a1
                   [[1 2 3]] :a2
                   :else :a3)))

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

(defn repeat-n 
  {:style/indent :defn}
  [n s]
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





(require '[meander.match.delta :as r.match]
         '[meander.match.ir.delta :as r.ir]
         '[meander.matrix.delta :as r.matrix]
         '[meander.syntax.delta :as r.syntax]
         '[meander.util.delta :as r.util]
         '[meander.util.delta :as r.util])

(defn analyze-compile
  {:style/indent :defn}
  ([patterns]
   (analyze-compile patterns :find))
  ([kind patterns]
   (let [analyzer (case kind
                    :match r.match/analyze-match-args
                    :find r.match/analyze-find-args
                    :search r.match/analyze-search-args)
         analysis (analyzer (cons 'target patterns))
         matrix (:matrix analysis)
         ir (r.match/compile ['target] matrix)
         ir* (r.ir/rewrite (r.ir/rewrite ir))
         code (r.ir/compile ir* nil kind)]
     {:matrix matrix
      :ir ir
      :ir* ir*
      :code code})))


(analyze-compile :search
  (quote
   ({?key ?value}
    (str " " (name ?key) "=\"" (name ?value) "\""))))

(analyze-compile :match
  (quote ({:preferred-address {:zip (pred some? !zips)
                               :city (pred some? !cities)}
           :other-addresses [{:zip (pred some? !zips)
                              :city (pred some? !cities)} ...]}
          {:zips (distinct !zips)
           :cities (distinct !cities)})))

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

(m/compile
 (syntax/parse

  (quote
   ({:preferred-address {:zip (pred some? !zips)
                         :city (pred some? !cities)}
     :other-addresses [{:zip (pred some? !zips)
                        :city (pred some? !cities)} ...]}
    {:zips (distinct !zips)
     :cities (distinct !cities)}))
  )
 'fail 'match)

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
  
(defn reformat-preferred-address [person]
  (m/match person
    {:preferred-address 
     {:address1 ?address1
      :address2 ?address2
      :city ?city
      :state ?state
      :zip ?zip}}
    
    {:address {:line1 ?address1
               :line2 ?address2}
     :city-info {:city ?city
                 :state ?state
                 :zipcode ?zip}}))


(defn same-zip-preferred [person]
  (let [zip (get-in person [:preferred-address :zip])]
    (filter #(= (:zip %) zip) (:other-addresses person))))



(def people
  [{:name "jimmy"
    :preferred-address
    {:address1 "123 street ave"
     :address2 "apt 2"
     :city "Townville"
     :state "IN"
     :zip "46203"}
    :other-addresses 
    [{:address1 "432 street ave"
      :address2 "apt 7"
      :city "Cityvillage"
      :state "New York"
      :zip "12345"}
     {:address1 "534 street ave"
      :address2 "apt 5"
      :city "Township"
      :state "IN"
      :zip "46203"}]}
   {:name "joel"
    :preferred-address
    {:address1 "123 street ave"
     :address2 "apt 2"
     :city "Townville"
     :state "IN"
     :zip "46203"}
    :other-addresses 
    [{:address1 "432 street ave"
      :address2 "apt 7"
      :city "Cityvillage"
      :state "New York"
      :zip "12345"}
     {:address1 "534 street ave"
      :address2 "apt 5"
      :city "Township"
      :state "IN"
      :zip "46203"}]}])


(def person
  {:name "jimmy"
   :preferred-address 
   {:address1 "123 street ave"
    :address2 "apt 2"
    :city "Townville"
    :state "IN"
    :zip "46203"}
   :other-addresses 
   [{:address1 "432 street ave"
     :address2 "apt 7"
     :city "Cityvillage"
     :state "New York"
     :zip "12345"}
    {:address1 "534 street ave"
     :address2 "apt 5"
     :city "Townville"
     :state "IN"
     :zip "46203"}
    {:address1 "123 street ave"
     :address2 "apt 2"
     :city "Township"
     :state "IN"
     :zip "46203"}]})


(defn distinct-zips-and-cities [person]
  (let [preferred-address-zip (get-in person [:preferred-address :zip])
        preferred-address-city (get-in person [:preferred-address :city])
        other-zips (map :zip (:other-addresses person))
        other-cities (map :city (:other-addresses person))]
    {:zips (filter some? (distinct (cons preferred-address-zip other-zips)))
     :cities (filter some? (distinct (cons preferred-address-city other-cities)))}))

(defn distinct-zips-and-cities [person]
  (let [zip (get-in person [:preferred-address :zip])
        city (get-in person [:preferred-address :city])
        addresses (:other-addresses person)]
    (reduce (fn [acc address] 
              (assoc acc 
                     :zips (conj (:zips acc) (:zip address))
                     :cities (conj (:cities acc) (:city address)))) 
            {:zips [zip]
             :cities []}
            addresses)))


(add-watch #'distinct-zips-and-cities-mm :pref 
           (fn [_ _ _ _]
             (time
              (doseq [n (range 100000)]
                (distinct-zips-and-cities-mm person))) ))


(defn distinct-zips-and-cities [person]
  (let [addresses (conj (:other-addresses person) 
                        (:preferred-address person))]
    {:zips (filter some? (distinct (map :zip addresses)))
     :cities (filter some? (distinct (map :city addresses)))}))


(macroexpand
 (quote
  (m/match person
    {:preferred-address {:zip (pred some? !zips)
                         :city (pred some? !cities)}
     :other-addresses [{:zip (pred some? !zips)
                        :city (pred some? !cities)} ...]}
    {:zips (distinct !zips)
     :cities (distinct !cities)})))

(defn same-zip-preferred [person]
  )


(def zip "86753")
(defn find-people-with-zip [people zip]
  (->> (for [person people]
         (for [address (:addresses person)
               :when (= (:zip address) zip)]
           {:name (:name person)
            :address address}))
       (mapcat identity)))


(defn person-with-address-comb [person]
  (map (fn [address]
         {:name (:name person)
          :address address})
       (:addresses person)))

(defn find-people-with-zip [people zip]
  (->> people
       (mapcat person-with-address-comb)
       (filter (comp #{zip} :zip :address))))

(mapcat (fn [person]
          (let [matching-addresses (filter (comp #{zip} :zip) (:addresses person))]
            (map (fn [address name]
                   {:name name
                    :address address})
                 matching-addresses
                 (repeat (:name person))))) 
        people)

(defn find-people-with-zip [people zip]
  (m/search people
    (scan {:name ?name
           :addresses (scan {:zip ~zip :as ?address})})
    {:name ?name
     :address ?address}))



(def people
  [{:name "jimmy"
    :addresses [{:address1 "123 street ave"
                 :address2 "apt 2"
                 :city "Townville"
                 :state "IN"
                 :zip "46203"
                 :preferred true}
                {:address1 "534 street ave",
                 :address2 "apt 5",
                 :city "Township",
                 :state "IN",
                 :zip "46203"
                 :preferred false}
                {:address1 "543 Other St",
                 :address2 "apt 50",
                 :city "Town",
                 :state "CA",
                 :zip "86753"
                 :preferred false}]}
   {:name "joel"
    :addresses [{:address1 "2026 park ave"
                 :address2 "apt 200"
                 :city "Town"
                 :state "CA"
                 :zip "86753"
                 :preferred true}]}])



(def data
  {:people 
   [{:name "jimmy" :id 1}
    {:name "joel" :id 2}
    {:name "tim" :id 3}]
   :addresses
   {1 [{:address1 "123 street ave"
        :address2 "apt 2"
        :city "Townville"
        :state "IN"
        :zip "46203"
        :preferred true}
       {:address1 "534 street ave",
        :address2 "apt 5",
        :city "Township",
        :state "IN",
        :zip "46203"
        :preferred false}]
    2 [{:address1 "2026 park ave"
        :address2 "apt 200"
        :city "Town"
        :state "CA"
        :zip "86753"
        :preferred true}]
    3 [{:address1 "1448 street st"
        :address2 "apt 1"
        :city "City"
        :state "WA"
        :zip "92456"
        :preferred true}]}
   :visits {1 [{:date "12-31-1900"
                :geo-location {:zip "46203"}}]
            2 [{:date "1-1-1970"
                :geo-location {:zip "12345"}}
               {:date "1-1-1970"
                :geo-location {:zip "86753"}}]
            3 [{:date "4-4-4444"
                :geo-location {:zip "54221"}}
               {:date "4-4-4444"
                :geo-location {:zip "92456"}}]}})



(def addresses-by-person-id
  )


(m/search data
  {:people (scan {:id ?id :name ?name})
   :addresses {?id (scan {:preferred true :zip ?zip})}
   :visits {?id (scan {:geo-location {:zip (and (not ?zip) ?bad-zip)}
                       :date ?date})}}
  {:name ?name
   :id ?id
   :zip ?bad-zip
   :date ?date})

(defn extract-zips [{:keys [people addresses visits]}]
  (->> people
       (map (comp addresses :id))) )


(def person
  {:name "jimmy"
   :preferred-address 
   {:address1 "123 street ave"
    :address2 "apt 2"
    :city "Townville"
    :state "IN"
    :zip "46203"}
   :other-addresses 
   [{:address1 "432 street ave"
     :address2 "apt 7"
     :city "Cityvillage"
     :state "New York"
     :zip "12345"}
    {:address1 "534 street ave"
     :address2 "apt 5"
     :city "Townville"
     :state "IN"
     :zip "46203"}
    {:address1 "123 street ave"
     :address2 "apt 2"
     :city "Township"
     :state "IN"
     :zip "46203"}]})


(defn distinct-zip-codes [person])

(defn distinct-zips-and-cities-m
  [person]
  
  (m/match person
    {:preferred-address {:zip (pred some? !zips)
                         :city (pred some? !cities)}
     :other-addresses [{:zip (pred some? !zips)
                        :city (pred some? !cities)} ...]}
    {:zips (distinct !zips)
     :cities (distinct !cities)}))


(m/search person
  {:preferred-address {:zip ?zip}
   :other-addresses (scan {:zip ?zip :as ?address})}
  ?address)

(defn reformat-preferred-address [person]
  (let [address (:preferred-address person)]
    {:address {:line1 (:address1 person)
               :line2 (:address2 person)}
     :city-info {:city (:city address)
                 :state (:state address)
                 :zipcode (:zip address)}}))












(defn power [b n]
  (if (zero? n)
    1.0
    (* b (power b (dec n)))))

(power 2 3)





;; http://scala-lms.github.io/tutorials/02_basics.html#sec220:evalOrder
;; crazed hacked together, but poc

(def env (atom []))

(defn perform [stm]
  [:lms/perform stm])

(defn accumulate [res]
  (if (and (vector? res) (= (first res) :lms/perform))
    (let [x (gensym "x")]
      (swap! env  conj x (second res))
      (accumulate x))
    res))



(defn compute [x]
  x)

(defn power' [b n]
  (if (zero? n)
    (perform 1.0)
    (perform `(* ~(accumulate b) ~(accumulate (power' b (dec n)))))))

(defmacro attempt-stage [expr]
  (reset! env [])
  (let [result# (eval expr)]
    `(let ~(deref env)
       ~(second result#))))

(macroexpand
 (quote
  (attempt-stage
   (let [x (accumulate (perform '(compute 1)))]
     (power' (accumulate (perform `(+ (compute 2) ~(accumulate x)))) 5)))))

(macroexpand
 (quote
  (attempt-stage
   (power' 2 6))))


(def person
  {:name "jimmy"
   :preferred-address
   {:address1 "123 street ave"
    :address2 "apt 2"
    :city "Townville"
    :state "IN"
    :zip "46203"}
   :other-addresses 
   [{:address1 "432 street ave"
     :address2 "apt 7"
     :city "Cityvillage"
     :state "New York"
     :zip nil}
    {:address1 "534 street ave"
     :address2 "apt 5"
     :city "Township"
     :state "IN"
     :zip "46203"}]})



z(defn distinct-zips-and-cities [person]
  (m/match person
    {:preferred-address {:zip (or nil !zips)
                         :city (or nil !cities)}
     :other-addresses [{:zip (or nil !zips)
                        :city (or nil !cities)} ...]}
    {:zips (distinct !zips)
     :cities (distinct !cities)}))


(defn distinct-zip-codes [person]
  (m/match person
    {:preferred-address {:zip (or nil !zips)}
     :other-addresses [{:zip (or nil !zips)} ...]}
    (distinct !zips)))


(distinct-zips-and-cities person)

(def people
  [{:name "jimmy"
    :addresses [{:address1 "123 street ave"
                 :address2 "apt 2"
                 :city "Townville"
                 :state "IN"
                 :zip "46203"
                 :preferred true}
                {:address1 "534 street ave",
                 :address2 "apt 5",
                 :city "Township",
                 :state "IN",
                 :zip "46203"
                 :preferred false}
                {:address1 "543 Other St",
                 :address2 "apt 50",
                 :city "Town",
                 :state "CA",
                 :zip "86753"
                 :preferred false}]}
   {:name "joel"
    :addresses [{:address1 "2026 park ave"
                 :address2 "apt 200"
                 :city "Town"
                 :state "CA"
                 :zip "86753"
                 :preferred true}]}])


(defn find-people-with-zip [people zip]
  (for [person people
        address (:addresses person)
        :when (= (:zip address) zip)]
    {:name (:name person)
     :address address}))




(m/match [0 [:increment]]
 [?n [:increment]] (inc ?n)
 [?n [:decrement]] (dec ?n))



(find-people-with-zip people "46203")


(meander.match.ir.delta/run-star-vec
 val__37382
 [!zips !cities]
 1
 (clojure.core/fn
   [[!zips !cities] input__37387]
   (clojure.core/let
       [nth_0__37388 (clojure.core/nth input__37387 0)]
     (if
         (clojure.core/map? nth_0__37388)
       (clojure.core/let
           [val__37389
            (clojure.core/get nth_0__37388 :zip)]
         (if
             (some? val__37389)
           (clojure.core/letfn
               [(save__37390 [] meander.match.ir.delta/FAIL)]
             (clojure.core/let
                 [!zips (clojure.core/conj !zips val__37389)]
               (clojure.core/let
                   [val__37391
                    (clojure.core/get nth_0__37388 :city)]
                 (if
                     (some? val__37391)
                   (clojure.core/letfn
                       [(save__37392 [] (save__37390))]
                     (clojure.core/let
                         [!cities
                          (clojure.core/conj !cities val__37391)]
                       [!zips !cities]))
                   (save__37390)))))
           meander.match.ir.delta/FAIL))
       meander.match.ir.delta/FAIL)))
 (clojure.core/fn
   [[!zips !cities]]
   {:zips (distinct !zips),
    :cities (distinct !cities)}))

(meta #'r/n-times)


(defn n-times
  {:style/indent :defn}
  [n s]
  (apply r/pipe (clojure.core/repeat n s)))

(def simplify-addition
  (r/until =
    (r/bottom-up
     (r/attempt
      (r/rewrite
       (+ ?x 0) ?x
       (+ 0 ?x) ?x)))))

(def simplify-addition
  (r/rewrite
   (+ ?x 0) ?x
   (+ 0 ?x) ?x))

(def simplify-addition-bu
  (r/until =
    (r/bottom-up
     (r/trace
      (r/attempt simplify-addition)))))

(def simplify-addition-td
  (r/until =
    (r/trace
     (r/top-down
      (r/attempt simplify-addition)))))

(do
  (println "\n\n\n\n")
  (simplify-addition-bu '(+ (+ 0 3) 0)))






(do
  (println "\n\n\n\n")
  (simplify-addition '(+ 0 2)))

(def simplify-twice
  (r/n-times 2 simplify-addition))

(simplify-twice )

(simplify-addition '(+ 0 3)) ;; 3
(simplify-addition '(+ 3 0)) ;; 3

(simplify-addition
 (simplify-addition '(+ 0 (+ 0 3))))



(simplify-twice '(+ 0 3))


(def find-x
  (r/rewrite
   [?x] ?x
   [?x ?y] ?x
   [?x ?y ?z] ?x))

(find-x [1]) ;; 1
(find-x [1 2]) ;; 1
(find-x [1 2 3]) ;; 1


(def one-to-two
  (r/rewrite
   1 2))


(time
 (doseq [n (range 10000)]
   (m/search [:q :r :s :t :a :b :r :t :c :a :c :d :a :b :c :d :a :b]
     [_ ... :a :b . _ ... :c :d . _ ...]
     :yep)))

(defn search-1 [xs]
  (m/search xs
      [_ ... :a :b . !xs ...]
    !xs))

(defn search-2 [xs]
  (m/search xs
    [_ ... :c :d . _ ...]
    :yep))

(defn p1 [x]
  (m/match x
    [:a :b] true
    _ false))

((complement p1) [:q :r :s :t :a :b :r :t :c :a :c :d :a :b :c :d :a :b])


(take-while (complement p1)  [:q :r :s :t :a :b :r :t :c :a :c :d :a :b :c :d :a :b])


(search-1 [:q :r :s :t :a :b :r :t :c :a :c :d :a :b :c :d :a :b])

(def search-it
  (comp (partial mapcat search-2) search-1) )

(time
 (doseq [n (range 10000)]
   (search-it
    [:q :r :s :t :a :b :r :t :c :a :c :d :a :b :c :d :a :b])))


[_ ... :a :b . _ ... :c :d . _ ...]

[:q :r :s :t :a :b :r :t :c :a :c :d :a :b :c :d :a :b]

[[:a :b :r :t :c :a :c :d :a :b :c :d :a :b]
 [:a :c :d :a :b :c :d :a :b]
 [:a :b :c :d :a :b]]

[[:b :r :t :c :a :c :d :a :b :c :d :a :b]
 [:b :c :d :a :b]]

[[:c :a :c :d :a :b :c :d :a :b]
 [:c :d :a :b :c :d :a :b]
 [:c :d :a :b]
 [:c :d :a :b]]

[[:d :a :b :c :d :a :b]
 [:d :a :b]
 [:d :a :b]]



[_ ... :a :b . _ ... :c :d . _ ...]

[:q :r :s :t :a :b :r :t :c :a :c :d :a :b :c :d :a :b]

[[:r :t :c :a :c :d :a :b :c :d :a :b]
 [:c :d :a :b]]

[:c :d :a :b :c :d :a :b]
[:c :d :a :b]
[:c :d :a :b]


