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
