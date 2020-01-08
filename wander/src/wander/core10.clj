(ns wander.core10
  (:require [meander.match.ir.epsilon :as ir]
            [meander.strategy.epsilon :as r]
            [meander.strategy.epsilon :as strat]
            [meander.epsilon :as m]
            [meander.match.epsilon :as match]
            [clojure.string :as string]))
(do
  (println "\n\n\n")

  (analyze
   (m/match [1 2 23 2]
     [(m/pred number? !xs) ...]
     !xs)

   ))




(defn do-it4 []
  (clojure.core/let
      [target__35682 (into [] (range 10)) ]
    (clojure.core/let
        [!xs (transient [])]
      (clojure.core/let
          [ret__32941__auto__
           (meander.match.runtime.epsilon/run-star-1
             target__35682
             [!xs]
             (clojure.core/fn
               [[!xs] input__35685]
               (clojure.core/let
                   [input__35685_nth_0__ (input__35685 0)]
                 (if
                     (number? input__35685_nth_0__)
                   (clojure.core/let
                       [!xs (clojure.core/conj! !xs input__35685_nth_0__)]
                     [!xs])
                   meander.match.runtime.epsilon/FAIL)))
             (clojure.core/fn [[!xs]] (persistent! !xs)))]
        (if
            (meander.match.runtime.epsilon/fail? ret__32941__auto__)
          ((clojure.core/fn
             []
             (throw
              (clojure.core/ex-info "non exhaustive pattern match"))))
          ret__32941__auto__)))))

(m/match [1 2 23 2]
  [(m/pred number? !xs) ...]
  !xs)

(analyze
   (m/match [1 2 23 2]
     [(m/some !xs) ...]
     !xs)
)

(analyze

(m/find (quote [1 2 3 4 5 6]) [!xs ..?n !ys ..?n] [!xs !ys])
(m/find (quote (1 2 3 4 5 6)) (!xs ..?n !ys ..?n) [!xs !ys])

)


(analyze
 (m/match (into [] (range 1000))
   (m/gather (m/some !xs) ...)
   !xs))

(analyze
 (m/match [1 2 2 nil 2]
   (m/seqable (m/or nil !xs) ...)
   !xs))


(analyze
 (m/match (into [] (range 10))
   [(m/or nil !xs) ...]
   !xs))

(defn do-it []
  (m/match coll
    [(m/pred number? !xs) ...]
    !xs))

(defn do-it2 []
  (filterv number? (into [] (range 10))))

(defn do-it3 []
  (m/match (into [] (range 10))
    (m/gather (m/some !xs) ...)
    !xs))

(time 
 (dotimes [x 100000000]
   (let [[!xs] [[1 2 3]]]
     !xs)))
(time 
 (dotimes [x 100000000]
   (let [!xs (nth [[1 2 3]] 0)]
     !xs)))

(def xs (vector [1 2 3]))


(m/match [1 2 2 2]
  (m/gather (m/some !xs) ...)
  !xs)

(time
 (dotimes [x 10000]
   (do-it)))


(time
 (dotimes [x 10000]
   (do-it2)))

(time
 (dotimes [x 10000]
   (do-it3)))

(time
 (dotimes [x 10000]
   (do-it4)))


(analyze
 (m/match {:foo 1, :bar 2, :baz 3}
   {:foo 1}
   true))

(analyze 
 (m/match 2
   (m/or {} _)
   :yep))


(analyze
 (m/match '(a a a a)
   (?a ..1)
   !a))



(m/search {{:thing :stuff} :stuff
          :thing ?stuff}

  {:thing ?stuff
   {:thing ?stuff} _}
  [?stuff ?thing])



(m/match '(1 1)
  (?x ~?x)
  ?x)


(m/match {[1 2 3] :hello
          :answer 3}
  {:answer ?x
   [1 2 ?x] ?thing}
  [?thing])

map?

(m/search {1 3
           2 3
           3 9}
  {!xs ?x}
  [!xs ?x])


(m/match {:me :thing :stuff {:thing {:a {:b {:c {:d {:e :me}}}}}}}
  {?thing :thing
   :stuff {:thing {:a {:b {:c {:d {:e ?thing}}}}}}}

  [?thing])

(defn favorite-food-info [user foods-by-name]
  (m/match {:user user
            :foods-by-name foods-by-name}
    {:foods-by-name {?food {:popularity ?popularity
                            :calories ?calories}}

     :user
     {:name ?name
      :favorite-food {:name ?food}}}
    {:name ?name
     :favorite {:food ?food
                :popularity ?popularity
                :calories ?calories}}))
(println "\n\n\n\n")

(favorite-food-info 
 {:name "jimmy" :favorite-food {:name :nachos}}
 {:nachos {:popularity :very
           :calories 2000}})


(m/search [1 2 [ 3 4 1]]
  (m/$ [_ ... 1])
  :yep)





(m/search '[(+ 2 3) (* 3 5 0) (- 2 3) (/ 2 0)]
  [_ ... . (m/and (m/scan 0) !steps) . !steps ... . (/ _ 0 :as !steps)]
  !steps)


(m/match []
  (m/or {} _)
  :yep)
[{:cols 
  [{:tag :vec, :prt {:tag :prt, :left 
                     {:tag :cat, :elements 
                      [{:tag :lvr, :symbol ?name}
                       {:tag :map, :as nil, 
                        :rest-map nil, 
                        :map {{:tag :lvr, :symbol ?name} 
                              {:tag :lvr, :symbol ?prop}}}]}, 
                     :right {:tag :cat, :elements []}}, :as nil}], 
  :rhs {:value {:name ?name}, :op :return}, :env #{}, :refs {}, :ref-specs {}}]


{:cols [{:tag :set, 
         :elements ({:tag :cat, :elements
                     [{:tag :lit, :value :name} 
                      {:tag :lvr, :symbol ?name}]} 
                    {:tag :cat, :elements [{:tag :lvr, :symbol ?name}
                                           {:tag :lvr, :symbol ?thing}]})} 
        {:tag :any, :symbol _}], 
 :rhs {:value {:name ?name}, :op :return}, :env #{}, :refs {}, :ref-specs {}}


[{:cols [{:tag :map, :as nil, :rest-map nil, 
          :map {{:tag :lvr, :symbol ?name} {:tag :lvr, :symbol ?prop}}}], 
  :rhs {:value {:name ?name}, :op :return}, :env #{{:tag :lvr, :symbol ?name}}, :refs {}, :ref-specs {}}]

(analyze
 (m/match ["jimmy" {"jimmy" :name}]
   [?name {?name ?prop}]
   {:name ?name}))

(analyze)
(m/match {:stuff 1
          :name "jimmy"
          "jimmy" :thing}
  {:name ?name
   ?name ?thing}
  {:name ?name
   :thing ?thing})



[{:cols [{:tag :map,
          :as nil, 
          :rest-map nil, 
          :map {{:tag :lit, :value :name} 
                {:tag :lvr, :symbol ?name}, 
                {:tag :lvr, :symbol ?name}
                {:tag :lvr, :symbol ?thing}}}], 
  :rhs {:value {:name ?name}, :op :return}, :env #{}, :refs {}, :ref-specs {}}]

(defn analyze-compile
  {:style/indent :defn}
  ([patterns]
   (analyze-compile :find patterns))
  ([kind patterns]
   (analyze-compile kind patterns 'target))
  ([kind patterns expr]
   (let [analyzer (case kind
                    :match match/analyze-match-args
                    :find match/analyze-find-args
                    :search match/analyze-search-args)
         analysis (analyzer (cons 'target patterns))
         matrix (:matrix analysis)
         clauses (:clauses analysis)
         final-clause (:final-clause analysis)
         fail `(fn [] 
                 ~(if (some? final-clause)
                    (ir/compile (match/compile ['target] [final-clause]) nil :match)
                    `(throw (ex-info "non exhaustive pattern match" ))))

         target (gensym "target__")
         ir (match/compile [target] matrix)
         ir* (ir/rewrite 
              (ir/op-bind target (ir/op-eval expr) ir))
         code (ir/compile ir* `(~fail) kind)]
     {:clauses clauses
      :matrix matrix
      :ir ir
      :ir* ir*
      :code code})))

(defmacro analyze [expr]
  (m/match expr
    (~'m/match ?expr & ?body)
    `(analyze-compile :match (quote ~?body) (quote ~?expr))
    (~'m/search ?expr & ?body)
    `(analyze-compile :search (quote ~?body) (quote ~?expr))
    (~'m/find ?expr & ?body)
    `(analyze-compile :find (quote ~?body) (quote ~?expr))))








(analyze
 (m/match false
   (m/or true false)
   true))

(analyze
 (m/match [1 2 3]
   (m/gather !xs 4)
   :okay

   _
   :fail))



[!firsts ..1 
 (m/or (m/<> "VAN" "DER") (m/<> "DE LA"))
 !lasts ...]



(m/match reddit
  {:data
   {:children 
    (m/gather {:data
               {:title !title
                :permalink !link
                :preview {:images
                          [{:source {:url !image}} & _]}}})}}

  (m/subst
    [:div {:class :container}
     .
     [:div
      [:p [:a {:href (m/app str "https://reddit.com" !link)} 
           !title]]
      [:img {:src (m/app unescape !image)}]]
     ...]))





(m/search (parse-js example)
  (m/$ (m/or
        {:type "FunctionDeclaration"
         :id {:name ?name}
         :loc ?loc}

        {:type "VariableDeclarator"
         :id {:name ?name}
         :loc ?loc
         :init {:type (m/or "FunctionExpression" 
                            "ArrowFunctionExpression")}}))
  {:name ?name
   :loc ?loc})


(def coll [[1,2], [2, 3], [1]])

(->> coll
     (filter #(>= (count %) 2))
     (mapcat (partial drop 1))
     (map (partial * 2)))

(def expr '(+ (+ 0 (+ 0 0)) 0))
(def expr '(+ (+ 0 (+ 0 3)) 0))
(def expr '(+ (+ 0 (+ 0 (+ 3 (+ 2 0)))) 0))

(def simplify-addition
  (m/match expr
    (m/with [%add (m/or (+ 0 %add)
                        (+ %add 0)
                        (+ %add %add)
                        !xs)]
      %add)
    (m/rewrite !xs
      [] 0
      [?x] ?x
      _ (+ . !xs ...))))

(= [1 2 3] '(1 2 3))


(analyze)
(m/match (list false) [false] :ok)

(analyze)
(m/match [false] (false) :ok)

(analyze)
(m/match (list false) [?x] :ok)


(case (list false)
  ([false]) :ok)

(case (list false)
)

(= (list false) [false])
(= '(1) [1])


(m/match pokemon
  {:itemTemplates (m/gather {:pokemonSettings
                             {:rarity (m/some !rarity)
                              :pokemonId !pokemon
                              :form !form
                              :stats {:as !stats}}})}

  (m/subst [{:pokemon !pokemon 
             :form !form
             :rarity !rarity
             :stats !stats} ...]))





(m/match expr
  (m/with [%addition (m/or (+ . (m/or 0 %addition) ...) 
                           !xs)]
    %0)
  (m/rewrite !xs
    [] 0
    [?x] ?x
    _ (+ . !xs ...)))






(require '[meander.epsilon :as m])

(defn favorite-food-info [foods-by-name user]
  (m/match {:user user
            :foods-by-name foods-by-name}
    {:user
     {:name ?name
      :favorite-food {:name ?food}}
     :foods-by-name {?food {:popularity ?popularity
                            :calories ?calories}}}
    {:name ?name
     :favorite {:food ?food
                :popularity ?popularity
                :calories ?calories}}))


(def foods-by-name
  {:nachos {:popularity :high
            :calories :lots}
   :smoothie {:popularity :high
              :calories :less}})






(defn favorite-foods-info [foods-by-name user]
  (m/search {:user user
             :foods-by-name foods-by-name}
    {:user
     {:name ?name
      :favorite-foods (m/scan {:name ?food})}
     :foods-by-name {?food {:popularity ?popularity
                            :calories ?calories}}}
    {:name ?name
     :favorite {:food ?food
                :popularity ?popularity
                :calories ?calories}}))

(favorite-foods-info
 foods-by-name
 {:name :alice 
  :favorite-foods [{:name :nachos} 
                   {:name :smoothie}]})



(comment
  {:http/query {:name ?name}}

  (str "Hello " ?name)



  {:http/query {:id ?id}}

  let ?user = query :db/table :users
                    :db/where {:= :id ?id}

  match ?user:
    nil {:http/status 404
         :http/body {:message "User not found."}}

    {:name ?name} {:http/status 200
                   :http/body {:greeting (str "Hello " ?name)}}


 )

(analyze  
 (m/match names
      [!firsts ... ?last]
   [!firsts ?last]))

(defn split-full-name
  [full-name]
  (let [names (string/split full-name #" ")]
    (m/match names
      [!firsts ... ?last]
      [!firsts ?last])))


(split-full-name "JOHN PAUL DOE")
[["JOHN" "PAUL"] "DOE"]   ;;; Works as expected
(split-full-name "MADONNA")
nil         ;;; Works as expected
(split-full-name "JOHN DOE")





(defn- split-full-name
  "Split a `full-name` string into a first and last name. Correctly
  deals with multi-part first names (e.g., 'John Paul') as well as
  multi-part last names (e.g., 'Von Gunten'). If there is only a
  single name (e.g., 'Sting' or 'Madonna') then return it as both the
  first and last name."
  [full-name]
  (let [names  (string/split full-name #" ")]
    (m/find names
      ;; Match a prefix that starts the last name and everything that
      ;; follows is part of the last name
      [!firsts ..1
       (m/and (m/re #"VAN|VON|DE|LA|DELA|VANDER|O'|MV|MAC|ST|ST\.") ?p) .
       !lasts ...]
      [(string/join !firsts) (str (if (= ?p "ST.") "ST" ?p)
                                  (string/join !lasts))]

      ;; no prefixes, but at least two words
      [!firsts ..1 ?last]
      [(string/join !firsts) ?last]

      ;; single name (e.g., Sting or Madonna)
      [?single]
      [?single ?single])))

(split-full-name "JOHN VAN DAM")



(def data
  [{:event-type :pictures-liked
    :user       {:kind :user
                 :id   123
                 :name "blah"}
    :data       [{:kind :picture 
                  :id   345 
                  :src  "someurl"}
                 {:kind :picture
                  :id   678
                  :src  "someurl"}]}
   
   {:event-type :comment-added
    :user       {:kind :user
                 :id   321 
                 :name "bleh"}
    :data       {:target  {:kind   :note
                           :id     876
                           :title  "foo"
                           :body   "bar"
                           :author {:kind :user
                                    :id   543
                                    :name "baz"}}
                 :comment {:kind   :comment
                           :id     987
                           :body   "comment body"
                           :author {:kind :user
                                    :id   321
                                    :name "bleh"}}}}
   ])


(m/match data
  [(m/or {:event-type :pictures-liked 
          :user 
          {:kind !kind
           :id !id
           :name !name}
          :data [(m/let [!eid (gensym)]
                   {:kind !kind
                    :id !id
                    :src !name}) ...]}

         _) ...]
  (let [?user-id (gensym)]
    (m/subst
      [[?user-id :kind !kind]
       [?user-id :id !id]
       [?user-id :name !name]

       []])))







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
     :city "Township"
     :state "IN"
     :zip "46203"}]})



(defn distinct-zips-and-cities [person]
  (m/match person
    {:preferred-address {:zip (m/or nil !zips)
                         :city (m/or nil !cities)}
     :other-addresses [{:zip (m/or nil !zips)
                        :city (m/or nil !cities)} ...]}
    {:zips (doall (distinct !zips))
     :cities (doall (distinct !cities))}))


(analyze
 (m/match person
   {:preferred-address {:zip !zips
                        :city !cities}
    :other-addresses [{:zip !zips
                       :city !cities} ...]}
   {:zips (doall (distinct !zips))
    :cities (doall (distinct !cities))}))


(defn distinct-zips-and-cities' [person]
  (let [addresses (cons (:preferred-address person) 
                        (:other-addresses person))]
    {:zips (doall (filter some? (distinct (map :zip addresses))))
     :cities (doall (filter some? (distinct (map :city addresses))))}))


(time
 (dotimes [x 10000]
   (distinct-zips-and-cities person)))


(time
 (dotimes [x 10000]
   (distinct-zips-and-cities' person)))


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



(analyze
 (m/search data
   {:people (m/scan {:id ?id :name ?name})
    :addresses {?id (m/scan {:preferred true :zip ?zip})}
    :visits {?id (m/scan {:geo-location {:zip (m/and (m/not ?zip) ?bad-zip)}
                          :date ?date})}}
   {:name ?name
    :id ?id
    :zip ?bad-zip
    :date ?date}))

(defn find-potential-bad-visits [data]
  (m/search data
    {:people (m/scan {:id ?id :name ?name})
     :addresses {?id (m/scan {:preferred true :zip ?zip})}
     :visits {?id (m/scan {:geo-location {:zip (m/and (m/not ?zip) ?bad-zip)}
                           :date ?date})}}
    {:name ?name
     :id ?id
     :zip ?bad-zip
     :date ?date}))





(defn find-potential-bad-visits'' [data]
  (clojure.core/let
 [target__34959 data val__34960 (target__34959 :people)]
 (clojure.core/let
     [result__34961  val__34960]
  (clojure.core/mapcat
   (clojure.core/fn
    [result__34961_parts__]
    (clojure.core/let
     [val__34964
      (result__34961_parts__ :id)
      ?id
      val__34964
      val__34965
      (result__34961_parts__ :name)
      ?name
      val__34965
      val__34966
      (target__34959 :addresses)
      val__34967
      (val__34966 ?id)]
     (clojure.core/let
         [result__34968  val__34967]
      (clojure.core/mapcat
       (clojure.core/fn
        [result__34968_parts__]
        (clojure.core/let
         [val__34971 (result__34968_parts__ :preferred)]
         (clojure.core/case
          val__34971
          (true)
          (clojure.core/let
           [val__34972
            (result__34968_parts__ :zip)
            ?zip
            val__34972
            val__34973
            (target__34959 :visits)
            val__34974
            (val__34973 ?id)]
           (clojure.core/let
               [result__34975  val__34974]
            (clojure.core/mapcat
             (clojure.core/fn
              [result__34975_parts__]
              (clojure.core/let
               [val__34978
                (result__34975_parts__ :geo-location)
                val__34979
                (val__34978 :zip)]
               (clojure.core/letfn
                [(save__34980 [] :fail)
                 (f__35229
                  []
                  (clojure.core/let
                   [?bad-zip val__34979]
                   (clojure.core/let
                    [val__34981
                     (result__34975_parts__ :date)
                     ?date
                     val__34981]
                    (clojure.core/list
                     {:name ?name,
                      :id ?id,
                      :zip ?bad-zip,
                      :date ?date}))))]
                (if
                 (clojure.core/= ?zip val__34979)
                 (save__34980)
                 (f__35229)))))
             result__34975)))
          :fail)))
       result__34968))))
   result__34961))))

(time
 (dotimes [x 10000]
   (find-potential-bad-visits data)))


(time
 (dotimes [x 10000]
   (find-potential-bad-visits' data)))

(time
 (dotimes [x 10000]
   (find-potential-bad-visits'' data)))




(defn find-non-matching-visits [address visits]
  (filter (comp (complement #{(:zip address)}) :zip :geo-location) visits))

(defn find-bad-visits-for-person [addresses visits person]
  (let [preferred-address (first (filter :preferred addresses))
        non-matching (find-non-matching-visits preferred-address visits)]
    (map (fn [visit] {:name (:name person)
                      :id (:id person)
                      :zip (get-in visit [:geo-location :zip])
                      :date (:date visit)})
        non-matching)))

(defn find-potential-bad-visits' [{:keys [addresses visits people]}]
  (mapcat (fn [{:keys [id] :as person}] 
            (find-bad-visits-for-person 
             (addresses id)
             (visits id) 
             person))
          people))



(m/find [{:name "Alice" :id 1} {:name "Bob" :id 2} {:name "Bob" :id 3}]
  (m/scan {:name "Bob" :as ?bobs-info})
  ?bobs-info)

(m/rewrite {:xs [1 2 3 4 5]
            :ys [6 7 8 9 10]}
  {:xs [!xs ...]
   :ys [!ys ...]}
  [!xs ... . !ys ...])


(m/rewrites {:name "entity1"
             :status :complete
             :history [{:value 100} {:value 300} {:value 700}]
             :future [{:value 1000} {:value 10000}]}
  {:name ?name
   :status ?status
   :history (m/scan {:value ?value})
   :future [{:value !values} ...]}
  [:div
   [:h3 ?name]
   [:strong (m/app name ?status) - ?value]
   [:ul .
    [:li !values] ...]])

(m/rewrites {:name "entity1"
             :status :complete
             :history [{:id 1} 
                       {:id 2}]
             :values {1 [{:value 100 :status :failed} 
                         {:value 200 :status :failed}]
                      2 [{:value 300 :status :complete}
                         [:value 400 :staus :complete]]}}
  {:name ?name
   :status ?status
   :history (m/scan {:id ?id})
   :values {?id (m/gather {:value !values 
                           :status (m/not :failed)})}}

  {:name ?name
   :status ?status
   :values [!values ]})





(let [hiccup [:div
              [:p {"foo" "bar"}
               [:strong "Foo"]
               [:em {"baz" "quux"} "Bar"
                [:u "Baz"]]]
              [:ul
               [:li "Beef"]
               [:li "Lamb"]
               [:li "Pork"]
               [:li "Chicken"]]]]
  ;; meander.match.delta/find
  (m/find hiccup
          (m/with [%h1 [!tags {:as !attrs} . %hiccup ...]
                   %h2 [!tags . %hiccup ...]
                   %h3 !xs
                   %hiccup (m/or %h1 %h2 %h3)]
                  %hiccup)
          [!tags !attrs !xs]))
;; =>
[[:div :p :strong :em :u :ul :li :li :li :li]
 [{"foo" "bar"} {"baz" "quux"}]
 ["Foo" "Bar" "Baz" "Beef" "Lamb" "Pork" "Chicken"]]




(m/rewrite (range 8)
  (!xs ...)
  [[[!xs !xs] [!xs !xs]] [!xs ...]])



(def db {:user {1 {:name "jimmy" :boss 2 :address 3}
                2 {:name "falcon"}}
         :address {3 {:street "1234 street"
                      :country 5}}
         :country {5 {:country-code "US"}}})

(m/match db
  
  {:user {1 {:name ?name :boss ?boss :address ?address}
          ?boss {:name ?boss-name}}
   :address {?address {:street ?street
                       :country ?country}}
   :country {?country {:country-code ?country-code}}}

  {:name ?name
   :boss ?boss-name
   :address {:street ?street
             :country-code ?country-code}})





(m/match (list 1 2 3 4)
  (!xs ... ?x)
  [!xs ?x])



(m/match {:x {:y {:z 2}}}
  {:x
   {:y
    {:z 1}}}
  :match)

(match-it)




          {:players [{:name "Jimmer"
                      :class :warrior}
                     {:name "Sir Will"
                      :class :knight}
                     {:name "Dalgrith"
                      :class :rogue}]
           }


(def game-info
  {:players {1 {:name "Jimmer"
                :class :warrior
                :weapon :short-sword
                :reports-to 2}
             2 {:name "Sir Will"
                :class :knight
                :weapon :short-sword}
             3 {:name "Dalgrith"
                :class :rogue
                :reports-to 1
                :weapon :short-sword}}
   :stats {:short-sword {:attack-power 2
                         :upgrades []}
           :long-sword {:attack-power 4
                        :upgrades [:reach]}
           :unbeatable {:attack-power 9001
                        :upgrades [:indestructible]}}})



(m/rewrite [1 2 3 4]
  [?a ?b ?c ?d]
  [?d ?b ?c ?a])


(m/rewrites game-info
  {:players {?id {:reports-to ?reports-to
                  :weapon ?report-weapon
                  :as ?report}
             ?reports-to {:weapon ?commander-weapon
                          :as ?commander}}
   :stats (m/and {?report-weapon ?report-stats}
                 {?commander-weapon ?commander-stats})}
  
  [:report-and-commander
   {:report ?report
    :commander ?commander
    :report-weapon {:name ?report-weapon & ?report-stats}
    :commander-weapon {:name ?commander-weapon & ?commander-stats}}]

  {:players {?id {:reports-to nil
                  :weapon ?weapon
                  :as ?commander}}
   :stats {?weapon ?stats}}

  [:top-level
   {:commander ?commander
    :weapon {:name ?weapon & ?stats}}])







(m/match [1 2 3 4 5 6]
  [!xs ...]
  !xs)

(let [!xs [1 2 3 4 5 6]]
  (m/subst
    [[!xs !xs] ...]))


(m/search {:a 2 :b 2}
  {:a 2}
  :ok)


(m/search {:a 2 :b 3 :c 5}
  (m/not {:a 2})
  :ok)

(m/search {:a 1} 
  (m/not {:a 2})
  :ok)




(m/search [[:a 2] [:b 3] [:c 4]]
  (m/not (m/scan [:a 2]))
  :ok)


(m/search {:a 2 :b 3 :c 4}
  {:a}
  :ok)

(m/search [1 2 3]
  [_ ... . (m/not 1) . _ ...]
  :yep)



(def my-map
  {:a :a
   :b :d
   :c :a
   :d :e})

(m/search my-map
  {?a ?a} [?a]
  
  {?a ?b
   ?b (m/some ?c)} [?a ?b ?c])


(m/search #{1 2 3}
  #{(m/and ?x (m/not 1))}
  ?x)


#{} #{1} #{2} #{3} #{1 2} #{2 3} #{1 2 3}

(m/search #{1 2 3} 
  (m/not #{1})
  :ok)


(comment
  ;; Here are some semantics for search.
  ;; Let our input be V.

  ;; ∃ v ∈ V: p(v)
  [_ ... . p . _ ...]

  ;; ∃ v ∈ V: !p(v)
  [_ ... . (not p) . _ ...]

  ;; !∀ v ∈ V: p(v) <=> ∃ v ∈ V: !p(v)
  (not [_ ... . p . _ ...])


  ;; ∃ v ∈ V: p(v)
  #{p}

  ;; ∃ v ∈ V: !p(v)
  #{(not p)}

  ;; !∀ v ∈ V: p(v) <=> ∃ v ∈ V: !p(v)
  (not #{p}))






(m/search [1 1 2]
  (m/not [_ ... . 1 . _ ...])
  :yep)




(m/search #{1 2 3 4}
  #{(m/pred even? ?x) (m/pred odd? ?y)}
  #{?x ?y})


(analyze)
(m/search nil
  (m/not (m/some _))
  true)


(m/rewrite {:records  [{:name "A" :value 1 :foo "not-important"} 
	               {:name "B" :value 2 :bar "not-important"}]}

  {:records  [(m/or (m/let [!value 0] {:name (m/and "A" !name)})
                    {:name !name :value !value}) ...]}
  {:records [{:name !name :value !value} ...]})



(m/rewrite {:records  [{:name "A" :value 1 :foo "not-important"} 
	               {:name "B" :value 2 :bar "not-important"}]}

  (m/with [%record (m/or (m/let [!value 0]
                           {:name (m/and "A" !name)})
                         
                         {:name !name :value !value})]
    
    {:records [%record ...]})
  {:records [{:name !name :value !value} ...]})




(m/defsyntax if
  [pred true-branch false-branch]
  `(meander.epsilon/or (meander.epsilon/and ~pred ~true-branch)
                       ~false-branch))


(m/defsyntax if
  [pred true-branch false-branch]
  `(meander.epsilon/or (meander.epsilon/and ~pred ~true-branch)
                       ~false-branch))



(let [x "B"]
  (m/match x
    (wander.core10/if "A" 
      (m/let [?x "A"] _)
      (m/let [?x "B"] _))
    ?x))




(m/rewrite {:records [{:name "A" :value 1 :foo "not-important"} 
	              {:name "B" :value 2 :bar "not-important"}]}

  (m/with [%record (wander.core10/if {:name "A"}
                     (m/let [!value 0] {:name !name})
                     {:name !name :value !value})]
    
    {:records [%record ...]})
  {:records [{:name !name :value !value} ...]})





(m/rewrite {:records [{:name "A" :value 1 :foo "not-important"} 
	              {:name "B" :value 2 :bar "not-important"}]}
  {:records [(m/cata !record) ...]}
  {:records [!record ...]}
  
  
  {:name (m/and "A" ?name)} {:name ?name :value 0}
 
  {:name ?name :value ?value} {:name ?name :value ?value})




(m/defsyntax sub-rewrite
  [var & patterns]
  `(meander.epsilon/app
    (meander.strategy.epsilon/rewrite ~@patterns)
    ~var))


(defn transform-record [record]
  (m/rewrite record
    {:name (m/and "A" ?name)} {:name ?name :value 0}
    {:name ?name :value ?value} {:name ?name :value ?value}))


(m/rewrite {:records [{:name "A" :value 1 :foo "not-important"} 
	              {:name "B" :value 2 :bar "not-important"}]})


(m/rewrite {:records [{:name "A" :value 1 :foo "not-important"} 
	              {:name "B" :value 2 :bar "not-important"}]}

  {:records [(m/app transform-record !record) ...]}

  {:records [!record ...]})


(m/rewrite [1 2 3 4 5 6]
  [!xs ...]
  {& [[!xs [!xs !xs]] ...]})




(m/search [1 2 3 4 5]
  [1 . _ ... 2 . _ ... ?x & ?c]
  [?x ?c])


(m/rewrite [1 2 3 4 5 6]
  [!xs ...]
  [[!xs !xs] ...])

(m/rewrite [[1 2 3 4] [5 6 7 8] [9 10 11 12]]
  [[!xs ...] ..?n]
  [[!xs !xs !xs] ..?n])


(m/rewrite [1 2 3 4 5 6]
  [!xs ...]
  [[!xs !xs] ...])



(m/match [0 1 2 5 8]
  [(m/and !num !parity)
   ...]
  [!num !parity])




(m/search {:a {1 "a1"
               2 "a2"}
           :b {3 "b3"}}
  (m/scan [?k1 (m/scan [?k2 ?v])])
  [?k1 ?k2 ?v])


(m/search [1 2 3 4 5]
  [1 . _ ... 2 . _ ... ?x & ?c]
  [ ?x ?c])



(m/with [%my-var (make-var [] (fn [k] (fn [v] (fn [acc] (update acc k conj v)))))]
  [%my-var %my-var ...])

(m/rewrite {:a {:b :c}}
  {!k1 {!k2 !v}}                      
  {!k2 {!k1 !v}})

(meander.match.runtime.epsilon/partitions
 2
 [2 3 4 5])



[[1 [2 3]]
 [4 [5 6]]]

{2 [3 4]
 5 [6 1]}


(m/rewrite [:a [1 2 3] :b [4 5]]
  [!k [!x ..!n] ..!m]
  [!k [!x ..!n] ..!m])
;; => [:a [1 2 3] :b [4 5]]


(m/match [[1 [2 3]]
          [3 [4 5]]]
 
  {:k1 !k1
   :k2 !k2
   :v !v})


(m/rewrite [:a [1 2 3] :b [4 5]]
  [!k [!x ...] ...]
  [!k [!x ...] ...])


(m/match [:a [1 2 3] :b [4 5]]
  [!ks [!vs ...] ...]
  (m/subst [!ks [!vs ..1] ..2]))


(m/match {:a 1 :b 2}
  {:a 1 & ?rest}
  ?rest)

;; =>
{:b 2}

(m/rewrite [:a 1 :b 2]
  [!xs ...]
  {& [[!xs ..2] ...]})




(make-vars [!k !v] {} (fn [acc [!k !v] (update acc !k conj !v)]))


(m/match [:a [1 2]
          :b [2 1 3]]
  [!xs [!ys ...] !ys [!xs ...]]
  {:xs !xs
   :ys !ys})

[:a :a 2 1 3]
[1 2 :b :b :b]

(m/match [[1 2 3] [4 5]]
  [[!xs ..!n] [!ys ..!n]]
  [!xs !xs !n])

{}

(m/match [2 :one :two :three]
  [?x . !xs ..?x]
  [?x !xs])

(m/match [2 :one :two]
  [?x . !xs ..?x]
  [!xs ?x])


(m/match [1 2 3]
  [!xs ..?x]
  [!xs ?x])

(m/match [2 1 1]
  [?x . !xs ..?x]
  [?x !xs])

(m/match [[:a [[1 "a1"]
               [2 "a2"]]]
          [:b [[3 "b3"]]]]
  [[!k1 [[!k1.k2 !k1.v] ...]] ...]
  {:k1s !k1})

;; =>

{:k1s [:a [{:k2 1 :v "a1"} {:k2 2 :v "a2"}]
       :b [{:k2 3 :v "b3"}]]}





