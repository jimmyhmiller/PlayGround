(ns wander.core10
  (:require [meander.match.ir.epsilon :as ir]
            [meander.strategy.epsilon :as r]
            [meander.strategy.epsilon :as strat]
            [meander.epsilon :as m]
            [meander.match.epsilon :as match]
            [clojure.string :as string]))


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
        
       ]
     {:clauses clauses
      :matrix matrix
      :ir ir
      
      })))

(defmacro analyze [expr]
  (m/match expr
    (~'m/match ?expr & ?body)
    `(analyze-compile :match (quote ~?body) (quote ~?expr))))

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
