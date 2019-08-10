(ns wander.core5
  (:require [cheshire.core :as json]
            [meander.epsilon :as m]
            [meander.syntax.epsilon :as r.syntax]
            [meander.match.syntax.epsilon :as r.match.syntax]
            [meander.strategy.epsilon :as r]
             [meander.strategy.epsilon :as strat]
            [hiccup.core :as hiccup]
            [clojure.string :as string]))


(defn unescape [s]
  (string/replace s #"&amp;" "&"))


(def reddit
  (json/parse-string
   (slurp "/Users/jimmyhmiller/Downloads/reddit.json") true))


(hiccup/html)

(m/rewrite reddit
  {:data
   {:children
    (gather {:data
             {:title !title
              :permalink !link
              :preview {:images
                        [{:source {:url !image}} & _]}}})}}

  [:div {:class :container}
   .
   [:div
    [:p [:a {:href (m/app str "https://reddit.com" !link)}
         !title]]
    [:img {:src (m/app unescape !image)}]]
   ...])

(def pokemon (json/parse-string (slurp "/Users/jimmyhmiller/Desktop/GAME_MASTER.json") true))


(r.syntax/defsyntax gather
  ([pattern]
   (gather pattern '...))
  ([pattern repeat-pattern]
   (case (::r.syntax/phase &env)
     :meander/match
     `(m/seqable (m/or ~pattern ~'_) ~repeat-pattern)
     ;; else
     `[~pattern ...])))


(r.syntax/defsyntax some
  ([pattern repeat-pattern]
   (case (::r.syntax/phase &env)
     (let [!xs (gensym "!xs")]
       `(m/and (m/seqable (m/or (m/and ~pattern ~!xs) ~'_) ...)
               (m/guard (pos? (count ~!xs)))
               (m/and ~repeat-pattern (m/app count ~!xs))))
     ;; else
     #_`[~pattern ...])))

(r.syntax/defsyntax not-nil
  [pattern]
  (case (::r.syntax/phase &env)
    :meander/match
    `(m/and (m/not nil) ~pattern)
    ;; else
    pattern))

(r.syntax/defsyntax map-of
  ([key-pattern value-pattern]
   (map-of key-pattern value-pattern '...))
  ([key-pattern value-pattern repeat-pattern]
   (case (::r.syntax/phase &env)
     :meander/match
     `(m/pred map? (m/seqable [~key-pattern ~value-pattern] ~repeat-pattern))
     ;; else
     `{& [[~key-pattern ~value-pattern] ~repeat-pattern]})))




(m/match [:a 2]
  (some (m/pred number? !xs))
  !xs)



(m/match [:b 1 :s]
  (m/and [(m/or (m/and (m/pred number? !xs) !ys) _) ...]
         (m/guard (>= (count !ys) 1)))
  !xs)




(m/match [:a :b :c]
  (m/and
   (m/let [*xs 0] _)
   [(m/and (m/let [*xs (inc *xs)] !xs)) ...])
  *xs)


(m/match {:a 1 :b 2}
  {& (m/and !rest !rest2)}
  !rest)


(m/match [1 :a 3]
  (m/and
   (m/seqable (meander.epsilon/or (m/and (m/pred keyword? !xs) *x) _) ..?n)
   (meander.epsilon/guard (pos? ?n)))
  [!xs *x])

(m/rewrite [:a 1]
  (m/and (m/seqable (m/or (m/pred number? !xs) _) ..?n)
         (m/guard (pos? ?n)))
  !xs)


(/ 0 1)

(with [%x]
        %x)
;; Matrix
(m/rewrite [[1 2 3] [4 5 6] [4 3 2]]
  [[!xs ..!n] ..?m]
  [[!xs ..?m] ..!n])



(m/match {:a 1 :b 2}
  (map-of (m/pred keyword? !ks) (m/pred number? !vs) ..!a)
  (m/subst (map-of !ks !vs ..!a)))

(m/rewrite [1 [2]]
  [?x (gather ?x)]
  ?x)


(defn all-pokemon [pokemon])
(m/rewrite pokemon
  {:itemTemplates (gather {:pokemonSettings
                           {:pokemonId !pokemon
                            :evolutionBranch (some {:evolution !evolutions
                                                    :evolutionItemRequirement (m/not nil)} !n)}})}
  [[!pokemon [!evolutions ..!n]] ..3])



(m/rewrite
 {:things
  [{:something-else :thing}
   {:name "jimmy"
    :properties [{:valid true
                  :name "This one"}
                 {:valid false}]}
   {:name "falcon"
    :properties [{:valid true
                  :name "Also this"}
                 {:valid false}
                 {:valid true
                  :name "And this"}]}]}


 {:things [(m/or {:name !names
                  :properties [!properties ..!n]}
                 _)
           ...]}

 [{:name !names
   :properties [!properties ..!n]} ...])





(m/rewrite
 {:things
  [{:something-else :thing}
   {:name "jimmy"
    :properties [{:valid true
                  :name "This one"}
                 {:valid false}]}
   {:name "falcon"
    :properties [{:valid true
                  :name "Also this"}
                 {:valid false}
                 {:valid true
                  :name "And this"}]}]}


 {:things [(m/or {:name !names
                  :properties [(m/or {:valid true :name !properties} _) ..!n]}
                 _)
           ...]}

 [{:name !names
   :properties [!properties ..!n]} ...])







(m/rewrite pokemon
  {:itemTemplates (gather {:pokemonSettings
                           {:pokemonId !pokemon
                            :form !form
                            :rarity (not-nil !rarity)
                            :stats {:as !stats}}})}

  (gather {:pokemon !pokemon
           :form !form
           :rarity !rarity
           :stats !stats}))








(m/rewrite {:foo [:child1 :child2]
            :bar [:child3 :child4]}
  (m/seqable [!parent [!children ..!]] ..!)
  [(<> !parent . !children ..!) ..!])



(defn grab-all-foods [user]
  {:favorite-foods [{:name !foods} ...]
   :recipes [{:title !foods} ...]
   :meal-plan {:breakfast [{:food !foods} ...]
               :lunch [{:food !foods} ...]
               :dinner [{:food !foods} ...]}}

  !foods)

(def point [1 2])


(m/match point
  (m/with [%num? (m/pred number?)]
    (m/or [%num? (m/pred number? ?y)]
          [%num? (m/pred number? ?y) %num?]))
  ?y)


(m/defsyntax num?
  ([] `(m/pred number?))
  ([pattern] `(m/pred number? ~pattern)))

(m/match point
  [(num?) (num? ?y)] ?y
  [(num?) (num? ?y) (num?)] ?y)


(defn favorite-food-info [user foods]
  (m/search {:user user
            :foods foods}

    {:user
     {:name ?name
      :favorite-foods (m/scan {:name ?food})}
     :foods {?food {:popularity ?popularity
                    :calories ?calories}}}

    {:name ?name
     :favorite-food {:food ?food
                     :popularity ?popularity
                     :calories ?calories}}))


(favorite-food-info 
 {:name "jimmy" 
  :favorite-foods [{:name :nachos} {:name :tacos}]}
 {:nachos {:popularity :pop
           :calories 300}
  :tacos {:popularity :pop
          :calories 3000}})



(def eliminate-zeros
  (strat/rewrite
   (+ ?x 0) ?x
   (+ 0 ?x) ?x))

(def apply-to-all
  (strat/bottom-up
   (strat/attempt eliminate-zeros)))
