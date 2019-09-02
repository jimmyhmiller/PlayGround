(ns wander.core5
  (:require [cheshire.core :as json]
            [meander.epsilon :as m]
            [meander.syntax.epsilon :as r.syntax]
            [meander.match.syntax.epsilon :as r.match.syntax]
            [meander.strategy.epsilon :as r]
            [meander.strategy.epsilon :as strat]
            [hiccup.core :as hiccup]
            [clojure.string :as string]
            [com.rpl.specter :as specter]))



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





(r.syntax/defsyntax all-steps-between [first-pattern second-pattern mvar]
  `[~'_ ... . (m/and ~first-pattern ~mvar) . ~mvar ... . (m/and ~second-pattern ~mvar)])



(m/find '[(/ (- 4 2) (* 3 5 (- 2 2))) 
          (* 3 5 (- 2 2)) 
          (* 3 5 0) 
          (- 4 2)
          (/ 2 0)]
  (all-steps-between (m/scan 0) (/ _ 0) !steps)
  !steps)





(def pokemon (json/parse-string (slurp "/Users/jimmyhmiller/Desktop/GAME_MASTER.json") true))


`m/scan` is an incredibly useful operator when searching for things. It would be great to have something like it for match. More or less, this would act like filter. I made an operator and called it gather. Here is the preliminary definition.

(r.syntax/defsyntax gather
  ([pattern]
   (gather pattern '...))
  ([pattern repeat-pattern]
   (case (::r.syntax/phase &env)
     :meander/match
     `(m/seqable (m/or ~pattern ~'_) ~repeat-pattern)
     ;; else
     `[~pattern ...])))

This definition works in the simple case (there is in epsilon bug with or and maps being combined, but we will ignore that). Below are some simple cases.


```clojure
(m/match [1 2 3 4 5 6]
  (gather (m/pred even? !xs) ...)
  !xs)
;; => [2 4 6

(m/match [[1 2 3] [1 2] [3 2] [56] 3]
  (gather [_ !xs] ...)
  !xs)
;; => [2 2]
```

The problem with this definition is that the repeat-pattern doesn't work the way you'd intuitively think it would.
For example:

```clojure

(m/match [1 1 1 1 1 1 1 1]
  (gather (m/pred even? !xs) ..1)
  !xs)

;; => []

```
Ideally, this pattern would fail. Because what we want to express is we found 1 or more even numbers, not just one or more item in the collection. So far, I have been unable to figure out a way to write this.

I was hoping maybe someone could help. I think this is an incredibly useful operator, but only if the repeat symantics work properly. This is especially important when capturing groups (ie ..!n).



(r.syntax/defsyntax some
  ([pattern]
   (some pattern '...))
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


(m/match [1 2 3 2]
  [(not-nil !xs) ...]
  !xs)

(r.syntax/defsyntax map-of
  ([key-pattern value-pattern]
   (map-of key-pattern value-pattern '...))
  ([key-pattern value-pattern repeat-pattern]
   (case (::r.syntax/phase &env)
     :meander/match
     `(m/pred map? (m/seqable [~key-pattern ~value-pattern] ~repeat-pattern))
     ;; else
     `{& [[~key-pattern ~value-pattern] ~repeat-pattern]})))




(m/match {"test" 2 "thing" 4 }
  (map-of (m/pred string? !ks) (m/pred number? !vs) ..2)
  (m/subst (map-of !ks !vs ..2)))


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
                            :evolutionBranch (gather {:evolution !evolutions
                                                      :evolutionItemRequirement (m/not nil)} ..!n)}})}
  [[!pokemon [!evolutions ..!n]] ..3])


(m/rewrite [{:name "jimmy"
             :addresses [{:address "Stuff"
                          :business true}
                         {:address "Other Stuff"
                          :business true}]}
            {:name "falcon"
             :addresses [{:address "Falcons Address"
                          :business true}]}]

  (gather {:name (m/pred string? !name)
           :addresses (gather {:business true
                               :as !addresses} ..!n)})

  [!name [!addresses ..!n] ...])





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
                            :stats {:as !stats}}
                           & !rest})}

  (gather {:pokemon !pokemon
           :form !form
           :rarity !rarity
           :stats !stats
           :rest !rest}))








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

(def eliminate-all-zeros
  (strat/bottom-up
   (strat/attempt eliminate-zeros)))

(eliminate-all-zeros '(+ (+ 3 (+ 2 (+ (+ 0 (+ 0 0)) (+ 2 0)))) 0))

(fn [] "do things")


{:rule (my-function ?x ?y)}
=> 
(println ?x ?y)


{:expr (?rule & ?args)
 :result [:weird :data]}
=>
(println ?rule ?args)


{:execution-history
 [(scan 0 :as !steps) ...!steps (/ _ 0 :as !steps)]}
=>
!steps



(defn my-function [])


(def game-info
  {:players [{:name "Jimmer"
              :class :warrior}
             {:name "Sir Will"
              :class :knight}
             {:name "Dalgrith"
              :class :rogue}]
   :weapons [{:name :long-sword
              :allowed-classes #{:warrior :knight}
              :standard-upgrade :power}
             {:name :short-sword
              :allowed-classes #{:warrior :rogue}
              :standard-upgrade :speed}
             {:name :unbeatable
              :allowed-classes #{:rogue}
              :standard-upgrade :nuclear}]
   :stats {:short-sword {:attack-power 2
                         :upgrades []}
           :long-sword {:attack-power 4
                        :upgrades [:reach]}
           :unbeatable {:attack-power 9001
                        :upgrades [:indestructible]}}
   :third-party #{:unbeatable}})



(defn weapons-for-class [class weapons]
  (filter (fn [{:keys [allowed-classes]}] 
            (contains? allowed-classes class)) 
          weapons))

(defn gather-weapon-info [class {:keys [weapons stats third-party] :as info}]
  (->> weapons
       (weapons-for-class class)
       (filter #(not (contains? third-party (:name %))))
       (map #(assoc % :stats (stats (:name %))))))

(defn player-with-weapons [{:keys [weapons stats third-party] :as info} player]
  (map (fn [weapon player]
         {:name (:name player)
          :weapon (:name weapon)
          :class (:class player)
          :attack-power (get-in weapon [:stats :attack-power])
          :upgrades (conj (get-in weapon [:stats :upgrades])
                          (get-in weapon [:standard-upgrade]))})
       (gather-weapon-info (:class player) info)
       (repeat player)))

(defn players-with-weapons [{:keys [players weapons stats third-party] :as info}]
  (mapcat (partial player-with-weapons info) players))

(defn players-with-weapons' [game-info]
  (m/search game-info
    {:players (m/scan {:name ?name
                       :class ?class})
     :weapons (m/scan {:name ?weapon
                       :allowed-classes #{?class}
                       :standard-upgrade !upgrades})
     :stats {?weapon {:attack-power ?attack-power
                      :upgrades [!upgrades ...]}}
     :third-party (m/not #{?weapon})}

    {:name ?name
     :weapon ?weapon
     :class ?class
     :attack-power ?attack-power
     :upgrades !upgrades}))


(time
 (players-with-weapons game-info))



(time
 (players-with-weapons' game-info))






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

(time
 (find-potential-bad-visits data)) 



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
  (let [addresses (cons (:preferred-address person) 
                        (:other-addresses person))]
    {:zips (doall (filter some? (distinct (map :zip addresses))))
     :cities (doall (filter some? (distinct (map :city addresses))))}))


(defn distinct-zips-and-cities' [person]
  (m/match person
    {:preferred-address {:zip (m/or nil !zips)
                         :city (m/or nil !cities)}
     :other-addresses [{:zip (m/or nil !zips)
                        :city (m/or nil !cities)} ...]}
    {:zips (distinct !zips)
     :cities (distinct !cities)}))

(time
 (distinct-zips-and-cities person))

(time
 (distinct-zips-and-cities' person))


(m/search '[(+ 2 3) (+ 5 3) (+ 7 0) (+ 4 3) (/ 2 0)]
  (m/scan (m/and (m/scan 0) !steps) . !steps ... (m/and (/ _ 0) !steps))
  !steps)
