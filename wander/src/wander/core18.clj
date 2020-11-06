(ns wander.core18
  (:require [meander.epsilon :as m]
            [criterium.core :refer [quick-bench]]))

(def data
  {:people [{:id 1 :name "Bob"} {:id 2 :name "Alice" :inactive true}]
   :addresses 
   [{:type :business :person-id 1 :info ""}
    {:type :other :person-id 1 :info ""}
    {:type :business :person-id 2 :info ""}
    {:type :vacation :person-id 2 :info ""}]})


(def addresses  () [{:type :business :person-id 1 :info ""}
                 {:type :other :person-id 1 :info ""}
                 {:type :business :person-id 2 :info ""}
                 {:type :vacation :person-id 2 :info ""}])

(defn meander-solution [{:keys [people addresses]}]
  (m/rewrite data

    {:people (m/some (m/gather {:inactive true :as !person}))}
    (m/cata {:person !person
             :addresses ~addresses})
    
    (m/and {:person {:name ?name :id ?id}}
           {:addresses (m/gather {:person-id ?id :as !address})})
    {:name ?name
     :found 1
     :address !address}))

(defn meander-solution2 [data]
  (m/rewrite data

    {:people [_ ... {:inactive true :name ?name :id ?id} & _]
     :addresses [_ ... {:person-id ?id :as ?address} & _]}
    {:name ?name
     :found 1
     :address ?address}))

(seqable? nil)

(quick-bench )

(defn vanilla-solution [{:keys [people addresses]}]
  (->> people
       (filter :inactive)
       (some (fn [{:keys [name id]}]
               (when-let [address (some (fn [{:keys [person-id] :as address}]
                                          (when (= person-id id)
                                            address))
                                        addresses)]
                 {:name name
                  :found 1
                  :address address})))))

(defn index-by
  "Like group by but assumes attr picks out a unique identity."
  [attr coll]
  (into {} (map (juxt attr identity) coll)))

(assoc data :people
       (m/rewrites (update data :bonus #(index-by :name %)) 
         {:people (m/scan {:name ?name :as ?person}) 
          :bonus (m/or {?name {:amount ?amount}}
                       (m/let [?amount nil]
                         (m/not {?name _}) ))}
         
         {:amount ?amount & ?person}))

(def data
  {:people [{:name :john
             :age  10}
            {:name :jen
             :age  11}
            {:name :jack
             :age  12}]
   :bonus  [{:name   :john
             :amount 100}
            {:name   :jack
             :amount 200}]})


(def output 
  {:people [{:name :john
             :bonus-amount 100
             :age 10}
            {:name :jen
             :age  11}
            {:name :jack
             :bonus-amount 200
             :age 12}]
   :bonus  [{:name   :john
             :amount 100}
            {:name   :jack
             :amount 200}]})

(assoc data :people
       (m/search data
         (m/or
          {:people (m/scan {:name ?name :as ?person})
           :bonus (m/or) (m/scan {:name ?name :amount ?amount})}
          
          (m/let [?amount nil]
            {:people (m/scan {:name ?name :as ?person})
             :bonus (m/not (m/scan {:name ?name }))}))
         
         (if ?amount
           (assoc ?person :bonus-amount ?amount)
           ?person)))




(m/rewrite data
  {:people [{:name !name :as !person} ...]
   :bonus [{:name !name :amount !amount} ...]}
  
  {:people [{:bonus-amount !amount & !person} ...]})


(m/rewrite [3 4 5 6 7 8]
  [3 4 . !xs !ys ...]
  [!xs !ys ...])

(meander-solution2 data)

(vanilla-solution data)

;; 3us
(quick-bench
 (meander-solution data))

;; 450ns
(quick-bench 
 (vanilla-solution data))

(quick-bench
 (meander-solution2  data))



(defn meander-solution2 []
  (let*
      [R__22169
       (letfn*
        [C__22102
         (fn*
          C__22102
          ([data]
           (letfn*
            [state__22151
             (fn*
              state__22151
              ([]
               (let*
                   [!person []]
                 (letfn*
                  [D__22106
                   (fn*
                    D__22106
                    ([T__22117 !person]
                     (letfn*
                      [state__22153
                       (fn*
                        state__22153
                        ([]
                         (if
                             (map? T__22117)
                           (let*
                               [T__22118
                                (. T__22117 valAt :inactive)]
                             (letfn*
                              [state__22155
                               (fn*
                                state__22155
                                ([]
                                 (let*
                                     [G__22170 T__22118]
                                   (case*
                                    G__22170
                                    0
                                    0
                                    (state__22156)
                                    {1231
                                     [true
                                      (let*
                                          [T__22119
                                           (dissoc
                                            T__22117
                                            :inactive)]
                                        (let*
                                            [!person
                                             (conj
                                              !person
                                              T__22117)]
                                          [!person]))]}
                                    :compact
                                    :hash-equiv
                                    nil))))
                               state__22156
                               (fn*
                                state__22156
                                ([]
                                 (let*
                                     [T__22119
                                      (dissoc T__22117)]
                                   [!person])))]
                              (state__22155)))
                           (state__22154))))
                       state__22154
                       (fn* state__22154 ([] [!person]))]
                      (state__22153))))]
                  (if (map? data)
                    (let*
                        [T__22107 (. data valAt :people)]
                      (let*
                          [R__22160
                           (let*
                               [G__22171 T__22107]
                             (case*
                              G__22171
                              0
                              0
                              meander.match.runtime.epsilon/FAIL
                              {0 [nil true]}
                              :compact
                              :hash-equiv
                              nil))]
                        (if (meander.match.runtime.epsilon/fail?
                             R__22160)
                          (if (seqable? T__22107)
                            (let*
                                [SEQ__22109 (seq T__22107)]
                              (let*
                                  [R__22159
                                   (meander.match.runtime.epsilon/run-star-1
                                    SEQ__22109
                                    [!person]
                                    (fn*
                                     ([p__22172 input__22112]
                                      (let*
                                          [vec__22173
                                           p__22172
                                           !person
                                           (.nth vec__22173 0 nil)]
                                        (let*
                                            [input__22112_nth_0__
                                             (nth input__22112 0)]
                                          (let*
                                              [R__22158
                                               (D__22106
                                                input__22112_nth_0__
                                                !person)]
                                            (if
                                                (meander.match.runtime.epsilon/fail?
                                                 R__22158)
                                              meander.match.runtime.epsilon/FAIL
                                              (let*
                                                  [vec__22176
                                                   R__22158
                                                   !person
                                                   (nth
                                                    vec__22176
                                                    0
                                                    nil)]
                                                [!person])))))))
                                    (fn*
                                     ([p__22179]
                                      (let*
                                          [vec__22180
                                           p__22179
                                           !person
                                           (.nth vec__22180 0 nil)]
                                        (let*
                                            [T__22108
                                             (dissoc data :people)]
                                          (try
                                            [(let*
                                                 [!person__counter
                                                  (meander.substitute.runtime.epsilon/iterator
                                                   !person)]
                                               (let*
                                                   [R__22103
                                                    (C__22102
                                                     {:person
                                                      (if
                                                          (.
                                                           !person__counter
                                                           hasNext)
                                                        (.
                                                         !person__counter
                                                         next)),
                                                      :addresses
                                                      addresses})]
                                                 (if
                                                     (meander.match.runtime.epsilon/fail?
                                                      R__22103)
                                                   (throw
                                                    meander.substitute.runtime.epsilon/FAIL)
                                                   (nth
                                                    R__22103
                                                    0))))]
                                            (catch
                                                java.lang.Exception
                                                e__16611__auto__
                                              (if
                                                  (identical?
                                                   e__16611__auto__
                                                   meander.substitute.runtime.epsilon/FAIL)
                                                meander.match.runtime.epsilon/FAIL
                                                (throw
                                                 e__16611__auto__)))))))))]
                                (if
                                    (meander.match.runtime.epsilon/fail?
                                     R__22159)
                                  (state__22152)
                                  R__22159)))
                            (state__22152))
                          (state__22152))))
                    (state__22152))))))
             state__22152
             (fn*
              state__22152
              ([]
               (let*
                   [!address []]
                 (letfn*
                  [D__22124
                   (fn*
                    D__22124
                    ([T__22147 !address ?id]
                     (letfn*
                      [state__22161
                       (fn*
                        state__22161
                        ([]
                         (if
                             (map? T__22147)
                           (let*
                               [T__22148
                                (. T__22147 valAt :person-id)]
                             (letfn*
                              [state__22163
                               (fn*
                                state__22163
                                ([]
                                 (if
                                     (= ?id T__22148)
                                   (let*
                                       [T__22149
                                        (dissoc
                                         T__22147
                                         :person-id)]
                                     (let*
                                         [!address
                                          (conj
                                           !address
                                           T__22147)]
                                       [!address ?id]))
                                   (state__22164))))
                               state__22164
                               (fn*
                                state__22164
                                ([]
                                 (let*
                                     [T__22149
                                      (dissoc T__22147)]
                                   [!address ?id])))]
                              (state__22163)))
                           (state__22162))))
                       state__22162
                       (fn* state__22162 ([] [!address ?id]))]
                      (state__22161))))]
                  (if (map? data)
                    (let*
                        [T__22126
                         (. data valAt :addresses)
                         T__22125
                         (. data valAt :person)]
                      (if (map? T__22125)
                        (let*
                            [T__22129
                             (. T__22125 valAt :id)
                             T__22128
                             (. T__22125 valAt :name)]
                          (let*
                              [?name T__22128]
                            (let*
                                [?id T__22129]
                              (let*
                                  [T__22130
                                   (dissoc T__22125 :name :id)]
                                (if
                                    (seqable? T__22126)
                                  (let*
                                      [SEQ__22131 (seq T__22126)]
                                    (let*
                                        [R__22167
                                         (meander.match.runtime.epsilon/run-star-1
                                          SEQ__22131
                                          [!address]
                                          (fn*
                                           ([p__22183 input__22134]
                                            (let*
                                                [vec__22184
                                                 p__22183
                                                 !address
                                                 (nth
                                                  vec__22184
                                                  0
                                                  nil)]
                                              (let*
                                                  [input__22134_nth_0__
                                                   (nth
                                                    input__22134
                                                    0)]
                                                (let*
                                                    [R__22166
                                                     (D__22124
                                                      input__22134_nth_0__
                                                      !address
                                                      ?id)]
                                                  (if
                                                      (meander.match.runtime.epsilon/fail?
                                                       R__22166)
                                                    meander.match.runtime.epsilon/FAIL
                                                    (let*
                                                        [vec__22187
                                                         R__22166
                                                         !address
                                                         (nth
                                                          vec__22187
                                                          0
                                                          nil)
                                                         ?id
                                                         (nth
                                                          vec__22187
                                                          1
                                                          nil)]
                                                      [!address])))))))
                                          (fn*
                                           ([p__22190]
                                            (let*
                                                [vec__22191
                                                 p__22190
                                                 !address
                                                 (nth
                                                  vec__22191
                                                  0
                                                  nil)]
                                              (let*
                                                  [T__22127
                                                   (dissoc
                                                    data
                                                    :person
                                                    :addresses)]
                                                [{:name ?name,
                                                  :found 1,
                                                  :address
                                                  (nth
                                                   !address
                                                   0
                                                   nil)}])))))]
                                      (if
                                          (meander.match.runtime.epsilon/fail?
                                           R__22167)
                                        meander.match.runtime.epsilon/FAIL
                                        R__22167)))
                                  meander.match.runtime.epsilon/FAIL)))))
                        meander.match.runtime.epsilon/FAIL))
                    meander.match.runtime.epsilon/FAIL)))))]
            (state__22151))))]
        (let*
            [R__22168 (C__22102 data)]
          (if (meander.match.runtime.epsilon/fail? R__22168)
            meander.match.runtime.epsilon/FAIL
            (let*
                [vec__22194 R__22168 R__22104 (.nth vec__22194 0 nil)]
              R__22104))))]
    (if (meander.match.runtime.epsilon/fail? R__22169) nil R__22169)))



(m/search [[1 2] [1 3] [2 3]]
  (m/scan [?x (m/app dec ?x)])
  [?x])

(m/search [[1 2] [1 3] [2 3] [2 4]]
  (m/scan [(m/app #(* 2 %) ?x) ?x])
  [?x])
