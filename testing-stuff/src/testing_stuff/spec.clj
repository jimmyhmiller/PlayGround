(ns testing-stuff.spec
  (:require [clojure.spec :as s]))


(s/def ::binding
  (s/cat
    :name  symbol?
    :value (constantly true)))

(s/def ::bindings
  (s/and vector?
         #(-> % count even?)
         (s/* ::binding)))


(s/def ::description string?)
(s/def ::given (s/? (s/cat
                     :given-kw #{'given}
                     :bindings ::bindings
                     :and (s/* (s/cat
                                :and-kw #{'and}
                                :bindings ::bindings)))))


(s/def ::when (s/cat
               :when-kw #{'when}
               :bindings ::bindings))


(s/def ::then (s/cat
               :then-kw #{'then}
               :variable symbol?
               :should-key #{'should-be}
               :pred (constantly true)))

(s/def ::and (s/* (s/cat
                   :and-kw #{'and}
                   :variable symbol?
                   :should-key #{'should-be}
                   :pred (constantly true))))

(s/def ::scenario
  (s/cat
   :description ::description
   :given ::given
   :when ::when
   :then ::then
   :and ::and))




;; (defGenMethod scenario [desc]
;;   (? (:given context) (* :and contexts))
;;   (:when event)
;;   (cat (:then expr) (:should-be val))
;;   (* (:and exprs) (:should-be val)))

(s/fdef scenario
    :args ::scenario
    :ret any?)


(defmacro scenario [& args]
  (let [info (s/conform ::scenario args)]
    (run-scenario info)))


(defn run-scenario [scenario]
  (let [given-bindings (s/unform ::bindings (-> scenario :given :bindings))
        and-bindings (->> scenario
                          :given
                          :and
                          (map :bindings)
                          flatten
                          (s/unform ::bindings))
        when-bindings (s/unform ::bindings (-> scenario :when :bindings))
        bindings (into [] (concat given-bindings and-bindings when-bindings))
        then (:then scenario)
        first-case [(:variable then) (:pred then)]
        rest-case (map (fn [x] [(:variable x) (:pred x)]) (:and scenario))
        cases (concat [first-case] rest-case)
        equals (map (fn [[var val]] `(= ~var ~val)) cases)]
     `(let ~bindings (assert (and ~@equals)))))





(s/valid? ::given '(given [x 3] and [y 4]))
(s/valid? ::given '(given [x 3]))


(scenario "1 + 2 = 3"
          given [x 1]
          and [y 2]
          when [x (+ x y)]
          then x should-be 3)


(s/conform ::scenario
          '("1 + 2 = 3"
          :given [x 1]
          :and [y 2]
          :when [x (+ x y)]
          :then x :should-be 3
          :and y :should-be 1))



