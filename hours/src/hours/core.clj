(ns hours.core
  (:require [propaganda.generic-operators :as go]
            [propaganda.support-values :as support-values]))

(use 'propaganda.system)
(use 'propaganda.values)
(use '[propaganda.intervals.common :exclude [extend-merge]])
(use 'propaganda.intervals.system)


(defn plus-interval
  [x y]
  (make-interval (+ (:lo x) (:lo y)) (+ (:hi x) (:hi y))))


(def generic-plus (doto (go/generic-operator +)
                   (go/assign-operation plus-interval
                                        interval? interval?)
                   (go/assign-operation (coercing ->interval plus-interval)
                                        number? interval?)
                   (go/assign-operation (coercing ->interval plus-interval)
                                        interval? number?)))


(defn sub-interval
  [x y]
  (make-interval (- (:lo x) (:hi y)) (- (:hi x) (:lo y))))


(def generic-sub (doto (go/generic-operator -)
                   (go/assign-operation sub-interval
                                        interval? interval?)
                   (go/assign-operation (coercing ->interval sub-interval)
                                        number? interval?)
                   (go/assign-operation (coercing ->interval sub-interval)
                                        interval? number?)))



(doseq [generic-op [generic-plus generic-sub]]
  ;; supported values support
  (go/assign-operation generic-op
                       (support-values/supported-unpacking generic-op)
                       support-values/supported? support-values/supported?)
  (go/assign-operation generic-op
                       (coercing support-values/->supported generic-op)
                       support-values/supported? flat?)
  (go/assign-operation generic-op
                       (coercing support-values/->supported generic-op)
                       flat? support-values/supported?))


(def plus (function->propagator-constructor generic-plus))
(def sub (function->propagator-constructor generic-sub))



(defn sum
  ([system x y total]
   (-> system
       (plus x y total)
       (plus y x total)
       (sub total x y)
       (sub total y x)))

  ([system xs total]
   (if (= (count xs) 2)
     (let [[a b] xs]
       (sum system a b total))
     (let [new-total (gensym)]
       (sum
        (sum system (rest xs) new-total)
        (first xs) new-total total)))))


(def jimmy {:name jimmy
            :hours
            {:monday [0 6]
             :tuesday [0 6]
             :wednesday [0 6]
             :thursday [0 6]
             :friday [0 6]
             :saturday [0 8]
             :sunday [0 0]}
            :total [20 25]})

(def system (make-system (doto (default-merge) extend-merge) (default-contradictory?)))


(defn hours->system [system hours]
  (reduce (fn [system [name [x y]]]
            (add-value system name (make-interval x y)))
          system
          hours))

(defn total-hours [system hours total]
  (-> system
      (sum (keys hours) total)))


(defn vec->interval [[x y]]
  (make-interval x y))




(defn person->system [system person]
  (-> system
      (hours->system (:hours person))
      (total-hours (:hours person) :total)
      (add-value :total (vec->interval (:total jimmy)))))

(-> system
    (add-value :jimmy (person->system system jimmy))
    (add-value :test (person->system system jimmy)))

(defn system->namespaced-system [system n]
  (assoc system
    :values (into {} (map (fn [[k v]] [(keyword (str n "/" (name k))) v]) (:values system)))
    :propagators (into {} (map (fn [[k v]] [(keyword (str n "/" (name k))) v]) (:propagators system)))))

(let [s (person->system system jimmy)]

(system->namespaced-system (add-value system :a 2) "jimmy"))






(hours->system (make-system (doto (default-merge) extend-merge) (default-contradictory?)) (:hours jimmy))



(let [custom-merge (doto (default-merge) extend-merge)
      system (make-system custom-merge (default-contradictory?))
      result-system
        (-> system
            (sum-more :monday :tuesday :wednesday :thursday :friday :saturday :sunday :total)
            (add-value :monday (make-interval 0 6))
            (add-value :tuesday (make-interval 0 6))
            (add-value :wednesday (make-interval 0 6))
            (add-value :thursday (make-interval 0 6))
            (add-value :friday (make-interval 0 6))
            (add-value :saturday (make-interval 0 8))
            (add-value :total (make-interval 20 25))
            (sum-more :monday-2 :tuesday-2 :wednesday-2 :thursday-2 :friday-2 :saturday-2 :sunday-2 :total)
            (add-value :monday-2 (make-interval 0 6))
            (add-value :tuesday-2 (make-interval 0 6))
            (add-value :wednesday-2 (make-interval 0 6))
            (add-value :thursday-2 (make-interval 0 6))
            (add-value :friday-2 (make-interval 0 6))
            (add-value :saturday-2 (make-interval 0 8))
            (add-value :total-2 (make-interval 30 35))
            (sum :total :total-2 :total3)
            (sum :monday :monday-2 :monday-total)
            (add-value :monday-total 8)
            (add-value :total3 50)
            (add-value :monday 4)
            (add-value :tuesday 4)
            (add-value :thursday 5))]

    {:monday-2 (get-value result-system :monday-2)
     :monday (get-value result-system :monday)
     :monday-total (get-value result-system :monday-total)
     :tuesday-2 (get-value result-system :tuesday-2)
     :wednesday-2 (get-value result-system :wednesday-2)
     :thursday-2 (get-value result-system :thursday-2)
     :friday-2 (get-value result-system :friday-2)
     :saturday-2 (get-value result-system :saturday-2)
     :total (get-value result-system :total)
     :total-2 (get-value result-system :total-2)})
