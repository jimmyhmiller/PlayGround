(ns wander.core6
  (:require [meander.epsilon :as m]
            [meander.strategy.epsilon :as r]
            [meander.match.syntax.epsilon :as match.syntax]
            [clojure.test.check.generators :as gen]
            [clojure.spec.alpha :as s]))



(defn my-tuple [& args]
  (println (map gen/generator? args))
  (apply gen/tuple args))


(gen/sample
 (gen/let
     [?x (clojure.spec.alpha/gen number?)]
   (apply
    gen/tuple
    (list (gen/return 1) (gen/return ?x) (gen/return 2) (gen/return ?x)))))





(defn pre-transform [env]
  (r/rewrite
   {:tag :meander.match.syntax.epsilon/pred,
    :form ?pred
    :arguments ({:tag :lvr :symbol ?x})} ~(let [generator `(s/gen ~?pred)] 
                                            (swap! env assoc ?x generator)
                                            {:tag :lvr :symbol ?x})


   {:tag :lvr :symbol ?x :as ?lvr} ~(do (when-not (get @env ?x)
                                          (swap! env assoc ?x 'gen/any))
                                        ?lvr)))

(def create-lvr-generators
  (r/rewrite
   {:env (m/seqable [!keys !vals] ...) :expr ?expr}
   (gen/let [!keys !vals ...]
     ?expr)))


(def generator-transform
  (r/rewrite
   {:tag :lvr :symbol ?x} (gen/return ?x)
   {:tag :lit :value ?x} (gen/return ?x)
   {:tag :cat :elements (!elements ...)} (gen/tuple . !elements ...)
   {:tag :prt :left ?left} ~?left
   {:tag :vec :prt ?prt} (gen/fmap vec ?prt)))

(defn create-generator [lhs]
  (let [env (atom {})]
    (let [expr (match.syntax/parse lhs {})
          expr ((r/top-down (r/attempt (pre-transform env))) expr)
          value (create-lvr-generators 
                 {:env @env :expr ((r/bottom-up (r/attempt generator-transform)) expr)})]
      (eval value))))

(gen/sample
 (create-generator '[1 ?x 2 (m/pred pos-int? ?x)]))


(s/gen number?)
(s/gen
 (deref
  (resolve 'number?)))


(gen/sample
 (gen/fmap vec (apply gen/tuple (list (gen/return 1)))))


(gen/sample
 (gen/fmap (in)
           (apply gen/tuple (list (gen/return 1)))))
