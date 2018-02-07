(require '[experiment-spec.match :refer
           [match make-helper eval-many]])
(require '[experiment-spec.compiler :refer [compile lambda->num]])
(refer-clojure :exclude [eval])

(defn eval [expr env]
  (match [expr]
         [s symbol?] (get env s)
         ('fn [x] body) (fn [arg] (eval body (assoc env x arg)))
         (f x) ((eval f env) (eval x env))))



(compile '(letrec [eval (fn [expr env]
                         (ifn ))]
                  (eval 1 2)))
