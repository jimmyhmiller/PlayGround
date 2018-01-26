(require '[experiment-spec.match :refer [match make-helper eval-many]])
(refer-clojure :exclude [eval])

(def globals (atom {}))

(defn add-global [var val]
  (swap! globals assoc var val)
  nil)

(defn reset-globals []
  (reset! globals {}))

(defn lookup [env var]
  (get env var (get @globals var)))

(defn add-var [env var val]
  (assoc env var val))


(defn eval [expr env]
  (match [expr]
         [n number?] n
         [b boolean?] b
         [s symbol?] (lookup env s)
         ('clj body) (clojure.core/eval body)
         ('def var val) (add-global var (eval val env))
         ('let [var val] body) (eval body 
                                     (add-var env var val))
         ('fn [x] body) (fn [arg] (eval body (add-var env x arg)))
         ('fn [x y] body) (fn [arg1 arg2] 
                            (eval body (add-var 
                                        (add-var env x arg1) 
                                        y arg2)))
         (f x) ((eval f env) (eval x env))
         (f x y) ((eval f env) (eval x env) (eval y env)) 
         ('if pred t f) (if (eval pred env)
                          (eval t env)
                          (eval f env))))







(eval-many
 ;std-lib
 (def + (clj +))
 (def - (clj -))
 (def * (clj *))
 (def / (fn [x y] ((clj float) ((clj /) x y))))
 (def = (clj =))
 ;end std-lib
 (def infinite (fn [x] (infinite x)))
 (def ignore-first (fn [x y] y))

 (ignore-first (infinite 2) 2))


