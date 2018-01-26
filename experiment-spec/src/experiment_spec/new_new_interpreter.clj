(require '[experiment-spec.match :refer 
           [match make-helper eval-many]])

(refer-clojure :exclude [eval])

(def globals (atom {}))

(defn reset-globals []
  (reset! globals {})
  nil)

(defn add-global [var val]
  (swap! globals assoc var val)
  nil)

(defn lookup [env var]
  (get env var (get @globals var)))

(defn add-var [env var value]
  (assoc env var value))


(defn eval [expr env]
  (match [expr]
   [n number?] n
   [b boolean?] b
   [s string?] s
   [n nil?] n
   [var symbol?] (lookup env var)
   ('clj thing) (resolve thing)
   ('def var val) (add-global var (eval val env))
   ('let var va body) (eval body 
                                (add-var env var (eval val env)))
   ('if pred t f) (if (eval pred env)
                    (eval t env)
                    (eval f env))
   ('fn [x] body) (fn [x'] (eval body (add-var env x x')))
   ('fn [x y] body) (fn [x' y'] 
                      (eval body (add-var 
                                  (add-var env x x') 
                                  y y')))
   (f x) ((eval f env) (eval x env))
   (f a b) ((eval f env) 
            (eval a env)
            (eval b env))))

(reset-globals)


(eval-many
 ;-----std-lib----
 (def * (clj *))
 (def println (clj println))
 (def / ((clj comp) (clj float) (clj /)))
 (def + (clj +))
 (def - (clj -))
 ;end------std-lib----

 (println (- (/ 10 20) (+ 3 4))))
