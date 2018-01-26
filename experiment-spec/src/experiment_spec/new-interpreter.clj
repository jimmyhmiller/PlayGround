(require '[experiment-spec.match :refer [match make-helper eval-many]])
(refer-clojure :exclude [eval])

(def globals (atom {}))

(defn add-global [name val]
  (swap! globals assoc name val))

(defn get-global [name]
  (get @globals name))

(defn reset-globals []
  (reset! globals {}))

(defn lookup [name env]
  (let [local (get env name)]
    (if local
      local
      (get-global name))))

(defn apply-lambda [f args env]
  (apply (eval f env) 
         (map (fn [val] (eval val env)) args)))

(defn add-var [env name val]
  (assoc env name val))

(defn add-vars [env names vals]
  (reduce (fn [env [name val]] 
            (add-var env name val))
          env
          (map vector names vals)))

(defn create-lambda [args body env]
  (fn [& vals] (eval body (add-vars env args vals))))

(defn eval [expr env]
  (match [expr]
         [n number?]      n
         [b boolean?]     b
         [s string?]      s
         [s symbol?]      (lookup s env)
         ('def name val)  (add-global name (eval val env))
         ('if pred t f)   (if (eval pred env)
                            (eval t env)
                            (eval f env))
         ('fn [& args]
          body)           (create-lambda args body env)
         (f & args)       (apply-lambda f args env)))

(reset-globals)
(add-global '= =)
(add-global '+ +)
(add-global '* *)
(add-global '- -)

(eval-many
 (def fact
   (fn [x]
     (if (= x 0) 
       1
       (* x (fact (- x 1))))))
 (fact 5))
