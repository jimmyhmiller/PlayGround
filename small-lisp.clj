;; Anything you type in here will be executed
;; immediately with the results shown on the
;; right.


(defn boolean? [b]
  (or (true? b) (false? b)))

(defn atom? [x]
  (not (coll? x)))



(defn error [& message]
  (throw (Exception. (apply str message))))


(def empty-begin 813)

(defn pair? [coll]
  (not (empty? coll)))

(defn eprogn [exps env]
  (if (pair? exps)
    (if (pair? (rest exps))
      (do (eval (first exps) env)
        (eprogn (rest exps) env))
      (eval (first exps) env))
    empty-begin))



(defn create-env
  ([] (create-env {}))
  ([coll] (atom coll)))


(defn update! [id env value]
  (swap! env assoc id value)
  value)

(defn extend [env bindings]
  (create-env (merge @env bindings)))


(defn evlis [exps env]
  (if (pair? exps)
    (cons (eval (first exps) env)
          (evlis (rest exps) env))
    nil))

(defn invoke [f args]
  (if (fn? f)
    (f args)
    (error "not a function")))



(defn make-function [variables body env]
  (fn [values]
    (eprogn body (extend env (zipmap variables values)))))

(eval 1 {})
(eval \s {})
(eval true {})
(eval 'a (create-env {'a 2}))
(eval "string" {})
(eval '(quote 1) {})
(eval '(if true 0 1) {})
(eval '(if false 0 1) {})
(eval '(if a 0 1) (create-env {'a true}))
(eval '(if a 0 1) (create-env {'a false}))
(eval '(begin 1 2 3) {})
(eval '(begin (set! a 2) a) (create-env {'a 3}))
(eval '((lambda (f) (f)) (lambda (x) 2))
      (create-env {'list (partial apply list)}))





(eval' 1 {})
(eval' \s {})
(eval' true {})
(eval' 'a (create-env {'a 2}))
(eval' "string" {})
(eval' '(quote 1) {})
(eval' '(if true 0 1) {})
(eval' '(if false 0 1) {})
(eval' '(if a 0 1) (create-env {'a true}))
(eval' '(if a 0 1) (create-env {'a false}))
(eval' '(begin 1 2 3) {})
(eval' '(begin (set! a 2) a) (create-env {'a 3}))
(eval' '((lambda (f) (f)) (lambda (x) 2))
      (create-env {'list (partial apply list)}))









(defmulti eval' (fn [e env] (atom? e)))


(defmethod eval' true [e env] (eval-atom e env))
(defmethod eval' false [e env] (eval-list e env))


(defmulti eval-atom (fn [e env] (symbol? e)))
(defmulti eval-list (fn [e env] (first e)))



;symbol
(defmethod eval-atom true [e env] (get @env e))

;constant
(defmethod eval-atom false [e env]
  (if (or (number? e) (string? e) (char? e) (boolean? e))
    e
    (error "cannot evaluate " e)))



;quote
(defmethod eval-list 'quote [[_ body] env]
  (eval' body env))


;if
(defmethod eval-list 'if [[_ pred true-case false-case] env]
  (if (eval' pred env)
    (eval' true-case env)
    (eval' false-case env)))


;begin
(defmethod eval-list 'begin [[_ & body] env]
  (eprogn' body env))



;set!
(defmethod eval-list 'set! [[_ var value] env]
  (update! var env (eval' value env)))


;lambda
(defmethod eval-list 'lambda [[_ args & body] env]
  (make-function' args body env))


;function
(defmethod eval-list :default [[f & args] env]
  (invoke (eval f env)
          (evlis' args env)))







(def eprogn' nil)
(def make-function' nil)
(def evlis' nil)




(defn eval [e env]
  (if (atom? e)
    (cond
     (symbol? e) (get @env e)
     (or (number? e) (string? e) (char? e) (boolean? e)) e
     :else (error "cannot evaluate " e))
    (case (first e)
      quote (second e)
      if (if (eval (nth e 1) env)
           (eval (nth e 2) env)
           (eval (nth e 3) env))
      begin (eprogn (rest e) env)
      set! (update! (nth e 1) env (eval (nth e 2) env))
      lambda (make-function (second e) (nthrest e 2) env)
      (invoke (eval (first e) env)
              (evlis (rest e) env)))))
