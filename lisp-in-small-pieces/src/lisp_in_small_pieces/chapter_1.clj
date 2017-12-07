(ns lisp-in-small-pieces.chapter-1)

(defn wrong [message exp]
  (throw (Exception. message)))

(defn atom? [exp]
  (not (seq? exp)))

(defn eprogn [exps env]
  (last (map #(evaluate % env) exps)))

(defn evlis [exps env]
  (map #(evaluate % env) exps))

(defn lookup [id env]
  (get @env id))

(defn update! [id env val]
  (swap! env assoc id val))

(defn extend* [env names vals]
  (apply assoc env (flatten (map vector names vals))))

(defn extend [env names vals]
  (atom (extend* @env names vals)))

(defn evaluate [e env]
  (if (atom? e)
    (cond 
      (symbol? e) (lookup e env)
      (or (number? e) (string? e) (boolean? e) (vector? e)) e
      :else (wrong "Cannot evaluate" e))
    (case (first exp)
      'quote (second e)
      'if (if (evaluate (nth e 1) env)
            (evaluate (nth e 2) env)
            (evaluate (nth e 3) env))
      'do (eprogn (rest e) env)
      'set! (update! (nth e 1) env (evaluate (nth e 2) env))
      'fn (make-function (nth e 1) (rest (rest e)) env)
      :else (invoke (evaluate (first e) env)
                    (evlis (rest e) env)))))


(rest (rest (list 1 2 3)))
(nth (list 1 2 3 4) 3)
