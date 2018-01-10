(ns experiment-spec.compiler
  (:require [clojure.core.match :as core.match]))


;http://matt.might.net/articles/compiling-up-to-lambda-calculus/

(defn convert-match [case]
  (core.match/match [case]
                    [else :guard keyword?] else
                    [[sym pred]] [sym :guard pred]
                    [s-expr] [(list (into [] s-expr) :seq)]))

(defmacro match [c & cases]
  `(core.match/match ~c ~@(mapcat (fn [[k v]] [(convert-match k) v]) (partition 2 cases))))

(def VOID `(~'fn [void#] void#))



(def ERROR `(~'fn [_]
              ((~'fn [f#] (f# f#)) (~'fn [f#] (f# f#)))))

(def TRUE `(~'fn [t#] (~'fn [f#] (t# ~VOID))))
(def FALSE `(~'fn [t#] (~'fn [f#] (f# ~VOID))))


(def apply-n 
  (fn [f n z]
    (cond 
      (= n 0) z
      :else `(~f ~(apply-n f (- n 1) z)))))

(defn church-numerals [n]
  (cond
    (= n 0) `(~'fn [f#] (~'fn [z#] z#))
    :else   (let [f (gensym)
                  z (gensym)]
              `(~'fn [~f] (~'fn [~z]
                          ~(apply-n f n z))))))

(church-numerals 10)

(def ZERO? `(~'fn [n#]
             ((n# (~'fn [_#] ~FALSE)) ~TRUE)))

(def SUM `(~'fn [n#]
            (~'fn [m#]
              (~'fn [f#]
                (~'fn [z#]
                  ((m# f#) ((n# f#) z#)))))))

(def MUL `(~'fn [n#]
            (~'fn [m#]
              (~'fn [f#]
                (~'fn [z#]
                  ((m# (n# f#)) z#))))))

(def PRED `(~'fn [n#]
             (~'fn [f#]
               (~'fn [z#]
                 (((n# (~'fn [g#] (~'fn [h#] (h# (g# f#)))))
                   (~'fn [u#] z#))
                  (~'fn [u#] u#))))))

(def SUB `(~'fn [n#]
            (~'fn [m#]
              ((m# ~PRED) n#))))

(def CONS `(~'fn [car#]
             (~'fn [cdr#]
               (~'fn [on-cons#]
                 (~'fn [on-nil#]
                   ((on-cons# car#) cdr#))))))

(def NIL `(~'fn [on-cons#]
            (~'fn [on-nil#]
              (on-nil# ~VOID))))

(def CAR `(~'fn [list#]
            ((list# (~'fn [car#]
                     (~'fn [cdr#]
                       (car#))))
             ~ERROR)))

(def CDR `(~'fn [list#]
            ((list# (~'fn [car#]
                     (~'fn [cdr#]
                       cdr#)))
             ~ERROR)))

(def PAIR? `(~'fn [list#]
              ((list# (~'fn [_#] (~'fn [_#] ~TRUE)))
               (~'fn [_] ~FALSE))))

(def NULL? `(~'fn [list#]
              ((list# (~'fn [_#] (~'fn [_#] ~FALSE)))
               (~'fn [_#] ~TRUE))))

(def Y `((fn [y#] (fn [F#] (F# (fn [x#] (((y# y#) F#) x#)))))
         (fn [y#] (fn [F#] (F# (fn [x#] (((y# y#) F#) x#)))))))


(defn compile [expr]
  (match [expr]
         [_ (partial = true)] TRUE
         [_ (partial = false)] FALSE
         [_ symbol?] expr
         [_ number?] (church-numerals expr)
         ('if cond t f) (compile `(~cond (~'fn [] ~t) (~'fn [] ~f)))
         ('and a b) (compile `(~'if ~a ~b false))
         ('+ x y) `((~SUM ~(compile x)) ~(compile y))
         ('* x y) `((~MUL ~(compile x)) ~(compile y))
         ('- x y) `((~SUB ~(compile x)) ~(compile y))
         ('* x y) `((~MUL ~(compile x)) ~(compile y))
         ('= x y) (compile `(~'and 
                             (~'zero? (~'- ~x ~y))
                             (~'zero? (~'- ~y ~x))))
         ('zero? exp) `(~ZERO? ~(compile exp))
         ('let [var exp] body) (compile `((~'fn [~var] ~body) ~exp))
         ('letrec [f lam] body) (compile`(~'let [~f (~Y (~'fn [~f] ~lam))] ~body))
         (fn [] exp) `(~'fn [_#] ~(compile exp))
         (fn [v] exp) `(~'fn [~v] ~(compile exp))
         (f) (compile `(~(compile f) ~VOID))
         (f exp) `(~(compile f) ~(compile exp))
         (f exp1 exp2) (compile `((~f ~exp1) ~exp2))))

(def fib 
  (compile '(letrec
             [fib (fn [n] 
                    (if (= n 0)
                      1
                      (* n (fib (- n 1)))))]
             (fib 5))))

(((eval fib) inc) 0)




