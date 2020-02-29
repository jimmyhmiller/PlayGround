(ns wander.core16
  (:require [meander.epsilon :as m]
            [clojure.walk :as walk]))



(do
  (def rules
    '((rule fact
            (fact 1) 1
            (fact ?n) (* ?n (fact (- ?n 1))))
      
      (rule add
            (add ?x ?y) (+ ?x ?y))

      ;; Need to modify eval-match not to use regular eval
      ;; this rule is an example of why. It would infinitely recurse.
      ;; Actually is that right? Maybe I need to instead ensure I
      ;; separate meta rules? Need to think about this more.

      #_(rule print-fact
            ((fact ?x) => ?y)  (println ?x ?y))

      (rule print-fact
            ((fact ?x) => ?y) (:clj (println (quote ?x) (quote ?y))))

      #_(rule print
            (println ?x ?y) (:clj (println (quote ?x) (quote ?y))))
      (rule subject
            (- ?x ?y) (:clj (- ?x ?y)))
      
      (rule multiply
            (* ?x ?y) (:clj (* ?x ?y)))))

  (def parsed-rules
    (m/rewrite rules
      ((rule !name . !left !right ..!n) ..!m)
      ({:type :rule
        :name !name
        :clauses [{:left !left :right !right} ..!n]}
       ..!m))))



(defn matches [expr {:keys [left] :as clause}]
  (let [env
        (m/rewrite [expr left]
          [(!expr ..?n) (!left ..?n)]
          {& [(m/cata [!expr !left]) ...]}

          [?x (m/and ?y
                     (m/pred symbol?)
                     (m/pred #(clojure.string/starts-with? (name %) "?") ?str))]
          [?y ?x]

          {:fail true :as ?map} {:fail true}

          [?x ?x] {}
          
          ?x {:fail true})]
    (if (:fail env)
      nil
      {:clause clause :env env})))


(matches '((fact 1 => 1)) '{:left ((fact 1 => 1))})


;; @Ugly
(defn find-matching-rules [rules expr]
  (filter identity
          (map (fn [rule]
                 (let [ms (filter identity (map (partial matches expr) (:clauses rule)))]
                   (when (seq ms)
                     (assoc rule :matches ms :expr expr)))) 
               rules)))

(defn substitute [expr env]
  (walk/postwalk-replace env expr))


(defn eval-if-clj [expr]
  (if (and (seq? expr) (= (first expr) :clj))
    (eval (second expr))
    expr))

(mapcat (partial find-matching-rules parsed-rules) (tree-seq coll? seq '(* 2 (* 3 4))))

(defn eval-expr [rules expr]
  (let [matching (reverse (mapcat (partial find-matching-rules rules) (tree-seq coll? seq expr)))
        rule (first matching)
        clauses (:matches rule)
        clause (first clauses)
        result (eval-if-clj (substitute (get-in clause [:clause :right]) (get-in clause [:env]))) ]
   #_ (assert (= 1 (count matching) (count clauses)) "not doing multiple yet")
   {:expr expr
    :sub-expression (:expr rule)
    :clause clause
    :rule rule
     ;; probably wrong to do this given side effects?
    :result (substitute expr {(:expr rule) result})}))

(defn build-result-expr [evaled-expr]
  (m/rewrite evaled-expr
    {:expr ?expr
     :result ?result}
    (?expr => ?result)))


(defn match-on-eval [rules evaled-expr]
  (let [result-expr (build-result-expr evaled-expr)
        matching (find-matching-rules rules result-expr)
        rule (first matching)
        clauses (:matches rule)
        clause (first clauses)]
    #_(assert (= 1 (count matching) (count clauses)) "not doing multiple yet")
    (if-not rule
      evaled-expr
      (assoc evaled-expr
             :eval-match
             {:rule rule
              :expr result-expr
              :clause clause
              :result (substitute (get-in clause [:clause :right]) (get-in clause [:env])) }))))


;; Need to think about the fact that I will need to refire
;; rewrite results when things normalize.
;; Meaning fact n = n * fact n - 1
;; that maybe should print,
;; but so should more exact things
(defn step [rules expr]
  (->> expr
       (eval-expr rules)
       (match-on-eval rules)))


(def result-atom (atom nil))

;; @Ugly
(defn fixed-point-eval-match [rules eval-match]
  (let [result (:result eval-match)]
    (reset! result-atom result)
    (cond
      (or (not result) (= result '(quote nil))) 
      :no-result

      (= (first result) :clj)
      (do
        
        ;; Need to figure out how to properly print code values
        (eval (second result))
        nil)
      
      :else (let [stepped (step rules result)]
              (when (:eval-match stepped)
                (fixed-point-eval-match rules (:eval-match stepped)))
              (recur rules stepped)))))


(defn full-step [rules expr]
  (let [result (step rules expr)]
    (when-let [eval-match (:eval-match result)]
      (fixed-point-eval-match rules eval-match))
    (dissoc result :eval-match)))

(defn n-step [n rules expr results]
  (if (zero? n)
    (conj results expr)
    (recur (dec n) rules (:result (full-step rules expr)) (conj results expr))))

(defn fixed-point-steps [rules expr results]
  (let [result (:result (full-step rules expr))]
    (if (= result (last results))
      results
      (recur rules result (conj results expr)))))

(n-step 20 parsed-rules '(fact 5) [])


(fixed-point-steps parsed-rules '(fact (* 1 5)) [])

(full-step parsed-rules
           (:result
            (full-step parsed-rules '(fact 2))))

;; Next we need to look for things that are matching on results
;; To optimize we would need to associate clauses with their result
;; rules. But for now we can just check after every step.


;; After that we need to think about how to schedule injection of new rules.
;; Probably a simple schedule that looks like what I've heard the erlang one does.
;; "processes" get a certain amount of gas then we switch.
;; Checking often is probably best for interactivity. Don't want waiting ever.
;; Probably can get away with single threaded though.
;; We will have to see what it would look like to check every reduction.


;; I need a repl server.
;; Not going to rely on clojure repl stuff.
;; Need to be able to send to that server from emacs?
;; Benefits of that is getting editing stuff for free
;; Downsides are not having complete control
;; Also having to write emacs lisp to get things done.
;; Not being able to make the UI I really want.

;; Before we get to that we will start with a clojure process
;; that I can send things to by invoking a function.
;; And maybe consider simple little swing/javafx windows?
;; Or a simple web based ui? Really not sure.


;; Can rules dynamically create rules?

;; Need to think about replacing evaluation results
