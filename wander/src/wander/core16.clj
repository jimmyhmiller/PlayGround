(ns wander.core16
  (:require [meander.epsilon :as m]))




(def rules
  '((rule fact
          (fact 1) 1
          (fact ?n) (* ?n (fact (- ?n 1))))
    (rule add
          (add ?x ?y) (+ ?x ?y))))

(def parsed-rules
  (m/rewrite rules
    ((rule !name . !left !right ..!n) ..!m)
    ({:type :rule
      :name !name
      :clauses [{:left !left :right !right} ..!n]}
     ..!m)))



;; I need a repl server.
;; Not going to rely on clojure repl stuff.
;; Need to be able to send to that server from emacs?
;; Benefits of that is getting editing stuff for free
;; Downsides are not having complete control
;; Also having to write emacs lisp to get things done.
;; Not being able to make the UI I really want.

;; Before we get to that we will start with a clojure process
;; that I can send tings to by invoking a function.
;; And maybe consider simple little swing/javafx windows?
;; Or a simple web based ui? Really not sure.



(defn matches [expr {:keys [left] :as clause}]
  (if (= expr left)
    {:clause clause
     :env {}}))

;; @Ugly
(defn find-matching-rules [expr rules]
  (filter identity
          (map (fn [rule]
                 (let [ms (filter identity (map (partial matches expr) (:clauses rule)))]
                   (when (seq ms)
                     (assoc rule :matches ms)))) 
               rules)))


(defn eval-expr [expr rules]
  (let [matching (find-matching-rules expr rules)
        rule (first matching)
        clauses (:matches rule)
        clause (first clauses)]
    (assert (= 1 (count matching) (count clauses)) "not doing multiple yet")
    {:expr expr
     :clause clause
     :rule rule
     :result (get-in clause [:clause :right])}))

(eval-expr '(fact 1) parsed-rules)

;; Next we need to look for things that are matching on results
;; To optimize we would need to associate clauses with their result
;; rules. But for now we can just check after every step.

;; After that we need to think about how to schedule injection of new rules.
;; Probably a simple schedule that looks like what I've heard the erlang one does.
;; "processes" get a certain amount of gas then we switch.
;; Checking often is probably best for interactivity. Don't want waiting ever.
;; Probably can get away with single threaded though.
;; We will have to see what it would look like to check every reduction.

;; Can rules dynamically create rules?



