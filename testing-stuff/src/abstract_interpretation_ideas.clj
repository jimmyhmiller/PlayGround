(require '[clojure.string :as string])


(defn fizz-buzz-bad [n]
  (cond 
    (= (mod n 3) 0) "fizz"
    (= (mod n 5) 0) "buzz"
    (and (= (mod n 3) 0) 
         (= (mod n 5) 0)) "fizzbuzz"
    :else n))


(fizz-buzz-bad 3)
(fizz-buzz-bad 5)
(fizz-buzz-bad 7)


;; I'd like to be able to detect that the "and" condition here is
;; unreachable. I'd also like to use the examples below to infer
;; types. I could infer the return type of mod based on that.
;; I'd also love to be able to generate example inputs that match each
;; case, ideally showing some sort of table representation of the
;; condition.


(defn pull-out-parts [coll]
  (let [x (:x coll)
        y (:y coll)]
    (+ x y)))

;; Here we should be able to infer that coll is a map with :x and :y
;; that are both numbers. We should be able to make examples. We
;; should also be able to show what happens if either value is nil or
;; if the whole thing is nil.


(defn reducer [state action]
  (case (:type action)
    :increment (inc state)
    :decrement (dec state)
    :increment-n (+ state (:n action))
    state))

;; Here I should be able to infer the possible shapes of the action. I
;; should assume that the action only has `:n` if the type is
;; `:increment-n`. I should know that state is a number.


(def name-lookup 
  {:jimmy :stuff
   :james :thing})

(defn lookup-by-name [some-name]
  (name-lookup (keyword (string/lower-case some-name))))

;; Here I want to infer that the only meaningful values are the keys
;; in the lookup. `some-name` can of course be other values, but those
;; are the only real valid ones. I should also detect the null
;; reference exception here. 

(defn conjoin-path [some-name]
  (when some-name
    (str "/my-folder/" some-name)))


;; Here, we should surface the fact that when does not handle the
;; empty string case. This should be shown in a nice table of results


(defn lower-names [user]
  (update user :name string/lower-case))

(defn assign-active [user]
  (assoc user :active (>= 1 (:status user))))

(defn filter-users [users]
  (->> users
       (map lower-names)
       (map assign-active)
       (filter :active)))

;; In this example were should infer that this is a collection of
;; maps. But showing different collections is not really the point. If
;; we were aggregating, maybe multiple collections would be
;; helpful. Here we want to show the output for each map in the
;; collection. But ideally, we also show the steps in between.

;; In many ways what I'm thinking about is a hybrid abstract
;; interpreter and actual interpreter. But I don't think an actual
;; interpreter will need to be written. We can just leverage the real
;; language for that. What we will do is use abstract interpretation
;; to create constraint and create example inputs that we will then
;; feed to program fragments (also derived for the abstract
;; interpreter. We want to do this in a safe way, trying as much as we
;; can to prevent infinite loops and catching exceptions. All of this
;; can be done in the background with good explicit timeouts. We do
;; need to be careful about side effects when using the actual
;; language.


