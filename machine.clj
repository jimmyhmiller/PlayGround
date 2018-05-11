(use 'clojure.set)



(defn dissoc-in
  "Dissociates an entry from a nested associative structure returning a new
  nested structure. keys is a sequence of keys. Any empty maps that result
  will not be present in the new structure."
  [m [k & ks :as keys]]
  (if ks
    (if-let [nextmap (get m k)]
      (let [newmap (dissoc-in nextmap ks)]
        (if (seq newmap)
          (assoc m k newmap)
          (dissoc m k)))
      m)
    (dissoc m k)))


(defn string->list [string]
  (map str (into [] string)))


(def machine {:start :q1
              :q1 (fn [x] (if (even? x) :q1 :q2))
              :q2 {1 :q2
                   0 :q3
                   :accept "yes"}
              :q3 {0 :q2
                   1 :q2}})





(def n-machine {:start :q1
                :q1 {1 [:q1 :q2]
                     0 [:q1]}
                :q2 {:accept "yes"}})

(def e-machine {:start [:q1]
                :q1 {0 [:q1]
                     1 [:q1 :q2]}
                :q2 {0 [:q3]
                     :e [:q3]}
                :q3 {1 [:q4]}
                :q4 {0 [:q4]
                     1 [:q4]
                     :accept "yes"}})


(def machine-ab-repeat {:start :q1
                        :q1 {"a" :q1
                             "b" :q2}
                        :q2 {"b" :q2
                             :accept "yes"}})


(def machine-odd-ones {:start :q1
                       :q1 {0 :q1
                            1 :q2}
                       :q2 {0 :q2
                            1 :q1
                            :accept "yes"}})



(def c-machine {:start [:q1]
                :q1 {:e {:e [[:q2 :$]]}}
                :q2 {0 {:e [[:q2 0]]}
                     1 {0 [[:q3 :e]]}}
                :q3 {1 {0 [[:q3 :e]]}
                     :e {:$ [[:q4 :e]]}}
                :q4 {:accept "yes"}})


(defn cons-e [head tail]
  (let [tail' (if (nil? tail) '() tail)]
    (if (and (not= head :e) (not= head nil))
      (cons head tail')
      tail')))





(defn accept? [machine state]
  (contains? (machine state) :accept))


(defn stack-e [[state stack]]
  [state (cons :e stack)])

(defn transition-c-single 
  [machine input [state [head & tail]]]
  (-> [[s h]]
      (fn [s (cons-e h tail)])
      (map (get (get (get machine state) input) head))
      (concat
       (map (fn [[s h]] [s (cons-e h (cons-e head tail))]) (get (get (get machine state) input) :e)))))

(defn transition-c [machine input states]
   (into #{} (mapcat (partial transition-c-single machine input) states)))



(transition-c c-machine :e (map stack-e [[:q1 '()] [:q3 '(0 0 0 :$)]]))
(transition-c c-machine 1 [[:q2 '(0 0 :$)] [:q3 '(0 0 0 :$)]])

(follow-e-c c-machine #{} #{[:q1 '()]})


(defn follow-e-c [machine visited states]
  (let [unvisited (difference states visited)
        new-states (transition-c machine :e unvisited)]
    (if (subset? new-states unvisited)
      (union visited unvisited)
      (follow-e-c machine (union visited unvisited) new-states))))


(defn automate-c [machine inputs states]
   (println inputs states (follow-e-c machine #{} states))
   (if (empty? inputs)
     (not= (some true? (map (partial accept? machine) (map first (follow-e-c machine #{} states)))) nil)
     (automate-c machine
                 (rest inputs)
                 (union
                  (transition-c machine (first inputs) (follow-e-c machine #{} states))
                  (follow-e-c machine #{} (transition-c machine (first inputs) states))))))

(automate-c c-machine [0 0 0 0 1 1 1 1] #{[:q1 '()]})
(follow-e-c c-machine #{} #{[:q1 '()]})
(transition-c c-machine 0 (follow-e-c c-machine #{} #{[:q1 '()]}))

(defn transition-n [machine input states]
  (into #{} (mapcat #(% input) (map machine states))))


(defn follow-e [machine visited states]
  (let [unvisited (difference states visited)
        new-states (transition-n machine :e unvisited)]
    (if (subset? new-states unvisited)
      (union visited unvisited)
      (follow-e machine (union visited unvisited) new-states))))


(defn automate-n [machine inputs states]
  (if (empty? inputs)
    (not= (some true? (map (partial accept? machine) (follow-e machine #{} states))) nil)
    (automate-n machine
                (rest inputs)
                (union
                 (transition-n machine (first inputs) (follow-e machine #{} states))
                 (follow-e machine #{} (transition-n machine (first inputs) states))))))



(defn start-automate-n [machine inputs]
  (automate-n machine inputs (machine :start)))



(defn transition [machine input state]
  ((machine state) input))


(defn automate [machine inputs state]
  (if (empty? inputs)
    (accept? machine state)
    (automate machine
              (rest inputs)
              (transition machine
                          (first inputs)
                          state))))



(defn start-automate [machine inputs]
  (automate machine inputs (machine :start)))



(defn single-letter-nfa [letter]
  (let [q1 (gensym :q)
        accept (gensym :q)]
    {:start [q1]
     q1 {letter [accept]}
     accept {:accept "yes"}}))



(defn find-accept [nfa]
  (->> nfa
       (filter (fn [[k v]] (contains? v :accept)))
       (map first)))

(defn remove-accept [nfa]
  (dissoc-in nfa (concat (find-accept nfa) [:accept])))


(defn add-e [nfa state new-state]
  (assoc-in nfa (concat state [:e]) new-state))

(defn remove-start [nfa]
  (dissoc nfa :start))

(defn concat-nfa [nfa1 nfa2]
  (merge
   (remove-accept (add-e nfa1 (find-accept nfa1) (:start nfa2)))
   (remove-start nfa2)))


(defn union-nfa [nfa1 nfa2]
  (let [new-state [(gensym :q)]]
    (merge
     (add-e {:start new-state} new-state (concat (:start nfa1) (:start nfa2)))
     (remove-start nfa1)
     (remove-start nfa2))))


(defn string->nfa
  ([string]
   (let [string-list (string->list string)]
     (string->nfa (single-letter-nfa (first string-list)) (rest string-list))))
  ([nfa string-list]
   (if (empty? string-list)
     nfa
     (string->nfa
      (concat-nfa nfa (single-letter-nfa (first string-list)))
      (rest string-list)))))


(start-automate-n
 (concat-nfa
  (string->nfa "xy")
  (union-nfa
   (string->nfa "ab")
   (string->nfa "bc")))
 (string->list "xyab"))

(start-automate-n 
 (concat-nfa (single-letter-nfa "a") 
             (single-letter-nfa "b")) 
 (string->list "ab"))


(start-automate-n e-machine [1 1])
(start-automate machine [1 1 0 1])
(start-automate machine-ab-repeat (string->list "aaaabbbb"))
(start-automate machine-odd-ones [0 0 0 0 1 1 0 0 0 0 1 0 0 0 1 1])

