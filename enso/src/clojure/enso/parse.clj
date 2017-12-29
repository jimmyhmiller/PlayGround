(ns enso.parse
  (:require [clojure.string :as string]
            [clojure.zip :as z]))

; This parser only handles single argument commands.

(defn no-space? [text]
  (not (string/includes? text " ")))

(defn subtract-text [text command-name]
  (subs command-name (count text)))

(defn index-bounds [text word from]
  (let [index (string/index-of text word from)] 
    [index (+ index (count word))]))

(defn keep-track-of-index [text [indexes start] word]
  (let [new-start (+ start (count word))]
    [(cons (index-bounds text word start) indexes)
     new-start]))

(defn find-word-locations [text words]
  (->> words
       (reduce (partial keep-track-of-index text) ['() 0])
       (first)
       (reverse)
       (flatten)))

(defn add-text-ends [text locations]
  (distinct (conj (into [] (cons 0 locations)) (count text))))

(defn split-string-at [string [fst snd & locations]]
  (if (nil? locations)
    [(subs string fst snd)]
    (cons (subs string fst snd)
          (split-string-at string (cons snd locations)))))

(defn sanitize-text [text]
  (string/lower-case (name text)))

(defn combine-elements [current sibling]
  (if (= (second sibling) " ")
    [(first current) (str (second current) " ")]
    current))

(defn remove-empty [loc]
  (if (or (= (second (z/node loc)) " ")
          (= (second (z/node loc)) ""))
    (z/remove loc)
    loc))

(defn normalize [parsed]
  (loop [zp (-> (z/vector-zip parsed) z/down)]
    (let [current (-> zp z/right)
          sibling (-> current z/right)]
      (cond
        (nil? current) (into [] (z/root zp))
        (nil? sibling) (into [] (z/root (remove-empty current)))
        :else
        (-> current
            (z/edit combine-elements (z/node sibling))
            remove-empty
            recur)))))

(defn parse-with-space [text {command-name :name args :args} suggestion]
  (let [arg (first args)
        suggestion (str (sanitize-text command-name) " " 
                        (sanitize-text (arg suggestion)))]
    (->> (string/split text #" ")
         (find-word-locations suggestion)
         (add-text-ends suggestion)
         (split-string-at suggestion)
         (map vector (cycle [:match :unmatch]))
         (into [:line])
         (normalize))))

(defn parse-no-space [text {command-name :name args :args}]
  (normalize 
   [:line 
    [:match text] 
    [:unmatch (subtract-text text (name command-name))]
    [:arg (str " " (name (or (first args) "")))]]))

(defn parse [text command suggestion]
  (let [cleaned-text (string/trim text)]
    (cond 
      (empty? text) []
      (nil? command) [:line [:match text]]
      (no-space? cleaned-text) (parse-no-space cleaned-text command)
      (empty? suggestion) [:line [:match text]]
      :else (parse-with-space cleaned-text command suggestion))))

