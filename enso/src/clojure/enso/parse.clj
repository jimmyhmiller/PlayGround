;; This buffer is for Clojure experiments and evaluation.
;; Press C-j to evaluate the last expression.

(require '[clojure.string :as string])



; This parser only handles single argument commands.


(def answers 
  [[["op"
     {:name :open :args [:target]} 
     nil] 
    [:line [:match "op"] [:unmatch "en"] [:arg "target"]]]
   [["open f" 
     {:name :open :args [:target]} 
     {:type :application :target "Firefox"}]
    [:line [:match "open "][:match "f"] [:unmatch "irefox"]]]
   [["open f" 
     {:name :open :args [:target]} 
     {:type :application :target "Mozilla Firefox"}]
    [:line 
     [:match "open"] 
     [:unmatch " mozilla "] 
     [:match "f"] 
     [:unmatch "irefox"]]]
   [["open firefox" 
     {:name :open :args [:target]} 
     {:type :application :target "firefox"}]
    [:line 
     [:match "open "]
     [:match "firefox"]]]])

(defn no-space? [text]
  (not (string/includes? text " ")))

(defn subtract-text [text command-name]
  (subs command-name (count text)))

(defn parse-no-space [text {command-name :name args :args}]
  [:line 
   [:match text] 
   [:unmatch (subtract-text text (name command-name))]
   [:arg (name (first args))]])

(defn index-bounds [text word]
  (let [index (string/index-of text word)] 
    [index (+ index (count word))]))

(defn find-word-locations [text words]
  (mapcat (partial index-bounds text) words))

(defn add-text-ends [text locations]
  (distinct (conj (into [] (cons 0 locations)) (count text))))

(defn split-string-at [string [fst snd & locations]]
  (if (nil? locations)
    [(subs string fst snd)]
    (cons (subs string fst snd)
          (split-string-at string (cons snd locations)))))

(defn sanitize-text [text]
  (string/lower-case (name text)))

(defn parse-with-space [text {command-name :name args :args} suggestion]
  (let [arg (first args)
        suggestion (str (sanitize-text command-name) " " 
                        (sanitize-text (arg suggestion)))]
    (->> (string/split text #" ")
         (find-word-locations suggestion)
         (add-text-ends suggestion)
         (split-string-at suggestion)
         (map vector (cycle [:match :unmatch]))
         (into [:line]))))

(defn combine-elements [current sibling]
  (if (= (second sibling) " ")
    [(first current) (str (second current) " ")]
    current))

(defn remove-empty [loc]
  (if (= (second (z/node loc)) " ")
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

(defn parse [text command suggestion]
  (normalize
   (cond
     (no-space? text) (parse-no-space text command)
     :else (parse-with-space text command suggestion))))


(defn run-scenario [[[text command suggestion]]]
  (parse text command suggestion))

(defn check-answer [[_ answer :as scenario]]
  (= (run-scenario scenario) answer))



(map run-scenario answers)
(map check-answer answers)
