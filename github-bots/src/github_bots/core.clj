(ns github-bots.core
  (:require [tentacles.repos :as repos]
            [tentacles.core :as core]
            [tentacles.pulls :as pulls]
            [tentacles.issues :as issues]
            [clj-yaml.core :as yaml]
            [dnddice.core :as dice])
  (:use [clojure.string :only [lower-case split]]
        [clojure.repl :only [source]]))

;; (def user "nomicness")
;; (def repo "a-whole-new-world")

(def user "nomic-awesome-bot")
(def repo "a-whole-new-world")

(defn roll [roll-description]
  (-> roll-description
      dice/parse-roll
      dice/perform-roll
      :total))

(defn roll-comment [description user]
  (str "@" user " requested a roll \n\n"
       "Below are the results:\n\n"
       "`|" (roll description) "|`"))

(defn map-if [pred f coll]
  (reduce (fn [xs x]
            (if (pred x)
              (concat xs (list (f x)))
              (concat xs (list x))))
          '() coll))

(defn distinct-by
  "Removes duplicates as determines by pred. Keeps last duplicate."
  [pred coll]
  (->> coll
       (group-by pred)
       (map (comp last last))))

(defn words [word]
  (split word #" "))


(defn in? [coll target]
  (some #(= target %) coll))

(defn vote? [com]
  (or
   (= (lower-case com) "yay")
   (= (lower-case com) "nay")))

(defn proposal? [title]
  (not (nil? (re-matches #"^[0-9]+.*" title))))

(defn merged? [pr-event]
  (let [action (:action pr-event)
        merged (:pull_request pr-event)]
    (and (= action "closed") merged)))

(defn get-proposal-number [title]
  (Integer/parseInt (re-find #"^[0-9]+" title)))

(defn get-proposer [issue]
  (-> issue :user :login))

(defn vote-count [votes]
  (->> votes
       (map :body)
       (map (comp keyword lower-case))
       (group-by identity)
       (map (fn [[k v]] [k (count v)]))
       (into {})))

(defn calculate-proposer-points [proposal-number votes]
  (let [{:keys [yay nay]} (vote-count votes)
        ratio (/ yay (+ yay nay))]
    (Math/round
     (double
      (* (- proposal-number 291) ratio)))))

(defn percent [a b]
  (if (zero? b) 0
    (* (/ a b) 100)))

(defn vote-percent [votes]
  (let [{:keys [yay nay] :or {yay 0 nay 0}} (vote-count votes)]
    {:yay (percent yay (+ yay nay))
     :nay (percent nay (+ yay nay))}))

(defn get-vote-status [votes]
  (let [vote-percents (vote-percent votes)
        {:keys [yay]} vote-percents]
    (cond (> yay 50) :passing
          (< yay 50) :failing
          :else :tied)))

(def vote-status->label
  {:failing "Failing"
   :passing "Passing"
   :tied "Tied"})



(def labels-by-issue-number
  #(issues/issue-labels user repo %))

(defn swap-vote-status-label [issue-number]
  (let [labels (labels-by-issue-number issue-number)
        vote-status (get-vote-status (get-votes issue-number))
        status-labels (vals vote-status->label)
        new-label {:name (vote-status->label vote-status)}]
    (->> labels
         (remove #(in? status-labels (:name %)))
         (cons new-label)
         (set-labels issue-number))))

(defn quorum? [votes active-players]
  (> (percent (count votes) (count active-players))
     50))



(def get-labels
  #(issues/repo-labels user repo))

(def get-label
  #(issues/specific-label user repo % {}))

(def add-label
  #(issues/add-labels user repo %1 [%2] {}))

(def set-labels
  #(issues/replace-labels user repo %1 %2 {}))

(defn add-comment [issue-number body]
  (issues/create-comment user repo issue-number body {}))

(def comments-by-issue-number
  #(issues/issue-comments user repo % {:all-pages true}))

(defn get-players-file []
  (repos/contents user repo "players.yaml" {:str? true}))

(defn get-players
  ([] (get-players (get-players-file)))
  ([file]
   (-> file
       :content
       (yaml/parse-string))))


(defn get-active-players
  ([] (get-active-players (get-players)))
  ([players]
   (:activePlayers players)))


(defn active?
  ([player-name] (active? (get-active-players) player-name))
  ([active-players player-name]
   (not (nil? (some #{player-name} (map :name active-players))))))


(defn get-votes
  ([issue-number] (get-votes issue-number (get-active-players)))
  ([issue-number active-players]
   (->> issue-number
        comments-by-issue-number
        (filter (comp vote? :body))
        (distinct-by :user)
        (filter (comp (partial active? active-players) :login :user)))))


(defn update-points-by-player [player]
  (update player :points + (roll "2d6+1")))


(defn add-vote-points [active-players votes]
  (let [participated (set (map (comp :login :user) votes))]
    (->> active-players
         (map-if (comp #(contains? participated %) :name) update-points-by-player))))


(defn add-proposer-points [active-players issue votes]
  (let [proposer-name (get-proposer issue)
        proposal-number (get-proposal-number (:title issue))
        proposer-points (calculate-proposer-points proposal-number votes)]
    (map-if #(= (:name %) proposer-name) #(update % :points + proposer-points) active-players)))

(defn update-player-points [file new-points]
  (let [players (get-players file)
        new-player-info (assoc players :activePlayers new-points)]
    (repos/update-contents user repo "players.yaml" "update points"
                           (yaml/generate-string new-player-info :dumper-options {:flow-style :block})
                           (:sha file))))

(defn check-quorum [comment]
  (if (proposal? (-> comment :issue :title))
    (let [active-players (get-active-players)
          votes (get-votes (-> comment :issue :number) active-players)]
      (quorum? votes active-players))
    false))

(defn update-points [pr-event]
  (if (and
       (proposal? (-> pr-event :pull_request :title))
       (merged? pr-event))
    (let [active-players (get-active-players)
          votes (get-votes (-> pr-event :pull_request :number) active-players)
          file (get-players-file)]
      (update-player-points file (add-vote-points active-players votes)))))



