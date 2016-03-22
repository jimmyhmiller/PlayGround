(ns github-bots.core
  (:require [tentacles.repos :as repos]
            [tentacles.core :as core]
            [tentacles.pulls :as pulls]
            [tentacles.issues :as issues]
            [clj-yaml.core :as yaml]
            [dnddice.core :as dice])
  (:use [clojure.string :only [lower-case]]))

(def user "nomicness")
(def repo "a-whole-new-world")


(defn roll [roll-description]
  (-> roll-description
      dice/parse-roll
      dice/perform-roll
      :total))


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

(defn vote? [com]
  (or (= (lower-case com) "yay") (= (lower-case com) "nay")))

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
  (* (/ a b) 100))

(defn quorum? [votes active-players]
  (> (percent (count votes) (count active-players))
     50))

(def comments-by-issue-number
  (partial issues/issue-comments user repo))

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


(defn active? [active-players player-name]
  (not (nil? (some #{player-name} (map :name active-players)))))


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


