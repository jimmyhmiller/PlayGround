(ns github-bots.commands
  (:require [tentacles.core :as core])
  (:use [github-bots.core]))


(def auth "nomic-awesome-bot:correct horse nomic staple")

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


(core/with-defaults {:auth auth :branch "feature/test-out-bot" :ref "feature/test-out-bot"}
  (update-points {:pull_request {:title "304" :number 35}
                  :action "closed"}))
