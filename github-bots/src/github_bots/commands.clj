(ns github-bots.commands
  (:require [tentacles.core :as core]
            [environ.core :refer [env]]
            [tentacles.issues :as issues]
            [clojure.pprint :as pprint :refer [pprint print-table]])
  (:use [github-bots.core]
        [clojure.string :only [split lower-case]]))

(def auth (str (env :username) ":" (env :password)))



(defmulti command (fn [[command & _] issue-number comment] command))

(defmethod command "/open" [_ issue-number _]
  (add-label issue-number "Open For Voting"))

(defmethod command "/roll" [[_ description] issue-number {:keys [user]}]
  (add-comment issue-number (roll-comment description (:login user))))

(defmethod command "yay" [_ issue-number {:keys [user]}]
  (when (active? (:login user))
    (swap-vote-status-label issue-number)))

(defmethod command "nay" [_ issue-number _]
  (when (active? (:login user))
    (swap-vote-status-label issue-number)))

(defn comment->command [comment]
  (let [{:keys [issue body]} comment]
    (command (words (lower-case body)) (:number issue) comment)))

(core/with-defaults {:auth auth}
  (comment->command {:body "nay"
                     :issue {:number 1}
                     :user {:login "jimmyhmiller"}}))

(defn spit-clipboard [text]
  (.setContents (get-clipboard) (java.awt.datatransfer.StringSelection. text) nil))


(core/with-defaults {:auth auth :accept "application/vnd.github.squirrel-girl-preview"}
  (->>
   (issues/my-issues {:filter "created"})
   (filter #(and (> (-> % :assignees count) 0) (= (-> % :reactions :+1) (-> % :assignees count))))
   (map :url)))

(core/with-defaults {:auth auth :accept "application/vnd.github.squirrel-girl-preview"}
  (->>
   (issues/my-issues {:filter "created"})
   (filter #(and (> (-> % :assignees count) 0) (< (-> % :reactions :+1) (-> % :assignees count))))
   (filter #(contains? % :pull_request))
   (map :url)))

