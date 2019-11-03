(ns about
  (:require [cljs.loader :as loader]
            [noxt.lib :refer [Link]]))

(defn ^:export main []
  [:div [:h1 "About"]
   [Link {:page :index} "index"]])


;; Need to make a different loader module
;; Should autogen these
(loader/set-loaded! :about)

