(ns index
  (:require [cljs.loader :as loader]
            [noxt.lib :refer [Link]]))


(defn ^:export main []
  [:div [:h1 "Index"]
   [Link {:page :about} "about"]])


(loader/set-loaded! :index)
