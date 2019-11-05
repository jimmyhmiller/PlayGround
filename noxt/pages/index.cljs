(ns index
  (:require [noxt.lib :refer [Link]]))

(defn main []
  [:div [:h1 "Index"]
   [Link {:page :about} "about"]])
