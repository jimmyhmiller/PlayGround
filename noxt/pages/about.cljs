(ns about
  (:require [noxt.lib :refer [Link]]))

(defn main []
  [:div [:h1 "About"]
   [Link {:page :index} "index"]])
