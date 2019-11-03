(ns about
  (:require [cljs.loader :as loader]
            [noxt.lib :refer [Link]]))

(defn ^:export main []
  [:div [:h1 "About"]
   [Link {:page :index} "index"]])

(loader/set-loaded! :about)
