(ns try-npm-deps.core
  (:require [reagent.core :as reagent]))

(enable-console-print!)

(println "This text is printed from src/try-npm-deps/core.cljs. Go ahead and edit it and see reloading in action.")

(defonce app-state (reagent/atom {:text "Hello world!"}))

(defn mount-root []
  (reagent/render [:div "Hello World!"] (.getElementById js/document "app")))

(defn init! []
  (mount-root))

(init!)
