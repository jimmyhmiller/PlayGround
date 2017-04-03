(ns hello-world.core
  (:require [cljs.nodejs :as nodejs])
  (:use-macros [hello-world.macros :only (??)]))

(nodejs/enable-util-print!)


(def props #js {:friends #js []})

(defn -main [& args]
  (println (?? props user friends 0 friends)))

(set! *main-cli-fn* -main)
