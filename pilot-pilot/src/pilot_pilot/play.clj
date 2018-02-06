(ns pilot-pilot.play)




(set! (var hello-world) (fn [] (println "hello var")))

(defn add-function [fn-name fn]
  (intern *ns* fn-name fn))

(add-function 'hello-world
              (fn [] (println "hello world")))
