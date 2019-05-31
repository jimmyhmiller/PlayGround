(ns shadow-debug.worker
  (:require [cljs.js :as cljs]
            [cljs.env :as env]
            [shadow.cljs.bootstrap.browser :as boot]
            [cljs.reader :refer [read-string]]))


(defonce compile-state-ref (env/default-compiler-env))


(defn eval-str [source cb]
  (cljs/eval-str
   compile-state-ref
   (str source)
   "[test]"
   {:eval cljs/js-eval
    :load (partial boot/load compile-state-ref)
    :ns   (symbol "shadow-debug.eval-env")}
   cb))

(defonce initialize-eval
  (js/Promise. 
   (fn [resolve error]
     (try
       (boot/init compile-state-ref
                  {:path         "/js/bootstrap"
                   :load-on-init '#{meander-editor.eval-env}}
                  resolve)
       (catch js/Object e
         (println "error initializing" e))))))


(.then initialize-eval
  (fn [_])
   (eval-str "(+ 2 2)" println))