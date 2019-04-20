(ns meander-editor.worker
  (:require [cljs.js :as cljs]
            [cljs.env :as env]
            [shadow.cljs.bootstrap.browser :as boot]
            [cljs.reader :refer [read-string]]))


(defonce compile-state-ref (env/default-compiler-env))


(defn eval-str [source cb]
  (cljs/eval-str
   compile-state-ref
   (str "(let [results (atom nil)] (reset! results " source ") @results)")
   "[test]"
   {:eval cljs/js-eval
    :load (partial boot/load compile-state-ref)
    :ns   (symbol "meander-editor.eval-env")}
   cb))

(defonce initialize-eval 
  (js/Promise. (fn [resolve]
                 (boot/init compile-state-ref
                            {:path         "/js/bootstrap"
                             :load-on-init '#{meander-editor.eval-env}}
                            resolve))))


(defn eval-promise [source]
    (js/Promise. 
     (fn [resolve]
       (eval-str source (comp resolve (fn [{:keys [value message] :as result}]
                                        (if value 
                                          [:success value] 
                                          [:error result])))))))

(defn eval-meander-many [input lhs rhs]
  (.then (eval-promise (str "(fn [coll] (try [:success (meander/match coll " lhs " " rhs ")] (catch js/Object e [:error e])))"))
         (fn [[_ meander-fn]]
           (let [result (atom nil)
                 answer (map meander-fn (read-string input))]
             
             (reset! result answer)
            
             @result))))


(def meander-fn (atom nil))

(def has-updated (atom false))

(def state (atom {}))

(defmulti handle-message first)

(defmethod handle-message :lhs [[_ lhs]]
  (reset! has-updated true)
  (swap! state assoc :lhs lhs))

(defmethod handle-message :rhs [[_ rhs]]
  (println rhs)
  (reset! has-updated true)
  (swap! state assoc :rhs rhs))

(defmethod handle-message :input [[_ input]]
  (reset! has-updated true)
  (swap! state assoc :input (read-string input)))

(defmethod handle-message :default [data]
  (println "not found " data))


(defn create-function [lhs rhs]
  (eval-promise
   (str "(fn [coll] (try [:success (meander/match coll " lhs " " rhs ")] 
              (catch js/Object e [:error (:message e)])))")))

(add-watch state
           :compute 
           (fn [_ _ old-state state]

             (if (or (not= (:rhs old-state) (:rhs state))
                     (not= (:lhs old-state) (:lhs state)))
               (.then (create-function (:lhs state) (:rhs state))
                      (fn [[_ f]]
                        (reset! has-updated false)
                        (reset! meander-fn f))))))

(add-watch meander-fn :compute
           (fn [_ _ _ meander-fn]
             (js/postMessage (prn-str
                              (take-while #(not @has-updated)
                                          (map meander-fn (:input @state)))))))


;; If I save the input instead of sending it everytime things should be much faster.

(js/self.addEventListener "message"
  (fn [^js e]
    (let [message (read-string (.. e -data))]
      (println "Recieved message")
      (.then initialize-eval
             (fn [response]
               (handle-message message))))))
