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

;; This works except for this issue:
;; https://github.com/thheller/shadow-cljs/blob/da2fc8f8b10daeee0c984a10f9437a579933931c/src/main/shadow/cljs/bootstrap/browser.cljs#L37-L43


;; If I save the input instead of sending it everytime things should be much faster.

(js/self.addEventListener "message"
  (fn [^js e]
    (let [message (.. e -data)
          {:keys [input lhs rhs]} (read-string message)]
      (println "Recieved message")
      (.then initialize-eval
             (fn [response]
               (println "initialized")
               (try
                 (.then (eval-meander-many input lhs rhs)
                        (fn [x] (js/postMessage (prn-str x))))
                 (catch js/Object e (println e))))))))
