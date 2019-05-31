(ns meander-editor.worker
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
    :ns   (symbol "meander-editor.eval-env")}
   cb))

(defonce initialize-eval
  (do
    (js/Promise. (fn [resolve error]
                   (println "initializing2")
                   (try
                     (boot/init compile-state-ref
                                {:path         "/js/bootstrap"
                                 :load-on-init '#{meander-editor.eval-env}}
                                (fn [x] (println x)  (resolve x)))
                     (catch js/Object e
                       (println "error initializing" e)))))))


(defn eval-promise [source]
  (println source)
  (js/Promise. 
   (fn [resolve]
     (eval-str source (comp resolve (fn [{:keys [value message] :as result}]
                                      (if value 
                                        [:success value] 
                                        [:error result])))))))



(defn handle-error [error]
  (if (-> error :error ex-data :tag)
    (-> error :error ex-cause ex-cause cljs.repl/Error->map (select-keys [:cause :data]))
    (ex-message error)))

(defn eval-main [input]
  (println input)
  (.then (eval-promise (str input 
                            "\n-main"))
         (fn [[_ meander-fn]]
           (if (fn? meander-fn)
             (try (meander-fn)
                  (catch js/Object e (handle-error e)))
             (handle-error meander-fn)))))



(defn handle-message [input]
  (.then
   (eval-main input)
   (fn [output]
     (js/postMessage
      (prn-str output)))))


(js/self.addEventListener "message"
  (fn [^js e]
    (let [message (read-string (.. e -data))]
      (println "Recieved message" message)
      (.catch
       (.then initialize-eval
              (fn [response]
                (println "got here")
                (handle-message message)))
       (fn [error]
         (println error))))))
