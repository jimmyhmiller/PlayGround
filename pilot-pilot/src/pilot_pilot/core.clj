(ns pilot-pilot.core)



;; This is a complete joke right now.
;; Things that need to be extensible
;; get-advice
;; apply-advice


(def state
  (atom {:functions {}
         :stack '()
         :history {}
         :advice {}}))


(defmacro def-function [name args & body]
  (swap! state assoc-in [:functions name] 
         {:args args
          :body body
          :fn (eval `(fn ~args ~@body))})
  nil)

(def-function apply-advice [advice state]
  (println "advice" advice)
  (when advice
    ((:advice advice) state)))

(def-function get-advice [name state]
  (get-in state [:advice name]))

(defn call-function* [name args]
  ;(println "call" name)
  (let [current-state @state
        function (get-in current-state [:functions name :fn] (constantly nil))
        advice (if (or (= name 'get-advice) (= name 'apply-advice)) 
                 (get-in current-state [:advice name])
                 (call-function* 'get-advice [name current-state]))]
    (swap! state update :stack conj {:name name})
    (if advice
      (or (call-function* 'apply-advice [advice state])
          (apply function args))      
      (apply function args))))

(defmacro call-function [name & args]
  `(call-function* (quote ~name) ~args))

(defn def-advice* [advice-name fn-name advice]
  (swap! state assoc-in [:advice fn-name] 
         {:name advice-name
          :advice advice})
  nil)

(defmacro def-advice [advice-name fn-name advice]
  `(def-advice* (quote ~advice-name) (quote ~fn-name) ~advice))

(def-advice stop-em hello-world
  (fn [state] 
    (println "nope")
    :nope))

(def-advice all-advice get-advice
  (fn [state]
    (println "all-advice")
    (let [last-call (:name (first (drop 3 (:stack @state))))]
      (println (:stack @state))
      (if (and 
           (not= last-call 'get-advice)
           (not= last-call 'apply-advice))
        (println "asdfsadfdsfdsf")
        {:advice :end}))))

(def-function hello-world [] 
  (println "hello world"))

(call-function hello-world)
