(ns pilot-pilot.another-attempt)


(defn get-advice [{:keys [advice] :as state} {:keys [name] :as context}]
  {:state (update-in state [:functions 'call-function] 
                     (fn [call-function]
                       (fn [state context]
                         (call-function
                          state
                          (assoc context :advice (advice name))))))
   :context context})

(defn call-function [{:keys [functions] :as state} {:keys [name] :as context}]
  ((functions name) state context))


(def initial-state {:functions {'get-advice get-advice
                                'call-function call-function}
                    :advice {'get-advice [:advice]}})

(defn command [state command]
  ((get-in state [:functions 'call-function]) state command))


(command initial-state 
         {:name 'get-advice
          :args []})
