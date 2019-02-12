(ns micro-clj.server)

(defmulti -main 
  (fn [{:keys [uri request-method]}]
    [request-method uri]))

(defmethod -main [:get "/thing"] [req]
  {:thing true})

(defmethod -main :default [req]
  {:thing false})
