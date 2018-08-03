(ns micro-clj.server)

(defn -main [req]
  {:status  200
   :headers {"Content-Type" "application/json"}
   :body    {:test "hello http!"}})
