(ns micro-clj.server)

(defmulti -main (fn [{:keys [uri]}] uri))

(defmethod -main "/thing" [req]
  {:thing true})

(defmethod -main :default [req]
  {:thing false})
