(ns bud.core)


;; 101ad1985ebccbd5f814365a89c6b5af293b5dbe

(defn scratch [name keys cols]
  (table name keys cols :scratch))

(defn table
  ([name keys cols]
   (table name keys cols :table))
  ([name keys cols persist]))





(def state
  (atom {:tables {}
         :strata []
         :channels {}
         :budtime 0
         :ip nil
         :port nil
         :connections {}
         :inbound []
         :periodics (table :periodics [:name] [:ident :duration])
         :vars (table :vars [:name] [:value])
         :tmpvars (scratch :tmpvars [:name] [:value])}))




