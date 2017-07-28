(ns trinary)

(defn valid-trinary? [bin]
  (every? #{\2 \1 \0} bin))

(defn to-decimal [tri]
  (if (valid-trinary? tri)
    (read-string (str "3r" tri))
    0))
