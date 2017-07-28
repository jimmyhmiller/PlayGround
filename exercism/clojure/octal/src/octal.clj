(ns octal)

(defn valid-trinary? [bin]
  (every? #{\7 \6 \5 \4 \3 \2 \1 \0} bin))

(defn to-decimal [tri]
  (if (valid-trinary? tri)
    (read-string (str "8r" tri))
    0))
