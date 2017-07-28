(ns binary)

(defn valid-binary? [bin]
  (every? #{\1 \0} bin))

(defn to-decimal [bin]
  (if (valid-binary? bin)
    (read-string (str "2r" bin))
    0))
