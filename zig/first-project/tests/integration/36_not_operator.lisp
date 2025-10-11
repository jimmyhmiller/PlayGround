(def result1 (: Int) (if (not true) 1 0))
(def result2 (: Int) (if (not false) 1 0))
(printf (c-str "%d %d\n") result1 result2)
