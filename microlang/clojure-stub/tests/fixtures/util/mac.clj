(ns util.mac)
(defmacro twice [x] (list '+ x x))
(defn sq [n] (* n n))
