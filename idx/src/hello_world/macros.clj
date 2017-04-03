(ns hello-world.macros)

(defn string-or-int [si]
  (if (symbol? si) 
    (name si)
    si))

(defmacro ?? 
  ([obj prop1]
   (let [p1 (string-or-int prop1)]
     `(.. ~obj ~p1)))
  ([obj prop1 prop2]
   (let [p1 (string-or-int prop1)
         p2 (string-or-int prop2)]
     `(if (nil? (aget ~obj ~prop1))
        (aget ~obj ~p1)
        (aget ~obj ~p1 ~p2))))
  ([obj prop1 prop2 & props]
   (let  [p1 (string-or-int prop1)
          p2 (string-or-int prop2)]
     `(if (nil? (aget ~obj ~p1))
        (aget ~obj ~p1)
        (?? (aget ~obj ~p1) ~p2 ~@props)))))
