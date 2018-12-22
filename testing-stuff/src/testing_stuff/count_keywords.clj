(ns testing-stuff.count-keywords
  (:require [taoensso.carmine :as car]))


(def server {:pool {}
             :spec {:uri "redis://localhost"}})

(car/wcar server (car/ping))



(car/wcar server
          (dotimes [i 10000000]
            (car/sadd "b" (rand-int 100000000))))




(car/wcar server
          (car/sadd "a" 1)
          (car/sadd "a" 2)
          (car/sadd "a" 3)
          (car/sadd "b" 1)
          (car/sadd "b" 2))


(let [result (str (gensym))]
  (car/wcar server
            (car/sinterstore result "a" "b")
            (car/del result)))

(str (gensym))

(car/wcar 
 server
 (car/ping)
 (car/set "foo" "bar")
 (car/get "foo"))
