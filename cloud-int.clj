;; Anything you type in here will be executed
;; immediately with the results shown on the
;; right.


(def init-int
  [false 0 0])

(def server (atom init-int))



(defn add [n [r b d]]
  [r b (+ d n)])

(defn set [n [r b d]]
  [true n 0])

(defn get [[r b d]]
  (+ b d))


(defn fork [[r b d]]
  [[r b d] [false (+ d b) 0]])

(defn join [[r1 b1 d1] [r2 b2 d2]]
  (if r2
    [true b2 d2]
    [r1 b1 (+ d1 d2)]))


(defn send-to-server [state]
  (let [forked (fork state)
        server-state (first forked)
        local-state (last forked)]
    (swap! server join server-state)
    local-state))


(->> init-int
     (add 2)
     (add 2)
     (send-to-server)
     (send-to-server)
     (send-to-server)
     (send-to-server)
     (send-to-server)
     (add 3)
     (send-to-server)
     (set 3)
     (set 1)
     (add 2)
     (send-to-server))


(get @server)
