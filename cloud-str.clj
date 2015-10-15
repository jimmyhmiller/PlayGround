;; Anything you type in here will be executed
;; immediately with the results shown on the
;; right.


(def init-str
  [:bottom ""])


(def server (atom init-str))



(defn send-to-server [state]
  (let [forked (fork state)
        server-state (first forked)
        local-state (last forked)]
    (swap! server join server-state)
    local-state))


(defn set [s [r t]]
  [:written s])

(defn set-if-empty [s [r t]]
  (cond
   (and (= r :written) (= t "")) [:written s]
   (and (= r :bottom) (= t "")) [[:cond s] s]
   (and (= r :bottom) (not= t "")) [[:cond s] t]
   :else [r t]))

(defn get [[r s]]
  s)

(defn fork [[r s]]
  [[r s] [:bottom s]])

(defn get-cond [[[_ s] _]]
  s)

(defn join [[r1 s1] [r2 s2]]
  (cond
   (= r2 :written) [:written s2]
   (and (= r1 :written) (= s1 "") (= (first r2) :cond)) [:written (get-cond r2)]
   (and (= r1 :bottom) (= s1 "") (= (first r2) :cond)) [[:cond (get-cond r2)] (get-cond r2)]
   (and (= r1 :bottom) (not= s1 "") (= (first r2) :cond)) [[:cond (get-cond r2)] s1]
   :else [r1 s1]))

(->> init-str
     (set "test")
     (set-if-empty "test2")
     (set "")
     (set-if-empty "test2")
     (send-to-server)
     (set-if-empty "test3")
     (send-to-server)
     (set "stuff")
     (send-to-server))


@server