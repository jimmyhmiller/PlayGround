(ns cellular-traffic.core
  (:require ))

(enable-console-print!)

(println "This text is printed from src/cellular-traffic/core.cljs. Go ahead and edit it and see reloading in action.")

;; define your app data so that it doesn't get over-written on reload

(defonce app-state (atom {:text "Hello world!"}))


(defrecord Car [position velocity])

(defn build-cars-list [cars]
  (conj (into [] cars) (first cars)))

(def cars (build-cars-list [(Car. 0 0) (Car. 1 0)]))

(def v-max 5)
(def track-length 24)



(defn steps [track-length v-max cars]
  (let [next-cars (many-step track-length v-max cars)
        last-car (step track-length v-max (last cars) (first next-cars))]
    (sort-by :position (conj next-cars last-car))))




(defn n-steps [track-length v-max n cars]
  (if (zero? n)
    (butlast cars)
    (let [new-cars (many-step track-length v-max cars)]
      (n-steps track-length v-max (dec n) (build-cars-list new-cars)))))

(defn many-step [track-length v-max cars]
  (into [] (map (partial step track-length v-max) cars (next cars))))


(defn calculate-gap [track-length car next-car]
  (let [current-position (:position car)
        next-car-position (:position next-car)
        gap (- next-car-position current-position 1)]
    (if (neg? gap)
      (+ track-length gap)
      gap)))





(defn calculate-velocity [velocity gap v-max]
  (cond (and  
         (< velocity v-max)
         (>= gap (+ velocity 1))) (+ velocity 1)
        (> velocity (- gap 1)) gap
        :else velocity))

(defn calculate-position [position velocity track-length]
  (mod (+ velocity position) track-length))





(defn step [track-length v-max car next-car]
  (let [gap (calculate-gap track-length car next-car)
        velocity (calculate-velocity (:velocity car) gap v-max)]
    (-> car
        (update :position calculate-position velocity track-length)
        (assoc :velocity velocity))))





(take 11 (iterate (partial steps track-length v-max) 
                  [(Car. 0 3) 
                   (Car. 4 2)
                   (Car. 7 2)
                   (Car. 11 3)
                   (Car. 13 1)
                   (Car. 17 1)
                   (Car. 20 3)]))



(defn on-js-reload []
  ;; optionally touch your app-state to force rerendering depending on
  ;; your application
  ;; (swap! app-state update-in [:__figwheel_counter] inc)
)
