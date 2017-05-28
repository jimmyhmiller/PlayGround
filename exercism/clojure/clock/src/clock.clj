(ns clock)

(defn calculate-hours [hours minutes]
  (-> hours
      (+ (/ minutes 60))
      (mod 24)
      int))

(defn pad-zeros [time]
  (format "%02d" time))

(defn clock [hours minutes]
  [(calculate-hours hours minutes) (mod minutes 60)])

(defn clock->string [[hours minutes]]
  (str (pad-zeros hours) ":" (pad-zeros minutes)))

(defn add-time [[hours minutes] more-minutes]
  (clock hours (+ minutes more-minutes)))
