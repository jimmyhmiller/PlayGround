(ns space-age)

(def earth-year 31557600)

(defn converter [ratio]
  (fn [seconds]
    (/ seconds (* earth-year ratio))))

(def on-earth (converter 1))

(def on-mercury (converter 0.2408467))

(def on-venus (converter 0.61519726))

(def on-mars (converter 1.8808158))

(def on-jupiter (converter 11.862615))

(def on-saturn (converter 29.447498))

(def on-uranus (converter 84.016846))

(def on-neptune (converter 164.79132))
