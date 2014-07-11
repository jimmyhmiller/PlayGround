(ns flock.core
  (:require
   [figwheel.client :as fw]
   [om.core :as om :include-macros true]
   [om.dom :as dom :include-macros true]
   [sablono.core :as html :refer-macros [html]]))

(enable-console-print!)


(defonce pi Math/PI)
(defonce MAXACC 10)
(defonce MAXDISTANCE 100)
(defonce MAXSTEER (/ pi 100))
(defonce SCALE 1)

(defn square [x]
  (* x x))

(defn y-comp [{:keys [v O]}]
  (let [com (* v (Math/cos O))]
    (* -1 com)))


(defn x-comp [{:keys [v O]}]
  (let [com (* v (Math/sin O))]
    com))


(defn move [x vx a t]
  (+
   x
   (* vx t)
   (* .5 a (square t))))


(defn move-x [{:keys [x v]} t]
  (let [vx (x-comp v)]
    (move x vx 0 t)))

(defn move-y [{:keys [y v]} t]
  (let [vy (y-comp v)]
    (move y vy 0 t)))


(defn avg [coll]
  (/ (reduce + coll) (count coll)))


(defn direction [object]
  (:O (:v object)))

(defn avg-heading [coll]
  (->> coll
       (map direction)
       (avg)))

(defn distance [object1 object2]
  (Math/sqrt
   (+
    (square (- (:x object1) (:x object2)))
    (square (- (:y object1) (:y object2))))))



(defn neighbors [object coll]
  (filter
   #(and (< (distance object %) MAXDISTANCE)
         (not= (distance object %) 0))
   coll))

(defn steer [object coll]
  (let [direction (direction object)
        heading (avg-heading (neighbors object coll))
        distance (Math/abs (- direction heading))]
    (cond
     (< distance MAXSTEER)
     distance
     (< direction heading)
     MAXSTEER
     (> direction heading)
     (* -1 MAXSTEER)
     :else 0)))



(defn step [object coll t]
  (let [x  (move-x object t)
        y  (move-y object t)
        v (:v (:v object))
        O (mod (+ (:O (:v object)) (steer object coll) ) (* 2 pi))]
    (assoc object
      :x x
      :y y
      :v {:v v :O O})))





(defn triangle [x y direction key]
  [:div
   {:key key
    :style
    {:position "absolute"
     :left (* x SCALE)
     :top (* y SCALE)
     :width 0
     :height 0
     :font-size 5
     :border-left "5px solid transparent"
     :border-right "5px solid transparent"
     :border-bottom "5px solid black"
     :transform (str "rotate(" direction "rad) scaleY(" 3 ")")}}])



(defn main-loop [app]
  (let [objects (:objects @app)]
    (om/update! app [:objects] (map #(step % objects 0.032) objects))))


(defn flock [app owner]
  (reify
    om/IWillMount
    (will-mount
     [this]
     (.addEventListener js/window "resize"
                        #(om/update! app [:browser]
                                     {:width (aget js/window "innerWidth")
                                      :height (aget js/window "innerHeight")}))
     (om/update! app [:browser]
                 {:width (aget js/window "innerWidth")
                  :height (aget js/window "innerHeight")})
     (.setInterval js/window #(main-loop app) 110))
    om/IRender
    (render
     [this]
     (html [:div (triangle 20 20 pi)
            (map (fn [{:keys [x y v key]}] (triangle (mod x (->> app :browser :width)) y (:O v) key)) (:objects app))]))))



(defonce app-state (atom {:objects
                          (map (fn [d] {:x 300 :y 300 :v {:v 50 :O d} :key d} ) (range 0 (* 2 pi) 0.5))

                          :browser {:width 0
                                    :height 0}}))



(om/root
 flock
 app-state
 {:target (. js/document (getElementById "app"))})


(fw/watch-and-reload
 :jsload-callback (fn []
                    ;; (stop-and-start-my app)
                    ))
