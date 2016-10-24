(ns platformer.core
  (:require [om.core :as om :include-macros true]
            [om.dom :as dom :include-macros true]
            [sablono.core :as html :refer-macros [html]])
  (:use [platformer.utils :only [key-code]]))


(enable-console-print!)


(defn abs [x]
  (.abs js/Math x))

(defn pow [n x] (reduce * (repeat n x)))

(def square #(pow % 2))

(defn multi-assoc-in [coll ks kvs]
  (reduce (fn [coll [k v]]
            (assoc-in coll (conj ks k) v))
          coll
          kvs))


(defn compose-props [& reducers]
  (fn [object app]
    (->> reducers
      (map #(partial % object))
      (reduce (fn [a f] (f a)) app))))


(defn gravity [object app]
  (if (true? (:standing object))
    app
    (let [t (:t app)
          vy (max (+ (:vy object) (* -9.8 t)) 0)
          y (max
             (+ (:y object)
                vy
                (* .5 -9.8 (square t)))
             0)
          x (+ (:x object) (* (:vx object) t))]
      (multi-assoc-in app
                      [:objects (:name object)] {:y y :vy vy :x x}))))


(defn walk-with-velocity [velocity]
    (fn [object app]
      (let [keys (:keys app)
            vx (cond
                (contains? keys :right) velocity
                (contains? keys :left) (* -1 velocity)
                :else 0)
            vy (if
                 (and
                  (contains? keys :up)
                  (true? (:standing object)))
                 15
                 (:vy object))
            standing false]
        (multi-assoc-in app
                        [:objects (:name object)]
                        {:vx vx :vy vy :standing standing}))))

(def walk (walk-with-velocity 150))

(defn apply-attr [obj attr app]
  ((attr obj) obj app))

(defn remove-overrides [object app]
  (assoc-in app [:objects (:name object) :overrides] {}))

(defn main-character [object app]
  (let [new-object (merge object (:overrides object))
        walk (:walk new-object)
        gravity (:gravity new-object)]
    (->> app
         (walk object)
         (gravity object)
         (remove-overrides object))))


(defn top [object]
  (+ (:y object) (:height object)))

(defn bottom [object]
  (:y object))


(defn right [object]
  (+ (:x object) (:width object)))

(defn left [object]
  (:x object))


(defn between [x y z]
  (or
   (and (< x y) (> x z))
   (and (> x y) (< x z))))

(defn between-inclusive [x y z]
  (or
   (and (<= x y) (>= x z))
   (and (>= x y) (<= x z))))


(defn nearest [x y z]
  (if
    (>
     (.abs js/Math (- x y))
     (.abs js/Math (- x z)))
    z
    y))


(defn center-y [object]
  (+
   (:y object)
   (* 0.5 (:height object))))


(defn opposite-velocity [v]
  (+ v (* 2 (- 9.8 v))))


(defn fix-y [obj platform]
  (let [near (nearest
              (center-y obj) (top platform) (bottom platform))]
    (if (= near (bottom platform))
      (- (bottom platform) (:height obj))
      (top platform))))

(defn fix-left [obj platform]
  (assoc obj :x (- (left platform) (:width obj))))

(defn fix-right [obj platform]
  (assoc obj :x (right platform)))


(defn bounce [obj floor]
  (assoc obj
    :y (top floor)
    :vy (* .88 (opposite-velocity (:vy obj)))
    :standing true))


(defn stand [obj floor]
    (assoc obj
      :y (fix-y obj floor)
      :vy 9.8
      :standing (= (fix-y obj floor) (top floor))))

(defn fast [obj floor]
  (assoc obj
    :y (fix-y obj floor)
    :vy 9.8
    :standing (= (fix-y obj floor) (top floor))
    :overrides {:walk (walk-with-velocity 250)}))


(defn portal [x obj floor]
  (assoc obj
    :y (top floor)
    :vy (opposite-velocity (:vy obj))
    :x x
    :standing true))


(defn constrain-side
  [object platform side & {:keys [inclusive] :or {inclusive false}}]
  (let [sides {:right right
               :left left
               :top top
               :bottom bottom}
        opposites {:right left
                   :left right
                   :top bottom
                   :bottom top}
        _side (side sides)
        _opposite-side (side opposites)
        between-check (if (true? inclusive) between-inclusive between)]
    (between-check
     (_side object)
     (_side platform)
     (_opposite-side platform))))





(defn constrain-left [obj platform]
  (constrain-side obj platform :left {:inclusive true}))


(defn constrain-right [obj platform]
  (constrain-side obj platform :right {:inclusive true}))


(defn bound-horizontal [obj platform]
  (or
    (constrain-left obj platform)
    (constrain-right obj platform)))


(defn constrain-top [obj platform]
   (constrain-side obj platform :top))


(defn constrain-bottom [obj platform]
   (constrain-side obj platform :bottom))


(defn bound-vertical [obj platform]
  (or
   (constrain-bottom obj platform)
   (constrain-top obj platform)))


(defn bound [obj platform]
  (and
   (bound-horizontal obj platform)
   (bound-vertical obj platform)))



(defn nearest-side [obj platform]
  (let [sides (sides-constrained obj platform)
        diffs {:right (abs (- (left obj) (right platform)))
               :left (abs (- (right obj) (left platform)))
               :bottom (abs (- (top obj) (bottom platform)))
               :top (abs (- (bottom obj) (top platform)))}
        candidates (filter (fn [[k v]] (contains? sides k)) diffs)
        closest (reduce (fn [[ka a] [kb b]] (if (< a b) [ka a] [kb b])) candidates)]
    (first closest)))



(defn sides-constrained [obj platform]
  (let [sides {:right (constrain-left obj platform)
               :left (constrain-right obj platform)
               :bottom (constrain-top obj platform)
               :top (constrain-bottom obj platform)}]
    (->> sides
         (filter (fn [[k v]] (true? v)))
         (map (fn [[k _]] k))
         (into #{}))))



(defn collision [platform app]
  (let [result
        (into
         {}
         (map (fn [[k obj]]
                [k (bounds-fixes obj platform)])
              (:objects app)))]
    (assoc app :objects result)))


(defn bounds-fixes [obj platform]
  (if (bound obj platform)
      (let [fixes (:bounds platform)
            side (nearest-side obj platform)
            fix (side fixes)]
        (fix obj platform))
    obj))



(defn main-floor [object app]
  (->> app
      (collision object)))




(def character
  {:name :character
   :width 10
   :height 10
   :gravity gravity
   :walk walk
   :x 0
   :y 100
   :vy 9.8
   :vx 0
   :color "#000"
   :shape :circle
   :standing true
   :main main-character})


(def floor
  {:name :floor
   :width 600
   :height 10
   :x 0
   :y 0
   :color "#000"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top stand
            :bottom stand}})


(def floor1
  {:name :floor1
   :width 100
   :height 10
   :x 20
   :y 70
   :color "#000"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top stand
            :bottom stand}})


(def floor2
  {:name :floor2
   :width 100
   :height 10
   :x 70
   :y 140
   :color "#000"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top stand
            :bottom stand}})


(def floor3
  {:name :floor3
   :width 100
   :height 30
   :x 200
   :y 10
   :color "#F00"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top (partial portal 400)
            :bottom stand}})


(def floor4
  {:name :floor4
   :width 100
   :height 30
   :x 350
   :y 10
   :color "#00F"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top (partial portal 250)
            :bottom stand}})


(def floor5
  {:name :floor5
   :width 100
   :height 10
   :x 450
   :y 200
   :color "#000"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top stand
            :bottom stand}})


(def floor6
  {:name :floor6
   :width 400
   :height 10
   :x 600
   :y 0
   :color "#0F0"
   :shape :square
   :main main-floor
   :bounds {:left fix-left
            :right fix-right
            :top bounce
            :bottom stand}})


(defn create-object [object browser]
  (html [:div
         {:style
          {:width (:width object)
           :height (:height object)
           :background-color (:color object)
           :border-radius (if (= (:shape object) :circle) "50%" "0%")
           :position "absolute"
           :left (:x object)
           :top (- (:height browser) (:y object) (:height object))}}]))


(defn game-loop [app objects]
  (if
    (zero? (count objects))
    app
    (let [[_ object] (first objects)]
      (game-loop ((:main object) object app) (rest objects)))))


(defn update-game [app]
  (om/transact! app #(game-loop % (merge (:objects %) (:immoveable %)))))


(defn update-key-down [e app]
  (.preventDefault e)
  (om/transact! app [:keys] (fn [keys] (conj keys (key-code (aget e "keyCode"))))))


(defn update-key-up [e app]
  (.preventDefault e)
  (om/transact! app [:keys] (fn [keys] (disj keys (key-code (aget e "keyCode"))))))


(defn game [app owner]
  (reify
    om/IWillMount
    (will-mount [this]
                (.addEventListener js/window "resize"
                                   #(om/update! app [:browser]
                                                {:width (aget js/window "innerWidth")
                                                 :height (aget js/window "innerHeight")}))
                (.addEventListener js/window "keydown" #(update-key-down % app))
                (.addEventListener js/window "keyup" #(update-key-up % app))
                (om/update! app [:browser]
                            {:width (aget js/window "innerWidth")
                             :height (aget js/window "innerHeight")})
                (.setInterval js/window #(update-game app) 16))
    om/IRender
    (render [this]
            (html [:div
                   (let [create
                         (fn [[k v]]
                           (create-object v (:browser app)))]
                     (concat
                      (map create (:objects app))
                      (map create (:immoveable app))))]))))


(def app-state (atom {:t 0.016
                      :keys #{}
                      :immoveable {:floor floor
                                   :floor1 floor1
                                   :floor2 floor2
                                   :floor3 floor3
                                   :floor4 floor4
                                   :floor5 floor5
                                   :floor6 floor6}
                      :objects {:character character}
                      :browser {:width 0
                                :height 0}}))


(om/root
 game
 app-state
 {:target (. js/document (getElementById "app"))})


