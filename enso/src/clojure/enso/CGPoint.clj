(ns enso.CGPoint)

(gen-class 
 :name "enso.CGPoint"
 :extends com.sun.jna.Structure
 :constructors {[] []
                [double double] []}
 :prefix "cgpoint-"
 :init "init"
 :state state)


(defn cgpoint-init
  ([] (cgpoint-init 0 0))
  ([x y] [[] (atom {:x x :y y})]))

(defn cgpoint-getX [this]
  (get @(.state this) :x))

(defn cgpoint-getY [this]
  (get @(.state this) :y))

(defn cgpoint-setX [this x]
  (swap! (.state this) :assoc :x x))

(defn cgpoint-setY [this y]
  (swap! (.state this) :assoc :x y))

(compile (symbol "enso.CGPoint"))

(enso.CGPoint.)
