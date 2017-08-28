(ns machine-learning.tensorflow
  (:require [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.layers :as layer]
            [clojure-tensorflow.optimizers :as optimize]
            [clojure-tensorflow.core :refer [run with-graph with-session]]))



(def w (tf/variable [0.3]))
(def b (tf/variable [-0.3]))
(def x (tf/placeholder :x tf/float32))

(def linear-model (tf/add (tf/mult w x) b))

(run (tf/global-variables-initializer))

(run linear-model {:x [1. 2. 3. 4.]})

(def y (tf/placeholder :y tf/float32))

(def squared-deltas (tf/square (tf/sub linear-model y)))

(def loss (tf/mean squared-deltas))
(run (tf/assign w [-1.]))

(extend-type org.tensorflow.Output clojure.lang.IFn
             (-invoke [this args] (run this args)))

(type loss)

(run loss {:x [1. 2. 3.] :y [4. 3. 6.]})

