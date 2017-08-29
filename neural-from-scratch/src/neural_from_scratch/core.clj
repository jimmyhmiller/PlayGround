(ns neural-from-scratch.core
  (:require [clojure.core.matrix :refer [transpose] :as corem]
            [clojure.core.matrix.operators :refer [* + -]]
            [nd4clj.matrix :as imp]))


(corm/set-current-implementation :nd4j)

(def learning-rate 0.2)

(def input-neurons [1 0])
(def input-hidden-strengths [[0.12 0.2 0.13]
                             [0.01 0.02 0.03]])
(def hidden-neurons [0 0 0])
(def hidden-output-strengths [[0.15 0.16]
                              [0.02 0.03]
                              [0.01 0.02]])



(def activation-fn (fn [x] (Math/tanh x)))
(def dactivation-fn (fn [y] (- 1.0 (* y y))))

(defn layer-activation [inputs strengths]
  "forward propagate the input of a layer"
  (mapv activation-fn
      (mapv #(reduce + %)
       (* inputs (transpose strengths)))))



(def new-hidden-neurons
  (layer-activation input-neurons input-hidden-strengths))

(def new-output-neurons
  (layer-activation new-hidden-neurons hidden-output-strengths))

(def targets [0 1])


(defn output-deltas [targets outputs]
  "measures the delta errors for the output layer (Desired value – actual value) and multiplying it by the gradient of the activation function"
  (* (mapv dactivation-fn outputs)
     (- targets outputs)))

(output-deltas targets new-output-neurons)

(def odeltas (output-deltas targets new-output-neurons))


(defn hlayer-deltas [odeltas neurons strengths]
  (* (mapv dactivation-fn neurons)
     (mapv #(reduce + %)
           (* odeltas strengths))))


(def hdeltas (hlayer-deltas
              odeltas
              new-hidden-neurons
              hidden-output-strengths))



(defn update-strengths [deltas neurons strengths lrate]
  (+ strengths (* lrate
                  (mapv #(* deltas %) neurons))))


(def new-hidden-output-strengths
  (update-strengths
       odeltas
       new-hidden-neurons
       hidden-output-strengths
       learning-rate))


(def new-input-hidden-strengths
  (update-strengths
       hdeltas
       input-neurons
       input-hidden-strengths
       learning-rate))


(def nn [[0 0] 
         input-hidden-strengths 
         hidden-neurons 
         hidden-output-strengths 
         [0 0]])

(defn feed-forward [input network]
  (let [[in i-h-strengths h h-o-strengths out] network
        new-h (layer-activation input i-h-strengths)
        new-o (layer-activation new-h h-o-strengths)]
    [input i-h-strengths new-h h-o-strengths new-o]))



(defn update-weights [network target learning-rate]
  (let [[ in i-h-strengths h h-o-strengths out] network
        o-deltas (output-deltas target out)
        h-deltas (hlayer-deltas o-deltas h h-o-strengths)
        n-h-o-strengths (update-strengths
                         o-deltas
                         h
                         h-o-strengths
                         learning-rate)
        n-i-h-strengths (update-strengths
                         h-deltas
                         in
                         i-h-strengths
                         learning-rate)]
    [in n-i-h-strengths h n-h-o-strengths out]))



(defn train-network [network input target learning-rate]
  (update-weights (feed-forward input network) target learning-rate))


(def n1 (-> nn
     (train-network [1 0] [0 1] 0.5)
     (train-network [0.5 0] [0 0.5] 0.5)
     (train-network [0.25 0] [0 0.25] 0.5)))


(defn ff [input network]
  (last (feed-forward input network)))



(defn train-data [network data learning-rate]
  (if-let [[input target] (first data)]
    (recur
     (train-network network input target learning-rate)
     (rest data)
     learning-rate)
    network))


(defn inverse-data []
  (let [n (rand 1)]
    [[n 0] [0 n]]))



(defn gen-strengths [to from]
  (let [l (* to from)]
    (map vec (partition from (repeatedly l #(rand (/ 1 l)))))))

(defn construct-network [num-in num-hidden num-out]
  (vec (map vec [(repeat num-in 0)
             (gen-strengths num-in num-hidden)
             (repeat num-hidden 0)
             (gen-strengths num-hidden num-out)
             (repeat num-out 0)])))


(def tnn (construct-network 2 3 2))
(def n5 (train-data tnn (repeatedly 1000 inverse-data) 0.2))
(ff [1 0] n5)
