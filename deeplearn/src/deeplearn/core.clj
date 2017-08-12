(ns deeplearn.core
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.datavec.api.util ClassPathResource]
           [org.datavec.api.records.reader.impl.csv CSVRecordReader]
           [org.datavec.api.split FileSplit]
           [java.io File]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.optimize.api IterationListener]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.nd4j.linalg.activations Activation]))


(def batch-size 50)
(def num-inputs 2)
(def num-outputs 2)
(def num-hidden-nodes 20)

(def train-filename 
  (-> (ClassPathResource. "linear_data_train.csv")
      (.getFile)
      (.getPath)))

(def test-filename
  (-> (ClassPathResource. "linear_data_eval.csv")
      (.getFile)
      (.getPath)))



(def rr (doto (CSVRecordReader.)
          (.initialize (FileSplit. (File. train-filename)))))

(def rr-test (doto (CSVRecordReader.)
               (.initialize (FileSplit. (File. test-filename)))))


(def train-iter (RecordReaderDataSetIterator. rr batch-size 0 2))
(def test-iter (RecordReaderDataSetIterator. rr-test batch-size 0 2))


(def first-layer 
  (-> (DenseLayer$Builder.)
      (.nIn num-inputs)
      (.nOut num-hidden-nodes)
      (.build)))

(def output-layer
  (-> (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
      (.weightInit WeightInit/XAVIER)
      (.activation Activation/SOFTMAX)
      (.weightInit WeightInit/XAVIER)
      (.nIn num-hidden-nodes)
      (.nOut num-outputs)
      (.build)))


(def conf
  (-> (NeuralNetConfiguration$Builder.)
      (.iterations 1)
      (.weightInit WeightInit/XAVIER)
      (.activation "relu")
      (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
      (.learningRate 0.05)
      (.list)
      (.layer 0 first-layer)
      (.layer 1 output-layer)
      (.backprop true)
      (.build)))


(def model (MultiLayerNetwork. conf))

(doto model
  (.init)
  (.setListeners (into-array IterationListener [(ScoreIterationListener. 10)])))


(.fit model train-iter)

(.reset test-iter)

(.output model (.getFeatures (.next test-iter)))
(.score model)


