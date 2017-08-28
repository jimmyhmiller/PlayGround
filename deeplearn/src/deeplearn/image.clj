(ns deeplearn.image
  (:require [clojure.reflect :as r])
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
           [org.deeplearning4j.optimize.listeners 
            ScoreIterationListener PerformanceListener]
           [org.deeplearning4j.optimize.api IterationListener]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.nd4j.linalg.activations Activation]
           [org.datavec.image.loader BaseImageLoader]
           [java.util Random]
           [org.datavec.api.io.labels ParentPathLabelGenerator]
           [org.datavec.api.io.filters BalancedPathFilter]
           [org.datavec.image.recordreader ImageRecordReader]
           [org.deeplearning4j.zoo.model GoogLeNet VGG16 LeNet ResNet50 SimpleCNN]
           [org.nd4j.linalg.dataset.api.preprocessor VGG16ImagePreProcessor]
           [org.deeplearning4j.ui.api UIServer]
           [org.deeplearning4j.ui.storage InMemoryStatsStorage]
           [org.deeplearning4j.ui.stats StatsListener]))


(def ui-server (UIServer/getInstance))
(def stats-storage (InMemoryStatsStorage.))


(def allowed-extensions BaseImageLoader/ALLOWED_FORMATS)
(def seed 12345)
(def rand-num-gen (Random. seed))
(def height 244)
(def width 244)
(def channels 3)

(def parent-dir (File. "/Users/jimmy.miller/Desktop/images"))

(def label-maker (ParentPathLabelGenerator.))

(def files-in-dir (FileSplit. parent-dir allowed-extensions rand-num-gen))

(def path-filter (BalancedPathFilter. rand-num-gen allowed-extensions label-maker))

(def files-in-dir-split (.sample files-in-dir path-filter (double-array [80 20])))



(def train-data (aget files-in-dir-split 0))
(def test-data (aget files-in-dir-split 1))





(def record-reader (ImageRecordReader. height width channels label-maker))
(.initialize record-reader train-data)

(def test-record-reader (ImageRecordReader. height width channels label-maker))
(.initialize test-record-reader test-data)


(def output-num (.numLabels record-reader))
(def test-output-num (.numLabels record-reader))

(def batch-size 5)
(def label-index 1)

(def data-iter 
  (RecordReaderDataSetIterator. record-reader batch-size label-index output-num))

(def test-iter 
  (RecordReaderDataSetIterator. 
   test-record-reader 
   batch-size 
   label-index 
   test-output-num))



(def iterations 5)

(def model 
  (-> (doto (LeNet. output-num seed iterations)
        (.setInputShape (into-array (map int-array [[3 244 244]]))))
      (.init)))



(comment (.setListeners model (into-array IterationListener 
                                          [(ScoreIterationListener. 10) 
                                           (PerformanceListener. 1 true)])))


(.setListeners 
 model (into-array IterationListener [(StatsListener. stats-storage 1)]))

(.attach ui-server stats-storage)

(comment)

(while (.hasNext data-iter)
  (.fit model (.next data-iter)))



(comment 

  (.evaluate model test-iter))
 
