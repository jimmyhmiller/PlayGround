(defproject deeplearn "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.deeplearning4j/deeplearning4j-zoo "0.9.0"]
                 [org.nd4j/nd4j-native-platform "0.9.0"]
                 [org.nd4j/nd4j-api "0.9.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.9.0" 
                  :exclusions [org.nd4j/nd4j-api org.apache.commons/commons-compress]]
                 [org.datavec/datavec-api "0.9.0" 
                  :exclusions [org.apache.commons/commons-compress]]
                 [com.stuartsierra/log.dev "0.2.0"]
                 [org.deeplearning4j/deeplearning4j-ui_2.10 "0.9.0"]])
