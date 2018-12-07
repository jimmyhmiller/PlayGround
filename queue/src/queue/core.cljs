(ns queue.core
  (:require [faunadb]
            [dotenv]
            [clojure.pprint :as pprint]))

(.config dotenv)

(def q (.. faunadb -query))
(def Client (.-Client faunadb))
(def CreateClass (.-CreateClass q))
(def Create (.-Create q))
(def CreateIndex (.-CreateIndex q))
(def Index (.-Index q))
(def Class (.-Class q))
(def Get (.-Get q))
(def Match (.-Match q))
(def Update (.-Update q))
(def Delete (.-Delete q))
(def Select (.-Select q))
(def Append (.-Append q))
(def Let (.-Let q))
(def Var (.-Var q))

(def client (Client. #js{:secret (.. js/process -env -SECRET)}))
(defn query [client query]
  (.query client query))

(def debug (atom nil))

(defn print-result [p]
  (-> p
      (.then (fn [val] (reset! debug val) val))
      (.catch (fn [val] (reset! debug val) val))
      (.then (comp pprint/pprint js->clj))
      (.catch (comp pprint/pprint js->clj))))

(print-result
 (.query client (CreateClass #js{:name "queues"})))


(print-result
 (query client (Create (Class "queues") (clj->js {:data {:name "my-queue" :queue []}}))))

(print-result
 (query client (CreateIndex (clj->js {:name "queue-by-name" 
                                      :source (Class "queues")
                                      :terms [{:field [:data :name]}]
                                      :values [{:field [:data :name]}
                                               {:field [:data :queue]}]
                                      :unique true}))))

(print-result
 (query client (Delete (Index "queue-by-name"))))

(print-result
 (query client
        (Let (clj->js {:queue (Get (Match (Index "queue-by-name") "my-queue"))}) 
             (Update (Select (clj->js ["ref"]) (Var "queue"))
                     (clj->js 
                      {:data {:queue 
                              (Append (clj->js ["asdfdsafas"]) 
                                      (Select (clj->js [:data :queue]) (Var "queue")))}})))))


(print-result 
 (query client (Get (Match (Index "queue-by-name") "my-queue"))))

(pprint/pprint)
(js->clj
 (.parse js/JSON
         (.stringify js/JSON @debug)))
