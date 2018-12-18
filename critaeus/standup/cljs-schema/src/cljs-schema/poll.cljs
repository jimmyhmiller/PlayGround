(ns cljs-schema.poll
  (:require [faunadb]
            [dotenv]
            [clojure.pprint :as pprint]
            [clojure.string :as string]))

(.config dotenv)

(def q (.. faunadb -query))
(def Client (.-Client faunadb))
(def CreateClass (.-CreateClass q))
(def CreateDatabase (.-CreateDatabase q))
(def CreateFunction (.-CreateFunction q))
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
(def Date (.-Date q))
(def Ref (.-Ref q))
(def Paginate (.-Paginate q))
(def Map (.-Map q))
(def Lambda (.-Lambda q))
(def Casefold (.-Casefold q))
(def Query (.-Query q))
(def Call (.-Call q))
(def Function (.-Function q))

(def client (Client. #js{:secret (.. js/process -env -SECRET)}))

(defn query [client query]
  (.query client (clj->js query)))

(def debug (atom nil))

(defn print-result [p]
  (-> p
      (.then (fn [val] (reset! debug val) val))
      (.catch (fn [val] (reset! debug val) val))
      (.then (comp pprint/pprint js->clj))
      (.catch (comp pprint/pprint js->clj))))


(print-result
 (query client (CreateDatabase #js {:name "polls"})))


(print-result
 (query client (CreateClass #js {:name "polls"})))

(print-result
 (query client (Create (Class "polls") 
                       (clj->js {:data {:callback_id "123"
                                        :question "What is is is?"
                                        :anonymous true
                                        :options [{:value "Thing"
                                                   :votes ["jimmy"]}
                                                  {:value "Stuff"
                                                   :votes ["bob"]}]}}))))


(defn get-poll-by-callback-id [callback-id]
  (Get (Match (Index "poll-by-callback") callback-id)))



(defn add-vote [poll position voter]
  (update-in
   poll
   [:data :options position :votes]
   conj voter))

(add-vote (js->clj @debug :keywordize-keys true) 0 "bob")

(print-result
 (query client
        (Let (clj->js {:callback-id "123"
                       :position 0
                       :voter "bob" 
                       :poll (get-poll-by-callback-id (Var "callback-id"))})
             
             (Var "poll"))))

(print-result
 (query client
        (CreateIndex
         (clj->js
          {:name "poll-by-callback"
           :source (Class "polls")
           :terms [{:field ["data" "callback_id"]}]
           :values [{:field ["data" "question"]}
                    {:field ["data" "anonymous"]}
                    {:field ["data" "options"]}]}))))
