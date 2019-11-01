(ns cljs-schema.poll
  (:require [faunadb]
            [dotenv]
            [clojure.pprint :as pprint]
            [clojure.string :as string]))

(do
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
  (def Filter (.-Filter q))
  (def Lambda (.-Lambda q))
  (def Casefold (.-Casefold q))
  (def Query (.-Query q))
  (def Call (.-Call q))
  (def Function (.-Function q))
  (def Equals (.-Equals q))
  (def Not (.-Not q))
  (def If (.-If q))

  (def client (Client. #js{:secret (.. js/process -env -SECRET)}))

  (defn query [client query]
    (.query client (clj->js query)))

  (def debug (atom nil))

  (defn print-result [p]
    (-> p
        (.then (fn [val] (reset! debug val) val))
        (.catch (fn [val] (reset! debug val) val))
        (.then (comp pprint/pprint js->clj))
        (.catch (comp pprint/pprint js->clj)))))


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
                                                   :votes ["jimmy"]
                                                   :index 0}
                                                  {:value "Stuff"
                                                   :votes ["bob"]
                                                   :index 1}]}}))))


(defn get-poll-by-callback-id [callback-id]
  (Get (Match (Index "poll-by-callback") callback-id)))


(print-result)

(defn delete-all [index]
  (.then
   (query client (Paginate (Match (Index index)) #js {:size 1000}))
   (fn [xs]
     (doseq [x (.-data xs)]
       (query client (Delete x))))))





(defn filter-votes [option votes]
  (Filter (Var "votes")
          (Lambda "vote"
                  (Not (Equals (Var "voter") (Var "vote"))))))



(print-result
 (query client
        (Update
         (Function "remove-vote")
         (clj->js {:body
                   (Query
                    (Lambda #js ["options" "voter"]
                            (Map (Var "options")
                                 (Lambda "option"
                                         (Let (clj->js {:votes (Select #js ["votes"] (Var "option"))
                                                        :value (Select #js ["value"] (Var "option"))
                                                        :index (Select #js ["index"] (Var "option"))})
                                              (clj->js 
                                               {:votes (filter-votes (Var "votes") (Var "voter"))
                                                :value (Var "value")
                                                :index (Var "index")}))))))}))))


(print-result
 (query client
        (Update
         (Function "add-vote")
         (clj->js {:body
                   (Query
                    (Lambda #js ["options" "voter" "position"]
                            (Map (Var "options")
                                 (Lambda "option"
                                         (Let (clj->js {:index (Select #js ["index"] (Var "option"))
                                                        :votes (Select #js ["votes"] (Var "option"))
                                                        :value (Select #js ["value"] (Var "option"))})
                                              (clj->js 
                                               {:value (Var "value")
                                                :index (Var "index")
                                                :votes
                                                (If (Equals (Var "index") (Var "position"))
                                                    (Append (Var "votes")
                                                            #js [(Var "voter")])
                                                    (Var "votes"))}))))))}))))

(print-result
 (query client
        (Delete (Ref (Class "polls") "219394599553073677"))))


(print-result
 (query client
        (get-poll-by-callback-id "123")))



(print-result
 (query client
        (Update (Function "vote")
         (clj->js {:body
                  (Query
                   (Lambda #js ["callback-id" "position" "voter"]
                           (Let (clj->js {:poll (get-poll-by-callback-id (Var "callback-id"))
                                          :ref (Select #js ["ref"] (Var "poll"))
                                          :options (Select #js ["data" "options"] (Var "poll"))
                                          :removed (Call (Function "remove-vote") 
                                                         #js[(Var "options") 
                                                             (Var "voter")])
                                          :added (Call (Function "add-vote")
                                                       #js [(Var "removed")
                                                            (Var "voter")
                                                            (Var "position")])})
                                (Update (Var "ref")
                                        (clj->js {:data {:options (Var "added")}})))))}))))

(print-result
 (query client
        (Call (Function "vote")
              #js ["123" 0 "jimmy"])))

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



(print-result
 (query client
        (CreateIndex
         (clj->js
          {:name "user-by-access-token"
           :source (Class "users")
           :terms [{:field ["data" "access_token"]}]}))))


