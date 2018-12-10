(ns cljs-schema.core
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
 (query client
        (Create (Class "teams") {:data {:name "stuff"}})))

(print-result
 (query client
        (Create (Class "statuses")
                (clj->js
                 {:data {:user (Ref (Class "users") "217993811723289100")
                         :date (Date (today))
                         :team (Ref (Class "teams") "217993851172815371")
                         :status "I did something today"}}))))




(print-result
 (query client
        (Update (Ref (Class "statuses") "218148332424397323")
                {:data {:status "I really did nothing today"}})))

(print-result
 (query client
        (CreateIndex
         (clj->js
          {:name "statuses-by-team-date"
           :source (Class "statuses")
           :terms [{:field ["data" "team"]}
                   {:field ["data" "date"]}]
           :values [{:field ["data" "user"]}
                    {:field ["data" "status"]}]}))))



(print-result
 (query client
        (CreateIndex
         (clj->js
          {:name "team-by-name"
           :source (Class "teams")
           :terms [{:field ["data" "name"] :transform "casefold"}]
           :values [{:field ["ref"]}]
           :unique true}))))


(defn team-by-name [name]
  (Select #js ["ref"]
          (Get (Match (Index "team-by-name")
                      (Casefold name)))))

(print-result
 (query client
        (CreateIndex
         (clj->js
          {:name "user-by-name"
           :source (Class "users")
           :terms [{:field ["data" "name"] :transform "casefold"}]
           :values [{:field ["ref"]}]
           :unique true}))))

(defn user-by-name [name]
  (Select #js ["ref"]
          (Get (Match (Index "user-by-name")
                      (Casefold name)))))



(print-result
 (query client
        (Let (clj->js {:user (user-by-name "jimmy")
                      :team (team-by-name "Awesome")})
             (Create (Class "statuses")
                     (clj->js
                      {:data {:user (Var "user")
                              :date (Date (today))
                              :team (Var "team")
                              :status "I made a function today"}})))))


(->> (user-by-name "jimmy")
     (query client)
     print-result)

(print-result
 (query client))

(->
 (query client
        (CreateIndex
         (clj->js
          {:name "status-index"
           :source (Class "statuses")}))))

(js->clj
 (.parse js/JSON (.stringify js/JSON @debug)))

(print-result
 (query client (Get (Ref (Class "teams") "217993811723289100"))))


(->>
 (Get (Function "add-status"))
 (query client)
 print-result)

(->>
 (CreateFunction
  (clj->js {:name "add-status"
                   :body
                   (Query
                    (Lambda #js ["user-name" "team-name" "date" "status"]
                            (Let (clj->js {:user (user-by-name (Var "user-name"))
                                           :team (team-by-name (Var "team-name"))})
                                 (Create (Class "statuses")
                                         (clj->js
                                          {:data {:user (Var "user")
                                                  :date (Date (Var "date"))
                                                  :team (Var "team")
                                                  :status (Var "status")}})))))}))
 (query client)
 print-result)

(-> 
 (query client
        (CreateFunction
         (clj->js {:name "get-team-status" 
                   :body 
                   (Query (Lambda #js ["date" "team-name"]
                                  (Let (clj->js {:team (team-by-name (Var "team-name"))})
                                       (Map
                                        (Paginate
                                         (Match
                                          (Index "statuses-by-team-date")
                                          (Var "team")
                                          (Date (Var "date"))))
                                        (Lambda (clj->js ["user" "status"]) 
                                                #js [(Select (clj->js ["data", "name"]) (Get (Var "user")))
                                                     (Var "status")])))))})))
 print-result)

(->
 (.query client
         (Let (clj->js {:team (Select #js ["ref"] (Get (Match (Index "team-by-name") (Casefold "Awesome"))))})
              (Map
               (Paginate
                (Match
                 (Index "statuses-by-team-date")
                 (Var "team")
                 (Date "2018-12-09")))
               (Lambda (clj->js ["user" "status"]) 
                       #js [(Select (c-lj->js ["data", "name"]) (Get (Var "user")))
                            (Var "status")]))))
 (.then (fn [x] (println x))))


(defn get-team-status [date team-name]
  (->> (Call (Function "get-team-status") 
             #js [date team-name])
       (query client)))

(defn add-status [user-name team-name date status]
  (->> (Call (Function "add-status") 
             #js [user-name team-name date status])
       (query client)))


(js->clj (.parse js/JSON (.stringify js/JSON @debug)))

(type @debug)

(.keys js/Object (aget (.-data @debug) 0))

(.-tail (aget (.-data @debug) 0))

(first (second (first (js->clj @debug))))
 
(defn today []
  (-> (js/Date.)
      .toISOString
      (string/split #"T")
      first))


;; Get Status by team and date
;; Make Status
