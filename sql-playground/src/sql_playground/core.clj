(ns sql-playground.core
  (:require [next.jdbc :as jdbc]
            [honey.sql :as sql]
            [dotenv :as env]
            [next.jdbc.result-set :as rs])
  (:import [org.postgresql Driver]))







(def db {:dbtype "postgres"
         :dbname "defaultdb"
         :user "vercel"
         :host "super-lemur-4402.5xj.cockroachlabs.cloud"
         :port "26257"
         :password (slurp ".password")})

(jdbc/execute!
 db
 (sql/format {:create-table [:calories]
              :with-columns [[:name :string]
                             [:calories :int]
                             [:timestamp :timestamptz]]}))


(jdbc/execute!
 db
 (sql/format {:insert-into [:calories]
              :columns [:name :calories :timestamp]
              :values [["burrito" 1200 [:now]]]}))


(jdbc/execute!
 db
 (sql/format
  {:delete-from [:calories]}))


(jdbc/execute-one!
 db
 (sql/format {:select [:*]
              :from [:calories]})
 {:builder-fn rs/as-unqualified-lower-maps})


(jdbc/execute-one!
 db
 (sql/format {:select [[[:sum :calories] :total] ]
              :from [:calories]})
 {:builder-fn rs/as-unqualified-lower-maps})


(jdbc/execute-one!
 db
 (sql/format {:select [[[:count [:distinct [[:date_trunc [:inline "day"] :timestamp]]]] :days]]
              :from [:calories]})
 {:builder-fn rs/as-unqualified-lower-maps})

(jdbc/execute-one!
 db
 (sql/format {:select [[[:count [:distinct [:date_trunc [:inline "week"] :timestamp]]] :weeks]]
              :from [:calories]})
 {:builder-fn rs/as-unqualified-lower-maps})



(jdbc/execute-one!
 db
 (sql/format {:select [[[:sum :calories] :total]]
              :from [:calories]
              :where [[:= [:date_trunc [:inline "day"] :timestamp]
                       [:cast [:param :timestamp] :timestamptz]]]} {:params {:timestamp "2022-11-23"}})
 {:builder-fn rs/as-unqualified-lower-maps})
