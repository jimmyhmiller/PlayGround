(ns testing-stuff.transactions
  (:require [clojure.edn :as edn]
            [clj-time.core :as t]
            [clj-time.format :as f]
            [clj-time.predicates :as pr]
            [clojure.java.shell :as sh]
            [clojure.string :refer [index-of]]
            [clj-webdriver.taxi :as br]
            [cheshire.core :refer [parse-stream]]
            [environ.core :refer [env]]
            [clojure.string :as string])
  (:import [com.xero.api 
            Config 
            JsonConfig 
            OAuthRequestToken 
            OAuthAuthorizeToken 
            OAuthAccessToken
            XeroClient]))


(def meals "6380 - Meals and Entertainment")
(def image-prefix "/Users/jimmymiller/Desktop/transactions/")

(System/setProperty "webdriver.chrome.driver", "/Users/jimmymiller/Downloads/chromedriver")


(br/set-driver! {:browser :chrome})

(defn login-to-simple []
  (br/to "https://signin.simple.com/")
  (br/quick-fill-submit {"#login_username" (:simple-user env)
                         "#login_password" (:simple-pass env)})
  (br/submit "form"))


(defn take-picture-transaction [transaction-id]
  (br/to (str "https://bank.simple.com/transactions/" transaction-id))
  (Thread/sleep 1000)
  (br/take-screenshot :file (str "/Users/jimmymiller/Desktop/transactions/" transaction-id ".png")))




(defn login-to-xero []
  (br/to "https://login.xero.com/")
  (br/quick-fill-submit {"#email" (:xero-user env)
                         "#password" (:xero-pass env)})
  (br/submit "form"))


(defn make-field-id 
  ([invoice-id prefix]
   (make-field-id invoice-id prefix nil))
  ([invoice-id prefix suffix]
   (str "#" prefix "_" (string/replace invoice-id "-" "") (if suffix (str "_" suffix) ""))))


(defn add-receipt [{:keys [company-name date description category amount image]}]
  (println [company-name date description category amount image])
  (br/to "https://go.xero.com/Expenses/EditReceipt.aspx")
  (let [field-id (br/value (br/element "#NewInvoiceID"))
        other-id (br/value (br/element ".xoLineItemIDs"))]
    (br/clear (make-field-id other-id "UnitAmount"))
    (br/quick-fill-submit {(make-field-id field-id "PaidToName" "value") company-name
                           (make-field-id field-id "InvoiceDate") date
                           (make-field-id other-id "Description") description
                           (make-field-id other-id "Account" "value") category
                           (make-field-id other-id "UnitAmount") amount})
    (br/click "#ext-gen20")
    (br/input-text "input[type='file']" image)
    (br/click "body")
    (Thread/sleep 1000)
    (br/click "#ext-gen33")
    (Thread/sleep 1000)))







(comment 
  (login-to-xero)
  (add-receipt "Vardagan" "12/31/1991" "coffee" meals "4.52" "/Users/jimmymiller/Desktop/transactions/1b51f7c2-2887-30e3-9774-2b9cc4f65269.png"))




(defn fix-date [date]
  (clojure.string/replace date " " "T"))

(defn parse-date [date]
  (f/parse (f/formatters :date-hour-minute-second-fraction) date))

(defn weekday? [transaction]
  (let [time (-> transaction :times :when_recorded_local)]
    (pr/weekday? time)))

(defn lift-date [f] 
  (fn [transaction]
    (-> transaction :times :when_recorded_local f)))


(defn get-transactions []
  (-> "/Users/jimmymiller/Desktop/transactions.json"
      (clojure.java.io/reader)
      (parse-stream true)
      :transactions))

(def weekday-transactions
  (->> (get-transactions)
       (map #(update-in % [:times :when_recorded_local] (comp parse-date fix-date)))
       (filter weekday?)
       (filter #(t/after? (-> % :times :when_recorded_local) (t/date-time 2017 3 14)))))



(defn lunch-time? [date]
  (< 11 (t/hour date) 14))

(defn restaurant? [transaction]
  (some #(#{"Restaurants" "Fast Food"} (:name %)) (:categories transaction)))

(defn coffee? [transaction]
  (#{"Sp Vardagen Com" "Soho Cafe & Gallery"} (:description transaction)))


(def lunch-transactions
  (->> weekday-transactions
       (filter (lift-date lunch-time?))
       (filter restaurant?)
       (filter #(< 70000 (-> % :amounts :amount) 200000))))

(def date-formatter (f/formatter "MM/dd/yyyy"))

(def coffee-transactions
  (->> weekday-transactions
       (filter coffee?)))

(defn amount->string [amount]
  (str (float (/ amount 10000))))

(defn extract-info [type transaction]
  {:amount (amount->string (-> transaction :amounts :amount))
   :date (f/unparse date-formatter (-> transaction :times :when_recorded_local))
   :company-name (-> transaction :description)
   :id (-> transaction :uuid)
   :category meals
   :description type
   :image (str image-prefix (:uuid transaction) ".png")})

(login-to-xero)
(->> coffee-transactions
     (map (partial extract-info "coffee"))
     (map add-receipt))




(login-to-simple)
(->> coffee-transactions
     (map :uuid)
     (map take-picture-transaction))


(->> lunch-transactions
     (map :uuid)
     (map take-picture-transaction))


(->> lunch-transactions
     (map (partial extract-info "lunch"))
     (map (comp read-string :amount))
     (reduce +))

