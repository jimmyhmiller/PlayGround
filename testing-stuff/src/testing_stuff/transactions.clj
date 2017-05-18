(ns testing-stuff.transactions
  (:require [clojure.edn :as edn]
            [clj-time.core :as t]
            [clj-time.format :as f]
            [clj-time.predicates :as pr]
            [clojure.java.shell :as sh]
            [clojure.string :refer [index-of]]
            [clj-webdriver.taxi :as br]
            [environ.core :refer [env]])
  (:import [com.xero.api 
            Config 
            JsonConfig 
            OAuthRequestToken 
            OAuthAuthorizeToken 
            OAuthAccessToken
            XeroClient]))



(System/setProperty "webdriver.chrome.driver", "/Users/jimmymiller/Downloads/chromedriver")


(br/set-driver! {:browser :chrome})

(br/to "https://simple.com")

(br/click "a[href*='signin']")

(br/quick-fill-submit {"#login_username" (:simple-user env)
                       "#login_password" (:simple-pass env)})

(br/submit "form")


(defn take-picture-transaction [transaction-id]
  (br/to (str "https://bank.simple.com/transactions/" transaction-id))
  (Thread/sleep 1000)
  (br/take-screenshot :file (str "/Users/jimmymiller/Desktop/transactions/" transaction-id ".png")))


(def config (JsonConfig/getInstance))

(def request-token (OAuthRequestToken. config))
(.execute request-token)
(def token-info (.getAll request-token))
(def temp-token (get token-info "tempToken"))
(def temp-token-secret (get token-info "tempTokenSecret"))
(def auth-token (OAuthAuthorizeToken. config (.getTempToken request-token)))
(clojure.java.browse/browse-url (.getAuthUrl auth-token))
(def notification-response 
  (:err (sh/sh "terminal-notifier" "-message" "Insert Access Code" "-reply" "Access Code")))
(def access-code (subs notification-response (inc (index-of notification-response "@"))))

(def access-token (OAuthAccessToken. config))

(.execute (.build access-token access-code temp-token temp-token-secret))

(.isSuccess access-token)
(def token (.getAll access-token))

(def client (XeroClient.))
(.setOAuthToken client (get token "token") (get token "tokenSecret"))

(def receipts (.getReceipts client))

(def expenseClaims (.getExpenseClaims client))

(map (comp bean) expenseClaims)

(map (comp bean) receipts)

(def attachments (.getAttachments client "Receipts" "a333443d-ad32-4a14-a338-3ee77dcaa34c"))

(map (comp bean) attachments)





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

(def weekday-transactions
  (->> "/Users/jimmymiller/Desktop/transactions.edn"
       slurp
       edn/read-string
       :transactions
       (map #(update-in % [:times :when_recorded_local] (comp parse-date fix-date)))
       (filter weekday?)
       (filter #(t/after? (-> % :times :when_recorded_local) (t/date-time 2017 3 14)))))

(defn lunch-time? [date]
  (< 11 (t/hour date) 14))

(defn restaurant? [transaction]
  (some #(= (:name %) "Restaurants") (:categories transaction)))

(defn coffee? [transaction]
  (#{"Sp Vardagen Com" "Soho Cafe & Gallery"} (:description transaction)))


(def lunch-transactions
  (->> weekday-transactions
       (filter (lift-date lunch-time?))
       (filter restaurant?)))

(def coffee-transactions
  (->> weekday-transactions
       (filter coffee?)))

(first coffee-transactions)

(->> coffee-transactions
     (map :uuid)
     (take 10)
     (map take-picture-transaction))

