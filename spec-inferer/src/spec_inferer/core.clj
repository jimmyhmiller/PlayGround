(ns spec-inferer.core
  (:require [spec-provider.provider :as sp]
            [spec-provider.stats :as stats]
            [clojure.spec.alpha :as s]
            [spec-tools.core :as st]
            [spec-tools.json-schema :as jsc]
            [codax.core :as codax]
            [com.rpl.specter :as specter]
            [clojure.walk :refer [postwalk]]))

(def db (codax/open-database "demo-databases"))

(defn gen-spec-name []
  (keyword "inferred-spec" (str (gensym "spec"))))

; https://stackoverflow.com/a/35290636
(defn- pretty-demunge
  [fn-object]
  (let [dem-fn (clojure.repl/demunge (str fn-object))
        pretty (second (re-find #"(.*?\/.*?)[\-\-|@].*" dem-fn))]
    (if pretty pretty dem-fn)))

(defn clean-stats [stats]
  (postwalk (fn [k]
              (if (fn? k)
                (str "fn/" (pretty-demunge k))
                k)) stats))

(defn dirty-stats [stats]
  (postwalk (fn [k]
              (if (and (string? k) (clojure.string/includes? k "fn/") )
                @(resolve (symbol (pretty-demunge (subs k 3))))
                k)) stats))

(defn new-stats [db endpoint data]
  (let [spec-name (gen-spec-name)]
    (codax/assoc-at! db [endpoint]
                     {:spec spec-name
                      :stats (clean-stats (stats/collect data))})))

(defn get-stats [db endpoint]
  (codax/get-at! db [endpoint]))

(defn set-stats [db endpoint stats]
  (codax/assoc-at! db [endpoint :stats] (clean-stats stats)))

(defn update-stats [db endpoint data]
  (let [old-stats (get-stats db endpoint)
        new-stats (stats/update-stats (:stats old-stats) data {})]
    (set-stats db endpoint new-stats)))

(defn get-specs [stats]
  (sp/summarize-stats (dirty-stats (:stats stats)) (:spec stats)))

(defn create-specs! [specs]
  (eval specs)) ; Should I worry about this eval?

(defn to-json-schema [stats]
  (create-specs! (get-specs stats))
  (jsc/transform (:spec stats)))


(sp/summarize-stats (stats/collect [{:a 2} {:a 3}]) :test)

(dirty-stats (get-stats db "/test"))

(new-stats db "/test" [{:a 2} {:a 3}])
(get-specs (get-stats db "/test"))

(update-stats db "/test" {:a 2 :b 3})

(to-json-schema (get-stats db "/test"))
