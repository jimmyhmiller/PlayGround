(ns testing-stuff.talk
  (:require [clojure.spec :as s]
            [clojure.spec.gen :as gen]
            [clojure.spec.test :as stest]
            [juxt.dirwatch :refer (watch-dir)]
            [clojure.string :as string]))


(defn get-last [coll]
  (cond
    (zero? (count coll)) nil
    (= (count coll) 1) (first coll)
    (= (first coll) 42) 42
    :else (get-last (rest coll))))



(s/fdef get-last
  :args (s/cat :vals (s/coll-of int?))
  :ret (s/nilable int?)
  :fn #(= (first (reverse (-> % :args :vals))) (-> % :ret)))

(get-last [1 2 3])

(stest/abbrev-result (first (stest/check `get-last)))



(defn handle-file [{:keys [file count action]}]
  (when (string/ends-with? file ".multi")
    (let [contents (slurp file)]
      (when (string/starts-with?  contents "##")
        (let [new-contents (slurp (str (.getParent file) "/" (subs contents 2)))]
          (spit file new-contents))))))

(spit (clojure.java.io/file "/Users/jimmyhmiller/Desktop/test.multi") "hello")

(handle-file {:file (clojure.java.io/file "/Users/jimmyhmiller/Desktop/test.multi")})

(def watcher-agent (watch-dir handle-file (clojure.java.io/file "/Users/jimmyhmiller/Desktop/testmulti")))
