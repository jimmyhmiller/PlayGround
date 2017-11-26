(ns phone-number.routes_test
  (:require [clojure.test :refer :all]
            [cognitect.transcriptor :as xr :refer (check!)]))


(deftest test-app
  (->> (clojure.java.io/file ".")
       .getCanonicalPath
       xr/repl-files
       (map xr/run)
       doall))
