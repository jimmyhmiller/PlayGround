(ns github-bots.core-test
  (:require [clojure.test :refer :all]
            [github-bots.core :refer :all]))

(deftest proposal?-test
  (is (false? (proposal? "test")))
  (is (true? (proposal? "304. adsfadf"))))


(deftest vote?-test
  (is (false? (vote? "test")))
  (is (false? (vote? "naY asdfasdf")))
  (is (true? (vote? "yay")))
  (is (true? (vote? "YAY")))
  (is (true? (vote? "NAY"))))



(is (zero? (:fail (run-tests))))
