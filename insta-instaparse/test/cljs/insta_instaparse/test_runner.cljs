(ns insta-instaparse.test-runner
  (:require
   [doo.runner :refer-macros [doo-tests]]
   [insta-instaparse.core-test]))

(enable-console-print!)

(doo-tests 'insta-instaparse.core-test)
