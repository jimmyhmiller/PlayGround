(ns two-ultimate.test-runner
  (:require
   [doo.runner :refer-macros [doo-tests]]
   [two-ultimate.core-test]
   [two-ultimate.common-test]))

(enable-console-print!)

(doo-tests 'two-ultimate.core-test
           'two-ultimate.common-test)
