;; Test - copied from parser.lisp

(ns test-copy-parser)

(require [types :as types])

;; C library functions we need
(declare-fn atoll [str (Pointer U8)] -> I64)
(declare-fn malloc [size I32] -> (Pointer U8))

;; Simple test
(def test-val (: (-> [] I32))
  (fn []
    (let [v (allocate types/Value (types/make-nil))]
      0)))
