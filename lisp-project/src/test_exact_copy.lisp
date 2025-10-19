;; Exact copy from parser.lisp

(ns test-exact-copy)

(require [types :as types])

(declare-fn malloc [size I32] -> (Pointer U8))

;; Copied from parser.lisp line 56
(def parse-list (: (-> [(Pointer U8)] (Pointer types/Value)))
  (fn [p]
    (let [tok (types/Token types/TokenType/EOF pointer-null 0)]
      (if (= (. tok type) types/TokenType/RightParen)
        (let [_ (: I32) 0]
          (allocate types/Value (types/make-nil)))
        (allocate types/Value (types/make-nil))))))

;; Test
;;(parse-list pointer-null)
