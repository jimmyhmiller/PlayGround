;; Parse test files using the full pipeline
;; This demonstrates parsing actual op/block files from tests/

(ns parse-test-files)

(require [types :as types])
(require [parser :as parser])
(require [mlir-ast :as ast])

;; C library functions
(declare-fn malloc [size I32] -> (Pointer U8))

;; Test parsing a simple expression
(def test-simple-expr (: (-> [] I32))
  (fn []
    (printf (c-str "=== Test 1: Parse simple list (foo bar) ===\n"))

    ;; Create tokens manually for: (foo bar)
    (let [tokens (malloc 128)]

      ;; Token 0: (
      (pointer-field-write! tokens type types/TokenType/LeftParen)
      (pointer-field-write! tokens text (c-str "("))
      (pointer-field-write! tokens length 1)

      ;; Token 1: foo
      (let [t1 (+ (cast I64 tokens) 24)]
        (pointer-field-write! t1 type types/TokenType/Symbol)
        (pointer-field-write! t1 text (c-str "foo"))
        (pointer-field-write! t1 length 3)

        ;; Token 2: bar
        (let [t2 (+ (cast I64 tokens) 48)]
          (pointer-field-write! t2 type types/TokenType/Symbol)
          (pointer-field-write! t2 text (c-str "bar"))
          (pointer-field-write! t2 length 3)

          ;; Token 3: )
          (let [t3 (+ (cast I64 tokens) 72)]
            (pointer-field-write! t3 type types/TokenType/RightParen)
            (pointer-field-write! t3 text (c-str ")"))
            (pointer-field-write! t3 length 1)

            ;; Create parser and parse
            ;; Note: Can't use qualified type names, so we call make-parser which returns the right type
            (let [p (parser/make-parser tokens 4)
                  result (parser/parse-value p)]
              (printf (c-str "Parsed: "))
              (parser/print-value-ptr result)
              (printf (c-str "\n\n"))
              0)))))
    0))

;; Test parsing an op form
(def test-op-form (: (-> [] I32))
  (fn []
    (printf (c-str "=== Test 2: Parse and inspect op form ===\n"))
    (printf (c-str "Input: (op \"arith.constant\" [\"i32\"] [] {} [])\n"))

    ;; Manually build the op form: (op "arith.constant" ["i32"] [] {} [])
    (let [nil-val (allocate types/Value (types/make-nil))

          ;; Build empty regions vector: []
          regions (types/make-empty-vector)
          rest5 (types/make-cons regions nil-val)

          ;; Build empty attributes map: {}
          attrs (types/make-empty-map)
          rest4 (types/make-cons attrs rest5)

          ;; Build empty operands vector: []
          operands (types/make-empty-vector)
          rest3 (types/make-cons operands rest4)

          ;; Build result types vector: ["i32"]
          result-types (types/make-empty-vector)
          rest2 (types/make-cons result-types rest3)

          ;; Build name string: "arith.constant"
          name (allocate types/Value (types/make-string (c-str "arith.constant")))
          rest1 (types/make-cons name rest2)

          ;; Build op symbol
          op-sym (allocate types/Value (types/make-symbol (c-str "op")))
          op-form (types/make-cons op-sym rest1)]

      ;; Test if it's recognized as an op
      (let [is-op-result (: I32) (ast/is-op op-form)]
        (printf (c-str "Is this an op form? %s\n")
                (if (= is-op-result 1) (c-str "YES") (c-str "NO"))))

      ;; Extract and print the op name
      (let [extracted-name (ast/get-op-name op-form)
            name-tag (pointer-field-read extracted-name tag)]
        (if (= name-tag types/ValueTag/String)
          (let [name-str (pointer-field-read extracted-name str_val)]
            (printf (c-str "Op name: \"%s\"\n") name-str))
          (printf (c-str "ERROR: Could not extract op name\n"))))

      ;; Parse into OpNode structure
      (let [op-node (ast/parse-op op-form)]
        (if (!= (cast I64 op-node) 0)
          (let [parsed-name (pointer-field-read op-node name)]
            (printf (c-str "Parsed OpNode with name: \"%s\"\n") parsed-name))
          (printf (c-str "ERROR: parse-op failed\n"))))

      (printf (c-str "\n"))
      0)))

;; Test parsing a block form
(def test-block-form (: (-> [] I32))
  (fn []
    (printf (c-str "=== Test 3: Parse and inspect block form ===\n"))
    (printf (c-str "Input: (block [] [])\n"))

    ;; Manually build: (block [] [])
    (let [nil-val (allocate types/Value (types/make-nil))

          ;; Build empty operations vector: []
          operations (types/make-empty-vector)
          rest2 (types/make-cons operations nil-val)

          ;; Build empty block-args vector: []
          block-args (types/make-empty-vector)
          rest1 (types/make-cons block-args rest2)

          ;; Build block symbol
          block-sym (allocate types/Value (types/make-symbol (c-str "block")))
          block-form (types/make-cons block-sym rest1)]

      ;; Test if it's recognized as a block
      (let [is-block-result (: I32) (ast/is-block block-form)]
        (printf (c-str "Is this a block form? %s\n")
                (if (= is-block-result 1) (c-str "YES") (c-str "NO"))))

      ;; Parse into BlockNode structure
      (let [block-node (ast/parse-block block-form)]
        (if (!= (cast I64 block-node) 0)
          (printf (c-str "Successfully parsed BlockNode\n"))
          (printf (c-str "ERROR: parse-block failed\n"))))

      (printf (c-str "\n"))
      0)))

;; Main function
(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "=== MLIR AST Parser Demo ===\n\n"))
    (printf (c-str "This demo shows parsing of MLIR-style op and block forms\n"))
    (printf (c-str "using the modular parser and AST libraries.\n\n"))

    (test-simple-expr)
    (test-op-form)
    (test-block-form)

    (printf (c-str "=== Demo Complete ===\n"))
    0))

(main-fn)
