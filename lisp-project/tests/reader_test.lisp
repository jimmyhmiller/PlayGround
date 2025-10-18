;; ============================================================================
;; Reader Test - Simple demonstration of the reader
;; ============================================================================
;;
;; NOTE: Due to lisp0's lack of a module system, this test file contains
;; forward declarations to demonstrate usage. In a full implementation,
;; you would need to concatenate data.lisp, tokenizer.lisp, and reader.lisp
;; together before this test code.
;; ============================================================================

(include-header "stdio.h")
(include-header "stdlib.h")

(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Forward declarations from data.lisp
(declare-fn make-nil [] -> (Pointer Value))
(declare-fn make-number [I32] -> (Pointer Value))
(declare-fn make-string [(Pointer U8)] -> (Pointer Value))
(declare-fn make-symbol [(Pointer U8)] -> (Pointer Value))
(declare-fn make-empty-list [] -> (Pointer Value))
(declare-fn cons [(Pointer Value) (Pointer Value)] -> (Pointer Value))
(declare-fn get-number [(Pointer Value)] -> I32)
(declare-fn get-symbol [(Pointer Value)] -> (Pointer U8))
(declare-fn is-empty-list [(Pointer Value)] -> I32)
(declare-fn list-head [(Pointer Value)] -> (Pointer Value))
(declare-fn list-tail [(Pointer Value)] -> (Pointer Value))

;; Forward declarations from tokenizer.lisp
(declare-type Token)
(declare-fn tokenize [(Pointer U8)] -> (Pointer Token))

;; Forward declarations from reader.lisp
(declare-type Value)
(declare-fn read [(Pointer Token)] -> (Pointer Value))
(declare-fn read-all [(Pointer Token)] -> (Pointer Value))

;; ============================================================================
;; Helper Functions for Testing
;; ============================================================================

(def TAG_NIL (: I32) 0)
(def TAG_NUMBER (: I32) 1)
(def TAG_STRING (: I32) 2)
(def TAG_SYMBOL (: I32) 3)
(def TAG_LIST (: I32) 4)
(def TAG_VECTOR (: I32) 5)
(def TAG_MAP (: I32) 6)

(def Value (: Type)
  (Struct
    [tag I32]
    [data (Pointer U8)]))

;; Print a value (simple version for testing)
(def print-value (: (Fn [(Pointer Value)] -> Nil))
  (fn [v]
    (let [tag (: I32 (field-access v tag))]
      (if (= tag TAG_NUMBER)
        (printf (cast (Pointer U8) (const-str "%d\n")) (get-number v))
        (if (= tag TAG_SYMBOL)
          (printf (cast (Pointer U8) (const-str "%s\n")) (get-symbol v))
          (if (= tag TAG_LIST)
            (if (= (is-empty-list v) 1)
              (printf (cast (Pointer U8) (const-str "()\n")))
              (printf (cast (Pointer U8) (const-str "(list ...)\n"))))
            (if (= tag TAG_VECTOR)
              (printf (cast (Pointer U8) (const-str "[vector ...]\n")))
              (if (= tag TAG_MAP)
                (printf (cast (Pointer U8) (const-str "{map ...}\n")))
                (printf (cast (Pointer U8) (const-str "nil\n")))))))))))

;; ============================================================================
;; Test Cases
;; ============================================================================

(def test-simple-number (: (Fn [] -> Nil))
  (fn []
    (begin
      (printf (cast (Pointer U8) (const-str "Test 1: Reading a simple number\n")))
      (let [input (: (Pointer U8) (cast (Pointer U8) (const-str "42")))]
        (let [tokens (: (Pointer Token) (tokenize input))]
          (let [value (: (Pointer Value) (read tokens))]
            (begin
              (printf (cast (Pointer U8) (const-str "Result: ")))
              (print-value value)
              (printf (cast (Pointer U8) (const-str "\n"))))))))))

(def test-simple-symbol (: (Fn [] -> Nil))
  (fn []
    (begin
      (printf (cast (Pointer U8) (const-str "Test 2: Reading a simple symbol\n")))
      (let [input (: (Pointer U8) (cast (Pointer U8) (const-str "hello")))]
        (let [tokens (: (Pointer Token) (tokenize input))]
          (let [value (: (Pointer Value) (read tokens))]
            (begin
              (printf (cast (Pointer U8) (const-str "Result: ")))
              (print-value value)
              (printf (cast (Pointer U8) (const-str "\n"))))))))))

(def test-simple-list (: (Fn [] -> Nil))
  (fn []
    (begin
      (printf (cast (Pointer U8) (const-str "Test 3: Reading a simple list\n")))
      (let [input (: (Pointer U8) (cast (Pointer U8) (const-str "(+ 1 2)")))]
        (let [tokens (: (Pointer Token) (tokenize input))]
          (let [value (: (Pointer Value) (read tokens))]
            (begin
              (printf (cast (Pointer U8) (const-str "Result: ")))
              (print-value value)
              (printf (cast (Pointer U8) (const-str "\n"))))))))))

(def test-vector (: (Fn [] -> Nil))
  (fn []
    (begin
      (printf (cast (Pointer U8) (const-str "Test 4: Reading a vector\n")))
      (let [input (: (Pointer U8) (cast (Pointer U8) (const-str "[1 2 3]")))]
        (let [tokens (: (Pointer Token) (tokenize input))]
          (let [value (: (Pointer Value) (read tokens))]
            (begin
              (printf (cast (Pointer U8) (const-str "Result: ")))
              (print-value value)
              (printf (cast (Pointer U8) (const-str "\n"))))))))))

(def test-map (: (Fn [] -> Nil))
  (fn []
    (begin
      (printf (cast (Pointer U8) (const-str "Test 5: Reading a map\n")))
      (let [input (: (Pointer U8) (cast (Pointer U8) (const-str "{:name \"John\" :age 30}")))]
        (let [tokens (: (Pointer Token) (tokenize input))]
          (let [value (: (Pointer Value) (read tokens))]
            (begin
              (printf (cast (Pointer U8) (const-str "Result: ")))
              (print-value value)
              (printf (cast (Pointer U8) (const-str "\n"))))))))))

(def test-nested (: (Fn [] -> Nil))
  (fn []
    (begin
      (printf (cast (Pointer U8) (const-str "Test 6: Reading nested structures\n")))
      (let [input (: (Pointer U8) (cast (Pointer U8) (const-str "(def x [1 2 {:a 3}])")))]
        (let [tokens (: (Pointer Token) (tokenize input))]
          (let [value (: (Pointer Value) (read tokens))]
            (begin
              (printf (cast (Pointer U8) (const-str "Result: ")))
              (print-value value)
              (printf (cast (Pointer U8) (const-str "\n"))))))))))

;; ============================================================================
;; Main - Run all tests
;; ============================================================================

(def main (: (Fn [] -> I32))
  (fn []
    (begin
      (printf (cast (Pointer U8) (const-str "===============================================\n")))
      (printf (cast (Pointer U8) (const-str "Reader Test Suite\n")))
      (printf (cast (Pointer U8) (const-str "===============================================\n\n")))

      (test-simple-number)
      (test-simple-symbol)
      (test-simple-list)
      (test-vector)
      (test-map)
      (test-nested)

      (printf (cast (Pointer U8) (const-str "===============================================\n")))
      (printf (cast (Pointer U8) (const-str "Tests complete!\n")))
      (printf (cast (Pointer U8) (const-str "===============================================\n")))
      0)))
