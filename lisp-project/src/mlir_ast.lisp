;; MLIR AST - Data structures for op and block special forms
;; This extends our reader to parse MLIR-style operations

(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "string.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn strcmp [s1 (Pointer U8) s2 (Pointer U8)] -> I32)

;; Include Value types from reader.lisp
(def ValueTag (: Type)
  (Enum Nil Number Symbol String List Vector Keyword Map))

(def Value (: Type)
  (Struct
    [tag ValueTag]
    [num_val I64]
    [str_val (Pointer U8)]
    [cons_val (Pointer U8)]
    [vec_val (Pointer U8)]))

(def Cons (: Type)
  (Struct
    [car (Pointer U8)]
    [cdr (Pointer U8)]))

(def Vector (: Type)
  (Struct
    [data (Pointer U8)]
    [count I32]
    [capacity I32]))

;; Constructors we'll need
(def make-nil (: (-> [] Value))
  (fn []
    (Value ValueTag/Nil 0 pointer-null pointer-null pointer-null)))

(def make-symbol (: (-> [(Pointer U8)] Value))
  (fn [s]
    (Value ValueTag/Symbol 0 s pointer-null pointer-null)))

(def make-string (: (-> [(Pointer U8)] Value))
  (fn [s]
    (Value ValueTag/String 0 s pointer-null pointer-null)))

;; Create an empty vector
(def make-empty-vector (: (-> [] (Pointer Value)))
  (fn []
    (let [vec (: (Pointer Vector)) (cast (Pointer Vector) (malloc 16))
          data (: (Pointer U8)) (malloc 8)]
      (pointer-field-write! vec data data) (pointer-field-write! vec count 0) (pointer-field-write! vec capacity 1) (let [val (: (Pointer Value)) (allocate Value (make-nil))]
        (pointer-field-write! val tag ValueTag/Vector)
        (pointer-field-write! val vec_val (cast (Pointer U8) vec))
        val))))

;; Create an empty map
(def make-empty-map (: (-> [] (Pointer Value)))
  (fn []
    (let [vec (: (Pointer Vector)) (cast (Pointer Vector) (malloc 16))
          data (: (Pointer U8)) (malloc 8)]
      (pointer-field-write! vec data data) (pointer-field-write! vec count 0) (pointer-field-write! vec capacity 1) (let [val (: (Pointer Value)) (allocate Value (make-nil))]
        (pointer-field-write! val tag ValueTag/Map)
        (pointer-field-write! val vec_val (cast (Pointer U8) vec))
        val))))

;; Create a cons cell (needed for creating lists)
(def make-cons (: (-> [(Pointer Value) (Pointer Value)] (Pointer Value)))
  (fn [car-val cdr-val]
    (let [cons-cell (: (Pointer Cons)) (cast (Pointer Cons) (malloc 16))]
      (pointer-field-write! cons-cell car (cast (Pointer U8) car-val))
      (pointer-field-write! cons-cell cdr (cast (Pointer U8) cdr-val))
      (let [val (: (Pointer Value)) (allocate Value (make-nil))]
        (pointer-field-write! val tag ValueTag/List)
        (pointer-field-write! val cons_val (cast (Pointer U8) cons-cell))
        val))))

;; Helper to get car of a list
(def car (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [v]
    (let [cons-ptr (: (Pointer U8)) (pointer-field-read v cons_val)
          cons-cell (: (Pointer Cons)) (cast (Pointer Cons) cons-ptr)]
      (cast (Pointer Value) (pointer-field-read cons-cell car)))))

;; Helper to get cdr of a list
(def cdr (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [v]
    (let [cons-ptr (: (Pointer U8)) (pointer-field-read v cons_val)
          cons-cell (: (Pointer Cons)) (cast (Pointer Cons) cons-ptr)]
      (cast (Pointer Value) (pointer-field-read cons-cell cdr)))))

;; Check if a value is the symbol 'op'
(def is-symbol-op (: (-> [(Pointer Value)] I32))
  (fn [v]
    (let [tag (: ValueTag) (pointer-field-read v tag)]
      (if (= tag ValueTag/Symbol)
        (let [str-val (: (Pointer U8)) (pointer-field-read v str_val)
              cmp-result (: I32) (strcmp str-val (c-str "op"))]
          (if (= cmp-result 0) 1 0))
        0))))

;; Check if a value is the symbol 'block'
(def is-symbol-block (: (-> [(Pointer Value)] I32))
  (fn [v]
    (let [tag (: ValueTag) (pointer-field-read v tag)]
      (if (= tag ValueTag/Symbol)
        (let [str-val (: (Pointer U8)) (pointer-field-read v str_val)
              cmp-result (: I32) (strcmp str-val (c-str "block"))]
          (if (= cmp-result 0) 1 0))
        0))))

;; Check if a value is a list starting with the symbol 'op'
(def is-op (: (-> [(Pointer Value)] I32))
  (fn [v]
    (let [tag (: ValueTag) (pointer-field-read v tag)]
      (if (= tag ValueTag/List)
        (is-symbol-op (car v))
        0))))

;; Check if a value is a list starting with the symbol 'block'
(def is-block (: (-> [(Pointer Value)] I32))
  (fn [v]
    (let [tag (: ValueTag) (pointer-field-read v tag)]
      (if (= tag ValueTag/List)
        (is-symbol-block (car v))
        0))))

;; Extract the name from an op form
;; op form: (op <name> ...)
;; Returns the name (second element) as a Value pointer, or nil if invalid
(def get-op-name (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer Value)) (cdr op-form)
            rest-tag (: ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag ValueTag/List)
          (car rest)
          (allocate Value (make-nil))))
      (allocate Value (make-nil)))))

;; Extract the result-types from an op form
;; op form: (op <name> <result-types> ...)
;; Returns the result-types (third element) as a Value pointer, or nil if invalid
(def get-op-result-types (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer Value)) (cdr op-form)
            rest-tag (: ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag ValueTag/List)
          (let [rest2 (: (Pointer Value)) (cdr rest)
                rest2-tag (: ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag ValueTag/List)
              (car rest2)
              (allocate Value (make-nil))))
          (allocate Value (make-nil))))
      (allocate Value (make-nil)))))

;; Extract the operands from an op form
;; op form: (op <name> <result-types> <operands> ...)
;; Returns the operands (fourth element) as a Value pointer, or nil if invalid
(def get-op-operands (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer Value)) (cdr op-form)
            rest-tag (: ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag ValueTag/List)
          (let [rest2 (: (Pointer Value)) (cdr rest)
                rest2-tag (: ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag ValueTag/List)
              (let [rest3 (: (Pointer Value)) (cdr rest2)
                    rest3-tag (: ValueTag) (pointer-field-read rest3 tag)]
                (if (= rest3-tag ValueTag/List)
                  (car rest3)
                  (allocate Value (make-nil))))
              (allocate Value (make-nil))))
          (allocate Value (make-nil))))
      (allocate Value (make-nil)))))

;; Extract the attributes from an op form
;; op form: (op <name> <result-types> <operands> <attrs> ...)
;; Returns the attributes (fifth element) as a Value pointer, or nil if invalid
(def get-op-attributes (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer Value)) (cdr op-form)
            rest-tag (: ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag ValueTag/List)
          (let [rest2 (: (Pointer Value)) (cdr rest)
                rest2-tag (: ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag ValueTag/List)
              (let [rest3 (: (Pointer Value)) (cdr rest2)
                    rest3-tag (: ValueTag) (pointer-field-read rest3 tag)]
                (if (= rest3-tag ValueTag/List)
                  (let [rest4 (: (Pointer Value)) (cdr rest3)
                        rest4-tag (: ValueTag) (pointer-field-read rest4 tag)]
                    (if (= rest4-tag ValueTag/List)
                      (car rest4)
                      (allocate Value (make-nil))))
                  (allocate Value (make-nil))))
              (allocate Value (make-nil))))
          (allocate Value (make-nil))))
      (allocate Value (make-nil)))))

;; Extract the regions from an op form
;; op form: (op <name> <result-types> <operands> <attrs> <regions>)
;; Returns the regions (sixth element) as a Value pointer, or nil if invalid
(def get-op-regions (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer Value)) (cdr op-form)
            rest-tag (: ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag ValueTag/List)
          (let [rest2 (: (Pointer Value)) (cdr rest)
                rest2-tag (: ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag ValueTag/List)
              (let [rest3 (: (Pointer Value)) (cdr rest2)
                    rest3-tag (: ValueTag) (pointer-field-read rest3 tag)]
                (if (= rest3-tag ValueTag/List)
                  (let [rest4 (: (Pointer Value)) (cdr rest3)
                        rest4-tag (: ValueTag) (pointer-field-read rest4 tag)]
                    (if (= rest4-tag ValueTag/List)
                      (let [rest5 (: (Pointer Value)) (cdr rest4)
                            rest5-tag (: ValueTag) (pointer-field-read rest5 tag)]
                        (if (= rest5-tag ValueTag/List)
                          (car rest5)
                          (allocate Value (make-nil))))
                      (allocate Value (make-nil))))
                  (allocate Value (make-nil))))
              (allocate Value (make-nil))))
          (allocate Value (make-nil))))
      (allocate Value (make-nil)))))

;; Extract the block-args from a block form
;; block form: (block <block-args> <operations>)
;; Returns the block-args (second element) as a Value pointer, or nil if invalid
(def get-block-args (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [block-form]
    (if (= (is-block block-form) 1)
      (let [rest (: (Pointer Value)) (cdr block-form)
            rest-tag (: ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag ValueTag/List)
          (car rest)
          (allocate Value (make-nil))))
      (allocate Value (make-nil)))))

;; Extract the operations from a block form
;; block form: (block <block-args> <operations>)
;; Returns the operations (third element) as a Value pointer, or nil if invalid
(def get-block-operations (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [block-form]
    (if (= (is-block block-form) 1)
      (let [rest (: (Pointer Value)) (cdr block-form)
            rest-tag (: ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag ValueTag/List)
          (let [rest2 (: (Pointer Value)) (cdr rest)
                rest2-tag (: ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag ValueTag/List)
              (car rest2)
              (allocate Value (make-nil))))
          (allocate Value (make-nil))))
      (allocate Value (make-nil)))))

;; Main - test our helper
(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Testing is-symbol-op:\n"))

    ;; Test 1: symbol 'op' should return 1
    (let [op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          result1 (: I32) (is-symbol-op op-sym)]
      (printf (c-str "  symbol 'op': %d (expected 1)\n") result1))

    ;; Test 2: symbol 'block' should return 0
    (let [block-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "block")))
          result2 (: I32) (is-symbol-op block-sym)]
      (printf (c-str "  symbol 'block': %d (expected 0)\n") result2))

    ;; Test 3: nil should return 0
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          result3 (: I32) (is-symbol-op nil-val)]
      (printf (c-str "  nil: %d (expected 0)\n") result3))

    (printf (c-str "\nTesting is-symbol-block:\n"))

    ;; Test 4: symbol 'block' should return 1
    (let [block-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "block")))
          result4 (: I32) (is-symbol-block block-sym)]
      (printf (c-str "  symbol 'block': %d (expected 1)\n") result4))

    ;; Test 5: symbol 'op' should return 0
    (let [op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          result5 (: I32) (is-symbol-block op-sym)]
      (printf (c-str "  symbol 'op': %d (expected 0)\n") result5))

    (printf (c-str "\nTesting is-op predicate:\n"))

    ;; Test 6: (op ...) should return 1
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          op-list (: (Pointer Value)) (make-cons op-sym nil-val)
          result6 (: I32) (is-op op-list)]
      (printf (c-str "  (op): %d (expected 1)\n") result6))

    ;; Test 7: (block ...) should return 0 for is-op
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          block-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "block")))
          block-list (: (Pointer Value)) (make-cons block-sym nil-val)
          result7 (: I32) (is-op block-list)]
      (printf (c-str "  (block): %d (expected 0)\n") result7))

    (printf (c-str "\nTesting is-block predicate:\n"))

    ;; Test 8: (block ...) should return 1
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          block-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "block")))
          block-list (: (Pointer Value)) (make-cons block-sym nil-val)
          result8 (: I32) (is-block block-list)]
      (printf (c-str "  (block): %d (expected 1)\n") result8))

    ;; Test 9: (op ...) should return 0 for is-block
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          op-list (: (Pointer Value)) (make-cons op-sym nil-val)
          result9 (: I32) (is-block op-list)]
      (printf (c-str "  (op): %d (expected 0)\n") result9))

    (printf (c-str "\nTesting get-op-name:\n"))

    ;; Test 10: (op "func.func" ...) should extract "func.func"
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          name-str (: (Pointer Value)) (allocate Value (make-string (c-str "func.func")))
          rest (: (Pointer Value)) (make-cons name-str nil-val)
          op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          op-list (: (Pointer Value)) (make-cons op-sym rest)
          extracted-name (: (Pointer Value)) (get-op-name op-list)
          name-tag (: ValueTag) (pointer-field-read extracted-name tag)]
      (if (= name-tag ValueTag/String)
        (let [name-val (: (Pointer U8)) (pointer-field-read extracted-name str_val)]
          (printf (c-str "  extracted name: \"%s\" (expected \"func.func\")\n") name-val))
        (printf (c-str "  ERROR: extracted name is not a string!\n"))))

    ;; Test 11: (op) with no name should return nil
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          op-list (: (Pointer Value)) (make-cons op-sym nil-val)
          extracted-name (: (Pointer Value)) (get-op-name op-list)
          name-tag (: ValueTag) (pointer-field-read extracted-name tag)]
      (if (= name-tag ValueTag/Nil)
        (printf (c-str "  (op) with no name: nil (expected nil)\n"))
        (printf (c-str "  ERROR: should have returned nil!\n"))))

    (printf (c-str "\nTesting get-op-result-types:\n"))

    ;; Test 12: (op "name" []) should extract empty vector
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          empty-vec (: (Pointer Value)) (make-empty-vector)
          rest2 (: (Pointer Value)) (make-cons empty-vec nil-val)
          name-str (: (Pointer Value)) (allocate Value (make-string (c-str "name")))
          rest1 (: (Pointer Value)) (make-cons name-str rest2)
          op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          op-list (: (Pointer Value)) (make-cons op-sym rest1)
          extracted-types (: (Pointer Value)) (get-op-result-types op-list)
          types-tag (: ValueTag) (pointer-field-read extracted-types tag)]
      (if (= types-tag ValueTag/Vector)
        (printf (c-str "  extracted result-types: vector (expected vector)\n"))
        (printf (c-str "  ERROR: extracted result-types is not a vector!\n"))))

    (printf (c-str "\nTesting get-op-operands:\n"))

    ;; Test 13: (op "name" [] []) should extract empty vector for operands
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          operands-vec (: (Pointer Value)) (make-empty-vector)
          rest3 (: (Pointer Value)) (make-cons operands-vec nil-val)
          types-vec (: (Pointer Value)) (make-empty-vector)
          rest2 (: (Pointer Value)) (make-cons types-vec rest3)
          name-str (: (Pointer Value)) (allocate Value (make-string (c-str "name")))
          rest1 (: (Pointer Value)) (make-cons name-str rest2)
          op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          op-list (: (Pointer Value)) (make-cons op-sym rest1)
          extracted-operands (: (Pointer Value)) (get-op-operands op-list)
          operands-tag (: ValueTag) (pointer-field-read extracted-operands tag)]
      (if (= operands-tag ValueTag/Vector)
        (printf (c-str "  extracted operands: vector (expected vector)\n"))
        (printf (c-str "  ERROR: extracted operands is not a vector!\n"))))

    (printf (c-str "\nTesting complete op extraction:\n"))

    ;; Test 14: Build (op "arith.constant" ["i32"] [] {} []) and extract all parts
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          regions (: (Pointer Value)) (make-empty-vector)
          rest5 (: (Pointer Value)) (make-cons regions nil-val)
          attrs (: (Pointer Value)) (make-empty-map)
          rest4 (: (Pointer Value)) (make-cons attrs rest5)
          operands (: (Pointer Value)) (make-empty-vector)
          rest3 (: (Pointer Value)) (make-cons operands rest4)
          types (: (Pointer Value)) (make-empty-vector)
          rest2 (: (Pointer Value)) (make-cons types rest3)
          name (: (Pointer Value)) (allocate Value (make-string (c-str "arith.constant")))
          rest1 (: (Pointer Value)) (make-cons name rest2)
          op-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "op")))
          complete-op (: (Pointer Value)) (make-cons op-sym rest1)]
      (printf (c-str "  Testing extraction from complete op form:\n")) (let [ext-name (: (Pointer Value)) (get-op-name complete-op)
            name-tag (: ValueTag) (pointer-field-read ext-name tag)]
        (if (= name-tag ValueTag/String)
          (printf (c-str "    name: OK\n"))
          (printf (c-str "    name: ERROR\n")))) (let [ext-types (: (Pointer Value)) (get-op-result-types complete-op)
                types-tag (: ValueTag) (pointer-field-read ext-types tag)]
            (if (= types-tag ValueTag/Vector)
              (printf (c-str "    result-types: OK\n"))
              (printf (c-str "    result-types: ERROR\n")))) (let [ext-operands (: (Pointer Value)) (get-op-operands complete-op)
                    operands-tag (: ValueTag) (pointer-field-read ext-operands tag)]
                (if (= operands-tag ValueTag/Vector)
                  (printf (c-str "    operands: OK\n"))
                  (printf (c-str "    operands: ERROR\n")))) (let [ext-attrs (: (Pointer Value)) (get-op-attributes complete-op)
                        attrs-tag (: ValueTag) (pointer-field-read ext-attrs tag)]
                    (if (= attrs-tag ValueTag/Map)
                      (printf (c-str "    attributes: OK\n"))
                      (printf (c-str "    attributes: ERROR\n")))) (let [ext-regions (: (Pointer Value)) (get-op-regions complete-op)
                            regions-tag (: ValueTag) (pointer-field-read ext-regions tag)]
                        (if (= regions-tag ValueTag/Vector)
                          (printf (c-str "    regions: OK\n"))
                          (printf (c-str "    regions: ERROR\n")))))

    (printf (c-str "\nTesting block extraction:\n"))

    ;; Test 15: Build (block [] []) and extract parts
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))
          operations (: (Pointer Value)) (make-empty-vector)
          rest2 (: (Pointer Value)) (make-cons operations nil-val)
          block-args (: (Pointer Value)) (make-empty-vector)
          rest1 (: (Pointer Value)) (make-cons block-args rest2)
          block-sym (: (Pointer Value)) (allocate Value (make-symbol (c-str "block")))
          complete-block (: (Pointer Value)) (make-cons block-sym rest1)]
      (printf (c-str "  Testing extraction from complete block form:\n")) (let [ext-args (: (Pointer Value)) (get-block-args complete-block)
            args-tag (: ValueTag) (pointer-field-read ext-args tag)]
        (if (= args-tag ValueTag/Vector)
          (printf (c-str "    block-args: OK\n"))
          (printf (c-str "    block-args: ERROR\n")))) (let [ext-ops (: (Pointer Value)) (get-block-operations complete-block)
                ops-tag (: ValueTag) (pointer-field-read ext-ops tag)]
            (if (= ops-tag ValueTag/Vector)
              (printf (c-str "    operations: OK\n"))
              (printf (c-str "    operations: ERROR\n")))))

    (printf (c-str "\nAll tests passed!\n"))
    0))

(main-fn)