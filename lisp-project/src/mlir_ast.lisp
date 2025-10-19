;; MLIR AST - Data structures for op and block special forms
;; This extends our reader to parse MLIR-style operations

(ns mlir-ast)

(include-header "string.h")

(require [types :as types])

;; C library functions we need
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn strcmp [s1 (Pointer U8) s2 (Pointer U8)] -> I32)

;; OpNode - structured representation of a parsed op form
(def OpNode (: Type)
  (Struct
    [name (Pointer U8)]           ; string - the operation name (e.g. "arith.constant")
    [result-types (Pointer types/Value)] ; vector of type strings
    [operands (Pointer types/Value)]     ; vector of SSA value references
    [attributes (Pointer types/Value)]   ; map of attribute key-value pairs
    [regions (Pointer types/Value)]))    ; vector of region vectors

;; BlockNode - structured representation of a parsed block form
(def BlockNode (: Type)
  (Struct
    [args (Pointer types/Value)]         ; vector of [name type] pairs
    [operations (Pointer types/Value)])) ; vector of operations

;; Check if a value is the symbol 'op'
(def is-symbol-op (: (-> [(Pointer types/Value)] I32))
  (fn [v]
    (let [tag (: types/ValueTag) (pointer-field-read v tag)]
      (if (= tag types/ValueTag/Symbol)
        (let [str-val (: (Pointer U8)) (pointer-field-read v str_val)
              cmp-result (: I32) (strcmp str-val (c-str "op"))]
          (if (= cmp-result 0) 1 0))
        0))))

;; Check if a value is the symbol 'block'
(def is-symbol-block (: (-> [(Pointer types/Value)] I32))
  (fn [v]
    (let [tag (: types/ValueTag) (pointer-field-read v tag)]
      (if (= tag types/ValueTag/Symbol)
        (let [str-val (: (Pointer U8)) (pointer-field-read v str_val)
              cmp-result (: I32) (strcmp str-val (c-str "block"))]
          (if (= cmp-result 0) 1 0))
        0))))

;; Check if a value is a list starting with the symbol 'op'
(def is-op (: (-> [(Pointer types/Value)] I32))
  (fn [v]
    (let [tag (: types/ValueTag) (pointer-field-read v tag)]
      (if (= tag types/ValueTag/List)
        (is-symbol-op (types/car v))
        0))))

;; Check if a value is a list starting with the symbol 'block'
(def is-block (: (-> [(Pointer types/Value)] I32))
  (fn [v]
    (let [tag (: types/ValueTag) (pointer-field-read v tag)]
      (if (= tag types/ValueTag/List)
        (is-symbol-block (types/car v))
        0))))

;; Extract the name from an op form
;; op form: (op <name> ...)
;; Returns the name (second element) as a Value pointer, or nil if invalid
(def get-op-name (: (-> [(Pointer types/Value)] (Pointer types/Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer types/Value)) (types/cdr op-form)
            rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag types/ValueTag/List)
          (types/car rest)
          (allocate types/Value (types/make-nil))))
      (allocate types/Value (types/make-nil)))))

;; Extract the result-types from an op form
;; op form: (op <name> <result-types> ...)
;; Returns the result-types (third element) as a Value pointer, or nil if invalid
(def get-op-result-types (: (-> [(Pointer types/Value)] (Pointer types/Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer types/Value)) (types/cdr op-form)
            rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag types/ValueTag/List)
          (let [rest2 (: (Pointer types/Value)) (types/cdr rest)
                rest2-tag (: types/ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag types/ValueTag/List)
              (types/car rest2)
              (allocate types/Value (types/make-nil))))
          (allocate types/Value (types/make-nil))))
      (allocate types/Value (types/make-nil)))))

;; Extract the operands from an op form
;; op form: (op <name> <result-types> <operands> ...)
;; Returns the operands (fourth element) as a Value pointer, or nil if invalid
(def get-op-operands (: (-> [(Pointer types/Value)] (Pointer types/Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer types/Value)) (types/cdr op-form)
            rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag types/ValueTag/List)
          (let [rest2 (: (Pointer types/Value)) (types/cdr rest)
                rest2-tag (: types/ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag types/ValueTag/List)
              (let [rest3 (: (Pointer types/Value)) (types/cdr rest2)
                    rest3-tag (: types/ValueTag) (pointer-field-read rest3 tag)]
                (if (= rest3-tag types/ValueTag/List)
                  (types/car rest3)
                  (allocate types/Value (types/make-nil))))
              (allocate types/Value (types/make-nil))))
          (allocate types/Value (types/make-nil))))
      (allocate types/Value (types/make-nil)))))

;; Extract the attributes from an op form
;; op form: (op <name> <result-types> <operands> <attrs> ...)
;; Returns the attributes (fifth element) as a Value pointer, or nil if invalid
(def get-op-attributes (: (-> [(Pointer types/Value)] (Pointer types/Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer types/Value)) (types/cdr op-form)
            rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag types/ValueTag/List)
          (let [rest2 (: (Pointer types/Value)) (types/cdr rest)
                rest2-tag (: types/ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag types/ValueTag/List)
              (let [rest3 (: (Pointer types/Value)) (types/cdr rest2)
                    rest3-tag (: types/ValueTag) (pointer-field-read rest3 tag)]
                (if (= rest3-tag types/ValueTag/List)
                  (let [rest4 (: (Pointer types/Value)) (types/cdr rest3)
                        rest4-tag (: types/ValueTag) (pointer-field-read rest4 tag)]
                    (if (= rest4-tag types/ValueTag/List)
                      (types/car rest4)
                      (allocate types/Value (types/make-nil))))
                  (allocate types/Value (types/make-nil))))
              (allocate types/Value (types/make-nil))))
          (allocate types/Value (types/make-nil))))
      (allocate types/Value (types/make-nil)))))

;; Extract the regions from an op form
;; op form: (op <name> <result-types> <operands> <attrs> <regions>)
;; Returns the regions (sixth element) as a Value pointer, or nil if invalid
(def get-op-regions (: (-> [(Pointer types/Value)] (Pointer types/Value)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [rest (: (Pointer types/Value)) (types/cdr op-form)
            rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag types/ValueTag/List)
          (let [rest2 (: (Pointer types/Value)) (types/cdr rest)
                rest2-tag (: types/ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag types/ValueTag/List)
              (let [rest3 (: (Pointer types/Value)) (types/cdr rest2)
                    rest3-tag (: types/ValueTag) (pointer-field-read rest3 tag)]
                (if (= rest3-tag types/ValueTag/List)
                  (let [rest4 (: (Pointer types/Value)) (types/cdr rest3)
                        rest4-tag (: types/ValueTag) (pointer-field-read rest4 tag)]
                    (if (= rest4-tag types/ValueTag/List)
                      (let [rest5 (: (Pointer types/Value)) (types/cdr rest4)
                            rest5-tag (: types/ValueTag) (pointer-field-read rest5 tag)]
                        (if (= rest5-tag types/ValueTag/List)
                          (types/car rest5)
                          (allocate types/Value (types/make-nil))))
                      (allocate types/Value (types/make-nil))))
                  (allocate types/Value (types/make-nil))))
              (allocate types/Value (types/make-nil))))
          (allocate types/Value (types/make-nil))))
      (allocate types/Value (types/make-nil)))))

;; Extract the block-args from a block form
;; block form: (block <block-args> <operations>)
;; Returns the block-args (second element) as a Value pointer, or nil if invalid
(def get-block-args (: (-> [(Pointer types/Value)] (Pointer types/Value)))
  (fn [block-form]
    (if (= (is-block block-form) 1)
      (let [rest (: (Pointer types/Value)) (types/cdr block-form)
            rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag types/ValueTag/List)
          (types/car rest)
          (allocate types/Value (types/make-nil))))
      (allocate types/Value (types/make-nil)))))

;; Extract the operations from a block form
;; block form: (block <block-args> <operations>)
;; Returns the operations (third element) as a Value pointer, or nil if invalid
(def get-block-operations (: (-> [(Pointer types/Value)] (Pointer types/Value)))
  (fn [block-form]
    (if (= (is-block block-form) 1)
      (let [rest (: (Pointer types/Value)) (types/cdr block-form)
            rest-tag (: types/ValueTag) (pointer-field-read rest tag)]
        (if (= rest-tag types/ValueTag/List)
          (let [rest2 (: (Pointer types/Value)) (types/cdr rest)
                rest2-tag (: types/ValueTag) (pointer-field-read rest2 tag)]
            (if (= rest2-tag types/ValueTag/List)
              (types/car rest2)
              (allocate types/Value (types/make-nil))))
          (allocate types/Value (types/make-nil))))
      (allocate types/Value (types/make-nil)))))

;; Parse an op form into an OpNode struct
;; Returns pointer to OpNode, or null pointer (cast 0) if invalid
(def parse-op (: (-> [(Pointer types/Value)] (Pointer OpNode)))
  (fn [op-form]
    (if (= (is-op op-form) 1)
      (let [name-val (: (Pointer types/Value)) (get-op-name op-form)
            name-tag (: types/ValueTag) (pointer-field-read name-val tag)]
        (if (= name-tag types/ValueTag/String)
          (let [name (: (Pointer U8)) (pointer-field-read name-val str_val)
                result-types (: (Pointer types/Value)) (get-op-result-types op-form)
                operands (: (Pointer types/Value)) (get-op-operands op-form)
                attributes (: (Pointer types/Value)) (get-op-attributes op-form)
                regions (: (Pointer types/Value)) (get-op-regions op-form)
                node (: (Pointer OpNode)) (cast (Pointer OpNode) (malloc 40))]
            (pointer-field-write! node name name)
            (pointer-field-write! node result-types result-types)
            (pointer-field-write! node operands operands)
            (pointer-field-write! node attributes attributes)
            (pointer-field-write! node regions regions)
            node)
          (cast (Pointer OpNode) 0)))
      (cast (Pointer OpNode) 0))))

;; Parse a block form into a BlockNode struct
;; Returns pointer to BlockNode, or null pointer (cast 0) if invalid
(def parse-block (: (-> [(Pointer types/Value)] (Pointer BlockNode)))
  (fn [block-form]
    (if (= (is-block block-form) 1)
      (let [args (: (Pointer types/Value)) (get-block-args block-form)
            operations (: (Pointer types/Value)) (get-block-operations block-form)
            node (: (Pointer BlockNode)) (cast (Pointer BlockNode) (malloc 16))]
        (pointer-field-write! node args args)
        (pointer-field-write! node operations operations)
        node)
      (cast (Pointer BlockNode) 0))))

;; Main - test our helper
(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Testing is-symbol-op:\n"))

    ;; Test 1: symbol 'op' should return 1
    (let [op-sym (: (Pointer types/Value)) (allocate types/Value (types/make-symbol (c-str "op")))
          result1 (: I32) (is-symbol-op op-sym)]
      (printf (c-str "  symbol 'op': %d (expected 1)\n") result1))

    ;; Test 2: symbol 'block' should return 0
    (let [block-sym (: (Pointer types/Value)) (allocate types/Value (types/make-symbol (c-str "block")))
          result2 (: I32) (is-symbol-op block-sym)]
      (printf (c-str "  symbol 'block': %d (expected 0)\n") result2))

    ;; Test 3: nil should return 0
    (let [nil-val (: (Pointer types/Value)) (allocate types/Value (types/make-nil))
          result3 (: I32) (is-symbol-op nil-val)]
      (printf (c-str "  nil: %d (expected 0)\n") result3))

    (printf (c-str "\nTesting is-symbol-block:\n"))

    ;; Test 4: symbol 'block' should return 1
    (let [block-sym (: (Pointer types/Value)) (allocate types/Value (types/make-symbol (c-str "block")))
          result4 (: I32) (is-symbol-block block-sym)]
      (printf (c-str "  symbol 'block': %d (expected 1)\n") result4))

    ;; Test 5: symbol 'op' should return 0
    (let [op-sym (: (Pointer types/Value)) (allocate types/Value (types/make-symbol (c-str "op")))
          result5 (: I32) (is-symbol-block op-sym)]
      (printf (c-str "  symbol 'op': %d (expected 0)\n") result5))

    (printf (c-str "\nTesting is-op predicate:\n"))

    ;; Test 6: (op ...) should return 1
    (let [nil-val (: (Pointer types/Value)) (allocate types/Value (types/make-nil))
          op-sym (: (Pointer types/Value)) (allocate types/Value (types/make-symbol (c-str "op")))
          op-list (: (Pointer types/Value)) (types/make-cons op-sym nil-val)
          result6 (: I32) (is-op op-list)]
      (printf (c-str "  (op): %d (expected 1)\n") result6))

    ;; Test 7: (block ...) should return 0 for is-op
    (let [nil-val (: (Pointer types/Value)) (allocate types/Value (types/make-nil))
          block-sym (: (Pointer types/Value)) (allocate types/Value (types/make-symbol (c-str "block")))
          block-list (: (Pointer types/Value)) (types/make-cons block-sym nil-val)
          result7 (: I32) (is-op block-list)]
      (printf (c-str "  (block): %d (expected 0)\n") result7))

    (printf (c-str "\nTesting is-block predicate:\n"))

    ;; Test 8: (block ...) should return 1
    (let [nil-val (: (Pointer types/Value)) (allocate types/Value (types/make-nil))
          block-sym (: (Pointer types/Value)) (allocate types/Value (types/make-symbol (c-str "block")))
          block-list (: (Pointer types/Value)) (types/make-cons block-sym nil-val)
          result8 (: I32) (is-block block-list)]
      (printf (c-str "  (block): %d (expected 1)\n") result8))

    ;; Test 9: (op ...) should return 0 for is-block
    (let [nil-val (: (Pointer types/Value)) (allocate types/Value (types/make-nil))
          op-sym (: (Pointer types/Value)) (allocate types/Value (types/make-symbol (c-str "op")))
          op-list (: (Pointer types/Value)) (types/make-cons op-sym nil-val)
          result9 (: I32) (is-block op-list)]
      (printf (c-str "  (op): %d (expected 0)\n") result9))

    (printf (c-str "\nAll tests passed!\n"))
    0))

;; Commented out - this is now a library module
;; (main-fn)
