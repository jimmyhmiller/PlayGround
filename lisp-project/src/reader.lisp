;; Lisp Reader - Incremental Build
;; Starting with basic value types

(include-header "stdio.h")
(include-header "stdlib.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))

;; Vector structure - dynamic array
(def Vector (: Type)
  (Struct
    [data (Pointer U8)]  ; array of Value pointers
    [count I32]
    [capacity I32]))

;; Value type tags
(def ValueTag (: Type)
  (Enum Nil Number Symbol String List Vector))

;; Value structure - represents any Lisp value
;; Supporting Nil, Number, Symbol, String, List, and Vector
;; We use (Pointer U8) for cons_val and vec_val to avoid circular dependency
(def Value (: Type)
  (Struct
    [tag ValueTag]
    [num_val I64]
    [str_val (Pointer U8)]
    [cons_val (Pointer U8)]
    [vec_val (Pointer U8)]))

;; Cons cell for lists (car/cdr)
;; Uses void pointers that we'll cast to Value*
(def Cons (: Type)
  (Struct
    [car (Pointer U8)]
    [cdr (Pointer U8)]))

;; Constructors

(def make-nil (: (-> [] Value))
  (fn []
    (Value ValueTag/Nil 0 pointer-null pointer-null pointer-null)))

(def make-number (: (-> [I64] Value))
  (fn [n]
    (Value ValueTag/Number n pointer-null pointer-null pointer-null)))

(def make-symbol (: (-> [(Pointer U8)] Value))
  (fn [s]
    (Value ValueTag/Symbol 0 s pointer-null pointer-null)))

(def make-string (: (-> [(Pointer U8)] Value))
  (fn [s]
    (Value ValueTag/String 0 s pointer-null pointer-null)))

;; Create a vector with given count and capacity
;; For simplicity, we build vectors mutably then treat as immutable
(def make-vector-with-capacity (: (-> [I32 I32] (Pointer Value)))
  (fn [count capacity]
    (let [vec (: (Pointer Vector)) (cast (Pointer Vector) (malloc 16))]
      (let [data (: (Pointer U8)) (malloc (* capacity 8))]  ; 8 bytes per pointer
        (pointer-field-write! vec data data)
        (pointer-field-write! vec count count)
        (pointer-field-write! vec capacity capacity)
        (let [val (: (Pointer Value)) (allocate Value (make-nil))]
          (pointer-field-write! val tag ValueTag/Vector)
          (pointer-field-write! val vec_val (cast (Pointer U8) vec))
          val)))))

;; Set element at index in vector (mutable, for building)
(def vector-set (: (-> [(Pointer Value) I32 (Pointer Value)] I32))
  (fn [vec-val index elem]
    (let [vec-ptr (: (Pointer U8)) (pointer-field-read vec-val vec_val)]
      (let [vec (: (Pointer Vector)) (cast (Pointer Vector) vec-ptr)]
        (let [data (: (Pointer U8)) (pointer-field-read vec data)]
          (let [elem-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) (* index 8)))]
            (let [elem-ptr (: (Pointer (Pointer Value))) (cast (Pointer (Pointer Value)) elem-loc)]
              (pointer-write! elem-ptr elem)
              0)))))))

;; Create a cons cell (immutable)
;; We allocate on the heap and copy the values
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
    (let [cons-ptr (: (Pointer U8)) (pointer-field-read v cons_val)]
      (let [cons-cell (: (Pointer Cons)) (cast (Pointer Cons) cons-ptr)]
        (cast (Pointer Value) (pointer-field-read cons-cell car))))))

;; Helper to get cdr of a list
(def cdr (: (-> [(Pointer Value)] (Pointer Value)))
  (fn [v]
    (let [cons-ptr (: (Pointer U8)) (pointer-field-read v cons_val)]
      (let [cons-cell (: (Pointer Cons)) (cast (Pointer Cons) cons-ptr)]
        (cast (Pointer Value) (pointer-field-read cons-cell cdr))))))

;; Forward declaration for mutual recursion
(declare-fn print-value-ptr [v (Pointer Value)] -> I32)

;; Print a list recursively
(def print-list-contents (: (-> [(Pointer Value)] I32))
  (fn [v]
    (let [tag (: ValueTag) (pointer-field-read v tag)]
      (if (= tag ValueTag/Nil)
          (printf (c-str ")"))
          (if (= tag ValueTag/List)
              (let [_1 (: I32) (print-value-ptr (car v))]
                (let [next (: (Pointer Value)) (cdr v)]
                  (let [next-tag (: ValueTag) (pointer-field-read next tag)]
                    (if (= next-tag ValueTag/Nil)
                        (printf (c-str ")"))
                        (let [_2 (: I32) (printf (c-str " "))]
                          (print-list-contents next))))))
              (let [_1 (: I32) (printf (c-str ". "))]
                (let [_2 (: I32) (print-value-ptr v)]
                  (printf (c-str ")")))))))))

;; Helper function to print vector contents
(def print-vector-contents (: (-> [(Pointer Value) I32] I32))
  (fn [vec-val index]
    (let [vec-ptr (: (Pointer U8)) (pointer-field-read vec-val vec_val)]
      (let [vec (: (Pointer Vector)) (cast (Pointer Vector) vec-ptr)]
        (let [count (: I32) (pointer-field-read vec count)]
          (if (>= index count)
              (printf (c-str "]"))
              (let [data (: (Pointer U8)) (pointer-field-read vec data)]
                (let [elem-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) (* index 8)))]
                  (let [elem-ptr-ptr (: (Pointer (Pointer Value))) (cast (Pointer (Pointer Value)) elem-loc)]
                    (let [elem (: (Pointer Value)) (dereference elem-ptr-ptr)]
                      (print-value-ptr elem)
                      (if (< (+ index 1) count)
                          (let [_1 (: I32) (printf (c-str " "))]
                            (print-vector-contents vec-val (+ index 1)))
                          (printf (c-str "]")))))))))))))

;; Print a value from a pointer
(def print-value-ptr (: (-> [(Pointer Value)] I32))
  (fn [v]
    (let [tag (: ValueTag) (pointer-field-read v tag)]
      (if (= tag ValueTag/Nil)
          (printf (c-str "nil"))
          (if (= tag ValueTag/Number)
              (printf (c-str "%lld") (pointer-field-read v num_val))
              (if (= tag ValueTag/Symbol)
                  (printf (c-str "%s") (pointer-field-read v str_val))
                  (if (= tag ValueTag/String)
                      (printf (c-str "\"%s\"") (pointer-field-read v str_val))
                      (if (= tag ValueTag/List)
                          (let [_1 (: I32) (printf (c-str "("))]
                            (print-list-contents v))
                          (if (= tag ValueTag/Vector)
                              (let [_1 (: I32) (printf (c-str "["))]
                                (print-vector-contents v 0))
                              (printf (c-str "<unknown>")))))))))))

;; Test functions

(def print-value (: (-> [Value] I32))
  (fn [v]
    (if (= (. v tag) ValueTag/Nil)
        (printf (c-str "nil\n"))
        (if (= (. v tag) ValueTag/Number)
            (printf (c-str "%lld\n") (. v num_val))
            (if (= (. v tag) ValueTag/Symbol)
                (printf (c-str "%s\n") (. v str_val))
                (if (= (. v tag) ValueTag/String)
                    (printf (c-str "\"%s\"\n") (. v str_val))
                    (printf (c-str "<unknown>\n"))))))))

;; Main - test our basic value types
(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Testing basic value types:\n"))

    (let [v1 (: Value) (make-nil)]
      (printf (c-str "nil value: "))
      (print-value v1))

    (let [v2 (: Value) (make-number 42)]
      (printf (c-str "number value: "))
      (print-value v2))

    (let [v3 (: Value) (make-number -100)]
      (printf (c-str "negative number: "))
      (print-value v3))

    (let [v4 (: Value) (make-symbol (c-str "foo"))]
      (printf (c-str "symbol: "))
      (print-value v4))

    (let [v5 (: Value) (make-string (c-str "hello world"))]
      (printf (c-str "string: "))
      (print-value v5))

    (printf (c-str "\nTesting lists:\n"))

    ;; Build list (1 2 3) from inside out
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))]
      (let [num3 (: (Pointer Value)) (allocate Value (make-number 3))]
        (let [num2 (: (Pointer Value)) (allocate Value (make-number 2))]
          (let [num1 (: (Pointer Value)) (allocate Value (make-number 1))]
            (let [list3 (: (Pointer Value)) (make-cons num3 nil-val)]
              (let [list2 (: (Pointer Value)) (make-cons num2 list3)]
                (let [list1 (: (Pointer Value)) (make-cons num1 list2)]
                  (printf (c-str "list (1 2 3): "))
                  (print-value-ptr list1)
                  (printf (c-str "\n")))))))))

    ;; Build list (foo bar)
    (let [nil-val (: (Pointer Value)) (allocate Value (make-nil))]
      (let [bar (: (Pointer Value)) (allocate Value (make-symbol (c-str "bar")))]
        (let [foo (: (Pointer Value)) (allocate Value (make-symbol (c-str "foo")))]
          (let [list2 (: (Pointer Value)) (make-cons bar nil-val)]
            (let [list1 (: (Pointer Value)) (make-cons foo list2)]
              (printf (c-str "list (foo bar): "))
              (print-value-ptr list1)
              (printf (c-str "\n")))))))

    (printf (c-str "\nTesting vectors:\n"))

    ;; Build vector [1 2 3]
    (let [vec (: (Pointer Value)) (make-vector-with-capacity 3 3)]
      (let [num1 (: (Pointer Value)) (allocate Value (make-number 1))]
        (let [num2 (: (Pointer Value)) (allocate Value (make-number 2))]
          (let [num3 (: (Pointer Value)) (allocate Value (make-number 3))]
            (vector-set vec 0 num1)
            (vector-set vec 1 num2)
            (vector-set vec 2 num3)
            (printf (c-str "vector [1 2 3]: "))
            (print-value-ptr vec)
            (printf (c-str "\n"))))))

    ;; Build vector [:foo :bar]
    (let [vec (: (Pointer Value)) (make-vector-with-capacity 2 2)]
      (let [foo (: (Pointer Value)) (allocate Value (make-symbol (c-str ":foo")))]
        (let [bar (: (Pointer Value)) (allocate Value (make-symbol (c-str ":bar")))]
          (vector-set vec 0 foo)
          (vector-set vec 1 bar)
          (printf (c-str "vector [:foo :bar]: "))
          (print-value-ptr vec)
          (printf (c-str "\n")))))

    0))

(main-fn)