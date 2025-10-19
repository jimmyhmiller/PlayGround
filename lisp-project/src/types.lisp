;; Shared type definitions for the reader/parser system
;; This module contains the core data structures used throughout the codebase

(ns types)

(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "string.h")
(include-header "ctype.h")

;; Standard library function declarations (printf is builtin)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn strncpy [dest (Pointer U8) src (Pointer U8) n I32] -> (Pointer U8))
(declare-fn isdigit [c I32] -> I32)
(declare-fn atoll [str (Pointer U8)] -> I64)
(declare-fn strcmp [s1 (Pointer U8) s2 (Pointer U8)] -> I32)

;; Token types - used by tokenizer/parser
(def TokenType (: Type)
  (Enum LeftParen RightParen LeftBracket RightBracket LeftBrace RightBrace Number Symbol String Keyword EOF))

(def Token (: Type)
  (Struct
    [type TokenType]
    [text (Pointer U8)]
    [length I32]))

;; Value types - core data structures for representing Lisp values
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

;; Basic value constructors
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

;; Create a cons cell
(def make-cons (: (-> [(Pointer Value) (Pointer Value)] (Pointer Value)))
  (fn [car-val cdr-val]
    (let [cons-cell (: (Pointer Cons)) (cast (Pointer Cons) (malloc 16))]
      (pointer-field-write! cons-cell car (cast (Pointer U8) car-val))
      (pointer-field-write! cons-cell cdr (cast (Pointer U8) cdr-val))
      (let [val (: (Pointer Value)) (allocate Value (make-nil))]
        (pointer-field-write! val tag ValueTag/List)
        (pointer-field-write! val cons_val (cast (Pointer U8) cons-cell))
        val))))

;; Create an empty vector
(def make-empty-vector (: (-> [] (Pointer Value)))
  (fn []
    (let [vec (: (Pointer Vector)) (cast (Pointer Vector) (malloc 16))
          data (: (Pointer U8)) (malloc 8)]
      (pointer-field-write! vec data data)
      (pointer-field-write! vec count 0)
      (pointer-field-write! vec capacity 1)
      (let [val (: (Pointer Value)) (allocate Value (make-nil))]
        (pointer-field-write! val tag ValueTag/Vector)
        (pointer-field-write! val vec_val (cast (Pointer U8) vec))
        val))))

;; Create a vector with capacity
(def make-vector-with-capacity (: (-> [I32 I32] (Pointer Value)))
  (fn [count capacity]
    (let [vec (: (Pointer Vector)) (cast (Pointer Vector) (malloc 16))
          data (: (Pointer U8)) (malloc (* capacity 8))]
      (pointer-field-write! vec data data)
      (pointer-field-write! vec count count)
      (pointer-field-write! vec capacity capacity)
      (let [val (: (Pointer Value)) (allocate Value (make-nil))]
        (pointer-field-write! val tag ValueTag/Vector)
        (pointer-field-write! val vec_val (cast (Pointer U8) vec))
        val))))

;; Create an empty map
(def make-empty-map (: (-> [] (Pointer Value)))
  (fn []
    (let [vec (: (Pointer Vector)) (cast (Pointer Vector) (malloc 16))
          data (: (Pointer U8)) (malloc 8)]
      (pointer-field-write! vec data data)
      (pointer-field-write! vec count 0)
      (pointer-field-write! vec capacity 1)
      (let [val (: (Pointer Value)) (allocate Value (make-nil))]
        (pointer-field-write! val tag ValueTag/Map)
        (pointer-field-write! val vec_val (cast (Pointer U8) vec))
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

;; Copy string from token
(def copy-string (: (-> [(Pointer U8) I32] (Pointer U8)))
  (fn [src len]
    (let [dest (: (Pointer U8)) (malloc (+ len 1))]
      (strncpy dest src len)
      ;; Null terminate
      (let [null-pos (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 dest) (cast I64 len)))]
        (pointer-write! (cast (Pointer U8) null-pos) (cast U8 0)))
      dest)))

;; Check if a token represents a number
(def is-number-token (: (-> [Token] I32))
  (fn [tok]
    (if (= (. tok length) 0)
      0
      (let [first-char (: U8) (dereference (. tok text))
            is-digit (: I32) (isdigit (cast I32 first-char))]
        (if (!= is-digit 0)
          1
          ;; Check for negative number
          (if (and (= first-char (cast U8 45))  ; '-' character
            (> (. tok length) 1))
            (let [second-char-ptr (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 (. tok text)) 1))
                  second-char (: U8) (dereference second-char-ptr)
                  is-digit2 (: I32) (isdigit (cast I32 second-char))]
              (if (!= is-digit2 0) 1 0))
            0))))))

;; Helper to set element in vector
(def vector-set (: (-> [(Pointer Value) I32 (Pointer Value)] I32))
  (fn [vec-val index elem]
    (let [vec-ptr (: (Pointer U8)) (pointer-field-read vec-val vec_val)
          vec (: (Pointer Vector)) (cast (Pointer Vector) vec-ptr)
          data (: (Pointer U8)) (pointer-field-read vec data)
          elem-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) (* index 8)))
          elem-ptr (: (Pointer (Pointer Value))) (cast (Pointer (Pointer Value)) elem-loc)]
      (pointer-write! elem-ptr elem)
      0)))
