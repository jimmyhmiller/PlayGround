;; Parser - builds Value data structures from tokens
;; This combines with tokenizer.lisp to form the complete reader

(ns parser)

(require [types :as types])

;; C library functions we need
(declare-fn atoll [str (Pointer U8)] -> I64)
(declare-fn malloc [size I32] -> (Pointer U8))

;; Parser state (simple - just an array of tokens and position)
(def Parser (: Type)
  (Struct
    [tokens (Pointer types/Token)]
    [position I32]
    [count I32]))

(def make-parser (: (-> [(Pointer types/Token) I32] (Pointer Parser)))
  (fn [tokens count]
    (let [p (: (Pointer Parser)) (allocate Parser (Parser pointer-null 0 0))]
      (pointer-field-write! p tokens tokens)
      (pointer-field-write! p position 0)
      (pointer-field-write! p count count)
      p)))

;; Peek current token - read fields individually to avoid struct alignment issues
(def peek-token (: (-> [(Pointer Parser)] types/Token))
  (fn [p]
    (let [pos (: I32) (pointer-field-read p position)
          count (: I32) (pointer-field-read p count)]
      (if (>= pos count)
        (types/Token types/TokenType/EOF pointer-null 0)
        (let [tokens (: (Pointer types/Token)) (pointer-field-read p tokens)]
          ;; Token struct: type (4 bytes) + padding (4) + text (8 bytes) + length (4) + padding (4) = 24 bytes?
          ;; Let's try reading fields directly
          (let [token-ptr (: (Pointer types/Token)) (cast (Pointer types/Token) (+ (cast I64 tokens) (* (cast I64 pos) 24)))
                ttype (: types/TokenType) (pointer-field-read token-ptr type)
                ttext (: (Pointer U8)) (pointer-field-read token-ptr text)
                tlen (: I32) (pointer-field-read token-ptr length)]
            (types/Token ttype ttext tlen)))))))


;; Advance parser position
(def advance-parser (: (-> [(Pointer Parser)] I32))
  (fn [p]
    (let [pos (: I32) (pointer-field-read p position)]
      (pointer-field-write! p position (+ pos 1))
      0)))

;; Forward declarations for mutual recursion
(declare-fn parse-value [p (Pointer Parser)] -> (Pointer types/Value))
(declare-fn parse-vector-elements [p (Pointer Parser) vec (Pointer types/Value) index I32] -> I32)

;; Parse a list until )
(def parse-list (: (-> [(Pointer Parser)] (Pointer types/Value)))
  (fn [p]
    (let [tok (: types/Token) (peek-token p)]
      (if (= (. tok type) types/TokenType/RightParen)
        (let [_ (: I32) (advance-parser p)]
          (allocate types/Value (types/make-nil)))
        (let [first (: (Pointer types/Value)) (parse-value p)
              rest (: (Pointer types/Value)) (parse-list p)]
          (types/make-cons first rest))))))

;; Parse vector elements until ]
(def parse-vector-elements (: (-> [(Pointer Parser) (Pointer types/Value) I32] I32))
  (fn [p vec index]
    (let [tok (: types/Token) (peek-token p)]
      (if (= (. tok type) types/TokenType/RightBracket)
        (let [_ (: I32) (advance-parser p)]
          index)
        (let [elem (: (Pointer types/Value)) (parse-value p)]
          (types/vector-set vec index elem)
          (parse-vector-elements p vec (+ index 1)))))))

;; Parse a vector until ]
(def parse-vector (: (-> [(Pointer Parser)] (Pointer types/Value)))
  (fn [p]
    ;; Create vector with initial capacity of 16
    (let [vec (: (Pointer types/Value)) (types/make-vector-with-capacity 16 16)
          final-count (: I32) (parse-vector-elements p vec 0)]
      ;; Update the actual count
      (let [vec-ptr (: (Pointer U8)) (pointer-field-read vec vec_val)
            vec-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)]
        (pointer-field-write! vec-struct count final-count) vec))))

;; Parse map elements until }
(def parse-map-elements (: (-> [(Pointer Parser) (Pointer types/Value) I32] I32))
  (fn [p map-vec index]
    (let [tok (: types/Token) (peek-token p)]
      (if (= (. tok type) types/TokenType/RightBrace)
        (let [_ (: I32) (advance-parser p)]
          index)
        (let [elem (: (Pointer types/Value)) (parse-value p)]
          (types/vector-set map-vec index elem)
          (parse-map-elements p map-vec (+ index 1)))))))

;; Parse a map until }
(def parse-map (: (-> [(Pointer Parser)] (Pointer types/Value)))
  (fn [p]
    ;; Create map (using vector structure) with initial capacity of 32 for key-value pairs
    (let [map-vec (: (Pointer types/Value)) (types/make-vector-with-capacity 32 32)
          final-count (: I32) (parse-map-elements p map-vec 0)]
      ;; Update the actual count
      (let [vec-ptr (: (Pointer U8)) (pointer-field-read map-vec vec_val)
            vec-struct (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)]
        (pointer-field-write! vec-struct count final-count) ;; Create a Map value using the same vec_val but different tag
        (let [map-val (: (Pointer types/Value)) (allocate types/Value (types/make-nil))]
          (pointer-field-write! map-val tag types/ValueTag/Map)
          (pointer-field-write! map-val vec_val vec-ptr)
          map-val)))))

;; Parse a value (recursive)
(def parse-value (: (-> [(Pointer Parser)] (Pointer types/Value)))
  (fn [p]
    (let [tok (: types/Token) (peek-token p)]
      (if (= (. tok type) types/TokenType/LeftParen)
        (let [_ (: I32) (advance-parser p)]
          (parse-list p))
        (if (= (. tok type) types/TokenType/LeftBracket)
          (let [_ (: I32) (advance-parser p)]
            (parse-vector p))
          (if (= (. tok type) types/TokenType/LeftBrace)
            (let [_ (: I32) (advance-parser p)]
              (parse-map p))
            (if (= (. tok type) types/TokenType/Symbol)
              (let [_ (: I32) (advance-parser p)]
                ;; Check if it's actually a number
                (if (!= (types/is-number-token tok) 0)
                  (let [str (: (Pointer U8)) (types/copy-string (. tok text) (. tok length))
                        num (: I64) (atoll str)]
                    (allocate types/Value (types/make-number num)))
                  (let [str (: (Pointer U8)) (types/copy-string (. tok text) (. tok length))]
                    (allocate types/Value (types/make-symbol str)))))
              (if (= (. tok type) types/TokenType/String)
                (let [_ (: I32) (advance-parser p)]
                  ;; String token includes quotes, so skip first char and reduce length by 2
                  (let [str-start (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 (. tok text)) 1))
                        str-len (: I32) (- (. tok length) 2)
                        str (: (Pointer U8)) (types/copy-string str-start str-len)]
                    (allocate types/Value (types/Value types/ValueTag/String 0 str pointer-null pointer-null))))
                (if (= (. tok type) types/TokenType/Keyword)
                  (let [_ (: I32) (advance-parser p)]
                    ;; Keyword token includes ':', copy as-is
                    (let [str (: (Pointer U8)) (types/copy-string (. tok text) (. tok length))]
                      (allocate types/Value (types/Value types/ValueTag/Keyword 0 str pointer-null pointer-null))))
                  ;; Unknown token type - advance parser to avoid infinite loop
                  (let [_ (: I32) (advance-parser p)]
                    (allocate types/Value (types/make-nil))))))))))))

;; Print a value (simplified from reader.lisp)
(declare-fn print-value-ptr [v (Pointer types/Value)] -> I32)
(declare-fn print-vector-contents [vec-val (Pointer types/Value) index I32] -> I32)
(declare-fn print-map-contents [map-val (Pointer types/Value) index I32] -> I32)

(def print-vector-contents (: (-> [(Pointer types/Value) I32] I32))
  (fn [vec-val index]
    (let [vec-ptr (: (Pointer U8)) (pointer-field-read vec-val vec_val)
          vec (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
          count (: I32) (pointer-field-read vec count)]
      (if (>= index count)
        (printf (c-str "]"))
        (let [data (: (Pointer U8)) (pointer-field-read vec data)
              elem-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) (* index 8)))
              elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-loc)
              elem (: (Pointer types/Value)) (dereference elem-ptr-ptr)]
          (print-value-ptr elem) (if (< (+ index 1) count)
            (let [_ (: I32) (printf (c-str " "))]
              (print-vector-contents vec-val (+ index 1)))
            (printf (c-str "]"))))))))

(def print-map-contents (: (-> [(Pointer types/Value) I32] I32))
  (fn [map-val index]
    (let [vec-ptr (: (Pointer U8)) (pointer-field-read map-val vec_val)
          vec (: (Pointer types/Vector)) (cast (Pointer types/Vector) vec-ptr)
          count (: I32) (pointer-field-read vec count)]
      (if (>= index count)
        (printf (c-str "}"))
        (let [data (: (Pointer U8)) (pointer-field-read vec data)
              elem-loc (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 data) (* index 8)))
              elem-ptr-ptr (: (Pointer (Pointer types/Value))) (cast (Pointer (Pointer types/Value)) elem-loc)
              elem (: (Pointer types/Value)) (dereference elem-ptr-ptr)]
          (print-value-ptr elem) (if (< (+ index 1) count)
            (let [_ (: I32) (printf (c-str " "))]
              (print-map-contents map-val (+ index 1)))
            (printf (c-str "}"))))))))

(def print-value-ptr (: (-> [(Pointer types/Value)] I32))
  (fn [v]
    (let [tag (: types/ValueTag) (pointer-field-read v tag)]
      (if (= tag types/ValueTag/Nil)
        (printf (c-str "nil"))
        (if (= tag types/ValueTag/Number)
          (printf (c-str "%lld") (pointer-field-read v num_val))
          (if (= tag types/ValueTag/Symbol)
            (printf (c-str "%s") (pointer-field-read v str_val))
            (if (= tag types/ValueTag/String)
              (printf (c-str "\"%s\"") (pointer-field-read v str_val))
              (if (= tag types/ValueTag/Keyword)
                (printf (c-str "%s") (pointer-field-read v str_val))
                (if (= tag types/ValueTag/List)
                  (let [_ (: I32) (printf (c-str "("))]
                    (print-list-contents v))
                  (if (= tag types/ValueTag/Vector)
                    (let [_ (: I32) (printf (c-str "["))]
                      (print-vector-contents v 0))
                    (if (= tag types/ValueTag/Map)
                      (let [_ (: I32) (printf (c-str "{"))]
                        (print-map-contents v 0))
                      (printf (c-str "<unknown>")))))))))))))

(def print-list-contents (: (-> [(Pointer types/Value)] I32))
  (fn [v]
    (let [tag (: types/ValueTag) (pointer-field-read v tag)]
      (if (= tag types/ValueTag/Nil)
        (printf (c-str ")"))
        (if (= tag types/ValueTag/List)
          (let [cons-ptr (: (Pointer U8)) (pointer-field-read v cons_val)
                cons-cell (: (Pointer types/Cons)) (cast (Pointer types/Cons) cons-ptr)
                car-ptr (: (Pointer types/Value)) (cast (Pointer types/Value) (pointer-field-read cons-cell car))
                cdr-ptr (: (Pointer types/Value)) (cast (Pointer types/Value) (pointer-field-read cons-cell cdr))]
            (print-value-ptr car-ptr) (let [cdr-tag (: types/ValueTag) (pointer-field-read cdr-ptr tag)]
              (if (= cdr-tag types/ValueTag/Nil)
                (printf (c-str ")"))
                (let [_ (: I32) (printf (c-str " "))]
                  (print-list-contents cdr-ptr)))))
          (let [_ (: I32) (printf (c-str ". "))
                _ (: I32) (print-value-ptr v)]
            (printf (c-str ")"))))))))

;; Test
(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Parser test - building Value structures from tokens\n\n"))

    ;; Test 1: (foo bar)
    (printf (c-str "Test 1: (foo bar)\n"))
    (let [tokens (: (Pointer types/Token)) (cast (Pointer types/Token) (malloc 128))
          t0 (: (Pointer types/Token)) tokens]
      (pointer-field-write! t0 type types/TokenType/LeftParen) (pointer-field-write! t0 text (c-str "(")) (pointer-field-write! t0 length 1) (let [t1 (: (Pointer types/Token)) (cast (Pointer types/Token) (+ (cast I64 tokens) 24))]
        (pointer-field-write! t1 type types/TokenType/Symbol)
        (pointer-field-write! t1 text (c-str "foo"))
        (pointer-field-write! t1 length 3)
        (let [t2 (: (Pointer types/Token)) (cast (Pointer types/Token) (+ (cast I64 tokens) 48))]
          (pointer-field-write! t2 type types/TokenType/Symbol)
          (pointer-field-write! t2 text (c-str "bar"))
          (pointer-field-write! t2 length 3)
          (let [t3 (: (Pointer types/Token)) (cast (Pointer types/Token) (+ (cast I64 tokens) 72))]
            (pointer-field-write! t3 type types/TokenType/RightParen)
            (pointer-field-write! t3 text (c-str ")"))
            (pointer-field-write! t3 length 1)
            (let [parser (: (Pointer Parser)) (make-parser tokens 4)
                  result (: (Pointer types/Value)) (parse-value parser)]
              (printf (c-str "  Result: ")) (print-value-ptr result) (printf (c-str "\n\n")))))))

    0))

;; Commented out - this is now a library module
;; (main-fn)
