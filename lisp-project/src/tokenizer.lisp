;; Simple Tokenizer for Clojure-style syntax
;; Incrementally building - starting with basic tokens


(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "string.h")
(include-header "ctype.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn isdigit [c I32] -> I32)
(declare-fn isspace [c I32] -> I32)
(declare-fn strlen [s (Pointer U8)] -> I32)

;; Token types
(def TokenType (: Type)
  (Enum LeftParen RightParen LeftBracket RightBracket LeftBrace RightBrace Number Symbol String Keyword EOF))

;; Token structure
(def Token (: Type)
  (Struct
    [type TokenType]
    [text (Pointer U8)]
    [length I32]))

;; Tokenizer state
(def Tokenizer (: Type)
  (Struct
    [input (Pointer U8)]
    [position I32]
    [length I32]))

;; Create a tokenizer
(def make-tokenizer (: (-> [(Pointer U8)] (Pointer Tokenizer)))
  (fn [input]
    (let [tok (: (Pointer Tokenizer)) (allocate Tokenizer (Tokenizer pointer-null 0 0))]
      (pointer-field-write! tok input input)
      (pointer-field-write! tok position 0)
      (pointer-field-write! tok length (strlen input))
      tok)))

;; Peek current character
(def peek-char (: (-> [(Pointer Tokenizer)] I32))
  (fn [tok]
    (let [pos (: I32) (pointer-field-read tok position)
          len (: I32) (pointer-field-read tok length)]
      (if (>= pos len)
        (cast I32 0)  ; null char means EOF
        (let [input (: (Pointer U8)) (pointer-field-read tok input)
              char-loc (: I64) (+ (cast I64 input) (cast I64 pos))
              char-ptr (: (Pointer U8)) (cast (Pointer U8) char-loc)
              byte-val (: U8) (dereference char-ptr)]
          (cast I32 byte-val))))))

;; Advance position
(def advance (: (-> [(Pointer Tokenizer)] I32))
  (fn [tok]
    (let [pos (: I32) (pointer-field-read tok position)]
      (pointer-field-write! tok position (+ pos 1))
      0)))

;; Skip to end of line (for comments)
(def skip-to-eol (: (-> [(Pointer Tokenizer)] I32))
  (fn [tok]
    (let [c (: I32) (peek-char tok)]
      (if (and (!= c 0) (!= c 10))  ; 10 is newline
        (let [_1 (: I32) (advance tok)]
          (skip-to-eol tok))
        0))))

;; Skip whitespace and comments
(def skip-whitespace (: (-> [(Pointer Tokenizer)] I32))
  (fn [tok]
    (let [c (: I32) (peek-char tok)]
      (if (= c 59)  ; 59 is ';'
        (let [_1 (: I32) (skip-to-eol tok)]
          (skip-whitespace tok))
        (if (and (!= c 0) (!= (isspace c) 0))
          (let [_1 (: I32) (advance tok)]
            (skip-whitespace tok))
          0)))))

;; Create a token
(def make-token (: (-> [TokenType (Pointer U8) I32] Token))
  (fn [type text length]
    (Token type text length)))

;; Get next token
(def next-token (: (-> [(Pointer Tokenizer)] Token))
  (fn [tok]
    (skip-whitespace tok)
    (let [c (: I32) (peek-char tok)]
      (if (= c 0)
        (make-token TokenType/EOF pointer-null 0)
        (if (= c 40)  ; '('
          (let [_1 (: I32) (advance tok)]
            (make-token TokenType/LeftParen (c-str "(") 1))
          (if (= c 41)  ; ')'
            (let [_1 (: I32) (advance tok)]
              (make-token TokenType/RightParen (c-str ")") 1))
            (if (= c 91)  ; '['
              (let [_1 (: I32) (advance tok)]
                (make-token TokenType/LeftBracket (c-str "[") 1))
              (if (= c 93)  ; ']'
                (let [_1 (: I32) (advance tok)]
                  (make-token TokenType/RightBracket (c-str "]") 1))
                (if (= c 123)  ; '{'
                  (let [_1 (: I32) (advance tok)]
                    (make-token TokenType/LeftBrace (c-str "{") 1))
                  (if (= c 125)  ; '}'
                    (let [_1 (: I32) (advance tok)]
                      (make-token TokenType/RightBrace (c-str "}") 1))
                    (if (= c 34)  ; '"'
                      (read-string tok)
                      (if (= c 58)  ; ':'
                        (read-keyword tok)
                        ;; For now, treat everything else as a symbol
                        (read-symbol tok)))))))))))))

;; Helper to read a symbol token
(def read-symbol (: (-> [(Pointer Tokenizer)] Token))
  (fn [tok]
    (let [start-pos (: I32) (pointer-field-read tok position)
          input (: (Pointer U8)) (pointer-field-read tok input)
          start-ptr (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 input) (cast I64 start-pos)))
          len (: I32) 0]
      ;; Read until whitespace or delimiter
      (while (and (!= (peek-char tok) 0)
        (and (= (isspace (peek-char tok)) 0)
          (and (!= (peek-char tok) 40)
            (and (!= (peek-char tok) 41)
              (and (!= (peek-char tok) 91)
                (!= (peek-char tok) 93))))))
        (advance tok)
        (set! len (+ len 1))) ;; Return the token with the accumulated length
      (make-token TokenType/Symbol start-ptr len))))

;; Helper to read a string token
(def read-string (: (-> [(Pointer Tokenizer)] Token))
  (fn [tok]
    (let [start-pos (: I32) (pointer-field-read tok position)
          input (: (Pointer U8)) (pointer-field-read tok input)
          start-ptr (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 input) (cast I64 start-pos)))]
      ;; Skip opening quote
      (let [_ (: I32) (advance tok)
            len (: I32) 1]
        ; Start with 1 for opening quote
        ;; Read until closing quote
        (while (and (!= (peek-char tok) 0)
          (!= (peek-char tok) 34))  ; 34 is "
          (advance tok)
          (set! len (+ len 1))) ;; Advance past closing quote and include it in length
        (let [_ (: I32) (advance tok)]
          (make-token TokenType/String start-ptr (+ len 1)))))))

;; Helper to read a keyword token
(def read-keyword (: (-> [(Pointer Tokenizer)] Token))
  (fn [tok]
    (let [start-pos (: I32) (pointer-field-read tok position)
          input (: (Pointer U8)) (pointer-field-read tok input)
          start-ptr (: (Pointer U8)) (cast (Pointer U8) (+ (cast I64 input) (cast I64 start-pos)))
          len (: I32) 0]
      ;; Read ':' and following characters until whitespace or delimiter
      (while (and (!= (peek-char tok) 0)
        (and (= (isspace (peek-char tok)) 0)
          (and (!= (peek-char tok) 40)
            (and (!= (peek-char tok) 41)
              (and (!= (peek-char tok) 91)
                (and (!= (peek-char tok) 93)
                  (and (!= (peek-char tok) 123)
                    (!= (peek-char tok) 125))))))))
        (advance tok)
        (set! len (+ len 1))) ;; Return the token with the accumulated length
      (make-token TokenType/Keyword start-ptr len))))

;; Test
(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Testing tokenizer:\n"))

    (let [tok (: (Pointer Tokenizer)) (make-tokenizer (c-str ";; comment\n(foo bar)"))
          t1 (: Token) (next-token tok)]
      (printf (c-str "Token 1: type=%d text='%.*s'\n") (cast I32 (. t1 type)) (. t1 length) (. t1 text)) (let [t2 (: Token) (next-token tok)]
        (printf (c-str "Token 2: type=%d text='%.*s'\n") (cast I32 (. t2 type)) (. t2 length) (. t2 text))
        (let [t3 (: Token) (next-token tok)]
          (printf (c-str "Token 3: type=%d text='%.*s'\n") (cast I32 (. t3 type)) (. t3 length) (. t3 text))
          (let [t4 (: Token) (next-token tok)]
            (printf (c-str "Token 4: type=%d text='%.*s'\n") (cast I32 (. t4 type)) (. t4 length) (. t4 text))
            (let [t5 (: Token) (next-token tok)]
              (printf (c-str "Token 5: type=%d text='%.*s'\n") (cast I32 (. t5 type)) (. t5 length) (. t5 text))
              (let [t6 (: Token) (next-token tok)]
                (printf (c-str "Token 6: type=%d text='%.*s'\n") (cast I32 (. t6 type)) (. t6 length) (. t6 text))
                (let [t7 (: Token) (next-token tok)]
                  (printf (c-str "Token 7: type=%d text='%.*s'\n") (cast I32 (. t7 type)) (. t7 length) (. t7 text))
                  (let [t8 (: Token) (next-token tok)]
                    (printf (c-str "Token 8: type=%d text='%.*s'\n") (cast I32 (. t8 type)) (. t8 length) (. t8 text))
                    0))))))))

    0))

(main-fn)