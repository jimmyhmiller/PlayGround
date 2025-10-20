;; Parse test files using the full pipeline
;; This demonstrates parsing actual op/block files from tests/

(ns parse-test-files)

(require [types :as types])
(require [parser :as parser])
(require [mlir-ast :as ast])
(require [tokenizer :as tokenizer])

(include-header "stdio.h")
(include-header "stdlib.h")

;; C library functions
(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn fopen [filename (Pointer U8) mode (Pointer U8)] -> (Pointer U8))
(declare-fn fclose [file (Pointer U8)] -> I32)
(declare-fn fseek [file (Pointer U8) offset I32 whence I32] -> I32)
(declare-fn ftell [file (Pointer U8)] -> I32)
(declare-fn fread [ptr (Pointer U8) size I32 count I32 file (Pointer U8)] -> I32)
(declare-fn rewind [file (Pointer U8)] -> Nil)

;; Read entire file into a string
(def read-file (: (-> [(Pointer U8)] (Pointer U8)))
  (fn [filename]
    (let [file (: (Pointer U8)) (fopen filename (c-str "r"))]
      (if (= (cast I64 file) 0)
        (let [_ (: I32) (printf (c-str "Error: Could not open file %s\n") filename)]
          (cast (Pointer U8) 0))
        (let [;; Seek to end to get file size
              _ (: I32) (fseek file 0 2)  ; SEEK_END = 2
              size (: I32) (ftell file)
              _ (: Nil) (rewind file)

              ;; Allocate buffer (size + 1 for null terminator)
              buffer (: (Pointer U8)) (malloc (+ size 1))

              ;; Read file contents
              read-count (: I32) (fread buffer 1 size file)
              _ (: I32) (fclose file)]

          ;; Null terminate the buffer
          (let [null-pos (: I64) (+ (cast I64 buffer) (cast I64 size))
                null-ptr (: (Pointer U8)) (cast (Pointer U8) null-pos)]
            (pointer-write! null-ptr (cast U8 0))
            buffer))))))

;; Collect all tokens from a file into an array
(def tokenize-file (: (-> [(Pointer U8)] (Pointer types/Token)))
  (fn [content]
    (let [tok (: (Pointer tokenizer/Tokenizer)) (tokenizer/make-tokenizer content)
          ;; Allocate space for up to 1000 tokens
          max-tokens (: I32) 1000
          token-size (: I32) 24  ; sizeof(Token)
          tokens (: (Pointer types/Token)) (cast (Pointer types/Token) (malloc (* max-tokens token-size)))
          count (: I32) 0]

      ;; Collect tokens until EOF
      (while (< count max-tokens)
        (let [token (: types/Token) (tokenizer/next-token tok)
              token-type (: types/TokenType) (. token type)
              ;; Calculate offset for this token
              token-offset (: I64) (* (cast I64 count) (cast I64 token-size))
              token-ptr (: (Pointer types/Token)) (cast (Pointer types/Token) (+ (cast I64 tokens) token-offset))]

          ;; Always write the token (including EOF)
          (pointer-field-write! token-ptr type (. token type))
          (pointer-field-write! token-ptr text (. token text))
          (pointer-field-write! token-ptr length (. token length))
          (set! count (+ count 1))

          ;; Break if we just wrote EOF
          (if (= token-type types/TokenType/EOF)
            (set! count max-tokens)  ; Break loop
            (set! count count))))

      ;; Return the tokens array
      tokens)))

;; Parse a single file
(def parse-single-file (: (-> [(Pointer U8)] I32))
  (fn [filename]
    (printf (c-str "=== Parsing: %s ===\n") filename)

    ;; Read the file
    (let [content (: (Pointer U8)) (read-file filename)]
      (if (= (cast I64 content) 0)
        (let [_ (: I32) (printf (c-str "ERROR: Failed to read file\n"))]
          1)
        (let [_ (: I32) (printf (c-str "File content:\n%s\n\n") content)

              ;; Tokenize the content
              _ (: I32) (printf (c-str "Tokenizing...\n"))
              tokens (: (Pointer types/Token)) (tokenize-file content)

              ;; Count tokens
              token-count (: I32) 0
              found-eof (: I32) 0]

          ;; Count tokens loop (count until we hit EOF, including the EOF token)
          (while (and (< token-count 1000) (= found-eof 0))
            (let [token-offset (: I64) (* (cast I64 token-count) 24)
                  token-ptr (: (Pointer types/Token)) (cast (Pointer types/Token) (+ (cast I64 tokens) token-offset))
                  token-type (: types/TokenType) (pointer-field-read token-ptr type)]
              (set! token-count (+ token-count 1))
              (if (= token-type types/TokenType/EOF)
                (set! found-eof 1)
                (set! found-eof 0))))

          (let [_ (: I32) (printf (c-str "Found %d tokens\n\n") token-count)

                ;; Parse the tokens
                _ (: I32) (printf (c-str "Parsing...\n"))
                p (: (Pointer parser/Parser)) (parser/make-parser tokens token-count)]

            ;; Parse all top-level values and recursively convert to data structures
            (while (!= (cast I32 (. (parser/peek-token p) type)) (cast I32 types/TokenType/EOF))
              (let [result (: (Pointer types/Value)) (parser/parse-value p)]
                (printf (c-str "\nRecursively parsing entire tree:\n"))
                ;; Recursively parse and print the entire tree
                (ast/parse-and-print-recursive result 0)
                (printf (c-str "\n"))))

            (printf (c-str "\n"))
            0))))))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "=== MLIR AST Parser Demo ===\n\n"))
    (printf (c-str "This demo shows parsing of MLIR-style op and block forms\n"))
    (printf (c-str "using the modular parser and AST libraries.\n\n"))

    ;; Parse all test files
    (parse-single-file (c-str "tests/simple.lisp"))
    (parse-single-file (c-str "tests/add.lisp"))
    (parse-single-file (c-str "tests/fib.lisp"))

    (printf (c-str "=== All Files Parsed Successfully ===\n"))
    0))

(main-fn)