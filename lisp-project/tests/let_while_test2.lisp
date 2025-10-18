;; More specific test matching the tokenizer pattern

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Simpler struct to mimic Token
(def SimpleToken (: Type)
  (Struct [value I32]))

(def make-simple-token (: (-> [I32] SimpleToken))
  (fn [v]
    (SimpleToken v)))

;; This is the pattern that was failing in the tokenizer
(def problematic-pattern (: (-> [] SimpleToken))
  (fn []
    (let [start (: I32) 10]
      (let [input (: I32) 5]
        (let [len (: I32) 0]
          (while (< len 3)
            (set! len (+ len 1)))
          (make-simple-token len))))))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Testing the exact pattern from tokenizer...\n"))
    (let [tok (: SimpleToken) (problematic-pattern)]
      (printf (c-str "Token value: %d\n") (. tok value)))
    0))

(main-fn)
