;; Example 6: Loops and Iteration
;; Demonstrates: while loops, c-for loops, mutation, control flow

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Factorial using while loop
(def factorial-while (: (-> [I32] I32))
  (fn [n]
    (let [result (: I32) 1]
      (let [i (: I32) n]
        (while (> i 1)
          (set! result (* result i))
          (set! i (- i 1)))
        result))))

;; Factorial using c-for loop
(def factorial-for (: (-> [I32] I32))
  (fn [n]
    (let [result (: I32) 1]
      (c-for [i (: I32) 2] (<= i n) (set! i (+ i 1))
        (set! result (* result i)))
      result)))

;; Count down using while
(def countdown (: (-> [I32] I32))
  (fn [n]
    (while (> n 0)
      (printf (c-str "%d...\n") n)
      (set! n (- n 1)))
    (printf (c-str "Blast off!\n"))))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Factorial (while): 5! = %d\n") (factorial-while 5))
    (printf (c-str "Factorial (for): 7! = %d\n") (factorial-for 7))
    (printf (c-str "\nCountdown:\n"))
    (countdown 5)
    0))

(main-fn)
