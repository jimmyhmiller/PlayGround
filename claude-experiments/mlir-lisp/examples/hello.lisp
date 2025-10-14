;; Simple test program
;; Run: cargo run --bin mlir-lisp examples/hello.lisp

;; Import the core dialect
(import lisp-core)

;; Define main function that will be executed
(defn main [] i32
  (+ (* 10 20) 30))
