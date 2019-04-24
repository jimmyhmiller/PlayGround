(load "test-driver.scm")
(load "tests-1.1-req.scm")

(define (emit-program x)
  (unless (integer? x) (raise "Not an integer"))
  (emit "    .text")
  (emit "    .globl _scheme_entry")
  (emit "_scheme_entry:")
  (emit "    movl $~a, %eax" x)
  (emit "    ret"))

