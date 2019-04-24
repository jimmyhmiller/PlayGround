(load "test-driver.scm")
(load "tests/tests-1.1-req.scm")
(load "tests/tests-1.2-req.scm")


(define fxshift 2)
(define fxmask #x03)
(define bool_f #x2F)
(define bool_t #x6F)
(define null-rep #x3F)
(define char_tag #x0F)
(define charshift 8)
(define wordsize 4)

(define fixnum-bits (- (* wordsize 8) fxshift))
(define fxlower (- (expt 2 (- fixnum-bits 1))))
(define fxupper (sub1 (expt 2 (- fixnum-bits 1))))

(define (fixnum? x)
  (and (integer? x) (exact? x) (<= fxlower x fxupper)))

(define (immediate? x)
  (or (fixnum? x)
      (boolean? x)
      (char? x)
      (null? x)))

(define (char-rep x)
  (bitwise-ior (ash (char->integer x) charshift) char_tag))

(define (immediate-rep x)
  (cond
   [(fixnum? x) (ash x fxshift)]
   [(equal? x #f) bool_f]
   [(equal? x #t) bool_t]
   [(char? x) (char-rep x)]
   [(null? x) null-rep]))


(define (emit-program x)
  (unless (immediate? x) (raise "Not immediate"))
  (emit "    .text")
  (emit "    .globl _scheme_entry")
  (emit "_scheme_entry:")
  (emit "    movl $~s, %eax" (immediate-rep x))
  (emit "    ret"))

(test-all)
