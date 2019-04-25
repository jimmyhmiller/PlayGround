(load "test-driver.scm")
(load "tests/tests-1.1-req.scm")
;;(load "tests/tests-1.2-req.scm")
;;(load "tests/tests-1.3-req.scm")
;;(load "tests/tests-1.4-req.scm")

(define fxtag #x00)
(define fxshift 2)
(define fxmask #x03)
(define boolmask #xBF)
(define bool-f #x2F)
(define bool-t #x6F)
(define null-rep #x3F)
(define chartag #x0F)
(define charshift 8)
(define charmask #x3F)
(define word-size 4)
(define bool-bit 6)

(define fixnum-bits (- (* word-size 8) fxshift))
(define fxlower (- (expt 2 (- fixnum-bits 1))))
(define fxupper (sub1 (expt 2 (- fixnum-bits 1))))

(define (fixnum? x)
  (and (integer? x)
       (exact? x)
       (<= fxlower x fxupper)))

(define (immediate? x)
  (or (fixnum? x)
      (boolean? x)
      (char? x)
      (null? x)))

(define (char-rep x)
  (bitwise-ior (ash (char->integer x) charshift) chartag))

(define (immediate-rep x)
  (cond
   [(fixnum? x) (ash x fxshift)]
   [(equal? x #f) bool-f]
   [(equal? x #t) bool-t]
   [(char? x) (char-rep x)]
   [(null? x) null-rep]))


(define-syntax define-primitive
  (syntax-rules ()
    [(_ (prim-name stack-index arg* ...) b b* ...)
     (begin
       (putprop 'prim-name '*is-prim* #t)
       (putprop 'prim-name '*arg-count* (length '(arg* ...)))
       (putprop 'prim-name '*emitter*
                (lambda (stack-index arg* ...) b b* ...)))]))

(define (primitive? x)
  (and (symbol? x)
       (getprop x '*is-prim*)))

(define (primitive-emitter x)
  (or (getprop x '*emitter*) (raise "No emitter found")))

(define (primcall? expr)
  (and (pair? expr) (primitive? (car expr))))


(define-primitive (fxadd1 stack-index arg)
  (emit-expr stack-index arg)
  (emit "    addl $~s, %eax" (immediate-rep 1)))

(define-primitive (fxsub1 stack-index arg)
  (emit-expr stack-index arg)
  (emit "    subl $~s, %eax" (immediate-rep 1)))

(define-primitive (fixnum->char stack-index arg)
  (emit-expr stack-index arg)
  (emit "    shll $~s, %eax" (- charshift fxshift))
  (emit "    orl $~s, %eax" chartag))

(define-primitive (char->fixnum stack-index arg)
  (emit-expr stack-index arg)
  (emit "    shrl $~s, %eax" (- charshift fxshift))
  (emit "    orl $~s, %eax" fxtag))

(define-primitive (fixnum? stack-index arg)
  (emit-expr stack-index arg)
  (emit "    and $~s, %al" fxmask)
  (emit "    cmp $~s, %al" fxtag)
  (emit "    sete %al")
  (emit "    movzbl %al, %eax")
  (emit "    sal $~s, %al" bool-bit)
  (emit "    or $~s, %al" bool-f))

(define-primitive (fxzero? stack-index arg)
  (emit-expr stack-index arg)
  (emit "    cmp $~s, %eax" fxtag)
  (emit "    sete %al")
  (emit "    movzbl %al, %eax")
  (emit "    sal $~s, %al" bool-bit)
  (emit "    or $~s, %al" bool-f))

(define-primitive (null? stack-index arg)
  (emit-expr stack-index arg)
  (emit "    cmp $~s, %eax" null-rep)
  (emit "    sete %al")
  (emit "    movzbl %al, %eax")
  (emit "    sal $~s, %al" bool-bit)
  (emit "    or $~s, %al" bool-f))

(define-primitive (boolean? stack-index arg)
  (emit-expr stack-index arg)
  (emit "    and $~s, %eax" boolmask)
  (emit "    cmp $~s, %eax" bool-f)
  (emit "    sete %al")
  (emit "    movzbl %al, %eax")
  (emit "    sal $~s, %al" bool-bit)
  (emit "    or $~s, %al" bool-f))


(define-primitive (char? stack-index arg)
  (emit-expr stack-index arg)
  (emit "    and $~s, %eax" charmask)
  (emit "    cmp $~s, %eax" chartag)
  (emit "    sete %al")
  (emit "    movzbl %al, %eax")
  (emit "    sal $~s, %al" bool-bit)
  (emit "    or $~s, %al" bool-f))

(define-primitive (not stack-index arg)
  (emit-expr stack-index arg)
  (emit "    cmp $~s, %eax" bool-f)
  (emit "    sete %al")
  (emit "    movzbl %al, %eax")
  (emit "    sal $~s, %al" bool-bit)
  (emit "    or $~s, %al" bool-f))

(define-primitive (fxlognot stack-index arg)
  (emit-expr stack-index arg)
  (emit "    shr $~s, %eax" fxshift)
  (emit "    not %eax")
  (emit "    shl $~s, %eax" fxshift))


(define unique-label
  (let ([count 0])
    (lambda ()
      (let ([L (format "L_~s" count)])
        (set! count (add1 count))
        L))))

(define (if? expr)
  (and (pair? expr)
       (equal? (car expr) 'if)))

(define (if-test expr)
  (cadr expr))

(define (if-conseq expr)
  (caddr expr))

(define (if-altern expr)
  (cadddr expr))

(define (emit-if stack-index expr)
  (let ([alt-label (unique-label)]
        [end-label (unique-label)])
    (emit-expr stack-index (if-test expr))
    (emit "    cmp $~s, %al" bool-f)
    (emit "    je ~a" alt-label)
    (emit-expr stack-index (if-conseq expr))
    (emit "    jmp ~a" end-label)
    (emit "~a:" alt-label)
    (emit-expr stack-index (if-altern expr))
    (emit "~a:" end-label)))


(define (check-primcall-args prim args)
  #t)

(define (emit-primcall stack-idnex expr)
  (let ([prim (car expr)]
        [args (cdr expr)])
    (check-primcall-args prim args)
    (apply (primitive-emitter prim) stack-index args)))

(define (emit-immediate expr)
  (emit "    movl $~s, %eax" (immediate-rep expr)))

(define (emit-expr stack-index expr)
  (cond
   [(immediate? expr) (emit-immediate  expr)]
   [(if? expr) (emit-if stack-index expr)]
   [(primcall? expr) (emit-primcall stack-index expr)]
   [else (raise (condition (make-error)
                           (make-message-condition (format "Expression not recognized: ~s" expr))))]))

(define (emit-function-header name)
  (emit "    .text")
  (emit "    .globl ~s" name)
  (emit "~s:" name))


;; 64 bit vs 32 bit is causing issues
;; Do I try to forge ahead or figure out how to run this 32bit?
(define (emit-program expr)
  (emit-function-header "_L_scheme_entry")
  (emit-expr (- word-size) expr)
  (emit "    ret")
  (emit-function-header "_scheme_entry")
  (emit "    movl %esp, %ecx")
  (emit "    movl 4(%esp), %esp")
  (emit "    call _L_scheme_entry")
  (emit "    movl %ecx, %esp")
  (emit "    ret"))



(test-all)
