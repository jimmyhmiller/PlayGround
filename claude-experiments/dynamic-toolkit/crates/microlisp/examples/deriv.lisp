;; deriv.lisp — symbolic differentiation in microlisp
;;
;; Demonstrates:
;;   - A non-trivial macro (`case`) whose body calls a user-defined helper
;;     (`case-expand`) at expansion time, in the same JIT image.
;;   - Recursive `cond` macro that re-emits itself.
;;   - Heavy cons-cell allocation through a recursive `deriv` function.
;;   - Quote-literals embedded in source that go through the GC-traced
;;     LiteralPool — the program continues to work after a moving GC
;;     relocates them.
;;
;; Run normally:
;;     cargo run -p microlisp -- crates/microlisp/examples/deriv.lisp
;;
;; Run with aggressive GC (collects between every top-level form):
;;     MICROLISP_GC_STRESS=1 cargo run -p microlisp -- \
;;         crates/microlisp/examples/deriv.lisp

;; ── Macros ──────────────────────────────────────────────────────

(defmacro cond clauses
  (if (null? clauses)
      'nil
      (let ((first (car clauses))
            (rest (cdr clauses)))
        (if (eq? (car first) 'else)
            `(begin ,@(cdr first))
            `(if ,(car first)
                 (begin ,@(cdr first))
                 (cond ,@rest))))))

;; Helpers used by the case macro at expansion time. These must be
;; defined BEFORE the `case` macro so the macro body can resolve them
;; to JIT-compiled FuncRefs when it's compiled.

(define (member? x lst)
  (if (null? lst)
      #f
      (if (eq? x (car lst))
          #t
          (member? x (cdr lst)))))

(define (case-expand var clauses)
  (if (null? clauses)
      'nil
      (let ((c (car clauses))
            (rest (cdr clauses)))
        (if (eq? (car c) 'else)
            (cons 'begin (cdr c))
            (list 'if
                  (list 'member? var (list 'quote (car c)))
                  (cons 'begin (cdr c))
                  (case-expand var rest))))))

(defmacro case (scrut . clauses)
  (let ((g (gensym 'case)))
    `(let ((,g ,scrut))
       ,(case-expand g clauses))))

;; ── List utilities ──────────────────────────────────────────────

(define (cadr x)  (car (cdr x)))
(define (caddr x) (car (cdr (cdr x))))

;; ── Symbolic differentiator ────────────────────────────────────
;;
;; Rules:
;;   d(c)/dx       = 0           (c a number)
;;   d(x)/dx       = 1
;;   d(y)/dx       = 0           (y any other symbol)
;;   d(u + v)/dx   = du/dx + dv/dx
;;   d(u * v)/dx   = u*(dv/dx) + (du/dx)*v

(define (deriv expr var)
  (cond
    ((number? expr) 0)
    ((symbol? expr) (if (eq? expr var) 1 0))
    (else
      (case (car expr)
        ((+) (list '+
                   (deriv (cadr expr) var)
                   (deriv (caddr expr) var)))
        ((*) (list '+
                   (list '* (cadr expr)             (deriv (caddr expr) var))
                   (list '* (deriv (cadr expr) var) (caddr expr))))
        (else 'unknown-op)))))

;; ── Heavy allocation driver ─────────────────────────────────────
;; Calls `deriv` n times; each call allocates ~10 cons cells, all of
;; which become unreachable garbage after the iteration. Stresses the
;; GC by filling from-space.

(define (loop-deriv n)
  (if (eq? n 0)
      nil
      (begin
        (deriv '(+ (* x x) (* 3 x)) 'x)
        (loop-deriv (- n 1)))))

;; ── Driver ──────────────────────────────────────────────────────
;; Each `print` is a separate top-level form, so under GC stress mode
;; a moving collection runs between them. The quote-literals embedded
;; in `deriv`'s source ('+, '*, etc.) are scanned and relocated; the
;; emitted `gc_literal` loads pick up the new addresses on the next
;; call without any code patching.

(print (deriv 'x 'x))                     ;; => 1
(print (deriv 'y 'x))                     ;; => 0
(print (deriv 5 'x))                      ;; => 0

(print (deriv '(+ x x) 'x))               ;; => (+ 1 1)
(print (deriv '(* x x) 'x))               ;; => (+ (* x 1) (* 1 x))

(loop-deriv 500)

(print (deriv '(* x (* x x)) 'x))         ;; survives 500 iterations of garbage
(print (deriv '(+ (* x x) (* 3 x)) 'x))

(loop-deriv 1000)

(print (deriv '(* (+ x x) (* x x)) 'x))   ;; still works after another 1000

(print 'done)
