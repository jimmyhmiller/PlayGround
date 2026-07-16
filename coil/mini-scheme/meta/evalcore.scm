;; A first-order metacircular Scheme evaluator. Portable Scheme (runs on Chez) AND
;; the exact subset our Scheme->Coil metaprogram compiles. Closures/prims are DATA
;; (tagged lists); the global environment is a single mutable variable.
(define genv (quote ()))
(define (defvar! name val) (set! genv (cons (cons name val) genv)))
(define (cadr* x) (car (cdr x)))
(define (caddr* x) (car (cdr (cdr x))))
(define (cadddr* x) (car (cdr (cdr (cdr x)))))

(define (glookup s g)
  (if (null? g) s (if (eq? (car (car g)) s) (cdr (car g)) (glookup s (cdr g)))))
(define (lookup s env)
  (if (null? env) (glookup s genv)
      (if (eq? (car (car env)) s) (cdr (car env)) (lookup s (cdr env)))))
(define (bind ps as env)
  (if (null? ps) env (bind (cdr ps) (cdr as) (cons (cons (car ps) (car as)) env))))
(define (evlist es env)
  (if (null? es) (quote ()) (cons (seval (car es) env) (evlist (cdr es) env))))

(define (seval e env)
  (if (number? e) e
  (if (symbol? e) (lookup e env)
  (if (eq? (car e) (quote quote)) (cadr* e)
  (if (eq? (car e) (quote if))
      (if (seval (cadr* e) env) (seval (caddr* e) env) (seval (cadddr* e) env))
  (if (eq? (car e) (quote lambda))
      (cons (quote closure) (cons (cadr* e) (cons (caddr* e) env)))
  (if (eq? (car e) (quote define))
      (defvar! (cadr* e) (seval (caddr* e) env))
      (sapply (seval (car e) env) (evlist (cdr e) env)))))))))

(define (sapply f as)
  (if (eq? (car f) (quote prim)) (doprim (cadr* f) as)
      (seval (caddr* f) (bind (cadr* f) as (cdddr* f)))))
(define (cdddr* x) (cdr (cdr (cdr x))))
(define (doprim op as)
  (if (eq? op (quote +)) (+ (car as) (cadr* as))
  (if (eq? op (quote -)) (- (car as) (cadr* as))
  (if (eq? op (quote *)) (* (car as) (cadr* as))
  (if (eq? op (quote <)) (< (car as) (cadr* as))
  (if (eq? op (quote =)) (= (car as) (cadr* as)) 0))))))

(define (setup)
  (defvar! (quote +) (cons (quote prim) (cons (quote +) (quote ()))))
  (defvar! (quote -) (cons (quote prim) (cons (quote -) (quote ()))))
  (defvar! (quote *) (cons (quote prim) (cons (quote *) (quote ()))))
  (defvar! (quote <) (cons (quote prim) (cons (quote <) (quote ()))))
  (defvar! (quote =) (cons (quote prim) (cons (quote =) (quote ())))))

;; the TARGET program the evaluator runs: fib, as data
(define target
  (quote (begin
    (define fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))
    (fib 25))))

(define (run)
  (setup)
  (seval (cadr* target) (quote ()))              ; the (define fib ...)
  (seval (caddr* target) (quote ())))            ; the (fib 25)
(display (run))
(newline)
