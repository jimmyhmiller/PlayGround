#lang rosette
;; ===========================================================================
;; lambda-Tally, Stage D / E3-E4: connecting the STATIC multiplicity checker to
;; DYNAMIC resource safety, verified with Rosette/Z3.
;;
;; The memory model (memory-safety-rosette.rkt) verified the *operational*
;; discipline. This file closes the loop to the *type system*: it has a real
;; term language, a leftover-context multiplicity TYPE CHECKER, and an
;; environment EVALUATOR with a resource heap, and proves the central
;; resource-soundness theorem (the operational core of GraD / docs/05):
;;
;;     if the checker ACCEPTS a closed program, then evaluating it
;;        (a) never double-frees / uses-after-free a resource, and
;;        (b) leaks nothing (every resource created is consumed).
;;
;; This is exactly "the multiplicities deliver the operational guarantee" -- the
;; thing that makes the whole idea worth pursuing. We verify it for ALL
;; well-typed programs up to a term-depth bound (symbolic terms, bottom-up over a
;; fixed tree skeleton -- the standard tractable encoding; substitution is
;; avoided by using an environment evaluator, per the E3 plan in docs/06).
;;
;; To show the linearity check is load-bearing we also run a BROKEN checker that
;; keeps the type checks but DROPS the "used exactly once" rule: Z3 then finds a
;; program it accepts that leaks or double-frees.
;;
;; Language (de Bruijn), types U (unit) and R (linear resource):
;;   unit | new | use e | var i | seq e1 e2 | let e1 e2
;;   new : R         (allocate a resource)         use e : U   (consume e:R)
;;   seq e1 e2       (e1:U discarded, then e2)      let e1 e2  (bind x=e1 in e2)
;; Typing is linear: every bound variable must be used exactly once.
;;
;; Run:  racket resource-soundness-rosette.rkt
;; ===========================================================================

;; ---- constructors / types ------------------------------------------------
(define UNIT 0) (define NEW 1) (define VAR 2) (define USE 3) (define SEQ 4) (define LET 5)
(define U 0) (define R 1)                       ; types

;; ---- terms: a node carries a (concrete) position used as a resource id ----
(struct node (tag data left right pos) #:transparent)
(define POS (box 0))
(define (reset!) (set-box! POS 0))
(define (fresh!) (let ([p (unbox POS)]) (set-box! POS (add1 p)) p))
(define (leaf? t) (not (node-left t)))          ; CONCRETE: depth-0 node

;; ---- small list utilities (symbolic-index safe over concrete-length lists) -
(define (zeros n)  (make-list n 0))
(define (falses n) (make-list n #f))
(define (get lst i dflt)                        ; lst[i] or dflt; i may be symbolic
  (for/fold ([acc dflt]) ([x lst] [j (in-naturals)]) (if (= j i) x acc)))
(define (setx lst i v) (for/list ([x lst] [j (in-naturals)]) (if (= j i) v x)))
(define (unit-vec n i) (for/list ([j (in-range n)]) (if (= j i) 1 0)))
(define (vec+ a b) (map + a b))
(define (conj lst) (foldl (lambda (x acc) (&& x acc)) #t lst))

;; ---- the multiplicity TYPE CHECKER (leftover context) --------------------
;; returns (list ok? type usage) ; usage : per-context-var use counts.
;; `strict` = enforce linearity (each bound var used exactly once).
(define (typeof ctx t strict)
  (define tag (node-tag t)) (define n (length ctx))
  (cond
    [(= tag UNIT) (list #t U (zeros n))]
    [(= tag NEW)  (list #t R (zeros n))]
    [(= tag VAR)
     (define i (node-data t))
     (list (&& (>= i 0) (< i n)) (get ctx i U) (unit-vec n i))]
    [(leaf? t) (list #f U (zeros n))]                    ; concrete cutoff: no children
    [(= tag USE)
     (define r (typeof ctx (node-left t) strict))
     (list (&& (first r) (= (second r) R)) U (third r))] ; arg must be R; result U
    [(= tag SEQ)
     (define r1 (typeof ctx (node-left t) strict))
     (define r2 (typeof ctx (node-right t) strict))
     (list (&& (first r1) (first r2) (= (second r1) U)) (second r2)
           (vec+ (third r1) (third r2)))]
    [(= tag LET)
     (define r1 (typeof ctx (node-left t) strict))
     (define r2 (typeof (cons (second r1) ctx) (node-right t) strict))  ; bind x : type-of-e1
     (define used-x (first (third r2)))                  ; x is de Bruijn 0
     (list (&& (first r1) (first r2) (if strict (= used-x 1) #t))        ; LINEARITY here
           (second r2) (vec+ (third r1) (rest (third r2))))]
    [else (list #f U (zeros n))]))

;; ---- the EVALUATOR (environment machine + resource heap) -----------------
;; value: kind U/R and a resource id (-1 for unit). heap `live`: live?[id].
;; result threaded as (list value live err?) where err? records any UAF/double-free.
(struct val (kind id) #:transparent)
(define VUNIT (val U -1))
(define (ev env t live err)
  (define tag (node-tag t))
  (cond
    [(= tag UNIT) (list VUNIT live err)]
    [(= tag NEW)  (list (val R (node-pos t)) (setx live (node-pos t) #t) err)] ; allocate
    [(= tag VAR)  (list (get env (node-data t) VUNIT) live err)]
    [(leaf? t)    (list VUNIT live err)]
    [(= tag USE)
     (define r (ev env (node-left t) live err))
     (define id (val-id (first r)))
     (list VUNIT (setx (second r) id #f)                ; free it
           (|| (third r) (! (get (second r) id #f))))]  ; err if it was NOT live (UAF/double)
    [(= tag SEQ)
     (define r (ev env (node-left t) live err))
     (ev env (node-right t) (second r) (third r))]
    [(= tag LET)
     (define r (ev env (node-left t) live err))
     (ev (cons (first r) env) (node-right t) (second r) (third r))]
    [else (list VUNIT live err)]))

;; structural well-formedness of a symbolic term (tag ranges; var index >= 0)
(define (wf t)
  (if (leaf? t)
      (&& (>= (node-tag t) 0) (<= (node-tag t) 2) (>= (node-data t) 0))   ; leaf: UNIT/NEW/VAR
      (&& (>= (node-tag t) 0) (<= (node-tag t) 5) (>= (node-data t) 0)
          (wf (node-left t)) (wf (node-right t)))))

;; ---- build a symbolic term of depth d (complete skeleton) ----------------
(define (sym-term d)
  (define-symbolic* tag integer?)
  (define-symbolic* dat integer?)
  (if (= d 0)
      (node tag dat #f #f (fresh!))
      (let* ([p (fresh!)] [l (sym-term (- d 1))] [r (sym-term (- d 1))])
        (node tag dat l r p))))

;; ===========================================================================
;; THE THEOREM: accepts(e) => run(e) has no UAF/double-free and no leak.
;; Verified for every well-typed closed program of depth <= D.
;; ===========================================================================
(define (verify-resource-soundness strict D)
  (reset!)
  (define root (sym-term D))
  (define N (unbox POS))                          ; node count = resource-id space
  (define tr (typeof '() root strict))            ; (ok type usage), closed: ctx '()
  (define ok (first tr)) (define ty (second tr))
  (define res (ev '() root (falses N) #f))
  (define final-live (second res)) (define err (third res))
  (define sol
    (verify
     (begin
       (assume (wf root))
       (assume ok)                                ; the checker accepts it
       (assume (= ty U))                          ; a complete program returns unit
       (assert (&& (! err)                        ; (a) no use-after-free / double-free
                   (conj (map ! final-live)))))))  ; (b) no leak
  (cond
    [(unsat? sol)
     (printf "  ~a checker, depth<=~a : SOUND -- accepts(e) => run(e) is leak-free and UAF-free\n"
             (if strict "STRICT (linear)" "BROKEN (no linearity)") D)]
    [else
     (printf "  ~a checker, depth<=~a : COUNTEREXAMPLE (accepted yet unsafe)\n"
             (if strict "STRICT (linear)" "BROKEN (no linearity)") D)
     (printf "        ~a\n" (show (evaluate root sol)))]))

;; pretty-print a concrete (post-solve) term
(define (show t)
  (define nm (vector "unit" "new" "var" "use" "seq" "let"))
  (define tag (node-tag t))
  (cond
    [(not (and (integer? tag) (<= 0 tag 5))) "?"]
    [(= tag UNIT) "unit"] [(= tag NEW) "new"] [(= tag VAR) (format "x~a" (node-data t))]
    [(leaf? t) "?"]
    [(= tag USE) (format "(use ~a)" (show (node-left t)))]
    [(= tag SEQ) (format "(seq ~a ~a)" (show (node-left t)) (show (node-right t)))]
    [(= tag LET) (format "(let = ~a in ~a)" (show (node-left t)) (show (node-right t)))]
    [else "?"]))

;; ---- concrete sanity tests (validate checker + evaluator logic) ----------
(define (mk tag data . kids)
  (if (null? kids) (node tag data #f #f (fresh!))
      (node tag data (first kids) (second kids) (fresh!))))
(define (cunit) (mk UNIT 0)) (define (cnew) (mk NEW 0)) (define (cvar i) (mk VAR i))
(define (cuse e) (mk USE 0 e (cunit)))
(define (cseq a b) (mk SEQ 0 a b)) (define (clet a b) (mk LET 0 a b))

(define (concrete-check name t)
  (reset!) (set-box! POS 0)
  (define tr (typeof '() t #t)) (define ok (first tr)) (define ty (second tr))
  (define N 64)
  (define res (ev '() t (falses N) #f))
  (define leak (ormap (lambda (b) b) (second res)))
  (printf "  ~a : typechecks=~a type=~a | runtime-error=~a leak=~a\n"
          name ok (if (equal? ty U) 'U 'R) (third res) leak))

(printf "=== concrete sanity checks (strict checker) ===\n")
(concrete-check "use new                 " (cuse (cnew)))
(concrete-check "let x=new in use x       " (clet (cnew) (cuse (cvar 0))))
(concrete-check "let x=new in unit  (leak)" (clet (cnew) (cunit)))
(concrete-check "let x=new in use x;use x " (clet (cnew) (cseq (cuse (cvar 0)) (cuse (cvar 0)))))
(newline)

;; ---- the verification ----------------------------------------------------
(printf "=== METATHEOREM (E3/E4): resource soundness, all well-typed programs ===\n")
(verify-resource-soundness #t 2)
(verify-resource-soundness #t 3)
(verify-resource-soundness #t 4)
(verify-resource-soundness #f 2)
(printf "\nSTRICT = the multiplicity checker (linear). BROKEN = same type checks but the\n")
(printf "'used exactly once' rule dropped. The contrast shows linearity is exactly what\n")
(printf "turns a type checker into a memory-safety guarantee.\n")
