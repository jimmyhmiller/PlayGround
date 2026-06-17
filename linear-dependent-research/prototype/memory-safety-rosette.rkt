#lang rosette
;; ===========================================================================
;; lambda-Tally, Stage D: BOUNDED-EXHAUSTIVE memory-safety verification (Rosette)
;;
;; This complements the random fuzzing of memory-model.rkt (PLT Redex) with
;; SMT-backed *exhaustive* checking up to a program-length bound: Z3 considers
;; EVERY operation sequence of length <= k, not a random sample. It sits between
;; Redex (random) and a full Agda/Rocq proof (unbounded induction) on the
;; assurance ladder -- see docs/05.
;;
;; It targets the precise soundness crux identified in the research notes (the
;; one cell no existing system fills): when linear mutable VIEWS live inside a
;; DEPENDENT type theory, can an ERASED (multiplicity-0) reference to a
;; location's contents-type survive a linear STRONG UPDATE and cause a stale,
;; unsound access?
;;
;; The lambda-Tally claim (docs/02): no -- because runtime access requires a
;; LINEAR (multiplicity-1) view, while an erased proof (multiplicity 0) cannot
;; by itself authorise a read. We test that claim adversarially by running TWO
;; variants of the access discipline:
;;
;;   * SOUND  : every read requires a live linear view (this mirrors the types
;;              of read/write/free in the Stage-C prelude: they take View A l at
;;              multiplicity 1).
;;   * BROKEN : a read may be authorised by an erased proof alone (no view).
;;              This is "what if we got the multiplicity wrong".
;;
;; Expected: Rosette VERIFIES the sound discipline safe for all programs up to
;; length k, and FINDS A COUNTEREXAMPLE for the broken one -- demonstrating both
;; that the hazard is real/detectable and that linearity is exactly what removes
;; it. A counterexample in the SOUND variant would be a genuine soundness bug.
;;
;; Model: 2 locations, 2 runtime type-tags (A=1, B=2). Per location we track the
;; heap (alive?, actual stored type) and the program's resources (does it hold a
;; live linear view? of what type? an erased proof? of what type?). Each step is
;; a symbolic operation; its typing precondition is the GUARD (ill-typed steps
;; are no-ops, so only well-typed steps execute -- preservation by construction).
;;
;; Run:  racket memory-safety-rosette.rkt
;; ===========================================================================

(define L 2)          ; number of locations
;; runtime type tags: 1 = A, 2 = B ; 0 = uninitialised/freed (a distinct "type")
;; opcodes
(define ALLOC 0) (define WRITE 1) (define READ 2)
(define FREE 3)  (define MKPROOF 4) (define PROOFREAD 5)
(define NOPS 6)

;; ---- per-location state, fixed L=2, functional updates -------------------
(struct st (alive htype vview vtype pproof ptype) #:transparent)
(define (nth  lst i)   (if (= i 0) (first lst) (second lst)))
(define (setn lst i v) (if (= i 0) (list v (second lst)) (list (first lst) v)))
(define (init)
  (st (list #f #f) (list 0 0) (list #f #f) (list 0 0) (list #f #f) (list 0 0)))

;; ---- one interpreter step ------------------------------------------------
;; Returns the next state; READ/PROOFREAD emit a safety assertion that the type
;; actually stored at the location matches the type the access *claims*.
(define (step s op loc ty broken?)
  (define alive  (st-alive s))  (define htype (st-htype s))
  (define vview  (st-vview s))  (define vtype (st-vtype s))
  (define pproof (st-pproof s)) (define ptype (st-ptype s))
  (cond
    ;; alloc: only at a not-currently-live location; creates one LINEAR view.
    [(= op ALLOC)
     (if (not (nth alive loc))
         (st (setn alive loc #t) (setn htype loc ty)
             (setn vview loc #t) (setn vtype loc ty) pproof ptype)
         s)]
    ;; write = STRONG UPDATE: needs a live view; changes the stored type and
    ;; hands back a view at the NEW type. The view is consumed and re-produced
    ;; (still multiplicity 1). Any erased proof is left STALE on purpose.
    [(= op WRITE)
     (if (and (nth vview loc) (nth alive loc))
         (st alive (setn htype loc ty) vview (setn vtype loc ty) pproof ptype)
         s)]
    ;; read: needs a live view; SAFETY = stored type matches the view's claim.
    [(= op READ)
     (when (and (nth vview loc) (nth alive loc))
       (assert (= (nth htype loc) (nth vtype loc))))
     s]
    ;; free: needs a live view; reclaims the location (stored type -> 0).
    [(= op FREE)
     (if (and (nth vview loc) (nth alive loc))
         (st (setn alive loc #f) (setn htype loc 0)
             (setn vview loc #f) vtype pproof ptype)
         s)]
    ;; mkproof: forms an ERASED (mult-0) proof of the current type. Uses the
    ;; view at multiplicity 0, so it does NOT consume it (this is the subtle,
    ;; QTT-legal move: 0 + 1 = 1, the view survives for a later real use).
    [(= op MKPROOF)
     (if (and (nth vview loc) (nth alive loc))
         (st alive htype vview vtype (setn pproof loc #t) (setn ptype loc (nth vtype loc)))
         s)]
    ;; proofread: the adversarial access.
    [(= op PROOFREAD)
     (if broken?
         ;; BROKEN: an erased proof alone authorises a read -- no live view, no
         ;; aliveness required. SAFETY = stored type matches the PROOF's claim.
         (begin (when (nth pproof loc)
                  (assert (= (nth htype loc) (nth ptype loc))))
                s)
         ;; SOUND: a read still needs a live linear view (proof adds nothing).
         (begin (when (and (nth vview loc) (nth alive loc))
                  (assert (= (nth htype loc) (nth vtype loc))))
                s))]
    [else s]))

;; ---- run a whole symbolic program ---------------------------------------
(define (run ops locs tys broken?)
  (for/fold ([s (init)]) ([o ops] [l locs] [t tys])
    (step s o l t broken?)))

;; ---- one experiment: verify a variant for programs of length k ----------
(define (label broken?) (if broken? "BROKEN  (proof authorises read)" "SOUND   (read needs linear view)"))
(define op-name (vector "alloc" "write" "read" "free" "mkproof" "proofread"))

(define (run-experiment broken? k)
  (define ops  (for/list ([i (in-range k)]) (define-symbolic* o integer?) o))
  (define locs (for/list ([i (in-range k)]) (define-symbolic* l integer?) l))
  (define tys  (for/list ([i (in-range k)]) (define-symbolic* t integer?) t))
  (define sol
    (verify
     (begin
       (for ([o ops] [l locs] [t tys])
         (assume (&& (>= o 0) (< o NOPS) (>= l 0) (< l L) (>= t 1) (<= t 2))))
       (run ops locs tys broken?))))
  (cond
    [(unsat? sol)
     (printf "  [k=~a] ~a : VERIFIED safe (no counterexample in any program)\n" k (label broken?))]
    [else
     (printf "  [k=~a] ~a : COUNTEREXAMPLE\n" k (label broken?))
     (define cops (evaluate ops sol)) (define clocs (evaluate locs sol)) (define ctys (evaluate tys sol))
     (for ([o cops] [l clocs] [t ctys] [i (in-naturals)])
       (when (and (integer? o) (<= 0 o) (< o NOPS))
         (printf "        ~a) ~a  loc=~a type=~a\n" i (vector-ref op-name o) l t)))]))

;; ---- main ----------------------------------------------------------------
(printf "lambda-Tally bounded memory-safety check (Rosette/Z3)\n")
(printf "locations=~a, type-tags={A=1,B=2}; probing erased-reference + strong-update.\n\n" L)

(printf "SOUND discipline (mirrors the Stage-C primitive types):\n")
(for ([k '(4 8 12 16)]) (run-experiment #f k))

(printf "\nBROKEN discipline (erased proof may authorise a read):\n")
(for ([k '(3 4)]) (run-experiment #t k))

(printf "\nReading the result: 'SOUND ... VERIFIED' means linearity provably prevents\n")
(printf "the stale-erased-reference hazard for all programs up to that length. The\n")
(printf "BROKEN counterexample shows the same hazard is real once a read can be\n")
(printf "authorised by an erased (multiplicity-0) proof instead of a linear view.\n")
