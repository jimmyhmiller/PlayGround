#lang rosette
;; ===========================================================================
;; lambda-Tally, Stage D: BOUNDED-EXHAUSTIVE memory-safety verification (Rosette)
;;
;; Complements the random fuzzing of memory-model.rkt (PLT Redex) with SMT-backed
;; *exhaustive* checking up to a program-length bound: Z3 considers EVERY
;; operation sequence of length <= k, not a random sample. It sits between Redex
;; (random) and a full Agda/Rocq proof (unbounded induction) on the assurance
;; ladder -- see docs/05.
;;
;; This version is genuinely DEPENDENT: locations hold length-indexed arrays, and
;; a view carries the length. An element access requires a static bounds check
;; (arg < the view's claimed length) -- our stand-in for a dependent proof that
;; the index `i < n`. `resize` is a STRONG UPDATE that *changes the dependent
;; index* n. So the failure mode is now a real OUT-OF-BOUNDS memory access (or a
;; use-after-free), not mere tag confusion.
;;
;; It targets the soundness crux no existing system addresses (docs/05): when a
;; linear mutable view lives inside a dependent type theory, can a STALE claim
;; about a location's index survive a linear strong update and authorise an
;; unsound access? Two ways to get a stale claim -- an ERASED (multiplicity-0)
;; dependent proof "l : Array n", or a DUPLICATED linear view (aliasing) -- are
;; the SAME hazard here: "a length claim that outlives the mutation". We model
;; both with one secondary-capability slot `scap`, and run two disciplines:
;;
;;   * SOUND  : a read requires a live LINEAR view (mirrors the Stage-C primitive
;;              types: read/write/free take View A l at multiplicity 1). A stale
;;              claim cannot authorise a read.
;;   * BROKEN : a stale claim alone authorises a read ("what if we mis-graded the
;;              view, or allowed an erased proof to be used at runtime").
;;
;; Expected: Rosette VERIFIES SOUND safe for all programs up to length k, and
;; finds a COUNTEREXAMPLE for BROKEN -- showing the hazard is real/detectable and
;; that linearity is exactly what removes it. A SOUND counterexample would be a
;; genuine soundness bug.
;;
;; Each step's typing precondition is the GUARD: ill-typed steps are no-ops, so
;; only well-typed steps execute (preservation by construction). Per location we
;; track the heap (alive?, actual current length) and the program's capabilities
;; (a live linear view + its claimed length; a secondary stale claim + its len).
;;
;; Run:  racket memory-safety-rosette.rkt
;; ===========================================================================

(define L 2)        ; number of locations
(define MAXLEN 3)   ; array lengths / indices range over 0..MAXLEN
;; opcodes
(define ALLOC 0) (define READ 1) (define WRITE 2) (define RESIZE 3)
(define FREE 4)  (define MKCLAIM 5) (define STALEREAD 6)
(define NOPS 7)
;; disciplines
(define SOUND 0) (define BROKEN 1)

;; ---- per-location state, fixed L=2, functional updates -------------------
;;   alive  : is the location live?
;;   len    : its actual current array length (the runtime index)
;;   vview  : does the program hold a live LINEAR view of this location?
;;   vlen   : the length that view claims  (invariant under SOUND: vlen = len)
;;   scap   : does the program hold a SECONDARY claim (erased proof / dup view)?
;;   slen   : the length that secondary claim asserts (may go STALE on resize/free)
(struct st (alive len vview vlen scap slen) #:transparent)
(define (nth  lst i)   (if (= i 0) (first lst) (second lst)))
(define (setn lst i v) (if (= i 0) (list v (second lst)) (list (first lst) v)))
(define (init)
  (st (list #f #f) (list 0 0) (list #f #f) (list 0 0) (list #f #f) (list 0 0)))

;; The capability an op reads through, as (have? . claimed-length):
;;   have?   : is the capability present (and, for a view, the loc live)?
;;   claimed : the length it asserts -> the STATIC bounds check (arg < claimed),
;;             our stand-in for a dependent `i < n` proof.
(define (cap-of s op loc mode)
  (cond
    [(or (= op READ) (= op WRITE))
     (cons (&& (nth (st-vview s) loc) (nth (st-alive s) loc)) (nth (st-vlen s) loc))]
    [(= op STALEREAD)
     (if (= mode SOUND)
         (cons (&& (nth (st-vview s) loc) (nth (st-alive s) loc)) (nth (st-vlen s) loc))
         (cons (nth (st-scap s) loc) (nth (st-slen s) loc)))]   ; BROKEN: stale claim alone
    [else (cons #f 0)]))                                         ; non-accessing op

;; SAFETY: whenever an access fires (have? && arg<claimed), it must actually be
;; in-bounds of live memory (alive && arg < the REAL current length).
(define (safe-step? s op loc arg mode)
  (define c (cap-of s op loc mode))
  (==> (&& (car c) (< arg (cdr c)))
       (&& (nth (st-alive s) loc) (< arg (nth (st-len s) loc)))))

;; ---- one interpreter step (PURE: state transition only) ------------------
(define (step s op loc arg mode)
  (define alive (st-alive s)) (define len  (st-len s))
  (define vview (st-vview s)) (define vlen (st-vlen s))
  (define scap  (st-scap s))  (define slen (st-slen s))
  (cond
    ;; alloc an array of length `arg`; yields ONE linear view; clears any claim.
    [(= op ALLOC)
     (if (not (nth alive loc))
         (st (setn alive loc #t) (setn len loc arg)
             (setn vview loc #t) (setn vlen loc arg)
             (setn scap loc #f)  (setn slen loc 0))
         s)]
    ;; resize = STRONG UPDATE of the index: heap length AND the view's claim move
    ;; together; any secondary claim is intentionally left STALE.
    [(= op RESIZE)
     (if (&& (nth vview loc) (nth alive loc))
         (st alive (setn len loc arg) vview (setn vlen loc arg) scap slen)
         s)]
    ;; free: consume the view, reclaim (length -> 0). Secondary claim left stale.
    [(= op FREE)
     (if (&& (nth vview loc) (nth alive loc))
         (st (setn alive loc #f) (setn len loc 0) (setn vview loc #f) vlen scap slen)
         s)]
    ;; mkclaim: form a SECONDARY claim of the current length -- either an erased
    ;; (multiplicity-0) dependent proof, or a duplicated view (aliasing). Uses
    ;; the view at multiplicity 0, so it does NOT consume it (0 + 1 = 1).
    [(= op MKCLAIM)
     (if (&& (nth vview loc) (nth alive loc))
         (st alive len vview vlen (setn scap loc #t) (setn slen loc (nth vlen loc)))
         s)]
    ;; read / write / staleread do not change the state (only `safe-step?` cares).
    [else s]))

;; ---- run a whole symbolic program (asserting safety at each step) --------
(define (run ops locs args mode)
  (for/fold ([s (init)]) ([o ops] [l locs] [a args])
    (assert (safe-step? s o l a mode))
    (step s o l a mode)))

;; ---- one experiment: verify a discipline for programs of length k --------
(define (label mode) (if (= mode SOUND)
                         "SOUND  (read needs live linear view)"
                         "BROKEN (stale claim authorises read)"))
(define op-name (vector "alloc" "read" "write" "resize" "free" "mkclaim" "staleread"))

(define (run-experiment mode k #:forbid [forbid -1])  ; forbid: an opcode to exclude
  (define ops  (for/list ([i (in-range k)]) (define-symbolic* o integer?) o))
  (define locs (for/list ([i (in-range k)]) (define-symbolic* l integer?) l))
  (define args (for/list ([i (in-range k)]) (define-symbolic* a integer?) a))
  (define sol
    (verify
     (begin
       (for ([o ops] [l locs] [a args])
         (assume (&& (>= o 0) (< o NOPS) (not (= o forbid))
                     (>= l 0) (< l L) (>= a 0) (<= a MAXLEN))))
       (run ops locs args mode))))
  (cond
    [(unsat? sol)
     (printf "  [k=~a] ~a : VERIFIED safe (no counterexample in any program)\n" k (label mode))]
    [else
     (printf "  [k=~a] ~a : COUNTEREXAMPLE\n" k (label mode))
     (define cops (evaluate ops sol)) (define clocs (evaluate locs sol)) (define cargs (evaluate args sol))
     (for ([o cops] [l clocs] [a cargs] [i (in-naturals)])
       (when (and (integer? o) (<= 0 o) (< o NOPS))
         (define kind (if (or (= o ALLOC) (= o RESIZE)) "len" "idx"))
         (printf "        ~a) ~a  loc=~a ~a=~a\n" i (vector-ref op-name o) l kind a)))]))

;; ===========================================================================
;; THE METATHEOREM: an INDUCTIVE store-typing invariant.
;;
;; The bounded checks above say "safe for every program up to length k". This is
;; stronger: we exhibit an invariant Inv(state) and prove in Z3 that
;;   (P) every single operation, from ANY state satisfying Inv, (a) performs no
;;       unsafe access and (b) leaves a state still satisfying Inv.
;; The initial state satisfies Inv, so by induction Inv -- hence safety -- holds
;; at every reachable state, for programs of ANY length and arrays of ANY size.
;; The trace length is now UNBOUNDED; only the number of locations stays finite
;; (and locations are independent, so the argument is uniform in that too).
;;
;; Inv: holding a live linear view of a location implies the location is alive
;; and the view's claimed length equals the actual length (store-typing
;; consistency). That is exactly what makes a SOUND read in-bounds.
;; ===========================================================================
(define (==> a b) (|| (! a) b))
(define (all lst) (foldl (lambda (x acc) (&& x acc)) #t lst))

(define (inv s)
  (all (for/list ([loc (in-range L)])
         (==> (nth (st-vview s) loc)
              (&& (nth (st-alive s) loc)
                  (= (nth (st-vlen s) loc) (nth (st-len s) loc)))))))

(define (sym-bool) (define-symbolic* b boolean?) b)
(define (sym-nat)  (define-symbolic* n integer?) n)
(define (arb-state)                       ; a completely arbitrary machine state
  (st (list (sym-bool) (sym-bool)) (list (sym-nat) (sym-nat))
      (list (sym-bool) (sym-bool)) (list (sym-nat) (sym-nat))
      (list (sym-bool) (sym-bool)) (list (sym-nat) (sym-nat))))
(define (nats-nonneg s)                   ; lengths are naturals (otherwise UNBOUNDED)
  (all (for/list ([n (append (st-len s) (st-vlen s) (st-slen s))]) (>= n 0))))

(define (verify-inductive mode)
  (define s (arb-state))
  (define-symbolic* op integer?) (define-symbolic* loc integer?) (define-symbolic* arg integer?)
  (define sol
    (verify
     (begin
       (assume (nats-nonneg s))
       (assume (inv s))                                  ; arbitrary Inv-state
       (assume (&& (>= op 0) (< op NOPS) (>= loc 0) (< loc L) (>= arg 0)))
       (assert (&& (safe-step? s op loc arg mode)         ; safety of this access
                   (inv (step s op loc arg mode)))))))    ; + preservation of Inv
  (cond
    [(unsat? sol)
     (printf "  ~a : INDUCTIVE INVARIANT HOLDS -> safe for programs of ANY length\n" (label mode))]
    [else
     (printf "  ~a : BREAKS -- counterexample (one step from a well-typed state):\n" (label mode))
     (printf "        pre : alive=~a len=~a vview=~a vlen=~a scap=~a slen=~a\n"
             (evaluate (st-alive s) sol) (evaluate (st-len s) sol) (evaluate (st-vview s) sol)
             (evaluate (st-vlen s) sol) (evaluate (st-scap s) sol) (evaluate (st-slen s) sol))
     (define o (evaluate op sol))
     (printf "        op  : ~a loc=~a arg=~a\n"
             (if (and (integer? o) (<= 0 o) (< o NOPS)) (vector-ref op-name o) o)
             (evaluate loc sol) (evaluate arg sol))]))

;; ===========================================================================
;; E2 -- ERASURE SOUNDNESS, as a non-interference theorem.
;;
;; The secondary claim `scap`/`slen` models the multiplicity-0 (erased) fragment.
;; Erasure soundness = the erased part is computationally IRRELEVANT: it cannot
;; influence runtime behaviour. We prove it relationally: take two states that
;; agree on every RUNTIME-relevant field (alive,len,vview,vlen) but may differ
;; arbitrarily on the erased scap/slen; run the SAME operation on both; then they
;; still agree on the relevant fields AND one is safe iff the other is. So the
;; erased data can be deleted without changing what the program does or whether
;; it is safe -- the formal justification for erasing the 0-fragment at runtime.
;;
;; Under SOUND this holds (no runtime op consults scap). Under BROKEN it FAILS
;; (staleread reads scap), i.e. allowing an erased value to drive a read makes
;; erasure unsound -- exactly the property the multiplicity discipline must keep.
;; ===========================================================================
(define (<=> a b) (&& (==> a b) (==> b a)))
(define (agree-relevant s1 s2)
  (all (append
        (for/list ([l (in-range L)]) (<=> (nth (st-alive s1) l) (nth (st-alive s2) l)))
        (for/list ([l (in-range L)]) (= (nth (st-len  s1) l) (nth (st-len  s2) l)))
        (for/list ([l (in-range L)]) (<=> (nth (st-vview s1) l) (nth (st-vview s2) l)))
        (for/list ([l (in-range L)]) (= (nth (st-vlen s1) l) (nth (st-vlen s2) l))))))

(define (verify-erasure mode)
  (define s1 (arb-state)) (define s2 (arb-state))
  (define-symbolic* op integer?) (define-symbolic* loc integer?) (define-symbolic* arg integer?)
  (define sol
    (verify
     (begin
       (assume (nats-nonneg s1)) (assume (nats-nonneg s2))
       (assume (agree-relevant s1 s2))         ; identical except the erased fields
       (assume (&& (>= op 0) (< op NOPS) (>= loc 0) (< loc L) (>= arg 0)))
       (assert (&& (agree-relevant (step s1 op loc arg mode) (step s2 op loc arg mode))
                   (<=> (safe-step? s1 op loc arg mode)
                        (safe-step? s2 op loc arg mode)))))))
  (printf "  ~a : ~a\n" (label mode)
          (if (unsat? sol)
              "ERASURE SOUND -> the multiplicity-0 fragment is computationally irrelevant"
              "erasure FAILS -> erased data influences runtime behaviour")))

;; ---- main ----------------------------------------------------------------
(printf "lambda-Tally bounded memory-safety check (Rosette/Z3)\n")
(printf "~a locations; length-indexed arrays (lengths/indices 0..~a).\n" L MAXLEN)
(printf "Probing: stale dependent-index claim across a linear strong update (resize)/free.\n\n")

(printf "=== METATHEOREM 1: inductive store-typing invariant (UNBOUNDED length) ===\n")
(printf "  base case: initial state satisfies Inv? ~a\n" (inv (init)))
(verify-inductive SOUND)
(verify-inductive BROKEN)
(printf "\n")

(printf "=== METATHEOREM 2 (E2): erasure soundness / non-interference ===\n")
(verify-erasure SOUND)
(verify-erasure BROKEN)
(printf "\n")

(printf "=== Bounded-exhaustive reachability (complementary, finite length) ===\n")
(printf "SOUND discipline (mirrors the Stage-C primitive types):\n")
(for ([k '(4 8 12 16)]) (run-experiment SOUND k))

(printf "\nBROKEN discipline (stale claim may authorise a read):\n")
(for ([k '(3 4)]) (run-experiment BROKEN k))

(printf "\nBROKEN with `free` disabled (forces the DEPENDENT out-of-bounds witness):\n")
(run-experiment BROKEN 4 #:forbid FREE)

(printf "\nReading the result: 'SOUND ... VERIFIED' means linearity provably prevents\n")
(printf "the stale-index hazard for every program up to that length -- including\n")
(printf "alloc; mkclaim; resize; staleread (strong update under a dependent reference).\n")
(printf "The BROKEN counterexample is a genuine OUT-OF-BOUNDS access (or use-after-\n")
(printf "free): a length claim outlived the mutation. The same `scap` models both an\n")
(printf "erased (multiplicity-0) dependent proof and a duplicated view (aliasing) --\n")
(printf "linearity, by keeping a single live view, rules out both.\n")
