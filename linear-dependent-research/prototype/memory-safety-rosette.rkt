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

;; An element access through some capability:
;;   `have?`  : is the capability present (and, for a view, the loc live)?
;;   `claimed`: the length the capability asserts -> used for the STATIC bounds
;;              check (arg < claimed), our stand-in for a dependent `i < n` proof.
;; SAFETY OBLIGATION: the access is actually in-bounds of live memory
;;   (alive AND arg < the REAL current length). If the capability's claim is
;;   honest (claimed = len), this always holds; a stale claim breaks it.
(define (access s loc arg have? claimed)
  (when (&& have? (< arg claimed))
    (assert (&& (nth (st-alive s) loc) (< arg (nth (st-len s) loc)))))
  s)

;; ---- one interpreter step ------------------------------------------------
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
    ;; read / write element `arg`: need a live view; bounds-check against vlen.
    [(= op READ)  (access s loc arg (&& (nth vview loc) (nth alive loc)) (nth vlen loc))]
    [(= op WRITE) (access s loc arg (&& (nth vview loc) (nth alive loc)) (nth vlen loc))]
    ;; resize = STRONG UPDATE of the index: consume the view, reissue at the new
    ;; length `arg`. Heap length AND the view's claim move together; any
    ;; secondary claim is intentionally left STALE.
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
    ;; staleread: read element `arg` via the SECONDARY claim.
    [(= op STALEREAD)
     (if (= mode SOUND)
         ;; SOUND: a read still needs the live linear view; the claim adds nothing.
         (access s loc arg (&& (nth vview loc) (nth alive loc)) (nth vlen loc))
         ;; BROKEN: the stale claim alone authorises the read (no view, no
         ;; liveness), bounds-checking against its possibly-stale length `slen`.
         (access s loc arg (nth scap loc) (nth slen loc)))]
    [else s]))

;; ---- run a whole symbolic program ---------------------------------------
(define (run ops locs args mode)
  (for/fold ([s (init)]) ([o ops] [l locs] [a args])
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

;; ---- main ----------------------------------------------------------------
(printf "lambda-Tally bounded memory-safety check (Rosette/Z3)\n")
(printf "~a locations; length-indexed arrays (lengths/indices 0..~a).\n" L MAXLEN)
(printf "Probing: stale dependent-index claim across a linear strong update (resize)/free.\n\n")

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
