#lang rosette
;; ===========================================================================
;; lambda-Tally, Stage D / E5: INTER-LOCATION dependency.
;;
;; The hardest case for a linear+dependent system (docs/05, docs/06): a dependent
;; type at one location that mentions ANOTHER location's value. Here a dependent
;; capability lets you read array `darr` at any index < the VALUE stored at a
;; different location `dsrc`. So the access bound on one location comes from
;; another location's contents -- "types depend on values" spanning two cells.
;;
;; The hazard: after forming that dependent capability, mutate `dsrc` to a larger
;; value (or `darr` to a smaller length); the dependent read bound grows past the
;; array -> out of bounds. This is the interaction most likely to be unsound.
;;
;; The lambda-Tally claim: forming a dependency on `dsrc` must CONSUME its linear
;; view (freeze it), so `dsrc` cannot be mutated while the dependency is live;
;; likewise `darr`. We test two disciplines:
;;   * SOUND  : mkdep freezes both src and arr (consumes their views).
;;   * BROKEN : mkdep does NOT freeze -- src/arr stay mutable.
;;
;; As in E1, we prove an INDUCTIVE invariant (unbounded length) for SOUND, and a
;; bounded-reachability search exhibits a concrete unsafe trace for BROKEN.
;;
;; State (L=2 locations): per location alive?, len (value / array length), a
;; plain linear view (vview) and its claimed length (vlen); plus a single
;; dependent capability (dcap) tying location `darr`'s read-bound to `dsrc`'s len.
;;
;; Run:  racket interloc-dependency-rosette.rkt
;; ===========================================================================

(define L 2)
;; opcodes
(define ALLOC 0) (define READ 1) (define WRITE 2) (define RESIZE 3)
(define FREE 4)  (define MKDEP 5) (define DEPREAD 6) (define ENDDEP 7)
(define NOPS 8)
(define SOUND 0) (define BROKEN 1)
(define op-name (vector "alloc" "read" "write" "resize" "free" "mkdep" "depread" "enddep"))

(struct st (alive len vview vlen dcap dsrc darr) #:transparent)
(define (nth  lst i)   (if (= i 0) (first lst) (second lst)))
(define (setn lst i v) (if (= i 0) (list v (second lst)) (list (first lst) v)))
(define (==> a b) (|| (! a) b))
(define (all lst) (foldl (lambda (x acc) (&& x acc)) #t lst))
(define (init) (st (list #f #f) (list 0 0) (list #f #f) (list 0 0) #f 0 0))

;; ---- SAFETY: plain reads and the dependent read must be in-bounds ---------
(define (safe-step? s op loc arg mode)
  (define alive (st-alive s)) (define len (st-len s))
  (define vview (st-vview s)) (define vlen (st-vlen s))
  (cond
    [(|| (= op READ) (= op WRITE))
     (==> (&& (nth vview loc) (nth alive loc) (< arg (nth vlen loc)))
          (&& (nth alive loc) (< arg (nth len loc))))]
    [(= op DEPREAD)                       ; read darr at index arg, bound by len[dsrc]
     (define d (st-dsrc s)) (define a (st-darr s))
     (==> (&& (st-dcap s) (< arg (nth len d)))
          (&& (nth alive a) (< arg (nth len a))))]
    [else #t]))

;; ---- one pure interpreter step -------------------------------------------
(define (step s op loc arg mode)
  (define alive (st-alive s)) (define len (st-len s))
  (define vview (st-vview s)) (define vlen (st-vlen s))
  (define dcap (st-dcap s)) (define dsrc (st-dsrc s)) (define darr (st-darr s))
  (cond
    [(= op ALLOC)
     (if (! (nth alive loc))
         (st (setn alive loc #t) (setn len loc arg)
             (setn vview loc #t) (setn vlen loc arg) dcap dsrc darr)
         s)]
    [(= op RESIZE)                         ; strong update of a value/length
     (if (&& (nth vview loc) (nth alive loc))
         (st alive (setn len loc arg) vview (setn vlen loc arg) dcap dsrc darr)
         s)]
    [(= op FREE)
     (if (&& (nth vview loc) (nth alive loc))
         (st (setn alive loc #f) (setn len loc 0) (setn vview loc #f) vlen dcap dsrc darr)
         s)]
    [(= op MKDEP)                          ; loc = src, arg = arr (a location index)
     (define src loc) (define arr arg)
     (define ok (&& (< arr L) (! (= src arr)) (! dcap)
                    (nth vview src) (nth alive src)
                    (nth vview arr) (nth alive arr)
                    (<= (nth len src) (nth len arr))))   ; the bound must fit the array
     (if ok
         (if (= mode SOUND)
             (st alive len (setn (setn vview src #f) arr #f) vlen #t src arr)  ; FREEZE both
             (st alive len vview vlen #t src arr))                            ; BROKEN: no freeze
         s)]
    [(= op ENDDEP)                         ; end the borrow: give the views back
     (if dcap
         (st alive len (setn (setn vview dsrc #t) darr #t) vlen #f dsrc darr)
         s)]
    [else s]))                             ; read / write / depread: no state change

;; ---- run a symbolic program (asserting safety each step) -----------------
(define (run ops locs args mode)
  (for/fold ([s (init)]) ([o ops] [l locs] [a args])
    (assert (safe-step? s o l a mode))
    (step s o l a mode)))

;; ===========================================================================
;; The INDUCTIVE invariant (unbounded length).
;;   plain : vview[l] => alive[l] AND vlen[l] = len[l]
;;   dep   : dcap => src,arr in range & distinct & both alive & len[src] <= len[arr]
;;                   & NEITHER has a live view (so neither can be mutated)
;;                   & vlen matches len for both (so ENDDEP can restore soundly)
;; ===========================================================================
(define (inv s)
  (define alive (st-alive s)) (define len (st-len s))
  (define vview (st-vview s)) (define vlen (st-vlen s))
  (define d (st-dsrc s)) (define a (st-darr s))
  (&& (all (for/list ([l (in-range L)])
             (==> (nth vview l) (&& (nth alive l) (= (nth vlen l) (nth len l))))))
      (==> (st-dcap s)
           (&& (>= d 0) (< d L) (>= a 0) (< a L) (! (= d a))
               (nth alive d) (nth alive a)
               (<= (nth len d) (nth len a))
               (! (nth vview d)) (! (nth vview a))
               (= (nth vlen d) (nth len d)) (= (nth vlen a) (nth len a))))))

(define (sb) (define-symbolic* b boolean?) b)
(define (sn) (define-symbolic* n integer?) n)
(define (arb-state) (st (list (sb) (sb)) (list (sn) (sn)) (list (sb) (sb)) (list (sn) (sn))
                        (sb) (sn) (sn)))
(define (nats s) (all (append (map (lambda (n) (>= n 0)) (st-len s))
                              (map (lambda (n) (>= n 0)) (st-vlen s))
                              (list (>= (st-dsrc s) 0) (>= (st-darr s) 0)))))

(define (verify-inductive mode)
  (define s (arb-state))
  (define-symbolic* op integer?) (define-symbolic* loc integer?) (define-symbolic* arg integer?)
  (define sol
    (verify
     (begin
       (assume (nats s)) (assume (inv s))
       (assume (&& (>= op 0) (< op NOPS) (>= loc 0) (< loc L) (>= arg 0)))
       (assert (&& (safe-step? s op loc arg mode) (inv (step s op loc arg mode)))))))
  (cond
    [(unsat? sol)
     (printf "  ~a : INDUCTIVE INVARIANT HOLDS -> inter-location dependency safe at ANY length\n"
             (if (= mode SOUND) "SOUND " "BROKEN"))]
    [else
     (printf "  ~a : BREAKS -- one step from a well-typed state:\n" (if (= mode SOUND) "SOUND " "BROKEN"))
     (define o (evaluate op sol))
     (printf "        pre alive=~a len=~a vview=~a vlen=~a dcap=~a dsrc=~a darr=~a | op=~a loc=~a arg=~a\n"
             (evaluate (st-alive s) sol) (evaluate (st-len s) sol) (evaluate (st-vview s) sol)
             (evaluate (st-vlen s) sol) (evaluate (st-dcap s) sol) (evaluate (st-dsrc s) sol)
             (evaluate (st-darr s) sol)
             (if (and (integer? o) (<= 0 o) (< o NOPS)) (vector-ref op-name o) o)
             (evaluate loc sol) (evaluate arg sol))]))

;; ---- bounded reachability: exhibit a concrete BROKEN trace ---------------
(define (reach mode k)
  (define ops  (for/list ([i (in-range k)]) (define-symbolic* o integer?) o))
  (define locs (for/list ([i (in-range k)]) (define-symbolic* l integer?) l))
  (define args (for/list ([i (in-range k)]) (define-symbolic* a integer?) a))
  (define sol
    (verify (begin
              (for ([o ops] [l locs] [a args])
                (assume (&& (>= o 0) (< o NOPS) (>= l 0) (< l L) (>= a 0) (<= a 4))))
              (run ops locs args mode))))
  (cond
    [(unsat? sol) (printf "  ~a, k=~a : no unsafe trace\n" (if (= mode SOUND) "SOUND " "BROKEN") k)]
    [else
     (printf "  ~a, k=~a : UNSAFE TRACE\n" (if (= mode SOUND) "SOUND " "BROKEN") k)
     (for ([o (evaluate ops sol)] [l (evaluate locs sol)] [a (evaluate args sol)] [i (in-naturals)])
       (when (and (integer? o) (<= 0 o) (< o NOPS))
         (define lab (cond [(= o MKDEP) (format "src=~a arr=~a" l a)]
                           [(= o DEPREAD) (format "idx=~a" a)]
                           [(or (= o ALLOC) (= o RESIZE)) (format "loc=~a len=~a" l a)]
                           [else (format "loc=~a" l)]))
         (printf "        ~a) ~a  ~a\n" i (vector-ref op-name o) lab)))]))

;; ---- main ----------------------------------------------------------------
(printf "lambda-Tally E5: inter-location dependency (a type at one cell uses another's value)\n\n")
(printf "=== METATHEOREM: inductive invariant (UNBOUNDED length) ===\n")
(printf "  base case: init satisfies Inv? ~a\n" (inv (init)))
(verify-inductive SOUND)
(verify-inductive BROKEN)
(printf "\n=== Bounded reachability: a concrete unsafe BROKEN trace ===\n")
(reach BROKEN 5)
(printf "\nReading it: under SOUND, forming the dependency freezes BOTH cells (consumes their\n")
(printf "linear views), so neither can be resized or freed while a dependent read relies on\n")
(printf "them -- the inductive invariant holds for any length. Under BROKEN neither is frozen,\n")
(printf "so mutating the source (resize larger) or the array (resize smaller / free) after\n")
(printf "mkdep makes the dependent read run off the end or hit freed memory.\n")
