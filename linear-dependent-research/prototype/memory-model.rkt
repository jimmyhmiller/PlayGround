#lang racket/base
;; ===========================================================================
;; lambda-Tally, Stage B: an executable PLT Redex model of the MEMORY layer.
;;
;; This is the "get a handle on it" artifact for docs/02-memory-views.md. Where
;; Stage A (qtt_checker.py) is about the *static* multiplicity discipline, this
;; is about the *operational* memory-safety payoff: a heap + the four primitives
;;     new / read / write / free
;; with capabilities ("views") threaded LINEARLY, in the style of L3
;; (Ahmed/Fluet/Morrisett) and ATS views.
;;
;; The headline it demonstrates:
;;   * a well-formed program allocates, uses, and frees -> reduces to a value
;;     with an EMPTY heap (no leak);
;;   * a use-after-free configuration is operationally STUCK (the deref of a
;;     freed location has no reduction) -- i.e. the hazard is real;
;;   * the LINEAR type system rejects exactly the programs that could reach
;;     such a stuck state: you cannot still hold (cap l) after (free _ (cap l)),
;;     because the capability is linear. Double-free and use-after-free become
;;     untypeable, not merely unchecked.
;;
;; Pointers are UNRESTRICTED (type Ptr, used any number of times); capabilities
;; are LINEAR (type Cap, used exactly once). That split is the whole trick.
;;
;; Scope note: to stay small and runnable we use an OPAQUE Cap (no location in
;; the type) and Unit payloads. That is enough for use-after-free / double-free
;; / leak safety, which follow from linearity alone. Location-indexed caps
;; (Cap l T) for STRONG UPDATE / wrong-pointer safety are the documented next
;; step (docs/02 S3, docs/04 Phase 2). Unifying this linear discipline with the
;; dependent core of Stage A is also future work (docs/04 Phase 4).
;;
;; Requires only the non-GUI Redex core. Run:  racket memory-model.rkt
;; ===========================================================================
(require redex/reduction-semantics)

;; ---------------------------------------------------------------------------
;; 1. Syntax
;; ---------------------------------------------------------------------------
(define-language L3
  (e ::= x
         unit
         (lam (x T) e)
         (e e)
         (pair e e)
         (let-pair (x x) e e)
         (new e)            ; allocate, init -> (pair Ptr Cap)
         (read e e)         ; (read ptr cap)     -> (pair Unit Cap)
         (write e e e)      ; (write ptr cap v)  -> Cap            (in-place update)
         (free e e)         ; (free ptr cap)     -> Unit           (reclaim)
         (loc l)            ; runtime pointer value
         (cap l))           ; runtime capability token (linear)
  (v ::= unit (lam (x T) e) (loc l) (cap l) (pair v v))
  (T ::= Unit (-> T T) Ptr Cap (Prod T T))
  ;; evaluation contexts: left-to-right, call-by-value
  (E ::= hole
         (E e) (v E)
         (pair E e) (pair v E)
         (let-pair (x x) E e)
         (new E)
         (read E e) (read v E)
         (write E e e) (write v E e) (write v v E)
         (free E e) (free v E))
  (H ::= ((l v) ...))       ; heap: location -> value
  (Gamma ::= ((x T) ...))   ; typing context (used for both linear & unrestricted)
  (x l ::= variable-not-otherwise-mentioned)
  #:binding-forms
  (lam (x T) e #:refers-to x)
  (let-pair (x_1 x_2) e_1 e_2 #:refers-to (shadow x_1 x_2)))

;; ---------------------------------------------------------------------------
;; 2. Heap metafunctions
;; ---------------------------------------------------------------------------
(define-metafunction L3
  extend : H l v -> H
  [(extend ((l_1 v_1) ...) l v) ((l v) (l_1 v_1) ...)])

(define-metafunction L3
  lookup : H l -> v
  [(lookup ((l_1 v_1) ... (l v) (l_2 v_2) ...) l) v])

(define-metafunction L3
  store : H l v -> H
  [(store ((l_1 v_1) ... (l v_0) (l_2 v_2) ...) l v)
   ((l_1 v_1) ... (l v) (l_2 v_2) ...)])

(define-metafunction L3
  delete : H l -> H
  [(delete ((l_1 v_1) ... (l v) (l_2 v_2) ...) l)
   ((l_1 v_1) ... (l_2 v_2) ...)])

(define-metafunction L3
  in-dom? : H l -> boolean
  [(in-dom? ((l_1 v_1) ... (l v) (l_2 v_2) ...) l) #t]
  [(in-dom? H l) #f])

;; ---------------------------------------------------------------------------
;; 3. Operational semantics:  (H e) --> (H e)
;;    Note: read/write/free only fire when the location is LIVE (in-dom?).
;;    A capability for a freed location therefore gets STUCK -- which is the
;;    whole point: only the type system's linearity stops you ever holding one.
;; ---------------------------------------------------------------------------
(define red
  (reduction-relation
   L3
   #:domain (H e)
   (--> (H (in-hole E ((lam (x T) e) v)))
        (H (in-hole E (substitute e x v)))
        "beta")
   (--> (H (in-hole E (let-pair (x_1 x_2) (pair v_1 v_2) e)))
        (H (in-hole E (substitute (substitute e x_1 v_1) x_2 v_2)))
        "let-pair")
   (--> (H (in-hole E (new v)))
        ((extend H l v) (in-hole E (pair (loc l) (cap l))))
        (where l ,(variable-not-in (term H) (term a)))
        "new")
   (--> (H (in-hole E (read (loc l) (cap l))))
        (H (in-hole E (pair (lookup H l) (cap l))))
        (side-condition (term (in-dom? H l)))
        "read")
   (--> (H (in-hole E (write (loc l) (cap l) v)))
        ((store H l v) (in-hole E (cap l)))
        (side-condition (term (in-dom? H l)))
        "write")
   (--> (H (in-hole E (free (loc l) (cap l))))
        ((delete H l) (in-hole E unit))
        (side-condition (term (in-dom? H l)))
        "free")))

;; ---------------------------------------------------------------------------
;; 4. The LINEAR type system (leftover-context formulation).
;;    Judgment:  (types Gamma_unrestricted Gamma_linear e T Gamma_linear_out)
;;    Threading the linear context as input AND output makes it algorithmic
;;    (no nondeterministic context splitting). A linear var is REMOVED when
;;    used; using it twice fails (it is gone); never using it leaves it in the
;;    output (a leak), which the top-level check forbids.
;;    Pointers/Units/functions are unrestricted (live in Gamma_u); Caps are
;;    linear (live in Gamma_l), routed by `linear-type?`.
;; ---------------------------------------------------------------------------
(define-metafunction L3
  linear-type? : T -> boolean
  [(linear-type? Cap) #t]
  [(linear-type? (Prod T_1 T_2)) ,(or (term (linear-type? T_1)) (term (linear-type? T_2)))]
  [(linear-type? T) #f])

;; add (x : T) to the unrestricted ctx iff T is NOT linear
(define-metafunction L3
  ext-u : Gamma x T -> Gamma
  [(ext-u Gamma x T) Gamma                       (side-condition (term (linear-type? T)))]
  [(ext-u ((x_1 T_1) ...) x T) ((x T) (x_1 T_1) ...)])

;; add (x : T) to the linear ctx iff T IS linear
(define-metafunction L3
  ext-l : Gamma x T -> Gamma
  [(ext-l ((x_1 T_1) ...) x T) ((x T) (x_1 T_1) ...) (side-condition (term (linear-type? T)))]
  [(ext-l Gamma x T) Gamma])

(define-metafunction L3
  mem? : Gamma x -> boolean
  [(mem? ((x_1 T_1) ... (x T) (x_2 T_2) ...) x) #t]
  [(mem? Gamma x) #f])

(define-judgment-form L3
  #:mode (types I I I O O)
  #:contract (types Gamma Gamma e T Gamma)

  ;; unrestricted variable: context unchanged
  [(where (any_1 ... (x T) any_2 ...) Gamma_u)
   ------------------------------------------------ "var-u"
   (types Gamma_u Gamma_l x T Gamma_l)]

  ;; linear variable: consume it (remove from the linear context)
  [------------------------------------------------ "var-l"
   (types Gamma_u (any_1 ... (x T) any_2 ...) x T (any_1 ... any_2 ...))]

  [------------------------------------------------ "unit"
   (types Gamma_u Gamma_l unit Unit Gamma_l)]

  [(where Gamma_u2 (ext-u Gamma_u x T_1))
   (where Gamma_l2 (ext-l Gamma_l x T_1))
   (types Gamma_u2 Gamma_l2 e T_2 Gamma_l)        ; body returns leftover to original (arg consumed)
   ------------------------------------------------ "lam"
   (types Gamma_u Gamma_l (lam (x T_1) e) (-> T_1 T_2) Gamma_l)]

  [(types Gamma_u Gamma_l e_1 (-> T_1 T_2) Gamma_l1)
   (types Gamma_u Gamma_l1 e_2 T_1 Gamma_l2)
   ------------------------------------------------ "app"
   (types Gamma_u Gamma_l (e_1 e_2) T_2 Gamma_l2)]

  [(types Gamma_u Gamma_l e_1 T_1 Gamma_l1)
   (types Gamma_u Gamma_l1 e_2 T_2 Gamma_l2)
   ------------------------------------------------ "pair"
   (types Gamma_u Gamma_l (pair e_1 e_2) (Prod T_1 T_2) Gamma_l2)]

  [(types Gamma_u Gamma_l e_1 (Prod T_1 T_2) Gamma_l1)
   (where Gamma_u2 (ext-u (ext-u Gamma_u x_1 T_1) x_2 T_2))
   (where Gamma_l2 (ext-l (ext-l Gamma_l1 x_1 T_1) x_2 T_2))
   (types Gamma_u2 Gamma_l2 e_2 T Gamma_l3)
   (side-condition (not (term (mem? Gamma_l3 x_1))))   ; linear components must be consumed
   (side-condition (not (term (mem? Gamma_l3 x_2))))
   ------------------------------------------------ "let-pair"
   (types Gamma_u Gamma_l (let-pair (x_1 x_2) e_1 e_2) T Gamma_l3)]

  [(types Gamma_u Gamma_l e Unit Gamma_l1)
   ------------------------------------------------ "new"
   (types Gamma_u Gamma_l (new e) (Prod Ptr Cap) Gamma_l1)]

  [(types Gamma_u Gamma_l e_p Ptr Gamma_l1)
   (types Gamma_u Gamma_l1 e_c Cap Gamma_l2)
   ------------------------------------------------ "read"
   (types Gamma_u Gamma_l (read e_p e_c) (Prod Unit Cap) Gamma_l2)]

  [(types Gamma_u Gamma_l e_p Ptr Gamma_l1)
   (types Gamma_u Gamma_l1 e_c Cap Gamma_l2)
   (types Gamma_u Gamma_l2 e_v Unit Gamma_l3)
   ------------------------------------------------ "write"
   (types Gamma_u Gamma_l (write e_p e_c e_v) Cap Gamma_l3)]

  [(types Gamma_u Gamma_l e_p Ptr Gamma_l1)
   (types Gamma_u Gamma_l1 e_c Cap Gamma_l2)
   ------------------------------------------------ "free"
   (types Gamma_u Gamma_l (free e_p e_c) Unit Gamma_l2)])

;; A closed source program is well-typed iff it checks with empty contexts AND
;; leaves NO unused linear resource (empty linear leftover = no leak).
(define (well-typed? e)
  (not (null? (judgment-holds (types () () ,e T ()) T))))

;; ---------------------------------------------------------------------------
;; 5. Helpers for the safety demonstration
;; ---------------------------------------------------------------------------
(define (normal-forms cfg) (apply-reduction-relation* red cfg))

(define (value? t)                       ; is t a (H v) configuration?
  (and (pair? t) (redex-match? L3 v (cadr t))))

;; "safe" = every terminal configuration reached is a value (never stuck).
(define (safe-run? e)
  (andmap value? (normal-forms (term (() ,e)))))

(define (stuck? cfg)                      ; a normal form that is NOT a value
  (and (null? (apply-reduction-relation red cfg)) (not (value? cfg))))

;; ---------------------------------------------------------------------------
;; 6. Example programs
;; ---------------------------------------------------------------------------
;; GOOD: allocate, read, write, then free. Threads the linear cap the whole way.
(define good
  (term (let-pair (p c) (new unit)
          (let-pair (u c2) (read p c)
            (let-pair (c3 dummy) (pair (write p c2 unit) unit)   ; write returns a cap
              (free p c3))))))

;; GOOD-simple: allocate then free.
(define good-simple
  (term (let-pair (p c) (new unit) (free p c))))

;; BAD-leak: allocate but never free -> linear cap `c` left unconsumed.
(define bad-leak
  (term (let-pair (p c) (new unit) unit)))

;; BAD-double-free: free twice -> needs the cap twice -> linearity violation.
(define bad-double-free
  (term (let-pair (p c) (new unit)
          (let-pair (z1 z2) (pair (free p c) (free p c)) unit))))

;; BAD-use-after-free: read using the SAME cap that free consumed -> uses c twice.
(define bad-uaf
  (term (let-pair (p c) (new unit)
          (let-pair (z1 z2) (pair (free p c) (read p c)) unit))))

;; A hand-built RUNTIME configuration showing the operational hazard directly:
;; the location has been freed (heap empty) but we still hold its capability.
;; No reduction rule applies -> STUCK. The type system's job is to make this
;; configuration unreachable.
(define dangling (term (() (read (loc l) (cap l)))))

;; ---------------------------------------------------------------------------
;; 7. Tests
;; ---------------------------------------------------------------------------
(module+ main
  (printf "=== operational semantics ===~n")
  (printf "good-simple reduces to (empty-heap, unit)? ~a~n"
          (equal? (normal-forms (term (() ,good-simple))) (list (term (() unit)))))
  (printf "good        reduces to (empty-heap, unit)? ~a~n"
          (equal? (normal-forms (term (() ,good))) (list (term (() unit)))))
  (printf "dangling (use-after-free) config is STUCK? ~a~n" (stuck? dangling))
  (newline)

  (printf "=== linear type system (accept good, reject unsafe) ===~n")
  (for ([nm '(good good-simple)] [e (list good good-simple)])
    (printf "  [accept] ~a well-typed? ~a~n" nm (well-typed? e)))
  (for ([nm '(bad-leak bad-double-free bad-use-after-free)]
        [e (list bad-leak bad-double-free bad-uaf)])
    (printf "  [reject] ~a well-typed? ~a~n" nm (well-typed? e)))
  (newline)

  ;; Redex regression tests (test-->> checks full reduction to the given term).
  (test-->> red (term (() ,good-simple)) (term (() unit)))
  (test-->> red (term (() ,good))        (term (() unit)))
  (test-equal (well-typed? good)            #t)
  (test-equal (well-typed? good-simple)     #t)
  (test-equal (well-typed? bad-leak)        #f)
  (test-equal (well-typed? bad-double-free) #f)
  (test-equal (well-typed? bad-uaf)         #f)
  (test-equal (stuck? dangling)             #t)
  ;; The connection: well-typed source programs run without getting stuck.
  (test-equal (safe-run? good)        #t)
  (test-equal (safe-run? good-simple) #t)

  ;; --- redex-check: the "Run Your Research" methodology -------------------
  ;; Property (true): a value configuration is a normal form (cannot step).
  ;; redex-check fuzzes random (H v) and tries to break it. This illustrates
  ;; how you'd hunt counterexamples to a *metatheorem* automatically; scaling
  ;; this to "well-typed ==> safe" needs a typed-term generator (docs/04).
  (redex-check L3 (H v)
               (null? (apply-reduction-relation red (term (H v))))
               #:attempts 500)

  (test-results))
