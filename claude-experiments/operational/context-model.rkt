#lang rosette

;; =============================================================================
;; CONTEXT-AWARE JAVASCRIPT SUBSET MODEL
;; =============================================================================
;;
;; This models a JS-like language where:
;; - Functions have metadata about which context they belong to
;; - A context stack tracks the current execution context
;; - Values are coerced when crossing context boundaries
;; - We use Rosette to find counterexamples to security properties
;;
;; EXTENSION POINTS (search for "EXTENSION POINT" to find them):
;; 1. should-push-context?          - decides when a call pushes a new context
;; 2. make-function-metadata        - assigns metadata to functions
;; 3. make-object-metadata          - assigns metadata to objects
;; 4. coerce-value                  - transforms values crossing boundaries
;; 5. can-cross-boundary?           - decides if a value can cross at all
;; 6. should-coerce-property-access? - decides if property access crosses boundary
;; 7. coerce-for-property-access    - coerces value when reading property

(require racket/match)

;; =============================================================================
;; PART 1: CORE DATA STRUCTURES
;; =============================================================================

;; A Context represents an execution environment with security properties.
;; - id: unique identifier for this context
;; - level: numeric security level (higher = more privileged)
;; - capabilities: list of symbols representing what this context can do
(struct context (id level capabilities) #:transparent)

;; Metadata attached to functions (and later, objects).
;; - owner-context: the context this function "belongs to"
;; - tags: arbitrary key-value pairs for your policies
(struct fn-metadata (owner-context tags) #:transparent)

;; A runtime value that tracks where it came from.
;; - data: the actual JS value (number, string, function, etc.)
;; - origin: the context where this value was created
;; - tainted: whether this value has been through untrusted code
(struct tracked-val (data origin tainted) #:transparent)

;; A function in our model.
;; - params: list of parameter names
;; - body: the function body (an expression)
;; - closure-env: captured environment
;; - metadata: fn-metadata struct
(struct fn-val (params body closure-env metadata) #:transparent)

;; Metadata attached to objects.
;; - owner-context: the context this object "belongs to"
;; - tags: arbitrary key-value pairs for your policies
;; - sealed: if #t, properties cannot be added/modified from other contexts
(struct obj-metadata (owner-context tags sealed) #:transparent)

;; An object in our model (like a JS object).
;; - properties: hash table mapping property names to tracked-vals
;; - prototype: another obj-val or #f (for prototype chain)
;; - metadata: obj-metadata struct
;;
;; NOTE: We use a mutable hash for properties to model JS mutation semantics.
;; For symbolic execution, you may want an immutable version.
(struct obj-val (properties prototype metadata) #:transparent)

;; Helper to create a new empty object
(define (make-empty-object ctx annotations)
  (obj-val (make-hash)
           #f
           (make-object-metadata ctx annotations)))

;; The context stack - tracks where we are in execution.
;; Just a list of contexts, head is current.
(define current-context-stack (make-parameter '()))

;; Helper to get current context
(define (current-context)
  (if (null? (current-context-stack))
      (error 'current-context "Empty context stack!")
      (car (current-context-stack))))

;; Helper to push a context
(define (push-context ctx thunk)
  (parameterize ([current-context-stack (cons ctx (current-context-stack))])
    (thunk)))

;; =============================================================================
;; PART 2: EXTENSION POINTS - CUSTOMIZE THESE FOR YOUR SYSTEM
;; =============================================================================

;; -----------------------------------------------------------------------------
;; EXTENSION POINT 1: When do we push a new context?
;; -----------------------------------------------------------------------------
;; Given: the function being called, its metadata, arguments, current context
;; Returns: #f to stay in current context, or a context to push
;;
;; DEFAULT: Always push to the function's owner context
(define (should-push-context? fn args current-ctx)
  (fn-metadata-owner-context (fn-val-metadata fn)))

;; EXAMPLE ALTERNATIVE: Only push if crossing security levels
#;(define (should-push-context? fn args current-ctx)
    (let ([fn-ctx (fn-metadata-owner-context (fn-val-metadata fn))])
      (if (= (context-level fn-ctx) (context-level current-ctx))
          #f  ; Same level, don't push
          fn-ctx)))

;; -----------------------------------------------------------------------------
;; EXTENSION POINT 2: How do we assign metadata to functions?
;; -----------------------------------------------------------------------------
;; Given: the context where the function is defined, any annotations
;; Returns: fn-metadata struct
;;
;; DEFAULT: Function belongs to context where it was defined
(define (make-function-metadata defining-context annotations)
  (fn-metadata defining-context annotations))

;; EXAMPLE ALTERNATIVE: Inherit from annotations if present
#;(define (make-function-metadata defining-context annotations)
    (let ([explicit-ctx (assoc 'context annotations)])
      (fn-metadata (if explicit-ctx (cdr explicit-ctx) defining-context)
                   annotations)))

;; -----------------------------------------------------------------------------
;; EXTENSION POINT 3: How do we assign metadata to objects?
;; -----------------------------------------------------------------------------
;; Given: the context where the object is created, any annotations
;; Returns: obj-metadata struct
;;
;; DEFAULT: Object belongs to context where it was created, not sealed
(define (make-object-metadata defining-context annotations)
  (let ([sealed? (assoc 'sealed annotations)])
    (obj-metadata defining-context
                  annotations
                  (if sealed? (cdr sealed?) #f))))

;; EXAMPLE ALTERNATIVE: Objects from high contexts are always sealed
#;(define (make-object-metadata defining-context annotations)
    (obj-metadata defining-context
                  annotations
                  (> (context-level defining-context) 5)))  ; Sealed if high-privilege

;; -----------------------------------------------------------------------------
;; EXTENSION POINT 4: How do we coerce values crossing boundaries?
;; -----------------------------------------------------------------------------
;; Given: the value, source context, target context
;; Returns: the coerced value (possibly modified/wrapped/sanitized)
;;
;; This is where your "security barrier" logic lives!
;;
;; DEFAULT: Mark as tainted when going from high to low
(define (coerce-value val from-ctx to-ctx)
  (cond
    ;; Same context, no coercion needed
    [(equal? (context-id from-ctx) (context-id to-ctx))
     val]

    ;; Going from high privilege to low privilege - taint it
    [(> (context-level from-ctx) (context-level to-ctx))
     (tracked-val (tracked-val-data val)
                  (tracked-val-origin val)
                  #t)]  ; Mark tainted

    ;; Going from low to high - keep as is (but track origin)
    [else val]))

;; EXAMPLE ALTERNATIVE: Wrap in a proxy-like structure
#;(define (coerce-value val from-ctx to-ctx)
    (if (equal? (context-id from-ctx) (context-id to-ctx))
        val
        (make-membrane-wrapped val from-ctx to-ctx)))

;; EXAMPLE ALTERNATIVE: Deep copy and sanitize
#;(define (coerce-value val from-ctx to-ctx)
    (tracked-val (deep-sanitize (tracked-val-data val) to-ctx)
                 to-ctx  ; Change origin to target
                 #f))

;; -----------------------------------------------------------------------------
;; EXTENSION POINT 4: Can a value cross this boundary at all?
;; -----------------------------------------------------------------------------
;; Given: the value, source context, target context
;; Returns: #t if allowed, #f if blocked
;;
;; DEFAULT: Always allow (rely on coercion to sanitize)
(define (can-cross-boundary? val from-ctx to-ctx)
  #t)

;; EXAMPLE ALTERNATIVE: Block functions from crossing
#;(define (can-cross-boundary? val from-ctx to-ctx)
    (not (fn-val? (tracked-val-data val))))

;; EXAMPLE ALTERNATIVE: Check capabilities
#;(define (can-cross-boundary? val from-ctx to-ctx)
    (let ([required-cap (get-required-capability val)])
      (or (not required-cap)
          (member required-cap (context-capabilities to-ctx)))))

;; -----------------------------------------------------------------------------
;; EXTENSION POINT 6: When does property access trigger coercion?
;; -----------------------------------------------------------------------------
;; Given: the object, property name, current context
;; Returns: #f if no coercion needed, or the object's context if coercion required
;;
;; This is called when reading obj.prop - do we need to coerce the result?
;;
;; DEFAULT: Coerce if object belongs to a different context
(define (should-coerce-property-access? obj prop-name accessor-ctx)
  (let ([obj-ctx (obj-metadata-owner-context (obj-val-metadata obj))])
    (if (equal? (context-id obj-ctx) (context-id accessor-ctx))
        #f
        obj-ctx)))

;; EXAMPLE ALTERNATIVE: Only coerce for specific "sensitive" properties
#;(define (should-coerce-property-access? obj prop-name accessor-ctx)
    (let ([obj-ctx (obj-metadata-owner-context (obj-val-metadata obj))]
          [sensitive-props '(password token secret)])
      (if (and (member prop-name sensitive-props)
               (not (equal? (context-id obj-ctx) (context-id accessor-ctx))))
          obj-ctx
          #f)))

;; -----------------------------------------------------------------------------
;; EXTENSION POINT 7: How do we coerce when reading a property?
;; -----------------------------------------------------------------------------
;; Given: the value, object's context, accessor's context, property name
;; Returns: the coerced value
;;
;; This may differ from regular coerce-value because property access
;; might have different semantics than function call boundaries.
;;
;; DEFAULT: Use the same coercion as function boundaries
(define (coerce-for-property-access val obj-ctx accessor-ctx prop-name)
  (coerce-value val obj-ctx accessor-ctx))

;; EXAMPLE ALTERNATIVE: Return a "view" wrapper instead of tainting
#;(define (coerce-for-property-access val obj-ctx accessor-ctx prop-name)
    (tracked-val (list 'property-view prop-name (tracked-val-data val))
                 accessor-ctx
                 #t))

;; -----------------------------------------------------------------------------
;; EXTENSION POINT 8: Can we write to a property from this context?
;; -----------------------------------------------------------------------------
;; Given: the object, property name, new value, current context
;; Returns: #t if allowed, #f if blocked
;;
;; DEFAULT: Block if object is sealed and we're in a different context
(define (can-write-property? obj prop-name val accessor-ctx)
  (let* ([meta (obj-val-metadata obj)]
         [obj-ctx (obj-metadata-owner-context meta)]
         [same-ctx? (equal? (context-id obj-ctx) (context-id accessor-ctx))])
    (or same-ctx?
        (not (obj-metadata-sealed meta)))))

;; EXAMPLE ALTERNATIVE: Only owner context can write
#;(define (can-write-property? obj prop-name val accessor-ctx)
    (let ([obj-ctx (obj-metadata-owner-context (obj-val-metadata obj))])
      (equal? (context-id obj-ctx) (context-id accessor-ctx))))

;; =============================================================================
;; PART 3: THE BOUNDARY CROSSING LOGIC
;; =============================================================================

;; This is called whenever a value crosses a context boundary.
;; It checks if crossing is allowed, then applies coercion.
(define (cross-boundary val from-ctx to-ctx direction)
  ;; direction is 'arg (going into function) or 'return (coming back)

  (when (not (can-cross-boundary? val from-ctx to-ctx))
    (error 'cross-boundary
           "Value ~a cannot cross from ~a to ~a (~a)"
           val from-ctx to-ctx direction))

  (coerce-value val from-ctx to-ctx))

;; =============================================================================
;; PART 4: THE INTERPRETER
;; =============================================================================

;; Expressions in our language:
;; - (const v)           : literal value
;; - (var x)             : variable reference
;; - (fn (params...) body annotations) : function definition
;; - (call f args...)    : function call
;; - (if test then else) : conditional
;; - (binop op e1 e2)    : binary operation
;; - (new ((prop expr) ...) annotations) : object creation
;; - (get obj-expr prop-name)    : property read
;; - (set obj-expr prop-name val-expr) : property write
;; - (method-call obj-expr method-name args...) : call method with obj as `this`

;; Evaluate an expression in an environment
(define (eval-expr expr env)
  (match expr
    ;; Literal constant - wrap in tracked-val with current context
    [`(const ,v)
     (tracked-val v (current-context) #f)]

    ;; Variable lookup
    [`(var ,x)
     (let ([binding (assoc x env)])
       (if binding
           (cdr binding)
           (error 'eval-expr "Unbound variable: ~a" x)))]

    ;; Function definition - capture closure, attach metadata
    [`(fn ,params ,body ,annotations)
     (let* ([meta (make-function-metadata (current-context) annotations)]
            [fn (fn-val params body env meta)])
       (tracked-val fn (current-context) #f))]

    ;; Function call - THIS IS WHERE THE MAGIC HAPPENS
    [`(call ,f-expr ,arg-exprs ...)
     (let* ([f-val (eval-expr f-expr env)]
            [fn (tracked-val-data f-val)]
            [arg-vals (map (λ (e) (eval-expr e env)) arg-exprs)])
       (call-function fn arg-vals))]

    ;; Conditional
    [`(if ,test-expr ,then-expr ,else-expr)
     (let ([test-val (eval-expr test-expr env)])
       ;; Note: using Rosette's if for symbolic execution
       (if (tracked-val-data test-val)
           (eval-expr then-expr env)
           (eval-expr else-expr env)))]

    ;; Binary operations
    [`(binop ,op ,e1 ,e2)
     (let* ([v1 (eval-expr e1 env)]
            [v2 (eval-expr e2 env)]
            [result ((get-binop op)
                     (tracked-val-data v1)
                     (tracked-val-data v2))])
       ;; Result is tainted if either operand was tainted
       (tracked-val result
                    (current-context)
                    (or (tracked-val-tainted v1)
                        (tracked-val-tainted v2))))]

    ;; Object creation: (new ((prop1 expr1) (prop2 expr2) ...) annotations)
    ;; Creates a new object in the current context
    [`(new ,prop-defs ,annotations)
     (let* ([obj (make-empty-object (current-context) annotations)]
            [props (obj-val-properties obj)])
       ;; Evaluate each property expression and store it
       (for ([prop-def prop-defs])
         (match prop-def
           [`(,prop-name ,prop-expr)
            (let ([val (eval-expr prop-expr env)])
              (hash-set! props prop-name val))]))
       ;; Return the object wrapped in tracked-val
       (tracked-val obj (current-context) #f))]

    ;; Property read: (get obj-expr prop-name)
    ;; May coerce the result if object is from a different context
    [`(get ,obj-expr ,prop-name)
     (let* ([obj-tracked (eval-expr obj-expr env)]
            [obj (tracked-val-data obj-tracked)])
       (get-property obj prop-name))]

    ;; Property write: (set obj-expr prop-name val-expr)
    ;; May be blocked if object is sealed or from different context
    [`(set ,obj-expr ,prop-name ,val-expr)
     (let* ([obj-tracked (eval-expr obj-expr env)]
            [obj (tracked-val-data obj-tracked)]
            [val (eval-expr val-expr env)])
       (set-property! obj prop-name val))]

    ;; Method call: (method-call obj-expr method-name args...)
    ;; Looks up the method, binds `this`, and calls with context handling
    [`(method-call ,obj-expr ,method-name ,arg-exprs ...)
     (let* ([obj-tracked (eval-expr obj-expr env)]
            [obj (tracked-val-data obj-tracked)]
            [method-val (get-property obj method-name)]
            [fn (tracked-val-data method-val)]
            [arg-vals (map (λ (e) (eval-expr e env)) arg-exprs)])
       ;; Call the method with the object as an implicit first argument (this)
       (call-method fn obj-tracked arg-vals))]))

;; Binary operation lookup
(define (get-binop op)
  (case op
    [(+) +]
    [(-) -]
    [(*) *]
    [(/) /]
    [(==) equal?]
    [(<) <]
    [(>) >]
    [else (error 'get-binop "Unknown operator: ~a" op)]))

;; =============================================================================
;; PART 5: FUNCTION CALL MECHANICS
;; =============================================================================

;; This is the core of the context system.
;; When calling a function:
;; 1. Decide if we need to push a new context
;; 2. If crossing contexts, coerce all arguments
;; 3. Execute the function body
;; 4. If we crossed contexts, coerce the return value back
(define (call-function fn arg-vals)
  (let* ([caller-ctx (current-context)]
         [target-ctx (should-push-context? fn arg-vals caller-ctx)]
         [will-cross? (and target-ctx
                           (not (equal? (context-id target-ctx)
                                        (context-id caller-ctx))))])

    (if (not target-ctx)
        ;; No context switch - just call in current context
        (execute-fn-body fn arg-vals)

        ;; Context switch - handle boundary crossing
        (let* (;; Coerce arguments from caller context to function context
               [coerced-args
                (if will-cross?
                    (map (λ (arg)
                           (cross-boundary arg caller-ctx target-ctx 'arg))
                         arg-vals)
                    arg-vals)]

               ;; Execute in the new context
               [raw-result
                (push-context target-ctx
                              (λ () (execute-fn-body fn coerced-args)))]

               ;; Coerce return value back to caller context
               [coerced-result
                (if will-cross?
                    (cross-boundary raw-result target-ctx caller-ctx 'return)
                    raw-result)])

          coerced-result))))

;; Actually run the function body with args bound
(define (execute-fn-body fn arg-vals)
  (let* ([params (fn-val-params fn)]
         [body (fn-val-body fn)]
         [closure-env (fn-val-closure-env fn)]
         [bindings (map cons params arg-vals)]
         [extended-env (append bindings closure-env)])
    (eval-expr body extended-env)))

;; =============================================================================
;; PART 5b: OBJECT PROPERTY ACCESS
;; =============================================================================

;; Read a property from an object.
;; Handles prototype chain lookup and cross-context coercion.
(define (get-property obj prop-name)
  (let* ([props (obj-val-properties obj)]
         [proto (obj-val-prototype obj)]
         [accessor-ctx (current-context)]
         [raw-val (hash-ref props prop-name #f)])

    (cond
      ;; Property exists on this object
      [raw-val
       (let ([obj-ctx (should-coerce-property-access? obj prop-name accessor-ctx)])
         (if obj-ctx
             ;; Crossing context boundary - coerce the value
             (coerce-for-property-access raw-val obj-ctx accessor-ctx prop-name)
             ;; Same context - return as-is
             raw-val))]

      ;; Check prototype chain
      [proto
       (get-property proto prop-name)]

      ;; Property not found
      [else
       (tracked-val 'undefined (current-context) #f)])))

;; Write a property to an object.
;; May be blocked based on sealing/context policies.
(define (set-property! obj prop-name val)
  (let ([accessor-ctx (current-context)])
    ;; Check if write is allowed
    (when (not (can-write-property? obj prop-name val accessor-ctx))
      (error 'set-property!
             "Cannot write property ~a on sealed object from context ~a"
             prop-name (context-id accessor-ctx)))

    ;; Check if we need to coerce the value being written
    (let* ([obj-ctx (obj-metadata-owner-context (obj-val-metadata obj))]
           [coerced-val (if (equal? (context-id obj-ctx) (context-id accessor-ctx))
                            val
                            (coerce-value val accessor-ctx obj-ctx))])
      (hash-set! (obj-val-properties obj) prop-name coerced-val)
      ;; Return the value (like JS assignment)
      val)))

;; Call a method on an object.
;; The object is bound to `this` in the function body.
;; Context handling considers both the function's context and the object's context.
(define (call-method fn obj-tracked arg-vals)
  (let* ([caller-ctx (current-context)]
         [obj (tracked-val-data obj-tracked)]
         [fn-ctx (fn-metadata-owner-context (fn-val-metadata fn))]
         [obj-ctx (obj-metadata-owner-context (obj-val-metadata obj))]

         ;; Decide which context to execute in
         ;; DEFAULT: Use function's owner context (you may want to customize this)
         [target-ctx (should-push-context? fn arg-vals caller-ctx)]

         [will-cross? (and target-ctx
                           (not (equal? (context-id target-ctx)
                                        (context-id caller-ctx))))])

    (if (not target-ctx)
        ;; No context switch
        (execute-method-body fn obj-tracked arg-vals)

        ;; Context switch - coerce this and args
        (let* ([coerced-this (if will-cross?
                                 (cross-boundary obj-tracked caller-ctx target-ctx 'this)
                                 obj-tracked)]
               [coerced-args (if will-cross?
                                 (map (λ (arg)
                                        (cross-boundary arg caller-ctx target-ctx 'arg))
                                      arg-vals)
                                 arg-vals)]

               [raw-result
                (push-context target-ctx
                              (λ () (execute-method-body fn coerced-this coerced-args)))]

               [coerced-result
                (if will-cross?
                    (cross-boundary raw-result target-ctx caller-ctx 'return)
                    raw-result)])

          coerced-result))))

;; Execute method body with `this` bound
(define (execute-method-body fn this-val arg-vals)
  (let* ([params (fn-val-params fn)]
         [body (fn-val-body fn)]
         [closure-env (fn-val-closure-env fn)]
         ;; Bind `this` plus regular parameters
         [bindings (cons (cons 'this this-val)
                         (map cons params arg-vals))]
         [extended-env (append bindings closure-env)])
    (eval-expr body extended-env)))

;; =============================================================================
;; PART 6: SYMBOLIC EXECUTION & PROPERTY CHECKING
;; =============================================================================

;; Create a symbolic context for testing
(define (make-symbolic-context name)
  (define-symbolic* level integer?)
  (context name level '()))

;; Create a symbolic tracked value
(define (make-symbolic-value ctx)
  (define-symbolic* data integer?)
  (define-symbolic* tainted boolean?)
  (tracked-val data ctx tainted))

;; -----------------------------------------------------------------------------
;; PROPERTY: Information doesn't flow from high to low without coercion
;; -----------------------------------------------------------------------------
(define (check-no-leak-property)
  (define high-ctx (context 'high 10 '()))
  (define low-ctx (context 'low 1 '()))

  ;; A function in the low context
  (define low-fn
    (fn-val '(x)
            '(var x)  ; Just returns its argument
            '()
            (fn-metadata low-ctx '())))

  ;; A value from the high context
  (define high-value (tracked-val 42 high-ctx #f))

  ;; Call low function with high value
  (parameterize ([current-context-stack (list high-ctx)])
    (let ([result (call-function low-fn (list high-value))])
      ;; Property: result should be tainted since it crossed from high to low
      (assert (tracked-val-tainted result)
              "High value passed through low function should be tainted"))))

;; -----------------------------------------------------------------------------
;; PROPERTY: Coercion is applied symmetrically (args and returns)
;; -----------------------------------------------------------------------------
(define (check-symmetric-coercion)
  (define ctx-a (context 'a 5 '()))
  (define ctx-b (context 'b 10 '()))

  ;; Function in ctx-b that returns its input
  (define fn-in-b
    (fn-val '(x)
            '(var x)
            '()
            (fn-metadata ctx-b '())))

  (define-symbolic* input-data integer?)
  (define input-val (tracked-val input-data ctx-a #f))

  (parameterize ([current-context-stack (list ctx-a)])
    (let ([result (call-function fn-in-b (list input-val))])
      ;; After round-trip, data should be same but origin tracking updated
      (assert (equal? (tracked-val-data result) input-data)
              "Data should survive round-trip"))))

;; -----------------------------------------------------------------------------
;; PROPERTY: Reading property from different context causes coercion
;; -----------------------------------------------------------------------------
(define (check-object-property-coercion)
  (define high-ctx (context 'high 10 '()))
  (define low-ctx (context 'low 1 '()))

  ;; Create an object in the high context
  (define high-obj
    (push-context high-ctx
                  (λ ()
                    (let ([obj (make-empty-object high-ctx '())])
                      (hash-set! (obj-val-properties obj)
                                 'secret
                                 (tracked-val 42 high-ctx #f))
                      obj))))

  ;; Read the property from low context
  (parameterize ([current-context-stack (list low-ctx)])
    (let ([result (get-property high-obj 'secret)])
      ;; Property should be tainted when read from low context
      (assert (tracked-val-tainted result)
              "Property from high object should be tainted when read from low context"))))

;; -----------------------------------------------------------------------------
;; PROPERTY: Sealed objects cannot be modified from other contexts
;; -----------------------------------------------------------------------------
(define (check-sealed-object-protection)
  (define owner-ctx (context 'owner 5 '()))
  (define other-ctx (context 'other 5 '()))

  ;; Create a sealed object
  (define sealed-obj
    (push-context owner-ctx
                  (λ ()
                    (make-empty-object owner-ctx '((sealed . #t))))))

  ;; Attempt to write from another context should fail
  ;; (We test this by checking can-write-property? directly since
  ;; the actual set-property! would throw an error)
  (parameterize ([current-context-stack (list other-ctx)])
    (let ([can-write (can-write-property? sealed-obj 'x (tracked-val 1 other-ctx #f) other-ctx)])
      (assert (not can-write)
              "Should not be able to write to sealed object from different context"))))

;; -----------------------------------------------------------------------------
;; PROPERTY: Method calls coerce `this` appropriately
;; -----------------------------------------------------------------------------
(define (check-method-this-coercion)
  (define obj-ctx (context 'obj-ctx 10 '()))
  (define caller-ctx (context 'caller-ctx 1 '()))

  ;; Create object with a method that returns `this`
  (define obj-with-method
    (push-context obj-ctx
                  (λ ()
                    (let ([obj (make-empty-object obj-ctx '())])
                      (hash-set! (obj-val-properties obj)
                                 'getData
                                 (tracked-val
                                  (fn-val '()
                                          '(var this)  ; Returns this
                                          '()
                                          (fn-metadata obj-ctx '()))
                                  obj-ctx #f))
                      (hash-set! (obj-val-properties obj)
                                 'value
                                 (tracked-val 999 obj-ctx #f))
                      obj))))

  ;; Call method from a different context
  (parameterize ([current-context-stack (list caller-ctx)])
    (let* ([method-val (get-property obj-with-method 'getData)]
           [fn (tracked-val-data method-val)]
           [result (call-method fn (tracked-val obj-with-method obj-ctx #f) '())])
      ;; The returned `this` should be coerced
      (assert (tracked-val-tainted result)
              "`this` returned from method should be tainted when crossing to lower context"))))

;; =============================================================================
;; PART 7: RUNNING VERIFICATION
;; =============================================================================

;; Verify a property - returns #f if OK, or a counterexample
(define (verify-property prop-thunk description)
  (printf "Checking: ~a\n" description)
  (let ([result (verify (prop-thunk))])
    (if (unsat? result)
        (begin (printf "  ✓ Property holds\n") #f)
        (begin (printf "  ✗ COUNTEREXAMPLE FOUND:\n")
               (printf "    ~a\n" result)
               result))))

;; Run all property checks
(define (run-all-checks)
  (printf "\n=== RUNNING VERIFICATION ===\n\n")

  (printf "--- Function Call Properties ---\n")
  (verify-property check-no-leak-property
                   "High values get tainted when passing through low context")
  (verify-property check-symmetric-coercion
                   "Data survives round-trip through different context")

  (printf "\n--- Object Properties ---\n")
  (verify-property check-object-property-coercion
                   "Property access across contexts triggers coercion")
  (verify-property check-sealed-object-protection
                   "Sealed objects protected from foreign writes")
  (verify-property check-method-this-coercion
                   "`this` is coerced when method crosses context boundary")

  (printf "\n=== DONE ===\n"))

;; =============================================================================
;; PART 8: EXAMPLE USAGE
;; =============================================================================

;; Example: Set up a simple two-context system and run some code
(define (example-run)
  (define trusted-ctx (context 'trusted 10 '(read write)))
  (define untrusted-ctx (context 'untrusted 1 '(read)))

  (printf "\n=== EXAMPLE 1: Function Call Across Contexts ===\n\n")

  ;; A function defined in the untrusted context
  (define untrusted-fn-expr
    '(fn (x) (binop + (var x) (const 1)) ()))

  ;; Evaluate it in untrusted context
  (define untrusted-fn
    (push-context untrusted-ctx
                  (λ () (eval-expr untrusted-fn-expr '()))))

  ;; Call it from trusted context with a trusted value
  (define trusted-value (tracked-val 100 trusted-ctx #f))

  (printf "Calling untrusted function from trusted context...\n")
  (push-context trusted-ctx
                (λ ()
                  (let ([result (call-function (tracked-val-data untrusted-fn)
                                               (list trusted-value))])
                    (printf "Result: ~a\n" result)
                    (printf "Tainted? ~a\n" (tracked-val-tainted result))
                    result)))

  (printf "\n=== EXAMPLE 2: Object Property Access Across Contexts ===\n\n")

  ;; Create a trusted object with sensitive data
  (define trusted-obj
    (push-context trusted-ctx
                  (λ ()
                    (let ([obj (make-empty-object trusted-ctx '())])
                      (hash-set! (obj-val-properties obj)
                                 'secret
                                 (tracked-val "password123" trusted-ctx #f))
                      (hash-set! (obj-val-properties obj)
                                 'public
                                 (tracked-val "hello" trusted-ctx #f))
                      obj))))

  (printf "Reading trusted object properties from untrusted context...\n")
  (push-context untrusted-ctx
                (λ ()
                  (let ([secret (get-property trusted-obj 'secret)]
                        [public (get-property trusted-obj 'public)])
                    (printf "Secret value: ~a, Tainted? ~a\n"
                            (tracked-val-data secret)
                            (tracked-val-tainted secret))
                    (printf "Public value: ~a, Tainted? ~a\n"
                            (tracked-val-data public)
                            (tracked-val-tainted public)))))

  (printf "\n=== EXAMPLE 3: Method Call With This Binding ===\n\n")

  ;; Create object with a method
  (define obj-with-method
    (push-context trusted-ctx
                  (λ ()
                    (let ([obj (make-empty-object trusted-ctx '())])
                      (hash-set! (obj-val-properties obj)
                                 'value
                                 (tracked-val 42 trusted-ctx #f))
                      ;; Method that reads this.value
                      (hash-set! (obj-val-properties obj)
                                 'getValue
                                 (tracked-val
                                  (fn-val '()
                                          '(get (var this) value)
                                          '()
                                          (fn-metadata trusted-ctx '()))
                                  trusted-ctx #f))
                      obj))))

  (printf "Calling method from untrusted context...\n")
  (push-context untrusted-ctx
                (λ ()
                  (let* ([method-tracked (get-property obj-with-method 'getValue)]
                         [method (tracked-val-data method-tracked)]
                         [result (call-method method
                                              (tracked-val obj-with-method trusted-ctx #f)
                                              '())])
                    (printf "Method result: ~a, Tainted? ~a\n"
                            (tracked-val-data result)
                            (tracked-val-tainted result)))))

  (printf "\n=== DONE ===\n"))

;; =============================================================================
;; PART 9: HELPERS FOR BUILDING YOUR MODEL
;; =============================================================================

;; Record boundary crossings for debugging/analysis
(define crossing-log (make-parameter '()))

(define (log-crossing val from to direction)
  (crossing-log
   (cons (list 'crossing
               (tracked-val-data val)
               (context-id from)
               (context-id to)
               direction)
         (crossing-log))))

;; Wrap cross-boundary with logging
(define (cross-boundary/logged val from-ctx to-ctx direction)
  (log-crossing val from-ctx to-ctx direction)
  (cross-boundary val from-ctx to-ctx direction))

;; Print the crossing log
(define (print-crossings)
  (printf "\n=== BOUNDARY CROSSINGS ===\n")
  (for ([entry (reverse (crossing-log))])
    (match entry
      [(list 'crossing data from to dir)
       (printf "  ~a: ~a → ~a (value: ~a)\n" dir from to data)])))

;; =============================================================================
;; UNCOMMENT TO RUN
;; =============================================================================

;; (example-run)
;; (run-all-checks)
