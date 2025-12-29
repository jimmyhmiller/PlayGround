# Context-Aware JavaScript Operational Model

A Rosette-based formal model for reasoning about JavaScript-like languages with **context boundaries** and **value coercion**. Use this to find counterexamples to security properties in your context isolation system.

## What This Is

This models a language where:
- **Functions and objects belong to contexts** (security domains)
- **Calling across contexts triggers coercion** on arguments and return values
- **Property access across contexts triggers coercion** on the values read
- **Some objects can be sealed** to prevent modification from foreign contexts

The goal is to encode your real system's rules, then use Rosette's solver to find inputs that violate properties you expect to hold.

## Quick Start

```bash
# Run examples
racket -it context-model.rkt
> (example-run)

# Run verification (find counterexamples)
> (run-all-checks)
```

## Core Concepts

### Contexts

A context represents a security domain with an ID, privilege level, and capabilities:

```racket
(struct context (id level capabilities))

(define trusted-ctx (context 'trusted 10 '(read write)))
(define untrusted-ctx (context 'untrusted 1 '(read)))
```

### Tracked Values

Every runtime value tracks its origin context and taint status:

```racket
(struct tracked-val (data origin tainted))
```

### The Context Stack

Execution maintains a stack of contexts. When you call a function that belongs to a different context, the stack is pushed:

```
trusted-ctx calls untrusted-fn:
  [trusted] → [untrusted, trusted]
                ↑ current context
```

### Boundary Crossings

When values cross context boundaries (as arguments, returns, or property accesses), they go through:

1. **`can-cross-boundary?`** - Is this crossing allowed at all?
2. **`coerce-value`** - Transform the value for the target context

## Extension Points

These are the functions you modify to match your real system. Each has a default implementation and commented alternatives.

### Function Calls

| Function | When Called | Default Behavior |
|----------|-------------|------------------|
| `should-push-context?` | Before every call | Push to function's owner context |
| `make-function-metadata` | When function is defined | Inherit defining context |
| `coerce-value` | Args entering, returns exiting | Taint when going high→low |
| `can-cross-boundary?` | Before any crossing | Always allow |

### Objects

| Function | When Called | Default Behavior |
|----------|-------------|------------------|
| `make-object-metadata` | When object is created | Inherit context, check `sealed` annotation |
| `should-coerce-property-access?` | Reading any property | Coerce if different context |
| `coerce-for-property-access` | After deciding to coerce | Same as `coerce-value` |
| `can-write-property?` | Before property write | Block if sealed + different context |

## Adding Your Own Coercion Logic

Find `EXTENSION POINT 4` in the file and modify `coerce-value`:

```racket
;; Example: Wrap in a membrane instead of tainting
(define (coerce-value val from-ctx to-ctx)
  (cond
    [(equal? (context-id from-ctx) (context-id to-ctx))
     val]  ; Same context, no change

    [else
     ;; Create a membrane-wrapped value
     (tracked-val (membrane-wrap (tracked-val-data val) from-ctx to-ctx)
                  to-ctx
                  #f)]))
```

## Adding New Properties to Verify

Properties are functions that set up a scenario and assert what should hold:

```racket
(define (check-my-property)
  ;; 1. Set up contexts
  (define ctx-a (context 'a 5 '()))
  (define ctx-b (context 'b 10 '()))

  ;; 2. Create values/functions/objects
  (define secret-val (tracked-val 42 ctx-b #f))

  ;; 3. Use define-symbolic* for inputs Rosette should explore
  (define-symbolic* unknown-level integer?)

  ;; 4. Run some operations
  (parameterize ([current-context-stack (list ctx-a)])
    (let ([result (some-operation secret-val)])
      ;; 5. Assert what should be true
      (assert (some-property? result)
              "Description of what went wrong"))))
```

Then add it to `run-all-checks`:

```racket
(verify-property check-my-property
                 "Description for output")
```

If the property can be violated, Rosette returns a **counterexample** showing the specific inputs that break it.

## Adding New Expression Types

The interpreter uses pattern matching. Add new cases to `eval-expr`:

```racket
(define (eval-expr expr env)
  (match expr
    ;; ... existing cases ...

    ;; Your new expression type
    [`(try ,body-expr ,catch-expr)
     (with-handlers ([exn:fail? (λ (e) (eval-expr catch-expr env))])
       (eval-expr body-expr env))]

    ;; etc.
    ))
```

## Adding New Metadata

To track additional information on functions/objects:

1. Extend the metadata struct:
```racket
(struct fn-metadata (owner-context tags my-new-field) #:transparent)
```

2. Update `make-function-metadata` to populate it

3. Use it in your extension points:
```racket
(define (should-push-context? fn args current-ctx)
  (if (fn-metadata-my-new-field (fn-val-metadata fn))
      (fn-metadata-owner-context (fn-val-metadata fn))
      #f))
```

## Debugging

Enable crossing logs to see all boundary crossings:

```racket
;; In your test code, use cross-boundary/logged instead of cross-boundary
;; Then call:
(parameterize ([crossing-log '()])
  ;; ... your operations ...
  (print-crossings))
```

Output:
```
=== BOUNDARY CROSSINGS ===
  arg: trusted → untrusted (value: 100)
  return: untrusted → trusted (value: 101)
```

## File Structure

```
context-model.rkt
├── PART 1: Core Data Structures
│   └── context, tracked-val, fn-val, obj-val, obj-metadata
├── PART 2: Extension Points (MODIFY THESE)
│   └── should-push-context?, make-*-metadata, coerce-*, can-*
├── PART 3: Boundary Crossing Logic
│   └── cross-boundary
├── PART 4: The Interpreter
│   └── eval-expr with all expression types
├── PART 5: Function Call Mechanics
│   └── call-function, execute-fn-body
├── PART 5b: Object Property Access
│   └── get-property, set-property!, call-method
├── PART 6: Properties to Verify (ADD YOURS HERE)
│   └── check-no-leak-property, check-*-coercion, etc.
├── PART 7: Verification Runner
│   └── verify-property, run-all-checks
├── PART 8: Examples
│   └── example-run
└── PART 9: Helpers
    └── crossing-log, print-crossings
```

## Common Patterns

### "Find me inputs where X escapes to Y"

```racket
(define (check-no-escape)
  (define-symbolic* secret integer?)
  (define high-ctx (context 'high 10 '()))
  (define low-ctx (context 'low 1 '()))

  ;; Set up scenario where secret might escape
  ;; ...

  ;; Assert it doesn't appear uncoerced in low context
  (assert (=> (in-low-context? result)
              (tracked-val-tainted result))))
```

### "Verify coercion is idempotent"

```racket
(define (check-idempotent)
  (define-symbolic* data integer?)
  (define val (tracked-val data ctx-a #f))

  (let* ([once (coerce-value val ctx-a ctx-b)]
         [twice (coerce-value once ctx-a ctx-b)])  ; coerce again
    (assert (equal? once twice))))
```

### "Check that capabilities are respected"

```racket
(define (check-capabilities)
  (define no-write-ctx (context 'limited 5 '(read)))  ; no 'write
  (define obj (make-sealed-object ...))

  (parameterize ([current-context-stack (list no-write-ctx)])
    (assert (not (can-write-property? obj 'x some-val no-write-ctx)))))
```

## Tips

1. **Start small** - Model 2 contexts, 1 coercion rule, 1 property. Expand from there.

2. **Use `define-symbolic*`** - This tells Rosette to explore all possible values. Great for finding edge cases.

3. **Properties should be falsifiable** - Write properties that *could* fail. If Rosette always says "holds", your property might be trivially true.

4. **Check the counterexamples** - When Rosette finds one, it shows the specific values that break your property. This is the gold.

5. **Beware mutation** - Objects use mutable hashes. For pure symbolic execution, consider immutable alternatives.

## Next Steps

Ideas for extending this model:

- **Prototype chains** - The `prototype` field exists but isn't fully implemented
- **Async/callbacks** - Model what happens when callbacks cross contexts
- **Membrane patterns** - Replace simple tainting with revocable membrane wrappers
- **Multiple coercion strategies** - Different coercion for different value types
- **Capability attenuation** - Model how capabilities degrade across boundaries
