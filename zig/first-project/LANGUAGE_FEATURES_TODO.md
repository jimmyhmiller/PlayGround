# Language Features for Self-Hosting

## Critical Features

### 1. Pointers
```lisp
(def ptr (: (Ptr Int)) (& x))          ; address-of
(def value (: Int) (* ptr))            ; dereference
(def null-ptr (: (Ptr Int)) nullptr)   ; null pointer
```

### 2. Arrays
```lisp
(def arr (: (Array Int 10)) ...)       ; fixed-size array type
(def elem (: Int) ([] arr 5))          ; array indexing
```

### 3. Loops
```lisp
(while condition
  body)

(for [i 0] (< i 10) (+ i 1)
  body)
```

### 4. Mutable Variables
```lisp
(var x (: Int) 10)                     ; mutable variable
(set! x 20)                            ; assignment
```

### 5. Pattern Matching / Switch
```lisp
(match value
  (0 "zero")
  (1 "one")
  (_ "other"))
```

### 6. Better Enums (Tagged Unions)
```lisp
(def TokenType (: Type)
  (Enum
    (Number Int)
    (String []const u8)
    (Symbol []const u8)
    EOF))

(match token
  ((Number n) (process-number n))
  ((String s) (process-string s))
  ...)
```

### 7. Option/Maybe Type
```lisp
(def result (: (Option Int)) (Some 42))
(def nothing (: (Option Int)) None)

(match result
  ((Some val) val)
  (None 0))
```

### 8. Multiple Return Values / Tuples
```lisp
(def get-pair (: (-> [] (Tuple Int Int)))
  (fn [] (tuple 1 2)))

(let [(a b) (get-pair)]
  (+ a b))
```

### 9. Logical Operators
```lisp
(and a b)
(or a b)
(not x)
```

### 10. Break/Continue in Loops
```lisp
(while true
  (if (some-condition)
    break
    continue))
```

## Important Features

### 11. Type Aliases
```lisp
(defalias String (Ptr u8))
(defalias CharPtr (Ptr char))
```

### 12. Const/Immutable Pointers
```lisp
(Ptr const T)         ; const pointer
(const Ptr T)         ; pointer to const
```

### 13. Variadic Functions (for C interop)
```lisp
(def printf (: (-> [(Ptr const u8) ...] Int)) ...)
```

### 14. Casts / Type Conversions
```lisp
(cast Int u32-value)
(as (Ptr void) some-ptr)
```

### 15. Sizeof
```lisp
(sizeof Int)
(sizeof (Struct ...))
```

## Nice to Have

### 16. Do Blocks / Sequencing
```lisp
(do
  (action1)
  (action2)
  result)
```

### 17. Destructuring in Let
```lisp
(let [(Point x y) point]
  (+ x y))
```

### 18. String Literals as Arrays
```lisp
"hello" => (Array u8 6) with null terminator
```

### 19. Bitwise Operators
```lisp
(bit-and a b)
(bit-or a b)
(bit-xor a b)
(bit-not x)
(bit-shl x n)
(bit-shr x n)
```

### 20. Increment/Decrement
```lisp
(inc! x)  ; x = x + 1
(dec! x)  ; x = x - 1
```

## Syntax Examples for Self-Hosting

### Lexer Example
```lisp
(var i (: Int) 0)
(var tokens (: (Ptr Token)) (malloc (* (sizeof Token) 1000)))

(while (< i (strlen source))
  (match ([] source i)
    (#\( (do
           (set! ([] tokens token-count) (Token OpenParen))
           (set! token-count (+ token-count 1))))
    (#\) (do
           (set! ([] tokens token-count) (Token CloseParen))
           (set! token-count (+ token-count 1))))
    ...)
  (set! i (+ i 1)))
```

### Hash Map Example
```lisp
(def HashMap (: Type)
  (Struct
    [buckets (Ptr (Ptr Entry))]
    [size Int]
    [capacity Int]))

(def get (: (-> [HashMap (Ptr const u8)] (Option (Ptr void))))
  (fn [map key]
    (var hash (: Int) (hash-string key))
    (var index (: Int) (% hash map.capacity))
    (var entry (: (Ptr Entry)) ([] map.buckets index))
    
    (while (!= entry nullptr)
      (if (== 0 (strcmp entry.key key))
        (return (Some entry.value))
        (set! entry entry.next)))
    
    None))
```
