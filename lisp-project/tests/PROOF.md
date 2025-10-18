# PROOF: The Real Issue with the Tokenizer

## I was WRONG about my diagnosis

I claimed the issue was:
> "In let bodies, you can only have ONE expression"

But the tests prove this is FALSE:

### Test 1: Multiple expressions in let body WORKS ✅
```bash
$ build0/bin/lisp0 tests/let_while_test2.lisp --run
Testing the exact pattern from tokenizer...
Token value: 3
```

The pattern `(let [x 0] (while ...) (make-token ...))` works fine!

## The REAL Issue: `and` only takes 2 arguments!

### Test 2: `and` with 6 arguments FAILS ❌
```bash
$ build0/bin/lisp0 tests/and_test.lisp --run
ERROR: Function body expression #1 failed: error.InvalidTypeAnnotation
  Expression: (if (and true true true true true true) ...)
Type error at line 13 (expr #3): InvalidTypeAnnotation
```

## The Actual Problem in the Tokenizer

This code:
```lisp
(while (and (!= (peek-char tok) 0)
           (= (isspace (peek-char tok)) 0)
           (!= (peek-char tok) 40)
           (!= (peek-char tok) 41)
           (!= (peek-char tok) 91)
           (!= (peek-char tok) 93))  ; <-- 6 arguments to `and`!
  ...)
```

Should be:
```lisp
(while (and (!= (peek-char tok) 0)
           (and (= (isspace (peek-char tok)) 0)
                (and (!= (peek-char tok) 40)
                     (and (!= (peek-char tok) 41)
                          (and (!= (peek-char tok) 91)
                               (!= (peek-char tok) 93))))))
  ...)
```

## Conclusion

The constraint is: **`and` and `or` are binary operators (2 args only)**, not variadic.

This is documented in LANGUAGE_REFERENCE.md but I missed it!
