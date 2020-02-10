```
rule fibonacci:
    fib 0 => 0
    fib 1 => 1
    fib ?n => fib (?n - 1) + fib (?n - 2)

rule cache-fibonacci:
    (fib ?n => ?x) and not(@fibs.?n)
        @fibs.?n = ?x

    fib ?n and @fibs.?n
        @fibs.?n

rule catch-negative-fib:
    fib ?n | ?n < 0 => error "fib cannot be less than one"

rule test-fib:
    (fib 5 => ?x) => assert ?x == 5
    (fib 6 => ?x) => assert ?x == 8

rule test-helper:
    test ?fn ?arg = ?answer => ((?fn ?arg => ?x) => assert ?x == ?answer)

rule test-fib:
    test fib 7 = 13
    test fib 8 = 21

rule display-test:
    (fib ?x => error ?message) => error "fib of ${?x} failed with message ${message}"

rule change-test-display:
    display((?fn ?arg) => assert ?x == ?y) => display(test ?fn ?arg = ?y)
```