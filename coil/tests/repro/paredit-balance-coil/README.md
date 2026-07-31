# `paredit-like balance` moves delimiters in valid following Coil forms

Reproduced with `paredit-like 0.1.0`.

`input.coil` has exactly one intended defect: the `loop` in `locally-broken`
is missing one closing parenthesis. `already-valid` is a balanced Coil function
using the compiler's normal indentation for multiline `cond` predicates.

Run:

```sh
paredit-like balance tests/repro/paredit-balance-coil/input.coil --diff
```

The local repair is correct, but the command also removes two closing
parentheses from `already-valid` and appends them at the end of that function.
That output is saved verbatim as `actual.coil`. The desired output is
`expected.coil`: add one `)` to the broken loop and leave the following form
byte-for-byte unchanged.

Useful checks:

```sh
coil check tests/repro/paredit-balance-coil/input.coil     # unclosed `(`, expected
coil check tests/repro/paredit-balance-coil/expected.coil  # succeeds
coil check tests/repro/paredit-balance-coil/actual.coil    # fails after unwanted rewrite
```

The likely trigger is the indentation relationship between a multiline `cond`
predicate and its result expression. Once balancing is active, the Parinfer-style
pass interprets the result as nested inside the predicate, even though the input
form was already structurally balanced. A safe repair should preserve balanced
top-level forms following the damaged form, or at minimum stop once balance is
restored at the next top-level boundary.
