# jspartial - Partial Evaluator for JavaScript

A multi-pass partial evaluator targeting a subset of JavaScript, focused on specializing bytecode interpreters.

## Guiding Principles

### 1. Semantic Preservation is Non-Negotiable
At ALL times, every intermediate representation must be valid JavaScript with **identical semantics** to the input program. If we can't preserve semantics, we don't transform. This is the most important invariant.

### 2. Multi-Pass Architecture
Transformations happen in discrete passes. Each pass does one thing well:
- Parse → AST
- Pass 1: Constant propagation
- Pass 2: Constant folding
- Pass 3: Dead code elimination
- Pass N: ...
- Emit → JavaScript

After every single pass, we must be able to emit valid JS and run it.

### 3. No Special-Casing for Examples
We define general semantics for JavaScript operators and constructs. We NEVER special-case behavior for specific examples or benchmarks. If a transformation doesn't work in general, we don't do it.

### 4. Builtins are First-Class
JavaScript operators like `+`, `-`, `*`, array access `[]`, etc. need explicit semantic definitions. These definitions should be:
- Declarative (describe what they do, not how to optimize them)
- Complete (handle all cases the operator handles)
- Testable (we can verify our semantic model matches JS)

### 5. Conservative Over Clever
When in doubt, don't transform. A correct but slower program beats an incorrect but fast one. We can always add more transformations later.

### 6. Binding-Time Analysis
Values are either:
- **Static**: Known at partial evaluation time (can be computed/inlined)
- **Dynamic**: Unknown until runtime (must remain in residual program)

Mixed expressions require careful handling - we can partially reduce but must preserve dynamic parts.

## Focus Area: Bytecode Interpreters

Our primary use case is specializing interpreters like:
```javascript
function run(program) {
    let pc = 0;
    let stack = [];
    while (pc < program.length) {
        switch (program[pc]) {
            case 0: // PUSH
                stack.push(program[pc + 1]);
                pc += 2;
                break;
            case 1: // ADD
                stack.push(stack.pop() + stack.pop());
                pc += 1;
                break;
        }
    }
    return stack;
}
```

When `program` is known, we should be able to unroll the loop and eliminate the dispatch overhead.

## Testing Strategy

All tests use Node.js to verify semantic equivalence:
1. Run original code with `node -e`
2. Parse → Transform → Emit
3. Run transformed code with `node -e`
4. Assert: both must succeed AND produce identical stdout

This is the gold standard - we don't check string equality of code, we check behavioral equivalence.

## Supported JavaScript Subset (Initial)

- Literals: numbers, strings, booleans, arrays
- Variables: `let`, `var`, `const`
- Operators: arithmetic, comparison, logical
- Control flow: `if`/`else`, `while`, `switch`/`case`
- Functions: declarations, calls
- Arrays: literals, indexing, `.push()`, `.pop()`, `.length`
- Objects: literals, property access (stretch goal)

## Commands

```bash
# Run tests
cargo test

# Run on a file
cargo run -- input.js

# Run with verbose pass output
cargo run -- input.js --verbose
```

## Important Notes

- **DO NOT use `/dev/stdin`** - it will cause the process to freeze. Always write test code to a temporary file first.
