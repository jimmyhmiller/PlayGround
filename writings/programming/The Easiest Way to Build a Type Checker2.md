# The Easiest Way to Build a Type Checker

Type checkers are a piece of software that, at once, feel incredibly simple, yet incredibly complex. Seeing [Hindley Milner written in a logic programming language](https://github.com/webyrd/hindley-milner-type-inferencer/blob/master/type-inferencer.scm) is almost magical, but it never helped me understand how it was implemented. Nor does actually trying to read anything about Algorithm W. But thanks to [placeholder], I have discovered a setup for type checking that is so conceptually simple that it demystified the whole thing for me. It goes by the name Bidirectional Type Checking.

## Bidirectional Type Checking

The two directions in this type checker are inferring types and checking types. Our type inference can let us not have to put types on every single variable. But we still need some type annotations, but these are just for function definitions. (I'm sure for some complicated type systems, there are things we can't infer). So we can write code like:

```typescript
function sillyCalculator(x: number): number {
  let a = 10;
  let b: number = 20;
  let c = 30;
  let d: number = 40;
  let e = a;
  let f: number = b;
  let g = "hello";
  let h: string = "world";
  let i = 100;
  let j: number = 200;
  return x;
}
```

Depending on what we consider valid in our type system (can you add a number and a string?), we can consider this properly type checked or not. I think the easiest way to understand this is just to see some code implementing a very simple system, but don't worry, we will explain each part bit by bit.

```typescript
export type Type =
  | { kind: 'number' }
  | { kind: 'string' }
  | { kind: 'function', arg: Type, returnType: Type }

export type Expr =
  | { kind: 'number', value: number }
  | { kind: 'string', value: string }
  | { kind: 'varLookup', name: string }
  | { kind: 'function', param: string, body: Expr }
  | { kind: 'call', fn: Expr, arg: Expr }
  | { kind: 'let', name: string, value: Expr, type?: Type }
  | { kind: 'block', statements: Expr[], return: Expr }

export type Context = Map<string, Type>

export function synth(ctx: Context, expr: Expr): Type {
  switch (expr.kind) {
    case 'number':
      return { kind: 'number' }
    
    case 'string':
      return { kind: 'string' }

    case 'varLookup':
      const type = ctx.get(expr.name)
      if (!type) throw new Error(`Unbound variable: ${expr.name}`)
      return type
    
    case 'call':
      const fnType = synth(ctx, expr.fn)
      if (fnType.kind !== 'function') {
        throw new Error('Cannot call non-function')
      }
      check(ctx, expr.arg, fnType.arg)
      return fnType.returnType
    
    case 'function':
      throw new Error('Cannot synthesize type for function without annotation')

    case 'let':
      const valueType = synth(ctx, expr.value)
      if (expr.type) {
        if (!typesEqual(valueType, expr.type)) {
          throw new Error(`Type mismatch in let binding: expected ${JSON.stringify(expr.type)}, got ${JSON.stringify(valueType)}`)
        }
      }
      ctx.set(expr.name, valueType)
      return valueType

    case 'block':
      let blockCtx = new Map(ctx)
      for (const stmt of expr.statements) {
        synth(blockCtx, stmt)
      }
      return synth(blockCtx, expr.return)
  }
}

export function check(ctx: Context, expr: Expr, expected: Type): void {
  switch (expr.kind) {
    case 'function':
      if (expected.kind !== 'function') {
        throw new Error('Function must have function type')
      }
      const newCtx = new Map(ctx)
      newCtx.set(expr.param, expected.arg)
      check(newCtx, expr.body, expected.returnType)
      break

    case 'block':
      let blockCtx = new Map(ctx)
      for (const stmt of expr.statements) {
        synth(blockCtx, stmt)
      }
      check(blockCtx, expr.return, expected)
      break

    default:
      const actual = synth(ctx, expr)
      if (!typesEqual(actual, expected)) {
        throw new Error(`Type mismatch: expected ${expected}, got ${actual}`)
      }
  }
}

export function typesEqual(a: Type, b: Type): boolean {
  if (a.kind !== b.kind) return false
  if (a.kind === 'function' && b.kind === 'function') {
    return typesEqual(a.arg, b.arg) && typesEqual(a.returnType, b.returnType)
  }
  return true
}
```

Here we have, in ~100 lines, a fully functional type checker for a small language. Is it without flaw? Is it feature complete? Not at all. In a real type checker, you might not want to know only if something typechecks, but you might want to decorate the various parts with their type; we don't do that here. We don't do a lot of things. But I've found that this tiny bit of code is enough to start extending to much larger, more complicated code examples.

## Explanation

If you aren't super familiar with the implementation of programming languages, some of this code might strike you as a bit odd, so let me very quickly walk through the implementation. First, we have our data structures for representing our code:

```typescript
export type Type =
  | { kind: 'number' }
  | { kind: 'string' }
  | { kind: 'function', arg: Type, returnType: Type }

export type Expr =
  | { kind: 'number', value: number }
  | { kind: 'string', value: string }
  | { kind: 'varLookup', name: string }
  | { kind: 'function', param: string, body: Expr }
  | { kind: 'call', fn: Expr, arg: Expr }
  | { kind: 'let', name: string, value: Expr, type?: Type }
  | { kind: 'block', statements: Expr[], return: Expr }
```

Using this data structure, we can write code in a way that is much easier to work with than the actual string that we use to represent code. This kind of structure is called an "abstract syntax tree". For example

```typescript
// double(5)
{
  kind: 'call',
  fn: { kind: 'varLookup', name: 'double' },
  arg: { kind: 'number', value: 5 }
}
```

This structure makes it easy to walk through our program and check things bit by bit.

### Context

```typescript
export type Context = Map<string, Type>
```

This simple line of code is the key to how all variables, all functions, etc, work. When we enter a function or a block, we make a new Map that will let us hold the local variables and their types. We pass this map around and know we know the types of things that came before it. If we wanted to let you define functions out of order, we'd simply need to do two passes over the tree. The first to gather up the top-level functions, and the next to type-check the whole program. (This code gets more complicated with nested function definitions, but we'll ignore that here)

### Synthesis

Each little bit of synth may seem a bit trivial. So, to explain it, let's add a new feature, addition.

```typescript
// add this into our Expr type
| { kind: 'add', left: Expr, right: Expr }
```

Now we have something just a bit more complicated, so how would we write our synth for this? Let's start by assuming our add can only add numbers together.

```typescript
case 'add':
  const leftType = check(ctx, expr.left, {kind: "number"})
  const rightType = check(ctx, expr.right, {kind: "number"})
  return {kind: "number"};
```

This may seem a bit magical. How does check make this just work? Imagine that we have the following expression:
```typescript
// 2 + 3 + 4
 {
    kind: 'add',
    left: {
      kind: 'add',
      left: { kind: 'number', value: 2 },
      right: { kind: 'number', value: 3 }
    },
    right: { kind: 'number', value: 4 }
  }
```

 There is no special handling in check for add in check, so we end up at 

```typescript
default:
  const actual = synth(ctx, expr)
  if (!typesEqual(actual, expected)) {
    throw new Error(`Type mismatch: expected ${expected}, got ${actual}`)
  }
```

If you trace out the recursion (once you get used to recursion, you don't actually need to do this, but I've found it helps people who aren't used to it), we get something like

```typescript
 synth(2 + 3 + 4)
    check(2 + 3, number)
      synth(2 + 3)
        check(2, number)
          synth(2) → number
        check(3, number)
          synth(3) → number
    check(4, number)
      synth(4) → number
```



So now for our first left, we will recurse back to synth, then to check, and finally bottom out in some simple thing we know how to synth. This is the beauty of our bidirectional checker. We can interleave these synth and check calls at will!

How would we change our add to work with strings? Or coerce between number and string? I leave that as an exercise to the reader. It only takes just a little bit more code.

## Making it Feel Real

I know for a lot of people this might all seem a bit abstract. So here is a very quick, simple proof of concept that uses this same strategy above for a subset of TypeScript syntax (it does not try to recreate the TypeScript semantics for types).

