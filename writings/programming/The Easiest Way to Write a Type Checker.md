# The Easiest Way to Write a Type Checker

As with all things in the compiler literature, there are countless resources that teach you how to write a type checker, if you already know how to write one. I'm not saying it's impossible to understand statements like this:

![Screenshot 2025-08-30 at 5.02.36 PM](/Users/jimmyhmiller/Desktop/Screenshot 2025-08-30 at 5.02.36 PM.png)

But personally, I never find these presentations enlightening[^1]. It turns out though, that type checking, just like most things in compilers, can be understood more directly without all the unnecessary complication. 

## The Key Insight

We are going to be implementing a bidirectional type checker. A bidirectional type checker isn't a particular algorithm like `algorithm w` for Hindley Milner. Instead if is a general way of looking at implementing a type checker that quite naturally expands from the simple kind of type system we will implement all the way to dependent types.

The "two directions" of the bidirectional type checker are `check` and `infer`. Check does exactly what it says on the tin, we are given a type and we are going to check if it is right. We use inference, when we don't know the type. This inference isn't guessing, it doesn't backtrace, it is looking at the expression we have and figuring out the type. These two sides of the problem, play together to create our type checker.

### A quick example

The goal if this article will be to create a simple, but non-trivial type checker. But before we get to the details of the code, let's try to get the intuition for how our method is going to work on a small scale. Consider the simple expression `2 + 3` and let's tackle it from both angles. If we don't know the type of this expression, we can try to infer it.

```javascript
// inferring type of 2 + 3
function infer(x) {
  if (isNumber(x)) { return "int" }
  ...
}

let lhs = infer(2)
let rhs = infer(3)
if (lhs == "int" && rhs == "int") {
  return int
} else {
  throw new Error("lhs and rhs must be int")
}
```

Check is even easier.

```javascript
// checking type of 2 + 3
function check(x, type) {
  if infer(x) != type {
    throw `${x} is not an ${type}`
  }
}
// for + both sides must be int
let lhs = check(2, "int")
let rhs = check(3, "int")
```

For these trivial cases, this may seem like almost nothing. But using these simple mechanisms, we can introduce a rich type system. But we do make a tradeoff, unlike Hindley Milner, we do not get complete type inference. Instead, we need to annotate functions with their types. How far you can you take type inference with a bidirectional system while keeping the deterministic and clear benefits it offers? That's unclear to me.[^2]

## A Real Implementation

In this tutorial we are going to aim to get a very simple example working. It's an example that falls short of a full implementation, but that hopefully gives you enough to make it clear how you handle additions to it.

```typescript
function log(message: string) {
  // does some logging
}

function calculateShipping(weight: number, distance: number): number {
  let baseRate = 5.00;
  let weightFee = weight * 0.50;
  let distanceFee = distance * 0.25;
  let freeShippingThreshold = 75.0;
  
  let total = baseRate + weightFee + distanceFee;
  
  if (total > freeShippingThreshold) {
    log("Free shipping applied! Saved: $" + total)
    return 0.0;
  }
  
  return total;
}
```

You will probably recognize this as typescript. We will of course not be implementing typescript's type system. Doing so is [no small feat](https://github.com/microsoft/TypeScript/blob/main/src/compiler/checker.ts). Instead we are are just going to use typescripts syntax so we don't have to write our own parse and so people who don't like parenthesis aren't annoyed.

### Checking simple types

In order to check our types we need a way to represent them.

```typescript
type PrimitiveType = { 
  kind: 'primitive';
  name: 'number' | 'string' | 'boolean'
};
type Type = PrimitiveType;

const NUMBER_TYPE: Type = { kind: 'primitive', name: 'number' };
const STRING_TYPE: Type = { kind: 'primitive', name: 'string' };
const BOOLEAN_TYPE: Type = { kind: 'primitive', name: 'boolean' };

const typesEqual = (a: Type, b: Type): boolean => {
  if (a.kind !== b.kind) {
    return false;
  }
  if (a.kind === 'primitive' && b.kind === 'primitive'){
    return a.name === b.name;
  }
  return false;
};
```

We will start with numbers, strings, and booleans, but we know in the future we are going to have functions as well. So we have intentionally made a system that will make it easy to expand the future (hence the currently pointless `Type = PrimitiveType`). Now that we have some of our types defined we need two things, 1) A way to actually parse some code, 2) a way to check the types of it. Let's start top down here and actually implement our main function that will parse our code and check it.

```typescript
// a top of file
import * as ts from 'typescript';

function parseAndCheckSimpleProgram(code: string, type: Type) {
  const sourceFile = ts.createSourceFile('temp.ts', code, ts.ScriptTarget.Latest, true);
  checkSimpleProgram(sourceFile, type);
}

function main() {
  parseAndCheck("2", NUMBER_TYPE);
  parseAndCheck("true", BOOLEAN_TYPE);
  parseAndCheck("'hello'", STRING_TYPE);
}
```

Here we are using typescripts api to parse our code. If you aren't familiar with parsing, all you need to know is that this going to convert our code into a big tree called an [abstract syntax tree](https://astexplorer.net/#/gist/92c7e41b058764c983b97502bd8c29e2/9146943079dcecfd6fb16827d0b53001c7b85ff2), think of it like a big json object that represents our code.

Now we just need to walk this tree and check that our types match. At this point, we are make some simple assumptions. We are just going to have a single statement that is an expression statement.

```typescript
function checkSimpleProgram(sourceFile: ts.SourceFile, expectedType: Type): void {
  const statement = sourceFile.statements[0];
  
  if (!ts.isExpressionStatement(statement)) {
    throw new Error("Only expression statements supported");
  }
  
  check(statement.expression, expectedType);
}
```







[^1]: I was however luckily enough to find [David Christiansen's presentation](https://www.youtube.com/watch?v=utyBNDj7s2w) where he translates a bidirectional type system into code. It wasn't fully complete, but enough for me to go on. The audio has a terrible high pitch squeal, so can't recommend it as a nice listen.

[^2]: I'm not exactly sure how to even phrase this question. Of course, the code inside a bidirectional type checker is turing complete, so it could do whatever it wanted to infer a type. But maybe there is some formalism we could keep to? I know in David Christiansen's paper he says this "Sometimes, explicit type annotations will need to be within a term rather
than at the top level. In particular, explicit function abstractions that are
being applied directly may require a type annotation....We can solve this by providing specialized inference rules for
certain limited forms of λ-abstractions and conditionals, and we may even be
successful in the simply-typed context. While this is useful, it will not scale
to more interesting type systems for which type inference is undecidable. It
may, however, be possible to make a quite useful system in practice, where
users only need to annotate complicated code."

