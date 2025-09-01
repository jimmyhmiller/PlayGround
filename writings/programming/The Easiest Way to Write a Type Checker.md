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
function check(x, type) {
  if infer(x) != type {
    throw `${x} is not an ${type}`
  }
}
// for + both sides must be int
let lhs = check(2, "int")
let rhs = check(3, "int")
```

For these trivial cases, this may seem like almost nothing. But using these simple mechanisms, we can introduce a rich type system. But we do make a tradeoff, unlike Hindley Milner, we do not get complete type inference. Instead, we need to annotate functions with their types. How far you can you take type inference with a bidirectional system while keeping the deterministic and clear benefits it offers? That's unclear to me.

## A Real Implementation



[^1]: I was however luckily enough to find [David Christiansen's presentation](https://www.youtube.com/watch?v=utyBNDj7s2w) where he translates a bidirectional type system into code. It wasn't fully complete, but enough for me to go on. The audio has a terrible high pitch squeal, so can't recommend it as a nice listen.

