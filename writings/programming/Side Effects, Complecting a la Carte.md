#Side Effects, Complecting a la Carte

There are many concepts we learn as we continue our adventure in programming languages. Objects, polymorphism, inheritance, and a whole host of others. Often these concepts are taught with the goal of making our code simpler, yet more often than not, they makes our code more complex, in other words, they [complect](beautiful-code.html) them. In these post we dive into where most of the inessential complexity of our programs come from, side effects.

##Programming Functions are Algebraic Functions

A function in a programming language is the exact same thing as the function you learned about in mathematics.

```Javascript
f(x) = 2x
const f = (x) => x*2; 
```

The first example above is a mathematical function. The second is the exact same function in javascript. Functions serve one purpose, to take an input, apply a transformation, and return the new value. When we start thinking about functions in programming as mathematical functions we can smuggle mathematical concepts into our programming.

	(g ∘ f)(x) = g(f(x))

Above is the mathematical definition of function composition. It states that composing g and f is the same thing as first applying f to some value, then apply g to the return of f. A simple implementation in javascript would be the following:

```javascript
var compose = (g, f) => (x) => g(f(x))
```

This code says that compose is a function that takes two functions f and g. Compose returns a function that takes an x, and applies f  to x, and then g to the result of f applied to x.

###What mathematics gains us

Our compose function has a limitation, it can only compose two functions, but it seems like we should be able to chain compositions together to get more than two things composed as follows:

```javascript
compose(h, compose(g,f));
```

This seems correct enough but couldn't we also do the following?

```javascript
compose(compose(h,g), f);
```

Are these different? How can we tell? Well, we could try a few examples.

```javascript
compose(add2, compose(add4,add3))(2) 
// 11
compose(compose(add2, add4), add3)(2)
// 11
```

Well, it seems to hold in the case we thought of, but how do we know if it holds in general? This is what mathematics gives us. Since our functions are exactly the same as mathematical functions the laws which apply to mathematical functions apply to programming functions. In this case the pertinent law is associativity.

	 f ∘ (g ∘ h) = (f ∘ g) ∘ h

Associativity tells us that how we group our composition makes no difference to the result. Given this knowledge extending our composition function to and indefinite number of arguments is actually quite easy.

```javascript
const compose2 = (g, f) => (x) => g(f(x));
const compose = (...fns) => fns.reduce(compose2);
```

First we renamed our old compose function. We use an es2015 feature here, rest parameter syntax. It just says take the rest of the arguments and pack them into an array for us.  Now we can simply compose as follows:

```javascript
compose(add2, add4, add3)(2);
```

##When Functions Aren't

Functions when they are mathematical objects give us ways to reason about our code. Even if we don't bust out the associative law all the time we do get some guarantees from functions. For instance:

```javascript
f(x) === f(x)
```

Seems rather obvious, but it is very easy to make this not hold. In other words, sometimes functions can cease to be functions.

```javascript
var enabled = true;
var f = function (x) {
	if (enabled) {
		enabled = false;
		return x+2;
	}
	else {
		return x;
	}
}
f(2) === f(2)
// false
```

Here "functions" in javascript do something that algebraic functions certainly can't do, side effects. Side effects occur when our function accesses something outside itself. This reaching out of itself often involves changing the value of some variable, but can involve causing an effect, such as printing to the console or making an http request.

###But They Are Still Functions

Calling these side-effecting functions not "functions" is not great terminologically. Javascript is going to call them functions whether they have side effects or not. So while it may be true in the mathematics sense, it doesn't hold in javascript land. So instead let's refer to functions in the algebraic sense as *pure functions* and side effecting functions as *impure functions*.

Pure functions are the perfect example of decomplected code. Pure functions do one thing, transform input. Impure functions are complecting a la carte. Inside an impure function we can change anything, we can return different results depending on the time of the day, in fact, we can return nothing. These side effects weave together the notion of an effect with the computation that needs to be done.

###But My Programs Need to do Something

Side-effects are of course necessary. Our programs must communicate with the outside world in some way. But unconstrained side-effects can incredibly complicate our code. Mixing our data transformations with side-effects causes to lose the ability to reason about our code accurately. Mutating variables causes us to have to keep track of more and more state as variables change. In fact, it is my conjecture that a large majority of hard to trace bugs come from some variable somewhere being changed to some value we did not expect.

##Conclusion
So how do we constrain side-effects? That deserves a whole post or three on that very topic but for now we can keep it simple. First, always strive to write pure functions. Secondly, avoid mutation if at all possible. Finally move all side effects to the edges of your programs. Future posts will explore these topics showing how constraining our side-effects eliminates whole classes of bugs commonly encountered in programming and leads to simpler code.






