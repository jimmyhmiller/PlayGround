# Protomorphism

The last two posts ventured off into the hypothetical world. In that world we had a language very much like javascript but with variants and protocols. Unfortunately, that language isn't real, but that doesn't mean we can't sneak some of those ideas into our javascript. Today we are going to explore protocols further using a library called `Protomorphism` that adds protocols into javascript. 

```javascript
import protocol from 'protomorphism'

const Mapper = protocol({
  map: (x, f) => {
   // maps f over x
  }
});

Mapper.implementation(Array, {
  map: (arr, f) => arr.map(f)
})

const map = (f, x) => Mapper.map(x, f);

map(x => x + 2, [1,2,3]) // [3,4,5]
```

Here we can see protomorphism in action with our Mapper protocol from last post. This actually isn't too different from the code we would write in our imaginary language. What we see here is basically all of protomorphism, it is a simple library that does one thing. In fact, it is only 31 lines of code. But fewer lines doesn't mean less powerful. As promised in our last post, we are going to create our own lodash like library, but our library, using the power of protocols, will work whether we use normal Javascript Arrays, ImmutableJs Lists, or any type that implements our protocol.

## Sequence Protocol

```javascript
const Sequence = protocol({
  cons: (coll, elem) => {
    // prepends elem to coll
  },
  first: (coll) => {
    // gets first element of coll
  },
  rest: (coll) => {
    // returns all but first element
  },
  isEmpty: (coll) => {
    // returns true if empty
  },
  empty: (coll) => {
    // given a coll, it will return 
    // an empty collection of the same type
  }
})
```

Here is our Sequence protocol from which we will build all our lodash like functions. It is a simple protocol, with only five methods, each of which are fairly straight forward. Using these we can start building up more and more useful functions. Let's start off with some very simple ones.

### Examples

```javascript
const cons = Sequence.cons
const first = Sequence.first
const rest = Sequence.rest
const isEmpty = Sequence.isEmpty
const empty = Sequence.empty

const second = (coll) => first(rest(coll))
const ffirst = (coll) => first(first(coll))

const last = (coll) => {
  if (isEmpty(coll)) {
    return undefined
  } else (isEmpty(rest(coll))) {
	  return first(coll)
  } else {
    return last(rest(coll))
  }
}
```

We start off with simple aliases to our Sequence functions we need to use. This is purely for convenience sake and not necessary. Next we implement two very simple functions, `second` and` ffirst`. Second does what it says, it gives of the second element of a collection; ffirst gives us the first of the first element of the collection. Below should illustrate the difference clearly.

```javascript
const coll = [[1], [2], [3]]
second(coll) // [2]
ffirst(coll) // 1
```

The `last` function is a little more involved, but if you are familiar with recursion it is very simple. If we are passed an empty collection, there is no last, so we return undefined. If we are passed a collection with only one thing in it, we return that thing. Otherwise, we take one item off the collection and find the last of that collection.

One thing to note is that these functions are perfectly comprehensible and sensible and yet we have not mentioned at all what datastructure these functions are for. As far as our code is concerned, it doesn't matter if this is an array, an immutable list, or any other type. All that matters for the functions above is that the data structure implements the Sequence protocol.

## Implementations

```javascript
Sequence.implementation(Immutable.List, {
    cons: (coll, elem) => coll.unshift(elem),
    empty: (coll) => Immutable.List.of(),
    first: (coll) => coll.first(),
    rest: (coll) => coll.rest(),
    isEmpty: (coll) => coll.isEmpty(),
});
```

Above is our implementation of the sequence protocol for ImmutableJs Lists. Our Sequence protocol assumes that each of our functions has no side effects, so ImmutableJs is a perfect fit here. In fact, there are methods that correspond exactly to the methods on our the Sequence protocol. Now we can use the functions we wrote on ImmutableJs Lists.

```javascript
const coll = Immutable.fromJS([[1], [2], [3]])
second(coll) // List [2]
ffirst(coll) // 1
```

This works exactly the same as the example above. In our first example, we just assumed we had an implementation of Sequence for Javascript Arrays, let's go ahead and write one now.

```javascript
Seq.implementation(Array, {
    cons: (coll, elem) => {
        coll = coll.slice() // copy
        coll.unshift(elem)
        return coll
    },
    empty: (coll) => [],
    first: (coll) => coll[0],
    rest: (coll) => {
        coll = coll.slice() // copy
        coll.shift(0)
        return coll
    },
    isEmpty: (coll) => coll.length == 0
});
```

The definition for Arrays is a tad bit uglier. This is due mainly to that fact that our protocol's methods are assumed to be side-effect free, where as Arrays methods mutate. So in order to do cons and rest, we must copy the array. Now, that we have defined the Sequence protocol for Arrays, all functions that just use protocol methods will work with Arrays.

## More Functions

```javascript
const map = (f, coll) => {
    if (isEmpty(coll)) {
        return coll;
    } else {
        return cons(f(first(coll)), map(f, rest(coll)));
    }
}

const filter = (pred, coll) => {
    if (isEmpty(coll)) {
        return coll;
    } else if (pred(first(coll))) {
        return cons(first(coll), filter(pred, rest(coll)));
    } else {
        return filter(pred, rest(coll));
    }
}

const reduce = (f, init, coll) => {
    if (isEmpty(coll)) {
        return init;
    } else {
        return f(reduce(f, init, rest(coll)), first(coll)) 
    }
}
```

Here we have the three power house lodash functions. By showing that we can implement these, it becomes easy to see how we can begin to implement all the functionality that lodash supports, but without depending on a concrete implementation. 

## Conclusion

Protocols give us the ability to reason at a higher level of abstractions. They provide us a way to extend functionality to new code that we never planned for. This level of programming allows our code be clear, yet powerful. In our next pos,t we are going to explore a similar, yet slightly different way to provide flexibility and extensibility, multi-methods.