# Variants and Protocols

In our last post we explored variants and the way in which they allow us to express choice. We saw that variants are much more powerful than enums because they allow you to pass values. Pattern matching on variants allows code to be explicit yet concise. However, there is still a way to make variants even more power, the ability to write functions that apply to multiple types of variants.

```haskell
data Maybe = Nothing | Something(thing)
```

```javascript
fn map {
  (f, Something(x)) => Something(f(x))
  (f, Nothing) => Nothing
}
```

Above is our definition of the Maybe variant and its associated function, map. Just as map for a list applies a function to every element in the list, map for Maybe applies the function if there is an element. But if we defined map as we do above, it would conflict with the definition of List because they have the same name. We could just move each map definition into a module, but then we lose some of the benefit behind map. To see what that is, let's explore some different structures that work with map.

```haskell
data Either = Error(e) | Success(x)
```

```javascript
fn map {
  (f, Error(e)) => Error(e)
  (f, Success(x)) => Success(f(x))
}
```

This is our first example, the Either Variant. Either allows us to have a value that is either a success or an error. If we have a success, then we want to apply our function to the successful value. If we have an error, applying a function wouldn't make much sense.

```Haskell
data List = Nil | Cons(x, tail)
```

```javascript
fn map {
  (f, Nil) => Nil
  (f, (Cons(x, tail)) => Cons(f(x), map(f, tail))
}
```

Here is actually the map you are probably most familiar with, mapping over a list. And yet I'm sure this definition is new for many of you. List here is our own custom list instead of the Array from javascript. This list is a linked list, and sticking with the naming convention used for a long time, each link is built up using a constructor called `Cons`. In order to map over our list, we apply it to the first element and recurse over the rest of the list.

```haskell
data Tree = Leaf(x) | Node(left, x, right)
```

```javascript
fn map {
    (f, Leaf(x)) => Leaf(f(x))
    (f, Node(left, x, right)) => Node(map(f, left), f(x), map(f, right))
}
```

Here we have a representation of a Tree. Mapping over a tree acts almost exactly like lists, a function is applied to every element, but with trees the structure is branched, so recursion needs to happen on both sides.

```haskell
data Identity = Id(x)
```

```Javascript
fn map {
  (f, Id(x)) => Id(f(x))
}
```

This is the Identity variant. It has a completely trivial map function. We take out the x and apply f to it and then wrap it back up. This may seem pointless (there are uses), but it does show yet another use of map.

## Unifying map

Now that we've seen just some of the instances of how we could use map, it seems clear that just point this in separate modules will lead to ugly code. We will have to refer to map using fully qualified names (e.g. Maybe.map, Either.map), this makes our code verbose, but also limits its reusability. As far as map is concerned, we shouldn't care if we have Maybe, Either, or Identity, as long as we have an implementation of map. In other words, we want map to be a polymorphic function.

Protocols allow us to do exactly that, write functions which are polymorphic over a given datatype. When we pass a datatype to a function implemented as a protocol, it finds its type and dispatches to the proper function. Let's look at the Mapper protocol.

```javascript
protocol Mapper {
  fn map(x, f) {
    """maps f over x"""
  }
}

implement Mapper(Maybe) {
  fn map {
    (Something(x), f) => Something(f(x))
  	(Nothing, f) => Nothing
  }
}

const map = (f, x) => Mapper.map(x, f);
```

Here is our first protocol, the Mapper protocol. Mapper is simple, in order to implement the Mapper protocol, you need to define map. One thing to note, however, is that our definition in the protocol does differ from our definitions before by one small detail, the arguments are flipped. Protocols require the type they are going to dispatch on to be the first argument. That is why we define a simple auxiliary function that flips them back around.

Now that we've made our protocol and defined an implementation for Maybe, we can use it on any maybe values.

```javascript
map(x => x + 2, Something(2)) // Something(4)
map(x => x + 2, Nothing) // Nothing
```

What is special about this version of map is that as long as we define a Mapper implementation for a variant, we can pass a value of that variant to the map function and it will work.

```javascript
implement Mapper(Either) {
  fn map {
    (Error(e), f) => Error(e)
  	(Success(x), f) => Success(f(x))
  }
}

map(x => x + 2, Error("error")) // Error("error")
map(x => x + 2, Success(2)) // Success(4)
```

We can see that our map does the right thing when passed a Maybe or an Either. This is a feature with no direct counter-part in javascript. Protocols allow us to extend functionality to new datatypes, they allow us to build common interfaces with which we can interact, and they allow of this without a nested class hierarchy or any sort of monkey patching. Protocols offer a clean way to extend functionality through out our programs. They give us a way to add new capabilities to a library as well as to use old functions in new ways.

## Conclusion

We have only seen a tiny glimpse into what protocols can do for us. The real power behind protocols comes when we group multiple fns together in a protocol and then build new functions that depend on that protocol. That may sound a bit abstract, but in our next post, we will dive in and implement a lodash like library that works on both built-in javascript datastructures and ImmutableJs datastructures all powered by protocols.