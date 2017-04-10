# Multi-Methods

Functions, as we've seen earlier posts, are some of the basic building blocks upon which we define our programs. Functions allow us to build our applications out of small reusable parts, they allows to reason carefully about our programs, they document and explain our code, and yet functions are not without limitations. Functions fall down primary in the area of extensibility.

Functions are not extensible. Once you have defined a function, it is set in stone. The type of arguments it can take and the types it can return are entirely determined by the code you have placed inside it, but often times we find ourselves wanting to add new behavior to a function without having to modify it. Perhaps a new situation came up which we could not have predicted. Having to change an existing function can be annoying, or worse, if the function lies in someone else code, change a function can be impossible.

We saw a technique in our [previous post](protocols.html) that allows extensibility of functions based on type. Protocols allow us to define grouped operations that apply to a particular type, allowing us to implement a whole host of functions while abstracting out the particulars.  But protocols are intentionally limited, dispatching based on a type means that we often can't extend the functionality of our functions in ways we might want. Protocols are optimzed for use with abstract datatypes instead of just generic data. When we find protocols don't give us the necessary flexiblity, we can instead reach for multi-methods.

## Extensible Functions

Here we enter into territory not explored by javascript yet again. So as we did in our previous exploration of protocols, we will begin with a fictional language which has first class support for multi-methods and then a follow-up post, we will see a library encoding of multi-methods.

```javascript
multi area {
  (s) => s.shape
}

method area('circle') {
  (s) => Math.PI * Math.pow(s.radius, 2)
}

method area('square') {
  (s) => Math.pow(s.side, 2)
}

area({shape: 'circle', radius: 1})
// 3.141592653589793
area({shape: 'square', side: 1})
// 1
```

Here is essentially the `hello world` of multi-methods. We want to be able to find the area of a shape, but as you know, each shape needs its own are function. So rather than trying to figure out every shape ourselves, we want our function to be extensible so that other users can add their own shapes later. Our shapes are just an object that has a shape property. So, we start with our multi-methods dispatch property. In order to determine what function we need to call, we just get its shape property. Then we implement our function for different values of the shape property. First we do it for circle and then next for square. As we can see, every object can have completely different properties, but if they define a shape property, we can compute their area.

Imagine what we would do without multiple-methods. Any time we wanted to add a new shape, we would have to modify our existing function. And if we packaged this up in a library, we would have users asking for their new favorite shape to be added. They would be powerless to add new functionality to our library themselves.

### Practical uses

Multi-methods can be incredibly useful and practical in many different situations, but their is one in particular I found interesting given its popularity. If you've been in the React community, you may have heard of Redux. Redux is a state management library. It abstracts away state management from your library or framework of choice and provides a simple api for accessing state and making state transitions.

The fundamental concept around redux is that of a reducer. From the redux documentation here is a very simple reducer.

```javascript
function counter(state = 0, action) {
  switch (action.type) {
  case 'INCREMENT':
    return state + 1
  case 'DECREMENT':
    return state - 1
  default:
    return state
  }
}
```

This reducer tells us how to transition our counter given an `INCREMENT` action, a `DECREMENT` action, any other action, or no action at all (our initial state). Let's see how we could implement the exact same reducer using a multi-method.

```javascript
multi counter {
  (_, action) => action.type
}
method counter('INCREMENT') {
  (state, _) => state + 1
}
method counter('DECREMENT') {
  (state, _) => state - 1
}
method counter(_) {
  (state = 0, _) => state
}
```

Our multi-method behaves exactly the same way that our vanilla javascript reducer does. It is a put more verbose, but that is mainly due to the syntax we chose. But our multi-method has a property our vanilla function does not have, namely extensiblity. Imagine if someone wanted to add a `RESET` action to our reducer. In order to do this in without multi-methods, we would need to modify the original code, but now we can simple write the following.

```javascript
method counter('RESET') {
  (_, _) => 0
}
```

