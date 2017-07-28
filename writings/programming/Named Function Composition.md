# Named Function Composition

I've just released a little library on npm called fluent-compose. I've had some mixed feelings about my creation, I know that it will be met with some skepticism, in fact, I think had I seen this library from someone else, I would have dismissed it. And yet, I think I've stumbled onto a fairly decent idea. By decent idea, I mean a hack. But before we dive into this hack, let's look at the situation that gave rise to it.

There is this fantastic, little known library called [Zaphod](https://zaphod.surge.sh/). The idea behind zaphod is to mirror Clojure's immutable data API. This makes it incredibly simple to do immutable updates on plain javascript objects.

```javascript
import { update, inc } from 'zaphod/compat';
const state = { count: 0 };
update(state, 'count', inc) // { count : 1}
```

The way I've written the code above is actually not the default way Zaphod works. I imported the `compat` part of zaphod. By default, the functions are exposed to take advantage of the function bind operator `::`.

```javascript
import { update, inc, dec } from 'zaphod';
const state = { count: 0, otherCount: 0 };
state::update('count', inc) // { count : 1}

state
    ::update('count', inc)
    ::update('otherCount', dec)
// {count: 1, otherCount: -1}
```

This is actually some really neat functionality. It allows you to chain your operators together. We can build pipelines by continuing to bind. Unfortunately, we don't get function bind syntax for free. Function bind is still a stage 0 proposal. This means there is a very good possibility it will never make it into javascript. So there is definitely risk involved in its use. It also means, that `create-react-app` does not have support for it. 

But function bind syntax also has flaws even if it were accepted into the language. Function bind syntax abuses `this` the most misunderstood keyword in all of javascript. The functions you write with function binding in mind, must use this, they can't be normal functions. Of course, you can wrap up those functions, but if we need to wrap functions up, why not wrap them in a way that doesn't require function bind?

This is where `fluent-compose` comes in. Let's look at an example.

```javascript
import * as zaphod, { update } from 'zaphod/compat';
import { threadFirst } from 'fluent-compose';

const transform = threadFirst(zaphod);

const transformer = transform
    .update('count', inc)
    .update('otherCount', dec)
    .set('anotherCounter', 2)

transformer({count: 0, otherCount: 0, anotherCounter: 0})
// {count: 1, otherCount: -1, anotherCounter: 2}

update({ counters: {count: 0, otherCount: 0}}, 'counters', transformer)
// {counters: {count: 1, otherCount: -1, anotherCounter: 2}}
```

Here we see the `threadFirst` function in use. This allows us to take a collection of functions, in this case `zaphod`, and wrap them up into a fluent interface. But what does this fluent interface do? It is just function composition. At the end, we get a function back. We can now use this function to pass our data through the pipeline. Since, what we get back is just a function, we can also pass this function around. We can see its use on line 14 as just a normal function that lets us perform a series of transformations on data.

This is a fairly simple use of `fluent-compose`, let's take it one step further.

```javascript
import * as zaphod from 'zaphod/compat';
import * as lodashFpCollection from 'lodash/fp/collection';
import { threadFirst, threadLast, fluentCompose } from 'fluent-compose';

const zaphodTransform = threadFirst(zaphod);
const lodashTransform = threadLast(lodashFpCollection);

const transform = fluentCompose({
  ...zaphodTransform,
  ...lodashTransform,
})

const transformer = transform
  .map(x => x + 2)
  .filter(x => x % 2 === 2)
  .set(0, 3)

transformer([1,2,3,4]) // [3, 6]
```

Here we can see a combination of two totally separate libraries. In fact, I even used `lodash/fp` because rather than taking its primary argument first, it takes it last. Yet, we were still able to compose these libraries in a simple, yet flexible way. 

And yet, `fluent-compose` holds still more power. This time, we will be using some of the lower level features of `fluent-compose`, explaining them here would be beyond the scope of this posts, so I've omitted some implementations, but you can find the full source [here]().

```javascript
import { fluentCompose } from 'fluent-compose';

const baseReducer = ...
const initialState = ...
const reduce = ...

const INCREMENT = 'INCREMENT';
const DECREMENT = 'DECREMENT';

const increment = () => ({
  type: INCREMENT
})

const decrement = () => ({
  type: DECREMENT
})

const reducer = fluentCompose({ initialState, reduce }, baseReducer)
const counter = reducer
  .initialState(0)
  .reduce(INCREMENT, x => x + 1)
  .reduce(DECREMENT, x => x - 1)

console.log(
  counter(0, increment())
)
```

Using `fluent-compose` we've made a fluent reducer for redux! No longer would we need to write switch statements in order to make a reducer. In fact, since `fluent-compose` just makes functions, you can use this reducer with combine reducers. But another really cool thing you can do with it is add on the reducer after the fact. On feature to note with this implementation, is that it actually short circuits, as soon as it fines the action that matches the type, it returns, so there is no wasted computation. And while I omitted the implementation here to not distract from the main point, the implementation is just about 10 lines of code. 

## Why do I call this a hack?

I really do think this library is really useful. In fact, I'm very excited about wrapping up functions to work in this way. But at the same time, I can't help but feel a little weird about this library. In order to make this library work, I have to take advantage of the fact that functions are objects. I am making a function and then assigning methods to it. This definitely a strange thing to do. Now, I do avoid mutating the functions passed into, I "copy" them before I assign properties to them, but it still feels like the wrong means for accomplishing the task of creating a pipeline.

In fact, that is the thing that makes this library a hack; it is the wrong means. This library was created out of the limitation javascript imposes on us. How would we accompolish similar things in other languges? Here are just a couple examples.

```clojure
; Clojure
(->> (range 100)
     (map (partial + 2))
     (filter even?))
```

```haskell
-- Haskell
(|>) :: a -> (a -> b) -> b
(|>) a f = f a

range 100
|> map (+2)
|> filter even
```

Above we see how we could accomplish similar things in Haskell and Clojure. Almost all functional programming languges have a way to do this. In fact, there are some much more powerful techniques for function composition in both Haskell and Clojure.

### Still interesting

At the same time, this method has some interesting features all on its own. What we have done is allow our functions to have special ways in which they compose. Each function can determine for itself special compostion points. At each point along the way, we keep these composition properties, allowing us to compose further. Each of these composition methods have a named, hence "named function composition". 

### Future Steps

I can see this expanding in a few different ways. First, I think releasing libraries that use this technique could be useful to the community at large. These could simply be combinations of existing libraries into a nice pipeline, or something with a bit more substance to it. Secondly, I'd like to explore how far this style of function composition takes us. I've already make a quick and dirty implementation of the state monad. I think it would even be possible to make an implementation of the parse monad. This would definitely results in some fairly dense code, but might be an interesting experiment to test the limits of `fluent-compose`.

But perhaps most importantly, I need to come to a better understanding of the library myself in order to document how it works. I based the internal off redux-middleware. As I've been writing these helper methods though, I think they may be a better way. In some ways, as I'm writing them I'm reminded of transducers, maybe there is a relation there? Another thing that always comes to mind as I'm writing these is recursion schemes. Can these things form a catamorphism? Maybe once I have a better understanding of those I will know.