# What Does the React Compiler Even Do?

This article is mainly for me. I want to understand more about the react compiler. To explore some of its edges. To see how it currently optimizes code and how it could be extended in the future. But I've found time and time again the way I best learn is by explaining things to others. So that's what I'll set out to do. But who exactly are these others? Who am I teaching this to.

Honestly, this question is one of the things that stops me from writing blog posts. Especially those with deeper technical content. The two blog posts that took me longer than any others were my [Machine Code Isn't Scary](https://jimmyhmiller.com/machine-code-isnt-scary) and [The Easiest Way to Build a Type Checker](https://jimmyhmiller.com/easiest-way-to-build-type-checker). I didn't need to do any research on those topics. What I instead had to do was draft and draft, thought after thought, edit after edit where I finally accomplished a few things. 1) I didn't add too many caveats 2) I was clear enough 3) I did not overwhelm with information.

I give all this preamble, because since this post is mainly for me, I might fail at exactly those 3 points. If you good things about the react internals, you will find blogs posts that are much more of an introduction into the internals. Blog posts that don't assume you know things about ASTs blog posts that give proper motivation. What you will find here is instead things I find interesting.

## The Output

I have never found starting at main to be helpful. I've never wanted to trace a program through all its steps and understand it that way. In fact, I have a problem in how I understand things in general that causes people to be a bit confused with me. For example, when I read books, I never know character names. I can read whole books, whole series, tell you all about what I read, what the characters did, but not tell you their names. I just don't pronounce them in my head. Even before the AI era, I could never tell you where something is in the file structure of a codebase. I could never remember the names of structs or classes. I could functionally work through the code. I had a good mental grasp of where to make changes, what to grep for. But it was never put in words I could easily communicate to others.

So if I'm going to learn a codebase it is always by exploring the output. Not the internals. The internals will come as I need to solve a problem until I see the output, I don't know what the problem might be.

So let's start with our easiest input that will show us some meaningful output:

```jsx
export function Thing() {
  return <button onClick={() => console.log("click") } />
}

function clickHandler() {
  console.log("other click")
}
```

turns into

```jsx
import { c as _c } from "react/compiler-runtime";
export function Thing() {
  const $ = _c(1);
  let t0;
  if ($[0] === Symbol.for("react.memo_cache_sentinel")) {
    t0 = <button onClick={_temp} />;
    $[0] = t0;
  } else {
    t0 = $[0];
  }
  return t0;
}
function _temp() {
  return console.log("click");
}
```

_c is a function that let's us a get cache slots for our component. So in this case, we jsut need one slot. The other thing to notice is the "outlining" (opposite of inlining) of the function, turning it from an inline handler to a top level function. `Symbol.for("react.memo_cache_sentinel"))` is basically the zero value of the cache. This first if just asks if the cache slot is empty. If it is, we are going to construction our jsx object (i.e. call the jsx function) get its return value and save it. otherwise, return what is in the cache.

If you are used to react code, this can feel a bit like overkill. It would be like writing memo for everything.

```jsx
import { useMemo } from "react";

export function Thing() {
  const memod = useMemo(() => <button />, []);
  return memod;
}
```

But as silly as this may seem there is nothing wrong with it. And if you are a compiler, why not write the most optimal thing

## Limitations of Hooks the Compiler Gets Around

```jsx
function Thing({ thing, stuff }) {
  if (thing === "thing") {
    return "my thing";
  }
  return <button>{stuff}</button>
}

```

turns into

```jsx
import { c as _c } from "react/compiler-runtime";
export function Thing(t0) {
  const $ = _c(2);
  const { thing, stuff } = t0;
  if (thing === "thing") {
    return "my thing";
  }
  let t1;
  if ($[0] !== stuff) {
    t1 = <button>{stuff}</button>;
    $[0] = stuff;
    $[1] = t1;
  } else {
    t1 = $[1];
  }
  return t1;
}
```

The [rules of hooks](https://react.dev/reference/rules/rules-of-hooks) give us no clear solution for a use memo ehere. We can't put it after the first if, because this fires conditionally. If we put it before the if, we are making something that is unnecessary for the early return case. But this much lower level api for react gives more flexibility in getting these exact optimizations. It's always interesting to see the kinds of code that are more optimal and the tradeoffs library authors make in what they choose to expose, and allow in order to try to steer people towards correct code.

## Mutate Together

One final interesting output is its ability to group mutations together. 

```jsx
function Toolbar({ user, canEdit }) {
  const actions = getActions(user); // Returns a fresh array

  if (canEdit) {
    actions.push("Edit");
  }

  return <Menu actions={actions} />;
}
```

turns into

```jsx
import { c as _c } from "react/compiler-runtime";
function Toolbar(t0) {
  const $ = _c(5);
  const { user, canEdit } = t0;
  let actions;
  if ($[0] !== canEdit || $[1] !== user) {
    actions = getActions(user);

    if (canEdit) {
      actions.push("Edit");
    }
    $[0] = canEdit;
    $[1] = user;
    $[2] = actions;
  } else {
    actions = $[2];
  }
  let t1;
  if ($[3] !== actions) {
    t1 = <Menu actions={actions} />;
    $[3] = actions;
    $[4] = t1;
  } else {
    t1 = $[4];
  }
  return t1;
}

```

Here you can see that our little cach block keeps the actions.push inside the same cached case. If you change that to actions2.push, it is no longer looking at the same mutation scope and will move outside the if. 

## How does it do all of this?

