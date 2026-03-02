# Confusing Dynamic and Static Code Structure

I recently published a blog post on how I [learn a new code base](/learn-codebase-visualizer) by building a visualizer. I was super happy with the response it got. I got few emails about related projects and had relatively positive hacker news comments. But In both of these cases I saw a confusion I've seen a number of times in my career. A confusion that causes people to fail to understand a code base, that causes countless bugs. A confusion that on it's surface feels so obvious yet seems to happen to the best of us, the confusion of the static with the dynamic.

Sadly our debates about typing have confused us on what the words static and dynamic actually mean. I am not at all referring to types here. A statically typed language still has dynamic behavior. The dynamic behavior of a program is what it does when it is run.  Consider recursive fibonacci:

```typescript
function fib(n: number): number {
  if (n <= 1) {
    return n;
  }
  return fib(n - 1) + fib(n - 2);
}
```

The source itself is rather simple, but the runtime behavior is something that beginners often trip up on. It isn't at all obvious at first glance that this forms a tree. Nor is it obvious how your programs runtime will concretely run this program. There is no mention of a stack here, in fact, the stack isn't even necessary to understand this program. Dynamically however, it is important to understand if you want to know how your progam truly runs.

## An Obsession with Static Stucture



