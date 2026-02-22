# A Common Mistake: Confusing Dynamic and Static Code Structure

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

## Real World Programs

Real world programs are orders of magnitude harder to understand than fibonacci. But when we are seeking to understand them, we make the same mistake. Instead of focusing on what the code does at runtime, we focus on how the program is defined statically. Consider the [GitHub Next Repo Visualizer](https://githubnext.com/projects/repo-visualization/) it make some cool pictures. It doesn't actually help us work on a code base. Now perhaps a higher fidelity static represenation would help more. But static visualizations are making a major mistake, they are visualizing what the codebase is rather than what it does.

When trying to understand a code base, my goal is not to understand the code, it is to understand the program. What problem is this program trying to solve? What data does it work with? How does it run? What other systems is it talking to? Where does it allocate? How many events does it process? How do various queues and caches grow?

### Advice on Custom Visualizations

The number one advice I'd give if you want to understand a programs behavior is to start by thinking about time. Programs unfold in time. This is the fundamental aspect of the dynamic behavior of a system that isn't captured in its static representation. (My [cohost](https://feelingof.com/episodes/) might say we just need [better programming systems](https://ivanish.ca/hest/podcast/)). There are of course times where you will want to ignore time. But if you don't capture it at the beginning, you will miss something crucial about your program, the way in which it changes and grows is incredibly valuable.

