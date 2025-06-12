# The Beauty of Cheney's Algorithm

I'm in the early stages of building a programming language. Fortunately, I've already gotten to tackle some really interesting problems. But by far the most fun, most painful, and most satisfying work I've done has been creating some Garbage Collectors. I've actually created multiple implementations of at least 3 different garbage collection methods. I plan on making a post about all them, but I want to start with what is perhaps the most elegant of them all. This method is often called a "semi-space" collector. But for our purposes, we will call it a compacting garbage collector. 

If you are already a low level, compiler, language runtime kind of perons, this post may be entirely too long for you. If that's case, check out the much short, to the point explanation at [wingolog](https://wingolog.org/archives/2022/12/10/a-simple-semi-space-collector). If you are like me and originally had a web/high-level background, I hope you will find the over-explanations helpful.

## A Different Approach to GC

Fundamentally garbage collection is trying to give the illusion that we have unlimited memory. My general understanding of GC always assumed things worked something like this:

1. Allocate until you run out of memory
2. Scan memory and find which objects are dead
3. Reuse the space the dead objects took up

If we add in some details, what we have is called a mark and sweep algorithm for garbage collection. But there are some tradeoffs that we have to make here. FIrst, there is the idea of fragmentation. As we allocate more and more objects, the free spaces will get spread out. In fact, we can end up with free space that is too small for any object. Secondly, in order to mark all the objects that aren't alive, we need to actually walk over all the memory and mark them as dead. This can take a long time.

What a semi-space collector does instead is actually rather simple. Rather than keeping objects in place, it moves those objects. Given that we know which objects are still live, we can copy those live objects over to new memory, allocating them in a nice contiguous fashion. Once everything is moved, we can throw away the old memory. This is the key to our "compacting garbage collector". But the beauty comes from how we achieve this efficiently.

## How do you know what you've copied?

Some times in programming, its easier to start an explanation in the middle or at the end of a process. For example, I struggled to create languages for a long time, until I start by making the lowest level, code generator first. It wasn't that making a parser was super hard, it's that I didn't understand my end goal. Working backwards made it all make sense for me. I think the same is true here. There are a ton of details that go into building a garbage collector, but rather than starting from the language and working in, let's start with our end goal.

![Screenshot 2025-04-16 at 2.57.37 PM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2025-04-16 at 2.57.37 PM.png)

Here we have an simplistic view of our memory. Each of the objects I've colored green are the objects that are alive. On the right, we have an empty space we are going to copy these objects to. So once we have copied things over, it is going to look like this.

![Screenshot 2025-04-16 at 2.58.50 PM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2025-04-16 at 2.58.50 PM.png)

Here we've compacted our objects. Now, if this were all there was to our problem, it would be quite easy to solve. But unfortunately, our problem is a bit more complicated. We don't know from the beginning which objects are live. What we have are called the "roots". These roots are the live values used by the current program. (They often come from the stack, but we will talk about that more later). Each of these root objects, may reference other objects. We need to keep those alive as well. So what our memory really looks like is something like this:

![Screenshot 2025-04-16 at 3.07.54 PM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2025-04-16 at 3.10.33 PM.png)

Here rather than all the live obejcts being colored green, I've colored our roots green. What we need to do is make sure that all the objects our roots point to are get copied into our new location and all the objects they point to, and so on and so forth. If you trace the lines here, starting at the roots, you'll find that our end result after copying looks something like this:

![Screenshot 2025-04-16 at 3.13.34 PM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2025-04-16 at 3.13.34 PM.png)

<details>
 	<summary>
  Look a little too abstract, take a look at some example code.
  </summary>
  <div>
    ```javascript
    function doThingsWithTempContext() {
      let point = [1, 2]
      let tempConfig = {
        point: point,
        valid: true,
        count: 10,
        used: false,
      }
      let tempConstants = [point];
      ...
    }
    doThingsWithTempContext();
    let dimensions = [0, 1, 2, 3];
    let constants = [dimensions];
    let configuration = {
      count: 2,
      dims: dimensions,
      flat: false,
    }
    let context = {
      constants: constants,
      config: configuration,
    }
    ... Garbage Collect here
    ```
    I've certainly simplified things a bit, but hopefully, you can see how this maps onto our memory. The very first box in both our diagrams is `configuration`. If something is a simple value, we need to arrows connecting it. If it has an arrow, that means it refers to another object. Our garbage came from calling doThingsWithTempContext which created a bunch of objects that we no longer have references to.
  </div>
</details>
Now that we've copied the live objects, we are left with some extra space to allocate objects. But our structure is completely preserved. So how do we accomplish this? 

### How to Copy Objects With No Extra Space Requirements

If you start trying to write this yourself, you quickly realize one tricky requirement is to not copy things multiple times. Consider the third object in our little diagram. There are two things that point to it. This means that as we are walking our reference we will encounter this object twice and rather than copy it twice, we want to record where we copied it and make use of that reference. This means we are going to need to something to do our book keeping. Maybe this is a list we do a linear lookup or a hashmap of address to forwarding address. These options though require extra memory. This is where the beauty of Cheney's algorithm comes in. Rather than utilizing more memory to track this process, we will cleverly reuse what we already have to keep track of all our state.

