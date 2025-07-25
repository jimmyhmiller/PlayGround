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

### Bookkeeping

If you start trying to write this yourself, you quickly realize one tricky requirement is to not copy things multiple times. Consider the third object in our little diagram. There are two things that point to it. This means that as we are walking our reference we will encounter this object twice and rather than copy it twice, we want to record where we copied it and make use of that reference. This means we are going to need to something to do our book keeping.

Typically book keeping comes with some sort of memory overhead. We need to store our data somewhere. That could be a list or a hashmap or a set or whatever structure makes sense for our use case. But Cheney's algorithm gives us a way to solve this without using any extra space.

## Cheney's Algorithm

The key trick to Cheney's algorithm is to realize one key feature: Once we've copied our data, we don't need it anymore. But to understand how to efficiently take advantage of this fact, we need to dive one step deeper into how languages represent objects.

Here is a fairly simple model of how a language might represent objects in memory:

![Screenshot 2025-07-17 at 5.35.51 PM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2025-07-17 at 5.35.51 PM.png)

The header here records data about our object. It typically will be broken up into some different fields.

![Screenshot 2025-07-17 at 5.40.10 PM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2025-07-17 at 5.40.10 PM.png)

Here we have a made up example. The only important part for us here is this forward bit. This is a bit saved specifically for our GC. This bit and small detail about how machines allocate memory are all we need to do all the book keeping without any extra storage.

## The actual algorithm

```python
def cheney_gc(roots, from_space, to_space):
    for i, root in enumerate(roots):
        roots[i] = forward(root, to_space)

    scan = to_space.base_ptr
    while scan < to_space.cursor:
        header = read_header(to_space.data, scan)
        size = get_size(header)
        fields = get_pointer_fields(to_space.data, scan, size)
        for i, field in enumerate(fields):
            new_ptr = forward(field, from_space, to_space)
            write_pointer_field(to_space.data, scan, i, new_ptr)
        scan += size

def forward(ptr, to_space):
    if ptr is None:
        return None

    header = read_header(ptr)
    if is_forwarded(header):
        return header

    size = get_size(header)
    new_ptr = to_space.alloc(size)
    copy_bytes(ptr, new_ptr, size)

    write_header(ptr, new_ptr | 1)
    return new_ptr
```

I know when I read blog posts about code, I almost never read the whole code listing. So let me pull out the key insight: we can reuse our header to store the forwarded pointer. This means we know 1) if we have copied an object already and 2) where we copied it to. 

Finally, one fairly surprising feature of this algorithm is how it only requires us to first copy our roots, and
