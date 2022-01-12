# Towards Aesthetics Elements of Programming

Programming styles are immediately recognizable. And yet, the elements that contribute to these styles remain unnamed. What is it that separates the classic Scheme style of programming from a modern OO java style of programming? It is not the language. It is possible to write scheme, javascript, or whatever in the style of a modern OO java program. Nor is it just the paradigm, it isn’t as if all OO programs have the same style. So what precisely are we picking up on when we note the style of a codebase?

## Identifying Elements
Ideally, to understand these styles, we can consider elements that combine to form a certain style. If we identify the right elements, we should be able to take them, vary them, and combine them to get a particular, recognizable style. Perhaps even, we can see what combinations of these elements are rare, or unseen in the programming world.

This is what I hope to explore in the essay. I doubt I will get the elements right from the beginning. But perhaps by throwing some out there we can start to think more clearly about what the elements really are. We can find better ways to divide things and start having an understanding of style.


## Verbose vs Terse
We can significantly alter the way we communicate in code by varying our verbosity. This might be at the level of a whole program or in a particular section of code. We might adopt patterns that require a certain level of verbosity (objective C named methods, early redux culture) or adopt whole languages that focus on terseness (j/k/apl).  

Verbose code can help with clarity. Long, expressive function names can help communicate the semantic meaning of the code underneath. Repetitive elements in code can help reinforce patterns, making it clear what code belongs where. Verbose code can guard against errors, preventing simple transpositions from happening by requiring fully specifying the context of a value.

Terseness can be its own kind of clarity. Terseness can convey unity (eg. mathematical expressions). Terseness can help convey the lack of semantic meaning (eg. single-letter variables). By being terse we can make clear what matters and what doesn’t. With verbose code, it can often be unclear the exact shape of the code. It can be unclear what the relations between parts of the code are, terseness helps us here.

## Dense vs Sparse
Whereas verbose and terseness are related to the way elements are spelled out, density and sparsity are about the relations between these elements. A sparse codebase spreads its functionality out. This can take many different forms, many small classes, spread out across many files is a common example. A dense codebase packs many things into a singular place. 

In some ways, dense vs sparse might seem merely superficial, just concerned with the presentation of code rather than concepts in the code. I think this is far from true. Deciding where we place elements (regardless of what “where” here means) has a big impact on how code is understood. By grouping some things and keeping others separate, we are signaling something to other programmers. 

But this is not merely a signal. We may intentionally make our code dense to, for example, discourage reuse. Large single blocks of code may only serve a very specific purpose that won’t be useful to others. We might keep our code sparse to enable layering of functionality. This is often seen in classes with large inheritance hierarchy structures.

## Structured vs Unstructured
The contrast here between structure vs unstructured is not referring to “structured programming” in the “goto considered harmful sense”. Instead what I am referring to is a program-level structure. Some codebases provide an overarching structure, perhaps providing categories for where each sort of item should go (eg. MVC). Then some codebases are more free-flowing. These don’t provide some sort of meta-narrative about exactly where a bit of code belongs.

Highly structured code can be a means of communication. By placing elements inside this structure, we are communicating something about these bits of code, we are telling the reader what we consider to be the most important way bits of code are related.

Less structured code does not communicate any less. By keeping our code less structured, we can signal several things. First, we may be saying that this code is not in its final form, it is in flux and the organization may change. We might also be communicating that this code is a complete whole and does not need to be divided into parts.

## Direct vs Indirect
There are many different ways that directness can display itself in our codebases. First, we may call a function directly, or we might find some indirect means of calling this same function. Perhaps we define an interface and inject something with that interface, that then delegates to our function. 

But direct and indirect are not merely about function calls. We can for example have code that builds up some tree or hierarchical structure and then interprets it, rather than code that directly runs a computation. We might convey data onto a queue, with some consumer elsewhere rather than locally deal with our data. Directness of course comes in degrees and is relative to a particular structure. Some parts of our code will need to be direct in some way, but they may live inside of a larger indirect structure.

## Open vs Closed
To avoid confusion I should mention from the outset that I don’t mean the Open-Closed principle from solid. Open and Closed here are a bit broader in their meaning. A system might be open by being completely introspectable. It may be closed, by encapsulating or protecting some parts of its data. But this open and closed spectrum does not need to be enforced by the language to be an aspect of our codebase. A codebase can treat a structure that is technically open as if it were closed.

Open vs closed does not merely refer to data protecting either. A system can be open by accepting data it is unaware of. A closed system may guard the borders, only allowing particular data in the exact shape specified. An open system may expose reflective ability, it may provide metrics, or tracing. A closed system might provide strict guarantees, correctness, performance, memory usage. 

## Generic vs Specific
Our code can be Generic or Specific on many different levels. Our function may be polymorphic, allowing it to work on different inputs. Our codebase may be solving a generic problem (a renderer) or a very specific one (a renderer for the Mandelbrot set). 
Even if we are solving a generic problem, our code can be specific in its details. We may provide specific instantiations of our more generic method. We could on the other hand leave our code entirely generic, acting as a framework for which people can make more specific code.

## Concrete vs Abstract
Concrete vs Abstract may sound a lot like generic vs specific, but I think there are important differences. First, concrete vs abstract is about the subject matter or the concepts used in the application, rather than their capabilities. To give an example, Car may be considered Concrete, whereas Monad might not. Both Car and Monad can be used generically, with more specific instances utilized elsewhere in the codebase. 
Admittedly, it is much easier to see how abstract code coincides with generic code. But this need not be the case. Perhaps we are writing abstract code about partially ordered sets, but using them for a very specific purpose (proving that certain properties hold).

Regardless, the use of concretions and abstractions in our codebase is a powerful way to communicate. By making something concrete we put bounds on it, we invite readers to apply their common conceptual categories to it. By making things abstract, we invite readers to expand the bounds of what they consider.

## An Example
I think these elements are great candidates for being elements of programming style. Let’s take a random combination of them and think about what a program written in that style might look like.

* Verbose
* Sparse
* Structured
* Indirect
* Closed
* Specific
* Abstract

There is a lot to unpack here. First, it must be said that of course no codebase is uniformly all these things in all its parts, so will be generalizing a bit.  Second, trying to think about all these items at once is hard, so let's pair them up a bit and think through the implications of those pairings.
Combing verbose and sparse might look like a library with a decent amount of boilerplate, where these boilerplate elements are rather spread out, perhaps even orthogonal to each other. We might imagine a verbose and sparse Redux or Rails codebase.

Structured and Indirect might combine to a structured means of dispatch. We can imagine a dependency injection setup like hibernate. Nothing calls anything directly, but everything has a clear hierarchy. Here again Redux might be a good candidate to consider. A reducer structure with many sections combined with a hierarchy of selectors could meet this criterion.
Closed and Structured can be combined as a means of making illegal states unrepresentable. Here, we can ensure that no one can trigger any illegal actions, everything is clearly defined, and it cannot be changed without accounting for our structure.

Specific and Abstract are a bit harder to imagine. Imagine a web app designed to do a particular thing. It is not meant to be generally applicable or configurable, and yet the entities we find in our program are not concrete and familiar. Instead, we find things like executors, factories, or perhaps our code is organized around a free monad. While the concepts we are employing may be able to be used generally, our code is not doing this. Instead, we have very specific uses of these abstract concepts. Think, about the way people might make interfaces for everything, and yet only have one instance of them

So given this set, we can think about what it would be like working on a codebase following these styles. First, errors would be dealt with via our structural and closed system. Next, we would not introduce many different concrete types, instead, we would make use of abstract concepts, perhaps certain event types, or operations in and algebra. Further, things would not directly call other things or directly depend on things. Instead, these would be passed around to use or gathered through some abstract means. Further, as we introduce new concepts, we would not colocate them with existing concepts, but instead, following a pattern established by our structure, introduce them in some separate section of the codebase.

This feels very much like some Java Spring apps I’ve worked on, but also some redux apps I’ve used. These two could not be further from each other in terms of paradigm, and yet, there is a shared sense of style here. Now that does not mean they are equal, it does not mean working on them feels the same, just that they share something in common. Nor do I mean to suggest all Java spring apps or all Redux apps follow this style.

## Conclusion
Hopefully, these elements ring true to you. I think the most questionable distinction here is between abstract and concrete being different from specific and general. And yet, the more I try to get rid of this element, the more I see examples of it. For example, we might write general code, that we give concrete names. Maybe our code just works on some general notion of distance, but we name our parameters `distance-in-seconds`. Here we have a general, concrete coding style.

But as I think more about this, explore more codebases and get feedback from people. Perhaps I will come up with a new set, or replace elements on here. Regardless, I think these concepts are useful and helpful for discussing code style. This is something we need to do more of.