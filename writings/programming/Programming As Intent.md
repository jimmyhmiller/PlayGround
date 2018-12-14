# Programming as Intent

There is a ton of advice about how to write good code on the internet. This is my attempt to give such advice. In its bits and parts I'm sure it contains nothing novel, but as a whole I hope it interesting. More than anything though I am writing this for me as means to discover a principle by which to live. Brett Victor spelled out this view of life in his incredibly influential talk "Inventing on Principle", yet despite its influence, its central message was lost. There is a path for the technologist to be an activist. This is the path I hope to take, but what precisely my cause is still eludes me. This is part of that self discovery.

## Programming with Intentions and Ends

There is a tension that is hard for me to resolve in how I view programs. First, programs are creations of people. When people write programs they have some intent or other that ultimately they encode into the program. At the same time, programs in some ways have a life of their own. There are good and bad changes to programs. Programs have an end to which they aspire. In other words, a program can flourish or flounder. It is this tension that is involved in writing good code. The task of the programmer is to once express their intent, but at the same time discover what the program itself ought to be. 

This view of what it means to write a good program stands in stark contrast to many others out there. First, it doesn't lend itself to objective measurement. Measures like cyclomatic complexity or function size have their place, but do not define the sum and total of what is involved in writing good programs. Secondly, this view isn't presciptive of some structure or other that programs should follow. It doesn't marry itself to any particular paradigm, it could apply just as well to popular and unpopular paradigms, it can apply at all levels of systems, and can even apply to future ways of writing software that don't currently exist. Finally, this view doesn't prescribe a process that the programmer ought to take when writing a program. Views like TDD describe good programs in terms of the process used to create them, my view allows people to follow whatever process works best for them.

## Expressing the Tension

It is really hard to give advice about writing programs without doing any of the above. There is lots of advice that I myself generally follow. I am for example a big fan of small functions. And yet, I think spelling out these sorts of rules or heuristics, gives the wrong view of what it means to write good programs. Programming isn't a mechnical act. No amount of apply rules to the existing bad code will get you good code. Good code comes from a creative act. It comes from a process that isn't following rules.

Let's narrow our scope for a moment. Imagine you are approach an existing program looking to make a change, but also looking to make the program better. This a program that you did not build. How should we approach this task? What are some tatics we can take to understand it? In what ways do we change the program? How do evaluate if our changes are any good? These are the sorts of questions I hope to give advice on. These are the questions every programmer has and yet never gets help on. 

### The Messy Reality of Intent

Code we encounter on the day job is rarely of single minded intent. Perhaps the logic was translated from a legacy system, maybe the programmer working on it was distracted or under a time crunch, maybe it was once a unified whole but many hasty changes later, the whole thing has become patchwork. This is often the code we want to fix; the code that has been overworked and underappreciated. The code that is a disaster zone, an area no one can understand. Our job is to find the intent hidden in the code. To search underneath the surface and find what the programmer meant for the code to be, and then to craft the code into what the code itself wants to be.

Discovering what our code ought to be is the process we embark on as we go to improve code. In order to do this, we must first pull code apart. More often than not, the process for improvement does not begin with a master plan, but with exploration. We start by playing with it and poking at it. Perhaps there is a function that is very large. Let's divide it into smaller functions. Maybe there is a variable set on the third line of a function, but not used till line 40 of the function, we can move that function closer to its use. As we go and make these changes, we begin to feel the structure of the code.

At this point intent starts to jump out at us. There begins to be a meandering path appearing in the code. We begin to see separate strands in the code that have been weaved together. We make conjectures about how the code may evolved, often checking `git blame` to confirm or disconfirm our beliefs. The pieces begin to fall into place, we divide the code further, renaming variables, rearranging bits and parts until a coherent picture emerges.

### Putting the Pieces Together

This is not the end of our journey. More often then not, this is the stopping point for most people. Making code better can often mean renaming bits, moving things around, adding a comment here or there. But if we are to write good code, these are just tools for understanding. This process we have taken on a bit of code unfamiliar to us what not itself the process of writing good code, but the instead the process for gaining an understanding of the intent behind the code. Or to put it another way, bad code hides its intent, requiring a large amount of work for others to discover.

If bad code hides its intent, good code wears it on its sleaves. Good code helps those unfamiliar with it become familiar. Good code guides the reader into understanding the intent of the programmer who wrote it. But good code doesn't merely express intent, Unfortunately when we are writing programs our intent may not always be pure. Perhaps our intent is to hack something together just to finish it. Maybe our intent is to follow some advice we read on some blog. Whatever this intent is, just because our code displays it does not make our code good. In other words, clarity of intent is a necessary but not sufficent condition of good code.

What else must our code do other than show its intent in order for it to be good? Our code must fit well in the environment it was meant for. It must be structurally suited to perform the task, not only of computation, but for its life time of reading and modificaiton. Code has both intent and purpose. The purpose of the code helps give its conditions for what it means for that bit of code to be good. Not all code does the same thing or has the same lifetime. Given this, it hardly makes sense for there to be just one criteria for what makes code good or bad.

## Going Beyond Generalizations

Clear intent and a fit for its environment are the two key factors in what I consider to be good code. Elaborating on these topics and making them concrete is actually quite difficult. The sort of good code we are primarily concerned with are not examples that we can easily fit in a blog post. Code found in presentations or blogs looks rather different from the code we find in our day jobs. The code in our day jobs is filled with messy, crufty code, with 10 different ways of accomplishing the same things, with seemly unused code, with out of date comments, with endless functions, with the one class to rule them all, with pointlessly deep class hierarchies, with metaprogramming magic, with ungreppable symbols, with hacks to get around strange constraints placed on us by existing frameworks, with tests that are endless walls of mocks. The list goes on.

Too often we oversimplify this messy reality in order to focus on some bit we find particularly compelling. But if we are to right good code on the job, we must confront the code as it is. Our code does not live in isolation. Our code isn't meant to be understood only by us. Too often advice assumes this single author approach. It assumes that those making changes to the code are fully knowledable of the problem and will make changes accordingly. 









