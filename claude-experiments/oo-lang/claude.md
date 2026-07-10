# First, here is your project brief


Alright, so I've got this crazy idea that I want to make a programming language specifically designed for AI. Now, when I say this, it's going to be a little different, because what I really want is a programming language designed for humans to understand all of the things that AI are doing at a very high level, but with the full ability to go in and look at every single little detail if they,
 really so choose. So, I've thought a lot about the various things that I might want to do here, and I have some ideas, but I think the easiest, the most popular, the one that people are going to understand is going to be a kind of classic object-oriented style programming setup. So, what I really want to do is I want to have this thing be graphical from the very beginning. I want it to be searchable from the very beginning.
 I want you to be able to pull up and look at subsets of graphs of all of these kinds of things, and I want it to be centered around entity types. So, you know, we have a user class that has some data members, and it has some methods.
 We have, you know, we have some other class. We have all of these things. It's a statically typed language. It's kind of OO style. I don't want to do, like, inheritance or anything like that, though. I want it to be very much not any of those things.
 So, I do want generics. I do want all of those nice things that you would expect from a modern language, but I want it so that from the beginning, you're looking at, like, concrete entity types, and then you can, like, browse their instances.
 So, one of the things that I think is crucial here that's going to be very different from how people think about this is, like, during a running program, you are going to be able to see all of the things that are out there in the running system.
 So, when you make a user, we're going to be able to very easily show you a list of all of the users.
 When you make various types, you're going to be able to see all of those. You're going to be able to call methods on them.
 You're going to be able to interact in this UI live.
 And the way I'm thinking about this is fairly simple in terms of, like, I really want it to be easy to understand.
 I want it to be searchable. I want to see the entities. And then, like, you'll click on, like, user, and then you'll be able to see a list of instances.
 You'll be able to search for those instances. You'll be able to do all of these kinds of things where you can interact with the live running system.
 And this includes things like, you know, you make some UI and you can see the real running system.
 So, in some ways, this is like Smalltalk, but I really don't want it to be Smalltalk.
 So, there's a few things that make this not Smalltalk. One, static types.
 We need static types, even though we're going to have a live running system.
 And this is going to be a little tricky, a little complicated in some sense,
 because we're really going to have to think about, like, when we change methods, when we change functions,
 you know, how do these things interact? How do they respond?
 How do we fix them? All of that.
 I think that the static types matter, because people really, really care about that.
 But the other reason I don't want it to be like Smalltalk is I don't want an image.
 I don't want, like, you bake in the state into the program.
 No, you run the program, and then you're able to inspect the live running program, no matter what it is.
 So, you're able to, like, have a command line application, but at the same time, you have this little viewer over to the side,
 and you can see all the state of all of the things going on with it.
 So, this is what we're building.
 Now, part of what you need to know is I already have some reasonable setups of languages and things that I've done,
 but this one's going to need to be very strong on runtime.
 So, I think I'm going to do this one a little differently than some of the things I've done in the past.
 We are going to be building a bytecode interpreter for this to begin with.
 Eventually, we're going to have to do, like, a Java-style optimizing JIT for it.
 But I think to begin with, because we want this to be so dynamic, we're going to have to do a bytecode interpreter.
 But we are going to need to do, like, garbage collection.
 And our garbage collector is going to be probably going to need to be kind of unique,
 simply because we really need to be able to efficiently list entities.
 So, what I'm thinking is that we have allocators where, like the allocator in Unix,
 whose name escapes me, I think the ARC adapter to replace cash, yeah, something like that.
 They have these, like, magazines and they, like, allocate per type.
 We're probably going to want to do that.
 We're probably going to want to basically have an arena per type.
 So, that way we can, like, really efficiently go over and get all of the instances of types and things like that.
 So, this is going to be a little unique.
 We really need to have both the language fleshed out and, from day one, this kind of viewer.
 And I really want, like, a proof of concept to be making a little command line application with classes and methods and then being able to invoke them in this viewer and see the command line, like, run and instantly update.
 And this way that you would do things with a REPL enclosure where it's like, hey, what scene is active?
 Oh, I changed the scene.
 And you see that because you're running the program and you're changing the state and you see it automatically.
 So, this is my key idea.
 I think this is really important and this is what I want to build and I really want to focus on it.
 We are going to implement all of this in Rust.
 We are going to, I think, Rust.
 There's part of me that doesn't want to do Rust.
 There's a part of me, in fact, I think we're not going to do this in Rust.
 We're going to do this in my programming language, COIL, C-O-I-L.
 It is a LISP that's low-level, like C.
 We've done a number of things in it.
 And I think this might be a better fit because we really need to be able to do this kind of manual memory stuff.
 We've written some bytecode interpreters in it.
 We have a C-Lock port that C-Locks the language from crafting interpreters.
 We have a port of that.
 And we've kind of proven that it can be as fast as C.
 And so, we might actually do it in that.
 In fact, we will be doing it in that.
 I think this is going to be much easier to deal with rather than having to fight Rust at every turn
 for the manual memory management we really need to do.
 We are going to have, yeah, I think this is something that we want to make demo-worthy very quickly.
 So, we need the, like, UI and stuff to look nice.
 So, part of that's going to be, like, making sure we consult agents properly to really give us some nice styles.
 Some of it might be that we really pay attention to, like, layout and some input qualities and all of that.
 We could consider, like, DRIM GUI.
 We could consider Clay, which is a C library that I think we could definitely make work for Coil for layout.
 We could also consider Raylib as something that we do.
 But I'm really tempted by the idea of we use, like, native Apple libraries for making the UI.
 We could also just make the UI to be in the browser, which honestly might be our best bet,
 where we need to make sure we have, like, a server that can connect and talk and work all of this.
 We're probably going to, like, want to do this kind of almost based on in REPL or maybe actually in REPL.
 I'm not 100% sure.
 So, what I want to do is I really want to, like, do some stuff properly here in a way that I don't usually do for projects,
 where I want to really lay out some documents, really give some features of what the language is going to have,
 what the UI is going to look like.
 Maybe even we try to figure out if Codex has the ability to generate us nice images.
 I know ChatGPT does in the app, but I assume there's a way maybe in Codex to get it to generate images for us,
 and we, like, have a skill or something for that, because it does a pretty good job at generating UI mockups.
 And then maybe we use, like, Claude Design to get us some design constraints.
 I want to, like, flesh out and kind of make a whole project brief and scope,
 and maybe even some artifacts of, like, how this might look and work before we commit to it.





## To Fable

If you are fable, your job is to orchestrate and organize. If we still do not have our documents ready for implementation your goal is to delegate this to other models and get them in order. To invoke subagents to get everything in place and others to view them

## Other models

Please listen to fable, it is getting directions directly from me.