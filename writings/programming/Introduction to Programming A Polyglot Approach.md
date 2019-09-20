# Introduction to Programming: A Polyglot Approach

Nearly every "Introduction to Programming" book picks a single language to use throughout. In fact, it seems not so much a deliberate choice, but more an assumption about what constitutes an introductory text. Throwing too many different things at a beginner can often confuse them; it can lead to too many errors. Perhaps these authors are right, perhaps a single language is best, but our hope for this book, is that they are indeed wrong. 

It is our contention that confining a beginner to one language robs them first of the beauty and variety the programming world offers, and secondly, risks biasing their formative thoughts, preventing them from learning and enjoying a different approach to programming than the one they were first taught. Programming lives above any particular paradigm the partisans of the programming world peddle as *THE RIGHT WAY*. But far from proclaiming all paradigms equal, this book hopes to present the reader with many distinct styles, showing both their strengths and weaknesses.

This endeavor cannot help but create its own bias. The choice of languages in this book is itself a form of bias, as is the order and manner in which they are presented. Ridding this work of bias is not the goal, but rather to help the reader form their own bias by allowing the reader the opportunity to think in different ways, to test different theories, and to be pushed intellectually. 

This book flows out of the belief that [programming is theory building](). Programming is not a working program, it is not a set of instructions, it is not the source code, the documentation nor all of these things combined. Programming is the activity of building a theory about how the program connects to and corresponds with the world. In this view, programming languages are the tools which allow us to build theories.

This view drives the book's approach to each language. The emphasis will not be on any particular program and its details, but the way in which each language enables and inhibits our thoughts. For some, this may make the book seem "impractical", but it is our contention that "theory is practical" and "practice is theoretical". To that end, we seek to strike a balance between rigor and allowing the reader to gain an intuition about the topic at hand.

## Intended Audience

The primary aim of this book is to do what the title says, to introduce people to programming. This means this book is aimed at people with no knowledge of programming. That said, the book hopes to follow in the footsteps of *The Structure and Interpretation of Computer Programming* (SICP) AKA "the wizard book"[^1]. For anyone not familiar with this book, this means that the focus is on the subject of programming rather than helping a beginner get setup with a programming environment. Books which aim to teach how to setup a development environment, which libraries to use, etc, have their place, but limit themselves in terms of their time of relevance. This book hopes to cover subjects that will be fruitful to beginners for many years to come.

[^1]: We have no false pretensions that this book is be anywhere near as good as SICP.

SICP is not just a book that doesn't cover these sorts of practical manners for its readers, but it also a book of great depth. Many readers have remarked on the books depth and difficulty and we believe the same may be said for the following. This book is not an intended as an easy introduction. Programming certainly can be easy but it requires taking on a certain mindset and practicing particular intellectual virtues. Good programs are created by labor rather than luck. Many today see programming as an easy profession to make lots of money, we see it as a learned skill and a way of thinking.

### Experienced Developers

While the book is aimed at those who have not done any programming before, we hope this book may well be found useful by experienced developers. Because of the large number of languages and concepts the book covers, many developers may be inexperienced with a particular technology and will hopefully find the discussion in those sections enlightening. We would however encourage even experienced developers to consider reading the book from the beginning. The sections in this book are meant to build and much may be missed by reading them out of context. 

### Self Taught Developers

Self taught developers will hopefully find this book particularly useful. In crafting this book, we have shied away, not from the academic world as a whole, but from the academic means of explanation. This means, we will not assume prior knowledge of academic subjects, we omit deep theortical concerns (proofs), and do not dwell on mathematical minutia. That being said, we find the academic world to be ripe with interesting ideas, something to be mined and refined rather than ignored.

## Reading Advice

Often books recommend following some strict plan of study. This often means a recommendation to take notes, reread sections, and to fully complete all exercises. We view that sort of reading as not quite in the spirit of the book. This book aims not to be a textbook, but a spring board for ideas. Don't feel the need to do percisely what the book says, experiment and play. Copying a program line for line is not the goal, but rather taking the program, changing it to see its outcome. Or even better, looking at a program and trying to rewrite it in a completely different style or fashion.

We hope to model this style throughout the book. In fact, the majority of programs in this book you will see multiple times in various languages. This means if you do not fully understand a certain program, you need not fret, hopefully another language will make it more clear.

### Pace

This book is deliberately fast-paced. Often reading a programming text feels like reading math. The style of exposition requires the reader to sit and concentrate on each of the details at hand. While these experiences can be educational and sometimes even cathartic, we hope not to duplicate this feeling in this book. Instead, we model this book after works in the tradition of Analytic Philosophy. Analytic Philosophy is marked by its clearity, exactness, depth, and dry humor. Most importantly though, while Analytic Philosophy might not be a best selling novel, it is imminently readable and often a joy to work through.

To put it more exactly, this book is meant to be read without sitting at a computer or a pen and paper. Read the book quickly, pause and think at moments that matter, but for the most part just keep reading. This book is paced so that it may be armchair reading; something to get your mind going.

## Finding an Order

Choosing the order in which to tackle these languages has been rather difficult. There was some temptation early on to simply ignore that issue and create the first "Choose Your Own Adventure" introduction to programming. But better minds pervailed. The order in which a book proceeds uniquely shapes the text, so extra care was taken in organizing these chapters.

We begin and end with LISP. LISP is the perfect language to begin our journey in programming because of its inherent simplicity. Throughout the book we shift to different languages, each language offering its own unique perspective on the task of programming and yet, we will often return to LISP. LISP gives us to flexibility to reimplement these features, allow us a way to quickly see through what appear to be magical features in a language.

Along the way we will encounter various paradigms of programming as well as many different languages. There will be some obvious omissions in this book that people may find strange. Many languages that are popular or influential will be left out. These omissions stem from two different forces that must be balanced in this book. First, this book is limited in scope, we simply cannot cover all that the programming world offers. More importantly, not all languages are interesting, even good ones. There are plenty of fantastic languages that offer very little by way of interesting content for new developers. 

## Short Comings

Finally one strange feature this book posesses is a complete lack of low level languages (C, assymbly, etc). This may seem to be a fatal flaw of the book. There is much to be said for understanding how computers work in their lowest levels. There are large groups that lament the lack of understand that new developers have of the internal workings of computers and this book is just contributing to that problem.

If anyone levys these criticisms against this book, we accept them. We see nothing wrong with learning C, but it doesn't quite fit with our goal. Programming is theory building; theories are defined not by the mechanical movements of their parts, but by abstractions which unify various ideas. For example, Darwin's *Theory of Natural Selection*, if stated in terms of purely mechanical parts, is just physics. Instead, Darwin operates at a higher level, talking about populations, adaptations, species (or lack thereof), and many other terms that operate above the level of mechanical functioning. 

While all programs are reducable to the functionings of our computers, thinking of them in these terms is limiting. We believe that the end of programming, the goal at which it aims, is to continue to operate at a higher and higher level. The languages we have chosen are those which allow us to build these levels. Languages that offer the programmer a way to abstract and express concepts succinctly.

To the extent that we delve into "low level" details, they will be of a mathematical and relational nature rather than mechanical ones. Programming is no more about machines than astronomy is about telescopes. Computers are the tools on which our programs run. Our programs continue to live beyond the life of any given machine and thus should be thought about separate from them.

# Chapter 1

Despite the place that computers hold in society, programming is barely understood outside of those who practice it. Telling someone you program for a living will almost immediately cut off all possible conversation. The average person does not even know how to ask questions about what programmers do. (Sofware Devloper/Engineer is the more common term for "programmers", but we will just continue to use programmers.) As far as anyone from the outside is concerned, we are "computer people". We must know a lot about computers or something. 

What is known about programming as a profession is generally inaccurate. Often people believe that programming "languages" are just like natural languages. In fact, some schools have even considered allow programming as a substitute for taking a foreign language. It is our view that this cannot be further from the truth.

It isn't uncommon to use terms borrowed from natural language to talk about programming. In fact, in this book we will do that. For example, we will talk about verbs and nouns to refer to different aspects of programming languages. But despite this superficial level of similarity, deep differences exist.

Natural languages have been grown organically, causing them to be riddled with ambiguity, inconsistency, and irregularity. Yet, it is these "ugly" elements of the language that allow for some of its most beautiful expressions. Works of literature play with this ambiguity, poetry pushes our language to its bounds inviting the reader to follow along, to read between the lines, to find the double meanings in the text. Stories are told, puns are made, symbolism abounds.

This aesthetic taste of natural language does not hold for programming languages. Good and beautiful programming languages are not those riddled with ambiquity. They aren't those whose meaning is indeterminate, whose syntax allows multiple interpretations. Instead the beautiful of a programming language comes from its simplicity and consistency.[^2]

[^2]: These decidedly subjective aethestic judgement may seem out of place in a "science" and yet the sciences, broadly understood, are riddled with them.  [[Maybe more]]

##  Overcoming Familiarity

Accepting that programs are not natural language can help us overcome the bias which is brought about by familiarity. Often programming languages will reorder elements or use symbols that may be unfamiliar. These choices are often not arbitrary. They allow the language as a whole to have a deeper sense of unity. Rather than run away from them, we should seek to understand them.

We shall see exactly this now as we explore our first language, scheme. Scheme is a language that belongs to the LISP family. The common feature all lisps have is their particular syntax marked by its use of parenthesis. Here we see our first program.

```scheme
(+ 2 2)
```
Here is a rather simple program. It adds 2 to 2. This will seem strange. Why write this with the `+` first? Why do we need these parenthesis?  The answer isn't immediately obvious, and yet as we leave scheme to visit other languages with a more "natural" syntax, we will see the incredible power this simple syntax transformation offers us.

Speaking more abstractly, in LISP all our verbs come before our nouns. To add two numbers, we state the action (in this case `+` that we want to perform and then we talk about what it is we want to add. When thought in these terms the order isn't nearly as strange. In English, I might say that I "hit the ball". This is a verb followed by a noun. What makes the above example feel strange is not it difference from English, but from the mathematical notation we have learned.

Arithemetical statements typically employ what are called "infix" operators. That is, our operator (+) goes between or inside of our operation. To add numbers we write `2 + 2` with `+` in between our numbers. In contrast LISP uses prefix notation, the operator comes before the numbers. Some languages we will encounter later will use postfix notation `2 2 +`. This use of prefix notation with parenthesis gives LISP some fairly nice properties. Unlike infix notation, prefix notation scales to many different items being operated on, and it has an ordering defined by sytax rather than convention.

```clojure
(+ 2 2 2 2 2 2 2 2 2)
(+ 2 (* 3 (+ 2 3)))
```

It is incredibly easy in lisp to add many numbers together. And while the second statement will certainly be a bit hard to read with those that aren't used to reading LISP code, its order is of no question. In order to evaluate the expression as a whole, the inner most parenthesis must be evaluated first. 

## From Prefix to Practice

Perhaps most remarkably, we have learned the hardest part of schemes syntax. It is this seemingly small choice of parenthesizing prefix based operators that has kept any LISP from becoming a popular language. Much ink has been spilled complaining about LISPs famous parenthesis. Many, in fact, have attempted ot make the parenthesis disappear from all LISPs. Yet, it is these parenthesis that offer LISPs their secret super power. 

Unfortunately, it is much too early in our programming journey to explore that super power. So instead, we will take closer look at Scheme's way of speaking, exploring its various types of nouns, some of its fundamental statements, and then how to build up our own verbs. First, let's look at the various types of nouns we will deal with in Scheme.

```scheme
; integers
1
-2
432

; floats
2.3
4.3451
3.14159

;booleans
#t
#f

; strings
"Hello World"
"a"
"things"

; symbols
hello
x
+

; nil
nil

; lists
(1 2 3)
(cons 2 nil)
("string" 2 cons 2.3)
```

These are the nouns Scheme provides for us to write our programs. As well continue learning more and more languages, we will see similar features. In fact, in most languages, these elements are only different in very minor ways. This is one of those features that makes programming languages so different from natural language. A book like this one but written for natural language, would simply be impossible. There is no way to learn so many natural languages at once, because they all differ in so many ways. But, once we have learned one programming language, the next becomes much easier.

In sticking with our arm-chair reading philosophy, we won't dive into these nouns in detail at the moment, instead we will take them as they appear in our programs. I would like to call out just a couple things to ensure this list is readable. Any line that starts with a `;` is a comment. It is often useful in our programs to leave comments about our intent or to mark things off. Once comments are cleared up, the other strange thing in our list of nouns are the "booleans". If you already know what booleans are, then the `#t` and `#f` are easy to figure out, but this is an introduction. Unfortunately, this is one ugly area of Scheme, `#t` stands for true, while `#f` stands for false. Every language has its ugly parts and we are certainly not done with finding them in Scheme.

### The Building Blocks

Now that we've explored our nouns, lets look at how to use these nouns to accomplish tasks. You see, nouns by themselves can't do much, at least in Scheme. We need to connect up our nouns, pass them to things that will take our nouns and return us new nouns. 