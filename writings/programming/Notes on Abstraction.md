# Notes on Abstraction

I haven't really enjoyed talks I've heard from programmers about abstraction. It seems there is a lot of confusion and I'd love to try and examine it myself. I think having an account of abstraction would certainly be useful, though I worry there is no one account. I do think that what we call abstract is at least 2 or 3 different things, but perhaps we can pull these apart. To help me think about what abstract is, I will first talk about what it is not.

##Cheng Lou's Proposal

[Cheng Lou's talk](https://www.youtube.com/watch?v=mVVNJKv9esE&t=912s) on abstraction contrasts "abstract" with "useful". Something that is more abstract is less useful in the sense that that we can't "take it off the shelf" and use it. We must build on the abstraction in order to suit our concrete use case. Here abstract corresponds with "power". It isn't really clear what power here is supposed to be. It is actually defined in the talk in a seemingly circular way. But it seems from the rest of the talk that power has to do with having fewer constraints. Or to put it in a different way, a more abstract thing can cover more concrete use cases. 

I don't think the talk is very clear, I have tried listening to it around 4 times now to try it get a more complete picture, but still the details confuse me. I think perhaps his most clear example is the discussion of build systems. He constrasts a declarative DSL build system with one that lets you just use functions. The declarative DSL would be considered less abstract, because it is more specialized, more constrained. The building system allowing you to use your own functions would be more abstract because it is more powerful, less constrained. The latter build system will cover more cases, but require more work to cover a particular concrete use case.

So it would seem from the discussion in the talk that if something is more powerful, it is more abstract. Now their are two ways to take such a claim, as a technical definition or as a way to capture our intuitions. Let's explore the latter first. Let's compare two concepts in programming and decide which is more abstract than the other. For some elements, this may be rather difficult, but in general, it seems given the theory above which should be able to compare many elements. The two I'd like to compare are monads and functions.

It would seem that on our intuitions, most people would agree that monads are more abstract than functions. In fact, it seems that *maybe* Cheng Lou agrees with this as well. In the talk he mentions Monads as one example of things that are "way too powerful". He calls Monads "too general to be popular". He even suggests we can see this by reasoning using his trees of abstractions. In a slide not presented in the talk he shows monads at the very top of a tree of abstraction.

So it seems that Cheng Lou agrees with our intuition that monads are more abstract than functions, but yet, monads are clearly less powerful than functions. There are all sorts of things that may be expressed with functions that cannot be expressed by monads. In fact, monads just are functions constrained. So the definition above fails to capture our intuition properly. Likewise, if meant as a technical definition, it does not seem to be followed consistently, because it ought to conclude that monads are less abstract than functions.

## Zach Tellman's Proposal

I am not clear at all what the suggestion is supposed to be other than that it involves models, interfaces, and environments. It involves ignoring some details.

## Ways in which we talk about Abstraction

We talk about something being abstract vs concrete. We also talk about things being more or less abstract. We talk about "abstracting over" some detail or other. We discuss ideas "in the abstract". We talk about "good abstractions" and "leaky abstractions". We discuss the "cost" of a particular abstraction. We say that some things are "too abstract" or that something looks like a "premature abstraction". We talk about data abstractions as well as abstract data types. We have abstract base classes. We say that something is an abstraction simplicitor. We can compare abstractions, saying this is a "better abstraction" for accomplishing X than that is. We can discuss equivalent abstractions. Two languages may have the same abstraction with surfaces features that differ and yet, we can see a "sameness" in them. We can have proper and improper implementations of an abstraction. We can have abstract representations. We can discuss the purpose of an abstraction.

These are just some of the ways we use the words abstract and abstraction. This sort of flexibility and variety in our use of the concept suggests that an adequate theory of abstraction will be rather difficult. We must support judgements and higher order operations on our notion of abstraction. We must have some sort of ability to talk about the identity of a particular abstraction, we must have some notion of ordering or hierarchy. In fact, this variety suggests that there are perhaps multiple notions at play here. For example, is the abstraction of art the same notion as abstraction in computer science? In what ways might they be related?

Of course, a definition of a difficult term like abstraction shouldn't just capture our intuitions, it should also help us clarify areas that may have been unclear. Hopefully this definition will allow us to weigh-in on areas where disagreement abounds. This is a tall order for a definition and is the sort of definition not normally given in software. Instead, a minimal technical definition is given, which has its uses, but is not illuminating in the way we desire.

## Fundamental Subjectivity

There are certain ways we talk about abstraction that are necessarily subjective. We may talk about something being "too abstract" simplicitor or "too abstract" to be useful/popular/understood etc. This sort of talk is coupling a certain understanding of what it means for something to be abstract with a value judgement. It also make quite a few assumptions about people and their purposes and abilities. It should not be expected that any theory of abstraction will help us decide questions like is this thing "too abstract", but a theory may give us insight into why people may think of something as "too abstract". If a theory allows us to define a spretrum of any sort, we may be able to find properties these "too abstract" things have in common.

## Abstraction is Relative

I have this idea that I think might be right. Let's start with the notion of a set. In java, a set is an abstraction. There is the java.util.Set interface that helps us define the abstraction (not an exhaustive definition). There are a few things that make this an abstraction, but one necessary condition for it being an abstraction is that it is built out of other things. This is of course not sufficient. A car is not an abstraction nor is a person and they are both built. But what we cannot say from this is that sets are abstractions simplicitor. In ZFC, sets are not an abstraction, sets are fundamental. In fact, in ZFC, integers would be an abstraction and sets would not be, whereas in java it would be the exact opposite. 

So to say something is an abstraction implies that is it not fundamental relative to some base. This reminds me quite a lot of "Making Things Up" by Karen Bennett. There definitely seems to be a relationship.

So what about degrees of abstraction? Let's take a famous programming example, monads. As I mentioned in Cheng Lou's proposal monads are a pretty famous example of "very" abstract things. And yet Cheng Lou's notion of power would actually make monads less abstract than functions. So to keep our language precise, let's talk about monads vs functions in haskell. In haskell, why would a monad be more abstract than a function? Here it actually seems pretty straight forward. A monad is built out of functions. If Y is built out of X then Y is more abstract than X. We will end up with a partial order, even when restricting our bases. I need to review making things up because I think there is a lot that is useful there.

### Random Thoughts

* Are abstraction and reification related?
* Conceptual Engineering and abstraction?

### Abstraction Criteria

* A primitive is not an abstraction
* An abstraction must be multi-realizable
* Abstractions have instances which are the realizers
* Abstractions are built

