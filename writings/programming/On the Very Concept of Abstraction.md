# On the Very Concept of Abstraction

Providing an analysis of a term as fundamental as “abstraction” is a tall task. Yet, it seems this task has not be recognized in the literature. This is not to say that attemtps to characterize abstraction have not been made. Perhaps the most direct can be found in Timothy Colburn and Gary Shute’s Abstraction in Computer Science. Here we find perhaps the most complete description of the most common definition of abstraction given, Information Hiding. What exactly is information hiding? We aren’t given much in the way of explication. The primary contrast is between hiding and neglect. Colburn and Shute claim that abstraction in mathematics is charactized by neglect while in computer science it is characterized by hiding.

We could investigate this claim and see if it is really true that abstraction in computer science is about information hiding, but I think a more important issue lurks here. What makes for a good analysis of a concept? Is “Information Hiding” supposed to be a complete description of the term abstraction? Certainly we can make intuitive sense of it, but does this analysis help us understand abstraction at a deeper level? Does it provide guidence in how we ought answer difficult questions about abstraction? It isn’t clear that it does.

Suppose I am planning a surprise party for a loved one and choose to hide this information, have I created an abstraction? Perhaps it seems obvious to the reader than I have not, but where does that determination come from? It is from our intuition about abstraction rather than from the analysis offered. If “Information Hiding” is to be a true analysis of abstraction, it must at the very least give us guidence as to what kind of “information hiding” counts. 

For my part, I will not focus on making precise the notion of information hiding, though I think this endevour worthwhile. Instead I intend to focus on a quite different anaylsis of abstraction. One, which while incomplete, will hopefully begin to help us to form a more precise notion of abstraction. My hope is that we will arrive at a picture of abstraction that is more precise, but also that helps us make sense of our intuitions regarding the subject.

## How we speak about abstraction

Abstraction discourse comes in many forms in the programming world. Here we find a wholly incomplete sample of the sorts of ways we can talk about abstraction.

- Lists are an abstraction.
- That is a leaky abstraction.
- Programming is about finding the right abstractions.
- This code is too abstract.
- Monads are a great abstraction for asynchronous programming.
- Monads are so abstract no one can understand them.
- That isn’t real, it’s merely an abstraction.
- We need to think about this at the right level of abstraction.

Most notably in this list is the word “abstraction” a noun and the  adjectival form “abstract”. We also find “abstraction” and “level of abstraction” two related but clearly different uses of the word abstraction. An analysis of abstraction ought to help us make sense of these and many more utterences regarding abstractions. Information hiding, even if not precise definition is forth coming, does seem to help us in this regard. Take “This is a leaky abstraction”, on the information hiding view, this is straight forward, the abstraction “leaks” some of the information it was trying to hide.

But while an analysis must pay attention to the linguistic data, it is not necessarily beholden to it. It should  come as no surprise that linguistic utterings of various people do not form a perfectly coherent concept. A technical analysis will be necessity exclude some things included in the folk definition, but we should aim that these elements are not at the core of the theory, but ones for which there is already disagreement or confusion.

### Abstractions and The Abstract

I want to focus in on one element I believe is crucial to any understanding of abstractions. One which we must explain and not merely set aside. This is our subjective ascriptions of something being “too abstract”  or overly abstract. Consider the complaint above about monads. While we might not all feel monads are too abstract, it is certainly true that monads are not only an abstraction, but a rather abstract abstraction. What do we mean by these words?

Let’s begin with trying to see if the information hiding view can help us here. Perhaps the most obvious suggestion that comes to mind is that X is more abstract than Y just in case that X hides more information than Y. So on this suggestion Monads must hide more information than other abstractions we find relevant. But is this the case? Consider functions (we will later reconsider the status of functions as abstractions in greater detail). Functions (in many languages) are arguably abstractions. But they are also generally judged to be less abstract than monads. But which of these hides more information? A function. Monads are constrained by laws, they have a defined interface, given a monad, I can tell you much more information about it, than a function.

So much for that suggestion of information hiding. Perhaps there is a way to rescue the informaiton hiding view here, but at the very least, it isn’t obvious that the information hiding view tries to answers this question. What other answers could we give? Perhaps monads are more abstract because of the number of non-trivial implementations they allow. By non-trivial I mean to rule out implementations that differ only in unimportant details or that contain unnecessary elements. There is the state monad, the io monad, the continuation monad, etc. But what about lists or trees? Are there not just as many ways lists can be implemented? Do we thereby find them more abstract?

It is clear that whatever answer we give to this quesiton will not be definitive. Perhaps we could create some function which orders all abstractions from less abstract to more abstract, but doubtless there will be some cases where there is disagreement. An analysis of abstraction would not need to provide such a function, but it ought to explain something about how these judgements relate.

### Abstraction and Mere Abstraction

Among those who advocate for the information hiding view of abstraction, there is a related notion of an interface. For many these terms are intimately related. To abstract something is to provide an interface for it that hide the details of the implementation, taking something from a concrete thing, to an abstract thing. But is this move the only way to create an abstraction? Consider the difference between a java.util.List and a linked list in C. A java.util.List defines a clear interface that must be implemented for something to count as a list, but a list in C has no such clarity. We can imagine many different ways of creating such a list.

Consider a simple implementation consisting of a struct with a data field and a pointer to the next node. Is this an abstraction? Well, perhaps it is lacking somethings. Consider now that we add various operations, (head, tail, prepend, etc) to our list. Is it now an abstraction? On the information hiding view, we’d need to ask what information is being hidden. Perhaps the answer is that all sorts of information is being hidden, structs aren’t really a thing in computers, which registers are being allocated is obscured, the abi of the function calls are unknown. But this is merely a confusion.

We did not ask tout court if there are abstractions here. We are asking if our C list is itself an abstraction. Is our C list hiding some information? Well as it stands now, it doesn’t seems so. Perhaps it might be argued that the head function “hides” its implementation, but on what grounds?  We have no specificied that anything is restricted in anyway. But there is one surefire way to take our C list and turn it into an “information hding abstraction”, to restrict access to the underlying struct.

Why does restricting access to the underlying struct now take our C list from the status of not abstraction to abstraction? Does this mean, that within the implementation of the C list, the list is not an abstraction, but when used by others it becomes an abstracton? I am sure some will bite this bullet, but here I think we have been led astray. A linked list implemented using a struct with a data and pointer field is an abstraction regardless of the protection of the internals. It is just what we might call a mere abstraction, an abstraction that wears its structure on its sleeves.

#### Leaks and Mere Abstractions

For the information hiding advocate it might seem that we have been a bit too hasty. Of course lists in C implemented in that fashion are abstractions, they are just leaky abstractions.  Consider a linked that provides access to the next node pointer for any node, but does not expose the underlying struct. By giving us access to this pointer, the information the linked list was attempting to hide leaks out. The same is true of exposing the underlying struct. 

But what are we to make of this suggestion? Something is an abstraction if it hides information, but sometimes that information can leak out. How much of that information can leak? If an abstraction leaks all of its information, is it still an abstraction? Then in what sense is abstraciton “information hiding” if none of the information is actually hidden?

#### Fixing the Leak

I don’t think there is an easy answer here. Imagine that someone creates a linked list and accidentally allows full access to the internals. This was not their intent and it is in fact a property of the build system that these internals are leaked. Did they now fail in making an abstraction? If they talk about the abstractions in their program are they now speaking falsly? This seems wrong. But yet, if we take seriously the idea that abstractions *must* hide information, it seems like a clear consequence.

One fix here would be to say that abstractions have to hide information, but that quantity does not have to be non-zero. While technically a fix for the problem, it does not have much to recommend it. Allowing that zero information is hidden, ruins the motivation behind the information hiding view. But perhaps there is another fix, something is an abstraction is someone intends to hide information through its use. Inadvertent failure to do so, is no longer a barrier to something being an abstraction. But is this retreat something we truly want? Should the status of an abstraction depend on the intention of its author? Do we need to inquire into the authorship of a piece of code to know if it offers an abstraction? 

## Sketching the alternative

As should be clear I find that the information hiding view has a number of difficulties. Simply pointing to these difficulties is I think progress in itself; though I am not suggesting that these difficulties are insurmountable. Advocates of the information hiding view should offer responses these problems, if only to make their view more precise. I am also not suggesting these two problems I meantioned are exhustive with respect to the information hiding view. But I do think they are quite problematic and any alternative view should offer better answers to these problems.

I attend to offer such a view and what follows for the rest of the paper will be the sketching out of this view. A sketch it will indeed be. Abstraction is a rich topic which deserves a book length treatment rather than a simple exploration in a paper. But for now, we will at least start to work towards putting clothes on the view.

### Abstractions are Relative

We will begin with a restriction I believe any adequate view of abstraction but contain and yet has been ignored (as far as I can tell) in the literature up to this point, abstractions are relative to some base. To ask the question, “is the number 1 an abstraction” in complete isloation is to ask a misformed question. In order to consider if the number 1 is an abstraction, we must be given a context. Is the number 1 an abstraction in set theory? It seems clear that it is. Is the number one an abstraction in C? It clearly isn’t.

Now this may be contended. “Of course 1 is an abstraction in C, it is actually represented completely different depending on the architecture of the computer it is running on”, but this isn’t an objection, merely a misunderstanding. The context here isn’t “is the number 1 an abstraction relative to computers”, it is in C. In C, the number 1 is not defined in terms of anything else, unlike in set theory. In C, 1 is a primitive concept.

We already saw this same complication when considering whether C linked lists hide information. Relative to computer as a whole, everything we work with in programming languages is an abstraction. Numbers are binary not decimal, operations are actually excuted out of order, memory is virtualized. When we work in languages, we are presented with abstractions relative to the computer that give us various features. But within these languages themselves, we are offered elements as primitives.

