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

But while an analysis must pay attention to the linguistic data, it is not necessarily beholden to it. It should  