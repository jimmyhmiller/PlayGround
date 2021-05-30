# Abstraction

Abstraction is perhaps the most used, least defined word in software. Now this may strike you as suprising, as I'm sure you believe you know a definition of abstraction. Yet, what I hoped to show is that this common definition is insufficient to account for the ways we use "abstraction". I don't mean to suggest I have some better definition. Instead I want to call people to examine abstraction a bit more closely. Abstraction is fundamental to programming, and yet, as I hope to show, we don't understand it.

Now even before we get to this common definition of abstraction that I find inadequate, there is a problem in specifying what sort of abstraction I'm talking about. Since the point of this essay is to show that we don't quite understand abstraction, how can I point to what it is I am talking about? Well, we can start with what I don't mean, and further provide examples of the sort of speech I hope to cover.

Firstly, by abstraction I am not talking about function abstraction in the lambda calculus sense, nor any similar formal definition. I am instead looking to discuss more informal notions. Examples of this notion of abstraction include things like:

- Lists are an abstraction.
- That is a leaky abstraction.
- Programming is about finding the right abstractions.
- This code is too abstract.
- Monads are a great abstraction for asynchronous programming.
- Monads are so abstract no one can understand them.
- That isn't real, it's merely an abstraction.
- We need to think about this at the right level of abstraction.

Now technically not all of these sentences mention abstraction, some instead use the word abstract. Perhaps these aren't related, I think otherwise. I find it rather hard to imagine code that is "too abstract" that is not full of abstractions. Given that, let's look at the standard definition of abstraction and see if it really matches up with these sentences above.

## The Standard Definition

The standard definition given for abstraction has something to do with "information hiding". Typically what people say the mean by abstraction is that by making an abstraction we are hiding details that don't matter. So for example, if a java programmer says that lists are an abstraction, what they mean is that lists can be implemented multiple ways. In Java there is "List" the abstraction and then there are implementations of "List", ArrayList, Stack, LinkedList, etc.

So far so good. And this definition works quite well for some of the items on the list. Take "That is a leaky abstraction". This can mean that some of the information you were trying to hide did not stay hidden. We can see the implementation details through the abstraction. And yet, despite its initial success, it doesn't actually cover all the examples above.

### Information Hiding and Comparisons

To see where this definition falls apart, let's start with the comparison statements in the list above. "This code is too abstract". So, if we are to take the information hiding abstraction seriously, it would seem that this code is "too abstract" must mean that it hides more information. It would be rather weird for one abstraction to be "more abstract" than another while hiding less information. And yet, this is precisely what we see.

Monads are actually a great example here. Let's start with some hopefully uncontroversial statements.

1. Functions are abstractions.
2. Monads are abstractions.
3. Monads are more abstract than functions.

I think most people will find themselves agreeing here, and yet when we ask the further question:

> Which hides more information, a function or a monad?

It is fairly obvious that the answer is the function. Given some arbitrary function and some arbitrary monad, we can tell you a lot more about the monad. You see, the monad hides less, because it makes more guarantees. We know what operations it has and what laws it follows. But the same isn't true about the function. So if information hiding is the essence of abstraction why do our comparisons fail to uphold that relationship?

### Information Hiding and Mere Abstraction

But comparisons aren't the only place where the standard definition falls short. Take for example "This isn't real, it is merely an abstraction". What exactly could this mean if abstraction is information hiding? What real thing are we constrasting this with? Perhaps you think this sentence is unrealistic. Take instead "A list is an abstraction". Now imagine that a Java programmer said this. Would they mean something different by that sentence than a C programmer saying the same thing? What about "A list is a mere abstraction"?

I think we can all make sense of the notion of a "mere abstraction". Lists can be abstraction in two senses. The first, is the java sense. Lists support certain operations and anything that supports this can be called a list. In other words, an abstraction is some functional definition. In the C sense, what we mean is not that there is some interface. Instead, we are saying more or less that lists are an illusion. Lists are nothing more than pointers.

A list being a mere abstraction isn't about it hiding information. When we consider lists in C, we aren't hiding the fact that they are pointers. We can keep all the internals of our list exposed, force users to do the pointer chasing, and yet still call lists an abstraction.

### Abstractions Relative to a Base

But inadequacy of defining abstraction as information hiding doesn't just end with our linguistic uses of the word. If we instead consider how we would answer certain questions about what is and isn't an abstraction, I think we will find a surprising feature about abstractions, they are relative to a particular base we are talking about.

Consider first integers. Are integers an abstraction? Well, it really depends on the context. In c are integers an abstraction? I think most of us would say no. Now, we could try to argue about how the computer really operates on electrical signals, and so integers an abstraction. But that is to ignore the context we setup. We are talking about in C, not in a computer.

What about in the lambda calculus? Are integers an abstraction here?  I think the obvious answer is yes. Is that because we have hidden some information? No, the lambda terms are transparent, nothing about their structure is hidden.

These examples abound. Not only are certain things abstractions or not relative to a base, but it is clear they live on different levels of abstraction relative to a base. Compare functions in haskell to functions in java. Regardless of if you think functions are an abstraction in haskell or not, it is very clear that java functions in java, are more abstraction that haskell functions are in haskell. Now this way of putting it is a little awkward, but my point is that if we look at the layers of abstraction in each language, we can see that functions in java are on a much higher layer. Haskell functions are very near or on the base layer. Java functions have several layers above them.

## Why This Matters

If you as pedantic as I am, perhaps you see immediately how important this fact is. We have been assuming we understand a notion, when our practices betray the fact that our supposed understanding is wrong. Incorrect understanding invariably leads to incorrect application. But what is even worse than incorrect understanding, is thinking stopping any investigation into the issue because everyone believes its settled. This is precisely what we have done.

Abstractions are incredibly important to programming. I don't think anyone would dispute that. If we don't understand what abstractions are, how we can we know their limits? How can we reason about which abstractions are good or bad? How can we we advance programming if we have misunderstood such a fundamental part of it?

