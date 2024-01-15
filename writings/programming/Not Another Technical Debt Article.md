# Not Another Technical Debt Article
There are three types of technical debt articles:
1. Taxonomies of Technical Debt
2. Redefinitions of Technical Debt
3. Technical Debt is (Good|Bad)

None of these articles ultimately help you with technical debt. Instead, they continue our endless dialogue around technical debt, perpetuating a programming culture that loves to talk about technical debt but does nothing about it. Let’s look at each of these article types in turn.

## The Taxonomy

Taxonomies aren’t wholly bad. They can help separate distinct concepts allowing you to divide and conquer. And yet, taxonomies can also mislead us. It seems very unlikely that subsections of Technical Debt form a [natural kind](https://plato.stanford.edu/entries/natural-kinds/) and so we must recognize that any divisions we want to make are going to be intermixed with our own desires and goals. Taxonomies created by others may not include your own or your teams' concerns. These ideas can then blind you from seeing categories that might be more useful in your local socio-technical environment.

## The Redefinition

Redefinitions come in two varieties. Those embracing and extending the debt metaphor, and those that completely reject it. But what they share in common is they want to claim a "folk" understanding of technical debt is incorrect. These articles typically do very little to help us understand the things we are calling technical debt. Whether the metaphor holds or not is beside the point. Nor do we need some crystal clear definition of technical debt to recognize it or do something about it. Instead, these articles simply increase the amount of bikeshedding over what is and isn’t technical debt.

## Declaring Normative Status

Articles that focus on the goodness or badness of technical debt do nothing to move the dialogue forward and nothing to solve the problem we are hoping to solve. Of course, no article actually states in absolute terms that all technical debt is bad or all is good. Instead, they combine elements of the last two types to give you a taxonomy and claim that those elements fall into two categories, good and bad. Or they redefine technical debt to be only the good/bad parts of the concept. In this way, these articles reduce to the former categories.

## How to Deal with Technical Debt

Regardless of which type of technical debt article you are reading, they almost always have some section like this one. How should we deal with technical debt? The advice is nearly always in three parts. 

1. Tell your stakeholders about all the info in this article
2. Categorize all your technical debt and put it in priority order
3. Get dedicated time to tackle it (almost always 20% of your time/points)

This is nearly always a recipe to never deal with your problems. The only way you can tackle technical debt is to *just fix it*. Perhaps this sounds like non-advice. Being told just to fix technical debt might seem meaningless, but I have found it to be anything but.

As software engineers, our job is not to "do what we are told" or "build what the business decides". Our job is the build good software. Good software isn’t defined by some overly simplistic metrics, but by looking at the whole of the product. If we recognize technical debt, we ought to fix it. Some bits of technical debt will be easy to fix. They might include things like updating a dependency everyone has neglected to upgrade which is now causing slowdowns in other parts of development. It might be fixing up an n^2 algorithm to be linear. But some of these issues might be much larger. 

That doesn’t mean we need to go ask the business for dedicated time to fix them. That means we have to think through them, come up with a plan, and work our way to the end state. We don’t need cards to track this work, we don’t need estimations, initiatives, a call out on the roadmap, praise, or recognition by higher-ups. Some of these are good things, but they aren’t about the problem, they are orthogonal to it. What we need is the persistence and willingness to fix the problems that we see.

## Social Dynamics

Most technical debt action plans have one failure mode, bikeshedding that leads to nothing getting fixed. This plan has a different one. Fixing "problems" without agreement that what is being changed is a problem. If you follow this plan of just fixing, you need to be sure you don’t ignore this. Being right is not all that matters.  It is tempting to fix things you believe strongly are issues even if no one else sees them as such. 

In these cases, you must proceed slowly. Show the problem, show the solution. Try and convince those around you to see things your way. Don’t foist your solutions on others. Further, be willing to admit when you are wrong. Your solution might not be the right one, the "problem" you identified might not be a problem. Don’t mistake change for progress.

## Fixing Things as a Way of Helping Others

Time and time again people don’t fix problems they recognize out of fear. What if someone questions why I’m working on this rather than the card I was assigned? What if other engineers think I’m wasting time? What if I can’t fix this problem and end up spending so much time and accomplishing nothing? All these fears are completely understandable.

Yet in my experience putting those fears aside leads to a reward that is greater than the risk. These fears hold us back, but just like the fear of raising your hand in class to ask a "stupid" question, once you finally do it you see the fear was unfounded. Minimally, you get a better understanding. But more often than not you find that others had the same issue. This is what fixing technical debt does, it brings us closer as a team. It helps others see that you too struggled to understand a piece of code. You too had problems making that build process run. Fixing things helps bond us together. It is an act that helps unite and ultimately helps others succeed.

We can move past the bikeshedding. We can fix our broken systems. We can have codebases that are enjoyable to work in. It just requires us to stop considering the "meta-work" involved and get involved directly with the work of fixing it.
