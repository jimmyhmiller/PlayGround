# Programming and Theory Building

If we are to take this idea of programming as theory building seriously, we need to think about the implications it has. I think this view is actually quite a bit more radical than it may seem at first glance. We must consider the ways in which the business can be organized to hinder or support this activity. We have to think about how our practices can limit the theory building we are capable of. We have to consider how lines of communication can prevent others from gaining their own theories, or how they can even cause them to gain inaccurate theories.

Think about how you would organize a group of people if you believed their primary activity was not outputing an artifact but instead building theories. Would processes like Kanban and Scrum make sense? I think the answer would be a clear no. I’ve detailed elsewhere some of the problems that these methodologies face, so I won’t dwell on it here. Instead what I want to suggest is that we need to rethink our processes starting from the idea that they are about building theories. This will be a rather difficult task. Because as opposed to building a concrete object, building a theory is not a straight-forward endeavor. 

## Accepting the Messiness of Theory Building

Building a theory requires lots of experiments. It is full of false-starts, dead-ends, and unexpected turns. Further, the journey needed to expand the theory is only clear to those who already know the theory. What problems exist, what solution are feasible, what avenues of exploration will be fruitful are completely opaque to those outside the community that shares that theory. Given this, we must admit that having an external group that is in charge of what the theoreticians ought to work on next seems rather counterproductive.

There is of course a danger here. As soon as we say that some group should not have their work decided for them, we tend to make certain assumptions. Am I here arguing that the theoreticians are smarter than others? Am I arguing that they are superior or ought to be in charge of others? Not at all. I in fact want to resist this conclusion at all costs. It is not that some group of people is better or worse or should have more or less power. It is that we have areas of specialization. We deal with complex problems that only those on the inside can truly understand.

### Understanding the Complexity of Building a Theory

What sort of complexity are we talking about? I want to be clear that when I talk about complexity here, I do not mean the complexity of programming languages, but of the concrete system we have built and are building. The theories we are building as software engineers are not generic theories about computer science fundamentals, but instead radically specific theories about the actual code bases, processes, and systems we work in. We are looking to take our business domain (aka some part of the world) and our system and try to find a way to fit these two together. 

The interesting thing about software engineering is that our theory building includes building in two directions. These directions have been called the “world to mind fit” and the “mind to world fit”. In the former, we are trying to shape the world to fit the way we think of things. In the latter we are trying to fit our mind to the way the world is. Engineering clearly involves both. We start with the given, with the facts, the immutable truths, the limits, the theorems, and we attempt to create.

But software engineering in particular makes these lines rather messy. Unlike the physical world, the digital world is incredibly malleable. Changing a program is a seemingly trivial thing. And yet, as every software engineer knows, our programs don’t just easily yield to change, instead they push back. The smallest changes can cause a cascade of failures. The consequences of our change might not be seen for days, months, or years down the line. Changes made in absence of a theory are bound to cause issues.

### Achieving Fit

These two directions of fit are what determin this complexity we face as software engineers. We might find a way to satisfy both directions at once. We must understand the world enough to begin to create the world we need. But now that our creation is part of the world, the world as we have known it has changed. It would almost seem that since we knew the world before and we know our creation, we should know the world + creation. And yet as anyone who has caused a production outage from deploying codes knows, this could not be further from the case.

This is where taking the theory building approach helps. First, it helps us understand that just because the code is done, doesn’t mean that we are, and it doesn’t mean that we ought to ship it. Second, it makes clear why these things would surprise us. Complex theories involve complex interconnected relations. There is no simple addition to a theory. Instead as we add new entities, we must understand they ways in which they could interact with all the parts.

Achieving this fit is difficult. It involves trying out code, evaluating its inclusion in our theory and deciding whether this modification makes sense. It also involves keeping track of all the changes going on to the world around us. The changes done by our fellow engineers. (Given this lens does code review serve a different purpose?). But the other important component of achieving fit is deciding what in our mind we want the world to fit to. What should software be? How should it operate? We are not merely creators, we are designers, we are activists, we are product. 

## Never Achieving Fit

There has been a lot written about software projects that fail. How is failure defined here? Typically it is code that fails to ship, or “ships” but fails in some spectacular way. Instead of talking about this, I want to talk about a failure to build a theory. Not because someone didn’t try, or wasn’t given time, but because “world to mind fit” was never achieved. Since world to mind fit is about shaping the world to fit our mind, wouldn’t a failure to achieve fit just be a failure to write the code we needed to write? Well, it could, but I want to talk about a more subtle issue, where code ostensibly works for the business, but fails to fit for the engineer. You might know this by another name, “hacks”.

Here I mean hacks in a very broad sense. A whole entire system could be a hack. Further, hacks by definition here are person relative. It is this person relative nature of hacks that makes them invisible and ignored. They become to many a necessary cost of getting the business value they want in the time they want. But if we take the theory building view that what matter is not the code but theory being built, we start to see how damaging they can be.

Software engineers who are forced to constantly hack, fail to ever achieve building a theory. Their efforts are thwarted not by a lack of understanding of the problem, but because their desires cannot be fulfilled. They have a world to mind fit they want to achieve. They have goals, they have convictions about how the world ought to be and are being forced to make the world be otherwise. Because of this, the theories they attempt to create become muddled. They are conflicted between the beautiful vision in their head of what this theory could be, and the harsh reality they’ve been forced to inhabit.

## Constraints on Organization

Given this complexity, given these two directions of fit, how can company organizations help or hinder theory building. The first rather radical constraint we see is that imposing of goals on software engineers can actually stop you from getting the main end product you should truly be after, a theory. Now this isn’t to say that you can’t give direction, cast a vision, convince others. In fact, this exactly what needs to be done. Getting engineers to agree with you on how the world should be, is the first step to ensuring they build a theory that fits with your own desires.

Further, we must recognize that any organization structure that keep knowledge out of the hands of the engineers can stunt the theory building process. Building a theory requires full knowledge of the way the world is. This information needs to be as raw as possible. Being fed information through [cards](/card-driven-development) that contain information already filtered through some theoritical lens, will not help a theory be built. Nor will preventing engineers from truly understand the customer needs or not allowing them to understand how other engineering teams at the company work.

Finally, it must be recognized that if engineers are constantly unhappy with the code they are writing, constantly unhappy with the system they must work in, they will fail to form a theory of any sort. Companies that keep this going for years on end are throwing money down the drain, never achieving the true end of what they are paying for, a theory built by a software engineer. Instead, they get people who stay for 1-2 years before moving on, hoping finally for that shop that understands. Hopefully finally (perhaps no in these worlds) that they can properly build a theory. That they can not only achieve world to mind fit, but finally make a world they want to live in.





