# What follows from empirical software research?

There is a growing body of literature studying software engineering from the empirical perspective. There is perhaps no greater resource for discovering this content than [It Will Never Work in Theory](https://neverworkintheory.org/). Here you will find short reviews of countless empirical studies about software engineering. Some reviewed quite favorably, and others questioned quite vigorously.

What I want to do here is not question any of these studies, not even survey their results. Instead, I want to ask a meta-question, what follows from these studies? Assume for a second that a study of deep relevance to practitioners is replicated, and its conclusions accepted by the research community. It has large sample sizes, a beautiful research plan, a statistically sound system for controlling for various other explanatory factors; whatever it is that we need to proclaim it to be a good study. (If I got the criteria above incorrect, please substitute your own). Given all of this, we have on our hands some sort of scientific fact. What should we do with it?

## Modern TDD Research

As a starting point, we will look at [Greg Wilson’s review of “Modern Software Engineering”](https://neverworkintheory.org/2022/05/06/not-quite-modern.html). Here we find a beautiful confluence of so much. First Wilson tells us that despite its title the book “feels a bit old-fashioned”. He specifically points to the book's lack of inclusion of more recent research regarding TDD. The book cites studies that claim TDD helps quite a lot. But Wilson points us to a meta-analysis that found the following:

> (a) the more rigorous the study, the weaker the evidence for or against TDD and (b) overall, there is no evidence that TDD makes anything better or worse

Not only that, a more recent (and Wilson adds “very careful”) study found very similar results.

> (a) there's no difference between test-first and test-after but (b) interleaving short bursts of coding and testing is more effective than working in longer iterations

Specifically, this study found the following:

> The Kruskal-Wallis tests did not show any significant difference between TDD and TLD in terms of testing effort (p-value = .27), external code quality (p-value = .82), and developers' productivity (p-value = .83). 

Just to head off an objection I might get from TDD advocates, Wilson himself is a fan of TDD he writes in his review of this more recent study the following:

> I don't program very much anymore, but when I do, I feel that writing tests up front makes me more productive. I'd like to be able to say that these researchers and others must be measuring the wrong things, or measuring things the wrong way, but after so many years and so many different studies, those of us who believe might just have to accept that our self-assessment is wrong.

## What follows?

Let us set aside the conclusions of these studies for a moment. Let’s instead play in our imaginary world taking each possibility in turn. These possibilities are as follows:

```
P1: TDD has a positive effect on testing effort, external code quality, and developers’ productivity.
P2. TDD has no effect on testing effort, external code quality, and developers’ productivity. 
P3. TDD has a negative effect on testing effort, external code quality, and developers’ productivity.
```

As practitioners, we might want to know how we should change our practice given these various empirical findings. For example, we might ask the question “Should I practice TDD?”. Do P1, P2, or P3 give us any answer to this question? For shorthand and clarity, we will turn P1, P2, and P3 into a function of the effect P(positive) picks out P1, P(no) P2, and P(negative) P3. Now we can take these as premises in the argument and see if we can find a logically valid argument that answers our question.

```
P(positive)
Therefore I should practice TDD.
```

Hopefully, we can all see this is an invalid argument form. How can we make it valid? Well, quite easily, we can simply use Modus Ponens.

```
P4: If P(positive) then I should practice TDD
P5: P(positive)
C:  Therefore I should practice TDD
```

So now we have a valid argument! And we can imagine pretty easily how to get the rest of the arguments depending on the conclusions of our empirical studies (well `no` could be a bit tricky, but set it aside for now). So problem solved right? Well not quite. How can we justify premise P4? Isn’t P4 the exact question we were trying to ask? We can’t assume P4 is true, so while this is a valid argument we have no reason to believe it is sound.

So what would it look make a logically valid argument for p4? Well, we have actually come upon a classic philosophical problem, the is/ought problem. This was a question raised by David Hume where (on some readings) he showed that there is no way to go from a statement of fact “is” to a logical conclusion of what one ought to do. Not wanting to get off topic we will not dive into this issue. But, it is important that we have this difficulty in mind. 

### A Practical Argument

Let’s perhaps loosen our grip on searching for logical soundness and instead try to aim for a bit more practical argument. Kant makes a distinction between what is called a “Categorical Imperative” and a “Hypothetical Imperative”.  Basically, a Categorical Imperative is an ought statement that is true regardless of our intentions or desires. “Everyone ought to respect human dignity”. A Hypothetical Imperative is relative to our desires “If I want to cook at home tonight, I ought to go grocery shopping”. Hypothetical imperatives are what we use to reason about achieving our goals.

So we can make a better argument using this hypothetical.

```
P6: If you care about testing effort, external code quality, and developer productivity, you should practice TDD if p(positive) is true.
P7: P(positive) is true
C:  Therefore, if you care about testing effort, external code quality, and developer productivity then you should practice TDD
```

Perhaps not the most elegant syllogism, but I think it gets the point across.  Is this a good practical argument for TDD given the empirical evidence? Well, no. You may care about these metrics, but perhaps not as much as you value other aspects of your life. Perhaps you really don’t enjoy TDD at all and would rather enjoy your work. Perhaps it actually gives you anxiety or demotivates you from wanting to work. Perhaps your boss hates TDD and if they find out you are doing it will fire you. There are any number of confounding factors that can make the first premise false. 

We could continue to Chisholm this argument by adding clauses about “all things being equal” or something like that. But I’m not sure we’d benefit from that. As we made our claim narrower and narrower we’d move further and further away from the idea we are interested in “How should we change our practice given these various empirical findings?”.  It seems that for the positive case the answer is it depends. It depends on our goals, desires, etc, and if those align. But what about the negative or neutral case?

For the neutral case, it is hard to see how it would be any different. If TDD made no difference in terms of the criteria listed, then it is a matter of preference. But what about a negative finding? Don’t we have a greater obligation to not negatively impact testing effort, external code quality, and developer productivity? Let’s assume we do, does it follow that we ought to not practice TDD? Well, no. Perhaps TDD does lower these metrics, but there may be other metrics we care more about which TDD improves. We may have the obligation, but we might have other conflicting obligations. Perhaps TDD lowers those metrics, but it is also the only way we can motivate ourselves to program at all. Further, let’s supposed our obligation to program something is greater than our obligation to not negatively impact these metrics. Here, we are obligated to practice TDD despite these empirical findings of its negativity.

### What we can say, believe, and know

So even on the contentious assumption that we have certain obligations towards not negatively impacting these metrics, it doesn’t follow from the empirical evidence that we ought not practice TDD. So what could follow? Well, we might get a hint from Greg Wilson’s quote above about his sadness regarding the conclusion TDD research has come to.

> those of us who believe [that TDD makes us more productive] might just have to accept that our self-assessment is wrong.

Perhaps this is what follows. Rather than any particular action being obligated by TDD, we must have a change in belief structure. Well, there seems to be something clearly right about that. Let’s say that before I read the research I believed P(positive). If the research actually says P(negative) and the research meets my evidentiary standards, it seems I should believe P(negative). But it must be pointed out that this is markedly different from a statement like Wilson made.

P(negative) is a statement about the statistical effect on various metrics by TDD. What it isn’t is a statement that no one person can be more productive when practicing TDD. In other words, we are not warranted in the claim that “I can’t be more productive with TDD” follows from P(negative). Perhaps instead we should believe the following “given P(negative) there is a low probability that I am more productive with TDD”.  But do we even have a basis for this conclusion?

It is hard to see how we would. To make this clear, let’s imagine that it has been shown in a statistically rigorous way, using all the criteria above that dogs reduce stress in people. Let’s call that find D. So equivalently should we believe the following “given D there is a low probability that dogs stress me out”? Well, it is hard to know what to make of that. Do we mean low probability as in, if I am stressed by dogs I am in the minority? Or do we mean an epistemic probability, as in I should conclude that I am likely not stressed by dogs? Well, the first may be true, but the second could clearly be false. Perhaps I know for a fact that I am very afraid of dogs and that I get incredibly stressed when I am around them.

So if by “given P(negative) there is a low probability that I am more productive with TDD” we mean, I am in a minority, that is a fine belief. But if we mean that I should conclude that I am likely not more productive, we have made a confusion. We can’t conclude anything about this situation without looking at our background beliefs and past experience. Wilson (and those like him) should rest easy. We need more than just empirical research to show that we ourselves are not more productive with TDD.

Now I fear what I just said will be misunderstood. Imagine this, someone does an empirical study on only me. They measure my productivity in a way I agree is a good measure, their procedures are solid, and everything is up to my standards. Imagine now that they find that I am not more productive with TDD. Is what I just said above license to ignore the finding? Of course not. The point is that we can’t go from a statistical case to a personal conclusion. Not that empirical evidence has no barring.

#### TDD advocacy

Perhaps instead of a change in belief, we find a change in behavior around advocacy. If we find that P(negative) is true, perhaps we shouldn’t advocate for people to practice TDD. Yet again it is hard to see how that would be the case. TDD may have other benefits, we may have other goals with our advocacy of TDD other than the metrics listed in P. What does follow is that we can’t advocate for TDD by stating that P(positive) is true. That would be disingenuous.

Perhaps it doesn’t obligate us to no longer advocate for TDD, but shouldn’t it lower the intensity of our advocacy? Maybe we shouldn’t be as strident advocates as we once were. Well, that depends. Was our prior level of intensity determined in a substantial way by a belief that P(positive) was true? If so, then yes, it seems we should lower our level of intensity. But without that, no we are under no such obligation. 

#### Know vs believe

Perhaps all that has changed is a relation to the proposition. Imagine before reading the research we believed that P(positive) and after reading the research we find that in fact, the research shows that P(positive) is true. The suggestion made by some advocates of empirical software methodology is now we have gained something. Before we just “believed” that P(positive) was true, now, we “know” that P(positive) is true. Perhaps this is what follows, a much better world based on knowledge rather than “opinions” or “superstition”.

What is knowledge? Well for about 2000+ years it was generally regarded in the western philosophical tradition to be “justified true belief” until [Edmond Gettier showed this definition to be false](https://fitelson.org/proseminar/gettier.pdf). So the definition is a bit up in the air. For our purposes those, we can go with this definition: “Knowledge is warranted, true belief”. Hardly better if we don’t know what “warrant” is. But again, this isn’t a philosophical text. Warrant in our case is just something that turns a belief into knowledge.

So in this case, it is supposed that before we read the research our belief that P(positive) is true was not warranted. After we read the research our belief becomes warranted and hence is knowledge. But why think that? Perhaps we think that warrant means “has adequate evidence for” and before we read the research we lacked adequate evidence. Unfortunately in Gettier’s paper he actually shows that this definition isn’t sufficient. So just because we may have gained adequate evidence, doesn’t mean we for sure know P(positive). But I won’t argue that here, it seems pretty likely that given our belief that P(positive) is true and our acceptance of the empirical evidence for P(positive), we do in fact know P(positive).


Well, one way of taking P(positive) is a rather particular claim. It is a statistical claim about the effect of TDD on some population of engineers. Taken that way it seems unlikely that before reading the research you believed that P(positive) was true. What we probably believed was something a bit more vague than that way of taking P(positive) something like TDD has the ability to increase code quality, productivity, and reduce testing effort for some people in some circumstances. Can we know a claim like this without a study? Can we know that dogs relieve stress for some people in some circumstances? It seems we can.

Perhaps you think I am being too kind to TDD advocates. Perhaps they believed before the research that TDD would improve these metrics for most people most of the time. Perhaps then they weren’t warranted in that belief. But after the research, should they now claim to know that belief to be true? In other words, is it now warranted? Well, it would seem not. Because that isn’t what P(positive) says. It is a claim about a statistically significant effect. It is much more precise than the prior belief.


What doesn’t follow is a move from “superstition” and “opinion” to knowledge. We can accept a study and yet still believe something on the basis of “superstition” and “opinion”. We can have knowledge of many things without having an empirical study. To suggest otherwise is to misunderstand how knowledge works and to overly simplify the noetic structure of human beings. We can all agree that believing things for the right reasons is good. That evidence is good. But these claims about “superstition” or “opinion” are much too strong.

## So What?

We’ve covered a number of things that don’t follow from the results of P(x). Do these results generalize? For example, many fans of empirical studies point to code review as an example of a great empirical result. It turns out code review is very effective at reducing bug count. Doesn’t that mean we should do code review? By itself? As I hope I’ve shown, no that doesn’t follow. But neither does the idea that we shouldn’t do code review. 

Empirical results give us a data point to use in our practical reasoning. They guard us from overly grandiose claims. They can cause us to rethink them. But we aren’t warranted in moving from research suggests X has Y benefits, to therefore we ought to do X. In fact, I will make a stronger statement. No empirical finding about software alone should make us conclude anything about what we ought to do. What we ought to do depends on facts like our background beliefs, our desires, and our particular circumstances.

So why care about empirical results at all? Well, I enjoy knowing things. I think the more we understand the better. But do empirical results have a large role to play in questions like the following?

* Should we use static or dynamic types?
* Should we practice TDD?
* Should we do agile development?
* Should we use a memory safe-language?

Alone they cannot. Only when our goals, desires, and background beliefs align with what the research shows does it at all follow that we have any answer to these questions. These questions must be answered by individuals. There will be no global ought for the industry.

Just as many advocates of empirical software studies implore us, we must not be taken in by salespeople, by the sophists who push their own solutions. This includes the appeal to the rigor of empirical studies. We must not let ourselves be led astray by rhetoric like “superstition” thrown around as an attempt to cast negative aspersions. Empirical study does not exhaust rationality.

# Predicted questions

**Q:** Do you just hate empirical software studies?

**A:** No. I have nothing against it at all. Studying the world is great. I just see a common implicit conflation of what science tells us is true and what we ought to do. These things are not one and the same and we should pay attention to that.

**Q:** But if science tells us X is more productive, shouldn’t we just do it? We want to be productive.

**A:** We do want to be productive, I guess. But we also want to be happy. We want to get along with our coworkers. We (I) want to drink nice coffee. We have all sorts of desires and these can conflict. I personally don’t care that much about productivity. It is not at the top of my list. I am fairly productive on the things I want to be productive on.

**Q:** But if science tells us X increases code quality, that just seems like a clear win.

**A:** What I said above still applies. But I also want to take this opportunity to remove the idealistic assumptions we’ve made. We have to be careful about the meaning of words here. The meanings of “external code quality”, “testing effort”, and “developer productivity” in the study mentioned above have massively different meanings from what I take the ordinary meanings of those words to be. Perhaps I am mistaken and others would see them as tracking the ordinary definitions. But I will say, I don’t particularly care about the attributes they have measured.

**Q:** Isn’t this just a sophisticated way of saying you are going to ignore evidence?

**A:** I’m not sure how it could be that. I have no skepticism toward general scientific consensus. Nor would I have skepticism towards the findings of empirical software studies. My point is just that we have to do an extra step in thinking about applying these findings and that extra step involved judgment, weighing of goods, and introspection of our desires.

**Q:** Sure, as an individual you might not want to do these things. But shouldn’t businesses use this to guide their decision-making? Businesses desire productivity and efficiency, anything that has been shown to produce that should be adopted and enforced in a place of business.

**A:** I just think that is too simplistic of a picture. The “scientific management” of John Taylor showed all sorts of results. Perhaps you think they don’t have the scientific rigor of today, but I don’t think that matters. Businesses are complicated, complex systems. Simple-minded, thin rules aren’t going to solve our problems. 

**Q:** But isn’t this whole thing a strawman? Consider project Aristotle from google. Here we see scientists making direct recommendations on how we “ought” to act. They measured all of these things you are talking about, desires, goals, etc, and found concrete real recommendations.

**A:** If anything that reinforces my point. Project Aristotle is pretty careful in parts to be clear that their recommendations are relative to the environment of google. For example, they note that team size didn’t seem to matter at google, but did elsewhere. In other cases, they are much less careful, for example in their recommendations around OKRs. 

Perhaps the suggestion here is that research that takes goals, desires, etc into an account can offer “oughts” and so there is no is/ought problem in the neighborhood. Let’s assume they can legitimately off these oughts. They can only do so as a hypothetical imperative, “if you want X you should do Y”. But that just brings us back to the same question, do I want X? Is there some other thing Z that I want more than X that conflicts with my attaining X via Y? We haven’t escaped the problem.

**Q:** Doesn’t leaving these things up to judgment just mean we will end up with no progress?

**A:** Does judgment never give us progress? Was C an improvement on BCPL because of scientific empirical study? Did the early pioneers of personal computing use their judgment to bring us a future others didn’t see? Further, are the sciences devoid of judgment? I don’t think there is any reason to think allowing for judgment prevents progress, in fact, I think (but have not argued for) the opposite.

**Q:** Wouldn’t it be better to just follow what science has shown, even if it is suboptimal rather than leaving it up to the whims of people?


**A:** I think we need to be clear that judgments and whims are not the same things. In our everyday life, we can see a difference between acting on a whim and making a considered decision. The same is true in programming and business. I personally don’t think it would be better to just “follow the science” (whatever that means) rather than use our judgment. Partially because I don’t think there is anything called “following the science” that is devoid of judgment, so if we think we are doing that we are deceiving ourselves.