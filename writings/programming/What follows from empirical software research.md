# What Follows from Empirical Software Research?

There is a growing body of literature studying software engineering from the empirical perspective. There is perhaps no greater resource for discovering this content than [It Will Never Work in Theory](https://neverworkintheory.org/). Here you will find short reviews of countless empirical studies about software engineering. Some reviewed quite favorably, and others questioned quite vigorously.

What I want to do here is not question any of these studies, not even survey their results. Instead, I want to ask a meta-question, what follows from these studies? Assume for a second that a study of deep relevance to practitioners is replicated, and its conclusions accepted by the research community. It has large sample sizes, a beautiful research plan, a statistically sound system for controlling for various other explanatory factors; whatever it is that we need to proclaim it to be a good study. (If I got the criteria above incorrect, please substitute your own). Given all of this, we have on our hands some sort of scientific fact. What should we do with it?

## Modern TDD Research

As a starting point, we will look at [Greg Wilson’s review of “Modern Software Engineering”](https://neverworkintheory.org/2022/05/06/not-quite-modern.html). Here we find a beautiful confluence. First Wilson tells us that despite its title the book “feels a bit old-fashioned”. He specifically points to the book’s lack of inclusion of more recent research regarding TDD. The book cites studies that claim TDD helps quite a lot, but Wilson points us to a meta-analysis that found the following:

> (a) the more rigorous the study, the weaker the evidence for or against TDD and (b) overall, there is no evidence that TDD makes anything better or worse

Not only that, a more recent (and Wilson adds “very careful”) study found very similar results.

> (a) there’s no difference between test-first and test-after but (b) interleaving short bursts of coding and testing is more effective than working in longer iterations

Specifically, this study found the following:

> The Kruskal-Wallis tests did not show any significant difference between TDD and TLD in terms of testing effort (p-value = .27), external code quality (p-value = .82), and developers’ productivity (p-value = .83).

Now it may seem like Wilson has some sort of anti-TDD agenda here, but Wilson is a fan of TDD. He writes in his review of this more recent study the following:

> I don’t program very much anymore, but when I do, I feel that writing tests up front makes me more productive. I’d like to be able to say that these researchers and others must be measuring the wrong things, or measuring things the wrong way, but after so many years and so many different studies, those of us who believe might just have to accept that our self-assessment is wrong.

This review sets up a perfect context for our discussion. How are we to understand results like this? Particularly in relation to our own feelings about practices. How should empirical results affect the way we program?

## What Should We Do With Research Results?

Let us set aside the precise conclusions of these studies for a moment. Let’s instead play in our imaginary world taking each possibility in turn. These possibilities are as follows:

```other
P1: TDD has a positive effect on testing effort, external code quality, and developers’ productivity.
P2. TDD has no effect on testing effort, external code quality, and developers’ productivity. 
P3. TDD has a negative effect on testing effort, external code quality, and developers’ productivity.
```

As practitioners, we might want to know how we should change our practice given these various empirical findings. For example, we might ask the question “Should I practice TDD?”. Do P1, P2, or P3 give us any answer to this question? For shorthand and clarity, we will turn P1, P2, and P3 into a function of the effect P(positive) picks out P1, P(no) P2, and P(negative) P3. Now we can take these as premises in the argument and see if we can find a logically valid argument that answers our question.

```other
P(positive)
Therefore I should practice TDD.
```

Hopefully, we can all see this is an invalid argument form. How can we make it valid? Well, quite easily, we can simply use Modus Ponens.

```other
P4: If P(positive) then I should practice TDD
P5: P(positive)
C:  Therefore I should practice TDD
```

So now we have a valid argument! We can further imagine pretty easily how to get the rest of the arguments depending on the conclusions of our empirical studies. (A neutral statement might be a bit harder, but will ignore that for now.) So problem solved right? Well not quite. How can we justify premise P4? Isn’t P4 the exact question we were trying to ask? We can’t assume P4 is true, so while this is a valid argument we have no reason to believe it is sound.

So what would it look make a logically valid argument for p4? Well, we have actually come upon a classic philosophical problem, the is/ought problem. This was a question raised by David Hume where (on some readings) he showed that there is no way to go from a statement of fact “is” to a logical conclusion of what one ought to do. Not wanting to get off topic we will not dive into this issue, but it is important that we have this difficulty in mind.

### A Practical Argument for TDD

Let’s perhaps loosen our grip on searching for logical soundness and instead try to aim for a bit more practical argument. Kant makes a distinction between what is called a “Categorical Imperative” and a “Hypothetical Imperative”.  A Categorical Imperative is an ought statement that is true regardless of our intentions or desires. “Everyone ought to respect human dignity”. A Hypothetical Imperative is relative to our desires “If I want to cook at home tonight, I ought to go grocery shopping”. Hypothetical imperatives are what we use to reason about achieving our goals.

So we can make a better argument by utilizing a hypothetical imperative.

```other
P6: If you care about testing effort, external code quality, and developer productivity, you should practice TDD if p(positive) is true.
P7: P(positive) is true
C:  Therefore, if you care about testing effort, external code quality, and developer productivity then you should practice TDD
```

Perhaps not the most elegant syllogism, but I think it gets the point across.  Is this a good practical argument for TDD given the empirical evidence? Well, no. Sadly with hypothetical, practical arguments like this, there can be confounding factors. For example, you may care about these metrics, but not as much as you value other aspects of your life. Perhaps you really don’t enjoy TDD at all and would rather enjoy your work. Perhaps it actually gives you anxiety or demotivates you from wanting to work. Perhaps your boss hates TDD and if they find out you are doing it will fire you. The first premise will only be true if all of these confounding factors are controlled for.

We could continue to Chisholm this argument by adding clauses about “all things being equal” or something like that. But I’m not sure we’d benefit from that. As we made our claim narrower and narrower we’d move further and further away from the idea we are interested in “How should we change our practice given these various empirical findings?”.  It seems that for the positive case the answer is it depends. It depends on our goals, desires, etc, and if those align. Does this result generalize beyond P(positive) to P(negative) and P(no)?

### Negative and Neutral Cases

For the neutral case (P(no)), it is hard to see how it would be any different. If TDD made no difference in terms of the criteria listed, then how could it be anything other than a matter of preference?  But what about a negative finding? Don’t we have a greater obligation to not negatively impact testing effort, external code quality, and developer productivity? Let’s assume we do, does it follow that we ought to not practice TDD?

It seems clear that just as with the positive case, there can be confounding factors. What if TDD does in fact lower these metrics, but increases others we care about more? What if not practicing TDD causes us to feel anxiety? Imagine in the extreme case where we can’t even bring ourselves to program if we don’t do TDD. If we have an obligation to program, we would actually have an obligation to do TDD despite the empirical evidence telling us these metrics are negatively impacted.

## Other Consequences

So even on the contentious assumption that we have certain obligations towards not negatively impacting these metrics, it doesn’t follow from the empirical evidence that we ought not practice TDD. So what could follow? Well, we might get a hint from Greg Wilson’s quote above about his sadness regarding the conclusion TDD research has come to.

### The Personal Consequence

> those of us who believe [that TDD makes us more productive] might just have to accept that our self-assessment is wrong.

Perhaps this is what follows. Rather than any particular action being obligated by TDD, we must have a change in belief structure. Well, there seems to be something clearly right about that. Let’s say that before I read the research I believed P(positive). If the research actually says P(negative) and the research meets my evidentiary standards, it seems I should believe P(negative). But it must be pointed out that this is markedly different from a statement like Wilson made.

P(negative) is a statement about the statistical effect on various metrics by the practice of TDD. What it isn’t is a statement that no one person can be more productive when practicing TDD. In other words, we are not warranted in the claim that “I can’t be more productive with TDD” follows from P(negative). Perhaps instead we should believe the following “given P(negative) there is a low probability that I am more productive with TDD”.  But do we even have a basis for this conclusion?

It is hard to see how we would. To make this clear, let’s imagine that it has been shown in a statistically rigorous way, using all the criteria above that dogs reduce stress in people. Let’s call that finding D. So equivalently should we believe the following “given D there is a low probability that dogs stress me out”? Well, it is hard to know what to make of that. Do we mean low probability as in, if I am stressed by dogs I am in the minority? Or do we mean an epistemic probability, as in I should conclude that I am likely not stressed by dogs? Well, the first isn’t exactly right, it would definitely be a bit more complicated than that. But the second is clearly false. Not only do I know facts about this study, I know facts about myself. If I knew that I was in fact afraid of dogs (I am not), I would know that it is very likely that dogs stress me out, despite the study.

So if “given P(negative) there is a low probability that I am more productive with TDD” I mean that, given everything I know, I am likely not more productive, I have made a confusion. We can’t conclude anything about empirical studies like this without looking at our background beliefs and past experience. Wilson (and those like him) should rest easy. We need more than just empirical research to show that we ourselves are not more productive with TDD.

It is very easy to confuse the statistical effect a study has found with a personal conclusion. There are of course times when it makes sense to apply results from studies to ourselves. But these cases depend on our background knowledge. A study might find a difference in performance on a test from an amount of sleep. If we know of no relevant facts that would make us think we would do any better or worse, we might tentatively conclude that we would probably do the same. But the procedure here is the same, we must check our background beliefs.

Now I fear what I just said will be misunderstood. Imagine this, someone does an empirical study on only me. They measure my productivity in a way I agree is a good measure, their procedures are solid, and everything is up to my standards. Imagine now that they find that I am not more productive with TDD. Is what I just said above license to ignore the finding? Of course not. The point is that we can’t go from a statistical case to a personal conclusion, not that empirical evidence has no bearing.

### Change in Advocacy

Perhaps instead of a change in belief, we find a change in behavior around advocacy. If we find that P(negative) is true, perhaps we shouldn’t advocate for people to practice TDD. But why exactly think that? Perhaps our advocacy was based on the belief that P(positive) was true. If so, then it would definitely make sense to rethink our behaviors. But that seems unlikely as we will explore more in a bit. Absent that, has P(negative) given us any reason to abandon our advocacy? Well no. We can clearly advocate for things that we don’t believe increase these metrics. Perhaps we simply enjoy TDD. Isn’t that enough for advocacy?

But even if it doesn’t obligate us to no longer advocate for TDD, shouldn’t P(negative) lower the intensity of our advocacy? Should we be as strident in our advocacy of TDD? Well, that depends. Was our prior level of intensity determined in a substantial way by a belief that P(positive) was true? If so, then yes, it seems we should lower our level of intensity. But absent that, why think so? It is hard to see a reason why.

### Change from Belief to Knowledge

If our advocacy behavior needn’t change, maybe all that has changed is our relation to the proposition. Imagine before reading the research we believed that P(positive) and after reading the research we find that in fact, we were right, the research shows that P(positive) is true. The suggestion made by some advocates of empirical software methodology is now we have gained something. Before we just “believed” that P(positive) was true, now, we “know” that P(positive) is true. Now our actions can be based on knowledge rather than “opinions” or “superstition”.

There’s a bit to unpack here. While this may seem like one claim there are actually quite a few packed in. First is the claim that believing on the basis of an empirical study can give us knowledge. This claim feels fairly straightforward. It definitely seems true, particularly given our ideal assumptions, that after reading an empirical study someone can definitely come to know all sorts of facts about software. If the claim stopped here, there would be no need to comment further, but the claim goes deeper. The claim is not just that we can know, but that before reading the study, we didn’t know. That we “believed” or that we had mere “superstition”. If we are to understand the importance of this claim, we must explore a bit of what knowledge is.

For about 2000+ years there was general agreement on what knowledge amounted to. It was a definition offered by Plato’s Socrates that knowledge was “Justified True Belief”. This definition held sway until [Edmond Gettier published a three-page paper ](https://fitelson.org/proseminar/gettier.pdf)that showed this to in fact be false. What exactly is the definition of knowledge accepted today? Well, there isn’t one. For our purposes, we aren’t going to dive into this debate. But it is important to bring up for two reasons. First, while knowledge might not be justified true belief, it is generally agreed that knowledge is at least true belief. In other words, you can’t have knowledge of something false, and you can’t know something you don’t believe. Secondly, in the same paper that shows knowledge isn’t justified true belief, Gettier showed that knowledge isn’t true belief that has adequate evidence supporting it.

In other words, this discussion about studies taking us from “I believe X” to “I know X” is founded on a confusion. In order to know X, we must also believe X. What about superstition? Well presumeably the assumption is that if we didn't know on the basis of study, it must be superstition. But is that the only option? Let's explore what we could have known absent a study and what we couldn’t. While we may not have the exact criteria of knowledge, for our purposes we will use the following formulation, knowledge is warranted true belief. What is warrant? Well, we aren’t sure of the details, but it is whatever it is that turns “mere true belief” into knowledge. So the question becomes, what can we be warranted in believing? Let’s look at the details.

### What We Knew Before

Could we have known P(positive) without having read the research? Well, one way of taking P(positive) is a rather particular claim. It is a statistical claim about the effect of TDD on some population of engineers. Taken that way it seems unlikely that before the research anyone knew P(positive). But it is also equally unlikely that anyone believed P(positive) to be true. What they probably believed was something a bit vaguer than that. They might have believed that some people could become more productive with TDD. But this isn’t the same as the claim that there would be a statistically significant result in a controlled test.

Could we know that some people can be more productive when practicing TDD absent a study? It seems a rather bold claim to say we couldn’t. Can we know what makes ourselves more productive absent a study? Can we know things like “dogs relieve stress for some people” without a study? What reasons could we have for thinking this isn’t true? How would we prevent these reasons from having globally skeptical conclusions that show we can’t know anything absent empirical studies? Perhaps there is an argument the be had here but is hard to see how one would go.

You might think I am being too kind to TDD advocates. Their claims might not have been so modest. Perhaps they believed before the research that TDD would improve these metrics for most people most of the time. If so, it seems they weren’t warranted in that belief. But after the research, should they now claim that belief to be knowledge? In other words, is it now warranted? Well, it would seem not, because that isn’t what P(positive) says. It is a claim about a statistically significant effect. It is much more precise than the prior belief. If I summarized P(positive) as “TDD improves these metrics for most people, most of the time” people would rightly claim I misrepresented the research.

Perhaps I am merely being too pedantic here. Before the TDD advocate read the research, they believed something about TDD that was not backed by research. Maybe it was a modest claim, or maybe it was grandiose. Regardless, after reading the research, their view is changed. Let’s assume that P(positive) turns out to be the case. Now the TDD advocate can genuinely advocate for TDD with good evidence. This of course seems true. The question I want to ask, is what about the prior beliefs the TDD advocate had? Should they now give them up in light of P(positive)? Should they confine their advocacy to only mention P(positive)? Well if their prior beliefs could have a positive epistemic status without knowing P(positive), it seems hard to see how P(positive) would change this. If they weren’t positive, then of course they should have given them up regardless of P(positive)’s truth value.

#### False Belief on the Basis of Empirical Evidence

We can’t only ask about our beliefs before we read the research, we need to ask about our beliefs after reading the research. Are these studies a surefire way to prevent us from forming new immodest beliefs? Do they ensure our new beliefs truly rise to the level of knowledge? Here we might be tempted to point out that studies can often have flaws, but remember, we are intentionally making idealistic assumptions. Assuming an ideal study, with all the features we’d want for the best possible science, can we still form unwarranted beliefs?

It is clear we can. It is fairly common to take solid empirical evidence and over-generalize. We may not pay proper attention to the parameters of the study. We may draw conclusions that seem to us to follow, but actually have a flaw in one of our major premises. This may happen despite our best efforts and intentions.

But of course, everyone recognizes this. What is usually said is that while empirical studies don’t lead us infallibly to knowledge, it is by far the best way we know to lead us to knowledge, and further it is a self-correcting mechanism. Here you will find no argument from me. In matters of fact, empirical evidence is a great mechanism. I will just highlight again that we must not mistake questions of “is” with questions of “ought”. Nor should we conflate what an empirical study shows with our everyday beliefs.

### Superstition, Knowledge, and Rhetoric

The important point to underscore here is that regardless of a change in our belief structure, what we haven’t seen is a move from “superstition” and “opinion” to knowledge. These suggestions are mere rhetoric. They oversimplify the noetic structures of human beings. Not only can we know the more modest claims we often believe without studies, but our more grandiose claims also don’t ever rise to the level of knowledge. Further, even if we accept a study, that doesn’t preclude us from believing on the basis of “superstition” or “opinion”. We might accept a study, not on its merits, but because it makes us feel good. We are complicated believers that cannot be fit into a simple binary.

We must move beyond these rhetorical oppositions. It is of course good to base our beliefs on evidence but to suggest the only evidence that exists is controlled studies is to misunderstand the way our knowledge generally works. Did no one know anything before the first controlled study in 1747? Clearly, no one on either side of this debate would think that is true. We must make that clear in our speech and not rely on shorthand when discussing these fraught issues.

## So What?

We’ve covered a number of things that don’t follow from the results of P(x). Do these results generalize? For example, many fans of empirical studies point to code review as an example of a great empirical result. It turns out code review is very effective at reducing bug count. Doesn’t that mean we should do code review? By itself? As I hope I’ve shown that is an ill-formed idea. Facts about the world don’t tell us what we ought to do. They must be married with our desires, beliefs, background knowledge, etc.

Empirical results give us a data point to use in our practical reasoning. They guard us against overly grandiose claims. They can cause us to rethink things we might have assumed. But we aren’t warranted in moving from research suggests X has Y benefits, to therefore we ought to do X. In fact, I will make a stronger statement. No empirical finding about software alone should make us conclude anything about what we ought to do. What we ought to do depends on facts like our background beliefs, our desires, and our particular circumstances.

So why care about empirical results at all? Well, learning more about the world is always a good thing. Truth is valuable. But the question still remains, what role does empirical knowledge have in answering the questions we ask like:

- Should we use static or dynamic types?
- Should we practice TDD?
- Should we do agile development?
- Should we use a memory safe-language?

That answer depends on our goals, desires, and background beliefs. If these align with what research has shown then perhaps they will be beneficial for our own personal decision-making process. But these results alone, absent these broader considerations, don’t tell us anything about what we ought to do. Given that, there can be no global ought for the industry, for there is no goal or desire we all share.

Just as many advocates of empirical software studies implore us, we must not be taken in by salespeople, by the sophists who push their own solutions. This includes the appeal to the rigor of empirical studies. We must not let ourselves be led astray by rhetoric like “superstition” thrown around as an attempt to cast negative aspersions. Empirical study does not exhaust rationality. We cannot escape the need to make judgment calls on the basis of the values we hold.

# Predicted (blunt) questions

**Q:** Do you just hate empirical software studies?

**A:** No. I have nothing against it at all. Studying the world is great. I just see a common implicit conflation of what science tells us is true and what we ought to do. These things are not one and the same and we should pay attention to that.

**Q:** But if science tells us X is more productive, shouldn’t we just do it? We want to be productive.

**A:** We do want to be productive, I guess. But we also want to be happy. We want to get along with our coworkers. We (I) want to drink nice coffee. We have all sorts of desires and these can conflict. I personally don’t care that much about productivity. It is not at the top of my list. I am fairly productive on the things I want to be productive on and not really interested in maximizing that.

**Q:** But if science tells us X increases code quality, that just seems like a clear win.

**A:** What I said above still applies. But I also want to take this opportunity to remove the idealistic assumptions we’ve made. We have to be careful about the meaning of words here. The meanings of “external code quality”, “testing effort”, and “developer productivity” in the study mentioned above have massively different meanings from what I take the ordinary meanings of those words to be. Perhaps I am mistaken and others would see them as tracking the ordinary definitions. But I will say, I don’t particularly care about the attributes they have measured.

**Q:** Isn’t this just a sophisticated way of saying you are going to ignore evidence?

**A:** I’m not sure how it could be that. I have no skepticism toward general scientific consensus. Nor would I have skepticism towards the findings of empirical software studies. My point is just that we have to do an extra step in thinking about applying these findings and that extra step involved judgment, weighing of goods, and introspection of our desires.

**Q:** Sure, as an individual you might not want to do these things. But shouldn’t businesses use this to guide their decision-making? Businesses desire productivity and efficiency, anything that has been shown to produce that should be adopted and enforced in a place of business.

**A:** I just think that is too simplistic of a picture. The “scientific management” of John Taylor showed all sorts of results. Perhaps you think they don’t have the scientific rigor of today, but I don’t think that matters. Businesses are complicated, complex systems. Simple-minded, thin rules aren’t going to solve our problems.

**Q:** But isn’t this whole thing a strawman? Consider project Aristotle from google. Here we see scientists making direct recommendations on how we “ought” to act. They measured all of these things you are talking about, desires, goals, etc, and found concrete real recommendations.

**A:** If anything that reinforces my point. Project Aristotle is pretty careful in parts to be clear that their recommendations are relative to the environment of google. For example, they note that team size didn’t seem to matter at google, but did elsewhere. In other cases, they are much less careful, for example in their recommendations around OKRs.

Perhaps the suggestion here is that research that takes goals, desires, etc into an account can offer “oughts” and so there is no is/ought problem in the neighborhood. Let’s assume they can legitimately offer these oughts. They can only do so as a hypothetical imperative, “if you want X you should do Y”. But that just brings us back to the same question, do I want X? Is there some other thing Z that I want more than X that conflicts with my attaining X via Y? We haven’t escaped the problem.

**Q:** Doesn’t leaving these things up to judgment just mean we will end up with no progress?

**A:** Does judgment never give us progress? Was C an improvement on BCPL because of scientific empirical study? Did the early pioneers of personal computing not use their judgment to bring us a future others didn’t see? Are the sciences devoid of judgment? I don’t think there is any reason to think allowing for judgment prevents progress, in fact, I think (but have not argued for) the opposite.

**Q:** Wouldn’t it be better to just follow what science has shown, even if it is suboptimal rather than leaving it up to the whims of people?

**A:** I think we need to be clear that judgments and whims are not the same things. In our everyday life, we can see a difference between acting on a whim and making a considered decision. The same is true in programming and business. I personally don’t think it would be better to just “follow the science” (whatever that means) rather than use our judgment. Partially because I don’t think there is anything called “following the science” that is devoid of judgment, so if we think we are doing that we are deceiving ourselves.

**Q:** Let’s concede your points above that empirical research alone doesn’t give us an ought. Doesn’t that mean you should apply this same logic to things outside of software? For example, seatbelts have been shown to reduce deaths in accidents. By your logic, that fact alone doesn’t show that we should wear seatbelts. In fact, wouldn’t your logic result in the further claim that seatbelts shouldn’t be mandated by law, since after all, not everyone has the same goals, desires, etc?

**A:** Well, yes this same logic does apply outside of software. We do in fact have to look at our desires, beliefs, goals, etc when determining what to do with empirical facts. Knowing that an umbrella prevents me from getting wet doesn’t give me any action I ought to do. But, if it is coupled with my desire to not get wet, then I can see I might want to use an umbrella. So for seatbelts, we see the same thing. If we want to be safe, we should wear a seatbelt. I always do. As for the law, that doesn’t follow at all from what I said. There are tons of different ways to justify why governments would have a legitimate reason to mandate seatbelts. Going into those is definitely beyond the scope of this post.

**Q:** No one believes that empirical studies alone can tell us what we ought to do. They just implicitly assume we share common goals. Are our goals really that divergent?

**A:** If that’s true, I am relieved. I do think it is a minority view. But it does seem to me that some people advocate for it. As for sharing common goals, I don’t know. It seems to me that we do often have divergent goals. The generative artist and the Site Reliability Engineer seem to be doing quite different activities. Even the functional programming advocate and the object-oriented programming advocate seem to be in opposition. If there are some overarching goals that we all share in this matter, they have been lost on me.
