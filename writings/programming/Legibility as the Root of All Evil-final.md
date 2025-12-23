# Legibility as the Root of All Evil

Thanks to Sean Goedecke's [Seeing Like a Software Company](https://www.seangoedecke.com/seeing-like-a-software-company/) I can finally finish this draft that has been sitting in my "write a blog post about this" pile for far too long. I never could bring myself to write explaining and defending why companies care so much about legibility. But Sean has done that for me. In what follows I don't assume you have read Sean's post. But you should, it is fantastic.

> By “legible”, I mean work that is predictable, well-estimated, has a paper trail, and doesn’t depend on any contingent factors (like the  availability of specific people). Quarterly planning, OKRs, and Jira all exist to make work legible. Illegible work is everything else: asking for and giving favors, using tacit knowledge that isn’t or can’t be written down, fitting in unscheduled changes, and drawing on interpersonal relationships.

Sean gives us this wonderfully simple notion of what legible means in practice in a software company. In the abstract, legibility is a process by which large organizations impose structure in order to understand and control. It is been my contention for a quite sometime that almost of the things that make modern software engineering frustrating stems from our acceptance of practices specifically designed for legibility. But more than that, that the only escape we have is not more legibility, but an embrace of the illegible.

This article has gone through more drafts than any article I've previously written. Mostly for this one reason: how do I convince you that I'm not just an annoying person who doesn't want to follow process and therefore declares legibility evil? I can't. I have tried giving the story from high school all through my career of grappling with the confusing landscape legibility causes. The weird incentives to not finish your work early. The way in which company politics takes its shape from these desires of legibility. But at every turn I felt the need to hedge, to defend, to convince you I'm not that person. It made for very tedious annoying writing. So instead, I will start by presenting a number of common, some small, some large, problems we face, recast them in a legibility lense. My hope here is not to be exhaustive, but perhaps to give you a interpretive lense to apply to yet more cases.

## Enterprise Software

Enterprise software development practices are legibility taken to its greatest extreme and the end result is clear: lots of money, very poor software. Enterprise software is awful to use, but it is often even worse to develop. The codebases are enormous. They are a [complete mess](https://jimmyhmiller.com/ugliest-beautiful-codebase). It is generally recognized the role deadlines (a kind of legibility) in the creation this situation, but I want to point to some less considered ones.

> Don't touch that code unless you want to own it

Every bit of code in the system is either 1) abandoned or 2) owned by someone. This is the simplifying legibile way of viewing a codebase that leads to perverse incentives. Bugs linger, known about for years by many, workarounds to the bug proliferate in the source code all because no one wants to touch that piece of code, put their name on it and now forever be on the hook for it.

> We must follow written down coding and testing standards

What could be better for code quality than wriiten, clear standards? I actually think these standards themselves are bad. But I've found that people aren't convinced by that point. So instead, let's look at this again from a legibility standpoint. We can all agree that code that follows the standard perfectly can be of varying quality. But that notion isn't legible. Instead, we replace this human judgement with checkbox. As long as people meet the standards, their code is good enough. Anyone objecting to standard compliant code has the burden of proof, or more realistically a burden to try to change the standards. A process that can't be done by you.

> All new features must have integration tests

Simplistic rules like this are legible. Perhaps this isn't the one in your head, replace it with another. At enterprise companies, test runs can take hours or days. The number of test far exceeds what is useful. But what test is useful or not is not a legibile property. "Did we add tests for this feature?" is.

> All work must be tracked by a single system

This may seem like a simple way to communicate with your teammates. But ultimately it is a means of allowing the business to cancel work they deem unimportant. This work may be the most important work you will do. This isn't done maliciously, but simply because people have different perspectives. The legibility requirement comes with a simplifying assumption that if work is worth doing, non-engineers will recoginize or can be convinced that is worth doing. This is simply not true and prevents the work that needs to be done from being prioritized.

### How to Fix Enterprise Software

Fixing enterprise software to not be awful requires illegible work. Not just bits and parts here and there that our systems implicitly allow. It needs the kind of illegible work that our legibility requirements push out. Understanding a system takes time. Finding the right joints at which to carve a system requires know how, particular knowlege. Having a vision for the future of a system requires phases in which the future is unclear, uncertain, in which the plan cannot be fully articulated. It requires the full attention of a group of people. It requires extended periods of illegible time. And as Sean points out:

> Even when siloed to a temporary team, sanctioned illegibility still coexists awkwardly with the rest of the organization. Engineers outside the team don’t like seeing other engineers given the freedom to work without the burden of process: either because they’re jealous, or because they’re believers in process and think that such work is unacceptably dangerous. Managers also don’t like extending that level of trust. That’s why sanctioned efforts like this are almost always temporary.

Making a software system built over 20+ years better requires a longer period of illegibility than companies can tolerate. It can't be accomplished in a quarter. It doesn't have measurable OKRs. It does not allow for fungibility of engineers. It can't be accomplished with in the rules, because it is those very rules that create the incentives to make the problem worse. The changes that are needed will have to go against the rules but this cannot be tolerated for long enough to make an effective change.

## Legibility Outside The Enterprise

I have placed the blame here on large enterprise software because it is an easy target, but I have seen many of these same legibility failure modes play out at small companies.

> Projects should be planned and estimated.

Estimates are a constraint. Not simply on the amount of time that something will take, but the kinds of solutions we can consider. Once work as been estimated, we are committing to do that work. But maybe 2 weeks into a 6 week project we discover the whole project is a bad idea. What to do now? At many companies, even small ones, projects are tracked as first class things, a failed project is a bad thing. A successful project that turns out to be a bad feature, costs the company money and eventually causes rework or instability? Not so much.

> We should have one set of written coding and testing standards

People never like that I disagree with this one. But I want to be clear, it is not that I'm disagree with having standards. It is not that I am disagreeing with writting things down. It is that software quality cannot be determined by these kinds of legible standards and by writing these down, we are giving in the temptation to believe they can. Never have I seen these documented standards actually result in better code in the large. Instead they result in fights, power struggles, and resentment. Or they are just ignored.

> Standups make sure work is happening efficiently

Standups are about legibility. You are supposed to take your work and bundle it up into a nice little package to talk its current status (or not, feel free to debate the purpose of standup, but this is what happens). But this isn't always something that can be done. And to attempt to do it can actually lead to detremental affects, perhaps for your reputation if others don't believe your status. This is why so many standups inevitability go over because someone wants to talk at length about something. Or why people are quiet at standup, but then go talk to someone about their problems afterwards.

## But What's The Alternative?

I'm honestly always a bit shocked that for many people it is hard to imagine the alternative. It seems that many people believe the opposite legible processes is pure chaos. Natural forests are examples of things that were illegible to the German government, so they made these industrial monocultureal, straight lined forests. This ultimately led to ecological disaster. The forest had its own logic, its own balance, which was ignored. Instead the needs of administrators were privildeged to the detriment of the forest.

Our software has needs. We the people building the software have needs. There are structures that will help our software flourish and structures that will help us as individuals and as members of a team flourish. Legibility is about homogenous practices. It is about simplifying assumptions. It is about building abstractions that are inherently, knowingly leaky. The alternative isn't chaos. It is a non-homogenous process. It is about looking at the particulars of our software, the particulars of our team and accounting for them. It is about giving up the need to measure and instead focusing on being effective.





