# Legibility is Ruining your Codebase

In Seeing Like a State, James C Scott coined the term legibility to refer to the way in which large organizations (in many cases the state) try to make sense of complicated phenomenon. Legibility does not involve merely abstracting over these activities, but fundamentally changing them, simplifying them, forcing them to be measurable. The quintessential example of this is the "scientifically managed" forests of 18th/19th century germany. Rather than a natural forest full of underbrush, diverse species of trees planted in a naturally occurring pattern, Germany created industrials "forests" in neat rows, cleared of underbrush. This worked remarkably well for the first generation of trees, but as time progressed the lack of diversity led to massive problems, including "forest death".

[Sean Goedecke](https://www.seangoedecke.com/) has recently published an article of the aptly named [Seeing like a Software Company](https://www.seangoedecke.com/seeing-like-a-software-company/) that does a fantastic job exploring the ways in which and the reasons for legibile practices being applied inside software companies. Sean's description provides a balanced perspective. It does not attempt to cover up the downsides, but also explains why companies often feel the tradeoffs are worth the downsides. But despite agreeing with almost everything Sean said, I think there a dimension lost. Just like the "scientific management" of the "forest" ruined the forest, legibility is ruining your codebase.

## Two Distinct Activities

Forests are used to provide us with wood. Even before the creation of "scientifically managed" forests, humans cultivated forests for their particular needs. Yet, to say that a forest is "for" providing humans with wood is a bit of a stretch. The forest is a natural phenomenon. If it has a purpose, it is certainly a multifaceted one. It is provides a shelter for animals, nurishment for tree, oxygen to the earth. A codebase is not natural, and yet it is no less complicated to describe its "purpose".

Consider the recent relicensing of Terraform. Terraform was a library created by Hashicorp to make Hashicorp money. But clearly that can't be its only purpose. Terraform is a tool "that lets you build, change, and version infrastructure safely and efficiently." But even this is too simplistic. Terraform, because it was open source and had community input, participated in a complex ecosystem where many different people depended on it for different needs. Its parts served different roles to different people, different companies, some playing a minor role and other essential to the function of those activities.

A proprietary, fully in-house, codebase is no different. Different teams work on them. Different parts of the codebase satisfy the needs of different customers. Different parts of the codebase have different constraints, have different requirements, require different upkeep. Yes the company made this codebase "to make money" just as we cultivated forests to "make wood" but that does not stop the codebase from having its own internal structure, from having different relationships to different things. 

A codebase can be ruined while a company is successful. In what follows I will be assuming this at all points. This point may seem obvious and not worth stating. But the natural reaction to justify any bad practices is that it is "what the business wants and they pay your pay check". That may well be true. But it doesn't prevent their actions from having negative consequences. Should companies care about these negative consequences? I don't know.

## Types of Legibility that Ruin Codebases

Legibility is about making something clear and measurable to a business. It is a process of control to make predictable the "outputs" of a process. But it can come in many forms. Some of which we all recognize as destructive (but often ignore because they are "necessary for the business") and others that are overlooked but no less harmful.

### Deadlines

This is perhaps the easiest case. Deadlines cause sloppy, hacky code. This kind of code can cost way more to fix than merely changing the text. Sometimes this causes months of rework. I am reminded of a deadline driven decision by managers at a company I worked at that decided we don't need to canonicalize customer information. Each time a customer entered their address and phone number, we would just make a new entry. Later a new requirement came in to know how many transactions each customer had had. At this point there were millions of duplicated unmatched records and the work to clean it up lasted well over a year.

### Estimates

I have been called cynical for suggesting that most estimates are fake. But luckily now, thanks to Sean Geodecke I am in good company.

> Project estimates are largely fantasy. More accurately, they’re performative: **the initial estimate determines the kind of engineering work that gets done to deliver by that estimate, not the other way around**....However, these assumptions are true enough for their purpose, which is to provide legibility to the executives in charge of the company. Whether the project estimate is accurate or not, it can be used to plan and to communicate with other large organizations (who are themselves typically aware that these estimates ought not to be taken completely seriously). (Emphasis in original post)

I can certainly say Sean and I have had markedly different experiences on how seriously estimates are taken by executives. But setting that aside, Sean doesn't write about one of the largest negative downsides to estimates. The fact that it implicitly commits a team to doing the work they estimated.

It isn't uncommon to discover while working on a feature that the feature is completely unneeded. This might be because the problem itself can be eliminated rather than fixed. That there is already something else in the system that serves this same function. It turns out no one actually wants this feature. An upstream data provider that you depend on for this feature to work doesn't actually have the data. The feature will make the system too slow and there is an alternative that fullfills the need without this implication.

Getting out of doing a project for which you provided an estimate for is a really tricky endeavor. Many times there are stakeholders who faught and argued to get this project prioritized. By killing it, you are making them take a hit to their reputation. So even if you are someone willing to fight these fights, you will have to choose your battles and will not win them all. This fact often leaves the codebase in a much worse place, full of things it didn't need and often leaves the company with a maintanence burden it didn't expect.

### Standup

It is easy to see standup as a simple daily meeting and the fact that engineers would ever complain about them shows just how much of divas we are. But standup is the quintessential legibility play. Can all of software of my software day be reduced to what I worked on yesterday, blockers, and what I plan to work on today? Of course it can't. And we see this reality in the "failure modes" of standup. You can read countless articles telling you to not let your standup have "unstructured chatter" and their example always is one engineer talking about problems they had and others trying to help them. Perhaps rather than reducing software engineering down to three "essential questions" we could actually encourage these more meaningful conversations. Why don't we? Because they aren't legibile. 

But how does this affect your codebase? Standup provides a negative incentive for sharing true deep problems. Coming to standup day after day saying that you are working on the same problem and don't have meaningful progress to share is considered a very bad thing. The idea is that all problems can be unblocked by simply talking to another team member. This creates a culture of workarounds, of being unwilling to sit with hard problems. Of not giving the time people need to put things on the backburner.

### The Backlog

Writing down problems in your software, ideas for what you want to work on in the future, feature requests, etc can be incredibly valuable. But a backlog is almost always more than this. I have been at companies where the backlog is *explicitly owned* by product and not engineering. Engineering is allowed to participate in "grooming" the backlog. What exactly is that? It is almost always the process of making the backlog more legibile. Story pointing, categorizing, prioritzing are not for the team, but for executives to make decisions on what should be done when.

It hard to list all the ways that backlogs can create negative consequences for your codebase. The first obvious on is a practice I saw quite commonly at a former company, having a backlog meant that *any* team could work on a feature. This was one of the explicit goals of the backlog to spread out who was able to work on features. What this meant is that code became incredibly confusing. Engineer A knew about features X, Y, and Z and had been assigned X and Y this sprint. Assuming they would work on Z, they made to code so that it could easily accomidate Z. But now engineer B is assigned Z. Not knowing much about X and Y, they implement Z whole cloth, duplicating effort and code they didn't know existed.

The temptation here is to respond that clearly this isn't a problem with the backlog, but the assingment of tasks to work on the backlog. Or perhaps, it is a breakdown of backlog review. Or perhaps it is a breakdown of linking related tasks in the backlog. We are enigineers who can come up with countless ways to structure things to "prevent" these problems. But the problem itself occured because of the legibility of the backlog. If instead A had worked on X and Y and needed to work on something else, they could have simply talked to B and told them about X and Y and how to do Z. Instead, we believed the simplifying legible myth that each "story" is an isolated unit of work and ended up in this scenario.

### OKRs / KPIs

OKRs and KPIs are perhaps one of the most obvious forms of legibility. They are quite literally an attempt to ignore all the true complexity for a team, a group, an individual and instead assign numbers that are meant to be a proxy, for these more complicated facts-on-the-ground. I think a large number of software developers hate these so I won't be labor the point. But I will tell you about a time I failed my OKR.

I worked on a reporting system at a small startup. Our CEO had decided that OKRs (or was it KPIs I don't even remember) were incredibly important now for everyone to do. Anyways, it had been decided that our goal was to "lower median report end to end time by 10%". If we did not meet that goal it was a failure. So what we started with was talking to the customer who used reports the most. They had like 60k Reports a month. We asked them why they had so many they said "What? Umm that must be a bug". The next week the number of reports they ran dropped to like 60 a month and our median time decreased!

Having been so successful, we went to the next highest user. Same exact thing! They were surprised at how much they were using reports and drastically reduced their usage. But this time, the median time increased....You see, most of those reports they were running returned no data, and so they were fast. Now that we got rid of them, our median time got higher. So we failed that okr despite making the system better.

### Coding Standards

Coding standards are an attempt at legibility by making software engineers interchangable resources. By standardizing software development practices, languages, style, tools, we make it easier for programmers to moved around. We allow the business to ignore us as individuals. This is the great irony of the whole situation. Many people feel that coding standards are **the** way to keep a codebase clean. But within the context of a business, they also serve the purpose of the illusion that we can have anyone work on any part of any codebase, leading to the decline in quality.

### Others

There are countless other practices I could list here. Nearly every software methodolgy I can think of has its roots in legibility. For some reason we keep ignoring the fact that our methods are explicitly designed around this feature. They aren't made to make our code better. They aren't made to make our products more enjoyable. Of course you hardcore agile apologist might disagree. But I have receipts.

> “agile methodology”: a system of methods designed to allow the  development team to match and track the business needs, especially in a context where business needs change frequently, important facts change, or where we are obliged to adapt to important uncontrolled factors. - [Alister Cockburn](https://www.satisfice.com/blog/archives/5175#comment-205)

Our practices are not meant for us. They are meant for the business. They are attempts to reign in software engineers, at the expense of the goals of those software engineers for the benfits of the company. Even processes we now consider "best practices" have this exact logic built into them.

## Being Illegible

I am bit hesitant to give people advice. Particularly advice that goes against the norm. So I instead I will rephrase it. One valuable thing to ask yourself is "Do I like being legible?". In my experience, some people really enjoy it. Some people want this external recognition that comes at each point along the way that is only possible if your work is legible. Some people love knowing that their work aligns with the business. Some people love the process, love the oversight.

But for others of us, legibility is the death of motivation. Personally, I will do way more work if that work is not legible. If that work is self-guided, self-chosen, and not accounted for, I will work hard. I will be excited. As soon as my work becomes the bogged down with the trappings of legibilty, as soon as I feel this lose of control, that my work could be "reprioritized" at any minute, that the very thing that made the work valuable (to the customer) could be striped away, I lose all motiviation to do that work. That doesn't mean I don't do it. It just means I don't enjoy it.

Being illegible can be good for the business. It can help the business achieve its goals in a way it itself has blocked off. It can get you in trouble as well. But it is also a way to succeed. I have seen countless projects fail only to be saved by the illegible work of a lone engineer who knew there was a better way to solve the problem.





