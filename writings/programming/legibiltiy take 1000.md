# Legibility is Ruining your Codebase

James C Scott in Seeing Like a State coined the term legibility to refer to the way in which large organizations (in many cases the state) try to make sense of complicated phenomenon. Legibility does not involve merely abstracting over these activities, but fundamentally changing them, simplifying them, forcing them to be measurable. The quintessential example of this is the "scientifically managed" forests of 18th/19th century germany. Rather than a natural forest full of underbrush, diverse species of trees planted in a naturally occurring pattern, Germany created industrials "forests" in neat rows, cleared of underbrush. This worked remarkably well for the first generation of trees, but as time progressed the lack of diversity led to massive problems, including "forest death".

[Sean Goedecke](https://www.seangoedecke.com/) has recently published an article of the aptly named [Seeing like a Software Company](https://www.seangoedecke.com/seeing-like-a-software-company/) that does a fantastic job exploring the ways in which and the reasons for legibile practices inside software companies. Sean's description provides a balanced perspective. It does not attempt to cover up the downsides, but also explains why companies often feel the tradeoffs are worth the downsides. But despite agreeing with everything Sean said, I think there a dimension lost. Just like the "scientific management" of the "forest" ruined the forest, legibility is ruining your codebase.

## Two Distinct Activities

Forests are used to provide us with wood. Even before the creation of "scientifically managed" forests, humans cultivated forests for their particular needs. Yet, to say that a forest is "for" providing wood for humans is a bit of a stretch. The forest is a natural phenomenon. If it has a purpose, it is certainly a multifaceted one. It is provides a shelter for animals, nurishment for tree, oxygen to the earth. A codebase is not natural, and yet it is no less complicated to describe its "purpose".

Consider the recent relicensing of Terraform. Terraform was a library created by Hashicorp to make Hashicorp money. But clearly that can't be its only purpose. Terraform is a tool "that lets you build, change, and version infrastructure safely and efficiently." But even this is too simplistic. Terraform because it was open source and had community input participated in a complex ecosystem where many different people depended on it. Its parts served different roles to different people, different companies.

A propreitary, fully in-house, codebase is no different. Different teams work on it. Parts of the codebase satisfy the needs of different customers. Different parts of the code need different constraints, have different requirements, require different upkeep. Yes the company made this codebase "to make money" just as we cultivated forests to "make wood" but that does not stop the codebase from having its own internal structure, from have different relationships to different things. 

A codebase can be ruined while a company is successful. It what follows I will be assuming this at all points. This point may seem obvious and not worth stating. But the natural reaction to justify any bad practices is that it is "what the business wants and they pay your pay check". That may well be true. But it doesn't prevent their actions from having negative consequences. Should companies care about these negative consequences? Hopefully we will see that they should way more than they currently do.

## Types of Legibility that Ruin Codebases

Legibility is about making something clear and measurable to a business. It is a process of control to make predictable the "outputs" of a process. But it can come in many forms. Some of which we all recognize as destructive (but often ignore because they are "necessary for the business") and others that are overlooked but no less harmful.

### Deadlines

This is perhaps the easiest case. Deadlines cause sloppy, hacky code. This kind of code can cost way more to fix than merely changing the text. Sometimes this causes months of rework. I am reminded of a deadline driven decision by managers at a company I worked at that decided we don't need to canonicalize customer information. Each time a customer entered their address and phone number, we would just make a new entry. Later a new requirement came in to know how many transactions each customer had had. At this point there were millions of duplicated unmatched records and the work to clean it up lasted a year.

### Estimates

I have been called cynical for suggesting that most estimates are fake. But luckily now, thanks to Sean Geodecke I am in good company.

> Project estimates are largely fantasy. More accurately, they’re performative: **the initial estimate determines the kind of engineering work that gets done to deliver by that estimate, not the other way around**....However, these assumptions are true enough for their purpose, which is to provide legibility to the executives in charge of the company. Whether the project estimate is accurate or not, it can be used to plan and to communicate with other large organizations (who are themselves typically aware that these estimates ought not to be taken completely seriously). (Emphasis in original post)

I can certainly say Sean and I have had markedly different experiences on how seriously estimates are taken by executives. But setting that aside, I think Sean doesn't write about one of the largest negative downsides to estimates. The fact that it implicitly commits a team to doing the work they estimated.

It isn't uncommon to discover while working on a feature that the feature is completely unneeded. This might be because the problem itself can be eliminated rather than fixed. There is already something else in the system that serves this same function. It turns out no one actually wants this feature. An upstream data provider that you depend on for this feature to work doesn't actually have the data. The feature will make the system too slow and there is an alternative that fullfills the need without this implication. 

Getting out of doing a project for you provided an estimate for is a really tricky endeavor. Many times there are stakeholders who faught and argued to get this project prioritized. By killing it, you are making them take a hit to their reputation. So even if you are someone willing to fight these fights, you will have to choose your battles and will not win them all. This fact often leaves the codebase in a much worse place, full of things it didn't need and often leaves the company with a maintanence burden it didn't expect.

(Note: Maybe talk about all the things estimates don't capture)

### Standup



