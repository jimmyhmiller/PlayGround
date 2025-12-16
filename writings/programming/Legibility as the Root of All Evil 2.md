# Legibility as the Root of All Evil

If you've ever worked in enterprise software, you know that [it is all awful](https://jimmyhmiller.github.io/ugliest-beautiful-codebase). But why is that? It's not uncommon to blame it on some inherent feature of the problem. Large software systems are big complicated beasts. Large groups of people are hard to push in one direction. The people who buy the software are not the people who use it. But none of these ever sat right with me. They all have this kind of fatalism that is really unattractive to my ears.

I don't like to accept that things are inevitable.

So instead I'm going to offer a competing explanation. One that is no easier to solve, but it has one thing in its favor; it doesn't accept the idea that enterprise software is doomed. Instead, it focuses on our implicit practices. Those things we almost never question. Those things that have the feeling of inevitability but are actually accidental. Luckily all these bad practices, all these things that hold us back from achieving the goals we want in software fall under one category: legibility.

## Legibility

I have had this post sitting in my drafts for quite some time never quite sure how to write it. I faced two problems. First I needed to explain legibility and then I needed to defend it. If you merely tear something down without starting with its virtues, no one will care. But I simply couldn't bring myself to write that. Luckily I have been saved this pain by [Sean Goedecke](https://www.seangoedecke.com/). In his post [Seeing Like a Software Company](https://www.seangoedecke.com/seeing-like-a-software-company/) he lays out legibility quite clearly and does an excellent job telling you all the good things about it. Why we need it. 

I will not assume you've read his post. But if you need the opposite perspective, feel free to jump over there.[^1] Luckily though Sean has offer us an exceptionally clear explanation of legibility and its opposite:

> By “legible”, I mean work that is predictable, well-estimated, has a paper trail, and doesn’t depend on any contingent factors (like the  availability of specific people). Quarterly planning, OKRs, and Jira all exist to make work legible. Illegible work is everything else: asking for and giving favors, using tacit knowledge that isn’t or can’t be written down, fitting in unscheduled changes, and drawing on interpersonal relationships.

Making work "legible" is not about making it clean and tidy. It about making that work understandable to "the business". It is about making a homogoneous view of the work so that it can be monitored and controlled from the top down. The head of engineering is able to set direction, they are able to measure how productive a group has been, they can determine which work is most valuable to do when. Large organizations value legibility. But so do many software engineers. People who advocate for legible software practices often refer to these processes as "growing up". When you are small you can get away with illegibility, but now that we are large, we need these legible processes, we need standards, we need review, we need approvals, we need estimations, we need visibility.

## The Evil

My goal in arguing against legibility is not to convince you that it has no benefit. Sean has does a fantastic job laying out many of the benefits it in fact has. What instead want to argue is that it legibility actively fights against the goal of making good enterprise software. This is intentionally crafted to be narrow. I don't mean that you can't create good software using legibile processes. What I'm arguing is that once you have some of this awful enterprise software, no amount of legibility is going to fix your problems. In fact, it will just make them worse.





[^1]: I don't actually disagree with much Sean has said in his post. This is a different emphasis not a rebuttal or anything. 

