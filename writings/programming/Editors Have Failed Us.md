# Text Editors Have Failed Us

To say that all programming text editors look similar would be an understandment. All text editors look identical. The only thing that *might* set one apart from another is their default theme. But generally, baring that, it is almost impossible to tell one editor from another merely by looking at them. Yet despite this, all these editors behave quite differently. 

Or do they?

Given the religious ferver over our editors, you'd think so. But once you look a bit closer you can see most of these difference as actually rather minor. Some editors are faster than others, some have more visible UI. Some have modal editing, some don't. Some integrate by default with various tools, others don't. But all of them at the end of the day are panes of text with perhaps multiple columns. They really aren't that different.

And perhaps this is the way it should be. Text editors work after all. I mean, if there were a better way to do them, we would have come up with it by now. People adopt tools that solve problems they have. Text editors are the ultimate, utilitarian piece of softare. The ideal all software should aspire to reach. But what about all the stuff editors leave out?

## The Things Editors Forgot

Editors are the home from which we create software. This to me is their crucial function. Our editors are the softare we as programmers use most. They are the piece of software we know best, that we modify. We even are willing to spend time practicing using our editor in order to be more efficent. Our editors should provide us what we need to build our software. They should enable us to be efficient in all activities directly related to working on a codebase. Yet for some reason, they forgot so many of the things we do with code.

### The Many Ways of Reading

It is a tired cliche to claim we spend more time reading code than writing code. But this is generally where we stop. What exactly are we doing when we read code? What is the purpose of that reading? Do we always read code in the same way? How do our editors help or hinder us in the reading of code? 

#### Reading to Assess

When I was a consultant I spent quite a bit of time reading code. My job for new customers was almost always to spend time looking through their codebase, given them a sort of global review of the code. I would look for obvious errors in the codebase, assess the general state of the code, and make recommendations. If you have never done this task before it is not entirely clear how you should be begin it. Do I just look around randomly? How do I make sure I haven't missed anything? Over time you developer tricks and in my case tools. But regardless of your technique, your purpose for reading is quite particular. You are not reading to understand, not to debug, not to modify. You are reading to assess. You do not have time to comprehend all 60k lines, yet you must be sure not to miss obvious things. How does your editor help you with this task?

#### Reading to Write

For some reason software companies have decided that [Cards](/card-driven-development) are the correct unit of work. So if you spend your time in any "tech leadership" role, you will at one point or other find yourself tasked with writing a card about some change that needs to be made to the codebase. If the card is technical in nature and the people doing the work are not super familiar with the codebase, it is often useful to read through the codebase and offer suggestions for how a change might be implemented. This kind of reading is not meant to help you fully understand, but to survey. Your goal is the find the relavant parts of the program quickly, non-exhaustively and point at various bits that might be relevant. How do you find those bits if you don't already know where they are?

#### Reading to Understand

Luckily, we don't only have to spend our time writing or working on cards. Understanding a codebase, just in itself, not in order to do anything, is also part of our job. In order to be prepared for the questions we may be asked about how something functions, or the changes we may be asked to make, we need to have a solid foundation in how the program works. Here we are looking for both detail and broad generalizations. It may be important that we learn nuances like our retry strategy for failed requests. (Do we use jitter? Do we cap the exponential backoff? Where does work end up if it fails its retries?). But we must also be able to make broad generalizations that may be inprecise. (We tend to isloate side-effects from computation. In general, we avoid caching. We try to avoid n+1 queries). How do we find this information? How do we record it? How do we share it?

#### Reading to Review

Code review is perhaps one of the most loved and hated practices in all of software. It is almost always agreed that it is a good thing, yet no one wants to do it and no one wants to have it done to their code. Without a doubt this tension comes from the social aspects of our work. But there is still a lot to be said for the tooling side of it. A thorough review involves at the very least, pulling down the change someone made, running it, testing things out, considering alternative implementations, asking questions of the code. More often than not we skip this kind of thorough review. But is that reason merely time? Or is the flow just too interrupting?

#### Reading to Debug

We often think about debugging as if it is a singularly focus activity. What is debugging? It is running our code with a debugger or with print statements to determine the cause of a bug. But debugging is much more inclusive than that. 

