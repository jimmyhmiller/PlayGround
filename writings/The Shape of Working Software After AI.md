# The Chasm: The Shape of Unfinished AI Codebases

It's 2 am. You've stayed up all night trying to get your program working. You finally figured out that one bug. The singular thing stopping you from running your new program end to end. The relief! You have accomplished something incredible.

The next day, still feeling the high from this sense of accomplishment, you go to show your loved one the program you've made. After breaking a world record for the number of caveats and hedge words a single sentence can contain, you show them a few flashing words in a terminal window. The glaze passes over their eyes. Your accomplishment, however great it is, is simply that, yours. No one else is going to understand.

## Understanding What is Done

This style of progress is one of the big barriers that I've seen keeping people out of programming. Countless teaching tools have been made that try to circumvent this problem, to give learners something that they can *see* as a big accomplishment and that they can share with others who can see as well. This is one reason games are such a popular format for teaching programming. But the reality has always been that progress in programming starts invisibly. Software in development would always be in a state of not working, not demoable, not flashy, long before it became "real software".

AI has flipped this on its head. Creating something flashy, something demoable, something that appears to work is perhaps the easiest thing to do. But it is more than that. Even for software with no flash, the gaps in the program, the parts that don't work yet, take on a completely different shape than they do in a human-authored program.

### Cracks or Chasms

In a program written by a person (especially a singular person), there are just obvious whole parts missing. A parser might not accept full input, a button might not do anything at all, a page may just be missing. Now, of course, exactly which parts are missing will depend a bit on the person, what they want to work on, what they want to demo, or circumstances like that. But in my experience, they are mostly predictable.

AI codebases are simply not this way. Or, to put it more exactly, they are predictable in an entirely different way from human predictability. AI written programs are great at giving the illusion of a fully written, fully working solution. They will write endless tests, give endless benchmarks, show immediate speed improvements. And yet, if you try to use the program for anything other than the demo, it can immediately crash, or leak memory, or hang indefinitely. Not that human-written programs don't do that. But for me, I have a good intuition of where these things will occur that has now been violated.

This isn't a comment on the code quality of AI codebases. Nor a comment on what can be ultimately achieved with fully AI written, AI reviewed codebases. It is a comment about this shape incomplete AI codebases often take. For human-authored code, you can see the cracks forming before you drop off the cliff. For AI codebases, the chasms are deep, hidden, and often impossible to climb out of.

### The Need for Rewrite

One of the things I remember about my earlier adventures in programming was the constant rewrites I found myself doing. A program would rise in complexity as I kept working on it until it would collapse under its own weight. I would start seeing how much harder it was to add new features and either abandon or rewrite it. As I got more experienced, this happened less often, but when I would try projects that pushed me out of my comfort zone, this pattern reemerged.

I've found this same pattern with AI rewritten software. But here, because I am being hands-off, it is much less easy to see it occurring. Rather than a cognitive wall being hit where I can no longer hold the program in my head enough to make progress, I see an AI agent consistently lying to me about the progress that is being made. Whole entire sets of features of the application may no longer work, tests may have been rewritten to "pass" in the face of failure, and if I'm not careful, I won't even notice.

Or worse, I see real improvements in every area of my program I ask to test, but rather than true proper engineering, we have special-cased our way to something wholly unusable for anything other than showing off that our software is the "fastest" solution to ever exist. Attempting to rescue these now utterly ruined pieces of software becomes a nightmare. No amount of lints, tests, or metrics will climb you out of the chasm your agent has been so thoughtfully and cheerfully carving for you. So what can we do? Rewrite.

## The Alternative?

I didn't write this post with some pat moral lesson in mind. Nor did I make it to bash vibe-coding. But simply to call out the pattern I've seen. I have found that this is happening to me less often now than in my earlier explorations of agents. Not only with smarter models either. I think it's because I've begun to be able to predict where those shapes are going to be. To nudge things in the right direction. But for me, and the kinds of problems I enjoy, I'm still not sure if there is a good answer on how to codify this. I think instead my intuition for where these problems will occur, when to check on them, how to nudge in the right direction has just been honed. So if you find yourself in the same state, first know you're not alone. But if I had to offer any advice, think back to those novice moments. Think back to when programming was new and hard. Think back to the patience it took to learn a new way of thinking. Perhaps there are valuable lessons to be learned there.