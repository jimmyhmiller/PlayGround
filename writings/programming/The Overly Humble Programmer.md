# The Overly Humble Programmer

> The competent programmer is fully aware of the strictly limited size of his own skull; therefore he approaches the programming task in full humility - Edsger W. Dijkstra The Humble Programmer

Humble is a rather unlikely term to be found in a description of the "modern programmer". Can I suggest, proud, grandiose, delusional, self-absorbed, fad-chaser, bro-y. Even if we ignore the egregious cases of the cryptobro and the AI doomer, programmer discourse is full of complete dismissals of others and overly grand claims about one's own (or one's tribe's) work. Programmers always have the answers. They always know best practices, they are always sure of their technology choices; they could always build it better.

Even when we step outside of this online bubble, when we peel back this toxic, frustrating environment, we might still feel the pull that we are not humble enough. Aren't we always so sure of ourselves? Aren't we always over-confident in our abilities, always believing our code is good enough, good enough without more testing, without proof, without formal methods? Don't we all need to learn the lesson Dijkstra tried to teach us a half-century ago? Shouldn't we be more humble?

It is a diagnosis many of us believe, and the cure we have sought is precisely what [Dijkstra prescribed](https://www.cs.utexas.edu/~EWD/transcriptions/EWD03xx/EWD340.html).

> I now suggest that we confine ourselves to the design and implementation of intellectually manageable programs.

Linters, type checkers, libraries, well-designed languages, these are all tools that we've made since the times of Dijkstra to limit the intellectual burden of writing our programs. The beautiful thing about software is that the size and complexity of "intellectually manageable programs" can change over time as we develop better technologies. This approach of building abstractions, as the key to managing software, has taken us far.

But I can't help but feel we may have **taken the advice too seriously**. Or at least I have, at one point in my career. Not directly, of course, I did not sit and read the humble programmer and devote myself to its teachings. But Dijkstra's advice, filtered through the culture, made its way to me. I believed it deeply. And I took from it a lesson I'm sure Dijkstra never intended: don't try hard things.

## Seeing our Humility

Perhaps the bros in Silicon Valley could use some humility, but for the rest of us, we are already quite a bit too humble. We've created a culture in which "library authors" are the ones equipped and ready to handle complexity. We've taken the fantastic idea of libraries to the extreme. Leftpad has become a common punchline in this regard, but we have to ask ourselves, why do programmers feel the need to grab these kinds of micro libraries? It could be that they are lazy. Or perhaps because it has been ingrained in them over and over again, "smarter people have already solved this problem".

This myth of "smarter people" is the retort made to anyone who dares attempt to write things from scratch. When I talked about [building an editor in rust](/editor-experience), I was asked why I would do that. Editors are hard and smart people have already made them. When I wanted to learn machine code, I just kept being told to write assembly instead, but it turns out [machine code isn't scary](https://jimmyhmiller.com/machine-code-isnt-scary). But I'm not upset at these people. They, too, have internalized the message we've all been told. All of us. Even famously accomplished programmers.

> "Nobody can build a C++ compiler. That’s not a thing. Why are you even messing around with this?" - [Episode 205: Chris Lattner Interview Transcript](https://atp.fm/205-chris-lattner-interview-transcript)

As someone who started coding for the "web", I always believed that I wasn't smart enough to understand the layers on which my software was built; the layers written in those arcane "low-level" languages. But not only that, I believed that we ought to transcend low-level programming. That the correct answer for all problems was a sufficiently high-level programming language and a sufficiently smart compiler. I believed that I had to stand on the shoulders of giants. That I could not forge my own path, create my own tools, start from scratch. Giving up on this belief has brought me back to the fun difficulties that brought me into programming to begin with. I had to start being less humble.

## Being Less Humble

Being less humble does not have to mean giving up on all dependencies. It's not about going back to C99 and swearing off all modern programming languages. It does not mean giving up on garbage collection, giving up on structured programming, and creating a commune where you worship the sacred numbers. It is about questioning those doubts about your own abilities, about thinking through your problem from first principles. It is about being willing to learn the things one level deeper than your current abstraction boundary. To not accept that "how things work" is how they need to work. It is about deal with the pain the abstraction is attempting to fix.

We have increasingly grown accustomed to (and are encouraged to) treat the abstractions we build on as black boxes. We need not do that. Our software hierarchy does not need to create a social one. You are not worse than the kernel hacker, the compiler engineer, or the game engine programmer. You, too, can build whatever you want to build. All software is just software. It's time we treated it that way.
