# Stuck? Build Your Language Backwards

I can't count the number of times I started trying to build a programming language just to get stuck in the parser. Some of this was certainly my lack of experience writing parsers, but it was also just how many different choices I had to make. Which way would I parse things? What syntax am I going to use for X? How do I deal with parse errors? Inevitably I abandoned the project before I ever got to the parts I really wanted to learn, namely, all the rest of building a compiler.

That is until I stumbled upon, by accident, the technique that has worked flawlessly ever since. Building the language backwards.

This means picking the lowest level you intend to target and then writing code that produces it. So, if you are building a bytecode-interpreted language, write a simple bytecode interpreter and something that spits out bytecode. If you are targeting machine code, write a little assembler. If it's wasm, write some code that generates wasm. I have found this an incredible way to get myself unstuck.

It may sound counter-intuitive. You probably already have an idea of the language you want to build in mind, so why not start with what you already know? The key for me was that it was so unclear, in my head, how to get from high level -> low level, but once I had the low-level thing already in place, I could see the tediousness, the repetition, the bookkeeping, all the things my higher level language would get rid of. Bridging that gap became much easier.

Ever since accidentally stumbling upon this idea, I have had way more success in getting my language projects past that initial hump and I learned quite a lot more along the way. So if you're stuck, try it out and let me know if it works for you.