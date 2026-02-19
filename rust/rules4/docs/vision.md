We have created here a little language for term rewriting. But right now we have fairly limited syntactic support. One thing I want is that we could easily create our own arbitrary syntax and provide ways in which it reduces. So we could end up having maps and sets and all sorts of stuff like this.

But also, you could end up just defining things like for loops in the traditional style and our semantics makes them work.

## Concurrency

One thing I don't think we currently do is concurrently rewrite across scopes. If I have two scopes @a and @b rules should be reducing concurrently across them

## Daedalus Inspiration

We should be able to do things that the berekely orders of magnitude project did. For example we should be able to make a raft implementation here there is some scope we write to that handles the network, but we are able to keep the log and look at sequential time and all this

## PuzzleScript

Given all these features and flexibility we should also be able to make our own little puzzlescript dialect that lets us make sokoban games.

## Demo

All of these kinds of things need to be put into a nice little demo notebook where people can explore and learn and walk through how this system works