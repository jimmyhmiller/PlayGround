# Code as Data; Data as Code

Programming is a rather peculiar activity. Once programming begins, you've left the world of nature language behind. Ambiguity ceases to exist, common sense fades, all that remains is pure, cold, logical obedience. Consider the famous "shampoo bottle" example, "lather, rinse repeat". As has been quipped a million times over, in programming this is called an infinite loop. But more important the shampoo example is supposed to explain to us what code *is*. Code is said to be simply a set of instructions for the computer to execute. In this post I want to explode that idea. We will look at code not as instructions, not as commands, but as data. And we will see that not only is code data, but data is code.

## Building blocks

Before we can arrive at this marvelous conclusion we need to gather our building materials. Unfortunately, unlike my other posts, I can't lean on javascript here. Javascript is a fine language and certainly this point could be made using it, but the message will be muddied. So, instead of working directly with javascript, we'll build up our own tool kit, making reference back to javascript when analogy allows.

### The Simplest Data Structure

Perhaps the most basic data structure in existence is the list. Javascript itself does not have lists, but the concept isn't hard to understand. A list is very similar to an array. But lists are limited. Lists only allow you to add and take off from the front. Lists do not let you access a random index. Let's see some examples of lists.
	
	(1 2 3)
	(jimmy janice tanner)
	((1) (2) (3))
	(1, jimmy, (3))

Above we have four different list. The first is a list that contains three numbers. The second contains three symbols (we will get to those later). The third contains three lists. And the last contains a number, a symbol and a list. As you can see lists here are surrounded by parentheses. They are whitespace delimited, no commas necessary. But we can add commas if we like.


### Vectors (AKA arrays)

The next data structure we need for our purposes is a vector. Vectors are basically arrays. In fact they look almost identical.

	[1 2 3]
	[jimmy janice tanner
	[[1] [2] [3]]
	[1, jimmy, [3]]

Above are the equivalent vectors to the lists we saw earlier. Vectors unlike lists do allow random lookup. Vectors are most efficient when adding to the end rather than the front. If you know javascript arrays, you know vectors.

### Maps (Kinda Like Object)

The final data structure in our list will look rather familiar. It is called a map. It looks remarkably similar to an object, but doesn't have all the properties objects have. In fact, they are so similar that when you are using json, you are using an object as if it were a map! So what exactly is a map? Well, it is a set of key value pairs.

	{:stuff 1 :other-stuff 2}
	{1 fish, 2 seal}
	{[1] (2) (2) [1]}
	{{"a" "b"} 3}

The above might be a little strange looking. But is actually quite easy to understand. Maps have keys and values, but unlike json, maps keys can be any type. The first map has a key of keywords  and a value of number. A keyword is kind of like a string constant. Each keyword is only stored once in memory, so they make for great keys in maps. The second has a key that is a number and value is a symbol. Third uses vectors and lists as keys. And the final one uses another map as a key. As before commas are optional.

## Enough Prelude

I admit that was a rather long prelude but hopefully there will some good pay off. We've defined our data structures and look at some data. So why don't we try some code now. We won't be writing code in javascript. In fact, our code may look a little weird, but as you will see, it is actually quite familiar.

	(def x 2)

This code. What this code does is assign x to the value 2. In other words, def here means define, define a variable. This code is basically equivalent to the following javascript.

	var x = 2;

But this code looks suspiciously familiar. It is surrounded by parentheses and each token is separated by a space, just like a list, that's because this code is actually just a list. This "code" is data, it is a list that contains two symbols, def and x, followed by a number. Let's look at a more complex example.

	(defn double [x]
		(* x 2))

Here we have a function that doubles any number it gets. But again really what we have is a list with two symbols, a vector containing the symbol x, and then a list with two symbols and a number. This code clearly shows its dual nature on its sleeve. At one glance it is code, at the next it is data.

### Moving forward

But it is far from obvious how this could be useful, in order to understand we need to dive in a bit to the operators on our data. All the data structures we've listed above have numerous operators defined for them. But the most important