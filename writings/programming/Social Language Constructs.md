# Social Language Constructs
As programmers, we love to act as if everything is cut and dry, unaffected by social circumstances. The mindless execution of our code makes us believe that all aspects of coding practice are equally mindless, equally free from social influence and opinion. Perhaps we don’t consciously think this, we aren’t completely naive. And yet, I think, many of us would admit to that proclivity, that default assumption that code issues are cut and dry.

I want to call attention to one element of programming that is much more determined by our social situation than we generally recognize. A feature of programming that may feel set in stone that I believe is radically different depending on the social circumstances you find yourself in. These features of programming I want to dub "Social Programming Language Constructs".

Let’s begin with the Wikipedia definition of language constructs:

> A language construct is a syntactically allowable part of a program that may be formed from one or more lexical tokens in accordance with the rules of a programming language.

Here we find that cut and dryness we love about programming. We have two constraints placed on what makes something a language construct. It must be "syntactically allowable", and must be used "in accordance with the rules of the programming language". Perhaps this definition is lacking in some regards because these two rules don’t precisely identify what does and doesn’t count. It gives necessary, but not sufficient criteria for what is and isn’t a language construct.

And yet, I think most of us have a good grasp of this. Controls flow operators are language constructs. Classes (in java) are language constructs. Type Classes in Haskell are language constructs. As follows directly from the definition, what is and isn’t a language construct is language-dependent. 

## The Languages we use
My goal here is not to dispute this definition. Not in its wording nor even in spirit. There are of course these cut and dry language constructs, they are real. But what I want to call attention to is another notion of language construct that I think are equally real, and perhaps more impactful. The notion of a Social Language Construct. Simply put, it is the idea that what is and isn’t "syntactically allowable" depends not just on your compiler/interpreter, but on the social environment you write code in.

Perhaps this claim seems ridiculous, let’s start by making it seem less so. What is and isn’t syntactically allowable depends on your language's compiler/interpreter. But not just the language in general, but the concrete version you use. If you are programming in K&R C, there are certain syntactic patterns that aren’t valid in more modern C implementations. If you are programming in plain vanilla, uncompiled javascript, what is and isn’t syntactically valid depends on what browsers you want to support.

But in real systems, syntactically valid often become more strict than just what the compiler/interpreter allows. We add extra layers onto our syntactic validity checking via linting. Once we have added errors to our linter, we have not added additional constraints on what is syntactically valid.

In other words, we have made a new language. If a programming language at a bare minimum decides what things are syntactically valid, our new restriction has made a new language, that is a subset of our existing language. Now because it is a subset, it isn’t radically different. Hopefully, we have made the language simpler in some regards, eliminating certain foot guns.

## What Counts as Syntactically Valid
Hopefully, a linter defining a new meaning of syntactically valid doesn’t seem too radical. But in what way is a linter social? Well if the linter's rules were not decided by you, but instead more people, through some social negotiation process, or perhaps through a power relation, those lint rules owe their existence to social factors. Or in other words, absent that particular social situation, what is syntactically valid for you would change.

But are linter's the only way we can change what is "syntactically valid". Is something only syntactically invalid if a process returns a non-zero exit value? Or can code review change what it means to be "syntactically valid". If my code can never be merged to master, never be deployed because it doesn’t meet some check, is it meaningfully syntactically valid?

Trying to make a meaningful difference between these things seems to be a waste of time. Absent an automated linter, if my code won’t ever be merged if, for example, I use `goto`, the language I’m working with does not meaningfully have `goto`. `goto` is no longer syntactically valid for me.

## A Socially Defined Language
If social structures can merely limit what is syntactically valid, and therefore limit what counts as a programming language construct, they might not be that interesting. But it goes beyond that. Limitations were merely meant as a non-controversial way to introduce the concept. Is it not possible to add to our languages socially?

Well, our rules can go beyond removal. Our social rules for our programming languages can make new features. Consider MVC. Most languages do not have a "Modle" language construct. Yet in our codebases, we often have rules about what counts as a model. We can talk about modles as if they are parts of our language. As if they are the building blocks we use. 

This goes further than merely abstractions like modles though. Working in a codebase with dependency injection, you may find that you have your own module system. No longer do you use your language's `import` or `require` system. You have your own way of constructing modules. No one removed the module system from the languages compiler/interpreter. And yet when writing code in that codebase, it isn’t "syntactically valid" to require a "service" or a "module" because your code will not be merged. 

### Limitations of Socially Defined Languages
Much like non-social language constructs, the limitations of social language construction are implementation-dependent. First-class functions might be a very attractive language construct, but if the implementation of them comes at a 10x cost in performance and memory, their value might not be worth the cost. But perhaps performance doesn’t matter for our use case? If so, a poor implementation can still be useful for our goals.

Social language constructs are implemented in the social systems that regulate the codebase in question. The question of quality here becomes a bit fuzzier. General as programmers we can agree that more memory and worse performance are less desirable than their opposites. But in social structures, it is often hard to find such agreement on implementation values.

Perhaps some members of a social structure value uniformity, then the implementation’s quality to them will be determined by its ability to ensure nothing is missed and the construct is applied consistently. Others may value plurality, here the perceived implementation quality may depend on the flexibility and ability to deviate if desired. These differences in values can often lead to conflict that leads the implementation to deviate over time.

Given the complexity offered by social situations, social language constructs offer much less stability than those codified in the language. Of course, language choice is part of the social fabric, but generally, changing languages is a difficult process. Social situations, particularly at a work environment, are changeable through many external means. So, these social language constructs do not rest on solid footing. This means as time passes, we may lose understanding of the functions these constructs once played.


### Benefits of Social Language Constructs
How can we use Social Language Constructs for good? By focusing first on their implementation. How have we decided to implement our social language constructs? Is it via informed consensus? Via senior engineers deciding what is acceptable? Via deference to our code ancestors? Via hype and trend following? 

Once we honestly consider these structures we can consider if they properly reflect the values we hold. Is our decision to enforce X based on a deeply held value, is it enforced in a manner that is consistent with our values, does it update as our values change over time?

By going through this process we can end up with Social Language Constructs that not only reflect our values but help others see them more clearly and practice them themselves. Perhaps we have a construct involving code comments. If we have consciously thought about this construct and the way it fits into our values, we should find that by reading and writing these comments, people feel the values we claim to hold.

## Downsides of Social Language Constructs
Social language constructs can be detrimental especially with they reflect the power structures of the social situation they are in embedded in. For example, there may be a code in a codebase that cannot be changed. The author may be a very important more senior developer who doesn’t take kindly to people messing around in "their" section of the code. There may also be things not expressible simply because the valid means of expressing them have been deemed forbidden by a committee on code style.

Those who do not have a voice in the social structures at large may now find themselves also marginalized in the code. This is I think one of the most ignored features of our codebases, how we assert social pressure and conformity. Codebases are not neutral repositories of code. Code style guides are not merely rules for what goes in text files. People exert control and social influence through the code they write.

These forms of control are seemingly ignored by many. It is quite understandable. It is easy to mistake a "desire to not be controlled via social language constructs" with "I don’t want a linter, so I can write whatever code I want". Without the conceptual framework offered above, how could anyone hear complaints about linters and style guides as anything other than that?

## Conclusion
While I am happy that I spilled this amount of ink to discuss a not often talked about phenomenon I noticed. I must admit that my reason for doing so was largely because of my opposition to these structures. At the same time writing this down has helped me see that not all these structures are bad. 

Instead I want to suggest a way we might consider looking at these structures. On a minimal casting a Liberalism (I do not mean anything related to modern politics by using these words), a fundamental principle is that there is a plurality of conceptions of the good. There is no one universal value system that all citizens of a country hold to, or ought to hold to. There may be universal principles and rights that must be upheld, but conceptions of the good will always differ.

This to me is the way we ought to consider the goods in our codebases as well. There values various engineers hold that they want to reflect in their code. These differ from person to person. They differ from team to team, organization to organization. We must consider that others have different conceptions of the good of a codebase than ourselves. We must then ask ourselves how should we proceed?

I claim that we should lean towards allowing this plurality. We need those who care about performance. We need those who care about security. We need those who care about approachability. We need those who care about debuggability. We need a multiplicity of values in order to have a flourishing engineering organization. So my advice is to stop trying to create your own language through power structures. Instead let the plurality of goods flow.