# Meander: Declarative Explorations at the Limits of FP

As functional programmers we love to pay lip service to declarative data transformation. Yet this allegiance rarely plays out in our actual code. Data transforming combinators (map, filter, reduce, etc) offer significant expressive power over manual loops and mutation, but when combined in complex ways, lose their declarative power.  Lost in a web of nested pipelines, the shape of our data becomes obscured; our code becomes hard to follow. We wind up, yet again, needing to play computer in our heads to understand what our code is doing.

Meander is an exploration into truly declarative data manipulation created as a library in Clojure. Borrowing ideas from logic programming and term rewriting, Meander allows declarative descriptions of arbitrarily complex data; enabling you to search, match, remember, join, and transform any part of your data directly. This talk will show you how to leverage Meander to declaratively solve real-world data transformation problems, give you insight into how Meander remains both performant and expressive, and finally cast a vision for what a more declarative future for functional programming may hold.



## What you will learn

Attendees will first and foremost learn about a different way to look at data transformation. The audience will see how Meander's pattern matching on steroids opens up new avenues expressing intent. We will look at a more mundane, real world example, showing how Meander's declarative approach makes it resilent in the face of changes to input and output, allows us to treat our ordinary data structures in a relational manner, and frees us to think in terms of small transformations of data.

Next we will explore a bit under the hood of Meander. Meander features a rich intermediate representation that allows for optimization of patterns and simplified code generation. We will discuss how this drives Meanders more advanced and experimental features namely strategy combinators and `with` patterns. Strategy combinators are a feature borrowed from term rewriting that allows programmers control over the flow of transformations. This enables incremental rewrites of tree like data structures. Finally we look at `with` patterns which extend Meander's with recursion, opening up Meander to become a full-fledge programming language.

Meander is a picture into what functional programming with first class data manipulation might look like. Just as functional programming has freed us from the drudgeries of imperative manipulation and mutation of data structures, Meander hopes to free us from the linear, nested transformation pipelines by giving us the capability to talk about our data directly.