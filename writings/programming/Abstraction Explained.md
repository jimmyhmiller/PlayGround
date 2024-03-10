# Abstraction Explained

Abstraction is information hiding. Or so the traditional wisdom goes. But how are we to understand this view? What kind of information are we hiding? Is all information hiding abstraction? Despite the vast literature dedicated to abstraction in computer science, these answers are not forth coming. Hitherto, no precisifcation of this theory has been offered. In this paper I will take this as my starting point. First to attempt to make the "information hiding" view of abstraction as precise as we can. Next to evaluate the view. Does the information hiding view do justice to the ways in which we use "abstract" in all its variations? Finally I will sketch the beginnings of an alternative theory of abstraction.

## Information Hiding

In many ways, it is difficult to call the information hiding view of abstraction a "theory" or an "analysis". To my knowledge no work has attempted to truly define what is meant by "information hiding". It is instead taken for granted. Consider Timothy Colburn and Gary Shute’s "Abstraction in Computer Science". The paper sets out to contrast abstraction in computer science vs abstraction in mathematics. The former is marked by information hiding, but the latter by information neglect. Yet, we are never told what information hiding is. Instead, we rely on our intuitive notions through examples like the follow:

> From a software point of view, a bit, or binary digit, is the interpretation given to a piece of hardware, called a flip-flop, that is always in one and only one of two possible states. It is an abstraction that hides the details of how the flip-flop is built.

Colburn and Shute continue on to describe the layers upon layers built upon this abstraction, each hiding more and more information. We can certainly make sense of these notions absent more precise specfication of what is meant. But how can we be sure this view has fully captured the essence of abstraction in computer science? Do all abstractions hide information? Does their information hiding play an explanatory role in the ways we think about various aspects of abstractions? We can only answer these questions by first making it clear what we in fact mean by this theory.

### Attempting a Definition

Making clear a concept that has been kept vague for half a century is a tall order. Over this time, the meaning attached this phrase have certainly grown to include quite distinct notions. So my attempt to make things precise should not be considered the final word. In fact, if I had one hope for this paper it would be that others will attempt to fix the mistakes in my analysis of information hiding. To shore up the areas I have failed to attend to. 

But despite the high potential for failure, we must procede. Beginning with "is". How are we supposed to under the "is" in "abstraction is information hiding"? Does the "is" here function as a categorical statement like in "pizza is food"? As an identity statement like "pizza is a flat round dough baked with toppings"? Or as a predicative statement like "Pizza is served hot"? Let's begin with the categorical reading. What would it mean for abstraction to fall under the category of "information hiding"? It is hard to see how to make sense of this. We don't think of information hiding as a categorical term like food or color.

So we are left with predicatve and identity. Of this two predicative is the more modest statement. Under this reading on attribute of abstraction is that it hides information. But even this is imprecise. If information hiding is an attribute of abstraction, is it a necessary or contigent attribute of abstraction? In other words, can there be abstraction with no information hidden? Consider, "pizza is hot". This may be seen as a true attribution of pizza, but it isn't a necessary predicate of pizza. Pizza doesn't cease to be pizza if it cools down. (Some weird people prefer day old, cold pizza). But pizza is edible seems to be a necessary attribute of pizza. A plastic look alike of pizza isn't in fact pizza.

So, on the information hiding view of abstraction, does abstraction necessarily involve hiding information? It is hard to see how it could be otherwise. It would be rather surprising if the view that "abstraction is information hiding" did not consider information hiding to be a necessary component of abstraction. so given, this are we warranted in taking this farther is abstraction identical to information hiding? For now, let us not take a stance here. Instead we will offer two different readings of this view, the strong view, on which abstraction just is "information hiding" (a term we still need to make precise) and the weak view on which abstraction is not identical to information hiding, but must necesarily involve information hiding. 

On the strong view, all information hiding (of the right sort), will count as an abstraction. On the weak view, there maybe some information hiding that does rise to the level of abstraction. This distinction will be important later. But both the strong and the weak views entail:

> 1. Abstraction necessarily involves information hiding

#### What is information hiding?

This singular criteria may hardly seem to be an advance at all. But it can often be helpful to state the obvious in order to uncover the non-obvious. Here the non-obvious is what do we actually mean by "information hiding". Supposed I am planning a surprise birthday party for my best friend and conceal this fact from them. Am I "hiding information" in the sense imagined above? It would seem not. What proponents of the information hiding view have in mind, is not all information that might be hidden, but specifically implementation details that are hidden. 

> Every module in the second decomposition is characterized by its knowledge of a design decision which it hides from all others. Its interface or definition was chosen to reveal as little as possible about its inner workings.  "On the Criteria To Be Used in Decomposing Systems into Modules" - D.L. Parnas 

A system which restricts access to a secret token isn't an abstraction, despite the fact that it is hiding information. Abstractions hide information about their implementation or other words, their internal workings. So we get 

> 2. The information being hidden must include implementation details

Hiding a surprise birthday party is not information hiding in this sense because it isn't about hiding implementation details, nor is hiding the auth key. But this hardly seems sufficient. Imagine I have a completely closed system which, has implementation details that are hidden from you. In fact, you cannot interact with the system at all. Is this abstraction? No, abstraction must involve not only hiding, but expose

> 3. Abstraction must involve exposing some means of interaction

Do these criteria jointly create necessary and sufficient criteria for abstraction? 