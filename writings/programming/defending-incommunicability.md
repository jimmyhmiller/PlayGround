# Defending the Incommunicability of Programs

Peter Naur's essay "Programming as Theory Building" is an exploration into the essence of programming. What is the activity of programming? How does this notion affect our practice? As Naur sees it, understanding the nature of this activity is paramount, for, "If our understanding is inappropriate we will misunderstand the difficulties that arise in the activity and our attempts to overcome them will give rise to conflicts and frustrations." Unfortunately, despite Naur's consciousness raising essay, we find ourselves in exactly this predicament.

As the title suggestions, Naur claims that programming is first and formost a process of theory building. What Naur means by this is that what is most fundamental to the process of programming is not the source code, the documentation, or even the running program, it is instead the knowledge the programmer has.

> One way of stating the main point I want to make is that programming in this sense primarily must be the programmers’ building up knowledge of a certain kind, knowledge taken to be basically the programmers’ immediate possession, any documentation being an auxiliary, secondary product.

This notion of programming as theory building has recieved in general wide praise in secondary literature. It seems to be recognized widely that Naur's view brings quite a bit to the table. But there is an aspect of Naur's view that has not recieved quite so much praise. This aspect may be called the "incommunicability thesis". Naur puts it this way:

> A main claim of the Theory Building View of programming is that an essential part of any program, the theory of it, is something that could not be conceivably be expressed, but is inextricably bound to human beings.

This seemingly radical thesis is not just a consequence of Naur's view, but rather a main claim of it. In other words, to view programming as theory building, one must see that these theories are in some sense incommunicable. Given the strong importances Naur places on this thesis, it is suprising that commenters who are even supportive of the view tend to reject this major claim. For instance, Donald Knuth writes:

> My experiences agree well with Peter Naur's hypothesis that programming is "theory building," and they strongly support his conclusion that programmers should be accorded high professional status. But I do not share Naur's unproved assertion that "reestablishing the theory of a program merely from the documentation is strictly impossible." On the contrary, I believe that improved methods of documentation are able to communicate everything necessary for the maintenance and modification of programs.

The goal of this essay is to defend Naur's view that theory is incommunicable. Naur himself does provide such a defense, but his defense took for granted philosophical background knowledge not shared by his readers. Not only this, but Naur's defense is indirect. Here we will make explicit what was implicit in Naur, namely his reliance on Gilbert Ryle's notion of theory building and division of knowledge into two distinct kinds. Having explicated the background Naur assumed, we will examine more closely his claims.

### Ryle's Theory Building

Gilbert Ryle is most known for his critique of Cartesian Dualism, the notion that the mind is an immaterial substance separate from the body. Ryle's work, while remembered for its groundbreaking critique of dualism has a much broader scope. Ryle's behaviorist theory, requires that he separate mental talk from observable behavior. This self-imposed constraint requires Ryle to re-examine intellectual activity broadly and from this we get beautiful descriptions of various intellectual activities, most important for our purpose, the activity of theory building.

It is here that difficulties arise. Theory is a word with many meanings and uses not all of which match Ryle's. In fact, what we have in Ryle is a technical notion of theory. Theory for Ryle, while multifaceted has particular meaning, one described in Ryle's work not by definition, but by explicating its relations. In order to understand what Ryle means by a theory, we must pay close attention to these various relations and constraints he places upon it. We must distinguish between the act of building a theory, operations on a theory, and the theory itself.

Ryle's notion of theory extends accross disciplines, Marx and Sherlock Holmes, while differing in subject matter and method, built theories. Someone laboring to discover how to lay carpet in a room, making measurements, determining which way the carpet ought to be laid, is caught up in the act of theory building. The historian, as he studies the accounts of a battle, is building a theory about the battle's procedings. Given this broad view of theory building, it should be not be a surprise that programming will fall into this category as well.

#### Theory and Communication

But what sort of thing is a theory that these people are building? We may be to tempted to identify a theory with some set of statements. For example, we may talk about Newton's theory of motion by stating his three laws. We may speak of Sherlock's theory of a case by citing a passage in which he lays out his conclusion as well as the twists and turns along the way that lead him to this conclusion. How can a theory be incommuncable if a theory just is a statement of some position?

This is where Ryle's attention to detail pays off. Ryle helps us by making more precise the notion of theory, separating it from its manifestions and operations upon the theory. First Ryle wants to separate out building a theory from having a theory. 

> To have a theory or a plan is not itself to be doing or saying anything, any more than to have a pen is to be writing with it. To have a pen is to be in a position to write with it, if occasion arises to do so; and to have a theory or plan is to be prepared either to tell it or to apply it, if occasion arises to do so. The work of building a theory or plan is the work of getting oneself so prepared.

Here Ryle contrasts the process of building the theory, from having a theory by talking about our abilities after we have a theory, the ability to state or apply our theory. From this it would seem that a theory is some sort of proposition. If this is so, the work of building a theory would be that of memorization in order to recite a propositional statement. But this isn't quite right.

> Having a theory or plan is not merely being able to tell what one’s theory or plan is. Being able to tell a theory is, in fact, being able to make just one, namely the didactic exploitation of it. Mastery of Euclid’s theorems is not merely ability to cite them; it is also ability to solve riders to them, meet objections to them and find out the dimensions of fields with their aid.

Having a theory must go beyond mere recitation. Theories are things which can be applied and to have a theory requires the ability to apply that theory. Theories are varied in their presentation and use and a certain level of mastery is required in order to claim possesion of a theory.  To put these in Ryle's terms, having a theory involves aspects of knowing how and knowing that.

#### Knowing How and Knowing That

While not the originator of the idea, Ryle offers a spirited defense of the distinction between knowing how and knowing that. Roughing speaking to "know how", is to have the ability to perform an action. Whereas "knowing that" is to justifiably believe a true fact. In *The Concept of Mind*, Ryle intends to defend two claims concerning these types of knowledge. First, "knowing how" and "knowing that" are not reducible to each other, they are two distinct forms of knowing. Secondly, "knowing how" doesn't require prior instances of "knowing that". This second claim we will not explore further, but in order to understand Ryle's notion of theory, we must explore more Ryle's separation of these two forms of knowing.

The stereotypical example used in philosophy when discussing this topic is that of a juggler. We are led to imagine a skilled juggler, who with ease can juggle various objects in various amounts. His hands rise and fall in perfect time with the objects. This is a perfect example of "knowing how", our juggler has a certain intuitive understand of the objects he is manipulating. He knows how high to throw an object, how quickly to move his hands to catch them as they fall; his knowledge is made evident by his performance. 

There are some who suggest "knowing how" is just a species of "knowing that". In the case of our juggler, the suggestion might be that there are certain propositions that the juggler knows such as "If I am juggling N objects, I need to throw them up X feet at Y angle." In fact, if "knowing how" is reducible to "knowing that", our juggler knows all sorts of propositions of this type. But, as Ryle argues, one could know all possible propositions about juggling and still themselves not be a skilled juggler. Juggling requires practice, it requires "muscle memory", it requires us to build up knowledge of how, not just propositions about the activity of juggling.

> To be a Newtonian was not just to say what Newton had said, but also to say and do what Newton would have said and done. Having a theory is being prepared to make a variety of moves, only some of which are teachings; and to teach something to someone, oneself or another, is, in its turn, to prepare him for a variety of tasks, only some of which will be further teachings.

Here we can see Ryle's insistence that theories require a sort of "knowing how". We are to be "prepared to make a variety of moves". Having a theory requires the ability to know how to wield this theory. It requires being able to put the theory into practice, to use it for its particular ends. This is not merely to sit in an arm chair and draw conclusions from it, but to know our theory so well, we know how to teach it, how to answer queries about it, how to relate it to other things, how to modify it in the face of new evidence. Each different type of theory has its own use that for which it must be employed.

> Sherlock Holmes’ theories were primarily intended to be applied in the apprehension and conviction of criminals, the thwarting of planned crimes and the exculpation of innocent suspects….His theories were applied, if further deductions were actually made from them, and if criminals were arrested and suspects released in accordance with them.


### Programming as "Knowing how"

With this philosophical background laid out, it becomes much easier to see what Naur might mean by statements like:

> A main claim of the Theory Building View of programming is that an essential part of any program, the theory of it, is something that could not be conceivably be expressed, but is inextricably bound to human beings.

Having a theory is to know how to do something. If a programmer has a theory about a program, he will know how to make changes to that program, we will be able to answer questions about that program, he will be able to explain parts of the program to others. But none of these presentations or applications of the theory of the program is the theory itself. 

If no presentation of a theory is the theory itself, it follows that no documentation can capture fully the theory of any program. If having a theory requires being able to perform certain moves, in the case of programming, making modifications to a program, then no program contains its own theory. In fact, no text, no video, no media of any sort can contain the theory of a programming. Having a theory is something only a human can do.

### Implications

The theory of the programs we write live inside us. As we move onto new projects, leave old companies, or forget about code we once wrote, those theories die. The code we wrote might live on. People may continue to run it, read it, modify it, but some understanding of that code lives (or lived) only in our heads. Try as we might, we cannot communicate this theory in its full.

As we inherit code bases writen by others, we encounter the same things, code bases whose theory are gone to time or locked away in a location unaccessible to us. Programs that have been modified under many different theories, often incompatible ones. Bugs begin to find their way in, often at the intersection of two theories. Hacks begin to emerge as needs evolve, but theory is lacking. The code becomes worse and worse.

These are the facts we face as professional software engineers. We write code whose theories will one day be lost and we inherit code bases whose theories are long forgotten. If we are to do programming well, we most adopt practices that address these issues.
