# Defense and Critique of "Programming as Theory Building"

Peter Naur's essay "Programming as Theory Building" is an exploration into the essence of programming. What is the activity of programming? How does this notion affect our practice? As Naur sees it, understanding the nature of this activity is paramount, for, "If our understanding is inappropriate we will misunderstand the difficulties that arise in the activity and our attempts to overcome them will give rise to conflicts and frustrations." Unfortunately, despite Naur's consciousness raising essay, we find ourselves in exactly this predicament.

This essay is an attempt to explicate Naur's notion of "Programming as Theory Building" in order to clear up misunderstanding. We will begin by reviewing common themes in the secondary liturature. In general the reception of Naur's essay was positive and yet, even in works favorably disposed towards it, major criticism are livied which belie a misunderstanding of the theory itself. Insufficent focus has been put on Naur's philosophical underpinnings. 

Naur's essay has also attracted a host of industry focused commentators, mainly those in the XP and Agile circles. While industry use of Naur's concept should definitely be encouraged, it seems to largely be used in order to bolster the already held beliefs of these authors. After having explained further Naur's philosophical background, we confront these applications, showing in what ways they are misunderstandings. Finally the paper presents novel applications , which  are not only more faithful to Naur's work, but also pay proper attention to its philosphical underpinnings.

## Common Objections

The secondary literature is riddled with positive mentions of the overall thrust of Naur's essay, yet few if any authors take it upon themselves to expand on Naur's notions. Instead, commentators immediately either weaken Naur's thesis by removing some of its more bold statements, or believe that Naur, while insightful, was ultimately aiming at the wrong target. 

#### Communicating Theory - (Needs rewritten)

The primary target of weakening might be called the "Incommunicablity of Theory". Naur makes a number of bold statements concerning the limits of communicating a theory that has been built by a programmer. 

> A main claim of the Theory Building View of programming is that an essential part of any program, the theory of it, is something that could not be conceivably be expressed, but is inextricably bound to human beings.
>
> …the theory could not, in principle, be expressed in terms of rules
>
> ..the knowledge possessed by the programmer by virtue of his or her having the theory necessarily, and in an essential manner, transcends that which is recorded in the documented products.
>
> …reestablishing the theory of a program merely from the documentation is strictly impossible.

Because of this strong claim, it may be tempting to think that Naur isn't exactly being clear with this claim. Perhaps the second quote is all Naur was getting out; maybe we need more than rules to express the theory, we also need metaphors and all the beauty natural language offers. But Naur makes certain he is not misunderstood, by continually repeating this claim throughout the essay.

Given its counter-intuitive nature, it isn't surprising that people seek to weaken this thesis. Unfortunately it isn't clear how one could abandon this claim without also abandoning Naur's larger project. The incommunicability of theory is a necessary consequence of Naur's notion of theory.

#### Social Aspects of Programming

The second criticism levied against the Theory Building View as presented by Naur, is not in what it includes, but what it excludes. As Naur's critics paint it, the Theory Building View takes an entirely individualistic approach. Today, it is claimed, software isn't created by isolated individuals, but rather by teams. Software isn't even just created by "programmers", instead, a whole host of people are included in the construction of software – project mangers, stakeholders, scrum masters, and even the customer. Naur's theory with its focus on the individual programmer misses much that is essential to modern day software development.

While this criticism does not explicitly weaken any part of Naur's thesis, for many authors, once this social aspect is explored, it has the same consequences as the first criticism. How could a theory be incommunicable if it is necessarily built socially? Rather than a particular theory built by a programmer, a team of creative people must come to a "shared theory" (Anticipating Reality Construction). This "shared theory" involves people communicating their individual theories in order to build a common language they can use to talk about the problem at hand. We will show that these social criticisms face a dilemma, either they are reducible to the first criticism, or they fail to criticize Naur's thesis altogether.

## Philosophical Background

What may be remarkable about Naur's essay for many in computer science is its extensive reliance of philosophical texts. Nearly half of Naur's citations are philosophical texts not directly related in anyway to computer science. Unfortunately, the length of Naur's essay does not allow him to articulate more clearly exactly how the content of his ideas relate to this philosophical background. Instead, Naur assumes his readers are intimately familar with the works he cites, and further assumes agreement with the content of these texts. While Naur cannot be faulted for doing so, it is this unfortunate circumstance that has led this the vast misunderstandings of this groundbreaking paper.

In this section we dive into three different philosophers that Naur's work relies on. First and most importantly we will explore Gilbert Ryle's notion of theory building. This work by Ryle forms the backbone of Naur's entire thesis. Without this philosophical background, making sense of Naur's paper seems a nearly impossible task. Second we shall take up two great thinkers in the philosophy of science, Popper and Kuhn. Here we will give an outline of their works that are certainly assumed in Naur's paper, but not explicitly relied upon. Naur cites both of these authors but in a rather focused context, as support for Ryles notion of theory building. This will not be our focus, instead we will use these sorts to draw new implications from Naur's thesis. These implications, with the explicit reliance on the philisophical heritage Naur has adopted, remain faithful to the project Naur began.

### Ryle's Theory Building

Gilbert Ryle is most known for his critique of Cartesian Dualism, the notion that the mind is an immaterial substance separate from the body. Ryle is famous for coining the phrase "the ghost in the machine" as a rather unflattering way of describing what Ryle believes dualists are commited to. Luckily enough for us, the details of Ryle's critique are not important to Naur's work. In fact, depsite Naur's citation of Ryle's *The Concept of Mind* nothing Naur mentions has anything to do with the mind-body problem. Ryle's work, while remembered for its groundbreaking critique has a much broader scope. Ryle's behaviorist theory, requires that he separate mental talk from observable behavior. This self-imposed constraint requires Ryle to re-examine intellectual activity broadly and from this we get beautiful descriptions of various intellectual activities, most important for our purpose, the activity of theory building.

It is here that difficulties arise. Theory is a word with many meanings and uses not all of which match Ryle's. In fact, what we have in Ryle is a technical notion of theory. Theory for Ryle, while multifaceted has particular meaning, one described in Ryles work not by definition, but by explicating its relations. In order to understand what Ryle means by a theory, we must pay close attention to these various relations and constraints he places upon it. We must distinguish between the act of building a theory, operations on a theory, and the theory itself.

Ryle does not intend his notion of theory to extend only to a particular discipline, Marx and Sherlock Holmes, while differing in subject matter and method, surely built theories. Someone laboring to discover how to lay carpet in a room, making measurements, determining which way the carpet ought to be laid, is caught up in the act of theory building. The historian, as he studies the accounts of a battle, is building a theory about the battle's procedings. Given this broad view of theory building, it should be not be a surprise that programming will fall into this category as well.

#### Theory and Communication

But what sort of thing is a theory that these people are building? We may be to tempted to identify a theory with some set of statements. For example, we may talk about Newton's theory of motion by stating his three laws. We may speak of Sherlock's theory of a case by citing a passage in which he lays out his conclusion as well as the twists and turns along the way that lead him to this conclusion. How can a theory be incommuncable if a theory just is a statement of some position?

This is where Ryle's attention to detail pays off. Ryle helps us by making more precise the notion of theory, separating it from its manifestions and operations upon the theory. First Ryle wants to separate out building a theory from having a theory. 

> To have a theory or a plan is not itself to be doing or saying anything, any more than to have a pen is to be writing with it. To have a pen is to be in a position to write with it, if occasion arises to do so; and to have a theory or plan is to be prepared either to tell it or to apply it, if occasion arises to do so. The work of building a theory or plan is the work of getting oneself so prepared.

Here Ryle contrasts the process of building the theory, from having a theory by talking about our abilities after we have a theory, the ability to state or apply our theory. From this it would seem that a theory is some sort of proposition. If this is so, the work of building a theory would be that of memorization in order to recite a propositional statement. But this isn't quite right.

> Having a theory or plan is not merely being able to tell what one’s theory or plan is. Being able to tell a theory is, in fact, being able to make just one, namely the didactic exploitation of it. Mastery of Euclid’s theorems is not merely ability to cite them; it is also ability to solve riders to them, meet objections to them and find out the dimensions of fields with their aid.

To put this in more technical terms, being able to "tell what one’s theory or plan is" is a necessary, but not sufficient criteria for having a theory. Having a theory must go beyond mere recitation. Theories are things which can be applied and to have a theory requires the ability to apply that theory. Theories are varied in their presentation and use and a certain level of mastery is required in order to claim possesion of a theory.  To put these in Ryle's terms, having a theory involves aspects of knowing how and knowing that.

####Knowing How and Knowing That

While not the originator of the idea, Ryle offers a spirited defense of the distinction between knowing how and knowing that. Roughing speaking to "know how", is to have the ability to perform an action. Whereas "knowing that" is to justifiably believe a true fact. In *The Concept of Mind*, Ryle intends to defend two claims concerning these types of knowledge. First, "knowing how" and "knowing that" are not reducible, they are two distinct forms of knowing. Secondly, "knowing how" doesn't require prior instances of "knowing that".  This second claim we will not explore further, but in order to understand Ryle's notion of theory, we must explore more Ryle's separation of these two forms of knowing.

The quintessential example used in philosophy when discussing this topic is that of a juggler. We are led to imagine a skilled juggler, who with ease can juggle various objects in various amounts. His hands rise and fall in perfect time with the objects. This is a perfect example of "knowing how", our juggler has a certain intuitive understand of the objects he is manipulating. He knows how high to throw an object, how quickly to move his hands to catch them as they fall; his knowledge is made evident by his performance. 

There are some who suggest "knowing how" is just a species of "knowing that". In the case of our juggler, the suggestion might be that there are certain propositions that the juggler knows such as "If I am juggling N objects, I need to throw them up X feet at Y angle." In fact, if "knowing how" is reducible to "knowing that", our juggler knows all sorts of propositions of this type. But, as Ryle argues, one could know all possible propositions about juggling and still themselves not be a skilled juggler. Juggling requires practice, it requires "muscle memory", it requires us to build up knowledge of how, not just propositions about the activity of juggling.

> To be a Newtonian was not just to say what Newton had said, but also to say and do what Newton would have said and done. Having a theory is being prepared to make a variety of moves, only some of which are teachings; and to teach something to someone, oneself or another, is, in its turn, to prepare him for a variety of tasks, only some of which will be further teachings.

Here we can see Ryle's insistence that theories require a sort of "knowing how". We are to be "prepared to make a variety of moves". Having a theory requires the ability to know how to wield this theory. It requires being able to put the theory into practice, to use it for its particular ends. This is not merely to sit in an arm chair and draw conclusions from it, but to know our theory so well, we know how to teach it, how to answer queries about it, how to relate it to other things, how to modify it in the face of new evidence. Each different type of theory has its own use that for which it must be employed.

> Sherlock Holmes’ theories were primarily intended to be applied in the apprehension and conviction of criminals, the thwarting of planned crimes and the exculpation of innocent suspects….His theories were applied, if further deductions were actually made from them, and if criminals were arrested and suspects released in accordance with them.

### Popper

Naur's citation of Popper is used as support for Ryle's notion of theory. Exploring exactly what Naur means by "unembodied World 3 objects" would certainly be an interesting investigation and perhaps enable us to support Ryle's thesis more, but that is unfortuantely outside the scope of this essay. Instead of focusing on the aspects of Popper's philosophy Ryle explicitly cites, we will shift our focus to Popper's more famous work, falsification. Falsification did not make its way into Naur's writing nor does it seemed to be implicit in any of his statements. Yet, as shall later be shown, if programming is theory building, falisification may be an illuminating way to view certain activities programmers routinely participate in.

#### Demarcation

Popper's primary work rests squarely in the field of philosophy of science. For those unfamiliar with the discipline, philosophy of science approaches science from a meta position asking questions like how is scientific knowledge possible, how might we metaphyiscally interpret our scientific discoveries, and what properly counts as science. It is this last question that Popper was most concerned with. There are tons of things that masquarade as science, things that today may be called pseudo-science. Sometimes the line to draw can be rather obvious, very few people today believe that Astrology or Alchemy ought to be considered sciences, but mere agreement doesn't provide us a criteria. It would seem that what is and isn't science shouldn't fall to popular opinion. 

Popper believed that a definite criteria for what was science and what was not was critically important. Scientific progress depended on our ability to distinguish these to make a demarcation criteria. If not our time and effort would be wasted on unscientific investigation. Popper proposed a criteria known as Falsification. A theory is scientific just in the case that it makes predictions that could be shown to not occur. In other words, if our theory predicts that A will occur in circumstance C, we can falsify our theory by creating circumstance C and if A does not occur, our theory is false.

Popper motivation for this criteria came from a few different places. First, verificationism had remained a dominant view in Popper's time. The basic idea behind verificationism is that we must be able to verify any statements that we make. Unfortunately as Popper pointed out, we cannot verify empirical generalizations. Just because a ball has fallen when dropped 1000 times before, does not mean it will in the future. Secondly and more controversially, Popper saw Freudian psychology as unscientific. Freudian psychology appeared to have an answer for everything. No matter the circumstance, a Freudian could give an explanation. There was no way in practice to show that Freudian psychology was wrong.

Popper was obviously not the last word on the demarcation criteria. There have been many arguments against Popper's criteria. Some argue that Popper's criteria allows things which aren't science to slip in. While others believe Popper has actually excluded quite a bit of what is scientific. We are not here interested in settling this debate, but one point seems fairly uncontroversial. Falsification is a possitive attribute for any theory. A theory gains strength as it makes predictions that are falsifiable, but when testing them, they fail to be falsified. If programming is theory building, this feature may be important to pay attention to.

### Kuhn

As with Popper, Naur doesn't cite Kuhn in support of Ryle. Unlike Popper, the work cited for Kuhn is his most famous most influencial work. While not explicit, Naur's citation of Kuhn, shows that he was not opposed to thinking of programming from a Kuhnian lens. In a later section, we shall take up this idea, show quite a different application of Kuhn to the real of programming than is normally discussed, but first we must make it clear what Kuhn was exploring with his discussion of "paradigms". 

#### Paradigmatic Paradigms

Kuhn like Popper was concerned with the question of demarcation. He explicitly disagreed with Popper that  normal science functioned through falsification. In order to explain his demarcation criteria, Kuhn needed to first explain how science operates. For Kuhn this was not discoverable by a set of abstract principles like Bacon's scientific method, but instead, required historical and socialogical work. It is this work we shall focus on and later show in what ways it applies to programming and in what applications are misguided.

Kuhn's way of looking at science was to divide the progress of various disciplines into periods of what he called "normal science" that are punctuated by paradigm shifts. A paradigm shift transformed "normal science" into an entirely different way of operating. A standard example of a historical paradigm shift was from Newtonian Classical Mechanics to Einsteinian Relativity. The change to a relativistic view of physics was not merely a change in some premise or some observed fact, it was an entirely different way of interpreting the scientific enterprise of physics.

Once the Einsteinian paradigm has been accepted, there is no easy way to recover a purely Newtonian viewpoint in order to make an apples to apples comparison. The conceptual apparatus by which we view the world has changed, the problems we are solving having changed, our language takes on new meanings. These features make it hard if not impossible to have a criteria for measuring the success or failure of a theory apart from the paradigm it inhabits. 

For Kuhn, these paradigm shifts were the chief mechanism by which scientific progress happened. They were historical moments. They showed that science was not a linear march towards truth, but rather required puncuated breaks from the past, complicating science's relationship to truth and rationality. But, despite this, these paradigms shifts did offer a way forward. A paradigm shift only occurs when some viewpoint or other gains a widespread popularity and becomes the dominant way a discipline is viewed. When we later investigate Kuhnian paradigms in programming, we must keep this notion of historical progress in mind.

##Objections Answered

### Theory as "Knowledge How"

With this philosophical background laid out, it becomes much easier to see what Naur might mean by statements like:

> A main claim of the Theory Building View of programming is that an essential part of any program, the theory of it, is something that could not be conceivably be expressed, but is inextricably bound to human beings.

Having a theory is to know how to do something. If a programmer has a theory about a program, he will know how to make changes to that program, we will be able to answer questions about that program, he will be able to explain parts of the program to others. But none of these presentations or applications of the theory of the program is the theory itself.  Donald Knuth criticizes Naur by suggestion that with literate programming we can in fact communicate the theory of a a program. But given Ryles notion of  theory, this can be seen to be a category mistake. Documentation, no matter how comprehensive, is just a presentation of the theory the programmer has.

Given that a theory is a "knowing how" and not a "knowing that", it is hard to understand the criticisms of this aspect of Naur's position. Is the disagreement about whether knowing how is reducible to knowing that? It doesn't seem to be that. Is the disagreement a matter of not seeing programming as a processing of theory building? This does not seem to be where the disagreement lies either as most authors are very positive about this fact. It seems instead this disagreement comes from a misunderstanding of what a theory is for Naur. If that is the case, it seems hard to take this as a serious critique.

### Social Theory Building in Naur

The social critique of Naur seems to claim that Naur has ignored something that is essential to software construction, namely the social aspects of it. By failing to attend to these, Naur has come to incorrect conclusions. These criticisms allege that rather than an individual theory, we must come to a "shared theory" about the software we are creating. Further, we must also recognize that more than just programmers are involved in the software construction process.

What are we to make of the notion of a shared theory? How can there be a shared theory given that a theory is a kind of "knowing how"? Let's take jugglers as an example. One way you might supposed that jugglers could share a kind of "knowing how" would simply be that they have the same abilities. Two jugglers that can each juggle three balls using the same technique may be said to "share" knowledge. Is this is what is meant by a "shared theory"? Well, given the variety of people that are supposed to be part of this "software construction" process that can't be the case. The project manager and programmer do not share the same skillset.

Another way jugglers might "share" in their "knowing how" is to juggle together. But we must supposed in order to for this case not to reduce to the first that the jugglers are doing different things, perhaps the one is throwing high and the other is throwing low. Is this is what would be meant by a "shared theory"? If so it hardly seems to disagree with Naur. Nothing in Naur claimed their couldn't be colaboration between different people. In the jugglers case there is still an incommunicable element, a "knowing how" that isn't reducible to rules and if this is equivilant to the "software construction" case, then Naur's contention remains.

### Locating the Disagreement

It is actually quite difficult to locate the real disagreement that is happening with these criticisms. Take for example this passage:

> At this point Naur's approach proves unsatisfactory. After all, the need to build a theory in software projects and to reflect upon it arises not least because of the necessity of reaching a common understanding of the various requirements and different perspectives among the persons involved. Thus the specific implications of cooperation in software development for the process of theory building must be systematically taken into account as well.

This passage is far from clear. What does he mean by "the need to build a theory"? If programming just is theory building, then there isn't a need for which the theory is built, the act of programming just is the act of building a theory. In other words, a theory would be built even if there were no "necessity of reaching a common understanding". Rather than trying to understand the passages criticizing Naur, let's look at the positive theory put forth in place of Naur's.

>		Shared theory is, basically, communicated individual theory. The expertise required to determine and evaluate the use purpose of a software object can only be acquired in the course of a communicated, mutual learning process. This applies particularly where not only objective and economic considerations have to be taken into account, but also the individual and collective quality requirements of the users.

It is incredibly confusing to see how this passage makes sense in light of understanding theory has a "knowing how". But if we read closer into other passages, we can see that this idea of theory is actually just given up or perhaps never understood to begin with.

> Anticipations of purpose can be seen as conceptions, as ideas about essential features of objects for future use and of the activities which they mediate
>
> I view the establishment of a new anticipation of purpose as "theory building" about the use purposes of the future software objects.
>
> In participative software projects, 'theory' means a common and moreover a mutually mediated anticipation of purpose as a frame of reference for the cooperative work processes. 

These difficult to understand passages suggest that a theory is a kind of "knowing that". Theory has to do with "ideas about essential features of objects for future use and of the activities which they mediate". So a theory is not a "knowing how". So is this a criticism of Naur? It is hard to see how it could be.

## Inappropriate Applications

### The Attempt to Bolster Agile

## New Applications

### Kuhn's Paradigms are programs

### Popper and TDD

### Rewrites, Resurrection, and Baxandall

##Conclusion