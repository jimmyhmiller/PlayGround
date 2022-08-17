# The Space Between Programs
The “space” between our programs is an aspect we almost entirely overlook. Yet, it is crucial to many aspects of our design. By “space”, I don’t mean the interface that the program exposes. Nor do I mean the means of communication (http, a queue, a log, etc). I  am instead looking at something more amorphous. If we think about programs and their connectors taking up space on a 2d plane, what I want to talk about is the negative space. Or in other words, what do our programs leave out, what do our programs leave room for, and what, by their existence, have they excluded?

These are the question we rarely ask ourselves when drawing an architecture diagram, and yet these problems crop up time and time again. There are simple examples, the monolith that takes up all our conceptual space so that nothing else can exist. Then there are complex interactions: the placements of these three services make it so that our new service must live in a crevice shoved between them all. 

This goes beyond the technical and conceptual into the social. Programs that live too close or too far from each other have impact on how the teams that maintain them relate. Improper spacing of code can lead to improper organization relations (a sort of reverse Conway’s law). Finding the correct amount of space is incredibly difficult.

## Artificial Space
There are programs where the space between them has been forced. Two programs that really ought to be one. I don’t want to dive into the criteria of what makes it so that we ought to unify programs, but in companies, there are some symptoms. When there is constant confusion or fighting about which team should be responsible for what, that is a good sign that the space is artificial. The typical advice is to make a defined interface. This is a bandaid fix.

The other answer is to eliminate the space between these programs. To make it so there are no longer two programs. Typically the answer here is not to combine the teams. There are usually organizational reasons for the division. Instead, if you are on this team, start finding different but related responsibilities that are truly separable. Begin to work on those, and give up your old responsibilities to the other team.

## Vast Space
Then there are programs for which there should be space, but the amount of space is just too great. There are two options for remedying this issue, make the programs move closer to each other, or introduce a program in the middle. It might seem like the latter option doesn’t decrease the space. But by making the space easier to traverse, the distance has been essentially reduced.

There are circumstances where simply bringing the two programs closer is the right answer. But I do think that leaning towards adding an intermediary is nearly always the right answer. First, it may turn out that there is a third program out there that could use its space reduced as well. Secondly, it is much easier to combine programs than to split them. So, starting with the intermediary gives us more flexibility to design when and where to combine.

## No Space
"No space" can be seen as the classic monolith. Secretly in our one program, there are many programs, looking to free themselves. Finding these programs can sometimes seem difficult. But it often isn’t. What is difficult is having the courage to split them out and the vision to see the consequences of doing so.

Nearly always the answer for splitting things out is to make some sort of lever you can control, in near real-time, to decide whether things go down the old code path or get redirected to our new program. Absent this, you are making a migration for which there is no going back. That is a dangerous prospect. Instead, start by being able to direct 10% of your traffic to the new program, and ramp up and down. Ensure your new program has adequate monitoring and alerting. 

## What Space Creates

We often focus on the technical properties that technologies enable, but we forget to think about the human aspects. By having space, we make it easier to think certain thoughts and harder to think others. Space creates possibilities, but also hinders accessibility. 

Space also determines social circumstances. Creating space between two systems can bring about competition between two groups. Each looking to own, or not own, some part of the process. Lack of space can create entirely artificial conflict, as decisions that should be local become much larger. With a lack of space, conformity becomes the norm.

## Conclusion

Space is a complex topic for which there are no easy metrics or measures we can use to decide on how to resolve the issues. Nor to even recognize the problems. These decisions aren’t merely decisions that should be made based on things like performance. Reducing system design down to a set of objective metrics ignores so much of what is important in software design. We must consider the larger implications of our choices, the impact they have on the future, and the social implications of making these choices.