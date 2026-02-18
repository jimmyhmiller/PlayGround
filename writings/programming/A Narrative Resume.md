# A Narrative Resume

I've never been happy with a resume. The roles I've had, the technologies I've worked with, don't seem to fully define my career. Each job was more than just the technologies I used. More than the responsibilities I had. They shaped who I am as a software engineer. A resume does not tell you if you want to work with me. It does not tell you if I share the values your company holds. This is my attempt to do that. 

But I will help you out. If you actually are someone whose recieved my regular resume and chosen to come visit this page, each section will be the highlights. In fact if you hit the button just the right of resume. All the narrative collapses. And what you are left with is the resume as we normally concieve it. With one caveat, it is in chronological order. Rather than seeing what I've done lately you can see how I've grown.

## Heartland Payment Systems (2012-2014)

While I didn't write this directly in the blog post. It isn't a stretch to figure out my ["We ran out of columns" - The best, worst codebase](https://jimmyhmiller.com/ugliest-beautiful-codebase) was about my time Heartland Payment Systems. My first tech job. In fact, you can hear more about it about in [my interview with the Changelog](https://changelog.com/podcast/609) or if you are an audiobook listener, you can join 1M+ listeners and hear it [read to you by the Primagen](https://www.youtube.com/watch?v=uPrXEtvKFoI).

I won't recount all the details of this codebase here. But I'll give you some highlights of the tech stack. It was primarily C# codebase. But quite a bit of Visual Basic, a sprinkle of delphi, and a number of other old technologies like JScript.

After my internship quickly turned into a Junior Developer job, I was the junior developer mentoring the interns in a support, bug fixing adjacent role. Basically, the task of our group (1 Senior developer, 1 Analyst, 1 produce person, 5 interns and myself) was to fix the bugs and handle the manual data requests  that the business didn't want to spend money on. In practice, we did a ton of Brownfield development. Our software was made for internal customer support people and we talked to them directly. Gather requirements and built them software with in the existing monolith.

In this role I got to do all sorts of fun things. We had paid a contractor for custom time tracking software used by all the employees. But we had lost the source for it. So we had checked in decompiled sources for the software in source control. My job was to add new features using the decompiled sources as my source code. I spent a brief stint as the solo developer on a greenfield project. They brought in a senior developer to the team, but he was reassigned before doing to much work. The project pivoted.

Looking back, I didn't do a great job on the project but also about what I'd expect for an engineer with my experience and this level of instruction. This was in an era where rest was peak hype. I was told to make two different rest api servers for this one project. The first would access the database directly and expose the database tables with a one to one rest interface. The second would be a rest api that would talk to this data layer rest api. Then I was supposed to write a single page application in Backbone js or something similar. Instead, I used C# mvc with Razor as the templating language. 

I don't think either of these were the right choice. But I'm not sure looking back that there was a great choice for this project. It was ill defined, mostly demo ware. Disconnected from what customers needed. In many ways this was foreshadowing for projects to come later.

### What I Learned

I learned the power of small groups who are given freedom and have direct feedback on their work. We worked directly with our "customers" (fellow employees) to build them software particularly tailored to their jobs. We were able to save people hours of work with relatively simple solutions. But perhaps the biggest lesson that stuck in my head came from Justin the senior developer on my team. All of our customer support people used the Merchant Search page. An absolutely massive page that over the years had gotten slower and slower. Without being asked, Justin rewrote the page to go from an average page load of 3 mins, to 3 seconds. The massive impact a single developer can have just by doing what is right has stuck with me to this day.

## NextGear Capital  (2014-2016)

NextGear was a radical departure from the simple, informal structure I was used to at Heartland. We had the full suite of agile stuff. We had 4 different "scrum teams" working on the same project split across two managers. We had scrum masters. We had product people. We had the outside agile consultants telling us the "right way" to do things. But what we didn't have when I initially joined was actual work for 20+ newly higher engineers and QA.

NextGear was a newly created entity. Cox, the large private company, and had bought a bunch of smaller companies and stuck them together. These smaller companies had all their engineering contracted and now it was being brought in house. I joined as a mid-level software engineer and quickly found myself facing a leadership role I was ill-prepared for.

This was my transitions very rapidly from Junior to Senior. I lacked a lot of the patience and people skills I now understand are important for these roles (in my defense I was 22). But I get involved with a bunch of technolgies I hadn't been exposed to before. First, there was Java. Now I honeslty knew Java fairly well. First it really isn't that different from C#, but second I had asked for a java book for my birthday when I was in 8th grade. I read that thing cover to cover and vowed never to write java. I didn't until the AP exam (I did get a 5).

But the java I knew was not the complicated Spring mess I encountered here. Our stack was angular 1 and Spring. The whole project was a mess. It was greenfield development. But the constant struggle between the teams to get work, the decisions of spreading out ownership constantly hopping back and forth between different teams caused the whole prokect to be inconsistent. We also were expressly forbidden from talking to customers (who were employees at our own company). 

My favorite technical aspect of this job was a part I got to work on separate from this main spring monolith. It was a queue for feeding data into the system. At the time we were told to expect 1000 messages a day at peak. So the devops group setup a simple rabbitmq setup and I made a single server (docker container) to read from. The first day in production I woke up, drove to work, checked the queue and it had 1.2 million messages and counting.

We did the math and our single worker was not going to handle the messages in time. In fact, the rabbitmq server was not going to last either. Worse, it didn't have persistence turned on and it was running out of memory. But there was a problem, ordering mattered. Not globally, but per entity. Luckily I had begged to attend strangeloop just a month earlier where I watched ["Building Scalable Stateful Services" by Caitie McCaffrey](https://youtu.be/H0i_bXKwujQ?t=882). There I learned about consistent hashing. To keep a long story short, we had do to some dancing of servers to get a rabbitmq with consistent hashing setup, had to code up a worker that would work with the fan out. But by 5pm that same day we had deployed a fix and were chopping down the queue.

### What I Learned

I learned a lot about the ways in which politics can affect a software project. I learned a lot about my own need to control my frustration. That perception of your actions can often be as important as your actions. I learned a ton about how to write code that a team can understand, not just myself. I learned a lot about bad practices to avoid in testing. But most importantly I learned that [things can change](https://jimmyhmiller.com/never-change). That people can make a difference if they work together.

## Trabian (2016-2017)

I was a big Clojure fan at this time (still love it). I had been searching on Linkedin to try and see of anyone locally (Indianapolis) was using Clojuire. I doubted that I'd find any companies. But I was considering starting a funcitonal programming meetup (which I did later do) and wanted to see if Individuals were using it. I found the CEO of a local company who was using it. I clicked on his profile, didn't think much of it. But after a bit he reached out to me.

Trabian was a react consultancy for fintech startups, banks, and credit unions. The CEO had been the primary programmer and wanted someone else to take this role and build out a team. That was my job. it was a very small company. When I joined I was the only full time engineer other than the CEO, we grew the team with two other engineers and two interns. We pumped out prototype after prototype. On a good day I was reviewing 8 PRs and write 3 myself. We wrote clean react. Each project we tried one new library to find what stack we believed worked best for our needs.

I also work in very early react native on Android, maintaining a our own fork for some not great reasons. Outside of client work we were working some server side work prototyping in both Clojure and Elixir around some graphql work utilizing CUFX. We did a head to head comparison and found Clojure better for our use case. Sadly this work didn't get off the ground before Trabian ran out of money. On a Wednesday I was told that my paycheck on Friday wasn't coming. By Monday, I had a contract position back at NextGear.

### What I Learned

I had been doing React since my days at Heartland. I taught a class on it at NextGear. At Trabian I got to see what a small team dedicated to writing good software can do. We use React and Redux, played with all the middleware you can imagine. I was able to finally have a job not stuck in one technology stack outside my control. I learned what it takes to make the right choices and how quickly you can fix bad ones if you plan ahead. I also learned the joy of quick prototyping.

## NextGear (2017)

I will keep this short as it was short. I joined NextGear for a quick stint after Trabian ran out of money as a contractor. There I worked on a neural network setup for classifying documents. The project was a bit of a mess so not much got done in those six months. But there was a big lesson learned and I will focus on that.

### What I Learned

At this point the project I worked on had been in production without me being around to see it. I saw how much had changed. I saw how poor decisions early had added up. But most importantly I saw the impact not being able to talk to customers had. On the wall in the room was a 3x5 index card with feedback from the "customers" who were just a few doors down. On it there was one card that will will never forget

>  Remember you are supposed to make our lives better, not worse - Customer feedback

## HealthFinch (2017-2018)

Heathfinch is one of the best companies I ever had the pleasure of working with. The people at this company deeply cared. The technology that was made there was fantastic. The culture was unmatched. It was of course not without its problems. But I will always look back fondly on that team. HealthFinch was my first fully remote job. My first job fully in Clojure. It was the first job where the software I made felt good to make.

HealthFinch created a rules engine that would help with prescription renewals. If a patient needed a certain test before renewing their prescription, our system would figure that out and tell the doctor about these requirements. It was a sophisticated piece of software that would explain precisely why it made those recommendations.

One of my favorite projects while I was there was actually working on a legacy ruby application that formed the template of an old version of the software. The templates were incredibly complex. But I [wrote a visualizer](/learn-codebase-visualizer) for them that showed you precisely where each bit of data had come from. Making a change that tool quite a while before almost instaneous.

### What I Learned

I learned just how productive Clojure can be. I learned that languages are not silver bullets for solving all your technical problems. I learned that the values people hold about code can lead them to vastly different solutions. I learned the value of mentoring as a means to learn yourself. I learned about logic programming and the power of state machines as an abstraction.

## Adzerk/Kevel

When I joined the company it was called Adzerk. By the time I left it had rebranded to one of the silliest names. Here I worked for the first time with people who I had known by reputation before ever working with them. Kevzerk had some big names in Clojure working for them. Many people who had worked directly with Rich Hickey. I have to admit, I was a bit surprised by the relative mess I had found when I joined. I was being handed off from the former maintainer the reporting side of this ad tech business. This involved hadoop clusters, redshift databases, a ton of queues and services. All of which were manually deployed, CI was non-existent, the servers were pets, not cattle. 

As the former maintainer steps onto other things, I was on a team of one for a while before we hired [Grzegorz](https://github.com/nabacg) to join the team. This was an absolute massive change. Together we were able to transform this hand built system into something with complete cloudformation defined services, all with CI, autoscaling, automatic deployments, automatic rollback, etc. Not only that, we did a live migration of all the old api traffic to our complete rewrite, all with zero downtime and no customer visiable breakages. 

I also wrote some very awful software using lambdas. There was a new process at the company obsessed with writing big long documents and I was given a project for which the document was needed. I quickly learned just how not value I found the process. Documents can absolutely be wonderful. And I've worked with many engineers who are great at producing them. But they are not how I think, I need to [discover the code](https://jimmyhmiller.com/discovery-coding) first, then write the document.

