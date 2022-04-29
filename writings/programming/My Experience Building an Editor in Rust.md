# My Experience Building an Editor in Rust

I've always wanted to buid a text editor. I've played around before with trying to modify existing editors like codemirror. But ultimately those just felt incredibly unsatisfying. While I was able to make some fun little experiments with them, I was just gluing code together. As I started to play with Rust, it seemed like the perfect opportunity to go beyond glue code. To write a GUI text editor from "scratch". Overall, I'd say it worked out pretty here's screenshot of where things are right now. (We'll explain what you see in a bit.)

![editor](/Users/jimmyhmiller/Downloads/editor.png)

The question of course is, what does "scratch". In my case I decided that "scratch" was going to defined as using sdl2. So I had a fairly high level way of dealing with things like drawing to a screen and rendering fonts, but nothing text editor specific. This choice I think was actually pretty good for me to actually get stuff going. I have no experience with graphics apis and had I started there, I might have just stayed there. 

From the other angle sdl2 was also good in that it didn't do too many things for me. There were no text edit widgets I was trying to control, no framework, just a library for windowing, drawing, font rendering, and eventing. I really did have to understand what I was making, consider how to make my app work well, 

## Background and Goals

Before we talk about general path I took, let's talk about my general goals. Intially my goal was to just build an editor. I wanted color syntax, I wanted general editing capabilities, that was about it. As time grew on, my goals became less and less clear. What I discovered as I kept developing is that building a text editor was a bit more finicky than I had expected, but also way easier than I ever imagined. 

Rust is a really fast language. In my career, I've mostly worked in slow langauges. Most worked on problems where speed was important, but not the most important. For this project I wanted performance. I use emacs to do my clojure development, and to be honest, it's terrible. I like paredit, I like my cider setup, but emacs freezes constantly. When I try to print data out in the repl, I can completely lock the editor. That wasn't something I'd allow my editor to do, so I thought I'd have to be super clever to make that happen. Turns out, I don't.

### Influences

#### Casey Muratori

As I was looking to under take this project, Casey Muratori had put out his perhaps [video](https://www.youtube.com/watch?v=hxM8QmyZXtg) about building a fast terminal. He walks through how he made the rendering fast by default using a standard technique called a font atlas. So that's where I started. If you like me had never heard of a font atlas, it is a simple idea. Draw a picture of all the characters you need and as you are rendering, you just reference those characters. No need to render the font each time. 

In my case, I focused on ascii and just made a bit text of all those characters. It was blazingly fast. Turning off vsync, I had 1000s of frames per second. I played around with manually adding some color and find the performance held. Good sign, now I had two things to figure out. 1) What data structure to use for editing. 2) How to do color syntax? Luckily for the first there was some prior art that lead me down an unexpected path.

#### Jamie Brandon

After I started my project, I found out that Jamie had implemented a very similar idea but in zig. Because it was so similar though (used sdl, a font atlas, etc) I didn't want to look too closely at it and spoil in fun. But what I did do was read his blog posts on it and the one on [text handling](https://www.scattered-thoughts.net/writing/focus-text/) was particularly interesting. In the article, he says he just kept the text as a string and it was fast enough. Given my simplification of only handling ascii, I decided to try a `Vec<u8>` and it turns out, that was incredibly fast. 

Modern computers are way faster than I think many of us realize. I was able to implement text editing operations just by using the Vec. When I insert text in the middle of a line, I insert into the Vec, shifting the whole thing down. And yet, every file feels smooth. Didn't need to use a rope or any fancy data structure like that. Of course those have other benefits, but in this case, I could keep focusing on my text editor.

### Color Syntax

At this point I had edit and displaying but no color syntax. To figure out line breaks, I parsed the and made a very inefficient line array of tuples with `(start, end)`. I think this is one of those choices I wish I had done differently, mostly just on how I wrote the code, but it worked. One thing it let me do was only render the lines that were visible, so my first instinct for color syntax was to take advantage of that. I knew that editors like sublime use a regex based solution that only looks a line at a time. So I thought, maybe I should just use that approach and take something off the shelf.

### Failed Attempts

#### Syntect

I first looked at using [Syntect](https://github.com/trishume/syntect) for my highlighting. It works with sublime syntax files, so I'd be able to support many languages out of the box. It was incredibly easy to add to my project and well documented. I was able to integrate it very quickly and very quickly learn it was much too slow for what I wanted to do. 

Now this isn't to fault Syntect. The sublime syntax format is based on textmate's format, which rely heavily on regex. Given the complexity and constraints of these formats, there is only so much you can do. But it wasn't cutting it for me. You see at this point in the project, I wanted to keep things as simple as possible. Syntect though would often completely miss the frame deadline. If I was in the middle of editing a file and needed to resyntax highlight it, it would kill the performance. So I had to look elsewhere. 

#### Tree Sitter

[Tree Sitter](https://tree-sitter.github.io/tree-sitter/) is a very interesting project out of github. It does incremental parsing with really robust error handling of many lanugages. So, if I added tree-sitter to my editor, I should get syntax highlighting for cheap, but also, not have a performance issue as I'm editting. Or so I thought.

First getting tree sitter setup was far from straight forward. The packaging and build situation for tree-sitter was a bit weird. But once I got that going, I was actually quite sad to find out that the highlighting portion of tree-sitter [was not incremental at all](https://github.com/tree-sitter/tree-sitter/issues/1540). I looked for a little while at making my own custom integration and I knew it was possible, but also didn't sound like fun. So I took a different path building my own custom setup.

### Tokenizing Every Frame

I started with the simpliest possible thing I could do, make a custom tokenizer and tokenize on every single frame. So I did that, wrote a really terrible, representitive tokenizer, and revamped my rendering to use it. Turns out, that was actually really fast and really easy! Even with doing the incredibly naive thing of parsing every single frame, I was able to have the largest source files I could find on my machine open and editable in 60fps. Honestly, I was pretty blown away with how well that setup worked. Admittedly, my tokenizer is not very good now right. But as they say, that's just a matter of programming. I know that I can keep and even improve the performance while making it more feature rich.

## Moving Beyond the Traditional Editor

At this point I had the basics and wanted to play. First quesiton, since I'm a big fan of [Muse](https://museapp.com/), what if my text editor was a canvas. Implementing that was very straight forward, if a bit finicky, and moved me directly into the more intersting things I  now wanted to do. 

### Token Pane

![Screen Shot 2022-04-28 at 6.39.23 PM](/Users/jimmyhmiller/Desktop/Screen Shot 2022-04-28 at 6.39.23 PM.png)

As I was working on my tozenizer, I wanted to be able to see the output of the tokens right in the app. So I created what I called the token pane. If there is a pane called `token_pane` Its contents are defined as the raw tokens of the active pane. So now I could see exactly what things were tokenizing into. Incredibly useful for debugging.

### Action Pane

![Screen Shot 2022-04-28 at 6.41.52 PM](/Users/jimmyhmiller/Desktop/Screen Shot 2022-04-28 at 6.41.52 PM.png)Next was the action pane. Quite a bit trickier. Here I would display every action that happened in the app. But, what about scrolling the action pane? Well, if I did that, then as I scrolled the action pane would constantly get new actions. The whole thing was a bit of a mess. In fact, in general I found I really needed to filter some actions out. 

The other hard part of this setup was that I original didn't have a great way to refer to panes. My actions would be something like "MoveActivePane". But what was the active pane, or more percisely, when? Well, if I was looking at the action_pane, it was the active pane, so now as I'm filtering out action_pane actions, I would filter out all active pane actions! not what I wanted. So I had to setup a system where you actions resolve to ids. 

### Draw Panes

Ultimately what I want out of an editor more than anything was capture in this early blog post on [LightTable.](http://lighttable.com/2012/05/21/the-future-is-specific/). In it, they imagine an editor configurable in itself. But I wanted a different flavor. What if you could extend the editor in any language? I already had seen ways of taking the internals of the editor and using panes as output. What if I could do the opposite, use panes as input?  I later discovered a nice name for this, [afterburner rendering ala Mary Rose Cook](https://maryrosecook.notion.site/Afterburner-To-Dos-129f967a4d1343f390b78a56fc0fc7a0).  Here's an example. What I you can see here is some javascript that prints output like `rect 4 10 100 100`, that output is then parsed by the editor and drawn as rectangles to the screen.

![Screen Shot 2022-04-28 at 6.51.57 PM](/Users/jimmyhmiller/Desktop/Screen Shot 2022-04-28 at 6.51.57 PM.png)

### Text Space Drawing

Obviously rectangles aren't the most useful thing to draw. But I also played with rendering in text space. For example, here is a quick proof of concept of underlining text mentioned in rust compiler output. On the right you see some unused imports. On the left, a quick bash script for drawing that to the screen.

![Screen Shot 2022-04-28 at 6.58.15 PM](/Users/jimmyhmiller/Desktop/Screen Shot 2022-04-28 at 6.58.15 PM.png)

The dream is that as things evolved, your editor could gain new powers simply by code you have running in panes. No need for an extension language. Simply output things to stdout and you can control and extend the editor. What I found with this experiment is that even with the most naive, unoptimized code doing things that way was entirely possible.

### Accessing Pane Content

One fun experiment I played with was a way for any language to get the contents of a pane. Obviously, if a pane is backed by a file, you can read that file. But that wasn't good enough to me. I want you to be able to access the contents before changes have been saved. Further, you should be able to access it with just standard tools. So, I exposed the editor as an http service.

![Screen Shot 2022-04-29 at 11.10.40 AM](/Users/jimmyhmiller/Desktop/Screen Shot 2022-04-29 at 11.10.40 AM.png)

Honestly, as weird as it may seem, it was pretty easy to do, not expensive computationally and made it really easy to access the data. Ultimately, I'd love to even expose more things. Like being able to control the editor via http requests. Now, external programs can interact with the editor in a way I've never seen supported. Once we have that, it means we have the full unix power accessible in our editor in a very first class way. 

## What Went Well

### Rust

Rust was a wonderful language to write this in. I've gotten passed the learning curve where I basically can write Rust without thinking too much about it. With all of these features I basically was just thinking about the problems and not the language. While of course Rust's ownership model pushed me in one direction, it never felt like it limited me. 

One thing I really have grown to enjoy are rusts explicit clones. Without GC, clones can be expensive in a tight loop. Making them explicit let me be able to easily spot my bottlenecks. A number of times I could do a quick clone to get things working and then come back and move the code around to make the ownership clear and avoid the clone.

### SDL

SDL2 was really easy to get going on this project. From day one I had things drawing on the screen. The primitives sdl provides where just what I needed to focus on my task and not worry about the details.

### My Willingness to Let the Code be Messy

I often get to a point in projects where things just stall out because I want to clean up my code and I don't know exactly how I want things to be. I know that the path I'm on is not the right one and instead of forging ahead with working code, I stop and consider, then lose interest and stop working on it. I didn't do that here. The code base is a mess. There is a lot of duplicated code. And as it stands right now, things brokens from unfinished experiments. But, I got a lot done with this spare time project. And even with all the duplicated and poorly designed code, I only have 3442 lines of code.



