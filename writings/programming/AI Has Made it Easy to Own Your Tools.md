# AI Has Made it Easy to Own Your Tools

I am a digital hoarder. Probably not an extreme one. I don't have some fancy complicated raid system for endless terabytes of media. But when I find a link or a pdf to some interesting article, I want to save it. But I almost never review it. That isn't to say I don't read articles. I read plenty. But I also collect them. But I've always had a problem: how do I organize them, classify them, keep them?

For links, my answer was pocket. Well, you can imagine how that went. My exported list is living in Raindrop for the time being. My PDFs had been living in [Muse](https://museapp.com/). A nice piece of software, but it has diverged from the minimalism I enjoyed about it. So I've been on the search for alternatives. But there has always been this nagging feeling that what I want is not just some different piece of software, but a custom, fully owned setup for myself. But where would I ever find the time to do that?

## Some AI tooling

I admit that I'm far from having the tools I want. The tools I've built so far are rudimentary. But they were built in minutes, and they were exactly what I asked for. And they did things that a few years ago would not have been possible.

### Tool 1 - Local LLM Sorting

Local LLMs are getting better and better. Seeing that trend, I bought a [framework desktop](https://frame.work/desktop). Using it, I was able to have Claude write a [quite simple script](https://github.com/jimmyhmiller/PlayGround/blob/0c4c04287c0c30e2c9a5f48782ecfd39f8618d09/claude-experiments/reading-tools/scripts/find_programming_pdfs_local_llm.sh) that found every pdf on my machine, grabbed some of the initial text from them, and passed them to gpt-120b and asked, is this pdf about programming? Now I can find all those programming PDFs. But I needed to sort them.

### Tool 2 - A Swift Application

Now that I have the initial collection of potential PDFs. How was I going to sort them? I didn't want a tagging system. There's a chance that later I will. But for now, I wanted discrete categories. But what categories? I'd only find out once I started organizing them. So I asked Claude and got an app specifically designed to let me categorize them.

![Screenshot 2025-11-12 at 2.23.14 PM](/Users/jimmyhmiller/Desktop/Screenshot 2025-11-12 at 2.23.14 PM.png)

### Tool 3 - PDF sync

A simple tool for syncing PDFs by hash to S3.

### Tool 4 - PDF indexer

Some of the PDFs have nice metadata built in. So we can just go ahead and extract that. That's why this tool.

### Tool 5 - LLM OCR metadata extractor

For the rest, we pass to Qwen3-VL-30B to grab the title and author.

### Tool 6

A swift application compiled for my Mac and iPad that lets me annotate PDFs.

![Screenshot 2025-12-24 at 11.15.36 PM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2025-12-24 at 11.15.36 PM.png)

The app is far from fully featured yet. But the fact that it syncs between my Mac and iPad seamlessly is wonderful. I use this mainly for the podcast, so I haven't gotten to do a ton with it yet. But having it be something I can customize to my needs already has me excited.

### Tool 7

A page on my website that statically [generates the archive](/readings) for all to browse.

## The Unsung Hero

We've yet to reach the point where local models can quite replace Claude for coding. But having a local model where I never had to worry about the cost to send it experiments, one where it ran not on my laptop, but in the background on a "server", was such an enabling feature. I would have been very hesitant to send all these PDFs to a model. But with a local model, I had so much flexibility to use it wherever judgment could be a substitute for deterministic processing.

## Conclusion

Nothing here was groundbreaking. Nothing here is something I couldn't have made myself. But they are all things I put off. All things I would have never prioritized, but wanted made. They are all imperfect tools. Many of them are one-offs. There were actually countless other small tools made along the way. Cleaning up titles, a tool that chose between the metadata title and the OCR ones (ocr were usually better). Any one of these little bottlenecks might have been enough for me to stop working on this project.

I see lots of discussions about AI all having to do with "production code". I'm sure I'll write my thoughts about that at some point. But I also think it is important that we remember that this isn't the only code that exists. In this case, [it's personal code](https://www.robinsloan.com/notes/home-cooked-app/). Code, I enjoy having the ability to modify. Code that I am happy isn't robust; it doesn't try to handle every edgecase. It doesn't need a complicated sync process. Doesn't need granular permissions. 

This is just the start of what I expect to be a continual evolution of my pdf (and eventually link) management software. But for the first time in my programming life (not career, not everything is a business transaction), I don't feel the weight of maintenance I've created for myself. I feel a sense of freedom to build more without incurring such a heavy cost. This, to me, is one of the most exciting features of our new AI capabilities.
