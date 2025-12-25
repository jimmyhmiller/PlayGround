# AI Has Made it Easy to Own Your Tools

I am a digital hoarder. Probably not an extreme one. I don't have some fancy complicated raid system for endless terabytes of media. But when I find a link or a pdf to some interesting article, I want to save it. But I almost never review it. That isn't to say I don't read articles. I read plenty. But I also collect them. But I've always had a problem, how do I organize them, classify them, keep them.

For links my answer was pocket. Well, you can imagine how that went. My exported list is living in raindrop for the time being. But I've never been happy with that. My PDFs had been living in [Muse](https://museapp.com/). A nice piece of software, but it has diverged from the minimalism I enjoyed about it. So I've been on search for alternatives. But there has always been this nagging feeling that what I want is not just some different piece of software, but a custom, fully owned setup for myself. But where would I ever find the time to do that.

## Some AI tooling

I admit that I'm far from having the tools I want. The tools I've built so far are rudimentary. But they were built in minutes, they were exactly what I asked for. And they did things that a few years ago would not have been possible.

### Tool 1 - Local LLM Sorting

Local LLMs are getting better and better. Seeing that trend, I bought a [framework desktop](https://frame.work/desktop). Using it I was able to have claude write a [quite simple script](https://github.com/jimmyhmiller/PlayGround/blob/0c4c04287c0c30e2c9a5f48782ecfd39f8618d09/claude-experiments/reading-tools/scripts/find_programming_pdfs_local_llm.sh) that found every pdf on my machine, grabbed some of the initial text from them and passed the to gpt-120b and asked, is this pdf about programming? Now I could find all those programming pdfs. But I needed to sort them.

### Tool 2 - A Swift Application

Now that I had the initial collection of potential pdfs. How was I going to sort them. I didn't want a tagging system. There's a chance that later I will. But for now I wanted descrete categories. But what categories? I'd only find out once I started organizing them. So I asked claude and got an app specifically designed to let me categorize them.

![Screenshot 2025-11-12 at 2.23.14 PM](/Users/jimmyhmiller/Desktop/Screenshot 2025-11-12 at 2.23.14 PM.png)

### Tool 3 - PDF sync

A simple tool for syncing pdfs by hash to s3.

### Tool 4 - PDF indexer

Some of the pdfs have nice metadata built-in. So we can just go ahead an extract that. That's why this tool.

### Tool 5 - LLM OCR metadata extractor

For the rest, we pass to Qwen3-VL-30B to grab the title and author.

### Tool 6

A swift application compiled for my mac and ipad that let's me annotate pdfs.

![Screenshot 2025-12-24 at 11.15.36 PM](/Users/jimmyhmiller/Library/Application Support/typora-user-images/Screenshot 2025-12-24 at 11.15.36 PM.png)

The app is far from fully featured yet. But the fact that it syncs between my mac and ipad seamlessly is wonderful. I use this mainly for the podcast, so I haven't gotten to do a ton with it yet. But having it be something I can customize to my needs already has me excited.
