# Scythe Robotics Interview Assignment

## How does your system work? (if not addressed in comments in source)

My program uses tokio to spin up a task for each connection. That task then listens for messages, parses them, and responds accordingly. I preprocess the file before allowing connections to find all the byte ranges for lines. I then use syncfile to let me access those byte ranges concurrently.

## How will your system perform as the number of requests per second increases?

It seems to do fairly well in the benchmarks I quickly ran. If I wanted to be robust, I could generate traffic and make a load tester, or find a nice existing one. I did try out `tcpkali` and was able to spin up a decent amount of traffic for the server. Ultimately, it will hit some sort of bottleneck depending on your machine and the number of threads/bandwidth it can handle. But the whole things is fairly minimal in its overhead.

## How will your system perform with a 1 GB file? a 100 GB file? a 1,000 GB file?

I could have saved more space in my newline parsing by just storing the start of the newline and then at runtime scanning for the next newline character. As it stands, I store 2 usizes per line. So you would need 16 bytes per line on a 64bit machine. I tried a 1 GB file. It took about 4 seconds for booting up on my m2 mac. That is way slower than the theoretical limit. Definitely not the fastest way to parse new lines. Retrieval was perfectly fine though. Of course, I didn't do any of the rigorously because this is an interview take home project. This is all way less than a business days worth of work. But at no point do we load the whole file into memory. So we aren't limited by ram size.

## What documentation, websites, papers, etc did you consult in doing this assignment?

I googled things like tokio tcp server, rust logging, concurrent file access rust. That was about it. 

## What third-party libraries or other tools does the system use?

You can see them all listed in the Cargo.toml file

## How long did you spend on this exercise?

2-3 hours, mostly interrupted time. I could, of course, make this better with more time. I could add tests. I could gold plate this whole thing. But I also don't believe in over engineering. I'm sure there is some subtle (or obvious) bug that I missed. But we all have to make trade offs on our time so for this exercise I tried to time box it to what I could get done with my free time today.


## Things to note

There are no tests. Maybe should add them. Mostly around parsing the new lines and making sure I handle error cases. I think a generative integration test would probably be the most useful.

The code isn't crazy clean or abstracted out in the absolutely best fashion. As I was writing it, I didn't have the server or connection structs, but the main function looked ugly. So I added them which did make the nesting of the main function way less gnarley. Still not sure they are needed. Or that they are setup the way I'd like them to be.

I haven't rigorously tested this or benchmarked it. I didn't add any caching or anything like that because I don't know anything about the access patterns. I'm not personally a big fan of the error messages, as I would have liked them to be descriptive, but I wanted to follow what the spec said.

Also, my machine is way more capable from the quoted machine, so I didn't try to get any exact performance numbers as that is very machine/environment dependent.

Finally, the requirement about the program being "observable" was for me met just by adding some logging. My logging isn't perfect and of course in a production environment you'd want better stats and stuff. But again, keeping this minimal and time boxing it.

If you think I missed something obvious or did something wrong, let me know. I did almost miss the \r\n requirement as I haven't had to deal with that since my window days.