# keep-running

A terminal session manager. Like `dtach`, but with names you can remember and a CLI that doesn't fight you.

Start something, walk away, come back, it's still running.

## Install

```sh
cargo install --path .
```

## Use

```sh
keep-running run -- npm run dev        # start a session
keep-running shell                     # start one with your shell
keep-running list                      # see what's running
keep-running fuzzy-penguin             # attach by name (prefix works too)
keep-running kill fuzzy-penguin
```

Inside an attached session:

```
Ctrl+a d        detach (leave it running)
Ctrl+a k        kill it
Ctrl+a Ctrl+a   send a literal Ctrl+a
```

That's it.
