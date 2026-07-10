# Scry demo - the interactive assistant (Phase 7)

A ~5-minute sit-down. The thesis: **we run an ordinary interactive program, and for free we get a
live viewer into it - and we can change its code from the viewer while it runs.** The app
(`examples/assistant.scry`) is a Claude-Code-like terminal assistant. It contains zero viewer or
server code; the runtime injects the eval server for any program (like NREPL).

## 0. Build and launch

```
coil build                       # writes ./scry
./scry run examples/assistant.scry
```

It prints `viewer: http://localhost:7357` (or the next free port). Open that URL in a browser and
put it side-by-side with the terminal. The terminal shows:

```
Scry assistant - a Claude-Code-like REPL. Type 'help', or 'research <topic>'.
you>
```

Leave it sitting at the prompt.

## 1. It's a real REPL - type at it

In the terminal, type each of these and press Enter:

```
you> hello
you> help
you> what is scry?
```

You get canned-but-causal replies (the `ScriptedModel` keys off `hello`, `?`, etc.). Point out:
this is a genuine `Console.readLine()` loop - the process is really blocked waiting for your input.

## 2. …but the viewer is live THE WHOLE TIME (STW works while it waits for input)

This is the first "aha". While the terminal sits at `you> ` (blocked in `readLine`), the viewer is
fully responsive. In the browser:

- Watch the **type index** (left rail): `Agent`, `Conversation`, `Message`, `ScriptedModel`,
  `Session`, `Orchestrator`, `SubAgentWorker`, `Tool` implementors. There is **1 Agent** right now.
- Click `Agent` → the one instance `assistant`. Open it. Note its `conversation` size.

In the viewer's **REPL dock** (backtick to toggle) paste, one at a time:

```
types()
Agent.instance(0).conversation.size()
Agent.instance(0).say("probe question?")
Agent.instance(0).conversation.size()
```

The last size is +2. **You just mutated the running program from the viewer while its main thread
sat waiting for keyboard input.** That only works because `readLine` is safepoint-cooperative - it
never parks in a blocking syscall, so a stop-the-world eval lands within ~20 ms even mid-prompt.

## 3. Sub-agents on real threads - watch them appear and work in the viewer

Back in the terminal:

```
you> research quantum computing
```

The orchestrator prints a "delegating…" line and spawns **two sub-agents on two real OS threads**
(researcher, summarizer). They print interleaved, color-prefixed per-agent lines as they work. In
the viewer, *while they run*:

- The `Agent` count jumps **1 → 3** (refresh/watch the left rail). Two new instances,
  `researcher` and `summarizer`, appear.
- Open `Message` → the count **climbs** poll-to-poll as each sub-agent appends to its conversation.
- Open a sub-agent's `Agent` detail → its `status` goes `Running` → `Done`; its `conversation`
  grows.

Crucially: **the prompt is still live while they work.** Type another line mid-research:

```
you> thanks
```

You get a reply immediately, interleaved with the sub-agents' output. The main loop never blocked
on the sub-agents.

## 4. THE two-way beat - redefine a method live and a new UI element pops up

The app has an intentional extension point: `Session.suggest(input)` ships returning `""` (nothing
printed after each response). We'll make an autocomplete/suggestions box appear **without
restarting**.

In the viewer, open any `Session` instance detail and hit **✎ edit source** (or use the REPL dock).
Paste this **verbatim** (it is the exact snippet in the `examples/assistant.scry` header):

```
class Session {
  history: List<String>
  fn init() { self.history = List<String>() }
  fn renderPrompt() -> String { "you> " }
  fn suggest(input: String) -> String {
    if self.history.len() == 0 { "" }
    else { "  [suggestions: help | research <topic> | exit  (last: " + self.history.get(self.history.len() - 1) + ")]" }
  }
}
```

Hit define. The viewer confirms `✓ Session redefined - now at generation 1`. Now, in the terminal:

```
you> anything at all
```

Under the reply, a suggestions box now appears:

```
  [suggestions: help | research <topic> | exit  (last: anything at all)]
```

**No restart. The running program grew a new UI element because we changed its code from the
outside.** Every existing call site to `suggest` reaches the new body (generations, not diffs).
Optional flourish: redefine again changing `renderPrompt`'s body to return `"assistant> "` - the
prompt itself changes on the next line.

(If you paste a deliberately-broken body - e.g. `self.history + 5` - the viewer rejects it as a
`TypeError` and the running program is untouched: a rejected edit is a strict no-op.)

## 5. Exit - and it's still browsable

```
you> exit
```

The orchestrator joins any outstanding sub-agents, prints `goodbye`, and `main` returns. The
runtime keeps the process alive:

```
main() finished - the heap is still live and browsable. Serving evals; press Ctrl-C to exit.
```

The viewer still works - every instance, every conversation, every message from the whole session
is still there to inspect. Ctrl-C in the terminal ends it.

---

### One-line pitch to land

"That was an ordinary command-line program. We never wrote a line of UI or server code in it - the
runtime gives every program a live, mutable window into itself, and we just reached in and changed
its behavior while it was running."
