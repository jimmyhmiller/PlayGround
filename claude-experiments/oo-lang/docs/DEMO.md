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

---

# Scry demo, Phase 8c - a REAL agent loop (watch/pause/hot-swap a live agent)

Phase 8c upgrades the assistant's brain from a canned string to a **model-driven tool-use loop**.
Offline it uses a deterministic `ScriptedModel`; with a key it uses a real `AnthropicModel` that
calls Claude and does live tool use. The loop is identical either way.

## 0. Offline first - the loop is real with zero network

```
./scry run examples/assistant.scry
brain: ScriptedModel - offline (set DEEPSEEK_API_KEY for the live model)
you> what is 17 times 23?
[agent] -> tool_use: calculate({"a":17,"b":23,"op":"mul"})
[agent] <- tool_result: calculate => 17 * 23 = 391
assistant Here you go: 17 * 23 = 391
you> weather in Tokyo?
[agent] -> tool_use: get_weather({"location":"Tokyo"})
[agent] <- tool_result: get_weather => Tokyo: 18C, cloudy
assistant Here you go: Tokyo: 18C, cloudy
```

The model **chose** the tool and its arguments; the tool ran for real (17*23=391 is arithmetic, not
a canned string); the result fed back and shaped the final answer. That is the whole loop.

## 1. With a key - a LIVE model actually calls the tool

Default target is **DeepSeek's Anthropic-compatible endpoint** (`deepseek-v4-pro` over
`https://api.deepseek.com/anthropic`) — same `/v1/messages` tool-use protocol, so the only thing
that changes is the key. Overridable: `ANTHROPIC_BASE_URL`, `SCRY_MODEL`, and the key resolves from
`DEEPSEEK_API_KEY` | `DEEPSEEK_KEY` | `ANTHROPIC_API_KEY`. (To target Anthropic instead:
`export ANTHROPIC_BASE_URL=https://api.anthropic.com SCRY_MODEL=claude-sonnet-5 ANTHROPIC_API_KEY=sk-ant-...`.)

```
export DEEPSEEK_API_KEY=sk-...
./scry run examples/assistant.scry
brain: AnthropicModel (deepseek-v4-pro @ https://api.deepseek.com/anthropic) - LIVE
you> what is 17 times 23?
[agent] -> tool_use: calculate({"a":17,"b":23,"op":"mul"})   # the live model emitted this tool_use
[agent] <- tool_result: calculate => 17 * 23 = 391            # our loop ran the tool + fed it back
assistant 17 × 23 = **391**.                                  # the model's end_turn answer
```

Same prompts, same loop - now Claude is the brain. Every request/response is a browsable
`HttpResponse` entity, and because the HTTP multi-loop is STW-cooperative, an agent parked mid-API-
call stays fully inspectable in the viewer.

## 2. Watch + pause the loop live (the payoff)

```
you> loop weather in Tokyo        # a repeating agent loop on a background OS thread
```

In the viewer (`http://localhost:PORT`): `Message.instances()` climbs as the loop turns. Find the
`looper` Agent and invoke `pause()` on it - the message count **freezes**; the loop visibly stalls.
`resume()` and it climbs again. You paused a running agent from the outside, with no restart.

## 3. Hot-swap a tool's behavior mid-loop

Paste a redefinition of a tool's `run` into the viewer's code panel WHILE the loop runs:

```
class WeatherTool implements Tool {
  source: String
  fn name() -> String { "get_weather" }
  fn description() -> String { "..." }
  fn inputSchema() -> String { "..." }
  fn run(argsJson: String) -> String { "PATCHED: it's always sunny" }
}
```

The very next tool call the running agent makes uses the new body - no restart. (A broken edit is
rejected as a `TypeError` and the running program is untouched, same strict no-op as Phase 6.)

The pitch: a real agent - model-driven tool use over a live API - that you can watch, pause, and
rewrite while it runs, because it is an ordinary Scry program and the runtime gives every program a
live, mutable window into itself.

---

# Scry demo, Phase 9 - see the code BEFORE it runs (the class graph)

The same viewer that shows a live heap can show a program's **static structure without running it** -
and it's literally the same graph, which just fills in once you run it.

## 0. Inspect a program that has never run

```
./scry inspect examples/assistant.scry
# viewer: http://localhost:7400
#   (inspect: schema only, program not running - press Ctrl-C to exit)
# (as of Phase 10 the fixed :7357 belongs to the PORTAL; run/inspect take an ephemeral :7400+)
```

`main()` never runs - no `you>` prompt, no agent output. It typechecks, builds the (empty) arenas,
and serves the schema. Open the URL: the landing view is the **class graph**.

## 1. Read the architecture at a glance

The graph is the whole program's shape, laid out and readable:

- **class** nodes (blue), the **object** `Json` (gold), **interface** nodes (dashed teal: `Tool`,
  `Model`, `Runnable`) and **enum** nodes (purple: `AgentStatus`, `JsonValue`).
- **solid edges** = a field references another type - `Agent -> Conversation`, `Agent -> Model`,
  `Agent -> Tool` (through `List<Tool>`), `Conversation -> Message`.
- **dashed edges** = `implements` - `ShellTool`/`SearchTool`/`CalcTool`/`WeatherTool -> Tool`,
  `ScriptedModel`/`AnthropicModel -> Model`.
- every node has a **live-count badge**: all `0`, because nothing is running yet.

Hover a node to light up its relationships. Click `Tool` (an interface) and a static card shows its
methods and its four implementors. Click `AgentStatus` and see its variants. This is the typechecker's
knowledge, browsable - no execution required.

## 2. Now run the SAME file and watch the nodes populate

```
# Ctrl-C the inspect, then:
./scry run examples/assistant.scry
```

Open the viewer - **same graph, same layout** (the positions are computed from the schema shape, so
nothing jumps). Type `research quantum computing` at the prompt and watch the **badges climb**: the
`Agent` node ticks 1 -> 3 as sub-agents spawn, `Message` climbs as they work. Now click the `Agent`
node: because it has live instances, it drills straight into the **instance table** you already know
from the earlier demos - rows, fields, click through to a detail, invoke a method.

One view, two states: the static schema you inspected and the live heap you're now steering are the
**same node-link graph**. That's the pitch - the class diagram isn't a separate artifact you keep in
sync by hand; it IS the running program, before and during the run.


---

# Scry demo, Phase 10 - the PORTAL: sit at one page, watch programs pop up (DECISIONS #13)

The UI model change the owner asked for: launch a hub, sit at ONE page, and every program you run or
inspect appears as a card you click into. No per-program URL juggling - one origin (`:7357`) that
reverse-proxies the eval channel to whichever program you picked.

## 0. Launch the portal and leave it open

```
./scry portal
# portal: http://localhost:7357  - sit here; run/inspect programs pop up as cards. Ctrl-C to exit.
```

Open **http://localhost:7357**. It's empty - "no programs yet" - with the two commands to try.

## 1. In OTHER terminals, start programs - watch cards POP UP

```
# terminal 2
./scry run examples/assistant.scry
# viewer: http://localhost:7400
# portal: http://localhost:7357     <- it found the portal and registered

# terminal 3
./scry inspect examples/agents.scry
# viewer: http://localhost:7401
# portal: http://localhost:7357
```

Watch the portal page (it polls every second): a **`assistant.scry` card** appears with a *running*
badge, then an **`agents.scry` card** with an *inspect* badge. Each card shows a live status dot, the
port, start time, and cheap live stats (instance + type counts, fetched through the proxy). New cards
animate in - it feels alive.

## 2. Click a card -> the inspector, proxied

Click the `assistant.scry` card. You land in the **class graph** (the Phase 9 default view) for THAT
program - nodes, edges, live badges - except every eval now flows through `/p/<id>/eval` and the
portal routes it to that process. Everything you learned in the earlier demos works verbatim:
drill a live node into its instance table, open a detail, invoke a method, pop the REPL dock. Hit
**`← portal`** (top-left) to go back to the grid and click into the `agents.scry` inspector instead
(all-zero badges - it's inspect-only, never ran main()).

## 3. It still works standalone - the portal is ADDITIVE, never required

Kill the portal (Ctrl-C in terminal 1). The programs keep running and **each still serves its own
viewer** on its printed `:7400`/`:7401` URL - open either directly and it's the exact same inspector.
Registration is best-effort: if no portal is up when you `scry run`, it simply prints its direct URL
and nothing else. The portal adds a hub; it never becomes a dependency.

Back on the portal page, kill one program (Ctrl-C its terminal): within a second its card **greys out**
("exited") - the portal health-probes each program's port on every poll, so a dead program visibly
drops out of the live set.

### One-line pitch to land

> Launch the portal, sit at one page, and run programs in other terminals - they pop up as cards you
> click into and get the full live inspector, all through one origin. Kill the portal and every
> program still stands on its own. The hub is a lens, not a leash.
