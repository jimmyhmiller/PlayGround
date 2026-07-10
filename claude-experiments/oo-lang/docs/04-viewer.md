# 04 — Viewer: UX and Wire Protocol

The viewer is not a debugger UI bolted onto the runtime. It is the runtime's second
interface — stdin/stdout is the first, the browser is the second — and both read the same
live objects. This doc specifies the screens, the update model that keeps them honest, the
invoke/eval flow, and the JSON wire protocol that makes all of it work. Examples throughout
use the agent-TUI demo's types (`Agent`, `Conversation`, `Message`, `Tool`, `ToolCall`,
`Task`) from `00-vision.md`.

## 1. Information architecture

Four screens, one drill path, always reachable by URL (the viewer is a normal web app —
back button works, a detail view is bookmarkable as `/agents/Agent%237`):

```
Entity-type index  →  Instance table  →  Instance detail  →  (methods, references)
        ▲                                        │
        └──────────── REPL/eval dock (always docked, not a 5th screen) ──┘
```

**Entity-type index** (landing screen). One row per class in the schema, live count, and a
small trend indicator (climbing / falling / flat, computed client-side from the last few
count deltas — not a stored history). A search box filters the type list by name as you
type; this is a client-side filter over the schema, not a server round-trip.

```
🔍 search types…

Agent           3 live
Conversation    3 live
Message        47 live   ▲
Tool             5 live
ToolCall        12 live   ▲
Task             8 live
```

**Instance table** (click a type). Paged, sortable, filterable table. Columns are the
type's fields in declaration order, plus an `id` column pinned left. The filter bar takes a
predicate over fields (`status == Paused`, `name contains "research"`, `messages.length >
20`) — a small expression subset of the language itself (field access, comparisons,
`and`/`or`, method calls with zero args), evaluated server-side against every instance in
the arena. Reference-typed columns (`conversation: Conversation`) render as clickable links
showing the target's id and a one-line summary (`Conversation#2 (14 messages)`), not a raw
pointer.

**Instance detail** (click a row, or follow a reference link). Three sections:

- *Fields* — name, static type, current value. Reference fields are links that navigate
  (pushing a breadcrumb: `Agent#7 › conversation › Conversation#2`); collection fields
  (`List<Tool>`) render as an expandable inline list of link chips, not a sub-table, unless
  the list has more than ~20 elements, in which case it's a link to an instance-table view
  pre-filtered to that collection.
- *Methods* — every method with its signature (`fn setPriority(p: Int) -> Void`). Clicking
  one expands an inline invocation form in place (§3) — never a modal, so you can see the
  instance's current field values while filling in arguments.
- *Recently changed* — the fields that changed in roughly the last 5 seconds, listed with
  their previous value struck through, so a viewer who glances over mid-demo sees not just
  current state but "what just happened."

**REPL/eval dock**. A persistent, collapsible bottom drawer (toggled with a keyboard
shortcut, ` — the "console key" convention from Quake/Source-engine games and browser
devtools), not a separate page. It carries an implicit `self` binding equal to whatever
instance is currently open in the detail view (empty at the type-index/table screens), so
`self.conversation.messages.length` just works when you're looking at an `Agent`. Output is
a scrollback of `expr → value` pairs, each value rendered with the same typed-value
renderer used for fields (so a returned `Task` reference is a clickable link, not a string).

Global search (top bar, always visible) jumps directly to an instance by id
(`Agent#7`) or does a cross-type substring search over field values, returning a flat list
of matches grouped by type. This is the "I don't know which screen it's on" escape hatch.

## 2. Live-update model

The instruction to not stream every allocation is a hard constraint, not a suggestion —
`ToolCall#4812` being born and dying inside one agent turn should not be a network event
unless something is actually looking at `ToolCall`s right now. The model:

**Nothing is pushed unless something is subscribed.** The wire protocol has exactly three
subscribable views:

1. `type-index` — count (and only count) per type. One subscription serves the whole
   landing screen.
2. `instance-list` — a specific `(type, filter, page)` triple. Subscribing to the `Agent`
   table with no filter is a different subscription than subscribing to it filtered to
   `status == Paused`; the server evaluates the filter, not the client.
3. `instance-detail` — a single instance id.

There is deliberately no "subscribe to everything" or "subscribe to all mutations of type
X" — that's the firehose the brief warns against, and per-type arenas make it *possible*,
which is exactly why it must be refused at the protocol level, not just left slow.

**Coalesced dirty-tracking, not per-write pushes.** Every arena slot carries a generation
counter and a per-field dirty bitmask, set on write (cheap — the interpreter already knows
the field offset for the `setfield` opcode). Between publish ticks — a fixed cadence,
proposed at 66ms (~15Hz, fast enough to read as "live," slow enough that a tight loop
mutating a field 10,000 times a second still produces one push) — the server walks the
dirty set once per active subscription, diffs against what that subscription last saw, and
emits exactly one delta message per changed instance per tick. Ticks with no dirty
instances for a given subscription send nothing. This bounds outbound traffic to O(active
subscriptions × changed instances actually in view), independent of total mutation rate.

**Snapshot + delta with sequence numbers, not poll-on-focus.** Each subscription gets a
monotonic per-subscription sequence: `subscribed` returns a full snapshot at `seq = N`;
every subsequent push for that `subId` carries `seq = N+1, N+2, …`. The client applies
deltas in order and asserts contiguity. If a message is dropped (reconnect, slow consumer,
server restart) the client notices a seq gap and requests a fresh snapshot rather than
trying to patch a hole. This is chosen over poll-on-focus deliberately: the demo's second
beat ("Message and ToolCall counts visibly climbing") and third beat ("the terminal updates
the instant I invoke pause()") both require the screen to move *without the viewer being
interacted with* — a poll-on-focus model would show a stale count until the user jiggles
the mouse, which kills exactly the moment the demo is built around. The cost (a resync
round-trip on the rare dropped-message case) is worth paying.

**One connection, many subscriptions.** A single WebSocket per browser tab multiplexes all
subscriptions by `subId`; opening a new detail view is one `subscribe` frame, not a new
socket. Closing a tab lets the server garbage-collect subscriptions on disconnect (no
explicit unsubscribe required, though the client sends one on navigation-away so a
long-lived tab with many past views doesn't accumulate server-side subscriptions).

**What actually fires the tick, given 02-runtime.md's queue-drained-at-safepoint model.**
02-runtime.md §7 describes the interpreter as reactive: the only described trigger for
servicing anything is the dispatch loop's `safepoint-poll` noticing a *pending queued
request*. A wall-clock 66ms cadence needs a concrete way to become a queued request rather
than a spontaneous wake-up inside the interpreter (there is no second thread that's safe to
touch arenas — that's the whole point of §7's single-writer discipline). The mechanism:
the server thread's own timer enqueues a synthetic `tick` entry onto the same lock-free
completion queue viewer requests and async-I/O completions already use, once per 66ms; the
interpreter only ever learns about it at its own next `safepoint-poll`, exactly like any
other queued item. **Tradeoff, stated plainly, not hidden:** this makes the 66ms figure a
*floor*, not a guarantee — an interpreter parked in a blocking foreign call (02-runtime.md
§7's "the genuinely hard part") or simply not hitting a safepoint for a stretch will miss
one or more ticks, and the next publish lands late, coalescing whatever piled up. This is
the same failure window that already stalls `invoke`/`eval` for the same reason, so a
viewer watching counts freeze during that window is reading a real, already-acknowledged
interpreter-availability gap, not a new bug introduced by the tick model.

**Terminology note vs. 02-runtime.md §8.** That doc's introspection table says `types`
"pushes a delta whenever any `arena-alloc`/`arena-free` fires for a type the client is
watching." Read literally that's a per-mutation network send, which is exactly the firehose
this section spends its whole argument refusing. It should be read as describing the
*dirty-marking* step only — the single branch on the "any viewer attached" flag plus the
`watched` bit (02-runtime.md §8, same paragraph) that happens synchronously inside the
store/alloc/free path — not as a claim that a message crosses the wire at that moment. The
actual transmission is exactly the coalesced, tick-gated send described above: dirty is set
on write, sent on the next tick, at most once per instance per tick regardless of write
volume in between. This doc is authoritative for what actually crosses the wire and when;
02-runtime.md §8's phrasing should be understood as shorthand for "marks dirty for the next
tick to pick up," not "pushes immediately."

This ties to the concurrency model of the interpreted program, which is **OPEN** (see
`DECISIONS.md`): dirty-tracking and publish ticks are cheap under a single-threaded
cooperative-turn model, and need re-checking (likely: per-arena or per-object locking
around the dirty bitmask) if agent turns end up running on real OS threads. Flagged here,
not resolved here.

## 3. Method invocation

Invoking is inline, not modal, and typed:

- **Zero-arg methods** (`resume()`, `pause()`) show a single "Invoke" button, no form.
- **Typed scalar params** (`Int`, `String`, `Bool`, `Float`) get a matching input widget
  (number field, text field, checkbox) with the static type shown as a placeholder hint.
- **Enum params** get a dropdown of the enum's variants — the schema message carries enum
  variants (§4) precisely so the viewer never has to guess valid values.
- **Reference-typed params** (e.g. `assignTool(t: Tool)`) get a type-ahead picker: typing
  searches live instances of that type by id or field match (same query the instance table
  uses), and picking one fills the argument with that instance's id.
- **Anything the form-builder doesn't have a widget for** (a `List<T>` argument, a
  user-defined struct literal) falls back to a "raw" toggle on that argument: a single text
  field taking a JSON-ish literal, parsed and typechecked server-side before the call runs.
  This is the pressure valve — it must exist so the form-builder never blocks a valid call,
  but it's opt-in per-argument, not the default experience.

**Results.** A successful invoke shows the return value inline, right below the method,
using the same typed-value renderer as fields (so a returned `Task` is a link), tagged with
a timestamp and a fading highlight. An error shows as a red inline panel directly under the
form (never a toast, never a separate log) with the error kind, message, and a bytecode
call-stack trace rendered as `Type.method (line N)` frames — source-level, not raw Coil
frames, because the whole point of this system is that the person looking at it may not be
the one who wrote the interpreter.

**stdout.** The viewer never captures or redirects the target process's stdout/stderr —
per the non-negotiable "ordinary process" decision, the program's stdout belongs to the
terminal it was launched from, full stop. Invoking `pause()` from the viewer does not print
anything to the browser; what the viewer shows is the *structured* result of the call (the
return value, or the field deltas it caused, arriving a tick later over the
`instance-detail`/`instance-list` subscriptions already open). If the invoked method
happens to call something that writes to stdout, that text appears in the terminal the
program is running in, same as if the program had triggered it itself — because to the
runtime, it did. The demo's "TUI reflects it instantly" moment is the TUI's own render loop
reacting to the mutated `AgentStatus` field, not the viewer piping anything into it.

## 4. Wire protocol

Transport: one WebSocket per client at `ws://localhost:<port>/ws`, alongside a plain HTTP
server on the same port serving the viewer's static assets and one REST-ish endpoint
(`GET /snapshot`) used only for the very first paint before the socket is up. Every
protocol message is a single JSON object with an `op` field. Request/response pairs
(anything the client asks for) carry a client-chosen `reqId` echoed back on the response so
concurrent in-flight requests can be matched; push messages carry a `subId` and `seq`
instead.

### Relationship to 05-milestones.md's build order

Everything below is this doc's answer to "what should the wire protocol look like once the
observability product is finished" — `subscribe`/`subscribed`, per-subscription `seq`,
dirty-bitmask-driven pushes, `instance-updated`/`instance-created`/`instance-collected`/
`count-updated`. It is **not** a claim that M1–M3 in `05-milestones.md` build this verbatim,
and the two documents must not be read as describing the same wire traffic at the same
point in the build. `05-milestones.md` ships a deliberately smaller, named subset first,
polling instead of subscribing, and is explicit that the subscribe/dirty-tracking layer
below may be cut entirely for the PoC ("revisit only if polling proves visibly laggy in
rehearsal" — M1's OUT list). The mapping:

| PoC op (`05-milestones.md`) | Nearest equivalent below | What's missing/simplified in the PoC cut |
|---|---|---|
| `{"op":"types"}` → `{"op":"types","rows":[...]}` (M1, client polls every 250ms) | `type-index` subscription's snapshot content (§4 `subscribe`/`subscribed`) | request/response instead of subscribe; no `subId`/`seq`; freshness is a poll interval, not a tick-driven push |
| `{"op":"instances","type":...,"query":...}` (M1, polled) | `list-instances` request/response | same request/response shape survives almost as-is; M1's query grammar is narrower (four predicate forms, no `and`/`or`) than §1's filter-bar description |
| `{"op":"instance","id":...}` (M1, polled) | `instance-detail` one-shot fetch | near-identical shape (`id` vs `target` naming aside); no `instance-detail` *subscription* variant, no live push — the client just re-polls |
| `{"op":"invoke",...}` → `{"op":"result",...}` + `{"op":"changed",...}` (M2) | `invoke`/`invoke-result` + `instance-updated` push | M2's `changed` is a single flat diff scoped to "the instance just invoked on" (no dirty-tracking system, no `subId`/`seq`), not a subscribed push |
| `{"op":"eval",...}` (M2) | `eval`/`eval-result` | same idea, deliberately narrower — "call a method or read a field on a known id," not an arbitrary expression subset |

None of M1–M3's IN lists introduce `subscribe`, `subId`, `seq`, or per-field dirty
bitmasks — so, per the "if it isn't in some milestone's IN list, it does not exist for the
PoC" rule in `05-milestones.md`, the full protocol below should be read as the durable
design this doc commits to for the product, with the left column of the table above as the
concrete, possibly-final, wire surface the PoC actually ships. If a later milestone (or
post-PoC work) does upgrade to subscriptions, the upgrade path is additive, not a breaking
rename: the PoC's `reqId`-keyed request/response ops keep working unchanged, and
`subscribe`/`subscribed`/`seq`-carrying pushes are layered in as new op types alongside
them, exactly as `invoke`/`eval` already coexist with `subscribe` in the message catalogue
below.

### `hello` (server → client, unsolicited, first message on connect)

```json
{
  "op": "hello",
  "protocolVersion": 1,
  "runtime": { "pid": 51234, "program": "agents.oo", "started": "2026-07-09T14:02:11Z" },
  "schemaVersion": 3,
  "enums": [
    { "name": "AgentStatus", "variants": ["Running", "Paused", "Waiting", "Done"] }
  ],
  "types": [
    {
      "name": "Agent",
      "fields": [
        { "name": "name", "type": "String" },
        { "name": "model", "type": "String" },
        { "name": "status", "type": "AgentStatus" },
        { "name": "conversation", "type": "ref:Conversation" },
        { "name": "tools", "type": "list:ref:Tool" }
      ],
      "methods": [
        { "name": "pause", "params": [], "returns": "Void" },
        { "name": "resume", "params": [], "returns": "Void" },
        { "name": "currentTask", "params": [], "returns": "ref:Task" }
      ],
      "liveCount": 3
    },
    {
      "name": "Conversation",
      "fields": [
        { "name": "task", "type": "ref:Task" },
        { "name": "messages", "type": "list:ref:Message" }
      ],
      "methods": [
        { "name": "append", "params": [{ "name": "m", "type": "ref:Message" }], "returns": "Void" },
        { "name": "tokenEstimate", "params": [], "returns": "Int" }
      ],
      "liveCount": 3
    }
  ]
}
```

### `list-instances` (client → server request, paged + filtered)

```json
{
  "op": "list-instances",
  "reqId": "r1",
  "type": "Agent",
  "filter": "status == Paused",
  "sort": { "field": "name", "dir": "asc" },
  "page": { "offset": 0, "limit": 50 }
}
```

```json
{
  "op": "list-instances",
  "reqId": "r1",
  "type": "Agent",
  "total": 1,
  "page": { "offset": 0, "limit": 50 },
  "rows": [
    {
      "id": "Agent#7",
      "fields": {
        "name": "researcher",
        "model": "fable-5",
        "status": "Paused",
        "conversation": { "ref": "Conversation#2", "summary": "Conversation#2 (14 messages)" },
        "tools": { "count": 2 }
      }
    }
  ]
}
```

### `subscribe` / `subscribed` / `unsubscribe`

```json
{ "op": "subscribe", "subId": "s1", "view": "type-index" }
{ "op": "subscribe", "subId": "s2", "view": "instance-list",
  "type": "Agent", "filter": "status == Paused", "page": { "offset": 0, "limit": 50 } }
{ "op": "subscribe", "subId": "s3", "view": "instance-detail", "target": "Agent#7" }
```

```json
{
  "op": "subscribed",
  "subId": "s2",
  "seq": 1042,
  "snapshot": {
    "type": "Agent", "total": 1, "page": { "offset": 0, "limit": 50 },
    "rows": [ { "id": "Agent#7", "fields": { "name": "researcher", "status": "Paused" } } ]
  }
}
```

```json
{ "op": "unsubscribe", "subId": "s2" }
```

### `instance-detail` (one-shot fetch, used for the detail screen's first paint)

```json
{ "op": "instance-detail", "reqId": "r2", "target": "Agent#7" }
```

```json
{
  "op": "instance-detail",
  "reqId": "r2",
  "target": "Agent#7",
  "type": "Agent",
  "fields": {
    "name": "researcher",
    "model": "fable-5",
    "status": "Paused",
    "conversation": { "ref": "Conversation#2", "summary": "Conversation#2 (14 messages)" },
    "tools": {
      "list": [
        { "ref": "Tool#1", "summary": "web_search" },
        { "ref": "Tool#4", "summary": "code_exec" }
      ]
    }
  },
  "methods": [
    { "name": "pause", "params": [], "returns": "Void" },
    { "name": "resume", "params": [], "returns": "Void" },
    { "name": "currentTask", "params": [], "returns": "ref:Task" }
  ]
}
```

### `invoke` / `invoke-result` / `invoke-error`

```json
{ "op": "invoke", "reqId": "r3", "target": "Agent#7", "method": "resume", "args": [] }
```

```json
{ "op": "invoke-result", "reqId": "r3", "target": "Agent#7", "method": "resume", "value": null }
```

Typed argument, on a different instance:

```json
{
  "op": "invoke", "reqId": "r4", "target": "Task#3", "method": "setPriority",
  "args": [ { "type": "Int", "value": 1 } ]
}
```

```json
{
  "op": "invoke-error",
  "reqId": "r5",
  "target": "Agent#7",
  "method": "resume",
  "error": {
    "kind": "RuntimeError",
    "message": "resume() requires status == Paused, was Running",
    "trace": [ { "type": "Agent", "method": "resume", "line": 12 } ]
  }
}
```

### `eval` / `eval-result` (the REPL dock)

```json
{
  "op": "eval", "reqId": "r6",
  "context": { "self": "Agent#7" },
  "source": "self.conversation.messages.length"
}
```

```json
{ "op": "eval-result", "reqId": "r6", "value": { "type": "Int", "value": 14 } }
```

### `code-change` (live redefinition — see `03-live-semantics.md`)

```json
{
  "op": "code-change",
  "reqId": "r7",
  "target": { "type": "Agent", "method": "summarize" },
  "source": "fn summarize() -> String {\n  \"TL;DR: \" + self.conversation.lastMessage().content\n}"
}
```

```json
{
  "op": "code-change-result",
  "reqId": "r7",
  "status": "applied",
  "type": "Agent",
  "method": "summarize",
  "schemaVersion": 4
}
```

```json
{
  "op": "code-change-result",
  "reqId": "r7",
  "status": "rejected",
  "type": "Agent",
  "method": "summarize",
  "diagnostics": [ { "line": 2, "col": 34, "message": "expected String, found Int" } ]
}
```

### Push messages (server → client, always carry `subId` + `seq`)

```json
{ "op": "count-updated", "subId": "s1", "seq": 501, "counts": { "Message": 48, "ToolCall": 13 } }
```

```json
{
  "op": "instance-created", "subId": "s2", "seq": 1043,
  "row": { "id": "Message#48", "fields": { "role": "assistant", "conversation": { "ref": "Conversation#2" } } }
}
```

```json
{ "op": "instance-updated", "subId": "s3", "seq": 217, "target": "Agent#7", "fields": { "status": "Running" } }
```

```json
{ "op": "instance-collected", "subId": "s2", "seq": 1044, "target": "ToolCall#4812" }
```

```json
{
  "op": "type-changed", "seq": 88, "type": "Agent", "schemaVersion": 4,
  "methods": [ { "name": "summarize", "params": [], "returns": "String" } ]
}
```

```json
{ "op": "resync-required", "subId": "s2" }
```

## 5. Visual and UX direction

Dark-mode-first, built to demo well on a shared screen, not to be a general-purpose admin
panel skin:

- **Layout**: fixed left rail (entity types, ~240px) + main content + a bottom drawer for
  the REPL that slides up over the content rather than pushing it, so opening the console
  never reflows the table you were reading. Generous whitespace, hairline 1px borders
  instead of drop shadows or card elevation — flat, not skeuomorphic. Breadcrumbs at the
  top of the main content are the only navigation chrome besides the rail; no tabs, no
  hamburger menus.
- **Typography**: two faces only. A monospaced face for anything that *is data* — ids,
  field values, code, the REPL — because these are meant to read as "the actual bits," and
  a humanist sans for UI chrome (labels, buttons, section headers) so chrome and data are
  never visually confusable. Small type by default (13–14px base) in the density style of
  Warp/Linear, not a marketing site's oversized type.
- **Color**: near-black background (`#0d0e12`-ish, not pure `#000`), a single accent hue
  reserved *exclusively* for "this is live/changing" (a field that just updated, a count
  that's climbing, the connection-status dot) so liveness reads as a specific, consistent
  signal rather than decoration. Values get quiet type-coded color (strings, numbers, enum
  pills, reference links each a distinct but desaturated tone) — legible at a glance,
  never a rainbow.
- **Motion as feedback, not decoration**: a field that changes flashes its background for
  ~500ms and fades, rather than snapping — this is what makes "the count is climbing" and
  "the invoke landed" *readable* to an audience watching over someone's shoulder rather
  than requiring them to have been staring at the exact pixel. No other animation
  (no page transitions, no skeleton shimmer) — motion is spent entirely on signaling state
  change, which is the one thing this product is about.
- **Connection status** is a permanent, small, literal indicator (`● live` / `○
  reconnecting…`) in the top bar — because the trust model of this whole tool depends on
  the viewer never silently going stale without telling you.

## OPEN

- **Safety of the invoke/eval channel while the interpreter is running a turn.** An
  `invoke` or `eval` message arriving mid-turn needs a precise semantics: does the runtime
  finish the current bytecode dispatch quantum and run the call at the next safepoint, does
  it require a full stop-the-world, or does invoke run on its own execution context
  concurrently with the program's? This is pinned **OPEN** in `DECISIONS.md` and is not
  decided here — it belongs in `02-runtime.md` alongside the safepoint/GC design, since the
  same safepoint mechanism that makes GC possible is the natural place to also make invoke
  safe. This doc assumes invoke *eventually* completes and produces either a
  `invoke-result`/`invoke-error` plus the field deltas it caused; it does not assume how
  quickly, and the protocol's request/response `reqId` model tolerates an arbitrarily
  delayed answer without changes.
- **Concurrency model of the interpreted program** (also OPEN per `DECISIONS.md`) directly
  affects §2's dirty-tracking cost model, as noted there.
