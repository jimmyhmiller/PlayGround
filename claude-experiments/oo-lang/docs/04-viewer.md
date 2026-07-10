# 04 — Viewer: UX and the Eval Channel

The viewer is not a message-viewer bolted onto the runtime, and it is not consuming a feed.
**We are in the running program.** The browser is a second interface to the same process
stdin/stdout is the first — and the only thing that ever crosses the wire between them is an
`eval`: an expression or definition in oo-lang itself, sent to the live interpreter, run
against the live heap at a safepoint, with the result serialized back. Every screen below —
type index, instance table, instance detail, method invocation, the REPL dock, even live code
change — is sugar over that one operation. This is `DECISIONS.md` #8, stated as an emphatic
correction to an earlier draft of this doc that modeled the viewer as a subscriber to a
message/delta feed. That model is gone, not simplified — there is no subscribe, no `subId`,
no sequence numbers, no dirty bitmasks, no server-initiated push of any kind. Examples
throughout use the agent-TUI demo's types (`Agent`, `Conversation`, `Message`, `Tool`,
`ToolCall`, `Task`) from `00-vision.md`.

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
count deltas it has itself observed — not a stored server-side history). A search box
filters the type list by name as you type; this is a client-side filter over the schema
already in hand, not a new eval per keystroke.

```
🔍 search types…

Agent           3 live
Conversation    3 live
Message        47 live   ▲
ShellTool        3 live
SearchTool       2 live
ToolCall        12 live   ▲
Task             8 live
```

**Instance table** (click a type). Paged, sortable, filterable table. Columns are the
type's fields in declaration order, plus an `id` column pinned left. The filter bar takes a
predicate over fields (`status == Paused`, `name contains "research"`) — a small expression
subset of the language itself (field access, comparisons, `and`/`or`, zero-arg method
calls), passed as a string argument into the `instances(...)` eval described in §4, so
"the server evaluates the filter, not the client" is true by construction: there is no
separate filter-evaluation code path, it's the same `eval` everything else uses. Reference-
typed columns (`conversation: Conversation`) render as clickable links showing the target's
id and a one-line summary (`Conversation#2 (14 messages)`), not a raw pointer — never the
whole referenced object inlined (§4's depth rule).

**Instance detail** (click a row, or follow a reference link). Three sections:

- *Fields* — name, static type, current value, refreshed by re-running the same `at(...)`
  eval that painted the page (§2). Reference fields are links that navigate (pushing a
  breadcrumb: `Agent#7 › conversation › Conversation#2`); collection fields (`List<Tool>`)
  render as an expandable inline list of link chips, not a sub-table, unless the list has
  more than ~20 elements, in which case it's a link to an instance-table view pre-filtered
  to that collection.
- *Methods* — every method with its signature (`fn setPriority(p: Int) -> Void`), read off
  the same reflection eval that built the type index (§3.2). Clicking one expands an inline
  invocation form in place (§3) — never a modal, so you can see the instance's current
  field values while filling in arguments.
- *Recently changed* — fields whose value differs from the last re-eval the client itself
  observed, shown with the previous value struck through for a few seconds before fading.
  There is no server-side "changed" concept to reconcile with this — it is a client-side
  diff of two eval results, nothing more (§2).

**REPL/eval dock**. A persistent, collapsible bottom drawer (toggled with a keyboard
shortcut, ` — the "console key" convention from Quake/Source-engine games and browser
devtools), not a separate page. Whatever instance is open in the detail view is bound to
`self` for anything typed here (§3.3 shows the mechanism — it's a wire-level trick, not a
server feature); empty at the type-index/table screens, so `self.conversation.messages.length`
just works when you're looking at an `Agent`. Output is a scrollback of `expr → value`
pairs, each value rendered with the same typed-value renderer used for fields (so a
returned `Task` reference is a clickable link, not a string) — because under the hood it
*is* the same eval channel, just with raw user-typed source instead of a canned expression
the UI generated.

Global search (top bar, always visible) jumps directly to an instance by id (`Agent#7`,
parsed into an `at(...)` eval) or does a cross-type substring search over field values —
itself just another eval over `instances(...)` for every type, run client-side in parallel
and merged. This is the "I don't know which screen it's on" escape hatch.

## 2. Refresh model: re-eval, not push

**Nothing is pushed, ever.** There is no subscription, no dirty tracking, no server-
initiated message of any kind — the server only ever answers an eval that arrived. A pane
gets fresh data by re-issuing the same eval that produced what's on screen. Concretely:

- **On an interval, while the pane is focused.** The type index re-evals `types()` every
  ~250ms–1s; an open instance table re-evals its `instances(...)` call; an open detail
  view re-evals its `at(...)` call. Backgrounding a tab or navigating away stops the
  timer — there's no server-side subscription to leak, because there was never a
  subscription to begin with.
- **On focus.** Returning to a tab (or a pane regaining focus) triggers an immediate
  re-eval rather than waiting for the next tick, so nothing shows stale-on-return.
- **Immediately after any action.** Invoking a method, submitting a code change, or typing
  a new filter all re-issue the relevant eval(s) right away, in addition to whatever
  interval is running — this is what makes "the terminal updates the instant I invoke
  `pause()`" (`00-vision.md`'s second demo beat) true: the invoke's own eval mutates state,
  and the detail pane's *next* eval (fired immediately, not on the next tick) reads the new
  value back. There is no separate "changed" push riding along with the invoke response;
  the freshness comes from asking again, right after.

**Why this is honest, not a downgrade.** `02-runtime.md`'s entire thesis is that arena
enumeration is cheap by design — "list every live `Agent`" is an O(live count) slab walk,
`at(slot, gen)` is one `arena-slot-ptr` plus a generation compare, `instances(filter, ...)`
is a linear scan over one type's arena evaluating an O(1)-per-field predicate (§8's own
table). None of that got more expensive by dropping push; push was never buying cheapness,
it was buying the *appearance* of not having to ask. Re-asking a cheap question on a
250ms–1s human-paced interval costs nothing an arena walk wasn't already going to cost on
every "tick" the old dirty-bitmask design proposed — the difference is there's no bitmask,
no per-subscription diff, no seq numbers to keep consistent, because the interpreter never
has to remember what a particular browser tab last saw. Every eval is stateless from the
server's point of view; all "did anything change" logic lives client-side, as a diff
between this poll's JSON and last poll's JSON it already has in memory (this is exactly
what powers the field-flash in §1's "Recently changed" and §5's motion design — nothing
new is asked of the runtime for it).

**Safepoint honesty, carried over from the old design because it's still true.** Every eval
executes only when the single interpreter thread reaches a safepoint (`02-runtime.md` §7 —
a check at every loop back-edge and call site). Evals arriving concurrently from multiple
open panes are serialized through the same completion queue viewer requests already share
with async-I/O completions: each one runs to completion before the next is serviced, so an
`instances(...)` eval never observes a half-mutated object mid-invoke, and two evals never
interleave their effects. The cost this buys is the same one the old design named and did
not hide: if the interpreter is parked inside a blocking foreign call (`02-runtime.md` §7's
"genuinely hard part"), every pending eval — a poll tick, an invoke, a REPL keystroke —
queues behind it until the next safepoint. A viewer reading stale counts during that window
is observing a real, already-acknowledged interpreter-availability gap, not a bug in the
refresh model.

## 3. Method invocation and code change, both just eval

### 3.1 Invoking a method

Invoking is inline, not modal, and typed, exactly as before — only the mechanism under the
"Invoke" button changed:

- **Zero-arg methods** (`resume()`, `pause()`) show a single "Invoke" button. Clicking it
  builds the source string `Agent.at(7, 3).resume()` — the instance's own stable id/
  generation pair (already in hand from the eval that painted the row) spliced into a
  literal method call — and sends it as an ordinary eval.
- **Typed scalar params** (`Int`, `String`, `Bool`, `Float`) get a matching input widget;
  their values are spliced as literals into the same call expression
  (`Task.at(3, 1).setPriority(1)`).
- **Enum params** get a dropdown of the enum's variants, read off the same reflection eval
  that lists the method's signature (§3.2) — the viewer never has to guess valid values.
- **Reference-typed params** (e.g. `assign(t: Task)`) get a type-ahead picker over
  `Task.instances(...)` (the same eval the instance table uses); picking one splices
  `Task.at(slot, gen)` as that argument's source text — an argument is never anything but
  more oo-lang source.
- **Anything the form-builder doesn't have a widget for** falls back to a "raw" toggle: a
  text field whose contents are spliced verbatim into the call expression, typechecked
  server-side (by the ordinary typechecker, on the ordinary eval) before it runs. This is
  the pressure valve — it must exist so the form-builder never blocks a valid call, but
  it's opt-in per argument.

**Results.** A successful invoke's eval response (`{"value": ...}`, §4) is rendered inline
right below the method, using the same typed-value renderer as fields, tagged with a
timestamp and a fading highlight. An error (`{"error": ...}`, §4) shows as a red inline
panel directly under the form — never a toast, never a separate log — with the error kind,
message, and a source-level call-stack trace (`Type.method (line N)` frames, not raw Coil
frames), because the whole point of this system is that the person looking at it may not be
the one who wrote the interpreter.

**stdout.** The viewer never captures or redirects the target process's stdout/stderr —
per the "ordinary process" decision, the program's stdout belongs to the terminal it was
launched from, full stop. Invoking `pause()` prints nothing to the browser; the viewer
shows the eval's structured `value`/`error`, and the detail pane's next re-eval (§2) is
what shows the mutated field. If the invoked method happens to write to stdout, that text
appears in the terminal the program is running in — because to the runtime, an invoke *is*
a call, indistinguishable from one the program made itself.

### 3.2 Schema discovery — also just eval

There is no `hello` handshake, no schema-versioned push. The question "what types, fields,
methods exist" is answered by a reflection eval, proposed here as a minimal addition to
`01-language.md`'s surface since that doc does not currently define one:

```
types()                     -> List<TypeDescriptor>   // name, fields, methods, liveCount — everything
                                                       // the type index and a fresh detail pane need
fields(typeName: String)    -> List<FieldDescriptor>  // convenience: one type's shape without the rest
methods(typeName: String)   -> List<MethodDescriptor>
```

`TypeDescriptor`/`FieldDescriptor`/`MethodDescriptor` are plain values (never arena-backed,
never carry a `ref`) — structural metadata, not entities. `types()` alone is enough to paint
the whole landing screen in one eval; `fields`/`methods` exist so a detail pane refreshing
its method list doesn't have to re-fetch every other type's schema too.

**OPEN, flagged here rather than silently decided:** whether `types()`/`fields()`/
`methods()` (and §4's `instances()`/`at()`) are ordinary language-level stdlib — callable
from inside a running `.oo` program itself, for self-inspection — or a builtin surface
reserved to the eval channel only, never visible to ordinary program source. The former is
more uniform (one mechanism, no privileged namespace) and is what's assumed by default
throughout this doc; the latter avoids ordinary program logic coming to depend on viewer-
only reflection machinery. `01-language.md` should pin this; nothing here requires one
answer over the other.

### 3.3 The REPL dock's `self` binding — a wire-level trick, not a server feature

`self` in the REPL is bound to whatever instance the detail view has open — but the request
shape (§4) is exactly `{id, source}`, with no separate "context" field to carry it. The
binding is done by the *viewer*, textually, before the eval is ever sent: given
`Agent#7`/generation `3` open and the user typing `self.conversation.messages.length`, the
viewer sends

```
{ let self = Agent.at(7, 3)
  self.conversation.messages.length }
```

as the `source` — an ordinary block expression, using nothing `01-language.md` §1.4 doesn't
already define. No protocol feature was needed for `self` to work; it falls out of eval
being real language source and the language already having block-scoped `let`.

### 3.4 Code change — a definition, evaluated

Live redefinition (`03-live-semantics.md`) is not a special op either. The `source` sent is
a class (or, if the language later grows standalone method-target syntax, a method)
definition — the same text you'd write in a `.oo` file — and the runtime's response is
either the definition's acceptance message as an ordinary `value`, or a rejection as an
`error` carrying `diagnostics` (§4.3). See §4.4 for concrete request/response pairs.

## 4. The eval channel

Transport: a plain HTTP endpoint, `POST /eval`, alongside static assets served over
ordinary HTTP. A persistent connection (keep-alive, or a WebSocket used purely as a
request/response pipe with zero push semantics) is a legitimate implementation choice to
cut per-request overhead under the ~250ms poll cadence of §2, but it is exactly that — an
implementation detail. The message shape below is identical either way, and nothing in the
protocol depends on a connection surviving between requests, because nothing is ever
pushed down it. This is a real simplification worth naming against `02-runtime.md` §7's
flagged risk: that section names the WebSocket handshake (RFC 6455, SHA-1, base64, frame
masking) as unbuilt, hand-rolled machinery sitting under every milestone from M1 on. None of
that is required by this protocol anymore — a request/response HTTP endpoint has zero
handshake crypto and zero frame-masking surface. Whether to build the WS pipe anyway, purely
as a performance optimization once profiling asks for it, is `05-milestones.md`'s call, not
a protocol requirement.

There is exactly one message shape in each direction, and no `op` field, because there is
nothing to discriminate:

**Request:**
```json
{ "id": "e1", "source": "types()" }
```

**Response — success:**
```json
{ "id": "e1", "value": { /* §4.1 */ } }
```

**Response — failure:**
```json
{ "id": "e1", "error": { "kind": "...", "message": "...", "trace": [...] } }
```

`source` is any expression or definition legal at top level in oo-lang (`01-language.md`).
`id` is client-chosen and echoed back, so concurrent in-flight requests (several panes
polling at once) can be matched to their responses — the one piece of request/response
bookkeeping this protocol needs, and it needs no `subId`/`seq` alongside it because nothing
is ever unsolicited.

### 4.1 Result serialization

Every value carries a `type` tag so the viewer's renderer never has to guess:

| Kind | Shape | Notes |
|---|---|---|
| `Int`/`Float`/`Bool`/`String` | `{"type":"Int","value":14}` | direct |
| `Void` | `{"type":"Void"}` | method calls with no return |
| enum, no payload | `{"type":"AgentStatus","case":"Paused"}` | `type` is the enum's own name |
| enum, with payload | `{"type":"Shape","case":"Circle","payload":[{"type":"Float","value":2.0}]}` | payload is positional, per the case's declared fields |
| entity, **depth 0** | `{"type":"Agent","ref":"Agent#7","generation":3,"fields":{...}}` | full field expansion — see depth rule below |
| entity, **depth ≥ 1** | `{"type":"ref","class":"Agent","ref":"Agent#7","generation":3,"summary":"researcher"}` | collapsed to a clickable link, never the object graph |
| `List<T>`/collection | `{"type":"list","elementType":"Message","length":47,"truncated":true,"items":[...]}` | truncates at ~20 items; `length` is always the true count |
| `Map<K,V>` | `{"type":"map","keyType":"String","valueType":"Int","length":3,"truncated":false,"entries":[[k,v],...]}` | same truncation rule |

`Option<T>`/`Result<T,E>` need no special case at all — they're ordinary enums
(`01-language.md` §3.3/§3.4), so `None` serializes as `{"type":"Option","case":"None"}` and
`Some(v)` as `{"type":"Option","case":"Some","payload":[v]}` under the enum rule above. One
serialization mechanism, not a second one for "maybe absent."

**The depth rule** is what keeps the object graph from being serialized wholesale, and it's
one rule applied uniformly: *depth 0* is the value(s) the eval directly returns — including,
if the returned value is itself a list or map, each of its elements (this is what makes the
instance table's rows show full fields, not links-to-themselves). Anything reached by
stepping through one entity-typed **field** — a reference field on a depth-0 entity, an
element of a list that is itself such a field — is depth ≥ 1 and always renders collapsed,
regardless of how deep the actual object graph goes from there. Clicking a collapsed
reference fetches it properly by issuing a fresh eval — typically `Type.at(slot, gen)` —
which makes the clicked object depth 0 for that new request. There is no server-side depth
limit to configure; the limit is structural (one field-hop) and the same for every type.

Total size is bounded the same way collections are: if a single response would exceed a
byte budget (a large `List<Message>`, a `String` field holding a big blob), the server
truncates and sets `truncated: true`; the viewer's "show more" affordance is another eval —
`Conversation.at(2, 1).messages` sliced with an offset, or a dedicated paged
`instances(...)` call scoped to that collection — never a second protocol mechanism.

### 4.2 The `instances`/`at` builtins these evals lean on

Every entity class gets two compiler-synthesized static members, the same way a
memberwise constructor is synthesized (`01-language.md` §1.1) — this is the minimal
reflection surface §3.2 flags as OPEN for language-level-vs-eval-channel-only, extended to
per-instance access:

```
Agent.instances(filter: String, sort: String, offset: Int, limit: Int) -> List<Agent>
Agent.at(slot: Int, generation: Int) -> Agent
```

`instances` walks the type's arena (`02-runtime.md` §5's `arena-for-each-live`), evaluating
`filter` as the same small predicate subset `05-milestones.md`'s query language already
scopes (`field == literal`, numeric comparisons, `contains`) — no new query language, the
one from the old design's instance-table filter bar is reused verbatim, just as a string
argument to an ordinary function instead of a side-channel query field. `at` resolves the
identity triple exactly as `02-runtime.md` §5 describes: `arena-slot-ptr` plus a generation
compare. A generation mismatch — the slot was freed and possibly reused — is not a language-
level `Option`/error; it's an eval-level failure, because "the reference you held went
stale" is a property of holding a handle across time from *outside* the process, which is
exactly the boundary this channel exists to police:

```json
{ "id": "e6", "error": { "kind": "StaleReference",
  "message": "Agent#7 (generation 2) is no longer live — slot has been freed or reused" } }
```

### 4.3 Errors

```json
{
  "id": "e5",
  "error": {
    "kind": "RuntimeError",
    "message": "resume() requires status == Paused, was Running",
    "trace": [ { "type": "Agent", "method": "resume", "line": 12 } ]
  }
}
```

Code-change rejections reuse the same envelope with a `diagnostics` array instead of/in
addition to a `trace`, for both flavors `03-live-semantics.md` names:

```json
{ "id": "e8", "error": { "kind": "TypeError",
  "message": "Agent.summarize: expected String, found Int",
  "diagnostics": [ { "line": 9, "col": 34, "message": "expected String, found Int" } ] } }
```

```json
{ "id": "e8", "error": { "kind": "LiveStateError",
  "message": "field 'notes' added with no default and no migration function; 17 live Agent instances exist" } }
```

### 4.4 Worked examples, end to end

**Type index refresh** (§1, §2 — one eval covers the whole landing screen):

```json
→ { "id": "e1", "source": "types()" }
← { "id": "e1", "value": { "type": "list", "elementType": "TypeDescriptor", "length": 7, "truncated": false,
    "items": [
      { "type": "TypeDescriptor", "name": "Agent", "liveCount": 3,
        "fields": [ { "name": "name", "type": "String" }, { "name": "status", "type": "AgentStatus" },
                    { "name": "conversation", "type": "ref:Conversation" } ],
        "methods": [ { "name": "pause", "params": [], "returns": "Void" },
                     { "name": "resume", "params": [], "returns": "Void" } ] },
      { "type": "TypeDescriptor", "name": "Conversation", "liveCount": 3, "fields": [...], "methods": [...] },
      { "type": "TypeDescriptor", "name": "Message", "liveCount": 47, "fields": [...], "methods": [] }
    ] } }
```

**Instance table** (§1, filtered):

```json
→ { "id": "e2", "source": "Agent.instances(filter: \"status == Paused\", sort: \"name\", offset: 0, limit: 50)" }
← { "id": "e2", "value": { "type": "list", "elementType": "Agent", "length": 1, "truncated": false,
    "items": [
      { "type": "Agent", "ref": "Agent#7", "generation": 3, "fields": {
          "name": { "type": "String", "value": "researcher" },
          "model": { "type": "String", "value": "fable-5" },
          "status": { "type": "AgentStatus", "case": "Paused" },
          "conversation": { "type": "ref", "class": "Conversation", "ref": "Conversation#2",
                             "generation": 1, "summary": "Conversation#2 (14 messages)" },
          "tools": { "type": "list", "elementType": "Tool", "length": 2, "truncated": false,
            "items": [
              { "type": "Tool", "case": "Shell", "payload": [
                  { "type": "ref", "class": "ShellTool", "ref": "ShellTool#1", "generation": 1, "summary": "shell" } ] },
              { "type": "Tool", "case": "Search", "payload": [
                  { "type": "ref", "class": "SearchTool", "ref": "SearchTool#4", "generation": 1, "summary": "search" } ] }
            ] } } }
    ] } }
```

**Instance detail, following the `conversation` link** (§1, §4.2):

```json
→ { "id": "e3", "source": "Conversation.at(2, 1)" }
← { "id": "e3", "value": { "type": "Conversation", "ref": "Conversation#2", "generation": 1, "fields": {
      "task": { "type": "ref", "class": "Task", "ref": "Task#3", "generation": 1, "summary": "Task#3 (scrape docs)" },
      "messages": { "type": "list", "elementType": "Message", "length": 34, "truncated": true,
        "items": [
          { "type": "ref", "class": "Message", "ref": "Message#1", "generation": 1, "summary": "user: \"summarize the PR\"" },
          { "type": "ref", "class": "Message", "ref": "Message#2", "generation": 1, "summary": "assistant: \"Here's what I found...\"" }
          /* … 18 more, then truncated: true */
        ] } } } }
```

**Invoke** (§3.1):

```json
→ { "id": "e4", "source": "Agent.at(7, 3).resume()" }
← { "id": "e4", "value": { "type": "Void" } }
```

**REPL dock, `self`-scoped** (§3.3):

```json
→ { "id": "e7", "source": "{ let self = Agent.at(7, 3)\n  self.conversation.messages.length }" }
← { "id": "e7", "value": { "type": "Int", "value": 14 } }
```

**Code change** (§3.4):

```json
→ { "id": "e8", "source": "class Agent {\n  name: String\n  model: String\n  status: AgentStatus\n  conversation: Conversation\n  tools: List<Tool>\n\n  fn pause() { self.status = AgentStatus.Paused }\n  fn resume() { self.status = AgentStatus.Waiting }\n  fn summarize() -> String {\n    \"TL;DR: \" + self.conversation.lastMessage().content\n  }\n}" }
← { "id": "e8", "value": { "type": "String", "value": "method Agent.summarize replaced — next call uses new code" } }
```

### Relationship to `05-milestones.md`'s build order

This simplifies the milestone story rather than complicating it, because there is only one
wire mechanism to build, not a request/response layer plus a later subscription upgrade.

- **M1 (viewer, read-only):** the server exists, `eval` exists, and the UI issues a fixed,
  canned set of expressions it constructs itself — `types()`, `Type.instances(...)`,
  `Type.at(...)`. No user-authored source ever reaches the interpreter yet; the REPL dock
  and per-method invoke buttons aren't built, so there's no way to *send* an arbitrary
  eval even though the channel underneath is the same one that will carry them later.
- **M2 (interaction):** the same channel is unlocked to carry arbitrary source —
  the REPL dock ships, and invoke buttons construct method-call expressions (§3.1) instead
  of the UI being restricted to the canned M1 set. No new op, no new response shape, no
  migration of anything: M1's requests keep working unchanged because they were always
  just evals.
- **M3 (live change):** `source` is allowed to be a definition instead of an expression
  (§3.4). Same channel, same envelope, still no new op.

Nothing here is a wire-protocol distinction — it is entirely which expressions the UI is
willing to construct and send, which is a front-end scoping question, not a backend one.
`05-milestones.md` should read M1→M2→M3 as "the UI's vocabulary of self-issued evals grows,"
not as three different protocols to reconcile.

**Note vs. `02-runtime.md` §8.** That section's table still frames `types`/`instances`/
`instance`/`invoke` as distinct "wire ops," each resolving via arena machinery. The
machinery described there — `arena-for-each-live`, `arena-slot-ptr` + generation check,
`TypeInfo.methods` lookup — is exactly right and is what `types()`/`instances(...)`/
`at(...)`/method calls above compile down to. Read its table as documenting *interpreter-
side call targets*, not separate wire-level operations; this doc is authoritative for what
actually crosses the wire (one shape, `{id, source}` → `{id, value|error}`), and
`02-runtime.md`'s "op" language there should be understood as shorthand for "the builtin
this eval resolves to," the same way its §8 already reads once `04-viewer.md`'s eval-only
pin is applied.

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
  reserved *exclusively* for "this just changed" (a field whose latest re-eval differs from
  the one before it, a count that's climbing, the connection-status dot) so liveness reads
  as a specific, consistent signal rather than decoration. Values get quiet type-coded color
  (strings, numbers, enum pills, reference links each a distinct but desaturated tone) —
  legible at a glance, never a rainbow.
- **Motion as feedback, not decoration**: a field flashes its background for ~500ms and
  fades when it changes, rather than snapping. Since there's no server-pushed delta to key
  this off (§2), the client does the diff itself: every re-eval's serialized value is
  compared against the last one this pane received, field by field, entirely in the
  browser — the runtime is never asked "what changed," only "what is it right now," as
  often as the refresh model calls for. No other animation (no page transitions, no
  skeleton shimmer) — motion is spent entirely on signaling state change.
- **Connection status** is a permanent, small, literal indicator (`● live` / `○
  reconnecting…`) in the top bar. With no push channel to monitor, "live" means the most
  recent eval round-trip completed within its expected budget; "reconnecting" means the
  last request errored at the transport level (connection refused/reset) or is still
  outstanding well past that budget — still a simple, honest, always-visible signal that
  the tool depends on for trust, just derived from request/response timing instead of a
  socket's open/closed state.

## OPEN

- **Safety of the invoke/eval channel while the interpreter is running a turn.** An eval
  arriving mid-turn needs a precise semantics: does the runtime finish the current
  bytecode dispatch quantum and run it at the next safepoint, does it require a full
  stop-the-world, or does it run on its own execution context concurrently with the
  program's? This is pinned **OPEN** in `DECISIONS.md` and is not decided here — it belongs
  in `02-runtime.md` alongside the safepoint/GC design. This doc assumes every eval
  *eventually* completes and produces either `value` or `error`; it does not assume how
  quickly, and the `{id, source}`/`{id, value|error}` shape tolerates an arbitrarily
  delayed answer without changes.
- **Concurrency model of the interpreted program** (also OPEN per `DECISIONS.md`) directly
  affects how many evals can be in flight against one arena at once and whether `at`'s
  generation check needs to become atomic; §2's "cheap by design" argument assumes today's
  single-writer discipline (`02-runtime.md` §7).
- **Reflection surface — language-level or eval-channel-only** (§3.2): `types()`,
  `fields()`, `methods()`, `instances()`, `at()` are proposed here as compiler-synthesized
  members in the same family as the synthesized memberwise constructor, but whether
  ordinary `.oo` program source is allowed to call them too (self-inspection) or they're
  reserved to requests arriving over the eval channel is left to `01-language.md`.
- **Standalone method-target definition syntax** (§3.4): code-change examples here redefine
  a whole class; whether `01-language.md` grows a lighter `fn Agent.summarize() -> ... { }`
  form for a single-method edit, or whole-class redefinition is the only surface the PoC
  ships, is that doc's call, not this one's.
