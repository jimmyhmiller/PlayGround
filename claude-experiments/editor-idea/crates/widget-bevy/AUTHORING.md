# Authoring widgets

A *widget* is a pane that renders a retained UI tree (an `Element`) and
reacts to events. There are **two ways to host one**, and they share the
same `Element` vocabulary and the same set of interactions. They differ
only in where the code runs and how events are delivered:

| | In-process Rhai | Subprocess |
|---|---|---|
| Code | a `.rhai` script in `~/.terminal-bevy/widgets/` | any program speaking NDJSON on stdio |
| Runs on | a worker thread inside the app | its own OS process |
| Event delivery | calls into named script **handlers** (`on_click`, …) | one `HostEvent` JSON line per event on stdin |
| Frame delivery | script `render()` returns an `Element` | program writes a `frame` JSON line on stdout |
| Hot reload | yes (file watcher re-parses) | restart the process |
| Use it for | small, live-editable UI; the default | heavier logic, other languages, isolation |

Both paths produce the **exact same `Element` tree** (see
`src/protocol.rs` for the full catalog) and expose the **exact same
interactions**. The tables below line the two event models up so there's
no ambiguity about which interaction reaches your code as what.

---

## The event model

This is the part that has bitten people: **UI events and the Claude Code
event bus are different things.** A name like "on_event" sounds like it
means "any event", but the bus and the UI are separate channels. Keep
them straight:

- **UI interaction** — the user clicked a button, flipped a toggle,
  picked a tab, typed in an input. These come from *this widget's own
  rendered elements*.
- **Claude Code bus** — `pre_tool_use`, `user_prompt_submit`, `stop`,
  etc., mirrored from the central hook bus. *Every* widget sees *every*
  bus event; filter by `kind` and `payload.cwd`.
- **Widget↔widget bus** — control messages widgets send *each other*
  (`emit` / `on_message`). Scoped to one editor project. This is a
  THIRD, separate channel — not UI, not the Claude bus. See
  "[The widget↔widget bus](#the-widgetwidget-bus)" below.

### Rhai handlers ↔ subprocess `HostEvent`

| Interaction | Rhai handler | Subprocess `HostEvent` (`event` field) |
|---|---|---|
| Button / `ListItem` press | `on_click(x, y, shift, cmd, id)` | `click` `{id}` |
| Press on empty space | `on_click(x, y, shift, cmd, "")` | (n/a — no target) |
| `Toggle` flipped | `on_toggle(id, checked)` | `toggle` `{id, checked}` |
| `Tabs` selection | `on_tab_select(id, tab)` | `tab-select` `{id, tab}` |
| `Input` focus / blur | `on_input_focus(id, focused)` | `input-focus` `{id, focused}` |
| `Input` edited | `on_input_change(id, value)` | `input-change` `{id, value}` |
| `Input` Enter | `on_input_submit(id, value)` | `input-submit` `{id, value}` |
| drag / release | `on_drag(x, y)` / `on_release(x, y)` | (rhai only) |
| hover (x=inf on leave) | `on_hover(x, y)` | (rhai only) |
| nav key, no input focused | `on_key(key)` | (rhai only) |
| pane resized | `on_resize(w, h)` | `resize` `{width, height}` |
| per frame, while animating | `on_frame(dt)` | `tick` `{dt}` |
| **Claude Code bus** | **`on_bus(kind, payload)`** | `claude-event` `{kind, payload}` |
| **Widget↔widget bus** | **`on_message(topic, payload, sender)`** | `message` `{topic, payload, sender}` |
| lifecycle: spawned | `on_init()` | `init` `{width, height, state}` |
| lifecycle: closing | (worker `Shutdown`) | `close` |

`checked`, `tab`, and `value` are all computed host-side and handed to
you ready to use — `checked` is already the *new* value, you don't invert
it; `value` is the full new string, not a delta.

> `on_bus` was historically named `on_event`. That name is still
> accepted as a fallback but is deprecated, precisely because it implied
> "all events" and led authors to expect UI events there. Use `on_bus`.

## The widget↔widget bus

Several small widgets can act as one app — an editor pane, a results
pane, a schema browser — by sending each other control messages. This is
a general publish/subscribe channel, **separate from the Claude bus**
(`on_bus`) and from UI events.

### Publish

```rhai
emit("sql.run", #{ sql: state.query });   // native value — host serializes
emit("schema.changed");                    // payload-less signal
```

`emit(topic, payload)` is fire-and-forget. `payload` is any native Rhai
value (map `#{…}`, array, string, number, bool) — the **host** encodes
it, so you never touch JSON in a script. The message is broadcast to
every widget **in the same editor project** (panes in other projects
never see it).

### Receive

```rhai
fn on_message(topic, payload, sender) {
    if sender == my_id() { return; }       // ignore echoes of our own emits
    if topic == "sql.run" {
        run(payload.sql);
    }
}
```

Delivery is **pushed** — the host wakes your worker and calls
`on_message` directly. You do **not** need `set_animating` / `on_frame`;
the bus is fully event-driven. `sender` is the publishing widget's id
(`"tbmsg"` for the CLI); compare it to `my_id()` to skip your own
messages, or use it to address a targeted reply (e.g. a topic naming the
sender).

### Retained messages (late joiners)

A pane that opens *after* a message was sent would miss it. For state
that late joiners need — the current DB connection, the current query —
use `emit_retained`:

```rhai
emit_retained("conn.state", #{ host: "localhost", ok: true });
```

The host keeps the **last** retained value per topic and redelivers it to
each widget once, on init. So a results pane opened later immediately
learns the current connection without asking anyone. Retain is in-memory
only (it does not survive an app restart).

### Subprocess widgets

Same model over NDJSON: publish with `WidgetMsg::Emit { topic, payload,
retain }` on stdout; receive `HostEvent::Message { topic, payload,
sender }` on stdin.

### From the shell — `tbmsg`

```sh
tbmsg emit --project datalog-db --topic sql.run --json '{"sql":"select 1"}'
tbmsg emit --project datalog-db --topic conn.state --json '{"ok":true}' --retain
tbmsg tail --project datalog-db        # follow the bus live (one JSON line each)
```

Handy for driving a widget from a `proc_spawn`ed child or verifying flow
without the GUI. `--project` takes a project name or `active`. Messages
from the CLI arrive with `sender = "tbmsg"`.

### Suggested topic conventions

Dotted topic names keyed by concern, e.g. for a SQL IDE:

| topic | payload | direction |
|---|---|---|
| `sql.run` | `{sql}` | editor → results: execute |
| `sql.result` | `{columns, rows, error, ms}` | results → *: a query finished |
| `sql.set_editor` | `{sql}` | history → editor: load text |
| `schema.changed` | `{}` | after DDL → browser: refresh |
| `conn.state` | `{host, ok}` *(retained)* | conn → late joiners |

Keep payloads small — the bus carries **control signals** (~kilobytes),
not bulk data. Big result sets stay where they were produced (or go
through the datalog DB); the bus just says "ready".

### Single-line vs multi-line input

- **`Input`** is single-line. `Enter` fires `on_input_submit`.
- **`TextArea`** is multi-line: `Enter` inserts a newline and the box is
  `rows` lines tall (default 4). Submit (`on_input_submit`) is
  **Cmd/Ctrl+Enter** — the usual "run query" gesture. Arrows move the
  caret across lines; Home/End are line-aware. Hard newlines only (no
  soft wrap).

Both emit the *same* `on_input_change` / `on_input_submit` /
`on_input_focus` handlers (subprocess: `input-change` / `input-submit` /
`input-focus`), with `value` carrying the full string (newlines and
all). So a query box is just:

```rhai
#{ type: "textarea", id: "query", value: state.q, rows: 6,
   placeholder: "SELECT …" }

fn on_input_change(id, value) { state.q = value; request_render(); }
fn on_input_submit(id, value) { run_query(value); }   // Cmd/Ctrl+Enter
```

### Tables

`Element::Table` renders a header row + data rows on a grid. Columns
size to their content (capped, then the cell text wraps) unless you give
an explicit `width`; set per-column `align` for right-aligned numbers.
`zebra` stripes alternate rows.

```rhai
#{ type: "table", zebra: true,
   columns: [
     #{ header: "id",    width: 48.0, align: "end" },
     #{ header: "name" },
     #{ header: "role" },
     #{ header: "score", width: 70.0, align: "end" },
   ],
   rows: [
     ["1", "Ada Lovelace", "Engineer", "98"],
     ["2", "Alan Turing",  "Researcher", "91"],
   ] }
```

Cells are plain strings (row-major; a short row leaves later cells
empty). The table sizes to its content width rather than filling the
pane, so give wide columns an explicit `width` when you want a specific
layout.

### Focused-input ownership

While an `Input` or `TextArea` is focused, the **host owns** the live
edit buffer and the blinking caret (`WidgetInputFocus`). That means:

- Typing echoes instantly — you do **not** need to round-trip a frame to
  show keystrokes.
- You get `on_input_change` after each edit and `on_input_submit` on
  Enter; react to those (e.g. run a search, store the value in `state`).
- The element's `value` you render is only the *initial* / committed
  value; the host substitutes the live buffer while focused.
- Nav keys (arrows / Home / End) drive the caret and do **not** fire
  `on_key` while an input is focused.

---

## Writing a Rhai widget

Drop a `.rhai` file in `~/.terminal-bevy/widgets/`. The file watcher
re-parses on save. The top-level body runs **once per load** (initialize
`state`, define handler `fn`s). All handlers are optional — define only
what you need.

```rhai
// counter.rhai
if !("n" in state)    { state.n = 0; }
if !("dark" in state) { state.dark = false; }
if !("q" in state)    { state.q = ""; }

fn on_init() { request_render(); }

fn on_click(x, y, shift, cmd, id) {
    if id == "inc" { state.n += 1; }
    if id == "dec" { state.n -= 1; }
    request_render();
}

fn on_toggle(id, checked)     { if id == "dark" { state.dark = checked; } request_render(); }
fn on_input_change(id, value) { if id == "search" { state.q = value; } request_render(); }
fn on_input_submit(id, value) { if id == "search" { run_search(value); } }

fn on_bus(kind, payload) {
    // Claude Code bus — NOT UI events.
    if kind == "stop" { state.n = 0; request_render(); }
}

fn render(w, h) {
    #{ type: "vstack", gap: 8.0, pad: 12.0, children: [
        #{ type: "text", value: "count: " + state.n },
        #{ type: "hstack", gap: 4.0, children: [
            #{ type: "button", id: "dec", label: "-" },
            #{ type: "button", id: "inc", label: "+" },
        ]},
        #{ type: "toggle", id: "dark", label: "Dark", checked: state.dark },
        #{ type: "input",  id: "search", value: state.q, placeholder: "search…" },
    ]}
}
```

### `state` and persistence

`state` is a `Map` in scope, persisted across restarts and hot reloads
(round-tripped to JSON into the pane snapshot). Mutate it in place.

### Scheduling renders

Rhai widgets are **event-driven** — there is no per-frame poll by
default. After a handler mutates state, call `request_render()` to redraw
once. For continuous animation, call `set_animating(true)` to start
receiving `on_frame(dt)`; `set_animating(false)` to stop (idle widgets
cost zero CPU).

### Function scoping gotcha

User-defined `fn`s are pure: they do **not** see top-level `const`s, and
only host-invoked handlers receive `state`. Helpers take what they need
as parameters. (See `project_editor_idea_rhai_scoping` in memory.)

### Host functions available to scripts

`request_render`, `set_animating`, `time`, `rand`, `rand_int`,
`hash_str`, `host_env`, `host_log`, `clipboard_set`, `widget_asset`, the
widget↔widget bus `emit` / `emit_retained` / `my_id` (see "[The
widget↔widget bus](#the-widgetwidget-bus)"), and the generic subprocess
primitives `proc_spawn` / `proc_write` / `proc_read` / `proc_alive` /
`proc_kill`.

---

## Writing a subprocess widget

Speak NDJSON on stdio: read one `HostEvent` per line on stdin, write
`frame` / `state` / `title` messages on stdout. The enum definitions
(`HostEvent`, `WidgetMsg`, `Element`, `Style`) and their exact JSON shape
live in `src/protocol.rs`. The same interaction table above applies —
just delivered as JSON lines instead of handler calls.

---

## Where the code lives

- `src/protocol.rs` — `Element` catalog, `Style`, `HostEvent`,
  `WidgetMsg`. The single source of truth for the UI vocabulary.
- `src/rhai_widget.rs` — the in-process Rhai host: worker thread,
  `HostToWorker` channel, handler dispatch, hot reload. The module-level
  doc has the handler table inline.
- `src/lib.rs` — the subprocess host (`WidgetIO`, NDJSON pump) plus the
  shared rendering, hit-testing, scroll, and focused-input typing that
  both paths use.
- `src/render.rs` / `src/layout.rs` — `Element` → Bevy sprites (Taffy
  flexbox layout).
