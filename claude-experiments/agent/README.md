# flowline

A native agent-workflow runner built in Coil with Raylib. It loads a workflow program, derives node readiness from its dependency graph, launches real Codex agents, captures their final responses, and pauses at explicit human-approval nodes.

## Run

Requirements:

- `coil`
- Raylib 5.5 available to the system linker

On macOS with Homebrew Raylib installed:

```sh
coil run
```

Or build and run separately:

```sh
coil build
./flowline
```

Run commands from the repository root so the bundled fonts resolve from `assets/fonts`.

## Structure

- `src/main.coil` contains the application state, explicit state transitions, data, layout, and rendering.
- `src/workflow.coil` parses workflow programs and interprets their dependency graph at runtime.
- `src/raylib.coil` is the small audited C binding used by the application.
- `workflows/*.flow` are executable workflows discovered by the interface at startup and after creation.
- `assets/fonts` contains IBM Plex Mono and IBM Plex Sans at high rasterization sizes for clean UI text.
- `licenses/IBM-Plex-OFL.txt` contains the font license.

All durable application state lives in `AppState`. Raylib input is captured into an `InputFrame`, and `apply-input` is the single transition boundary.

## Create your first workflow

Start the app from the repository root:

```sh
coil build
./flowline
```

The catalog intentionally starts empty.

1. Click `+ NEW WORKFLOW` in the left sidebar or the center of the empty screen.
2. Use the workflow designer chat to describe a rough outcome, ask questions, and iterate. The designer can inspect this repository and recommend project-specific steps.
3. When the design is ready, click `BUILD THIS WORKFLOW`.
4. Wait for the creator status to change to `WORKFLOW CREATED`. The new workflow is selected automatically and appears in the sidebar.
5. Click `RUN WORKFLOW`. Select nodes to inspect their status and output, and approve explicit human-approval nodes when they pause.

Design and creation require the `codex` CLI to be installed and authenticated. The design conversation is recorded beneath `.flowline/`, and the creator turns that conversation into one project-specific program beneath `workflows/` only after you ask it to build.

## Workflow DSL

Generated programs use a line-oriented DSL supporting sequential nodes, parallel-ready branches, joins, and human approval:

```text
workflow Release Readiness
runner codex

agent build
title Build project
prompt Run the project build and report failures. Do not modify files.
after none

approval review
title Human review
after build

agent summary
title Prepare summary
prompt Read the build output and prepare the approved release summary.
after review
```

`after none` creates a root. Two dependency names separated by a comma create a join. Nodes that become ready together are launched concurrently.

Click `RUN WORKFLOW` to execute the selected program. Agent processes use `codex exec --ephemeral --sandbox workspace-write`. Prompts, event logs, completion sentinels, and final messages are written beneath `.flowline/`.

Inspect the parsed graph without opening the UI or running agents:

```sh
./flowline --check-catalog
./flowline --check-workflow workflows/my-workflow.flow
```
