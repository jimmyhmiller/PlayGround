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

## Workflow DSL

The included program demonstrates sequential nodes, parallel-ready branches, a join, and a human approval:

```text
workflow refund-resolver
runner codex

agent policy
title Policy lookup
prompt Read the request and policy, then decide eligibility.
after route

agent compose
title Draft resolution
prompt Compose the final decision and customer response.
after policy, history

approval review
title Human approval
after compose
```

`after none` creates a root. Two dependency names separated by a comma create a join. Nodes that become ready together are launched concurrently.

Click `+ NEW WORKFLOW`, describe what the workflow should accomplish in this project, and press Enter or `CREATE WORKFLOW`. A creator agent inspects the repository, writes a new DSL program under `workflows/`, and the interface refreshes the project catalog when creation completes.

Click `RUN WORKFLOW` to execute the selected program. Agent processes use `codex exec --ephemeral --sandbox workspace-write`. Prompts, event logs, completion sentinels, and final messages are written beneath `.flowline/`.

Inspect the parsed graph without opening the UI or running agents:

```sh
./flowline --check-workflow
./flowline --check-catalog
```
