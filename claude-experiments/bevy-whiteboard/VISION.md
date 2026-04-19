# Living Whiteboard — Vision

## User's Vision (verbatim)

> Okay, we are going to be building with bevy something a bit unique. We want a white board. But we want it to come to life. Think factorio, think a simulator. We are going to let you sketch a real system and see how it would operate. SO for example, you might be able to make a little load balancer setup. Data would flow in and then get distributed to the worker nodes. You'd be able to put probes on lines to see the flow rate. You'd be able to be able to simulate traffic coming in.
>
> From the beginning I want a little floating panel of tools on the right above the canvas. THen I want the ability to draw out these things. Have things that generate date and things that capture data.
>
> We need colors to be able to stand for different kinds of data.

## Claude's Understanding

This is a **live systems sketching tool** — part whiteboard, part discrete-event
simulator. The user draws a system (boxes, arrows) and it becomes executable:
packets travel visibly along the edges so you can *see* how the architecture
behaves under load.

### Core mental model

- **Nodes** are components of a system. At minimum:
  - **Generators / sources** — emit data (configurable rate, data type).
  - **Processors / workers** — consume input, optionally transform/delay, emit output.
  - **Sinks / captures** — terminal collectors (log, count, display).
  - **Routers** (e.g. a load balancer) — split traffic across downstream nodes by some policy.
- **Edges** connect nodes and carry packets. Packets are rendered as moving
  dots/shapes traveling along the edge so flow is visually obvious.
- **Probes** attach to an edge (or node) and display measurements —
  flow rate (pkts/s), latency, queue depth, etc. — live.
- **Color = data type.** A generator emits packets of a chosen color; a worker
  might only accept certain colors, or transform one color into another. This
  gives us typed flows without needing a text UI.

### UI shape (starting point)

- Infinite (or large) **canvas** that fills the window.
- **Floating tool palette** on the right, above the canvas (overlay, not a sidebar
  that steals canvas space). Tool buttons: select, generator, worker, sink, router,
  edge-draw, probe, color picker, play/pause.
- **Click-drag** to place nodes; **click node → click node** to draw an edge.
- Simulation is always running by default (play/pause toggle). Speed slider later.

### Inspirations

- **Factorio** — items physically travel on belts; you debug visually.
- **tldraw / Excalidraw** — lightweight infinite canvas feel.
- **Node-based tools** (Blender shader editor, Unreal Blueprint) — typed pin semantics.
- **Discrete-event simulators** (SimPy, OMNeT++) — the underlying model.

### Open questions (defer, don't block)

- Persistence / save-load format.
- How rich the policy language for routers becomes (round-robin → weighted → scripted).
- Whether probes can drive charts/graphs over time, not just an instantaneous number.
- Zoom / pan.

## Build plan (incremental)

1. **Bevy window + empty canvas** with clear background and camera pan/zoom.
2. **Floating tool palette** (bevy_egui overlay on the right).
3. **Place nodes** of different kinds from the palette.
4. **Draw edges** between nodes.
5. **Simulation tick**: generators emit packets; packets travel along edges and
   get consumed at sinks. Render packets as moving colored shapes.
6. **Router node** with round-robin split.
7. **Probes** on edges showing flow rate.
8. **Color as type** — generator color picker; worker/sink color filters.
