# Agent Harness Project Charter

This document is the durable context for every person and agent that works on this repository. Read it before designing, changing, or reviewing the system.

The project will evolve. Preserve the intent described here, but do not treat today's proposed abstractions as sacred. When experience disproves a design, update the design and this document together.

## What We Are Building

We are building a programmable agent execution platform in COIL.

It is not primarily a chat application, a thin wrapper around model APIs, or a collection of prompts. Its core is a headless runtime for long-lived, observable, steerable workflows composed from agents, tools, loops, branches, and graphs.

The platform must support:

- local execution on a laptop;
- remote execution on a server;
- multiple independent user interfaces, including a mobile client;
- synchronous, asynchronous, and background work;
- multiple model providers and model families;
- opinionated model selection based on capability, cost, latency, and task needs;
- tool execution with complete lifecycle visibility;
- durable workflows that can be inspected, paused, resumed, cancelled, and steered;
- communication and escalation among agents and from agents to a human;
- supervisor or meta-agents that observe other work and assess progress, quality, sentiment, suspicious shortcuts, and possible cheating;
- live inspection of every agent and every material action it takes;
- extension through small, composable libraries rather than framework lock-in.

We will own the orchestration and content model ourselves. Do not introduce an existing agent framework as the foundation of the system. Focused libraries for transport, encoding, persistence, cryptography, or other infrastructure may be appropriate, but their concepts must not become our domain model accidentally.

## Product Principles

### Headless first

The runtime must be fully usable without a graphical interface. A terminal client, web application, mobile application, automation, and tests must all interact with the same explicit application boundary. No essential behavior may live only in a UI.

### Observable by construction

Observability is part of execution semantics, not logging added later. Every meaningful state transition must produce structured information that can be persisted, streamed, queried, and rendered.

At minimum, the system must be able to represent:

- run creation, start, pause, resume, cancellation, completion, and failure;
- model request construction, dispatch, streaming progress, response, usage, and error;
- tool proposal, authorization, start, progress, result, timeout, cancellation, and error;
- agent creation, status, messages, delegation, escalation, and termination;
- workflow/node transitions and the reason each transition occurred;
- human input and steering;
- supervisor observations and judgments;
- cost, token usage, timing, provider, and model identity where available.

Prefer structured events over prose logs. Prose is useful for people; structured data is necessary for the system.

### Steerable, not merely autonomous

Long-running work must expose safe control points. A human or authorized supervisor must be able to inspect current state, provide information, change permitted direction, pause work, resume it, or cancel it. Do not design execution as an opaque function that disappears until it returns.

Steering must be explicit and recorded. It must not silently rewrite history.

### Durable execution

Important work must survive process boundaries and recoverable failures. Separate durable logical state from ephemeral in-process tasks. Design identifiers, events, checkpoints, and effects so that retries and recovery are possible.

Do not claim durability before restart and recovery behavior has been tested.

### Provider and model independence

Core workflow and agent logic must not depend on a provider-specific request or response type. Provider adapters translate between external APIs and a small internal model protocol.

Preserve provider-specific information when it is useful, but contain it at the boundary. Do not reduce all models to a false lowest common denominator: capabilities should be represented explicitly and selected intentionally.

### Opinionated model routing

The platform should eventually choose models intelligently using factors such as:

- required capabilities;
- expected reasoning difficulty;
- context and output limits;
- tool-use quality;
- latency;
- availability and rate limits;
- monetary or quota cost;
- prior measured performance;
- privacy and execution constraints.

A caller must also be able to pin a model when needed. Routing decisions must be observable and explainable. Begin with explicit policies; do not hide important choices behind unexplained heuristics.

### Composition over special cases

Loops, branches, retries, review, delegation, escalation, and supervision should emerge from a small set of composable runtime concepts. Avoid adding a new execution mechanism for every workflow shape.

Prefer a few orthogonal primitives with precise semantics over a large catalogue of convenient but overlapping features.

### Small pieces, clear boundaries

Build small libraries and modules that compose. Avoid god objects, central files that know every implementation, and modules that mix protocol translation, orchestration, persistence, and presentation.

No source file may grow beyond 4,000 lines. This is an emergency ceiling, not a target. Most files should be far smaller. Add an automated check early, and refactor before approaching the limit.

## Architectural Boundaries

Names will evolve, but the following responsibilities must remain distinguishable.

### Domain

The domain defines stable concepts and state transitions: runs, workflows, nodes, agents, messages, tool calls, model calls, events, policies, and outcomes. It must not import UI concerns or concrete provider clients.

### Runtime

The runtime advances work. It schedules runnable units, evaluates transitions, coordinates model and tool effects, handles cancellation, and records outcomes. Runtime behavior should operate against interfaces for time, identifiers, persistence, providers, and external effects so that it is deterministic under tests where practical.

### Model providers

Each provider adapter owns authentication, transport, wire formats, provider streaming, error translation, and usage extraction. It advertises capabilities rather than relying on scattered provider-name conditionals.

### Tools

A tool has an identity, description, input contract, output contract, execution policy, and implementation. Tool proposal is distinct from tool authorization, and authorization is distinct from execution.

Tools must support clear lifecycle states and cancellation where the underlying operation permits it. Side-effecting tools require special care around authorization, retries, and idempotency.

### Persistence

Persistence records durable state and history without deciding workflow policy. Storage representations must be versioned. Migrations and compatibility need deliberate treatment once persistent data exists.

### Transport/API

Remote clients interact through a versioned protocol. The protocol must support commands, queries, and event streaming without exposing in-process objects. Assume clients disconnect, reconnect, miss events, and submit duplicate requests.

### User interfaces

UIs render state and issue commands. They do not own canonical workflow state or secretly perform orchestration. Different UIs should be interchangeable views over the same runtime capabilities.

### Supervision and evaluation

Supervisors consume the same observable execution data available through supported interfaces. They produce recorded assessments or commands. They must not gain hidden, uninspectable mutation paths merely because they are implemented inside the system.

## Dependency Rules

Dependencies point inward toward stable concepts.

- Domain code must not depend on providers, databases, network transports, or UIs.
- Provider adapters must not decide workflow progression.
- Tools must not reach into scheduler internals.
- Persistence must not contain product orchestration policy.
- UIs must not import runtime internals as a substitute for an API.
- Cross-cutting capabilities should use explicit interfaces and events, not global mutable state.
- Provider-specific branching belongs in a provider adapter or capability policy, not throughout the codebase.
- Convenience APIs may wrap core primitives, but must not create a second execution model.

If a change violates one of these rules, document why before implementing it. Repeated exceptions indicate that the boundary or the implementation is wrong and must be reconsidered.

## Execution Model Requirements

Model and tool operations are effects with explicit lifecycles. At a conceptual level, the first complete vertical slice is:

```text
task
  -> model request
  -> streamed model output
  -> optional tool proposal
  -> authorization
  -> asynchronous tool execution
  -> tool result
  -> model continuation
  -> completion, failure, or cancellation
```

Every arrow must be observable. Every long-running step must have a stable identity. Terminal states must be explicit.

As the graph system develops:

- node inputs and outputs must have defined contracts;
- transition conditions must be explicit and inspectable;
- loops must declare termination or budget policy;
- retries must be bounded or governed by an explicit policy;
- cancellation must propagate predictably;
- parallel branches must define joining and failure behavior;
- failures must remain distinguishable from cancellations and policy rejections;
- graph definitions must be separate from individual run state;
- changes to a live workflow must be versioned and recorded;
- replay must never repeat external side effects accidentally.

Avoid recursion disguised as orchestration without budgets or termination rules. The runtime must be able to explain why a unit is running and what can cause it to stop.

## Event and State Discipline

Events are immutable facts. Commands are requests to change the world. State is a derived or persisted view of what is currently true. Keep these concepts separate.

Each event should normally include:

- a globally unique event identifier;
- event type and schema version;
- timestamp from an injectable clock;
- run and workflow identifiers where applicable;
- actor or origin;
- causation identifier;
- correlation identifier;
- structured payload;
- sequence or ordering information appropriate to its scope.

Do not place secrets, credentials, or unnecessary sensitive model context in events. Redaction is an explicit boundary responsibility, not a UI-only feature.

Use explicit state machines for lifecycles. Reject invalid transitions rather than repairing them silently. Preserve enough information to diagnose how a state was reached.

Ordering in a distributed system is scoped, not magical. Specify which stream or aggregate is ordered and design concurrent operations consciously.

## Asynchrony, Cancellation, and Backpressure

Asynchronous work must be structured and owned. Every spawned task needs:

- an owner or parent scope;
- a stable logical operation identifier;
- a completion path;
- error propagation behavior;
- cancellation behavior;
- resource and concurrency limits;
- observable status.

Do not create detached background work that can fail invisibly. Do not use unbounded queues or unbounded agent creation. Streaming consumers must have a backpressure, buffering, coalescing, or disconnect policy.

Cancellation is a normal outcome, not an exceptional afterthought. Cancellation requests and confirmed cancellation are different facts. External operations may complete after cancellation is requested; define how late results are recorded and whether they may affect the run.

## Tools and Effects

Tool execution is a security and correctness boundary.

- Validate tool inputs against a declared contract before execution.
- Record the exact authorized invocation.
- Separate read-only, reversible, and destructive effects.
- Make authorization policies composable and inspectable.
- Use idempotency keys where repeated requests could duplicate effects.
- Never retry a side effect blindly.
- Bound execution with timeouts, cancellation, and output limits.
- Treat tool output as untrusted input.
- Preserve stdout, stderr, exit status, timing, and structured results when relevant.
- Make secrets available only to the narrowest execution boundary that needs them.
- Avoid leaking secrets into prompts, logs, events, errors, or UI payloads.

Sandboxing and remote execution must be capabilities of the execution layer, not assumptions embedded in individual tools.

## Agent Communication and Supervision

Agent communication must use explicit, typed messages with identities and traceable causation. An agent may delegate work, request review, report a blocker, or escalate to a human. These are domain actions, not informal conventions hidden only in prompt text.

Supervisors may evaluate qualities such as:

- progress toward the requested outcome;
- unsupported claims;
- suspicious shortcuts or reward hacking;
- ignored requirements;
- ineffective repetition;
- unsafe actions;
- uncertainty, frustration, or other useful sentiment signals;
- whether human intervention is warranted.

Supervisor conclusions are assessments, not unquestionable truth. Store their evidence, confidence, model or policy identity, and resulting action. Critical interventions should be reviewable.

## COIL Engineering Practices

COIL is a language we control, and it may not yet provide every facility this project needs. Add missing capabilities deliberately as small, reusable libraries. Do not compensate for missing language features with opaque application-level machinery.

When work exposes a COIL limitation:

1. Identify whether it belongs in the language, standard library, runtime support, or this application.
2. Choose the smallest layer that gives the concept a clean, reusable home.
3. Specify behavior with focused tests before depending on it broadly.
4. Document limitations and failure semantics.
5. Keep application progress incremental; do not redesign the language casually during feature work.

Prefer explicit data and transformations. Isolate mutation. Avoid hidden dynamic scope, ambient configuration, and implicit control flow unless COIL semantics make them unavoidable and they are clearly documented.

## Code Quality Rules

### Size and cohesion

- Keep modules focused on one responsibility.
- Treat 4,000 lines as a hard maximum for any source file.
- Add CI or a repository check that rejects source files over the limit.
- Refactor based on conceptual boundaries, not arbitrary numbered fragments.
- Do not create generic `utils` dumping grounds. Name modules after the concept they own.

### Public contracts

- Document exported functions, types, protocols, and non-obvious invariants.
- State error, cancellation, ownership, and concurrency behavior.
- Prefer narrow interfaces shaped by consumers.
- Version persisted formats and remote protocols.
- Avoid breaking contracts casually; when breaking them early in development, change call sites, tests, examples, and documentation together.

### Error handling

- Use structured errors with stable categories where callers need to react.
- Preserve underlying causes and useful context.
- Never swallow failures from background work.
- Do not use generic retries as error handling.
- Distinguish invalid input, policy rejection, provider failure, timeout, cancellation, and internal defect.

### Configuration

- Configuration must be explicit, validated, and inspectable.
- Keep secrets separate from ordinary configuration.
- Do not let environment-variable reads spread through domain code.
- Record relevant non-secret configuration with a run so its behavior can be understood later.

### Comments and documentation

- Explain intent, invariants, and surprising tradeoffs, not syntax.
- Keep architectural documentation close to the code it governs.
- Update documentation in the same change as behavior.
- Include small examples for public abstractions whose use is not obvious.

## Testing Strategy

Testing is part of the architecture. Favor deterministic tests and controllable boundaries.

Every substantive feature should include the appropriate mix of:

- unit tests for pure domain behavior and state transitions;
- contract tests shared by all model providers, storage implementations, and tool executors;
- integration tests across real boundaries;
- end-to-end tests for critical execution paths;
- restart and recovery tests for durable behavior;
- concurrency tests for cancellation, races, ordering, and backpressure;
- failure-injection tests for timeouts, malformed streams, partial writes, disconnects, and duplicate delivery;
- replay tests that prove external effects are not duplicated;
- serialization compatibility tests for durable and remote formats.

Model-dependent tests must not rely only on live model behavior. Use deterministic fake providers for semantics and a smaller, clearly marked suite for real provider compatibility. Live tests must have cost and rate limits.

Tests should assert observable behavior and invariants, not private implementation trivia.

## Security and Privacy

Assume prompts, model outputs, tool outputs, remote messages, and persisted artifacts may contain hostile or sensitive content.

- Authenticate remote clients and authorize commands by capability and resource.
- Use least privilege for agents, tools, providers, and supervisors.
- Treat model output as data, never authority.
- Make approval requirements explicit for consequential effects.
- Redact secrets before persistence and transmission.
- Define retention and deletion behavior before storing sensitive histories broadly.
- Audit security-relevant actions with actor and causation.
- Bound resource consumption: tokens, money, time, storage, processes, network, agents, and retries.
- Fail closed when an authorization decision cannot be made safely.

Do not market monitoring, sandboxing, isolation, or durability guarantees that the implementation and tests do not yet provide.

## Model Economics and Budgets

Cost is a runtime concern and a product feature. Represent budgets directly rather than relying on prompt reminders.

Runs and sub-runs should be able to carry limits for:

- monetary spend;
- input and output tokens;
- wall-clock time;
- number of model calls;
- number and type of tool calls;
- retries;
- concurrent workers;
- recursion or delegation depth.

Budget consumption and estimates must be visible. When a provider does not report exact cost, label estimates as estimates. A routing policy must not trade away a required capability merely to select a cheaper model.

## Development Workflow

Build the system through thin, working vertical slices. The first target is one task that can call a model, stream output, execute an authorized asynchronous tool, continue the model turn, and expose the entire lifecycle as structured events.

For each change:

1. Read this charter and nearby documentation.
2. Inspect existing abstractions before creating new ones.
3. State the behavior and invariants being added or changed.
4. Implement the smallest coherent slice.
5. Add tests, including failure paths where relevant.
6. Run formatting, static checks, tests, and the file-size check.
7. Review dependency direction and public surface area.
8. Update documentation and examples.
9. Report what was verified and what remains uncertain.

Do not combine unrelated refactors with feature work. Do not leave duplicate old and new paths indefinitely. Temporary compatibility code must have a reason and a removal condition.

## Decision Records

Record architectural decisions that are costly to reverse or easy to misunderstand. A decision record should include:

- context and forces;
- the decision;
- alternatives considered;
- consequences and tradeoffs;
- current status;
- conditions that would justify revisiting it.

Decisions we expect to record early include the event model, persistence strategy, workflow representation, provider protocol, remote transport, tool authorization, identity model, and recovery semantics.

## Definition of Done

A change is not complete merely because its happy path runs. It is complete when:

- its responsibility belongs in the chosen module;
- public behavior and invariants are clear;
- relevant success, failure, cancellation, and concurrency paths are tested;
- actions and transitions are observable;
- costs and resources are bounded where applicable;
- security and secret-handling implications are addressed;
- docs and examples match the implementation;
- formatting, tests, static checks, and size checks pass;
- no essential logic has leaked into a particular UI or provider;
- remaining limitations are explicit rather than concealed.

## Guidance for Future Agents

Before writing code, determine which layer owns the behavior and which event makes it observable. Before adding an abstraction, find at least two real pressures that require it; otherwise prefer the direct implementation. Before adding a dependency, identify the boundary it serves and the domain concepts it must not own.

Never optimize solely for a convincing demo. Preserve the path from today's small implementation to a durable, remote, multi-client runtime. At the same time, do not build speculative infrastructure for distant possibilities. Establish one sound seam at a time and exercise it end to end.

When requirements are unclear, preserve information and reversibility. Use stable identities, explicit policies, versioned data, narrow interfaces, and recorded decisions. Surface uncertainty instead of silently inventing guarantees.

The standard for this project is not that an agent eventually produced an answer. The standard is that the system can show what happened, why it happened, what it cost, what may happen next, and how an authorized person or agent can intervene.
