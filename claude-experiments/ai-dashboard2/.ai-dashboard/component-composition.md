# Component Composition Ideas

## Core Concept

Enable components to compose together by allowing outputs from one (or multiple) components to flow into inputs of others. The composed result appears as a single component to the user, but internally leverages the power of multiple specialized components.

## Example: Test Results Composition

**Current State**: We have separate `commandRunner` and `testResults` widgets.

**Composed Vision**:
```
commandRunner("npm test")
  → transformer(parseTestOutput)
  → testResults(formatted data)
```

The user sees only `testResults`, but it's powered by composition under the hood.

## Key Questions to Explore

### 1. Component Communication Models

- **Point-to-point**: Component A → Component B (explicit wiring)
- **Broadcast/Subscribe**: Component publishes to channel, others subscribe
- **Local vs Global**: Should composition be dashboard-scoped or cross-dashboard?
- **Hybrid**: Mix of explicit pipes and broadcast channels?

**Questions:**
- Do we need both local (component-to-component) and global (broadcast) patterns?
- Should components declare their input/output "ports" explicitly?
- How do we handle multiple subscribers to one output?

### 2. Data Transformation Layer

- **Inline transformers**: Small JavaScript/WASM functions to reshape data
- **Transformer components**: Dedicated components that only transform data
- **Built-in transformers**: Common patterns (JSON parse, regex extract, format)

**Questions:**
- Should transformers be components themselves, or a special "pipe" concept?
- Do we need a visual pipeline editor to show data flow?
- How do we handle transformation errors?
- Should transformations be declarative (config) or imperative (code)?

### 3. Visibility & Debugging

**Questions:**
- When components are composed, can users "peek inside" to see intermediate states?
- Should there be a debug mode that expands composition into visible stages?
- How do we visualize the data flow graph?
- Can users override/inject data at intermediate steps for testing?

### 4. Component Interface Design

**Output Types:**
- Single output vs multiple named outputs?
- Typed outputs (JSON, string, binary, stream)?
- Real-time streams vs one-time values?

**Input Types:**
- Required vs optional inputs?
- Default values when no input connected?
- Can a component work standalone OR composed?

**Questions:**
- Do components need to declare schemas for their inputs/outputs?
- Should we support streaming/reactive data flows (component B updates when A changes)?
- How do we handle backpressure (slow consumer, fast producer)?

### 5. Composition Patterns to Support

#### Pattern: Command → Parser → Display
```
commandRunner("cargo test")
  → parseCargoTest()
  → testResults
```

#### Pattern: Multiple Sources → Aggregator → Display
```
commandRunner("jest") ─┐
commandRunner("pytest") → combineResults() → testResults
commandRunner("cargo") ─┘
```

#### Pattern: Single Source → Multiple Consumers
```
              ┌→ testResults
commandRunner → transformer → barChart
              └→ stat (test count)
```

#### Pattern: Feedback Loop
```
userInput → commandRunner → validator ──→ output
              ↑                         │
              └─────── (retry) ─────────┘
```

**Questions:**
- Which patterns should we prioritize first?
- Do we need special handling for loops/cycles?
- Should we support conditional flows (if-then-else)?

### 6. User Experience

**Configuration:**
- Visual graph editor vs JSON config?
- Templates for common compositions?
- AI assistance to suggest compositions?

**Questions:**
- How does a user create a composition? Drag connections? Edit JSON?
- Can the AI agent create compositions automatically?
- Should composed components be saveable as reusable templates?
- Do users need to understand the composition, or can it be fully abstracted?

### 7. Technical Architecture

**Implementation Approaches:**
- React context for data sharing?
- Event bus system?
- State management library (Redux, MobX, Zustand)?
- Custom message-passing system?

**Questions:**
- How do we handle component lifecycle (mount/unmount during composition)?
- Should composition be defined in widget config or separate composition config?
- Do we need dependency resolution (A requires B's output, so B runs first)?
- How do we handle async operations in the pipeline?

### 8. Advanced Scenarios

**Real-world Use Cases:**
```
# Live monitoring pipeline
commandRunner("docker stats")
  → parseDockerStats()
  → [barChart(CPU), stat(Memory), progress(Disk)]

# Multi-step build pipeline
commandRunner("npm build")
  → checkExitCode()
  → commandRunner("npm test")
  → testResults

# Data aggregation
fileList(./src) ─┐
gitStatus() ─────→ aggregateProjectHealth() → progress
todoList() ──────┘
```

**Questions:**
- Should we support sequential execution (A then B then C)?
- How do we handle error propagation through the pipeline?
- Can compositions be dynamic (add/remove stages based on runtime conditions)?

### 9. Configuration Schema Example

```json
{
  "id": "composed-test-results",
  "type": "composition",
  "label": "Test Results",
  "components": [
    {
      "id": "runner",
      "type": "commandRunner",
      "config": { "command": "npm test" },
      "outputs": { "stdout": "raw-output" }
    },
    {
      "id": "parser",
      "type": "transformer",
      "function": "parseJestOutput",
      "inputs": { "text": "runner.stdout" },
      "outputs": { "tests": "parsed-tests" }
    },
    {
      "id": "display",
      "type": "testResults",
      "inputs": { "data": "parser.tests" },
      "visible": true
    }
  ],
  "x": 0, "y": 0, "width": 400, "height": 300
}
```

**Questions:**
- Is this config too complex for users to hand-write?
- Should "composition" be a widget type, or a property of any widget?
- How do we reference outputs? String paths? Typed references?

### 10. First Steps

**Minimal Viable Composition:**

1. Pick one simple pattern (command → transform → display)
2. Hard-code the plumbing for that specific case
3. Validate the UX and performance
4. Generalize from lessons learned

**Prototype Ideas:**
- Start with `commandRunner` → `testResults` composition
- Use props/callbacks for now (no fancy event system yet)
- Allow AI agent to generate the transformer function
- Make the composition invisible to user (just works™)

**Questions to Answer First:**
- What's the simplest possible version we can ship?
- Which composition pattern would provide the most immediate value?
- Should we build the infrastructure first, or start with hard-coded compositions?

---

## Next Actions

- [ ] Pick one composition pattern to prototype
- [ ] Design the minimal API for component outputs/inputs
- [ ] Decide on visible vs hidden composition UX
- [ ] Build proof-of-concept with two components
- [ ] Test with real use case (test results?)
- [ ] Gather feedback and iterate

## Related Ideas

- **Component marketplace**: Share composed components as templates
- **AI-assisted composition**: "Create a widget that shows my test coverage over time"
- **Live editing**: Modify composition while it's running
- **Time-travel debugging**: Replay data through the composition pipeline
- **Performance**: Caching, memoization, lazy evaluation in pipelines
