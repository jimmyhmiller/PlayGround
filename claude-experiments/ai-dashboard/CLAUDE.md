# AI Dashboard - Project Context for Claude

## What This Is

An experimental dashboard application that demonstrates the **CSS Zen Garden** approach applied to modern AI-powered interfaces. The core concept: semantic HTML structure that never changes, while AI agents generate complete CSS stylesheets to create radically different visual presentations.

Think of it as "separation of concerns taken to the extreme" - structure and presentation are completely decoupled, allowing AI to be infinitely creative with styling without breaking functionality.

## Current State

**Status**: Early prototype / proof of concept (v0.1.0)

**What's Working**:
- ✅ Electron + React + Vite setup
- ✅ Internal style agent using Claude Agent SDK
- ✅ Style generation via Claude (uses user's Claude subscription, no API key needed)
- ✅ CSS injection and hot-swapping
- ✅ Semantic component library (Layout, Widgets, Metrics, Charts, Status)
- ✅ WebSocket bridge for external agents
- ✅ Example external agent (live-metrics-agent)
- ✅ Agent SDK for building custom agents
- ✅ Theme state management (Zustand)

**What's Experimental**:
- Custom component registration from external agents (architecture defined, implementation may be incomplete)
- Theme contract system for custom components
- Component sandboxing (planned but not implemented)
- Style persistence/marketplace (future idea)

**What's Not Implemented Yet**:
- Component sandbox (isolated-vm)
- Style marketplace (save/share/load themes)
- Real-time collaboration
- Custom shader support (WebGL)
- MCP protocol support
- HTTP/REST agent protocol

## Architecture Overview

### Three-Tier Architecture

```
┌─────────────────────────────────────┐
│   Dashboard (Electron + React)      │
│   - Semantic components (HTML)      │
│   - Style provider (CSS injection)  │
│   - WebSocket server                │
└─────────────────────────────────────┘
           ↕
┌─────────────────────────────────────┐
│   Internal Style Agent (Electron)   │
│   - Claude Agent SDK                │
│   - CSS generation                  │
│   - Theme iteration                 │
└─────────────────────────────────────┘
           ↕
┌─────────────────────────────────────┐
│   External Agents (Optional)        │
│   - Custom visualizations           │
│   - Live data sources               │
│   - Theme participation             │
└─────────────────────────────────────┘
```

### Key Design Decisions

**1. External Style Agent**
- The style agent runs as a separate process using the Claude Agent SDK
- Uses the user's Claude subscription (no API keys in dashboard)
- Can be upgraded/replaced independently
- Same architecture pattern as external data agents

**2. Semantic Components**
- Components always render identical HTML structure
- Only CSS class names and data attributes
- No inline styles, no dynamic styling logic
- This constraint enables infinite visual variety

**3. CSS Zen Garden Approach**
- Complete separation of structure and presentation
- AI generates entire stylesheets from scratch
- Same HTML can look like anything (art deco, cyberpunk, brutalist, etc.)
- Educational - users can inspect generated CSS

**4. External Agent Protocol**
- Agents connect via WebSocket
- Can register custom React components
- Participate in theming via "theme contracts"
- Sandboxed execution (planned)

## Project Structure

```
ai-dashboard/
├── src/                    # React dashboard application
│   ├── semantic/          # Semantic component library
│   │   ├── Layout.tsx     # .dashboard-root, .sidebar, .main-content
│   │   ├── WidgetContainer.tsx
│   │   ├── MetricDisplay.tsx
│   │   ├── DataSeries.tsx
│   │   └── StatusItem.tsx
│   ├── engine/            # Theme engine and style injection
│   │   └── StyleProvider.tsx
│   ├── store/             # State management (Zustand)
│   │   └── useThemeStore.ts
│   ├── ui/                # UI controls (style control panel)
│   └── types/             # TypeScript type definitions
│
├── electron/              # Electron main process
│   ├── main.ts           # Entry point, window management
│   ├── agents/           # Internal agents
│   │   └── style-agent.ts # Claude-powered CSS generator
│   ├── bridge/           # External agent communication
│   │   └── agent-bridge.ts # WebSocket server
│   └── preload.ts        # IPC bridge
│
├── agents/               # External agent implementations
│   └── style-agent/     # (May be deprecated - now in electron/agents/)
│
├── agent-sdk/           # SDK for building external agents
│   └── src/
│       ├── DashboardAgent.ts
│       └── index.ts
│
└── examples/            # Example external agents
    └── live-metrics-agent/ # Demo agent with custom component
```

## Core Concepts

### 1. Semantic Components

All dashboard components use semantic class names and render fixed HTML:

```tsx
// This NEVER changes
<div className="metric-display">
  <span className="metric-label">Success Rate</span>
  <span className="metric-value">99.4</span>
  <span className="metric-unit">%</span>
</div>
```

### 2. Style Generation

The style agent receives prompts like:
- "Art Deco with gold accents"
- "Cyberpunk with neon colors"
- "1960s NASA control room"

And generates complete CSS stylesheets that target the semantic class names.

### 3. Style Iteration

Users can iterate on existing themes:
- "Make it darker"
- "Add more glow effects"
- "Use a different font"

The agent receives context about the current theme and modifies it.

### 4. Custom Components (Experimental)

External agents can register React components:

```typescript
await agent.registerComponent({
  id: 'neural-flow',
  name: 'Neural Network Flow',
  semantic: 'custom',
  code: `export default function NeuralFlow({ data, theme }) { ... }`,
  themeContract: {
    uses: ['colors.accent', 'colors.secondary'],
    providesClasses: ['neural-flow', 'flow-node']
  }
});
```

The style agent sees the theme contract and generates CSS for those classes.

## Development Workflow

### Running the Dashboard

```bash
# Terminal 1: Start dashboard (no API key needed)
npm run dev

# Terminal 2: External agents can connect if needed
cd examples/live-metrics-agent
npm run dev
```

The dashboard includes the internal style agent (electron/agents/style-agent.ts) which uses the Claude Agent SDK.

### Style Generation Flow

1. User enters prompt in UI
2. UI sends request to Electron main process via IPC
3. Main process invokes StyleAgent.generateStyle()
4. StyleAgent uses Claude Agent SDK to query Claude
5. Claude generates CSS code
6. CSS is cleaned up and returned
7. Main process sends CSS back to renderer
8. StyleProvider injects CSS into document
9. Visual transformation happens instantly

### Agent SDK Usage

External agents can connect and:
- Register custom visualization components
- Send data updates
- Respond to queries
- Participate in theming

See `examples/live-metrics-agent/` for a working example.

## Technology Stack

- **Frontend**: React 18, TypeScript, Vite
- **Desktop**: Electron 28
- **AI**: Claude Agent SDK (@anthropic-ai/claude-agent-sdk)
- **State**: Zustand
- **Communication**: WebSocket (ws library)
- **Styling**: Pure CSS (AI-generated)

## Known Limitations

1. **No component sandboxing yet** - Custom components run in main React context
2. **No style persistence** - Generated styles are ephemeral
3. **No error recovery** - If style generation fails, need to manually retry
4. **Single user** - No collaboration features
5. **Memory only** - No database, everything in-memory

## Future Exploration Ideas

From the README:

- Component sandbox using isolated-vm
- Style marketplace (save/share/load themes)
- Real-time collaboration between users
- Custom shader support (WebGL/GLSL)
- Agent-generated animations and transitions
- Theme inheritance/mixing
- MCP (Model Context Protocol) support
- HTTP/REST agent protocol (alternative to WebSocket)

## Development Notes

### Style Agent Architecture

There are currently TWO style agents in the codebase:
1. `electron/agents/style-agent.ts` - The **active** internal agent used by the dashboard
2. `agents/style-agent/` - May be an earlier external agent prototype or example

The internal agent (in electron/) is the one actually used by the dashboard.

### Claude Agent SDK Integration

The style agent uses a clever dynamic import trick to load the ESM-only Claude Agent SDK:

```typescript
const importFunc = new Function('specifier', 'return import(specifier)');
const sdk = await importFunc('@anthropic-ai/claude-agent-sdk');
this.query = sdk.query;
```

This bypasses TypeScript transpilation issues with ESM modules.

### CSS Generation Prompt

The prompt sent to Claude is in `electron/agents/style-agent.ts` and includes:
- User's aesthetic request
- Optional current theme context for iteration
- List of all semantic component classes
- Requirements (modern CSS, readability, creativity)
- Strict format requirement (CSS only, no explanations)

### Adding New Semantic Components

When adding new components:
1. Create the React component in `src/semantic/`
2. Use only semantic class names
3. Update the style generation prompt in `style-agent.ts`
4. Export from `src/semantic/index.ts`
5. No TypeScript updates needed if using existing types

## Maintenance Guidelines

### When Modifying Components

- **NEVER** add inline styles or CSS-in-JS to semantic components
- **NEVER** change HTML structure based on theme/style
- **ALWAYS** use semantic, descriptive class names
- **ALWAYS** update style generation prompt when adding classes

### When Modifying the Style Agent

- Keep prompt concise but comprehensive
- Include all semantic classes
- Specify requirements clearly
- Test with multiple aesthetic prompts

### When Adding External Agent Features

- Update agent-sdk types
- Update WebSocket message protocol
- Test with example agent
- Document in README

## Questions for Future Development

1. Should custom components be sandboxed? (Security vs. simplicity trade-off)
2. How to handle style versioning and compatibility?
3. Should there be a "base" theme that agents extend?
4. How to handle responsive design in generated CSS?
5. Should agents be able to generate SVG defs for patterns/gradients?
6. What's the migration path from WebSocket to MCP?

## Debugging Tips

- **Style generation fails**: Check Electron console for error messages
- **WebSocket connection issues**: Verify port 3000 is available, check logs
- **CSS not applying**: Inspect generated CSS in DevTools, check for syntax errors
- **Agent SDK errors**: Verify Claude subscription is active
- **TypeScript errors**: Run `npm run typecheck` to see full error output

## Related Projects & Inspiration

- CSS Zen Garden (original inspiration)
- @anthropic-ai/claude-agent-sdk
- Electron + Vite + React template
- WebSocket-based agent architectures

---

**Last Updated**: 2025-11-19
**Version**: 0.1.0
**Status**: Early prototype / experimental
