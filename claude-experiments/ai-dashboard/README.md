# AI Dashboard

A CSS Zen Garden-style dashboard where AI agents generate complete visual styling and can contribute custom visualizations.

## Architecture

### Core Concepts

1. **Semantic Components** - Fixed HTML structure that never changes
2. **Style Agent** - Internal Claude-powered agent that generates complete CSS
3. **External Agents** - Out-of-process agents that can register custom components and data
4. **Theme Participation** - All components (built-in and custom) style through agent-generated CSS

### How It Works

```
User: "Make it look like art deco with gold"
  ↓
Style Agent generates complete CSS
  ↓
Same HTML, completely different visual presentation
```

## Getting Started

### Prerequisites

- Node.js 18+
- Claude subscription (NO API key needed!)

### Setup

1. **Install dependencies**

```bash
npm install
```

2. **Run the dashboard** (no API key needed)

```bash
npm run dev
```

This will:
- Start the Vite dev server (renderer)
- Compile and start Electron (main process)
- Start the WebSocket server for external agents (ws://localhost:3000)
- Wait for style agent to connect

3. **Run the style agent** (in a new terminal)

**NO API KEY NEEDED!**

```bash
cd agents/style-agent
npm install
npm run dev
```

### Using the Dashboard

1. **Generate a style** - Type a prompt like "cyberpunk with neon colors" and click "Generate Style"
2. **Try quick styles** - Click any of the quick style buttons
3. **Iterate** - Generate new styles to see the same components styled completely differently

### Example Prompts

- "Art Deco with gold accents"
- "Cyberpunk with neon colors"
- "Brutalist monochrome"
- "1960s NASA control room"
- "Vaporwave aesthetic"
- "Organic art nouveau"

## External Agents

### Running the Example Agent

The live metrics agent demonstrates external agent capabilities:

```bash
# In a new terminal
cd examples/live-metrics-agent
npm install
npm run dev
```

This agent will:
- Connect to the dashboard via WebSocket
- Register a custom "Network Activity" component
- Send live data updates every 2 seconds
- Respond to queries from the dashboard

### Creating Your Own Agent

1. **Install the SDK**

```bash
mkdir my-agent
cd my-agent
npm init -y
npm install @ai-dashboard/agent-sdk
```

2. **Create your agent**

```typescript
import { DashboardAgent } from '@ai-dashboard/agent-sdk';

const agent = new DashboardAgent({
  id: 'my-agent',
  name: 'My Custom Agent',
  connection: {
    type: 'websocket',
    url: 'ws://localhost:3000',
  },
});

await agent.connect();

// Register a component
await agent.registerComponent({
  id: 'my-viz',
  name: 'My Visualization',
  semantic: 'custom',
  code: `
    export default function MyViz({ data, theme }) {
      // Your React component
      // Has access to theme.colors, theme.fonts, etc.
      return <div>...</div>;
    }
  `,
  themeContract: {
    uses: ['colors.accent', 'fonts.body'],
    providesClasses: ['my-viz'],
  },
});

// Send data
await agent.updateData('my-data-source', { ... });
```

3. **The style agent will automatically style your component** based on the current theme!

## Project Structure

```
ai-dashboard/
├── src/                    # React app
│   ├── semantic/          # Semantic components (never change)
│   ├── engine/            # Theme engine, style provider
│   ├── store/             # State management
│   ├── ui/                # UI components
│   └── types/             # TypeScript types
│
├── electron/              # Electron main process
│   ├── main.ts           # Main entry point
│   ├── agents/           # Internal agents
│   │   └── style-agent.ts # Claude-powered style generator
│   └── bridge/           # External agent communication
│       └── agent-bridge.ts
│
├── agent-sdk/            # SDK for external agents
│   └── src/
│       └── DashboardAgent.ts
│
└── examples/             # Example external agents
    └── live-metrics-agent/
```

## How Theming Works

### Semantic HTML

Components always render the same HTML:

```tsx
<div className="metric-display">
  <span className="metric-label">Success Rate</span>
  <span className="metric-value">99.4%</span>
</div>
```

### Agent-Generated CSS

The style agent generates CSS for any aesthetic:

**Art Deco:**
```css
.metric-display {
  border: 3px solid #d4af37;
  background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
  font-family: 'Futura', sans-serif;
}
.metric-value {
  font-size: 4rem;
  color: #d4af37;
  text-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
}
```

**Cyberpunk:**
```css
.metric-display {
  border: 1px solid #00ffff;
  background: rgba(0, 0, 0, 0.8);
  font-family: 'Courier New', monospace;
  box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
}
.metric-value {
  font-size: 3rem;
  color: #ff00ff;
  text-shadow: 0 0 10px #ff00ff;
  animation: glitch 2s infinite;
}
```

Same HTML → Completely different visuals!

## Custom Components

External agents can provide custom React components that participate in theming:

```typescript
// Agent provides this code
await agent.registerComponent({
  id: 'neural-flow',
  code: `
    export default function NeuralFlow({ data, theme }) {
      // theme is injected automatically
      const nodeColor = theme.colors.accent;

      return (
        <svg className="neural-flow">
          {data.nodes.map(node => (
            <circle fill={nodeColor} ... />
          ))}
        </svg>
      );
    }
  `,
  themeContract: {
    uses: ['colors.accent', 'colors.secondary'],
    providesClasses: ['neural-flow', 'flow-node'],
  },
});
```

When the user requests a new style, the style agent:
1. Sees the `themeContract` from the custom component
2. Generates CSS for `.neural-flow` and `.flow-node`
3. Provides theme variables for `colors.accent` and `colors.secondary`
4. Custom component automatically uses the new theme!

## Development

```bash
# Run dashboard in dev mode
npm run dev

# Build for production
npm run build

# Type check
npm run typecheck
```

## Architecture Decisions

### Why External Style Agent?

- Uses Claude Agent SDK (no manual API management)
- Runs independently of dashboard
- Can be upgraded/replaced without touching dashboard
- Same architecture as other external agents
- Dashboard doesn't need API keys

### Why External Agents?

- Agents can run in any language
- Agents can run anywhere (local, remote, cloud)
- Sandboxed execution
- Easy to develop and test independently
- Clean separation of concerns

### Why CSS Zen Garden Approach?

- Separation of concerns (structure vs. presentation)
- Infinite visual variety from same HTML
- Agents can be creative without breaking functionality
- Educational - users can see generated CSS

## Future Ideas

- [ ] Component sandbox (isolated-vm)
- [ ] Style marketplace (save/share/load themes)
- [ ] Real-time collaboration
- [ ] Custom shader support (WebGL)
- [ ] Agent-generated animations
- [ ] Theme inheritance/mixing
- [ ] MCP protocol support
- [ ] HTTP/REST agent protocol

## License

MIT
