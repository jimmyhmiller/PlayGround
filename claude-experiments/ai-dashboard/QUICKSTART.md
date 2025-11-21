# Quick Start Guide

## Setup (5 minutes)

### 1. Run the dashboard

**No API key needed for the dashboard!**

```bash
npm run dev
```

This will:
- Start Vite dev server on http://localhost:5173
- Launch Electron window with the dashboard
- Start WebSocket server for external agents (auto-detects available port, starts at 3000)
- Print the WebSocket URL agents should connect to
- Wait for style agent to connect

**Check the console output** to see which port the bridge is using!

### 2. Run the style agent (in a new terminal)

**NO API KEY NEEDED** - Uses your Claude subscription!

```bash
cd agents/style-agent
npm install

# Optional: Set dashboard port if it's not 3000
# export DASHBOARD_PORT=3001

# Run the agent
npm run dev
```

**Note:** Check the dashboard console to see which port the bridge is using, and set `DASHBOARD_PORT` if needed.

The style agent will connect to the dashboard and be ready to generate styles.

### 3. Try generating a style

In the dashboard window:

1. Type a prompt like **"Art Deco with gold accents"**
2. Click **"Generate Style"**
3. Watch as the style agent generates complete CSS
4. See the same components styled completely differently!

### 4. (Optional) Run the example external agent

In a new terminal:

```bash
cd examples/live-metrics-agent
npm install
npm run dev
```

This agent will:
- Connect to the dashboard via WebSocket
- Register a custom "Network Activity" component
- Send live data updates every 2 seconds

## What to Try

### Style Prompts

Try these different aesthetics:

- "Cyberpunk with neon colors"
- "Brutalist monochrome"
- "1960s NASA control room"
- "Vaporwave aesthetic"
- "Organic art nouveau"
- "Swiss design, minimal"
- "Memphis design, bold colors"
- "Terminal from Blade Runner"

### Iteration

After generating a style, you can iterate:

- "Make it darker"
- "Add more color"
- "Use a different font"
- "Make it more minimal"
- "Add glow effects"

## Architecture

```
┌─────────────────────────────────────┐
│   Electron Dashboard (localhost)    │
│                                     │
│   - React UI (semantic components) │
│   - Style Agent (Claude SDK)       │
│   - WebSocket Server (:3000)       │
└─────────────────────────────────────┘
           ↕
┌─────────────────────────────────────┐
│   External Agents (optional)        │
│   - Live Metrics Agent              │
│   - Your Custom Agents              │
└─────────────────────────────────────┘
```

## How It Works

### 1. Semantic HTML (Never Changes)

Components always render the same structure:

```tsx
<div className="metric-display">
  <span className="metric-label">Success Rate</span>
  <span className="metric-value">99.4%</span>
</div>
```

### 2. Agent Generates CSS

Claude receives your prompt and generates complete CSS:

```css
.metric-display {
  border: 3px solid #d4af37;
  background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
  font-family: 'Futura', sans-serif;
  padding: 2rem;
}

.metric-value {
  font-size: 4rem;
  color: #d4af37;
  text-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
}
```

### 3. CSS Injected → Visual Transformation

Same HTML + Different CSS = Completely Different Look!

## Troubleshooting

### Dashboard doesn't start

- Check that you have Node.js 18+ installed
- Make sure port 5173 is not in use
- Check the terminal for error messages

### Style generation fails

- Verify your `ANTHROPIC_API_KEY` is set correctly in `.env`
- Check your API key has credits
- Look for error messages in the Electron console

### External agent won't connect

- Make sure the dashboard is running first
- Check that port 3000 is not in use
- Verify the agent is trying to connect to `ws://localhost:3000`

## Next Steps

1. **Experiment with styles** - Try wildly different aesthetics
2. **Build a custom agent** - Use the agent SDK to create your own visualizations
3. **Explore the code** - See how the style agent generates CSS
4. **Share your themes** - Export generated CSS and share with others

## Development

### Project Structure

```
ai-dashboard/
├── src/                    # React app
│   ├── semantic/          # Semantic components
│   ├── engine/            # Theme engine
│   └── ui/                # UI components
├── electron/              # Electron main
│   ├── agents/           # Internal style agent
│   └── bridge/           # External agent bridge
└── examples/             # Example agents
```

### Adding Custom Components

See `examples/live-metrics-agent/` for a complete example of:
- Connecting to the dashboard
- Registering a custom component
- Sending data updates
- Responding to theme changes

Happy styling! 🎨
