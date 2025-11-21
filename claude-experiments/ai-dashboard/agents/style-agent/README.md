# Style Agent

External agent that generates CSS styles using the Claude Agent SDK.

**NO API KEY REQUIRED** - This agent uses your Claude subscription when running in the Claude environment.

## Setup

1. **Install dependencies**

```bash
npm install
```

2. **Configure (Optional)**

If the dashboard is running on a non-default port:

```bash
export DASHBOARD_PORT=3001
```

3. **Build and run**

```bash
npm run dev
```

## How It Works

1. Connects to the dashboard at `ws://localhost:3000`
2. Registers as the `style-agent`
3. Listens for `generate-style` query messages
4. Uses Claude Agent SDK with the `provide_stylesheet` tool
5. Generates complete CSS based on aesthetic prompts
6. Sends the generated styles back to the dashboard

## Usage

1. Start the dashboard first (`npm run dev` from root)
2. Start this style agent (`npm run dev` from this directory)
3. In the dashboard UI, type a style prompt like "Art Deco with gold"
4. The agent will generate CSS and the dashboard will update

## Claude Agent SDK

This agent uses `@anthropic-ai/claude-agent-sdk` which:
- Handles interaction with Claude (using your subscription, no API key!)
- Manages tool calling and responses
- Provides structured agent workflows

No need to manually manage API calls - the Agent SDK handles it all!
