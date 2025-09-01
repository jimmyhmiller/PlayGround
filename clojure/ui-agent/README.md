# UI Agent

An autonomous Claude-powered agent that listens to messages and proactively creates helpful user interfaces.

## What it does

This agent receives open-ended messages via HTTP and autonomously decides what UI elements would be helpful, then creates them using Skija graphics. It's designed to be proactive - it doesn't wait to be asked to create UI, it analyzes the context and creates visualizations, tables, charts, or other interface elements that would make the information more accessible.

## Features

- **Autonomous UI Generation**: Analyzes messages and creates UI without explicit requests
- **Rich Graphics**: Uses Skija for drawing complex graphics, text, charts, and interactive elements  
- **Live Code Evaluation**: Can inspect its own state and execute Clojure code via nREPL
- **HTTP API**: Simple REST endpoint for sending messages
- **Message History**: Maintains context across conversations
- **Extensible Tools**: Rich set of drawing and evaluation tools available to the agent

## Quick Start

1. **Start the agent**:
   ```bash
   cd ui-agent
   clj -M:dev -m ui-agent.main
   ```

2. **Send messages using the script**:
   ```bash
   ./send-message.sh "Here's our Q1 sales data: North=150, South=200, East=175, West=125"
   ```

3. **Watch the UI window** - the agent will automatically create visualizations!

## Usage Examples

```bash
# Send data that could become a chart
./send-message.sh "Website traffic: Monday=1200, Tuesday=1150, Wednesday=1300, Thursday=1250, Friday=1400"

# Send a process that could become a flowchart  
./send-message.sh "Our deployment process: 1) Code review, 2) Run tests, 3) Deploy to staging, 4) Deploy to production"

# Send comparative data
./send-message.sh "Performance comparison - Old system: 2.5s response time, New system: 0.8s response time"

# Send any message - the agent will find something to visualize!
./send-message.sh "I'm working on three projects this week and feeling overwhelmed"
```

## API

**POST /message**
```json
{
  "message": "Your message text here",
  "metadata": {
    "any": "additional context"
  }
}
```

**GET /health**
Returns server status.

## How it Works

1. **Message Analysis**: The agent extracts patterns from incoming messages (numbers, lists, data structures)
2. **Context Building**: Combines current message with recent history and metadata
3. **Claude Processing**: Sends enriched prompt to Claude with access to drawing and evaluation tools
4. **Autonomous UI Creation**: Claude proactively creates helpful visualizations using the available tools
5. **Real-time Rendering**: UI updates appear immediately in the graphics window

## Architecture

- `ui-agent.core`: Graphics engine based on Skija/OpenGL
- `ui-agent.server`: HTTP server for receiving messages  
- `ui-agent.agent`: Main agent logic and message processing
- `ui-agent.claude`: Claude API integration and tool definitions
- `ui-agent.main`: Application entry point

The agent has access to powerful tools:
- Drawing primitives (rectangles, circles, text, paths)
- Code evaluation via nREPL
- State inspection and manipulation
- Error handling and debugging
- Canvas management

## Configuration

- Server port: Default 8080 (override with command line arg)
- nREPL port: 7889
- Claude API: Requires `CLAUDE_CODE_OAUTH_TOKEN` environment variable

## Requirements

- Clojure 1.12+
- macOS with ARM64 (for Skija natives)
- Claude API access
- httpie for the message script