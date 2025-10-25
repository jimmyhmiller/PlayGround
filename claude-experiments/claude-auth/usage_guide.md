# Claude Auth Usage Guide

A comprehensive guide to using the `claude-auth` Rust library for Claude API authentication and prompt caching.

## Table of Contents

- [Installation](#installation)
- [Authentication](#authentication)
- [Basic Usage](#basic-usage)
- [Prompt Caching](#prompt-caching)
- [Multi-turn Conversations](#multi-turn-conversations)
- [Working with Tools](#working-with-tools)
- [API Reference](#api-reference)
- [Examples](#examples)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
claude-auth = "0.1.0"
tokio = { version = "1", features = ["full"] }
serde_json = "1.0"
```

## Authentication

### Method 1: Environment Variable (Recommended)

Set the `CLAUDE_CODE_OAUTH_TOKEN` environment variable:

```bash
export CLAUDE_CODE_OAUTH_TOKEN="your-token-here"
```

Then create a client:

```rust
use claude_auth::ClaudeClient;

let mut client = ClaudeClient::from_env_token()?;
```

### Method 2: OAuth Flow with CLI

Use the included CLI tool for the OAuth flow:

```bash
# Start OAuth flow
cargo run --bin claude-auth-cli auth login --provider claude-pro-max

# Check token status
cargo run --bin claude-auth-cli token status --provider claude-pro-max

# Get current token
cargo run --bin claude-auth-cli token get --provider claude-pro-max
```

Then use the stored credentials:

```rust
let mut client = ClaudeClient::new("claude-pro-max")?;
```

## Basic Usage

### Simple Request

```rust
use claude_auth::{ClaudeClient, MessageContent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = ClaudeClient::from_env_token()?;

    let messages = vec![
        MessageContent::user("What is Rust?")
    ];

    let response = client.create_message(
        "claude-sonnet-4-20250514",
        messages,
        Some(1024),  // max tokens
        None,        // system message
    ).await?;

    if let Some(text) = &response.content[0].text {
        println!("Claude: {}", text);
    }

    println!("Tokens used: {} input, {} output",
        response.usage.input_tokens,
        response.usage.output_tokens
    );

    Ok(())
}
```

### With System Messages

```rust
use claude_auth::{ClaudeClient, MessageContent, SystemMessage};

let system = vec![
    SystemMessage::new("You are a helpful programming assistant specializing in Rust.")
];

let response = client.create_message(
    "claude-sonnet-4-20250514",
    messages,
    Some(1024),
    Some(system),
).await?;
```

## Prompt Caching

Prompt caching automatically reduces costs by up to 90% on repeated content.

### Automatic Caching (Default Behavior)

**System messages and tools are cached by default:**

```rust
use claude_auth::{SystemMessage, Tool, json};

// System messages are cached automatically (5-minute TTL)
let system = vec![
    SystemMessage::new("You are a helpful assistant.")
];

// Tools are also cached automatically
let tools = vec![
    Tool::new(
        "calculate",
        "Perform a calculation",
        json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        })
    )
];
```

### Cache Configuration

**5-minute cache (default):**
- Write cost: 1.25x base token price
- Read cost: 0.1x base token price
- Expires after 5 minutes of inactivity

**1-hour cache (beta):**
- Write cost: 2x base token price
- Read cost: 0.1x base token price
- Expires after 1 hour of inactivity

```rust
// Use 1-hour cache
let system = SystemMessage::new("Long context...").with_cache_1h();

// Disable caching if needed
let system = SystemMessage::new_no_cache("Frequently changing content");
let system = SystemMessage::new("Content").without_cache();
```

### Understanding Cache Usage

Check the `Usage` struct in responses:

```rust
let response = client.create_message(...).await?;

println!("Input tokens: {}", response.usage.input_tokens);
println!("Output tokens: {}", response.usage.output_tokens);

if let Some(cache_creation) = response.usage.cache_creation_input_tokens {
    println!("Cache creation: {} tokens (cached for reuse)", cache_creation);
}

if let Some(cache_read) = response.usage.cache_read_input_tokens {
    println!("Cache read: {} tokens (90% savings!)", cache_read);
}
```

## Multi-turn Conversations

Use `ConversationManager` for effortless multi-turn conversations with automatic caching.

### Basic Conversation

```rust
use claude_auth::{ClaudeClient, ConversationManager, SystemMessage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeClient::from_env_token()?;

    let system = vec![
        SystemMessage::new("You are a helpful assistant.")
    ];

    let mut conversation = ConversationManager::new(
        client,
        "claude-sonnet-4-20250514",
        Some(system),
        None,
    ).with_max_tokens(1024);

    // Turn 1
    let response1 = conversation.send("What is Rust?").await?;
    println!("Response: {}", response1.content[0].text.as_ref().unwrap());

    // Turn 2 - previous turn automatically cached
    let response2 = conversation.send("What are its main features?").await?;
    println!("Response: {}", response2.content[0].text.as_ref().unwrap());

    // Turn 3 - even more cache hits!
    let response3 = conversation.send("How does it compare to C++?").await?;

    Ok(())
}
```

### How Multi-turn Caching Works

1. **Turn 1**: System messages + tools + first message all cached
2. **Turn 2**: Previous turn read from cache (90% savings), new turn cached
3. **Turn 3+**: More cache hits as conversation grows

The caching happens automatically:
- System messages are always cached
- The last content block of the last message is marked for caching
- Previous cache markers remain valid (API uses longest cached prefix)
- Cache refreshes on each use (extends the 5-minute TTL)

### Managing Conversation History

```rust
// Get full history
let history = conversation.get_history();
println!("Conversation has {} messages", history.len());

// Clear history (start fresh)
conversation.clear_history();

// Access client for other operations
let client = conversation.client();
```

## Working with Tools

Tools allow Claude to perform actions or retrieve information.

### Defining Tools

```rust
use claude_auth::{Tool, json};

let tools = vec![
    Tool::new(
        "get_weather",
        "Get the current weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        })
    ),
    // More tools...
];
```

### Tool Use Flow

```rust
let response = client.create_message_with_cache(
    "claude-sonnet-4-20250514",
    messages,
    Some(1024),
    Some(system),
    Some(tools),
).await?;

// Check if Claude wants to use a tool
for block in &response.content {
    if block.r#type == "tool_use" {
        let tool_name = block.name.as_ref().unwrap();
        let tool_input = block.input.as_ref().unwrap();
        let tool_use_id = block.id.as_ref().unwrap();

        // Execute the tool (your implementation)
        let result = execute_tool(tool_name, tool_input);

        // Send result back
        messages.push(MessageContent::assistant_with_tool_calls(response.content.clone()));
        messages.push(MessageContent::user_tool_result(tool_use_id, &result));

        // Continue conversation
        let response = client.create_message_with_cache(...).await?;
    }
}
```

### Tools with ConversationManager

```rust
let tools = vec![
    Tool::new("calculate", "Perform calculation", json!({...}))
];

let mut conversation = ConversationManager::new(
    client,
    "claude-sonnet-4-20250514",
    Some(system),
    Some(tools),  // Tools are cached automatically
);
```

## API Reference

### Core Types

#### `ClaudeClient`

Main API client.

**Methods:**
- `new(provider: &str) -> AuthResult<Self>` - Create from stored OAuth token
- `from_env_token() -> AuthResult<Self>` - Create from environment variable
- `create_message(...)` - Simple message creation
- `create_message_with_tools(...)` - Message with tool support
- `create_message_with_cache(...)` - Full cache control (auto-marks for caching)

#### `ConversationManager`

Multi-turn conversation helper.

**Methods:**
- `new(client, model, system, tools)` - Create new conversation
- `with_max_tokens(n)` - Set max tokens
- `send(message) -> AuthResult<ClaudeResponse>` - Send user message
- `get_history()` - Get conversation history
- `clear_history()` - Reset conversation

#### `SystemMessage`

System prompt with caching.

**Methods:**
- `new(text)` - Create with default caching (5-min)
- `new_no_cache(text)` - Create without caching
- `with_cache()` - Enable 5-minute cache
- `with_cache_1h()` - Enable 1-hour cache
- `without_cache()` - Disable caching

#### `Tool`

Tool definition with caching.

**Methods:**
- `new(name, description, schema)` - Create with default caching (5-min)
- `new_no_cache(name, description, schema)` - Create without caching
- `with_cache()` - Enable 5-minute cache
- `with_cache_1h()` - Enable 1-hour cache
- `without_cache()` - Disable caching

#### `MessageContent`

Conversation message.

**Helper methods:**
- `user(text)` - Create user message
- `assistant_text(text)` - Create assistant text message
- `assistant_with_tool_calls(blocks)` - Create assistant message with tool uses
- `user_tool_result(id, result)` - Create tool result message

### Response Types

#### `ClaudeResponse`

```rust
pub struct ClaudeResponse {
    pub id: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}
```

#### `Usage`

```rust
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_creation_input_tokens: Option<u32>,  // Tokens written to cache
    pub cache_read_input_tokens: Option<u32>,      // Tokens read from cache
}
```

## Examples

The repository includes several examples:

```bash
# Simple API call
cargo run --example simple_example

# Detailed caching examples
cargo run --example with_caching

# Multi-turn conversation
cargo run --example conversation

# Using tools
cargo run --example with_tools

# REPL examples
cargo run --example repl
cargo run --example repl_with_tools
cargo run --example repl_complete
```

## Best Practices

### 1. Use ConversationManager for Multi-turn Chats

```rust
// ✅ Good - automatic caching
let mut conversation = ConversationManager::new(...);
conversation.send("Hello").await?;

// ❌ Manual management is more error-prone
let mut messages = vec![];
messages.push(MessageContent::user("Hello"));
// ... manual history management
```

### 2. Default Caching is Usually Right

```rust
// ✅ Good - uses sensible defaults
let system = SystemMessage::new("You are helpful.");
let tool = Tool::new("calculate", "Calculate", schema);

// ⚠️  Only disable if content changes frequently
let system = SystemMessage::new_no_cache("Current time: ...");
```

### 3. Use 1-hour Cache for Long-lived Context

```rust
// ✅ Good for large, stable context
let system = SystemMessage::new("
    Here is the entire codebase documentation...
    [10,000 tokens of docs]
").with_cache_1h();
```

### 4. Monitor Cache Usage

```rust
let response = conversation.send("Hello").await?;

// Track savings
if let Some(cache_read) = response.usage.cache_read_input_tokens {
    let savings = cache_read as f64 * 0.9; // 90% savings
    println!("Saved approximately {:.0} tokens worth of cost", savings);
}
```

## Troubleshooting

### Authentication Errors

```rust
// Error: CLAUDE_CODE_OAUTH_TOKEN not found
// Solution: Set environment variable
std::env::set_var("CLAUDE_CODE_OAUTH_TOKEN", "your-token");

// Or use OAuth flow
cargo run --bin claude-auth-cli auth login --provider claude-pro-max
```

### Token Refresh

The library automatically refreshes expired tokens:

```rust
// This handles token refresh automatically
let response = client.create_message(...).await?;
```

### Cache Not Working

Caching only applies when:
- Using `create_message_with_cache()` or `ConversationManager`
- Prompt is above minimum length (1024-4096 tokens depending on model)
- Content hasn't changed since last request

```rust
// Check cache usage in response
if response.usage.cache_read_input_tokens.is_none() {
    println!("No cache hit - prompt may be too short or content changed");
}
```

## Additional Resources

- [API Documentation](https://docs.rs/claude-auth) - Generated docs
- [Examples Directory](./examples/) - Working code examples
- [Claude Documentation](https://docs.anthropic.com) - Official Claude docs
- [Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) - Caching details

## License

See LICENSE file for details.
