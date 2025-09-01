# Claude Auth

A Rust library for Claude Code subscription authentication using OAuth.

## Features

- OAuth authentication flow for Claude Pro/Max subscriptions
- Automatic token refresh handling
- Simple API for making authenticated requests
- CLI utility for token management

## Library Usage

### Simple Usage
```rust
use claude_auth::{ClaudeClient, MessageContent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use environment token (recommended)
    let mut client = ClaudeClient::from_env_token()?;
    
    // Or use stored config
    // let mut client = ClaudeClient::new("claude-pro-max")?;
    
    let messages = vec![MessageContent::user("Hello Claude!")];
    
    let response = client.create_message(
        "claude-sonnet-4-20250514",
        messages,
        Some(100),
        None
    ).await?;
    
    for content_block in &response.content {
        if let Some(text) = &content_block.text {
            println!("Claude: {}", text);
        }
    }
    
    Ok(())
}
```

### With Tools
```rust
use claude_auth::{ClaudeClient, MessageContent, Tool, json};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = ClaudeClient::from_env_token()?;
    
    let tools = vec![
        Tool {
            name: "get_weather".to_string(),
            description: "Get weather for a city".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }),
        }
    ];
    
    let mut conversation = vec![MessageContent::user("What's the weather in NYC?")];
    
    // Initial request
    let response = client.create_message_with_tools(
        "claude-sonnet-4-20250514",
        conversation.clone(),
        Some(1000),
        None,
        Some(tools.clone())
    ).await?;
    
    // Add Claude's response to conversation
    conversation.push(MessageContent::assistant_with_tool_calls(response.content.clone()));
    
    // Handle tool calls
    for content_block in &response.content {
        if content_block.r#type == "tool_use" {
            if let (Some(id), Some(name), Some(input)) = (&content_block.id, &content_block.name, &content_block.input) {
                // Execute your tool logic here
                let result = "Weather: 68°F, cloudy";
                
                // Add tool result to conversation
                conversation.push(MessageContent::user_tool_result(id, result));
            }
        }
    }
    
    // Get final response
    let final_response = client.create_message_with_tools(
        "claude-sonnet-4-20250514",
        conversation,
        Some(1000),
        None,
        Some(tools)
    ).await?;
    
    // Process final response
    for content_block in &final_response.content {
        if let Some(text) = &content_block.text {
            println!("Claude: {}", text);
        }
    }
    
    Ok(())
}
```

## CLI Usage

### Authentication

```bash
# Start OAuth login flow
claude-auth-cli auth login --provider claude-pro-max

# Store tokens manually (if OAuth endpoint is blocked)
claude-auth-cli auth manual --provider claude-pro-max \
    --access-token "your-access-token" \
    --refresh-token "your-refresh-token" \
    --expires-in 3600

# List authenticated providers
claude-auth-cli auth list

# Remove authentication
claude-auth-cli auth remove --provider claude-pro-max
```

### Token Management

```bash
# Print current access token
claude-auth-cli token --provider claude-pro-max

# Write token to file
claude-auth-cli token --provider claude-pro-max --output /path/to/token.txt
```

## Configuration

Tokens are stored in `~/.config/claude-auth/config.json` in JSON format.

## OAuth Flow Details

- Uses PKCE (Proof Key for Code Exchange) for secure authentication
- Client ID: `9d1c250a-e61b-44d9-88ed-5944d1962f5e` (shared community client)
- Required scopes: `org:create_api_key user:profile user:inference`
- Redirect URI: `https://console.anthropic.com/oauth/code/callback`

## Manual Token Extraction

Due to Cloudflare protection on the token endpoint, you may need to extract tokens manually:

1. Complete OAuth flow in browser
2. Open browser developer tools
3. Go to Application tab > Local Storage > console.anthropic.com
4. Look for keys containing 'token', 'auth', or 'access'
5. Use `claude-auth-cli auth manual` to store tokens