use claude_auth::{ClaudeClient, MessageContent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client using environment token
    let mut client = ClaudeClient::from_env_token()?;
    
    // Simple text message
    let messages = vec![MessageContent::user("Hello Claude!")];
    
    let response = client.create_message(
        "claude-sonnet-4-20250514",
        messages,
        Some(100),
        None
    ).await?;
    
    // Print response
    for content_block in &response.content {
        if let Some(text) = &content_block.text {
            println!("Claude: {}", text);
        }
    }
    
    Ok(())
}