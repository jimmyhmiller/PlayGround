use claude_auth::{ClaudeClient, MessageContent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = ClaudeClient::new("claude-pro-max")?;
    
    let messages = vec![
        MessageContent::user("Hello, Claude!")
    ];
    
    match client.create_message(
        "claude-3-5-sonnet-20241022",
        messages,
        Some(100),
        None
    ).await {
        Ok(response) => {
            if let Some(content_block) = response.content.first() {
                if let Some(text) = &content_block.text {
                    println!("Claude: {}", text);
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}