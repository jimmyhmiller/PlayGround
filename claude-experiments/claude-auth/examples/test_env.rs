use claude_auth::{ClaudeClient, MessageContent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test using environment variable
    let mut client = ClaudeClient::from_env_token()?;
    
    let messages = vec![
        MessageContent::user("Hello! Just testing authentication. Please respond with 'Auth working'.")
    ];
    
    match client.create_message(
        "claude-3-5-sonnet-20241022",
        messages,
        Some(50),
        None
    ).await {
        Ok(response) => {
            println!("✅ Authentication successful!");
            if let Some(content_block) = response.content.first() {
                if let Some(text) = &content_block.text {
                    println!("Claude: {}", text);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Authentication failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}