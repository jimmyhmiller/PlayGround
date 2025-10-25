use claude_auth::{ClaudeClient, MessageContent};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Claude Auth REPL");
    println!("================");
    
    // Try to create client from environment variable first
    let mut client = match ClaudeClient::from_env_token() {
        Ok(client) => {
            println!("✅ Using CLAUDE_CODE_OAUTH_TOKEN from environment");
            client
        }
        Err(_) => {
            println!("🔍 No CLAUDE_CODE_OAUTH_TOKEN found, trying stored config...");
            match ClaudeClient::new("claude-pro-max") {
                Ok(client) => {
                    println!("✅ Using stored authentication for claude-pro-max");
                    client
                }
                Err(e) => {
                    eprintln!("❌ Authentication failed: {}", e);
                    eprintln!("Either:");
                    eprintln!("  1. Set CLAUDE_CODE_OAUTH_TOKEN environment variable");
                    eprintln!("  2. Run: claude-auth-cli auth login --provider claude-pro-max");
                    std::process::exit(1);
                }
            }
        }
    };

    println!();
    println!("Type your messages (empty line to quit):");
    println!();

    loop {
        print!("> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        let message = input.trim();
        if message.is_empty() {
            println!("Goodbye!");
            break;
        }

        let messages = vec![MessageContent::user(message)];
        
        match client.create_message(
            "claude-sonnet-4-20250514",
            messages,
            Some(1000),
            None
        ).await {
            Ok(response) => {
                if let Some(content_block) = response.content.first() {
                    if let Some(text) = &content_block.text {
                        println!("Claude: {}", text);
                    }
                }
                println!();
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
                println!();
            }
        }
    }

    Ok(())
}