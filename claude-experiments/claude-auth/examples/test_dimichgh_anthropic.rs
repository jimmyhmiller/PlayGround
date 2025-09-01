use anthropic_sdk::{Anthropic, MessageCreateBuilder};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get token from environment (using placeholder for testing)
    let token = env::var("CLAUDE_CODE_OAUTH_TOKEN")
        .unwrap_or_else(|_| {
            println!("⚠️ CLAUDE_CODE_OAUTH_TOKEN not found, using test placeholder");
            "test-oauth-token-placeholder".to_string()
        });

    println!("Testing dimichgh/anthropic-sdk-rust with OAuth token...");
    
    // Create client with custom token
    let client = Anthropic::new(&token)?;

    match client.messages()
        .create(
            MessageCreateBuilder::new("claude-sonnet-4-20250514", 100)
                .user("Hello! Please respond with 'OAuth working with dimichgh anthropic-rs'")
                .system("You are Claude Code, Anthropic's official CLI for Claude.")
                .build()
        )
        .await 
    {
        Ok(response) => {
            println!("✅ dimichgh/anthropic-sdk-rust works with OAuth token!");
            println!("Response: {:#?}", response);
            
            // Extract text from response content
            for (i, content_block) in response.content.iter().enumerate() {
                println!("Content block {}: {:?}", i, content_block);
            }
        }
        Err(e) => {
            println!("❌ dimichgh/anthropic-sdk-rust failed: {:?}", e);
            println!("This might be due to missing required OAuth headers:");
            println!("  - anthropic-beta: oauth-2025-04-20");
            println!("  - Proper User-Agent for Claude Code");
        }
    }

    Ok(())
}