use claude_auth::{ClaudeClient, ConversationManager, SystemMessage, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multi-Turn Conversation with Automatic Caching");
    println!("==============================================\n");

    // Create client from environment token
    let client = ClaudeClient::from_env_token()?;

    // Set up system messages (automatically cached by default)
    let system = vec![
        SystemMessage::new(
            "You are a helpful AI assistant with expertise in programming and technology. \
            Keep your responses concise and informative."
        )
    ];

    // Set up tools (also automatically cached by default)
    let tools = vec![
        Tool::new(
            "calculate",
            "Perform a mathematical calculation",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            })
        ),
    ];

    // Create conversation manager - handles all caching automatically!
    let mut conversation = ConversationManager::new(
        client,
        "claude-sonnet-4-20250514",
        Some(system),
        Some(tools),
    ).with_max_tokens(1024);

    println!("Starting conversation with automatic caching enabled...\n");

    // Turn 1: First message
    println!("👤 User: What is Rust?");
    let response1 = conversation.send("What is Rust?").await?;

    if let Some(text) = &response1.content[0].text {
        println!("🤖 Claude: {}\n", text);
    }

    println!("📊 Turn 1 Usage:");
    println!("   Input tokens: {}", response1.usage.input_tokens);
    println!("   Output tokens: {}", response1.usage.output_tokens);
    if let Some(creation) = response1.usage.cache_creation_input_tokens {
        println!("   Cache creation: {} tokens (system + tools cached)", creation);
    }
    if let Some(read) = response1.usage.cache_read_input_tokens {
        println!("   Cache read: {} tokens", read);
    }
    println!();

    // Turn 2: Follow-up question
    println!("👤 User: What are its main features?");
    let response2 = conversation.send("What are its main features?").await?;

    if let Some(text) = &response2.content[0].text {
        println!("🤖 Claude: {}\n", text);
    }

    println!("📊 Turn 2 Usage:");
    println!("   Input tokens: {} (just the new message)", response2.usage.input_tokens);
    println!("   Output tokens: {}", response2.usage.output_tokens);
    if let Some(creation) = response2.usage.cache_creation_input_tokens {
        println!("   Cache creation: {} tokens (new turn cached)", creation);
    }
    if let Some(read) = response2.usage.cache_read_input_tokens {
        println!("   Cache read: {} tokens (90% savings!)", read);
    }
    println!();

    // Turn 3: Another follow-up
    println!("👤 User: How does it compare to C++?");
    let response3 = conversation.send("How does it compare to C++?").await?;

    if let Some(text) = &response3.content[0].text {
        println!("🤖 Claude: {}\n", text);
    }

    println!("📊 Turn 3 Usage:");
    println!("   Input tokens: {}", response3.usage.input_tokens);
    println!("   Output tokens: {}", response3.usage.output_tokens);
    if let Some(creation) = response3.usage.cache_creation_input_tokens {
        println!("   Cache creation: {} tokens", creation);
    }
    if let Some(read) = response3.usage.cache_read_input_tokens {
        println!("   Cache read: {} tokens (even more savings!)", read);
    }
    println!();

    // Show conversation history
    println!("\n📜 Conversation History:");
    println!("   Total turns: {}", conversation.get_history().len());

    println!("\n✨ Key Benefits:");
    println!("   1. System messages and tools are cached automatically");
    println!("   2. Each conversation turn is incrementally cached");
    println!("   3. Previous turns are read from cache (90% token savings)");
    println!("   4. Cache is refreshed on each use (5-minute TTL)");
    println!("   5. Zero configuration - just use ConversationManager!");

    Ok(())
}
