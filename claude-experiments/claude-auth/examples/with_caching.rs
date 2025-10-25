use claude_auth::{ClaudeClient, MessageContent, SystemMessage, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Claude Prompt Caching Example");
    println!("==============================\n");

    // Create client from environment token
    let mut client = ClaudeClient::from_env_token()?;

    // Example 1: System Prompt Caching
    println!("Example 1: System Prompt Caching");
    println!("---------------------------------");

    // Create a system message - caching is enabled by default!
    // This is useful when you have a long system prompt that doesn't change
    let system_messages = vec![
        SystemMessage::new(
            "You are an expert software engineer specializing in Rust programming. \
            You have deep knowledge of best practices, common patterns, and performance optimization. \
            You provide clear, concise explanations and prefer idiomatic Rust code."
        ) // Automatically cached with 5-minute TTL
    ];

    let messages = vec![
        MessageContent::user("What's the difference between String and &str in Rust?")
    ];

    let response = client.create_message_with_cache(
        "claude-sonnet-4-20250514",
        messages,
        Some(1024),
        Some(system_messages),
        None,
    ).await?;

    println!("Response: {}", response.content[0].text.as_ref().unwrap());
    println!("\nUsage:");
    println!("  Input tokens: {}", response.usage.input_tokens);
    println!("  Output tokens: {}", response.usage.output_tokens);
    if let Some(cache_creation) = response.usage.cache_creation_input_tokens {
        println!("  Cache creation tokens: {} (will be cached for 5 minutes)", cache_creation);
    }
    if let Some(cache_read) = response.usage.cache_read_input_tokens {
        println!("  Cache read tokens: {} (90% savings!)", cache_read);
    }

    // Example 2: Tool Definitions Caching
    println!("\n\nExample 2: Tool Definitions Caching");
    println!("------------------------------------");

    // Tools are cached by default with 5-minute TTL
    // All tools will have caching enabled automatically
    let tools = vec![
        Tool::new(
            "read_file",
            "Read the contents of a file",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            })
        ),
        Tool::new(
            "write_file",
            "Write content to a file",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            })
        ),
        Tool::new(
            "list_directory",
            "List contents of a directory",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list"
                    }
                },
                "required": ["path"]
            })
        ),
    ];

    let messages2 = vec![
        MessageContent::user("What tools are available to me?")
    ];

    let response2 = client.create_message_with_cache(
        "claude-sonnet-4-20250514",
        messages2,
        Some(1024),
        None,
        Some(tools.clone()),
    ).await?;

    println!("Response: {}", response2.content[0].text.as_ref().unwrap());
    println!("\nUsage:");
    println!("  Input tokens: {}", response2.usage.input_tokens);
    println!("  Output tokens: {}", response2.usage.output_tokens);
    if let Some(cache_creation) = response2.usage.cache_creation_input_tokens {
        println!("  Cache creation tokens: {} (tool definitions cached)", cache_creation);
    }

    // Example 3: Using 1-hour cache for long-lived context
    println!("\n\nExample 3: 1-Hour Cache for Long Context");
    println!("-----------------------------------------");

    let long_context = SystemMessage::new(
        "You are analyzing this codebase. Here is the main file:\n\n\
        // This would be a very long file in practice\n\
        fn main() {\n\
            println!(\"Hello, world!\");\n\
        }\n\n\
        And here are the dependencies:\n\
        - tokio: async runtime\n\
        - serde: serialization\n\
        - reqwest: HTTP client\n\n\
        The project follows standard Rust conventions and uses cargo for builds."
    ).with_cache_1h(); // Use 1-hour cache for long-lived context

    let messages3 = vec![
        MessageContent::user("What async runtime does this project use?")
    ];

    let response3 = client.create_message_with_cache(
        "claude-sonnet-4-20250514",
        messages3,
        Some(512),
        Some(vec![long_context]),
        None,
    ).await?;

    println!("Response: {}", response3.content[0].text.as_ref().unwrap());
    println!("\nUsage:");
    println!("  Input tokens: {}", response3.usage.input_tokens);
    println!("  Output tokens: {}", response3.usage.output_tokens);
    if let Some(cache_creation) = response3.usage.cache_creation_input_tokens {
        println!("  Cache creation tokens: {} (cached for 1 hour)", cache_creation);
    }

    println!("\n\nKey Takeaways:");
    println!("--------------");
    println!("1. SystemMessage::new() and Tool::new() cache by default (5-minute TTL)");
    println!("2. Use .without_cache() or new_no_cache() to disable caching if needed");
    println!("3. Use .with_cache_1h() for longer-lived caches (beta feature)");
    println!("4. The API automatically marks the last message content for caching");
    println!("5. Cache creation costs 1.25x for 5-min cache, 2x for 1-hour cache");
    println!("6. Cache reads cost only 0.1x - that's 90% savings!");
    println!("7. Caches expire after 5 minutes (or 1 hour) of inactivity");
    println!("8. For multi-turn conversations, use ConversationManager!");
    println!("   (see examples/conversation.rs)");

    Ok(())
}
