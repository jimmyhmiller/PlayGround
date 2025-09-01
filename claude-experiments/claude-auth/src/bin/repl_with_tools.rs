use claude_auth::{ClaudeClient, MessageContent, Tool};
use serde_json::json;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Claude Auth REPL with Tools");
    println!("===========================");
    
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

    // Define some example tools
    let tools = vec![
        Tool {
            name: "get_time".to_string(),
            description: "Get the current time".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        Tool {
            name: "calculate".to_string(),
            description: "Perform a mathematical calculation".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
        },
    ];

    println!();
    println!("Available tools: get_time, calculate");
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
        
        match client.create_message_with_tools(
            "claude-sonnet-4-20250514",
            messages,
            Some(1000),
            None,
            Some(tools.clone())
        ).await {
            Ok(response) => {
                println!("Full response: {:#?}", response);
                println!();
                
                for content_block in &response.content {
                    match content_block.r#type.as_str() {
                        "text" => {
                            if let Some(text) = &content_block.text {
                                println!("Claude: {}", text);
                            }
                        }
                        "tool_use" => {
                            if let (Some(name), Some(input)) = (&content_block.name, &content_block.input) {
                                println!("🔧 Tool Call: {} with input: {}", name, input);
                                
                                // Simple tool implementations
                                let result = match name.as_str() {
                                    "get_time" => {
                                        format!("Current time: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"))
                                    }
                                    "calculate" => {
                                        if let Some(expr) = input.get("expression").and_then(|v| v.as_str()) {
                                            format!("Calculation result for '{}': [simulated result]", expr)
                                        } else {
                                            "Invalid calculation input".to_string()
                                        }
                                    }
                                    _ => format!("Unknown tool: {}", name)
                                };
                                
                                println!("🔧 Tool Result: {}", result);
                            }
                        }
                        _ => {
                            println!("Unknown content type: {}", content_block.r#type);
                        }
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