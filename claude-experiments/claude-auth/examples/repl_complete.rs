use claude_auth::{ClaudeClient, MessageContent, Tool};
use serde_json::json;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Claude Auth REPL with Complete Tool Flow");
    println!("=======================================");
    
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
        Tool::new(
            "get_time",
            "Get the current time",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        ),
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

    let mut conversation: Vec<MessageContent> = Vec::new();

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

        // Add user message to conversation
        conversation.push(MessageContent::user(message));
        
        println!("📤 Sending request:");
        println!("Model: claude-sonnet-4-5-20250929");
        println!("Messages: {:#?}", conversation);
        println!("Tools: {:#?}", tools);
        println!();
        
        match client.create_message_with_tools(
            "claude-sonnet-4-5-20250929",
            conversation.clone(),
            Some(1000),
            None,
            Some(tools.clone())
        ).await {
            Ok(response) => {
                println!("Full response: {:#?}", response);
                println!();
                
                let mut has_tool_calls = false;
                let mut tool_results = Vec::new();
                
                // Add assistant response to conversation
                conversation.push(MessageContent::assistant_with_tool_calls(response.content.clone()));
                
                for content_block in &response.content {
                    match content_block.r#type.as_str() {
                        "text" => {
                            if let Some(text) = &content_block.text {
                                println!("Claude: {}", text);
                            }
                        }
                        "tool_use" => {
                            has_tool_calls = true;
                            if let (Some(id), Some(name), Some(input)) = (&content_block.id, &content_block.name, &content_block.input) {
                                println!("🔧 Tool Call: {} (id: {}) with input: {}", name, id, input);
                                
                                // Execute tool
                                let result = match name.as_str() {
                                    "get_time" => {
                                        format!("Current time: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"))
                                    }
                                    "calculate" => {
                                        if let Some(expr) = input.get("expression").and_then(|v| v.as_str()) {
                                            // Simple evaluation (just echo for demo)
                                            format!("Calculation result for '{}': [simulated result - would need real evaluator]", expr)
                                        } else {
                                            "Invalid calculation input".to_string()
                                        }
                                    }
                                    _ => format!("Unknown tool: {}", name)
                                };
                                
                                println!("🔧 Tool Result: {}", result);
                                tool_results.push((id.clone(), result));
                            }
                        }
                        _ => {
                            println!("Unknown content type: {}", content_block.r#type);
                        }
                    }
                }
                
                // If there were tool calls, send results back to Claude
                if has_tool_calls && !tool_results.is_empty() {
                    println!();
                    println!("📤 Sending tool results back to Claude...");
                    
                    for (tool_use_id, result) in tool_results {
                        conversation.push(MessageContent::user_tool_result(&tool_use_id, &result));
                    }
                    
                    println!("Updated conversation with tool results: {:#?}", conversation);
                    println!();
                    
                    // Get Claude's final response
                    match client.create_message_with_tools(
                        "claude-sonnet-4-20250514",
                        conversation.clone(),
                        Some(1000),
                        None,
                        Some(tools.clone())
                    ).await {
                        Ok(final_response) => {
                            println!("Final response after tools: {:#?}", final_response);
                            println!();
                            
                            conversation.push(MessageContent::assistant_with_tool_calls(final_response.content.clone()));
                            
                            for content_block in &final_response.content {
                                if let Some(text) = &content_block.text {
                                    println!("Claude: {}", text);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ Error getting final response: {}", e);
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