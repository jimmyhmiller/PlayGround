use claude_auth::ClaudeClient;
use anthropic_sdk::{Anthropic, MessageCreateBuilder, Tool};
use serde_json::json;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Claude Auth REPL using dimichgh/anthropic-sdk-rust");
    println!("===============================================");
    
    // Try to create client from environment variable first
    let client = match std::env::var("CLAUDE_CODE_OAUTH_TOKEN") {
        Ok(token) => {
            println!("✅ Using CLAUDE_CODE_OAUTH_TOKEN from environment");
            Anthropic::with_claude_code_oauth(token)?
        }
        Err(_) => {
            println!("🔍 No CLAUDE_CODE_OAUTH_TOKEN found, trying stored config...");
            match ClaudeClient::new("claude-pro-max") {
                Ok(mut claude_client) => {
                    println!("✅ Using stored authentication for claude-pro-max");
                    
                    // Get fresh token and create anthropic-sdk-rust client
                    let token = claude_client.get_current_token().await?;
                    Anthropic::with_claude_code_oauth(token)?
                }
                Err(e) => {
                    println!("❌ Authentication failed: {}", e);
                    println!("Using placeholder token for demo...");
                    Anthropic::with_claude_code_oauth("demo-token-placeholder")?
                }
            }
        }
    };

    // Define the same tools we had before
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

    let mut conversation = Vec::new();

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
        conversation.push(anthropic_sdk::MessageParam {
            role: anthropic_sdk::Role::User,
            content: vec![anthropic_sdk::ContentBlockParam::Text { text: message.to_string() }],
        });
        
        println!("📤 Sending request to Claude via dimichgh/anthropic-sdk-rust...");
        println!("Model: claude-sonnet-4-20250514");
        println!("Messages: {:#?}", conversation);
        println!("Tools: {:#?}", tools);
        println!();
        
        match client.messages()
            .create(
                MessageCreateBuilder::new("claude-sonnet-4-20250514", 1000)
                    .messages(conversation.clone())
                    .tools(tools.clone())
                    .system("You are Claude Code, Anthropic's official CLI for Claude.")
                    .build()
            )
            .await 
        {
            Ok(response) => {
                println!("Full response: {:#?}", response);
                println!();
                
                let mut has_tool_calls = false;
                let mut tool_results = Vec::new();
                
                // Add assistant response to conversation
                let mut assistant_content = Vec::new();
                
                for content_block in &response.content {
                    match content_block {
                        anthropic_sdk::ContentBlock::Text { text } => {
                            println!("Claude: {}", text);
                            assistant_content.push(anthropic_sdk::ContentBlockParam::Text { text: text.clone() });
                        }
                        anthropic_sdk::ContentBlock::ToolUse { id, name, input } => {
                            has_tool_calls = true;
                            println!("🔧 Tool Call: {} (id: {}) with input: {}", name, id, input);
                            
                            // Execute tool
                            let result = match name.as_str() {
                                "get_time" => {
                                    format!("Current time: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"))
                                }
                                "calculate" => {
                                    if let Some(expr) = input.get("expression").and_then(|v| v.as_str()) {
                                        format!("Calculation result for '{}': [simulated result - would need real evaluator]", expr)
                                    } else {
                                        "Invalid calculation input".to_string()
                                    }
                                }
                                _ => format!("Unknown tool: {}", name)
                            };
                            
                            println!("🔧 Tool Result: {}", result);
                            tool_results.push((id.clone(), result));
                            assistant_content.push(anthropic_sdk::ContentBlockParam::ToolUse { 
                                id: id.clone(), 
                                name: name.clone(), 
                                input: input.clone() 
                            });
                        }
                        _ => {
                            println!("Unknown content type: {:?}", content_block);
                        }
                    }
                }
                
                // Add assistant message to conversation
                conversation.push(anthropic_sdk::MessageParam {
                    role: anthropic_sdk::Role::Assistant,
                    content: assistant_content,
                });
                
                // If there were tool calls, send results back to Claude
                if has_tool_calls && !tool_results.is_empty() {
                    println!();
                    println!("📤 Sending tool results back to Claude...");
                    
                    for (tool_use_id, result) in tool_results {
                        conversation.push(anthropic_sdk::MessageParam {
                            role: anthropic_sdk::Role::User,
                            content: vec![anthropic_sdk::ContentBlockParam::ToolResult { 
                                tool_use_id, 
                                content: Some(result),
                                is_error: Some(false)
                            }],
                        });
                    }
                    
                    println!("Updated conversation with tool results: {:#?}", conversation);
                    println!();
                    
                    // Get Claude's final response
                    match client.messages()
                        .create(
                            MessageCreateBuilder::new("claude-sonnet-4-20250514", 1000)
                                .messages(conversation.clone())
                                .tools(tools.clone())
                                .system("You are Claude Code, Anthropic's official CLI for Claude.")
                                .build()
                        )
                        .await 
                    {
                        Ok(final_response) => {
                            println!("Final response after tools: {:#?}", final_response);
                            println!();
                            
                            let mut final_assistant_content = Vec::new();
                            for content_block in &final_response.content {
                                match content_block {
                                    anthropic_sdk::ContentBlock::Text { text } => {
                                        println!("Claude: {}", text);
                                        final_assistant_content.push(anthropic_sdk::ContentBlockParam::Text { text: text.clone() });
                                    }
                                    _ => {}
                                }
                            }
                            
                            conversation.push(anthropic_sdk::MessageParam {
                                role: anthropic_sdk::Role::Assistant,
                                content: final_assistant_content,
                            });
                        }
                        Err(e) => {
                            eprintln!("❌ Error getting final response: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
            }
        }
        
        println!();
    }

    Ok(())
}