use claude_auth::{ClaudeClient, MessageContent, Tool, json};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client using environment token
    let mut client = ClaudeClient::from_env_token()?;
    
    // Define tools (cached by default)
    let tools = vec![
        Tool::new(
            "get_weather",
            "Get weather information for a city",
            json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            })
        )
    ];
    
    // Start conversation
    let mut conversation = vec![MessageContent::user("What's the weather in San Francisco?")];
    
    // First request
    let response = client.create_message_with_tools(
        "claude-sonnet-4-20250514",
        conversation.clone(),
        Some(1000),
        None,
        Some(tools.clone())
    ).await?;
    
    println!("Initial response: {:#?}", response);
    
    // Add Claude's response to conversation
    conversation.push(MessageContent::assistant_with_tool_calls(response.content.clone()));
    
    // Handle tool calls
    for content_block in &response.content {
        if content_block.r#type == "tool_use" {
            if let (Some(id), Some(name), Some(input)) = (&content_block.id, &content_block.name, &content_block.input) {
                println!("Tool call: {} with input: {}", name, input);
                
                // Execute tool (mock)
                let result = format!("Weather in San Francisco: 72°F, sunny");
                
                // Add tool result to conversation
                conversation.push(MessageContent::user_tool_result(id, &result));
            }
        }
    }
    
    // Send tool results back and get final response
    let final_response = client.create_message_with_tools(
        "claude-sonnet-4-20250514",
        conversation,
        Some(1000),
        None,
        Some(tools)
    ).await?;
    
    println!("Final response: {:#?}", final_response);
    
    // Print final text
    for content_block in &final_response.content {
        if let Some(text) = &content_block.text {
            println!("Claude: {}", text);
        }
    }
    
    Ok(())
}