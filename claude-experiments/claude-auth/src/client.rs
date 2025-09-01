use reqwest::{Client, header::{HeaderMap, HeaderValue}};
use serde::{Deserialize, Serialize};

use crate::auth::ClaudeAuth;
use crate::error::{AuthError, AuthResult};

pub struct ClaudeClient {
    auth: ClaudeAuth,
    client: Client,
    provider: String,
}

impl ClaudeClient {
    pub fn new(provider: &str) -> AuthResult<Self> {
        let auth = ClaudeAuth::new()?;
        let client = Client::new();
        
        Ok(Self {
            auth,
            client,
            provider: provider.to_string(),
        })
    }

    pub fn from_env_token() -> AuthResult<Self> {
        // Verify the environment variable exists
        std::env::var("CLAUDE_CODE_OAUTH_TOKEN")
            .map_err(|_| AuthError::ConfigError("CLAUDE_CODE_OAUTH_TOKEN not found".to_string()))?;
        
        let auth = ClaudeAuth::new()?;
        let client = Client::new();
        
        Ok(Self {
            auth,
            client,
            provider: "env".to_string(),
        })
    }

    pub async fn get_current_token(&mut self) -> AuthResult<String> {
        if self.provider == "env" {
            std::env::var("CLAUDE_CODE_OAUTH_TOKEN")
                .map_err(|_| AuthError::ConfigError("CLAUDE_CODE_OAUTH_TOKEN not found".to_string()))
        } else {
            self.auth.get_valid_token(&self.provider).await
        }
    }

    async fn get_headers(&mut self) -> AuthResult<HeaderMap> {
        let token = if self.provider == "env" {
            std::env::var("CLAUDE_CODE_OAUTH_TOKEN")
                .map_err(|_| AuthError::ConfigError("CLAUDE_CODE_OAUTH_TOKEN not found".to_string()))?
        } else {
            self.auth.get_valid_token(&self.provider).await?
        };
        
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", HeaderValue::from_static("application/json"));
        headers.insert("Authorization", HeaderValue::from_str(&format!("Bearer {}", token))?);
        headers.insert("anthropic-beta", HeaderValue::from_static("oauth-2025-04-20"));
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers.insert("User-Agent", HeaderValue::from_static("claude-auth-rust/0.1.0"));
        headers.insert("X-Stainless-Lang", HeaderValue::from_static("rust"));
        headers.insert("X-Stainless-Package-Version", HeaderValue::from_static("0.1.0"));
        
        Ok(headers)
    }

    pub async fn create_message(
        &mut self,
        model: &str,
        messages: Vec<MessageContent>,
        max_tokens: Option<u32>,
        system: Option<String>,
    ) -> AuthResult<ClaudeResponse> {
        self.create_message_with_tools(model, messages, max_tokens, system, None).await
    }

    pub async fn create_message_with_tools(
        &mut self,
        model: &str,
        messages: Vec<MessageContent>,
        max_tokens: Option<u32>,
        system: Option<String>,
        tools: Option<Vec<Tool>>,
    ) -> AuthResult<ClaudeResponse> {
        let headers = self.get_headers().await?;
        
        let request_body = CreateMessageRequest {
            model: model.to_string(),
            max_tokens: max_tokens.unwrap_or(4096),
            messages,
            system: system.unwrap_or_else(|| "You are Claude Code, Anthropic's official CLI for Claude.".to_string()),
            tools,
        };

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .headers(headers)
            .json(&request_body)
            .send()
            .await?;

        if response.status().is_success() {
            let claude_response: ClaudeResponse = response.json().await?;
            Ok(claude_response)
        } else if response.status().as_u16() == 401 {
            Err(AuthError::TokenExpired)
        } else {
            let status = response.status().as_u16();
            let error_text = response.text().await?;
            Err(AuthError::AuthenticationFailed(format!(
                "API request failed: {} - {}", status, error_text
            )))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageContent {
    pub role: String,
    pub content: Vec<ContentPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl MessageContent {
    pub fn user(text: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![ContentPart {
                r#type: "text".to_string(),
                text: Some(text.to_string()),
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
            }],
        }
    }
    
    pub fn assistant_text(text: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: vec![ContentPart {
                r#type: "text".to_string(),
                text: Some(text.to_string()),
                id: None,
                name: None,
                input: None,
                tool_use_id: None,
                content: None,
            }],
        }
    }

    pub fn assistant_with_tool_calls(content_blocks: Vec<ContentBlock>) -> Self {
        let content_parts: Vec<ContentPart> = content_blocks
            .into_iter()
            .map(|block| ContentPart {
                r#type: block.r#type.clone(),
                text: block.text,
                id: block.id,
                name: block.name,
                input: block.input,
                tool_use_id: None,
                content: None,
            })
            .collect();

        Self {
            role: "assistant".to_string(),
            content: content_parts,
        }
    }
    
    pub fn user_tool_result(tool_use_id: &str, result: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![ContentPart {
                r#type: "tool_result".to_string(),
                text: None,
                id: None,
                name: None,
                input: None,
                tool_use_id: Some(tool_use_id.to_string()),
                content: Some(result.to_string()),
            }],
        }
    }
}

#[derive(Debug, Serialize)]
struct CreateMessageRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<MessageContent>,
    system: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ClaudeResponse {
    pub id: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlock {
    pub r#type: String,
    pub text: Option<String>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}