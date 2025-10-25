//! Claude API client with automatic prompt caching support.
//!
//! This module provides the core API client for interacting with Claude,
//! including automatic prompt caching for optimal token efficiency.
//!
//! # Key Types
//!
//! - [`ClaudeClient`] - Main API client for making requests
//! - [`ConversationManager`] - Helper for multi-turn conversations with automatic caching
//! - [`SystemMessage`] - System prompt blocks (cached by default)
//! - [`Tool`] - Tool definitions (cached by default)
//! - [`MessageContent`] - User and assistant messages
//! - [`ClaudeResponse`] - API response with content and usage statistics
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```rust,no_run
//! use claude_auth::{ClaudeClient, MessageContent};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut client = ClaudeClient::from_env_token()?;
//! let messages = vec![MessageContent::user("Hello!")];
//! let response = client.create_message(
//!     "claude-sonnet-4-20250514",
//!     messages,
//!     Some(1024),
//!     None,
//! ).await?;
//! # Ok(())
//! # }
//! ```

use reqwest::{Client, header::{HeaderMap, HeaderValue}};
use serde::{Deserialize, Serialize};

use crate::auth::ClaudeAuth;
use crate::error::{AuthError, AuthResult};

/// Cache control configuration for prompt caching.
///
/// Controls how content is cached to reduce token costs.
/// Caching reduces costs by 90% on cache hits.
///
/// # Cache Pricing
///
/// - **5-minute cache**: Write costs 1.25x, read costs 0.1x
/// - **1-hour cache**: Write costs 2x, read costs 0.1x
///
/// # Examples
///
/// ```rust
/// use claude_auth::CacheControl;
///
/// // 5-minute cache (default)
/// let cache = CacheControl::ephemeral();
///
/// // 1-hour cache
/// let cache = CacheControl::ephemeral_1h();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl CacheControl {
    /// Creates a 5-minute ephemeral cache (default)
    pub fn ephemeral() -> Self {
        Self {
            r#type: "ephemeral".to_string(),
            ttl: None,
        }
    }

    /// Creates a 1-hour ephemeral cache (beta)
    pub fn ephemeral_1h() -> Self {
        Self {
            r#type: "ephemeral".to_string(),
            ttl: Some("1h".to_string()),
        }
    }

    /// Creates a custom cache with specified TTL
    pub fn with_ttl(ttl: &str) -> Self {
        Self {
            r#type: "ephemeral".to_string(),
            ttl: Some(ttl.to_string()),
        }
    }
}

/// Main client for interacting with the Claude API.
///
/// Handles authentication, token refresh, and API requests with automatic prompt caching.
///
/// # Examples
///
/// ```rust,no_run
/// use claude_auth::{ClaudeClient, MessageContent};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // From environment variable
/// let mut client = ClaudeClient::from_env_token()?;
///
/// // Or from stored OAuth token
/// let mut client = ClaudeClient::new("claude-pro-max")?;
///
/// let messages = vec![MessageContent::user("Hello, Claude!")];
/// let response = client.create_message(
///     "claude-sonnet-4-20250514",
///     messages,
///     Some(1024),
///     None,
/// ).await?;
/// # Ok(())
/// # }
/// ```
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
        // Convert string system to SystemMessage vector for backward compatibility
        let system_messages = system.map(|s| vec![SystemMessage::new(&s)]);
        self.create_message_with_cache(model, messages, max_tokens, system_messages, tools).await
    }

    /// Creates a message with full cache control support
    /// Automatically marks the last system message and last content block for caching
    pub async fn create_message_with_cache(
        &mut self,
        model: &str,
        mut messages: Vec<MessageContent>,
        max_tokens: Option<u32>,
        system: Option<Vec<SystemMessage>>,
        tools: Option<Vec<Tool>>,
    ) -> AuthResult<ClaudeResponse> {
        let headers = self.get_headers().await?;

        // Automatically mark the last content block of the last message for caching
        if let Some(last_message) = messages.last_mut() {
            if let Some(last_content) = last_message.content.last_mut() {
                // Only add cache_control if not already set
                if last_content.cache_control.is_none() {
                    last_content.cache_control = Some(CacheControl::ephemeral());
                }
            }
        }

        // Ensure system messages are provided (with default if none given)
        let mut system_messages = system.unwrap_or_else(|| vec![SystemMessage::new("You are Claude Code, Anthropic's official CLI for Claude.")]);

        // Automatically mark the last system message for caching if not already set
        if let Some(last_sys) = system_messages.last_mut() {
            if last_sys.cache_control.is_none() {
                last_sys.cache_control = Some(CacheControl::ephemeral());
            }
        }

        let request_body = CreateMessageRequest {
            model: model.to_string(),
            max_tokens: max_tokens.unwrap_or(4096),
            messages,
            system: Some(system_messages),
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

/// A message in the conversation with role and content parts.
///
/// Messages alternate between user and assistant roles in a conversation.
///
/// # Examples
///
/// ```rust
/// use claude_auth::MessageContent;
///
/// // Create a user message
/// let user_msg = MessageContent::user("What is Rust?");
///
/// // Create an assistant message
/// let assistant_msg = MessageContent::assistant_text("Rust is a systems programming language.");
/// ```
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
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
                cache_control: None,
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
                cache_control: None,
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
                cache_control: None,
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
                cache_control: None,
            }],
        }
    }
}

impl ContentPart {
    /// Enables caching for this content part with a 5-minute TTL
    pub fn with_cache(mut self) -> Self {
        self.cache_control = Some(CacheControl::ephemeral());
        self
    }

    /// Enables caching for this content part with a 1-hour TTL
    pub fn with_cache_1h(mut self) -> Self {
        self.cache_control = Some(CacheControl::ephemeral_1h());
        self
    }

    /// Enables caching for this content part with a custom TTL
    pub fn with_cache_ttl(mut self, ttl: &str) -> Self {
        self.cache_control = Some(CacheControl::with_ttl(ttl));
        self
    }
}

/// System message content block with automatic caching.
///
/// System messages provide instructions and context to Claude.
/// They are cached by default with a 5-minute TTL for optimal token efficiency.
///
/// # Examples
///
/// ```rust
/// use claude_auth::SystemMessage;
///
/// // Cached by default (5-minute TTL)
/// let system = SystemMessage::new("You are a helpful assistant.");
///
/// // Disable caching if needed
/// let system = SystemMessage::new_no_cache("You are a helpful assistant.");
///
/// // Use 1-hour cache
/// let system = SystemMessage::new("You are a helpful assistant.").with_cache_1h();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub r#type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemMessage {
    /// Creates a new system message with caching enabled by default
    pub fn new(text: &str) -> Self {
        Self {
            r#type: "text".to_string(),
            text: text.to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        }
    }

    /// Creates a new system message without caching
    pub fn new_no_cache(text: &str) -> Self {
        Self {
            r#type: "text".to_string(),
            text: text.to_string(),
            cache_control: None,
        }
    }

    /// Disables caching for this system message
    pub fn without_cache(mut self) -> Self {
        self.cache_control = None;
        self
    }

    /// Enables caching for this system message with a 5-minute TTL
    pub fn with_cache(mut self) -> Self {
        self.cache_control = Some(CacheControl::ephemeral());
        self
    }

    /// Enables caching for this system message with a 1-hour TTL
    pub fn with_cache_1h(mut self) -> Self {
        self.cache_control = Some(CacheControl::ephemeral_1h());
        self
    }

    /// Enables caching for this system message with a custom TTL
    pub fn with_cache_ttl(mut self, ttl: &str) -> Self {
        self.cache_control = Some(CacheControl::with_ttl(ttl));
        self
    }
}

#[derive(Debug, Serialize)]
struct CreateMessageRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
}

/// Tool definition with automatic caching.
///
/// Tools allow Claude to perform actions or retrieve information.
/// They are cached by default with a 5-minute TTL for optimal token efficiency.
///
/// # Examples
///
/// ```rust
/// use claude_auth::{Tool, json};
///
/// // Tools are cached by default
/// let tool = Tool::new(
///     "calculate",
///     "Perform a calculation",
///     json!({
///         "type": "object",
///         "properties": {
///             "expression": {"type": "string"}
///         },
///         "required": ["expression"]
///     })
/// );
///
/// // Disable caching if needed
/// let tool = Tool::new_no_cache("calculate", "Perform a calculation", json!({}));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Tool {
    /// Creates a new tool with the given name, description, and input schema
    /// By default, tools are created with caching enabled (5-minute TTL)
    pub fn new(name: &str, description: &str, input_schema: serde_json::Value) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema,
            cache_control: Some(CacheControl::ephemeral()),
        }
    }

    /// Creates a new tool without caching enabled
    pub fn new_no_cache(name: &str, description: &str, input_schema: serde_json::Value) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema,
            cache_control: None,
        }
    }

    /// Enables caching for this tool with a 5-minute TTL
    pub fn with_cache(mut self) -> Self {
        self.cache_control = Some(CacheControl::ephemeral());
        self
    }

    /// Disables caching for this tool
    pub fn without_cache(mut self) -> Self {
        self.cache_control = None;
        self
    }

    /// Enables caching for this tool with a 1-hour TTL
    pub fn with_cache_1h(mut self) -> Self {
        self.cache_control = Some(CacheControl::ephemeral_1h());
        self
    }

    /// Enables caching for this tool with a custom TTL
    pub fn with_cache_ttl(mut self, ttl: &str) -> Self {
        self.cache_control = Some(CacheControl::with_ttl(ttl));
        self
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

/// Helper for managing multi-turn conversations with automatic caching.
///
/// `ConversationManager` simplifies multi-turn conversations by:
/// - Automatically managing conversation history
/// - Caching system messages, tools, and conversation turns
/// - Providing 90% token savings on cache hits
/// - Refreshing cache on each use (5-minute TTL)
///
/// # Examples
///
/// ```rust,no_run
/// use claude_auth::{ClaudeClient, ConversationManager, SystemMessage};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = ClaudeClient::from_env_token()?;
/// let system = vec![SystemMessage::new("You are a helpful assistant.")];
///
/// let mut conversation = ConversationManager::new(
///     client,
///     "claude-sonnet-4-20250514",
///     Some(system),
///     None,
/// );
///
/// // Each turn is automatically cached
/// let response1 = conversation.send("What is Rust?").await?;
/// let response2 = conversation.send("Tell me more.").await?;
/// // Previous turn read from cache with 90% savings!
/// # Ok(())
/// # }
/// ```
pub struct ConversationManager {
    client: ClaudeClient,
    model: String,
    system_messages: Vec<SystemMessage>,
    tools: Option<Vec<Tool>>,
    conversation_history: Vec<MessageContent>,
    max_tokens: u32,
}

impl ConversationManager {
    /// Creates a new conversation manager
    pub fn new(
        client: ClaudeClient,
        model: &str,
        system_messages: Option<Vec<SystemMessage>>,
        tools: Option<Vec<Tool>>,
    ) -> Self {
        let system = system_messages.unwrap_or_else(|| vec![SystemMessage::new("You are Claude Code, Anthropic's official CLI for Claude.")]);

        Self {
            client,
            model: model.to_string(),
            system_messages: system,
            tools,
            conversation_history: Vec::new(),
            max_tokens: 4096,
        }
    }

    /// Sets the max tokens for responses
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Adds a user message and sends it, returning the assistant's response
    pub async fn send(&mut self, user_message: &str) -> AuthResult<ClaudeResponse> {
        // Add user message to history
        self.conversation_history.push(MessageContent::user(user_message));

        // Send the conversation with automatic caching
        let response = self.client.create_message_with_cache(
            &self.model,
            self.conversation_history.clone(),
            Some(self.max_tokens),
            Some(self.system_messages.clone()),
            self.tools.clone(),
        ).await?;

        // Add assistant response to history
        self.conversation_history.push(MessageContent::assistant_with_tool_calls(response.content.clone()));

        Ok(response)
    }

    /// Gets the full conversation history
    pub fn get_history(&self) -> &[MessageContent] {
        &self.conversation_history
    }

    /// Clears the conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Returns a reference to the client
    pub fn client(&mut self) -> &mut ClaudeClient {
        &mut self.client
    }

    /// Gets the current model
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Gets the system messages
    pub fn system_messages(&self) -> &[SystemMessage] {
        &self.system_messages
    }

    /// Gets the tools
    pub fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}