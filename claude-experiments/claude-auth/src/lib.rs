//! # Claude Auth
//!
//! A Rust library for authenticating with Claude Code subscriptions and making API calls
//! with automatic prompt caching for optimal token efficiency.
//!
//! ## Features
//!
//! - **OAuth 2.0 Authentication**: Full OAuth flow support for Claude Code subscriptions
//! - **Token Management**: Automatic token refresh and secure storage
//! - **Prompt Caching**: Automatic caching of system messages, tools, and conversation history
//! - **Multi-turn Conversations**: Built-in `ConversationManager` for effortless chat sessions
//! - **Tool Support**: Define and use tools with automatic caching
//! - **Zero Configuration**: Sensible defaults for maximum token savings
//!
//! ## Quick Start
//!
//! ### Simple API Call
//!
//! ```rust,no_run
//! use claude_auth::{ClaudeClient, MessageContent};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create client from environment variable
//!     let mut client = ClaudeClient::from_env_token()?;
//!
//!     // Send a message
//!     let messages = vec![MessageContent::user("Hello, Claude!")];
//!     let response = client.create_message(
//!         "claude-sonnet-4-20250514",
//!         messages,
//!         Some(1024),
//!         None,
//!     ).await?;
//!
//!     println!("{}", response.content[0].text.as_ref().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! ### Multi-turn Conversation with Automatic Caching
//!
//! ```rust,no_run
//! use claude_auth::{ClaudeClient, ConversationManager, SystemMessage};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = ClaudeClient::from_env_token()?;
//!
//!     // System messages are cached by default
//!     let system = vec![SystemMessage::new("You are a helpful assistant.")];
//!
//!     // ConversationManager handles caching automatically
//!     let mut conversation = ConversationManager::new(
//!         client,
//!         "claude-sonnet-4-20250514",
//!         Some(system),
//!         None,
//!     );
//!
//!     // Each turn is automatically cached
//!     let response1 = conversation.send("What is Rust?").await?;
//!     let response2 = conversation.send("Tell me more about ownership.").await?;
//!
//!     // Previous turns are read from cache with 90% token savings!
//!     Ok(())
//! }
//! ```
//!
//! ## Authentication
//!
//! Set the `CLAUDE_CODE_OAUTH_TOKEN` environment variable:
//!
//! ```bash
//! export CLAUDE_CODE_OAUTH_TOKEN="your-token-here"
//! ```
//!
//! Or use the CLI for OAuth flow:
//!
//! ```bash
//! cargo run --bin claude-auth-cli auth login --provider claude-pro-max
//! ```
//!
//! ## Caching
//!
//! Prompt caching is enabled by default for:
//! - **System messages** (`SystemMessage::new()`)
//! - **Tools** (`Tool::new()`)
//! - **Conversation history** (when using `ConversationManager` or `create_message_with_cache()`)
//!
//! Caching reduces costs by 90% on cache hits and automatically refreshes every 5 minutes.
//!
//! ## Examples
//!
//! See the `examples/` directory for more:
//! - `simple_example.rs` - Basic API usage
//! - `with_caching.rs` - Detailed caching examples
//! - `conversation.rs` - Multi-turn conversation with ConversationManager
//! - `with_tools.rs` - Using tools with Claude
//!
//! ## Module Organization
//!
//! - [`client`] - API client and conversation management
//! - [`auth`] - OAuth authentication and token management
//! - [`config`] - Token storage and configuration
//! - [`error`] - Error types and result aliases

pub mod auth;
pub mod client;
pub mod config;
pub mod error;

// Main library exports for external usage
pub use client::{ClaudeClient, MessageContent, ContentPart, Tool, ContentBlock, ClaudeResponse, Usage, CacheControl, SystemMessage, ConversationManager};
pub use error::{AuthError, AuthResult};
pub use auth::{ClaudeAuth, OAuthFlow};
pub use config::TokenInfo;

// Re-export for convenience
pub use serde_json::json;
