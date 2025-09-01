pub mod auth;
pub mod client;
pub mod config;
pub mod error;

// Main library exports for external usage
pub use client::{ClaudeClient, MessageContent, ContentPart, Tool, ContentBlock, ClaudeResponse, Usage};
pub use error::{AuthError, AuthResult};
pub use auth::{ClaudeAuth, OAuthFlow};
pub use config::TokenInfo;

// Re-export for convenience
pub use serde_json::json;
