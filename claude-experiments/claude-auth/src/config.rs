use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::error::{AuthError, AuthResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<u64>,
    pub client_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub tokens: HashMap<String, TokenInfo>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            tokens: HashMap::new(),
        }
    }
}

impl AuthConfig {
    pub fn config_dir() -> AuthResult<PathBuf> {
        dirs::config_dir()
            .ok_or_else(|| AuthError::ConfigError("Could not find config directory".to_string()))
            .map(|dir| dir.join("claude-auth"))
    }

    pub fn config_file() -> AuthResult<PathBuf> {
        Ok(Self::config_dir()?.join("config.json"))
    }

    pub fn load() -> AuthResult<Self> {
        let config_file = Self::config_file()?;
        
        if !config_file.exists() {
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&config_file)?;
        let config: AuthConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }

    pub fn save(&self) -> AuthResult<()> {
        let config_dir = Self::config_dir()?;
        fs::create_dir_all(&config_dir)?;
        
        let config_file = Self::config_file()?;
        let contents = serde_json::to_string_pretty(self)?;
        fs::write(&config_file, contents)?;
        
        Ok(())
    }

    pub fn store_token(&mut self, provider: &str, token_info: TokenInfo) {
        self.tokens.insert(provider.to_string(), token_info);
    }

    pub fn get_token(&self, provider: &str) -> Option<&TokenInfo> {
        self.tokens.get(provider)
    }

    pub fn remove_token(&mut self, provider: &str) -> Option<TokenInfo> {
        self.tokens.remove(provider)
    }
}