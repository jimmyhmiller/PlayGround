use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};
use url::Url;

use crate::config::{AuthConfig, TokenInfo};
use crate::error::{AuthError, AuthResult};

pub const CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const CLAUDE_AUTH_URL: &str = "https://claude.ai/oauth/authorize";
pub const CLAUDE_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";

#[derive(Debug, Clone)]
pub struct OAuthFlow {
    client_id: String,
    code_verifier: String,
    code_challenge: String,
    state: String,
    redirect_uri: String,
}

impl OAuthFlow {
    pub fn new_with_port(port: u16) -> Self {
        // Generate random bytes for PKCE
        let mut rng_bytes = [0u8; 32];
        getrandom::getrandom(&mut rng_bytes).expect("Failed to generate random bytes");
        
        let code_verifier = URL_SAFE_NO_PAD.encode(&rng_bytes);
        let code_challenge = URL_SAFE_NO_PAD.encode(&Sha256::digest(code_verifier.as_bytes()));
        
        // Generate state
        let mut state_bytes = [0u8; 16];
        getrandom::getrandom(&mut state_bytes).expect("Failed to generate random bytes");
        let state = URL_SAFE_NO_PAD.encode(&state_bytes);

        Self {
            client_id: CLAUDE_CLIENT_ID.to_string(),
            code_verifier,
            code_challenge,
            state,
            redirect_uri: format!("http://localhost:{}/callback", port),
        }
    }

    pub fn new() -> Self {
        Self::new_with_port(57032)
    }

    pub fn wait_for_callback(&self) -> AuthResult<String> {
        let port = self.redirect_uri.split(':').nth(2)
            .and_then(|s| s.split('/').next())
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(57032);
            
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
            .map_err(|e| AuthError::ConfigError(format!("Failed to bind to port {}: {}", port, e)))?;
        
        println!("Waiting for OAuth callback on {}...", self.redirect_uri);
        
        for stream in listener.incoming() {
            let stream = stream?;
            if let Ok(code) = self.handle_callback(stream) {
                return Ok(code);
            }
        }
        
        Err(AuthError::AuthenticationFailed("No authorization code received".to_string()))
    }

    fn handle_callback(&self, mut stream: TcpStream) -> Result<String, Box<dyn std::error::Error>> {
        let mut buffer = [0; 1024];
        stream.read(&mut buffer)?;
        
        let request = String::from_utf8_lossy(&buffer);
        let first_line = request.lines().next().unwrap_or("");
        
        if let Some(path) = first_line.split_whitespace().nth(1) {
            let url = Url::parse(&format!("http://localhost{}", path))?;
            let query_pairs: std::collections::HashMap<_, _> = url.query_pairs().collect();
            
            if let Some(code) = query_pairs.get("code") {
                let response = "HTTP/1.1 200 OK\r\n\r\n<html><body><h1>Success!</h1><p>You can close this window.</p></body></html>";
                let _ = std::io::Write::write_all(&mut stream, response.as_bytes());
                return Ok(code.to_string());
            }
            
            if let Some(error) = query_pairs.get("error") {
                let response = "HTTP/1.1 400 Bad Request\r\n\r\n<html><body><h1>Error</h1><p>Authorization failed</p></body></html>";
                let _ = std::io::Write::write_all(&mut stream, response.as_bytes());
                return Err(format!("OAuth error: {}", error).into());
            }
        }
        
        let response = "HTTP/1.1 400 Bad Request\r\n\r\n<html><body><h1>Invalid Request</h1></body></html>";
        let _ = std::io::Write::write_all(&mut stream, response.as_bytes());
        Err("Invalid callback request".into())
    }

    pub fn get_authorization_url(&self) -> AuthResult<String> {
        let mut url = Url::parse(CLAUDE_AUTH_URL)?;
        
        url.query_pairs_mut()
            .append_pair("client_id", &self.client_id)
            .append_pair("redirect_uri", &self.redirect_uri)
            .append_pair("response_type", "code")
            .append_pair("state", &self.state)
            .append_pair("scope", "user:inference")
            .append_pair("code_challenge", &self.code_challenge)
            .append_pair("code_challenge_method", "S256");

        Ok(url.to_string())
    }

    pub async fn exchange_code_for_tokens(&self, auth_code: &str) -> AuthResult<TokenInfo> {
        let client = Client::new();
        
        let data = TokenExchangeRequest {
            grant_type: "authorization_code".to_string(),
            client_id: self.client_id.clone(),
            code: auth_code.to_string(),
            code_verifier: self.code_verifier.clone(),
            redirect_uri: self.redirect_uri.clone(),
        };

        let response = client
            .post(CLAUDE_TOKEN_URL)
            .header("Content-Type", "application/json")
            .json(&data)
            .send()
            .await?;

        if response.status().is_success() {
            let token_response: TokenResponse = response.json().await?;
            let expires_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() + token_response.expires_in.unwrap_or(3600);

            Ok(TokenInfo {
                access_token: token_response.access_token,
                refresh_token: token_response.refresh_token,
                expires_at: Some(expires_at),
                client_id: Some(self.client_id.clone()),
            })
        } else {
            let error_text = response.text().await?;
            Err(AuthError::AuthenticationFailed(format!(
                "Token exchange failed: {}", error_text
            )))
        }
    }
}

#[derive(Debug, Serialize)]
struct TokenExchangeRequest {
    grant_type: String,
    client_id: String,
    code: String,
    code_verifier: String,
    redirect_uri: String,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
    token_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct RefreshTokenRequest {
    grant_type: String,
    refresh_token: String,
    client_id: String,
}

pub struct ClaudeAuth {
    config: AuthConfig,
    client: Client,
}

impl ClaudeAuth {
    pub fn new() -> AuthResult<Self> {
        let config = AuthConfig::load()?;
        let client = Client::new();
        
        Ok(Self { config, client })
    }

    pub fn get_token(&self, provider: &str) -> Option<&TokenInfo> {
        self.config.get_token(provider)
    }

    pub fn is_token_expired(&self, token_info: &TokenInfo) -> bool {
        if let Some(expires_at) = token_info.expires_at {
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            current_time >= expires_at.saturating_sub(300)
        } else {
            false
        }
    }

    pub async fn refresh_token(&mut self, provider: &str) -> AuthResult<()> {
        let token_info = self.config.get_token(provider)
            .ok_or_else(|| AuthError::ConfigError(format!("No token found for provider: {}", provider)))?
            .clone();

        let refresh_token = token_info.refresh_token.clone()
            .ok_or(AuthError::NoRefreshToken)?;

        let client_id = token_info.client_id.clone()
            .unwrap_or_else(|| CLAUDE_CLIENT_ID.to_string());

        let data = RefreshTokenRequest {
            grant_type: "refresh_token".to_string(),
            refresh_token,
            client_id: client_id.clone(),
        };

        let response = self.client
            .post(CLAUDE_TOKEN_URL)
            .header("Content-Type", "application/json")
            .json(&data)
            .send()
            .await?;

        if response.status().is_success() {
            let token_response: TokenResponse = response.json().await?;
            let expires_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() + token_response.expires_in.unwrap_or(3600);

            let new_token_info = TokenInfo {
                access_token: token_response.access_token,
                refresh_token: token_response.refresh_token.or(token_info.refresh_token),
                expires_at: Some(expires_at),
                client_id: Some(client_id),
            };

            self.config.store_token(provider, new_token_info);
            self.config.save()?;
            Ok(())
        } else {
            let status_code = response.status().as_u16();
            let error_text = response.text().await?;
            Err(AuthError::TokenRefreshFailed {
                status_code,
                message: error_text,
            })
        }
    }

    pub async fn get_valid_token(&mut self, provider: &str) -> AuthResult<String> {
        let token_info = self.config.get_token(provider)
            .ok_or_else(|| AuthError::ConfigError(format!("No token found for provider: {}", provider)))?
            .clone();

        if self.is_token_expired(&token_info) {
            self.refresh_token(provider).await?;
            let updated_token = self.config.get_token(provider)
                .ok_or_else(|| AuthError::ConfigError("Token not found after refresh".to_string()))?;
            Ok(updated_token.access_token.clone())
        } else {
            Ok(token_info.access_token)
        }
    }

    pub fn store_token(&mut self, provider: &str, token_info: TokenInfo) -> AuthResult<()> {
        self.config.store_token(provider, token_info);
        self.config.save()
    }

    pub fn remove_token(&mut self, provider: &str) -> AuthResult<()> {
        self.config.remove_token(provider);
        self.config.save()
    }

    pub fn list_providers(&self) -> Vec<String> {
        self.config.tokens.keys().cloned().collect()
    }
}