use thiserror::Error;

pub type AuthResult<T> = Result<T, AuthError>;

#[derive(Error, Debug)]
pub enum AuthError {
    #[error("Network request failed: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    #[error("Token refresh failed: {status_code} - {message}")]
    TokenRefreshFailed { status_code: u16, message: String },
    
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    
    #[error("Invalid token: {0}")]
    InvalidToken(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    #[error("URL parsing error: {0}")]
    UrlError(#[from] url::ParseError),
    
    #[error("Invalid header value: {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),
    
    #[error("No refresh token available")]
    NoRefreshToken,
    
    #[error("Token expired")]
    TokenExpired,
}