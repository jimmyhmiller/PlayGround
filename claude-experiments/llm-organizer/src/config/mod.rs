use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LLMConfig,
    pub filesystem: FilesystemConfig,
    pub database: DatabaseConfig,
    pub organization: OrganizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Endpoint URL for the LLM API (e.g., "http://localhost:8080")
    pub endpoint: String,

    /// Optional model name if the endpoint supports multiple models
    #[serde(default)]
    pub model: Option<String>,

    /// Maximum tokens for completion requests
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for generation (0.0 to 1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemConfig {
    /// Directories to watch for file changes
    pub watch_dirs: Vec<PathBuf>,

    /// Mount point for the FUSE filesystem
    pub mount_point: PathBuf,

    /// File extensions to ignore (e.g., [".tmp", ".swp"])
    #[serde(default)]
    pub ignore_extensions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to the SQLite database file
    pub path: PathBuf,

    /// Enable WAL mode for better concurrent access
    #[serde(default = "default_true")]
    pub wal_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationConfig {
    /// Master prompt that defines how files should be organized
    pub prompt: String,

    /// Whether to automatically analyze new files
    #[serde(default = "default_true")]
    pub auto_analyze: bool,

    /// Cache TTL for LLM responses in seconds
    #[serde(default = "default_cache_ttl")]
    pub cache_ttl_secs: u64,
}

// Default value functions
fn default_max_tokens() -> usize {
    2048
}

fn default_temperature() -> f32 {
    0.7
}

fn default_timeout() -> u64 {
    30
}

fn default_true() -> bool {
    true
}

fn default_cache_ttl() -> u64 {
    3600 // 1 hour
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())
            .context("Failed to read config file")?;

        toml::from_str(&contents)
            .context("Failed to parse config file")
    }

    /// Load configuration from default location (~/.config/llm-organizer/config.toml)
    pub fn load_default() -> Result<Self> {
        let config_path = Self::default_config_path()?;

        if !config_path.exists() {
            Self::create_default_config(&config_path)?;
        }

        Self::from_file(&config_path)
    }

    /// Get the default configuration file path
    pub fn default_config_path() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .context("HOME environment variable not set")?;

        Ok(PathBuf::from(home)
            .join(".config")
            .join("llm-organizer")
            .join("config.toml"))
    }

    /// Create a default configuration file
    fn create_default_config(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create config directory")?;
        }

        let default_config = Self::default();
        let toml_string = toml::to_string_pretty(&default_config)
            .context("Failed to serialize default config")?;

        std::fs::write(path, toml_string)
            .context("Failed to write default config file")?;

        log::info!("Created default configuration at {}", path.display());
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let base_path = PathBuf::from(&home).join(".local").join("share").join("llm-organizer");

        Self {
            llm: LLMConfig {
                endpoint: "http://localhost:8080".to_string(),
                model: None,
                max_tokens: default_max_tokens(),
                temperature: default_temperature(),
                timeout_secs: default_timeout(),
            },
            filesystem: FilesystemConfig {
                watch_dirs: vec![
                    PathBuf::from(&home).join("Documents"),
                ],
                mount_point: base_path.join("mount"),
                ignore_extensions: vec![
                    ".tmp".to_string(),
                    ".swp".to_string(),
                    ".DS_Store".to_string(),
                ],
            },
            database: DatabaseConfig {
                path: base_path.join("metadata.db"),
                wal_mode: default_true(),
            },
            organization: OrganizationConfig {
                prompt: "Organize files by topic, project, and date. \
                         Extract key information like authors, dates, and subjects. \
                         Tag documents with relevant keywords.".to_string(),
                auto_analyze: default_true(),
                cache_ttl_secs: default_cache_ttl(),
            },
        }
    }
}
