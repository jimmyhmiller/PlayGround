// Theme configuration for CodeGraph visualizations
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Complete theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    /// Theme metadata
    #[serde(default)]
    pub metadata: ThemeMetadata,

    /// Block colors
    #[serde(default)]
    pub blocks: BlockColors,

    /// Instruction attribute colors
    #[serde(default)]
    pub instruction_attributes: HashMap<String, String>,

    /// Heatmap configuration (for profiling data)
    #[serde(default)]
    pub heatmap: HeatmapConfig,

    /// Arrow/edge colors
    #[serde(default)]
    pub arrows: ArrowColors,
}

/// Theme metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeMetadata {
    /// Theme name
    #[serde(default = "default_name")]
    pub name: String,

    /// Theme description
    #[serde(default)]
    pub description: String,

    /// Target compiler (optional)
    #[serde(default)]
    pub compiler: Option<String>,
}

fn default_name() -> String {
    "Default".to_string()
}

impl Default for ThemeMetadata {
    fn default() -> Self {
        Self {
            name: default_name(),
            description: String::new(),
            compiler: None,
        }
    }
}

/// Block-level colors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockColors {
    /// Block header background
    #[serde(default = "default_block_header")]
    pub header: String,

    /// Loop header background
    #[serde(default = "default_loop_header")]
    pub loop_header: String,

    /// Backedge block background
    #[serde(default = "default_backedge")]
    pub backedge: String,

    /// Block border color
    #[serde(default = "default_border")]
    pub border: String,

    /// Text color
    #[serde(default = "default_text")]
    pub text: String,
}

fn default_block_header() -> String { "#0c0c0d".to_string() }
fn default_loop_header() -> String { "#1fa411".to_string() }
fn default_backedge() -> String { "#ff6600".to_string() }
fn default_border() -> String { "#000000".to_string() }
fn default_text() -> String { "#ffffff".to_string() }

impl Default for BlockColors {
    fn default() -> Self {
        Self {
            header: default_block_header(),
            loop_header: default_loop_header(),
            backedge: default_backedge(),
            border: default_border(),
            text: default_text(),
        }
    }
}

/// Instruction attribute colors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionColors {
    /// Attribute name to color mapping
    pub colors: HashMap<String, String>,
}

/// Heatmap configuration for profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapConfig {
    /// Enable heatmap visualization
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Hot color (high sample count)
    #[serde(default = "default_hot")]
    pub hot: String,

    /// Cool color (low sample count)
    #[serde(default = "default_cool")]
    pub cool: String,

    /// Threshold for hot/cool (0.0-1.0)
    #[serde(default = "default_threshold")]
    pub threshold: f64,
}

fn default_enabled() -> bool { true }
fn default_hot() -> String { "#ff849e".to_string() }
fn default_cool() -> String { "#ffe546".to_string() }
fn default_threshold() -> f64 { 0.2 }

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            hot: default_hot(),
            cool: default_cool(),
            threshold: default_threshold(),
        }
    }
}

/// Arrow/edge colors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrowColors {
    /// Normal forward edge
    #[serde(default = "default_arrow_normal")]
    pub normal: String,

    /// Backedge (loop)
    #[serde(default = "default_arrow_backedge")]
    pub backedge: String,

    /// Loop header edge
    #[serde(default = "default_arrow_loop_header")]
    pub loop_header: String,
}

fn default_arrow_normal() -> String { "#000000".to_string() }
fn default_arrow_backedge() -> String { "#ff0000".to_string() }
fn default_arrow_loop_header() -> String { "#1fa411".to_string() }

impl Default for ArrowColors {
    fn default() -> Self {
        Self {
            normal: default_arrow_normal(),
            backedge: default_arrow_backedge(),
            loop_header: default_arrow_loop_header(),
        }
    }
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            metadata: ThemeMetadata::default(),
            blocks: BlockColors::default(),
            instruction_attributes: HashMap::new(),
            heatmap: HeatmapConfig::default(),
            arrows: ArrowColors::default(),
        }
    }
}

/// Theme manager for loading and applying themes
pub struct Theme {
    config: ThemeConfig,
}

impl Theme {
    /// Load theme from TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read theme file: {}", e))?;

        let config: ThemeConfig = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse theme TOML: {}", e))?;

        Ok(Self { config })
    }

    /// Create theme from in-memory config
    pub fn from_config(config: ThemeConfig) -> Self {
        Self { config }
    }

    /// Create default theme
    pub fn default() -> Self {
        Self {
            config: ThemeConfig::default(),
        }
    }

    /// Get color for an instruction attribute
    pub fn instruction_attribute_color(&self, attr: &str) -> Option<&str> {
        self.config.instruction_attributes.get(attr).map(|s| s.as_str())
    }

    /// Get block header color
    pub fn block_header_color(&self) -> &str {
        &self.config.blocks.header
    }

    /// Get loop header color
    pub fn loop_header_color(&self) -> &str {
        &self.config.blocks.loop_header
    }

    /// Get backedge block color
    pub fn backedge_color(&self) -> &str {
        &self.config.blocks.backedge
    }

    /// Get heatmap config
    pub fn heatmap(&self) -> &HeatmapConfig {
        &self.config.heatmap
    }

    /// Get arrow color for edge type
    pub fn arrow_color(&self, is_backedge: bool, is_loop_header: bool) -> &str {
        if is_backedge {
            &self.config.arrows.backedge
        } else if is_loop_header {
            &self.config.arrows.loop_header
        } else {
            &self.config.arrows.normal
        }
    }

    /// Get the full config
    pub fn config(&self) -> &ThemeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_theme() {
        let theme = Theme::default();
        assert_eq!(theme.block_header_color(), "#0c0c0d");
        assert_eq!(theme.loop_header_color(), "#1fa411");
    }

    #[test]
    fn test_theme_serialization() {
        let theme = ThemeConfig::default();
        let toml_str = toml::to_string(&theme).unwrap();
        assert!(toml_str.contains("[blocks]"));
    }
}
