// Layout configuration for CodeGraph visualizations
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Block spacing
    #[serde(default)]
    pub blocks: BlockSpacing,

    /// Arrow/edge configuration
    #[serde(default)]
    pub arrows: ArrowConfig,

    /// Text rendering
    #[serde(default)]
    pub text: TextConfig,

    /// Backedge-specific layout
    #[serde(default)]
    pub backedge: BackedgeConfig,
}

/// Block spacing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSpacing {
    /// Horizontal margin between blocks
    #[serde(default = "default_margin_x")]
    pub margin_x: f64,

    /// Vertical margin between layers
    #[serde(default = "default_margin_y")]
    pub margin_y: f64,

    /// Internal padding
    #[serde(default = "default_padding")]
    pub padding: f64,

    /// Gap between blocks in same layer
    #[serde(default = "default_gap")]
    pub gap: f64,
}

fn default_margin_x() -> f64 { 20.0 }
fn default_margin_y() -> f64 { 30.0 }
fn default_padding() -> f64 { 20.0 }
fn default_gap() -> f64 { 44.0 }

impl Default for BlockSpacing {
    fn default() -> Self {
        Self {
            margin_x: default_margin_x(),
            margin_y: default_margin_y(),
            padding: default_padding(),
            gap: default_gap(),
        }
    }
}

/// Arrow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrowConfig {
    /// Arrow curve radius
    #[serde(default = "default_radius")]
    pub radius: f64,

    /// Track padding
    #[serde(default = "default_track_padding")]
    pub track_padding: f64,

    /// Joint spacing
    #[serde(default = "default_joint_spacing")]
    pub joint_spacing: f64,

    /// Port starting position
    #[serde(default = "default_port_start")]
    pub port_start: f64,

    /// Port spacing
    #[serde(default = "default_port_spacing")]
    pub port_spacing: f64,

    /// Header arrow pushdown
    #[serde(default = "default_header_pushdown")]
    pub header_pushdown: f64,
}

fn default_radius() -> f64 { 12.0 }
fn default_track_padding() -> f64 { 36.0 }
fn default_joint_spacing() -> f64 { 16.0 }
fn default_port_start() -> f64 { 16.0 }
fn default_port_spacing() -> f64 { 60.0 }
fn default_header_pushdown() -> f64 { 16.0 }

impl Default for ArrowConfig {
    fn default() -> Self {
        Self {
            radius: default_radius(),
            track_padding: default_track_padding(),
            joint_spacing: default_joint_spacing(),
            port_start: default_port_start(),
            port_spacing: default_port_spacing(),
            header_pushdown: default_header_pushdown(),
        }
    }
}

/// Text rendering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    /// Character width (for size estimation)
    #[serde(default = "default_char_width")]
    pub character_width: f64,

    /// Line height
    #[serde(default = "default_line_height")]
    pub line_height: f64,

    /// Font family
    #[serde(default = "default_font_family")]
    pub font_family: String,

    /// Font size
    #[serde(default = "default_font_size")]
    pub font_size: f64,
}

fn default_char_width() -> f64 { 7.2 }
fn default_line_height() -> f64 { 16.0 }
fn default_font_family() -> String { "monospace".to_string() }
fn default_font_size() -> f64 { 12.0 }

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            character_width: default_char_width(),
            line_height: default_line_height(),
            font_family: default_font_family(),
            font_size: default_font_size(),
        }
    }
}

/// Backedge-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackedgeConfig {
    /// Loop margin
    #[serde(default = "default_loop_margin")]
    pub loop_margin: f64,

    /// Backedge margin
    #[serde(default = "default_backedge_margin")]
    pub backedge_margin: f64,
}

fn default_loop_margin() -> f64 { 7.0 }
fn default_backedge_margin() -> f64 { 7.0 }

impl Default for BackedgeConfig {
    fn default() -> Self {
        Self {
            loop_margin: default_loop_margin(),
            backedge_margin: default_backedge_margin(),
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            blocks: BlockSpacing::default(),
            arrows: ArrowConfig::default(),
            text: TextConfig::default(),
            backedge: BackedgeConfig::default(),
        }
    }
}

impl LayoutConfig {
    /// Load layout config from TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read layout config: {}", e))?;

        let config: LayoutConfig = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse layout TOML: {}", e))?;

        Ok(config)
    }

    /// Create default layout config
    pub fn default_config() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_layout() {
        let layout = LayoutConfig::default();
        assert_eq!(layout.blocks.margin_x, 20.0);
        assert_eq!(layout.arrows.radius, 12.0);
    }

    #[test]
    fn test_layout_serialization() {
        let layout = LayoutConfig::default();
        let toml_str = toml::to_string(&layout).unwrap();
        assert!(toml_str.contains("[blocks]"));
        assert!(toml_str.contains("[arrows]"));
    }
}
