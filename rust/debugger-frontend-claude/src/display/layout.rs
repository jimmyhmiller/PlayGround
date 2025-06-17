/// Layout utilities for organizing debugger UI elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub direction: Direction,
    pub margin: f32,
    pub child_spacing: f32,
}

impl Layout {
    pub fn horizontal() -> Self {
        Self {
            direction: Direction::Horizontal,
            margin: 0.0,
            child_spacing: 0.0,
        }
    }

    pub fn vertical() -> Self {
        Self {
            direction: Direction::Vertical,
            margin: 0.0,
            child_spacing: 0.0,
        }
    }

    pub fn with_margin(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }

    pub fn with_child_spacing(mut self, child_spacing: f32) -> Self {
        self.child_spacing = child_spacing;
        self
    }
}

/// Helper for organizing text content into lines
pub fn lines_from_strings(lines: &[String]) -> Vec<&str> {
    lines.iter().map(|s| s.as_str()).collect()
}

/// Format a list of items with consistent indentation
pub fn format_list_with_prefix<T>(
    items: &[T],
    formatter: impl Fn(&T) -> String,
    prefix: &str,
) -> Vec<String> {
    items
        .iter()
        .map(|item| format!("{}{}", prefix, formatter(item)))
        .collect()
}