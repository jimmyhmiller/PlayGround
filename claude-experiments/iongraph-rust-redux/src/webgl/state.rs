//! Viewport and interaction state management
//!
//! Handles pan/zoom transformations and selection/hover state.

use std::collections::HashSet;

/// 2D vector for positions and offsets
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec2f {
    pub x: f32,
    pub y: f32,
}

impl Vec2f {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Viewport state for pan/zoom transformations
#[derive(Debug, Clone)]
pub struct Viewport {
    /// Offset in screen coordinates (pan position)
    pub offset: Vec2f,

    /// Zoom scale (1.0 = 100%)
    pub scale: f32,

    /// Screen dimensions
    pub screen_width: f32,
    pub screen_height: f32,
}

impl Viewport {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            offset: Vec2f::default(),
            scale: 1.0,
            screen_width: width,
            screen_height: height,
        }
    }

    /// Convert screen coordinates to world coordinates
    pub fn screen_to_world(&self, screen: Vec2f) -> Vec2f {
        Vec2f {
            x: (screen.x - self.offset.x) / self.scale,
            y: (screen.y - self.offset.y) / self.scale,
        }
    }

    /// Convert world coordinates to screen coordinates
    pub fn world_to_screen(&self, world: Vec2f) -> Vec2f {
        Vec2f {
            x: world.x * self.scale + self.offset.x,
            y: world.y * self.scale + self.offset.y,
        }
    }

    /// Zoom centered on a focal point in screen coordinates
    pub fn zoom_at(&mut self, factor: f32, focal: Vec2f) {
        // Convert focal point to world before zoom
        let world_focal = self.screen_to_world(focal);

        // Apply zoom (clamped to reasonable range)
        let new_scale = (self.scale * factor).clamp(0.1, 10.0);
        self.scale = new_scale;

        // Convert focal point back to screen after zoom
        let new_screen_focal = self.world_to_screen(world_focal);

        // Adjust offset so focal point stays in place
        self.offset.x += focal.x - new_screen_focal.x;
        self.offset.y += focal.y - new_screen_focal.y;
    }

    /// Pan by a screen delta
    pub fn pan(&mut self, delta: Vec2f) {
        self.offset.x += delta.x;
        self.offset.y += delta.y;
    }

    /// Reset to default view
    pub fn reset(&mut self) {
        self.offset = Vec2f::default();
        self.scale = 1.0;
    }

    /// Fit content to viewport with optional padding
    pub fn fit_to_content(&mut self, content_width: f32, content_height: f32, padding: f32) {
        let available_width = self.screen_width - padding * 2.0;
        let available_height = self.screen_height - padding * 2.0;

        let scale_x = available_width / content_width;
        let scale_y = available_height / content_height;
        self.scale = scale_x.min(scale_y).min(1.0); // Don't zoom in past 100%

        // Center content
        let scaled_width = content_width * self.scale;
        let scaled_height = content_height * self.scale;
        self.offset.x = (self.screen_width - scaled_width) / 2.0;
        self.offset.y = (self.screen_height - scaled_height) / 2.0;
    }

    /// Get the transformation matrix for WebGL shaders (column-major 4x4)
    pub fn get_transform_matrix(&self) -> [f32; 16] {
        // Validate screen dimensions to avoid division by zero
        if self.screen_width <= 0.0 || self.screen_height <= 0.0 {
            // Return identity matrix if screen is invalid
            return [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ];
        }

        // Transform from world coordinates to clip space (-1 to 1)
        // The matrix combines: scale, translate, and projection to clip space
        let sx = 2.0 * self.scale / self.screen_width;
        let sy = -2.0 * self.scale / self.screen_height; // Y flip for WebGL
        let tx = 2.0 * self.offset.x / self.screen_width - 1.0;
        let ty = -2.0 * self.offset.y / self.screen_height + 1.0;

        // Column-major 4x4 matrix
        [
            sx, 0.0, 0.0, 0.0, // column 0
            0.0, sy, 0.0, 0.0, // column 1
            0.0, 0.0, 1.0, 0.0, // column 2
            tx, ty, 0.0, 1.0, // column 3
        ]
    }

    /// Get visible world bounds
    pub fn get_visible_bounds(&self) -> super::element::Rect {
        let top_left = self.screen_to_world(Vec2f::new(0.0, 0.0));
        let bottom_right = self.screen_to_world(Vec2f::new(self.screen_width, self.screen_height));
        super::element::Rect::new(
            top_left.x,
            top_left.y,
            bottom_right.x - top_left.x,
            bottom_right.y - top_left.y,
        )
    }
}

/// Selection and hover state for interactivity
#[derive(Debug, Default)]
pub struct InteractionState {
    /// Currently selected block IDs (supports multi-select with ctrl/cmd)
    pub selected_blocks: HashSet<String>,

    /// Most recently selected block (for shift-click range selection)
    pub last_selected: Option<String>,

    /// Block currently under the cursor
    pub hovered_block: Option<String>,

    /// Arrow currently under the cursor (from_block, to_block)
    pub hovered_arrow: Option<(String, String)>,

    /// Whether selection changed this frame (for callbacks)
    pub selection_changed: bool,

    /// Whether hover changed this frame (for callbacks)
    pub hover_changed: bool,
}

impl InteractionState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Select a single block, clearing previous selection
    pub fn select_single(&mut self, block_id: &str) {
        self.selected_blocks.clear();
        self.selected_blocks.insert(block_id.to_string());
        self.last_selected = Some(block_id.to_string());
        self.selection_changed = true;
    }

    /// Toggle selection of a block (for ctrl/cmd-click)
    pub fn toggle_selection(&mut self, block_id: &str) {
        if self.selected_blocks.contains(block_id) {
            self.selected_blocks.remove(block_id);
        } else {
            self.selected_blocks.insert(block_id.to_string());
            self.last_selected = Some(block_id.to_string());
        }
        self.selection_changed = true;
    }

    /// Clear all selection
    pub fn clear_selection(&mut self) {
        if !self.selected_blocks.is_empty() {
            self.selected_blocks.clear();
            self.last_selected = None;
            self.selection_changed = true;
        }
    }

    /// Set hover state
    pub fn set_hover(&mut self, block_id: Option<&str>) {
        let new_hover = block_id.map(|s| s.to_string());
        if new_hover != self.hovered_block {
            self.hovered_block = new_hover;
            self.hover_changed = true;
        }
    }

    /// Check if a block is selected
    pub fn is_selected(&self, block_id: &str) -> bool {
        self.selected_blocks.contains(block_id)
    }

    /// Check if a block is hovered
    pub fn is_hovered(&self, block_id: &str) -> bool {
        self.hovered_block.as_deref() == Some(block_id)
    }

    /// Reset change flags (call after processing)
    pub fn reset_change_flags(&mut self) {
        self.selection_changed = false;
        self.hover_changed = false;
    }
}

/// Mouse state for gesture handling
#[derive(Debug, Default)]
pub struct MouseState {
    /// Current mouse position in screen coordinates
    pub position: Vec2f,

    /// Whether currently panning (middle mouse or space+drag)
    pub is_panning: bool,

    /// Position where pan started
    pub pan_start: Vec2f,

    /// Previous position for drag delta calculation
    pub prev_position: Vec2f,
}

impl MouseState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get drag delta since last update
    pub fn get_delta(&self) -> Vec2f {
        Vec2f {
            x: self.position.x - self.prev_position.x,
            y: self.position.y - self.prev_position.y,
        }
    }

    /// Update position and store previous
    pub fn update_position(&mut self, x: f32, y: f32) {
        self.prev_position = self.position;
        self.position = Vec2f::new(x, y);
    }

    /// Start panning
    pub fn start_pan(&mut self) {
        self.is_panning = true;
        self.pan_start = self.position;
    }

    /// Stop panning
    pub fn stop_pan(&mut self) {
        self.is_panning = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewport_coordinates() {
        let viewport = Viewport::new(800.0, 600.0);

        // At default zoom/pan, screen coords should equal world coords
        let world = viewport.screen_to_world(Vec2f::new(100.0, 200.0));
        assert_eq!(world.x, 100.0);
        assert_eq!(world.y, 200.0);
    }

    #[test]
    fn test_viewport_zoom() {
        let mut viewport = Viewport::new(800.0, 600.0);
        viewport.scale = 2.0;

        // At 2x zoom, world coords are half of screen coords (from origin)
        let world = viewport.screen_to_world(Vec2f::new(200.0, 100.0));
        assert_eq!(world.x, 100.0);
        assert_eq!(world.y, 50.0);
    }

    #[test]
    fn test_viewport_pan() {
        let mut viewport = Viewport::new(800.0, 600.0);
        viewport.offset = Vec2f::new(50.0, 100.0);

        // Pan offsets the coordinate system
        let world = viewport.screen_to_world(Vec2f::new(150.0, 200.0));
        assert_eq!(world.x, 100.0);
        assert_eq!(world.y, 100.0);
    }

    #[test]
    fn test_interaction_selection() {
        let mut state = InteractionState::new();

        state.select_single("block1");
        assert!(state.is_selected("block1"));
        assert!(!state.is_selected("block2"));

        state.toggle_selection("block2");
        assert!(state.is_selected("block1"));
        assert!(state.is_selected("block2"));

        state.toggle_selection("block1");
        assert!(!state.is_selected("block1"));
        assert!(state.is_selected("block2"));

        state.clear_selection();
        assert!(!state.is_selected("block2"));
    }
}
