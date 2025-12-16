//! Hit testing and event handling
//!
//! Handles mouse/keyboard events and determines what elements are under the cursor.

use super::element::Rect;
use super::scene::{DrawCallOwner, Scene};
use super::state::{InteractionState, MouseState, Vec2f, Viewport};

/// Hit box for a block (for fast point-in-rect testing)
#[derive(Debug, Clone)]
pub struct BlockHitBox {
    pub block_id: String,
    pub bounds: Rect,
}

/// Hit tester - finds elements at screen positions
pub struct HitTester {
    /// Block bounds in world coordinates
    blocks: Vec<BlockHitBox>,
}

impl HitTester {
    /// Create a new empty hit tester
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Build hit test data from a scene
    pub fn from_scene(scene: &Scene) -> Self {
        let mut blocks = Vec::new();

        for call in &scene.draw_calls {
            if let DrawCallOwner::Block { block_id } = &call.owner {
                // Look for the block background rect (first Rect primitive for each block)
                if let super::scene::Primitive::Rect { rect, .. } = &call.primitive {
                    // Only add if we haven't seen this block yet
                    if !blocks.iter().any(|b: &BlockHitBox| b.block_id == *block_id) {
                        blocks.push(BlockHitBox {
                            block_id: block_id.clone(),
                            bounds: *rect,
                        });
                    }
                }
            }
        }

        Self { blocks }
    }

    /// Find which block is at a world position
    pub fn find_block_at(&self, world_pos: Vec2f) -> Option<&str> {
        // Search in reverse order (later blocks are on top)
        for block in self.blocks.iter().rev() {
            if block.bounds.contains(world_pos.x, world_pos.y) {
                return Some(&block.block_id);
            }
        }
        None
    }

    /// Get all blocks
    pub fn blocks(&self) -> &[BlockHitBox] {
        &self.blocks
    }

    /// Clear hit test data
    pub fn clear(&mut self) {
        self.blocks.clear();
    }
}

impl Default for HitTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Event handler for mouse and keyboard input
pub struct EventHandler {
    /// Viewport state
    pub viewport: Viewport,

    /// Interaction state (selection, hover)
    pub interaction: InteractionState,

    /// Mouse state
    pub mouse: MouseState,

    /// Hit tester for finding blocks
    pub hit_tester: HitTester,

    /// Whether a redraw is needed
    pub needs_redraw: bool,
}

impl EventHandler {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            viewport: Viewport::new(width, height),
            interaction: InteractionState::new(),
            mouse: MouseState::new(),
            hit_tester: HitTester::new(),
            needs_redraw: true,
        }
    }

    /// Update hit tester from scene
    pub fn update_hit_tester(&mut self, scene: &Scene) {
        self.hit_tester = HitTester::from_scene(scene);
    }

    /// Handle mouse down event
    pub fn handle_mouse_down(&mut self, x: f32, y: f32, button: i16, ctrl: bool, shift: bool) {
        self.mouse.update_position(x, y);

        match button {
            0 => {
                // Left click
                let world_pos = self.viewport.screen_to_world(Vec2f::new(x, y));

                if let Some(block_id) = self.hit_tester.find_block_at(world_pos) {
                    if ctrl {
                        // Toggle selection
                        self.interaction.toggle_selection(block_id);
                    } else if shift {
                        // Add to selection (for range select, simplified here)
                        self.interaction.selected_blocks.insert(block_id.to_string());
                        self.interaction.selection_changed = true;
                    } else {
                        // Single select
                        self.interaction.select_single(block_id);
                    }
                    self.needs_redraw = true;
                } else {
                    // Click on empty space - start panning or clear selection
                    if !ctrl && !shift {
                        self.interaction.clear_selection();
                    }
                    self.mouse.start_pan();
                    self.needs_redraw = true;
                }
            }
            1 => {
                // Middle click - start panning
                self.mouse.start_pan();
            }
            _ => {}
        }
    }

    /// Handle mouse up event
    pub fn handle_mouse_up(&mut self, _x: f32, _y: f32, _button: i16) {
        self.mouse.stop_pan();
    }

    /// Handle mouse move event
    pub fn handle_mouse_move(&mut self, x: f32, y: f32) {
        self.mouse.update_position(x, y);

        if self.mouse.is_panning {
            let delta = self.mouse.get_delta();
            self.viewport.pan(delta);
            self.needs_redraw = true;
        } else {
            // Update hover state
            let world_pos = self.viewport.screen_to_world(Vec2f::new(x, y));
            let block_id = self.hit_tester.find_block_at(world_pos);
            self.interaction.set_hover(block_id);

            if self.interaction.hover_changed {
                self.needs_redraw = true;
            }
        }
    }

    /// Handle mouse wheel event
    pub fn handle_wheel(&mut self, x: f32, y: f32, delta_x: f32, delta_y: f32, ctrl: bool) {
        if ctrl {
            // Zoom - delta_y controls zoom factor
            let zoom_factor = (-delta_y * 0.01).exp();
            self.viewport.zoom_at(zoom_factor, Vec2f::new(x, y));
        } else {
            // Pan
            self.viewport.pan(Vec2f::new(-delta_x, -delta_y));
        }
        self.needs_redraw = true;
    }

    /// Handle key down event, returns true if event was handled
    pub fn handle_key_down(&mut self, key: &str, ctrl: bool, _shift: bool) -> KeyAction {
        match key {
            "Escape" => {
                self.interaction.clear_selection();
                self.needs_redraw = true;
                KeyAction::Handled
            }
            "0" if ctrl => {
                // Reset zoom
                self.viewport.reset();
                self.needs_redraw = true;
                KeyAction::Handled
            }
            "=" | "+" if ctrl => {
                // Zoom in
                let center = Vec2f::new(
                    self.viewport.screen_width / 2.0,
                    self.viewport.screen_height / 2.0,
                );
                self.viewport.zoom_at(1.2, center);
                self.needs_redraw = true;
                KeyAction::Handled
            }
            "-" if ctrl => {
                // Zoom out
                let center = Vec2f::new(
                    self.viewport.screen_width / 2.0,
                    self.viewport.screen_height / 2.0,
                );
                self.viewport.zoom_at(0.8, center);
                self.needs_redraw = true;
                KeyAction::Handled
            }
            "ArrowRight" | "f" => {
                // Next pass (handled by viewer)
                KeyAction::NextPass
            }
            "ArrowLeft" | "r" => {
                // Previous pass (handled by viewer)
                KeyAction::PrevPass
            }
            "ArrowUp" => {
                // Previous function (handled by viewer)
                KeyAction::PrevFunction
            }
            "ArrowDown" => {
                // Next function (handled by viewer)
                KeyAction::NextFunction
            }
            _ => KeyAction::Unhandled,
        }
    }

    /// Handle window resize
    pub fn handle_resize(&mut self, width: f32, height: f32) {
        self.viewport.screen_width = width;
        self.viewport.screen_height = height;
        self.needs_redraw = true;
    }

    /// Check and reset redraw flag
    pub fn take_needs_redraw(&mut self) -> bool {
        let needs = self.needs_redraw;
        self.needs_redraw = false;
        needs
    }

    /// Reset interaction change flags
    pub fn reset_change_flags(&mut self) {
        self.interaction.reset_change_flags();
    }
}

/// Result of key handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Event was handled internally
    Handled,
    /// Event was not handled
    Unhandled,
    /// Navigate to next pass
    NextPass,
    /// Navigate to previous pass
    PrevPass,
    /// Navigate to next function
    NextFunction,
    /// Navigate to previous function
    PrevFunction,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_tester() {
        let mut tester = HitTester::new();
        tester.blocks.push(BlockHitBox {
            block_id: "block1".to_string(),
            bounds: Rect::new(10.0, 20.0, 100.0, 50.0),
        });
        tester.blocks.push(BlockHitBox {
            block_id: "block2".to_string(),
            bounds: Rect::new(50.0, 60.0, 100.0, 50.0),
        });

        // Point inside block1
        assert_eq!(
            tester.find_block_at(Vec2f::new(50.0, 40.0)),
            Some("block1")
        );

        // Point inside block2
        assert_eq!(
            tester.find_block_at(Vec2f::new(100.0, 80.0)),
            Some("block2")
        );

        // Point outside both
        assert_eq!(tester.find_block_at(Vec2f::new(5.0, 5.0)), None);
    }

    #[test]
    fn test_event_handler_selection() {
        let mut handler = EventHandler::new(800.0, 600.0);
        handler.hit_tester.blocks.push(BlockHitBox {
            block_id: "block1".to_string(),
            bounds: Rect::new(10.0, 20.0, 100.0, 50.0),
        });

        // Click on block
        handler.handle_mouse_down(50.0, 40.0, 0, false, false);
        assert!(handler.interaction.is_selected("block1"));

        // Escape to clear
        handler.handle_key_down("Escape", false, false);
        assert!(!handler.interaction.is_selected("block1"));
    }
}
