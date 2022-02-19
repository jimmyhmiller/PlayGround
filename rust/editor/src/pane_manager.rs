use std::cmp::{max, min};

use rand::Rng;

use crate::{pane::{TextPane, Pane, AdjustablePosition}, Window, renderer::EditorBounds, ink::InkManager};


#[derive(Debug, Clone, Copy)]
pub enum PaneSelector {
    Active,
    Id(usize),
    AtMouse((i32, i32)),
    Scroll,
}


 // Does the manager include the blank space?
 // That might be a good conceptual model as I try to do things like ink.
 // Maybe if I had boards like muse, each pane manager would be each board
pub struct PaneManager {
    pub panes: Vec<Pane>,
    pub active_pane: usize,
    pub scroll_active_pane: Option<usize>,
    pub window: Window,
    pub dragging_start: (i32, i32),
    pub dragging_pane: Option<usize>,
    pub dragging_pane_start: (i32, i32),
    pub resize_start: (i32, i32),
    pub resize_pane: Option<usize>,
    pub create_pane_activated: bool,
    pub create_pane_start: (i32, i32),
    pub create_pane_current: (i32, i32),
    pub pane_id_counter: usize,
    pub scale_factor: f32,
    pub ink_manager: InkManager,
}

impl PaneManager {

    pub fn new(panes: Vec<Pane>, window: Window) -> Self {
        PaneManager {
            panes,
            window,
            active_pane: 0,
            scroll_active_pane: None,
            dragging_pane: None,
            dragging_start: (0, 0),
            dragging_pane_start: (0, 0),
            resize_pane: None,
            resize_start: (0, 0),
            create_pane_activated: false,
            create_pane_start: (0, 0),
            create_pane_current: (0, 0),
            pane_id_counter: 0,
            scale_factor: 1.0,
            ink_manager: InkManager::new(),
        }
    }

    pub fn get_pane_index_by_id(&self, pane_id: usize) -> Option<usize> {
        for (index, pane) in self.panes.iter().enumerate() {
            if pane.id() == pane_id {
                return Some(index);
            }
        }
        None
    }

    pub fn delete_pane(&mut self, pane_id: usize) {
        if let Some(pane) = self.get_pane_index_by_id(pane_id) {
            self.panes.remove(pane);
        }
    }



    pub fn set_active_by_id(&mut self, pane_id: usize) {
        if let Some(i) = self.get_pane_index_by_id(pane_id) {
            self.active_pane = i;
            if let Some(pane) = self.panes.get_mut(i) {
                pane.set_active(true);
            }
            for pane in self.panes.iter_mut() {
                if pane.id() != pane_id {
                    pane.set_active(false);
                }
            }
        }
    }

    pub fn set_scroll_active_by_id(&mut self, pane_id: usize) {
        if let Some(i) = self.get_pane_index_by_id(pane_id) {
            self.scroll_active_pane = Some(i);
        }
    }

    pub fn clear_scroll_active(&mut self) {
        self.scroll_active_pane = None;
    }

    pub fn get_pane_index_at_mouse(&self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> Option<usize> {
        for (i, pane) in self.panes.iter().enumerate().rev() {
            if pane.is_mouse_over(mouse_pos, bounds, self.scale_factor) {
                return Some(i);
            }
        }
        None
    }

    pub fn get_pane_at_mouse_mut(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> Option<&mut Pane> {
        if let Some(i) = self.get_pane_index_at_mouse(mouse_pos, bounds) {
            return self.panes.get_mut(i);
        }
        None
    }
    pub fn get_pane_at_mouse(&self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> Option<&Pane> {
        if let Some(i) = self.get_pane_index_at_mouse(mouse_pos, bounds) {
            return self.panes.get(i);
        }
        None
    }

    // TODO: This should be a Option<Pane> because there might not be any.
    pub fn get_active_pane_mut(&mut self) -> Option<&mut Pane> {
        self.panes.get_mut(self.active_pane)
    }

    pub fn get_scroll_active_pane_mut(&mut self) -> Option<&mut Pane> {
        self.panes.get_mut(self.scroll_active_pane?)
    }
    pub fn get_scroll_active_pane(&self) -> Option<&Pane> {
        self.panes.get(self.scroll_active_pane?)
    }

    pub fn get_active_pane(&self) -> Option<&Pane> {
        self.panes.get(self.active_pane)
    }

    pub fn get_dragging_pane_mut(&mut self) -> Option<&mut Pane> {
        if let Some(i) = self.dragging_pane {
            Some(&mut self.panes[i])
        } else {
            None
        }
    }

    pub fn get_resize_pane_mut(&mut self) -> Option<&mut Pane> {
        if let Some(i) = self.resize_pane {
            Some(&mut self.panes[i])
        } else {
            None
        }
    }

    pub fn set_dragging_start(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> bool {
        if let Some(i) = self.get_pane_index_at_mouse(mouse_pos, bounds) {
            let pane = self.panes.remove(i);
            self.panes.push(pane);

            let new_i = self.panes.len() - 1;
            if self.active_pane == i {
                self.active_pane = new_i;
            }
            self.dragging_start = mouse_pos;
            self.dragging_pane = Some(new_i);
            self.dragging_pane_start = self.panes[new_i].position();
            return true
        }
        false
    }

    pub fn update_dragging_position(&mut self, mouse_pos: (i32, i32)) {
        let (x, y) = mouse_pos;
        let (x_diff, y_diff) = (x - self.dragging_start.0, y - self.dragging_start.1);
        let (pane_x, pane_y) = self.dragging_pane_start;
        if let Some(pane) = self.get_dragging_pane_mut() {
            pane.set_position(pane_x as i32 + x_diff, pane_y as i32 + y_diff);
        }
    }

    pub fn stop_dragging(&mut self) {
        self.dragging_pane = None;
    }

    pub fn set_resize_start(&mut self, mouse_pos: (i32, i32), pane_id: usize) -> bool {
        if let Some(i) = self.get_pane_index_by_id(pane_id) {
            self.resize_start = mouse_pos;
            self.resize_pane = Some(i);
            self.update_resize_size(mouse_pos);
            true
        } else {
            false
        }
    }

    pub fn update_resize_size(&mut self, mouse_pos: (i32, i32)) {
        let (x, y) = mouse_pos;

        if let Some(pane) = self.get_resize_pane_mut() {

            let (current_x, current_y) = pane.position();

            if x < current_x || y < current_y {
                return;
            }

            let width = x - current_x;
            let height = y - current_y;

            pane.set_width(max(width as usize, 20));
            pane.set_height(max(height as usize, 20));
        }
    }

    pub fn stop_resizing(&mut self) {
        self.resize_pane = None;
    }

    pub fn set_create_start(&mut self, mouse_pos: (i32, i32)) {
        self.create_pane_activated = true;
        self.create_pane_start = mouse_pos;
        self.create_pane_current = mouse_pos;
    }

    pub fn update_create_pane(&mut self, mouse_pos: (i32, i32)) {
        if self.create_pane_activated {
            self.create_pane_current = mouse_pos;
        }
    }

    pub fn create_pane_raw(&mut self, pane_name: String, position: (i32, i32), width: usize, height: usize) -> usize{
        let id = self.new_pane_id();
        // Maybe I should have a pane type selector here.

        // self.panes.push(Pane::Empty(EmptyPane {
        //     id,
        //     name: pane_name,
        //     position,
        //     width,
        //     height,
        //     active: true,
        // }));

        // Can I use Pane::new here?
        
        self.panes.push(Pane::Text(
            TextPane::new(id, pane_name, position, (width, height), "", true)
        ));
        self.panes.len() - 1
    }

    pub fn create_pane(&mut self) {
        if self.create_pane_activated {
            self.create_pane_activated = false;


            let position_x = min(self.create_pane_start.0, self.create_pane_current.0);
            let position_y = min(self.create_pane_start.1, self.create_pane_current.1);
            let current_x = max(self.create_pane_start.0, self.create_pane_current.0);
            let current_y = max(self.create_pane_start.1, self.create_pane_current.1);
            let width = (current_x - position_x) as usize;
            let height = (current_y - position_y) as usize;
            if width < 20 || height < 20 {
                return;
            }

            let mut rng = rand::thread_rng();
            self.create_pane_raw(format!("temp-{}", rng.gen::<u32>()), (position_x, position_y), width, height);
            self.active_pane = self.panes.len() - 1;
        }
    }

    pub fn _remove(&mut self, i: usize) -> Pane {
        // Should swap_remove but more complicated
        let pane = self.panes.remove(i);
        if i < self.active_pane {
            self.active_pane -= 1;
        }
        pane
    }

    pub fn _insert(&mut self, i: usize, pane: Pane) {
        if pane._active() {
            self.active_pane = i;
        }
        self.panes.insert(i, pane);
    }


    pub fn _get_pane_mut(&mut self, index: usize) -> Option<&mut Pane> {
        self.panes.get_mut(index)
    }

    pub fn get_pane_by_name_mut(&mut self, pane_name: String) -> Option<&mut Pane> {
        for pane in self.panes.iter_mut() {
            if pane.name().starts_with(&pane_name) {
                return Some(pane);
            }
        }
        None
    }


    pub fn get_pane_by_id_mut(&mut self, pane_id: usize) -> Option<&mut Pane> {
        for pane in self.panes.iter_mut() {
            if pane.id() == pane_id {
                return Some(pane);
            }
        }
        None
    }
    pub fn get_pane_by_id(&self, pane_id: usize) -> Option<&Pane> {
        for pane in self.panes.iter() {
            if pane.id() == pane_id {
                return Some(pane);
            }
        }
        None
    }

    pub fn get_pane_by_name(&self, pane_name: &str) -> Option<&Pane> {
        for pane in self.panes.iter() {
            if pane.name().starts_with(&pane_name) {
                return Some(pane);
            }
        }
        None
    }

    pub fn new_pane_id(&mut self) -> usize {
        self.pane_id_counter += 1;
        self.pane_id_counter
    }

    pub fn get_pane_by_selector_mut(&mut self, pane_selector: &PaneSelector, editor_bounds: &EditorBounds) -> Option<&mut Pane> {
        match pane_selector {
            PaneSelector::Active => self.get_active_pane_mut(),
            PaneSelector::Id(id) => self.get_pane_by_id_mut(*id),
            PaneSelector::AtMouse(mouse_pos) => self.get_pane_at_mouse_mut(*mouse_pos, editor_bounds),
            PaneSelector::Scroll => self.get_scroll_active_pane_mut(),
        }
    }

    pub fn get_pane_by_selector(&self, pane_selector: &PaneSelector, editor_bounds: &EditorBounds) -> Option<&Pane> {
        match pane_selector {
            PaneSelector::Active => self.get_active_pane(),
            PaneSelector::Id(id) => self.get_pane_by_id(*id),
            PaneSelector::AtMouse(mouse_pos) => self.get_pane_at_mouse(*mouse_pos, editor_bounds),
            PaneSelector::Scroll => self.get_scroll_active_pane(),
        }
    }

    pub fn clear_active(&mut self) {
        self.active_pane = 0;
        for pane in self.panes.iter_mut() {
            pane.set_active(false);
        }
    }
}