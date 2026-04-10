use crate::anim::Easing;
use crate::animated::AnimatedValue;
use crate::scene::{GroupNode, Node};
use crate::tweakables::Tweakables;

/// Configuration for how items enter the collection.
pub struct EnterAnim {
    pub scale_from: f64,
    pub opacity_from: f64,
    pub duration: f64,
    pub easing: Easing,
}

impl Default for EnterAnim {
    fn default() -> Self {
        Self {
            scale_from: 0.0,
            opacity_from: 0.0,
            duration: 0.4,
            easing: Easing::BackOut,
        }
    }
}

/// Configuration for how items exit the collection.
pub struct ExitAnim {
    pub scale_to: f64,
    pub opacity_to: f64,
    pub duration: f64,
    pub easing: Easing,
}

impl Default for ExitAnim {
    fn default() -> Self {
        Self {
            scale_to: 0.0,
            opacity_to: 0.0,
            duration: 0.3,
            easing: Easing::CubicIn,
        }
    }
}

enum ItemState {
    Alive,
    Exiting { remaining: f64 },
}

struct CollectionItem {
    state: ItemState,
}

/// A managed collection of nodes with automatic enter/exit animations.
///
/// Wraps a GroupNode. When you push a node, it gets enter animations applied.
/// When you remove a node, it plays exit animations and is cleaned up after.
pub struct Collection {
    pub group: GroupNode,
    items: Vec<CollectionItem>,
    enter: EnterAnim,
    exit: ExitAnim,
}

impl Collection {
    pub fn new(id: &str) -> Self {
        let mut group = GroupNode::new();
        group.props.id = Some(id.to_string());
        Self {
            group,
            items: Vec::new(),
            enter: EnterAnim::default(),
            exit: ExitAnim::default(),
        }
    }

    pub fn with_enter(mut self, enter: EnterAnim) -> Self {
        self.enter = enter;
        self
    }

    pub fn with_exit(mut self, exit: ExitAnim) -> Self {
        self.exit = exit;
        self
    }

    /// Add a node to the collection with enter animation.
    /// Returns the index of the new item.
    pub fn push(&mut self, mut node: Node) -> usize {
        let props = node.props_mut();

        // Apply enter animation to scale
        props.scale = AnimatedValue::tween(
            self.enter.scale_from,
            1.0,
            self.enter.duration,
            self.enter.easing,
        );
        props.scale.fire();

        // Apply enter animation to opacity
        props.opacity = AnimatedValue::spring(self.enter.opacity_from, 200.0, 15.0);
        props.opacity.set_target(1.0);

        let idx = self.group.children.len();
        self.group.children.push(node);
        self.items.push(CollectionItem { state: ItemState::Alive });
        idx
    }

    /// Add a node without enter animation (already visible).
    pub fn push_immediate(&mut self, node: Node) -> usize {
        let idx = self.group.children.len();
        self.group.children.push(node);
        self.items.push(CollectionItem { state: ItemState::Alive });
        idx
    }

    /// Mark a node for removal. It will play exit animations and be cleaned up.
    pub fn remove(&mut self, idx: usize) {
        if idx >= self.items.len() {
            return;
        }
        if matches!(self.items[idx].state, ItemState::Exiting { .. }) {
            return; // already exiting
        }

        self.items[idx].state = ItemState::Exiting {
            remaining: self.exit.duration,
        };

        // Start exit animation
        let props = self.group.children[idx].props_mut();
        props.scale = AnimatedValue::tween(
            1.0,
            self.exit.scale_to,
            self.exit.duration,
            self.exit.easing,
        );
        props.scale.fire();
        props.opacity = AnimatedValue::spring(1.0, 200.0, 15.0);
        props.opacity.set_target(self.exit.opacity_to);
    }

    /// Get a reference to a live node by index.
    pub fn get(&self, idx: usize) -> Option<&Node> {
        if idx < self.items.len() && matches!(self.items[idx].state, ItemState::Alive) {
            Some(&self.group.children[idx])
        } else {
            None
        }
    }

    /// Get a mutable reference to a live node by index.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Node> {
        if idx < self.items.len() && matches!(self.items[idx].state, ItemState::Alive) {
            Some(&mut self.group.children[idx])
        } else {
            None
        }
    }

    /// Number of alive items (not counting exiting ones).
    pub fn alive_count(&self) -> usize {
        self.items.iter().filter(|i| matches!(i.state, ItemState::Alive)).count()
    }

    /// Total items including exiting.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Advance all animations and clean up finished exits.
    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        // Tick all child nodes
        for child in &mut self.group.children {
            child.tick(dt, tw);
        }

        // Count down exit timers
        for item in &mut self.items {
            if let ItemState::Exiting { remaining } = &mut item.state {
                *remaining -= dt;
            }
        }

        // Remove completed exits (iterate backwards to preserve indices)
        let mut i = self.items.len();
        while i > 0 {
            i -= 1;
            if let ItemState::Exiting { remaining } = &self.items[i].state {
                if *remaining <= 0.0 {
                    self.items.remove(i);
                    self.group.children.remove(i);
                }
            }
        }
    }
}
