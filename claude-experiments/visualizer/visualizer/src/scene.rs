use std::collections::HashMap;

use vello::kurbo::{Affine, Circle, Point, RoundedRect, Stroke};
use vello::peniko::{Color, Fill};
use vello::Scene;

use crate::animated::{AnimatedColor, AnimatedPos, AnimatedValue};
use crate::theme;
use crate::tweakables::Tweakables;

/// Common properties shared by all nodes.
pub struct NodeProps {
    pub id: Option<String>,
    pub pos: AnimatedPos,
    pub scale: AnimatedValue,
    pub opacity: AnimatedValue,
    pub visible: bool,
}

impl NodeProps {
    fn tick(&mut self, dt: f64, tw: &Tweakables) {
        self.pos.tick(dt, tw);
        self.scale.tick(dt, tw);
        self.opacity.tick(dt, tw);
    }
}

impl Default for NodeProps {
    fn default() -> Self {
        Self {
            id: None,
            pos: AnimatedPos::constant(0.0, 0.0),
            scale: AnimatedValue::constant(1.0),
            opacity: AnimatedValue::constant(1.0),
            visible: true,
        }
    }
}

pub struct RectNode {
    pub props: NodeProps,
    pub width: AnimatedValue,
    pub height: AnimatedValue,
    pub corner_radius: AnimatedValue,
    pub fill: AnimatedColor,
}

pub struct CircleNode {
    pub props: NodeProps,
    pub radius: AnimatedValue,
    pub fill: AnimatedColor,
}

pub struct TriangleNode {
    pub props: NodeProps,
    pub size: AnimatedValue,
    pub fill: AnimatedColor,
}

pub struct LineNode {
    pub props: NodeProps,
    pub x2: AnimatedValue,
    pub y2: AnimatedValue,
    pub stroke_width: AnimatedValue,
    pub color: AnimatedColor,
    pub from_node: Option<String>,
    pub to_node: Option<String>,
}

pub struct ArrowNode {
    pub props: NodeProps,
    pub x2: AnimatedValue,
    pub y2: AnimatedValue,
    pub stroke_width: AnimatedValue,
    pub head_size: AnimatedValue,
    pub color: AnimatedColor,
    pub from_node: Option<String>,
    pub to_node: Option<String>,
}

pub struct GroupNode {
    pub props: NodeProps,
    pub children: Vec<Node>,
}

pub enum Node {
    Rect(RectNode),
    Circle(CircleNode),
    Triangle(TriangleNode),
    Line(LineNode),
    Arrow(ArrowNode),
    Group(GroupNode),
}

// ── Constructors ──

impl RectNode {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self {
            props: NodeProps {
                pos: AnimatedPos::constant(x, y),
                ..Default::default()
            },
            width: AnimatedValue::constant(w),
            height: AnimatedValue::constant(h),
            corner_radius: AnimatedValue::constant(0.0),
            fill: AnimatedColor::constant(1.0, 1.0, 1.0, 1.0),
        }
    }
}

impl CircleNode {
    pub fn new(x: f64, y: f64, r: f64) -> Self {
        Self {
            props: NodeProps {
                pos: AnimatedPos::constant(x, y),
                ..Default::default()
            },
            radius: AnimatedValue::constant(r),
            fill: AnimatedColor::constant(1.0, 1.0, 1.0, 1.0),
        }
    }
}

impl TriangleNode {
    pub fn new(x: f64, y: f64, size: f64) -> Self {
        Self {
            props: NodeProps {
                pos: AnimatedPos::constant(x, y),
                ..Default::default()
            },
            size: AnimatedValue::constant(size),
            fill: AnimatedColor::constant(1.0, 1.0, 1.0, 1.0),
        }
    }
}

impl LineNode {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self {
            props: NodeProps {
                pos: AnimatedPos::constant(x1, y1),
                ..Default::default()
            },
            x2: AnimatedValue::constant(x2),
            y2: AnimatedValue::constant(y2),
            stroke_width: AnimatedValue::constant(2.0),
            color: AnimatedColor::constant(1.0, 1.0, 1.0, 1.0),
            from_node: None,
            to_node: None,
        }
    }
}

impl ArrowNode {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self {
            props: NodeProps {
                pos: AnimatedPos::constant(x1, y1),
                ..Default::default()
            },
            x2: AnimatedValue::constant(x2),
            y2: AnimatedValue::constant(y2),
            stroke_width: AnimatedValue::constant(2.0),
            head_size: AnimatedValue::constant(8.0),
            color: AnimatedColor::constant(1.0, 1.0, 1.0, 1.0),
            from_node: None,
            to_node: None,
        }
    }
}

impl GroupNode {
    pub fn new() -> Self {
        Self {
            props: Default::default(),
            children: Vec::new(),
        }
    }

    pub fn add(&mut self, node: Node) {
        self.children.push(node);
    }

    pub fn add_rect(&mut self, rect: RectNode) -> usize {
        let idx = self.children.len();
        self.children.push(Node::Rect(rect));
        idx
    }

    pub fn add_circle(&mut self, circle: CircleNode) -> usize {
        let idx = self.children.len();
        self.children.push(Node::Circle(circle));
        idx
    }

    pub fn add_triangle(&mut self, tri: TriangleNode) -> usize {
        let idx = self.children.len();
        self.children.push(Node::Triangle(tri));
        idx
    }

    pub fn add_group(&mut self, group: GroupNode) -> usize {
        let idx = self.children.len();
        self.children.push(Node::Group(group));
        idx
    }
}

// ── Node enum accessors ──

impl Node {
    pub fn props(&self) -> &NodeProps {
        match self {
            Node::Rect(n) => &n.props,
            Node::Circle(n) => &n.props,
            Node::Triangle(n) => &n.props,
            Node::Line(n) => &n.props,
            Node::Arrow(n) => &n.props,
            Node::Group(n) => &n.props,
        }
    }

    pub fn props_mut(&mut self) -> &mut NodeProps {
        match self {
            Node::Rect(n) => &mut n.props,
            Node::Circle(n) => &mut n.props,
            Node::Triangle(n) => &mut n.props,
            Node::Line(n) => &mut n.props,
            Node::Arrow(n) => &mut n.props,
            Node::Group(n) => &mut n.props,
        }
    }

    /// Find a node by ID in this subtree.
    pub fn find(&self, id: &str) -> Option<&Node> {
        if self.props().id.as_deref() == Some(id) {
            return Some(self);
        }
        if let Node::Group(g) = self {
            for child in &g.children {
                if let Some(found) = child.find(id) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Find a mutable node by ID in this subtree.
    pub fn find_mut(&mut self, id: &str) -> Option<&mut Node> {
        if self.props().id.as_deref() == Some(id) {
            return Some(self);
        }
        if let Node::Group(g) = self {
            for child in &mut g.children {
                if let Some(found) = child.find_mut(id) {
                    return Some(found);
                }
            }
        }
        None
    }

    pub fn as_rect(&self) -> Option<&RectNode> {
        match self { Node::Rect(n) => Some(n), _ => None }
    }

    pub fn as_rect_mut(&mut self) -> Option<&mut RectNode> {
        match self { Node::Rect(n) => Some(n), _ => None }
    }

    pub fn as_circle(&self) -> Option<&CircleNode> {
        match self { Node::Circle(n) => Some(n), _ => None }
    }

    pub fn as_circle_mut(&mut self) -> Option<&mut CircleNode> {
        match self { Node::Circle(n) => Some(n), _ => None }
    }

    pub fn as_line_mut(&mut self) -> Option<&mut LineNode> {
        match self { Node::Line(n) => Some(n), _ => None }
    }

    pub fn as_arrow_mut(&mut self) -> Option<&mut ArrowNode> {
        match self { Node::Arrow(n) => Some(n), _ => None }
    }

    pub fn as_group(&self) -> Option<&GroupNode> {
        match self { Node::Group(n) => Some(n), _ => None }
    }

    pub fn as_group_mut(&mut self) -> Option<&mut GroupNode> {
        match self { Node::Group(n) => Some(n), _ => None }
    }
}

// ── Tick ──

impl Node {
    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        match self {
            Node::Rect(n) => {
                n.props.tick(dt, tw);
                n.width.tick(dt, tw);
                n.height.tick(dt, tw);
                n.corner_radius.tick(dt, tw);
                n.fill.tick(dt, tw);
            }
            Node::Circle(n) => {
                n.props.tick(dt, tw);
                n.radius.tick(dt, tw);
                n.fill.tick(dt, tw);
            }
            Node::Triangle(n) => {
                n.props.tick(dt, tw);
                n.size.tick(dt, tw);
                n.fill.tick(dt, tw);
            }
            Node::Line(n) => {
                n.props.tick(dt, tw);
                n.x2.tick(dt, tw);
                n.y2.tick(dt, tw);
                n.stroke_width.tick(dt, tw);
                n.color.tick(dt, tw);
            }
            Node::Arrow(n) => {
                n.props.tick(dt, tw);
                n.x2.tick(dt, tw);
                n.y2.tick(dt, tw);
                n.stroke_width.tick(dt, tw);
                n.head_size.tick(dt, tw);
                n.color.tick(dt, tw);
            }
            Node::Group(n) => {
                n.props.tick(dt, tw);
                for child in &mut n.children {
                    child.tick(dt, tw);
                }
            }
        }
    }
}

// ── Draw ──

impl Node {
    pub fn draw(&self, scene: &mut Scene, tw: &Tweakables, parent_opacity: f32) {
        let props = self.props();
        if !props.visible {
            return;
        }

        let opacity = props.opacity.get(tw) as f32 * parent_opacity;
        if opacity < 0.001 {
            return;
        }

        let scale = props.scale.get(tw);
        if scale < 0.001 {
            return;
        }

        let [px, py] = props.pos.get(tw);

        let th = theme::current();
        let stroke_rgba = th.stroke;
        let stroke_width = th.stroke_width;
        let stroke_color =
            Color::new([stroke_rgba[0], stroke_rgba[1], stroke_rgba[2], stroke_rgba[3] * opacity]);

        match self {
            Node::Rect(n) => {
                let w = n.width.get(tw) * scale;
                let h = n.height.get(tw) * scale;
                let r = n.corner_radius.get(tw) * scale;
                let [cr, cg, cb, ca] = n.fill.get(tw);

                let x = px - w / 2.0;
                let y = py - h / 2.0;
                let rect = RoundedRect::new(x, y, x + w, y + h, r);
                scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    Color::new([cr, cg, cb, ca * opacity]),
                    None,
                    &rect,
                );
                if stroke_width > 0.0 {
                    scene.stroke(
                        &Stroke::new(stroke_width),
                        Affine::IDENTITY,
                        stroke_color,
                        None,
                        &rect,
                    );
                }
            }
            Node::Circle(n) => {
                let r = n.radius.get(tw) * scale;
                let [cr, cg, cb, ca] = n.fill.get(tw);

                let circle = Circle::new(Point::new(px, py), r);
                scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    Color::new([cr, cg, cb, ca * opacity]),
                    None,
                    &circle,
                );
                if stroke_width > 0.0 {
                    scene.stroke(
                        &Stroke::new(stroke_width),
                        Affine::IDENTITY,
                        stroke_color,
                        None,
                        &circle,
                    );
                }
            }
            Node::Triangle(n) => {
                let s = n.size.get(tw) * scale;
                let [cr, cg, cb, ca] = n.fill.get(tw);

                let path = vello::kurbo::BezPath::from_vec(vec![
                    vello::kurbo::PathEl::MoveTo(Point::new(px, py + s)),
                    vello::kurbo::PathEl::LineTo(Point::new(px - s, py - s)),
                    vello::kurbo::PathEl::LineTo(Point::new(px + s, py - s)),
                    vello::kurbo::PathEl::ClosePath,
                ]);
                scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    Color::new([cr, cg, cb, ca * opacity]),
                    None,
                    &path,
                );
                if stroke_width > 0.0 {
                    scene.stroke(
                        &Stroke::new(stroke_width),
                        Affine::IDENTITY,
                        stroke_color,
                        None,
                        &path,
                    );
                }
            }
            Node::Line(n) => {
                let x2 = n.x2.get(tw);
                let y2 = n.y2.get(tw);
                let sw = n.stroke_width.get(tw) * scale;
                let [cr, cg, cb, ca] = n.color.get(tw);
                let color = Color::new([cr, cg, cb, ca * opacity]);

                let path = vello::kurbo::Line::new(
                    Point::new(px, py),
                    Point::new(x2, y2),
                );
                scene.stroke(
                    &Stroke::new(sw),
                    Affine::IDENTITY,
                    color,
                    None,
                    &path,
                );
            }
            Node::Arrow(n) => {
                let x2 = n.x2.get(tw);
                let y2 = n.y2.get(tw);
                let sw = n.stroke_width.get(tw) * scale;
                let hs = n.head_size.get(tw) * scale;
                let [cr, cg, cb, ca] = n.color.get(tw);
                let color = Color::new([cr, cg, cb, ca * opacity]);

                // Draw the line
                let line = vello::kurbo::Line::new(
                    Point::new(px, py),
                    Point::new(x2, y2),
                );
                scene.stroke(
                    &Stroke::new(sw),
                    Affine::IDENTITY,
                    color,
                    None,
                    &line,
                );

                // Draw the arrowhead as a filled triangle
                let dx = x2 - px;
                let dy = y2 - py;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 0.001 {
                    let ux = dx / len;
                    let uy = dy / len;
                    // Perpendicular
                    let nx = -uy;
                    let ny = ux;
                    let tip = Point::new(x2, y2);
                    let left = Point::new(x2 - ux * hs + nx * hs * 0.5, y2 - uy * hs + ny * hs * 0.5);
                    let right = Point::new(x2 - ux * hs - nx * hs * 0.5, y2 - uy * hs - ny * hs * 0.5);

                    let head = vello::kurbo::BezPath::from_vec(vec![
                        vello::kurbo::PathEl::MoveTo(tip),
                        vello::kurbo::PathEl::LineTo(left),
                        vello::kurbo::PathEl::LineTo(right),
                        vello::kurbo::PathEl::ClosePath,
                    ]);
                    scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &head);
                }
            }
            Node::Group(n) => {
                for child in &n.children {
                    child.draw(scene, tw, opacity);
                }
            }
        }
    }
}

/// The top-level scene graph. Just a root group.
pub struct SceneGraph {
    pub root: GroupNode,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            root: GroupNode::new(),
        }
    }

    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        for child in &mut self.root.children {
            child.tick(dt, tw);
        }
        // Resolve line/arrow connections: update endpoints from referenced node positions
        self.resolve_connections(tw);
    }

    /// Collect id→position for all named nodes, then update any line/arrow
    /// that has from_node/to_node set.
    fn resolve_connections(&mut self, tw: &Tweakables) {
        let mut positions: HashMap<String, [f64; 2]> = HashMap::new();
        Self::collect_positions(&self.root.children, tw, &mut positions);
        Self::apply_connections(&mut self.root.children, tw, &positions);
    }

    fn collect_positions(nodes: &[Node], tw: &Tweakables, out: &mut HashMap<String, [f64; 2]>) {
        for node in nodes {
            if let Some(id) = &node.props().id {
                let pos = node.props().pos.get(tw);
                out.insert(id.clone(), pos);
            }
            if let Node::Group(g) = node {
                Self::collect_positions(&g.children, tw, out);
            }
        }
    }

    fn apply_connections(nodes: &mut [Node], _tw: &Tweakables, positions: &HashMap<String, [f64; 2]>) {
        for node in nodes.iter_mut() {
            match node {
                Node::Line(n) => {
                    if let Some(ref id) = n.from_node {
                        if let Some(&[x, y]) = positions.get(id) {
                            n.props.pos.x.set_immediate(x);
                            n.props.pos.y.set_immediate(y);
                        }
                    }
                    if let Some(ref id) = n.to_node {
                        if let Some(&[x, y]) = positions.get(id) {
                            n.x2.set_immediate(x);
                            n.y2.set_immediate(y);
                        }
                    }
                }
                Node::Arrow(n) => {
                    if let Some(ref id) = n.from_node {
                        if let Some(&[x, y]) = positions.get(id) {
                            n.props.pos.x.set_immediate(x);
                            n.props.pos.y.set_immediate(y);
                        }
                    }
                    if let Some(ref id) = n.to_node {
                        if let Some(&[x, y]) = positions.get(id) {
                            n.x2.set_immediate(x);
                            n.y2.set_immediate(y);
                        }
                    }
                }
                Node::Group(g) => {
                    Self::apply_connections(&mut g.children, _tw, positions);
                }
                _ => {}
            }
        }
    }

    pub fn draw(&self, scene: &mut Scene, tw: &Tweakables) {
        for child in &self.root.children {
            child.draw(scene, tw, 1.0);
        }
    }

    pub fn add(&mut self, node: Node) -> usize {
        let idx = self.root.children.len();
        self.root.children.push(node);
        idx
    }

    /// Find a node by ID anywhere in the tree.
    pub fn find(&self, id: &str) -> Option<&Node> {
        for child in &self.root.children {
            if let Some(found) = child.find(id) {
                return Some(found);
            }
        }
        None
    }

    /// Find a mutable node by ID anywhere in the tree.
    pub fn find_mut(&mut self, id: &str) -> Option<&mut Node> {
        for child in &mut self.root.children {
            if let Some(found) = child.find_mut(id) {
                return Some(found);
            }
        }
        None
    }
}
