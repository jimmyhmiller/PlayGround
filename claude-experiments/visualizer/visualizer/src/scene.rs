use std::collections::HashMap;

use vello::kurbo::{Affine, Circle, Point, QuadBez, RoundedRect, Stroke};
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
    pub curve: f64,
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
    pub curve: f64,
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

// ── Hover context ──

/// Passed into draw to control hover-based dimming.
pub struct HoverCtx<'a> {
    /// The node currently under the cursor, if any.
    pub hovered: Option<&'a str>,
    /// Set of node IDs that are connected to the hovered node
    /// (via from_node/to_node on arrows/lines).
    pub connected: &'a std::collections::HashSet<String>,
}

impl<'a> HoverCtx<'a> {
    /// Returns the opacity multiplier for a node based on hover state.
    /// - If nothing is hovered, returns 1.0 (no dimming).
    /// - If something is hovered, the hovered node and its connected nodes
    ///   get 1.0, everything else gets `dim`.
    pub fn opacity_for(&self, id: Option<&str>) -> f32 {
        let Some(_hovered) = self.hovered else { return 1.0 };
        let dim = 0.25_f32;
        match id {
            Some(id) if self.connected.contains(id) => 1.0,
            _ => dim,
        }
    }

    /// Returns the opacity multiplier for a line/arrow based on whether
    /// its from/to match the hovered node.
    pub fn opacity_for_connection(&self, from: Option<&str>, to: Option<&str>) -> f32 {
        let Some(hovered) = self.hovered else { return 1.0 };
        if from == Some(hovered) || to == Some(hovered) {
            1.0
        } else {
            0.08
        }
    }
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
            curve: 0.0,
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
            curve: 0.0,
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

    /// Hit-test: returns the ID of the topmost node containing (mx, my).
    /// Only tests rects and circles (not lines/arrows/groups).
    pub fn hit_test(&self, mx: f64, my: f64, tw: &Tweakables) -> Option<&str> {
        match self {
            Node::Group(g) => {
                // Test children in reverse order (topmost drawn last)
                for child in g.children.iter().rev() {
                    if let Some(id) = child.hit_test(mx, my, tw) {
                        return Some(id);
                    }
                }
                None
            }
            Node::Rect(n) => {
                let id = n.props.id.as_deref()?;
                if !n.props.visible || n.props.opacity.get(tw) < 0.01 {
                    return None;
                }
                let [px, py] = n.props.pos.get(tw);
                let scale = n.props.scale.get(tw);
                let w = n.width.get(tw) * scale;
                let h = n.height.get(tw) * scale;
                let x0 = px - w / 2.0;
                let y0 = py - h / 2.0;
                if mx >= x0 && mx <= x0 + w && my >= y0 && my <= y0 + h {
                    Some(id)
                } else {
                    None
                }
            }
            Node::Circle(n) => {
                let id = n.props.id.as_deref()?;
                if !n.props.visible || n.props.opacity.get(tw) < 0.01 {
                    return None;
                }
                let [px, py] = n.props.pos.get(tw);
                let r = n.radius.get(tw) * n.props.scale.get(tw);
                let dx = mx - px;
                let dy = my - py;
                if dx * dx + dy * dy <= r * r {
                    Some(id)
                } else {
                    None
                }
            }
            _ => None,
        }
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

// ── Curve helpers ──

/// Compute the quadratic bezier control point for an arc between p1 and p2.
/// `curve` is the perpendicular offset at the midpoint (positive = "left" of direction).
fn curve_control_point(p1: Point, p2: Point, curve: f64) -> Point {
    let mid = Point::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 {
        return mid;
    }
    // Perpendicular (rotated 90° CCW)
    let nx = -dy / len;
    let ny = dx / len;
    Point::new(mid.x + nx * curve, mid.y + ny * curve)
}

/// Tangent direction at the end of a quadratic bezier (p0, cp, p2).
/// Returns (dx, dy) normalized.
fn quad_end_tangent(cp: Point, p2: Point) -> (f64, f64) {
    let dx = p2.x - cp.x;
    let dy = p2.y - cp.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.001 {
        (1.0, 0.0)
    } else {
        (dx / len, dy / len)
    }
}

// ── Draw ──

impl Node {
    pub fn draw(&self, scene: &mut Scene, tw: &Tweakables, parent_opacity: f32, hover: &HoverCtx) {
        let props = self.props();
        if !props.visible {
            return;
        }

        let base_opacity = props.opacity.get(tw) as f32 * parent_opacity;
        if base_opacity < 0.001 {
            return;
        }

        let scale = props.scale.get(tw);
        if scale < 0.001 {
            return;
        }

        let [px, py] = props.pos.get(tw);

        let th = theme::current();
        let stroke_width = th.stroke_width;

        match self {
            Node::Rect(n) => {
                let hover_mul = hover.opacity_for(n.props.id.as_deref());
                let opacity = base_opacity * hover_mul;
                let stroke_rgba = th.stroke;
                let stroke_color =
                    Color::new([stroke_rgba[0], stroke_rgba[1], stroke_rgba[2], stroke_rgba[3] * opacity]);

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
                let hover_mul = hover.opacity_for(n.props.id.as_deref());
                let opacity = base_opacity * hover_mul;
                let stroke_rgba = th.stroke;
                let stroke_color =
                    Color::new([stroke_rgba[0], stroke_rgba[1], stroke_rgba[2], stroke_rgba[3] * opacity]);

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
                let hover_mul = hover.opacity_for(n.props.id.as_deref());
                let opacity = base_opacity * hover_mul;
                let stroke_rgba = th.stroke;
                let stroke_color =
                    Color::new([stroke_rgba[0], stroke_rgba[1], stroke_rgba[2], stroke_rgba[3] * opacity]);

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
                let hover_mul = hover.opacity_for_connection(
                    n.from_node.as_deref(),
                    n.to_node.as_deref(),
                );
                let opacity = base_opacity * hover_mul;
                let x2 = n.x2.get(tw);
                let y2 = n.y2.get(tw);
                let sw = n.stroke_width.get(tw) * scale;
                let [cr, cg, cb, ca] = n.color.get(tw);
                let color = Color::new([cr, cg, cb, ca * opacity]);

                let p1 = Point::new(px, py);
                let p2 = Point::new(x2, y2);

                if n.curve.abs() > 0.001 {
                    let cp = curve_control_point(p1, p2, n.curve);
                    let qb = QuadBez::new(p1, cp, p2);
                    scene.stroke(&Stroke::new(sw), Affine::IDENTITY, color, None, &qb);
                } else {
                    let line = vello::kurbo::Line::new(p1, p2);
                    scene.stroke(&Stroke::new(sw), Affine::IDENTITY, color, None, &line);
                }
            }
            Node::Arrow(n) => {
                let hover_mul = if n.from_node.is_some() || n.to_node.is_some() {
                    hover.opacity_for_connection(
                        n.from_node.as_deref(),
                        n.to_node.as_deref(),
                    )
                } else {
                    // Non-connected arrows (e.g. root pointers) use node-based hover
                    hover.opacity_for(n.props.id.as_deref())
                };
                let opacity = base_opacity * hover_mul;
                let x2 = n.x2.get(tw);
                let y2 = n.y2.get(tw);
                let sw = n.stroke_width.get(tw) * scale;
                let hs = n.head_size.get(tw) * scale;
                let [cr, cg, cb, ca] = n.color.get(tw);
                let color = Color::new([cr, cg, cb, ca * opacity]);

                let p1 = Point::new(px, py);
                let p2 = Point::new(x2, y2);

                if n.curve.abs() > 0.001 {
                    let cp = curve_control_point(p1, p2, n.curve);
                    let qb = QuadBez::new(p1, cp, p2);
                    scene.stroke(&Stroke::new(sw), Affine::IDENTITY, color, None, &qb);

                    // Arrowhead aligned to curve tangent at endpoint
                    let (ux, uy) = quad_end_tangent(cp, p2);
                    let nx = -uy;
                    let ny = ux;
                    let tip = p2;
                    let left = Point::new(tip.x - ux * hs + nx * hs * 0.5, tip.y - uy * hs + ny * hs * 0.5);
                    let right = Point::new(tip.x - ux * hs - nx * hs * 0.5, tip.y - uy * hs - ny * hs * 0.5);
                    let head = vello::kurbo::BezPath::from_vec(vec![
                        vello::kurbo::PathEl::MoveTo(tip),
                        vello::kurbo::PathEl::LineTo(left),
                        vello::kurbo::PathEl::LineTo(right),
                        vello::kurbo::PathEl::ClosePath,
                    ]);
                    scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &head);
                } else {
                    // Straight arrow
                    let line = vello::kurbo::Line::new(p1, p2);
                    scene.stroke(&Stroke::new(sw), Affine::IDENTITY, color, None, &line);

                    let dx = x2 - px;
                    let dy = y2 - py;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len > 0.001 {
                        let ux = dx / len;
                        let uy = dy / len;
                        let nx = -uy;
                        let ny = ux;
                        let tip = p2;
                        let left = Point::new(tip.x - ux * hs + nx * hs * 0.5, tip.y - uy * hs + ny * hs * 0.5);
                        let right = Point::new(tip.x - ux * hs - nx * hs * 0.5, tip.y - uy * hs - ny * hs * 0.5);
                        let head = vello::kurbo::BezPath::from_vec(vec![
                            vello::kurbo::PathEl::MoveTo(tip),
                            vello::kurbo::PathEl::LineTo(left),
                            vello::kurbo::PathEl::LineTo(right),
                            vello::kurbo::PathEl::ClosePath,
                        ]);
                        scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &head);
                    }
                }
            }
            Node::Group(n) => {
                for child in &n.children {
                    child.draw(scene, tw, base_opacity, hover);
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
        Self::apply_connections(&mut self.root.children, &positions);
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

    fn apply_connections(nodes: &mut [Node], positions: &HashMap<String, [f64; 2]>) {
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
                    Self::apply_connections(&mut g.children, positions);
                }
                _ => {}
            }
        }
    }

    /// Hit-test: returns the ID of the topmost node at (mx, my).
    pub fn hit_test(&self, mx: f64, my: f64, tw: &Tweakables) -> Option<String> {
        for child in self.root.children.iter().rev() {
            if let Some(id) = child.hit_test(mx, my, tw) {
                return Some(id.to_string());
            }
        }
        None
    }

    /// Collect the set of node IDs connected to `node_id` via arrows/lines.
    /// Includes `node_id` itself plus all from_node/to_node peers.
    pub fn connected_set(&self, node_id: &str) -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::new();
        set.insert(node_id.to_string());
        Self::collect_connected(&self.root.children, node_id, &mut set);
        set
    }

    fn collect_connected(nodes: &[Node], node_id: &str, set: &mut std::collections::HashSet<String>) {
        for node in nodes {
            let (from, to) = match node {
                Node::Line(n) => (n.from_node.as_deref(), n.to_node.as_deref()),
                Node::Arrow(n) => (n.from_node.as_deref(), n.to_node.as_deref()),
                _ => (None, None),
            };
            if from == Some(node_id) {
                if let Some(t) = to { set.insert(t.to_string()); }
            }
            if to == Some(node_id) {
                if let Some(f) = from { set.insert(f.to_string()); }
            }
            if let Node::Group(g) = node {
                Self::collect_connected(&g.children, node_id, set);
            }
        }
    }

    pub fn draw(&self, scene: &mut Scene, tw: &Tweakables, hover: &HoverCtx) {
        for child in &self.root.children {
            child.draw(scene, tw, 1.0, hover);
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
