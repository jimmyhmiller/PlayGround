use simple_easing::*;

#[derive(Clone, Copy)]
pub enum Easing {
    Linear,
    QuadIn,
    QuadOut,
    QuadInOut,
    CubicIn,
    CubicOut,
    CubicInOut,
    ElasticIn,
    ElasticOut,
    ElasticInOut,
    BounceIn,
    BounceOut,
    BounceInOut,
    BackIn,
    BackOut,
    BackInOut,
}

impl Easing {
    pub fn apply(self, t: f32) -> f32 {
        match self {
            Easing::Linear => t,
            Easing::QuadIn => quad_in(t),
            Easing::QuadOut => quad_out(t),
            Easing::QuadInOut => quad_in_out(t),
            Easing::CubicIn => cubic_in(t),
            Easing::CubicOut => cubic_out(t),
            Easing::CubicInOut => cubic_in_out(t),
            Easing::ElasticIn => elastic_in(t),
            Easing::ElasticOut => elastic_out(t),
            Easing::ElasticInOut => elastic_in_out(t),
            Easing::BounceIn => bounce_in(t),
            Easing::BounceOut => bounce_out(t),
            Easing::BounceInOut => bounce_in_out(t),
            Easing::BackIn => back_in(t),
            Easing::BackOut => back_out(t),
            Easing::BackInOut => back_in_out(t),
        }
    }
}

pub trait Lerp {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self;
}

impl Lerp for f64 {
    fn lerp(a: &f64, b: &f64, t: f32) -> f64 {
        a + (b - a) * t as f64
    }
}

impl Lerp for [f64; 2] {
    fn lerp(a: &[f64; 2], b: &[f64; 2], t: f32) -> [f64; 2] {
        [f64::lerp(&a[0], &b[0], t), f64::lerp(&a[1], &b[1], t)]
    }
}

impl Lerp for [f32; 4] {
    fn lerp(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]
    }
}

/// A tween animation from one value to another.
pub struct Tween<T> {
    pub from: T,
    pub to: T,
    pub duration: f64,
    pub easing: Easing,
    elapsed: f64,
}

impl<T: Lerp + Clone> Tween<T> {
    pub fn new(from: T, to: T, duration: f64, easing: Easing) -> Self {
        Self { from, to, duration, easing, elapsed: 0.0 }
    }

    pub fn value(&self) -> T {
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0) as f32;
        let eased = self.easing.apply(t);
        T::lerp(&self.from, &self.to, eased)
    }

    pub fn tick(&mut self, dt: f64) {
        self.elapsed += dt;
    }

    pub fn done(&self) -> bool {
        self.elapsed >= self.duration
    }

    pub fn reset(&mut self) {
        self.elapsed = 0.0;
    }
}

/// A damped spring simulation.
pub struct Spring {
    pub target: f64,
    pub position: f64,
    pub velocity: f64,
    /// Stiffness (higher = snappier)
    pub stiffness: f64,
    /// Damping ratio (1.0 = critically damped, <1 = bouncy, >1 = overdamped)
    pub damping: f64,
    /// Mass
    pub mass: f64,
}

impl Spring {
    pub fn new(initial: f64) -> Self {
        Self {
            target: initial,
            position: initial,
            velocity: 0.0,
            stiffness: 200.0,
            damping: 10.0,
            mass: 1.0,
        }
    }

    pub fn with_params(initial: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            stiffness,
            damping,
            ..Self::new(initial)
        }
    }

    pub fn set_target(&mut self, target: f64) {
        self.target = target;
    }

    pub fn tick(&mut self, dt: f64) {
        let force = -self.stiffness * (self.position - self.target);
        let damping_force = -self.damping * self.velocity;
        let acceleration = (force + damping_force) / self.mass;
        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;
    }

    pub fn settled(&self) -> bool {
        (self.position - self.target).abs() < 0.01 && self.velocity.abs() < 0.01
    }
}

/// A 2D spring for position animation.
pub struct Spring2D {
    pub x: Spring,
    pub y: Spring,
}

impl Spring2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x: Spring::new(x), y: Spring::new(y) }
    }

    pub fn with_params(x: f64, y: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            x: Spring::with_params(x, stiffness, damping),
            y: Spring::with_params(y, stiffness, damping),
        }
    }

    pub fn set_target(&mut self, x: f64, y: f64) {
        self.x.set_target(x);
        self.y.set_target(y);
    }

    pub fn position(&self) -> [f64; 2] {
        [self.x.position, self.y.position]
    }

    pub fn tick(&mut self, dt: f64) {
        self.x.tick(dt);
        self.y.tick(dt);
    }
}
