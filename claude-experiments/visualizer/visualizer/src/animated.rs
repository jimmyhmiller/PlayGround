use crate::anim::{Easing, Spring, Tween};
use crate::tweakables::Tweakables;

/// How a value is produced and animated.
enum Inner {
    /// Fixed value.
    Const(f64),

    /// Reads from a tweakable by name. Optionally smoothed by a spring.
    Tweakable {
        name: String,
        spring: Option<Spring>,
    },

    /// Spring that tracks a target. Call `set_target` to change it.
    Spring(Spring),

    /// Tween from A to B over a duration. Call `fire` to start/restart.
    Tween {
        tween: Tween<f64>,
        fired: bool,
    },

    /// Derives from another source, smoothed through a spring.
    /// The `source` closure returns the current target value.
    Derived {
        source: Box<dyn Fn(&Tweakables) -> f64>,
        spring: Spring,
    },

    /// Derives from another source, read live each frame (no spring).
    DerivedLive {
        source: Box<dyn Fn(&Tweakables) -> f64>,
    },
}

/// A single animated f64 value.
///
/// This is the core building block. Every property on a scene node
/// (x, y, width, opacity, etc.) is an `AnimatedValue`.
pub struct AnimatedValue {
    inner: Inner,
}

// ── Constructors ──

impl AnimatedValue {
    /// A constant value that never changes.
    pub fn constant(value: f64) -> Self {
        Self { inner: Inner::Const(value) }
    }

    /// Reads a tweakable value by name each frame.
    pub fn tweakable(name: &str) -> Self {
        Self {
            inner: Inner::Tweakable {
                name: name.to_string(),
                spring: None,
            },
        }
    }

    /// Reads a tweakable, smoothed through a spring so changes animate.
    pub fn tweakable_smoothed(name: &str, stiffness: f64, damping: f64) -> Self {
        Self {
            inner: Inner::Tweakable {
                name: name.to_string(),
                spring: Some(Spring::with_params(0.0, stiffness, damping)),
            },
        }
    }

    /// A spring-based value. Starts at `initial`, animates toward targets.
    pub fn spring(initial: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            inner: Inner::Spring(Spring::with_params(initial, stiffness, damping)),
        }
    }

    /// A tween that plays once when `fire()` is called.
    pub fn tween(from: f64, to: f64, duration: f64, easing: Easing) -> Self {
        Self {
            inner: Inner::Tween {
                tween: Tween::new(from, to, duration, easing),
                fired: false,
            },
        }
    }

    /// Derives from an arbitrary function of tweakables, smoothed through a spring.
    pub fn derived(
        source: impl Fn(&Tweakables) -> f64 + 'static,
        stiffness: f64,
        damping: f64,
    ) -> Self {
        Self {
            inner: Inner::Derived {
                source: Box::new(source),
                spring: Spring::with_params(0.0, stiffness, damping),
            },
        }
    }

    /// Reads from an arbitrary function of tweakables, fresh each frame (no spring).
    pub fn derived_live(source: impl Fn(&Tweakables) -> f64 + 'static) -> Self {
        Self {
            inner: Inner::DerivedLive {
                source: Box::new(source),
            },
        }
    }
}

// ── Reading ──

impl AnimatedValue {
    /// Get the current value.
    pub fn get(&self, tw: &Tweakables) -> f64 {
        match &self.inner {
            Inner::Const(v) => *v,
            Inner::Tweakable { name, spring } => {
                match spring {
                    Some(s) => s.position,
                    None => tw.get(name),
                }
            }
            Inner::Spring(s) => s.position,
            Inner::Tween { tween, fired } => {
                if *fired {
                    tween.value()
                } else {
                    tween.from
                }
            }
            Inner::Derived { spring, .. } => spring.position,
            Inner::DerivedLive { source } => source(tw),
        }
    }

    /// Is the animation settled (not moving)?
    pub fn settled(&self) -> bool {
        match &self.inner {
            Inner::Const(_) => true,
            Inner::Tweakable { spring, .. } => {
                spring.as_ref().map(|s| s.settled()).unwrap_or(true)
            }
            Inner::Spring(s) => s.settled(),
            Inner::Tween { tween, fired } => !fired || tween.done(),
            Inner::Derived { spring, .. } => spring.settled(),
            Inner::DerivedLive { .. } => true,
        }
    }
}

// ── Mutation ──

impl AnimatedValue {
    /// Set the target for a spring-based value. No-op on const/tween.
    pub fn set_target(&mut self, target: f64) {
        match &mut self.inner {
            Inner::Spring(s) => s.set_target(target),
            Inner::Tweakable { spring: Some(s), .. } => s.set_target(target),
            Inner::Derived { spring, .. } => spring.set_target(target),
            _ => {}
        }
    }

    /// Set to a constant value immediately (no animation).
    pub fn set_immediate(&mut self, value: f64) {
        match &mut self.inner {
            Inner::Const(v) => *v = value,
            Inner::Spring(s) => {
                s.position = value;
                s.velocity = 0.0;
                s.target = value;
            }
            Inner::Tweakable { spring: Some(s), .. } => {
                s.position = value;
                s.velocity = 0.0;
                s.target = value;
            }
            Inner::Derived { spring, .. } => {
                spring.position = value;
                spring.velocity = 0.0;
                // target stays whatever the source returns
            }
            _ => {}
        }
    }

    /// Fire a tween (start/restart it).
    pub fn fire(&mut self) {
        if let Inner::Tween { tween, fired } = &mut self.inner {
            tween.reset();
            *fired = true;
        }
    }

    /// Fire a tween with new from/to values.
    pub fn fire_with(&mut self, from: f64, to: f64) {
        if let Inner::Tween { tween, fired } = &mut self.inner {
            tween.from = from;
            tween.to = to;
            tween.reset();
            *fired = true;
        }
    }

    /// Update spring parameters live (e.g. from tweakables).
    pub fn set_spring_params(&mut self, stiffness: f64, damping: f64) {
        match &mut self.inner {
            Inner::Spring(s) => {
                s.stiffness = stiffness;
                s.damping = damping;
            }
            Inner::Tweakable { spring: Some(s), .. } => {
                s.stiffness = stiffness;
                s.damping = damping;
            }
            Inner::Derived { spring, .. } => {
                spring.stiffness = stiffness;
                spring.damping = damping;
            }
            _ => {}
        }
    }

    /// Advance the animation by `dt` seconds.
    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        match &mut self.inner {
            Inner::Const(_) => {}
            Inner::Tweakable { name, spring } => {
                if let Some(s) = spring {
                    s.set_target(tw.get(name));
                    s.tick(dt);
                }
            }
            Inner::Spring(s) => {
                s.tick(dt);
            }
            Inner::Tween { tween, fired } => {
                if *fired {
                    tween.tick(dt);
                }
            }
            Inner::Derived { source, spring } => {
                let target = source(tw);
                spring.set_target(target);
                spring.tick(dt);
            }
            Inner::DerivedLive { .. } => {} // always live, no state to advance
        }
    }
}

/// Convenience: a pair of animated values for 2D position.
pub struct AnimatedPos {
    pub x: AnimatedValue,
    pub y: AnimatedValue,
}

impl AnimatedPos {
    pub fn spring(x: f64, y: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            x: AnimatedValue::spring(x, stiffness, damping),
            y: AnimatedValue::spring(y, stiffness, damping),
        }
    }

    pub fn constant(x: f64, y: f64) -> Self {
        Self {
            x: AnimatedValue::constant(x),
            y: AnimatedValue::constant(y),
        }
    }

    pub fn get(&self, tw: &Tweakables) -> [f64; 2] {
        [self.x.get(tw), self.y.get(tw)]
    }

    pub fn set_target(&mut self, x: f64, y: f64) {
        self.x.set_target(x);
        self.y.set_target(y);
    }

    pub fn set_immediate(&mut self, x: f64, y: f64) {
        self.x.set_immediate(x);
        self.y.set_immediate(y);
    }

    pub fn settled(&self) -> bool {
        self.x.settled() && self.y.settled()
    }

    pub fn set_spring_params(&mut self, stiffness: f64, damping: f64) {
        self.x.set_spring_params(stiffness, damping);
        self.y.set_spring_params(stiffness, damping);
    }

    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        self.x.tick(dt, tw);
        self.y.tick(dt, tw);
    }
}

/// Convenience: 4-channel animated color [r, g, b, a].
pub struct AnimatedColor {
    pub r: AnimatedValue,
    pub g: AnimatedValue,
    pub b: AnimatedValue,
    pub a: AnimatedValue,
}

impl AnimatedColor {
    pub fn constant(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r: AnimatedValue::constant(r as f64),
            g: AnimatedValue::constant(g as f64),
            b: AnimatedValue::constant(b as f64),
            a: AnimatedValue::constant(a as f64),
        }
    }

    pub fn spring(r: f32, g: f32, b: f32, a: f32, stiffness: f64, damping: f64) -> Self {
        Self {
            r: AnimatedValue::spring(r as f64, stiffness, damping),
            g: AnimatedValue::spring(g as f64, stiffness, damping),
            b: AnimatedValue::spring(b as f64, stiffness, damping),
            a: AnimatedValue::spring(a as f64, stiffness, damping),
        }
    }

    pub fn get(&self, tw: &Tweakables) -> [f32; 4] {
        [
            self.r.get(tw) as f32,
            self.g.get(tw) as f32,
            self.b.get(tw) as f32,
            self.a.get(tw) as f32,
        ]
    }

    pub fn set_target(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.r.set_target(r as f64);
        self.g.set_target(g as f64);
        self.b.set_target(b as f64);
        self.a.set_target(a as f64);
    }

    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        self.r.tick(dt, tw);
        self.g.tick(dt, tw);
        self.b.tick(dt, tw);
        self.a.tick(dt, tw);
    }
}
