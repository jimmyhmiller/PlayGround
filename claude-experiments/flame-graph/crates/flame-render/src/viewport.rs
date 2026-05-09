/// Pan/zoom state for the flame graph timeline.
///
/// `start_ns` and `ns_per_pixel` are kept as `f64` for nanosecond-precision math.
/// They're converted to `f32` only when computing per-instance pixel rects.
#[derive(Clone, Debug)]
pub struct Viewport {
    pub start_ns: f64,
    pub ns_per_pixel: f64,
    pub size_px: (f32, f32),
    pub scroll_y_px: f32,
    pub row_height_px: f32,
}

impl Viewport {
    pub fn new(size_px: (f32, f32)) -> Self {
        Self {
            start_ns: 0.0,
            ns_per_pixel: 1.0,
            size_px,
            scroll_y_px: 0.0,
            row_height_px: 18.0,
        }
    }

    /// Fit `[start_ns, end_ns)` into the current viewport width with a small
    /// margin. Profiles whose first slice doesn't start at t=0 (e.g. Firefox
    /// `timeDeltas`-based traces) need this so the viewport lands on the
    /// actual data, not before it.
    pub fn fit_time(&mut self, start_ns: u64, end_ns: u64) {
        if end_ns <= start_ns || self.size_px.0 <= 0.0 {
            self.start_ns = start_ns as f64;
            self.ns_per_pixel = 1.0;
            return;
        }
        let usable_w = (self.size_px.0 - 16.0).max(1.0) as f64;
        let total = (end_ns - start_ns) as f64;
        // Pad both sides by 1% so the leftmost/rightmost slices aren't flush
        // with the screen edges.
        let pad = total * 0.01;
        self.ns_per_pixel = (total + pad * 2.0) / usable_w;
        self.start_ns = start_ns as f64 - pad;
    }

    pub fn end_ns(&self) -> f64 {
        self.start_ns + self.ns_per_pixel * self.size_px.0 as f64
    }

    pub fn ns_to_x(&self, ns: u64) -> f32 {
        ((ns as f64 - self.start_ns) / self.ns_per_pixel) as f32
    }

    pub fn x_to_ns(&self, x_px: f32) -> f64 {
        self.start_ns + x_px as f64 * self.ns_per_pixel
    }

    /// Multiplicative zoom around an anchor x. `factor < 1` zooms in.
    pub fn zoom_at(&mut self, anchor_x_px: f32, factor: f64) {
        let cursor_ns = self.x_to_ns(anchor_x_px);
        self.ns_per_pixel = (self.ns_per_pixel * factor).max(1e-3);
        self.start_ns = cursor_ns - anchor_x_px as f64 * self.ns_per_pixel;
    }

    pub fn pan_x_px(&mut self, dx_px: f32) {
        self.start_ns -= dx_px as f64 * self.ns_per_pixel;
    }

    pub fn pan_y_px(&mut self, dy_px: f32) {
        self.scroll_y_px = (self.scroll_y_px - dy_px).max(0.0);
    }

    pub fn resize(&mut self, w: f32, h: f32) {
        self.size_px = (w.max(1.0), h.max(1.0));
    }

    /// Cap zoom-out at "trace fills 75% of viewport" and keep a small margin
    /// of pannable space on each side. Must be called after every viewport
    /// mutation that originates from user input.
    pub fn clamp(&mut self, trace_start_ns: u64, trace_end_ns: u64) {
        if trace_end_ns <= trace_start_ns || self.size_px.0 <= 0.0 {
            return;
        }
        let total = (trace_end_ns - trace_start_ns) as f64;
        let view_w = self.size_px.0 as f64;
        let max_ns_per_pixel = total / (view_w * 0.75).max(1.0);
        if self.ns_per_pixel > max_ns_per_pixel {
            self.ns_per_pixel = max_ns_per_pixel;
        }
        let view_ns = view_w * self.ns_per_pixel;
        let slack = view_ns * 0.25;
        let min_start = trace_start_ns as f64 - slack;
        let max_start = trace_end_ns as f64 - view_ns + slack;
        if max_start < min_start {
            self.start_ns = (trace_start_ns as f64 + trace_end_ns as f64 - view_ns) * 0.5;
        } else {
            self.start_ns = self.start_ns.clamp(min_start, max_start);
        }
    }
}
