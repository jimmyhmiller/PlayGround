use std::cell::RefCell;

use vello::peniko::{
    Blob, Extend, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageSampler,
};

#[derive(Clone, Debug)]
pub struct Theme {
    pub name: String,
    pub background: [f32; 4],
    pub stroke: [f32; 4],
    pub stroke_width: f64,
    pub label: [f32; 4],
    pub accents: Vec<[f32; 4]>,
    pub grain_enabled: bool,
    pub grain_intensity: f32,
    // Motion
    pub spring_stiffness: f64,
    pub spring_damping: f64,
    pub tween_duration: f64,
    pub tween_easing: String,
}

fn hex(s: &str) -> [f32; 4] {
    let s = s.trim_start_matches('#');
    let r = u8::from_str_radix(&s[0..2], 16).unwrap() as f32 / 255.0;
    let g = u8::from_str_radix(&s[2..4], 16).unwrap() as f32 / 255.0;
    let b = u8::from_str_radix(&s[4..6], 16).unwrap() as f32 / 255.0;
    [r, g, b, 1.0]
}

impl Theme {
    pub fn iso50() -> Self {
        Self {
            name: "iso50".into(),
            background: hex("#1f2d35"),
            stroke: hex("#8aa39b"),
            stroke_width: 1.25,
            label: hex("#a8b5a8"),
            accents: vec![
                hex("#d47b6a"), // coral
                hex("#c9a063"), // mustard
                hex("#7fa99b"), // sage
                hex("#b8746b"), // dusty rose
                hex("#e0d4b8"), // cream
                hex("#4a6b7a"), // slate blue
                hex("#bf8a5a"), // amber
                hex("#6b8a82"), // muted teal
            ],
            grain_enabled: true,
            grain_intensity: 0.12,
            spring_stiffness: 140.0,
            spring_damping: 24.0,
            tween_duration: 0.8,
            tween_easing: "quad-in-out".into(),
        }
    }

    pub fn byrne() -> Self {
        Self {
            name: "byrne".into(),
            background: hex("#f3ecd8"),
            stroke: hex("#1a1a1a"),
            stroke_width: 1.5,
            label: hex("#1a1a1a"),
            accents: vec![
                hex("#d23b2b"),
                hex("#ffc107"),
                hex("#2d5fb8"),
                hex("#1a1a1a"),
                hex("#d23b2b"),
                hex("#ffc107"),
                hex("#2d5fb8"),
                hex("#1a1a1a"),
            ],
            grain_enabled: false,
            grain_intensity: 0.0,
            spring_stiffness: 500.0,
            spring_damping: 30.0,
            tween_duration: 0.25,
            tween_easing: "quad-out".into(),
        }
    }

    pub fn terminal() -> Self {
        Self {
            name: "terminal".into(),
            background: hex("#0a0e14"),
            stroke: hex("#4dd0a0"),
            stroke_width: 1.0,
            label: hex("#b3f5d8"),
            accents: vec![
                hex("#4dd0a0"),
                hex("#f5c04d"),
                hex("#4da6ff"),
                hex("#ff6b8a"),
                hex("#c58af5"),
                hex("#4dd0a0"),
                hex("#f5c04d"),
                hex("#ff6b8a"),
            ],
            grain_enabled: false,
            grain_intensity: 0.0,
            spring_stiffness: 400.0,
            spring_damping: 22.0,
            tween_duration: 0.3,
            tween_easing: "cubic-out".into(),
        }
    }

    pub fn paper() -> Self {
        Self {
            name: "paper".into(),
            background: hex("#efe8d5"),
            stroke: hex("#3a3026"),
            stroke_width: 1.25,
            label: hex("#3a3026"),
            accents: vec![
                hex("#b8543a"),
                hex("#d89030"),
                hex("#4a7a5c"),
                hex("#355d8a"),
                hex("#8a4a6c"),
                hex("#6c5a3a"),
                hex("#b87a3a"),
                hex("#3a5a6c"),
            ],
            grain_enabled: true,
            grain_intensity: 0.10,
            spring_stiffness: 200.0,
            spring_damping: 20.0,
            tween_duration: 0.6,
            tween_easing: "cubic-in-out".into(),
        }
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "iso50" => Some(Self::iso50()),
            "byrne" => Some(Self::byrne()),
            "terminal" => Some(Self::terminal()),
            "paper" => Some(Self::paper()),
            _ => None,
        }
    }

    pub fn accent(&self, i: usize) -> [f32; 4] {
        if self.accents.is_empty() {
            return [1.0, 1.0, 1.0, 1.0];
        }
        self.accents[i % self.accents.len()]
    }
}

thread_local! {
    static CURRENT: RefCell<Theme> = RefCell::new(Theme::iso50());
    static GRAIN: RefCell<Option<ImageData>> = RefCell::new(None);
}

pub fn current() -> Theme {
    CURRENT.with(|c| c.borrow().clone())
}

pub fn set(theme: Theme) {
    CURRENT.with(|c| *c.borrow_mut() = theme);
}

pub fn accent(i: usize) -> [f32; 4] {
    CURRENT.with(|c| c.borrow().accent(i))
}

/// Build (and cache) a tiled noise image used for the grain overlay.
///
/// Film-grain style: box-blurred white noise producing signed deviations
/// (both lighter and darker pixels), then mapped to premultiplied RGBA
/// where dark pixels have near-black color and light pixels have near-white.
/// Uses a gaussian-ish bias so extreme values are rare and the overall feel
/// is soft rather than salt-and-pepper.
pub fn grain_image() -> ImageData {
    GRAIN.with(|g| {
        if let Some(img) = g.borrow().as_ref() {
            return img.clone();
        }
        let size: usize = 512;
        let n = size * size;

        // Step 1: white noise centered on 0 (range ~[-1, 1]) with gaussian bias.
        // Using box-muller-ish: sum of 3 uniform samples → triangular distribution.
        let mut seed: u32 = 0x9E3779B9;
        let mut rand = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / ((1u32 << 24) as f32) // [0, 1)
        };

        let mut noise = vec![0.0f32; n];
        for v in noise.iter_mut() {
            // Sum of 3 uniforms → triangular-ish, values ~[-1.5, 1.5], peak at 0.
            let s = rand() + rand() + rand() - 1.5;
            *v = s;
        }

        // Step 2: two-pass separable box blur (radius 1, kernel 3) — cheap
        // low-pass filter. Run it twice for a smoother, more organic clump.
        fn box_blur(src: &[f32], dst: &mut [f32], size: usize) {
            // Horizontal
            let mut tmp = vec![0.0f32; size * size];
            for y in 0..size {
                for x in 0..size {
                    let xl = if x == 0 { size - 1 } else { x - 1 };
                    let xr = if x + 1 == size { 0 } else { x + 1 };
                    tmp[y * size + x] =
                        (src[y * size + xl] + src[y * size + x] + src[y * size + xr]) / 3.0;
                }
            }
            // Vertical
            for y in 0..size {
                let yu = if y == 0 { size - 1 } else { y - 1 };
                let yd = if y + 1 == size { 0 } else { y + 1 };
                for x in 0..size {
                    dst[y * size + x] =
                        (tmp[yu * size + x] + tmp[y * size + x] + tmp[yd * size + x]) / 3.0;
                }
            }
        }

        let mut blurred = vec![0.0f32; n];
        box_blur(&noise, &mut blurred, size);
        // Second pass for softer clumping
        box_blur(&blurred.clone(), &mut blurred, size);

        // Step 3: blurring reduced variance — renormalize so the output spans
        // roughly [-1, 1] again. Compute max abs value.
        let mut max_abs = 0.0f32;
        for &v in &blurred {
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }
        if max_abs > 0.0 {
            for v in blurred.iter_mut() {
                *v /= max_abs;
            }
        }

        // Step 4: map to premultiplied RGBA. Signed value s in [-1, 1]:
        //   s > 0 → light pixel (white), alpha = s * peak
        //   s < 0 → dark pixel (black), alpha = -s * peak
        // The `peak` shapes the distribution — lower = softer grain tile,
        // the draw-time `alpha` multiplier modulates intensity further.
        let peak: f32 = 0.85;
        let mut pixels = vec![0u8; n * 4];
        for (i, &s) in blurred.iter().enumerate() {
            // Apply a gentle curve so the bulk of pixels are near-zero and
            // only clumps stand out. pow(|s|, 1.4) emphasizes peaks.
            let mag = s.abs().powf(1.4) * peak;
            let a8 = (mag.clamp(0.0, 1.0) * 255.0) as u8;
            let c8 = if s >= 0.0 { a8 } else { 0 }; // white or black, premultiplied
            let o = i * 4;
            pixels[o] = c8;
            pixels[o + 1] = c8;
            pixels[o + 2] = c8;
            pixels[o + 3] = a8;
        }

        let img = ImageData {
            data: Blob::from(pixels),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::AlphaPremultiplied,
            width: size as u32,
            height: size as u32,
        };
        *g.borrow_mut() = Some(img.clone());
        img
    })
}

pub fn grain_brush(alpha: f32) -> ImageBrush {
    ImageBrush {
        image: grain_image(),
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: vello::peniko::ImageQuality::Low,
            alpha,
        },
    }
}
