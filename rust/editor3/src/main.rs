mod headless_editor;

use std::str::from_utf8;

use headless_editor::{Selector, SimpleTextBuffer, TextBuffer, TextView, ViewTextBuffer};
use rand::{thread_rng, Rng};
use skia_window::{App, Options};
use winit::{
    event::{MouseScrollDelta, WindowEvent},
    window::CursorIcon,
};

// "#D36247", "#FFB5A3", "#F58C73", "#B54226", "#D39147", "#FFD4A3", "#F5B873",
// "#7CAABD", "#4C839A", "#33985C", "#83CDA1", "#53B079", "#1C8245", "#353f38" "#39463e"

use skia_safe::{Color4f, Font, FontStyle, Paint, RRect, Rect, Typeface};

#[derive(Copy, Clone)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {
    pub fn as_paint(&self) -> Paint {
        Paint::new(Color4f::new(self.r, self.g, self.b, self.a), None)
    }

    pub fn as_color4f(&self) -> Color4f {
        Color4f::new(self.r, self.g, self.b, self.a)
    }

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    #[allow(unused)]
    pub fn as_sk_color(&self) -> skia_safe::Color {
        self.as_color4f().to_color()
    }

    #[allow(unused)]
    pub fn from_color4f(color4f: &Color4f) -> Self {
        Color {
            r: color4f.r,
            g: color4f.g,
            b: color4f.b,
            a: color4f.a,
        }
    }

    pub fn parse_hex(hex: &str) -> Color {
        let mut start = 0;
        if hex.starts_with('#') {
            start = 1;
        }

        let r = i64::from_str_radix(&hex[start..start + 2], 16).unwrap() as f32;
        let g = i64::from_str_radix(&hex[start + 2..start + 4], 16).unwrap() as f32;
        let b = i64::from_str_radix(&hex[start + 4..start + 6], 16).unwrap() as f32;
        Color::new(r / 255.0, g / 255.0, b / 255.0, 1.0)
    }
}

#[derive(Copy, Clone)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Copy, Clone)]
struct Size {
    width: f32,
    height: f32,
}

struct Pane {
    scroll_offset: f64,
    file: usize,
    position: Position,
    size: Size,
    selector: Selector,
    text_view: Option<TextView>,
}

impl Pane {
    fn is_inside(&self, mouse_position: Position) -> bool {
        mouse_position.x > self.position.x
            && mouse_position.x < self.position.x + self.size.width
            && mouse_position.y > self.position.y
            && mouse_position.y < self.position.y + self.size.height
    }
}

struct Editor {
    files: Vec<ViewTextBuffer<SimpleTextBuffer>>,
    panes: Vec<Pane>,
    mouse_position: Position,
}

impl Editor {}

impl App for Editor {
    fn on_window_create(
        &mut self,
        event_loop_proxy: winit::event_loop::EventLoopProxy<()>,
        size: skia_window::Size,
    ) {
    }

    fn add_event(&mut self, event: &winit::event::Event<()>) -> bool {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    return false;
                }
                WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                    for pane in self.panes.iter_mut() {
                        if pane.is_inside(self.mouse_position) {
                            let file = self.files.get_mut(pane.file).unwrap();
                            let line_count = file.line_count();
                            let mut rng = thread_rng();
                            // Exclusive range
                            let n: u32 = rng.gen_range(0..line_count as u32);
                        
                            file.insert_bytes(n as usize, 0, b"\n");

                            println!("inserted at line {}", n);

                        }
                    }
                }

                WindowEvent::CursorMoved { device_id, position } => {
                    self.mouse_position = Position {
                        x: position.x as f32,
                        y: position.y as f32,
                    };
                }
                WindowEvent::MouseWheel {
                    device_id,
                    delta,
                    phase,
                } => {
                    if let MouseScrollDelta::PixelDelta(delta) = delta {
                        for pane in self.panes.iter_mut() {
                            if pane.is_inside(self.mouse_position) {
                                pane.scroll_offset += delta.y;
                                if pane.scroll_offset > 0.0 {
                                    pane.scroll_offset = 0.0;
                                }
                            }
                        }
                    } else {
                        panic!("Not a Pixel Delta!")
                    }
                }
                _ => {}
            },
            _ => {}
        }

        true
    }

    fn exit(&mut self) {}

    fn draw(&mut self, canvas: &skia_safe::Canvas) {
        canvas.clear(Color::parse_hex("#39463e").as_color4f());

        let mut paint = Paint::default();
        paint.set_color(Color::parse_hex("#353f38").as_sk_color());

        let mut text_paint = Paint::default();
        text_paint.set_color(Color::parse_hex("#FFD4A3").as_sk_color());

        let font_size = 32.0;

        let margin_top = 30.0;
        let margin_left = 30.0;

        let font = Font::new(
            Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
            font_size,
        );

        let white = &Color::parse_hex("#dc9941").as_paint();

        for pane in self.panes.iter() {
            canvas.save();
            canvas.translate((pane.position.x, pane.position.y));
            let radius = 20.0;
            let bounds = Rect::from_xywh(0.0, 0.0, pane.size.width, pane.size.height);
            let bounds_rounded = RRect::new_rect_xy(bounds, radius, radius);
            canvas.draw_rrect(bounds_rounded, &paint);
            canvas.clip_rect(bounds, None, None);
            canvas.translate((margin_left, margin_top));

            let scroll_offset = pane.scroll_offset as f32;
            canvas.translate((0.0, scroll_offset as f32));
            let file = &self.files[pane.file];

            let lines = file.get_lines_for_text_view(&pane.text_view.as_ref().unwrap());
            for (index, line) in lines.enumerate() {
                canvas.draw_str(
                    from_utf8(line).unwrap(),
                    (0.0, (index + 1) as f32 * font_size),
                    &font,
                    white,
                );
            }
            canvas.restore();
        }
    }

    fn end_frame(&mut self) {}

    fn tick(&mut self) {}

    fn should_redraw(&mut self) -> bool {
        true
    }

    fn cursor_icon(&mut self) -> winit::window::CursorIcon {
        CursorIcon::Default
    }

    fn set_window_size(&mut self, size: skia_window::Size) {}
}

impl Editor {
    fn init(&mut self) {
        for pane in self.panes.iter_mut() {
            let file = self.files.get_mut(pane.file).unwrap();
            let text_view = file.text_view_for_selector(&pane.selector);
            pane.text_view = Some(text_view);
        }
    }
}


fn main() {
    let file = std::fs::read_to_string("src/main.rs").unwrap();

    let file = SimpleTextBuffer::new_with_contents(&file.as_bytes());
    let file = ViewTextBuffer::<SimpleTextBuffer>::new_simple(file);

    let mut app = Editor {
        mouse_position: Position { x: 0.0, y: 0.0 },
        files: vec![file],
        panes: vec![Pane {
            scroll_offset: 0.0,
            file: 0,
            position: Position { x: 300.0, y: 300.0 },
            size: Size {
                width: 900.0,
                height: 900.0,
            },
            selector: Selector::Lines(18, 24),
            text_view: None,
        }, 
        Pane {
            scroll_offset: 0.0,
            file: 0,
            position: Position { x: 1400.0, y: 300.0 },
            size: Size {
                width: 900.0,
                height: 900.0,
            },
            selector: Selector::Lines(73, 77),
            text_view: None,
        }],
    };

    app.init();
    app.create_window(
        "Editor 3",
        Options {
            vsync: false,
            width: 1200,
            height: 900,
            title: "Editor 3".to_string(),
            position: (213, 180),
        },
    );
}
