mod headless_editor;

use std::{ops::Add, str::from_utf8};

use headless_editor::{Selector, SimpleTextBuffer, TextBuffer, TextView, ViewTextBuffer};
use rand::{thread_rng, Rng};
use skia_window::{App, Options};
use winit::{
    event::{MouseScrollDelta, WindowEvent},
    window::CursorIcon,
};

// "#D36247", "#FFB5A3", "#F58C73", "#B54226", "#D39147", "#FFD4A3", "#F5B873",
// "#7CAABD", "#4C839A", "#33985C", "#83CDA1", "#53B079", "#1C8245", "#353f38" "#39463e"

use skia_safe::{font, textlayout::{FontCollection, Paragraph, ParagraphBuilder, ParagraphStyle, TextStyle}, Canvas, Color4f, Font, FontMgr, FontStyle, Paint, RRect, Rect, Typeface};

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

#[derive(Debug, Copy, Clone)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Copy, Clone)]
struct Size {
    width: f32,
    height: f32,
}


impl Size {
    fn to_rect(&self) -> Rect {
        Rect::from_wh(self.width, self.height)
    }
}

impl Add for Size {
    type Output = Size;

    fn add(self, other: Size) -> Size {
        Size {
            width: self.width + other.width,
            height: self.height + other.height,
        }
    }
}

struct Bounds {
    size: Size,
}

#[derive(Debug, Copy, Clone)]
struct Layout {
    position: Position,
    size: Size,
}

#[derive(Debug, Copy, Clone)]
enum Sides {
    Top(f32),
    Bottom(f32),
    Left(f32),
    Right(f32),
}

#[derive(Debug, Clone)]
struct Style {
    margin: Vec<Sides>,
    padding: Vec<Sides>,
}

impl Default for Style {
    fn default() -> Self {
        Style {
            margin: vec![],
            padding: vec![],
        }
    }
}


trait Drawable {
    fn draw(&self, canvas: &Canvas);
    fn compute_layout(&mut self, bounds: Bounds) -> Layout;
    fn get_layout(&self) -> Layout;
    fn get_styles(&self) -> &Style;
    fn get_styles_mut(&mut self) -> &mut Style;
    fn set_styles(&mut self, style: Style);
    fn margin(mut self, value: f32) -> Self where Self: Sized {
        let style = self.get_styles_mut();
        style.margin = vec![
            Sides::Top(value),
            Sides::Bottom(value),
            Sides::Left(value),
            Sides::Right(value),
        ];
        self
    }
    fn padding(mut self, value: f32) -> Self where Self: Sized {
        let style = self.get_styles_mut();
        style.padding = vec![
            Sides::Top(value),
            Sides::Bottom(value),
            Sides::Left(value),
            Sides::Right(value),
        ];
        self
    }
    fn margin_top(mut self, value: f32) -> Self where Self: Sized {
        self.get_styles_mut().margin.push(Sides::Top(value));
        self
    }
    fn margin_bottom(mut self, value: f32) -> Self where Self: Sized {
        self.get_styles_mut().margin.push(Sides::Bottom(value));
        self
    }
    fn margin_left(mut self, value: f32) -> Self where Self: Sized {
        self.get_styles_mut().margin.push(Sides::Left(value));
        self
    }
    fn margin_right(mut self, value: f32) -> Self where Self: Sized {
        self.get_styles_mut().margin.push(Sides::Right(value));
        self
    }
    fn padding_top(mut self, value: f32) -> Self where Self: Sized {
        self.get_styles_mut().padding.push(Sides::Top(value));
        self
    }
    fn padding_bottom(mut self, value: f32) -> Self where Self: Sized {
        self.get_styles_mut().padding.push(Sides::Bottom(value));
        self
    }
    fn padding_left(mut self, value: f32) -> Self where Self: Sized {
        self.get_styles_mut().padding.push(Sides::Left(value));
        self
    }
    fn padding_right(mut self, value: f32) -> Self where Self: Sized {
        self.get_styles_mut().padding.push(Sides::Right(value));
        self
    }
    fn draw_margin_top_left(&self, canvas: &Canvas) {
        let style = self.get_styles();
        for side in style.margin.iter() {
            match side {
                Sides::Top(value) => {
                    canvas.translate((0.0, *value));
                }
                Sides::Left(value) => {
                    canvas.translate((*value, 0.0));
                }
                _ => {}
            }
        }
    }

    fn draw_margin_bottom_right(&self, canvas: &Canvas) {
        let style = self.get_styles();
        for side in style.margin.iter() {
            match side {
                Sides::Bottom(value) => {
                    canvas.translate((0.0, -*value));
                }
                Sides::Right(value) => {
                    canvas.translate((-*value, 0.0));
                }
                _ => {}
            }
        }
    }

    fn margin_size(&self) -> Size {
        let style = self.get_styles();
        let mut width = 0.0;
        let mut height = 0.0;
        for side in style.margin.iter() {
            match side {
                Sides::Top(value) => {
                    height += value;
                }
                Sides::Bottom(value) => {
                    height += value;
                }
                Sides::Left(value) => {
                    width += value;
                }
                Sides::Right(value) => {
                    width += value;
                }
            }
        }
        Size { width, height }
    }
    fn draw_padding(&self, canvas: &Canvas) {
        let style = self.get_styles();
        for side in style.padding.iter() {
            match side {
                Sides::Top(value) => {
                    canvas.translate((0.0, *value));
                }
                Sides::Left(value) => {
                    canvas.translate((*value, 0.0));
                }
                _ => {}
            }
        }
    }

    fn padding_size(&self) -> Size {
        let style = self.get_styles();
        let mut width = 0.0;
        let mut height = 0.0;
        for side in style.padding.iter() {
            match side {
                Sides::Top(value) => {
                    height += value;
                }
                Sides::Bottom(value) => {
                    height += value;
                }
                Sides::Left(value) => {
                    width += value;
                }
                Sides::Right(value) => {
                    width += value;
                }
            }
        }
        Size { width, height }
    }
}


struct Text {
    content: String,
    paragraph: Option<Paragraph>,
    layout: Layout,
    style: Style,
}

impl Text {
    fn new(content: String) -> Text {
        Text {
            content,
            paragraph: None,
            layout: Layout {
                position: Position { x: 0.0, y: 0.0 },
                size: Size { width: 0.0, height: 0.0 },
            },
            style: Style::default(),
        }
    }
}

// TODO: Lots of work needed to make this setup work properly

impl Drawable for Text {
    fn draw(&self, canvas: &skia_safe::Canvas) {
        let paragraph = self.paragraph.as_ref().unwrap();
        paragraph.paint(canvas, (self.layout.position.x, self.layout.position.y));
    }

    fn compute_layout(&mut self, bounds: Bounds) -> Layout {

        // TODO: If I make the assumption of a monospaced font,
        // I don't really need to do this, just need to calculate
        // some stuff once and use those parameters
        if self.paragraph.is_none() {
                // Create a font collection
            let mut font_collection = FontCollection::new();
            let font_manager = FontMgr::new();

            let family_name = "Ubuntu Mono";
    
            font_collection.set_default_font_manager(font_manager, family_name);

            // Define the paragraph style
            let paragraph_style = ParagraphStyle::new();

            // Define the text style
            let mut text_style = TextStyle::new();
            text_style.set_font_size(32.0);

            // Build the paragraph
            let mut paragraph_builder = ParagraphBuilder::new(&paragraph_style, font_collection);
            paragraph_builder.push_style(&text_style);
            paragraph_builder.add_text(self.content.as_str());
            let mut paragraph = paragraph_builder.build();
            paragraph.layout(bounds.size.width as f32);
            self.paragraph = Some(paragraph);
        }
    
        let paragraph = self.paragraph.as_ref().unwrap();

        let height = paragraph.height();
        let width = paragraph.max_intrinsic_width();

        let layout = Layout {
            position: Position { x: 0.0, y: 0.0 },
            size: Size {
                width: width as f32,
                height: height as f32,
            },
        };
        self.layout = layout;
        layout
    }
    
    fn get_layout(&self) -> Layout {
        self.layout
    }

    fn get_styles(&self) -> &Style {
        &self.style
    }
    
    fn set_styles(&mut self, style: Style) {
        self.style = style;
    }
    
    fn get_styles_mut(&mut self) -> &mut Style {
        &mut self.style
    }
}

struct RoundedRect {
    layout: Layout,
    radius: f32,
    color: Color,
    style: Style,
    children: Vec<Box<dyn Drawable>>,
}

impl RoundedRect {
    fn new(radius: f32, color: Color) -> RoundedRect {
        RoundedRect {
            layout: Layout {
                position: Position { x: 0.0, y: 0.0 },
                size: Size { width: 0.0, height: 0.0 },
            },
            radius,
            color,
            style: Style::default(),
            children: vec![],
        }
    }
}

impl Drawable for RoundedRect {
    fn draw(&self, canvas: &skia_safe::Canvas) {

        canvas.save();
        self.draw_margin_top_left(canvas);
        let paint = self.color.as_paint();
        let rect = (self.padding_size() + self.layout.size).to_rect();
        let bounds_rounded = RRect::new_rect_xy(rect, self.radius, self.radius);
        canvas.draw_rrect(bounds_rounded, &paint);

        self.draw_padding(canvas);
        for child in self.children.iter() {
            let layout = child.get_layout();
            child.draw(canvas);
            canvas.translate((0.0, layout.size.height));
        }
        self.draw_margin_bottom_right(canvas);
        canvas.restore();
    }

    fn compute_layout(&mut self, bounds: Bounds) -> Layout {
        let mut y = 0.0;
        let mut width = 0.0;
        for child in self.children.iter_mut() {
            let layout = child.compute_layout(Bounds {
                size: Size {
                    width: bounds.size.width,
                    height: bounds.size.height,
                },
            });
            y += layout.size.height;
            width = layout.size.width.max(width);
        }
        self.layout =  Layout {
            position: Position { x: 0.0, y: 0.0 },
            size: Size {
                width: width,
                height: y,
            },
        };

        self.layout
    }

    fn get_layout(&self) -> Layout {
       self.layout
    }
    
    fn get_styles(&self) -> &Style {
        &self.style
    }
    
    fn set_styles(&mut self, style: Style) {
        self.style = style;
    }
    
    fn get_styles_mut(&mut self) -> &mut Style {
        &mut self.style
    }
}


struct Container {
    children: Vec<Box<dyn Drawable>>,
    layout: Layout,
    style: Style,
}

impl Container {
    fn new() -> Container {
        Container {
            children: vec![],
            layout: Layout {
                position: Position { x: 0.0, y: 0.0 },
                size: Size { width: 0.0, height: 0.0 },
            },
            style: Style::default(),
        }
    }
}

impl Drawable for Container {
    fn draw(&self, canvas: &skia_safe::Canvas) {
        canvas.save();
        for child in self.children.iter() {
            let layout = child.get_layout();
            child.draw(canvas);

            // TODO: This should probably just be factored as part of the layout
            let margin_height = child.margin_size().height;
            let padding_height = child.padding_size().height;
            canvas.translate((0.0, layout.size.height + margin_height + padding_height));
        }
        canvas.restore();
    }

    fn compute_layout(&mut self, bounds: Bounds) -> Layout {
        let mut y = 0.0;
        let mut width = 0.0;
        for child in self.children.iter_mut() {
            let layout = child.compute_layout(Bounds {
                size: Size {
                    width: bounds.size.width,
                    height: bounds.size.height,
                },
            });
            child.get_layout().position = Position { x: 0.0, y };
            y += layout.size.height;
            width = layout.size.width;
        }
        
        Layout {
            position: Position { x: 0.0, y: 0.0 },
            size: Size {
                width,
                height: y,
            },
        }
    }
    
    fn get_layout(&self) -> Layout {
        Layout {
            position: Position { x: 0.0, y: 0.0 },
            size: Size {
                width: 0.0,
                height: 0.0,
            },
        }
    }

    fn get_styles(&self) -> &Style {
        &self.style
    }
    
    fn set_styles(&mut self, style: Style) {
        self.style = style;
    }
    
    fn get_styles_mut(&mut self) -> &mut Style {
        &mut self.style
    }
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

        canvas.translate((300.0, 300.0));

        let mut container = self.render();
        container.compute_layout(Bounds {
            size: Size {
                width: 1800.0,
                height: 1800.0,
            },
        });

        canvas.save();
        container.draw(canvas);
        canvas.restore();
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

    fn render(&mut self) -> Container {
        let mut root = Container::new();
        for pane in self.panes.iter() {
            let mut container =
                RoundedRect::new(20.0, Color::parse_hex("#353f38"))
                    .margin(30.0)
                    .padding(30.0);
          
            let file = &self.files[pane.file];
            let lines = file.get_lines_for_text_view(&pane.text_view.as_ref().unwrap());
            for (index, line) in lines.enumerate() {
               container.children.push(Box::new(Text::new(
                   from_utf8(line).unwrap().to_string(),
               )));
            }
            root.children.push(Box::new(container));
        }
        root
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
