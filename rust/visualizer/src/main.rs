use std::{fs::{self, File}, collections::{HashMap}, hash::{Hash, Hasher}, io::{BufReader, BufRead}, cmp::max};


use block::{Block, CodeLocation};

use draw::Color;
use fps::FpsCounter;
use graph::{make_method_graph, call_graphviz_command_line, call_graphviz_in_new_thread, Promise};
use itertools::Itertools;
use serde::{Serialize, Deserialize};
use serde_json::Deserializer;
use skia_safe::{Rect, PaintStyle, Font, Typeface, FontStyle, Contains, Canvas, Image, Data};
use window::Driver;

mod window;
mod block;
mod native;
mod draw;
mod graph;
mod fps;


use syntect::{easy::HighlightLines, highlighting, parsing::SyntaxReference};
use syntect::parsing::SyntaxSet;
use syntect::highlighting::{ThemeSet};
use syntect::util::{LinesWithEndings};



fn draw_color_syntax(highlighted_code: Vec<Vec<(highlighting::Style, String)>>, canvas: &mut Canvas) -> usize {
    // Load these once at the start of your program
    let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(), 24.0);
    let mut x = 0;
    let mut y = 0;
    for line in highlighted_code.iter() {
        for (style, text) in line {
            let foreground = style.foreground;
            let color = Color::from_syntect_color(&foreground);
            canvas.draw_str(text, (x as f32, y as f32), &font, &color.to_paint());
            x += text.len() * 12;
        }
        y += 32;
        x = 0;
    }
    y
}

fn color_syntax_code(code: &str, ps: &SyntaxSet, syntax: &SyntaxReference) -> Vec<Vec<(highlighting::Style, String)>> {
    let ts = ThemeSet::load_defaults();

    let mut h = HighlightLines::new(syntax, &ts.themes["base16-ocean.dark"]);
    let mut lines = vec![];
    for line in LinesWithEndings::from(code) {
        let ranges = h.highlight_line(line, ps).unwrap();
        lines.push(ranges.into_iter().map(|(style, text)| (style, text.to_string())).collect());
    }
    lines
}

fn color_syntax_ruby(code: &str) -> Vec<Vec<(highlighting::Style, String)>> {
    // TODO: I'm only supposed load this once
    let ps = SyntaxSet::load_defaults_newlines();
    let syntax = ps.find_syntax_by_extension("rb").unwrap();
    color_syntax_code(code, &ps, syntax)
}

fn color_syntax_arm_assembly(code: &str) -> Vec<Vec<(highlighting::Style, String)>> {
    let ps = SyntaxSet::load_from_folder("./syntaxes").unwrap();
    let syntax = ps.find_syntax_by_name("ARM Assembly").unwrap();
    color_syntax_code(code, &ps, syntax)
}



struct Caches {
    ruby_source_method_cache: HashMap<Method, String>,
    ruby_source_highlight_cache: HashMap<Method, Vec<Vec<(highlighting::Style, String)>>>,
    assembly_highlight_cache: HashMap<Method, Vec<Vec<(highlighting::Style, String)>>>,
    ruby_method_image_cache: HashMap<Method, Image>,
}



#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
enum Scene {
    Overview,
    Method { file_name: String, method: Method },
}


pub struct Style {
    pub background_color: Color,
    pub primary_text_color: Color,
    pub exit_text_color: Color,
    pub outer_block_color: Color,
    pub primary_rect_color: Color,
}





struct Visualizer {
    scroll_offset_y: f64,
    scroll_offset_x: f64,
    code_files: Vec<CodeFile>,
    mouse_position: (f32, f32),
    max_draw_height: f64,
    scene: Scene,
    mouse_clicked: bool,
    style: Style,
    caches: Caches,
    records: Vec<Block>,
    timeline: HashMap<usize, Vec<Block>>,
    fps_counter: FpsCounter,
    canvas_size: (i32, i32),
    max_draw_width: f64,
    method_graph_promises: HashMap<Method, Promise<Vec<u8>>>,
}

impl Visualizer {

    fn get_method_ruby_source(&mut self, method: &Method) -> Option<String> {
        // We might have tried to get the file and we couldn't
        // in that case we store None in the cache so we don't
        // keep trying to read the file that might not exist.
        if self.caches.ruby_source_method_cache.contains_key(method) {
            if let Some(source) = self.caches.ruby_source_method_cache.get(method) {
                return Some(source.clone())
            }
            return None
        }

        if let Some(source) = method.get_ruby_source() {
            self.caches.ruby_source_method_cache.insert(method.clone(), source.clone());
            return Some(source);
        }
        None
    }

    fn get_image_for_method(&mut self, method: &Method) -> Option<Image> {
        if self.caches.ruby_method_image_cache.contains_key(method) {
            return self.caches.ruby_method_image_cache.get(method).cloned();
        }
        if self.method_graph_promises.contains_key(method) {
            return None
        }

        let graph = make_method_graph(&self.style, &self.records, method);
        let mut promise = call_graphviz_in_new_thread(&graph);

        if promise.ready() {
            let raw_data = promise.get().unwrap();
            let data = Data::new_copy(&raw_data);
            let image = Image::from_encoded(&data).unwrap();
            // let width: usize = image.width() as usize / 2;
            // let height: usize = image.height() as usize / 2;
            // let image_info = ImageInfo::new_n32_premul((width as i32, height as i32), None);
            // let row_bytes = width * 4;
            // let pixels = vec![0; height as usize * row_bytes as usize];
            // let pixmap = Pixmap::new(&image_info, &pixels, row_bytes);
            // if !image.scale_pixels(&pixmap, SamplingOptions::default(), None) {
            //     println!("Didn't scale pixels");
            // }
            // if let Some(image) = Image::from_raster_data(&image_info,Data::new_copy(&pixels), row_bytes) {
            //     self.caches.ruby_method_image_cache.insert(method.clone(), image.clone());
            // }
            self.caches.ruby_method_image_cache.insert(method.clone(), image.clone());
            return Some(image)
        } else {
            self.add_method_graph_promise(method.clone(), promise);
            None
        }



    }

    fn get_ruby_highlighted_source(&mut self, method: &Method) -> Option<Vec<Vec<(highlighting::Style, String)>>> {
        if self.caches.ruby_source_highlight_cache.contains_key(method) {
            return self.caches.ruby_source_highlight_cache.get(method).cloned();
        }
        let source = self.get_method_ruby_source(method)?;
        let highlighted = color_syntax_ruby(&source);
        self.caches.ruby_source_highlight_cache.insert(method.clone(), highlighted.clone());
        Some(highlighted)
    }

    fn get_assembly_highlight_source(&mut self, method: &Method) -> Option<Vec<Vec<(highlighting::Style, String)>>> {
        if self.caches.assembly_highlight_cache.contains_key(method) {
            return self.caches.assembly_highlight_cache.get(method).cloned();
        }
        let source = method.get_assembly_source();
        let highlighted = color_syntax_arm_assembly(&source);
        self.caches.assembly_highlight_cache.insert(method.clone(), highlighted.clone());
        Some(highlighted)
    }

    fn mouse_clicked_at(&self, canvas: &Canvas) -> bool {
        if self.mouse_clicked {
            if let Some(rect) = canvas.device_clip_bounds() {
                if rect.contains(&Rect::from_xywh(self.mouse_position.0, self.mouse_position.1, 1.0, 1.0)) {
                    return true
                }
            }
        }
        false
    }


    fn draw_overview(&mut self, canvas: &mut Canvas) {
        canvas.translate((0.0, 150.0));

        let exit_paint = self.style.exit_text_color.to_paint();
        let primary_paint = self.style.primary_rect_color.to_paint();
        let primary_text_paint = self.style.primary_text_color.to_paint();

        // let heading_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 128.0);
        // canvas.draw_str("YJIT", (0, 0), &heading_font, &primary_paint);

        // canvas.translate((0.0, 100.0));

        let heading_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 64.0);
        canvas.draw_str("Timeline", (0, 0), &heading_font, &primary_text_paint);
        canvas.translate((0.0, 20.0));


        let mut x = 0;

        let mut keys = self.timeline.keys().cloned().collect::<Vec<usize>>();
        keys.sort();

        let (first, last) = (keys.first().unwrap(), keys.last().unwrap());



        let border = Rect::from_xywh(0.0, 0.0, 2900.0, 400.0);
        let mut border_paint = primary_paint.clone();
        border_paint.set_style(PaintStyle::Stroke);
        border_paint.set_stroke_width(5.0);
        canvas.draw_rect(border, &border_paint);

        canvas.save();
        canvas.clip_rect(border.with_inset((30.0, 30.0)), None, None);


        canvas.translate((30.0, 30.0));



        let text_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(), 24.0);


        let height = 3;

        let y = 400.0;
        canvas.draw_str("0", (0.0, y - 32.0 * 2.0), &text_font, &primary_paint);
        canvas.draw_str("25", (0.0, y - 32.0 * 2.0 - height as f32 * 25.0), &text_font, &primary_text_paint);
        canvas.draw_str("50", (0.0, y - 32.0 * 2.0 - height as f32 * 50.0), &text_font, &primary_text_paint);
        canvas.draw_str("75", (0.0, y - 32.0 * 2.0 - height as f32 * 75.0), &text_font, &primary_text_paint);
        canvas.draw_str("100", (0.0, y - 32.0 * 2.0 - height as f32 * 100.0), &text_font, &primary_text_paint);
        canvas.translate((24.0*2.0 + 10.0, 0.0));

        for time in (*first..*last).rev().take(220).rev() {
            let width = 10;
            let mut y = 400;
            if let Some(entries) = self.timeline.get(&time) {

                for block in entries.iter() {
                    let paint = if block.is_exit {
                        &exit_paint
                    } else {
                        &primary_paint
                    };

                    let rect = Rect::new(
                        x as f32,
                        y as f32 - height as f32,
                        x as f32 + width as f32,
                        y as f32,
                    );
                    canvas.draw_rect(rect, paint);
                    y -= 1;
                }
            }
            x += width + 4;
        }

        canvas.restore();

        canvas.translate((0.0, 530.0));

        let heading_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 64.0);
        canvas.draw_str("Methods", (0, 0), &heading_font, &primary_text_paint);
        canvas.translate((0.0, 20.0));



        let width = 2600;
        let rect_width = 950.0;
        let margin = 30;

        let heading_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
        let text_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(), 24.0);


        let mut columns: [i32; 3] = [0, 0, 0];


        let mut x = 0;
        let mut rect_paint = self.style.primary_rect_color.to_paint();
        rect_paint.set_style(PaintStyle::Stroke);
        let mut column = 0;


        for file in self.code_files.iter() {

            let mut y = columns[column];

            if y as f64 + self.scroll_offset_y > self.canvas_size.1 as f64 {
                self.max_draw_height = y as f64 + 500.0;
                column = (column + 1) % columns.len();
                x += rect_width as i32 + margin;
                if x > width {
                    x = 0;
                }
                continue;

            }
            canvas.save();

            let text_paint = self.style.primary_text_color.to_paint();


            canvas.draw_str(&file.name, (x as f32 + 56.0, y as f32 + 56.0), &heading_font, &text_paint);

            let mut hr_paint = rect_paint.clone();
            hr_paint.set_style(PaintStyle::Fill);

            y += 64;

            canvas.draw_rect(Rect::from_xywh(x as f32, y as f32 + 32.0, rect_width, 9.0), &hr_paint);

            {
                canvas.save();
                canvas.translate((x, 0));


                let x = 32.0;
                y += 32 + 64;
                for method in file.methods.iter() {
                    let mut text_paint = self.style.primary_text_color.to_paint();

                    canvas.save();

                    canvas.clip_rect(Rect::from_xywh(0.0, y as f32 - 24.0, rect_width - margin as f32, 32.0), None, None);

                    if let Some(device_position) = canvas.device_clip_bounds() {
                        if device_position.contains(Rect::from_xywh(self.mouse_position.0, self.mouse_position.1, 1.0, 1.0)) {
                            text_paint = Color::parse_hex("#ffffff").to_paint();
                        }
                    }
                    if self.mouse_clicked_at(canvas) {
                        // I can't call a method with &mut here, but I can do this.
                        // I think this is the most recent case (maybe only) where I'm a bit
                        // annoyed with the borrow checker. I'm sure there is a reason,
                        // but it is hard for me to see how these are different.
                        self.scene = Scene::Method { file_name: file.name.clone(), method: method.clone() };
                        self.scroll_offset_y = 0.0;
                        self.scroll_offset_x = 0.0;
                    }

                    canvas.save();
                    canvas.clip_rect(Rect::from_xywh(0.0, y as f32 - 24.0, 480_f32, 32.0), None, None);

                    canvas.draw_str(&method.name, (x as f32, y as f32), &text_font, &text_paint);
                    canvas.restore();

                    {
                        let mut x = 0;
                        canvas.save();
                        canvas.translate((x as f32 + 500.0, y as f32 - 16.0_f32));

                        canvas.clip_rect(Rect::from_xywh(0.0, 0.0, rect_width - 500.0 - margin as f32, 24.0), None, None);

                        for block in method.blocks.iter() {
                            let size = block.disasm.len();
                            let rect_width = (size / 200 + 1) as f32;
                            let rect = Rect::from_xywh(x as f32, 0.0, rect_width, 24.0);
                            let mut rect_paint = if block.is_exit {
                                self.style.exit_text_color.to_paint()
                            } else {
                                rect_paint.clone()
                            };
                            rect_paint.set_style(PaintStyle::Fill);
                            canvas.draw_rect(rect, &rect_paint);
                            x += rect_width as i32 + 3;
                        }
                        canvas.restore();
                    }
                    y += 32;
                    canvas.restore();
                }

                canvas.restore();
            }


            let height = y - columns[column];

            let rect = Rect::from_xywh(x as f32, columns[column] as f32, rect_width, height as f32);
            canvas.draw_rect(rect, &rect_paint);



            columns[column] = y + margin;
            column = (column + 1) % columns.len();

            x += rect_width as i32 + margin;
            if x > width {
                x = 0;
            }


            canvas.restore();

        }

        self.max_draw_height = *columns.iter().max().unwrap() as f64;

    }

    fn draw_method(&mut self, canvas: &mut Canvas, method: &Method) {

        canvas.clear(self.style.background_color.to_color4f());

        canvas.translate((0.0, 100.0));
        let heading_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 128.0);

        let text_paint = self.style.primary_text_color.to_paint();

        if self.mouse_clicked_at(canvas) {
            self.change_scene(Scene::Overview);
        }

        let mut x = 0;
        let mut y = 0;

        canvas.draw_str(&method.name, (x as f32 + 56.0, y as f32 + 56.0), &heading_font, &text_paint);

        y += 100;


        // text_paint.set_style(PaintStyle::Fill);

        // canvas.draw_str(&format!("{:?}", method.location), (x as f32 + 56.0, y as f32 + 56.0), &text_font, &text_paint);
        y += 100;
        {
            canvas.save();
            canvas.clip_rect(Rect::from_xywh(0.0, y as f32 - 24.0, 950.0, 160000.0), None, None);
            if let Some(source) = self.get_ruby_highlighted_source(method) {
                canvas.save();
                canvas.translate((x as f32 + 56.0, y as f32 + 56.0));
                let height = draw_color_syntax(source, canvas);
                self.max_draw_height = max(self.max_draw_height as usize, height) as f64;
                canvas.restore();
            }


            canvas.restore();
        }
        // x += 1000;
        // {
        //     if let Some(source) = self.get_assembly_highlight_source(method) {
        //         canvas.save();
        //         canvas.translate((x as f32 + 56.0, y as f32 + 56.0));
        //         let height = draw_color_syntax(source, canvas);
        //         self.max_draw_height = max(self.max_draw_height as usize, height) as f64;
        //         canvas.restore();
        //     }
        // }

        x += 1000;

        if let Some(image) = self.get_image_for_method(method) {

            let image_height = image.height();
            let image_width = image.width();
            let canvas_width = self.canvas_size.0;
            let canvas_height = self.canvas_size.1;
            canvas.draw_image(image, (x as f32 + 56.0, y as f32 + 56.0), None);
            self.max_draw_height = max(self.max_draw_height as usize, y + (image_height as usize).saturating_sub(canvas_height as usize) + 300) as f64;
            self.max_draw_width = max(self.max_draw_width as usize, x + (image_width as usize).saturating_sub(canvas_width as usize) + 300) as f64;
        }

    }

    fn change_scene(&mut self, scene: Scene) {
        self.scene = scene;
        self.scroll_offset_y = 0.0;
        self.scroll_offset_x = 0.0;
    }

    fn add_method_graph_promise(&mut self, method: Method, promise: graph::Promise<Vec<u8>>)  {
       self.method_graph_promises.insert(method, promise);
    }


}


fn deindent(s: &str) -> String {
    let mut lines = s.lines();
    let first_line = lines.next().unwrap();
    let indent = first_line.chars().take_while(|c| c.is_whitespace()).count();
    let mut result = String::new();
    result.push_str(first_line.trim_start());
    result.push('\n');
    for line in lines {
        if line.len() > indent {
            result.push_str(&line[indent..]);
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }
    result
}


impl Driver for Visualizer {

    fn get_title(&self) -> String {
        "Visualizer".to_string()
    }

    fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_position = (x, y);
    }

    fn update(&mut self) {

        if self.scene == Scene::Overview {
            self.max_draw_width = 0.0;
        }

        if self.scroll_offset_y > 0.0 {
            self.scroll_offset_y = 0.0;
        }
        if self.scroll_offset_x > 0.0 {
            self.scroll_offset_x = 0.0;
        }


        if self.scroll_offset_y < -self.max_draw_height {
            self.scroll_offset_y = -self.max_draw_height;
        }

        if self.scroll_offset_x < -self.max_draw_width {
            self.scroll_offset_x = -self.max_draw_width;
        }

        let mut to_delete = vec![];

        for (method, promise) in self.method_graph_promises.iter_mut() {
            if promise.ready() {
                to_delete.push(method.clone());
                let raw_data = promise.get().unwrap();
                let data = Data::new_copy(&raw_data);
                let image = Image::from_encoded(&data).unwrap();
                self.caches.ruby_method_image_cache.insert(method.clone(), image.clone());
            }
        }
        for method in to_delete.iter() {
            self.method_graph_promises.remove(method);
        }



    }

    fn init(&mut self) {
        native::set_smooth_scroll();
    }

    fn add_event(&mut self, event: &winit::event::Event<'_, ()>) {
        match event {
            winit::event::Event::WindowEvent { window_id: _, event } => {
                use winit::event::WindowEvent::*;
                match event {
                    MouseWheel { device_id: _, delta, phase: _, .. } => {
                        match delta {
                            winit::event::MouseScrollDelta::LineDelta(_x, _y) => {
                                println!("Line?");
                            }
                            winit::event::MouseScrollDelta::PixelDelta(pos) => {
                                self.scroll_offset_y += pos.y;
                                self.scroll_offset_x -= pos.x;
                            }
                        }
                    }
                    MouseInput { device_id: _, state, button, .. } => {
                        use winit::event::*;
                        match (state, button) {
                            (ElementState::Pressed, MouseButton::Left) => {

                            }
                            (ElementState::Released, MouseButton::Left) => {
                                self.mouse_clicked = true;
                            }
                            _ => {}
                        }
                    },
                    _ => {}
                }
            },
           _ => {}
        }
    }

    fn before_draw(&mut self) {
        self.max_draw_height = 0.0;
    }

    fn draw(&mut self, canvas: &mut Canvas) {

        let size = canvas.base_layer_size();
        self.canvas_size = (size.width, size.height);

        self.fps_counter.tick();
        canvas.clear(self.style.background_color.to_color4f());

        // let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
        // let text_color = self.style.primary_text_color.to_paint();

        // canvas.draw_str(self.fps_counter.fps.to_string(), (3000.0 - 60.0, 30.0), &font, &text_color);

        canvas.translate((self.scroll_offset_x as i32 + 100, self.scroll_offset_y as i32));
        let scene = self.scene.clone();
        match &scene {
            Scene::Overview => {
                self.draw_overview(canvas);
            }
            Scene::Method { file_name: _, method } => {
                self.draw_method(canvas, &method.clone());
            }
        }
    }

    fn next_frame(&mut self) {
        self.mouse_clicked = false;

    }
}




#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct CodeFile {
    name: String,
    full_path: String,
    methods: Vec<Method>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Method {
    pub name: String,
    pub location: CodeLocation,
    pub blocks: Vec<Block>,
}

impl Method {
    fn get_ruby_source(&self) -> Option<String> {
        if let Some(file) = &self.location.file {
            if let Ok(file) = File::open(file) {
                let reader = BufReader::new(file);
                let lines = reader.lines();
                let mut line = 1;
                let mut source = String::new();
                for l in lines {
                    if let Ok(l) = l {
                        // TODO: Unindent total indention but keep relative indention
                        if line >= self.location.line_start.0 && line <= self.location.line_end.0 {
                            source.push_str(&l);
                            source.push('\n');
                        }
                        line += 1;
                    }
                }
                return Some(deindent(&source));
            }
        }
        None
    }

    fn get_assembly_source(&self) -> String {
        let mut records = self.blocks.clone();
        let mut text = String::new();
        records.sort_by_key(|x| x.block_id.idx);
        records.dedup_by_key(|x| x.disasm.clone());
        for record in records {
            text.push_str(&format!("{:?}\n", record.block_id.idx));
            text.push_str(&record.disasm);
            text.push('\n');
        }
       text
    }

}


fn main() {



    let path = "/Users/jimmyhmiller/Documents/Code/yjit-bench/yjit.log";
    // let path = "/Users/jimmyhmiller/Documents/Code/ruby/yjit.log";
    let contents = fs::read_to_string(path).unwrap();
    let mut records: Vec<_> = Deserializer::from_str(&contents)
        .into_iter::<Block>()
        .filter_map(|x| x.ok())
        .collect();

    for record in records.iter_mut() {
        record.is_exit = record.disasm.contains("exit to interpreter");
    }

    records.sort_by_key(|x| x.location.file.clone().unwrap_or_else(|| "unknown".to_string()));


    let mut code_files : Vec<CodeFile> = vec![];


    // TODO: Clean up
    for (file, group) in records.iter().group_by(|x| x.location.file.clone()).into_iter() {
        // let file = location.file.clone();
        let full_path = file.unwrap_or_else(|| "unknown".to_string()).to_string();
        let name = full_path.split('/').last().unwrap().to_string();
        let mut code_file = CodeFile {
            name,
            full_path,
            methods: vec![],
        };



        group
            .sorted_by_key(|x| x.location.method_name.as_ref())
            .group_by(|x| x.location.method_name.as_ref())
            .into_iter().for_each(|(method_name, group)| {

            let mut group = group.collect::<Vec<_>>();
            let location = group.first().unwrap().location.clone();

            group.sort_by_key(|x| (x.id, -(x.epoch as i32)));
            let group = group.iter().dedup_by(|x, y| x.id == y.id);


            let method = Method {
                name: (method_name.cloned().unwrap_or_else(|| "unknown".to_string())),
                location,
                blocks: group.map(|x| x.clone().clone()).collect(),
            };
            code_file.methods.push(method);
        });
        code_files.push(code_file);
    }

    code_files.sort_by_key(|x| x.methods.len());
    code_files.reverse();

    records.sort_by_key(|x| x.created_at);
    let timeline : HashMap<usize, Vec<Block>> = records
        .iter()
        .group_by(|x| x.created_at/25)
        .into_iter()
        .map(|(key, group)| (key, group.cloned().collect())).collect();

    let fps_counter = FpsCounter::new();

    let business_style = Style {
        background_color: Color::parse_hex("#ffffff"),
        primary_text_color: Color::parse_hex("#474747"),
        exit_text_color: Color::parse_hex("#fd5e53"),
        outer_block_color: Color::parse_hex("#5e5efd"),
        primary_rect_color: Color::parse_hex("#666666"),
    };

    let hacker_style = Style {
        background_color: Color::parse_hex("#210522"),
        primary_text_color: Color::parse_hex("#5a8a5e"),
        exit_text_color: Color::parse_hex("#fd5e53"),
        outer_block_color: Color::parse_hex("#5e5efd"),
        primary_rect_color: Color::parse_hex("#5a8a5e"),
    };


    let visualizer = Visualizer {
        scroll_offset_y: 0.0,
        scroll_offset_x: 0.0,
        records,
        code_files,
        mouse_position: (0.0, 0.0),
        max_draw_height: 0.0,
        max_draw_width: 0.0,
        scene: Scene::Overview,
        mouse_clicked: false,
        timeline,
        fps_counter,
        canvas_size: (0,0),
        style: hacker_style,
        caches: Caches {
            ruby_source_method_cache: HashMap::new(),
            ruby_source_highlight_cache: HashMap::new(),
            assembly_highlight_cache: HashMap::new(),
            ruby_method_image_cache: HashMap::new(),

        },
        method_graph_promises: HashMap::new(),
    };

    window::setup_window(visualizer);

}


// TODO:
// In each method, merge blocks that are the same (to be defined)
// Capture when a block has multiple version (in same method or different?)
// Make it so you can click on a method and see a detailed view.
// Allow you to read the disassembly of a block
// as well as see a graph
// That graph should let you see its transformation through time
// Add time stamps to data so I can have a timeline
// Capture things like block invalidation so I can show those as well.
// Add PID to the blocks so I can compare across different runs
// Actually send the data as the program is running
// Persist data to disk and don't require all data to be in memory.
