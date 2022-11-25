use std::{fs::{self, File}, collections::{hash_map::DefaultHasher, HashMap}, hash::{Hash, Hasher}, io::{BufReader, BufRead}, cmp::max};


use block::{Block, CodeLocation};

use draw::Color;
use itertools::Itertools;
use serde::{Serialize, Deserialize};
use serde_json::Deserializer;
use skia_safe::{Rect, PaintStyle, Font, Typeface, FontStyle, Contains, Canvas};
use window::Driver;

mod window;
mod block;
mod native;
mod draw;



struct Caches {
    ruby_source_method_cache: HashMap<Method, String>,
}



#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
enum Scene {
    Overview,
    Method { file_name: String, method: Method },
}


struct Style {
    background_color: Color,
    primary_text_color: Color,
}





struct Visualizer {
    scroll_offset: f64,
    code_files: Vec<CodeFile>,
    mouse_position: (f32, f32),
    max_draw_height: f64,
    scene: Scene,
    mouse_clicked: bool,
    style: Style,
    caches: Caches,
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

    fn mouse_clicked_at(&self, canvas: &Canvas) -> bool {
        if self.mouse_clicked {
            println!("Yes clicked");
            if let Some(rect) = canvas.device_clip_bounds() {
                if rect.contains(&Rect::from_xywh(self.mouse_position.0, self.mouse_position.1, 1.0, 1.0)) {
                    return true
                }
            }
        }
        false
    }


    fn draw_overview(&mut self, canvas: &mut Canvas) {
        canvas.translate((0.0, 100.0));

        // let jungle_green = Color::parse_hex("#62b4a6");


        let width = 2600;
        let rect_width = 950.0;
        let margin = 30;


        let heading_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
        let text_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(), 24.0);



        canvas.clear(self.style.background_color.to_color4f());

        let mut columns: [i32; 3] = [0, 0, 0];


        let mut x = 0;
        let mut rect_paint = self.style.primary_text_color.to_paint();
        rect_paint.set_style(PaintStyle::Stroke);
        let mut column = 0;


        for file in self.code_files.iter() {
            canvas.save();
            let mut y = columns[column];

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
                        self.scroll_offset = 0.0;
                    }

                    canvas.save();
                    canvas.clip_rect(Rect::from_xywh(0.0, y as f32 - 24.0, 480 as f32, 32.0), None, None);

                    canvas.draw_str(&method.name, (x as f32, y as f32), &text_font, &text_paint);
                    canvas.restore();

                    {
                        let mut x = 0;
                        canvas.save();
                        canvas.translate((x as f32 + 500.0, y as f32 - 16.0 as f32));

                        canvas.clip_rect(Rect::from_xywh(0.0, 0.0, rect_width - 500.0 - margin as f32, 24.0), None, None);


                        for block in method.blocks.iter() {
                            let size = block.disasm.len();
                            // let rect_width = if size < 100 {
                            //     1.0
                            // } else if size < 1000 {
                            //     4.0
                            // } else if size < 10000 {
                            //     8.0
                            // } else {
                            //     16.0
                            // };
                            let rect_width = (size / 200 + 1) as f32;
                            let rect = Rect::from_xywh(x as f32, 0.0, rect_width, 24.0);
                            canvas.draw_rect(rect, &text_paint);
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
                // y += rect_height as i32 + margin;
            }

            canvas.restore();

        }

        self.max_draw_height = columns.iter().max().unwrap().clone() as f64;

    }

    fn draw_method(&mut self, canvas: &mut Canvas, file_name: &String, method: &Method) {

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
        let text_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(), 32.0);
        let mut text_paint = self.style.primary_text_color.to_paint();
        text_paint.set_style(PaintStyle::Fill);

        canvas.draw_str(&format!("{:?}", method.location), (x as f32 + 56.0, y as f32 + 56.0), &text_font, &text_paint);
        y += 100;
        {
            canvas.save();
            canvas.clip_rect(Rect::from_xywh(0.0, y as f32 - 24.0, 950.0, 1600.0), None, None);
            let mut y = y;
            if let Some(source) = self.get_method_ruby_source(method) {
                for line in source.lines() {
                    canvas.draw_str(line, (x as f32 + 56.0, y as f32 + 56.0), &text_font, &text_paint);
                    y += 32;
                }
            }
            self.max_draw_height = max(self.max_draw_height as usize, y as usize) as f64;

            canvas.restore();
        }
        x += 1000;
        {
            let mut y = y;
            for line in method_to_text(method).lines() {
                canvas.draw_str(line, (x as f32 + 56.0, y as f32 + 56.0), &text_font, &text_paint);
                y += 32;
            }
            self.max_draw_height = max(self.max_draw_height as usize, y as usize) as f64;
        }

    }

    fn change_scene(&mut self, scene: Scene) {
        self.scene = scene;
        self.scroll_offset = 0.0;
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

    fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_position = (x, y);
    }

    fn update(&mut self) {
        if self.scroll_offset > 0.0 {
            self.scroll_offset = 0.0;
        }
        if self.scroll_offset < -self.max_draw_height {
            self.scroll_offset = -self.max_draw_height;
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
                                self.scroll_offset += pos.y;
                            }
                        }
                    }
                    MouseInput { device_id: _, state, button, .. } => {
                        use winit::event::*;
                        match (state, button) {
                            (ElementState::Pressed, MouseButton::Left) => {
                                // let mut rng = rand::thread_rng();
                                // let method = self.record_by_method.keys().collect::<Vec<_>>().choose(&mut rng).unwrap().clone();
                                // let records = &self.record_by_method[method];

                                // self.text = method_to_text(method, records);
                                // self.scroll_offset = 0.0;
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

        canvas.translate((100, self.scroll_offset as i32));
        let scene = self.scene.clone();
        match &scene {
            Scene::Overview => {
                self.draw_overview(canvas);
            }
            Scene::Method { file_name, method } => {
                self.draw_method(canvas, file_name, &method.clone());
            }
        }
    }

    fn next_frame(&mut self) {
        self.mouse_clicked = false;

    }
}





fn method_to_text(method: &Method) -> String {
    let mut records = method.blocks.clone();
    let mut text = String::new();
    records.sort_by_key(|x| x.block_id.idx);
    records.dedup_by_key(|x| x.disasm.clone());
    for record in records {
        text.push_str(&format!("{:?}\n", record.block_id.idx));
        text.push_str(&record.disasm);
        text.push_str("\n");
    }
   return text;
}


fn _calculate_hash<T: Hash>(t: &T) -> String {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish().to_string()
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct CodeFile {
    name: String,
    full_path: String,
    methods: Vec<Method>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct Method {
    name: String,
    location: CodeLocation,
    blocks: Vec<Block>,
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
                            source.push_str("\n");
                        }
                        line += 1;
                    }
                }
                return Some(deindent(&source));
            }
        }
        None
    }
}


fn main() {



    let path = "/Users/jimmyhmiller/Documents/Code/yjit-bench/yjit.log";
    // let path = "/Users/jimmyhmiller/Documents/Code/ruby/yjit.log";
    let contents = fs::read_to_string(path).unwrap();
    let mut records: Vec<_> = Deserializer::from_str(&contents)
        .into_iter::<Block>()
        .filter(|x| x.is_ok())
        .map(|x| x.unwrap())
        .collect();

    records.sort_by_key(|x| x.location.file.clone().unwrap_or_else(|| "unknown".to_string()));


    let mut code_files : Vec<CodeFile> = vec![];


    // TODO: Clean up
    for (file, group) in records.iter().group_by(|x| x.location.file.clone()).into_iter() {
        // let file = location.file.clone();
        let full_path = file.map(|x| x.clone()).unwrap_or_else(|| "unknown".to_string()).to_string();
        let name = full_path.split("/").last().unwrap().to_string();
        let mut code_file = CodeFile {
            name,
            full_path,
            methods: vec![],
        };



        group
            .sorted_by_key(|x| x.location.method_name.as_ref())
            .group_by(|x| x.location.method_name.as_ref())
            .into_iter().for_each(|(method_name, group)| {

            let group = group.collect::<Vec<_>>();

            let location = group.first().unwrap().location.clone();


            let method = Method {
                name: (method_name.map(|x| x.clone()).unwrap_or_else(|| "unknown".to_string())).to_string(),
                location: location.clone(),
                blocks: group.iter().map(|x| x.clone().clone()).collect(),
            };
            code_file.methods.push(method);
        });
        code_files.push(code_file);
    }

    code_files.sort_by_key(|x| x.methods.len());
    code_files.reverse();

    let visualizer = Visualizer {
        scroll_offset: 0.0,
        code_files,
        mouse_position: (0.0, 0.0),
        max_draw_height: 0.0,
        scene: Scene::Overview,
        mouse_clicked: false,
        style: Style {
            background_color: Color::parse_hex("#210522"),
            primary_text_color: Color::parse_hex("#5a8a5e"),
        },
        caches: Caches {
            ruby_source_method_cache: HashMap::new(),
        },
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
