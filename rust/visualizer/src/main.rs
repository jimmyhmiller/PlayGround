use std::{fs::{self}, collections::{hash_map::DefaultHasher}, hash::{Hash, Hasher}};


use block::{Block, CodeLocation};

use draw::Color;
use itertools::Itertools;
use serde_json::Deserializer;
use skia_safe::{Canvas, textlayout::{FontCollection, ParagraphStyle, ParagraphBuilder, TextStyle, Paragraph}, FontMgr, Rect, PaintStyle, Font, Typeface, FontStyle, Contains, Point};
use window::Driver;

mod window;
mod block;
mod native;
mod draw;



struct Visualizer {
    scroll_offset: f64,
    code_files: Vec<CodeFile>,
    mouse_position: (f32, f32),
    max_draw_height: f64,
}
impl Visualizer {
    // fn draw_paragraph(&mut self, color: Color, canvas: &mut Canvas) {

    //     if let Some(paragraph) = &self.paragraph {
    //         paragraph.paint(canvas, (0, self.scroll_offset as i32));
    //     } else {
    //         let mut font_collection = FontCollection::new();
    //         font_collection.set_default_font_manager(FontMgr::new(), "Ubuntu Mono");
    //         let paragraph_style = ParagraphStyle::new();
    //         let mut paragraph_builder = ParagraphBuilder::new(&paragraph_style, font_collection);
    //         let mut ts = TextStyle::new();
    //         ts.set_foreground_color(color.to_paint());
    //         ts.set_font_size(32.0);
    //         paragraph_builder.push_style(&ts);
    //         paragraph_builder.add_text(&self.text);

    //         let mut paragraph = paragraph_builder.build();
    //         paragraph.layout(1024.0);
    //         paragraph.paint(canvas, (0, self.scroll_offset as i32));
    //         self.paragraph = Some(paragraph);
    //     }


    // }
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
                    MouseWheel { device_id, delta, phase, modifiers } => {
                        match delta {
                            winit::event::MouseScrollDelta::LineDelta(x, y) => {
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
                            _ => {}
                        }
                    },
                    _ => {}
                }
            },
           _ => {}
        }
    }


    fn draw(&mut self, canvas: &mut skia_safe::Canvas) {

        canvas.translate((0.0, 100.0));

        // let jungle_green = Color::parse_hex("#62b4a6");


        let width = 2600;
        let rect_width = 950.0;
        let margin = 30;


        let yellow = Color::parse_hex("#5a8a5e");
        let dark_green = Color::parse_hex("#210522");
        let heading_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
        let text_font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(), 24.0);



        canvas.clear(dark_green.to_color4f());
        canvas.translate((100, self.scroll_offset as i32));

        let mut columns: [i32; 3] = [0, 0, 0];


        let mut x = 0;
        let mut rect_paint = yellow.to_paint();
        rect_paint.set_style(PaintStyle::Stroke);
        let mut column = 0;


        for file in self.code_files.iter() {
            canvas.save();
            let mut y = columns[column];

            let text_paint = yellow.to_paint();


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
                    let mut text_paint = yellow.to_paint();

                    canvas.save();

                    canvas.clip_rect(Rect::from_xywh(0.0, y as f32 - 24.0, rect_width - margin as f32, 32.0), None, None);

                    if let Some(device_position) = canvas.device_clip_bounds() {
                        if device_position.contains(Rect::from_xywh(self.mouse_position.0, self.mouse_position.1, 1.0, 1.0)) {
                            text_paint = Color::parse_hex("#ffffff").to_paint();
                        }
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
                            let rect_width = if size < 100 {
                                1.0
                            } else if size < 1000 {
                                4.0
                            } else if size < 10000 {
                                8.0
                            } else {
                                16.0
                            };
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
}





fn method_to_text(method: &CodeLocation, records: &Vec<Block>) -> String {
    let mut records = records.clone();
    let mut text = String::new();
    text.push_str(&format!("{}:{}\n", method.clone().file.unwrap_or_else(|| "unknown".to_string()), method.clone().method_name.unwrap_or_else(|| "unknown".to_string())));
    records.sort_by_key(|x| x.block_id.idx);
    for record in records {
        text.push_str(&format!("{:?}\n", record.block_id.idx));
        text.push_str(&record.disasm);
        text.push_str("\n");
    }
   return text;
}


fn calculate_hash<T: Hash>(t: &T) -> String {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish().to_string()
}


struct CodeFile {
    name: String,
    full_path: String,
    methods: Vec<Method>,
}

struct Method {
    name: String,
    blocks: Vec<Block>,
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

    for (name, group) in records.iter().group_by(|x| x.location.file.as_ref()).into_iter() {
        let full_path = name.map(|x| x.clone()).unwrap_or_else(|| "unknown".to_string()).to_string();
        let name = name.map(|x| x.clone()).unwrap_or_else(|| "unknown".to_string()).split("/").last().unwrap().to_string();
        let mut code_file = CodeFile {
            name,
            full_path,
            methods: vec![],
        };
        group.sorted_by_key(|x| x.location.method_name.as_ref()).group_by(|x| x.location.method_name.as_ref()).into_iter().for_each(|(method_name, group)| {
            let method = Method {
                name: (method_name.map(|x| x.clone()).unwrap_or_else(|| "unknown".to_string())).to_string(),
                blocks: group.map(|x| x.clone()).collect(),
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
    };
    window::setup_window(visualizer);

    println!("Hello, world!");
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
