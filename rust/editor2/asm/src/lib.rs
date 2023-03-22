use std::{error::Error, fs, str::from_utf8};

use framework::{App, Canvas, Color, Rect, KeyState, KeyCode};

use roxmltree::Node;
use serde::{Deserialize, Serialize};
mod framework;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AsmData {
    file_info: Vec<FileInfo>,
    offset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileInfo {
    name: String,
    asm: Vec<String>,
    desc: String,
    regdiagram: Vec<String>,
}

impl App for AsmData {
    type State = AsmData;

    fn init() -> Self {
        // local directory from cargo

        // let cargo_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        // let xml_file_path = format!("{}/{}", cargo_dir, "resources/onebigfile.xml");
        Self { file_info: vec![], offset: 0 }
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();

        let foreground = Color::parse_hex("#62b4a6");
        let background = Color::parse_hex("#530922");

        let bounding_rect = Rect::new(0.0, 0.0, 2500.0, 1500.0);

        canvas.save();
        canvas.set_color(&background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);
        canvas.set_color(&foreground);



        canvas.translate(50.0, 50.0);

        if self.file_info.is_empty() {
            canvas.draw_str("Click to load file", 0.0, 0.0);
        }

        for file_info in self.file_info.iter().skip(self.offset).take(5) {
            canvas.draw_str(&file_info.name, 0.0, 0.0);
            canvas.translate(0.0, 40.0);

            canvas.save();
            for regdiagram in file_info.regdiagram.iter() {
                let document = roxmltree::Document::parse(&regdiagram).unwrap();
                let root = document.root_element();
                let attrs = root.attributes().map(|x| format!("{}: {}", x.name(), x.value())).collect::<Vec<String>>().join(", ");

                
                canvas.draw_str(&format!("{} {}", root.tag_name().name(), attrs), 0.0, 0.0);
                canvas.translate(0.0, 40.0);
            }
            canvas.restore();
            canvas.save();
            canvas.translate(1200.0, 0.0);
            for regdiagram in file_info.regdiagram.iter() {
                let document = roxmltree::Document::parse(&regdiagram).unwrap();
                let root = document.root_element();
                let name = root.attribute("name").unwrap_or("");
                let use_name = root.attribute("usename");
                let hibit = root.attribute("hibit").unwrap();
                let width = root.attribute("width").unwrap_or("1").parse::<i32>().unwrap();
                let children = root.descendants().filter(|x| x.has_tag_name("c")).map(|x| {
                    x.text().map(|x| x.to_string())
                }).flatten().collect::<Vec<String>>();
                canvas.draw_str(hibit, 0.0, 0.0);
                if children.is_empty() {
                    canvas.draw_str(name, 0.0, 40.0);
                } else {
                    for (i, child) in children.iter().enumerate() {
                        canvas.draw_str(child, i as f32 * 70.0, 40.0);
                    }
                }
                if !children.is_empty() {
                    canvas.draw_str(name, 0.0, 80.0);
                }

                canvas.translate(40.0 * width as f32, 0.0);
            }
            canvas.restore();

            canvas.translate(0.0, file_info.regdiagram.len() as f32 * 40.0 + 30.0);
        }

        canvas.restore();
    }

    fn on_click(&mut self, _x: f32, _y: f32) {
        // grab the xml file
        // self.xml_file_text = current_dir().unwrap().to_str().unwrap().to_string();
        match self.get_xml_stuff() {
            Ok(_) => (),
            Err(e) => {
                
            }
        }
    }

    fn on_key(&mut self, input: KeyboardInput) {
        match input {
            KeyboardInput { state: KeyState::Pressed, key_code, .. }=> {
                match key_code {
                    KeyCode::R => {
                        self.offset = 0;
                    }
                    KeyCode::DownArrow => {
                        self.offset += 1;
                        if self.offset >= self.file_info.len() {
                            self.offset = 0;
                        }
                    }
                    KeyCode::UpArrow => {
                        if self.offset == 0 {
                            self.offset = self.file_info.len() - 1;
                        } else {
                            self.offset -= 1;
                        }

                    }
                    _ => {}
                }
               
            }
            
            _ => {}
        }
    }

    fn on_scroll(&mut self, _x: f64, _y: f64) {}

    fn get_state(&self) -> Self::State {
        self.clone()
    }

    fn set_state(&mut self, state: Self::State) {
        *self = state;
    }
}

impl AsmData {
    fn get_xml_stuff(&mut self) -> Result<(), Box<dyn Error>> {


        if !self.file_info.is_empty() {
            // let name : String = self.file_info.iter().map(|x| format!("{:#?} \n", x)).collect();
            // self.xml_file_text = name;
            return Ok(())
        }


        let before_read = std::time::Instant::now();
        let xml_file_bytes = fs::read("onebigfile.xml")?;
        let xml_file_text = from_utf8(&xml_file_bytes)?;
        println!("Read file in {}ms", before_read.elapsed().as_millis());
        let before_parse = std::time::Instant::now();
        let xml = roxmltree::Document::parse(xml_file_text.clone())?;
        println!("Parsed file in {}ms", before_parse.elapsed().as_millis());

        let before_find = std::time::Instant::now();
        let file_names = xml
            .descendants()
            .filter(|x| x.has_tag_name("iforms"))
            .find(|x| {
                x.attribute("title")
                    .unwrap_or("")
                    .contains("Base Instructions")
            })
            .unwrap()
            .descendants()
            .filter(|x| x.has_tag_name("iform"))
            .filter_map(|x| x.attribute("iformfile"));

        let mut found_file_nodes = vec![];
        for file_name in file_names {
            let file_ndoe = xml
                .descendants()
                .find(|x| x.attribute("file") == Some(file_name));
            if let Some(file_node) = file_ndoe {
                found_file_nodes.push(file_node);
            }
        }

        //     basic_insn_files.each do |filename|
        //     file_node = files_by_name[filename].first
        //     file_node.css("instructionsection").each do |section|
        //       asm = section.css("asmtemplate").map(&:text)
        //       desc = section.at_css("desc > brief").text.strip
        //       fname = section["id"].downcase
        //       files_and_classes << [fname, section["id"]]
        //       unless File.exist?  "lib/aarch64/instructions/#{fname}.rb"
        //         File.binwrite "lib/aarch64/instructions/#{fname}.rb", make_encode(section["id"], section["title"], desc, asm, section.at_css("regdiagram"))
        //       end
        //     end
        //   end

        let file_info : Vec<FileInfo> = found_file_nodes
            .iter()
            .flat_map(|x| {
                x.descendants()
                    .filter(|x| x.has_tag_name("instructionsection"))
            })
            .map(|x| {
                let name = x
                    .attribute("id")
                    .unwrap_or("No file found")
                    .to_ascii_lowercase();
                let asm = x
                    .descendants()
                    .filter(|x| x.has_tag_name("asmtemplate"))
                    .map(|x| xml_file_text[x.range()].to_string())
                    .collect();
                let desc = x
                    .descendants()
                    .find(|x| x.has_tag_name("desc"))
                    .and_then(|x| x.descendants().find(|x| x.has_tag_name("brief")))
                    .and_then(|x| x.descendants().find(|x| x.has_tag_name("para")))
                    .map(|x| x.text().unwrap_or(""))
                    .unwrap_or("")
                    .to_string();
                let regdiagram = x
                    .descendants()
                    .find(|x| x.has_tag_name("regdiagram"))
                    .map(|x| {
                        let boxes: Vec<Node> =
                            x.descendants().filter(|x| x.has_tag_name("box")).collect();
                        boxes
                    })
                    .unwrap_or_default()
                    .iter()
                    .map(|x| xml_file_text[x.range()].to_string())
                    .collect();
                FileInfo {
                    name,
                    asm,
                    desc,
                    regdiagram,
                }
            })
            .collect();
        self.file_info = file_info.clone();

        let name : String = file_info.iter().map(|x| format!("{:#?} \n", x)).collect();

        println!("Found file in {}ms", before_find.elapsed().as_millis());

        // self.xml_file_text = format!("Files: {}", name);
        Ok(())
    }
}

app!(AsmData);
