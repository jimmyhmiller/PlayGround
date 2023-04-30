use std::{error::Error, fs, str::from_utf8};

use framework::{App, Canvas, Color, Rect, KeyState, KeyCode};

use roxmltree::{Node, Document};
use serde::{Deserialize, Serialize};
mod framework;
use indoc::indoc;

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

use std::fmt::Write;


fn find_indent(s: &str) -> usize {
    let lines = s.lines();

    // Find the indent of the first non-empty line
    lines
        .clone()
        .filter(|line| !line.trim().is_empty())
        .next()
        .map(|line| line.chars().take_while(|c| c.is_whitespace()).count())
        .unwrap_or(0)
}

fn remove_common_indent(s: &str) -> String {

    // Find the indent of the first non-empty line
    let first_non_empty_indent = find_indent(s);
    
    let lines = s.lines();
    // Remove that indent from all lines
    lines
        .map(|line| {
            if line.len() >= first_non_empty_indent {
                &line[first_non_empty_indent..]
            } else {
                line
            }
        })
        .collect::<Vec<&str>>()
        .join("\n")
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
            let name = file_info.name.to_ascii_uppercase();
            canvas.draw_str(&format!("# {}", &name), 0.0, 0.0);
            canvas.translate(0.0, 40.0);

            canvas.draw_str(&format!( "# {}", &file_info.desc), 0.0, 0.0);
            canvas.translate(0.0, 40.0);
            

            for asm in file_info.asm.iter() {
                let asm = Document::parse(asm).unwrap();
                let texts = asm
                    .descendants()
                    .filter(|x| x.has_tag_name("a"))
                    .map(|x| x.text().unwrap_or(""));
                canvas.save();
                canvas.draw_str("#", 0.0, 0.0);
                canvas.translate(30.0, 0.0);
                for text in texts {
                    canvas.draw_str(text, 0.0, 0.0);
                    canvas.translate(80.0, 0.0);
                }
                canvas.restore();
                canvas.translate(0.0, 40.0);
            }

            canvas.translate(0.0, 40.0);


            struct Field {
                name: String,
                shift: u32,
                bits: String,
                width: u32,
                required: bool,
            }

            let mut fields = vec![];


            canvas.save();
            for regdiagram in file_info.regdiagram.iter() {
                let document = roxmltree::Document::parse(&regdiagram).unwrap();
                let root = document.root_element();
                let name = root.attribute("name").unwrap_or("");
                let use_name = root.attribute("usename");

                let bits : Vec<String> = root.descendants()
                    .filter(|child| !child.is_text())
                    .filter_map(|child| {
                        let text = child.text().unwrap_or_default().trim();
                        if !text.is_empty() {
                            Some(text.replace("(", "").replace(")", ""))
                        } else {
                            None
                        }
                    })
                    .collect();
        
                let hibit = root.attribute("hibit").unwrap().parse::<u32>().unwrap() + 1;
                let width = root.attribute("width").unwrap_or("1").parse::<u32>().unwrap();
                let shift = hibit - width;


                fields.push(Field {
                    name: name.to_string(),
                    shift,
                    bits: if bits.is_empty() {
                        "0".repeat(width as usize).to_string()
                    } else {
                        bits.join("")
                    },
                    width,
                    required: bits.is_empty(),
                })
            }

            let params = fields.iter().filter(|x| x.required).map(|x| format!("{}: u32", x.name)).collect::<Vec<String>>().join(", ");

            let indent = "                    ";
            let mut define_params = String::new();
            for field in fields.iter().filter(|x| x.required) {
                let name = field.name.to_ascii_lowercase();
                define_params.write_str(&format!("{}{}: u32,\n", indent, name)).unwrap();
            }
            if !define_params.is_empty() {
                define_params.pop();
                define_params = define_params[indent.len()..].to_string();
            }
                
            let indent = "                            ";
            // rd: check_mask(rd, 0x1f),
            let mut init_params = String::new();
            for field in fields.iter().filter(|x| x.required) {
                let mask = (1 << field.width) - 1;
                let name = field.name.to_ascii_lowercase();
                init_params.write_str(&format!("{}{}: check_mask({}, {:#02x}),\n", indent, name, name, mask)).unwrap();
            }
            if !init_params.is_empty() {
                init_params.pop();
                init_params = init_params[indent.len()..].to_string();
            }


            let mut encode = String::new();
            let all_bits = fields.iter().map(|x| x.bits.clone()).collect::<Vec<String>>().join("_");


            encode.write_str(&format!("0b{}\n", all_bits)).unwrap();
            // for lack of a better thing to do I copied from template
            let indent = "                        ";
            for field in fields.iter().filter(|x| x.required) {
                let name = field.name.to_ascii_lowercase();
                encode.write_str(&format!("{}| (self.{} << {})\n", indent, name, field.shift)).unwrap();
            }
            encode.pop();

            let template = remove_common_indent(&format!("
                pub struct {name} {{
                    {define_params}
                }}
                
                impl {name} {{
                    pub fn new({params}) -> Self {{
                        {name} {{
                            {init_params}
                        }}
                    }}
                
                    pub fn encode(&self) -> u32 {{
                        {encode}
                    }}
                }}
            "));

            for line in template.lines() {
                canvas.draw_str(line, 0.0, 0.0);
                canvas.translate(0.0, 40.0);
            }



            canvas.translate(0.0, file_info.regdiagram.len() as f32 * 5.0 + 30.0);
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

    fn create_template() -> String {
        indoc!{"
            pub struct {name} {
                {define_params}
            }
    
            impl {name} {
                pub fn new({params}) -> Self {
                    {name} {
                        {init_params}
                    }
                }
    
                pub fn encode(&self) -> u32 {
                   {encode}
                }
            }
        "}.to_string()
    }





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
