use std::{str::from_utf8, fs, env::current_dir, error::Error, io};

use framework::{App, Canvas, Color, Rect, KeyState, KeyCode};
use minidom::Element;
use roxmltree::{Node, Document};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
mod framework;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AsmData {
    xml_file_text: String,
}


impl App for AsmData {
    type State = String;

    fn init() -> Self {


        // local directory from cargo

        // let cargo_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        // let xml_file_path = format!("{}/{}", cargo_dir, "resources/onebigfile.xml");
        let xml_file_text = "".to_string();
        Self {
            xml_file_text
        }
    }

    fn draw(&mut self) {

        let canvas = Canvas::new();

        let foreground = Color::parse_hex("#62b4a6");
        let background = Color::parse_hex("#530922");

        let bounding_rect = Rect::new(0.0, 0.0, 2000.0, 800.0);


        canvas.save();
        canvas.set_color(&background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);
        canvas.set_color(&foreground);
        if self.xml_file_text.is_empty() {
            canvas.draw_str("No XML file loaded", 40.0, 40.0);
        } else {
            for line in self.xml_file_text.lines() {
                canvas.draw_str(line, 40.0, 40.0);
                canvas.translate(0.0, 40.0);
            }
        }

        canvas.restore();
    }

    fn on_click(&mut self, x: f32, y: f32) {

        self.xml_file_text = "clicked".to_string();
        // grab the xml file
        // self.xml_file_text = current_dir().unwrap().to_str().unwrap().to_string();
        match self.get_xml_stuff() {
            Ok(_) => (),
            Err(e) => {
                self.xml_file_text = format!("Failed {}", e.to_string());
            }
        }

    }

    fn on_key(&mut self, input: KeyboardInput) {

    }

    fn on_scroll(&mut self, x: f64, y: f64) {

    }

    fn get_state(&self) -> Self::State {
        self.xml_file_text.clone()
    }

    fn set_state(&mut self, state: Self::State) {
        self.xml_file_text = state;
    }

}


impl AsmData {
    fn get_xml_stuff(&mut self) -> Result<(), Box<dyn Error>>  {
        let before_read = std::time::Instant::now();
        let xml_file_bytes = fs::read("onebigfile.xml")?;
        let xml_file_text = from_utf8(&xml_file_bytes)?;
        println!("Read file in {}ms", before_read.elapsed().as_millis());
        let before_parse = std::time::Instant::now();
        let xml = roxmltree::Document::parse(xml_file_text.clone())?;
        println!("Parsed file in {}ms", before_parse.elapsed().as_millis());

        let file_names = xml.descendants()
            .filter(|x| x.has_tag_name("iforms"))
            .find(|x| x.attribute("title").unwrap_or("").contains("Base Instructions"))
            .unwrap()
            .descendants()
            .filter(|x| x.has_tag_name("iform"))
            .map(|x| x.attribute("iformfile"))
            .flatten();

        let mut found_file_nodes = vec![];
        for file_name in file_names {
            let file_ndoe = xml.descendants().find(|x| x.attribute("file") == Some(file_name));
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

        let name : String = found_file_nodes.iter()
            .flat_map(|x| x.descendants().filter(|x| x.has_tag_name("instructionsection")))
            .map(|x| x.attribute("id").unwrap_or("No file found"))
            .map(|x| format!("{} \n", x))
            .collect();


        self.xml_file_text = format!("Files: {}", name);
        Ok(())
    }

}


app!(AsmData);

