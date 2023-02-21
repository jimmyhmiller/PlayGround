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

        let bounding_rect = Rect::new(0.0, 0.0, 2000.0, 300.0);


        canvas.save();
        canvas.set_color(&background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);
        canvas.set_color(&foreground);
        if self.xml_file_text.is_empty() {
            canvas.draw_str("No XML file loaded", 40.0, 40.0);
        } else {
            canvas.draw_str(&format!("{}", &self.xml_file_text), 40.0, 40.0);
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
        println!("get_xml_stuff");
        let xml_file_bytes = fs::read("onebigfile.xml")?;
        let xml_file_text = from_utf8(&xml_file_bytes)?;
        let xml = roxmltree::Document::parse(xml_file_text.clone())?;

        let result = xml;
        let result = result.root().tag_name();
        self.xml_file_text = format!("Name: {}", result.name().to_string());
        Ok(())
    }

}


app!(AsmData);

