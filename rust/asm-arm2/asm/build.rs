use std::{
    collections::HashSet,
    error::Error,
    fs::{self, File},
    io::Write,
    process::Command,
    str::from_utf8,
};

use roxmltree::{Document, Node};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Instruction {
    title: String,
    name: String,
    asm: Vec<String>,
    description: String,
    regdiagram: Vec<String>,
    fields: Vec<Field>,
    argument_comments: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Kind {
    Register,
    Immediate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Field {
    name: String,
    shift: u32,
    bits: String,
    width: u32,
    required: bool,
    kind: Kind,
}

fn to_camel_case(name: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for c in name.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c.to_ascii_lowercase());
        }
    }

    result
}

// We learned from stp_gen that some instructions have multiple classes
// These need to be accounted for and are not.
// Problem: Our code is a mess and it is hard to refactor to deal with that


pub fn generate_template() -> Result<(), Box<dyn Error>> {
    let xml_file_path =
        "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/asm-arm2/asm/resources/onebigfile.xml";

    let xml_file_bytes = fs::read(xml_file_path)?;
    let xml_file_text = from_utf8(&xml_file_bytes)?;
    let xml = roxmltree::Document::parse(xml_file_text.clone())?;

    let file_names = xml
        .descendants()
        .filter(|x| x.has_tag_name("iforms"))
        .filter(|x| {
            let title = x.attribute("title").unwrap_or("");
            title.contains("Base Instructions")
                || title.contains("SIMD and Floating-point Instructions")
        })
        .flat_map(|x| x.descendants())
        .filter(|x| x.has_tag_name("iform"))
        .filter_map(|x| x.attribute("iformfile"));

    let mut file_names_hash = HashSet::new();
    file_names_hash.extend(file_names);

    let found_file_nodes = xml
        .descendants()
        .filter(|x| file_names_hash.contains(x.attribute("file").unwrap_or("Not Found")));



    let instructions: Vec<Instruction> = found_file_nodes
        // .take(5)
        .flat_map(|x| {
            x.descendants()
                .filter(|x| x.has_tag_name("instructionsection"))
        })
        .map(|x| {
            let title = x.attribute("title").unwrap_or("No file found").to_string();
            let name = to_camel_case(&x.attribute("id").unwrap_or("No file found").to_string());

            if name == "StpGen" {
                println!("HERE");
            }

            let asm = x
                .descendants()
                .filter(|x| x.has_tag_name("asmtemplate"))
                .map(|x| xml_file_text[x.range()].to_string())
                .collect::<Vec<String>>();

            let description = x
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
                .collect::<Vec<String>>();

            let mut fields = vec![];
            for regdiagram in regdiagram.iter() {
                let document = roxmltree::Document::parse(&regdiagram).unwrap();
                let root = document.root_element();
                let name = root.attribute("name").unwrap_or("").to_ascii_lowercase();
                // let use_name = root.attribute("usename");

                let bits: Vec<String> = root
                    .descendants()
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
                let width = root
                    .attribute("width")
                    .unwrap_or("1")
                    .parse::<u32>()
                    .unwrap();
                let shift = hibit - width;

                let mut constraints: Vec<String> = vec![];
                let mut new_bits: Vec<String> = vec![];

                for bit in bits {
                    let all_numbers = bit.chars().all(|x| x.is_numeric() || x == 'x');
                    if all_numbers {
                        new_bits.push(bit);
                    } else {
                        let width = bit
                            .chars()
                            .filter(|c| c.is_numeric())
                            .collect::<String>()
                            .len();
                        constraints.push(bit);
                        new_bits.push("0".repeat(width as usize).to_string())
                    }
                }

                let joined_bits = new_bits.join("");
                let field_template_bits = if new_bits.is_empty() {
                    "0".repeat(width as usize).to_string()
                } else {
                    joined_bits.replace("x", "0")
                };

                fields.push(Field {
                    name: name.to_string(),
                    shift,
                    bits: field_template_bits,
                    width,
                    required: joined_bits.is_empty() || joined_bits.contains("x"),
                    kind: if name.starts_with("r") {
                        Kind::Register
                    } else {
                        Kind::Immediate
                    },
                })
            }

            let mut argument_comments = vec![];

            for asm in asm.iter() {
                let asm = Document::parse(asm).unwrap();

                let texts = asm
                    .root_element()
                    .children()
                    .map(|x| x.text().unwrap_or("").to_string())
                    .collect::<Vec<String>>();

                argument_comments.push(texts.join(""));
            }

            Instruction {
                title,
                name,
                asm,
                description,
                regdiagram,
                fields,
                argument_comments,
            }
        })
        .collect();

    let template = liquid::ParserBuilder::with_stdlib()
        .build()
        .unwrap()
        .parse_file("./resources/asm.tpl")
        .unwrap();

    let globals = liquid::object!({
        "instructions": instructions,
    });

    let output = template.render(&globals).unwrap();
    let mut file = File::create("src/arm.rs")?;
    file.write_all(output.as_bytes())?;

    let result = Command::new("cargo")
        .arg("fmt")
        .arg("--")
        .arg("src/arm.rs")
        .spawn()?;

    println!("{:?}", result);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // generate_template()?;
    Ok(())
}
