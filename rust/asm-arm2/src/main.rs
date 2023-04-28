use mmap_rs::MmapOptions;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::{self, File},
    io::Write,
    mem,
    process::Command,
    str::from_utf8,
};

use roxmltree::{Document, Node};
use serde::{Deserialize, Serialize};

use arm::{Asm, Register};

use crate::arm::{Size, X0, X1};

mod arm;

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

fn generate_template() -> Result<(), Box<dyn Error>> {
    let xml_file_path = "resources/onebigfile.xml";

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

                let mut constraints = vec![];
                let mut new_bits = vec![];

                for bit in bits {
                    let all_numbers = bit.chars().all(|x| x.is_numeric());
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

                // != 1111

                fields.push(Field {
                    name: name.to_string(),
                    shift,
                    bits: if new_bits.is_empty() {
                        "0".repeat(width as usize).to_string()
                    } else {
                        new_bits.join("")
                    },
                    width,
                    required: new_bits.is_empty(),
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
    generate_template()?;
    use_the_assembler()?;
    Ok(())
}

fn print_u32_hex_le(value: u32) {
    let bytes = value.to_le_bytes();
    for byte in &bytes {
        print!("{:02x}", byte);
    }
    println!();
}

fn mov(destination: Register, input: u16) -> Asm {
    Asm::Movz {
        sf: destination.sf(),
        hw: 0,
        // TODO: Shouldn't this be a u16??
        imm16: input as u32,
        rd: destination,
    }
}

fn add(destination: Register, a: Register, b: Register) -> Asm {
    Asm::AddAddsubShift {
        sf: destination.sf(),
        shift: 0,
        imm6: 0,
        rn: a,
        rm: b,
        rd: destination,
    }
}

fn ret() -> Asm {
    Asm::Ret {
        rn: Register {
            size: Size::S64,
            index: 30,
        },
    }
}

fn compare(a: Register, b: Register) -> Asm {
    Asm::CmpSubsAddsubShift {
        sf: a.sf(),
        shift: 0,
        rm: a,
        imm6: 0,
        rn: b,
    }
}

fn jump_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 0,
    }
}
fn jump_not_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 1,
    }
}
fn jump_greater_or_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 10,
    }
}
fn jump_less_than(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 11,
    }
}
fn jump_greater(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 12,
    }
}
fn jump_less_or_equal(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 13,
    }
}
fn jump(destination: u32) -> Asm {
    Asm::BCond {
        imm19: destination,
        cond: 14,
    }
}
fn breakpoint() -> Asm {
    Asm::Brk { imm16: 30 }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Label {
    index: usize,
}

struct Lang {
    instructions: Vec<Asm>,
    label_index: usize,
    label_locations: HashMap<usize, usize>,
    labels: Vec<String>,
}

impl Lang {
    fn new() -> Self {
        Lang {
            instructions: vec![],
            label_locations: HashMap::new(),
            label_index: 0,
            labels: vec![],
        }
    }

    fn get_label_index(&mut self) -> usize {
        let current_label_index = self.label_index;
        self.label_index += 1;
        current_label_index
    }

    fn breakpoint(&mut self) {
        self.instructions.push(breakpoint())
    }

    fn mov(&mut self, destination: Register, input: u16) {
        self.instructions.push(mov(destination, input));
    }
    fn add(&mut self, destination: Register, a: Register, b: Register) {
        self.instructions.push(add(destination, a, b));
    }
    fn ret(&mut self) {
        self.instructions.push(ret());
    }
    fn compare(&mut self, a: Register, b: Register) {
        self.instructions.push(compare(a, b));
    }
    fn jump_equal(&mut self, destination: Label) {
        self.instructions.push(jump_equal(destination.index as u32));
    }
    fn jump_not_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_not_equal(destination.index as u32));
    }
    fn jump_greater_or_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_greater_or_equal(destination.index as u32));
    }
    fn jump_less_than(&mut self, destination: Label) {
        self.instructions
            .push(jump_less_than(destination.index as u32));
    }
    fn jump_greater(&mut self, destination: Label) {
        self.instructions
            .push(jump_greater(destination.index as u32));
    }
    fn jump_less_or_equal(&mut self, destination: Label) {
        self.instructions
            .push(jump_less_or_equal(destination.index as u32));
    }
    fn jump(&mut self, destination: Label) {
        self.instructions.push(jump(destination.index as u32));
    }
    fn new_label(&mut self, name: &str) -> Label {
        self.labels.push(name.to_string());
        Label {
            index: self.get_label_index(),
        }
    }
    fn write_label(&mut self, label: Label) {
        self.label_locations
            .insert(label.index, self.instructions.len());
    }

    fn compile(&mut self) -> &Vec<Asm> {
        self.patch_labels();
        &self.instructions
    }

    fn patch_labels(&mut self) {
        for (instruction_index, instruction) in self.instructions.iter_mut().enumerate() {
            match instruction {
                Asm::BCond { imm19, cond: _ } => {
                    let label_index = *imm19 as usize;
                    let label_location = self.label_locations.get(&label_index).unwrap();
                    let relative_position = label_location - instruction_index;
                    *imm19 = relative_position as u32;
                }
                _ => {}
            }
        }
    }
}


fn use_the_assembler() -> Result<(), Box<dyn Error>> {
    let mut lang = Lang::new();

    let label = lang.new_label("label");

    lang.breakpoint();
    lang.mov(X0, 100);
    lang.mov(X1, 100);
    lang.compare(X0, X1);
    lang.jump_equal(label);
    lang.add(X0, X1, X0);
    lang.write_label(label);
    lang.ret();

    let instructions = lang.compile();


    for instruction in instructions.iter() {
        print_u32_hex_le(instruction.encode());
    }

    let mut buffer = MmapOptions::new(MmapOptions::page_size())?.map_mut()?;
    let memory = &mut buffer[..];
    let mut bytes = vec![];
    for instruction in instructions.iter() {
        for byte in instruction.encode().to_le_bytes() {
            bytes.push(byte);
        }
    }
    for (i, byte) in bytes.iter().enumerate() {
        memory[i] = *byte;
    }
    // println!("{:?}", buffer.as_slice());
    let size = buffer.size();
    buffer.flush(0..size)?;

    let exec = buffer.make_exec().unwrap_or_else(|(_map, e)| {
        panic!("Failed to make mmap executable: {}", e);
    });

    let f: fn() -> u64 = unsafe { mem::transmute(exec.as_ref().as_ptr()) };

    println!("{}", f());

    Ok(())
}


// TODO:
// ORGANIZE!!!!


// We have a nice enum format
// but we need a none enum format

// We need better documentation on which
// instructions we actually like and care about.
// We need to abstract over all the different "add" and "mov"
// functions so I don't look dumb on stream

// We need to build a real build system

// Start writing some code for the real language
