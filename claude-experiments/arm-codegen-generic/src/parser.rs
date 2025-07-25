use roxmltree::{Document, Node};
use std::{collections::HashSet, error::Error, fs, str::from_utf8};

use crate::{ArmBox, Bits, Field, FieldKind, Instruction, RegisterDiagram};

fn is_power_of_two(x: u8) -> bool {
    (x & (x - 1)) == 0
}

fn to_camel_case(name: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for c in name.chars() {
        if c == '_' || c == '-' || c == ' ' || c == ',' {
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

fn get_files<'a>(xml: &'a Document<'a>) -> Vec<Node<'a, 'a>> {
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
        .filter(|x| file_names_hash.contains(x.attribute("file").unwrap_or("Not Found")))
        .collect();
    found_file_nodes
}

fn get_instruction_name_and_title(instruction_section: Node) -> (String, String) {
    let title = instruction_section
        .attribute("title")
        .unwrap_or("No file found")
        .to_string();
    let name = to_camel_case(
        instruction_section
            .attribute("id")
            .unwrap_or("No file found"),
    );
    (name, title)
}

fn get_instruction_comments(instruction_section: Node) -> Vec<String> {
    let mut argument_comments = vec![];

    let asm = instruction_section
        .descendants()
        .filter(|x: &Node| x.has_tag_name("asmtemplate"));

    for asm in asm {
        let texts = asm
            .children()
            .map(|x| x.text().unwrap_or("").to_string())
            .collect::<Vec<String>>();

        argument_comments.push(texts.join(""));
    }

    argument_comments
}

fn get_instruction_description(instruction_section: Node) -> String {
    let description = instruction_section
        .descendants()
        .find(|x| x.has_tag_name("desc"))
        .and_then(|x| x.descendants().find(|x| x.has_tag_name("brief")))
        .and_then(|x| x.descendants().find(|x| x.has_tag_name("para")))
        .map(|x| x.text().unwrap_or(""))
        .unwrap_or("")
        .to_string();
    description
}

fn get_register_diagrams(instruction_section: Node) -> Vec<RegisterDiagram> {
    let classes = instruction_section
        .descendants()
        .filter(|x| x.has_tag_name("iclass"));

    let mut result: Vec<RegisterDiagram> = vec![];
    for class in classes {
        let name = class.attribute("name").unwrap().to_string();
        let diagram = class
            .descendants()
            .find(|x| x.has_tag_name("regdiagram"))
            .unwrap();
        let boxes = diagram
            .descendants()
            .filter(|x| x.has_tag_name("box"))
            .map(parse_armbox)
            .collect::<Vec<ArmBox>>();

        result.push(RegisterDiagram { name, boxes })
    }

    result
}

fn parse_armbox(node: Node) -> ArmBox {
    let hibit = node.attribute("hibit").unwrap().parse::<u8>().unwrap() + 1;
    let width = node
        .attribute("width")
        .unwrap_or("1")
        .parse::<u8>()
        .unwrap();

    let bits: Vec<String> = node
        .descendants()
        .filter(|child| !child.is_text())
        .filter_map(|child| {
            let text = child.text().unwrap_or_default().trim();
            if !text.is_empty() {
                Some(text.to_string().replace("(", "").replace(")", ""))
            } else {
                None
            }
        })
        .collect();

    let bits = if bits.is_empty() {
        Bits::Empty(width)
    } else if bits.iter().any(|x| x.contains("!=")) {
        let bits = node.attribute("constraint").unwrap().to_string();
        let bits = bits.trim();
        let bits = bits.replace("!= ", "");
        let bits = bits.chars().map(|x| x.to_string()).collect::<Vec<_>>();
        Bits::Constraint(bits)
    } else if bits.iter().any(|x| x.chars().any(|y| y.is_alphabetic())) {
        Bits::HasVariable(bits)
    } else if bits.iter().all(|x| x.chars().all(|y| y.is_numeric())) {
        Bits::AllNums(bits.join(""))
    } else {
        panic!("Unknown bits: {}", bits.join("sep"))
    };

    let name = node.attribute("name").unwrap_or("").to_string();

    ArmBox {
        hibit,
        width,
        name,
        bits,
    }
}

impl Bits {
    pub fn render(&self) -> Option<String> {
        match self {
            Bits::Empty(width) => Some("0".repeat(*width as usize)),
            Bits::AllNums(nums) => Some(nums.to_string()),
            Bits::HasVariable(bits) => Some("0".repeat(bits.len())),
            Bits::Constraint(bits) => Some("0".repeat(bits.len())),
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Bits::Empty(_))
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, Bits::HasVariable(_) | Bits::Constraint(_))
    }
}

impl ArmBox {
    pub fn is_arg(&self) -> bool {
        self.bits.is_empty() || self.bits.is_variable()
    }

    pub fn kind(&self) -> FieldKind {
        if self.name.starts_with('R') {
            FieldKind::Register
        } else if self.name.starts_with("imm") {
            let num = &self.name[3..];
            let num = num.parse::<u8>();
            match num {
                Ok(num) => {
                    if is_power_of_two(num) {
                        FieldKind::Immediate
                    } else {
                        FieldKind::NonPowerOfTwoImm(num)
                    }
                }
                Err(_) => FieldKind::Immediate,
            }
        } else {
            FieldKind::Immediate
        }
    }
}

fn instruction_fields(instruction: &Instruction) -> Vec<Field> {
    let mut fields = vec![];
    for diagram in instruction.diagrams.iter() {
        for arm_box in diagram.boxes.iter() {
            let field = Field {
                name: arm_box.name.to_ascii_lowercase().clone(),
                shift: arm_box.shift(),
                bits: arm_box.bits.render(),
                width: arm_box.width,
                is_arg: arm_box.is_arg(),
                kind: arm_box.kind(),
            };
            if !fields.contains(&field) {
                fields.push(field)
            }
        }
    }
    if instruction.diagrams.len() > 1 {
        fields.push(Field {
            name: "class_selector".to_string(),
            shift: 0,
            bits: None,
            width: 0,
            is_arg: true,
            kind: FieldKind::ClassSelector(format!("{}Selector", instruction.name)),
        })
    }
    fields
}

impl PartialEq for Field {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// Load ARM instructions from the embedded XML data
pub fn load_arm_instructions() -> Result<Vec<Instruction>, Box<dyn Error>> {
    // For now, use the XML file from the original location
    let xml_file_path = "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/asm-arm2/asm/resources/onebigfile.xml";
    parse_xml(xml_file_path)
}

pub fn parse_xml(xml_file_path: &str) -> Result<Vec<Instruction>, Box<dyn Error>> {
    let xml_file_bytes = fs::read(xml_file_path)?;
    let xml_file_text = from_utf8(&xml_file_bytes)?;
    let xml = roxmltree::Document::parse(xml_file_text)?;

    let found_file_nodes = get_files(&xml);

    let instruction_sections = found_file_nodes.iter().flat_map(|x| {
        x.descendants()
            .filter(|x| x.has_tag_name("instructionsection"))
    });

    let mut instructions = vec![];

    for instruction_section in instruction_sections {
        let (name, title) = get_instruction_name_and_title(instruction_section);
        let comments = get_instruction_comments(instruction_section);
        let description = get_instruction_description(instruction_section);
        let diagrams = get_register_diagrams(instruction_section);
        
        let mut instruction = Instruction {
            name,
            title,
            comments,
            description,
            diagrams,
            fields: vec![],
        };
        
        instruction.fields = instruction_fields(&instruction);
        instructions.push(instruction);
    }

    Ok(instructions)
}