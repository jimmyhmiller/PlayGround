use std::{
    collections::HashSet,
    error::Error,
    fs::{self, File},
    io::Write,
    process::Command,
    str::from_utf8,
};

use std::time::SystemTime;

use codegen::{Scope, Type};
use roxmltree::{Document, Node};
use serde::{Deserialize, Serialize};

fn is_power_of_two(x: u8) -> bool {
    (x & (x - 1)) == 0
}

const TEMPLATE_PREFIX: &str = "

#![allow(clippy::identity_op)]
#![allow(clippy::unusual_byte_groupings)]

use std::ops::Shl;


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Size {
    S32,
    S64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Register {
    pub size: Size,
    pub index: u8,
}

impl Register {
    pub fn sf(&self) -> i32 {
        match self.size {
            Size::S32 => 0,
            Size::S64 => 1,
        }
    }
}

impl Register {
    pub fn encode(&self) -> u8 {
        self.index
    }
}

impl Shl<u32> for &Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

pub const SP: Register = Register {
    index: 31,
    size: Size::S64,
};

pub fn truncate_imm<T: Into<i32>, const WIDTH: usize>(imm: T) -> u32 {
    let value: i32 = imm.into();
    let masked = (value as u32) & ((1 << WIDTH) - 1);

    // Assert that we didn't drop any bits by truncating.
    if value >= 0 {
        assert_eq!(value as u32, masked);
    } else {
        assert_eq!(value as u32, masked | (u32::MAX << WIDTH));
    }

    masked
}
";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Kind {
    Register,
    Immediate,
    ClassSelector(String),
    NonPowerOfTwoImm(u8),
}

impl From<Kind> for String {
    fn from(val: Kind) -> Self {
        match val {
            Kind::Register => "Register".to_string(),
            Kind::Immediate => "i32".to_string(),
            Kind::ClassSelector(name) => name,
            Kind::NonPowerOfTwoImm(_) => "i32".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Field {
    name: String,
    shift: u8,
    bits: Option<String>,
    width: u8,
    is_arg: bool,
    kind: Kind,
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

// We learned from stp_gen that some instructions have multiple classes
// These need to be accounted for and are not.
// Problem: Our code is a mess and it is hard to refactor to deal with that

pub fn get_files<'a>(xml: &'a Document<'a>) -> Vec<Node<'a, 'a>> {
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
            .map(ArmBox::parse)
            .collect::<Vec<ArmBox>>();

        result.push(RegisterDiagram { name, boxes })
    }

    result
}

#[derive(Debug, PartialEq, Eq)]
enum Bits {
    Empty(u8),
    AllNums(String),
    HasVariable(Vec<String>),
    Constraint(String),
    Unknown(String),
}

impl Bits {
    fn render(&self) -> Option<String> {
        match self {
            Bits::Empty(width) => Some("0".repeat(*width as usize)),
            Bits::AllNums(nums) => Some(nums.to_string()),
            Bits::HasVariable(bits) => Some("0".repeat(bits.len())),
            Bits::Constraint(_) => None,
            Bits::Unknown(_) => None,
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Bits::Empty(_))
    }

    fn is_variable(&self) -> bool {
        matches!(self, Bits::HasVariable(_))
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ArmBox {
    hibit: u8,
    width: u8,
    name: String,
    bits: Bits,
}

#[derive(Debug, PartialEq, Eq)]
struct RegisterDiagram {
    name: String,
    boxes: Vec<ArmBox>,
}

#[derive(Debug, PartialEq, Eq)]
struct Instruction {
    name: String,
    title: String,
    comments: Vec<String>,
    description: String,
    diagrams: Vec<RegisterDiagram>,
}

impl ArmBox {
    fn shift(&self) -> u8 {
        self.hibit - self.width
    }

    fn parse(node: Node) -> ArmBox {
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
                    Some(text.to_string())
                } else {
                    None
                }
            })
            .collect();
        let bits = if bits.is_empty() {
            Bits::Empty(width)
        } else if bits.iter().any(|x| x.contains("!=")) {
            Bits::Constraint(bits.join(""))
        } else if bits.iter().any(|x| x.chars().any(|y| y.is_alphabetic())) {
            Bits::HasVariable(bits)
        } else if bits.iter().all(|x| x.chars().all(|y| y.is_numeric())) {
            Bits::AllNums(bits.join(""))
        } else {
            Bits::Unknown(bits.join("sep"))
        };

        let name = node.attribute("name").unwrap_or("").to_ascii_lowercase();

        ArmBox {
            hibit,
            width,
            name,
            bits,
        }
    }

    fn is_arg(&self) -> bool {
        self.bits.is_empty() || self.bits.is_variable()
    }

    fn kind(&self) -> Kind {
        if self.name.starts_with('r') {
            Kind::Register
        } else if self.name.starts_with("imm") {
            let num = &self.name[3..];
            let num = num.parse::<u8>();
            match num {
                Ok(num) => {
                    if is_power_of_two(num) {
                        Kind::Immediate
                    } else {
                        Kind::NonPowerOfTwoImm(num)
                    }
                }
                Err(_) => Kind::Immediate,
            }
        } else {
            Kind::Immediate
        }
    }
}

impl Instruction {
    fn fields(&self) -> Vec<Field> {
        let mut fields = vec![];
        for diagram in self.diagrams.iter() {
            for arm_box in diagram.boxes.iter() {
                let field = Field {
                    name: arm_box.name.clone(),
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
        if self.diagrams.len() > 1 {
            fields.push(Field {
                name: "class_selector".to_string(),
                shift: 0,
                bits: None,
                width: 0,
                is_arg: true,
                kind: Kind::ClassSelector(format!("{}Selector", self.name)),
            })
        }
        fields
    }

    fn args(&self) -> Vec<Field> {
        self.fields().into_iter().filter(|x| x.is_arg).collect()
    }

    fn get_comments(&self) -> String {
        let mut result = String::new();
        result.push_str(&format!("/// {}", self.title));
        result.push('\n');
        result.push_str(
            &self
                .description
                .lines()
                .map(|x| format!("/// {}", x))
                .collect::<Vec<String>>()
                .join("\n"),
        );
        result.push('\n');
        result.push_str(
            &self
                .comments
                .iter()
                .map(|s| format!("/// {}", s))
                .collect::<Vec<String>>()
                .join("\n"),
        );

        result
    }
}

fn generate_instruction_enum(instructions: &[Instruction]) -> String {
    let mut scope = Scope::new();

    let enum_gen = scope.new_enum("Asm");
    enum_gen.vis("pub");
    enum_gen.derive("Debug");

    for instruction in instructions.iter() {
        let variant = enum_gen.new_variant(instruction.name.clone());
        variant.annotation(instruction.get_comments());
        for field in instruction.fields().iter().filter(|x| x.is_arg) {
            variant.named(&field.name, Type::new(field.kind.clone()));
        }
    }
    scope.to_string()
}

fn generate_encoding_instructions(instructions: &[Instruction]) -> String {
    // let instructions = instructions.iter().take(5).collect::<Vec<_>>();
    let mut scope = Scope::new();
    let asm_impl = scope.new_impl("Asm");

    let function = asm_impl.new_fn("encode");
    function.vis("pub");
    function.arg_ref_self();
    function.ret("u32");
    function.line("match self {");

    for instruction in instructions.iter() {
        let arguments = instruction
            .args()
            .iter()
            .map(|x| x.name.clone())
            .collect::<Vec<String>>()
            .join(",");
        function.line(format!(
            "Asm::{} {{ {} }} => {{",
            instruction.name, arguments
        ));

        // TODO: Deduplicate
        if instruction.diagrams.len() == 1 {
            let diagram = &instruction.diagrams[0];
            let bits = diagram
                .boxes
                .iter()
                .filter_map(|x| x.bits.render())
                .collect::<Vec<String>>()
                .join("_");
            function.line(format!("0b{}", bits));
            for armbox in diagram.boxes.iter() {
                if !armbox.is_arg() {
                    continue;
                }
                match armbox.kind() {
                    Kind::Register => {
                        function.line(format!("| {} << {}", armbox.name, armbox.shift()));
                    }
                    Kind::Immediate => {
                        function.line(format!("| (*{} as u32) << {}", armbox.name, armbox.shift()));
                    }
                    Kind::ClassSelector(_) => {}
                    Kind::NonPowerOfTwoImm(n) => {
                        function.line(format!(
                            "| truncate_imm::<_, {}>(*{}) << {}",
                            n,
                            armbox.name,
                            armbox.shift()
                        ));
                    }
                }
            }
        } else {
            function.line("match class_selector {".to_string());
            for diagram in instruction.diagrams.iter() {
                function.line(format!(
                    "{}Selector::{} => {{",
                    instruction.name,
                    to_camel_case(&diagram.name)
                ));
                let bits = diagram
                    .boxes
                    .iter()
                    .filter_map(|x| x.bits.render())
                    .collect::<Vec<String>>()
                    .join("_");
                function.line(format!("0b{}", bits));
                for armbox in diagram.boxes.iter() {
                    if !armbox.is_arg() {
                        continue;
                    }
                    match armbox.kind() {
                        Kind::Register => {
                            function.line(format!("| {} << {}", armbox.name, armbox.shift()));
                        }
                        Kind::Immediate => {
                            function.line(format!(
                                "| (*{} as u32) << {}",
                                armbox.name,
                                armbox.shift()
                            ));
                        }
                        Kind::ClassSelector(_) => {}
                        Kind::NonPowerOfTwoImm(n) => {
                            function.line(format!(
                                "| truncate_imm::<_, {}>(*{}) << {}",
                                n,
                                armbox.name,
                                armbox.shift()
                            ));
                        }
                    }
                }
                function.line("}");
            }
            function.line("}");
        }

        function.line("}");
    }

    function.line("}");

    scope.to_string()
}

fn generate_class_selector_enums(instructions: &[Instruction]) -> String {
    let mut scope = Scope::new();

    for instruction in instructions {
        if instruction.diagrams.len() == 1 {
            continue;
        }

        let selector_enum = scope.new_enum(format!("{}Selector", instruction.name));
        selector_enum.vis("pub");
        selector_enum.derive("Debug");
        for diagram in instruction.diagrams.iter() {
            selector_enum.new_variant(to_camel_case(&diagram.name));
        }
    }

    scope.to_string()
}

fn generate_registers() -> String {
    let mut result = String::new();
    for i in 0..31 {
        result.push_str(&format!(
            "pub const X{i}: Register = Register {{
                index: {i},
                size: Size::S64,
            }};"
        ));
    }

    result.push_str(
        "
        pub const ZERO_REGISTER: Register = Register {
            index: 31,
            size: Size::S64,
        };",
    );

    result
}

pub fn generate_template() -> Result<(), Box<dyn Error>> {
    let xml_file_path =
        "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/asm-arm2/asm/resources/onebigfile.xml";

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
        instructions.push(Instruction {
            name,
            title,
            comments,
            description,
            diagrams,
        })
    }
    let registers = generate_registers();
    let instruction_enum = generate_instruction_enum(&instructions);
    let class_selector_enums = generate_class_selector_enums(&instructions);
    let encoding_instructions = generate_encoding_instructions(&instructions);

    let code = format!(
        "{}\n{}\n{}\n{}\n{}",
        TEMPLATE_PREFIX, registers, instruction_enum, class_selector_enums, encoding_instructions
    );
    let mut file = File::create("src/arm.rs")?;
    file.write_all(code.as_bytes())?;

    let result = Command::new("cargo")
        .arg("fmt")
        .arg("--")
        .arg("src/arm.rs")
        .spawn()?;

    println!("{:?}", result);

    Ok(())
}

fn get_last_modified_date(file_path: &str) -> Result<SystemTime, std::io::Error> {
    let metadata = fs::metadata(file_path)?;
    metadata.modified()
}

fn main() -> Result<(), Box<dyn Error>> {
    if get_last_modified_date("build.rs")? > get_last_modified_date("src/arm.rs")? {
        generate_template()?;
    }

    Ok(())
}
