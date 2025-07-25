use crate::{CodeGenerator, FieldKind, Instruction};
use codegen::{Scope, Type};

pub struct RustCodeGenerator;

impl CodeGenerator for RustCodeGenerator {
    fn generate_prefix(&self) -> String {
        r#"
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
"#
        .to_string()
    }

    fn generate_registers(&self) -> String {
        let mut result = String::new();
        for i in 0..31 {
            result.push_str(&format!(
                "pub const X{i}: Register = Register {{
    index: {i},
    size: Size::S64,
}};
"
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

    fn generate_instruction_enum(&self, instructions: &[Instruction]) -> String {
        let mut scope = Scope::new();

        let enum_gen = scope.new_enum("ArmAsm");
        enum_gen.vis("pub");
        enum_gen.derive("Debug");

        for instruction in instructions.iter() {
            let variant = enum_gen.new_variant(instruction.name.clone());
            let comments = self.get_instruction_comments(instruction);
            variant.annotation(comments);
            for field in instruction.fields.iter().filter(|x| x.is_arg) {
                let field_type = match &field.kind {
                    FieldKind::Register => "Register",
                    FieldKind::Immediate => "i32",
                    FieldKind::ClassSelector(name) => name,
                    FieldKind::NonPowerOfTwoImm(_) => "i32",
                };
                variant.named(&field.name, Type::new(field_type));
            }
        }
        scope.to_string()
    }

    fn generate_encoding_impl(&self, instructions: &[Instruction]) -> String {
        let mut scope = Scope::new();
        let asm_impl = scope.new_impl("ArmAsm");

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
                "ArmAsm::{} {{ {} }} => {{",
                instruction.name, arguments
            ));

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
                        FieldKind::Register => {
                            function.line(format!(
                                "| {} << {}",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::Immediate => {
                            function.line(format!(
                                "| (*{} as u32) << {}",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::ClassSelector(_) => {}
                        FieldKind::NonPowerOfTwoImm(n) => {
                            function.line(format!(
                                "| truncate_imm::<_, {}>(*{}) << {}",
                                n,
                                armbox.name.to_ascii_lowercase(),
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
                            FieldKind::Register => {
                                function.line(format!(
                                    "| {} << {}",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::Immediate => {
                                function.line(format!(
                                    "| (*{} as u32) << {}",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::ClassSelector(_) => {}
                            FieldKind::NonPowerOfTwoImm(n) => {
                                function.line(format!(
                                    "| truncate_imm::<_, {}>(*{}) << {}",
                                    n,
                                    armbox.name.to_ascii_lowercase(),
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

    fn generate_class_selector_enums(&self, instructions: &[Instruction]) -> String {
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
}

impl RustCodeGenerator {
    fn get_instruction_comments(&self, instruction: &Instruction) -> String {
        let mut result = String::new();
        result.push_str(&format!("/// {}", instruction.title));
        result.push('\n');
        result.push_str(
            &instruction
                .description
                .lines()
                .map(|x| format!("/// {}", x))
                .collect::<Vec<String>>()
                .join("\n"),
        );
        result.push('\n');
        result.push_str(
            &instruction
                .comments
                .iter()
                .map(|s| format!("/// {}", s))
                .collect::<Vec<String>>()
                .join("\n"),
        );

        result
    }
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