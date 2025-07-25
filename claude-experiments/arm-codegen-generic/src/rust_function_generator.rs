use crate::{CodeGenerator, FieldKind, Instruction};

pub struct RustFunctionGenerator;

impl CodeGenerator for RustFunctionGenerator {
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
        let mut result = String::new();
        
        for instruction in instructions {
            // Generate function signature
            let function_name = to_snake_case(&instruction.name);
            let mut params = Vec::new();
            
            // Add instruction-specific parameters
            for field in instruction.fields.iter().filter(|f| f.is_arg) {
                let param_type = match &field.kind {
                    FieldKind::Register => "Register",
                    FieldKind::Immediate => "i32",
                    FieldKind::ClassSelector(name) => name,
                    FieldKind::NonPowerOfTwoImm(_) => "i32",
                };
                params.push(format!("{}: {}", field.name, param_type));
            }
            
            // Generate function documentation
            result.push_str(&format!("/// {}\n", instruction.title));
            if !instruction.description.is_empty() {
                for line in instruction.description.lines() {
                    result.push_str(&format!("/// {}\n", line));
                }
            }
            for comment in &instruction.comments {
                if !comment.is_empty() {
                    result.push_str(&format!("/// {}\n", comment));
                }
            }
            
            // Generate function signature
            if params.is_empty() {
                result.push_str(&format!("pub fn {}() -> u32 {{\n", function_name));
            } else {
                result.push_str(&format!("pub fn {}({}) -> u32 {{\n", function_name, params.join(", ")));
            }
            
            // Generate function body based on instruction diagrams
            if instruction.diagrams.len() == 1 {
                let diagram = &instruction.diagrams[0];
                let bits = diagram
                    .boxes
                    .iter()
                    .filter_map(|x| x.bits.render())
                    .collect::<Vec<String>>()
                    .join("_");
                result.push_str(&format!("    let mut result = 0b{};\n", bits));
                
                for armbox in diagram.boxes.iter() {
                    if !armbox.is_arg() {
                        continue;
                    }
                    match armbox.kind() {
                        FieldKind::Register => {
                            result.push_str(&format!(
                                "    result |= ({}.encode() as u32) << {};\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::Immediate => {
                            result.push_str(&format!(
                                "    result |= ({} as u32) << {};\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::ClassSelector(_) => {}
                        FieldKind::NonPowerOfTwoImm(n) => {
                            result.push_str(&format!(
                                "    result |= truncate_imm::<_, {}>({}) << {};\n",
                                n,
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                    }
                }
                result.push_str("    result\n");
            } else {
                // Multiple diagrams - use match on class_selector
                result.push_str("    match class_selector {\n");
                for diagram in instruction.diagrams.iter() {
                    let selector_variant = to_camel_case(&diagram.name);
                    result.push_str(&format!("        {}Selector::{} => {{\n", instruction.name, selector_variant));
                    
                    let bits = diagram
                        .boxes
                        .iter()
                        .filter_map(|x| x.bits.render())
                        .collect::<Vec<String>>()
                        .join("_");
                    result.push_str(&format!("            let mut result = 0b{};\n", bits));
                    
                    for armbox in diagram.boxes.iter() {
                        if !armbox.is_arg() {
                            continue;
                        }
                        match armbox.kind() {
                            FieldKind::Register => {
                                result.push_str(&format!(
                                    "            result |= ({}.encode() as u32) << {};\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::Immediate => {
                                result.push_str(&format!(
                                    "            result |= ({} as u32) << {};\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::ClassSelector(_) => {}
                            FieldKind::NonPowerOfTwoImm(n) => {
                                result.push_str(&format!(
                                    "            result |= truncate_imm::<_, {}>({}) << {};\n",
                                    n,
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                        }
                    }
                    result.push_str("            result\n");
                    result.push_str("        }\n");
                }
                result.push_str("    }\n");
            }
            
            result.push_str("}\n\n");
        }
        
        result
    }

    fn generate_encoding_impl(&self, _instructions: &[Instruction]) -> String {
        // No encoding impl needed since each function directly returns u32
        String::new()
    }

    fn generate_class_selector_enums(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();

        for instruction in instructions {
            if instruction.diagrams.len() == 1 {
                continue;
            }

            result.push_str(&format!("#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n"));
            result.push_str(&format!("pub enum {}Selector {{\n", instruction.name));
            for diagram in instruction.diagrams.iter() {
                result.push_str(&format!("    {},\n", to_camel_case(&diagram.name)));
            }
            result.push_str("}\n\n");
        }

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

fn to_snake_case(name: &str) -> String {
    let mut result = String::new();
    let mut prev_was_lower = false;

    for c in name.chars() {
        if c.is_uppercase() && prev_was_lower {
            result.push('_');
            result.push(c.to_ascii_lowercase());
        } else if c == '-' || c == ' ' || c == ',' {
            result.push('_');
        } else {
            result.push(c.to_ascii_lowercase());
        }
        prev_was_lower = c.is_lowercase();
    }

    result
}