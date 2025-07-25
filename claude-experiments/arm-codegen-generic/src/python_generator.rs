use crate::{CodeGenerator, FieldKind, Instruction};

pub struct PythonCodeGenerator;

impl CodeGenerator for PythonCodeGenerator {
    fn generate_prefix(&self) -> String {
        r#"
from enum import Enum
from typing import Union

class Size(Enum):
    S32 = 0
    S64 = 1

class Register:
    def __init__(self, index: int, size: Size):
        self.index = index
        self.size = size
    
    def sf(self) -> int:
        return 1 if self.size == Size.S64 else 0
    
    def encode(self) -> int:
        return self.index

def truncate_imm(imm: int, width: int) -> int:
    masked = imm & ((1 << width) - 1)
    if imm >= 0:
        assert imm == masked, f"Value {imm} doesn't fit in {width} bits"
    else:
        assert imm == (masked | (0xFFFFFFFF << width)), f"Value {imm} doesn't fit in {width} bits"
    return masked
"#
        .to_string()
    }

    fn generate_registers(&self) -> String {
        let mut result = String::new();
        for i in 0..31 {
            result.push_str(&format!("X{} = Register({}, Size.S64)\n", i, i));
        }
        result.push_str("\nSP = Register(31, Size.S64)\n");
        result.push_str("ZERO_REGISTER = Register(31, Size.S64)\n");
        result
    }

    fn generate_instruction_enum(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();
        result.push_str("class ArmInstruction:\n");
        result.push_str("    def __init__(self, **kwargs):\n");
        result.push_str("        for key, value in kwargs.items():\n");
        result.push_str("            setattr(self, key, value)\n\n");

        for instruction in instructions {
            result.push_str(&format!("class {}(ArmInstruction):\n", instruction.name));
            
            // Add docstring
            if !instruction.title.is_empty() || !instruction.description.is_empty() {
                result.push_str("    \"\"\"\n");
                if !instruction.title.is_empty() {
                    result.push_str(&format!("    {}\n", instruction.title));
                }
                if !instruction.description.is_empty() {
                    result.push_str(&format!("    {}\n", instruction.description));
                }
                for comment in &instruction.comments {
                    if !comment.is_empty() {
                        result.push_str(&format!("    {}\n", comment));
                    }
                }
                result.push_str("    \"\"\"\n");
            }

            // Constructor
            let args: Vec<String> = instruction
                .fields
                .iter()
                .filter(|f| f.is_arg)
                .map(|f| {
                    let type_hint = match &f.kind {
                        FieldKind::Register => "Register",
                        FieldKind::Immediate => "int",
                        FieldKind::ClassSelector(_) => "str",
                        FieldKind::NonPowerOfTwoImm(_) => "int",
                    };
                    format!("{}: {}", f.name, type_hint)
                })
                .collect();
            
            if args.is_empty() {
                result.push_str("    def __init__(self):\n");
                result.push_str("        super().__init__()\n\n");
            } else {
                result.push_str(&format!("    def __init__(self, {}):\n", args.join(", ")));
                result.push_str("        super().__init__(\n");
                for (i, field) in instruction.fields.iter().filter(|f| f.is_arg).enumerate() {
                    if i == instruction.fields.iter().filter(|f| f.is_arg).count() - 1 {
                        result.push_str(&format!("            {}={}\n", field.name, field.name));
                    } else {
                        result.push_str(&format!("            {}={},\n", field.name, field.name));
                    }
                }
                result.push_str("        )\n\n");
            }
        }

        result
    }

    fn generate_encoding_impl(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();
        result.push_str("def encode_instruction(instruction) -> int:\n");
        result.push_str("    \"\"\"\n");
        result.push_str("    Encode an ARM instruction to its binary representation\n");
        result.push_str("    \"\"\"\n");
        
        for (i, instruction) in instructions.iter().enumerate() {
            let class_check = if i == 0 { "if" } else { "elif" };
            result.push_str(&format!("    {} isinstance(instruction, {}):\n", class_check, instruction.name));

            if instruction.diagrams.len() == 1 {
                let diagram = &instruction.diagrams[0];
                let bits = diagram
                    .boxes
                    .iter()
                    .filter_map(|x| x.bits.render())
                    .collect::<Vec<String>>()
                    .join("");
                result.push_str(&format!("        result = 0b{}\n", bits));
                
                for armbox in diagram.boxes.iter() {
                    if !armbox.is_arg() {
                        continue;
                    }
                    match armbox.kind() {
                        FieldKind::Register => {
                            result.push_str(&format!(
                                "        result |= instruction.{}.encode() << {}\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::Immediate => {
                            result.push_str(&format!(
                                "        result |= (instruction.{} & 0xFFFFFFFF) << {}\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::ClassSelector(_) => {}
                        FieldKind::NonPowerOfTwoImm(n) => {
                            result.push_str(&format!(
                                "        result |= truncate_imm(instruction.{}, {}) << {}\n",
                                armbox.name.to_ascii_lowercase(),
                                n,
                                armbox.shift()
                            ));
                        }
                    }
                }
                result.push_str("        return result\n");
            } else {
                result.push_str("        if instruction.class_selector == ");
                for (j, diagram) in instruction.diagrams.iter().enumerate() {
                    if j > 0 {
                        result.push_str("        elif instruction.class_selector == ");
                    }
                    result.push_str(&format!("'{}':\n", to_camel_case(&diagram.name)));
                    
                    let bits = diagram
                        .boxes
                        .iter()
                        .filter_map(|x| x.bits.render())
                        .collect::<Vec<String>>()
                        .join("");
                    result.push_str(&format!("            result = 0b{}\n", bits));
                    
                    for armbox in diagram.boxes.iter() {
                        if !armbox.is_arg() {
                            continue;
                        }
                        match armbox.kind() {
                            FieldKind::Register => {
                                result.push_str(&format!(
                                    "            result |= instruction.{}.encode() << {}\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::Immediate => {
                                result.push_str(&format!(
                                    "            result |= (instruction.{} & 0xFFFFFFFF) << {}\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::ClassSelector(_) => {}
                            FieldKind::NonPowerOfTwoImm(n) => {
                                result.push_str(&format!(
                                    "            result |= truncate_imm(instruction.{}, {}) << {}\n",
                                    armbox.name.to_ascii_lowercase(),
                                    n,
                                    armbox.shift()
                                ));
                            }
                        }
                    }
                    result.push_str("            return result\n");
                }
            }
        }

        result.push_str("    else:\n");
        result.push_str("        raise ValueError(f'Unknown instruction type: {type(instruction)}')\n");
        
        result
    }

    fn generate_class_selector_enums(&self, _instructions: &[Instruction]) -> String {
        // Python doesn't need separate enums for class selectors since we use strings
        String::new()
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