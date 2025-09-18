use crate::{CodeGenerator, FieldKind, Instruction};

pub struct CppFunctionGenerator;

impl CodeGenerator for CppFunctionGenerator {
    fn generate_prefix(&self) -> String {
        r#"
#pragma once
#include <cstdint>
#include <cassert>

namespace arm_asm {

enum class Size : uint8_t {
    S32 = 0,
    S64 = 1
};

struct Register {
    Size size;
    uint8_t index;
    
    constexpr Register(uint8_t idx, Size sz) : size(sz), index(idx) {}
    
    constexpr int sf() const {
        return static_cast<int>(size);
    }
    
    constexpr uint8_t encode() const {
        return index;
    }
};

// Register constants
constexpr Register SP{31, Size::S64};
constexpr Register ZERO_REGISTER{31, Size::S64};

template<int WIDTH>
constexpr uint32_t truncate_imm(int32_t imm) {
    static_assert(WIDTH > 0 && WIDTH <= 32, "Width must be between 1 and 32");
    const uint32_t masked = static_cast<uint32_t>(imm) & ((1U << WIDTH) - 1);
    
    // Assert that we didn't drop any bits by truncating
    if (imm >= 0) {
        assert(static_cast<uint32_t>(imm) == masked);
    } else {
        assert(static_cast<uint32_t>(imm) == (masked | (0xFFFFFFFFU << WIDTH)));
    }
    
    return masked;
}
"#
        .to_string()
    }

    fn generate_registers(&self) -> String {
        let mut result = String::new();
        
        // Generate X0-X30 register constants
        for i in 0..31 {
            result.push_str(&format!("constexpr Register X{}{{{}U, Size::S64}};\n", i, i));
        }
        
        result.push_str("\n");
        result
    }

    fn generate_instruction_enum(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();
        
        for instruction in instructions {
            // Generate function documentation
            result.push_str("/**\n");
            result.push_str(&format!(" * {}\n", instruction.title));
            if !instruction.description.is_empty() {
                result.push_str(&format!(" * {}\n", instruction.description));
            }
            for comment in &instruction.comments {
                if !comment.is_empty() {
                    result.push_str(&format!(" * {}\n", comment));
                }
            }
            result.push_str(" */\n");
            
            // Generate function signature
            let function_name = to_snake_case(&instruction.name);
            let mut params = Vec::new();
            
            // Add instruction-specific parameters
            for field in instruction.fields.iter().filter(|f| f.is_arg) {
                let param_type = match &field.kind {
                    FieldKind::Register => "Register",
                    FieldKind::Immediate => "int32_t",
                    FieldKind::ClassSelector(_name) => {
                        // For C++, we'll use an enum class
                        &format!("{}Selector", instruction.name)
                    },
                    FieldKind::NonPowerOfTwoImm(_) => "int32_t",
                };
                params.push(format!("{} {}", param_type, field.name));
            }
            
            // Generate function signature
            if params.is_empty() {
                result.push_str(&format!("constexpr uint32_t {}() noexcept {{\n", function_name));
            } else {
                result.push_str(&format!(
                    "constexpr uint32_t {}({}) noexcept {{\n", 
                    function_name, 
                    params.join(", ")
                ));
            }
            
            // Generate function body based on instruction diagrams
            if instruction.diagrams.len() == 1 {
                let diagram = &instruction.diagrams[0];
                let bits = diagram
                    .boxes
                    .iter()
                    .filter_map(|x| x.bits.render())
                    .collect::<Vec<String>>()
                    .join("");
                result.push_str(&format!("    uint32_t result = 0b{}U;\n", bits));
                
                for armbox in diagram.boxes.iter() {
                    if !armbox.is_arg() {
                        continue;
                    }
                    match armbox.kind() {
                        FieldKind::Register => {
                            result.push_str(&format!(
                                "    result |= static_cast<uint32_t>({}.encode()) << {}U;\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::Immediate => {
                            result.push_str(&format!(
                                "    result |= static_cast<uint32_t>({}) << {}U;\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::ClassSelector(_) => {}
                        FieldKind::NonPowerOfTwoImm(n) => {
                            result.push_str(&format!(
                                "    result |= truncate_imm<{}>({}) << {}U;\n",
                                n,
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                    }
                }
                result.push_str("    return result;\n");
            } else {
                // Multiple diagrams - use switch on class_selector
                result.push_str("    switch (class_selector) {\n");
                for diagram in instruction.diagrams.iter() {
                    let selector_variant = to_camel_case(&diagram.name);
                    result.push_str(&format!(
                        "    case {}Selector::{}: {{\n", 
                        instruction.name, 
                        selector_variant
                    ));
                    
                    let bits = diagram
                        .boxes
                        .iter()
                        .filter_map(|x| x.bits.render())
                        .collect::<Vec<String>>()
                        .join("");
                    result.push_str(&format!("        uint32_t result = 0b{}U;\n", bits));
                    
                    for armbox in diagram.boxes.iter() {
                        if !armbox.is_arg() {
                            continue;
                        }
                        match armbox.kind() {
                            FieldKind::Register => {
                                result.push_str(&format!(
                                    "        result |= static_cast<uint32_t>({}.encode()) << {}U;\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::Immediate => {
                                result.push_str(&format!(
                                    "        result |= static_cast<uint32_t>({}) << {}U;\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::ClassSelector(_) => {}
                            FieldKind::NonPowerOfTwoImm(n) => {
                                result.push_str(&format!(
                                    "        result |= truncate_imm<{}>({}) << {}U;\n",
                                    n,
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                        }
                    }
                    result.push_str("        return result;\n");
                    result.push_str("    }\n");
                }
                result.push_str("    default:\n");
                result.push_str("        // Should never reach here if all cases are handled\n");
                result.push_str("        return 0U;\n");
                result.push_str("    }\n");
            }
            
            result.push_str("}\n\n");
        }
        
        result
    }

    fn generate_encoding_impl(&self, _instructions: &[Instruction]) -> String {
        // No separate encoding implementation needed since each function directly returns uint32_t
        String::new()
    }

    fn generate_class_selector_enums(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();

        for instruction in instructions {
            if instruction.diagrams.len() == 1 {
                continue;
            }

            result.push_str(&format!("enum class {}Selector {{\n", instruction.name));
            for diagram in instruction.diagrams.iter() {
                result.push_str(&format!("    {},\n", to_camel_case(&diagram.name)));
            }
            result.push_str("};\n\n");
        }

        result
    }
}

impl CppFunctionGenerator {
    pub fn generate_with_namespace(&self, instructions: &[Instruction]) -> String {
        let mut output = String::new();
        output.push_str(&self.generate_prefix());
        output.push_str("\n");
        output.push_str(&self.generate_registers());
        output.push_str("\n");
        output.push_str(&self.generate_class_selector_enums(instructions));
        output.push_str("\n");
        output.push_str(&self.generate_instruction_enum(instructions));
        output.push_str("\n} // namespace arm_asm\n");
        output
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