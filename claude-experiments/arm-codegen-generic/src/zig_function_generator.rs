use crate::{CodeGenerator, FieldKind, Instruction};

pub struct ZigFunctionGenerator;

impl CodeGenerator for ZigFunctionGenerator {
    fn generate(&self, instructions: &[Instruction]) -> String {
        let mut output = String::new();
        output.push_str(&self.generate_prefix());
        output.push_str("\n\n");
        output.push_str(&self.generate_registers());
        output.push_str("\n\n");
        output.push_str(&self.generate_class_selector_enums(instructions));
        output.push_str("\n\n");
        output.push_str(&self.generate_instruction_enum(instructions));
        output
    }

    fn generate_prefix(&self) -> String {
        r#"const std = @import("std");

pub const Size = enum(u8) {
    S32 = 0,
    S64 = 1,
};

pub const Register = extern struct {
    size: Size,
    index: u8,

    pub fn init(idx: u8, sz: Size) Register {
        return .{ .size = sz, .index = idx };
    }

    pub fn sf(self: Register) u32 {
        return @intFromEnum(self.size);
    }

    pub fn encode(self: Register) u32 {
        return self.index;
    }
};

// Register constants
pub const SP = Register.init(31, .S64);
pub const ZERO_REGISTER = Register.init(31, .S64);"#
        .to_string()
    }

    fn generate_registers(&self) -> String {
        let mut result = String::new();

        // Generate X0-X30 register constants
        result.push_str("\n// 64-bit registers\n");
        for i in 0..31 {
            result.push_str(&format!("pub const X{} = Register.init({}, .S64);\n", i, i));
        }

        // Generate W0-W30 register constants
        result.push_str("\n// 32-bit registers\n");
        for i in 0..31 {
            result.push_str(&format!("pub const W{} = Register.init({}, .S32);\n", i, i));
        }

        result.push_str("\n");
        result
    }

    fn generate_instruction_enum(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();

        for instruction in instructions {
            // Generate function documentation
            result.push_str(&format!("/// {} -- A64\n", instruction.title));
            if !instruction.description.is_empty() {
                result.push_str(&format!("/// {}\n", instruction.description));
            }
            for comment in &instruction.comments {
                if !comment.is_empty() {
                    result.push_str(&format!("/// {}\n", comment));
                }
            }

            // Generate function signature
            let function_name = to_snake_case(&instruction.name);
            let mut params = Vec::new();

            // Add instruction-specific parameters - Zig-friendly version with exact widths
            // Filter out zero-width fields (fixed bits) and invalid fields
            for field in instruction.fields.iter().filter(|f| f.is_arg && f.width > 0) {
                let param_type = match &field.kind {
                    FieldKind::Register => "Register",
                    FieldKind::Immediate => {
                        // Use wider types for very narrow fields to handle shifts
                        if field.width <= 4 {
                            "u8"  // Use u8 minimum for fields 4 bits or smaller
                        } else {
                            &format!("u{}", field.width)
                        }
                    },
                    FieldKind::ClassSelector(_) => {
                        // For Zig, we'll use an enum
                        &format!("{}Selector", instruction.name)
                    },
                    FieldKind::NonPowerOfTwoImm(width) => {
                        // Use wider types for very narrow fields to handle shifts
                        if *width <= 4 {
                            "i8"  // Use i8 minimum for fields 4 bits or smaller
                        } else {
                            &format!("i{}", width)
                        }
                    },
                };
                params.push(format!("{}: {}", field.name, param_type));
            }

            // Add class_selector parameter for multi-diagram instructions
            if instruction.diagrams.len() > 1 {
                let selector_type = format!("{}Selector", instruction.name);
                params.push(format!("class_selector: {}", selector_type));
            }

            // Generate Zig-friendly internal function first
            if params.is_empty() {
                result.push_str(&format!("pub fn {}() u32 {{\n", function_name));
            } else {
                result.push_str(&format!(
                    "pub fn {}({}) u32 {{\n",
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
                // Check if we have any modifications to make
                let has_modifications = diagram.boxes.iter().any(|box_item| box_item.is_arg() && box_item.width > 0);
                if has_modifications {
                    result.push_str(&format!("    var result: u32 = 0b{};\n", bits));
                } else {
                    result.push_str(&format!("    const result: u32 = 0b{};\n", bits));
                }

                for armbox in diagram.boxes.iter() {
                    if !armbox.is_arg() || armbox.width == 0 {
                        continue;
                    }
                    match armbox.kind() {
                        FieldKind::Register => {
                            result.push_str(&format!(
                                "    result |= {}.encode() << {};\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::Immediate => {
                            let field_name = armbox.name.to_ascii_lowercase();
                            if armbox.width <= 4 {
                                // Need to truncate u8 parameter to actual field width and cast to u32 for shift
                                result.push_str(&format!(
                                    "    result |= (@as(u32, @as(u{}, @truncate({})))) << {};\n",
                                    armbox.width,
                                    field_name,
                                    armbox.shift()
                                ));
                            } else {
                                result.push_str(&format!(
                                    "    result |= (@as(u32, {})) << {};\n",
                                    field_name,
                                    armbox.shift()
                                ));
                            }
                        }
                        FieldKind::ClassSelector(_) => {}
                        FieldKind::NonPowerOfTwoImm(_width) => {
                            // For signed immediates, bitcast to unsigned of same width
                            let field_name = armbox.name.to_ascii_lowercase();
                            if armbox.width <= 4 {
                                // Need to truncate i8 parameter to actual field width, bitcast, then cast to u32 for shift
                                result.push_str(&format!(
                                    "    result |= (@as(u32, @as(u{}, @bitCast(@as(i{}, @truncate({})))))) << {};\n",
                                    armbox.width,
                                    armbox.width,
                                    field_name,
                                    armbox.shift()
                                ));
                            } else {
                                result.push_str(&format!(
                                    "    result |= (@as(u32, @as(u{}, @bitCast({})))) << {};\n",
                                    armbox.width,
                                    field_name,
                                    armbox.shift()
                                ));
                            }
                        }
                    }
                }
                result.push_str("    return result;\n");
            } else {
                // Multiple diagrams - use switch on class_selector
                result.push_str("    return switch (class_selector) {\n");
                for diagram in instruction.diagrams.iter() {
                    let selector_variant = to_camel_case(&diagram.name);
                    result.push_str(&format!(
                        "        .{} => blk: {{\n",
                        selector_variant
                    ));

                    let bits = diagram
                        .boxes
                        .iter()
                        .filter_map(|x| x.bits.render())
                        .collect::<Vec<String>>()
                        .join("");
                    // Check if we have any modifications to make
                    let has_modifications = diagram.boxes.iter().any(|box_item| box_item.is_arg() && box_item.width > 0);
                    if has_modifications {
                        result.push_str(&format!("            var result: u32 = 0b{};\n", bits));
                    } else {
                        result.push_str(&format!("            const result: u32 = 0b{};\n", bits));
                    }

                    for armbox in diagram.boxes.iter() {
                        if !armbox.is_arg() || armbox.width == 0 {
                            continue;
                        }
                        match armbox.kind() {
                            FieldKind::Register => {
                                result.push_str(&format!(
                                    "            result |= {}.encode() << {};\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::Immediate => {
                                let field_name = armbox.name.to_ascii_lowercase();
                                if armbox.width <= 4 {
                                    // Need to truncate u8 parameter to actual field width and cast to u32 for shift
                                    result.push_str(&format!(
                                        "            result |= (@as(u32, @as(u{}, @truncate({})))) << {};\n",
                                        armbox.width,
                                        field_name,
                                        armbox.shift()
                                    ));
                                } else {
                                    result.push_str(&format!(
                                        "            result |= (@as(u32, {})) << {};\n",
                                        field_name,
                                        armbox.shift()
                                    ));
                                }
                            }
                            FieldKind::ClassSelector(_) => {}
                            FieldKind::NonPowerOfTwoImm(_width) => {
                                let field_name = armbox.name.to_ascii_lowercase();
                                if armbox.width <= 4 {
                                    // Need to truncate i8 parameter to actual field width, then bitcast
                                    result.push_str(&format!(
                                        "            result |= (@as(u32, @as(u{}, @bitCast(@as(i{}, @truncate({})))))) << {};\n",
                                        armbox.width,
                                        armbox.width,
                                        field_name,
                                        armbox.shift()
                                    ));
                                } else {
                                    result.push_str(&format!(
                                        "            result |= (@as(u32, @as(u{}, @bitCast({})))) << {};\n",
                                        armbox.width,
                                        field_name,
                                        armbox.shift()
                                    ));
                                }
                            }
                        }
                    }
                    result.push_str("            break :blk result;\n");
                    result.push_str("        },\n");
                }
                result.push_str("    };\n");
            }

            result.push_str("}\n\n");

            // Generate C-compatible export wrapper
            self.generate_c_export_wrapper(instruction, &mut result);
        }

        result
    }

    fn generate_encoding_impl(&self, _instructions: &[Instruction]) -> String {
        // No separate encoding implementation needed since each function directly returns u32
        String::new()
    }

    fn generate_class_selector_enums(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();

        for instruction in instructions {
            if instruction.diagrams.len() == 1 {
                continue;
            }

            result.push_str(&format!("pub const {}Selector = enum(u32) {{\n", instruction.name));
            for diagram in instruction.diagrams.iter() {
                result.push_str(&format!("    {},\n", to_camel_case(&diagram.name)));
            }
            result.push_str("};\n\n");
        }

        // Add common condition codes and shift types
        result.push_str(r#"// Condition codes for conditional branches
pub const Condition = enum(u32) {
    EQ = 0b0000, // Equal
    NE = 0b0001, // Not equal
    CS = 0b0010, // Carry set (unsigned higher or same)
    CC = 0b0011, // Carry clear (unsigned lower)
    MI = 0b0100, // Minus (negative)
    PL = 0b0101, // Plus (positive or zero)
    VS = 0b0110, // Overflow set
    VC = 0b0111, // Overflow clear
    HI = 0b1000, // Unsigned higher
    LS = 0b1001, // Unsigned lower or same
    GE = 0b1010, // Signed greater than or equal
    LT = 0b1011, // Signed less than
    GT = 0b1100, // Signed greater than
    LE = 0b1101, // Signed less than or equal
    AL = 0b1110, // Always
};

// Shift types for shifted register operands
pub const ShiftType = enum(u32) {
    LSL = 0b00, // Logical shift left
    LSR = 0b01, // Logical shift right
    ASR = 0b10, // Arithmetic shift right
    ROR = 0b11, // Rotate right
};

// Extend types for extended register operands
pub const ExtendType = enum(u32) {
    UXTB = 0b000, // Unsigned extend byte
    UXTH = 0b001, // Unsigned extend halfword
    UXTW = 0b010, // Unsigned extend word
    UXTX = 0b011, // Unsigned extend doubleword
    SXTB = 0b100, // Signed extend byte
    SXTH = 0b101, // Signed extend halfword
    SXTW = 0b110, // Signed extend word
    SXTX = 0b111, // Signed extend doubleword
};

"#);

        result
    }
}


impl ZigFunctionGenerator {
    pub fn generate_with_header(&self, instructions: &[Instruction]) -> String {
        let mut output = String::new();
        output.push_str(&self.generate_prefix());
        output.push_str("\n");
        output.push_str(&self.generate_registers());
        output.push_str("\n");
        output.push_str(&self.generate_class_selector_enums(instructions));
        output.push_str("\n");
        output.push_str(&self.generate_instruction_enum(instructions));
        output
    }

    fn generate_c_export_wrapper(&self, instruction: &Instruction, result: &mut String) {
        let function_name = to_snake_case(&instruction.name);
        let c_function_name = format!("{}_c", function_name);

        // Generate C-compatible parameters
        let mut c_params = Vec::new();
        let mut call_args = Vec::new();

        for field in instruction.fields.iter().filter(|f| f.is_arg && f.width > 0) {
            match &field.kind {
                FieldKind::Register => {
                    c_params.push(format!("{}: Register", field.name));
                    call_args.push(field.name.clone());
                },
                FieldKind::Immediate => {
                    // Use standard C types
                    let c_type = if field.width <= 8 { "u8" }
                    else if field.width <= 16 { "u16" }
                    else { "u32" };
                    c_params.push(format!("{}: {}", field.name, c_type));
                    // Need to cast/truncate when calling the main function
                    if field.width <= 4 {
                        call_args.push(format!("@as(u{}, @truncate({}))", field.width, field.name));
                    } else if field.width < 32 {
                        call_args.push(format!("@as(u{}, @truncate({}))", field.width, field.name));
                    } else {
                        call_args.push(field.name.clone());
                    }
                },
                FieldKind::ClassSelector(_) => {
                    // Use u32 for C compatibility
                    c_params.push(format!("{}: u32", field.name));
                    call_args.push(format!("@as({}Selector, @enumFromInt({}))", instruction.name, field.name));
                },
                FieldKind::NonPowerOfTwoImm(width) => {
                    // Use standard signed C types
                    let c_type = if *width <= 8 { "i8" }
                    else if *width <= 16 { "i16" }
                    else { "i32" };
                    c_params.push(format!("{}: {}", field.name, c_type));
                    // Need to cast/truncate when calling the main function
                    if field.width <= 4 {
                        call_args.push(format!("@as(i{}, @truncate({}))", field.width, field.name));
                    } else if field.width < 32 {
                        call_args.push(format!("@as(i{}, @truncate({}))", field.width, field.name));
                    } else {
                        call_args.push(field.name.clone());
                    }
                },
            }
        }

        // Add class_selector parameter for multi-diagram instructions
        if instruction.diagrams.len() > 1 {
            // Use u32 for C compatibility
            c_params.push(format!("class_selector: u32"));
            call_args.push(format!("@as({}Selector, @enumFromInt(class_selector))", instruction.name));
        }

        // Generate C export wrapper function
        result.push_str(&format!("/// C-compatible wrapper for {}\n", function_name));
        if c_params.is_empty() {
            result.push_str(&format!("export fn {}() u32 {{\n", c_function_name));
            result.push_str(&format!("    return {}();\n", function_name));
        } else {
            result.push_str(&format!(
                "export fn {}({}) u32 {{\n",
                c_function_name,
                c_params.join(", ")
            ));
            result.push_str(&format!(
                "    return {}({});\n",
                function_name,
                call_args.join(", ")
            ));
        }
        result.push_str("}\n\n");
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

