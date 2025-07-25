use arm_codegen_generic::{ArmCodeGen, CodeGenerator, Instruction};

/// Example custom code generator that creates Python functions
struct PythonGenerator;

impl CodeGenerator for PythonGenerator {
    fn generate_prefix(&self) -> String {
        "# Generated ARM instruction encoders\n\n".to_string()
    }
    
    fn generate_registers(&self) -> String {
        let mut result = String::new();
        for i in 0..31 {
            result.push_str(&format!("X{} = {}\n", i, i));
        }
        result.push_str("SP = 31\n\n");
        result
    }
    
    fn generate_instruction_enum(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();
        
        for instruction in instructions {
            let func_name = to_snake_case(&instruction.name);
            result.push_str(&format!("def {}(", func_name));
            
            let params: Vec<String> = instruction.fields
                .iter()
                .filter(|f| f.is_arg)
                .map(|f| f.name.clone())
                .collect();
            
            result.push_str(&params.join(", "));
            result.push_str("):\n");
            result.push_str(&format!("    \"\"\"{}\"\"\"\n", instruction.title));
            result.push_str("    # Implementation would go here\n");
            result.push_str("    return 0\n\n");
        }
        
        result
    }
    
    fn generate_encoding_impl(&self, _instructions: &[Instruction]) -> String {
        String::new() // Not needed for this simple example
    }
    
    fn generate_class_selector_enums(&self, _instructions: &[Instruction]) -> String {
        String::new() // Not needed for this simple example
    }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🐍 Custom Python Generator Example");
    println!("===================================");
    
    let arm = ArmCodeGen::new()?;
    
    // Generate Python code using our custom generator
    let python_code = arm.generate(
        PythonGenerator,
        vec!["AddAddsubImm", "MovMovzImm", "LdrLitLit"]
    );
    
    println!("Generated Python code:");
    println!("{}", python_code);
    
    println!("✨ You can create generators for any language by implementing CodeGenerator!");
    
    Ok(())
}