use std::collections::HashSet;

pub mod parser;
pub mod rust_generator;
pub mod rust_function_generator;
pub mod cpp_function_generator;
pub mod zig_function_generator;
pub mod python_generator;

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub shift: u8,
    pub bits: Option<String>,
    pub width: u8,
    pub is_arg: bool,
    pub kind: FieldKind,
}

#[derive(Debug, Clone)]
pub enum FieldKind {
    Register,
    Immediate,
    ClassSelector(String),
    NonPowerOfTwoImm(u8),
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub name: String,
    pub title: String,
    pub comments: Vec<String>,
    pub description: String,
    pub fields: Vec<Field>,
    pub diagrams: Vec<RegisterDiagram>,
}

#[derive(Debug, Clone)]
pub struct RegisterDiagram {
    pub name: String,
    pub boxes: Vec<ArmBox>,
}

#[derive(Debug, Clone)]
pub struct ArmBox {
    pub hibit: u8,
    pub width: u8,
    pub name: String,
    pub bits: Bits,
}

#[derive(Debug, Clone)]
pub enum Bits {
    Empty(u8),
    AllNums(String),
    HasVariable(Vec<String>),
    Constraint(Vec<String>),
}

impl ArmBox {
    pub fn shift(&self) -> u8 {
        self.hibit - self.width
    }
}

impl Instruction {
    pub fn args(&self) -> Vec<&Field> {
        self.fields.iter().filter(|f| f.is_arg).collect()
    }
}

#[derive(Debug, Clone)]
pub struct InstructionFilter {
    allowed: Option<HashSet<String>>,
    blocked: Option<HashSet<String>>,
}

impl InstructionFilter {
    pub fn new() -> Self {
        Self {
            allowed: None,
            blocked: None,
        }
    }

    pub fn allow(mut self, names: Vec<String>) -> Self {
        self.allowed = Some(names.into_iter().collect());
        self
    }

    pub fn block(mut self, names: Vec<String>) -> Self {
        self.blocked = Some(names.into_iter().collect());
        self
    }

    pub fn should_include(&self, instruction: &Instruction) -> bool {
        if let Some(allowed) = &self.allowed {
            if !allowed.contains(&instruction.name) {
                return false;
            }
        }
        if let Some(blocked) = &self.blocked {
            if blocked.contains(&instruction.name) {
                return false;
            }
        }
        true
    }
}

pub trait CodeGenerator {
    fn generate_prefix(&self) -> String;
    
    fn generate_instruction_enum(&self, instructions: &[Instruction]) -> String;
    
    fn generate_encoding_impl(&self, instructions: &[Instruction]) -> String;
    
    fn generate_class_selector_enums(&self, instructions: &[Instruction]) -> String;
    
    fn generate_registers(&self) -> String;
    
    fn generate(&self, instructions: &[Instruction]) -> String {
        let mut output = String::new();
        output.push_str(&self.generate_prefix());
        output.push_str("\n\n");
        output.push_str(&self.generate_registers());
        output.push_str("\n\n");
        output.push_str(&self.generate_class_selector_enums(instructions));
        output.push_str("\n\n");
        output.push_str(&self.generate_instruction_enum(instructions));
        output.push_str("\n\n");
        output.push_str(&self.generate_encoding_impl(instructions));
        output.push_str("\n\n} // namespace arm_asm\n");
        output
    }
}

pub struct ArmCodeGen {
    instructions: Vec<Instruction>,
}

impl ArmCodeGen {
    /// Create a new ARM code generator with all available instructions
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instructions = parser::load_arm_instructions()?;
        Ok(Self { instructions })
    }
    
    /// Get all available instruction names
    pub fn available_instructions(&self) -> Vec<&str> {
        self.instructions.iter().map(|i| i.name.as_str()).collect()
    }
    
    /// Find instructions matching a pattern (case-insensitive)
    pub fn find_instructions(&self, pattern: &str) -> Vec<&str> {
        let pattern_lower = pattern.to_lowercase();
        self.instructions
            .iter()
            .filter(|i| i.name.to_lowercase().contains(&pattern_lower) || 
                       i.title.to_lowercase().contains(&pattern_lower))
            .map(|i| i.name.as_str())
            .collect()
    }
    
    /// Get instruction details
    pub fn instruction_info(&self, name: &str) -> Option<(&str, &str)> {
        self.instructions
            .iter()
            .find(|i| i.name == name)
            .map(|i| (i.name.as_str(), i.title.as_str()))
    }
    
    /// Generate code for specific instructions
    pub fn generate_filtered<G: CodeGenerator>(&self, generator: G, filter: InstructionFilter) -> String {
        let filtered_instructions: Vec<&Instruction> = self.instructions
            .iter()
            .filter(|i| filter.should_include(i))
            .collect();
        
        // Convert &Instruction to Instruction for the generator
        let owned_instructions: Vec<Instruction> = filtered_instructions
            .into_iter()
            .cloned()
            .collect();
            
        generator.generate(&owned_instructions)
    }
    
    /// Generate code for all instructions
    pub fn generate_all<G: CodeGenerator>(&self, generator: G) -> String {
        generator.generate(&self.instructions)
    }
    
    /// Generate code for specific instruction names
    pub fn generate<G: CodeGenerator>(&self, generator: G, instruction_names: Vec<&str>) -> String {
        let filter = InstructionFilter::new().allow(
            instruction_names.into_iter().map(|s| s.to_string()).collect()
        );
        self.generate_filtered(generator, filter)
    }
}