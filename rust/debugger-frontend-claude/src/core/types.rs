use serde::{Deserialize, Serialize};

/// Built-in types for the custom language with tagged pointer support
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuiltInTypes {
    Int,
    Float,
    String,
    Bool,
    Function,
    Closure,
    Struct,
    Array,
    Null,
    None,
}

impl BuiltInTypes {
    pub fn null_value() -> isize {
        0b111
    }

    pub fn tag(&self, value: isize) -> isize {
        let value = value << 3;
        let tag = self.get_tag();
        value | tag
    }

    pub fn get_tag(&self) -> isize {
        match self {
            BuiltInTypes::Int => 0b000,
            BuiltInTypes::Float => 0b001,
            BuiltInTypes::String => 0b010,
            BuiltInTypes::Bool => 0b011,
            BuiltInTypes::Function => 0b100,
            BuiltInTypes::Closure => 0b101,
            BuiltInTypes::Struct => 0b110,
            BuiltInTypes::Array => 0b111,
            BuiltInTypes::Null => 0b111,
            BuiltInTypes::None => 0,
        }
    }

    pub fn untag(pointer: usize) -> usize {
        pointer >> 3
    }

    pub fn get_kind(pointer: usize) -> Self {
        if pointer == Self::null_value() as usize {
            return BuiltInTypes::Null;
        }
        match pointer & 0b111 {
            0b000 => BuiltInTypes::Int,
            0b001 => BuiltInTypes::Float,
            0b010 => BuiltInTypes::String,
            0b011 => BuiltInTypes::Bool,
            0b100 => BuiltInTypes::Function,
            0b101 => BuiltInTypes::Closure,
            0b110 => BuiltInTypes::Struct,
            0b111 => BuiltInTypes::Array,
            _ => panic!("Invalid tag"),
        }
    }

    pub fn is_embedded(&self) -> bool {
        match self {
            BuiltInTypes::Int => true,
            BuiltInTypes::Float => true,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => true,
            BuiltInTypes::Function => false,
            BuiltInTypes::Struct => false,
            BuiltInTypes::Array => false,
            BuiltInTypes::Closure => false,
            BuiltInTypes::Null => true,
            BuiltInTypes::None => false,
        }
    }

    pub fn construct_int(value: isize) -> isize {
        if value > isize::MAX >> 3 {
            panic!("Integer overflow")
        }
        BuiltInTypes::Int.tag(value)
    }

    pub fn construct_boolean(value: bool) -> isize {
        let bool = BuiltInTypes::Bool;
        if value {
            bool.tag(1)
        } else {
            bool.tag(0)
        }
    }

    pub fn tag_size() -> i32 {
        3
    }

    pub fn is_heap_pointer(value: usize) -> bool {
        match BuiltInTypes::get_kind(value) {
            BuiltInTypes::Int => false,
            BuiltInTypes::Float => false,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => false,
            BuiltInTypes::Function => false,
            BuiltInTypes::Closure => true,
            BuiltInTypes::Struct => true,
            BuiltInTypes::Array => true,
            BuiltInTypes::Null => false,
            BuiltInTypes::None => false,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            BuiltInTypes::Int => "Int".to_string(),
            BuiltInTypes::Float => "Float".to_string(),
            BuiltInTypes::String => "String".to_string(),
            BuiltInTypes::Bool => "Bool".to_string(),
            BuiltInTypes::Function => "Function".to_string(),
            BuiltInTypes::Closure => "Closure".to_string(),
            BuiltInTypes::Struct => "Struct".to_string(),
            BuiltInTypes::Array => "Array".to_string(),
            BuiltInTypes::Null => "Null".to_string(),
            BuiltInTypes::None => "None".to_string(),
        }
    }
}

/// Generic value representation for debugger display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Integer(u64),
    String(String),
    Address(u64),
}

impl Value {
    pub fn to_string(&self) -> String {
        match self {
            Value::Integer(i) => format!("{}", i),
            Value::String(s) => s.clone(),
            Value::Address(addr) => format!("0x{:x}", addr),
        }
    }
}

/// Memory location with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub address: u64,
    pub value: u64,
    pub kind: BuiltInTypes,
}

impl Memory {
    pub fn new(address: u64, value: u64) -> Self {
        Self {
            address,
            value,
            kind: BuiltInTypes::get_kind(value as usize),
        }
    }

    pub fn to_string(&self) -> String {
        format!("0x{:x}: 0x{:x} {:?}", self.address, self.value, self.kind)
    }
}

/// CPU register representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Register {
    pub name: String,
    pub value: Value,
    pub kind: BuiltInTypes,
}

impl Register {
    pub fn new(name: String, value: Value, kind: BuiltInTypes) -> Self {
        Self { name, value, kind }
    }

    pub fn to_string(&self) -> String {
        format!(
            "{}: {} - {}",
            self.name,
            self.value.to_string(),
            self.kind.to_string()
        )
    }
}

/// Disassembled instruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub address: u64,
    pub hex: String,
    pub mnemonic: String,
    pub operands: Vec<String>,
    pub comment: String,
}

impl Instruction {
    pub fn new(address: u64, mnemonic: String, operands: Vec<String>) -> Self {
        Self {
            address,
            hex: String::new(),
            mnemonic,
            operands,
            comment: String::new(),
        }
    }

    pub fn to_string(&self, show_address: bool, show_hex: bool) -> String {
        let mut result = String::new();
        if show_address {
            result.push_str(&format!("0x{:x} ", self.address));
        }
        if show_hex && !self.hex.is_empty() {
            result.push_str(&format!("{} ", self.hex));
        }
        result.push_str(&format!("{:8} ", self.mnemonic));
        for operand in &self.operands {
            result.push_str(&format!("{} ", operand));
        }
        if !self.comment.is_empty() {
            result.push_str(&format!("  ; {}", self.comment));
        }
        result
    }
}

/// Function metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Function {
    Foreign {
        name: String,
        pointer: usize,
    },
    Builtin {
        name: String,
        pointer: usize,
    },
    User {
        name: String,
        address_range: (usize, usize),
        number_of_arguments: usize,
    },
}

impl Function {
    pub fn get_name(&self) -> String {
        match self {
            Function::Foreign { name, .. } => name.clone(),
            Function::Builtin { name, .. } => name.clone(),
            Function::User { name, .. } => name.clone(),
        }
    }

    pub fn get_address(&self) -> usize {
        match self {
            Function::Foreign { pointer, .. } => *pointer,
            Function::Builtin { pointer, .. } => *pointer,
            Function::User { address_range, .. } => address_range.0,
        }
    }
}

/// Source code label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub label: String,
    pub function_pointer: usize,
    pub label_index: usize,
    pub label_location: usize,
}

impl Label {
    pub fn new(label: String, function_pointer: usize, label_index: usize, label_location: usize) -> Self {
        Self {
            label,
            function_pointer,
            label_index,
            label_location,
        }
    }
}