use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Stack map details for function analysis
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct StackMapDetails {
    pub function_name: Option<String>,
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

/// Main message data types for debugging communication - matches original exactly
#[derive(Debug, Encode, Decode, Clone, Serialize, Deserialize)]
pub enum MessageData {
    ForeignFunction {
        name: String,
        pointer: usize,
    },
    BuiltinFunction {
        name: String,
        pointer: usize,
    },
    HeapSegmentPointer {
        pointer: usize,
    },
    UserFunction {
        name: String,
        pointer: usize,
        len: usize,
        number_of_arguments: usize,
    },
    Label {
        label: String,
        function_pointer: usize,
        label_index: usize,
        label_location: usize,
    },
    StackMap {
        pc: usize,
        name: String,
        stack_map: Vec<(usize, StackMapDetails)>,
    },
    Allocate {
        bytes: usize,
        stack_pointer: usize,
        kind: String,
    },
    Tokens {
        file_name: String,
        tokens: Vec<String>,
        token_line_column_map: Vec<(usize, usize)>,
    },
    Ir {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        token_range_to_ir_range: Vec<((usize, usize), (usize, usize))>,
    },
    Arm {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        ir_to_machine_code_range: Vec<(usize, (usize, usize))>,
    },
}

impl MessageData {
    pub fn to_display_string(&self) -> String {
        match self {
            MessageData::ForeignFunction { name, pointer } => {
                format!("{}: 0x{:x}", name, pointer)
            }
            MessageData::BuiltinFunction { name, pointer } => {
                format!("{}: 0x{:x}", name, pointer)
            }
            MessageData::HeapSegmentPointer { pointer } => {
                format!("0x{:x}", pointer)
            }
            MessageData::UserFunction {
                name,
                pointer,
                len,
                number_of_arguments: _,
            } => {
                format!("{}: 0x{:x} 0x{:x}", name, pointer, (pointer + len))
            }
            MessageData::Label {
                label,
                function_pointer,
                label_index,
                label_location,
            } => {
                format!(
                    "{}: 0x{:x} 0x{:x} 0x{:x}",
                    label, function_pointer, label_index, label_location
                )
            }
            MessageData::StackMap {
                pc,
                name,
                stack_map,
            } => {
                let stack_map_details_string = stack_map
                    .iter()
                    .map(|(key, details)| {
                        format!(
                            "0x{:x}: size: {}, locals: {}",
                            key, details.current_stack_size, details.number_of_locals
                        )
                    })
                    .collect::<Vec<String>>()
                    .join(" | ");
                format!("{}, 0x{:x}, {}", name, pc, stack_map_details_string)
            }
            MessageData::Allocate {
                bytes,
                stack_pointer,
                kind,
            } => {
                format!("{}: {} 0x{:x}", kind, bytes, stack_pointer)
            }
            MessageData::Tokens {
                file_name,
                tokens,
                token_line_column_map,
            } => {
                let tokens = tokens.join(" ");
                let token_line_column_map = token_line_column_map
                    .iter()
                    .map(|(line, column)| format!("{}:{}", line, column))
                    .collect::<Vec<String>>()
                    .join(" ");
                format!("{}: {} {}", file_name, tokens, token_line_column_map)
            }
            MessageData::Ir {
                function_pointer: _,
                file_name,
                instructions,
                token_range_to_ir_range,
            } => {
                let instructions = instructions.join(" ");
                let token_range_to_ir_range = token_range_to_ir_range
                    .iter()
                    .map(|((start, end), (start_ir, end_ir))| {
                        format!("{}-{}:{}-{}", start, end, start_ir, end_ir)
                    })
                    .collect::<Vec<String>>()
                    .join(" ");
                format!(
                    "{}: {} {}",
                    file_name, instructions, token_range_to_ir_range
                )
            }
            MessageData::Arm {
                function_pointer: _,
                file_name,
                instructions,
                ir_to_machine_code_range,
            } => {
                let instructions = instructions.join(" ");
                let ir_to_machine_code_range = ir_to_machine_code_range
                    .iter()
                    .map(|(ir, (start, end))| format!("{}:{}-{}", ir, start, end))
                    .collect::<Vec<String>>()
                    .join(" ");
                format!(
                    "{}: {} {}",
                    file_name, instructions, ir_to_machine_code_range
                )
            }
        }
    }
}

/// Complete debugging message with metadata
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct DebugMessage {
    pub kind: String,
    pub data: MessageData,
}

impl DebugMessage {
    pub fn new(kind: String, data: MessageData) -> Self {
        Self { kind, data }
    }

    pub fn to_string(&self) -> String {
        format!("{} {}", self.kind, self.data.to_display_string())
    }
}

/// Message wrapper for binary deserialization - matches original exactly
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Message {
    pub kind: String,
    pub data: MessageData,
}

impl Message {
    pub fn new(kind: String, data: MessageData) -> Self {
        Self { kind, data }
    }

    pub fn from_binary(buffer: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        bincode::decode_from_slice(buffer, bincode::config::standard()).map(|(msg, _)| msg)
    }

    pub fn to_binary(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::encode_to_vec(self, bincode::config::standard())
    }
}