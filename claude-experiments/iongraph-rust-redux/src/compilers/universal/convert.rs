// Converters from other IR formats to Universal format
use super::schema::{UniversalBlock, UniversalInstruction, UniversalIR, UNIVERSAL_VERSION};
use crate::compilers::ion::schema::{IonJSON, Pass};
use crate::json_compat::Value;
use std::collections::HashMap;

/// Helper to create a Value from a number
fn num_value(n: u32) -> Value {
    #[cfg(feature = "serde")]
    {
        serde_json::Value::Number(n.into())
    }
    #[cfg(not(feature = "serde"))]
    {
        Value::from(n)
    }
}

/// Helper to create a Value from a string
fn str_value(s: impl Into<String>) -> Value {
    #[cfg(feature = "serde")]
    {
        serde_json::Value::String(s.into())
    }
    #[cfg(not(feature = "serde"))]
    {
        Value::String(s.into())
    }
}

/// Helper to convert a Vec<u32> to a Value array
fn vec_value(v: &[u32]) -> Value {
    #[cfg(feature = "serde")]
    {
        serde_json::json!(v)
    }
    #[cfg(not(feature = "serde"))]
    {
        Value::Array(v.iter().map(|n| Value::from(*n)).collect())
    }
}

/// Convert Ion JSON to Universal format
pub fn ion_to_universal(ion: &IonJSON) -> UniversalIR {
    let mut blocks = Vec::new();

    // Extract metadata from Ion
    let mut metadata = HashMap::new();
    metadata.insert("version".to_string(), num_value(ion.version));

    if let Some(ref func_name) = ion.function {
        metadata.insert("name".to_string(), str_value(func_name.clone()));
    }

    // Convert all blocks from all functions and passes
    if !ion.functions.is_empty() {
        // New format: multiple functions
        for func in &ion.functions {
            for pass in &func.passes {
                if let Some(ref mir) = pass.mir {
                    for mir_block in &mir.blocks {
                        let block = UniversalBlock {
                            id: format!("{}", mir_block.id),
                            attributes: mir_block.attributes.clone(),
                            loop_depth: mir_block.loop_depth,
                            predecessors: mir_block
                                .predecessors
                                .iter()
                                .map(|p| format!("{}", p.0))
                                .collect(),
                            successors: mir_block
                                .successors
                                .iter()
                                .map(|s| format!("{}", s.0))
                                .collect(),
                            back_edges: Vec::new(), // Ion format doesn't have separate back edges
                            has_self_loop: false,   // Ion format doesn't track self-loops
                            instructions: mir_block
                                .instructions
                                .iter()
                                .map(|ins| UniversalInstruction {
                                    opcode: ins.opcode.clone(),
                                    attributes: ins.attributes.clone().unwrap_or_default(),
                                    type_: ins.type_.clone(),
                                    profiling: None,
                                    metadata: {
                                        let mut meta = HashMap::new();
                                        meta.insert("id".to_string(), num_value(ins.id));
                                        meta.insert("ptr".to_string(), num_value(ins.ptr.0));
                                        if !ins.inputs.is_empty() {
                                            meta.insert("inputs".to_string(), vec_value(&ins.inputs));
                                        }
                                        if !ins.uses.is_empty() {
                                            meta.insert("uses".to_string(), vec_value(&ins.uses));
                                        }
                                        meta
                                    },
                                })
                                .collect(),
                            metadata: {
                                let mut meta = HashMap::new();
                                meta.insert("ptr".to_string(), num_value(mir_block.ptr.0));
                                if mir_block.resume_point.is_some() {
                                    meta.insert(
                                        "resumePoint".to_string(),
                                        mir_block.resume_point.clone().unwrap(),
                                    );
                                }
                                meta
                            },
                        };
                        blocks.push(block);
                    }
                }

                // TODO: Handle LIR blocks similarly
            }
        }
    } else if !ion.passes.is_empty() {
        // Old format: single function
        for pass in &ion.passes {
            if let Some(ref mir) = pass.mir {
                for mir_block in &mir.blocks {
                    let block = UniversalBlock {
                        id: format!("{}", mir_block.id),
                        attributes: mir_block.attributes.clone(),
                        loop_depth: mir_block.loop_depth,
                        predecessors: mir_block
                            .predecessors
                            .iter()
                            .map(|p| format!("{}", p.0))
                            .collect(),
                        successors: mir_block
                            .successors
                            .iter()
                            .map(|s| format!("{}", s.0))
                            .collect(),
                        back_edges: Vec::new(), // Ion format doesn't have separate back edges
                        has_self_loop: false,   // Ion format doesn't track self-loops
                        instructions: mir_block
                            .instructions
                            .iter()
                            .map(|ins| UniversalInstruction {
                                opcode: ins.opcode.clone(),
                                attributes: ins.attributes.clone().unwrap_or_default(),
                                type_: ins.type_.clone(),
                                profiling: None,
                                metadata: {
                                    let mut meta = HashMap::new();
                                    meta.insert("id".to_string(), num_value(ins.id));
                                    meta.insert("ptr".to_string(), num_value(ins.ptr.0));
                                    if !ins.inputs.is_empty() {
                                        meta.insert("inputs".to_string(), vec_value(&ins.inputs));
                                    }
                                    if !ins.uses.is_empty() {
                                        meta.insert("uses".to_string(), vec_value(&ins.uses));
                                    }
                                    meta
                                },
                            })
                            .collect(),
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("ptr".to_string(), num_value(mir_block.ptr.0));
                            if mir_block.resume_point.is_some() {
                                meta.insert(
                                    "resumePoint".to_string(),
                                    mir_block.resume_point.clone().unwrap(),
                                );
                            }
                            meta
                        },
                    };
                    blocks.push(block);
                }
            }
        }
    }

    UniversalIR {
        format: UNIVERSAL_VERSION.to_string(),
        compiler: "ion".to_string(),
        metadata,
        blocks,
    }
}

/// Convert a single Pass to Universal format
pub fn pass_to_universal(pass: &Pass, function_name: &str) -> UniversalIR {
    let mut blocks = Vec::new();
    let mut metadata = HashMap::new();

    metadata.insert("name".to_string(), str_value(function_name.to_string()));
    metadata.insert("pass".to_string(), str_value(pass.name.clone()));

    if let Some(ref mir) = pass.mir {
        for mir_block in &mir.blocks {
            let block = UniversalBlock {
                id: format!("{}", mir_block.id),
                attributes: mir_block.attributes.clone(),
                loop_depth: mir_block.loop_depth,
                predecessors: mir_block
                    .predecessors
                    .iter()
                    .map(|p| format!("{}", p.0))
                    .collect(),
                successors: mir_block
                    .successors
                    .iter()
                    .map(|s| format!("{}", s.0))
                    .collect(),
                back_edges: Vec::new(), // Ion format doesn't have separate back edges
                has_self_loop: false,   // Ion format doesn't track self-loops
                instructions: mir_block
                    .instructions
                    .iter()
                    .map(|ins| UniversalInstruction {
                        opcode: ins.opcode.clone(),
                        attributes: ins.attributes.clone().unwrap_or_default(),
                        type_: ins.type_.clone(),
                        profiling: None,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("id".to_string(), num_value(ins.id));
                            meta
                        },
                    })
                    .collect(),
                metadata: HashMap::new(),
            };
            blocks.push(block);
        }
    }

    UniversalIR {
        format: UNIVERSAL_VERSION.to_string(),
        compiler: "ion".to_string(),
        metadata,
        blocks,
    }
}
