// Converters from other IR formats to Universal format
use super::schema::{UniversalIR, UniversalBlock, UniversalInstruction, UNIVERSAL_VERSION};
use crate::compilers::ion::schema::{IonJSON, Pass};
use std::collections::HashMap;

/// Convert Ion JSON to Universal format
pub fn ion_to_universal(ion: &IonJSON) -> UniversalIR {
    let mut blocks = Vec::new();

    // Extract metadata from Ion
    let mut metadata = HashMap::new();
    metadata.insert(
        "version".to_string(),
        serde_json::Value::Number(ion.version.into()),
    );

    if let Some(ref func_name) = ion.function {
        metadata.insert("name".to_string(), serde_json::Value::String(func_name.clone()));
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
                                        meta.insert(
                                            "id".to_string(),
                                            serde_json::Value::Number(ins.id.into()),
                                        );
                                        meta.insert(
                                            "ptr".to_string(),
                                            serde_json::Value::Number(ins.ptr.0.into()),
                                        );
                                        if !ins.inputs.is_empty() {
                                            meta.insert(
                                                "inputs".to_string(),
                                                serde_json::json!(ins.inputs),
                                            );
                                        }
                                        if !ins.uses.is_empty() {
                                            meta.insert(
                                                "uses".to_string(),
                                                serde_json::json!(ins.uses),
                                            );
                                        }
                                        meta
                                    },
                                })
                                .collect(),
                            metadata: {
                                let mut meta = HashMap::new();
                                meta.insert(
                                    "ptr".to_string(),
                                    serde_json::Value::Number(mir_block.ptr.0.into()),
                                );
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
                                    meta.insert(
                                        "id".to_string(),
                                        serde_json::Value::Number(ins.id.into()),
                                    );
                                    meta.insert(
                                        "ptr".to_string(),
                                        serde_json::Value::Number(ins.ptr.0.into()),
                                    );
                                    if !ins.inputs.is_empty() {
                                        meta.insert(
                                            "inputs".to_string(),
                                            serde_json::json!(ins.inputs),
                                        );
                                    }
                                    if !ins.uses.is_empty() {
                                        meta.insert("uses".to_string(), serde_json::json!(ins.uses));
                                    }
                                    meta
                                },
                            })
                            .collect(),
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert(
                                "ptr".to_string(),
                                serde_json::Value::Number(mir_block.ptr.0.into()),
                            );
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

    metadata.insert("name".to_string(), serde_json::Value::String(function_name.to_string()));
    metadata.insert("pass".to_string(), serde_json::Value::String(pass.name.clone()));

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
                            meta.insert(
                                "id".to_string(),
                                serde_json::Value::Number(ins.id.into()),
                            );
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

