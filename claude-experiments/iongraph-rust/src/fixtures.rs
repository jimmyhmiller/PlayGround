use crate::types::*;

pub fn create_mock_mir_block(overrides: Option<MIRBlockOverrides>) -> MIRBlock {
    let defaults = MIRBlock {
        id: BlockID(0),
        number: BlockNumber(0),
        loop_depth: 0,
        attributes: vec![],
        predecessors: vec![],
        successors: vec![],
        instructions: vec![
            MIRInstruction {
                id: 1,
                opcode: "Parameter".to_string(),
                attributes: vec![],
                inputs: vec![],
                uses: vec![],
                mem_inputs: vec![],
                instruction_type: "Int32".to_string(),
            },
            MIRInstruction {
                id: 2,
                opcode: "Return".to_string(),
                attributes: vec![],
                inputs: vec![1],
                uses: vec![1],
                mem_inputs: vec![],
                instruction_type: "None".to_string(),
            },
        ],
    };

    if let Some(overrides) = overrides {
        apply_mir_overrides(defaults, overrides)
    } else {
        defaults
    }
}

pub fn create_mock_lir_block(overrides: Option<LIRBlockOverrides>) -> LIRBlock {
    let defaults = LIRBlock {
        id: BlockID(0),
        number: BlockNumber(0),
        instructions: vec![
            LIRInstruction {
                id: 1,
                opcode: "move32 %eax, %r0".to_string(),
                defs: vec![0],
            },
            LIRInstruction {
                id: 2,
                opcode: "ret".to_string(),
                defs: vec![],
            },
        ],
    };

    if let Some(overrides) = overrides {
        apply_lir_overrides(defaults, overrides)
    } else {
        defaults
    }
}

// EXACT copies from TypeScript original
pub fn create_simple_pass() -> Pass {
    Pass {
        name: "TestPass".to_string(),
        mir: MIRData {
            blocks: vec![
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(0)),
                    number: Some(BlockNumber(0)),
                    successors: Some(vec![BlockNumber(1)]),
                    ..Default::default()
                })),
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(1)),
                    number: Some(BlockNumber(1)),
                    predecessors: Some(vec![BlockNumber(0)]),
                    ..Default::default()
                })),
            ],
        },
        lir: LIRData {
            blocks: vec![
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(0)),
                    number: Some(BlockNumber(0)),
                    ..Default::default()
                })),
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(1)),
                    number: Some(BlockNumber(1)),
                    ..Default::default()
                })),
            ],
        },
    }
}

pub fn create_loop_pass() -> Pass {
    Pass {
        name: "LoopPass".to_string(),
        mir: MIRData {
            blocks: vec![
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(0)),
                    number: Some(BlockNumber(0)),
                    successors: Some(vec![BlockNumber(1)]),
                    loop_depth: Some(0),
                    ..Default::default()
                })),
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(1)),
                    number: Some(BlockNumber(1)),
                    predecessors: Some(vec![BlockNumber(0), BlockNumber(2)]),
                    successors: Some(vec![BlockNumber(2)]),
                    loop_depth: Some(1),
                    attributes: Some(vec!["loopheader".to_string()]),
                    ..Default::default()
                })),
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(2)),
                    number: Some(BlockNumber(2)),
                    predecessors: Some(vec![BlockNumber(1)]),
                    successors: Some(vec![BlockNumber(1)]),
                    loop_depth: Some(1),
                    attributes: Some(vec!["backedge".to_string()]),
                    ..Default::default()
                })),
            ],
        },
        lir: LIRData {
            blocks: vec![
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(0)),
                    number: Some(BlockNumber(0)),
                    ..Default::default()
                })),
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(1)),
                    number: Some(BlockNumber(1)),
                    ..Default::default()
                })),
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(2)),
                    number: Some(BlockNumber(2)),
                    ..Default::default()
                })),
            ],
        },
    }
}

pub fn create_complex_pass() -> Pass {
    Pass {
        name: "ComplexPass".to_string(),
        mir: MIRData {
            blocks: vec![
                // Entry block
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(0)),
                    number: Some(BlockNumber(0)),
                    successors: Some(vec![BlockNumber(1)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 1,
                            opcode: "Parameter".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "Int32".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                // Conditional branch
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(1)),
                    number: Some(BlockNumber(1)),
                    predecessors: Some(vec![BlockNumber(0)]),
                    successors: Some(vec![BlockNumber(2), BlockNumber(3)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 2,
                            opcode: "Compare".to_string(),
                            attributes: vec![],
                            inputs: vec![1],
                            uses: vec![1],
                            mem_inputs: vec![],
                            instruction_type: "Bool".to_string(),
                        },
                        MIRInstruction {
                            id: 3,
                            opcode: "Branch".to_string(),
                            attributes: vec![],
                            inputs: vec![2],
                            uses: vec![2],
                            mem_inputs: vec![],
                            instruction_type: "None".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                // True branch
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(2)),
                    number: Some(BlockNumber(2)),
                    predecessors: Some(vec![BlockNumber(1)]),
                    successors: Some(vec![BlockNumber(4)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 4,
                            opcode: "Add".to_string(),
                            attributes: vec![],
                            inputs: vec![1, 1],
                            uses: vec![1, 1],
                            mem_inputs: vec![],
                            instruction_type: "Int32".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                // False branch
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(3)),
                    number: Some(BlockNumber(3)),
                    predecessors: Some(vec![BlockNumber(1)]),
                    successors: Some(vec![BlockNumber(4)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 5,
                            opcode: "Sub".to_string(),
                            attributes: vec![],
                            inputs: vec![1, 1],
                            uses: vec![1, 1],
                            mem_inputs: vec![],
                            instruction_type: "Int32".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                // Exit block
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(4)),
                    number: Some(BlockNumber(4)),
                    predecessors: Some(vec![BlockNumber(2), BlockNumber(3)]),
                    successors: Some(vec![]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 6,
                            opcode: "Return".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "None".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
            ],
        },
        lir: LIRData {
            blocks: vec![
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(0)),
                    number: Some(BlockNumber(0)),
                    ..Default::default()
                })),
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(1)),
                    number: Some(BlockNumber(1)),
                    ..Default::default()
                })),
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(2)),
                    number: Some(BlockNumber(2)),
                    ..Default::default()
                })),
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(3)),
                    number: Some(BlockNumber(3)),
                    ..Default::default()
                })),
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(4)),
                    number: Some(BlockNumber(4)),
                    ..Default::default()
                })),
            ],
        },
    }
}

pub fn create_switch_like_pass() -> Pass {
    Pass {
        name: "SwitchLike".to_string(),
        mir: MIRData {
            blocks: vec![
                // Switch block with many successors
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(0)),
                    number: Some(BlockNumber(0)),
                    successors: Some(vec![
                        BlockNumber(1), BlockNumber(2), BlockNumber(3), BlockNumber(4), BlockNumber(5)
                    ]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 1,
                            opcode: "Switch".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "None".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                // Case blocks (multiple branches)
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(1)),
                    number: Some(BlockNumber(1)),
                    predecessors: Some(vec![BlockNumber(0)]),
                    successors: Some(vec![BlockNumber(6)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 2,
                            opcode: "Case0".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "Int32".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(2)),
                    number: Some(BlockNumber(2)),
                    predecessors: Some(vec![BlockNumber(0)]),
                    successors: Some(vec![BlockNumber(6)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 3,
                            opcode: "Case1".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "Int32".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(3)),
                    number: Some(BlockNumber(3)),
                    predecessors: Some(vec![BlockNumber(0)]),
                    successors: Some(vec![BlockNumber(6)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 4,
                            opcode: "Case2".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "Int32".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(4)),
                    number: Some(BlockNumber(4)),
                    predecessors: Some(vec![BlockNumber(0)]),
                    successors: Some(vec![BlockNumber(6)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 5,
                            opcode: "Case3".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "Int32".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(5)),
                    number: Some(BlockNumber(5)),
                    predecessors: Some(vec![BlockNumber(0)]),
                    successors: Some(vec![BlockNumber(6)]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 6,
                            opcode: "Case4".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "Int32".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
                // Merge block
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(6)),
                    number: Some(BlockNumber(6)),
                    predecessors: Some(vec![
                        BlockNumber(1), BlockNumber(2), BlockNumber(3), BlockNumber(4), BlockNumber(5)
                    ]),
                    loop_depth: Some(0),
                    instructions: Some(vec![
                        MIRInstruction {
                            id: 7,
                            opcode: "Merge".to_string(),
                            attributes: vec![],
                            inputs: vec![],
                            uses: vec![],
                            mem_inputs: vec![],
                            instruction_type: "None".to_string(),
                        },
                    ]),
                    ..Default::default()
                })),
            ],
        },
        lir: LIRData {
            blocks: (0..7).map(|i| {
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(i)),
                    number: Some(BlockNumber(i)),
                    ..Default::default()
                }))
            }).collect(),
        },
    }
}

pub fn create_deep_loop_pass() -> Pass {
    Pass {
        name: "DeepLoops".to_string(),
        mir: MIRData {
            blocks: vec![
                // Entry
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(0)),
                    number: Some(BlockNumber(0)),
                    successors: Some(vec![BlockNumber(1)]),
                    loop_depth: Some(0),
                    ..Default::default()
                })),
                // Loop level 1
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(1)),
                    number: Some(BlockNumber(1)),
                    predecessors: Some(vec![BlockNumber(0), BlockNumber(5)]),
                    successors: Some(vec![BlockNumber(2)]),
                    loop_depth: Some(1),
                    attributes: Some(vec!["loopheader".to_string()]),
                    ..Default::default()
                })),
                // Loop level 2
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(2)),
                    number: Some(BlockNumber(2)),
                    predecessors: Some(vec![BlockNumber(1), BlockNumber(4)]),
                    successors: Some(vec![BlockNumber(3)]),
                    loop_depth: Some(2),
                    attributes: Some(vec!["loopheader".to_string()]),
                    ..Default::default()
                })),
                // Loop level 3 (deepest)
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(3)),
                    number: Some(BlockNumber(3)),
                    predecessors: Some(vec![BlockNumber(2)]),
                    successors: Some(vec![BlockNumber(4), BlockNumber(6)]),
                    loop_depth: Some(3),
                    attributes: Some(vec!["loopheader".to_string()]),
                    ..Default::default()
                })),
                // Backedge to level 2
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(4)),
                    number: Some(BlockNumber(4)),
                    predecessors: Some(vec![BlockNumber(3)]),
                    successors: Some(vec![BlockNumber(2)]),
                    loop_depth: Some(2),
                    attributes: Some(vec!["backedge".to_string()]),
                    ..Default::default()
                })),
                // Exit from inner loop to outer loop body  
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(5)),
                    number: Some(BlockNumber(5)),
                    predecessors: Some(vec![BlockNumber(3)]),
                    successors: Some(vec![BlockNumber(1)]),
                    loop_depth: Some(1),
                    attributes: Some(vec!["backedge".to_string()]),
                    ..Default::default()
                })),
                // Exit
                create_mock_mir_block(Some(MIRBlockOverrides {
                    id: Some(BlockID(6)),
                    number: Some(BlockNumber(6)),
                    predecessors: Some(vec![BlockNumber(3)]),
                    loop_depth: Some(0),
                    ..Default::default()
                })),
            ],
        },
        lir: LIRData {
            blocks: (0..7).map(|i| {
                create_mock_lir_block(Some(LIRBlockOverrides {
                    id: Some(BlockID(i)),
                    number: Some(BlockNumber(i)),
                    ..Default::default()
                }))
            }).collect(),
        },
    }
}

#[derive(Default)]
pub struct MIRBlockOverrides {
    pub id: Option<BlockID>,
    pub number: Option<BlockNumber>,
    pub loop_depth: Option<u32>,
    pub attributes: Option<Vec<String>>,
    pub predecessors: Option<Vec<BlockNumber>>,
    pub successors: Option<Vec<BlockNumber>>,
    pub instructions: Option<Vec<MIRInstruction>>,
}

#[derive(Default)]
pub struct LIRBlockOverrides {
    pub id: Option<BlockID>,
    pub number: Option<BlockNumber>,
    pub instructions: Option<Vec<LIRInstruction>>,
}

fn apply_mir_overrides(mut block: MIRBlock, overrides: MIRBlockOverrides) -> MIRBlock {
    if let Some(id) = overrides.id { block.id = id; }
    if let Some(number) = overrides.number { block.number = number; }
    if let Some(loop_depth) = overrides.loop_depth { block.loop_depth = loop_depth; }
    if let Some(attributes) = overrides.attributes { block.attributes = attributes; }
    if let Some(predecessors) = overrides.predecessors { block.predecessors = predecessors; }
    if let Some(successors) = overrides.successors { block.successors = successors; }
    if let Some(instructions) = overrides.instructions { block.instructions = instructions; }
    block
}

fn apply_lir_overrides(mut block: LIRBlock, overrides: LIRBlockOverrides) -> LIRBlock {
    if let Some(id) = overrides.id { block.id = id; }
    if let Some(number) = overrides.number { block.number = number; }
    if let Some(instructions) = overrides.instructions { block.instructions = instructions; }
    block
}