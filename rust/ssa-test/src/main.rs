use std::collections::{HashMap, HashSet};

use crate::ast::Ast;
use crate::instruction::{Block, BlockId, Instruction, Phi, PhiId, PhiReference, Value, Variable};
mod ast;
mod instruction;
mod syntax;
mod visualizer;
mod ssa_properties;

pub struct SSATranslator {
    pub definition: HashMap<String, HashMap<BlockId, Value>>,
    pub sealed_blocks: HashSet<BlockId>,
    pub incomplete_phis: HashMap<BlockId, HashMap<String, PhiId>>,
    pub phis: HashMap<PhiId, Phi>,
    pub blocks: Vec<Block>,
    pub next_variable_id: usize,
    pub next_phi_id: usize,
    pub current_block: BlockId, 
}

impl Default for SSATranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl SSATranslator {
    pub fn new() -> Self {
        let mut translator = SSATranslator {
            definition: HashMap::new(),
            sealed_blocks: HashSet::new(),
            incomplete_phis: HashMap::new(),
            phis: HashMap::new(),
            blocks: Vec::new(),
            next_variable_id: 0,
            next_phi_id: 0,
            current_block: BlockId(0), 
        };
        
        let initial_block = translator.create_block();
        translator.current_block = initial_block;
        translator
    }

    fn write_variable(&mut self, variable: String, block_id: BlockId, value: Value) {
        self.definition
            .entry(variable)
            .or_default()
            .insert(block_id, value);
    }

    fn read_variable(&mut self, variable: String, block_id: BlockId) -> Value {
        if self.definition.contains_key(&variable) {
            if let Some(value) = self
                .definition
                .get(&variable)
                .and_then(|v| v.get(&block_id))
            {
                return value.clone();
            }
        }
        self.read_variable_recursively(variable, block_id)
    }

    fn read_variable_recursively(&mut self, variable: String, block_id: BlockId) -> Value {
        let value: Value;
        if !self.sealed_blocks.contains(&block_id) {
            // Block not sealed - create incomplete phi and materialize to variable
            let phi_id = self.create_phi(block_id);
            let temp_var = self.materialize_phi(phi_id, block_id);
            value = Value::Var(temp_var);
            self.incomplete_phis
                .entry(block_id)
                .or_default()
                .insert(variable.clone(), phi_id);
        } else {
            let block = self
                .blocks
                .get(block_id.0)
                .expect("Block not found");
            if block.predecessors.len() == 1 {
                value = self.read_variable(variable.clone(), block.predecessors[0]);
            } else {
                // Multiple predecessors - create phi and materialize to variable
                let phi_id = self.create_phi(block_id);
                let temp_var = self.materialize_phi(phi_id, block_id);
                let var_value = Value::Var(temp_var.clone());
                self.write_variable(variable.clone(), block_id, var_value.clone());
                let result = self.add_phi_operands(&variable, phi_id);

                // If the phi was removed as trivial, use the replacement value directly
                // and clean up the unnecessary copy instruction
                if !self.phis.contains_key(&phi_id) {
                    // Phi was removed - update definition and remove the copy instruction
                    self.write_variable(variable.clone(), block_id, result.clone());
                    // Remove the materialized instruction (vN := replacement)
                    self.remove_copy_instruction(block_id, &temp_var);
                    value = result;
                } else {
                    value = var_value;
                }
            }
        }
        self.write_variable(variable.clone(), block_id, value.clone());
        value
    }

    /// Materialize a phi to a variable by creating an Assign instruction at block start
    fn materialize_phi(&mut self, phi_id: PhiId, block_id: BlockId) -> Variable {
        let temp_var = self.get_temp_variable("phi");
        // Insert phi assignment at the beginning of the block, after any existing phi assignments
        let insert_pos = self.find_phi_insert_position(block_id);
        self.blocks[block_id.0].instructions.insert(insert_pos, Instruction::Assign {
            dest: temp_var.clone(),
            value: Value::Phi(phi_id),
        });
        temp_var
    }

    /// Find the position to insert a new phi assignment (after existing phi assigns)
    fn find_phi_insert_position(&self, block_id: BlockId) -> usize {
        let block = &self.blocks[block_id.0];
        for (i, instr) in block.instructions.iter().enumerate() {
            if !matches!(instr, Instruction::Assign { value: Value::Phi(_), .. }) {
                return i;
            }
        }
        block.instructions.len()
    }

    /// Remove a copy instruction (dest := something) from a block
    fn remove_copy_instruction(&mut self, block_id: BlockId, dest: &Variable) {
        let block = &mut self.blocks[block_id.0];
        block.instructions.retain(|instr| {
            !matches!(instr, Instruction::Assign { dest: d, .. } if d == dest)
        });
    }

    fn add_phi_operands(&mut self, variable: &String, phi_id: PhiId) -> Value {
        // Check if phi still exists (might have been removed as trivial)
        let block_id = match self.phis.get(&phi_id) {
            Some(phi) => phi.block_id,
            None => {
                // Phi was already removed, return current definition
                if let Some(block_defs) = self.definition.get(variable) {
                    // Find a valid definition - this is a fallback
                    for (_, value) in block_defs {
                        return value.clone();
                    }
                }
                return Value::Undefined;
            }
        };

        // Check if this phi already has operands
        if let Some(phi) = self.phis.get(&phi_id) {
            if !phi.operands.is_empty() {
                return Value::Phi(phi_id);
            }
        }

        let block = self
            .blocks
            .get(block_id.0)
            .expect("Block not found");

        for predecessor in block.predecessors.clone() {
            let value = self.read_variable(variable.clone(), predecessor);
            if let Some(phi) = self.phis.get_mut(&phi_id) {
                phi.operands.push(value.clone());
            }
            if let Value::Phi(id) = value {
                self.add_phi_phi_use(phi_id, id);
            }
        }
        // Don't try to remove trivial phi immediately - wait until all phis are resolved
        return self.try_remove_trivial_phi(phi_id);
    }

    fn try_remove_trivial_phi(&mut self, phi_id: PhiId) -> Value {
        let phi = match self.phis.get(&phi_id) {
            Some(p) => p.clone(),
            None => return Value::Undefined, // Phi was already removed
        };

        // Don't try to remove incomplete phis - they haven't had operands filled in yet
        let is_incomplete = self.incomplete_phis.values()
            .any(|block_phis| block_phis.values().any(|&id| id == phi_id));
        if is_incomplete {
            return Value::Phi(phi_id);
        }

        // Don't try to remove phis with no operands - they're not yet resolved
        if phi.operands.is_empty() {
            return Value::Phi(phi_id);
        }

        let mut same: Option<Value> = None;

        for op in phi.operands.iter() {
            // Skip self-references and duplicates
            if Some(op) == same.as_ref() || op == &Value::Phi(phi_id) {
                continue;
            }
            // If we already found one operand and this is different, keep the phi
            if same.is_some() {
                return Value::Phi(phi_id);
            }
            same = Some(op.clone());
        }

        // Determine replacement value
        let replacement = if same.is_none() {
            Value::Undefined
        } else {
            same.unwrap()
        };

        // Remove the trivial phi from the map BEFORE replacing uses
        // This prevents infinite recursion
        self.phis.remove(&phi_id);

        // Replace all uses of this phi with the replacement
        // Use the cloned phi's uses since we already removed the phi from the map
        self.replace_phi_uses_with_list(&phi.uses, phi_id, replacement.clone());

        // Update the definition map - any variable defined as this phi should now use replacement
        let block_id = phi.block_id;
        for (_var_name, block_defs) in self.definition.iter_mut() {
            if let Some(value) = block_defs.get_mut(&block_id) {
                if value == &Value::Phi(phi_id) {
                    *value = replacement.clone();
                }
            }
        }

        // Collect phi users to process (avoiding borrow issues)
        let phi_users: Vec<PhiId> = phi.uses.iter()
            .filter_map(|user| {
                if let PhiReference::Phi(user_phi_id) = user {
                    // Don't process self or already-removed phis
                    if *user_phi_id != phi_id && self.phis.contains_key(user_phi_id) {
                        return Some(*user_phi_id);
                    }
                }
                None
            })
            .collect();

        // Try to recursively remove all phi users, which might have become trivial
        for user_phi_id in phi_users {
            self.try_remove_trivial_phi(user_phi_id);
        }

        replacement
    }

    fn seal_block(&mut self, block_id: BlockId) {
        self.sealed_blocks.insert(block_id);
        self.blocks[block_id.0].seal();

        let predecessor_count = self.blocks[block_id.0].predecessors.len();

        if let Some(phis) = self.incomplete_phis.remove(&block_id) {
            for (variable, phi_id) in phis {
                if predecessor_count == 1 {
                    // Single predecessor: no phi needed, just read from predecessor
                    let pred = self.blocks[block_id.0].predecessors[0];
                    let value = self.read_variable(variable.clone(), pred);

                    // Only update definition if there isn't already a local definition
                    // (the incomplete phi was for a READ, not a WRITE - don't overwrite writes)
                    let has_local_def = self.definition
                        .get(&variable)
                        .and_then(|m| m.get(&block_id))
                        .map(|v| v != &Value::Phi(phi_id))
                        .unwrap_or(false);

                    if !has_local_def {
                        self.write_variable(variable.clone(), block_id, value.clone());
                    }

                    // Replace uses of this phi with the value (in instructions)
                    if let Some(phi) = self.phis.get(&phi_id).cloned() {
                        self.replace_phi_uses_with_list(&phi.uses, phi_id, value.clone());
                    }

                    // Remove the unnecessary phi
                    self.phis.remove(&phi_id);
                } else {
                    // Multiple predecessors: need to add operands and potentially simplify
                    self.add_phi_operands(&variable, phi_id);
                }
            }
        }
    }

    fn get_temp_variable(&mut self, _prefix: &str) -> Variable {
        let variable = Variable(format!("v{}", self.next_variable_id));
        self.next_variable_id += 1;
        variable
    }

    fn create_phi(&mut self, block_id: BlockId) -> PhiId {
        let phi_id = PhiId(self.next_phi_id);
        self.next_phi_id += 1;
        
        
        let phi = Phi {
            id: phi_id,
            block_id,
            operands: Vec::new(),
            uses: Vec::new(),
        };
        
        self.phis.insert(phi_id, phi);
        let instruction_offset = self.blocks[block_id.0].instructions.len();
        self.add_phi_use(phi_id, block_id, instruction_offset);
        phi_id
    }

    fn add_phi_use(&mut self, phi_id: PhiId, block_id: BlockId, instruction_offset: usize) {
        if let Some(phi) = self.phis.get_mut(&phi_id) {
            phi.uses.push(PhiReference::Instruction {
                block_id,
                instruction_offset,
            });
        }
    }

    fn add_phi_phi_use(&mut self, phi_id: PhiId, user_phi_id: PhiId) {
        if let Some(phi) = self.phis.get_mut(&phi_id) {
            phi.uses.push(PhiReference::Phi(user_phi_id));
        }
    }

    fn replace_phi_uses_with_list(&mut self, uses: &[PhiReference], phi_id: PhiId, replacement: Value) {
        // Replace in tracked phi-to-phi uses
        for phi_ref in uses {
            match phi_ref {
                PhiReference::Instruction { block_id, instruction_offset } => {
                    self.replace_value_at_location(*block_id, *instruction_offset, phi_id, replacement.clone());
                }
                PhiReference::Phi(ph_id) => {
                    self.phis.get_mut(ph_id).map(|p| {
                        for operand in &mut p.operands {
                            if operand.is_same_phi(phi_id) {
                                *operand = replacement.clone();
                            }
                        }
                    });
                }
            }
        }

        // Also scan ALL instructions for this phi reference
        // This is needed because use tracking for incomplete phis may be incorrect
        self.replace_phi_in_all_instructions(phi_id, replacement);
    }

    fn replace_phi_in_all_instructions(&mut self, phi_id: PhiId, replacement: Value) {
        for block in &mut self.blocks {
            for instruction in &mut block.instructions {
                Self::replace_value_in_instruction_static(instruction, phi_id, replacement.clone());
            }
        }
    }

    fn replace_value_at_location(&mut self, block_id: BlockId, instruction_offset: usize, old_phi_id: PhiId, new_value: Value) {
        // First, replace in instructions
        if let Some(block) = self.blocks.get_mut(block_id.0) {
            if let Some(instruction) = block.instructions.get_mut(instruction_offset) {
                Self::replace_value_in_instruction_static(instruction, old_phi_id, new_value.clone());
            }
        }
        
        // Then, replace in other phis' operands
        for other_phi in self.phis.values_mut() {
            for operand in &mut other_phi.operands {
                if let Value::Phi(id) = operand {
                    if *id == old_phi_id {
                        *operand = new_value.clone();
                    }
                }
            }
        }
    }

    fn replace_value_in_instruction_static(instruction: &mut Instruction, old_phi_id: PhiId, new_value: Value) {
        match instruction {
            Instruction::Assign { value, .. } => {
                if let Value::Phi(id) = value {
                    if *id == old_phi_id {
                        *value = new_value;
                    }
                }
            }
            Instruction::BinaryOp { left, right, .. } => {
                if let Value::Phi(id) = left {
                    if *id == old_phi_id {
                        *left = new_value.clone();
                    }
                }
                if let Value::Phi(id) = right {
                    if *id == old_phi_id {
                        *right = new_value;
                    }
                }
            }
            Instruction::UnaryOp { operand, .. } => {
                if let Value::Phi(id) = operand {
                    if *id == old_phi_id {
                        *operand = new_value;
                    }
                }
            }
            Instruction::ConditionalJump { condition, .. } => {
                if let Value::Phi(id) = condition {
                    if *id == old_phi_id {
                        *condition = new_value;
                    }
                }
            }
            Instruction::Print { value } => {
                if let Value::Phi(id) = value {
                    if *id == old_phi_id {
                        *value = new_value;
                    }
                }
            }
            _ => {}
        }
    }

    fn translate(&mut self, ast: &Ast) -> Value {
        let current_block = self.current_block;
        match ast {
            Ast::Literal(value) => {
                let temp_var = self.get_temp_variable("lit");
                self.blocks[current_block.0].add_instruction(Instruction::Assign {
                    dest: temp_var.clone(),
                    value: Value::Literal(*value),
                });
                Value::Var(temp_var)
            }
            Ast::Variable(name) => {
                self.read_variable(name.clone(), current_block)
            }
            Ast::Block(statements) => {
                let mut value = Value::Undefined;
                for statement in statements {
                    value = self.translate(statement);
                }
                value
            }
            Ast::Assignment { variable, value } => {
                let rhs = self.translate(value);
                self.write_variable(variable.clone(), current_block, rhs.clone());
                rhs
            }
            Ast::BinaryOp { left, op, right } => {
                let left_value = self.translate(left);
                let right_value = self.translate(right);
                let dest_var = self.get_temp_variable(&format!("{:?}", op));
                let instruction = Instruction::BinaryOp {
                    dest: dest_var.clone(),
                    left: left_value,
                    op: op.clone(),
                    right: right_value,
                };
                self.blocks[current_block.0].add_instruction(instruction);
                Value::Var(dest_var)
            }
            Ast::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition_value = self.translate(condition);

                let then_block = self.create_block();
                let else_block = self.create_block();

                self.blocks[then_block.0].add_predecessor(current_block);
                self.blocks[else_block.0].add_predecessor(current_block);
                self.seal_block(current_block);
                self.seal_block(then_block);
                self.seal_block(else_block);

                self.blocks[current_block.0].add_instruction(Instruction::ConditionalJump {
                    condition: condition_value,
                    true_target: then_block,
                    false_target: else_block,
                });

                let merge_block = self.create_block();
                self.current_block = then_block;
                for stmt in then_branch {
                    self.translate(stmt);
                }

                self.blocks[merge_block.0]
                    .add_predecessor(self.current_block);

                self.blocks[self.current_block.0].add_instruction(Instruction::Jump {
                    target: merge_block,
                });

                self.current_block = else_block;
                if let Some(else_stmts) = else_branch {
                    for stmt in else_stmts {
                        self.translate(stmt);
                    }
                }
                self.blocks[merge_block.0]
                    .add_predecessor(self.current_block);

                self.blocks[self.current_block.0].add_instruction(Instruction::Jump {
                    target: merge_block,
                });

                self.seal_block(merge_block);

                self.current_block = merge_block;
                Value::Undefined
            }
            Ast::Print(value) => {
                let print_value = self.translate(value);
                let temp_var = self.get_temp_variable("print");
                self.blocks[self.current_block.0].add_instruction(Instruction::Assign {
                    dest: temp_var.clone(),
                    value: print_value,
                });
                self.blocks[self.current_block.0].add_instruction(Instruction::Print {
                    value: Value::Var(temp_var.clone()),
                });
                Value::Var(temp_var)
            }
            Ast::While { condition, body } => {
                let loop_start = self.create_block();
                let loop_body = self.create_block();
                let loop_end = self.create_block();

                // Add predecessor from entry to loop start
                self.blocks[loop_start.0].add_predecessor(current_block);
                
                // Jump to loop start
                self.blocks[current_block.0].add_instruction(Instruction::Jump {
                    target: loop_start,
                });
                
                // Seal the current block since all its successors are known
                self.seal_block(current_block);

                // Switch to loop start block
                self.current_block = loop_start;

                // Translate condition
                let condition_value = self.translate(condition);
                
                // Add conditional jump
                self.blocks[self.current_block.0].add_instruction(Instruction::ConditionalJump {
                    condition: condition_value,
                    true_target: loop_body,
                    false_target: loop_end,
                });

                // Add predecessors for loop body and loop end
                self.blocks[loop_body.0].add_predecessor(loop_start);
                self.blocks[loop_end.0].add_predecessor(loop_start);
                
                // Seal loop_end now - it won't get any more predecessors
                self.seal_block(loop_end);

                // Process loop body
                self.current_block = loop_body;
                for stmt in body {
                    self.translate(stmt);
                }

                // Jump back to loop start from body
                self.blocks[self.current_block.0].add_instruction(Instruction::Jump {
                    target: loop_start,
                });
                
                // Add the back-edge predecessor
                self.blocks[loop_start.0].add_predecessor(self.current_block);

                // Seal loop_body after processing it
                self.seal_block(loop_body);
                
                // Seal loop_start last, after the back-edge is added
                self.seal_block(loop_start);

                self.current_block = loop_end;
                Value::Undefined
            }
            ast => {
                println!("Unsupported AST node: {:?}", ast);
                unimplemented!();
            }
        }
    }

    fn create_block(&mut self) -> BlockId {
        let block_id = BlockId(self.blocks.len());
        self.blocks.push(Block::new(block_id));
        block_id
    }
}

fn main() {
    let program_lisp = program! {
        (set x 10)
        (set y 5)
        (set sum (+ (var x) (var y)))
        (if (> (var sum) 10)
            (if (> (var y) 0)
                (set result 1)
                (set result 2))
            (set result 1))
        (print (var result))
        (while (> (var x) 0)
            (set x (- (var x) 1)))
        (print (var x))
    };

    let mut ssa_translator = SSATranslator::new();
    ssa_translator.translate(&program_lisp);

    println!("{:#?}", program_lisp);

    ssa_properties::debug_ssa_state(&ssa_translator);

    
    let visualizer = visualizer::SSAVisualizer::new(&ssa_translator);

    
    if let Err(e) = visualizer.render_to_file("ssa_graph.dot") {
        eprintln!("Failed to write dot file: {}", e);
    }

    
    if let Err(e) = visualizer.render_to_png("ssa_graph.png") {
        eprintln!("Failed to render SSA PNG: {}", e);
    }
}
