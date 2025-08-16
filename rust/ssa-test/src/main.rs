use std::collections::{HashMap, HashSet};

use crate::ast::Ast;
use crate::instruction::{Block, BlockId, Instruction, Phi, PhiId, PhiReference, Value, Variable};
mod ast;
mod instruction;
mod syntax;
mod visualizer;

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
            let phi_id = self.create_phi(block_id);
            value = Value::new_phi(phi_id);
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
                let phi_id = self.create_phi(block_id);
                let val = Value::new_phi(phi_id);
                self.write_variable(variable.clone(), block_id, val.clone());
                value = self.add_phi_operands(&variable, phi_id);
            }
        }
        self.write_variable(variable.clone(), block_id, value.clone());
        value
    }

    fn add_phi_operands(&mut self, variable: &String, phi_id: PhiId) -> Value {
        let block_id = self.phis.get(&phi_id).unwrap().block_id;
        
        // Check if this phi already has operands
        if !self.phis.get(&phi_id).unwrap().operands.is_empty() {
            return Value::Phi(phi_id);
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
        let mut same: Option<Value> = None;
        let phi = self.phis.get(&phi_id).expect("Phi not found").clone();
        
        
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

        // Replace all uses of this phi with the replacement
        for user in phi.uses.iter().cloned() {
            self.replace_phi_uses(phi_id, replacement.clone());
        }

        // Try to recursively remove all phi users, which might have become trivial
        for user in phi.uses.iter().cloned() {
            match user {
                PhiReference::Phi(user_phi_id) => {
                    // Only process phis that are not incomplete (have been through addPhiOperands)
                    let is_incomplete = self.incomplete_phis.values()
                        .any(|block_phis| block_phis.values().any(|&phi_id| phi_id == user_phi_id));
                    
                    if !is_incomplete {
                        self.try_remove_trivial_phi(user_phi_id);
                    }
                }
                _ => {}
            }
        }
        
        replacement
    }

    fn seal_block(&mut self, block_id: BlockId) {
        self.sealed_blocks.insert(block_id);
        self.blocks[block_id.0].seal();

        if let Some(phis) = self.incomplete_phis.clone().get(&block_id) {
            for variable in phis.keys() {
                let phi_id = self
                    .incomplete_phis
                    .get(&block_id)
                    .unwrap()
                    .get(variable)
                    .cloned()
                    .expect("Phi not found");
                self.add_phi_operands(variable, phi_id);
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

    fn replace_phi_uses(&mut self, phi_id: PhiId, replacement: Value) {
        if let Some(phi) = self.phis.get(&phi_id).cloned() {
            let uses = phi.uses.clone();
            for phi_ref in uses {
                match phi_ref {
                    PhiReference::Instruction { block_id, instruction_offset } => {
                        self.replace_value_at_location(block_id, instruction_offset, phi_id, replacement.clone());
                    }
                    PhiReference::Phi(ph_id) => {
                        self.phis.get_mut(&ph_id).map(|p| {
                            for operand in &mut p.operands {
                                if operand.is_same_phi(phi_id) {
                                    *operand = replacement.clone();
                                }
                            }
                        });
                    }
                }
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

    fn remove_phi(&mut self, phi_id: PhiId) {
        self.phis.remove(&phi_id);
        
        // Remove from incomplete_phis if present
        for phis_in_block in self.incomplete_phis.values_mut() {
            phis_in_block.retain(|_, &mut id| id != phi_id);
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

    
    let visualizer = visualizer::SSAVisualizer::new(&ssa_translator);

    
    if let Err(e) = visualizer.render_to_file("ssa_graph.dot") {
        eprintln!("Failed to write dot file: {}", e);
    }

    
    if let Err(e) = visualizer.render_and_open("ssa_graph.png") {
        eprintln!("Failed to visualize SSA: {}", e);
    }
}
