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
        let block = self
            .blocks
            .get(block_id.0)
            .expect("Block not found");
        
        for predecessor in block.predecessors.clone() {
            let value = self.read_variable(variable.clone(), predecessor);
            println!("value {:?}", value);
            if let Some(phi) = self.phis.get_mut(&phi_id) {
                phi.operands.push(value);
            }
        }
        
        if let Some(phi) = self.phis.get(&phi_id) {
            println!("phi {:?}", phi);
        }
        
        self.try_remove_trivial_phi(phi_id)
    }

    fn try_remove_trivial_phi(&mut self, phi_id: PhiId) -> Value {
        let mut same: Option<Value> = None;
        
        println!("Trying to remove trivial phi: {:?}", phi_id);
        
        // First pass: check if phi is trivial
        if let Some(phi) = self.phis.get(&phi_id) {
            println!("Phi operands: {:?}", phi.operands);
            
            // Debug: print all phis
            println!("All phis:");
            for (id, phi) in &self.phis {
                println!("  {:?}: {:?}", id, phi);
            }
            for operand in phi.operands.iter() {
                println!("Checking operand: {:?}", operand);
                
                // Resolve phi operands recursively
                let resolved_operand = if let Value::Phi(op_phi_id) = operand {
                    if *op_phi_id == phi_id {
                        // Self-reference, skip
                        println!("Skipping self-reference");
                        continue;
                    }
                    // Recursively resolve the phi operand
                    let resolved = self.try_remove_trivial_phi(*op_phi_id);
                    println!("Resolved phi operand {:?} to {:?}", op_phi_id, resolved);
                    resolved
                } else {
                    operand.clone()
                };
                
                println!("Resolved operand: {:?}", resolved_operand);
                println!("same.as_ref(): {:?}", same.as_ref());
                
                if Some(&resolved_operand) == same.as_ref() {
                    println!("Skipping operand (same as previous)");
                    continue; // Same value as before
                }
                if same.is_some() {
                    println!("Phi is not trivial - merges multiple values");
                    return Value::Phi(phi_id); // The phi merges at least two values: not trivial
                }
                same = Some(resolved_operand);
                println!("Set same to: {:?}", same);
            }
        } else {
            return Value::Undefined; // Phi doesn't exist
        }
        
        if same.is_none() {
            same = Some(Value::Undefined); // The phi is unreachable or in the start block
        }
        
        let replacement = same.unwrap();
        
        // Get users before removing the phi (to avoid borrow issues)
        let users: Vec<PhiId> = if let Some(_phi) = self.phis.get(&phi_id) {
            self.phis.values()
                .filter(|other_phi| {
                    other_phi.operands.iter().any(|op| {
                        if let Value::Phi(id) = op {
                            *id == phi_id
                        } else {
                            false
                        }
                    })
                })
                .map(|other_phi| other_phi.id)
                .collect()
        } else {
            Vec::new()
        };
        
        // Replace all uses of phi with the replacement value and remove phi
        self.replace_phi_uses(phi_id, replacement.clone());
        self.remove_phi(phi_id);
        
        // Try to recursively remove all phi users, which might have become trivial
        for user_phi_id in users {
            if self.phis.contains_key(&user_phi_id) {
                self.try_remove_trivial_phi(user_phi_id);
            }
        }
        
        replacement
    }

    fn seal_block(&mut self, block_id: BlockId) {
        self.sealed_blocks.insert(block_id);

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
        phi_id
    }

    fn add_phi_use(&mut self, phi_id: PhiId, block_id: BlockId, instruction_offset: usize) {
        if let Some(phi) = self.phis.get_mut(&phi_id) {
            phi.uses.push(PhiReference {
                block_id,
                instruction_offset,
            });
        }
    }

    fn replace_phi_uses(&mut self, phi_id: PhiId, replacement: Value) {
        if let Some(phi) = self.phis.get(&phi_id).cloned() {
            let uses = phi.uses.clone();
            for phi_ref in uses {
                self.replace_value_at_location(phi_ref.block_id, phi_ref.instruction_offset, phi_id, replacement.clone());
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

                // TODO: I think I need to do this a bit better. It really should
                // be that I
                // 1. Change the current block
                // 2. Translate the if
                // 3. then setup the merge block
                // 4. then same on the else.
                // This will ensure that if things are nested they don't get all confused.

                self.blocks[then_block.0].predecessors.push(current_block);
                self.blocks[else_block.0].predecessors.push(current_block);
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
                    .predecessors
                    .push(self.current_block);

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
                    .predecessors
                    .push(self.current_block);

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
            // (if (> (var y) 0)
            //     (set result 1)
            //     (set result 2))
            (set result 0)
            0)
        (print (var result))
        // (while (> (var x) 0)
        //     (set x (- (var x) 1)))
    };

    let mut ssa_translator = SSATranslator::new();
    ssa_translator.translate(&program_lisp);

    println!("{:#?}", program_lisp);

    
    let visualizer = visualizer::SSAVisualizer::new(&ssa_translator);

    
    if let Err(e) = visualizer.render_to_file("ssa_graph.dot") {
        eprintln!("Failed to write dot file: {}", e);
    }

    
    // if let Err(e) = visualizer.render_and_open("ssa_graph.png") {
    //     eprintln!("Failed to visualize SSA: {}", e);
    // }
}
