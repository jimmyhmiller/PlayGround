use std::collections::{HashMap, HashSet};

use crate::ast::Ast;
use crate::instruction::{Block, BlockId, Instruction, Phi, Value, Variable};
mod ast;
mod instruction;
mod syntax;
mod visualizer;

pub struct SSATranslator {
    pub definition: HashMap<String, HashMap<BlockId, Value>>,
    pub sealed_blocks: HashSet<BlockId>,
    pub incomplete_phis: HashMap<BlockId, HashMap<String, Phi>>,
    pub blocks: Vec<Block>,
    pub next_variable_id: usize,
    pub current_block: BlockId, // The current block being processed
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
            blocks: Vec::new(),
            next_variable_id: 0,
            current_block: BlockId(0), // Will be set properly after creating first block
        };
        // Create the initial block
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
            value = Value::new_phi(block_id);
            self.incomplete_phis
                .entry(block_id)
                .or_default()
                .insert(variable.clone(), value.get_phi());
        } else {
            let block = self
                .blocks
                .get(block_id.0)
                .expect("Block not found");
            if block.predecessors.len() == 1 {
                value = self.read_variable(variable.clone(), block.predecessors[0]);
            } else {
                let val = Value::new_phi(block_id);
                self.write_variable(variable.clone(), block_id, val.clone());
                value = self.add_phi_operands(&variable, val.get_phi());
            }
        }
        self.write_variable(variable.clone(), block_id, value.clone());
        value
    }

    fn add_phi_operands(&mut self, variable: &String, phi: Phi) -> Value {
        let mut phi = phi;
        let block = self
            .blocks
            .get(phi.block_id.0)
            .expect("Block not found");
        for predecessor in block.predecessors.clone() {
            let value = self.read_variable(variable.clone(), predecessor);
            println!("value {:?}", value);
            phi.operands.push(value);
        }
        println!("phi {:?}", phi);
        self.try_remove_trivial_phi(phi)
    }

    fn try_remove_trivial_phi(&mut self, phi: Phi) -> Value {
        // TODO:
        let mut same: Option<&Value> = None;
        for operand in phi.operands.iter() {
            if Some(operand) == same || operand.is_same_phi(&phi) {
                continue;
            }
            if same.is_some() {
                return Value::PhiValue(Box::new(phi));
            }
            same = Some(operand);
        }
        if same.is_none() {
            same = Some(&Value::Undefined);
        }
        // TODO:
        // finish this
        // Rather than tracking uses, let's reify phis, give them ids
        // and then update in place

        Value::PhiValue(Box::new(same.unwrap().get_phi()))

        // Value::Literal(0)
    }

    fn seal_block(&mut self, block_id: BlockId) {
        self.sealed_blocks.insert(block_id);

        if let Some(phis) = self.incomplete_phis.clone().get(&block_id) {
            for variable in phis.keys() {
                let phi = self
                    .incomplete_phis
                    .get(&block_id)
                    .unwrap()
                    .get(variable)
                    .cloned()
                    .expect("Phi not found");
                self.add_phi_operands(variable, phi);
            }
        }
    }

    fn get_temp_variable(&mut self, _prefix: &str) -> Variable {
        let variable = Variable(format!("v{}", self.next_variable_id));
        self.next_variable_id += 1;
        variable
    }

    fn translate(&mut self, ast: &Ast) -> Value {
        let current_block = self.current_block;
        match ast {
            Ast::Literal(value) => {
                // Create a temporary variable for the literal value
                let temp_var = self.get_temp_variable("lit");
                // add the assignment instruction
                self.blocks[current_block.0].add_instruction(Instruction::Assign {
                    dest: temp_var.clone(),
                    value: Value::Literal(*value),
                });
                Value::Var(temp_var)
            }
            Ast::Variable(name) => {
                // Read the variable's value from the current block
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
                // Evaluate the condition in the current block
                let condition_value = self.translate(condition);

                // Create blocks for then, else, and merge
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

                // Add conditional jump to current block
                self.blocks[current_block.0].add_instruction(Instruction::ConditionalJump {
                    condition: condition_value,
                    true_target: then_block,
                    false_target: else_block,
                });

                let merge_block = self.create_block();
                // Process then branch
                self.current_block = then_block;
                for stmt in then_branch {
                    self.translate(stmt);
                }

                self.blocks[merge_block.0]
                    .predecessors
                    .push(self.current_block);

                // Add jump from then block to merge block
                self.blocks[self.current_block.0].add_instruction(Instruction::Jump {
                    target: merge_block,
                });

                // Process else branch
                self.current_block = else_block;
                if let Some(else_stmts) = else_branch {
                    for stmt in else_stmts {
                        self.translate(stmt);
                    }
                }
                self.blocks[merge_block.0]
                    .predecessors
                    .push(self.current_block);
                // Add jump from else block to merge block
                self.blocks[self.current_block.0].add_instruction(Instruction::Jump {
                    target: merge_block,
                });

                self.seal_block(merge_block);

                // Set current block to merge block for subsequent statements
                self.current_block = merge_block;

                // The merge block will be sealed when its containing block is done
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
            (if (> (var y) 0)
                (set result 1)
                (set result 2))
            (set result 0))
        (print (var result))
        // (while (> (var x) 0)
        //     (set x (- (var x) 1)))
    };

    let mut ssa_translator = SSATranslator::new();
    ssa_translator.translate(&program_lisp);

    println!("{:#?}", program_lisp);

    // Visualize the SSA graph
    let visualizer = visualizer::SSAVisualizer::new(&ssa_translator);

    // Generate dot file for inspection
    if let Err(e) = visualizer.render_to_file("ssa_graph.dot") {
        eprintln!("Failed to write dot file: {}", e);
    }

    // Generate and open PNG
    if let Err(e) = visualizer.render_and_open("ssa_graph.png") {
        eprintln!("Failed to visualize SSA: {}", e);
    }
}
