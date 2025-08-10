use std::collections::HashMap;
use std::io::{self, Write};

#[derive(Debug, Clone)]
enum Value {
    Number(f64),
    String(String),
}

#[derive(Debug)]
enum Statement {
    Let(String, Expr),
    Print(Expr),
    Goto(usize),
    If(Expr, usize),
    End,
}

#[derive(Debug)]
enum Expr {
    Value(Value),
    Var(String),
    BinOp(Box<Expr>, Op, Box<Expr>),
}

#[derive(Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

struct Interpreter {
    variables: HashMap<String, Value>,
    program: Vec<(usize, Statement)>,
}

impl Interpreter {
    fn new() -> Self {
        Interpreter {
            variables: HashMap::new(),
            program: Vec::new(),
        }
    }

    fn parse_line(&self, line: &str) -> Result<(usize, Statement), String> {
        let tokens: Vec<&str> = line.trim().split_whitespace().collect();
        if tokens.is_empty() {
            return Err("Empty line".to_string());
        }

        let line_number = tokens[0].parse::<usize>().map_err(|_| "Invalid line number")?;
        
        match tokens[1].to_uppercase().as_str() {
            "LET" => {
                if tokens.len() < 4 || tokens[3] != "=" {
                    return Err("Invalid LET statement".to_string());
                }
                let var_name = tokens[2].to_string();
                let expr = self.parse_expr(&tokens[4..])?;
                Ok((line_number, Statement::Let(var_name, expr)))
            }
            "PRINT" => {
                let expr = self.parse_expr(&tokens[2..])?;
                Ok((line_number, Statement::Print(expr)))
            }
            "GOTO" => {
                if tokens.len() < 3 {
                    return Err("Invalid GOTO statement".to_string());
                }
                let target = tokens[2].parse::<usize>().map_err(|_| "Invalid GOTO target")?;
                Ok((line_number, Statement::Goto(target)))
            }
            "IF" => {
                // Simple IF statement: IF <expr> THEN <line_number>
                if tokens.len() < 5 || tokens[tokens.len() - 2].to_uppercase() != "THEN" {
                    return Err("Invalid IF statement".to_string());
                }
                let target = tokens[tokens.len() - 1].parse::<usize>().map_err(|_| "Invalid IF target")?;
                let expr_tokens = &tokens[2..tokens.len() - 2];
                let expr = self.parse_expr(expr_tokens)?;
                Ok((line_number, Statement::If(expr, target)))
            }
            "END" => {
                Ok((line_number, Statement::End))
            }
            _ => Err("Unknown statement".to_string()),
        }
    }

    fn parse_expr(&self, tokens: &[&str]) -> Result<Expr, String> {
        if tokens.len() == 1 {
            // Try to parse as number first
            if let Ok(num) = tokens[0].parse::<f64>() {
                Ok(Expr::Value(Value::Number(num)))
            } else if tokens[0].starts_with('"') && tokens[0].ends_with('"') {
                // String literal
                Ok(Expr::Value(Value::String(tokens[0][1..tokens[0].len()-1].to_string())))
            } else {
                // Variable
                Ok(Expr::Var(tokens[0].to_string()))
            }
        } else if tokens.len() == 3 {
            let left = self.parse_expr(&[tokens[0]])?;
            let op = match tokens[1] {
                "+" => Op::Add,
                "-" => Op::Sub,
                "*" => Op::Mul,
                "/" => Op::Div,
                "=" => Op::Eq,
                "<>" => Op::Ne,
                "<" => Op::Lt,
                ">" => Op::Gt,
                "<=" => Op::Le,
                ">=" => Op::Ge,
                _ => return Err(format!("Unknown operator: {}", tokens[1])),
            };
            let right = self.parse_expr(&[tokens[2]])?;
            Ok(Expr::BinOp(Box::new(left), op, Box::new(right)))
        } else {
            Err("Invalid expression".to_string())
        }
    }

    fn eval_expr(&self, expr: &Expr) -> Result<Value, String> {
        match expr {
            Expr::Value(value) => Ok(value.clone()),
            Expr::Var(name) => {
                self.variables.get(name)
                    .cloned()
                    .ok_or_else(|| format!("Variable {} not defined", name))
            }
            Expr::BinOp(left, op, right) => {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                
                match (left_val, right_val, op) {
                    (Value::Number(l), Value::Number(r), Op::Add) => Ok(Value::Number(l + r)),
                    (Value::Number(l), Value::Number(r), Op::Sub) => Ok(Value::Number(l - r)),
                    (Value::Number(l), Value::Number(r), Op::Mul) => Ok(Value::Number(l * r)),
                    (Value::Number(l), Value::Number(r), Op::Div) => Ok(Value::Number(l / r)),
                    (Value::Number(l), Value::Number(r), Op::Eq) => Ok(Value::Number(if l == r { 1.0 } else { 0.0 })),
                    (Value::Number(l), Value::Number(r), Op::Ne) => Ok(Value::Number(if l != r { 1.0 } else { 0.0 })),
                    (Value::Number(l), Value::Number(r), Op::Lt) => Ok(Value::Number(if l < r { 1.0 } else { 0.0 })),
                    (Value::Number(l), Value::Number(r), Op::Gt) => Ok(Value::Number(if l > r { 1.0 } else { 0.0 })),
                    (Value::Number(l), Value::Number(r), Op::Le) => Ok(Value::Number(if l <= r { 1.0 } else { 0.0 })),
                    (Value::Number(l), Value::Number(r), Op::Ge) => Ok(Value::Number(if l >= r { 1.0 } else { 0.0 })),
                    _ => Err("Type mismatch in expression".to_string()),
                }
            }
        }
    }

    fn run_line(&mut self, line: &Statement) -> Result<bool, String> {
        // Returns true if program should continue, false if it should end
        match line {
            Statement::Let(var_name, expr) => {
                let value = self.eval_expr(expr)?;
                self.variables.insert(var_name.clone(), value);
                Ok(true)
            }
            Statement::Print(expr) => {
                let value = self.eval_expr(expr)?;
                match value {
                    Value::Number(n) => println!("{}", n),
                    Value::String(s) => println!("{}", s),
                }
                Ok(true)
            }
            Statement::Goto(target) => {
                // Find target line index
                for (i, (line_num, _)) in self.program.iter().enumerate() {
                    if line_num == target {
                        // We'll handle jumping in the run_program function
                        return Ok(true);
                    }
                }
                Err(format!("Line {} not found", target))
            }
            Statement::If(expr, target) => {
                let value = self.eval_expr(expr)?;
                match value {
                    Value::Number(n) => {
                        if n != 0.0 {
                            // We'll handle jumping in the run_program function
                            return Ok(true);
                        }
                    }
                    _ => return Err("IF condition must evaluate to a number".to_string()),
                }
                Ok(true)
            }
            Statement::End => Ok(false),
        }
    }

    fn load_program(&mut self, lines: Vec<String>) {
        self.program.clear();
        for line in lines {
            if let Ok(parsed_line) = self.parse_line(&line) {
                self.program.push(parsed_line);
            }
        }
        self.program.sort_by_key(|(line_num, _)| *line_num);
    }

    fn find_line_index(&self, line_number: usize) -> Option<usize> {
        self.program.iter().position(|(num, _)| *num == line_number)
    }

    fn run_program(&mut self) -> Result<(), String> {
        let mut current_index = 0;
        
        while current_index < self.program.len() {
            let (line_number, statement) = &self.program[current_index];
            
            // Handle GOTO and IF statements that might change the execution flow
            match statement {
                Statement::Goto(target) => {
                    if let Some(target_index) = self.find_line_index(*target) {
                        current_index = target_index;
                        continue;
                    } else {
                        return Err(format!("Line {} not found", target));
                    }
                }
                Statement::If(expr, target) => {
                    let value = self.eval_expr(expr)?;
                    match value {
                        Value::Number(n) => {
                            if n != 0.0 {
                                if let Some(target_index) = self.find_line_index(*target) {
                                    current_index = target_index;
                                    continue;
                                } else {
                                    return Err(format!("Line {} not found", target));
                                }
                            }
                        }
                        _ => return Err("IF condition must evaluate to a number".to_string()),
                    }
                }
                _ => {
                    if !self.run_line(statement)? {
                        break;
                    }
                }
            }
            
            current_index += 1;
        }
        
        Ok(())
    }
}

fn main() {
    let mut interpreter = Interpreter::new();
    
    println!("Simple BASIC Interpreter");
    println!("Enter your BASIC program line by line (enter 'RUN' to execute, 'EXIT' to quit):");
    
    let mut program_lines = Vec::new();
    
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        let input = input.trim();
        
        if input.to_uppercase() == "EXIT" {
            break;
        } else if input.to_uppercase() == "RUN" {
            interpreter.load_program(program_lines.clone());
            if let Err(e) = interpreter.run_program() {
                println!("Error: {}", e);
            }
            println!("Program execution finished.");
        } else {
            program_lines.push(input.to_string());
        }
    }
}