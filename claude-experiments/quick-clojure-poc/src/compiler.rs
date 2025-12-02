use crate::clojure_ast::Expr;
use crate::value::Value;
use crate::ir::{Instruction, IrValue, IrBuilder, Condition};
use std::collections::HashMap;

/// Clojure to IR compiler
///
/// Compiles Clojure AST to our IR, which is then compiled to ARM64.
pub struct Compiler {
    /// Global variables (def'd values)
    globals: HashMap<String, IrValue>,

    /// IR builder
    builder: IrBuilder,
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            globals: HashMap::new(),
            builder: IrBuilder::new(),
        }
    }

    /// Compile an expression and return the register containing the result
    pub fn compile(&mut self, expr: &Expr) -> Result<IrValue, String> {
        match expr {
            Expr::Literal(value) => self.compile_literal(value),
            Expr::Var(name) => self.compile_var(name),
            Expr::Def { name, value } => self.compile_def(name, value),
            Expr::If { test, then, else_ } => self.compile_if(test, then, else_),
            Expr::Do { exprs } => self.compile_do(exprs),
            Expr::Call { func, args } => self.compile_call(func, args),
            Expr::Quote(value) => self.compile_literal(value),
        }
    }

    fn compile_literal(&mut self, value: &Value) -> Result<IrValue, String> {
        let result = self.builder.new_register();

        match value {
            Value::Nil => {
                self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
            }
            Value::Bool(true) => {
                self.builder.emit(Instruction::LoadTrue(result));
            }
            Value::Bool(false) => {
                self.builder.emit(Instruction::LoadFalse(result));
            }
            Value::Int(i) => {
                // For now, just store the raw value - we'll add tagging later
                let tagged = (*i as isize) << 3;  // Simple 3-bit tag (000 for int)
                self.builder.emit(Instruction::LoadConstant(
                    result,
                    IrValue::TaggedConstant(tagged),
                ));
            }
            _ => {
                return Err(format!("Literal type not yet supported: {:?}", value));
            }
        }

        Ok(result)
    }

    fn compile_var(&mut self, name: &str) -> Result<IrValue, String> {
        // Look up in globals
        if let Some(value) = self.globals.get(name) {
            Ok(value.clone())
        } else {
            Err(format!("Undefined variable: {}", name))
        }
    }

    fn compile_def(&mut self, name: &str, value_expr: &Expr) -> Result<IrValue, String> {
        // Compile the value expression
        let value_reg = self.compile(value_expr)?;

        // Store in globals
        self.globals.insert(name.to_string(), value_reg.clone());

        // def returns the value
        Ok(value_reg)
    }

    fn compile_if(
        &mut self,
        test: &Expr,
        then: &Expr,
        else_: &Option<Box<Expr>>,
    ) -> Result<IrValue, String> {
        // Compile test expression
        let test_reg = self.compile(test)?;

        // Create labels for control flow
        let else_label = self.builder.new_label();
        let end_label = self.builder.new_label();

        // Result register
        let result = self.builder.new_register();

        // Jump to else if test is false
        self.builder.emit(Instruction::JumpIf(
            else_label.clone(),
            Condition::Equal,
            test_reg,
            IrValue::False,
        ));

        // Then branch
        let then_reg = self.compile(then)?;
        self.builder.emit(Instruction::Assign(result, then_reg));
        self.builder.emit(Instruction::Jump(end_label.clone()));

        // Else branch
        self.builder.emit(Instruction::Label(else_label));
        if let Some(else_expr) = else_ {
            let else_reg = self.compile(else_expr)?;
            self.builder.emit(Instruction::Assign(result, else_reg));
        } else {
            self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
        }

        // End label
        self.builder.emit(Instruction::Label(end_label));

        Ok(result)
    }

    fn compile_do(&mut self, exprs: &[Expr]) -> Result<IrValue, String> {
        let mut last_result = IrValue::Null;

        for expr in exprs {
            last_result = self.compile(expr)?;
        }

        Ok(last_result)
    }

    fn compile_call(&mut self, func: &Expr, args: &[Expr]) -> Result<IrValue, String> {
        // For now, only handle builtin functions
        // Later we'll handle user-defined functions

        if let Expr::Var(name) = func {
            match name.as_str() {
                "+" => self.compile_builtin_add(args),
                "-" => self.compile_builtin_sub(args),
                "*" => self.compile_builtin_mul(args),
                "/" => self.compile_builtin_div(args),
                "<" => self.compile_builtin_lt(args),
                ">" => self.compile_builtin_gt(args),
                "=" => self.compile_builtin_eq(args),
                _ => Err(format!("Unknown function: {}", name)),
            }
        } else {
            Err("Function calls must use a symbol".to_string())
        }
    }

    fn compile_builtin_add(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("+ requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        // Untag inputs
        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        // Add
        let sum = self.builder.new_register();
        self.builder.emit(Instruction::AddInt(sum, left_untagged, right_untagged));

        // Tag result (shift left 3 for int tag 000)
        let result = self.builder.new_register();
        let tag = IrValue::TaggedConstant(0);  // Int tag is 000
        self.builder.emit(Instruction::Tag(result, sum, tag));

        Ok(result)
    }

    fn compile_builtin_sub(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("- requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let diff = self.builder.new_register();
        self.builder.emit(Instruction::Sub(diff, left_untagged, right_untagged));

        let result = self.builder.new_register();
        let tag = IrValue::TaggedConstant(0);
        self.builder.emit(Instruction::Tag(result, diff, tag));

        Ok(result)
    }

    fn compile_builtin_mul(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("* requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let product = self.builder.new_register();
        self.builder.emit(Instruction::Mul(product, left_untagged, right_untagged));

        let result = self.builder.new_register();
        let tag = IrValue::TaggedConstant(0);
        self.builder.emit(Instruction::Tag(result, product, tag));

        Ok(result)
    }

    fn compile_builtin_div(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("/ requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let quotient = self.builder.new_register();
        self.builder.emit(Instruction::Div(quotient, left_untagged, right_untagged));

        let result = self.builder.new_register();
        let tag = IrValue::TaggedConstant(0);
        self.builder.emit(Instruction::Tag(result, quotient, tag));

        Ok(result)
    }

    fn compile_builtin_lt(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("< requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::LessThan));

        Ok(result)
    }

    fn compile_builtin_gt(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("> requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::GreaterThan));

        Ok(result)
    }

    fn compile_builtin_eq(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("= requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::Equal));

        Ok(result)
    }

    /// Get the generated IR instructions
    pub fn finish(self) -> Vec<Instruction> {
        self.builder.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read;
    use crate::clojure_ast::analyze;

    #[test]
    fn test_compile_add_generates_ir() {
        let mut compiler = Compiler::new();
        let val = read("(+ 1 2)").unwrap();
        let ast = analyze(&val).unwrap();

        compiler.compile(&ast).unwrap();
        let instructions = compiler.finish();

        // Should generate:
        // 1. LoadConstant for 1
        // 2. LoadConstant for 2
        // 3. Untag left
        // 4. Untag right
        // 5. AddInt
        // 6. Tag result
        println!("\nGenerated {} IR instructions for (+ 1 2):", instructions.len());
        for (i, inst) in instructions.iter().enumerate() {
            println!("  {}: {:?}", i, inst);
        }

        assert_eq!(instructions.len(), 6);
    }

    #[test]
    fn test_compile_nested() {
        let mut compiler = Compiler::new();
        let val = read("(+ (* 2 3) 4)").unwrap();
        let ast = analyze(&val).unwrap();

        compiler.compile(&ast).unwrap();
        let instructions = compiler.finish();

        println!("\nGenerated {} IR instructions for (+ (* 2 3) 4):", instructions.len());
        for (i, inst) in instructions.iter().enumerate() {
            println!("  {}: {:?}", i, inst);
        }

        // Should compile (* 2 3) first, then (+ result 4)
        assert!(instructions.len() > 10);
    }
}
