use crate::chunk::{Chunk, OpCode};
use crate::object::{GcHeap, ObjFunction};
use crate::scanner::{Scanner, Token, TokenType};
use crate::value::{Value, number_val, obj_val};

use std::ptr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum Precedence {
    None,
    Assignment,
    Or,
    And,
    Equality,
    Comparison,
    Term,
    Factor,
    Unary,
    Call,
    Primary,
}

impl Precedence {
    fn next(self) -> Precedence {
        match self {
            Precedence::None => Precedence::Assignment,
            Precedence::Assignment => Precedence::Or,
            Precedence::Or => Precedence::And,
            Precedence::And => Precedence::Equality,
            Precedence::Equality => Precedence::Comparison,
            Precedence::Comparison => Precedence::Term,
            Precedence::Term => Precedence::Factor,
            Precedence::Factor => Precedence::Unary,
            Precedence::Unary => Precedence::Call,
            Precedence::Call => Precedence::Primary,
            Precedence::Primary => Precedence::Primary,
        }
    }
}

type ParseFn = fn(&mut Compiler, bool);

struct ParseRule {
    prefix: Option<ParseFn>,
    infix: Option<ParseFn>,
    precedence: Precedence,
}

fn get_rule(token_type: TokenType) -> ParseRule {
    match token_type {
        TokenType::LeftParen => ParseRule {
            prefix: Some(Compiler::grouping),
            infix: Some(Compiler::call),
            precedence: Precedence::Call,
        },
        TokenType::RightParen => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::LeftBrace => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::RightBrace => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Comma => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Dot => ParseRule {
            prefix: None,
            infix: Some(Compiler::dot),
            precedence: Precedence::Call,
        },
        TokenType::Minus => ParseRule {
            prefix: Some(Compiler::unary),
            infix: Some(Compiler::binary),
            precedence: Precedence::Term,
        },
        TokenType::Plus => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Term,
        },
        TokenType::Semicolon => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Slash => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Factor,
        },
        TokenType::Star => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Factor,
        },
        TokenType::Bang => ParseRule {
            prefix: Some(Compiler::unary),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::BangEqual => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Equality,
        },
        TokenType::Equal => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::EqualEqual => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Equality,
        },
        TokenType::Greater => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Comparison,
        },
        TokenType::GreaterEqual => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Comparison,
        },
        TokenType::Less => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Comparison,
        },
        TokenType::LessEqual => ParseRule {
            prefix: None,
            infix: Some(Compiler::binary),
            precedence: Precedence::Comparison,
        },
        TokenType::Identifier => ParseRule {
            prefix: Some(Compiler::variable),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::String => ParseRule {
            prefix: Some(Compiler::string),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Number => ParseRule {
            prefix: Some(Compiler::number),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::And => ParseRule {
            prefix: None,
            infix: Some(Compiler::and_),
            precedence: Precedence::And,
        },
        TokenType::Class => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Else => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::False => ParseRule {
            prefix: Some(Compiler::literal),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::For => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Fun => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::If => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Nil => ParseRule {
            prefix: Some(Compiler::literal),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Or => ParseRule {
            prefix: None,
            infix: Some(Compiler::or_),
            precedence: Precedence::Or,
        },
        TokenType::Print => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Return => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Super => ParseRule {
            prefix: Some(Compiler::super_),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::This => ParseRule {
            prefix: Some(Compiler::this_),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::True => ParseRule {
            prefix: Some(Compiler::literal),
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Var => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::While => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Error => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
        TokenType::Eof => ParseRule {
            prefix: None,
            infix: None,
            precedence: Precedence::None,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FunctionType {
    Function,
    Script,
    Method,
    Initializer,
}

struct Local {
    name: String,
    depth: i32,
    is_captured: bool,
}

#[derive(Debug, Clone, Copy)]
struct Upvalue {
    index: u8,
    is_local: bool,
}

struct ClassCompiler {
    has_superclass: bool,
}

struct CompilerState {
    function: *mut ObjFunction,
    function_type: FunctionType,
    locals: Vec<Local>,
    upvalues: Vec<Upvalue>,
    scope_depth: i32,
}

impl CompilerState {
    fn new(function: *mut ObjFunction, function_type: FunctionType) -> Self {
        let mut state = CompilerState {
            function,
            function_type,
            locals: Vec::with_capacity(256),
            upvalues: Vec::with_capacity(256),
            scope_depth: 0,
        };
        // Slot 0 is for the VM's internal use
        let name = if function_type != FunctionType::Function {
            "this".to_string()
        } else {
            String::new()
        };
        state.locals.push(Local {
            name,
            depth: 0,
            is_captured: false,
        });
        state
    }
}

pub struct Compiler {
    scanner: Scanner,
    current: Token,
    previous: Token,
    had_error: bool,
    panic_mode: bool,
    heap: *mut GcHeap,
    states: Vec<CompilerState>,
    class_compilers: Vec<ClassCompiler>,
}

impl Compiler {
    pub fn compile(source: &str, heap: &mut GcHeap) -> Option<*mut ObjFunction> {
        let function = heap.alloc_function();
        let scanner = Scanner::new(source);

        let dummy_token = Token {
            token_type: TokenType::Eof,
            lexeme: String::new(),
            line: 0,
        };

        let mut compiler = Compiler {
            scanner,
            current: dummy_token.clone(),
            previous: dummy_token,
            had_error: false,
            panic_mode: false,
            heap,
            states: vec![CompilerState::new(function, FunctionType::Script)],
            class_compilers: Vec::new(),
        };

        compiler.advance();

        while !compiler.match_token(TokenType::Eof) {
            compiler.declaration();
        }

        compiler.emit_return();

        if compiler.had_error {
            None
        } else {
            Some(function)
        }
    }

    fn current_state(&self) -> &CompilerState {
        self.states.last().unwrap()
    }

    fn current_state_mut(&mut self) -> &mut CompilerState {
        self.states.last_mut().unwrap()
    }

    fn current_chunk(&mut self) -> &mut Chunk {
        unsafe { &mut (*self.current_state().function).chunk }
    }

    fn heap(&mut self) -> &mut GcHeap {
        unsafe { &mut *self.heap }
    }

    // ── Parsing helpers ──────────────────────────────────────────

    fn advance(&mut self) {
        self.previous = self.current.clone();
        loop {
            self.current = self.scanner.scan_token();
            if self.current.token_type != TokenType::Error {
                break;
            }
            let msg = self.current.lexeme.clone();
            self.error_at_current(&msg);
        }
    }

    fn consume(&mut self, token_type: TokenType, message: &str) {
        if self.current.token_type == token_type {
            self.advance();
            return;
        }
        self.error_at_current(message);
    }

    fn check(&self, token_type: TokenType) -> bool {
        self.current.token_type == token_type
    }

    fn match_token(&mut self, token_type: TokenType) -> bool {
        if !self.check(token_type) {
            return false;
        }
        self.advance();
        true
    }

    // ── Emitting bytecode ────────────────────────────────────────

    fn emit_byte(&mut self, byte: u8) {
        let line = self.previous.line;
        self.current_chunk().write(byte, line);
    }

    fn emit_bytes(&mut self, byte1: u8, byte2: u8) {
        self.emit_byte(byte1);
        self.emit_byte(byte2);
    }

    fn emit_return(&mut self) {
        if self.current_state().function_type == FunctionType::Initializer {
            self.emit_bytes(OpCode::GetLocal as u8, 0);
        } else {
            self.emit_byte(OpCode::Nil as u8);
        }
        self.emit_byte(OpCode::Return as u8);
    }

    fn emit_constant(&mut self, value: Value) {
        let constant = self.make_constant(value);
        self.emit_bytes(OpCode::Constant as u8, constant);
    }

    fn make_constant(&mut self, value: Value) -> u8 {
        let constant = self.current_chunk().add_constant(value);
        if constant > 255 {
            self.error("Too many constants in one chunk.");
            return 0;
        }
        constant as u8
    }

    fn emit_jump(&mut self, instruction: u8) -> usize {
        self.emit_byte(instruction);
        self.emit_byte(0xff);
        self.emit_byte(0xff);
        self.current_chunk().code.len() - 2
    }

    fn patch_jump(&mut self, offset: usize) {
        let jump = self.current_chunk().code.len() - offset - 2;
        if jump > u16::MAX as usize {
            self.error("Too much code to jump over.");
        }
        self.current_chunk().code[offset] = ((jump >> 8) & 0xff) as u8;
        self.current_chunk().code[offset + 1] = (jump & 0xff) as u8;
    }

    fn emit_loop(&mut self, loop_start: usize) {
        self.emit_byte(OpCode::Loop as u8);
        let offset = self.current_chunk().code.len() - loop_start + 2;
        if offset > u16::MAX as usize {
            self.error("Loop body too large.");
        }
        self.emit_byte(((offset >> 8) & 0xff) as u8);
        self.emit_byte((offset & 0xff) as u8);
    }

    // ── Declarations ─────────────────────────────────────────────

    fn declaration(&mut self) {
        if self.match_token(TokenType::Class) {
            self.class_declaration();
        } else if self.match_token(TokenType::Fun) {
            self.fun_declaration();
        } else if self.match_token(TokenType::Var) {
            self.var_declaration();
        } else {
            self.statement();
        }

        if self.panic_mode {
            self.synchronize();
        }
    }

    fn class_declaration(&mut self) {
        self.consume(TokenType::Identifier, "Expect class name.");
        let class_name = self.previous.clone();
        let name_constant = self.identifier_constant(&class_name.lexeme);
        self.declare_variable();

        self.emit_bytes(OpCode::Class as u8, name_constant);
        self.define_variable(name_constant);

        self.class_compilers.push(ClassCompiler {
            has_superclass: false,
        });

        if self.match_token(TokenType::Less) {
            self.consume(TokenType::Identifier, "Expect superclass name.");
            Compiler::variable(self, false);

            if class_name.lexeme == self.previous.lexeme {
                self.error("A class can't inherit from itself.");
            }

            self.begin_scope();
            self.add_local("super".to_string());
            self.define_variable(0);

            self.named_variable(&class_name.lexeme, false);
            self.emit_byte(OpCode::Inherit as u8);
            self.class_compilers.last_mut().unwrap().has_superclass = true;
        }

        self.named_variable(&class_name.lexeme, false);
        self.consume(TokenType::LeftBrace, "Expect '{' before class body.");

        while !self.check(TokenType::RightBrace) && !self.check(TokenType::Eof) {
            self.method();
        }

        self.consume(TokenType::RightBrace, "Expect '}' after class body.");
        self.emit_byte(OpCode::Pop as u8);

        let has_superclass = self.class_compilers.last().unwrap().has_superclass;
        if has_superclass {
            self.end_scope();
        }

        self.class_compilers.pop();
    }

    fn method(&mut self) {
        self.consume(TokenType::Identifier, "Expect method name.");
        let name = self.previous.lexeme.clone();
        let constant = self.identifier_constant(&name);

        let function_type = if name == "init" {
            FunctionType::Initializer
        } else {
            FunctionType::Method
        };
        self.function(function_type);
        self.emit_bytes(OpCode::Method as u8, constant);
    }

    fn fun_declaration(&mut self) {
        let global = self.parse_variable("Expect function name.");
        self.mark_initialized();
        self.function(FunctionType::Function);
        self.define_variable(global);
    }

    fn function(&mut self, function_type: FunctionType) {
        let function = self.heap().alloc_function();
        let name_str = self.previous.lexeme.clone();
        unsafe {
            (*function).name = if function_type != FunctionType::Script {
                self.heap().alloc_string(name_str)
            } else {
                ptr::null_mut()
            };
        }

        self.states
            .push(CompilerState::new(function, function_type));
        self.begin_scope();

        self.consume(TokenType::LeftParen, "Expect '(' after function name.");
        if !self.check(TokenType::RightParen) {
            loop {
                let arity = unsafe { (*self.current_state().function).arity as u32 };
                if arity >= 255 {
                    self.error_at_current("Can't have more than 255 parameters.");
                }
                unsafe { (*self.current_state().function).arity += 1 };
                let constant = self.parse_variable("Expect parameter name.");
                self.define_variable(constant);
                if !self.match_token(TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenType::RightParen, "Expect ')' after parameters.");
        self.consume(TokenType::LeftBrace, "Expect '{' before function body.");
        self.block();

        self.emit_return();

        let state = self.states.pop().unwrap();
        let upvalue_count = state.upvalues.len();
        unsafe {
            (*function).upvalue_count = upvalue_count as u16;
        }

        let constant = self.make_constant(obj_val(function as *mut crate::object::Obj));
        self.emit_bytes(OpCode::Closure as u8, constant);
        for upvalue in &state.upvalues {
            self.emit_byte(if upvalue.is_local { 1 } else { 0 });
            self.emit_byte(upvalue.index);
        }
    }

    fn var_declaration(&mut self) {
        let global = self.parse_variable("Expect variable name.");

        if self.match_token(TokenType::Equal) {
            self.expression();
        } else {
            self.emit_byte(OpCode::Nil as u8);
        }
        self.consume(
            TokenType::Semicolon,
            "Expect ';' after variable declaration.",
        );

        self.define_variable(global);
    }

    // ── Statements ───────────────────────────────────────────────

    fn statement(&mut self) {
        if self.match_token(TokenType::Print) {
            self.print_statement();
        } else if self.match_token(TokenType::For) {
            self.for_statement();
        } else if self.match_token(TokenType::If) {
            self.if_statement();
        } else if self.match_token(TokenType::Return) {
            self.return_statement();
        } else if self.match_token(TokenType::While) {
            self.while_statement();
        } else if self.match_token(TokenType::LeftBrace) {
            self.begin_scope();
            self.block();
            self.end_scope();
        } else {
            self.expression_statement();
        }
    }

    fn print_statement(&mut self) {
        self.expression();
        self.consume(TokenType::Semicolon, "Expect ';' after value.");
        self.emit_byte(OpCode::Print as u8);
    }

    fn for_statement(&mut self) {
        self.begin_scope();
        self.consume(TokenType::LeftParen, "Expect '(' after 'for'.");

        // Initializer
        if self.match_token(TokenType::Semicolon) {
            // No initializer
        } else if self.match_token(TokenType::Var) {
            self.var_declaration();
        } else {
            self.expression_statement();
        }

        let mut loop_start = self.current_chunk().code.len();

        // Condition
        let mut exit_jump = None;
        if !self.match_token(TokenType::Semicolon) {
            self.expression();
            self.consume(TokenType::Semicolon, "Expect ';' after loop condition.");
            exit_jump = Some(self.emit_jump(OpCode::JumpIfFalse as u8));
            self.emit_byte(OpCode::Pop as u8);
        }

        // Increment
        if !self.match_token(TokenType::RightParen) {
            let body_jump = self.emit_jump(OpCode::Jump as u8);
            let increment_start = self.current_chunk().code.len();
            self.expression();
            self.emit_byte(OpCode::Pop as u8);
            self.consume(TokenType::RightParen, "Expect ')' after for clauses.");

            self.emit_loop(loop_start);
            loop_start = increment_start;
            self.patch_jump(body_jump);
        }

        self.statement();
        self.emit_loop(loop_start);

        if let Some(exit_jump) = exit_jump {
            self.patch_jump(exit_jump);
            self.emit_byte(OpCode::Pop as u8);
        }

        self.end_scope();
    }

    fn if_statement(&mut self) {
        self.consume(TokenType::LeftParen, "Expect '(' after 'if'.");
        self.expression();
        self.consume(TokenType::RightParen, "Expect ')' after condition.");

        let then_jump = self.emit_jump(OpCode::JumpIfFalse as u8);
        self.emit_byte(OpCode::Pop as u8);
        self.statement();

        let else_jump = self.emit_jump(OpCode::Jump as u8);
        self.patch_jump(then_jump);
        self.emit_byte(OpCode::Pop as u8);

        if self.match_token(TokenType::Else) {
            self.statement();
        }
        self.patch_jump(else_jump);
    }

    fn return_statement(&mut self) {
        if self.current_state().function_type == FunctionType::Script {
            self.error("Can't return from top-level code.");
        }

        if self.match_token(TokenType::Semicolon) {
            self.emit_return();
        } else {
            if self.current_state().function_type == FunctionType::Initializer {
                self.error("Can't return a value from an initializer.");
            }
            self.expression();
            self.consume(TokenType::Semicolon, "Expect ';' after return value.");
            self.emit_byte(OpCode::Return as u8);
        }
    }

    fn while_statement(&mut self) {
        let loop_start = self.current_chunk().code.len();
        self.consume(TokenType::LeftParen, "Expect '(' after 'while'.");
        self.expression();
        self.consume(TokenType::RightParen, "Expect ')' after condition.");

        let exit_jump = self.emit_jump(OpCode::JumpIfFalse as u8);
        self.emit_byte(OpCode::Pop as u8);
        self.statement();
        self.emit_loop(loop_start);

        self.patch_jump(exit_jump);
        self.emit_byte(OpCode::Pop as u8);
    }

    fn expression_statement(&mut self) {
        self.expression();
        self.consume(TokenType::Semicolon, "Expect ';' after expression.");
        self.emit_byte(OpCode::Pop as u8);
    }

    fn block(&mut self) {
        while !self.check(TokenType::RightBrace) && !self.check(TokenType::Eof) {
            self.declaration();
        }
        self.consume(TokenType::RightBrace, "Expect '}' after block.");
    }

    // ── Expressions (Pratt parser) ───────────────────────────────

    fn expression(&mut self) {
        self.parse_precedence(Precedence::Assignment);
    }

    fn parse_precedence(&mut self, precedence: Precedence) {
        self.advance();
        let prefix_rule = get_rule(self.previous.token_type).prefix;
        let can_assign = precedence <= Precedence::Assignment;

        match prefix_rule {
            Some(prefix) => prefix(self, can_assign),
            None => {
                self.error("Expect expression.");
                return;
            }
        }

        while precedence <= get_rule(self.current.token_type).precedence {
            self.advance();
            let infix_rule = get_rule(self.previous.token_type).infix;
            if let Some(infix) = infix_rule {
                infix(self, can_assign);
            }
        }

        if can_assign && self.match_token(TokenType::Equal) {
            self.error("Invalid assignment target.");
        }
    }

    fn number(&mut self, _can_assign: bool) {
        let value: f64 = self.previous.lexeme.parse().unwrap();
        self.emit_constant(number_val(value));
    }

    fn string(&mut self, _can_assign: bool) {
        let lexeme = self.previous.lexeme.clone();
        let s = &lexeme[1..lexeme.len() - 1];
        let obj_str = self.heap().alloc_string(s.to_string());
        self.emit_constant(obj_val(obj_str as *mut crate::object::Obj));
    }

    fn variable(&mut self, can_assign: bool) {
        let name = self.previous.lexeme.clone();
        self.named_variable(&name, can_assign);
    }

    fn named_variable(&mut self, name: &str, can_assign: bool) {
        let (get_op, set_op, arg) = match self.resolve_local(self.states.len() - 1, name) {
            Some((arg, false)) => (OpCode::GetLocal, OpCode::SetLocal, arg),
            Some((arg, true)) => {
                self.error("Can't read local variable in its own initializer.");
                (OpCode::GetLocal, OpCode::SetLocal, arg)
            }
            None => {
                if let Some(arg) = self.resolve_upvalue(self.states.len() - 1, name) {
                    (OpCode::GetUpvalue, OpCode::SetUpvalue, arg)
                } else {
                    let arg = self.identifier_constant(name);
                    (OpCode::GetGlobal, OpCode::SetGlobal, arg)
                }
            }
        };

        if can_assign && self.match_token(TokenType::Equal) {
            self.expression();
            self.emit_bytes(set_op as u8, arg);
        } else {
            self.emit_bytes(get_op as u8, arg);
        }
    }

    fn grouping(&mut self, _can_assign: bool) {
        self.expression();
        self.consume(TokenType::RightParen, "Expect ')' after expression.");
    }

    fn unary(&mut self, _can_assign: bool) {
        let operator_type = self.previous.token_type;
        self.parse_precedence(Precedence::Unary);
        match operator_type {
            TokenType::Bang => self.emit_byte(OpCode::Not as u8),
            TokenType::Minus => self.emit_byte(OpCode::Negate as u8),
            _ => unreachable!(),
        }
    }

    fn binary(&mut self, _can_assign: bool) {
        let operator_type = self.previous.token_type;
        let rule = get_rule(operator_type);
        self.parse_precedence(rule.precedence.next());

        match operator_type {
            TokenType::BangEqual => {
                self.emit_bytes(OpCode::Equal as u8, OpCode::Not as u8);
            }
            TokenType::EqualEqual => self.emit_byte(OpCode::Equal as u8),
            TokenType::Greater => self.emit_byte(OpCode::Greater as u8),
            TokenType::GreaterEqual => {
                self.emit_bytes(OpCode::Less as u8, OpCode::Not as u8);
            }
            TokenType::Less => self.emit_byte(OpCode::Less as u8),
            TokenType::LessEqual => {
                self.emit_bytes(OpCode::Greater as u8, OpCode::Not as u8);
            }
            TokenType::Plus => self.emit_byte(OpCode::Add as u8),
            TokenType::Minus => self.emit_byte(OpCode::Subtract as u8),
            TokenType::Star => self.emit_byte(OpCode::Multiply as u8),
            TokenType::Slash => self.emit_byte(OpCode::Divide as u8),
            _ => unreachable!(),
        }
    }

    fn call(&mut self, _can_assign: bool) {
        let arg_count = self.argument_list();
        self.emit_bytes(OpCode::Call as u8, arg_count);
    }

    fn dot(&mut self, can_assign: bool) {
        self.consume(TokenType::Identifier, "Expect property name after '.'.");
        let name = self.identifier_constant(&self.previous.lexeme.clone());

        if can_assign && self.match_token(TokenType::Equal) {
            self.expression();
            self.emit_bytes(OpCode::SetProperty as u8, name);
        } else if self.match_token(TokenType::LeftParen) {
            let arg_count = self.argument_list();
            self.emit_bytes(OpCode::Invoke as u8, name);
            self.emit_byte(arg_count);
        } else {
            self.emit_bytes(OpCode::GetProperty as u8, name);
        }
    }

    fn literal(&mut self, _can_assign: bool) {
        match self.previous.token_type {
            TokenType::False => self.emit_byte(OpCode::False as u8),
            TokenType::Nil => self.emit_byte(OpCode::Nil as u8),
            TokenType::True => self.emit_byte(OpCode::True as u8),
            _ => unreachable!(),
        }
    }

    fn and_(&mut self, _can_assign: bool) {
        let end_jump = self.emit_jump(OpCode::JumpIfFalse as u8);
        self.emit_byte(OpCode::Pop as u8);
        self.parse_precedence(Precedence::And);
        self.patch_jump(end_jump);
    }

    fn or_(&mut self, _can_assign: bool) {
        let else_jump = self.emit_jump(OpCode::JumpIfFalse as u8);
        let end_jump = self.emit_jump(OpCode::Jump as u8);

        self.patch_jump(else_jump);
        self.emit_byte(OpCode::Pop as u8);

        self.parse_precedence(Precedence::Or);
        self.patch_jump(end_jump);
    }

    fn this_(&mut self, _can_assign: bool) {
        if self.class_compilers.is_empty() {
            self.error("Can't use 'this' outside of a class.");
            return;
        }
        Compiler::variable(self, false);
    }

    fn super_(&mut self, _can_assign: bool) {
        if self.class_compilers.is_empty() {
            self.error("Can't use 'super' outside of a class.");
        } else if !self.class_compilers.last().unwrap().has_superclass {
            self.error("Can't use 'super' in a class with no superclass.");
        }

        self.consume(TokenType::Dot, "Expect '.' after 'super'.");
        self.consume(TokenType::Identifier, "Expect superclass method name.");
        let name = self.identifier_constant(&self.previous.lexeme.clone());

        self.named_variable("this", false);

        if self.match_token(TokenType::LeftParen) {
            let arg_count = self.argument_list();
            self.named_variable("super", false);
            self.emit_bytes(OpCode::SuperInvoke as u8, name);
            self.emit_byte(arg_count);
        } else {
            self.named_variable("super", false);
            self.emit_bytes(OpCode::GetSuper as u8, name);
        }
    }

    fn argument_list(&mut self) -> u8 {
        let mut arg_count: u8 = 0;
        if !self.check(TokenType::RightParen) {
            loop {
                self.expression();
                if arg_count == 255 {
                    self.error("Can't have more than 255 arguments.");
                }
                arg_count += 1;
                if !self.match_token(TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenType::RightParen, "Expect ')' after arguments.");
        arg_count
    }

    // ── Variable resolution ──────────────────────────────────────

    fn identifier_constant(&mut self, name: &str) -> u8 {
        let obj_str = self.heap().alloc_string(name.to_string());
        self.make_constant(obj_val(obj_str as *mut crate::object::Obj))
    }

    fn parse_variable(&mut self, error_message: &str) -> u8 {
        self.consume(TokenType::Identifier, error_message);

        self.declare_variable();
        if self.current_state().scope_depth > 0 {
            return 0;
        }

        self.identifier_constant(&self.previous.lexeme.clone())
    }

    fn declare_variable(&mut self) {
        if self.current_state().scope_depth == 0 {
            return;
        }

        let name = self.previous.lexeme.clone();

        // Check for duplicate in current scope
        let scope_depth = self.current_state().scope_depth;
        for local in self.current_state().locals.iter().rev() {
            if local.depth != -1 && local.depth < scope_depth {
                break;
            }
            if local.name == name {
                self.error("Already a variable with this name in this scope.");
                return;
            }
        }

        self.add_local(name);
    }

    fn add_local(&mut self, name: String) {
        if self.current_state().locals.len() >= 256 {
            self.error("Too many local variables in function.");
            return;
        }
        self.current_state_mut().locals.push(Local {
            name,
            depth: -1,
            is_captured: false,
        });
    }

    fn mark_initialized(&mut self) {
        if self.current_state().scope_depth == 0 {
            return;
        }
        let depth = self.current_state().scope_depth;
        self.current_state_mut().locals.last_mut().unwrap().depth = depth;
    }

    fn define_variable(&mut self, global: u8) {
        if self.current_state().scope_depth > 0 {
            self.mark_initialized();
            return;
        }
        self.emit_bytes(OpCode::DefineGlobal as u8, global);
    }

    /// Returns Some((index, uninitialized)) if found.
    fn resolve_local(&self, state_idx: usize, name: &str) -> Option<(u8, bool)> {
        let state = &self.states[state_idx];
        for (i, local) in state.locals.iter().enumerate().rev() {
            if local.name == name {
                return Some((i as u8, local.depth == -1));
            }
        }
        None
    }

    fn resolve_upvalue(&mut self, state_idx: usize, name: &str) -> Option<u8> {
        if state_idx == 0 {
            return None;
        }

        // Check locals in the enclosing function
        if let Some((local, _)) = self.resolve_local(state_idx - 1, name) {
            self.states[state_idx - 1].locals[local as usize].is_captured = true;
            return Some(self.add_upvalue(state_idx, local, true));
        }

        // Check upvalues in the enclosing function
        if let Some(upvalue) = self.resolve_upvalue(state_idx - 1, name) {
            return Some(self.add_upvalue(state_idx, upvalue, false));
        }

        None
    }

    fn add_upvalue(&mut self, state_idx: usize, index: u8, is_local: bool) -> u8 {
        let state = &self.states[state_idx];
        let upvalue_count = state.upvalues.len();

        for (i, upvalue) in state.upvalues.iter().enumerate() {
            if upvalue.index == index && upvalue.is_local == is_local {
                return i as u8;
            }
        }

        if upvalue_count >= 256 {
            self.error("Too many closure variables in function.");
            return 0;
        }

        self.states[state_idx]
            .upvalues
            .push(Upvalue { index, is_local });
        upvalue_count as u8
    }

    // ── Scoping ──────────────────────────────────────────────────

    fn begin_scope(&mut self) {
        self.current_state_mut().scope_depth += 1;
    }

    fn end_scope(&mut self) {
        self.current_state_mut().scope_depth -= 1;
        let depth = self.current_state().scope_depth;

        while !self.current_state().locals.is_empty()
            && self.current_state().locals.last().unwrap().depth > depth
        {
            if self.current_state().locals.last().unwrap().is_captured {
                self.emit_byte(OpCode::CloseUpvalue as u8);
            } else {
                self.emit_byte(OpCode::Pop as u8);
            }
            self.current_state_mut().locals.pop();
        }
    }

    // ── Error handling ───────────────────────────────────────────

    fn error(&mut self, message: &str) {
        self.error_at(&self.previous.clone(), message);
    }

    fn error_at_current(&mut self, message: &str) {
        self.error_at(&self.current.clone(), message);
    }

    fn error_at(&mut self, token: &Token, message: &str) {
        if self.panic_mode {
            return;
        }
        self.panic_mode = true;
        eprint!("[line {}] Error", token.line);

        match token.token_type {
            TokenType::Eof => eprint!(" at end"),
            TokenType::Error => {}
            _ => eprint!(" at '{}'", token.lexeme),
        }

        eprintln!(": {}", message);
        self.had_error = true;
    }

    fn synchronize(&mut self) {
        self.panic_mode = false;

        while self.current.token_type != TokenType::Eof {
            if self.previous.token_type == TokenType::Semicolon {
                return;
            }

            match self.current.token_type {
                TokenType::Class
                | TokenType::Fun
                | TokenType::Var
                | TokenType::For
                | TokenType::If
                | TokenType::While
                | TokenType::Print
                | TokenType::Return => return,
                _ => {}
            }

            self.advance();
        }
    }
}
