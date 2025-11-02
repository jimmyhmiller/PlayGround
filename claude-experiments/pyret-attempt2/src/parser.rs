//! Pyret Parser
//!
//! Hand-written recursive descent parser implementing the complete Pyret grammar.
//! Each BNF rule maps to a corresponding parse function.
//!
//! Reference: /pyret-lang/src/js/base/pyret-grammar.bnf

use crate::ast::*;
use crate::error::{ParseError, ParseResult};
use crate::tokenizer::{Token, TokenType};

// ============================================================================
// SECTION 1: Parser Struct and Core Methods
// ============================================================================

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    position: usize, // Alias for current, used for checkpointing
    file_name: String,
}

impl Parser {
    pub fn new(tokens: Vec<Token>, file_name: String) -> Self {
        Parser {
            tokens,
            current: 0,
            position: 0,
            file_name,
        }
    }

    // ========== Core Navigation ==========

    fn peek(&self) -> &Token {
        if self.current < self.tokens.len() {
            &self.tokens[self.current]
        } else {
            // Return EOF token if we're past the end
            self.tokens.last().unwrap()
        }
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
            self.position = self.current;
        }
        &self.tokens[self.current - 1]
    }

    fn peek_ahead(&self, offset: usize) -> &Token {
        let pos = self.current + offset;
        if pos < self.tokens.len() {
            &self.tokens[pos]
        } else {
            // Return EOF token if we're past the end
            self.tokens.last().unwrap()
        }
    }

    fn expect(&mut self, token_type: TokenType) -> ParseResult<Token> {
        if self.matches(&token_type) {
            Ok(self.advance().clone())
        } else {
            Err(ParseError::expected(token_type, self.peek().clone()))
        }
    }

    fn matches(&self, token_type: &TokenType) -> bool {
        !self.is_at_end() && &self.peek().token_type == token_type
    }

    fn is_at_end(&self) -> bool {
        self.peek().token_type == TokenType::Eof
    }

    // ========== Checkpointing for Backtracking ==========

    fn checkpoint(&self) -> usize {
        self.current
    }

    fn restore(&mut self, checkpoint: usize) {
        self.current = checkpoint;
        self.position = checkpoint;
    }

    // ========== Location Helpers ==========

    fn current_loc(&self) -> Loc {
        let token = self.peek();
        Loc::new(
            self.file_name.clone(),
            token.location.start_line,
            token.location.start_col,
            token.location.start_pos,
            token.location.end_line,
            token.location.end_col,
            token.location.end_pos,
        )
    }

    fn make_loc(&self, start: &Token, end: &Token) -> Loc {
        Loc::new(
            self.file_name.clone(),
            start.location.start_line,
            start.location.start_col,
            start.location.start_pos,
            end.location.end_line,
            end.location.end_col,
            end.location.end_pos,
        )
    }

    /// Parse a field name (can be a Name token or a keyword used as identifier)
    /// In Pyret, keywords can be used as field names after a dot
    fn parse_field_name(&mut self) -> ParseResult<Token> {
        let token = self.peek().clone();
        // Accept Name tokens or any keyword token
        if matches!(token.token_type, TokenType::Name) || self.is_keyword(&token.token_type) {
            Ok(self.advance().clone())
        } else {
            Err(ParseError::expected(TokenType::Name, token))
        }
    }

    /// Check if a token type is a keyword that can be used as a field name
    fn is_keyword(&self, token_type: &TokenType) -> bool {
        matches!(
            token_type,
            TokenType::Method
                | TokenType::Fun
                | TokenType::Var
                | TokenType::Let
                | TokenType::Letrec
                | TokenType::Rec
                | TokenType::Data
                | TokenType::If
                | TokenType::Else
                | TokenType::ElseIf
                | TokenType::ElseColon
                | TokenType::When
                | TokenType::Block
                | TokenType::For
                | TokenType::From
                | TokenType::End
                | TokenType::Check
                | TokenType::CheckColon
                | TokenType::Where
                | TokenType::Import
                | TokenType::Provide
                | TokenType::ProvideColon
                | TokenType::ProvideTypes
                | TokenType::Include
                | TokenType::Sharing
                | TokenType::Shadow
                | TokenType::Type
                | TokenType::TypeLet
                | TokenType::Newtype
                | TokenType::Doc
                | TokenType::Cases
                | TokenType::Ask
                | TokenType::OtherwiseColon
                | TokenType::ThenColon
                | TokenType::Spy
                | TokenType::Reactor
                | TokenType::Table
                | TokenType::TableExtend
                | TokenType::TableExtract
                | TokenType::TableFilter
                | TokenType::TableOrder
                | TokenType::TableSelect
                | TokenType::TableUpdate
                | TokenType::LoadTable
                | TokenType::Sanitize
                | TokenType::Ref
                | TokenType::With
                | TokenType::Use
                | TokenType::Using
                | TokenType::Module
                | TokenType::Lam
                | TokenType::Lazy
                | TokenType::Row
                | TokenType::SourceColon
                | TokenType::Examples
                | TokenType::ExamplesColon
                | TokenType::Do
                | TokenType::Of
                | TokenType::By
                | TokenType::Hiding
                | TokenType::Ascending
                | TokenType::Descending
        )
    }
}

// ============================================================================
// SECTION 2: Program & Top-Level
// ============================================================================

impl Parser {
    /// program: prelude block
    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let start = self.peek().clone();

        // For now, parse empty prelude (no imports/provides)
        // TODO: Implement full prelude parsing when needed
        let _use = None;
        let _provide = Provide::SProvideNone {
            l: self.current_loc(),
        };
        let provided_types = ProvideTypes::SProvideTypesNone {
            l: self.current_loc(),
        };
        let provides = Vec::new();
        let imports = Vec::new();

        // Parse program body (statement block)
        let body = self.parse_block()?;

        // Ensure we've consumed all tokens
        if !self.is_at_end() {
            return Err(ParseError::general(
                self.peek(),
                "Unexpected tokens after program end",
            ));
        }

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        Ok(Program::new(
            self.make_loc(&start, &end),
            _use,
            _provide,
            provided_types,
            provides,
            imports,
            body,
        ))
    }

    /// prelude: [use-stmt] (provide-stmt|import-stmt)*
    fn parse_prelude(&mut self) -> ParseResult<()> {
        // TODO: Implement full prelude parsing
        // For now, just return Ok since we're skipping prelude
        Ok(())
    }

    /// block: stmt*
    /// Parses a sequence of statements and returns an SBlock expression
    fn parse_block(&mut self) -> ParseResult<Expr> {
        let start = self.peek().clone();
        let mut stmts = Vec::new();

        // Parse statements until EOF or until we can't parse any more
        while !self.is_at_end() {
            // Try to parse different statement types
            let stmt = if self.matches(&TokenType::Let) {
                // Explicit let binding: let x = 5
                self.parse_let_expr()?
            } else if self.matches(&TokenType::Var) {
                // Var binding: var x := 5
                self.parse_var_expr()?
            } else if self.matches(&TokenType::Name) {
                // Check if this is an implicit let binding: x = value
                // Look ahead to see if there's an = or := after the name
                let checkpoint = self.checkpoint();
                let _name = self.advance(); // Consume the name

                if self.matches(&TokenType::Equals) {
                    // Implicit let binding: x = value
                    self.restore(checkpoint);
                    self.parse_implicit_let_expr()?
                } else if self.matches(&TokenType::ColonEquals) {
                    // Implicit var binding: x := value
                    self.restore(checkpoint);
                    self.parse_implicit_var_expr()?
                } else {
                    // Not a binding, restore and parse as expression
                    self.restore(checkpoint);
                    match self.parse_expr() {
                        Ok(expr) => expr,
                        Err(_) => break,
                    }
                }
            } else {
                // Default: try to parse as expression
                match self.parse_expr() {
                    Ok(expr) => expr,
                    Err(_) => break, // Stop if we can't parse
                }
            };
            stmts.push(Box::new(stmt));
        }

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        Ok(Expr::SBlock {
            l: self.make_loc(&start, &end),
            stmts,
        })
    }
}

// ============================================================================
// SECTION 3: Import/Export Parsing
// ============================================================================

impl Parser {
    /// use-stmt: USE NAME import-source
    fn parse_use_stmt(&mut self) -> ParseResult<Use> {
        todo!("Implement parse_use_stmt")
    }

    /// import-stmt: INCLUDE | IMPORT | ...
    fn parse_import_stmt(&mut self) -> ParseResult<Import> {
        todo!("Implement parse_import_stmt")
    }

    /// provide-stmt: PROVIDE stmt END | ...
    fn parse_provide_stmt(&mut self) -> ParseResult<Provide> {
        todo!("Implement parse_provide_stmt")
    }

    /// provide-types-stmt: PROVIDE-TYPES ann | ...
    fn parse_provide_types_stmt(&mut self) -> ParseResult<ProvideTypes> {
        todo!("Implement parse_provide_types_stmt")
    }
}

// ============================================================================
// SECTION 4: Type Annotation Parsing
// ============================================================================

impl Parser {
    /// ann: name-ann | record-ann | arrow-ann | ...
    fn parse_ann(&mut self) -> ParseResult<Ann> {
        todo!("Implement parse_ann")
    }

    /// arrow-ann: (args -> ret)
    fn parse_arrow_ann(&mut self) -> ParseResult<Ann> {
        todo!("Implement parse_arrow_ann")
    }

    /// record-ann: { field, ... }
    fn parse_record_ann(&mut self) -> ParseResult<Ann> {
        todo!("Implement parse_record_ann")
    }

    /// tuple-ann: { ann; ... }
    fn parse_tuple_ann(&mut self) -> ParseResult<Ann> {
        todo!("Implement parse_tuple_ann")
    }
}

// ============================================================================
// SECTION 5: Binding Parsing
// ============================================================================

impl Parser {
    /// binding: name [:: ann]
    /// Parses a parameter binding like: x, x :: Number
    fn parse_bind(&mut self) -> ParseResult<Bind> {
        let name_token = self.expect(TokenType::Name)?;
        let name_str = name_token.value.clone();

        // Create Name node
        let name = Name::SName {
            l: Loc::new(
                self.file_name.clone(),
                name_token.location.start_line,
                name_token.location.start_col,
                name_token.location.start_pos,
                name_token.location.end_line,
                name_token.location.end_col,
                name_token.location.end_pos,
            ),
            s: name_str,
        };

        // Optional type annotation: :: ann
        let ann = if self.matches(&TokenType::ColonColon) {
            self.expect(TokenType::ColonColon)?;
            self.parse_ann()?
        } else {
            Ann::ABlank
        };

        Ok(Bind::SBind {
            l: Loc::new(
                self.file_name.clone(),
                name_token.location.start_line,
                name_token.location.start_col,
                name_token.location.start_pos,
                name_token.location.end_line,
                name_token.location.end_col,
                name_token.location.end_pos,
            ),
            shadows: false, // Default to false for lambda parameters
            id: name,
            ann,
        })
    }

    /// tuple-binding: { id; ... }
    fn parse_tuple_bind(&mut self) -> ParseResult<Bind> {
        todo!("Implement parse_tuple_bind")
    }

    /// let-binding: LET | VAR binding = expr
    fn parse_let_bind(&mut self) -> ParseResult<LetBind> {
        todo!("Implement parse_let_bind")
    }
}

// ============================================================================
// SECTION 6: Expression Parsing
// ============================================================================

impl Parser {
    /// expr: binop-expr | prim-expr | ...
    /// Top-level expression dispatcher
    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        // For now, just handle binop expressions which subsume primitive expressions
        self.parse_binop_expr()
    }

    /// Parse an expression and ensure all tokens are consumed (expects EOF)
    /// This is useful for standalone expression parsing where trailing tokens should be an error
    pub fn parse_expr_complete(&mut self) -> ParseResult<Expr> {
        let expr = self.parse_expr()?;

        // Ensure we've consumed all tokens
        if !self.is_at_end() {
            return Err(ParseError::general(
                self.peek(),
                "Expected end of input after expression",
            ));
        }

        Ok(expr)
    }

    /// binop-expr: expr binop expr (left-associative)
    /// Parse binary operator expressions with left-associativity
    /// Also handles function application (postfix operator)
    fn parse_binop_expr(&mut self) -> ParseResult<Expr> {
        // Start with a primary expression
        let mut left = self.parse_prim_expr()?;

        // Check for postfix operators (function application, dot access, and bracket access)
        // These can be chained together, e.g., obj.foo()[0].bar()
        loop {
            if self.matches(&TokenType::ParenNoSpace) {
                // Function application (no whitespace before paren)
                left = self.parse_app_expr(left)?;
            } else if self.matches(&TokenType::Dot) {
                // Dot access or tuple access
                let _dot_token = self.expect(TokenType::Dot)?;

                // Check if this is tuple access: .{number}
                if self.matches(&TokenType::LBrace) {
                    // Tuple access: x.{2}
                    self.expect(TokenType::LBrace)?;
                    let index_token = self.expect(TokenType::Number)?;
                    let index: usize = index_token.value.parse()
                        .map_err(|_| ParseError::invalid("tuple index", &index_token, "Invalid number"))?;
                    let index_loc = self.make_loc(&index_token, &index_token);
                    let end = self.expect(TokenType::RBrace)?;

                    let start_loc = match &left {
                        Expr::SNum { l, .. } => l.clone(),
                        Expr::SBool { l, .. } => l.clone(),
                        Expr::SStr { l, .. } => l.clone(),
                        Expr::SId { l, .. } => l.clone(),
                        Expr::SOp { l, .. } => l.clone(),
                        Expr::SParen { l, .. } => l.clone(),
                        Expr::SApp { l, .. } => l.clone(),
                        Expr::SConstruct { l, .. } => l.clone(),
                        Expr::SDot { l, .. } => l.clone(),
                        Expr::SBracket { l, .. } => l.clone(),
                        Expr::SObj { l, .. } => l.clone(),
                        Expr::STuple { l, .. } => l.clone(),
                        Expr::STupleGet { l, .. } => l.clone(),
                        Expr::SLam { l, .. } => l.clone(),
                        Expr::SBlock { l, .. } => l.clone(),
                        Expr::SUserBlock { l, .. } => l.clone(),
                        Expr::SIf { l, .. } => l.clone(),
                        Expr::SIfElse { l, .. } => l.clone(),
                        Expr::SWhen { l, .. } => l.clone(),
                        Expr::SFor { l, .. } => l.clone(),
                        Expr::SLetExpr { l, .. } => l.clone(),
                Expr::SLet { l, .. } => l.clone(),
                Expr::SVar { l, .. } => l.clone(),
                Expr::SAssign { l, .. } => l.clone(),
                        _ => self.current_loc(),
                    };

                    left = Expr::STupleGet {
                        l: Loc::new(
                            self.file_name.clone(),
                            start_loc.start_line,
                            start_loc.start_column,
                            start_loc.start_char,
                            end.location.end_line,
                            end.location.end_col,
                            end.location.end_pos,
                        ),
                        tup: Box::new(left),
                        index,
                        index_loc,
                    };
                } else {
                    // Regular dot access: obj.field
                    let field_token = self.parse_field_name()?;

                    let start_loc = match &left {
                        Expr::SNum { l, .. } => l.clone(),
                        Expr::SBool { l, .. } => l.clone(),
                        Expr::SStr { l, .. } => l.clone(),
                        Expr::SId { l, .. } => l.clone(),
                        Expr::SOp { l, .. } => l.clone(),
                        Expr::SParen { l, .. } => l.clone(),
                        Expr::SApp { l, .. } => l.clone(),
                        Expr::SConstruct { l, .. } => l.clone(),
                        Expr::SDot { l, .. } => l.clone(),
                        Expr::SBracket { l, .. } => l.clone(),
                        Expr::SObj { l, .. } => l.clone(),
                        Expr::STuple { l, .. } => l.clone(),
                        Expr::STupleGet { l, .. } => l.clone(),
                        Expr::SLam { l, .. } => l.clone(),
                        Expr::SBlock { l, .. } => l.clone(),
                        Expr::SUserBlock { l, .. } => l.clone(),
                        Expr::SIf { l, .. } => l.clone(),
                        Expr::SIfElse { l, .. } => l.clone(),
                        Expr::SWhen { l, .. } => l.clone(),
                        Expr::SFor { l, .. } => l.clone(),
                        Expr::SLetExpr { l, .. } => l.clone(),
                Expr::SLet { l, .. } => l.clone(),
                Expr::SVar { l, .. } => l.clone(),
                Expr::SAssign { l, .. } => l.clone(),
                        _ => self.current_loc(),
                    };

                    left = Expr::SDot {
                        l: Loc::new(
                            self.file_name.clone(),
                            start_loc.start_line,
                            start_loc.start_column,
                            start_loc.start_char,
                            field_token.location.end_line,
                            field_token.location.end_col,
                            field_token.location.end_pos,
                        ),
                        obj: Box::new(left),
                        field: field_token.value.clone(),
                    };
                }
            } else if self.matches(&TokenType::LBrack) {
                // Bracket access
                left = self.parse_bracket_expr(left)?;
            } else {
                // No more postfix operators
                break;
            }
        }

        // Parse operators and right-hand sides in a left-associative manner
        while self.is_binop() {
            let op_token = self.peek().clone();
            let op = self.parse_binop()?;

            // Parse right-hand side with postfix operators
            let mut right = self.parse_prim_expr()?;

            // Check for postfix operators on the right side (same as left side)
            loop {
                if self.matches(&TokenType::ParenNoSpace) {
                    right = self.parse_app_expr(right)?;
                } else if self.matches(&TokenType::Dot) {
                    let _dot_token = self.expect(TokenType::Dot)?;
                    let field_token = self.parse_field_name()?;

                    let start_loc = match &right {
                        Expr::SNum { l, .. } => l.clone(),
                        Expr::SBool { l, .. } => l.clone(),
                        Expr::SStr { l, .. } => l.clone(),
                        Expr::SId { l, .. } => l.clone(),
                        Expr::SOp { l, .. } => l.clone(),
                        Expr::SParen { l, .. } => l.clone(),
                        Expr::SApp { l, .. } => l.clone(),
                        Expr::SConstruct { l, .. } => l.clone(),
                        Expr::SDot { l, .. } => l.clone(),
                        Expr::SBracket { l, .. } => l.clone(),
                        Expr::SObj { l, .. } => l.clone(),
                        Expr::STuple { l, .. } => l.clone(),
                        Expr::STupleGet { l, .. } => l.clone(),
                        Expr::SLam { l, .. } => l.clone(),
                        _ => self.current_loc(),
                    };

                    right = Expr::SDot {
                        l: Loc::new(
                            self.file_name.clone(),
                            start_loc.start_line,
                            start_loc.start_column,
                            start_loc.start_char,
                            field_token.location.end_line,
                            field_token.location.end_col,
                            field_token.location.end_pos,
                        ),
                        obj: Box::new(right),
                        field: field_token.value.clone(),
                    };
                } else if self.matches(&TokenType::LBrack) {
                    // Bracket access
                    right = self.parse_bracket_expr(right)?;
                } else {
                    break;
                }
            }

            // Create location for the operator
            let op_l = Loc::new(
                self.file_name.clone(),
                op_token.location.start_line,
                op_token.location.start_col,
                op_token.location.start_pos,
                op_token.location.end_line,
                op_token.location.end_col,
                op_token.location.end_pos,
            );

            // Get location from left expression
            let start_loc = match &left {
                Expr::SNum { l, .. } => l.clone(),
                Expr::SBool { l, .. } => l.clone(),
                Expr::SStr { l, .. } => l.clone(),
                Expr::SId { l, .. } => l.clone(),
                Expr::SOp { l, .. } => l.clone(),
                Expr::SParen { l, .. } => l.clone(),
                Expr::SApp { l, .. } => l.clone(),
                Expr::SConstruct { l, .. } => l.clone(),
                Expr::SDot { l, .. } => l.clone(),
                _ => self.current_loc(),
            };

            // Get location from right expression
            let end_loc = match &right {
                Expr::SNum { l, .. } => l.clone(),
                Expr::SBool { l, .. } => l.clone(),
                Expr::SStr { l, .. } => l.clone(),
                Expr::SId { l, .. } => l.clone(),
                Expr::SOp { l, .. } => l.clone(),
                Expr::SParen { l, .. } => l.clone(),
                Expr::SApp { l, .. } => l.clone(),
                Expr::SConstruct { l, .. } => l.clone(),
                Expr::SDot { l, .. } => l.clone(),
                _ => self.current_loc(),
            };

            let loc = Loc::new(
                self.file_name.clone(),
                start_loc.start_line,
                start_loc.start_column,
                start_loc.start_char,
                end_loc.end_line,
                end_loc.end_column,
                end_loc.end_char,
            );

            left = Expr::SOp {
                l: loc,
                op_l,
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        // Check for check operators (is, raises, satisfies, violates)
        // These have lower precedence than binary operators
        if self.is_check_op() {
            let op = self.parse_check_op()?;

            // Parse the right-hand side (if needed)
            // Some check operators may not have a right side (e.g., "raises" without specific error)
            let right = if !self.is_at_end() && !matches!(self.peek().token_type, TokenType::Eof) {
                Some(Box::new(self.parse_prim_expr()?))
            } else {
                None
            };

            // Get location from left expression
            let start_loc = match &left {
                Expr::SNum { l, .. } => l.clone(),
                Expr::SBool { l, .. } => l.clone(),
                Expr::SStr { l, .. } => l.clone(),
                Expr::SId { l, .. } => l.clone(),
                Expr::SOp { l, .. } => l.clone(),
                Expr::SParen { l, .. } => l.clone(),
                Expr::SApp { l, .. } => l.clone(),
                Expr::SConstruct { l, .. } => l.clone(),
                Expr::SDot { l, .. } => l.clone(),
                Expr::SBracket { l, .. } => l.clone(),
                Expr::SObj { l, .. } => l.clone(),
                Expr::STuple { l, .. } => l.clone(),
                Expr::STupleGet { l, .. } => l.clone(),
                Expr::SLam { l, .. } => l.clone(),
                Expr::SBlock { l, .. } => l.clone(),
                Expr::SUserBlock { l, .. } => l.clone(),
                Expr::SIf { l, .. } => l.clone(),
                Expr::SIfElse { l, .. } => l.clone(),
                Expr::SWhen { l, .. } => l.clone(),
                Expr::SFor { l, .. } => l.clone(),
                Expr::SLetExpr { l, .. } => l.clone(),
                Expr::SLet { l, .. } => l.clone(),
                Expr::SVar { l, .. } => l.clone(),
                Expr::SAssign { l, .. } => l.clone(),
                _ => self.current_loc(),
            };

            // Get end location from right expression if it exists, otherwise from operator
            let end_loc = if let Some(ref right_expr) = right {
                match right_expr.as_ref() {
                    Expr::SNum { l, .. } => l.clone(),
                    Expr::SBool { l, .. } => l.clone(),
                    Expr::SStr { l, .. } => l.clone(),
                    Expr::SId { l, .. } => l.clone(),
                    Expr::SOp { l, .. } => l.clone(),
                    Expr::SParen { l, .. } => l.clone(),
                    Expr::SApp { l, .. } => l.clone(),
                    Expr::SConstruct { l, .. } => l.clone(),
                    Expr::SDot { l, .. } => l.clone(),
                    Expr::SBracket { l, .. } => l.clone(),
                    Expr::SObj { l, .. } => l.clone(),
                    Expr::STuple { l, .. } => l.clone(),
                    Expr::STupleGet { l, .. } => l.clone(),
                    Expr::SLam { l, .. } => l.clone(),
                    Expr::SBlock { l, .. } => l.clone(),
                    Expr::SUserBlock { l, .. } => l.clone(),
                    Expr::SFor { l, .. } => l.clone(),
                    Expr::SLetExpr { l, .. } => l.clone(),
                Expr::SLet { l, .. } => l.clone(),
                Expr::SVar { l, .. } => l.clone(),
                    _ => self.current_loc(),
                }
            } else {
                // If no right expression, use operator location
                match &op {
                    CheckOp::SOpIs { l } => l.clone(),
                    CheckOp::SOpIsRoughly { l } => l.clone(),
                    CheckOp::SOpIsNot { l } => l.clone(),
                    CheckOp::SOpIsNotRoughly { l } => l.clone(),
                    CheckOp::SOpIsOp { l, .. } => l.clone(),
                    CheckOp::SOpIsNotOp { l, .. } => l.clone(),
                    CheckOp::SOpSatisfies { l } => l.clone(),
                    CheckOp::SOpSatisfiesNot { l } => l.clone(),
                    CheckOp::SOpRaises { l } => l.clone(),
                    CheckOp::SOpRaisesOther { l } => l.clone(),
                    CheckOp::SOpRaisesNot { l } => l.clone(),
                    CheckOp::SOpRaisesSatisfies { l } => l.clone(),
                    CheckOp::SOpRaisesViolates { l } => l.clone(),
                }
            };

            let loc = Loc::new(
                self.file_name.clone(),
                start_loc.start_line,
                start_loc.start_column,
                start_loc.start_char,
                end_loc.end_line,
                end_loc.end_column,
                end_loc.end_char,
            );

            left = Expr::SCheckTest {
                l: loc,
                op,
                refinement: None,
                left: Box::new(left),
                right,
                cause: None,
            };
        }

        Ok(left)
    }

    /// prim-expr: num-expr | frac-expr | rfrac-expr | bool-expr | string-expr | id-expr | paren-expr | array-expr | lam-expr | block-expr
    fn parse_prim_expr(&mut self) -> ParseResult<Expr> {
        let token = self.peek().clone();

        match token.token_type {
            TokenType::Number => self.parse_num(),
            TokenType::Rational => self.parse_rational(),
            TokenType::RoughRational => self.parse_rough_rational(),
            TokenType::True | TokenType::False => self.parse_bool(),
            TokenType::String => self.parse_str(),
            TokenType::Name => self.parse_id_expr(),
            TokenType::ParenSpace | TokenType::LParen => self.parse_paren_expr(),
            TokenType::LBrack => self.parse_construct_expr(),
            TokenType::LBrace => self.parse_obj_expr(),
            TokenType::Lam => self.parse_lambda_expr(),
            TokenType::Block => self.parse_block_expr(),
            TokenType::If => self.parse_if_expr(),
            TokenType::When => self.parse_when_expr(),
            TokenType::For => self.parse_for_expr(),
            TokenType::Let => self.parse_let_expr(),
            TokenType::Var => self.parse_var_expr(),
            _ => Err(ParseError::unexpected(token)),
        }
    }

    /// num-expr: NUMBER
    /// Pyret represents all numbers as rationals, so:
    /// - Integers like "42" -> SNum with n=42.0
    /// - Decimals like "3.14" -> SFrac with num=157, den=50
    fn parse_num(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::Number)?;
        let loc = Loc::new(
            self.file_name.clone(),
            token.location.start_line,
            token.location.start_col,
            token.location.start_pos,
            token.location.end_line,
            token.location.end_col,
            token.location.end_pos,
        );

        // Check if this is a decimal number (contains '.')
        if token.value.contains('.') {
            // Convert decimal to rational
            let (num, den) = Self::decimal_to_rational(&token.value)
                .map_err(|e| ParseError::invalid("number", &token, &e))?;

            Ok(Expr::SFrac { l: loc, num, den })
        } else {
            // Integer - keep as SNum
            let n: f64 = token
                .value
                .parse()
                .map_err(|_| ParseError::invalid("number", &token, "Invalid number format"))?;

            Ok(Expr::SNum { l: loc, n })
        }
    }

    /// frac-expr: RATIONAL
    /// Parses explicit rational numbers like "3/2" or "-5/7"
    fn parse_rational(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::Rational)?;
        let loc = Loc::new(
            self.file_name.clone(),
            token.location.start_line,
            token.location.start_col,
            token.location.start_pos,
            token.location.end_line,
            token.location.end_col,
            token.location.end_pos,
        );

        // Parse "num/den" format
        let parts: Vec<&str> = token.value.split('/').collect();
        if parts.len() != 2 {
            return Err(ParseError::invalid(
                "rational",
                &token,
                "Expected format: numerator/denominator",
            ));
        }

        let num: i64 = parts[0]
            .parse()
            .map_err(|_| ParseError::invalid("rational", &token, "Invalid numerator"))?;
        let den: i64 = parts[1]
            .parse()
            .map_err(|_| ParseError::invalid("rational", &token, "Invalid denominator"))?;

        if den == 0 {
            return Err(ParseError::invalid(
                "rational",
                &token,
                "Denominator cannot be zero",
            ));
        }

        // Simplify the fraction
        let gcd = Self::gcd(num.abs(), den.abs());
        let (num, den) = (num / gcd, den / gcd);

        Ok(Expr::SFrac { l: loc, num, den })
    }

    /// rfrac-expr: ROUGHRATIONAL
    /// Parses rough (approximate) rational numbers like "~3/2" or "~-5/7"
    fn parse_rough_rational(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::RoughRational)?;
        let loc = Loc::new(
            self.file_name.clone(),
            token.location.start_line,
            token.location.start_col,
            token.location.start_pos,
            token.location.end_line,
            token.location.end_col,
            token.location.end_pos,
        );

        // The tokenizer includes the '~' in the value, so we need to strip it
        let value = token.value.trim_start_matches('~');

        // Parse "num/den" format
        let parts: Vec<&str> = value.split('/').collect();
        if parts.len() != 2 {
            return Err(ParseError::invalid(
                "rough rational",
                &token,
                "Expected format: ~numerator/denominator",
            ));
        }

        let num: i64 = parts[0]
            .parse()
            .map_err(|_| ParseError::invalid("rough rational", &token, "Invalid numerator"))?;
        let den: i64 = parts[1]
            .parse()
            .map_err(|_| ParseError::invalid("rough rational", &token, "Invalid denominator"))?;

        if den == 0 {
            return Err(ParseError::invalid(
                "rough rational",
                &token,
                "Denominator cannot be zero",
            ));
        }

        // Simplify the fraction
        let gcd = Self::gcd(num.abs(), den.abs());
        let (num, den) = (num / gcd, den / gcd);

        Ok(Expr::SRfrac { l: loc, num, den })
    }

    /// bool-expr: TRUE | FALSE
    fn parse_bool(&mut self) -> ParseResult<Expr> {
        let token = self.peek().clone();
        let b = match token.token_type {
            TokenType::True => {
                self.advance();
                true
            }
            TokenType::False => {
                self.advance();
                false
            }
            _ => return Err(ParseError::expected(TokenType::True, token)),
        };

        Ok(Expr::SBool {
            l: Loc::new(
                self.file_name.clone(),
                token.location.start_line,
                token.location.start_col,
                token.location.start_pos,
                token.location.end_line,
                token.location.end_col,
                token.location.end_pos,
            ),
            b,
        })
    }

    /// string-expr: STRING
    fn parse_str(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::String)?;

        Ok(Expr::SStr {
            l: Loc::new(
                self.file_name.clone(),
                token.location.start_line,
                token.location.start_col,
                token.location.start_pos,
                token.location.end_line,
                token.location.end_col,
                token.location.end_pos,
            ),
            s: token.value.clone(),
        })
    }

    /// id-expr: NAME
    fn parse_id_expr(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::Name)?;

        Ok(Expr::SId {
            l: Loc::new(
                self.file_name.clone(),
                token.location.start_line,
                token.location.start_col,
                token.location.start_pos,
                token.location.end_line,
                token.location.end_col,
                token.location.end_pos,
            ),
            id: Name::SName {
                l: Loc::new(
                    self.file_name.clone(),
                    token.location.start_line,
                    token.location.start_col,
                    token.location.start_pos,
                    token.location.end_line,
                    token.location.end_col,
                    token.location.end_pos,
                ),
                s: token.value.clone(),
            },
        })
    }

    /// Check if current token is a binary operator
    fn is_binop(&self) -> bool {
        matches!(
            self.peek().token_type,
            TokenType::Plus
                | TokenType::Dash
                | TokenType::Times
                | TokenType::Slash
                | TokenType::Leq
                | TokenType::Geq
                | TokenType::EqualEqual
                | TokenType::Spaceship
                | TokenType::EqualTilde
                | TokenType::Neq
                | TokenType::Lt
                | TokenType::Gt
                | TokenType::And
                | TokenType::Or
                | TokenType::Caret
        )
    }

    /// Check if current token is a check operator (is, raises, satisfies, violates)
    fn is_check_op(&self) -> bool {
        matches!(
            self.peek().token_type,
            TokenType::Is
                | TokenType::IsRoughly
                | TokenType::IsNot
                | TokenType::IsNotRoughly
                | TokenType::Satisfies
                | TokenType::Violates
                | TokenType::Raises
                | TokenType::RaisesOtherThan
                | TokenType::RaisesSatisfies
                | TokenType::RaisesViolates
        )
    }

    /// Parse a binary operator token and return the operator name
    fn parse_binop(&mut self) -> ParseResult<String> {
        let token = self.peek().clone();
        let op = match token.token_type {
            TokenType::Plus => "op+",
            TokenType::Dash => "op-",
            TokenType::Times => "op*",
            TokenType::Slash => "op/",
            TokenType::Leq => "op<=",
            TokenType::Geq => "op>=",
            TokenType::EqualEqual => "op==",
            TokenType::Spaceship => "op<=>",
            TokenType::EqualTilde => "op=~",
            TokenType::Neq => "op<>",
            TokenType::Lt => "op<",
            TokenType::Gt => "op>",
            TokenType::And => "opand",
            TokenType::Or => "opor",
            TokenType::Caret => "op^",
            _ => return Err(ParseError::unexpected(token)),
        };

        self.advance();
        Ok(op.to_string())
    }

    /// Parse a check operator token and return CheckOp enum
    fn parse_check_op(&mut self) -> ParseResult<CheckOp> {
        let token = self.advance().clone();
        let l = Loc::new(
            self.file_name.clone(),
            token.location.start_line,
            token.location.start_col,
            token.location.start_pos,
            token.location.end_line,
            token.location.end_col,
            token.location.end_pos,
        );

        let op = match token.token_type {
            TokenType::Is => CheckOp::SOpIs { l },
            TokenType::IsRoughly => CheckOp::SOpIsRoughly { l },
            TokenType::IsNot => CheckOp::SOpIsNot { l },
            TokenType::IsNotRoughly => CheckOp::SOpIsNotRoughly { l },
            TokenType::Satisfies => CheckOp::SOpSatisfies { l },
            TokenType::Violates => CheckOp::SOpSatisfiesNot { l }, // violates = satisfies-not
            TokenType::Raises => CheckOp::SOpRaises { l },
            TokenType::RaisesOtherThan => CheckOp::SOpRaisesOther { l },
            TokenType::RaisesSatisfies => CheckOp::SOpRaisesSatisfies { l },
            TokenType::RaisesViolates => CheckOp::SOpRaisesViolates { l },
            _ => return Err(ParseError::unexpected(token)),
        };

        Ok(op)
    }

    /// paren-expr: LPAREN expr RPAREN | PARENSPACE expr RPAREN
    /// Parenthesized expression (with whitespace before paren)
    fn parse_paren_expr(&mut self) -> ParseResult<Expr> {
        // Expect ParenSpace or LParen token
        let start = if self.matches(&TokenType::ParenSpace) {
            self.expect(TokenType::ParenSpace)?
        } else {
            self.expect(TokenType::LParen)?
        };

        let expr = self.parse_expr()?;
        let end = self.expect(TokenType::RParen)?;

        Ok(Expr::SParen {
            l: self.make_loc(&start, &end),
            expr: Box::new(expr),
        })
    }

    /// array-expr: LBRACK (expr COMMA)* RBRACK
    /// Array literal expression
    /// construct-expr: LBRACK construct-modifier binop-expr COLON trailing-opt-comma-binops RBRACK
    /// Construct expressions like [list: 1, 2, 3] or [lazy set: x, y]
    fn parse_construct_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::LBrack)?;

        // Check for optional LAZY modifier
        let modifier = if self.matches(&TokenType::Lazy) {
            self.advance();
            ConstructModifier::SConstructLazy
        } else {
            ConstructModifier::SConstructNormal
        };

        // Parse the constructor expression (like "list", "set", etc.)
        let constructor = self.parse_expr()?;

        // Expect colon
        self.expect(TokenType::Colon)?;

        // Parse comma-separated values (can be empty)
        let values = if self.matches(&TokenType::RBrack) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_expr())?
        };

        let end = self.expect(TokenType::RBrack)?;

        Ok(Expr::SConstruct {
            l: self.make_loc(&start, &end),
            modifier,
            constructor: Box::new(constructor),
            values: values.into_iter().map(Box::new).collect(),
        })
    }

    /// bracket-expr: expr LBRACK binop-expr RBRACK
    /// Bracket access expression like arr[0] or dict["key"]
    fn parse_bracket_expr(&mut self, obj: Expr) -> ParseResult<Expr> {
        let start = self.expect(TokenType::LBrack)?;

        // Parse the field expression (can be any expression)
        let field = self.parse_expr()?;

        let end = self.expect(TokenType::RBrack)?;

        // Get location from obj expression
        let obj_loc = match &obj {
            Expr::SNum { l, .. } => l.clone(),
            Expr::SBool { l, .. } => l.clone(),
            Expr::SStr { l, .. } => l.clone(),
            Expr::SId { l, .. } => l.clone(),
            Expr::SOp { l, .. } => l.clone(),
            Expr::SParen { l, .. } => l.clone(),
            Expr::SApp { l, .. } => l.clone(),
            Expr::SConstruct { l, .. } => l.clone(),
            Expr::SDot { l, .. } => l.clone(),
            Expr::SBracket { l, .. } => l.clone(),
            Expr::SObj { l, .. } => l.clone(),
            Expr::STuple { l, .. } => l.clone(),
            Expr::STupleGet { l, .. } => l.clone(),
            Expr::SLam { l, .. } => l.clone(),
            _ => self.current_loc(),
        };

        Ok(Expr::SBracket {
            l: Loc::new(
                self.file_name.clone(),
                obj_loc.start_line,
                obj_loc.start_column,
                obj_loc.start_char,
                end.location.end_line,
                end.location.end_col,
                end.location.end_pos,
            ),
            obj: Box::new(obj),
            field: Box::new(field),
        })
    }

    /// obj-expr: LBRACE obj-fields RBRACE | LBRACE RBRACE
    /// tuple-expr: LBRACE expr (SEMICOLON expr)+ RBRACE
    /// Disambiguates between object and tuple based on first separator
    /// Objects use colons: {x: 1, y: 2}
    /// Tuples use semicolons: {1; 2; 3}
    fn parse_obj_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::LBrace)?;

        // Check for empty object
        if self.matches(&TokenType::RBrace) {
            let end = self.expect(TokenType::RBrace)?;
            return Ok(Expr::SObj {
                l: self.make_loc(&start, &end),
                fields: Vec::new(),
            });
        }

        // Parse first element to determine if it's an object or tuple
        // We need to peek ahead to see if we get a colon or semicolon
        // For tuples, we expect: expr SEMICOLON
        // For objects, we expect: name COLON or REF name COLON

        // Try to determine the type by checking if the first token looks like an object field
        let is_tuple = if self.matches(&TokenType::Ref) ||
                          (self.matches(&TokenType::Name) && self.peek_ahead(1).token_type == TokenType::Colon) {
            // This looks like an object field (ref x: or name:)
            false
        } else {
            // Otherwise, parse an expression and check what follows
            // Save position to potentially backtrack
            let checkpoint = self.current;

            // Try parsing as expression
            match self.parse_expr() {
                Ok(_) => {
                    // Check what comes after the expression
                    let is_tuple = self.matches(&TokenType::Semi);
                    // Restore position to re-parse
                    self.current = checkpoint;
                    self.position = checkpoint;
                    is_tuple
                }
                Err(_) => {
                    // Failed to parse as expression, assume object
                    self.current = checkpoint;
                    self.position = checkpoint;
                    false
                }
            }
        };

        if is_tuple {
            self.parse_tuple_expr(start)
        } else {
            self.parse_obj_expr_fields(start)
        }
    }

    /// Parse object fields after determining it's an object
    fn parse_obj_expr_fields(&mut self, start: Token) -> ParseResult<Expr> {
        // Parse comma-separated fields with optional trailing comma
        let mut fields = Vec::new();
        loop {
            fields.push(self.parse_obj_field()?);

            if !self.matches(&TokenType::Comma) {
                break;
            }
            self.advance(); // consume comma

            // Check for trailing comma (followed by closing brace)
            if self.matches(&TokenType::RBrace) {
                break;
            }
        }

        let end = self.expect(TokenType::RBrace)?;

        Ok(Expr::SObj {
            l: self.make_loc(&start, &end),
            fields,
        })
    }

    /// tuple-expr: LBRACE expr (SEMICOLON expr)+ RBRACE
    /// Parse tuple expression: {1; 2; 3}
    fn parse_tuple_expr(&mut self, start: Token) -> ParseResult<Expr> {
        let mut fields = Vec::new();

        // Parse first expression
        fields.push(Box::new(self.parse_expr()?));

        // Parse remaining expressions (semicolon-separated)
        while self.matches(&TokenType::Semi) {
            self.advance(); // consume semicolon

            // Check for trailing semicolon (followed by closing brace)
            if self.matches(&TokenType::RBrace) {
                break;
            }

            fields.push(Box::new(self.parse_expr()?));
        }

        let end = self.expect(TokenType::RBrace)?;

        Ok(Expr::STuple {
            l: self.make_loc(&start, &end),
            fields,
        })
    }

    /// obj-field: key COLON binop-expr
    ///          | REF key [COLONCOLON ann] COLON binop-expr
    ///          | METHOD key fun-header (BLOCK|COLON) doc-string block where-clause END
    /// Parse a single object field (data, mutable, or method)
    fn parse_obj_field(&mut self) -> ParseResult<Member> {
        let token = self.peek().clone();

        match token.token_type {
            TokenType::Ref => {
                // Mutable field: REF key [COLONCOLON ann] COLON binop-expr
                let start = self.expect(TokenType::Ref)?;
                let name_token = self.expect(TokenType::Name)?;
                let name = name_token.value.clone();

                // Optional type annotation
                let ann = if self.matches(&TokenType::ColonColon) {
                    self.expect(TokenType::ColonColon)?;
                    self.parse_ann()?
                } else {
                    Ann::ABlank // No annotation
                };

                self.expect(TokenType::Colon)?;
                let value = self.parse_expr()?;

                Ok(Member::SMutableField {
                    l: self.make_loc(&start, &name_token),
                    name,
                    ann,
                    value: Box::new(value),
                })
            }
            TokenType::Method => {
                // Method field: METHOD key fun-header (BLOCK|COLON) doc-string block where-clause END
                self.parse_method_field()
            }
            TokenType::Name => {
                // Data field: key COLON binop-expr
                let name_token = self.expect(TokenType::Name)?;
                let name = name_token.value.clone();
                self.expect(TokenType::Colon)?;
                let value = self.parse_expr()?;

                Ok(Member::SDataField {
                    l: self.make_loc(&name_token, &name_token),
                    name,
                    value: Box::new(value),
                })
            }
            _ => Err(ParseError::unexpected(token)),
        }
    }

    /// method-field: METHOD key fun-header (BLOCK|COLON) doc-string block where-clause END
    /// Parse a method field in an object
    /// Example: method _plus(self, other): self.arr end
    fn parse_method_field(&mut self) -> ParseResult<Member> {
        let start = self.expect(TokenType::Method)?;

        // Parse method name
        let name_token = self.expect(TokenType::Name)?;
        let name = name_token.value.clone();

        // Parse fun-header: ty-params args return-ann
        // For simplicity, we skip ty-params (type parameters) for now

        // Expect opening paren (can be LParen or ParenSpace or ParenNoSpace)
        let paren_token = self.peek().clone();
        match paren_token.token_type {
            TokenType::LParen | TokenType::ParenSpace | TokenType::ParenNoSpace => {
                self.advance();
            }
            _ => {
                return Err(ParseError::expected(TokenType::LParen, paren_token));
            }
        }

        // Parse parameters (comma-separated bindings)
        let args = if self.matches(&TokenType::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_bind())?
        };

        // params is for type parameters (e.g., <T, U>), not function parameters
        // We don't parse type parameters yet, so this is always empty
        let params: Vec<Name> = Vec::new();

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Optional type annotation: -> ann
        let ann = if self.matches(&TokenType::ThinArrow) {
            self.expect(TokenType::ThinArrow)?;
            self.parse_ann()?
        } else {
            Ann::ABlank
        };

        // Parse body separator (COLON or BLOCK)
        let blocky = if self.matches(&TokenType::Block) {
            self.advance();
            true
        } else {
            self.expect(TokenType::Colon)?;
            false
        };

        // Parse doc string (usually empty, skip for now)
        let doc = String::new();

        // Parse method body (statements until END or WHERE)
        let mut body_stmts = Vec::new();
        while !self.matches(&TokenType::End)
            && !self.matches(&TokenType::Where)
            && !self.is_at_end()
        {
            let stmt = self.parse_expr()?;
            body_stmts.push(Box::new(stmt));
        }

        // Skip where clause if present
        let check = if self.matches(&TokenType::Where) {
            self.advance();
            // Parse where clause body
            let mut where_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                where_stmts.push(Box::new(stmt));
            }
            Some(Box::new(Expr::SBlock {
                l: self.current_loc(),
                stmts: where_stmts,
            }))
        } else {
            None
        };

        let end = self.expect(TokenType::End)?;

        // Wrap body in SBlock
        let body = Box::new(Expr::SBlock {
            l: self.current_loc(),
            stmts: body_stmts,
        });

        let check_loc = check.as_ref().map(|c| match c.as_ref() {
            Expr::SBlock { l, .. } => l.clone(),
            _ => self.current_loc(),
        });

        Ok(Member::SMethodField {
            l: self.make_loc(&start, &end),
            name,
            params,
            args,
            ann,
            doc,
            body,
            check_loc,
            check,
            blocky,
        })
    }

    /// app-expr: expr PARENNOSPACE (expr COMMA)* RPAREN
    /// Function application (no whitespace before paren)
    fn parse_app_expr(&mut self, base: Expr) -> ParseResult<Expr> {
        // Get location from base expression
        let base_loc = match &base {
            Expr::SNum { l, .. } => l.clone(),
            Expr::SBool { l, .. } => l.clone(),
            Expr::SStr { l, .. } => l.clone(),
            Expr::SId { l, .. } => l.clone(),
            Expr::SOp { l, .. } => l.clone(),
            Expr::SParen { l, .. } => l.clone(),
            Expr::SApp { l, .. } => l.clone(),
            Expr::SConstruct { l, .. } => l.clone(),
            Expr::SDot { l, .. } => l.clone(),
            Expr::SBracket { l, .. } => l.clone(),
            Expr::SObj { l, .. } => l.clone(),
            Expr::STuple { l, .. } => l.clone(),
            Expr::STupleGet { l, .. } => l.clone(),
            Expr::SLam { l, .. } => l.clone(),
            Expr::SBlock { l, .. } => l.clone(),
            Expr::SUserBlock { l, .. } => l.clone(),
            Expr::SIf { l, .. } => l.clone(),
            Expr::SIfElse { l, .. } => l.clone(),
            Expr::SWhen { l, .. } => l.clone(),
            Expr::SFor { l, .. } => l.clone(),
                        Expr::SLetExpr { l, .. } => l.clone(),
                Expr::SLet { l, .. } => l.clone(),
                Expr::SVar { l, .. } => l.clone(),
                Expr::SAssign { l, .. } => l.clone(),
            _ => self.current_loc(),
        };

        self.expect(TokenType::ParenNoSpace)?;

        // Parse arguments as comma-separated list
        let args = if self.matches(&TokenType::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_expr().map(Box::new))?
        };

        let end = self.expect(TokenType::RParen)?;

        Ok(Expr::SApp {
            l: Loc::new(
                self.file_name.clone(),
                base_loc.start_line,
                base_loc.start_column,
                base_loc.start_char,
                end.location.end_line,
                end.location.end_col,
                end.location.end_pos,
            ),
            _fun: Box::new(base),
            args,
        })
    }
}

// ============================================================================
// SECTION 7: Control Flow Parsing
// ============================================================================

impl Parser {
    /// block-expr: BLOCK COLON stmts END
    /// Parses user-defined block expressions like: block: 5 end, block: x = 1 x + 2 end
    fn parse_block_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Block)?;

        // Parse statements until we hit 'end'
        let mut stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_expr()?;
            stmts.push(Box::new(stmt));
        }

        let end = self.expect(TokenType::End)?;

        // Create the SBlock wrapper
        let block_body = Expr::SBlock {
            l: self.make_loc(&start, &end),
            stmts,
        };

        // Wrap in SUserBlock
        Ok(Expr::SUserBlock {
            l: self.make_loc(&start, &end),
            body: Box::new(block_body),
        })
    }

    /// if-expr: IF expr COLON body (ELSE-IF expr COLON body)* (ELSE-COLON body)? END
    /// Parses if expressions like: if true: 1 else: 2 end
    fn parse_if_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::If)?;

        // Parse the first branch (always present)
        let test = self.parse_expr()?;
        self.expect(TokenType::Colon)?;

        // Parse the body (statements until else/elseif/end)
        let mut then_stmts = Vec::new();
        while !self.matches(&TokenType::ElseColon)
            && !self.matches(&TokenType::ElseIf)
            && !self.matches(&TokenType::End)
            && !self.is_at_end()
        {
            let stmt = self.parse_expr()?;
            then_stmts.push(Box::new(stmt));
        }

        // Create the body as an SBlock
        let body = Expr::SBlock {
            l: self.current_loc(), // TODO: proper location
            stmts: then_stmts,
        };

        // Create the first branch
        let mut branches = vec![IfBranch {
            node_type: "s-if-branch".to_string(),
            l: self.current_loc(), // TODO: proper location
            test: Box::new(test),
            body: Box::new(body),
        }];

        // Parse optional else-if branches
        while self.matches(&TokenType::ElseIf) {
            self.advance();
            let test = self.parse_expr()?;
            self.expect(TokenType::Colon)?;

            let mut elseif_stmts = Vec::new();
            while !self.matches(&TokenType::ElseColon)
                && !self.matches(&TokenType::ElseIf)
                && !self.matches(&TokenType::End)
                && !self.is_at_end()
            {
                let stmt = self.parse_expr()?;
                elseif_stmts.push(Box::new(stmt));
            }

            let body = Expr::SBlock {
                l: self.current_loc(), // TODO: proper location
                stmts: elseif_stmts,
            };

            branches.push(IfBranch {
                node_type: "s-if-branch".to_string(),
                l: self.current_loc(), // TODO: proper location
                test: Box::new(test),
                body: Box::new(body),
            });
        }

        // Parse optional else clause
        let else_expr = if self.matches(&TokenType::ElseColon) {
            self.advance();

            let mut else_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                else_stmts.push(Box::new(stmt));
            }

            Some(Box::new(Expr::SBlock {
                l: self.current_loc(), // TODO: proper location
                stmts: else_stmts,
            }))
        } else {
            None
        };

        let end = self.expect(TokenType::End)?;
        let loc = self.make_loc(&start, &end);

        // Return SIfElse if there's an else clause, otherwise SIf
        if let Some(else_body) = else_expr {
            Ok(Expr::SIfElse {
                l: loc,
                branches,
                _else: else_body,
                blocky: false,
            })
        } else {
            Ok(Expr::SIf {
                l: loc,
                branches,
                blocky: false,
            })
        }
    }

    /// for-expr: FOR expr PARENNOSPACE [for-bind (COMMA for-bind)*] RPAREN return-ann (BLOCK|COLON) block END
    /// for-bind: binding FROM binop-expr
    /// Parses for expressions like: for map(x from lst): x + 1 end
    fn parse_for_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::For)?;

        // Parse the iterator expression (e.g., "map" or "lists.map2")
        // This needs to be a full expression to handle dot access
        // We'll manually parse just enough to get the iterator before the paren
        let mut iterator = self.parse_prim_expr()?;

        // Handle dot access for things like lists.map2
        while self.matches(&TokenType::Dot) {
            self.advance();
            let field_token = self.parse_field_name()?;

            let start_loc = match &iterator {
                Expr::SId { l, .. } => l.clone(),
                Expr::SDot { l, .. } => l.clone(),
                _ => self.current_loc(),
            };

            iterator = Expr::SDot {
                l: Loc::new(
                    self.file_name.clone(),
                    start_loc.start_line,
                    start_loc.start_column,
                    start_loc.start_char,
                    field_token.location.end_line,
                    field_token.location.end_col,
                    field_token.location.end_pos,
                ),
                obj: Box::new(iterator),
                field: field_token.value.clone(),
            };
        }

        // Expect opening paren (no space)
        self.expect(TokenType::ParenNoSpace)?;

        // Parse for-bindings: binding FROM expr (COMMA binding FROM expr)*
        let mut bindings = Vec::new();

        if !self.matches(&TokenType::RParen) {
            loop {
                // Parse the binding (name with optional type annotation)
                let bind = self.parse_bind()?;

                // Expect FROM keyword
                self.expect(TokenType::From)?;

                // Parse the value expression
                let value = Box::new(self.parse_expr()?);

                // Create ForBind
                bindings.push(ForBind {
                    node_type: "s-for-bind".to_string(),
                    l: self.current_loc(), // TODO: proper location from bind to value
                    bind,
                    value,
                });

                // Check for comma (more bindings) or close paren
                if self.matches(&TokenType::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Parse optional return annotation (for now, we'll use a-blank)
        let ann = Ann::ABlank;

        // Parse body separator (COLON or BLOCK)
        let blocky = if self.matches(&TokenType::Block) {
            self.advance();
            true
        } else {
            self.expect(TokenType::Colon)?;
            false
        };

        // Parse the body (statements until END)
        let mut body_stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_expr()?;
            body_stmts.push(Box::new(stmt));
        }

        // Create the body as an SBlock
        let body = Box::new(Expr::SBlock {
            l: self.current_loc(), // TODO: proper location
            stmts: body_stmts,
        });

        let end = self.expect(TokenType::End)?;
        let loc = self.make_loc(&start, &end);

        Ok(Expr::SFor {
            l: loc,
            iterator: Box::new(iterator),
            bindings,
            ann,
            body,
            blocky,
        })
    }

    /// let-expr: LET bind = expr
    ///          | LET bind = expr BLOCK body END
    /// Parses let bindings: x = 5
    fn parse_let_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Let)?;

        // Parse binding: name [:: type]
        let bind = self.parse_bind()?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        // Create LetBind
        let let_bind = LetBind::SLetBind {
            l: self.current_loc(),
            b: bind.clone(),
            value: Box::new(value.clone()),
        };

        // Check if there's a block body
        let body = if self.matches(&TokenType::Block) {
            self.expect(TokenType::Block)?;
            self.expect(TokenType::Colon)?;

            // Parse block body
            let mut body_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                body_stmts.push(Box::new(stmt));
            }

            self.expect(TokenType::End)?;

            Expr::SBlock {
                l: self.current_loc(),
                stmts: body_stmts,
            }
        } else {
            // No explicit body, just use the value
            value.clone()
        };

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        Ok(Expr::SLetExpr {
            l: self.make_loc(&start, &end),
            binds: vec![let_bind],
            body: Box::new(body),
            blocky: false,
        })
    }

    /// var-expr: VAR bind := expr
    /// Parses mutable variable bindings: var x := 5
    fn parse_var_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Var)?;

        // Parse binding: name [:: type]
        let bind = self.parse_bind()?;

        // Expect :=
        self.expect(TokenType::ColonEquals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        // Create VarBind
        let var_bind = LetBind::SVarBind {
            l: self.current_loc(),
            b: bind.clone(),
            value: Box::new(value.clone()),
        };

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        // Var expressions are represented as SLetExpr with SVarBind
        Ok(Expr::SLetExpr {
            l: self.make_loc(&start, &end),
            binds: vec![var_bind],
            body: Box::new(value), // Use the value as the body
            blocky: false,
        })
    }

    /// Implicit let binding: x = value (no "let" keyword)
    /// Creates an s-let statement (not s-let-expr)
    fn parse_implicit_let_expr(&mut self) -> ParseResult<Expr> {
        let start = self.peek().clone();

        // Parse binding: name [:: type]
        let bind = self.parse_bind()?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        Ok(Expr::SLet {
            l: self.make_loc(&start, &end),
            name: bind,
            value: Box::new(value),
            keyword_val: false,
        })
    }

    /// Implicit var binding: x := value (no "var" keyword)
    /// This is actually an assignment (s-assign), not a var declaration
    fn parse_implicit_var_expr(&mut self) -> ParseResult<Expr> {
        let start = self.peek().clone();

        // Parse name (just the identifier, no type annotation)
        let name = self.parse_name()?;

        // Expect :=
        self.expect(TokenType::ColonEquals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        // This is an assignment, not a var declaration
        Ok(Expr::SAssign {
            l: self.make_loc(&start, &end),
            id: name,
            value: Box::new(value),
        })
    }

    /// cases-expr: CASES (type) expr: branches ... END
    fn parse_cases_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_cases_expr")
    }

    /// when-expr: WHEN expr: block END
    /// Parses when expressions like: when true: print("yes") end
    fn parse_when_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::When)?;

        // Parse the test expression
        let test = self.parse_expr()?;

        // Expect colon
        self.expect(TokenType::Colon)?;

        // Parse the block (statements until end)
        let mut block_stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_expr()?;
            block_stmts.push(Box::new(stmt));
        }

        let end = self.expect(TokenType::End)?;
        let loc = self.make_loc(&start, &end);

        // Create the block
        let block = Expr::SBlock {
            l: self.current_loc(),
            stmts: block_stmts,
        };

        Ok(Expr::SWhen {
            l: loc,
            test: Box::new(test),
            block: Box::new(block),
            blocky: false,
        })
    }
}

// ============================================================================
// SECTION 8: Function Parsing
// ============================================================================

impl Parser {
    /// fun-expr: FUN name<typarams>(args) ann: doc body where END
    fn parse_fun_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_fun_expr")
    }

    /// lambda-expr: LAM LPAREN [args] RPAREN [COLONCOLON ann] COLON body END
    /// Parses lambda expressions like: lam(): 5 end, lam(x): x + 1 end
    fn parse_lambda_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Lam)?;

        // Expect opening paren (can be LParen or ParenSpace after lam keyword)
        if self.matches(&TokenType::LParen) {
            self.expect(TokenType::LParen)?;
        } else {
            self.expect(TokenType::ParenSpace)?;
        }

        // Parse parameters (comma-separated bindings)
        let args = if self.matches(&TokenType::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_bind())?
        };

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Optional type annotation: :: ann
        let ann = if self.matches(&TokenType::ColonColon) {
            self.expect(TokenType::ColonColon)?;
            self.parse_ann()?
        } else {
            Ann::ABlank
        };

        // Expect colon before body
        self.expect(TokenType::Colon)?;

        // Parse body expression
        let body_expr = self.parse_expr()?;

        // Expect end keyword
        let end = self.expect(TokenType::End)?;

        // Wrap body in SBlock (required by Pyret AST format)
        let body = Expr::SBlock {
            l: self.current_loc(),
            stmts: vec![Box::new(body_expr)],
        };

        Ok(Expr::SLam {
            l: self.make_loc(&start, &end),
            name: String::new(), // Anonymous lambda
            params: Vec::new(),  // Empty params (used for type parameters)
            args,
            ann,
            doc: String::new(),
            body: Box::new(body),
            check_loc: None,
            check: None,
            blocky: false,
        })
    }

    /// method-expr: METHOD(args) ann: doc body where END
    fn parse_method_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_method_expr")
    }

    /// Shared helper for function headers
    fn parse_fun_header(&mut self) -> ParseResult<(Vec<Name>, Vec<Bind>, Ann)> {
        todo!("Implement parse_fun_header")
    }

    /// where-clause: WHERE: body END
    fn parse_where_clause(&mut self) -> ParseResult<Option<Expr>> {
        todo!("Implement parse_where_clause")
    }
}

// ============================================================================
// SECTION 9: Data Definition Parsing
// ============================================================================

impl Parser {
    /// data-expr: DATA name<typarams>: variants sharing where END
    fn parse_data_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_data_expr")
    }

    /// data-variant: name(members) | name
    fn parse_variant(&mut self) -> ParseResult<Variant> {
        todo!("Implement parse_variant")
    }

    /// data-with: with: members END
    fn parse_data_with(&mut self) -> ParseResult<Vec<Member>> {
        todo!("Implement parse_data_with")
    }
}

// ============================================================================
// SECTION 10: Table Operation Parsing
// ============================================================================

impl Parser {
    /// table-expr: table: headers row: ... end
    fn parse_table_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_table_expr")
    }

    /// table-select: select columns from table
    fn parse_table_select(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_table_select")
    }

    /// load-table-expr: load-table: headers source: ... end
    fn parse_load_table_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_load_table_expr")
    }
}

// ============================================================================
// SECTION 11: Check/Test Parsing
// ============================================================================

impl Parser {
    /// check-expr: CHECK name: body END
    fn parse_check_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_check_expr")
    }

    /// check-test: expr is expr | expr raises expr | ...
    fn parse_check_test(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_check_test")
    }

    /// spy-stmt: spy: contents END
    fn parse_spy_stmt(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_spy_stmt")
    }
}

// ============================================================================
// SECTION 12: Helper Methods
// ============================================================================

impl Parser {
    /// Convert a decimal string to a rational (numerator, denominator)
    /// For example: "3.14" -> (157, 50), "2.5" -> (5, 2)
    fn decimal_to_rational(decimal_str: &str) -> Result<(i64, i64), String> {
        // Remove leading + or - sign
        let (sign, num_str) = if let Some(stripped) = decimal_str.strip_prefix('-') {
            (-1, stripped)
        } else if let Some(stripped) = decimal_str.strip_prefix('+') {
            (1, stripped)
        } else {
            (1, decimal_str)
        };

        // Check if there's a decimal point
        if let Some(dot_pos) = num_str.find('.') {
            let integer_part = &num_str[..dot_pos];
            let decimal_part = &num_str[dot_pos + 1..];

            // Parse integer and decimal parts
            let int_val: i64 = if integer_part.is_empty() {
                0
            } else {
                integer_part.parse().map_err(|_| "Invalid integer part")?
            };

            let dec_val: i64 = decimal_part.parse().map_err(|_| "Invalid decimal part")?;
            let dec_places = decimal_part.len() as u32;
            let denominator = 10_i64.pow(dec_places);

            // Calculate numerator: (integer_part * denominator) + decimal_part
            let numerator = sign * (int_val * denominator + dec_val);

            // Simplify the fraction
            let gcd = Self::gcd(numerator.abs(), denominator);
            Ok((numerator / gcd, denominator / gcd))
        } else {
            // No decimal point - it's an integer
            let int_val: i64 = num_str.parse().map_err(|_| "Invalid integer")?;
            Ok((sign * int_val, 1))
        }
    }

    /// Calculate greatest common divisor using Euclid's algorithm
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Parse comma-separated list
    fn parse_comma_list<T, F>(&mut self, parser: F) -> ParseResult<Vec<T>>
    where
        F: Fn(&mut Self) -> ParseResult<T>,
    {
        let mut items = Vec::new();

        loop {
            items.push(parser(self)?);

            if !self.matches(&TokenType::Comma) {
                break;
            }
            self.advance(); // consume comma
        }

        Ok(items)
    }

    /// Parse optional element
    fn parse_optional<T, F>(&mut self, parser: F) -> ParseResult<Option<T>>
    where
        F: Fn(&mut Self) -> ParseResult<T>,
    {
        match parser(self) {
            Ok(value) => Ok(Some(value)),
            Err(_) => Ok(None),
        }
    }

    /// Parse list until END token
    fn parse_until_end<T, F>(&mut self, parser: F) -> ParseResult<Vec<T>>
    where
        F: Fn(&mut Self) -> ParseResult<T>,
    {
        let mut items = Vec::new();

        while !self.matches(&TokenType::End) && !self.is_at_end() {
            items.push(parser(self)?);
        }

        Ok(items)
    }

    /// Parse NAME token into Name AST node
    fn parse_name(&mut self) -> ParseResult<Name> {
        let token = self.expect(TokenType::Name)?;
        let loc = Loc::new(
            self.file_name.clone(),
            token.location.start_line,
            token.location.start_col,
            token.location.start_pos,
            token.location.end_line,
            token.location.end_col,
            token.location.end_pos,
        );
        Ok(Name::SName {
            l: loc,
            s: token.value,
        })
    }
}
