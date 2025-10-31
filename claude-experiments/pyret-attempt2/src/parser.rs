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
    file_name: String,
}

impl Parser {
    pub fn new(tokens: Vec<Token>, file_name: String) -> Self {
        Parser {
            tokens,
            current: 0,
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
        }
        &self.tokens[self.current - 1]
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
}

// ============================================================================
// SECTION 2: Program & Top-Level
// ============================================================================

impl Parser {
    /// program: prelude block
    pub fn parse_program(&mut self) -> ParseResult<Program> {
        todo!("Implement parse_program")
    }

    /// prelude: [use-stmt] (provide-stmt|import-stmt)*
    fn parse_prelude(&mut self) -> ParseResult<()> {
        todo!("Implement parse_prelude")
    }

    /// block: stmt*
    fn parse_block(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_block")
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
    /// binding: name-binding | tuple-binding
    fn parse_bind(&mut self) -> ParseResult<Bind> {
        todo!("Implement parse_bind")
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
                // Dot access
                let _dot_token = self.expect(TokenType::Dot)?;
                let field_token = self.expect(TokenType::Name)?;

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
                    let field_token = self.expect(TokenType::Name)?;

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

    /// prim-expr: num-expr | frac-expr | rfrac-expr | bool-expr | string-expr | id-expr | paren-expr | array-expr
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
    /// if-expr: IF expr: block ... END
    fn parse_if_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_if_expr")
    }

    /// cases-expr: CASES (type) expr: branches ... END
    fn parse_cases_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_cases_expr")
    }

    /// when-expr: WHEN expr: block END
    fn parse_when_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_when_expr")
    }

    /// for-expr: FOR expr(bindings) ann: body END
    fn parse_for_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_for_expr")
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

    /// lambda-expr: LAM(args) ann: doc body where END
    fn parse_lambda_expr(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_lambda_expr")
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
