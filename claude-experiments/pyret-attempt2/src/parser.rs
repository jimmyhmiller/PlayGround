//! Pyret Parser
//!
//! Hand-written recursive descent parser implementing the complete Pyret grammar.
//! Each BNF rule maps to a corresponding parse function.
//!
//! Reference: /pyret-lang/src/js/base/pyret-grammar.bnf

use crate::ast::*;
use crate::error::{ParseError, ParseResult};
use crate::tokenizer::{Token, TokenType};

// Type alias for complex return type
type PreludeResult = (Option<Use>, Provide, ProvideTypes, Vec<ProvideBlock>, Vec<Import>);

// ============================================================================
// SECTION 1: Parser Struct and Core Methods
// ============================================================================

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    _file_id: FileId,
}

impl Parser {
    pub fn new(tokens: Vec<Token>, file_id: FileId) -> Self {
        // Filter out comments and block comments as they're not part of the AST
        let tokens: Vec<Token> = tokens
            .into_iter()
            .filter(|t| !matches!(t.token_type, TokenType::Comment | TokenType::BlockComment | TokenType::UnterminatedBlockComment))
            .collect();

        Parser {
            tokens,
            current: 0,
            _file_id: file_id,
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

    /// Get the last consumed token, or fallback if no tokens have been consumed
    fn last_token(&self, fallback: &Token) -> Token {
        if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            fallback.clone()
        }
    }

    // ========== Checkpointing for Backtracking ==========

    fn checkpoint(&self) -> usize {
        self.current
    }

    fn restore(&mut self, checkpoint: usize) {
        self.current = checkpoint;
    }

    // ========== Location Tracking Helpers ==========

    /// Get current location without cloning the entire token
    fn start_loc(&self) -> Loc {
        self.peek().location
    }

    /// Get location of the last consumed token
    fn last_loc(&self) -> Loc {
        if self.current > 0 {
            self.tokens[self.current - 1].location
        } else {
            self.peek().location
        }
    }

    /// Create a span from start to the last consumed token
    fn make_span(&self, start: Loc) -> Loc {
        start.span(self.last_loc())
    }

    /// Compute location for a block from its statements
    /// If block is empty, use fallback location (typically current token)
    fn block_loc(&self, stmts: &[Box<Expr>], fallback: Loc) -> Loc {
        if stmts.is_empty() {
            fallback
        } else {
            let first = stmts.first().unwrap().get_loc();
            let last = stmts.last().unwrap().get_loc();
            first.span(*last)
        }
    }

    // ========== Token Matching Helpers ==========

    /// Accept any form of left paren (LParen, ParenSpace, ParenNoSpace)
    fn expect_any_lparen(&mut self) -> ParseResult<Token> {
        let token_type = self.peek().token_type.clone();
        if matches!(token_type, TokenType::LParen | TokenType::ParenSpace | TokenType::ParenNoSpace) {
            Ok(self.advance().clone())
        } else {
            Err(ParseError::expected(TokenType::LParen, self.peek().clone()))
        }
    }

    // ========== Common Parsing Patterns ==========

    /// Parse optional generic type parameters: <T, U, V>
    /// Returns empty vec if no type parameters present
    fn parse_opt_type_params(&mut self) -> ParseResult<Vec<Name>> {
        if self.matches(&TokenType::Lt) || self.matches(&TokenType::LtNoSpace) {
            self.advance();
            let params = self.parse_comma_list(|p| p.parse_name())?;
            self.expect(TokenType::Gt)?;
            Ok(params)
        } else {
            Ok(Vec::new())
        }
    }

    /// Parse optional doc string: doc: "string"
    /// Returns empty string if no doc present
    fn parse_opt_doc_string(&mut self) -> ParseResult<String> {
        if self.matches(&TokenType::Doc) {
            self.advance();
            let doc_token = self.expect(TokenType::String)?;
            Ok(doc_token.value.clone())
        } else {
            Ok(String::new())
        }
    }

    /// Parse block separator (BLOCK or COLON)
    /// Returns true if block: was used, false if : was used
    fn parse_block_separator(&mut self) -> ParseResult<bool> {
        if self.matches(&TokenType::Block) {
            self.advance();
            Ok(true)
        } else {
            self.expect(TokenType::Colon)?;
            Ok(false)
        }
    }

    /// Parse optional return type annotation: -> Type
    /// Returns ABlank if no annotation present
    fn parse_opt_return_ann(&mut self) -> ParseResult<Ann> {
        if self.matches(&TokenType::ThinArrow) {
            self.advance();
            self.parse_ann()
        } else {
            Ok(Ann::ABlank)
        }
    }

    // ========== Name Helpers ==========

    /// Convert a token to a Name::SName
    fn token_to_name(&self, token: &Token) -> Name {
        Name::SName {
            l: token.location,
            s: token.value.clone(),
        }
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

    /// Keywords that can be used as field names in Pyret
    const FIELD_NAME_KEYWORDS: &'static [TokenType] = &[
        TokenType::Method,
        TokenType::Fun,
        TokenType::Var,
        TokenType::Let,
        TokenType::Letrec,
        TokenType::Rec,
        TokenType::Data,
        TokenType::If,
        TokenType::Else,
        TokenType::ElseIf,
        TokenType::ElseColon,
        TokenType::When,
        TokenType::Block,
        TokenType::For,
        TokenType::From,
        TokenType::End,
        TokenType::Check,
        TokenType::CheckColon,
        TokenType::Where,
        TokenType::Import,
        TokenType::Provide,
        TokenType::ProvideColon,
        TokenType::ProvideTypes,
        TokenType::Include,
        TokenType::Sharing,
        TokenType::Shadow,
        TokenType::Type,
        TokenType::TypeLet,
        TokenType::Newtype,
        TokenType::Doc,
        TokenType::Cases,
        TokenType::Ask,
        TokenType::OtherwiseColon,
        TokenType::ThenColon,
        TokenType::Spy,
        TokenType::Reactor,
        TokenType::Table,
        TokenType::TableExtend,
        TokenType::TableExtract,
        TokenType::TableFilter,
        TokenType::TableOrder,
        TokenType::TableSelect,
        TokenType::TableUpdate,
        TokenType::LoadTable,
        TokenType::Sanitize,
        TokenType::Ref,
        TokenType::With,
        TokenType::Use,
        TokenType::Using,
        TokenType::Module,
        TokenType::Lam,
        TokenType::Lazy,
        TokenType::Row,
        TokenType::SourceColon,
        TokenType::Examples,
        TokenType::ExamplesColon,
        TokenType::Do,
        TokenType::Of,
        TokenType::By,
        TokenType::Hiding,
        TokenType::Ascending,
        TokenType::Descending,
    ];

    /// Check if a token type is a keyword that can be used as a field name
    fn is_keyword(&self, token_type: &TokenType) -> bool {
        Self::FIELD_NAME_KEYWORDS.contains(token_type)
    }

    // ========== Helper: Parse hiding(...) clause ==========

    /// Parse optional `hiding(name1, name2, ...)` clause
    /// Returns empty vector if no hiding clause is present
    fn parse_hiding_clause(&mut self) -> ParseResult<Vec<Name>> {
        if !self.matches(&TokenType::Hiding) {
            return Ok(Vec::new());
        }

        self.advance(); // consume HIDING

        // Accept any form of left paren
        self.expect_any_lparen()?;

        let names = self.parse_comma_list(|p| p.parse_name())?;
        self.expect(TokenType::RParen)?;

        Ok(names)
    }
}

// ============================================================================
// SECTION 2: Program & Top-Level
// ============================================================================

impl Parser {
    /// program: prelude block
    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let start = self.start_loc();

        // Parse prelude (imports, provides, etc.)
        let (_use, _provide, provided_types, provides, imports) = self.parse_prelude()?;

        // Parse program body (statement block)
        let body = self.parse_block()?;

        // Ensure we've consumed all tokens
        if !self.is_at_end() {
            return Err(ParseError::general(
                self.peek(),
                "Unexpected tokens after program end",
            ));
        }

        Ok(Program::new(
            self.make_span(start),
            _use,
            _provide,
            provided_types,
            provides,
            imports,
            body,
        ))
    }

    /// prelude: [use-stmt] (provide-stmt|import-stmt)*
    fn parse_prelude(&mut self) -> ParseResult<PreludeResult> {
        let mut _use = None;

        let mut provides = Vec::new();
        let mut imports = Vec::new();
        let mut _provide = Provide::SProvideNone {
            l: self.peek().location,
        };
        let mut provided_types = ProvideTypes::SProvideTypesNone {
            l: self.peek().location,
        };

        // Parse use statement (can only appear once, at the beginning)
        if self.matches(&TokenType::Use) {
            _use = Some(self.parse_use_stmt()?);
        }

        // Parse provide and import statements in any order
        // per grammar: (provide-stmt|import-stmt)*
        loop {
            if self.matches(&TokenType::ProvideColon) {
                provides.push(self.parse_provide_block()?);
            } else if self.matches(&TokenType::Provide) {
                // Peek ahead to check if this is "provide from" (which creates a ProvideBlock)
                // or a regular provide statement
                if self.peek_ahead(1).token_type == TokenType::From {
                    provides.push(self.parse_provide_from_block()?);
                } else {
                    _provide = self.parse_provide_stmt()?;
                }
            } else if self.matches(&TokenType::ProvideTypes) {
                provided_types = self.parse_provide_types_stmt()?;
            } else if self.matches(&TokenType::Import) || self.matches(&TokenType::Include) {
                imports.push(self.parse_import_stmt()?);
            } else {
                // No more prelude statements
                break;
            }
        }

        Ok((_use, _provide, provided_types, provides, imports))
    }

    /// block: stmt*
    /// Parses a sequence of statements and returns an SBlock expression
    fn parse_block(&mut self) -> ParseResult<Expr> {
        let start = self.start_loc();
        let mut stmts = Vec::new();

        // Parse statements until EOF or until we can't parse any more
        while !self.is_at_end() {
            // Try to parse a statement
            match self.parse_block_statement() {
                Ok(stmt) => stmts.push(Box::new(stmt)),
                Err(_) => break, // Stop if we can't parse
            }
        }

        Ok(Expr::SBlock {
            l: self.make_span(start),
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
        let start = self.expect(TokenType::Use)?;
        let name = self.parse_name()?;
        let module = self.parse_import_source()?;

        let end = self.last_token(&start);

        Ok(Use {
            node_type: "s-use".to_string(),
            l: start.location.span(end.location),
            name,
            module,
        })
    }

    /// import-stmt: INCLUDE | IMPORT | ...
    /// Parses import statements like:
    /// - import module as name
    /// - include module
    /// - import { field1, field2 } from module
    fn parse_import_stmt(&mut self) -> ParseResult<Import> {
        let start = self.start_loc();
        let start_token_type = self.peek().token_type.clone();

        match start_token_type {
            TokenType::Import => {
                self.advance(); // consume IMPORT

                // We need to peek ahead to determine which import syntax we have:
                // 1. import x, y from source (comma-separated names)
                // 2. import source as name (regular import)

                // Try to parse the first name
                let first_name = self.parse_name()?;

                // Check what comes after the first name
                if self.matches(&TokenType::Comma) {
                    // import x, y, z from source - use comma list parser
                    let mut fields = vec![first_name];
                    self.advance(); // consume first comma
                    fields.extend(self.parse_comma_list_no_trailing(|p| p.parse_name())?);

                    self.expect(TokenType::From)?;
                    let module = self.parse_import_source()?;

                    Ok(Import::SImportFields {
                        l: self.make_span(start),
                        fields,
                        import: module,
                    })
                } else if self.matches(&TokenType::From) {
                    // import x from source (single name)
                    self.advance(); // consume FROM
                    let module = self.parse_import_source()?;

                    Ok(Import::SImportFields {
                        l: self.make_span(start),
                        fields: vec![first_name],
                        import: module,
                    })
                } else {
                    // import source as name (regular import)
                    // The first_name we parsed is actually the import source
                    // Extract location and string from Name enum
                    let (name_loc, name_str) = match &first_name {
                        Name::SName { l, s } => (*l, s.clone()),
                        _ => return Err(ParseError::general(
                            self.peek(),
                            "Expected identifier for import source",
                        )),
                    };

                    // Check for special import like file("...")
                    // If next token is ParenNoSpace, we need to parse it as special import
                    let module = if self.matches(&TokenType::ParenNoSpace) {
                        self.advance(); // consume (

                        let kind = name_str;

                        // Parse comma-separated string arguments
                        let first_arg = self.expect(TokenType::String)?;
                        let mut args = vec![first_arg.value];

                        // Parse additional arguments if present
                        if self.matches(&TokenType::Comma) {
                            self.advance();
                            args.extend(self.parse_comma_list_no_trailing(|p| {
                                Ok(p.expect(TokenType::String)?.value)
                            })?);
                        }

                        self.expect(TokenType::RParen)?;

                        ImportType::SSpecialImport {
                            l: self.make_span(start),
                            kind,
                            args,
                        }
                    } else {
                        ImportType::SConstImport {
                            l: name_loc,
                            module: name_str,
                        }
                    };

                    // Expect AS keyword
                    if !self.matches(&TokenType::As) {
                        return Err(ParseError::general(
                            self.peek(),
                            "Expected 'as' after import module",
                        ));
                    }
                    self.advance(); // consume AS

                    // Parse the alias name
                    let name = self.parse_name()?;

                    Ok(Import::SImport {
                        l: self.make_span(start),
                        import: module,
                        name,
                    })
                }
            }
            TokenType::Include => {
                self.advance(); // consume INCLUDE

                // Check for include-from: include from module: names
                if self.matches(&TokenType::From) {
                    self.advance();

                    // Parse dotted module path: PPX.PX or just PPX
                    let mut module_path = vec![self.parse_name()?];
                    while self.matches(&TokenType::Dot) {
                        self.advance(); // consume dot
                        module_path.push(self.parse_name()?);
                    }

                    self.expect(TokenType::Colon)?;

                    // Parse comma-separated include specs (or empty list)
                    let names = if !self.matches(&TokenType::End) {
                        self.parse_comma_list(|p| p.parse_include_spec())?
                    } else {
                        Vec::new()
                    };

                    self.expect(TokenType::End)?;

                    Ok(Import::SIncludeFrom {
                        l: self.make_span(start),
                        module_path,
                        names,
                    })
                } else {
                    // Simple include: include module
                    let module = self.parse_import_source()?;

                    Ok(Import::SInclude {
                        l: self.make_span(start),
                        import: module,
                    })
                }
            }
            _ => Err(ParseError::general(
                self.peek(),
                "Expected 'import' or 'include'",
            )),
        }
    }

    /// Parse import source (module name)
    /// Handles:
    /// - Simple name: `module-name`
    /// - Special import: `file("path.arr")` or `file("path.arr", "other")`
    fn parse_import_source(&mut self) -> ParseResult<ImportType> {
        let start = self.start_loc();
        let module_name = self.expect(TokenType::Name)?;

        // Check for special import: NAME PARENNOSPACE STRING (COMMA STRING)* RPAREN
        if self.matches(&TokenType::ParenNoSpace) {
            self.advance(); // consume (

            let kind = module_name.value;

            // Parse comma-separated string arguments
            let first_arg = self.expect(TokenType::String)?;
            let mut args = vec![first_arg.value];

            // Parse additional arguments if present
            if self.matches(&TokenType::Comma) {
                self.advance();
                args.extend(self.parse_comma_list_no_trailing(|p| {
                    Ok(p.expect(TokenType::String)?.value)
                })?);
            }

            self.expect(TokenType::RParen)?;

            Ok(ImportType::SSpecialImport {
                l: self.make_span(start),
                kind,
                args,
            })
        } else {
            // Simple const import
            Ok(ImportType::SConstImport {
                l: self.make_span(start),
                module: module_name.value,
            })
        }
    }

    /// provide-stmt: PROVIDE stmt END | PROVIDE *
    /// Parses provide statements like:
    /// - provide *
    /// - provide: expr end
    /// - provide from module: specs end
    fn parse_provide_stmt(&mut self) -> ParseResult<Provide> {
        let start = self.expect(TokenType::Provide)?;

        // Check for provide-all: provide *
        if self.matches(&TokenType::Times) {
            self.advance();

            Ok(Provide::SProvideAll {
                l: start.location.span(start.location),
            })
        } else if self.matches(&TokenType::From) {
            // provide from module: specs end
            // This is actually a provide-block, not a provide-stmt
            // This case is now handled by parse_provide_from_block()
            // which is called from parse_prelude()
            unreachable!("provide from should be handled by parse_provide_from_block");
        } else if self.matches(&TokenType::Colon) {
            // provide: block end
            self.advance(); // consume :

            // Parse the provide block
            let mut block_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_block_statement()?;
                block_stmts.push(Box::new(stmt));
            }

            let end = self.expect(TokenType::End)?;

            let block = Box::new(Expr::SBlock {
                l: self.peek().location,
                stmts: block_stmts,
            });

            Ok(Provide::SProvide {
                l: start.location.span(end.location),
                block,
            })
        } else {
            // provide stmt end (e.g., provide { x: 10 } end)
            let stmt = self.parse_block_statement()?;
            let end = self.expect(TokenType::End)?;

            // The provide block is the statement itself, not wrapped in s-block
            Ok(Provide::SProvide {
                l: start.location.span(end.location),
                block: Box::new(stmt),
            })
        }
    }

    /// provide-types-stmt: PROVIDE-TYPES record-ann | PROVIDE-TYPES (STAR|TIMES)
    /// Parses provide-types statements like:
    /// - provide-types *
    /// - provide-types { Foo:: Foo, Bar:: Bar }
    fn parse_provide_types_stmt(&mut self) -> ParseResult<ProvideTypes> {
        let start = self.expect(TokenType::ProvideTypes)?;

        // Check for provide-types-all: provide-types *
        if self.matches(&TokenType::Times) || self.matches(&TokenType::Star) {
            self.advance();

            Ok(ProvideTypes::SProvideTypesAll {
                l: start.location.span(start.location),
            })
        } else if self.matches(&TokenType::LBrace) {
            // Parse provide-types { Foo:: Foo, Bar:: Bar }
            self.advance(); // consume {

            let mut fields = Vec::new();

            // Parse comma-separated list of Name:: Ann
            if !self.matches(&TokenType::RBrace) {
                loop {
                    // Parse Name:: Ann
                    let name_tok = self.expect(TokenType::Name)?;
                    let name = name_tok.value.clone();
                    self.expect(TokenType::ColonColon)?;
                    let ann = self.parse_ann()?;

                    fields.push(AField {
                        node_type: "a-field".to_string(),
                        l: name_tok.location.span(name_tok.location),
                        name,
                        ann,
                    });

                    // Check for comma
                    if self.matches(&TokenType::Comma) {
                        self.advance();
                        // Allow trailing comma
                        if self.matches(&TokenType::RBrace) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }

            let end = self.expect(TokenType::RBrace)?;

            Ok(ProvideTypes::SProvideTypes {
                l: start.location.span(end.location),
                anns: fields,
            })
        } else {
            Err(ParseError::general(
                self.peek(),
                "Expected * or { after provide-types",
            ))
        }
    }

    /// provide-block: PROVIDECOLON [provide-spec (COMMA provide-spec)* [COMMA]] END
    /// Parses provide-block statements like:
    /// - provide: add, multiply end
    /// - provide: * end (provide all)
    fn parse_provide_block(&mut self) -> ParseResult<ProvideBlock> {
        let start = self.expect(TokenType::ProvideColon)?;

        // Parse comma-separated provide-specs (or empty list)
        let specs = if !self.matches(&TokenType::End) {
            self.parse_comma_list(|p| p.parse_provide_spec())?
        } else {
            Vec::new()
        };

        let end = self.expect(TokenType::End)?;

        Ok(ProvideBlock {
            node_type: "s-provide-block".to_string(),
            l: start.location.span(end.location),
            path: Vec::new(), // Empty path for basic provide-blocks
            specs,
        })
    }

    /// provide from module: specs end
    /// Parses provide-from-block statements like:
    /// - provide from csv-lib: parse-string end
    /// - provide from lists: map, filter end
    fn parse_provide_from_block(&mut self) -> ParseResult<ProvideBlock> {
        let start = self.expect(TokenType::Provide)?;
        self.expect(TokenType::From)?;

        let module_name = self.parse_name()?;
        self.expect(TokenType::Colon)?;

        // Parse comma-separated provide-specs (or empty list)
        let specs = if !self.matches(&TokenType::End) {
            self.parse_comma_list(|p| p.parse_provide_spec())?
        } else {
            Vec::new()
        };

        let end = self.expect(TokenType::End)?;

        Ok(ProvideBlock {
            node_type: "s-provide-block".to_string(),
            l: start.location.span(end.location),
            path: vec![module_name], // Module path for provide-from
            specs,
        })
    }

    /// Parse a single provide-spec
    /// provide-spec: provide-name-spec | provide-type-spec | provide-data-spec | provide-module-spec
    fn parse_provide_spec(&mut self) -> ParseResult<ProvideSpec> {
        // Check for TYPE, DATA, or MODULE keywords
        if self.matches(&TokenType::Type) {
            self.advance(); // consume TYPE
            let name_spec = self.parse_name_spec()?;
            return Ok(ProvideSpec::SProvideType {
                l: self.peek().location,
                name: name_spec,
            });
        }

        if self.matches(&TokenType::Data) {
            self.advance(); // consume DATA
            let name_spec = self.parse_name_spec()?;

            // Parse optional hiding(...) spec
            let hidden = self.parse_hiding_clause()?;

            return Ok(ProvideSpec::SProvideData {
                l: self.peek().location,
                name: name_spec,
                hidden,
            });
        }

        if self.matches(&TokenType::Module) {
            self.advance(); // consume MODULE
            let name_spec = self.parse_name_spec()?;
            return Ok(ProvideSpec::SProvideModule {
                l: self.peek().location,
                name: name_spec,
            });
        }

        // Otherwise, parse as provide-name-spec
        let name_spec = self.parse_name_spec()?;

        Ok(ProvideSpec::SProvideName {
            l: self.peek().location,
            name: name_spec,
        })
    }

    /// Parse a name-spec
    /// name-spec: (STAR|TIMES) [hiding-spec] | module-ref | module-ref AS NAME
    fn parse_name_spec(&mut self) -> ParseResult<NameSpec> {
        // Check for * (star)
        if self.matches(&TokenType::Times) || self.matches(&TokenType::Star) {
            let start = self.start_loc();
            self.advance();

            // Parse optional hiding-spec: * hiding (name1, name2)
            let hidden = self.parse_hiding_clause()?;

            return Ok(NameSpec::SStar {
                l: self.make_span(start),
                hidden,
            });
        }

        // Parse module-ref (can be dotted path like PD.BT)
        let mut path = vec![self.parse_name()?];

        // Check for additional dot-separated names (e.g., PD.BT)
        while self.matches(&TokenType::Dot) {
            self.advance(); // consume DOT
            path.push(self.parse_name()?);
        }

        // Check for AS NAME
        let as_name = if self.matches(&TokenType::As) {
            self.advance(); // consume AS
            Some(self.parse_name()?)
        } else {
            None
        };

        // The JSON format shows: "name-spec": {"type": "s-module-ref", "path": [{"type": "s-name", "name": "add"}], "as-name": null}
        Ok(NameSpec::SModuleRef {
            l: self.peek().location,
            path,
            as_name,
        })
    }

    /// Parse an include-spec
    /// include-spec: include-name-spec | include-type-spec | include-data-spec | include-module-spec
    /// For now, just handles include-name-spec (simple names)
    fn parse_include_spec(&mut self) -> ParseResult<IncludeSpec> {
        let start = self.start_loc();

        // Check for TYPE, DATA, or MODULE keywords
        if self.matches(&TokenType::Type) {
            // include-type-spec: TYPE name-spec
            self.advance(); // consume TYPE
            let name = self.parse_name_spec()?;

            Ok(IncludeSpec::SIncludeType {
                l: self.make_span(start),
                name,
            })
        } else if self.matches(&TokenType::Data) {
            // include-data-spec: DATA data-name-spec [hiding-spec]
            self.advance(); // consume DATA
            let name = self.parse_name_spec()?;

            // Parse optional hiding-spec: hiding (name1, name2, ...)
            let hidden = self.parse_hiding_clause()?;

            Ok(IncludeSpec::SIncludeData {
                l: self.make_span(start),
                name,
                hidden,
            })
        } else if self.matches(&TokenType::Module) {
            // include-module-spec: MODULE name-spec
            self.advance(); // consume MODULE
            let name = self.parse_name_spec()?;

            Ok(IncludeSpec::SIncludeModule {
                l: self.make_span(start),
                name,
            })
        } else {
            // include-name-spec: name-spec
            let name_spec = self.parse_name_spec()?;

            Ok(IncludeSpec::SIncludeName {
                l: self.peek().location,
                name: name_spec,
            })
        }
    }
}

// ============================================================================
// SECTION 4: Type Annotation Parsing
// ============================================================================

impl Parser {
    /// ann: name-ann | record-ann | arrow-ann | ...
    fn parse_ann(&mut self) -> ParseResult<Ann> {
        // Handle simple name annotations like "Either", "Number", "Any", etc.
        // Also handles dotted names like "E.Either", "List.T", etc.
        // Also handles arrow types like "(A -> B)" and "(A, B -> C)"
        // Also handles tuple annotations like "{A; B}"

        // Handle braced annotations: tuples {A; B; C} or records { field :: Type, ... }
        if self.matches(&TokenType::LBrace) {
            let start = self.advance().clone(); // consume {

            // Empty braces - return empty record (not tuple!)
            // In Pyret: { } is an empty record type, {;} or similar would be tuple
            if self.matches(&TokenType::RBrace) {
                let end = self.advance().clone();
                return Ok(Ann::ARecord {
                    l: start.location.span(end.location),
                    fields: vec![],
                });
            }

            // Look ahead to distinguish record from tuple
            // Record: starts with Name followed by ::
            // Tuple: starts with annotation followed by ; or }
            let is_record = self.matches(&TokenType::Name) && {
                let saved_pos = self.current;
                self.advance(); // skip name
                let has_coloncolon = self.matches(&TokenType::ColonColon);
                self.current = saved_pos; // restore position
                has_coloncolon
            };

            if is_record {
                // Parse record annotation: { field :: Type, field :: Type }
                let mut record_fields = Vec::new();

                loop {
                    // Parse field name
                    let field_name_token = self.expect(TokenType::Name)?;
                    let field_name = field_name_token.value.clone();

                    // Expect ::
                    self.expect(TokenType::ColonColon)?;

                    // Parse field type annotation
                    let field_ann = self.parse_ann()?;

                    // Create AField
                    let field_loc = field_name_token.location.span(field_name_token.location);
                    record_fields.push(AField {
                        node_type: "a-field".to_string(),
                        l: field_loc,
                        name: field_name,
                        ann: field_ann,
                    });

                    // Check for comma (continue) or closing brace (done)
                    if self.matches(&TokenType::Comma) {
                        self.advance(); // consume comma
                        // Allow trailing comma
                        if self.matches(&TokenType::RBrace) {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                let end = self.expect(TokenType::RBrace)?;
                return Ok(Ann::ARecord {
                    l: start.location.span(end.location),
                    fields: record_fields,
                });
            } else {
                // Parse tuple annotation: {A; B; C}
                let mut tuple_fields = Vec::new();

                tuple_fields.push(self.parse_ann()?);

                while self.matches(&TokenType::Semi) {
                    self.advance(); // consume semicolon
                    // Allow trailing semicolon
                    if self.matches(&TokenType::RBrace) {
                        break;
                    }
                    tuple_fields.push(self.parse_ann()?);
                }

                let end = self.expect(TokenType::RBrace)?;
                return Ok(Ann::ATuple {
                    l: start.location.span(end.location),
                    fields: tuple_fields,
                });
            }
        }

        // Handle parenthesized arrow types: (A -> B), (A, B -> C), ( -> C), etc.
        if self.matches(&TokenType::LParen) || self.matches(&TokenType::ParenSpace) {
            let start = self.advance().clone(); // consume (

            // Parse argument types (comma-separated)
            // Special case: ( -> RetType) has no args, just check for immediate arrow
            let args = if !self.matches(&TokenType::RParen) && !self.matches(&TokenType::ThinArrow) {
                // Parse first arg, then check for more with commas
                let first_arg = self.parse_ann()?;
                if self.matches(&TokenType::Comma) {
                    self.advance(); // consume first comma
                    let mut all_args = vec![first_arg];
                    all_args.extend(self.parse_comma_list_no_trailing(|p| p.parse_ann())?);
                    all_args
                } else {
                    vec![first_arg]
                }
            } else {
                Vec::new()
            };

            // Expect arrow
            if self.matches(&TokenType::ThinArrow) {
                self.advance(); // consume ->

                // Parse return type
                let ret = Box::new(self.parse_ann()?);

                let end = self.expect(TokenType::RParen)?;
                let loc = start.location.span(end.location);

                return Ok(Ann::AArrow {
                    l: loc,
                    args,
                    ret,
                    use_parens: true,
                });
            } else {
                // Just a parenthesized annotation, not an arrow
                self.expect(TokenType::RParen)?;
                // Return the single annotation (unwrap the parens)
                return Ok(args.into_iter().next().unwrap_or(Ann::ABlank));
            }
        }

        if self.matches(&TokenType::Name) {
            let name_token = self.advance().clone();
            let mut loc = name_token.location.span(name_token.location);

            // Check if it's the special "Any" type (before processing dots)
            if name_token.value == "Any" && !self.matches(&TokenType::Dot) {
                return Ok(Ann::AAny { l: loc });
            }

            let name = self.token_to_name(&name_token);

            // Handle dotted names: E.Either, Module.Type, etc.
            // Build up nested ADot nodes for chains like A.B.C
            let mut base_ann: Ann = Ann::AName {
                l: loc,
                id: name,
            };

            while self.matches(&TokenType::Dot) {
                self.advance(); // consume dot
                let field_token = self.expect(TokenType::Name)?;
                loc = name_token.location.span(field_token.location);

                // For ADot, we need the obj to be a Name, not an Ann
                // So extract the Name from the current annotation
                let obj_name = match &base_ann {
                    Ann::AName { id, .. } => id.clone(),
                    Ann::ADot { obj, field, .. } => {
                        // For nested dots like A.B.C, we need to create a Name from A.B
                        // The official parser represents this as nested ADot nodes
                        // For now, we'll create a dotted name
                        match obj {
                            Name::SName { l, s } => Name::SName {
                                l: *l,
                                s: format!("{}.{}", s, field),
                            },
                            other => other.clone(),
                        }
                    }
                    _ => return Err(ParseError::general(&field_token, "Invalid dotted type annotation")),
                };

                base_ann = Ann::ADot {
                    l: loc,
                    obj: obj_name,
                    field: field_token.value.clone(),
                };
            }

            // Check for type application: List<T>, Map<K, V>, etc.
            let mut result_ann = if self.matches(&TokenType::Lt) || self.matches(&TokenType::LtNoSpace) {
                self.advance(); // consume '<'
                let type_args = self.parse_comma_list(|p| p.parse_ann())?;
                let end_token = self.expect(TokenType::Gt)?; // consume '>'
                let app_loc = name_token.location.span(end_token.location);
                Ann::AApp {
                    l: app_loc,
                    ann: Box::new(base_ann),
                    args: type_args,
                }
            } else {
                base_ann
            };

            // Check for refinement: %(predicate)
            if self.matches(&TokenType::Percent) {
                self.advance(); // consume '%'
                // Expect any form of left paren
                self.expect_any_lparen()?;
                let predicate = self.parse_expr()?;
                let end_token = self.expect(TokenType::RParen)?; // consume ')'
                let pred_loc = name_token.location.span(end_token.location);
                result_ann = Ann::APred {
                    l: pred_loc,
                    ann: Box::new(result_ann),
                    exp: Box::new(predicate),
                };
            }

            Ok(result_ann)
        } else {
            // Default to blank annotation
            Ok(Ann::ABlank)
        }
    }
}

// ============================================================================
// SECTION 5: Binding Parsing
// ============================================================================

impl Parser {
    /// binding: name [:: ann]
    /// Parses a parameter binding like: x, x :: Number
    fn parse_bind(&mut self) -> ParseResult<Bind> {
        // binding: name-binding | tuple-binding
        // Check if this is a tuple binding
        if self.matches(&TokenType::LBrace) {
            // Tuple binding: {x; y; z}
            self.parse_tuple_bind()
        } else if self.matches(&TokenType::Shadow) {
            // Name binding with shadow keyword
            self.advance(); // consume 'shadow'

            // Check if this is a tuple binding after shadow
            if self.matches(&TokenType::LBrace) {
                // shadow {x; y}
                return Err(ParseError::general(
                    self.peek(),
                    "Shadow keyword cannot be used before tuple bindings"
                ));
            }

            self.parse_bind_with_shadow(true)
        } else {
            // Regular name binding
            self.parse_bind_with_shadow(false)
        }
    }

    fn parse_bind_with_shadow(&mut self, shadows: bool) -> ParseResult<Bind> {
        let name_token = self.expect(TokenType::Name)?;
        let name_str = name_token.value.clone();

        // Create Name node - use SUnderscore for "_"
        let name = if name_str == "_" {
            Name::SUnderscore {
                l: name_token.location,
            }
        } else {
            self.token_to_name(&name_token)
        };

        // Optional type annotation: :: ann
        let ann = if self.matches(&TokenType::ColonColon) {
            self.expect(TokenType::ColonColon)?;
            self.parse_ann()?
        } else {
            Ann::ABlank
        };

        Ok(Bind::SBind {
            l: name_token.location,
            shadows,
            id: name,
            ann: Box::new(ann),
        })
    }

    /// tuple-binding: LBRACE (binding SEMI)* binding [SEMI] RBRACE [AS name-binding]
    /// binding: name-binding | tuple-binding
    ///
    /// Parses tuple bindings which can contain nested tuples: {x; y} or {{x; y}; z}
    fn parse_tuple_bind(&mut self) -> ParseResult<Bind> {
        let start = self.expect(TokenType::LBrace)?;

        // Parse fields: binding; binding; ...
        // Each binding can be a name or another tuple
        let mut fields = Vec::new();

        // Parse first field
        if !self.matches(&TokenType::RBrace) {
            // First binding
            let field = self.parse_bind_in_tuple()?;
            fields.push(field);

            // Parse remaining fields
            while self.matches(&TokenType::Semi) {
                self.advance(); // consume semicolon

                // Check for trailing semicolon before }
                if self.matches(&TokenType::RBrace) {
                    break;
                }

                let field = self.parse_bind_in_tuple()?;
                fields.push(field);
            }
        }

        let end = self.expect(TokenType::RBrace)?;

        // Check for optional "as name" after the tuple
        let as_name = if self.matches(&TokenType::As) {
            self.advance(); // consume 'as'
            let name = self.parse_name()?;
            let ann = if self.matches(&TokenType::ColonColon) {
                self.expect(TokenType::ColonColon)?;
                self.parse_ann()?
            } else {
                Ann::ABlank
            };
            Some(Box::new(Bind::SBind {
                l: self.peek().location,
                shadows: false,
                id: name,
                ann: Box::new(ann),
            }))
        } else {
            None
        };

        Ok(Bind::STupleBind {
            l: start.location.span(end.location),
            fields,
            as_name,
        })
    }

    /// Helper to parse a single binding within a tuple (can be name or nested tuple)
    fn parse_bind_in_tuple(&mut self) -> ParseResult<Bind> {
        // Check for optional shadow keyword
        if self.matches(&TokenType::Shadow) {
            self.advance(); // consume 'shadow'

            // After shadow, can only be a name, not a nested tuple
            if self.matches(&TokenType::LBrace) {
                return Err(ParseError::general(
                    self.peek(),
                    "Shadow keyword cannot be used before tuple bindings"
                ));
            }

            return self.parse_bind_with_shadow(true);
        }

        // Check if this is a nested tuple
        if self.matches(&TokenType::LBrace) {
            self.parse_tuple_bind()
        } else {
            // Regular name binding
            self.parse_bind_with_shadow(false)
        }
    }
}

// ============================================================================
// SECTION 6: Expression Parsing
// ============================================================================

impl Parser {
    /// expr: binop-expr | prim-expr | ...
    /// Top-level expression dispatcher
    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        // Check for let/var bindings first (x = value or var x := value)
        // These can appear as standalone expressions, not just in blocks
        if self.matches(&TokenType::Let) {
            return self.parse_let_expr();
        }

        if self.matches(&TokenType::Var) {
            return self.parse_var_expr();
        }

        // Check for implicit let binding or assignment: name = value or name := value
        // We need to look ahead to distinguish from other expressions
        if self.matches(&TokenType::Name) {
            let checkpoint = self.checkpoint();
            let _name = self.advance(); // Consume the name

            if self.matches(&TokenType::Equals) {
                // This is a let binding: x = value
                self.restore(checkpoint);
                return self.parse_standalone_let_expr();
            } else if self.matches(&TokenType::ColonEquals) {
                // This is an assignment: x := value
                self.restore(checkpoint);
                return self.parse_implicit_var_expr();
            } else {
                // Not a binding, restore and continue with normal parsing
                self.restore(checkpoint);
            }
        }

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
            } else if self.matches(&TokenType::LtNoSpace) {
                // Generic type instantiation: expr<T, U, ...> (no whitespace before <)
                left = self.parse_instantiate_expr(left)?;
            } else if self.matches(&TokenType::Dot) {
                // Dot access, tuple access, or object extension
                let _dot_token = self.expect(TokenType::Dot)?;

                // Check if this is tuple access (.{number}) or object extension (.{fields})
                if self.matches(&TokenType::LBrace) {
                    self.expect(TokenType::LBrace)?;

                    // Distinguish between tuple access and object extension
                    // Tuple access: .{number}
                    // Object extension: .{field: value, ...}
                    if self.matches(&TokenType::Number) {
                        // Tuple access: x.{2}
                        let index_token = self.expect(TokenType::Number)?;
                        let index: usize = index_token.value.parse()
                            .map_err(|_| ParseError::invalid("tuple index", &index_token, "Invalid number"))?;
                        let index_loc = index_token.location.span(index_token.location);
                        let end = self.expect(TokenType::RBrace)?;

                        let start_loc = left.get_loc();

                        left = Expr::STupleGet {
                            l: start_loc.span(end.location),
                            tup: Box::new(left),
                            index,
                            index_loc,
                        };
                    } else {
                        // Object extension: obj.{ x: 1, y: 2 }
                        // Parse object fields (same as parse_obj_expr_fields but return fields)
                        let mut fields = Vec::new();

                        // Handle empty extension
                        if !self.matches(&TokenType::RBrace) {
                            loop {
                                fields.push(self.parse_obj_field()?);

                                if !self.matches(&TokenType::Comma) {
                                    break;
                                }
                                self.advance(); // consume comma

                                // Check for trailing comma
                                if self.matches(&TokenType::RBrace) {
                                    break;
                                }
                            }
                        }

                        let end = self.expect(TokenType::RBrace)?;

                        let start_loc = left.get_loc();

                        // In Pyret, both extension and update use the same syntax: obj.{fields}
                        // The semantic difference is: extension adds NEW fields, update MODIFIES existing fields
                        // We use SExtend as the default; the type checker determines the actual semantics
                        left = Expr::SExtend {
                            l: start_loc.span(end.location),
                            supe: Box::new(left),
                            fields,
                        };
                    }
                } else {
                    // Regular dot access: obj.field
                    let field_token = self.parse_field_name()?;

                    let start_loc = left.get_loc();

                    left = Expr::SDot {
                        l: start_loc.span(field_token.location),
                        obj: Box::new(left),
                        field: field_token.value.clone(),
                    };
                }
            } else if self.matches(&TokenType::BrackNoSpace) {
                // Bracket access (no whitespace before bracket)
                left = self.parse_bracket_expr(left)?;
            } else if self.matches(&TokenType::Bang) {
                // Bang operator for ref field access or update
                let _bang_token = self.expect(TokenType::Bang)?;
                if self.matches(&TokenType::LBrace) {
                    // Bang update: obj!{ field: value, ... }
                    self.expect(TokenType::LBrace)?;
                    let fields = if self.matches(&TokenType::RBrace) {
                        Vec::new()
                    } else {
                        self.parse_comma_list(|p| p.parse_obj_field())?
                    };
                    let end = self.expect(TokenType::RBrace)?;

                    let start_loc = left.get_loc();

                    left = Expr::SUpdate {
                        l: start_loc.span(end.location),
                        supe: Box::new(left),
                        fields,
                    };
                } else {
                    // Bang field access: obj!field
                    let field_token = self.parse_field_name()?;
                    let start_loc = left.get_loc();
                    left = Expr::SGetBang {
                        l: start_loc.span(field_token.location),
                        obj: Box::new(left),
                        field: field_token.value.clone(),
                    };
                }
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

                    // Check if this is tuple access (.{number}) or dot number access (.number)
                    if self.matches(&TokenType::LBrace) {
                        self.expect(TokenType::LBrace)?;

                        // Tuple access with braces: .{number}
                        if self.matches(&TokenType::Number) {
                            let index_token = self.expect(TokenType::Number)?;
                            let index: usize = index_token.value.parse()
                                .map_err(|_| ParseError::invalid("tuple index", &index_token, "Invalid number"))?;
                            let index_loc = index_token.location.span(index_token.location);
                            let end = self.expect(TokenType::RBrace)?;

                            let start_loc = right.get_loc();

                            right = Expr::STupleGet {
                                l: start_loc.span(end.location),
                                tup: Box::new(right),
                                index,
                                index_loc,
                            };
                        } else {
                            // Object extension: obj.{ x: 1, y: 2 }
                            // Parse object fields (same as parse_obj_expr_fields but return fields)
                            let mut fields = Vec::new();

                            // Handle empty extension
                            if !self.matches(&TokenType::RBrace) {
                                loop {
                                    fields.push(self.parse_obj_field()?);

                                    if !self.matches(&TokenType::Comma) {
                                        break;
                                    }
                                    self.advance(); // consume comma

                                    // Check for trailing comma
                                    if self.matches(&TokenType::RBrace) {
                                        break;
                                    }
                                }
                            }

                            let end = self.expect(TokenType::RBrace)?;

                            let start_loc = right.get_loc();

                            // In Pyret, both extension and update use the same syntax: obj.{fields}
                            // The semantic difference is: extension adds NEW fields, update MODIFIES existing fields
                            // We use SExtend as the default; the type checker determines the actual semantics
                            right = Expr::SExtend {
                                l: start_loc.span(end.location),
                                supe: Box::new(right),
                                fields,
                            };
                        }
                    } else {
                        // Regular dot access: obj.field
                        let field_token = self.parse_field_name()?;

                        let start_loc = right.get_loc();

                        right = Expr::SDot {
                            l: start_loc.span(field_token.location),
                            obj: Box::new(right),
                            field: field_token.value.clone(),
                        };
                    }
                } else if self.matches(&TokenType::LBrack) {
                    // Bracket access
                    right = self.parse_bracket_expr(right)?;
                } else if self.matches(&TokenType::Bang) {
                    // Bang operator for ref field access or update
                    let _bang_token = self.expect(TokenType::Bang)?;
                    if self.matches(&TokenType::LBrace) {
                        // Bang update: obj!{ field: value, ... }
                        self.expect(TokenType::LBrace)?;
                        let fields = if self.matches(&TokenType::RBrace) {
                            Vec::new()
                        } else {
                            self.parse_comma_list(|p| p.parse_obj_field())?
                        };
                        let end = self.expect(TokenType::RBrace)?;

                        let start_loc = right.get_loc();

                        right = Expr::SUpdate {
                            l: start_loc.span(end.location),
                            supe: Box::new(right),
                            fields,
                        };
                    } else {
                        // Bang field access: obj!field
                        let field_token = self.parse_field_name()?;
                        let start_loc = right.get_loc();
                        right = Expr::SGetBang {
                            l: start_loc.span(field_token.location),
                            obj: Box::new(right),
                            field: field_token.value.clone(),
                        };
                    }
                } else {
                    break;
                }
            }

            // Create location for the operator
            let op_l = op_token.location;

            // Get location from left expression
            let start_loc = left.get_loc();

            // Get location from right expression
            let end_loc = right.get_loc();

            let loc = start_loc.span(*end_loc);

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

            // Parse optional refinement: is%(refinement-fn)
            let refinement = if self.matches(&TokenType::Percent) {
                self.advance(); // consume %
                // Refinement is a parenthesized expression (function call)
                if self.matches(&TokenType::ParenNoSpace) || self.matches(&TokenType::ParenSpace) {
                    let refinement_expr = self.parse_prim_expr()?;
                    // Unwrap SParen to get the inner expression
                    let unwrapped = match refinement_expr {
                        Expr::SParen { expr, .. } => *expr,
                        other => other,
                    };
                    Some(Box::new(unwrapped))
                } else {
                    let token = self.peek().clone();
                    return Err(ParseError::general(
                        &token,
                        "Expected parenthesized expression after % in check operator",
                    ));
                }
            } else {
                None
            };

            // Parse the right-hand side (if needed)
            // Some operators like does-not-raise are unary and don't have a right-hand side
            let right = if matches!(op, CheckOp::SOpRaisesNot { .. }) {
                // does-not-raise is unary - no right-hand side
                None
            } else if !self.is_at_end() && !matches!(self.peek().token_type, TokenType::Eof | TokenType::End) && !self.is_check_op() {
                // Parse a binary expression (which includes primitives, postfix ops, and binary ops)
                // But don't parse another check operator (they're at the same/lower precedence level)
                let right_expr = self.parse_binop_expr_no_check()?;
                Some(Box::new(right_expr))
            } else {
                None
            };

            // Get location from left expression
            let start_loc = left.get_loc();

            // Get end location, considering refinement and right expression
            // Priority: right > refinement > operator
            let end_loc = if let Some(ref right_expr) = right {
                *right_expr.get_loc()
            } else if let Some(ref refinement_expr) = refinement {
                // If no right expression but has refinement, use refinement location
                *refinement_expr.get_loc()
            } else {
                // If no right expression and no refinement, use operator location
                match &op {
                    CheckOp::SOpIs { l } => *l,
                    CheckOp::SOpIsRoughly { l } => *l,
                    CheckOp::SOpIsNot { l } => *l,
                    CheckOp::SOpIsNotRoughly { l } => *l,
                    CheckOp::SOpIsOp { l, .. } => *l,
                    CheckOp::SOpIsNotOp { l, .. } => *l,
                    CheckOp::SOpSatisfies { l } => *l,
                    CheckOp::SOpSatisfiesNot { l } => *l,
                    CheckOp::SOpRaises { l } => *l,
                    CheckOp::SOpRaisesOther { l } => *l,
                    CheckOp::SOpRaisesNot { l } => *l,
                    CheckOp::SOpRaisesSatisfies { l } => *l,
                    CheckOp::SOpRaisesViolates { l } => *l,
                }
            };

            // Check for optional 'because' clause
            let cause = if self.matches(&TokenType::Because) {
                self.advance(); // consume 'because'
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };

            // Update end location if there's a cause
            let final_end_loc = if let Some(ref cause_expr) = cause {
                *cause_expr.get_loc()
            } else {
                end_loc
            };

            let loc = start_loc.span(final_end_loc);

            left = Expr::SCheckTest {
                l: loc,
                op,
                refinement,
                left: Box::new(left),
                right,
                cause,
            };
        }

        Ok(left)
    }

    /// Same as parse_binop_expr but stops before parsing check operators
    /// Used for parsing the right-hand side of check operators
    fn parse_binop_expr_no_check(&mut self) -> ParseResult<Expr> {
        // Start with a primary expression
        let mut left = self.parse_prim_expr()?;

        // Check for postfix operators (function application, dot access, and bracket access)
        loop {
            if self.matches(&TokenType::ParenNoSpace) {
                left = self.parse_app_expr(left)?;
            } else if self.matches(&TokenType::LtNoSpace) {
                // Generic type instantiation: expr<T, U, ...> (no whitespace before <)
                left = self.parse_instantiate_expr(left)?;
            } else if self.matches(&TokenType::Dot) {
                let _dot_token = self.expect(TokenType::Dot)?;
                if self.matches(&TokenType::LBrace) {
                    // Tuple access
                    self.expect(TokenType::LBrace)?;
                    let index_token = self.expect(TokenType::Number)?;
                    let index: usize = index_token.value.parse()
                        .map_err(|_| ParseError::invalid("tuple index", &index_token, "Invalid number"))?;
                    let index_loc = index_token.location.span(index_token.location);
                    let end = self.expect(TokenType::RBrace)?;
                    let start_loc = left.get_loc();
                    left = Expr::STupleGet {
                        l: start_loc.span(end.location),
                        tup: Box::new(left),
                        index,
                        index_loc,
                    };
                } else {
                    // Regular dot access
                    let field_token = self.parse_field_name()?;
                    let start_loc = left.get_loc();
                    left = Expr::SDot {
                        l: start_loc.span(field_token.location),
                        obj: Box::new(left),
                        field: field_token.value.clone(),
                    };
                }
            } else if self.matches(&TokenType::LBrack) {
                left = self.parse_bracket_expr(left)?;
            } else if self.matches(&TokenType::Bang) {
                // Bang operator for ref field access or update
                let _bang_token = self.expect(TokenType::Bang)?;
                if self.matches(&TokenType::LBrace) {
                    // Bang update: obj!{ field: value, ... }
                    self.expect(TokenType::LBrace)?;
                    let fields = if self.matches(&TokenType::RBrace) {
                        Vec::new()
                    } else {
                        self.parse_comma_list(|p| p.parse_obj_field())?
                    };
                    let end = self.expect(TokenType::RBrace)?;

                    let start_loc = left.get_loc();

                    left = Expr::SUpdate {
                        l: start_loc.span(end.location),
                        supe: Box::new(left),
                        fields,
                    };
                } else {
                    // Bang field access: obj!field
                    let field_token = self.parse_field_name()?;
                    let start_loc = left.get_loc();
                    left = Expr::SGetBang {
                        l: start_loc.span(field_token.location),
                        obj: Box::new(left),
                        field: field_token.value.clone(),
                    };
                }
            } else {
                break;
            }
        }

        // Parse binary operators (but NOT check operators)
        while self.is_binop() {
            let op_token = self.peek().clone();
            let op = self.parse_binop()?;
            let mut right = self.parse_prim_expr()?;

            // Postfix operators on right side
            loop {
                if self.matches(&TokenType::ParenNoSpace) {
                    right = self.parse_app_expr(right)?;
                } else if self.matches(&TokenType::LtNoSpace) {
                    // Generic type instantiation: expr<T, U, ...> (no whitespace before <)
                    right = self.parse_instantiate_expr(right)?;
                } else if self.matches(&TokenType::Dot) {
                    let _dot_token = self.expect(TokenType::Dot)?;
                    let field_token = self.parse_field_name()?;
                    let start_loc = right.get_loc();
                    right = Expr::SDot {
                        l: start_loc.span(field_token.location),
                        obj: Box::new(right),
                        field: field_token.value.clone(),
                    };
                } else if self.matches(&TokenType::LBrack) {
                    right = self.parse_bracket_expr(right)?;
                } else {
                    break;
                }
            }

            let op_l = op_token.location;
            let start_loc = left.get_loc();
            let end_loc = right.get_loc();
            let loc = start_loc.span(*end_loc);

            left = Expr::SOp {
                l: loc,
                op_l,
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// prim-expr: num-expr | frac-expr | rfrac-expr | bool-expr | string-expr | id-expr | paren-expr | array-expr | lam-expr | block-expr
    fn parse_prim_expr(&mut self) -> ParseResult<Expr> {
        let token = self.peek().clone();

        match token.token_type {
            TokenType::Number => self.parse_num(),
            TokenType::RoughNumber => self.parse_rough_num(),
            TokenType::Rational => self.parse_rational(),
            TokenType::RoughRational => self.parse_rough_rational(),
            TokenType::True | TokenType::False => self.parse_bool(),
            TokenType::String => self.parse_str(),
            TokenType::Name => self.parse_id_expr(),
            TokenType::ParenSpace | TokenType::LParen | TokenType::ParenNoSpace => self.parse_paren_expr(),
            TokenType::BrackSpace | TokenType::LBrack => self.parse_construct_expr(),
            TokenType::LBrace => {
                // Distinguish between curly brace lambda {(x): ...}, tuple {(x + 1); 2}, and object {x: 1}
                // Peek ahead to see if next token is LParen or ParenSpace
                let next_type = &self.peek_ahead(1).token_type;
                if next_type == &TokenType::LParen || next_type == &TokenType::ParenSpace {
                    // Could be either curly-brace lambda or tuple with parenthesized expression
                    // Try parsing as curly-brace lambda first
                    let checkpoint = self.checkpoint();
                    match self.parse_curly_lambda_expr() {
                        Ok(expr) => Ok(expr),
                        Err(_) => {
                            // Failed to parse as lambda, restore and try as object/tuple
                            self.restore(checkpoint);
                            self.parse_obj_expr()
                        }
                    }
                } else {
                    self.parse_obj_expr()
                }
            }
            TokenType::Lam => self.parse_lambda_expr(),
            TokenType::Fun => self.parse_fun_expr(),
            TokenType::Type => self.parse_type_expr(),
            TokenType::Newtype => self.parse_newtype_expr(),
            TokenType::Data => self.parse_data_expr(),
            TokenType::Block => self.parse_block_expr(),
            TokenType::If => self.parse_if_expr(),
            TokenType::Ask => self.parse_ask_expr(),
            TokenType::Cases => self.parse_cases_expr(),
            TokenType::When => self.parse_when_expr(),
            TokenType::For => self.parse_for_expr(),
            TokenType::Let => self.parse_let_expr(),
            TokenType::Shadow => self.parse_shadow_expr(),
            TokenType::Rec => self.parse_rec_expr(),
            TokenType::Letrec => self.parse_letrec_expr(),
            TokenType::Var => self.parse_var_expr(),
            TokenType::CheckColon => self.parse_check_expr(),
            TokenType::Check => self.parse_check_expr(),
            TokenType::ExamplesColon => self.parse_check_expr(),
            TokenType::Spy => self.parse_spy_stmt(),
            TokenType::Method => self.parse_method_expr(),
            TokenType::Table => self.parse_table_expr(),
            TokenType::TableExtract => self.parse_extract_expr(),
            TokenType::LoadTable => self.parse_load_table_expr(),
            TokenType::Reactor => self.parse_reactor_expr(),
            TokenType::DotDotDot => self.parse_template_expr(),
            _ => Err(ParseError::unexpected(token)),
        }
    }

    /// num-expr: NUMBER
    /// Pyret represents all numbers as rationals, so:
    /// - Integers like "42" -> SNum with n=42.0
    /// - Decimals like "3.14" -> SNum with n=3.14 (NOT SFrac - decimals are stored as floats)
    ///
    ///   Note: The official Pyret parser converts decimals to fractions internally but
    ///   represents them as s-num in JSON with fraction string values like "157/50"
    fn parse_num(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::Number)?;
        let loc = token.location;

        // Store as string to support arbitrary precision (like SFrac/SRfrac)
        Ok(Expr::SNum { l: loc, value: token.value.clone() })
    }

    /// rough-num-expr: ROUGHNUMBER
    /// Parses rough (approximate) numbers like ~0.8 or ~42
    /// Represented as SNum with the tilde preserved in the value string
    fn parse_rough_num(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::RoughNumber)?;
        let loc = token.location;

        // Store as string INCLUDING the tilde to support arbitrary precision
        Ok(Expr::SNum { l: loc, value: token.value.clone() })
    }

    /// frac-expr: RATIONAL
    /// Parses explicit rational numbers like "3/2" or "-5/7"
    fn parse_rational(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::Rational)?;
        let loc = token.location;

        // Parse "num/den" format
        let parts: Vec<&str> = token.value.split('/').collect();
        if parts.len() != 2 {
            return Err(ParseError::invalid(
                "rational",
                &token,
                "Expected format: numerator/denominator",
            ));
        }

        // Strip leading '+' from numerator (Pyret normalizes +3 to 3)
        let num_str = parts[0].trim_start_matches('+');
        let den_str = parts[1];

        // Validate that denominator is not zero (check string representation)
        if den_str == "0" {
            return Err(ParseError::invalid(
                "rational",
                &token,
                "Denominator cannot be zero",
            ));
        }

        // Note: We don't simplify fractions to match official Pyret parser behavior
        // The official parser keeps explicit fractions in their original form (e.g., 6/3 not 2/1)
        // However, decimals converted to fractions ARE simplified (e.g., 2.5 → 5/2, not 25/10)
        // That simplification happens in JSON serialization, not here
        Ok(Expr::SFrac {
            l: loc,
            num: num_str.to_string(),
            den: den_str.to_string()
        })
    }

    /// rfrac-expr: ROUGHRATIONAL
    /// Parses rough (approximate) rational numbers like "~3/2" or "~-5/7"
    fn parse_rough_rational(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::RoughRational)?;
        let loc = token.location;

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

        // Strip leading '+' from numerator (Pyret normalizes +3 to 3)
        let num = parts[0].trim_start_matches('+').to_string();
        let den = parts[1].to_string();

        // Validate that denominator is not zero (check string representation)
        if den == "0" {
            return Err(ParseError::invalid(
                "rough rational",
                &token,
                "Denominator cannot be zero",
            ));
        }

        // Note: We don't simplify fractions to match official Pyret parser behavior
        // The official parser keeps fractions in their original form (e.g., ~8/10 not ~4/5)
        // We store numerator and denominator as strings to support arbitrary precision
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
            l: token.location,
            b,
        })
    }

    /// string-expr: STRING
    fn parse_str(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::String)?;

        Ok(Expr::SStr {
            l: token.location,
            s: token.value.clone(),
        })
    }

    /// template-expr: ...
    /// Represents a placeholder/template in function bodies
    fn parse_template_expr(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::DotDotDot)?;

        Ok(Expr::STemplate {
            l: token.location,
        })
    }

    /// id-expr: NAME
    fn parse_id_expr(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::Name)?;
        let loc = token.location;

        // Check if this is an underscore wildcard (for partial application)
        let id = if token.value == "_" {
            Name::SUnderscore { l: loc }
        } else {
            self.token_to_name(&token)
        };

        Ok(Expr::SId {
            l: loc,
            id,
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
                | TokenType::IsSpaceship
                | TokenType::IsEqualEqual
                | TokenType::IsEqualTilde
                | TokenType::IsNotSpaceship
                | TokenType::IsNotEqualEqual
                | TokenType::IsNotEqualTilde
                | TokenType::Satisfies
                | TokenType::Violates
                | TokenType::Raises
                | TokenType::DoesNotRaise
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
        let l = token.location;

        let op = match token.token_type {
            TokenType::Is => CheckOp::SOpIs { l },
            TokenType::IsRoughly => CheckOp::SOpIsRoughly { l },
            TokenType::IsNot => CheckOp::SOpIsNot { l },
            TokenType::IsNotRoughly => CheckOp::SOpIsNotRoughly { l },
            // is<op> variants
            TokenType::IsSpaceship => CheckOp::SOpIsOp { l, op: "op<=>".to_string() },
            TokenType::IsEqualEqual => CheckOp::SOpIsOp { l, op: "op==".to_string() },
            TokenType::IsEqualTilde => CheckOp::SOpIsOp { l, op: "op=~".to_string() },
            // is-not<op> variants
            TokenType::IsNotSpaceship => CheckOp::SOpIsNotOp { l, op: "op<=>".to_string() },
            TokenType::IsNotEqualEqual => CheckOp::SOpIsNotOp { l, op: "op==".to_string() },
            TokenType::IsNotEqualTilde => CheckOp::SOpIsNotOp { l, op: "op=~".to_string() },
            // other check operators
            TokenType::Satisfies => CheckOp::SOpSatisfies { l },
            TokenType::Violates => CheckOp::SOpSatisfiesNot { l }, // violates = satisfies-not
            TokenType::Raises => CheckOp::SOpRaises { l },
            TokenType::DoesNotRaise => CheckOp::SOpRaisesNot { l }, // does-not-raise = raises-not
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
        // Expect ParenSpace, LParen, or ParenNoSpace token
        let start = if self.matches(&TokenType::ParenSpace) {
            self.expect(TokenType::ParenSpace)?
        } else if self.matches(&TokenType::ParenNoSpace) {
            self.expect(TokenType::ParenNoSpace)?
        } else {
            self.expect(TokenType::LParen)?
        };

        let expr = self.parse_expr()?;
        let end = self.expect(TokenType::RParen)?;

        Ok(Expr::SParen {
            l: start.location.span(end.location),
            expr: Box::new(expr),
        })
    }

    /// array-expr: LBRACK (expr COMMA)* RBRACK
    /// Array literal expression
    /// construct-expr: LBRACK construct-modifier binop-expr COLON trailing-opt-comma-binops RBRACK
    /// Construct expressions like [list: 1, 2, 3] or [lazy set: x, y]
    fn parse_construct_expr(&mut self) -> ParseResult<Expr> {
        // Accept both BrackSpace and LBrack for construct expressions
        let start = if self.matches(&TokenType::BrackSpace) {
            self.expect(TokenType::BrackSpace)?
        } else {
            self.expect(TokenType::LBrack)?
        };

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
            l: start.location.span(end.location),
            modifier,
            constructor: Box::new(constructor),
            values: values.into_iter().map(Box::new).collect(),
        })
    }

    /// bracket-expr: expr LBRACK binop-expr RBRACK
    /// Bracket access expression like arr[0] or dict["key"]
    fn parse_bracket_expr(&mut self, obj: Expr) -> ParseResult<Expr> {
        let _start = self.expect(TokenType::BrackNoSpace)?;

        // Parse the field expression (can be any expression)
        let field = self.parse_expr()?;

        let end = self.expect(TokenType::RBrack)?;

        // Get location from obj expression
        let obj_loc = obj.get_loc();

        Ok(Expr::SBracket {
            l: obj_loc.span(end.location),
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
                l: start.location.span(end.location),
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
            let checkpoint = self.checkpoint();

            // Try parsing as expression
            match self.parse_expr() {
                Ok(_) => {
                    // Check what comes after the expression
                    let is_tuple = self.matches(&TokenType::Semi);
                    // Restore position to re-parse
                    self.restore(checkpoint);
                    is_tuple
                }
                Err(_) => {
                    // Failed to parse as expression, assume object
                    self.restore(checkpoint);
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
            l: start.location.span(end.location),
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
            l: start.location.span(end.location),
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
                    l: start.location.span(name_token.location),
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
                    l: name_token.location.span(name_token.location),
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
        // Parse optional type parameters: <T, U, V>
        let params = self.parse_opt_type_params()?;

        // Expect any form of left paren
        self.expect_any_lparen()?;

        // Parse parameters (comma-separated bindings)
        let args = if self.matches(&TokenType::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_bind())?
        };

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Optional type annotation: -> ann
        let ann = self.parse_opt_return_ann()?;

        // Parse body separator (COLON or BLOCK)
        let blocky = self.parse_block_separator()?;

        // Parse optional doc string: doc: "string"
        let doc = self.parse_opt_doc_string()?;

        // Parse method body (statements until END or WHERE)
        let mut body_stmts = Vec::new();
        while !self.matches(&TokenType::End)
            && !self.matches(&TokenType::Where)
            && !self.is_at_end()
        {
            let stmt = self.parse_block_statement()?;
            body_stmts.push(Box::new(stmt));
        }

        // Parse optional where clause
        let (check, check_loc) = if self.matches(&TokenType::Where) {
            let where_token = self.advance().clone();
            // The check-loc should just point to the WHERE keyword itself
            let check_loc = where_token.location.span(where_token.location);
            let mut where_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                where_stmts.push(Box::new(stmt));
            }
            let check_block = Box::new(Expr::SBlock {
                l: check_loc,
                stmts: where_stmts,
            });
            (Some(check_block), Some(check_loc))
        } else {
            (None, None)
        };

        let end = self.expect(TokenType::End)?;

        // Wrap body in SBlock
        let body = Box::new(Expr::SBlock {
            l: self.peek().location,
            stmts: body_stmts,
        });

        Ok(Member::SMethodField {
            l: start.location.span(end.location),
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
        let base_loc = base.get_loc();

        self.expect(TokenType::ParenNoSpace)?;

        // Parse arguments as comma-separated list (no trailing commas allowed)
        let args = if self.matches(&TokenType::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list_no_trailing(|p| p.parse_expr().map(Box::new))?
        };

        let end = self.expect(TokenType::RParen)?;

        Ok(Expr::SApp {
            l: base_loc.span(end.location),
            _fun: Box::new(base),
            args,
        })
    }

    /// Parse generic type instantiation: expr<T, U, ...>
    fn parse_instantiate_expr(&mut self, base: Expr) -> ParseResult<Expr> {
        // Get location from base expression
        let base_loc = base.get_loc();

        self.expect(TokenType::LtNoSpace)?;

        // Parse type parameters as comma-separated list
        let params = if self.matches(&TokenType::Gt) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_ann())?
        };

        let end = self.expect(TokenType::Gt)?;

        Ok(Expr::SInstantiate {
            l: base_loc.span(end.location),
            expr: Box::new(base),
            params,
        })
    }
}

// ============================================================================
// SECTION 7: Control Flow Parsing
// ============================================================================

impl Parser {
    /// block-expr: BLOCK COLON stmts END
    /// Helper function to parse a single statement in a block context
    /// This handles let bindings, var bindings, and other statements
    fn parse_block_statement(&mut self) -> ParseResult<Expr> {
        if self.matches(&TokenType::Type) {
            // Type alias: type Name = Type or type Name<T> = Type
            self.parse_type_expr()
        } else if self.matches(&TokenType::Newtype) {
            // Newtype: newtype Foo as FooT
            self.parse_newtype_expr()
        } else if self.matches(&TokenType::Let) {
            // Explicit let binding: let x = 5
            self.parse_let_expr()
        } else if self.matches(&TokenType::Rec) {
            // Rec binding: rec x = { foo: 1 }
            self.parse_rec_expr()
        } else if self.matches(&TokenType::Var) {
            // Var binding: var x := 5
            self.parse_var_expr()
        } else if self.matches(&TokenType::CheckColon) || self.matches(&TokenType::Check) || self.matches(&TokenType::ExamplesColon) {
            // Check block: check: ... end or check "name": ... end or examples: ... end
            self.parse_check_expr()
        } else if self.matches(&TokenType::LBrace) {
            // Check if this is a tuple destructuring: {a; b} = {1; 2}
            // We need to look ahead to see if there's an = after the tuple
            let checkpoint = self.checkpoint();

            // Try to parse the tuple pattern and see if = follows
            if self.parse_tuple_for_destructure().is_ok()
                && self.matches(&TokenType::Equals) {
                    // Yes! This is tuple destructuring
                    self.restore(checkpoint);
                    return self.parse_tuple_destructure_expr();
                }

            // Not tuple destructuring, restore and parse as expression
            self.restore(checkpoint);
            self.parse_expr()
        } else if self.matches(&TokenType::Shadow) {
            // Shadow binding: shadow x = value
            self.parse_shadow_let_expr()
        } else if self.matches(&TokenType::Name) {
            // Check if this is an implicit let binding: x = value or x :: Type = value
            // OR a contract statement: x :: Type
            // Look ahead to see if there's a :: or = or := after the name
            let checkpoint = self.checkpoint();
            let _name = self.advance(); // Consume the name

            // Check for type annotation first
            if self.matches(&TokenType::ColonColon) {
                // Has type annotation - could be contract or let binding
                // Need to look further ahead to check for = after the type
                self.advance(); // consume '::'

                // Check for generic parameters first: name :: <T, U>
                if self.matches(&TokenType::Lt) {
                    // Has generic parameters, so it's definitely a contract statement
                    self.restore(checkpoint);
                    self.parse_contract_stmt()
                } else {
                    // Try to parse the type annotation
                    if self.parse_ann().is_ok() {
                        if self.matches(&TokenType::Equals) {
                            // Has = after type, so it's a let binding: x :: Type = value
                            self.restore(checkpoint);
                            self.parse_implicit_let_expr()
                        } else {
                            // No = after type, so it's a contract statement: x :: Type
                            self.restore(checkpoint);
                            self.parse_contract_stmt()
                        }
                    } else {
                        // Failed to parse type annotation, restore and try as expression
                        self.restore(checkpoint);
                        self.parse_expr()
                    }
                }
            } else if self.matches(&TokenType::Equals) {
                // Implicit let binding: x = value
                self.restore(checkpoint);
                self.parse_implicit_let_expr()
            } else if self.matches(&TokenType::ColonEquals) {
                // Implicit var binding: x := value
                self.restore(checkpoint);
                self.parse_implicit_var_expr()
            } else {
                // Not a binding, restore and parse as expression
                self.restore(checkpoint);
                self.parse_expr()
            }
        } else {
            // Default: try to parse as expression
            self.parse_expr()
        }
    }

    /// Parses user-defined block expressions like: block: 5 end, block: x = 1 x + 2 end
    /// NOTE: "block:" is tokenized as a single Block token that includes the colon
    fn parse_block_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Block)?;
        // No need to expect Colon - it's included in the Block token

        // Parse statements until we hit 'end'
        let mut stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_block_statement()?;
            stmts.push(Box::new(stmt));
        }

        let end = self.expect(TokenType::End)?;

        // Create the SBlock wrapper
        let block_body = Expr::SBlock {
            l: start.location.span(end.location),
            stmts,
        };

        // Wrap in SUserBlock
        Ok(Expr::SUserBlock {
            l: start.location.span(end.location),
            body: Box::new(block_body),
        })
    }

    /// if-expr: IF expr COLON body (ELSE-IF expr COLON body)* (ELSE-COLON body)? END
    /// Parses if expressions like: if true: 1 else: 2 end
    fn parse_if_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::If)?;

        // Parse the first branch (always present)
        let test = self.parse_expr()?;

        // Check if using block: or just :
        // Note: TokenType::Block already includes the colon ("block:")
        let blocky = if self.matches(&TokenType::Block) {
            self.advance(); // Consume "block:" token
            true
        } else {
            self.expect(TokenType::Colon)?;
            false
        };

        // Parse the body (statements until else/elseif/end)
        let body_start = self.start_loc();
        let mut then_stmts = Vec::new();
        while !self.matches(&TokenType::ElseColon)
            && !self.matches(&TokenType::ElseIf)
            && !self.matches(&TokenType::End)
            && !self.is_at_end()
        {
            let stmt = self.parse_block_statement()?;
            then_stmts.push(Box::new(stmt));
        }

        // Create the body as an SBlock
        let body_loc = self.block_loc(&then_stmts, body_start);
        let body = Expr::SBlock {
            l: body_loc,
            stmts: then_stmts,
        };

        // Create the first branch (span from test to body)
        let branch_loc = test.get_loc().span(body_loc);
        let mut branches = vec![IfBranch {
            node_type: "s-if-branch".to_string(),
            l: branch_loc,
            test: Box::new(test),
            body: Box::new(body),
        }];

        // Parse optional else-if branches
        while self.matches(&TokenType::ElseIf) {
            self.advance();
            let test = self.parse_expr()?;
            self.expect(TokenType::Colon)?;

            let elseif_body_start = self.start_loc();
            let mut elseif_stmts = Vec::new();
            while !self.matches(&TokenType::ElseColon)
                && !self.matches(&TokenType::ElseIf)
                && !self.matches(&TokenType::End)
                && !self.is_at_end()
            {
                let stmt = self.parse_block_statement()?;
                elseif_stmts.push(Box::new(stmt));
            }

            let elseif_body_loc = self.block_loc(&elseif_stmts, elseif_body_start);
            let body = Expr::SBlock {
                l: elseif_body_loc,
                stmts: elseif_stmts,
            };

            let elseif_branch_loc = test.get_loc().span(elseif_body_loc);
            branches.push(IfBranch {
                node_type: "s-if-branch".to_string(),
                l: elseif_branch_loc,
                test: Box::new(test),
                body: Box::new(body),
            });
        }

        // Parse optional else clause
        let else_expr = if self.matches(&TokenType::ElseColon) {
            self.advance();

            let else_start = self.start_loc();
            let mut else_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_block_statement()?;
                else_stmts.push(Box::new(stmt));
            }

            let else_loc = self.block_loc(&else_stmts, else_start);
            Some(Box::new(Expr::SBlock {
                l: else_loc,
                stmts: else_stmts,
            }))
        } else {
            None
        };

        let end = self.expect(TokenType::End)?;
        let loc = start.location.span(end.location);

        // Return SIfElse if there's an else clause, otherwise SIf
        if let Some(else_body) = else_expr {
            Ok(Expr::SIfElse {
                l: loc,
                branches,
                _else: else_body,
                blocky,
            })
        } else {
            Ok(Expr::SIf {
                l: loc,
                branches,
                blocky,
            })
        }
    }

    /// ask-expr: ASK COLON [PIPE test THENCOLON block]+ [PIPE OTHERWISECOLON block] END
    /// Parses ask expressions (if-pipe-else in AST):
    /// ask:
    ///   | x > 0 then: "positive"
    ///   | x < 0 then: "negative"
    ///   | otherwise: "zero"
    /// end
    fn parse_ask_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Ask)?;
        self.expect(TokenType::Colon)?;

        let mut branches = Vec::new();
        let mut else_branch = None;

        // Parse branches (at least one required)
        while self.matches(&TokenType::Bar) {
            self.advance(); // consume |

            // Check if this is an "otherwise" branch
            let is_otherwise = if self.matches(&TokenType::OtherwiseColon) {
                self.advance(); // consume "otherwise:"
                true
            } else if self.peek().token_type == TokenType::Name && self.peek().value == "otherwise" {
                self.advance(); // consume "otherwise"
                self.expect(TokenType::Colon)?; // consume ":"
                true
            } else {
                false
            };

            if is_otherwise {
                // Parse the else body
                let mut else_stmts = Vec::new();
                while !self.matches(&TokenType::End)
                    && !self.matches(&TokenType::Bar)
                    && !self.is_at_end()
                {
                    let stmt = self.parse_block_statement()?;
                    else_stmts.push(Box::new(stmt));
                }

                else_branch = Some(Box::new(Expr::SBlock {
                    l: self.peek().location,
                    stmts: else_stmts,
                }));

                // After otherwise, we should be at end
                break;
            } else {
                // Parse test expression
                let test = self.parse_expr()?;

                // Expect "then:" - could be ThenColon token or Name("then") + Colon
                if self.matches(&TokenType::ThenColon) {
                    self.advance();
                } else if self.peek().token_type == TokenType::Name && self.peek().value == "then" {
                    self.advance(); // consume "then"
                    self.expect(TokenType::Colon)?; // consume ":"
                } else {
                    return Err(ParseError::general(self.peek(), "Expected 'then:' after test expression"));
                }

                // Parse the body
                let mut body_stmts = Vec::new();
                while !self.matches(&TokenType::End)
                    && !self.matches(&TokenType::Bar)
                    && !self.is_at_end()
                {
                    let stmt = self.parse_block_statement()?;
                    body_stmts.push(Box::new(stmt));
                }

                let body = Expr::SBlock {
                    l: self.peek().location,
                    stmts: body_stmts,
                };

                branches.push(IfPipeBranch {
                    node_type: "s-if-pipe-branch".to_string(),
                    l: self.peek().location,
                    test: Box::new(test),
                    body: Box::new(body),
                });
            }
        }

        let end = self.expect(TokenType::End)?;
        let loc = start.location.span(end.location);

        // Return SIfPipeElse if there's an else clause, otherwise SIfPipe
        if let Some(else_body) = else_branch {
            Ok(Expr::SIfPipeElse {
                l: loc,
                branches,
                _else: else_body,
                blocky: false,
            })
        } else {
            Ok(Expr::SIfPipe {
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

        // Parse the iterator expression - can be any postfix expression:
        // - Simple: map
        // - Dot access: lists.map2
        // - Function call: make-iterator(config)
        // - Chained: make-iterator(config).with-logging
        let mut iterator = self.parse_prim_expr()?;

        // Handle postfix operators (dot access, function calls, etc.)
        loop {
            if self.matches(&TokenType::Dot) {
                self.advance();
                let field_token = self.parse_field_name()?;

                let start_loc = iterator.get_loc();

                iterator = Expr::SDot {
                    l: start_loc.span(field_token.location),
                    obj: Box::new(iterator),
                    field: field_token.value.clone(),
                };
            } else if self.matches(&TokenType::ParenNoSpace) {
                // Check if this paren starts for-bindings (contains FROM) or is a function call
                // Look ahead to see if we have "name FROM" pattern
                let saved_pos = self.current;
                self.advance(); // consume (

                // Check if next tokens look like "binding FROM" (for-bindings)
                // or just an expression (function call args)
                let is_for_binding = if self.matches(&TokenType::Shadow) {
                    // Starts with "shadow" - definitely a for-binding
                    // Pattern: "shadow name FROM" or "shadow name :: Type FROM"
                    true
                } else if self.matches(&TokenType::Name) {
                    // Peek ahead to see if there's a FROM or :: after the name
                    // Pattern can be: "name FROM" or "name :: Type FROM"
                    let next_tok = self.peek_ahead(1);
                    next_tok.token_type == TokenType::From
                        || next_tok.token_type == TokenType::ColonColon  // Type annotation
                } else if self.matches(&TokenType::LBrace) {
                    // Could be tuple binding {x; y} FROM ...
                    // For now, assume it's a for-binding
                    true
                } else {
                    false
                };

                // Restore position
                self.current = saved_pos;

                if is_for_binding {
                    // This is the start of for-bindings, stop postfix parsing
                    break;
                }

                // This is a function call
                self.advance(); // consume (

                let args: Vec<Box<Expr>> = if !self.matches(&TokenType::RParen) {
                    self.parse_comma_list(|p| Ok(Box::new(p.parse_binop_expr()?)))?
                } else {
                    Vec::new()
                };

                let rparen = self.expect(TokenType::RParen)?;

                let start_loc = iterator.get_loc();

                iterator = Expr::SApp {
                    l: start_loc.span(rparen.location),
                    _fun: Box::new(iterator),
                    args,
                };
            } else if self.matches(&TokenType::LtNoSpace) {
                // Type instantiation: for fold<T, U>(...)
                iterator = self.parse_instantiate_expr(iterator)?;
            } else {
                break;
            }
        }

        // Now expect opening paren (no space) for the for bindings
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

                // Create ForBind (location spans from bind to value)
                let bind_loc = bind.get_loc().span(*value.get_loc());
                bindings.push(ForBind {
                    node_type: "s-for-bind".to_string(),
                    l: bind_loc,
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
        let blocky = self.parse_block_separator()?;

        // Parse the body (statements until END)
        let body_start = self.start_loc();
        let mut body_stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_block_statement()?;
            body_stmts.push(Box::new(stmt));
        }

        // Create the body as an SBlock
        let body_loc = self.block_loc(&body_stmts, body_start);
        let body = Box::new(Expr::SBlock {
            l: body_loc,
            stmts: body_stmts,
        });

        let end = self.expect(TokenType::End)?;
        let loc = start.location.span(end.location);

        Ok(Expr::SFor {
            l: loc,
            iterator: Box::new(iterator),
            bindings,
            ann,
            body,
            blocky,
        })
    }

    /// multi-let-expr: LET let-binding (COMMA let-binding)* (BLOCK|COLON) block END
    /// let-binding: let-expr | var-expr
    /// let-expr: toplevel-binding EQUALS binop-expr
    /// var-expr: VAR toplevel-binding EQUALS binop-expr
    ///
    /// Parses multi-let expressions: let x = 5, y = 10 block: ... end
    fn parse_let_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Let)?;

        // Parse first binding: binding = binop-expr
        let mut binds = Vec::new();

        // First binding (always a let-binding)
        let bind = self.parse_bind()?;
        self.expect(TokenType::Equals)?;
        let value = self.parse_binop_expr()?;
        binds.push(LetBind::SLetBind {
            l: self.peek().location,
            b: bind.clone(),
            value: Box::new(value.clone()),
        });

        // Additional bindings (comma-separated)
        // Each can be either: binding = expr  OR  var binding = expr
        if self.matches(&TokenType::Comma) {
            self.advance(); // consume first comma
            binds.extend(self.parse_comma_list_no_trailing(|p| {
                // Check if this is a var binding
                let is_var = p.matches(&TokenType::Var);
                if is_var {
                    p.expect(TokenType::Var)?;
                    let bind = p.parse_bind()?;
                    p.expect(TokenType::Equals)?;
                    let value = p.parse_binop_expr()?;
                    Ok(LetBind::SVarBind {
                        l: p.peek().location,
                        b: bind,
                        value: Box::new(value),
                    })
                } else {
                    // Regular let binding
                    let bind = p.parse_bind()?;
                    p.expect(TokenType::Equals)?;
                    let value = p.parse_binop_expr()?;
                    Ok(LetBind::SLetBind {
                        l: p.peek().location,
                        b: bind,
                        value: Box::new(value),
                    })
                }
            })?);
        }

        // Must have a body with (BLOCK|COLON) block END
        let blocky = self.matches(&TokenType::Block);
        if blocky {
            self.expect(TokenType::Block)?;
        } else {
            self.expect(TokenType::Colon)?;
        }

        // Parse block body
        let mut body_stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_block_statement()?;
            body_stmts.push(Box::new(stmt));
        }

        let end = self.expect(TokenType::End)?;

        let body = Expr::SBlock {
            l: start.location.span(end.location),
            stmts: body_stmts,
        };

        Ok(Expr::SLetExpr {
            l: start.location.span(end.location),
            binds,
            body: Box::new(body),
            blocky,
        })
    }

    /// shadow-expr: SHADOW bind = expr
    ///             | SHADOW bind = expr BLOCK body END
    /// Parses shadow bindings: shadow x = 10
    fn parse_shadow_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Shadow)?;

        // Parse binding with shadows=true: name [:: type]
        let bind = self.parse_bind_with_shadow(true)?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        // Create LetBind
        let let_bind = LetBind::SLetBind {
            l: self.peek().location,
            b: bind.clone(),
            value: Box::new(value.clone()),
        };

        // Check if there's a block body (either "block:" or just ":")
        let body = if self.matches(&TokenType::Block) || self.matches(&TokenType::Colon) {
            // Consume "block" if present
            if self.matches(&TokenType::Block) {
                self.expect(TokenType::Block)?;
            }

            self.expect(TokenType::Colon)?;

            // Parse block body
            let mut body_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                body_stmts.push(Box::new(stmt));
            }

            self.expect(TokenType::End)?;

            Expr::SBlock {
                l: self.peek().location,
                stmts: body_stmts,
            }
        } else {
            // No explicit body, just use the value
            value.clone()
        };

        let end = self.last_token(&start);

        Ok(Expr::SLetExpr {
            l: start.location.span(end.location),
            binds: vec![let_bind],
            body: Box::new(body),
            blocky: false,
        })
    }

    /// var-expr: VAR bind = expr
    /// Parses mutable variable bindings: var x = 5
    fn parse_var_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Var)?;

        // Parse binding: name [:: type]
        let bind = self.parse_bind()?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        let end = self.last_token(&start);

        // Var bindings are s-var statements
        Ok(Expr::SVar {
            l: start.location.span(end.location),
            name: bind,
            value: Box::new(value),
        })
    }

    /// rec-expr: REC toplevel-binding EQUALS binop-expr
    /// Parses recursive bindings: rec x = { foo: 1 }
    fn parse_rec_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Rec)?;

        // Parse binding: name [:: type]
        let bind = self.parse_bind()?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        let end = self.last_token(&start);

        // Rec bindings are s-rec statements
        Ok(Expr::SRec {
            l: start.location.span(end.location),
            name: bind,
            value: Box::new(value),
        })
    }

    /// Helper function to parse a tuple pattern for destructuring (lookahead only)
    /// Returns Ok(()) if it successfully parses a tuple pattern, Err otherwise
    fn parse_tuple_for_destructure(&mut self) -> ParseResult<()> {
        self.expect(TokenType::LBrace)?;

        // Parse first field (could be name or nested tuple)
        if !self.matches(&TokenType::RBrace) {
            self.parse_tuple_field_for_destructure()?;

            // Parse remaining fields
            while self.matches(&TokenType::Semi) {
                self.advance(); // consume semicolon

                // Check for trailing semicolon before }
                if self.matches(&TokenType::RBrace) {
                    break;
                }

                self.parse_tuple_field_for_destructure()?;
            }
        }

        self.expect(TokenType::RBrace)?;
        Ok(())
    }

    /// Helper to parse a single field in tuple destructuring (can be name or nested tuple)
    fn parse_tuple_field_for_destructure(&mut self) -> ParseResult<()> {
        // Check for optional shadow keyword
        if self.matches(&TokenType::Shadow) {
            self.advance(); // consume 'shadow'
        }

        // Check if this is a nested tuple
        if self.matches(&TokenType::LBrace) {
            // Recursively parse nested tuple
            self.parse_tuple_for_destructure()
        } else {
            // Parse name
            self.parse_name()?;

            // Check for optional type annotation: :: ann
            if self.matches(&TokenType::ColonColon) {
                self.advance(); // consume ::
                self.parse_ann()?;
            }

            Ok(())
        }
    }

    /// Parses tuple destructuring: {a; b; c} = {1; 2; 3}
    fn parse_tuple_destructure_expr(&mut self) -> ParseResult<Expr> {
        let start = self.start_loc();

        // Parse the tuple bind pattern: {a; b; c}
        let tuple_bind = self.parse_tuple_bind()?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse the value expression
        let value = self.parse_expr()?;

        // Create an s-let with tuple bind
        Ok(Expr::SLet {
            l: self.make_span(start),
            name: tuple_bind,
            value: Box::new(value),
            keyword_val: false,
        })
    }

    /// letrec-expr: LETREC let-expr (COMMA let-expr)* (BLOCK|COLON) block END
    /// Parses recursive let bindings: letrec f = lam(n): ... end: body end
    fn parse_letrec_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Letrec)?;

        // Parse first binding
        let mut binds = Vec::new();

        // Parse first let-expr: binding = expr
        let bind = self.parse_bind()?;
        self.expect(TokenType::Equals)?;
        let value = self.parse_expr()?;

        binds.push(LetrecBind {
            node_type: "s-letrec-bind".to_string(),
            l: self.peek().location,
            b: bind,
            value: Box::new(value),
        });

        // Parse additional bindings if present (COMMA let-expr)*
        if self.matches(&TokenType::Comma) {
            self.advance(); // consume first comma
            binds.extend(self.parse_comma_list_no_trailing(|p| {
                let bind = p.parse_bind()?;
                p.expect(TokenType::Equals)?;
                let value = p.parse_expr()?;
                Ok(LetrecBind {
                    node_type: "s-letrec-bind".to_string(),
                    l: p.peek().location,
                    b: bind,
                    value: Box::new(value),
                })
            })?);
        }

        // Parse body: (BLOCK|COLON) block END
        // Note: TokenType::Block is "block:" as a single token
        let blocky = if self.matches(&TokenType::Block) {
            self.advance(); // Consume "block:"
            true
        } else {
            self.expect(TokenType::Colon)?; // Consume ":"
            false
        };

        // Parse block body
        let mut body_stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_expr()?;
            body_stmts.push(Box::new(stmt));
        }

        let end = self.expect(TokenType::End)?;

        // Body is always wrapped in an s-block
        let body = Expr::SBlock {
            l: self.peek().location,
            stmts: body_stmts,
        };

        Ok(Expr::SLetrec {
            l: start.location.span(end.location),
            binds,
            body: Box::new(body),
            blocky,
        })
    }

    /// Implicit let binding: x = value (no "let" keyword)
    /// Creates an s-let statement (not s-let-expr)
    /// Used in block contexts where we want statement-style bindings
    fn parse_implicit_let_expr(&mut self) -> ParseResult<Expr> {
        let start = self.start_loc();

        // Parse binding: name [:: type]
        let bind = self.parse_bind()?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        Ok(Expr::SLet {
            l: self.make_span(start),
            name: bind,
            value: Box::new(value),
            keyword_val: false,
        })
    }

    /// Shadow let binding: shadow x = value
    /// Creates an s-let statement with shadows = true
    fn parse_shadow_let_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Shadow)?;

        // Parse binding: name [:: type]
        let mut bind = self.parse_bind()?;

        // Update the bind to set shadows = true
        match &mut bind {
            Bind::SBind { shadows, .. } => {
                *shadows = true;
            }
            Bind::STupleBind { fields, .. } => {
                // Set shadows for all fields in tuple binding
                for field in fields.iter_mut() {
                    if let Bind::SBind { shadows, .. } = field {
                        *shadows = true;
                    }
                }
            }
        }

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        let end = self.last_token(&start);

        Ok(Expr::SLet {
            l: start.location.span(end.location),
            name: bind,
            value: Box::new(value),
            keyword_val: false,
        })
    }

    /// Contract statement: name :: Type
    /// Grammar: contract-stmt: NAME COLONCOLON ty-params (ann | noparen-arrow-ann)
    /// Example: foo :: Number -> String
    fn parse_contract_stmt(&mut self) -> ParseResult<Expr> {
        let start = self.start_loc();

        // Parse contract name
        let name = self.parse_name()?;

        // Expect ::
        self.expect(TokenType::ColonColon)?;

        // Parse optional type parameters <T, U>
        let params = self.parse_opt_type_params()?;

        // Parse type annotation
        // Contract statements allow noparen-arrow-ann: args -> ret without parentheses
        let ann = self.parse_contract_ann()?;

        Ok(Expr::SContract {
            l: self.make_span(start),
            name,
            params,
            ann,
        })
    }

    /// Parse annotation for contract statements
    /// Supports noparen-arrow-ann: Type -> Type without requiring parentheses
    fn parse_contract_ann(&mut self) -> ParseResult<Ann> {
        let start_pos = self.current;

        // Parse comma-separated annotations (arguments)
        let first_arg = self.parse_ann()?;
        let args = if self.matches(&TokenType::Comma) {
            self.advance(); // consume first comma
            let mut all_args = vec![first_arg];
            all_args.extend(self.parse_comma_list_no_trailing(|p| p.parse_ann())?);
            all_args
        } else {
            vec![first_arg]
        };

        // Check if there's an arrow (for noparen-arrow-ann)
        if self.matches(&TokenType::ThinArrow) {
            self.advance(); // consume ->

            // Parse return type
            let ret = Box::new(self.parse_ann()?);

            let end_pos = self.current - 1;
            let end_token = &self.tokens[end_pos];
            let loc = self.tokens[start_pos].location.span(end_token.location);

            return Ok(Ann::AArrow {
                l: loc,
                args,
                ret,
                use_parens: false, // Not using parentheses for contract annotations
            });
        }

        // No arrow, just return the annotation as-is (should be single arg)
        if args.len() == 1 {
            Ok(args.into_iter().next().unwrap())
        } else {
            Err(ParseError::general(
                self.peek(),
                "Contract annotation requires arrow (->) when using multiple arguments"
            ))
        }
    }

    /// Standalone let binding: x = value (no "let" keyword)
    /// Creates an s-let-expr (expression form that returns a value)
    /// Used for standalone expression parsing
    fn parse_standalone_let_expr(&mut self) -> ParseResult<Expr> {
        let start = self.start_loc();

        // Parse binding: name [:: type]
        let bind = self.parse_bind()?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        // Return SLet (simple let binding statement)
        // This is for implicit let bindings like: x = 5
        Ok(Expr::SLet {
            l: self.make_span(start),
            name: bind,
            value: Box::new(value),
            keyword_val: false,
        })
    }

    /// Implicit var binding: x := value (no "var" keyword)
    /// This is actually an assignment (s-assign), not a var declaration
    fn parse_implicit_var_expr(&mut self) -> ParseResult<Expr> {
        let start = self.start_loc();

        // Parse name (just the identifier, no type annotation)
        let name = self.parse_name()?;

        // Expect :=
        self.expect(TokenType::ColonEquals)?;

        // Parse value expression
        let value = self.parse_expr()?;

        // This is an assignment, not a var declaration
        Ok(Expr::SAssign {
            l: self.make_span(start),
            id: name,
            value: Box::new(value),
        })
    }

    /// cases-expr: CASES (type) expr: branches ... END
    /// Parses cases expressions like: cases(Either) e: | left(v) => v | right(v) => v end
    fn parse_cases_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Cases)?;

        // Expect any form of left paren
        self.expect_any_lparen()?;

        // Parse type annotation
        let typ = self.parse_ann()?;

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Parse value expression
        let val = self.parse_expr()?;

        // Expect colon or block
        let blocky = self.parse_block_separator()?;

        // Parse branches (each starts with |)
        let mut branches = Vec::new();
        let mut else_branch: Option<Box<Expr>> = None;

        while self.matches(&TokenType::Bar) {
            self.advance(); // consume |

            // Check for else branch (| else => expr)
            if self.matches(&TokenType::Else) {
                self.advance();
                self.expect(TokenType::ThickArrow)?;

                // Parse else body (statements until end)
                let mut else_stmts = Vec::new();
                while !self.matches(&TokenType::End) && !self.is_at_end() {
                    let stmt = self.parse_block_statement()?;
                    else_stmts.push(Box::new(stmt));
                }

                else_branch = Some(Box::new(Expr::SBlock {
                    l: self.peek().location,
                    stmts: else_stmts,
                }));
                break;
            }

            // Parse branch pattern name
            let pattern_start = self.peek().clone();
            let name_token = self.expect(TokenType::Name)?;
            let name = name_token.value.clone();
            let mut pattern_loc = pattern_start.location.span(name_token.location);

            // Check for arguments: name(args)
            // Track whether parentheses were present to distinguish:
            // - | foo => ... (singleton, no parens)
            // - | foo() => ... (regular branch with empty args, parens present)
            let (has_parens, args) = if self.matches(&TokenType::LParen) || self.matches(&TokenType::ParenNoSpace) {
                self.advance(); // consume (

                let args = if self.matches(&TokenType::RParen) {
                    Vec::new()
                } else {
                    // Parse comma-separated bindings
                    self.parse_comma_list(|p| {
                        // Check for optional 'ref' keyword
                        let field_type = if p.matches(&TokenType::Ref) {
                            p.advance(); // consume 'ref'
                            CasesBindType::SRef
                        } else {
                            CasesBindType::SNormal
                        };

                        // Parse binding - could be tuple bind { x; y } or regular bind
                        let bind = if p.matches(&TokenType::LBrace) {
                            p.parse_tuple_bind()?
                        } else {
                            p.parse_bind()?
                        };

                        let l = match &bind {
                            Bind::SBind { l, .. } => *l,
                            Bind::STupleBind { l, .. } => *l,
                        };
                        Ok(CasesBind {
                            node_type: "s-cases-bind".to_string(),
                            l,
                            field_type,
                            bind,
                        })
                    })?
                };

                let rparen = self.expect(TokenType::RParen)?;
                pattern_loc = pattern_start.location.span(rparen.location);
                (true, args) // parens were present
            } else {
                (false, Vec::new()) // no parens
            };

            // Expect =>
            self.expect(TokenType::ThickArrow)?;

            // Parse body (statements until next | or end)
            let mut body_stmts = Vec::new();
            while !self.matches(&TokenType::Bar) && !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_block_statement()?;
                body_stmts.push(Box::new(stmt));
            }

            let body = Box::new(Expr::SBlock {
                l: self.peek().location,
                stmts: body_stmts,
            });

            // Create the appropriate branch type
            // Singleton only if NO parentheses were present
            let branch = if !has_parens {
                CasesBranch::SSingletonCasesBranch {
                    l: self.peek().location,
                    pattern_loc,
                    name,
                    body,
                }
            } else {
                // Has parentheses (even if args list is empty)
                CasesBranch::SCasesBranch {
                    l: self.peek().location,
                    pattern_loc,
                    name,
                    args,
                    body,
                }
            };

            branches.push(branch);
        }

        let end = self.expect(TokenType::End)?;
        let loc = start.location.span(end.location);

        // Return the appropriate cases expression type
        if let Some(else_expr) = else_branch {
            Ok(Expr::SCasesElse {
                l: loc,
                typ,
                val: Box::new(val),
                branches,
                _else: else_expr,
                blocky,
            })
        } else {
            Ok(Expr::SCases {
                l: loc,
                typ,
                val: Box::new(val),
                branches,
                blocky,
            })
        }
    }

    /// when-expr: WHEN expr: block END
    /// Parses when expressions like: when true: print("yes") end
    fn parse_when_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::When)?;

        // Parse the test expression
        let test = self.parse_expr()?;

        // Expect colon or block
        let blocky = self.parse_block_separator()?;

        // Parse the block (statements until end)
        let mut block_stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_block_statement()?;
            block_stmts.push(Box::new(stmt));
        }

        let end = self.expect(TokenType::End)?;
        let loc = start.location.span(end.location);

        // Create the block
        let block = Expr::SBlock {
            l: self.peek().location,
            stmts: block_stmts,
        };

        Ok(Expr::SWhen {
            l: loc,
            test: Box::new(test),
            block: Box::new(block),
            blocky,
        })
    }
}

// ============================================================================
// SECTION 8: Function Parsing
// ============================================================================

/// Helper struct for parsing function-like constructs (fun, lam, method)
struct FunctionParts {
    name: String,
    type_params: Vec<Name>,
    args: Vec<Bind>,
    ann: Ann,
    blocky: bool,
    doc: String,
    body: Box<Expr>,
    check_loc: Option<Loc>,
    check: Option<Box<Expr>>,
    end_loc: Loc,
}

impl Parser {
    /// fun-expr: FUN name<typarams>(args) ann: doc body where END
    /// Parses function declarations like: fun f(x): x + 1 end
    /// type-expr: TYPE NAME ty-params EQUALS ann
    /// Parses type alias declarations like:
    /// - type N = Number
    /// - type Loc = SL.Srcloc
    /// - type A<T> = T
    fn parse_type_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Type)?;

        // Parse type alias name
        let name = self.parse_name()?;

        // Parse optional type parameters <T, U, V>
        let params: Vec<Name> = if self.matches(&TokenType::Lt) || self.matches(&TokenType::LtNoSpace) {
            self.advance(); // consume '<'
            let type_params = self.parse_comma_list(|p| p.parse_name())?;
            self.expect(TokenType::Gt)?; // consume '>'
            type_params
        } else {
            Vec::new()
        };

        // Expect equals
        self.expect(TokenType::Equals)?;

        // Parse the type annotation
        let ann = self.parse_ann()?;

        let end = self.last_token(&start);

        Ok(Expr::SType {
            l: start.location.span(end.location),
            name,
            params,
            ann,
        })
    }

    /// newtype-expr: NEWTYPE NAME AS NAME
    /// Creates a type alias with a new name (newtype semantics)
    /// Example: newtype Foo as FooT
    fn parse_newtype_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Newtype)?;

        // Parse the source type name
        let name = self.parse_name()?;

        // Expect 'as'
        self.expect(TokenType::As)?;

        // Parse the new type name
        let namet = self.parse_name()?;

        let end = self.last_token(&start);

        Ok(Expr::SNewtype {
            l: start.location.span(end.location),
            name,
            namet,
        })
    }

    /// Shared parsing logic for function-like constructs
    /// Configurable via flags to handle differences between fun/lam/method
    fn parse_function_body(
        &mut self,
        parse_name: bool,
        parse_type_params: bool,
        parse_where: bool,
        allow_coloncolon_ann: bool,  // Lambda allows :: in addition to ->
    ) -> ParseResult<FunctionParts> {
        // 1. Parse optional name
        let name = if parse_name {
            let name_token = self.expect(TokenType::Name)?;
            name_token.value.clone()
        } else {
            String::new()
        };

        // 2. Parse optional type parameters <T, U, V>
        let type_params = if parse_type_params {
            self.parse_opt_type_params()?
        } else {
            Vec::new()
        };

        // 3. Parse parameters (args)
        self.expect_any_lparen()?;
        let args = if self.matches(&TokenType::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_bind())?
        };
        self.expect(TokenType::RParen)?;

        // 4. Parse optional return annotation (-> or :: for lambdas)
        let ann = if self.matches(&TokenType::ThinArrow) {
            self.advance();
            self.parse_ann()?
        } else if allow_coloncolon_ann && self.matches(&TokenType::ColonColon) {
            self.advance();
            self.parse_ann()?
        } else {
            Ann::ABlank
        };

        // 5. Parse block separator (: or block:)
        let blocky = self.parse_block_separator()?;

        // 6. Parse optional doc string
        let doc = self.parse_opt_doc_string()?;

        // 7. Parse body statements until END or WHERE
        let mut body_stmts = Vec::new();
        while !self.matches(&TokenType::End)
            && !self.matches(&TokenType::Where)
            && !self.is_at_end()
        {
            let stmt = self.parse_block_statement()?;
            body_stmts.push(Box::new(stmt));
        }

        // 8. Parse optional where clause
        let (check, check_loc) = if parse_where && self.matches(&TokenType::Where) {
            let where_token = self.advance().clone();
            let check_loc = where_token.location.span(where_token.location);
            let mut where_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                where_stmts.push(Box::new(stmt));
            }
            let check_block = Box::new(Expr::SBlock {
                l: check_loc,
                stmts: where_stmts,
            });
            (Some(check_block), Some(check_loc))
        } else {
            (None, None)
        };

        // 9. Expect END keyword
        let end = self.expect(TokenType::End)?;

        // 10. Wrap body statements in SBlock
        let body = Box::new(Expr::SBlock {
            l: self.peek().location,
            stmts: body_stmts,
        });

        Ok(FunctionParts {
            name,
            type_params,
            args,
            ann,
            blocky,
            doc,
            body,
            check_loc,
            check,
            end_loc: end.location,
        })
    }

    fn parse_fun_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Fun)?;

        // Use shared helper: fun has name, type params, where clause, no :: ann
        let parts = self.parse_function_body(true, true, true, false)?;

        Ok(Expr::SFun {
            l: start.location.span(parts.end_loc),
            name: parts.name,
            params: parts.type_params,
            args: parts.args,
            ann: parts.ann,
            doc: parts.doc,
            body: parts.body,
            check_loc: parts.check_loc,
            check: parts.check,
            blocky: parts.blocky,
        })
    }

    /// lambda-expr: LAM [LT type-params GT] LPAREN [args] RPAREN [COLONCOLON ann] COLON body END
    /// Parses lambda expressions like: lam(): 5 end, lam(x): x + 1 end, lam<A>(x :: A): x end
    fn parse_lambda_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Lam)?;

        // Use shared helper: lam has no name, has type params, no where clause, allows :: ann
        let parts = self.parse_function_body(false, true, false, true)?;

        Ok(Expr::SLam {
            l: start.location.span(parts.end_loc),
            name: String::new(), // Anonymous lambda
            params: parts.type_params,
            args: parts.args,
            ann: parts.ann,
            doc: parts.doc,
            body: parts.body,
            check_loc: None,  // Lambdas don't have where clauses
            check: None,
            blocky: parts.blocky,
        })
    }

    /// curly-lambda-expr: {(args): body}
    /// Shorthand syntax for lambda expressions using curly braces instead of lam/end
    /// Example: {(x): x + 1} is equivalent to lam(x): x + 1 end
    fn parse_curly_lambda_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::LBrace)?;

        // Expect any form of left paren
        self.expect_any_lparen()?;

        // Parse parameters (comma-separated bindings)
        let args = if self.matches(&TokenType::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_bind())?
        };

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Check if this is a blocky lambda {(x) block: ...} or regular {(x): ...}
        let blocky = self.parse_block_separator()?;

        // Parse body as a block of statements
        // Body continues until we hit the closing brace
        let mut stmts = Vec::new();
        while !self.matches(&TokenType::RBrace) && !self.is_at_end() {
            let stmt = self.parse_block_statement()?;
            stmts.push(Box::new(stmt));
        }

        // Expect closing brace
        let end = self.expect(TokenType::RBrace)?;

        // Body is the block of statements
        let body = Expr::SBlock {
            l: self.peek().location,
            stmts,
        };

        Ok(Expr::SLam {
            l: start.location.span(end.location),
            name: String::new(), // Anonymous lambda
            params: Vec::new(),  // No type parameters in curly brace syntax
            args,
            ann: Ann::ABlank,    // No return type annotation in curly brace syntax
            doc: String::new(),  // No doc string in curly brace syntax
            body: Box::new(body),
            check_loc: None,
            check: None,
            blocky,              // Set based on whether block: or : was used
        })
    }

    /// method-expr: METHOD(args) ann: doc body where END
    fn parse_method_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Method)?;

        // Use shared helper: method has no name/type params, has where clause, no :: ann
        let parts = self.parse_function_body(false, false, true, false)?;

        Ok(Expr::SMethod {
            l: start.location.span(parts.end_loc),
            name: String::new(),   // Methods have no name
            params: Vec::new(),    // Methods have no type parameters
            args: parts.args,
            ann: parts.ann,
            doc: parts.doc,
            body: parts.body,
            check_loc: parts.check_loc,
            check: parts.check,
            blocky: parts.blocky,
        })
    }
}

// ============================================================================
// SECTION 9: Data Definition Parsing
// ============================================================================

impl Parser {
    /// data-expr: DATA name<typarams>: variants sharing where END
    fn parse_data_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Data)?;

        // Parse data type name
        let name_token = self.expect(TokenType::Name)?;
        let name = name_token.value.clone();

        // Parse optional type parameters <T, U, V>
        let params: Vec<Name> = if self.matches(&TokenType::Lt) || self.matches(&TokenType::LtNoSpace) {
            self.advance(); // consume '<'
            let type_params = self.parse_comma_list(|p| p.parse_name())?;
            self.expect(TokenType::Gt)?; // consume '>'
            type_params
        } else {
            Vec::new()
        };

        // Expect colon
        self.expect(TokenType::Colon)?;

        // Parse variants (separated by |)
        let mut variants = Vec::new();

        // First variant may be preceded by |
        if self.matches(&TokenType::Bar) {
            self.advance();
        }

        // Parse first variant
        if !self.matches(&TokenType::End)
            && !self.matches(&TokenType::Sharing)
            && !self.matches(&TokenType::Where)
            && !self.matches(&TokenType::With) {
            variants.push(self.parse_variant()?);
        }

        // Parse remaining variants (each preceded by |)
        while self.matches(&TokenType::Bar) {
            self.advance();
            if !self.matches(&TokenType::End)
                && !self.matches(&TokenType::Sharing)
                && !self.matches(&TokenType::Where)
                && !self.matches(&TokenType::With) {
                variants.push(self.parse_variant()?);
            }
        }

        // Parse optional sharing clause
        let (shared_members, mixins) = if self.matches(&TokenType::Sharing) {
            self.advance(); // consume "sharing:"

            // Parse shared members (comma-separated method/field definitions)
            // Grammar: fields: field (COMMA field)* [COMMA]
            let mut members = Vec::new();
            while !self.matches(&TokenType::End) && !self.matches(&TokenType::Where) && !self.is_at_end() {
                members.push(self.parse_obj_field()?);

                // Check for optional comma separator
                if self.matches(&TokenType::Comma) {
                    self.advance(); // consume comma
                    // If we see END or WHERE after comma, it's a trailing comma - stop parsing
                    if self.matches(&TokenType::End) || self.matches(&TokenType::Where) {
                        break;
                    }
                } else {
                    // No comma, so we're done with members
                    break;
                }
            }
            (members, Vec::new())
        } else if self.matches(&TokenType::With) {
            // Parse optional with clause
            let with_members = self.parse_data_with()?;
            (with_members, Vec::new())
        } else {
            (Vec::new(), Vec::new())
        };

        // Parse optional where clause
        let (check, check_loc) = if self.matches(&TokenType::Where) {
            let where_token = self.advance().clone();
            // The check-loc should just point to the WHERE keyword itself
            let check_loc = where_token.location.span(where_token.location);
            let mut where_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                where_stmts.push(Box::new(stmt));
            }
            let check_block = Box::new(Expr::SBlock {
                l: check_loc,
                stmts: where_stmts,
            });
            (Some(check_block), Some(check_loc))
        } else {
            (None, None)
        };

        let end = self.expect(TokenType::End)?;

        Ok(Expr::SData {
            l: start.location.span(end.location),
            name,
            params,
            mixins,
            variants,
            shared_members,
            check_loc,
            check,
        })
    }

    /// data-variant: name(members) | name
    fn parse_variant(&mut self) -> ParseResult<Variant> {
        let start = self.start_loc();
        let name_token = self.expect(TokenType::Name)?;
        let name = name_token.value.clone();
        let constr_start = name_token.location;

        // Check if this is a constructor variant with arguments
        if self.matches(&TokenType::LParen)
            || self.matches(&TokenType::ParenSpace)
            || self.matches(&TokenType::ParenNoSpace) {
            self.advance(); // consume opening paren

            // Parse variant members (comma-separated)
            let members = if self.matches(&TokenType::RParen) {
                Vec::new()
            } else {
                self.parse_comma_list(|p| p.parse_variant_member())?
            };

            let rparen = self.expect(TokenType::RParen)?;
            let constr_loc = constr_start.span(rparen.location);

            // Parse optional with clause
            let with_members = if self.matches(&TokenType::With) {
                self.parse_data_with()?
            } else {
                Vec::new()
            };

            Ok(Variant::SVariant {
                l: self.make_span(start),
                constr_loc,
                name,
                members,
                with_members,
            })
        } else {
            // Singleton variant (no arguments)
            // Parse optional with clause
            let with_members = if self.matches(&TokenType::With) {
                self.parse_data_with()?
            } else {
                Vec::new()
            };

            Ok(Variant::SSingletonVariant {
                l: self.make_span(start),
                name,
                with_members,
            })
        }
    }

    /// Parse a variant member (parameter in a data constructor)
    /// Can be: ref binding | binding | { fields }
    fn parse_variant_member(&mut self) -> ParseResult<VariantMember> {
        let start = self.start_loc();

        // Check for ref modifier
        let member_type = if self.matches(&TokenType::Ref) {
            self.advance();
            VariantMemberType::SMutable
        } else {
            VariantMemberType::SNormal
        };

        // Parse the binding - could be regular bind or tuple bind
        let bind = if self.matches(&TokenType::LBrace) {
            // Tuple pattern: { x; y; z }
            self.parse_tuple_bind()?
        } else {
            // Regular binding: x or x :: Number
            self.parse_bind()?
        };

        Ok(VariantMember {
            node_type: "s-variant-member".to_string(),
            l: self.make_span(start),
            member_type,
            bind,
        })
    }

    /// data-with: with: members END
    fn parse_data_with(&mut self) -> ParseResult<Vec<Member>> {
        self.expect(TokenType::With)?;
        // Note: TokenType::With already includes the colon ("with:")
        // so we don't need to expect a separate Colon token

        let mut members = Vec::new();
        while !self.matches(&TokenType::End)
            && !self.matches(&TokenType::Bar)
            && !self.matches(&TokenType::Sharing)
            && !self.matches(&TokenType::Where)
            && !self.is_at_end() {
            members.push(self.parse_obj_field()?);
            // Members might be separated by commas
            if self.matches(&TokenType::Comma) {
                self.advance();
            }
        }

        Ok(members)
    }
}

// ============================================================================
// SECTION 10: Table Operation Parsing
// ============================================================================

impl Parser {
    /// table-expr: table: headers row: ... end
    fn parse_table_expr(&mut self) -> ParseResult<Expr> {
        // table: col1, col2, col3
        //   row: val1, val2, val3
        //   row: val1, val2, val3
        // end
        let start = self.expect(TokenType::Table)?; // consume "table:"

        // Parse column headers (comma-separated field names)
        let mut headers = Vec::new();

        // Empty table is allowed
        if !self.matches(&TokenType::Row) && !self.matches(&TokenType::End) {
            loop {
                let name_token = self.expect(TokenType::Name)?;
                let name = name_token.value.clone();
                let field_loc = name_token.location.span(name_token.location);

                // For now, headers don't have type annotations in basic tables
                // The annotation field is just a-blank
                headers.push(FieldName {
                    node_type: "s-field-name".to_string(),
                    l: field_loc,
                    name,
                    ann: Ann::ABlank,
                });

                // Check for comma (more headers) or row/end (done with headers)
                if self.matches(&TokenType::Comma) {
                    self.advance(); // consume comma
                } else {
                    break;
                }
            }
        }

        // Parse rows
        let mut rows = Vec::new();
        while self.matches(&TokenType::Row) {
            let row_start = self.advance().clone(); // consume "row:"

            // Parse row elements (comma-separated expressions)
            let mut elems = Vec::new();
            if !self.matches(&TokenType::Row) && !self.matches(&TokenType::End) {
                loop {
                    let expr = self.parse_binop_expr()?;
                    elems.push(Box::new(expr));

                    if self.matches(&TokenType::Comma) {
                        self.advance(); // consume comma
                    } else {
                        break;
                    }
                }
            }

            // Use row_start for location - good enough for now
            rows.push(TableRow {
                node_type: "s-table-row".to_string(),
                l: row_start.location.span(row_start.location),
                elems,
            });
        }

        let end = self.expect(TokenType::End)?;
        Ok(Expr::STable {
            l: start.location.span(end.location),
            headers,
            rows,
        })
    }

    /// extract-expr: EXTRACT column FROM table END
    /// Parses extract expressions like: extract state from obj end
    fn parse_extract_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::TableExtract)?;

        // Parse column name
        let column = self.parse_name()?;

        // Expect FROM keyword
        self.expect(TokenType::From)?;

        // Parse table expression
        let table = Box::new(self.parse_expr()?);

        // Expect END keyword
        let end = self.expect(TokenType::End)?;

        Ok(Expr::STableExtract {
            l: start.location.span(end.location),
            column,
            table,
        })
    }

    /// load-table-expr: LOAD-TABLE COLON headers spec* END
    /// Parses a load-table expression like:
    ///   load-table: name, age
    ///     source: "data.csv"
    ///     sanitize name using string-sanitizer
    ///   end
    fn parse_load_table_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::LoadTable)?;
        // Note: LoadTable token already includes the colon, like "load-table:"

        // Parse comma-separated column headers
        let mut headers = Vec::new();
        if !self.matches(&TokenType::SourceColon) && !self.matches(&TokenType::Sanitize) && !self.matches(&TokenType::End) {
            loop {
                let name_token = self.expect(TokenType::Name)?;
                let name = name_token.value.clone();

                // Optional type annotation
                let ann = if self.matches(&TokenType::ColonColon) {
                    self.expect(TokenType::ColonColon)?;
                    self.parse_ann()?
                } else {
                    Ann::ABlank
                };

                headers.push(FieldName {
                    node_type: "s-field-name".to_string(),
                    l: name_token.location.span(name_token.location),
                    name,
                    ann,
                });

                if !self.matches(&TokenType::Comma) {
                    break;
                }
                self.advance(); // consume comma
            }
        }

        // Parse spec lines (source, sanitize, etc.)
        let mut spec = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            if self.matches(&TokenType::SourceColon) {
                // source: expr
                let source_tok = self.expect(TokenType::SourceColon)?;
                let src_expr = self.parse_expr()?;
                spec.push(LoadTableSpec::STableSrc {
                    l: source_tok.location.span(source_tok.location),
                    src: Box::new(src_expr),
                });
            } else if self.matches(&TokenType::Sanitize) {
                // sanitize name using expr
                let sanitize_tok = self.expect(TokenType::Sanitize)?;
                let name_token = self.expect(TokenType::Name)?;
                self.expect(TokenType::Using)?;
                let sanitizer_expr = self.parse_expr()?;

                spec.push(LoadTableSpec::SSanitize {
                    l: sanitize_tok.location.span(name_token.location),
                    name: self.token_to_name(&name_token),
                    sanitizer: Box::new(sanitizer_expr),
                });
            } else {
                break;
            }
        }

        let end = self.expect(TokenType::End)?;

        Ok(Expr::SLoadTable {
            l: start.location.span(end.location),
            headers,
            spec,
        })
    }
}

// ============================================================================
// SECTION 11: Check/Test Parsing
// ============================================================================

impl Parser {
    /// reactor-expr: REACTOR COLON fields END
    /// Parses a reactor expression like:
    ///   reactor:
    ///     init: 0,
    ///     on-tick: lam(state): state + 1 end
    ///   end
    fn parse_reactor_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Reactor)?;
        self.expect(TokenType::Colon)?;

        // Parse comma-separated fields (same as object fields)
        let mut fields = Vec::new();

        // Check if there are any fields
        if !self.matches(&TokenType::End) {
            loop {
                fields.push(self.parse_obj_field()?);

                if !self.matches(&TokenType::Comma) {
                    break;
                }
                self.advance(); // consume comma

                // Check for trailing comma
                if self.matches(&TokenType::End) {
                    break;
                }
            }
        }

        let end = self.expect(TokenType::End)?;

        Ok(Expr::SReactor {
            l: start.location.span(end.location),
            fields,
        })
    }

    /// check-expr: CHECK [name] COLON body END
    /// Parses a check block like:
    ///   check:
    ///     1 + 1 is 2
    ///   end
    /// or with a name:
    ///   check "my test":
    ///     1 + 1 is 2
    ///   end
    fn parse_check_expr(&mut self) -> ParseResult<Expr> {
        let start_token = self.peek().clone();

        // Check for three patterns:
        // 1. check: ... end (CheckColon token)
        // 2. check "string": ... end (Check String Colon tokens)
        // 3. examples: ... end (ExamplesColon token) - keyword-check is false for examples
        let (start, name, keyword_check) = if self.matches(&TokenType::CheckColon) {
            let tok = self.advance().clone();
            (tok, None, true)
        } else if self.matches(&TokenType::ExamplesColon) {
            let tok = self.advance().clone();
            (tok, None, false) // examples: has keyword-check = false
        } else if self.matches(&TokenType::Check) {
            self.advance(); // consume CHECK

            // Expect a string for the name
            let name_token = self.expect(TokenType::String)?;
            let name_str = name_token.value.clone();

            // Expect colon
            self.expect(TokenType::Colon)?;

            (start_token, Some(name_str), true)
        } else {
            return Err(ParseError::general(
                &start_token,
                "Expected 'check:', 'examples:', or 'check \"name\":'",
            ));
        };

        // Parse the body as a block of statements (can include let bindings and check tests)
        let mut stmts = Vec::new();

        while !self.matches(&TokenType::End) && !self.is_at_end() {
            // Parse as block statement to handle both let bindings and check tests
            let stmt = self.parse_block_statement()?;
            stmts.push(Box::new(stmt));
        }

        let end = self.expect(TokenType::End)?;

        // Create the block for the body
        let body_loc = start.location.span(end.location);
        let body = Box::new(Expr::SBlock {
            l: body_loc,
            stmts,
        });

        Ok(Expr::SCheck {
            l: start.location.span(end.location),
            name,
            body,
            keyword_check,
        })
    }

    /// check-test: expr is expr | expr raises expr | ...
    /// Parses check test expressions like:
    ///   1 + 1 is 2
    ///   foo() raises "error"
    ///
    /// Note: parse_expr() will create an SCheckTest when it sees a check operator in parse_binop_expr.
    /// spy-stmt: SPY [string] COLON spy-contents END
    /// spy-contents: spy-expr [COMMA spy-expr]* [COMMA]
    /// spy-expr: NAME COLON expr | expr (where expr is NAME creates implicit label)
    fn parse_spy_stmt(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Spy)?;

        // Check for optional message expression (not just string literal)
        // The message can be any expression like "label" or "iteration " + to-string(i)
        let message = if !self.matches(&TokenType::Colon) {
            // Parse expression until we hit the colon
            let msg_expr = self.parse_binop_expr()?;
            Some(Box::new(msg_expr))
        } else {
            None
        };

        self.expect(TokenType::Colon)?;

        // Parse spy contents (comma-separated list of spy-expr)
        let mut contents = Vec::new();

        if !self.matches(&TokenType::End) {
            contents = self.parse_comma_list(|p| p.parse_spy_expr())?;
        }

        let end = self.expect(TokenType::End)?;

        Ok(Expr::SSpyBlock {
            l: start.location.span(end.location),
            message,
            contents,
        })
    }

    /// Parse a single spy expression: either "name: expr" or just "expr"
    /// If just "expr" and expr is an identifier, use implicit label
    fn parse_spy_expr(&mut self) -> ParseResult<SpyField> {
        // Try to parse as "name: expr" pattern
        // We need to look ahead to see if there's a colon after the name
        if self.matches(&TokenType::Name) {
            let next = self.peek_ahead(1);

            if next.token_type == TokenType::Colon {
                // This is "name: expr" pattern
                let name_tok = self.advance();
                let name = name_tok.value.clone();
                let name_loc = name_tok.location;
                self.expect(TokenType::Colon)?;
                let value_expr = self.parse_binop_expr()?;
                let value_loc = *value_expr.get_loc();
                let value = Box::new(value_expr);

                // Build the location
                let l = name_loc.span(value_loc);

                return Ok(SpyField {
                    node_type: "s-spy-expr".to_string(),
                    l,
                    name: Some(name),
                    value,
                    implicit_label: false,
                });
            }
        }

        // Otherwise, parse as just an expression
        let expr = self.parse_binop_expr()?;

        // If the expression is an identifier, use its name as implicit label
        let (name, implicit_label) = if let Expr::SId { id, .. } = &expr {
            // id is a Name enum, extract the string from it
            let name_str = match id {
                Name::SName { s, .. } => Some(s.clone()),
                Name::SUnderscore { .. } => Some("_".to_string()),
                _ => None,
            };
            (name_str, true)
        } else {
            (None, false)
        };

        Ok(SpyField {
            node_type: "s-spy-expr".to_string(),
            l: *expr.get_loc(),
            name,
            value: Box::new(expr),
            implicit_label,
        })
    }

}

// ============================================================================
// SECTION 12: Helper Methods
// ============================================================================

impl Parser {
    /// Parse comma-separated list
    /// Handles trailing commas: [1, 2, 3,] is valid
    fn parse_comma_list<T, F>(&mut self, parser: F) -> ParseResult<Vec<T>>
    where
        F: Fn(&mut Self) -> ParseResult<T>,
    {
        let mut items = Vec::new();

        // Parse first item
        items.push(parser(self)?);

        // Parse remaining items
        while self.matches(&TokenType::Comma) {
            self.advance(); // consume comma

            // Check if this is a trailing comma (followed by closing delimiter)
            // Try to parse the next item, but if it fails, that's okay - it's just a trailing comma
            match parser(self) {
                Ok(item) => items.push(item),
                Err(_) => break, // Trailing comma - stop here
            }
        }

        Ok(items)
    }

    /// Parse comma-separated list without allowing trailing commas
    /// Used for function arguments where trailing commas are not allowed
    fn parse_comma_list_no_trailing<T, F>(&mut self, parser: F) -> ParseResult<Vec<T>>
    where
        F: Fn(&mut Self) -> ParseResult<T>,
    {
        let mut items = Vec::new();

        // Parse first item
        items.push(parser(self)?);

        // Parse remaining items
        while self.matches(&TokenType::Comma) {
            self.advance(); // consume comma
            // After a comma, we MUST have another item (no trailing commas)
            items.push(parser(self)?);
        }

        Ok(items)
    }

    /// Parse NAME token into Name AST node
    fn parse_name(&mut self) -> ParseResult<Name> {
        let token = self.expect(TokenType::Name)?;
        let loc = token.location;

        // Check if this is an underscore wildcard
        if token.value == "_" {
            Ok(Name::SUnderscore { l: loc })
        } else {
            Ok(self.token_to_name(&token))
        }
    }
}
