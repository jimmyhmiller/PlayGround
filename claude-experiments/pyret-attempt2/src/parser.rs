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
        // Filter out comments and block comments as they're not part of the AST
        let tokens: Vec<Token> = tokens
            .into_iter()
            .filter(|t| !matches!(t.token_type, TokenType::Comment | TokenType::BlockComment))
            .collect();

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
    fn parse_prelude(
        &mut self,
    ) -> ParseResult<(
        Option<Use>,
        Provide,
        ProvideTypes,
        Vec<ProvideBlock>,
        Vec<Import>,
    )> {
        let mut _use = None;

        let mut provides = Vec::new();
        let mut imports = Vec::new();
        let mut _provide = Provide::SProvideNone {
            l: self.current_loc(),
        };
        let mut provided_types = ProvideTypes::SProvideTypesNone {
            l: self.current_loc(),
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
        let start = self.peek().clone();
        let mut stmts = Vec::new();

        // Parse statements until EOF or until we can't parse any more
        while !self.is_at_end() {
            // Try to parse a statement
            match self.parse_block_statement() {
                Ok(stmt) => stmts.push(Box::new(stmt)),
                Err(_) => break, // Stop if we can't parse
            }
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
        let start = self.expect(TokenType::Use)?;
        let name = self.parse_name()?;
        let module = self.parse_import_source()?;

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        Ok(Use {
            node_type: "s-use".to_string(),
            l: self.make_loc(&start, &end),
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
        let start = self.peek().clone();

        match start.token_type {
            TokenType::Import => {
                self.advance(); // consume IMPORT

                // We need to peek ahead to determine which import syntax we have:
                // 1. import x, y from source (comma-separated names)
                // 2. import source as name (regular import)

                // Try to parse the first name
                let first_name = self.parse_name()?;

                // Check what comes after the first name
                if self.matches(&TokenType::Comma) {
                    // import x, y, z from source
                    let mut fields = vec![first_name];

                    while self.matches(&TokenType::Comma) {
                        self.advance(); // consume comma
                        fields.push(self.parse_name()?);
                    }

                    self.expect(TokenType::From)?;
                    let module = self.parse_import_source()?;

                    let end = if self.current > 0 {
                        self.tokens[self.current - 1].clone()
                    } else {
                        start.clone()
                    };

                    Ok(Import::SImportFields {
                        l: self.make_loc(&start, &end),
                        fields,
                        import: module,
                    })
                } else if self.matches(&TokenType::From) {
                    // import x from source (single name)
                    self.advance(); // consume FROM
                    let module = self.parse_import_source()?;

                    let end = if self.current > 0 {
                        self.tokens[self.current - 1].clone()
                    } else {
                        start.clone()
                    };

                    Ok(Import::SImportFields {
                        l: self.make_loc(&start, &end),
                        fields: vec![first_name],
                        import: module,
                    })
                } else {
                    // import source as name (regular import)
                    // The first_name we parsed is actually the import source
                    // Extract location and string from Name enum
                    let (name_loc, name_str) = match &first_name {
                        Name::SName { l, s } => (l.clone(), s.clone()),
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
                        let mut args = Vec::new();

                        // Parse first string argument
                        let first_arg = self.expect(TokenType::String)?;
                        args.push(first_arg.value);

                        // Parse optional additional string arguments
                        while self.matches(&TokenType::Comma) {
                            self.advance(); // consume ,
                            let arg = self.expect(TokenType::String)?;
                            args.push(arg.value);
                        }

                        let end_paren = self.expect(TokenType::RParen)?;

                        ImportType::SSpecialImport {
                            l: self.make_loc(&start, &end_paren),
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

                    let end = if self.current > 0 {
                        self.tokens[self.current - 1].clone()
                    } else {
                        start.clone()
                    };

                    Ok(Import::SImport {
                        l: self.make_loc(&start, &end),
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
                    let module = self.parse_import_source()?;

                    self.expect(TokenType::Colon)?;

                    // Parse include specs (comma-separated names ending with END)
                    let mut names = Vec::new();

                    if !self.matches(&TokenType::End) {
                        // Parse first include spec
                        names.push(self.parse_include_spec()?);

                        // Parse additional specs with comma separators
                        while self.matches(&TokenType::Comma) {
                            self.advance(); // consume comma

                            // Allow optional trailing comma
                            if self.matches(&TokenType::End) {
                                break;
                            }

                            names.push(self.parse_include_spec()?);
                        }
                    }

                    let end = self.expect(TokenType::End)?;

                    Ok(Import::SIncludeFrom {
                        l: self.make_loc(&start, &end),
                        import: module,
                        names,
                    })
                } else {
                    // Simple include: include module
                    let module = self.parse_import_source()?;

                    let end = if self.current > 0 {
                        self.tokens[self.current - 1].clone()
                    } else {
                        start.clone()
                    };

                    Ok(Import::SInclude {
                        l: self.make_loc(&start, &end),
                        import: module,
                    })
                }
            }
            _ => Err(ParseError::general(
                &start,
                "Expected 'import' or 'include'",
            )),
        }
    }

    /// Parse import source (module name)
    /// Handles:
    /// - Simple name: `module-name`
    /// - Special import: `file("path.arr")` or `file("path.arr", "other")`
    fn parse_import_source(&mut self) -> ParseResult<ImportType> {
        let start = self.peek().clone();
        let module_name = self.expect(TokenType::Name)?;

        // Check for special import: NAME PARENNOSPACE STRING (COMMA STRING)* RPAREN
        if self.matches(&TokenType::ParenNoSpace) {
            self.advance(); // consume (

            let kind = module_name.value;
            let mut args = Vec::new();

            // Parse first string argument
            let first_arg = self.expect(TokenType::String)?;
            args.push(first_arg.value);

            // Parse optional additional string arguments
            while self.matches(&TokenType::Comma) {
                self.advance(); // consume ,
                let arg = self.expect(TokenType::String)?;
                args.push(arg.value);
            }

            let end = self.expect(TokenType::RParen)?;

            Ok(ImportType::SSpecialImport {
                l: self.make_loc(&start, &end),
                kind,
                args,
            })
        } else {
            // Simple const import
            Ok(ImportType::SConstImport {
                l: self.make_loc(&start, &module_name),
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
                l: self.make_loc(&start, &start),
            })
        } else if self.matches(&TokenType::From) {
            // provide from module: specs end
            // This is actually a provide-block, not a provide-stmt
            // We need to parse it here anyway since we already consumed PROVIDE
            self.advance(); // consume FROM

            // Parse module-ref (just a name for now)
            let module_name = self.parse_name()?;

            self.expect(TokenType::Colon)?;

            // Parse provide-specs
            let mut specs = Vec::new();

            if !self.matches(&TokenType::End) {
                specs.push(self.parse_provide_spec()?);

                while self.matches(&TokenType::Comma) {
                    self.advance(); // consume comma

                    if self.matches(&TokenType::End) {
                        break;
                    }

                    specs.push(self.parse_provide_spec()?);
                }
            }

            let end = self.expect(TokenType::End)?;

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
                l: self.current_loc(),
                stmts: block_stmts,
            });

            Ok(Provide::SProvide {
                l: self.make_loc(&start, &end),
                block,
            })
        } else {
            // provide stmt end (e.g., provide { x: 10 } end)
            let stmt = self.parse_block_statement()?;
            let end = self.expect(TokenType::End)?;

            // The provide block is the statement itself, not wrapped in s-block
            Ok(Provide::SProvide {
                l: self.make_loc(&start, &end),
                block: Box::new(stmt),
            })
        }
    }

    /// provide-types-stmt: PROVIDE-TYPES record-ann | PROVIDE-TYPES (STAR|TIMES)
    /// Parses provide-types statements like:
    /// - provide-types *
    /// - provide-types { Type1, Type2 } (TODO: record-ann not yet implemented)
    fn parse_provide_types_stmt(&mut self) -> ParseResult<ProvideTypes> {
        let start = self.expect(TokenType::ProvideTypes)?;

        // Check for provide-types-all: provide-types *
        if self.matches(&TokenType::Times) || self.matches(&TokenType::Star) {
            self.advance();

            Ok(ProvideTypes::SProvideTypesAll {
                l: self.make_loc(&start, &start),
            })
        } else {
            // TODO: Implement record-ann for provide-types { Type1, Type2 }
            // For now, just return an error
            Err(ParseError::general(
                self.peek(),
                "provide-types with specific types not yet implemented",
            ))
        }
    }

    /// provide-block: PROVIDECOLON [provide-spec (COMMA provide-spec)* [COMMA]] END
    /// Parses provide-block statements like:
    /// - provide: add, multiply end
    /// - provide: * end (provide all)
    fn parse_provide_block(&mut self) -> ParseResult<ProvideBlock> {
        let start = self.expect(TokenType::ProvideColon)?;

        // Parse provide-specs
        let mut specs = Vec::new();

        // Parse specs until we hit END
        if !self.matches(&TokenType::End) {
            // Parse first spec
            specs.push(self.parse_provide_spec()?);

            // Parse additional specs with comma separators
            while self.matches(&TokenType::Comma) {
                self.advance(); // consume comma

                // Allow optional trailing comma
                if self.matches(&TokenType::End) {
                    break;
                }

                specs.push(self.parse_provide_spec()?);
            }
        }

        let end = self.expect(TokenType::End)?;

        Ok(ProvideBlock {
            node_type: "s-provide-block".to_string(),
            l: self.make_loc(&start, &end),
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

        // Parse module-ref (just a name for now)
        let module_name = self.parse_name()?;

        self.expect(TokenType::Colon)?;

        // Parse provide-specs
        let mut specs = Vec::new();

        if !self.matches(&TokenType::End) {
            specs.push(self.parse_provide_spec()?);

            while self.matches(&TokenType::Comma) {
                self.advance(); // consume comma

                if self.matches(&TokenType::End) {
                    break;
                }

                specs.push(self.parse_provide_spec()?);
            }
        }

        let end = self.expect(TokenType::End)?;

        Ok(ProvideBlock {
            node_type: "s-provide-block".to_string(),
            l: self.make_loc(&start, &end),
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
                l: self.current_loc(),
                name: name_spec,
            });
        }

        if self.matches(&TokenType::Data) {
            self.advance(); // consume DATA
            let name_spec = self.parse_name_spec()?;
            return Ok(ProvideSpec::SProvideData {
                l: self.current_loc(),
                name: name_spec,
            });
        }

        if self.matches(&TokenType::Module) {
            self.advance(); // consume MODULE
            let name_spec = self.parse_name_spec()?;
            return Ok(ProvideSpec::SProvideModule {
                l: self.current_loc(),
                name: name_spec,
            });
        }

        // Otherwise, parse as provide-name-spec
        let name_spec = self.parse_name_spec()?;

        Ok(ProvideSpec::SProvideName {
            l: self.current_loc(),
            name: name_spec,
        })
    }

    /// Parse a name-spec
    /// name-spec: (STAR|TIMES) [hiding-spec] | module-ref | module-ref AS NAME
    fn parse_name_spec(&mut self) -> ParseResult<NameSpec> {
        // Check for * (star)
        if self.matches(&TokenType::Times) || self.matches(&TokenType::Star) {
            let start = self.peek().clone();
            self.advance();
            // TODO: Parse optional hiding-spec
            return Ok(NameSpec::SStar {
                l: self.make_loc(&start, &start),
                hidden: Vec::new(),
            });
        }

        // Parse module-ref (simple name for now, could be dotted path)
        let name = self.parse_name()?;

        // Check for AS NAME
        let as_name = if self.matches(&TokenType::As) {
            self.advance(); // consume AS
            Some(self.parse_name()?)
        } else {
            None
        };

        // The JSON format shows: "name-spec": {"type": "s-module-ref", "path": [{"type": "s-name", "name": "add"}], "as-name": null}
        Ok(NameSpec::SModuleRef {
            l: self.current_loc(),
            path: vec![name],
            as_name,
        })
    }

    /// Parse an include-spec
    /// include-spec: include-name-spec | include-type-spec | include-data-spec | include-module-spec
    /// For now, just handles include-name-spec (simple names)
    fn parse_include_spec(&mut self) -> ParseResult<IncludeSpec> {
        let start = self.peek().clone();

        // Check for TYPE, DATA, or MODULE keywords
        if self.matches(&TokenType::Type) {
            // include-type-spec: TYPE name-spec
            self.advance(); // consume TYPE
            let name = self.parse_name_spec()?;
            let end = if self.current > 0 {
                self.tokens[self.current - 1].clone()
            } else {
                start.clone()
            };

            Ok(IncludeSpec::SIncludeType {
                l: self.make_loc(&start, &end),
                name,
            })
        } else if self.matches(&TokenType::Data) {
            // include-data-spec: DATA data-name-spec [hiding-spec]
            self.advance(); // consume DATA
            let name = self.parse_name_spec()?;

            // TODO: Parse optional hiding-spec when needed
            // For now, just handle the name

            let end = if self.current > 0 {
                self.tokens[self.current - 1].clone()
            } else {
                start.clone()
            };

            Ok(IncludeSpec::SIncludeData {
                l: self.make_loc(&start, &end),
                name,
            })
        } else if self.matches(&TokenType::Module) {
            // include-module-spec: MODULE name-spec
            self.advance(); // consume MODULE
            let name = self.parse_name_spec()?;
            let end = if self.current > 0 {
                self.tokens[self.current - 1].clone()
            } else {
                start.clone()
            };

            Ok(IncludeSpec::SIncludeModule {
                l: self.make_loc(&start, &end),
                name,
            })
        } else {
            // include-name-spec: name-spec
            let name_spec = self.parse_name_spec()?;

            Ok(IncludeSpec::SIncludeName {
                l: self.current_loc(),
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

            // Empty braces - just return empty tuple
            if self.matches(&TokenType::RBrace) {
                let end = self.advance().clone();
                return Ok(Ann::ATuple {
                    l: self.make_loc(&start, &end),
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
                    let field_loc = self.make_loc(&field_name_token, &field_name_token);
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
                    l: self.make_loc(&start, &end),
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
                    l: self.make_loc(&start, &end),
                    fields: tuple_fields,
                });
            }
        }

        // Handle parenthesized arrow types: (A -> B), (A, B -> C), etc.
        if self.matches(&TokenType::LParen) || self.matches(&TokenType::ParenSpace) {
            let start = self.advance().clone(); // consume (

            // Parse argument types (comma-separated)
            let mut args = Vec::new();
            if !self.matches(&TokenType::RParen) {
                args.push(self.parse_ann()?);

                // Check for arrow or comma
                while self.matches(&TokenType::Comma) {
                    self.advance(); // consume comma
                    args.push(self.parse_ann()?);
                }
            }

            // Expect arrow
            if self.matches(&TokenType::ThinArrow) {
                self.advance(); // consume ->

                // Parse return type
                let ret = Box::new(self.parse_ann()?);

                let end = self.expect(TokenType::RParen)?;
                let loc = self.make_loc(&start, &end);

                return Ok(Ann::AArrow {
                    l: loc,
                    args,
                    ret,
                    use_parens: true,
                });
            } else {
                // Just a parenthesized annotation, not an arrow
                let end = self.expect(TokenType::RParen)?;
                // Return the single annotation (unwrap the parens)
                return Ok(args.into_iter().next().unwrap_or(Ann::ABlank));
            }
        }

        if self.matches(&TokenType::Name) {
            let name_token = self.advance().clone();
            let mut loc = self.make_loc(&name_token, &name_token);

            // Check if it's the special "Any" type (before processing dots)
            if name_token.value == "Any" && !self.matches(&TokenType::Dot) {
                return Ok(Ann::AAny { l: loc });
            }

            let name = Name::SName {
                l: loc.clone(),
                s: name_token.value.clone(),
            };

            // Handle dotted names: E.Either, Module.Type, etc.
            // Build up nested ADot nodes for chains like A.B.C
            let mut base_ann: Ann = Ann::AName {
                l: loc.clone(),
                id: name,
            };

            while self.matches(&TokenType::Dot) {
                self.advance(); // consume dot
                let field_token = self.expect(TokenType::Name)?;
                loc = self.make_loc(&name_token, &field_token);

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
                                l: l.clone(),
                                s: format!("{}.{}", s, field),
                            },
                            other => other.clone(),
                        }
                    }
                    _ => return Err(ParseError::general(&field_token, "Invalid dotted type annotation")),
                };

                base_ann = Ann::ADot {
                    l: loc.clone(),
                    obj: obj_name,
                    field: field_token.value.clone(),
                };
            }

            // Check for type application: List<T>, Map<K, V>, etc.
            let mut result_ann = if self.matches(&TokenType::Lt) {
                self.advance(); // consume '<'
                let type_args = self.parse_comma_list(|p| p.parse_ann())?;
                let end_token = self.expect(TokenType::Gt)?; // consume '>'
                let app_loc = self.make_loc(&name_token, &end_token);
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
                // Expect opening paren (can be LParen, ParenSpace, or ParenNoSpace)
                let paren_token = self.peek().clone();
                match paren_token.token_type {
                    TokenType::LParen | TokenType::ParenSpace | TokenType::ParenNoSpace => {
                        self.advance();
                    }
                    _ => {
                        return Err(ParseError::expected(TokenType::LParen, paren_token));
                    }
                }
                let predicate = self.parse_expr()?;
                let end_token = self.expect(TokenType::RParen)?; // consume ')'
                let pred_loc = self.make_loc(&name_token, &end_token);
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

        // Create Name node - use SUnderscore for "_"
        let name = if name_str == "_" {
            Name::SUnderscore {
                l: Loc::new(
                    self.file_name.clone(),
                    name_token.location.start_line,
                    name_token.location.start_col,
                    name_token.location.start_pos,
                    name_token.location.end_line,
                    name_token.location.end_col,
                    name_token.location.end_pos,
                ),
            }
        } else {
            Name::SName {
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
            }
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
        let start = self.expect(TokenType::LBrace)?;

        // Parse fields: name1; name2; name3
        let mut fields = Vec::new();

        // Parse first field
        let first_name = self.parse_name()?;
        fields.push(Bind::SBind {
            l: self.current_loc(),
            shadows: false,
            id: first_name,
            ann: Ann::ABlank,
        });

        // Parse remaining fields
        while self.matches(&TokenType::Semi) {
            self.advance(); // consume semicolon
            let name = self.parse_name()?;
            fields.push(Bind::SBind {
                l: self.current_loc(),
                shadows: false,
                id: name,
                ann: Ann::ABlank,
            });
        }

        let end = self.expect(TokenType::RBrace)?;

        Ok(Bind::STupleBind {
            l: self.make_loc(&start, &end),
            fields,
            as_name: None,
        })
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
                            Expr::SExtend { l, .. } => l.clone(),
                            Expr::SUpdate { l, .. } => l.clone(),
                            _ => self.current_loc(),
                        };

                        // In Pyret, both extension and update use the same syntax: obj.{fields}
                        // The semantic difference is: extension adds NEW fields, update MODIFIES existing fields
                        // We use SExtend as the default; the type checker determines the actual semantics
                        left = Expr::SExtend {
                            l: Loc::new(
                                self.file_name.clone(),
                                start_loc.start_line,
                                start_loc.start_column,
                                start_loc.start_char,
                                end.location.end_line,
                                end.location.end_col,
                                end.location.end_pos,
                            ),
                            supe: Box::new(left),
                            fields,
                        };
                    }
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
            // The right side is a full binary expression (can include +, *, etc.)
            // but check operators have lower precedence, so we DON'T recursively parse more check ops
            let right = if !self.is_at_end() && !matches!(self.peek().token_type, TokenType::Eof) && !self.is_check_op() {
                // Parse a binary expression (which includes primitives, postfix ops, and binary ops)
                // But don't parse another check operator (they're at the same/lower precedence level)
                let right_expr = self.parse_binop_expr_no_check()?;
                Some(Box::new(right_expr))
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

            // Get end location, considering refinement and right expression
            // Priority: right > refinement > operator
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
            } else if let Some(ref refinement_expr) = refinement {
                // If no right expression but has refinement, use refinement location
                match refinement_expr.as_ref() {
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
                // If no right expression and no refinement, use operator location
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

            // Check for optional 'because' clause
            let cause = if self.matches(&TokenType::Because) {
                self.advance(); // consume 'because'
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };

            // Update end location if there's a cause
            let final_end_loc = if let Some(ref cause_expr) = cause {
                match cause_expr.as_ref() {
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
                end_loc
            };

            let loc = Loc::new(
                self.file_name.clone(),
                start_loc.start_line,
                start_loc.start_column,
                start_loc.start_char,
                final_end_loc.end_line,
                final_end_loc.end_column,
                final_end_loc.end_char,
            );

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
            } else if self.matches(&TokenType::Dot) {
                let _dot_token = self.expect(TokenType::Dot)?;
                if self.matches(&TokenType::LBrace) {
                    // Tuple access
                    self.expect(TokenType::LBrace)?;
                    let index_token = self.expect(TokenType::Number)?;
                    let index: usize = index_token.value.parse()
                        .map_err(|_| ParseError::invalid("tuple index", &index_token, "Invalid number"))?;
                    let index_loc = self.make_loc(&index_token, &index_token);
                    let end = self.expect(TokenType::RBrace)?;
                    let start_loc = match &left {
                        Expr::SNum { l, .. } => l.clone(),
                        Expr::SBool { l, .. } => l.clone(),
                        Expr::SId { l, .. } => l.clone(),
                        Expr::SOp { l, .. } => l.clone(),
                        Expr::SParen { l, .. } => l.clone(),
                        Expr::SApp { l, .. } => l.clone(),
                        Expr::SDot { l, .. } => l.clone(),
                        Expr::SBracket { l, .. } => l.clone(),
                        Expr::STupleGet { l, .. } => l.clone(),
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
                    // Regular dot access
                    let field_token = self.parse_field_name()?;
                    let start_loc = match &left {
                        Expr::SNum { l, .. } => l.clone(),
                        Expr::SBool { l, .. } => l.clone(),
                        Expr::SId { l, .. } => l.clone(),
                        Expr::SOp { l, .. } => l.clone(),
                        Expr::SParen { l, .. } => l.clone(),
                        Expr::SApp { l, .. } => l.clone(),
                        Expr::SDot { l, .. } => l.clone(),
                        Expr::SBracket { l, .. } => l.clone(),
                        Expr::STupleGet { l, .. } => l.clone(),
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
                left = self.parse_bracket_expr(left)?;
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
                } else if self.matches(&TokenType::Dot) {
                    let _dot_token = self.expect(TokenType::Dot)?;
                    let field_token = self.parse_field_name()?;
                    let start_loc = match &right {
                        Expr::SNum { l, .. } => l.clone(),
                        Expr::SBool { l, .. } => l.clone(),
                        Expr::SId { l, .. } => l.clone(),
                        Expr::SOp { l, .. } => l.clone(),
                        Expr::SParen { l, .. } => l.clone(),
                        Expr::SApp { l, .. } => l.clone(),
                        Expr::SDot { l, .. } => l.clone(),
                        Expr::SBracket { l, .. } => l.clone(),
                        Expr::STupleGet { l, .. } => l.clone(),
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
                    right = self.parse_bracket_expr(right)?;
                } else {
                    break;
                }
            }

            let op_l = Loc::new(
                self.file_name.clone(),
                op_token.location.start_line,
                op_token.location.start_col,
                op_token.location.start_pos,
                op_token.location.end_line,
                op_token.location.end_col,
                op_token.location.end_pos,
            );
            let start_loc = match &left {
                Expr::SNum { l, .. } => l.clone(),
                Expr::SBool { l, .. } => l.clone(),
                Expr::SId { l, .. } => l.clone(),
                Expr::SOp { l, .. } => l.clone(),
                Expr::SParen { l, .. } => l.clone(),
                Expr::SApp { l, .. } => l.clone(),
                Expr::SDot { l, .. } => l.clone(),
                Expr::SBracket { l, .. } => l.clone(),
                _ => self.current_loc(),
            };
            let end_loc = match &right {
                Expr::SNum { l, .. } => l.clone(),
                Expr::SBool { l, .. } => l.clone(),
                Expr::SId { l, .. } => l.clone(),
                Expr::SOp { l, .. } => l.clone(),
                Expr::SParen { l, .. } => l.clone(),
                Expr::SApp { l, .. } => l.clone(),
                Expr::SDot { l, .. } => l.clone(),
                Expr::SBracket { l, .. } => l.clone(),
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
            TokenType::ParenSpace | TokenType::LParen => self.parse_paren_expr(),
            TokenType::LBrack => self.parse_construct_expr(),
            TokenType::LBrace => self.parse_obj_expr(),
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
            TokenType::Rec => self.parse_rec_expr(),
            TokenType::Letrec => self.parse_letrec_expr(),
            TokenType::Var => self.parse_var_expr(),
            TokenType::CheckColon => self.parse_check_expr(),
            TokenType::Check => self.parse_check_expr(),
            TokenType::ExamplesColon => self.parse_check_expr(),
            TokenType::Spy => self.parse_spy_stmt(),
            TokenType::Method => self.parse_method_expr(),
            TokenType::Table => self.parse_table_expr(),
            TokenType::LoadTable => self.parse_load_table_expr(),
            TokenType::Reactor => self.parse_reactor_expr(),
            _ => Err(ParseError::unexpected(token)),
        }
    }

    /// num-expr: NUMBER
    /// Pyret represents all numbers as rationals, so:
    /// - Integers like "42" -> SNum with n=42.0
    /// - Decimals like "3.14" -> SNum with n=3.14 (NOT SFrac - decimals are stored as floats)
    /// Note: The official Pyret parser converts decimals to fractions internally but
    /// represents them as s-num in JSON with fraction string values like "157/50"
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

        // Parse as float (integers and decimals)
        let n: f64 = token
            .value
            .parse()
            .map_err(|_| ParseError::invalid("number", &token, "Invalid number format"))?;

        // Store original string to preserve precision for large integers
        Ok(Expr::SNum { l: loc, n, original: Some(token.value.clone()) })
    }

    /// rough-num-expr: ROUGHNUMBER
    /// Parses rough (approximate) numbers like ~0.8 or ~42
    /// Represented as SNum with the tilde preserved in the original string
    fn parse_rough_num(&mut self) -> ParseResult<Expr> {
        let token = self.expect(TokenType::RoughNumber)?;
        let loc = Loc::new(
            self.file_name.clone(),
            token.location.start_line,
            token.location.start_col,
            token.location.start_pos,
            token.location.end_line,
            token.location.end_col,
            token.location.end_pos,
        );

        // Parse the number part (skip the ~)
        let num_str = token.value.trim_start_matches('~');
        let n: f64 = num_str
            .parse()
            .map_err(|_| ParseError::invalid("rough number", &token, "Invalid number format"))?;

        // Store original string INCLUDING the tilde
        Ok(Expr::SNum { l: loc, n, original: Some(token.value.clone()) })
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

        // Note: We don't simplify fractions to match official Pyret parser behavior
        // The official parser keeps fractions in their original form (e.g., 8/10 not 4/5)
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

        // Note: We don't simplify fractions to match official Pyret parser behavior
        // The official parser keeps fractions in their original form (e.g., ~8/10 not ~4/5)
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
                | TokenType::IsSpaceship
                | TokenType::IsEqualEqual
                | TokenType::IsEqualTilde
                | TokenType::IsNotSpaceship
                | TokenType::IsNotEqualEqual
                | TokenType::IsNotEqualTilde
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
        let _start = self.expect(TokenType::LBrack)?;

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

    /// Parse a member (same as obj-field, but used in data definitions)
    fn parse_member(&mut self) -> ParseResult<Member> {
        self.parse_obj_field()
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
            let stmt = self.parse_block_statement()?;
            body_stmts.push(Box::new(stmt));
        }

        // Parse optional where clause
        let (check, check_loc) = if self.matches(&TokenType::Where) {
            let where_token = self.advance().clone();
            // The check-loc should just point to the WHERE keyword itself
            let check_loc = self.make_loc(&where_token, &where_token);
            let mut where_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                where_stmts.push(Box::new(stmt));
            }
            let check_block = Box::new(Expr::SBlock {
                l: check_loc.clone(),
                stmts: where_stmts,
            });
            (Some(check_block), Some(check_loc))
        } else {
            (None, None)
        };

        let end = self.expect(TokenType::End)?;

        // Wrap body in SBlock
        let body = Box::new(Expr::SBlock {
            l: self.current_loc(),
            stmts: body_stmts,
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
            if let Ok(_) = self.parse_tuple_for_destructure() {
                if self.matches(&TokenType::Equals) {
                    // Yes! This is tuple destructuring
                    self.restore(checkpoint);
                    return self.parse_tuple_destructure_expr();
                }
            }

            // Not tuple destructuring, restore and parse as expression
            self.restore(checkpoint);
            self.parse_expr()
        } else if self.matches(&TokenType::Name) {
            // Check if this is an implicit let binding: x = value or x :: Type = value
            // Look ahead to see if there's a :: or = or := after the name
            let checkpoint = self.checkpoint();
            let _name = self.advance(); // Consume the name

            // Check for type annotation first
            if self.matches(&TokenType::ColonColon) {
                // Has type annotation - must be a let or var binding
                self.restore(checkpoint);
                self.parse_implicit_let_expr()
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
                let stmt = self.parse_block_statement()?;
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
                let stmt = self.parse_block_statement()?;
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
                    l: self.current_loc(),
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
                    l: self.current_loc(),
                    stmts: body_stmts,
                };

                branches.push(IfPipeBranch {
                    node_type: "s-if-pipe-branch".to_string(),
                    l: self.current_loc(),
                    test: Box::new(test),
                    body: Box::new(body),
                });
            }
        }

        let end = self.expect(TokenType::End)?;
        let loc = self.make_loc(&start, &end);

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
            let stmt = self.parse_block_statement()?;
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

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        // Var bindings are s-var statements
        Ok(Expr::SVar {
            l: self.make_loc(&start, &end),
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

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        // Rec bindings are s-rec statements
        Ok(Expr::SRec {
            l: self.make_loc(&start, &end),
            name: bind,
            value: Box::new(value),
        })
    }

    /// Helper function to parse a tuple pattern for destructuring (lookahead only)
    /// Returns Ok(()) if it successfully parses a tuple pattern, Err otherwise
    fn parse_tuple_for_destructure(&mut self) -> ParseResult<()> {
        self.expect(TokenType::LBrace)?;

        // Parse at least one field
        self.parse_name()?;

        // Parse remaining fields
        while self.matches(&TokenType::Semi) {
            self.advance();
            self.parse_name()?;
        }

        self.expect(TokenType::RBrace)?;
        Ok(())
    }

    /// Parses tuple destructuring: {a; b; c} = {1; 2; 3}
    fn parse_tuple_destructure_expr(&mut self) -> ParseResult<Expr> {
        let start = self.peek().clone();

        // Parse the tuple bind pattern: {a; b; c}
        let tuple_bind = self.parse_tuple_bind()?;

        // Expect =
        self.expect(TokenType::Equals)?;

        // Parse the value expression
        let value = self.parse_expr()?;

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        // Create an s-let with tuple bind
        Ok(Expr::SLet {
            l: self.make_loc(&start, &end),
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
            l: self.current_loc(),
            b: bind,
            value: Box::new(value),
        });

        // Parse additional bindings if present (COMMA let-expr)*
        while self.matches(&TokenType::Comma) {
            self.advance();

            let bind = self.parse_bind()?;
            self.expect(TokenType::Equals)?;
            let value = self.parse_expr()?;

            binds.push(LetrecBind {
                node_type: "s-letrec-bind".to_string(),
                l: self.current_loc(),
                b: bind,
                value: Box::new(value),
            });
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
            l: self.current_loc(),
            stmts: body_stmts,
        };

        Ok(Expr::SLetrec {
            l: self.make_loc(&start, &end),
            binds,
            body: Box::new(body),
            blocky,
        })
    }

    /// Implicit let binding: x = value (no "let" keyword)
    /// Creates an s-let statement (not s-let-expr)
    /// Used in block contexts where we want statement-style bindings
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

    /// Standalone let binding: x = value (no "let" keyword)
    /// Creates an s-let-expr (expression form that returns a value)
    /// Used for standalone expression parsing
    fn parse_standalone_let_expr(&mut self) -> ParseResult<Expr> {
        let start = self.peek().clone();

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

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        // Return SLetExpr with the value as the body
        Ok(Expr::SLetExpr {
            l: self.make_loc(&start, &end),
            binds: vec![let_bind],
            body: Box::new(value),
            blocky: false,
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
    /// Parses cases expressions like: cases(Either) e: | left(v) => v | right(v) => v end
    fn parse_cases_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Cases)?;

        // Expect opening paren (ParenSpace because cases sets paren_is_for_exp)
        if !self.matches(&TokenType::LParen) && !self.matches(&TokenType::ParenSpace) {
            return Err(ParseError::expected(TokenType::LParen, self.peek().clone()));
        }
        self.advance(); // consume the paren

        // Parse type annotation
        let typ = self.parse_ann()?;

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Parse value expression
        let val = self.parse_expr()?;

        // Expect colon or block
        let blocky = if self.matches(&TokenType::Block) {
            self.advance();
            true
        } else {
            self.expect(TokenType::Colon)?;
            false
        };

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
                    l: self.current_loc(),
                    stmts: else_stmts,
                }));
                break;
            }

            // Parse branch pattern name
            let pattern_start = self.peek().clone();
            let name_token = self.expect(TokenType::Name)?;
            let name = name_token.value.clone();
            let pattern_loc = self.make_loc(&pattern_start, &name_token);

            // Check for arguments: name(args)
            let args = if self.matches(&TokenType::LParen) || self.matches(&TokenType::ParenNoSpace) {
                self.advance(); // consume (

                let args = if self.matches(&TokenType::RParen) {
                    Vec::new()
                } else {
                    // Parse comma-separated bindings
                    self.parse_comma_list(|p| {
                        let bind = p.parse_bind()?;
                        let l = match &bind {
                            Bind::SBind { l, .. } => l.clone(),
                            Bind::STupleBind { l, .. } => l.clone(),
                        };
                        Ok(CasesBind {
                            node_type: "s-cases-bind".to_string(),
                            l,
                            field_type: CasesBindType::SNormal,
                            bind,
                        })
                    })?
                };

                self.expect(TokenType::RParen)?;
                args
            } else {
                Vec::new()
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
                l: self.current_loc(),
                stmts: body_stmts,
            });

            // Create the appropriate branch type
            let branch = if args.is_empty() {
                CasesBranch::SSingletonCasesBranch {
                    l: self.current_loc(),
                    pattern_loc,
                    name,
                    body,
                }
            } else {
                CasesBranch::SCasesBranch {
                    l: self.current_loc(),
                    pattern_loc,
                    name,
                    args,
                    body,
                }
            };

            branches.push(branch);
        }

        let end = self.expect(TokenType::End)?;
        let loc = self.make_loc(&start, &end);

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
        let blocky = if self.matches(&TokenType::Block) {
            self.advance();
            true
        } else {
            self.expect(TokenType::Colon)?;
            false
        };

        // Parse the block (statements until end)
        let mut block_stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_block_statement()?;
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
            blocky,
        })
    }
}

// ============================================================================
// SECTION 8: Function Parsing
// ============================================================================

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
        let params: Vec<Name> = if self.matches(&TokenType::Lt) {
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

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        Ok(Expr::SType {
            l: self.make_loc(&start, &end),
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

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        Ok(Expr::SNewtype {
            l: self.make_loc(&start, &end),
            name,
            namet,
        })
    }

    fn parse_fun_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Fun)?;

        // Parse function name
        let name_token = self.expect(TokenType::Name)?;
        let name = name_token.value.clone();

        // Parse optional type parameters <T, U, V>
        let params: Vec<Name> = if self.matches(&TokenType::Lt) {
            self.advance(); // consume '<'
            let type_params = self.parse_comma_list(|p| p.parse_name())?;
            self.expect(TokenType::Gt)?; // consume '>'
            type_params
        } else {
            Vec::new()
        };

        // Expect opening paren (can be LParen, ParenSpace, or ParenNoSpace)
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

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Optional return type annotation: -> ann
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

        // Parse doc string (skip for now, typically empty)
        let doc = String::new();

        // Parse function body (statements until END or WHERE)
        let mut body_stmts = Vec::new();
        while !self.matches(&TokenType::End)
            && !self.matches(&TokenType::Where)
            && !self.is_at_end()
        {
            let stmt = self.parse_expr()?;
            body_stmts.push(Box::new(stmt));
        }

        // Parse optional where clause
        let (check, check_loc) = if self.matches(&TokenType::Where) {
            let where_token = self.advance().clone();
            // The check-loc should just point to the WHERE keyword itself
            let check_loc = self.make_loc(&where_token, &where_token);
            let mut where_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                where_stmts.push(Box::new(stmt));
            }
            let check_block = Box::new(Expr::SBlock {
                l: check_loc.clone(),
                stmts: where_stmts,
            });
            (Some(check_block), Some(check_loc))
        } else {
            (None, None)
        };

        let end = self.expect(TokenType::End)?;

        // Wrap body in SBlock
        let body = Box::new(Expr::SBlock {
            l: self.current_loc(),
            stmts: body_stmts,
        });

        Ok(Expr::SFun {
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

        // Expect colon or block before body
        let blocky = if self.matches(&TokenType::Block) {
            self.advance();
            true
        } else {
            self.expect(TokenType::Colon)?;
            false
        };

        // Parse doc string (optional, we'll skip for now)
        // TODO: Parse doc strings

        // Parse body as a block of statements
        let mut stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.matches(&TokenType::Where) && !self.is_at_end() {
            let stmt = self.parse_block_statement()?;
            stmts.push(Box::new(stmt));
        }

        // Parse optional where clause
        // TODO: Handle where clauses in lambdas

        // Expect end keyword
        let end = self.expect(TokenType::End)?;

        // Body is the block of statements
        let body = Expr::SBlock {
            l: self.current_loc(),
            stmts,
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
            blocky,
        })
    }

    /// method-expr: METHOD(args) ann: doc body where END
    fn parse_method_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Method)?;

        // Expect opening paren (can be LParen, ParenSpace, or ParenNoSpace)
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
        // First should be `self`
        let args = if self.matches(&TokenType::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list(|p| p.parse_bind())?
        };

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Optional return type annotation: -> ann
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

        // Parse doc string (skip for now, typically empty)
        let doc = String::new();

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
            let check_loc = self.make_loc(&where_token, &where_token);
            let mut where_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                where_stmts.push(Box::new(stmt));
            }
            let check_block = Box::new(Expr::SBlock {
                l: check_loc.clone(),
                stmts: where_stmts,
            });
            (Some(check_block), Some(check_loc))
        } else {
            (None, None)
        };

        // Expect closing END
        let end = self.expect(TokenType::End)?;

        // Create location spanning entire method expression
        let loc = self.make_loc(&start, &end);

        // Create block body
        let body = Box::new(Expr::SBlock {
            l: loc.clone(),
            stmts: body_stmts,
        });

        Ok(Expr::SMethod {
            l: loc,
            name: String::new(), // Methods have no name
            params: Vec::new(),  // Methods have no type parameters
            args,
            ann,
            doc,
            body,
            check_loc,
            check,
            blocky,
        })
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
        let start = self.expect(TokenType::Data)?;

        // Parse data type name
        let name_token = self.expect(TokenType::Name)?;
        let name = name_token.value.clone();

        // Parse optional type parameters <T, U, V>
        let params: Vec<Name> = if self.matches(&TokenType::Lt) {
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

            // Parse shared members (method/field definitions)
            let mut members = Vec::new();
            while !self.matches(&TokenType::End) && !self.matches(&TokenType::Where) && !self.is_at_end() {
                members.push(self.parse_member()?);
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
            let check_loc = self.make_loc(&where_token, &where_token);
            let mut where_stmts = Vec::new();
            while !self.matches(&TokenType::End) && !self.is_at_end() {
                let stmt = self.parse_expr()?;
                where_stmts.push(Box::new(stmt));
            }
            let check_block = Box::new(Expr::SBlock {
                l: check_loc.clone(),
                stmts: where_stmts,
            });
            (Some(check_block), Some(check_loc))
        } else {
            (None, None)
        };

        let end = self.expect(TokenType::End)?;

        Ok(Expr::SData {
            l: self.make_loc(&start, &end),
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
        let start = self.peek().clone();
        let name_token = self.expect(TokenType::Name)?;
        let name = name_token.value.clone();
        let constr_loc = self.make_loc(&name_token, &name_token);

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

            self.expect(TokenType::RParen)?;

            // Parse optional with clause
            let with_members = if self.matches(&TokenType::With) {
                self.parse_data_with()?
            } else {
                Vec::new()
            };

            let end = if self.current > 0 {
                self.tokens[self.current - 1].clone()
            } else {
                start.clone()
            };

            Ok(Variant::SVariant {
                l: self.make_loc(&start, &end),
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

            let end = if self.current > 0 {
                self.tokens[self.current - 1].clone()
            } else {
                start.clone()
            };

            Ok(Variant::SSingletonVariant {
                l: self.make_loc(&start, &end),
                name,
                with_members,
            })
        }
    }

    /// Parse a variant member (parameter in a data constructor)
    /// Can be: ref binding | binding
    fn parse_variant_member(&mut self) -> ParseResult<VariantMember> {
        let start = self.peek().clone();

        // Check for ref modifier
        let member_type = if self.matches(&TokenType::Ref) {
            self.advance();
            VariantMemberType::SMutable
        } else {
            VariantMemberType::SNormal
        };

        // Parse the binding
        let bind = self.parse_bind()?;

        let end = if self.current > 0 {
            self.tokens[self.current - 1].clone()
        } else {
            start.clone()
        };

        Ok(VariantMember {
            node_type: "s-variant-member".to_string(),
            l: self.make_loc(&start, &end),
            member_type,
            bind,
        })
    }

    /// data-with: with: members END
    fn parse_data_with(&mut self) -> ParseResult<Vec<Member>> {
        self.expect(TokenType::With)?;
        self.expect(TokenType::Colon)?;

        let mut members = Vec::new();
        while !self.matches(&TokenType::End)
            && !self.matches(&TokenType::Bar)
            && !self.matches(&TokenType::Sharing)
            && !self.matches(&TokenType::Where)
            && !self.is_at_end() {
            members.push(self.parse_member()?);
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
                let field_loc = self.make_loc(&name_token, &name_token);

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
                l: self.make_loc(&row_start, &row_start),
                elems,
            });
        }

        let end = self.expect(TokenType::End)?;
        Ok(Expr::STable {
            l: self.make_loc(&start, &end),
            headers,
            rows,
        })
    }

    /// table-select: select columns from table
    fn parse_table_select(&mut self) -> ParseResult<Expr> {
        todo!("Implement parse_table_select")
    }

    /// load-table-expr: LOAD-TABLE COLON headers spec* END
    /// Parses a load-table expression like:
    ///   load-table: name, age
    ///     source: "data.csv"
    ///     sanitize name: string-sanitizer
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
                    l: self.make_loc(&name_token, &name_token),
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
                    l: self.make_loc(&source_tok, &source_tok),
                    src: Box::new(src_expr),
                });
            } else if self.matches(&TokenType::Sanitize) {
                // sanitize name: expr
                let sanitize_tok = self.expect(TokenType::Sanitize)?;
                let name_token = self.expect(TokenType::Name)?;
                self.expect(TokenType::Colon)?;
                let sanitizer_expr = self.parse_expr()?;

                spec.push(LoadTableSpec::SSanitize {
                    l: self.make_loc(&sanitize_tok, &name_token),
                    name: Name::SName {
                        l: self.make_loc(&name_token, &name_token),
                        s: name_token.value.clone(),
                    },
                    sanitizer: Box::new(sanitizer_expr),
                });
            } else {
                break;
            }
        }

        let end = self.expect(TokenType::End)?;

        Ok(Expr::SLoadTable {
            l: self.make_loc(&start, &end),
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
            l: self.make_loc(&start, &end),
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
        let body_loc = self.make_loc(&start, &end);
        let body = Box::new(Expr::SBlock {
            l: body_loc,
            stmts,
        });

        Ok(Expr::SCheck {
            l: self.make_loc(&start, &end),
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
    fn parse_check_test(&mut self) -> ParseResult<Expr> {
        // Simply parse the full expression (which will create SCheckTest when it sees the check operator)
        self.parse_expr()
    }

    /// spy-stmt: SPY [string] COLON spy-contents END
    /// spy-contents: spy-expr [COMMA spy-expr]* [COMMA]
    /// spy-expr: NAME COLON expr | expr (where expr is NAME creates implicit label)
    fn parse_spy_stmt(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::Spy)?;

        // Check for optional string message
        let message = if self.matches(&TokenType::String) {
            let str_tok = self.advance().clone();
            let str_value = str_tok.value.clone();
            let str_loc = self.make_loc(&str_tok, &str_tok);
            Some(Box::new(Expr::SStr {
                l: str_loc,
                s: str_value,
            }))
        } else {
            None
        };

        self.expect(TokenType::Colon)?;

        // Parse spy contents (comma-separated list of spy-expr)
        let mut contents = Vec::new();

        if !self.matches(&TokenType::End) {
            // Parse first spy-expr
            contents.push(self.parse_spy_expr()?);

            // Parse additional spy-exprs with comma separators
            while self.matches(&TokenType::Comma) {
                self.advance(); // consume comma

                // Allow optional trailing comma
                if self.matches(&TokenType::End) {
                    break;
                }

                contents.push(self.parse_spy_expr()?);
            }
        }

        let end = self.expect(TokenType::End)?;

        Ok(Expr::SSpyBlock {
            l: self.make_loc(&start, &end),
            message,
            contents,
        })
    }

    /// Parse a single spy expression: either "name: expr" or just "expr"
    /// If just "expr" and expr is an identifier, use implicit label
    fn parse_spy_expr(&mut self) -> ParseResult<SpyField> {
        let start_pos = self.current;

        // Try to parse as "name: expr" pattern
        // We need to look ahead to see if there's a colon after the name
        if self.matches(&TokenType::Name) {
            let name_tok = self.peek();
            let next = self.peek_ahead(1);

            if next.token_type == TokenType::Colon {
                // This is "name: expr" pattern
                let name_tok = self.advance();
                let name = name_tok.value.clone();
                let name_loc = name_tok.location.clone();
                self.expect(TokenType::Colon)?;
                let value_expr = self.parse_binop_expr()?;
                let value_loc = self.extract_loc(&value_expr);
                let value = Box::new(value_expr);

                // Build the location manually
                let l = Loc::new(
                    self.file_name.clone(),
                    name_loc.start_line,
                    name_loc.start_col,
                    name_loc.start_pos,
                    value_loc.end_line,
                    value_loc.end_column,
                    value_loc.end_char,
                );

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
            l: self.extract_loc(&expr),
            name,
            value: Box::new(expr),
            implicit_label,
        })
    }

    /// Extract location from an expression
    fn extract_loc(&self, expr: &Expr) -> Loc {
        match expr {
            Expr::SId { l, .. } => l.clone(),
            Expr::SIdVar { l, .. } => l.clone(),
            Expr::SIdLetrec { l, .. } => l.clone(),
            Expr::SNum { l, .. } => l.clone(),
            Expr::SFrac { l, .. } => l.clone(),
            Expr::SRfrac { l, .. } => l.clone(),
            Expr::SStr { l, .. } => l.clone(),
            Expr::SBool { l, .. } => l.clone(),
            Expr::SParen { l, .. } => l.clone(),
            Expr::SLam { l, .. } => l.clone(),
            Expr::SMethod { l, .. } => l.clone(),
            Expr::SExtend { l, .. } => l.clone(),
            Expr::SUpdate { l, .. } => l.clone(),
            Expr::SObj { l, .. } => l.clone(),
            Expr::SArray { l, .. } => l.clone(),
            Expr::SConstruct { l, .. } => l.clone(),
            Expr::SApp { l, .. } => l.clone(),
            Expr::SPrimApp { l, .. } => l.clone(),
            Expr::SAssign { l, .. } => l.clone(),
            Expr::SIf { l, .. } => l.clone(),
            Expr::SIfElse { l, .. } => l.clone(),
            Expr::SCases { l, .. } => l.clone(),
            Expr::SCasesElse { l, .. } => l.clone(),
            Expr::SOp { l, .. } => l.clone(),
            Expr::SCheckTest { l, .. } => l.clone(),
            Expr::SCheckExpr { l, .. } => l.clone(),
            Expr::SDot { l, .. } => l.clone(),
            Expr::SGetBang { l, .. } => l.clone(),
            Expr::SBracket { l, .. } => l.clone(),
            Expr::SData { l, .. } => l.clone(),
            Expr::SDataExpr { l, .. } => l.clone(),
            Expr::SFor { l, .. } => l.clone(),
            Expr::SBlock { l, .. } => l.clone(),
            Expr::SUserBlock { l, .. } => l.clone(),
            Expr::SFun { l, .. } => l.clone(),
            Expr::SType { l, .. } => l.clone(),
            Expr::SNewtype { l, .. } => l.clone(),
            Expr::SVar { l, .. } => l.clone(),
            Expr::SRec { l, .. } => l.clone(),
            Expr::SLet { l, .. } => l.clone(),
            Expr::SLetrec { l, .. } => l.clone(),
            Expr::SInstantiate { l, .. } => l.clone(),
            Expr::SLetExpr { l, .. } => l.clone(),
            Expr::SCheck { l, .. } => l.clone(),
            Expr::SReactor { l, .. } => l.clone(),
            Expr::STuple { l, .. } => l.clone(),
            Expr::STupleGet { l, .. } => l.clone(),
            Expr::SWhen { l, .. } => l.clone(),
            Expr::SContract { l, .. } => l.clone(),
            Expr::SSpyBlock { l, .. } => l.clone(),
            Expr::SIfPipe { l, .. } => l.clone(),
            Expr::SIfPipeElse { l, .. } => l.clone(),
            Expr::STypeLetExpr { l, .. } => l.clone(),
            Expr::STemplate { l, .. } => l.clone(),
            Expr::STableExtend { l, .. } => l.clone(),
            Expr::STableUpdate { l, .. } => l.clone(),
            Expr::STableSelect { l, .. } => l.clone(),
            Expr::STableOrder { l, .. } => l.clone(),
            Expr::STableFilter { l, .. } => l.clone(),
            Expr::STableExtract { l, .. } => l.clone(),
            Expr::STable { l, .. } => l.clone(),
            Expr::SLoadTable { l, .. } => l.clone(),
            _ => {
                // For any other variants, we'll need to add them explicitly
                // This is a temporary catch-all
                panic!("extract_loc not implemented for this Expr variant")
            }
        }
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
