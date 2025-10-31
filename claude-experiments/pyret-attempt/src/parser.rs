// Pyret Parser - Pest integration
use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use crate::ast::*;

#[derive(Parser)]
#[grammar = "pyret.pest"]
pub struct PyretParser;

pub type ParseError = pest::error::Error<Rule>;

/// Parse a Pyret program from source code
pub fn parse_program(source: &str) -> Result<Program, ParseError> {
    let pairs = PyretParser::parse(Rule::program, source)?;
    let pair = pairs.into_iter().next().unwrap();

    Ok(parse_program_node(pair, source))
}

fn make_loc(pair: &Pair<Rule>, source: &str) -> Loc {
    let span = pair.as_span();
    let start = span.start();
    let end = span.end();

    // Calculate line and column numbers
    let text = &source[..start];
    let start_line = text.matches('\n').count() + 1;
    let start_column = text.rfind('\n').map(|i| start - i - 1).unwrap_or(start) + 1;

    let text_end = &source[..end];
    let end_line = text_end.matches('\n').count() + 1;
    let end_column = text_end.rfind('\n').map(|i| end - i - 1).unwrap_or(end) + 1;

    Loc {
        source: "input".to_string(),
        start_line,
        start_column,
        start_char: start,
        end_line,
        end_column,
        end_char: end,
    }
}

fn parse_program_node(pair: Pair<Rule>, source: &str) -> Program {
    let loc = make_loc(&pair, source);
    let mut use_stmt = None;
    let mut provide = Provide::ProvideNone(loc.clone());
    let mut provided_types = ProvideTypes::ProvideTypesNone(loc.clone());
    let mut provide_blocks = Vec::new();
    let mut imports = Vec::new();
    let mut block = Expr::Block { loc: loc.clone(), stmts: Vec::new() };

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::prelude => {
                for prelude_item in inner.into_inner() {
                    match prelude_item.as_rule() {
                        Rule::use_stmt => use_stmt = Some(parse_use_stmt(prelude_item, source)),
                        Rule::provide_stmt => {
                            // Parse provide statement
                            // This could be provide, provide-types, or provide block
                            // For now, simplified
                        },
                        Rule::import_stmt => imports.push(parse_import_stmt(prelude_item, source)),
                        _ => {}
                    }
                }
            },
            Rule::block => block = parse_block(inner, source),
            Rule::EOI => {},
            _ => {}
        }
    }

    Program {
        loc,
        use_stmt,
        provide,
        provided_types,
        provide_blocks,
        imports,
        block,
    }
}

fn parse_use_stmt(pair: Pair<Rule>, source: &str) -> Use {
    let loc = make_loc(&pair, source);
    let mut name = Name::Name { loc: loc.clone(), name: "".to_string() };
    let mut module = ImportType::Name(Name::Name { loc: loc.clone(), name: "".to_string() });

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::NAME => name = parse_name(inner, source),
            Rule::import_source => module = parse_import_source(inner, source),
            _ => {}
        }
    }

    Use { loc, name, module }
}

fn parse_import_stmt(pair: Pair<Rule>, source: &str) -> Import {
    let loc = make_loc(&pair, source);
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::import_source => {
            let module = parse_import_source(inner, source);
            Import::Include { loc, module }
        },
        _ => Import::Include {
            loc: loc.clone(),
            module: ImportType::Name(Name::Name { loc: loc.clone(), name: "unknown".to_string() }),
        },
    }
}

fn parse_import_source(pair: Pair<Rule>, source: &str) -> ImportType {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::import_name => {
            let name = parse_name(inner.into_inner().next().unwrap(), source);
            ImportType::Name(name)
        },
        Rule::import_special => {
            let mut name = Name::Name {
                loc: make_loc(&inner, source),
                name: "".to_string()
            };
            let mut args = Vec::new();

            for part in inner.into_inner() {
                match part.as_rule() {
                    Rule::NAME => name = parse_name(part, source),
                    Rule::STRING => args.push(parse_string_value(&part)),
                    _ => {}
                }
            }

            ImportType::Special(name, args)
        },
        _ => ImportType::Name(Name::Name {
            loc: make_loc(&inner, source),
            name: "unknown".to_string(),
        }),
    }
}

fn parse_block(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut stmts = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::stmt {
            stmts.push(parse_stmt(inner, source));
        }
    }

    Expr::Block { loc, stmts }
}

fn parse_stmt(pair: Pair<Rule>, source: &str) -> Expr {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::type_expr => parse_type_expr(inner, source),
        Rule::newtype_expr => parse_newtype_expr(inner, source),
        Rule::let_expr => parse_let_expr(inner, source),
        Rule::fun_expr => parse_fun_expr(inner, source),
        Rule::var_expr => parse_var_expr(inner, source),
        Rule::rec_expr => parse_rec_expr(inner, source),
        Rule::assign_expr => parse_assign_expr(inner, source),
        Rule::when_expr => parse_when_expr(inner, source),
        Rule::data_expr => parse_data_expr(inner, source),
        Rule::check_expr => parse_check_expr(inner, source),
        Rule::check_test => parse_check_test(inner, source),
        Rule::contract_stmt => parse_contract_stmt(inner, source),
        Rule::spy_stmt => parse_spy_stmt(inner, source),
        _ => Expr::Template(make_loc(&inner, source)),
    }
}

fn parse_type_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut name = Name::Name { loc: loc.clone(), name: "".to_string() };
    let mut params = Vec::new();
    let mut ann = Ann::Blank(loc.clone());

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::NAME => name = parse_name(inner, source),
            Rule::ty_params => params = parse_ty_params(inner, source),
            Rule::ann => ann = parse_ann(inner, source),
            _ => {}
        }
    }

    Expr::Type { loc, name, params, ann }
}

fn parse_newtype_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut names: Vec<_> = pair.into_inner()
        .filter(|p| p.as_rule() == Rule::NAME)
        .map(|p| parse_name(p, source))
        .collect();

    let name = names.get(0).cloned().unwrap_or(Name::Name { loc: loc.clone(), name: "".to_string() });
    let as_name = names.get(1).cloned().unwrap_or(Name::Name { loc: loc.clone(), name: "".to_string() });

    Expr::Newtype { loc, name, as_name }
}

fn parse_let_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut binding = Binding::Name {
        loc: loc.clone(),
        shadow: false,
        name: Name::Name { loc: loc.clone(), name: "".to_string() },
        ann: None,
    };
    let mut value = Box::new(Expr::Template(loc.clone()));

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::toplevel_binding => binding = parse_binding(inner.into_inner().next().unwrap(), source),
            Rule::binop_expr => value = Box::new(parse_binop_expr(inner, source)),
            _ => {}
        }
    }

    Expr::LetStmt {
        loc,
        name: binding,
        value,
        keyword_val: false,
    }
}

fn parse_fun_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut name = String::new();
    let mut params = Vec::new();
    let mut args = Vec::new();
    let mut ann = Ann::Blank(loc.clone());
    let mut doc = String::new();
    let mut body = Box::new(Expr::Block { loc: loc.clone(), stmts: Vec::new() });
    let mut check = None;
    let mut blocky = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::NAME => name = inner.as_str().to_string(),
            Rule::fun_header => {
                let header = parse_fun_header(inner, source);
                params = header.0;
                args = header.1;
                ann = header.2;
            },
            Rule::doc_string => {
                if let Some(doc_pair) = inner.into_inner().next() {
                    doc = parse_string_value(&doc_pair);
                }
            },
            Rule::block => body = Box::new(parse_block(inner, source)),
            Rule::where_clause => {
                if let Some(check_block) = inner.into_inner().next() {
                    check = Some(Box::new(parse_block(check_block, source)));
                }
            },
            Rule::BLOCK | Rule::COLON => blocky = inner.as_rule() == Rule::BLOCK,
            _ => {}
        }
    }

    Expr::Fun {
        loc: loc.clone(),
        name,
        params,
        args,
        ann,
        doc,
        body,
        check_loc: None,
        check,
        blocky,
    }
}

fn parse_fun_header(pair: Pair<Rule>, source: &str) -> (Vec<Name>, Vec<Binding>, Ann) {
    let mut params = Vec::new();
    let mut args = Vec::new();
    let mut ann = Ann::Blank(make_loc(&pair, source));

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ty_params => params = parse_ty_params(inner, source),
            Rule::args => args = parse_args(inner, source),
            Rule::return_ann => {
                if let Some(ann_pair) = inner.into_inner().next() {
                    ann = parse_ann(ann_pair, source);
                }
            },
            _ => {}
        }
    }

    (params, args, ann)
}

fn parse_ty_params(pair: Pair<Rule>, source: &str) -> Vec<Name> {
    let mut params = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::comma_names {
            params = parse_comma_names(inner, source);
        }
    }

    params
}

fn parse_comma_names(pair: Pair<Rule>, source: &str) -> Vec<Name> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::NAME)
        .map(|p| parse_name(p, source))
        .collect()
}

fn parse_args(pair: Pair<Rule>, source: &str) -> Vec<Binding> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::binding)
        .map(|p| parse_binding(p, source))
        .collect()
}

fn parse_binding(pair: Pair<Rule>, source: &str) -> Binding {
    let loc = make_loc(&pair, source);
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::name_binding => parse_name_binding(inner, source),
        Rule::tuple_binding => parse_tuple_binding(inner, source),
        _ => Binding::Name {
            loc: loc.clone(),
            shadow: false,
            name: Name::Name { loc: loc.clone(), name: "unknown".to_string() },
            ann: None,
        },
    }
}

fn parse_name_binding(pair: Pair<Rule>, source: &str) -> Binding {
    let loc = make_loc(&pair, source);
    let mut shadow = false;
    let mut name = Name::Name { loc: loc.clone(), name: "".to_string() };
    let mut ann = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::SHADOW => shadow = true,
            Rule::NAME => name = parse_name(inner, source),
            Rule::ann => ann = Some(parse_ann(inner, source)),
            _ => {}
        }
    }

    Binding::Name { loc, shadow, name, ann }
}

fn parse_tuple_binding(pair: Pair<Rule>, source: &str) -> Binding {
    let loc = make_loc(&pair, source);
    let mut fields = Vec::new();
    let mut as_name = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::binding => fields.push(parse_binding(inner, source)),
            Rule::name_binding => as_name = Some(Box::new(parse_name_binding(inner, source))),
            _ => {}
        }
    }

    Binding::Tuple { loc, fields, as_name }
}

fn parse_var_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut binding = Binding::Name {
        loc: loc.clone(),
        shadow: false,
        name: Name::Name { loc: loc.clone(), name: "".to_string() },
        ann: None,
    };
    let mut value = Box::new(Expr::Template(loc.clone()));

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::toplevel_binding => binding = parse_binding(inner.into_inner().next().unwrap(), source),
            Rule::binop_expr => value = Box::new(parse_binop_expr(inner, source)),
            _ => {}
        }
    }

    Expr::Var { loc, name: binding, value }
}

fn parse_rec_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut binding = Binding::Name {
        loc: loc.clone(),
        shadow: false,
        name: Name::Name { loc: loc.clone(), name: "".to_string() },
        ann: None,
    };
    let mut value = Box::new(Expr::Template(loc.clone()));

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::toplevel_binding => binding = parse_binding(inner.into_inner().next().unwrap(), source),
            Rule::binop_expr => value = Box::new(parse_binop_expr(inner, source)),
            _ => {}
        }
    }

    Expr::Rec { loc, name: binding, value }
}

fn parse_assign_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut id = Name::Name { loc: loc.clone(), name: "".to_string() };
    let mut value = Box::new(Expr::Template(loc.clone()));

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::NAME => id = parse_name(inner, source),
            Rule::binop_expr => value = Box::new(parse_binop_expr(inner, source)),
            _ => {}
        }
    }

    Expr::Assign { loc, id, value }
}

fn parse_when_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut test = Box::new(Expr::Template(loc.clone()));
    let mut block = Box::new(Expr::Block { loc: loc.clone(), stmts: Vec::new() });
    let mut blocky = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::binop_expr => test = Box::new(parse_binop_expr(inner, source)),
            Rule::block => block = Box::new(parse_block(inner, source)),
            Rule::BLOCK | Rule::COLON => blocky = inner.as_rule() == Rule::BLOCK,
            _ => {}
        }
    }

    Expr::When { loc, test, block, blocky }
}

fn parse_data_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut name = Name::Name { loc: loc.clone(), name: "".to_string() };
    let mut params = Vec::new();
    let mut variants = Vec::new();
    let mut shared_members = Vec::new();
    let mut check = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::NAME => name = parse_name(inner, source),
            Rule::ty_params => params = parse_ty_params(inner, source),
            Rule::first_data_variant | Rule::data_variant => {
                // Simplified variant parsing
            },
            Rule::data_sharing => {
                // Parse shared members
            },
            Rule::where_clause => {
                if let Some(check_block) = inner.into_inner().next() {
                    check = Some(Box::new(parse_block(check_block, source)));
                }
            },
            _ => {}
        }
    }

    Expr::Data { loc, name, params, variants, shared_members, check }
}

fn parse_check_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut name = None;
    let mut body = Box::new(Expr::Block { loc: loc.clone(), stmts: Vec::new() });

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::STRING => name = Some(parse_string_value(&inner)),
            Rule::block => body = Box::new(parse_block(inner, source)),
            _ => {}
        }
    }

    Expr::Check { loc, name, body }
}

fn parse_check_test(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    // Simplified - just return the expression
    if let Some(first) = pair.into_inner().next() {
        if first.as_rule() == Rule::binop_expr {
            return parse_binop_expr(first, source);
        }
    }

    Expr::Template(loc)
}

fn parse_contract_stmt(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut name = Name::Name { loc: loc.clone(), name: "".to_string() };
    let mut params = Vec::new();
    let mut ann = Ann::Blank(loc.clone());

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::NAME => name = parse_name(inner, source),
            Rule::ty_params => params = parse_ty_params(inner, source),
            Rule::ann | Rule::noparen_arrow_ann => ann = parse_ann(inner, source),
            _ => {}
        }
    }

    Expr::Contract { loc, name, params, ann }
}

fn parse_spy_stmt(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut expr = None;
    let mut fields = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::binop_expr => expr = Some(Box::new(parse_binop_expr(inner, source))),
            Rule::spy_contents => {
                // Parse spy fields
            },
            _ => {}
        }
    }

    Expr::Spy { loc, expr, fields }
}

fn parse_binop_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let mut exprs: Vec<Expr> = Vec::new();
    let mut ops: Vec<(Loc, String)> = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::expr => exprs.push(parse_expr(inner, source)),
            Rule::binop => {
                let loc = make_loc(&inner, source);
                let op = inner.as_str().to_string();
                ops.push((loc, op));
            },
            _ => {}
        }
    }

    // Build left-associative tree
    if exprs.is_empty() {
        return Expr::Template(Loc::dummy());
    }

    let mut result = exprs.into_iter();
    let mut current = result.next().unwrap();

    for ((op_loc, op), right) in ops.into_iter().zip(result) {
        let loc = current.loc().clone();
        current = Expr::Op {
            loc,
            op_loc,
            op,
            left: Box::new(current),
            right: Box::new(right),
        };
    }

    current
}

fn parse_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::prim_expr => parse_prim_expr(inner, source),
        Rule::id_expr => parse_id_expr(inner, source),
        Rule::paren_expr => parse_paren_expr(inner, source),
        Rule::lambda_expr => parse_lambda_expr(inner, source),
        Rule::method_expr => parse_method_expr(inner, source),
        Rule::obj_expr => parse_obj_expr(inner, source),
        Rule::tuple_expr => parse_tuple_expr(inner, source),
        Rule::if_expr => parse_if_expr(inner, source),
        Rule::template_expr => Expr::Template(make_loc(&inner, source)),
        Rule::user_block_expr => parse_user_block_expr(inner, source),
        _ => Expr::Template(make_loc(&inner, source)),
    }
}

fn parse_prim_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let inner = pair.into_inner().next().unwrap();
    let loc = make_loc(&inner, source);

    match inner.as_rule() {
        Rule::num_expr => {
            let value = inner.as_str().parse().unwrap_or(0.0);
            Expr::Num { loc, value }
        },
        Rule::bool_expr => {
            let value = inner.as_str() == "true";
            Expr::Bool { loc, value }
        },
        Rule::string_expr => {
            let value = parse_string_value(&inner);
            Expr::Str { loc, value }
        },
        Rule::frac_expr => {
            let parts: Vec<&str> = inner.as_str().split('/').collect();
            let numerator = parts[0].parse().unwrap_or(0);
            let denominator = parts[1].parse().unwrap_or(1);
            Expr::Frac { loc, numerator, denominator }
        },
        Rule::rfrac_expr => {
            let s = inner.as_str().trim_start_matches('~');
            let parts: Vec<&str> = s.split('/').collect();
            let numerator = parts[0].parse().unwrap_or(0);
            let denominator = parts[1].parse().unwrap_or(1);
            Expr::RoughFrac { loc, numerator, denominator }
        },
        _ => Expr::Template(loc),
    }
}

fn parse_id_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let name = parse_name(pair.into_inner().next().unwrap(), source);

    Expr::Id { loc, name }
}

fn parse_paren_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::binop_expr {
            return parse_binop_expr(inner, source);
        }
    }

    Expr::Template(loc)
}

fn parse_lambda_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut params = Vec::new();
    let mut args = Vec::new();
    let mut ann = Ann::Blank(loc.clone());
    let mut doc = String::new();
    let mut body = Box::new(Expr::Block { loc: loc.clone(), stmts: Vec::new() });
    let mut check = None;
    let mut blocky = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::fun_header => {
                let header = parse_fun_header(inner, source);
                params = header.0;
                args = header.1;
                ann = header.2;
            },
            Rule::doc_string => {
                if let Some(doc_pair) = inner.into_inner().next() {
                    doc = parse_string_value(&doc_pair);
                }
            },
            Rule::block => body = Box::new(parse_block(inner, source)),
            Rule::where_clause => {
                if let Some(check_block) = inner.into_inner().next() {
                    check = Some(Box::new(parse_block(check_block, source)));
                }
            },
            Rule::BLOCK | Rule::COLON => blocky = inner.as_rule() == Rule::BLOCK,
            _ => {}
        }
    }

    Expr::Lambda {
        loc: loc.clone(),
        params,
        args,
        ann,
        doc,
        body,
        check_loc: None,
        check,
        blocky,
    }
}

fn parse_method_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut params = Vec::new();
    let mut args = Vec::new();
    let mut ann = Ann::Blank(loc.clone());
    let mut doc = String::new();
    let mut body = Box::new(Expr::Block { loc: loc.clone(), stmts: Vec::new() });
    let mut check = None;
    let mut blocky = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::fun_header => {
                let header = parse_fun_header(inner, source);
                params = header.0;
                args = header.1;
                ann = header.2;
            },
            Rule::doc_string => {
                if let Some(doc_pair) = inner.into_inner().next() {
                    doc = parse_string_value(&doc_pair);
                }
            },
            Rule::block => body = Box::new(parse_block(inner, source)),
            Rule::where_clause => {
                if let Some(check_block) = inner.into_inner().next() {
                    check = Some(Box::new(parse_block(check_block, source)));
                }
            },
            Rule::BLOCK | Rule::COLON => blocky = inner.as_rule() == Rule::BLOCK,
            _ => {}
        }
    }

    Expr::Method {
        loc: loc.clone(),
        params,
        args,
        ann,
        doc,
        body,
        check_loc: None,
        check,
        blocky,
    }
}

fn parse_obj_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut fields = Vec::new();

    // Parse object fields if present
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::obj_fields {
            // Simplified - just collect fields
        }
    }

    Expr::Obj { loc, fields }
}

fn parse_tuple_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut fields = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::tuple_fields {
            for field_pair in inner.into_inner() {
                if field_pair.as_rule() == Rule::binop_expr {
                    fields.push(parse_binop_expr(field_pair, source));
                }
            }
        }
    }

    Expr::Tuple { loc, fields }
}

fn parse_if_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut branches = Vec::new();
    let mut else_branch = None;
    let mut blocky = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::binop_expr => {
                // This is the test of the first if branch
                // We need to collect the following block
            },
            Rule::block => {
                // This is a branch body
            },
            Rule::else_if => {
                // Parse else-if branch
            },
            Rule::BLOCK | Rule::COLON => blocky = inner.as_rule() == Rule::BLOCK,
            _ => {}
        }
    }

    if let Some(else_expr) = else_branch {
        Expr::IfElse { loc, branches, else_branch: else_expr, blocky }
    } else {
        Expr::If { loc, branches, blocky }
    }
}

fn parse_user_block_expr(pair: Pair<Rule>, source: &str) -> Expr {
    let loc = make_loc(&pair, source);
    let mut body = Box::new(Expr::Block { loc: loc.clone(), stmts: Vec::new() });

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::block {
            body = Box::new(parse_block(inner, source));
        }
    }

    Expr::UserBlock { loc, body }
}

fn parse_ann(pair: Pair<Rule>, source: &str) -> Ann {
    let loc = make_loc(&pair, source);

    // Handle direct noparen_arrow_ann calls
    if pair.as_rule() == Rule::noparen_arrow_ann {
        let mut args = Vec::new();
        let mut ret = Box::new(Ann::Blank(loc.clone()));

        for part in pair.into_inner() {
            match part.as_rule() {
                Rule::ann => ret = Box::new(parse_ann(part, source)),
                Rule::arrow_ann_args => {
                    // Parse arrow_ann_args if needed
                }
                _ => {}
            }
        }

        return Ann::Arrow { loc, args, ret };
    }

    // For Rule::ann, get the base_ann child
    let inner = match pair.into_inner().next() {
        Some(i) => i,
        None => return Ann::Blank(loc),
    };
    let inner_loc = make_loc(&inner, source);

    match inner.as_rule() {
        Rule::name_ann => {
            let name = parse_name(inner.into_inner().next().unwrap(), source);
            Ann::Name { loc: inner_loc, name }
        },
        Rule::record_ann => {
            let mut fields = Vec::new();
            // Parse record fields
            Ann::Record { loc: inner_loc, fields }
        },
        Rule::tuple_ann => {
            let mut fields = Vec::new();
            for field in inner.into_inner() {
                if field.as_rule() == Rule::ann {
                    fields.push(parse_ann(field, source));
                }
            }
            Ann::Tuple { loc: inner_loc, fields }
        },
        Rule::arrow_ann => {
            let mut args = Vec::new();
            let mut ret = Box::new(Ann::Blank(inner_loc.clone()));

            for part in inner.into_inner() {
                match part.as_rule() {
                    Rule::ann => ret = Box::new(parse_ann(part, source)),
                    _ => {}
                }
            }

            Ann::Arrow { loc: inner_loc, args, ret }
        },
        Rule::dot_ann => {
            let names: Vec<_> = inner.into_inner()
                .filter(|p| p.as_rule() == Rule::NAME)
                .map(|p| parse_name(p, source))
                .collect();

            let obj = names.get(0).cloned().unwrap_or(Name::Name {
                loc: inner_loc.clone(),
                name: "".to_string()
            });
            let field = names.get(1).map(|n| n.to_string()).unwrap_or_default();

            Ann::Dot { loc: inner_loc, obj, field }
        },
        _ => Ann::Blank(inner_loc),
    }
}

fn parse_name(pair: Pair<Rule>, source: &str) -> Name {
    let loc = make_loc(&pair, source);
    let name_str = pair.as_str().to_string();

    if name_str == "_" {
        Name::Underscore(loc)
    } else {
        Name::Name { loc, name: name_str }
    }
}

fn parse_string_value(pair: &Pair<Rule>) -> String {
    let s = pair.as_str();
    // Remove quotes and unescape
    s[1..s.len()-1].to_string()
}
