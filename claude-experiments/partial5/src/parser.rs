//! JavaScript parser using swc

use anyhow::{Context, Result};
use swc_common::{
    errors::{ColorConfig, Handler},
    input::StringInput,
    sync::Lrc,
    FileName, SourceMap,
};
use swc_ecma_ast::*;
use swc_ecma_parser::{lexer::Lexer, Parser, Syntax};

/// Parse a JavaScript source string into an AST
pub fn parse_js(source: &str) -> Result<Module> {
    let cm: Lrc<SourceMap> = Default::default();
    let handler = Handler::with_tty_emitter(ColorConfig::Auto, true, false, Some(cm.clone()));

    let fm = cm.new_source_file(Lrc::new(FileName::Custom("input.js".into())), source.into());

    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::Es2020,
        StringInput::from(&*fm),
        None,
    );

    let mut parser = Parser::new_from(lexer);

    for e in parser.take_errors() {
        e.into_diagnostic(&handler).emit();
    }

    parser
        .parse_module()
        .map_err(|e| {
            e.into_diagnostic(&handler).emit();
            anyhow::anyhow!("Parse error")
        })
        .context("Failed to parse JavaScript")
}

/// Parse a JavaScript expression
pub fn parse_expr(source: &str) -> Result<Box<Expr>> {
    let cm: Lrc<SourceMap> = Default::default();
    let handler = Handler::with_tty_emitter(ColorConfig::Auto, true, false, Some(cm.clone()));

    let fm = cm.new_source_file(Lrc::new(FileName::Custom("input.js".into())), source.into());

    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::Es2020,
        StringInput::from(&*fm),
        None,
    );

    let mut parser = Parser::new_from(lexer);

    for e in parser.take_errors() {
        e.into_diagnostic(&handler).emit();
    }

    parser
        .parse_expr()
        .map_err(|e| {
            e.into_diagnostic(&handler).emit();
            anyhow::anyhow!("Parse error")
        })
        .context("Failed to parse expression")
}

/// Helper to get identifier name from an expression
pub fn expr_as_ident(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Ident(id) => Some(&id.sym),
        _ => None,
    }
}

/// Helper to get number from a literal expression
pub fn expr_as_number(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Lit(Lit::Num(n)) => Some(n.value),
        Expr::Unary(UnaryExpr {
            op: UnaryOp::Minus,
            arg,
            ..
        }) => expr_as_number(arg).map(|n| -n),
        _ => None,
    }
}

/// Check if an expression is an identifier with a specific name
pub fn is_ident(expr: &Expr, name: &str) -> bool {
    expr_as_ident(expr) == Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let source = "var x = 1 + 2;";
        let module = parse_js(source).unwrap();
        assert_eq!(module.body.len(), 1);
    }

    #[test]
    fn test_parse_while_switch() {
        let source = r#"
            while (state >= 0) {
                switch (state & 1) {
                    case 0: state = 1; break;
                    case 1: state = -1; break;
                }
            }
        "#;
        let module = parse_js(source).unwrap();
        assert_eq!(module.body.len(), 1);
    }
}
