//! Source instrumentation for the dataflow **soundness oracle**.
//!
//! Given a set of source spans (UTF-16 `start`/`end` offsets, matching the IR
//! trivia), wrap the r-value expression at each span in a probe call
//! `__p(start, <expr>)` that records the value's concrete runtime value and
//! returns it unchanged. Running the instrumented program then lets us check
//! that an analysis's abstract value at each span over-approximates every
//! observed concrete value.
//!
//! Only expression kinds that are always r-values *and* whose value is a fresh
//! primitive (literals, binary/unary/conditional, identifier reads) are wrapped
//! — so the wrap never changes syntax, `this`-binding, or assignment targets.

use std::collections::HashMap;

use swc_common::sync::Lrc;
use swc_common::{BytePos, FileName, SourceMap, Spanned, DUMMY_SP};
use swc_ecma_ast::{self as ast, EsVersion};
use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};
use swc_ecma_visit::{VisitMut, VisitMutWith};

use crate::codegen;

/// The probe harness prepended to the instrumented program: `__p` logs each
/// value (tagged by JS type so `NaN`/`null`/strings stay distinguishable) and
/// returns it; the trailing line dumps the log as JSON.
const HARNESS: &str = "var __log={};function __p(i,v){var k=''+i;(__log[k]=__log[k]||[]).push(typeof v==='number'?(v!==v?'n:NaN':(v===Infinity?'n:Infinity':(v===-Infinity?'n:-Infinity':'n:'+v))):typeof v==='string'?'s:'+v:typeof v==='boolean'?'b:'+v:v===null?'l:':typeof v==='bigint'?'g:'+v:'o:'+typeof v);return v;}\n";

/// Wrap the r-value expression at each `(start, end)` span with a probe call
/// tagged by the mapped probe id. Returns runnable JS (harness + instrumented
/// program + JSON dump), or `None` if the source doesn't parse.
pub fn instrument_probes(src: &str, probes: &HashMap<(i64, i64), i64>) -> Option<String> {
    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(Lrc::new(FileName::Anon), src.to_string());
    let file_start = fm.start_pos;
    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::EsNext,
        StringInput::from(&*fm),
        None,
    );
    let mut parser = Parser::new_from(lexer);
    let mut program = parser.parse_program().ok()?;
    drop(parser);

    let mut inst = Instrumenter { src, file_start, probes };
    program.visit_mut_with(&mut inst);

    let body = codegen(&program);
    Some(format!(
        "{HARNESS}{body}\nconsole.log('__PROBES__'+JSON.stringify(__log));"
    ))
}

struct Instrumenter<'a> {
    src: &'a str,
    file_start: BytePos,
    probes: &'a HashMap<(i64, i64), i64>,
}

impl Instrumenter<'_> {
    /// UTF-16 offset of a byte position (matches the IR's `start`/`end`).
    fn utf16(&self, bp: BytePos) -> i64 {
        let byte = (bp.0 - self.file_start.0) as usize;
        self.src
            .get(..byte)
            .map(|p| p.encode_utf16().count() as i64)
            .unwrap_or(byte as i64)
    }
}

/// Expression kinds that are always r-values and whose wrapping is semantically
/// transparent (no `this`-binding or l-value hazards).
fn is_safe_rval(e: &ast::Expr) -> bool {
    matches!(
        e,
        ast::Expr::Lit(_)
            | ast::Expr::Bin(_)
            | ast::Expr::Unary(_)
            | ast::Expr::Cond(_)
            | ast::Expr::Ident(_)
            | ast::Expr::Paren(_)
    )
}

impl VisitMut for Instrumenter<'_> {
    fn visit_mut_expr(&mut self, e: &mut ast::Expr) {
        // Children first, so a wrapped inner expression nests inside an outer one.
        e.visit_mut_children_with(self);
        if !is_safe_rval(e) {
            return;
        }
        let sp = e.span();
        let key = (self.utf16(sp.lo), self.utf16(sp.hi));
        if let Some(&id) = self.probes.get(&key) {
            let old = std::mem::replace(e, ast::Expr::Invalid(ast::Invalid { span: DUMMY_SP }));
            *e = wrap_probe(id, old);
        }
    }
}

fn wrap_probe(id: i64, inner: ast::Expr) -> ast::Expr {
    ast::Expr::Call(ast::CallExpr {
        span: DUMMY_SP,
        ctxt: Default::default(),
        callee: ast::Callee::Expr(Box::new(ast::Expr::Ident(ast::Ident::new(
            "__p".into(),
            DUMMY_SP,
            Default::default(),
        )))),
        args: vec![
            ast::ExprOrSpread {
                spread: None,
                expr: Box::new(ast::Expr::Lit(ast::Lit::Num(ast::Number {
                    span: DUMMY_SP,
                    value: id as f64,
                    raw: None,
                }))),
            },
            ast::ExprOrSpread { spread: None, expr: Box::new(inner) },
        ],
        type_args: None,
    })
}
