//! DCE comparison: **text → dead-code-eliminated** across the three Rust JS
//! ecosystems. Each runs its *DCE-only* path (not full minify):
//!   - ours: parse → JSIR → `jsir_ir::build::dce`
//!   - oxc:  parse → AST  → `Minifier::dce`
//!   - swc:  parse → AST  → resolver + `simplify::dce`
//!
//! Correctness gate: before timing, we render BOTH our DCE'd `Module` and oxc's
//! DCE'd AST back into a canonical, printer-independent statement list and assert
//! the two are **identical** — same surviving statements, in the same order, with
//! the same contents. So the speed numbers below compare runs that produce the
//! same output, not merely "similar" DCE. (swc is timed too but only sanity-
//! checked on count; its tree-shaker is a different, more aggressive pass.)
//!
//! Usage:  cargo run --release -p jsir-parse --example bench_dce [statements]

use std::hint::black_box;
use std::time::{Duration, Instant};

use jsir_ir::{Attr, IrRead, Module, OpId};

/// Canonical, whitespace-free rendering of a number so both front ends agree
/// regardless of how each stores the literal (raw token vs `f64` value).
fn fmt_num(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

// ── ours: render the DCE'd `Module`'s top-level statements ───────────────────

fn ours_num(m: &Module, op: OpId) -> String {
    for (k, a) in m.attrs_owned(op) {
        if k == "value" {
            if let Attr::F64(v) = a {
                return fmt_num(v);
            }
        }
    }
    "?".to_string()
}

/// Operands store `ValueId`s, which are NOT equal to `OpId`s here (the copying
/// DCE preserves source value-ids while renumbering ops), so we resolve each
/// operand to its defining op through a value-id → op map.
type DefMap = std::collections::HashMap<u32, OpId>;

fn ours_operand(m: &Module, def: &DefMap, op: OpId, i: usize) -> String {
    match def.get(&m.operands(op)[i].0) {
        Some(&d) => ours_expr(m, def, d),
        None => "?".to_string(),
    }
}

fn ours_expr(m: &Module, def: &DefMap, op: OpId) -> String {
    match m.op_name(op) {
        "jsir.numeric_literal" => ours_num(m, op),
        "jsir.identifier" | "jsir.identifier_ref" => {
            m.str_attr(op, "name").unwrap_or("?").to_string()
        }
        "jsir.binary_expression" | "jsir.assignment_expression" => {
            let l = ours_operand(m, def, op, 0);
            let r = ours_operand(m, def, op, 1);
            let o = m.str_attr(op, "operator_").unwrap_or("?");
            format!("{l}{o}{r}")
        }
        "jsir.variable_declarator" => {
            let l = ours_operand(m, def, op, 0);
            let r = ours_operand(m, def, op, 1);
            format!("{l}={r}")
        }
        other => format!("<{other}>"),
    }
}

fn ours_stmt(m: &Module, def: &DefMap, op: OpId) -> String {
    match m.op_name(op) {
        "jsir.variable_declaration" => {
            let kind = m.str_attr(op, "kind").unwrap_or("var");
            let mut decls = Vec::new();
            for r in m.regions(op) {
                for b in m.region_blocks(*r) {
                    for &o in m.block_ops(*b) {
                        if m.op_name(o) == "jsir.variable_declarator" {
                            decls.push(ours_expr(m, def, o));
                        }
                    }
                }
            }
            format!("{kind} {}", decls.join(","))
        }
        "jsir.expression_statement" => ours_operand(m, def, op, 0),
        other => format!("<{other}>"),
    }
}

/// The DCE'd module's program body as an ordered list of canonical statements.
fn ours_program(m: &Module) -> Vec<String> {
    let mut def: DefMap = DefMap::new();
    for i in 0..m.op_count() {
        let op = OpId(i as u32);
        for v in m.results(op) {
            def.insert(v.0, op);
        }
    }
    let file = m.root();
    let prog = m.block_ops(m.region_blocks(m.regions(file)[0])[0])[0];
    let body = m.region_blocks(m.regions(prog)[0])[0];
    // In the AST dialect, sub-expressions are emitted flat in the body block
    // (each with an SSA result `%N`); a *statement* is a body op with no result.
    m.block_ops(body)
        .iter()
        .filter(|&&op| m.results(op).is_empty())
        .map(|&op| ours_stmt(m, &def, op))
        .collect()
}

// ── oxc: render the DCE'd AST's top-level statements, same canonical form ─────

fn oxc_expr(e: &oxc_ast::ast::Expression) -> String {
    use oxc_ast::ast::{Expression, SimpleAssignmentTarget};
    match e {
        Expression::NumericLiteral(n) => fmt_num(n.value),
        Expression::Identifier(i) => i.name.to_string(),
        Expression::BinaryExpression(b) => {
            format!("{}{}{}", oxc_expr(&b.left), b.operator.as_str(), oxc_expr(&b.right))
        }
        Expression::AssignmentExpression(a) => {
            let lhs = match a.left.as_simple_assignment_target() {
                Some(SimpleAssignmentTarget::AssignmentTargetIdentifier(i)) => i.name.to_string(),
                _ => "<?>".to_string(),
            };
            format!("{}{}{}", lhs, a.operator.as_str(), oxc_expr(&a.right))
        }
        other => format!("<{:?}>", std::mem::discriminant(other)),
    }
}

fn oxc_stmt(s: &oxc_ast::ast::Statement) -> String {
    use oxc_ast::ast::{BindingPatternKind, Statement};
    match s {
        Statement::VariableDeclaration(d) => {
            let decls: Vec<String> = d
                .declarations
                .iter()
                .map(|decl| {
                    let name = match &decl.id.kind {
                        BindingPatternKind::BindingIdentifier(bi) => bi.name.to_string(),
                        _ => "<?>".to_string(),
                    };
                    match &decl.init {
                        Some(init) => format!("{name}={}", oxc_expr(init)),
                        None => name,
                    }
                })
                .collect();
            format!("{} {}", d.kind.as_str(), decls.join(","))
        }
        Statement::ExpressionStatement(e) => oxc_expr(&e.expression),
        other => format!("<{:?}>", std::mem::discriminant(other)),
    }
}

fn oxc_program(program: &oxc_ast::ast::Program) -> Vec<String> {
    program.body.iter().map(oxc_stmt).collect()
}

use oxc_allocator::Allocator;
use oxc_minifier::{CompressOptions, Minifier, MinifierOptions};
use oxc_parser::Parser;
use oxc_span::SourceType;

use swc_common::{Globals, Mark, GLOBALS};
use swc_ecma_ast::Program;
use swc_ecma_transforms_base::resolver;
use swc_ecma_transforms_optimization::simplify::dce::{dce, Config};
use swc_ecma_visit::VisitMutWith;

fn swc_stmt_count(p: &Program) -> usize {
    match p {
        Program::Module(m) => m.body.len(),
        Program::Script(s) => s.body.len(),
    }
}

/// Half the declarations are dead (declared, never read); half are live (read
/// via a following assignment). All three DCEs should drop the dead ones.
fn generate(stmts: usize) -> String {
    let mut s = String::with_capacity(stmts * 40);
    for i in 0..stmts {
        if i % 2 == 0 {
            s.push_str(&format!("var dead{i} = {i} + 1;\n")); // never referenced
        } else {
            s.push_str(&format!("var live{i} = {i};\n"));
            s.push_str(&format!("live{i} = live{i} + 2;\n")); // references live{i}
        }
    }
    s
}

fn bench(name: &str, bytes: usize, iters: u32, mut f: impl FnMut()) -> Duration {
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per = t.elapsed() / iters;
    let mbps = (bytes as f64 / (1 << 20) as f64) / per.as_secs_f64();
    println!("  {name:<30} {per:>10.2?}/iter   {mbps:>7.0} MiB/s");
    per
}

fn main() {
    let stmts: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let src = generate(stmts);
    let bytes = src.len();
    let iters = 30;
    println!("source: {} stmts, {:.2} MiB ({} dead vars)\n", stmts, bytes as f64 / (1 << 20) as f64, stmts / 2);

    // Sanity: copying DCE and in-place DCE remove the dead vars and agree.
    let m = jsir_parse::parse_to_module(&src).unwrap();
    let before = m.op_count();
    let (after_m, removed) = jsir_ir::build::dce(&m);
    let mut m2 = jsir_parse::parse_to_module(&src).unwrap();
    let removed2 = jsir_ir::build::dce_in_place(&mut m2);
    assert_eq!(removed, removed2, "in-place and copying DCE remove different counts");
    assert_eq!(after_m.print(), m2.print(), "in-place DCE output differs from copying DCE");
    println!("ours: removed {removed} dead var-decls ({before} ops); in-place == copying ✓");

    // Prove IDENTICAL OUTPUT vs oxc: compare the full ordered list of surviving
    // statements, rendered to the same canonical form from each side's result.
    let ours_out = ours_program(&after_m);
    let oxc_out = {
        let alloc = Allocator::default();
        let mut ret = Parser::new(&alloc, &src, SourceType::default()).parse();
        let before = ret.program.body.len();
        let opts = MinifierOptions { mangle: None, compress: Some(CompressOptions::default()) };
        Minifier::new(opts).dce(&alloc, &mut ret.program);
        println!("oxc:  {before} top-level stmts → {} after dce", ret.program.body.len());
        oxc_program(&ret.program)
    };
    if ours_out != oxc_out {
        let first = ours_out
            .iter()
            .zip(oxc_out.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(ours_out.len().min(oxc_out.len()));
        panic!(
            "IDENTICAL-OUTPUT FAILED: ours kept {} stmts, oxc kept {} — first diff at #{first}:\n  ours: {:?}\n  oxc:  {:?}",
            ours_out.len(),
            oxc_out.len(),
            ours_out.get(first),
            oxc_out.get(first),
        );
    }
    println!(
        "identical output as oxc ✓ — byte-for-byte same {} surviving statements (dropped {removed})\n",
        ours_out.len(),
    );

    println!("text → DCE'd (parse + DCE-only):");
    bench("ours: JSIR + dce (copying)", bytes, iters, || {
        let m = jsir_parse::parse_to_module(&src).unwrap();
        let (out, r) = jsir_ir::build::dce(&m);
        black_box((out.op_count(), r));
    });
    bench("ours: JSIR + dce (in-place)", bytes, iters, || {
        let mut m = jsir_parse::parse_to_module(&src).unwrap();
        let r = jsir_ir::build::dce_in_place(&mut m);
        black_box((m.op_count(), r));
    });
    bench("oxc: AST + Minifier::dce", bytes, iters, || {
        let alloc = Allocator::default();
        let mut ret = Parser::new(&alloc, &src, SourceType::default()).parse();
        let opts = MinifierOptions { mangle: None, compress: Some(CompressOptions::default()) };
        Minifier::new(opts).dce(&alloc, &mut ret.program);
        black_box(ret.program.body.len());
    });

    // swc: resolver (assign scopes/marks) + simplify::dce TreeShaker, in GLOBALS.
    GLOBALS.set(&Globals::default(), || {
        // correctness: confirm swc drops the dead vars
        let mut p = jsir_swc::parse(&src).unwrap();
        let before = swc_stmt_count(&p);
        let u = Mark::new();
        let t = Mark::new();
        p.visit_mut_with(&mut resolver(u, t, false));
        p.visit_mut_with(&mut dce(Config { top_level: true, ..Default::default() }, u));
        println!("\nswc: {before} top-level stmts → {} after dce", swc_stmt_count(&p));

        bench("swc: AST + simplify::dce", bytes, iters, || {
            let mut p = jsir_swc::parse(&src).unwrap();
            let u = Mark::new();
            let t = Mark::new();
            p.visit_mut_with(&mut resolver(u, t, false));
            p.visit_mut_with(&mut dce(Config { top_level: true, ..Default::default() }, u));
            black_box(swc_stmt_count(&p));
        });
    });
}
