//! Dead-branch elimination for statically-decidable `if` statements.
//!
//! This exists because of one shape, used by nearly every package that ships both
//! a development and a production build:
//!
//! ```js
//! if (process.env.NODE_ENV === 'production') module.exports = require('./cjs/react-dom-client.production.js');
//! else module.exports = require('./cjs/react-dom-client.development.js');
//! ```
//!
//! [`crate::vite_define`] turns the test into a comparison of two literals; this
//! pass resolves it and deletes the branch that cannot run. Both halves are
//! required: substitution alone leaves the dead `require(...)` in the tree, and
//! [`crate::parser`] collects dependency edges from a plain full-tree walk with no
//! reachability filter, so that dead call would still pull React's ENTIRE
//! development build into the graph — over half the emitted bytes of a real app.
//!
//! It runs as a source-to-source rewrite in
//! [`crate::bundler::ResolutionCache::apply_vite_replacements`], alongside the
//! other compile-time rewrites and BEFORE the module is parsed for dependencies,
//! which is what makes the dead edge disappear rather than merely go unexecuted.
//!
//! The evaluator is deliberately narrow. It folds only literal comparisons,
//! boolean/unary/logical combinations of them, and nothing else — no identifier
//! lookup, no `typeof`, no arithmetic. A test it cannot decide is simply left
//! alone. Being unable to shrink a bundle is acceptable; changing what a program
//! does is not.

use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::{Expression, Statement};
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_span::{GetSpan, SourceType, Span};

/// How many substitution passes to run before giving up. Each pass resolves one
/// nesting level (a hit does not descend into its own replacement), so a handful
/// covers any realistic nesting; the cap only stops a pathological file from
/// looping. Reaching it leaves the remaining branches in place, which is safe.
const MAX_PASSES: usize = 8;

/// Deletes every statically-dead `if` branch in `source`, returning the rewritten
/// source, or `None` when nothing is decidable.
pub fn transform(path: &Path, source: &str) -> Option<String> {
    // Cheap gate: nothing to decide without a branch to decide. `?` covers the
    // conditional-expression form (and harmlessly over-admits optional chaining
    // and nullish coalescing, which the pass then simply finds nothing to fold in).
    if !source.contains("if") && !source.contains('?') {
        return None;
    }
    let mut current: Option<String> = None;
    for _ in 0..MAX_PASSES {
        let text = current.as_deref().unwrap_or(source);
        match pass(path, text) {
            Some(rewritten) => current = Some(rewritten),
            None => break,
        }
    }
    current
}

/// One substitution pass: folds every decidable `if` whose dead branch is safe to
/// delete, outermost-first.
fn pass(path: &Path, source: &str) -> Option<String> {
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path)
        .unwrap_or_default()
        .with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    // A file that does not parse is not ours to rewrite; the real parse
    // downstream reports the error with proper diagnostics.
    if !parsed.diagnostics.is_empty() {
        return None;
    }
    let mut collector = BranchCollector {
        source,
        edits: Vec::new(),
    };
    collector.visit_program(&parsed.program);
    if collector.edits.is_empty() {
        return None;
    }
    Some(apply_edits(source, collector.edits))
}

struct BranchCollector<'s> {
    source: &'s str,
    edits: Vec<(Span, String)>,
}

impl<'a> Visit<'a> for BranchCollector<'_> {
    fn visit_statement(&mut self, statement: &Statement<'a>) {
        if let Statement::IfStatement(branch) = statement
            && let Some(test) = evaluate(&branch.test)
        {
            let taken = if test {
                Some(&branch.consequent)
            } else {
                branch.alternate.as_ref()
            };
            let dead = if test {
                branch.alternate.as_ref()
            } else {
                Some(&branch.consequent)
            };
            // A `var` or function declaration in the dead branch is hoisted out of
            // it, so deleting the branch would remove a binding the surrounding
            // code can still legally reference. Leave the whole statement alone
            // rather than risk a ReferenceError.
            if dead.is_none_or(|dead| !hoists_a_binding(dead)) {
                let replacement = taken.map_or(String::new(), |taken| {
                    // The taken branch is spliced in verbatim, braces included when
                    // it is a block. Keeping the block preserves the `let`/`const`
                    // scoping the branch already had.
                    self.source[taken.span().start as usize..taken.span().end as usize].to_string()
                });
                self.edits.push((branch.span, replacement));
                // Do not descend: the replacement's own nested `if`s are folded by
                // the next pass, which keeps the recorded spans non-overlapping.
                return;
            }
        }
        // An `else if` whose test is decidably false and which has no branch of
        // its own to take must disappear TOGETHER with the parent's `else`
        // keyword — replacing just the inner `if` with nothing leaves a
        // dangling `else` and a parse error (found live on Redux Toolkit's
        // `else if (process.env.NODE_ENV !== "production")` guard blocks).
        if let Statement::IfStatement(branch) = statement
            && let Some(Statement::IfStatement(inner)) = branch.alternate.as_ref()
            && let Some(test) = evaluate(&inner.test)
        {
            let taken = if test {
                Some(&inner.consequent)
            } else {
                inner.alternate.as_ref()
            };
            let dead = if test {
                inner.alternate.as_ref()
            } else {
                Some(&inner.consequent)
            };
            if taken.is_none() && dead.is_none_or(|dead| !hoists_a_binding(dead)) {
                self.edits.push((
                    Span::new(branch.consequent.span().end, inner.span.end),
                    String::new(),
                ));
                // The kept parts still deserve their own folds.
                self.visit_expression(&branch.test);
                self.visit_statement(&branch.consequent);
                return;
            }
        }
        walk::walk_statement(self, statement);
    }

    /// The expression form of the same dispatch. Packages that swap an
    /// implementation for a no-op in production use a ternary rather than an `if`:
    ///
    /// ```js
    /// var TanStackRouterDevtools = process.env.NODE_ENV !== "development" ? function () { return null; } : RealDevtools;
    /// ```
    ///
    /// Folding it is what makes the discarded arm's import unreferenced, so the
    /// module-level DCE can drop the entire devtools tree from the GRAPH. Left
    /// unfolded, the chunk minifier still collapses the ternary, but far too late:
    /// the module is already linked in, and its bytes ship.
    fn visit_conditional_expression(
        &mut self,
        conditional: &oxc_ast::ast::ConditionalExpression<'a>,
    ) {
        if let Some(test) = evaluate(&conditional.test) {
            let taken = if test {
                &conditional.consequent
            } else {
                &conditional.alternate
            };
            let span = taken.span();
            // Parenthesized so the spliced text is valid in every expression
            // position it could land in (a bare `function`/`object` literal would
            // otherwise be misread at the start of a statement).
            let text = &self.source[span.start as usize..span.end as usize];
            self.edits
                .push((conditional.span, format!("({text})")));
            return;
        }
        walk::walk_conditional_expression(self, conditional);
    }
}

/// The statically-known truthiness of `expression`, or `None` when undecidable.
fn evaluate(expression: &Expression<'_>) -> Option<bool> {
    match expression {
        Expression::BooleanLiteral(literal) => Some(literal.value),
        Expression::ParenthesizedExpression(inner) => evaluate(&inner.expression),
        Expression::UnaryExpression(unary) if unary.operator.is_not() => {
            Some(!evaluate(&unary.argument)?)
        }
        Expression::LogicalExpression(logical) => {
            use oxc_syntax::operator::LogicalOperator;
            let left = evaluate(&logical.left)?;
            match logical.operator {
                // Short-circuit: a decided left operand settles `&&`/`||` even when
                // the right operand is not itself a literal.
                LogicalOperator::And if !left => Some(false),
                LogicalOperator::Or if left => Some(true),
                LogicalOperator::And | LogicalOperator::Or => evaluate(&logical.right),
                LogicalOperator::Coalesce => None,
            }
        }
        Expression::BinaryExpression(binary) => {
            use oxc_syntax::operator::BinaryOperator;
            let left = literal_value(&binary.left)?;
            let right = literal_value(&binary.right)?;
            // Both operands are the same primitive kind here, so `==` and `===`
            // agree and no coercion is involved.
            match binary.operator {
                BinaryOperator::Equality | BinaryOperator::StrictEquality => Some(left == right),
                BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                    Some(left != right)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// A comparable primitive literal, tagged by kind so a string never compares equal
/// to a boolean of the same spelling.
#[derive(PartialEq)]
enum LiteralValue {
    String(String),
    Boolean(bool),
    Null,
}

fn literal_value(expression: &Expression<'_>) -> Option<LiteralValue> {
    match expression {
        Expression::StringLiteral(literal) => {
            Some(LiteralValue::String(literal.value.to_string()))
        }
        Expression::BooleanLiteral(literal) => Some(LiteralValue::Boolean(literal.value)),
        Expression::NullLiteral(_) => Some(LiteralValue::Null),
        Expression::ParenthesizedExpression(inner) => literal_value(&inner.expression),
        // A template literal with no substitutions is just a string; anything
        // interpolated is not statically known.
        Expression::TemplateLiteral(template)
            if template.expressions.is_empty() && template.quasis.len() == 1 =>
        {
            template.quasis[0]
                .value
                .cooked
                .as_ref()
                .map(|cooked| LiteralValue::String(cooked.to_string()))
        }
        _ => None,
    }
}

/// Whether `statement` declares anything that hoists out of its own block: a `var`
/// or a function declaration. Nested function and class bodies are not searched —
/// a `var` inside them belongs to that function, not to the branch.
fn hoists_a_binding(statement: &Statement<'_>) -> bool {
    let mut scan = HoistScan { found: false };
    scan.visit_statement(statement);
    scan.found
}

struct HoistScan {
    found: bool,
}

impl<'a> Visit<'a> for HoistScan {
    fn visit_variable_declaration(&mut self, declaration: &oxc_ast::ast::VariableDeclaration<'a>) {
        if declaration.kind.is_var() {
            self.found = true;
        }
        walk::walk_variable_declaration(self, declaration);
    }

    fn visit_function(
        &mut self,
        function: &oxc_ast::ast::Function<'a>,
        flags: oxc_semantic::ScopeFlags,
    ) {
        // The declaration itself hoists; its body's own `var`s do not escape it, so
        // the body is not searched.
        if function.is_declaration() {
            self.found = true;
        }
        let _ = flags;
    }

    fn visit_arrow_function_expression(
        &mut self,
        _arrow: &oxc_ast::ast::ArrowFunctionExpression<'a>,
    ) {
    }

    fn visit_class(&mut self, _class: &oxc_ast::ast::Class<'a>) {}
}

/// Applies non-overlapping `(span, replacement)` edits, left to right. The
/// collector never records a span inside another, so sorting by start and skipping
/// any overlap keeps the output well-formed.
fn apply_edits(source: &str, mut edits: Vec<(Span, String)>) -> String {
    edits.sort_by_key(|(span, _)| span.start);
    let mut output = String::with_capacity(source.len());
    let mut cursor = 0_usize;
    for (span, replacement) in edits {
        let start = span.start as usize;
        let end = span.end as usize;
        if start < cursor {
            continue;
        }
        output.push_str(&source[cursor..start]);
        output.push_str(&replacement);
        cursor = end;
    }
    output.push_str(&source[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(source: &str) -> Option<String> {
        transform(Path::new("m.js"), source)
    }

    #[test]
    fn the_package_build_dispatch_keeps_only_the_production_require() {
        // The shape this pass exists for, after `vite_define` substitution.
        let source = "if (\"production\" === 'production') { module.exports = require('./prod.js'); } else { module.exports = require('./dev.js'); }";
        let out = run(source).unwrap();
        assert!(out.contains("./prod.js"), "{out}");
        assert!(
            !out.contains("./dev.js"),
            "the dead require must be GONE from the source, not merely unexecuted: {out}"
        );
    }

    #[test]
    fn an_inverted_test_keeps_the_other_branch() {
        let out = run("if (\"production\" !== 'production') { dev(); } else { prod(); }").unwrap();
        assert!(out.contains("prod()"), "{out}");
        assert!(!out.contains("dev()"), "{out}");
    }

    #[test]
    fn a_dead_branch_without_an_else_becomes_nothing() {
        let out = run("before(); if (\"production\" !== 'production') { dev(); } after();").unwrap();
        assert!(out.contains("before()") && out.contains("after()"), "{out}");
        assert!(!out.contains("dev()"), "{out}");
    }

    #[test]
    fn an_undecidable_test_is_left_alone() {
        assert!(run("if (x === 1) { a(); } else { b(); }").is_none());
        assert!(run("if (typeof window !== 'undefined') { a(); }").is_none());
    }

    #[test]
    fn a_var_in_the_dead_branch_blocks_elimination() {
        // `var hoisted` is visible after the `if`, so deleting the branch would
        // turn a legal read into a ReferenceError.
        assert!(
            run("if (\"a\" === 'b') { var hoisted = 1; }\nuse(hoisted);").is_none(),
            "a hoisted var must block elimination"
        );
    }

    #[test]
    fn a_function_declaration_in_the_dead_branch_blocks_elimination() {
        assert!(
            run("if (\"a\" === 'b') { function hoisted() {} }\nuse(hoisted);").is_none(),
            "a hoisted function declaration must block elimination"
        );
    }

    #[test]
    fn a_var_inside_a_nested_function_does_not_block_elimination() {
        // That `var` belongs to the inner function, not to the branch.
        let out = run("if (\"a\" === 'b') { call(() => { var local = 1; return local; }); }").unwrap();
        assert!(!out.contains("call("), "{out}");
    }

    #[test]
    fn nested_branches_resolve_across_passes() {
        let source =
            "if (true) { if (\"production\" === 'production') { keep(); } else { drop(); } }";
        let out = run(source).unwrap();
        assert!(out.contains("keep()"), "{out}");
        assert!(!out.contains("drop()"), "{out}");
    }

    #[test]
    fn logical_operators_short_circuit() {
        let out = run("if (false && whatever) { drop(); } else { keep(); }").unwrap();
        assert!(out.contains("keep()") && !out.contains("drop()"), "{out}");
        let out = run("if (true || whatever) { keep(); } else { drop(); }").unwrap();
        assert!(out.contains("keep()") && !out.contains("drop()"), "{out}");
    }

    #[test]
    fn a_negated_test_folds() {
        let out = run("if (!(\"a\" === 'a')) { drop(); } else { keep(); }").unwrap();
        assert!(out.contains("keep()") && !out.contains("drop()"), "{out}");
    }

    #[test]
    fn a_string_never_equals_a_boolean_of_the_same_spelling() {
        // `"true" === true` is false in JS; the kind tag must preserve that.
        let out = run("if (\"true\" === true) { drop(); } else { keep(); }").unwrap();
        assert!(out.contains("keep()") && !out.contains("drop()"), "{out}");
    }

    #[test]
    fn a_ternary_dispatch_drops_the_unused_arm() {
        // The devtools shape: the real implementation becomes unreferenced, which
        // is what lets module-level DCE drop it from the graph.
        let source = "var D = \"production\" !== \"development\" ? function () { return null; } : RealDevtools;";
        let out = run(source).unwrap();
        assert!(
            !out.contains("RealDevtools"),
            "the discarded arm must be gone so its import goes unreferenced: {out}"
        );
        assert!(out.contains("return null"), "{out}");
    }

    #[test]
    fn a_folded_ternary_stays_valid_in_expression_position() {
        let out = run("call(\"a\" === \"a\" ? {k: 1} : other);").unwrap();
        assert!(out.contains("({k: 1})"), "must be parenthesized: {out}");
        assert!(!out.contains("other"), "{out}");
    }

    #[test]
    fn an_undecidable_ternary_is_left_alone() {
        assert!(run("var x = flag ? a : b;").is_none());
    }

    #[test]
    fn an_else_if_that_folds_to_nothing_removes_the_dangling_else() {
        let out = run("if (x) { a(); } else if (\"production\" !== 'production') { b(); }").unwrap();
        assert!(out.contains("if (x) { a(); }"), "{out}");
        assert!(!out.contains("else"), "no dangling else: {out}");
        // With a live third arm the generic splice already produces a valid
        // `else <arm>`; this must stay untouched by the new deletion.
        let out = run("if (x) { a(); } else if (\"a\" === 'b') { b(); } else { c(); }").unwrap();
        assert!(out.contains("else {c();}") || out.contains("else { c(); }"), "{out}");
    }

    #[test]
    fn an_else_if_chain_keeps_the_surviving_arm() {
        let out = run("if (\"a\" === 'b') { one(); } else if (\"c\" === 'c') { two(); } else { three(); }").unwrap();
        assert!(out.contains("two()"), "{out}");
        assert!(!out.contains("one()") && !out.contains("three()"), "{out}");
    }
}

