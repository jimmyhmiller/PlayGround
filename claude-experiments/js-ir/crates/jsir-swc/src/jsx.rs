//! Desugar JSX to `React.createElement(...)` calls on the swc AST, before
//! conversion to JSIR.
//!
//! Upstream jsir has no JSX ops, and for the React-compiler work we only need
//! the *semantics* of JSX (it allocates a fresh element from its props/children
//! each render, so it is a memoizable reference value whose dependencies are the
//! attribute and child expressions). The classic createElement desugaring gives
//! exactly that using ordinary calls/objects, which the rest of the pipeline
//! already understands.

use swc_common::DUMMY_SP;
use swc_ecma_ast as ast;
use swc_ecma_visit::{VisitMut, VisitMutWith};

/// Rewrite every JSX expression in `program` into a `React.createElement` call.
pub fn desugar(program: &mut ast::Program) {
    program.visit_mut_with(&mut Jsx);
}

struct Jsx;

impl VisitMut for Jsx {
    fn visit_mut_expr(&mut self, e: &mut ast::Expr) {
        // Desugar nested JSX (attribute values, children) first.
        e.visit_mut_children_with(self);
        match e {
            ast::Expr::JSXElement(el) => *e = element_to_call(el),
            ast::Expr::JSXFragment(fr) => *e = fragment_to_call(fr),
            _ => {}
        }
    }
}

fn ident(name: &str) -> ast::Ident {
    ast::Ident::new(name.into(), DUMMY_SP, Default::default())
}

fn member(obj: ast::Expr, prop: &str) -> ast::Expr {
    ast::Expr::Member(ast::MemberExpr {
        span: DUMMY_SP,
        obj: Box::new(obj),
        prop: ast::MemberProp::Ident(ast::IdentName::new(prop.into(), DUMMY_SP)),
    })
}

/// A string literal with a `raw` form (the converter requires `extra.raw`).
fn jstr(s: &str) -> ast::Str {
    ast::Str { span: DUMMY_SP, value: s.into(), raw: Some(format!("{s:?}").into()) }
}

fn str_lit(s: &str) -> ast::Expr {
    ast::Expr::Lit(ast::Lit::Str(jstr(s)))
}

fn arg(e: ast::Expr) -> ast::ExprOrSpread {
    ast::ExprOrSpread { spread: None, expr: Box::new(e) }
}

fn create_element(tag: ast::Expr, props: ast::Expr, children: Vec<ast::ExprOrSpread>) -> ast::Expr {
    let callee = member(ast::Expr::Ident(ident("React")), "createElement");
    let mut args = vec![arg(tag), arg(props)];
    args.extend(children);
    ast::Expr::Call(ast::CallExpr {
        span: DUMMY_SP,
        ctxt: Default::default(),
        callee: ast::Callee::Expr(Box::new(callee)),
        args,
        type_args: None,
    })
}

fn element_to_call(el: &ast::JSXElement) -> ast::Expr {
    let tag = element_name(&el.opening.name);
    let props = attrs_to_object(&el.opening.attrs);
    let children = children_to_args(&el.children);
    create_element(tag, props, children)
}

fn fragment_to_call(fr: &ast::JSXFragment) -> ast::Expr {
    let tag = member(ast::Expr::Ident(ident("React")), "Fragment");
    let children = children_to_args(&fr.children);
    create_element(tag, ast::Expr::Lit(ast::Lit::Null(ast::Null { span: DUMMY_SP })), children)
}

/// `<div>` -> string "div"; `<Comp>` -> identifier `Comp`; `<a.b>` -> member.
fn element_name(name: &ast::JSXElementName) -> ast::Expr {
    match name {
        ast::JSXElementName::Ident(i) => {
            let s = i.sym.as_str();
            if s.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
                str_lit(s)
            } else {
                ast::Expr::Ident(i.clone())
            }
        }
        ast::JSXElementName::JSXMemberExpr(m) => jsx_member(m),
        ast::JSXElementName::JSXNamespacedName(n) => str_lit(&format!("{}:{}", n.ns.sym, n.name.sym)),
    }
}

fn jsx_member(m: &ast::JSXMemberExpr) -> ast::Expr {
    let obj = match &m.obj {
        ast::JSXObject::Ident(i) => ast::Expr::Ident(i.clone()),
        ast::JSXObject::JSXMemberExpr(inner) => jsx_member(inner),
    };
    member(obj, m.prop.sym.as_str())
}

fn attrs_to_object(attrs: &[ast::JSXAttrOrSpread]) -> ast::Expr {
    if attrs.is_empty() {
        return ast::Expr::Lit(ast::Lit::Null(ast::Null { span: DUMMY_SP }));
    }
    let mut props: Vec<ast::PropOrSpread> = Vec::new();
    for a in attrs {
        match a {
            ast::JSXAttrOrSpread::SpreadElement(s) => {
                props.push(ast::PropOrSpread::Spread(ast::SpreadElement {
                    dot3_token: DUMMY_SP,
                    expr: s.expr.clone(),
                }));
            }
            ast::JSXAttrOrSpread::JSXAttr(attr) => {
                let key = match &attr.name {
                    ast::JSXAttrName::Ident(i) => i.sym.to_string(),
                    ast::JSXAttrName::JSXNamespacedName(n) => format!("{}:{}", n.ns.sym, n.name.sym),
                };
                let value = match &attr.value {
                    None => ast::Expr::Lit(ast::Lit::Bool(ast::Bool { span: DUMMY_SP, value: true })),
                    // Rebuild the string from its cooked value via `jstr` so the emitted
                    // `raw` is JS-escaped. Cloning `s` keeps swc's JSX `raw`, which may hold
                    // a literal newline (legal in JSX, illegal in a JS string literal) and
                    // would emit an unterminated string.
                    Some(ast::JSXAttrValue::Str(s)) => match s.value.as_str() {
                        Some(v) => str_lit(v),
                        // Lone surrogates can't go through `jstr`; keep swc's original.
                        None => ast::Expr::Lit(ast::Lit::Str(s.clone())),
                    },
                    Some(ast::JSXAttrValue::JSXExprContainer(c)) => match &c.expr {
                        ast::JSXExpr::Expr(e) => (**e).clone(),
                        ast::JSXExpr::JSXEmptyExpr(_) => ast::Expr::Lit(ast::Lit::Bool(ast::Bool { span: DUMMY_SP, value: true })),
                    },
                    Some(ast::JSXAttrValue::JSXElement(e)) => element_to_call(e),
                    Some(ast::JSXAttrValue::JSXFragment(f)) => fragment_to_call(f),
                };
                props.push(ast::PropOrSpread::Prop(Box::new(ast::Prop::KeyValue(ast::KeyValueProp {
                    key: prop_key(&key),
                    value: Box::new(value),
                }))));
            }
        }
    }
    ast::Expr::Object(ast::ObjectLit { span: DUMMY_SP, props })
}

fn prop_key(name: &str) -> ast::PropName {
    // A plain identifier key when possible, else a string key.
    let ident_ok = !name.is_empty()
        && name.chars().next().map(|c| c.is_alphabetic() || c == '_' || c == '$').unwrap_or(false)
        && name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '$');
    if ident_ok {
        ast::PropName::Ident(ast::IdentName::new(name.into(), DUMMY_SP))
    } else {
        ast::PropName::Str(jstr(name))
    }
}

fn children_to_args(children: &[ast::JSXElementChild]) -> Vec<ast::ExprOrSpread> {
    let mut out = Vec::new();
    for c in children {
        match c {
            ast::JSXElementChild::JSXText(t) => {
                // Collapse JSX whitespace: drop pure-whitespace text, trim the rest.
                let raw = t.value.as_str();
                let trimmed = collapse_jsx_text(raw);
                if !trimmed.is_empty() {
                    out.push(arg(str_lit(&trimmed)));
                }
            }
            ast::JSXElementChild::JSXExprContainer(c) => match &c.expr {
                ast::JSXExpr::Expr(e) => out.push(arg((**e).clone())),
                ast::JSXExpr::JSXEmptyExpr(_) => {}
            },
            ast::JSXElementChild::JSXSpreadChild(s) => {
                out.push(ast::ExprOrSpread { spread: Some(DUMMY_SP), expr: s.expr.clone() });
            }
            ast::JSXElementChild::JSXElement(e) => out.push(arg(element_to_call(e))),
            ast::JSXElementChild::JSXFragment(f) => out.push(arg(fragment_to_call(f))),
        }
    }
    out
}

/// JSX text whitespace handling (simplified): a run of whitespace containing a
/// newline is dropped at the edges; interior whitespace collapses to a space.
fn collapse_jsx_text(s: &str) -> String {
    let has_nl = s.contains('\n');
    let t = if has_nl { s.trim() } else { s };
    // collapse internal whitespace runs to single spaces
    let mut out = String::new();
    let mut in_ws = false;
    for ch in t.chars() {
        if ch.is_whitespace() {
            if !in_ws {
                out.push(' ');
            }
            in_ws = true;
        } else {
            out.push(ch);
            in_ws = false;
        }
    }
    out
}
