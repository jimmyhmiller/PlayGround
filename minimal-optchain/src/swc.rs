//! The hand-written **direct** swc transform — the same `?.` desugar coded
//! straight against the swc AST, for comparison with the generic one.

use swc_common::util::take::Take;
use swc_common::{SyntaxContext, DUMMY_SP};
use swc_ecma_ast::{CallExpr, Callee, Expr, ExprOrSpread, Ident, Lit, MemberProp, OptChainBase, Str};
use swc_ecma_visit::{VisitMut, VisitMutWith};

use crate::SwcProgram;

fn str_lit(value: &str) -> Expr {
    Expr::Lit(Lit::Str(Str { span: DUMMY_SP, value: value.into(), raw: None }))
}

fn opt_call(obj: Expr, key: Expr) -> Expr {
    Expr::Call(CallExpr {
        span: DUMMY_SP,
        ctxt: SyntaxContext::empty(),
        callee: Callee::Expr(Box::new(Expr::Ident(Ident::new_no_ctxt("__opt".into(), DUMMY_SP)))),
        args: vec![
            ExprOrSpread { spread: None, expr: Box::new(obj) },
            ExprOrSpread { spread: None, expr: Box::new(key) },
        ],
        type_args: None,
    })
}

pub fn direct(program: &mut SwcProgram) {
    struct D;
    impl VisitMut for D {
        fn visit_mut_expr(&mut self, e: &mut Expr) {
            e.visit_mut_children_with(self);
            let Expr::OptChain(oc) = e else { return };
            let (obj, key) = match &mut *oc.base {
                OptChainBase::Call(_) => panic!("optional call not supported"),
                OptChainBase::Member(m) => {
                    let obj = *m.obj.take();
                    let key = match &mut m.prop {
                        MemberProp::Ident(id) => str_lit(id.sym.as_str()),
                        MemberProp::Computed(c) => *c.expr.take(),
                        MemberProp::PrivateName(_) => panic!("private not supported"),
                    };
                    (obj, key)
                }
            };
            *e = opt_call(obj, key);
        }
    }
    program.program.visit_mut_with(&mut D);
}
