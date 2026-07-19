//! The transform — one generic function, no backend-specific code. It matches
//! the [`OptChain`](crate::ExprRefMut::OptChain) arm of the *general* projection
//! [`Backend::expr_ref_mut`]; the same projection also exposes `Bin`, `Nullish`,
//! `Member`, `Paren`, so other transforms reuse it unchanged.

use crate::{Backend, ExprRefMut, MemberKeyMut, OptChainBaseMut};

/// Desugar every optional-**member** chain in `program` into `__opt(...)` calls,
/// in place. Post-order (so an inner link is already an `__opt(...)` call by the
/// time its parent is rebuilt); the object is *moved* in, so each base is
/// evaluated once.
#[inline]
pub fn desugar<'a, B: Backend + 'a>(cx: B::Ctx<'a>, program: &mut B::Program<'a>) {
    B::visit_exprs_mut(program, |expr| {
        let rebuilt = match B::expr_ref_mut(expr) {
            ExprRefMut::OptChain { base: OptChainBaseMut::Member { object, key }, .. } => {
                let object = B::take_expr(cx, object);
                let key = match key {
                    MemberKeyMut::Static(name) => B::string(cx, name),
                    MemberKeyMut::Computed(slot) => B::take_expr(cx, slot),
                    MemberKeyMut::Private(_) => panic!("optional private-field `?.#x` not supported"),
                };
                Some(B::call(cx, B::ident(cx, "__opt"), vec![object, key]))
            }
            ExprRefMut::OptChain { base: OptChainBaseMut::Call, .. } => {
                panic!("optional call `?.()` not supported")
            }
            _ => None,
        };
        if let Some(new) = rebuilt {
            *expr = new;
        }
    });
}
