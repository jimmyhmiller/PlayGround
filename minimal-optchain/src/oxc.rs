//! The hand-written **direct** oxc transform — the same `?.` desugar coded
//! straight against the oxc AST, for comparison with the generic one.
//!
//! Note: this module is named `oxc`, which shadows the `oxc` crate, so we reach
//! the crate through a leading `::oxc`.

use ::oxc::allocator::{Dummy, FromIn, GetAllocator, Vec as ArenaVec};
use ::oxc::ast::ast::{Argument, ChainElement, Expression, Program};
use ::oxc::ast::AstBuilder;
use ::oxc::ast::NONE;
use ::oxc::ast_visit::{walk_mut, VisitMut};
use ::oxc::span::SPAN;

pub fn direct<'a>(program: &mut Program<'a>, cx: AstBuilder<'a>) {
    struct D<'a> {
        cx: AstBuilder<'a>,
    }
    enum Key<'a> {
        Static(String),
        Computed(Expression<'a>),
    }
    impl<'a> D<'a> {
        fn take(&self, slot: &mut Expression<'a>) -> Expression<'a> {
            std::mem::replace(slot, Expression::dummy(self.cx.allocator()))
        }
        fn opt_call(&self, obj: Expression<'a>, key: Expression<'a>) -> Expression<'a> {
            let callee = self.cx.expression_identifier(SPAN, oxc_str::Ident::from_in("__opt", self.cx.allocator()));
            let mut args: ArenaVec<'a, Argument<'a>> = ArenaVec::new_in(&self.cx);
            args.push(Argument::from(obj));
            args.push(Argument::from(key));
            self.cx.expression_call(SPAN, callee, NONE, args, false)
        }
        fn rewrite(&self, e: &mut Expression<'a>) -> Option<Expression<'a>> {
            let (obj, key) = match e {
                Expression::ChainExpression(chain) => match &mut chain.expression {
                    ChainElement::StaticMemberExpression(m) => {
                        (self.take(&mut m.object), Key::Static(m.property.name.as_str().to_string()))
                    }
                    ChainElement::ComputedMemberExpression(m) => {
                        (self.take(&mut m.object), Key::Computed(self.take(&mut m.expression)))
                    }
                    ChainElement::CallExpression(_) => panic!("optional call not supported"),
                    ChainElement::PrivateFieldExpression(_) => panic!("private not supported"),
                    ChainElement::TSNonNullExpression(_) => return None,
                },
                Expression::StaticMemberExpression(m) if m.optional => {
                    (self.take(&mut m.object), Key::Static(m.property.name.as_str().to_string()))
                }
                Expression::ComputedMemberExpression(m) if m.optional => {
                    (self.take(&mut m.object), Key::Computed(self.take(&mut m.expression)))
                }
                Expression::CallExpression(c) if c.optional => panic!("optional call not supported"),
                Expression::PrivateFieldExpression(m) if m.optional => {
                    let _ = m;
                    panic!("private not supported")
                }
                _ => return None,
            };
            let key = match key {
                Key::Static(s) => self.cx.expression_string_literal(SPAN, oxc_str::Str::from_in(s.as_str(), self.cx.allocator()), None),
                Key::Computed(k) => k,
            };
            Some(self.opt_call(obj, key))
        }
    }
    impl<'a> VisitMut<'a> for D<'a> {
        fn visit_expression(&mut self, e: &mut Expression<'a>) {
            walk_mut::walk_expression(self, e);
            if let Some(new) = self.rewrite(e) {
                *e = new;
            }
        }
    }
    D { cx }.visit_program(program);
}
