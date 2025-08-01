use bumpalo::Bump;
use crate::{Term, TreeTerm, LiteralValue, BracketType, ParseError, BindingInfo, Associativity, Span};
use super::BindingEnv;

pub struct Enforester<'arena> {
    pub arena: &'arena Bump,
    pub env: BindingEnv<'arena>,
}

pub struct EnforestState<'arena> {
    pub combine: Box<dyn Fn(&'arena TreeTerm<'arena>) -> &'arena TreeTerm<'arena> + 'arena>,
    pub precedence: u32,
}

pub type OperatorStack<'arena> = Vec<EnforestState<'arena>>;

impl<'arena> Enforester<'arena> {
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            arena,
            env: BindingEnv::new(arena),
        }
    }

    pub fn enforest(
        &mut self,
        mut terms: Vec<Term>,
        combine: impl Fn(&'arena TreeTerm<'arena>) -> &'arena TreeTerm<'arena> + 'arena + 'static,
        min_precedence: u32,
        stack: &mut OperatorStack<'arena>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        if terms.is_empty() {
            return Err(ParseError::UnexpectedEndOfInput);
        }

        let first = terms.remove(0);
        
        match first {
            Term::Identifier(name, span) => {
                let scopes = self.env.current_scopes();
                
                match self.env.lookup(&name, &scopes) {
                    Some(BindingInfo::Macro(transformer)) => {
                        let expanded = transformer.transform(terms)?;
                        
                        if let Some(combine_state) = stack.pop() {
                            self.enforest(
                                expanded,
                                combine_state.combine,
                                combine_state.precedence,
                                stack,
                            )
                        } else {
                            self.enforest(expanded, combine, min_precedence, stack)
                        }
                    }
                    Some(BindingInfo::Variable(_)) | None => {
                        let name_str = self.arena.alloc_str(&name);
                        let var = self.arena.alloc(TreeTerm::Variable(name_str, scopes));
                        self.continue_enforest(var, terms, combine, min_precedence, stack)
                    }
                    Some(BindingInfo::BinaryOp { .. }) => {
                        Err(ParseError::UnexpectedOperator(name))
                    }
                }
            }
            Term::Literal(lit) => {
                let lit_term = self.arena.alloc(TreeTerm::Literal(lit));
                self.continue_enforest(lit_term, terms, combine, min_precedence, stack)
            }
            Term::Bracket(BracketType::Paren, inner_terms) => {
                if inner_terms.is_empty() {
                    let lit = self.arena.alloc(TreeTerm::Literal(LiteralValue::Nil));
                    self.continue_enforest(lit, terms, combine, min_precedence, stack)
                } else {
                    let (expr, remaining) = self.enforest(
                        inner_terms,
                        |x| x,
                        0,
                        &mut vec![],
                    )?;
                    
                    if !remaining.is_empty() {
                        return Err(ParseError::ExtraTermsInParens);
                    }
                    
                    self.continue_enforest(expr, terms, combine, min_precedence, stack)
                }
            }
            Term::Bracket(BracketType::Curly, inner_terms) => {
                let block = self.arena.alloc(TreeTerm::Block {
                    terms: inner_terms,
                    scope_id: self.env.current_scope,
                });
                self.continue_enforest(block, terms, combine, min_precedence, stack)
            }
            Term::Operator(op, span) => {
                Err(ParseError::UnexpectedOperator(op))
            }
            _ => Err(ParseError::UnexpectedTerm),
        }
    }

    fn continue_enforest(
        &mut self,
        left: &'arena TreeTerm<'arena>,
        mut remaining: Vec<Term>,
        combine: impl Fn(&'arena TreeTerm<'arena>) -> &'arena TreeTerm<'arena> + 'arena + 'static,
        min_precedence: u32,
        stack: &mut OperatorStack<'arena>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        let left = combine(left);

        if let Some(term) = remaining.first() {
            let (op_name, span) = match term {
                Term::Operator(op, span) => (op, *span),
                Term::Identifier(op, span) => (op, *span),
                _ => return Ok((left, remaining)),
            };

            let scopes = self.env.current_scopes();
            
            if let Some(BindingInfo::BinaryOp { precedence, assoc }) = 
                self.env.lookup(op_name, &scopes) {
                
                let (left_prec, right_prec) = match assoc {
                    Associativity::Left => (*precedence, precedence + 1),
                    Associativity::Right => (precedence + 1, *precedence),
                };

                if left_prec < min_precedence {
                    return Ok((left, remaining));
                }

                remaining.remove(0);

                let op_str = self.arena.alloc_str(op_name);
                let captured_left = left;
                let captured_span = span;
                let arena_ref = self.arena;
                
                let new_combine = move |right: &'arena TreeTerm<'arena>| -> &'arena TreeTerm<'arena> {
                    arena_ref.alloc(TreeTerm::BinaryOp {
                        op: op_str,
                        left: captured_left,
                        right,
                        span: captured_span,
                    })
                };

                self.enforest(remaining, new_combine, right_prec, stack)
            } else {
                Ok((left, remaining))
            }
        } else {
            Ok((left, remaining))
        }
    }
}