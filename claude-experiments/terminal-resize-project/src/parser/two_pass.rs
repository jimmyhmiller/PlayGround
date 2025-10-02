use crate::{Term, TreeTerm, AST, ScopeId, BindingId, BindingInfo, ParseError};
use super::{Parser, BindingEnv};

impl<'arena> Parser<'arena> {
    pub fn parse1(
        &mut self,
        mut terms: Vec<Term>,
        scope: ScopeId,
    ) -> Result<(Vec<&'arena TreeTerm<'arena>>, BindingEnv<'arena>), ParseError> {
        let mut results = Vec::new();
        let original_scope = self.enforester.env.current_scope;
        self.enforester.env.current_scope = scope;

        while !terms.is_empty() {
            let (tree_term, remaining) = self.enforester.enforest(
                terms,
                |x| x,
                0,
                &mut vec![],
            )?;

            match tree_term {
                TreeTerm::MacroCall { name, args, .. } if name == "macro" => {
                    // For simplicity, we'll skip macro definition registration in this implementation
                    results.push(tree_term);
                }
                TreeTerm::Function { name, params, body } => {
                    let binding_id = self.enforester.env.new_binding();
                    let scopes = self.enforester.env.current_scopes();
                    self.enforester.env.bind(
                        name.to_string(),
                        scopes,
                        BindingInfo::Variable(binding_id),
                    );
                    results.push(tree_term);
                }
                TreeTerm::BinaryOp { op: "=", left, right, .. } => {
                    if let TreeTerm::Variable(name, _) = left {
                        let binding_id = self.enforester.env.new_binding();
                        let scopes = self.enforester.env.current_scopes();
                        self.enforester.env.bind(
                            name.to_string(),
                            scopes,
                            BindingInfo::Variable(binding_id),
                        );
                    }
                    results.push(tree_term);
                }
                _ => {
                    results.push(tree_term);
                }
            }

            terms = remaining;
        }

        self.enforester.env.current_scope = original_scope;
        Ok((results, self.enforester.env.clone()))
    }

    pub fn parse2(
        &mut self,
        tree_terms: Vec<&'arena TreeTerm<'arena>>,
        bindings: BindingEnv<'arena>,
    ) -> Result<Vec<&'arena AST<'arena>>, ParseError> {
        self.enforester.env = bindings;
        
        tree_terms.into_iter()
            .map(|tree_term| self.complete_parse(tree_term))
            .collect()
    }

    pub fn complete_parse(&mut self, tree_term: &'arena TreeTerm<'arena>) -> Result<&'arena AST<'arena>, ParseError> {
        match tree_term {
            TreeTerm::Literal(lit) => {
                Ok(self.arena.alloc(AST::Literal(lit.clone())))
            }
            TreeTerm::Variable(name, scopes) => {
                match self.enforester.env.lookup(name, scopes) {
                    Some(BindingInfo::Variable(binding_id)) => {
                        Ok(self.arena.alloc(AST::ResolvedVar {
                            name,
                            binding_id: *binding_id,
                        }))
                    }
                    _ => Err(ParseError::UnboundVariable(name.to_string())),
                }
            }
            TreeTerm::Block { terms, scope_id } => {
                let block_scope = self.enforester.env.new_scope();
                let parsed_terms = self.parse(terms.clone())?;
                Ok(self.arena.alloc(AST::Block(parsed_terms)))
            }
            TreeTerm::Function { name, params, body } => {
                let body_scope = self.enforester.env.new_scope();
                let original_scope = self.enforester.env.current_scope;
                self.enforester.env.current_scope = body_scope;

                let param_bindings: Vec<_> = params.iter()
                    .map(|param| {
                        let binding_id = self.enforester.env.new_binding();
                        let scopes = self.enforester.env.current_scopes();
                        self.enforester.env.bind(
                            param.to_string(),
                            scopes,
                            BindingInfo::Variable(binding_id),
                        );
                        binding_id
                    })
                    .collect();

                let body_ast = self.complete_parse(body)?;
                
                self.enforester.env.current_scope = original_scope;

                Ok(self.arena.alloc(AST::FunctionDef {
                    binding_id: BindingId(0),
                    params: param_bindings,
                    body: body_ast,
                }))
            }
            TreeTerm::BinaryOp { op, left, right, .. } => {
                let left_ast = self.complete_parse(left)?;
                let right_ast = self.complete_parse(right)?;
                
                Ok(self.arena.alloc(AST::FunctionCall {
                    func: self.arena.alloc(AST::ResolvedVar {
                        name: op,
                        binding_id: BindingId(0),
                    }),
                    args: vec![left_ast, right_ast],
                }))
            }
            TreeTerm::If { condition, then_branch, else_branch } => {
                let condition_ast = self.complete_parse(condition)?;
                let then_ast = self.complete_parse(then_branch)?;
                let else_ast = if let Some(else_branch) = else_branch {
                    Some(self.complete_parse(else_branch)?)
                } else {
                    None
                };
                
                Ok(self.arena.alloc(AST::If {
                    condition: condition_ast,
                    then_branch: then_ast,
                    else_branch: else_ast,
                }))
            }
            TreeTerm::Call { func, args } => {
                let func_ast = self.complete_parse(func)?;
                let arg_asts: Result<Vec<_>, _> = args.iter()
                    .map(|arg| self.complete_parse(arg))
                    .collect();
                
                Ok(self.arena.alloc(AST::FunctionCall {
                    func: func_ast,
                    args: arg_asts?,
                }))
            }
            _ => Err(ParseError::UnexpectedTreeTerm),
        }
    }
}