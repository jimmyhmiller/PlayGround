use crate::ast::types::{FunctionSig, Parameter, StructDef, Type};
use crate::ast::expr::Stmt;

/// A function definition
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub sig: FunctionSig,
    pub body: Vec<Stmt>,
}

impl FunctionDef {
    pub fn new(
        name: impl Into<String>,
        params: Vec<Parameter>,
        return_type: Type,
        body: Vec<Stmt>,
    ) -> Self {
        Self {
            sig: FunctionSig {
                name: name.into(),
                params,
                return_type,
            },
            body,
        }
    }

    pub fn name(&self) -> &str {
        &self.sig.name
    }

    pub fn params(&self) -> &[Parameter] {
        &self.sig.params
    }

    pub fn return_type(&self) -> &Type {
        &self.sig.return_type
    }
}

/// A complete program
#[derive(Debug, Clone)]
pub struct Program {
    pub structs: Vec<StructDef>,
    pub functions: Vec<FunctionDef>,
}

impl Program {
    pub fn new(structs: Vec<StructDef>, functions: Vec<FunctionDef>) -> Self {
        Self { structs, functions }
    }

    pub fn find_struct(&self, name: &str) -> Option<&StructDef> {
        self.structs.iter().find(|s| s.name == name)
    }

    pub fn find_function(&self, name: &str) -> Option<&FunctionDef> {
        self.functions.iter().find(|f| f.sig.name == name)
    }
}
