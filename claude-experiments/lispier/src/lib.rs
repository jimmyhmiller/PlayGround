pub mod ast;
pub mod dialect;
pub mod ir_gen;
pub mod jit;
pub mod macro_compiler;
pub mod macros;
pub mod module_loader;
pub mod namespace;
pub mod parser;
pub mod reader;
pub mod runtime;
pub mod token;
pub mod tokenizer;
pub mod value;
pub mod value_ffi;

pub use ast::{
    AttributeValue, Binding, Block, BlockArgument, Compilation, Defmacro, Extern, FunctionType,
    LetExpr, LinkLibrary, Module, Node, Operation, Pass, Region, Require, RequireMacros, Target,
    Type, TypeAnnotation, TypedNumber,
};
pub use module_loader::{find_project_root, ModuleError, ModuleLoader};
pub use dialect::DialectRegistry;
pub use ir_gen::{GeneratorError, IRGenerator, SymbolTable};
pub use jit::{Jit, JitError};
pub use namespace::{Namespace, NamespaceScope};
pub use parser::{Parser, ParserError};
pub use reader::{Reader, ReaderError};
pub use runtime::{
    extract_compilation, extract_defmacros, extract_externs, extract_link_libraries, Backend,
    RuntimeEnv, RuntimeError,
};
pub use token::{Token, TokenType};
pub use tokenizer::{Tokenizer, TokenizerError};
pub use value::{Symbol, Value};
pub use macro_compiler::{CompiledMacros, MacroCompiler, MacroCompilerError};
pub use macros::{JitMacro, JitMacroFn, Macro, MacroError, MacroExpander, MacroRegistry};
pub use value_ffi::{get_value_ffi_functions, FfiFunction};
