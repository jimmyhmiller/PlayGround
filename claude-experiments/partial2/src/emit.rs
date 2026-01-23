// Re-export oxc's codegen for now
// We may want custom emit logic later

use oxc_ast::ast::Program;
use oxc_codegen::Codegen;

pub fn emit_js(program: &Program<'_>) -> String {
    Codegen::new().build(program).code
}
