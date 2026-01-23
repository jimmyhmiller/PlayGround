pub mod constant_fold;

use oxc_allocator::Allocator;
use oxc_ast::ast::Program;

/// A pass transforms a program into another program.
/// The key invariant: the output must have identical semantics to the input.
pub trait Pass {
    /// Apply the transformation, returning true if any changes were made.
    fn run<'a>(&self, allocator: &'a Allocator, program: &mut Program<'a>) -> bool;

    /// Name of this pass for debugging
    fn name(&self) -> &'static str;
}
