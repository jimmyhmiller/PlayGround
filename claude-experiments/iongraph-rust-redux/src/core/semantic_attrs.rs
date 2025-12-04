/// Universal semantic attributes for blocks (compiler-independent)
///
/// Different compilers use different attribute names, but the semantics are universal.
/// For example:
/// - Ion uses "loopheader", LLVM might use "loop.header"
/// - Ion uses "backedge", LLVM might use "loop.latch"
///
/// This enum provides a common vocabulary that all compilers can map to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticAttribute {
    /// Block is the header of a loop
    LoopHeader,
    /// Block is a backedge (jumps backward to loop header)
    Backedge,
    /// Block was created by edge splitting
    SplitEdge,
    /// Block is function entry point
    Entry,
    /// Block is unreachable
    Unreachable,
    /// Custom attribute (compiler-specific, no semantic meaning)
    Custom,
}

/// Extension trait for converting compiler-specific attribute strings to semantic meaning
pub trait AttributeSemantics {
    /// Map compiler-specific attribute strings to semantic meaning
    fn parse_attribute(attr: &str) -> SemanticAttribute;

    /// Check if a list of attributes contains a specific semantic attribute
    fn has_semantic_attribute(attributes: &[String], semantic: SemanticAttribute) -> bool {
        attributes.iter().any(|attr| Self::parse_attribute(attr) == semantic)
    }
}
