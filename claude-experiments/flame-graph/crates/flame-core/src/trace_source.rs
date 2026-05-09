use crate::builder::ProfileBuilder;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("unrecognized trace format")]
    UnknownFormat,
    #[error("parse error: {0}")]
    Parse(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type LoadResult<T> = std::result::Result<T, LoadError>;

/// Implemented by every format crate. The viewer iterates `&[&dyn TraceSource]`,
/// calling `detect()` until one matches, then `load()`.
pub trait TraceSource: Send + Sync {
    /// Short human-readable name (e.g. "Chrome Trace JSON").
    fn name(&self) -> &'static str;

    /// Cheap content/extension sniff. May read the first few KB of `input`.
    fn detect(&self, input: &[u8], filename: Option<&str>) -> bool;

    /// Parse `input` into the builder.
    fn load(&self, input: &[u8], builder: &mut ProfileBuilder) -> LoadResult<()>;
}
