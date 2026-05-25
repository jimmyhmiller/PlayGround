//! Real-world adapters for the constrained-language runtime.
//!
//! Each adapter implements the [`runtime::Adapter`] trait synchronously
//! (blocking on I/O for the duration of the call). For a v1 demo agent this
//! is enough; for concurrent workloads see the async-adapter rearchitecture
//! tracked separately.

pub mod fake_llm;
pub mod http_llm;
pub mod simple_tools;
pub mod stdin_lines;
pub mod stdout_notify;

pub use fake_llm::FakeLlmAdapter;
pub use http_llm::{AnthropicLlmAdapter, AnthropicError};
pub use simple_tools::SimpleToolsAdapter;
pub use stdin_lines::StdinLinesGenerator;
pub use stdout_notify::StdoutNotifyAdapter;
