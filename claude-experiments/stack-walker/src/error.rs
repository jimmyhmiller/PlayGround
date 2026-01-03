use thiserror::Error;

/// Errors that can occur during stack walking
#[derive(Error, Debug, Clone)]
pub enum StackWalkError {
    /// Failed to read memory at the specified address
    #[error("Memory read failed at address 0x{address:016x}")]
    MemoryReadFailed { address: u64 },

    /// Frame pointer is invalid (not monotonically increasing)
    #[error("Invalid frame pointer: 0x{fp:016x} (must be > previous 0x{previous_fp:016x})")]
    InvalidFramePointer { fp: u64, previous_fp: u64 },

    /// Stack pointer is invalid (not monotonically increasing)
    #[error("Invalid stack pointer: 0x{sp:016x} (must be > previous 0x{previous_sp:016x})")]
    InvalidStackPointer { sp: u64, previous_sp: u64 },

    /// Maximum frame count exceeded (possible infinite loop)
    #[error("Maximum frame count ({max}) exceeded - possible infinite loop")]
    MaxFramesExceeded { max: usize },

    /// End of stack reached (not an error, but signals termination)
    #[error("End of stack reached")]
    EndOfStack,
}

/// Result type for stack walking operations
pub type Result<T> = std::result::Result<T, StackWalkError>;
