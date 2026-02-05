use std::fmt;
use std::io;

#[allow(dead_code)]
#[derive(Debug)]
pub enum KeepError {
    SessionNotFound(String),
    SessionExists(String),
    NoSessions,
    ConnectionFailed(String),
    Protocol(String),
    Io(io::Error),
    ChildExited(i32),
    ChildSignaled,
    Other(String),
}

impl fmt::Display for KeepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeepError::SessionNotFound(s) => write!(f, "Session not found: {}", s),
            KeepError::SessionExists(s) => write!(f, "Session already exists: {}", s),
            KeepError::NoSessions => write!(f, "No sessions running"),
            KeepError::ConnectionFailed(s) => write!(f, "Failed to connect to session: {}", s),
            KeepError::Protocol(s) => write!(f, "Protocol error: {}", s),
            KeepError::Io(e) => write!(f, "IO error: {}", e),
            KeepError::ChildExited(code) => write!(f, "Child process exited with code: {}", code),
            KeepError::ChildSignaled => write!(f, "Child process killed by signal"),
            KeepError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for KeepError {}

impl From<io::Error> for KeepError {
    fn from(err: io::Error) -> Self {
        KeepError::Io(err)
    }
}

#[allow(dead_code)]
pub type Result<T> = std::result::Result<T, KeepError>;
