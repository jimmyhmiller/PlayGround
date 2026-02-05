use anyhow::{Context, Result};
use crossterm::cursor::Show;
use crossterm::execute;
use crossterm::terminal::{self, disable_raw_mode, enable_raw_mode};
use std::io::{self, Write};

/// Terminal state that restores on drop
pub struct RawModeGuard {
    was_raw: bool,
}

impl RawModeGuard {
    /// Enter raw mode
    pub fn enter() -> Result<Self> {
        enable_raw_mode().context("Failed to enable raw mode")?;
        Ok(Self { was_raw: true })
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        if self.was_raw {
            let _ = disable_raw_mode();
            // Ensure cursor is visible when exiting
            let _ = execute!(io::stdout(), Show);
        }
    }
}

/// Get current terminal size
pub fn get_size() -> Result<(u16, u16)> {
    let (cols, rows) = terminal::size().context("Failed to get terminal size")?;
    Ok((cols, rows))
}

/// Write directly to stdout (bypassing buffering)
pub fn write_stdout(data: &[u8]) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(data)?;
    stdout.flush()?;
    Ok(())
}
