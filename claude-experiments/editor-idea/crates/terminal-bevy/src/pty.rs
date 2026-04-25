//! Minimal pseudo-terminal wrapper.
//!
//! Ported from ghostling_rs/src/main.rs (libghostty-rs example). Spawns
//! the user's shell on a pty pair, sets the master fd non-blocking so
//! Bevy systems can drain it once per frame without stalling.

#![allow(unsafe_code)]

use std::{
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::process::CommandExt,
    },
    path::PathBuf,
    process::Command,
};

use libghostty_vt::Terminal;
use nix::{
    errno::Errno,
    fcntl::{self, OFlag},
    pty::ForkptyResult,
    sys::{signal, wait},
    unistd::{self, Pid},
};

/// Pixel-aware grid dimensions used to size the pty winsize struct.
#[derive(Clone, Copy, Debug)]
pub struct PtySize {
    pub cols: u16,
    pub rows: u16,
    pub cell_width_px: u16,
    pub cell_height_px: u16,
}

impl PtySize {
    fn to_winsize(self) -> nix::pty::Winsize {
        nix::pty::Winsize {
            ws_col: self.cols,
            ws_row: self.rows,
            ws_xpixel: self.cols.saturating_mul(self.cell_width_px),
            ws_ypixel: self.rows.saturating_mul(self.cell_height_px),
        }
    }
}

/// Owns the pty master file descriptor.
#[derive(Debug)]
pub struct Pty(OwnedFd);

impl Pty {
    /// `forkpty()` + exec the user's shell; returns the parent's master
    /// fd and the child's pid. `$SHELL` → passwd entry → `/bin/sh`.
    pub fn spawn(size: PtySize) -> std::io::Result<(Self, Child)> {
        match unsafe { nix::pty::forkpty(&size.to_winsize(), None)? } {
            ForkptyResult::Child => {
                let shell = match std::env::var_os("SHELL") {
                    Some(s) if !s.is_empty() => PathBuf::from(s),
                    _ => match unistd::User::from_uid(unistd::getuid()) {
                        Ok(Some(user)) => user.shell,
                        _ => PathBuf::from("/bin/sh"),
                    },
                };
                let arg0 = shell.file_name().unwrap_or(shell.as_os_str());
                let _ = Command::new(&shell)
                    .env("TERM", "xterm-256color")
                    .arg0(arg0)
                    .exec();
                std::process::exit(127);
            }
            ForkptyResult::Parent { child, master: fd } => {
                let raw_flags = fcntl::fcntl(&fd, fcntl::F_GETFL)?;
                let flags = OFlag::from_bits_retain(raw_flags) | OFlag::O_NONBLOCK;
                let _ = fcntl::fcntl(&fd, fcntl::F_SETFL(flags))?;
                Ok((Self(fd), Child::Active(child)))
            }
        }
    }

    /// Drain available bytes from the master fd into the terminal's VT
    /// parser. Returns `EndOfStream` on EOF / EIO so callers can transition
    /// the child state.
    ///
    /// Buffer sized to 64 KiB — bigger than the default macOS pty buffer,
    /// so `cat`-ing a large file typically drains in one or two syscalls
    /// per frame instead of dozens.
    pub fn read_into(&self, term: &mut Terminal) -> Result<(), PtyError> {
        let mut buf = [0u8; 65536];
        loop {
            match nix::unistd::read(&self.0, &mut buf) {
                Ok(0) => return Err(PtyError::EndOfStream),
                Ok(len) => term.vt_write(&buf[..len]),
                Err(Errno::EAGAIN) => return Ok(()),
                Err(Errno::EINTR) => continue,
                Err(Errno::EIO) => return Err(PtyError::EndOfStream),
                Err(err) => return Err(PtyError::Other(err)),
            }
        }
    }

    /// Best-effort write. Drops remaining bytes on `EAGAIN` / fatal errors —
    /// matches ghostling's policy under back-pressure.
    pub fn write(&self, data: &[u8]) {
        let mut remaining = data;
        while !remaining.is_empty() {
            match nix::unistd::write(&self.0, remaining) {
                Ok(len) => remaining = &remaining[len..],
                Err(Errno::EINTR) => continue,
                Err(_) => break,
            }
        }
    }

    pub fn resize(&self, size: PtySize) {
        nix::ioctl_write_ptr_bad!(tiocswinsz, nix::libc::TIOCSWINSZ, nix::pty::Winsize);
        let _ = unsafe { tiocswinsz(self.0.as_raw_fd(), &size.to_winsize()) };
    }
}

#[derive(Debug)]
pub enum Child {
    Active(Pid),
    Exited(Pid),
    Reaped(wait::WaitStatus),
}

impl Drop for Child {
    fn drop(&mut self) {
        match *self {
            Child::Active(pid) => {
                let _ = signal::kill(pid, signal::SIGHUP);
                let _ = wait::waitpid(pid, None);
            }
            Child::Exited(pid) => {
                let _ = wait::waitpid(pid, None);
            }
            Child::Reaped(_) => {}
        }
    }
}

#[derive(Clone, Debug)]
pub enum PtyError {
    EndOfStream,
    Other(Errno),
}
