//! Minimal pseudo-terminal wrapper.
//!
//! Ported from ghostling_rs/src/main.rs (libghostty-rs example). Spawns
//! the user's shell on a pty pair. Reads + parsing happen on the
//! per-terminal worker thread (see `worker.rs`); this struct is just
//! the fd handle + spawn / write / resize plumbing.

#![allow(unsafe_code)]

use std::{
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::process::CommandExt,
    },
    path::PathBuf,
    process::Command,
};

use nix::{
    errno::Errno,
    fcntl::{self, OFlag},
    pty::ForkptyResult,
    sys::{signal, wait},
    unistd::{self, Pid},
};

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
pub struct Pty(pub OwnedFd);

impl Pty {
    /// `forkpty()` + exec the user's shell. `$SHELL` → passwd entry → `/bin/sh`.
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
                Ok((Self(fd), Child::Active(child)))
            }
        }
    }

    /// Set the master fd to non-blocking. The worker thread polls it
    /// in a tight loop and uses `EAGAIN` to detect "no more data right
    /// now"; keystroke writes from the worker remain non-blocking but
    /// almost never produce `EAGAIN` because we write tiny payloads.
    pub fn set_nonblock(&self) -> std::io::Result<()> {
        let raw_flags = fcntl::fcntl(&self.0, fcntl::F_GETFL)?;
        let flags = OFlag::from_bits_retain(raw_flags) | OFlag::O_NONBLOCK;
        fcntl::fcntl(&self.0, fcntl::F_SETFL(flags))?;
        Ok(())
    }

    /// Best-effort write. Drops remaining bytes on `EAGAIN` / fatal
    /// errors — matches ghostling's policy under back-pressure.
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
