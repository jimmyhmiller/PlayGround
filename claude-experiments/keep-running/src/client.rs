use crate::protocol::{decode_message, encode_message, ClientMessage, DaemonMessage};
use crate::session::{self, SessionInfo};
use crate::terminal::{self, RawModeGuard};
use anyhow::{Context, Result};
use crossterm::tty::IsTty;
use crossterm::{execute, terminal::{Clear, ClearType}, cursor::MoveTo};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Detach sequence: Ctrl+a followed by 'd'
const CTRL_A: u8 = 0x01;

/// Connect to a session and run the interactive client
pub fn attach(session: &SessionInfo) -> Result<()> {
    // Check if we have a terminal
    if !io::stdin().is_tty() {
        anyhow::bail!("stdin is not a terminal - cannot attach to session");
    }

    let socket_path = session::socket_path(&session.name)?;
    let mut stream =
        UnixStream::connect(&socket_path).context("Failed to connect to session daemon")?;

    // Set read timeout for handshake
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_nonblocking(false)?;

    // Get terminal size
    let (cols, rows) = terminal::get_size()?;

    // Send attach message
    let msg = ClientMessage::Attach { cols, rows };
    let encoded = encode_message(&msg)?;
    stream.write_all(&encoded)?;

    // Wait for attached confirmation (blocking with timeout)
    let mut buf = [0u8; 8192];
    let n = stream.read(&mut buf).context("Failed to read attach confirmation")?;
    if n == 0 {
        anyhow::bail!("Connection closed while waiting for attach confirmation");
    }
    let mut msg_buf = buf[..n].to_vec();

    loop {
        if let Some((msg, consumed)) = decode_message::<DaemonMessage>(&msg_buf)? {
            msg_buf.drain(0..consumed);
            match msg {
                DaemonMessage::Attached => break,
                DaemonMessage::Error(e) => anyhow::bail!("Daemon error: {}", e),
                _ => {}
            }
        } else {
            let n = stream.read(&mut buf).context("Failed to read from daemon")?;
            if n == 0 {
                anyhow::bail!("Connection closed while waiting for attach confirmation");
            }
            msg_buf.extend_from_slice(&buf[..n]);
        }
    }

    // Clear timeout and switch to non-blocking for main loop
    stream.set_read_timeout(None)?;
    stream.set_nonblocking(true)?;

    // Clear screen and move cursor to top-left before showing session content
    execute!(io::stdout(), Clear(ClearType::All), MoveTo(0, 0))?;

    // Enter raw mode
    let _raw_guard = RawModeGuard::enter()?;

    // Set up signal handling for SIGWINCH (terminal resize)
    let resize_flag = Arc::new(AtomicBool::new(false));
    let resize_flag_clone = resize_flag.clone();

    // Install SIGWINCH handler
    unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGWINCH, move || {
            resize_flag_clone.store(true, Ordering::SeqCst);
        })?;
    }

    // Main I/O loop
    let mut input_buf = [0u8; 1024];
    let mut daemon_buf = [0u8; 8192];
    let mut daemon_msg_buf = msg_buf; // May have leftover data

    // State for detecting Ctrl+a d
    let mut saw_ctrl_a = false;

    // Get file descriptors for polling
    let stdin_fd = 0i32;
    let socket_fd = stream.as_raw_fd();

    loop {
        // Check for resize
        if resize_flag.swap(false, Ordering::SeqCst) {
            if let Ok((cols, rows)) = terminal::get_size() {
                let msg = ClientMessage::Resize { cols, rows };
                if let Ok(encoded) = encode_message(&msg) {
                    let _ = stream.write_all(&encoded);
                }
            }
        }

        // Use poll to check for data on stdin and socket
        let mut poll_fds = [
            libc::pollfd { fd: stdin_fd, events: libc::POLLIN, revents: 0 },
            libc::pollfd { fd: socket_fd, events: libc::POLLIN, revents: 0 },
        ];

        let poll_result = unsafe { libc::poll(poll_fds.as_mut_ptr(), 2, 10) }; // 10ms timeout

        if poll_result < 0 {
            let err = io::Error::last_os_error();
            if err.kind() != io::ErrorKind::Interrupted {
                return Err(err).context("poll failed");
            }
            continue;
        }

        // Check stdin
        if poll_fds[0].revents & libc::POLLIN != 0 {
            let n = unsafe {
                libc::read(stdin_fd, input_buf.as_mut_ptr() as *mut libc::c_void, input_buf.len())
            };

            if n == 0 {
                // EOF on stdin
                break;
            } else if n > 0 {
                let data = &input_buf[..n as usize];

                // Check for detach/kill sequence (Ctrl+a then d/k)
                // Batch regular bytes together for efficiency (critical for paste)
                let mut i = 0;
                while i < data.len() {
                    if saw_ctrl_a {
                        saw_ctrl_a = false;
                        let byte = data[i];
                        i += 1;

                        match byte {
                            b'd' | b'D' => {
                                // Detach!
                                let msg = ClientMessage::Detach;
                                if let Ok(encoded) = encode_message(&msg) {
                                    let _ = stream.write_all(&encoded);
                                }
                                eprintln!("\r\n[detached from {}]", session.name);
                                return Ok(());
                            }
                            b'k' | b'K' => {
                                // Kill session!
                                eprintln!("\r\n[killing session {}]", session.name);
                                unsafe {
                                    libc::kill(session.pid as i32, libc::SIGHUP);
                                }
                                return Ok(());
                            }
                            CTRL_A => {
                                // Double Ctrl+a - send a literal Ctrl+a
                                let msg = ClientMessage::Input(vec![CTRL_A]);
                                if let Ok(encoded) = encode_message(&msg) {
                                    let _ = stream.write_all(&encoded);
                                }
                            }
                            _ => {
                                // Not a command, send the Ctrl+a we held back, plus this byte
                                let msg = ClientMessage::Input(vec![CTRL_A, byte]);
                                if let Ok(encoded) = encode_message(&msg) {
                                    let _ = stream.write_all(&encoded);
                                }
                            }
                        }
                    } else {
                        // Find the next Ctrl+A or end of data
                        let start = i;
                        while i < data.len() && data[i] != CTRL_A {
                            i += 1;
                        }

                        // Send batch of regular bytes as a single message
                        if i > start {
                            let msg = ClientMessage::Input(data[start..i].to_vec());
                            if let Ok(encoded) = encode_message(&msg) {
                                let _ = stream.write_all(&encoded);
                            }
                        }

                        // If we stopped at Ctrl+A, consume it and set flag
                        if i < data.len() && data[i] == CTRL_A {
                            saw_ctrl_a = true;
                            i += 1;
                        }
                    }
                }
            }
        }

        // Check socket
        if poll_fds[1].revents & libc::POLLIN != 0 {
            match stream.read(&mut daemon_buf) {
                Ok(0) => {
                    // Daemon disconnected
                    eprintln!("\r\n[session ended]");
                    break;
                }
                Ok(n) => {
                    daemon_msg_buf.extend_from_slice(&daemon_buf[..n]);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => {
                    return Err(e).context("Error reading from daemon");
                }
            }
        }

        // Check for hangup on socket
        if poll_fds[1].revents & (libc::POLLHUP | libc::POLLERR) != 0 {
            eprintln!("\r\n[session ended]");
            break;
        }

        // Process any messages in the buffer
        loop {
            match decode_message::<DaemonMessage>(&daemon_msg_buf) {
                Ok(Some((msg, consumed))) => {
                    daemon_msg_buf.drain(0..consumed);

                    match msg {
                        DaemonMessage::Output(data) => {
                            terminal::write_stdout(&data)?;
                        }
                        DaemonMessage::ChildExited { code } => {
                            if let Some(c) = code {
                                eprintln!("\r\n[process exited with code {}]", c);
                            } else {
                                eprintln!("\r\n[process terminated by signal]");
                            }
                            return Ok(());
                        }
                        DaemonMessage::Error(e) => {
                            eprintln!("\r\n[daemon error: {}]", e);
                            return Ok(());
                        }
                        DaemonMessage::Attached => {
                            // Already handled
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("\r\n[protocol error: {}]", e);
                    break;
                }
            }
        }
    }

    Ok(())
}

/// Start a new session and immediately attach to it
pub fn run_and_attach(name: &str, command: &[String]) -> Result<()> {
    // Start the daemon
    crate::daemon::start_daemon(name.to_string(), command.to_vec())?;

    // Wait for session to be available
    std::thread::sleep(Duration::from_millis(100));

    // Load session info
    let session = session::load_session(name)?
        .context("Session not found after starting daemon")?;

    // Attach
    attach(&session)
}
