//! Live-editing across an FFI boundary — the GUI story in miniature.
//!
//! A tiny native "toolkit" (a stand-in for a windowing library) runs its own
//! event loop on its own thread, calling *up* into managed code once per frame
//! through the late-binding trampoline. Meanwhile another thread edits the
//! managed draw callback. The running loop switches to the new code mid-flight,
//! and the native window — created once by a `letonce` — is never reopened.
//!
//! It exercises every piece of the FFI model:
//!   * `foreign type Window` — an opaque handle the GC never traces.
//!   * `foreign fn open_window / draw` — the managed → native direction, with
//!     the handle pinned in a frame slot across the (uninterruptible) call.
//!   * `letonce win = open_window()` — native resource created once, surviving
//!     every hot edit (state persists, code reloads).
//!   * `Session::call("on_frame")` — the native → managed trampoline, resolving
//!     the *current* version of the callback each frame (late binding).
//!
//! Run with: `cargo run --bin livetype-ffi-gui`

use livetype::*;
use livetype_core::Session;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// The native side: our fake windowing library's state. Shared (behind a lock)
/// between the foreign functions the runtime calls and the harness that checks
/// what happened. In a real binding this would be `GLFWwindow*`s and a GL
/// context; here it is a counter and a log of what got drawn.
#[derive(Default)]
struct Toolkit {
    next_handle: u64,
    windows_opened: u64,
    /// (window handle, value drawn) — the "framebuffer" history.
    drawn: Vec<(u64, i64)>,
}

enum Cmd {
    Edit(String),
    Stop,
}

fn main() {
    println!("\n■ live-editing a draw callback across an FFI boundary\n");

    let toolkit = Arc::new(Mutex::new(Toolkit::default()));
    let mut session = Session::new();

    // ── 1. Load the FFI "header": declare the native type and functions. This
    //       assigns ids but runs nothing yet. ────────────────────────────────
    session
        .eval(
            r#"
            foreign type Window;
            foreign fn open_window() -> Window;
            foreign fn draw(w: Window, n: i64) -> ();
        "#,
        )
        .expect("declare foreign interface");
    println!("   · declared: foreign type Window; open_window(); draw(w, n)");

    // ── 2. Bind the native implementations (the host wires up the FFI). ──────
    let window_kind = session
        .foreign_kind("Window")
        .expect("Window kind assigned at declaration");
    {
        let tk = Arc::clone(&toolkit);
        session
            .register_foreign(
                "open_window",
                Box::new(move |_args| {
                    let mut t = tk.lock().unwrap();
                    t.next_handle += 1;
                    t.windows_opened += 1;
                    let ptr = t.next_handle;
                    println!("   · [native] open_window() -> window #{ptr}");
                    Value::Foreign { kind: window_kind, ptr }
                }),
            )
            .unwrap();
    }
    {
        let tk = Arc::clone(&toolkit);
        session
            .register_foreign(
                "draw",
                Box::new(move |args| {
                    let (Value::Foreign { ptr, .. }, Value::I64(n)) = (&args[0], &args[1]) else {
                        panic!("draw called with the wrong argument shapes");
                    };
                    tk.lock().unwrap().drawn.push((*ptr, *n));
                    Value::Unit
                }),
            )
            .unwrap();
    }
    println!("   · bound native open_window / draw");

    // ── 3. Load the program. The `letonce` runs open_window ONCE, storing the
    //       handle in a global that will survive every later edit. ────────────
    session
        .eval(
            r#"
            letonce win = open_window();
            fn on_frame() -> i64 {
                draw(win, 42);
                42
            }
        "#,
        )
        .expect("install program");
    println!("   · installed on_frame (draws 42)\n");

    // ── 4. Hand the session to the native event loop on its own thread. ──────
    let (tx, rx): (Sender<Cmd>, Receiver<Cmd>) = channel();
    let handle = {
        let toolkit = Arc::clone(&toolkit);
        thread::spawn(move || event_loop(session, rx, toolkit))
    };

    // ── 5. The "editor": let a few frames render, then edit the callback while
    //       the loop keeps running, then stop. ─────────────────────────────────
    thread::sleep(Duration::from_millis(600));
    println!("\n   ✎ LIVE EDIT (from this thread; the loop keeps running):");
    println!("       fn on_frame() -> i64 {{ draw(win, 84); 42 }}\n");
    tx.send(Cmd::Edit(
        "fn on_frame() -> i64 { draw(win, 84); 42 }".into(),
    ))
    .unwrap();

    thread::sleep(Duration::from_millis(600));
    tx.send(Cmd::Stop).unwrap();
    let _session = handle.join().unwrap();

    // ── 6. Verdict. ──────────────────────────────────────────────────────────
    let t = toolkit.lock().unwrap();
    let handles: Vec<u64> = {
        let mut h: Vec<u64> = t.drawn.iter().map(|(w, _)| *w).collect();
        h.dedup();
        h
    };
    let drew_42 = t.drawn.iter().any(|(_, n)| *n == 42);
    let drew_84 = t.drawn.iter().any(|(_, n)| *n == 84);
    let one_window = t.windows_opened == 1 && handles.len() == 1;
    println!(
        "\n   windows opened: {}   distinct handles drawn to: {:?}",
        t.windows_opened, handles
    );
    println!(
        "   drew 42 (before edit): {drew_42}    drew 84 (after edit): {drew_84}"
    );
    let ok = drew_42 && drew_84 && one_window;
    println!(
        "\n{} the callback hot-swapped 42 → 84 while the SAME window (#{}) kept \n  rendering — edited across the FFI boundary, native resource never reopened.\n",
        if ok { "✓" } else { "✗" },
        handles.first().copied().unwrap_or(0),
    );
}

/// The native toolkit's event loop: once per frame, call up into the current
/// `on_frame` through the trampoline. Between frames, drain and apply any edits
/// — an edit lands between two frames of a genuinely running loop.
fn event_loop(mut session: Session, rx: Receiver<Cmd>, toolkit: Arc<Mutex<Toolkit>>) -> Session {
    let mut frame = 0u64;
    let mut last_seen = 0usize;
    loop {
        // Apply edits that arrived since the last frame.
        loop {
            match rx.try_recv() {
                Ok(Cmd::Edit(src)) => match session.eval(&src) {
                    Ok(()) => println!("   · [live] edit applied between frames"),
                    Err(e) => println!("   · [live] edit rejected: {e}"),
                },
                Ok(Cmd::Stop) => return session,
                Err(_) => break,
            }
        }

        frame += 1;
        // The native → managed call. If the callback traps (a breaking edit),
        // we can't freeze a native frame — report and carry on with last-good;
        // the next frame picks up a repair.
        match session.call("on_frame", vec![]) {
            Ok(_) => {}
            Err(c) => println!("   · [native] frame {frame}: callback trapped ({c:?})"),
        }
        // Print the value the native draw() just recorded for this frame.
        {
            let t = toolkit.lock().unwrap();
            if t.drawn.len() > last_seen {
                last_seen = t.drawn.len();
                let (w, n) = t.drawn[last_seen - 1];
                println!("   frame {frame:>2}: window #{w} drew {n}");
            }
        }
        thread::sleep(Duration::from_millis(90));
        if frame >= 200 {
            return session; // safety cap
        }
    }
}
