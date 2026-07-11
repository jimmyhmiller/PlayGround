//! Editing a *genuinely running* program — a tight loop with no `yield` in it,
//! stepping continuously on a background thread while edits are injected from
//! another thread and applied between steps. The running loop's behavior
//! changes mid-flight, without pausing or restarting.
//!
//! Run with: `cargo run --bin livetype-liveloop`

use livetype::*;
use livetype_core::Session;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::Duration;

enum ToDriver {
    Edit(String),
}
enum FromDriver {
    Emit(i64),
    Note(String),
    Done,
}

fn main() {
    println!("\n■ a hot loop, edited while it runs (no yields — it never pauses itself)\n");

    let (to_driver, driver_edits) = channel::<ToDriver>();
    let (driver_events, events) = channel::<FromDriver>();

    let handle = thread::spawn(move || driver(driver_edits, driver_events));

    // The "editor": watch the running output; after a dozen ticks, change the
    // computation live; watch the output change without a restart.
    let mut vals = Vec::new();
    let mut edited = false;
    for evt in &events {
        match evt {
            FromDriver::Emit(n) => {
                vals.push(n);
                println!("   tick {:>2}: reading = {n}", vals.len());
                if vals.len() == 12 && !edited {
                    println!("\n   ✎ LIVE EDIT (from another thread, loop still running):");
                    println!("       fn read(s: Sensor) -> i64 {{ s.reading * 2 }}\n");
                    to_driver
                        .send(ToDriver::Edit(
                            "fn read(s: Sensor) -> i64 { s.reading * 2 }".into(),
                        ))
                        .unwrap();
                    edited = true;
                }
            }
            FromDriver::Note(s) => println!("   · {s}"),
            FromDriver::Done => break,
        }
    }
    handle.join().unwrap();

    let switched = vals.iter().any(|&v| v == 42) && vals.iter().any(|&v| v == 84);
    println!(
        "\n{} the same running loop reported {} then {} — edited in flight, never restarted.\n",
        if switched { "✓" } else { "✗" },
        vals.first().copied().unwrap_or(0),
        vals.last().copied().unwrap_or(0),
    );
}

/// Runs the program in a tight loop, stepping continuously. Between every step
/// it drains the edit channel and applies any edit — the whole point: an edit
/// lands between two instructions of a running loop, no safe point required.
fn driver(edits: Receiver<ToDriver>, events: Sender<FromDriver>) {
    let mut s = Session::new();
    s.eval(
        r#"
        struct Sensor { reading: i64 }
        fn read(s: Sensor) -> i64 { s.reading }
        fn main() -> i64 {
            let s = Sensor { reading: 42 };
            let i = 0;
            while i < 30 {
                emit(read(s));
                i = i + 1;
            }
            0
        }
    "#,
    )
    .unwrap();

    let main = s.fn_id("main").unwrap();
    let actor = s.runtime.spawn(main, vec![]).unwrap();
    let mut seen = 0;

    loop {
        // Apply any pending edits — between steps of the running loop.
        while let Ok(ToDriver::Edit(src)) = edits.try_recv() {
            match s.eval(&src) {
                Ok(()) => events.send(FromDriver::Note("edit applied to the live world".into())).ok(),
                Err(e) => events.send(FromDriver::Note(format!("edit rejected: {e}"))).ok(),
            };
        }
        match &s.runtime.actors[&actor].status {
            ActorStatus::Complete(_) => {
                let _ = events.send(FromDriver::Done);
                break;
            }
            ActorStatus::Runnable => {
                s.runtime.step(actor);
                if s.runtime.output.len() > seen {
                    seen = s.runtime.output.len();
                    if let Value::I64(n) = s.runtime.output[seen - 1] {
                        let _ = events.send(FromDriver::Emit(n));
                    }
                    // Pace so it's watchable (and so an edit can land mid-run).
                    thread::sleep(Duration::from_millis(30));
                }
            }
            ActorStatus::Paused(_) => {
                // A breaking edit would land here; wait for a repairing edit.
                let _ = events.send(FromDriver::Note("frozen — waiting for a repair edit".into()));
                match edits.recv() {
                    Ok(ToDriver::Edit(src)) => {
                        let _ = s.eval(&src);
                    }
                    Err(_) => break,
                }
            }
        }
    }
}
