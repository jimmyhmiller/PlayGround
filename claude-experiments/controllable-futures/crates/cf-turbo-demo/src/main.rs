//! Smallest possible turbo-tasks demo on cf-runtime.
//!
//! Goal: prove the whole stack links — cf-runtime executor, cf-tokio shim,
//! turbo-tasks library — by running something that touches turbo-tasks
//! types in a real async context.
//!
//! We don't construct a `TurboTasksApi` (that requires a backend
//! implementation hundreds of lines long); we just use turbo-tasks's
//! standalone primitives — its `Event` (a building block of its task
//! coordination) and its task-local plumbing — to confirm the linkage.

use cf_runtime::Runtime;
use std::time::Duration;
use turbo_tasks::event::Event;

fn main() {
    let rt = Runtime::new(2);

    // Spawn a producer/consumer using turbo-tasks's `Event` primitive.
    // This is the same Event that turbo-tasks itself uses internally for
    // task readiness signaling — it's built on tokio's primitives, which
    // in our build resolve to cf-tokio.
    rt.spawn("turbo-event-demo", async move {
        // Event::new takes a closure-of-closure (a description function).
        let event = std::sync::Arc::new(Event::new(|| || "demo-event".to_string()));
        let listener = event.listen_with_note(|| || "waiting".to_string());

        // Notify after a delay from a sibling task.
        let event_clone = event.clone();
        cf_tokio::spawn(async move {
            cf_runtime::time::sleep(Duration::from_millis(150)).await;
            event_clone.notify(usize::MAX);
        });

        listener.await;
        eprintln!("turbo-tasks Event was notified across cf-runtime workers ✓");
    });

    // Park forever; the host UI normally drives shutdown. For this demo
    // we just sleep enough to let the spawned future complete.
    std::thread::sleep(Duration::from_secs(1));

    let log = rt.handle().log();
    let reg = rt.handle().registry();
    eprintln!(
        "[cf-turbo-demo] events={} tasks={}",
        log.len(),
        reg.snapshot().len()
    );
}
