//! Verify the pause gate actually halts polling. Spawns a task that
//! repeatedly polls (yields and re-wakes), pauses the runtime, sleeps, and
//! checks that the task's poll count stops advancing while paused.

use cf_runtime::time::sleep;
use cf_runtime::Runtime;
use std::time::Duration;

#[test]
fn pause_halts_polling() {
    let rt = Runtime::new(2);
    let _h = rt.spawn("repolling", async {
        loop {
            sleep(Duration::from_millis(5)).await;
        }
    });

    // Let it run for a bit.
    std::thread::sleep(Duration::from_millis(100));
    let polls_before = rt
        .handle()
        .registry()
        .snapshot()
        .iter()
        .find(|t| t.name == "repolling")
        .unwrap()
        .poll_count;
    assert!(polls_before > 1, "task should have polled multiple times");

    // Pause and wait.
    rt.handle().controller().pause();
    std::thread::sleep(Duration::from_millis(100));
    let polls_during = rt
        .handle()
        .registry()
        .snapshot()
        .iter()
        .find(|t| t.name == "repolling")
        .unwrap()
        .poll_count;
    // Tolerance: at most one extra poll might have started before pause
    // landed. Anything more is a bug.
    assert!(
        polls_during <= polls_before + 1,
        "polling continued after pause: before={polls_before} during={polls_during}"
    );

    // Resume and verify polling restarts.
    rt.handle().controller().resume();
    std::thread::sleep(Duration::from_millis(100));
    let polls_after = rt
        .handle()
        .registry()
        .snapshot()
        .iter()
        .find(|t| t.name == "repolling")
        .unwrap()
        .poll_count;
    assert!(
        polls_after > polls_during + 2,
        "polling did not resume: during={polls_during} after={polls_after}"
    );
}
