//! Confirms the format dispatch table picks the right parser for each sample
//! file shipped under `samples/`.

use flame_core::{ProfileBuilder, TraceSource};

const SOURCES: &[&dyn TraceSource] = &[
    &flame_format_firefox::FirefoxSource,
    &flame_format_chrome::ChromeSource,
    &flame_format_speedscope::SpeedscopeSource,
    &flame_format_folded::FoldedSource,
];

fn dispatch(bytes: &[u8], filename: &str) -> &'static str {
    SOURCES
        .iter()
        .find(|s| s.detect(bytes, Some(filename)))
        .map(|s| s.name())
        .unwrap_or("UNKNOWN")
}

fn load(bytes: &[u8], filename: &str) -> flame_core::Profile {
    let src = SOURCES
        .iter()
        .find(|s| s.detect(bytes, Some(filename)))
        .expect("a source should match");
    let mut b = ProfileBuilder::new();
    src.load(bytes, &mut b).unwrap();
    b.finish()
}

#[test]
fn dispatch_folded() {
    let bytes = std::fs::read("../../samples/test.folded").unwrap();
    assert_eq!(dispatch(&bytes, "test.folded"), "Folded stacks");
    let p = load(&bytes, "test.folded");
    assert!(p.slices.len() >= 5, "got {} slices", p.slices.len());
}

#[test]
fn dispatch_chrome() {
    let bytes = std::fs::read("../../samples/test.chrome.json").unwrap();
    assert_eq!(dispatch(&bytes, "test.chrome.json"), "Chrome Trace JSON");
    let p = load(&bytes, "test.chrome.json");
    // 1 main B/E + 1 load B/E + 1 parse X + 1 render B/E + 2 X (draw, swap) = 6 slices
    assert_eq!(p.slices.len(), 6);
}

#[test]
fn dispatch_firefox_v24() {
    let bytes = std::fs::read("../../samples/firefox-processed-3.json").unwrap();
    assert_eq!(dispatch(&bytes, "firefox-processed-3.json"), "Firefox Profiler JSON");
    let p = load(&bytes, "firefox-processed-3.json");
    assert!(p.tracks.len() >= 3, "expected ≥3 tracks (3 threads), got {}", p.tracks.len());
    assert!(p.slices.len() > 0);
}

#[test]
fn dispatch_firefox_v60_shared() {
    let bytes = std::fs::read("../../samples/firefox-v60-min.json").unwrap();
    assert_eq!(dispatch(&bytes, "firefox-v60-min.json"), "Firefox Profiler JSON");
    let p = load(&bytes, "firefox-v60-min.json");
    // 3 sample slices + 1 marker = 4 slices, on 2 tracks (thread + markers).
    assert_eq!(p.slices.len(), 4);
    assert_eq!(p.tracks.len(), 2);
}

#[test]
fn dispatch_speedscope() {
    let bytes = std::fs::read("../../samples/test.speedscope.json").unwrap();
    assert_eq!(dispatch(&bytes, "test.speedscope.json"), "Speedscope JSON");
    let p = load(&bytes, "test.speedscope.json");
    // main(0..100) + compute(10..40) + io(45..60) + render(65..90) = 4 slices
    assert_eq!(p.slices.len(), 4);
}

/// 1M tiny slices on one row. Sub-pixel cull at full zoom-out should leave
/// well under 1280 visible (one per pixel of width).
#[test]
fn subpixel_cull_million_slices() {
    use flame_core::{ProfileBuilder, TrackKind, TrackId};
    let mut b = ProfileBuilder::new();
    let proc = b.add_process(0, "stress");
    let thread = b.add_thread(Some(proc), 0, "main");
    let track = b.add_track(TrackKind::Thread(thread), "stress", None);
    let cat = b.intern_category("default");
    let name = b.intern_string("tick");
    for i in 0..1_000_000u64 {
        b.add_complete_slice(track, 0, i * 1000, 500, name, cat, None);
    }
    let p = b.finish();
    assert_eq!(p.slices.len(), 1_000_000);

    // Simulate viewport culling at 1280px wide showing the full duration.
    let total = p.duration_ns();
    let view_w_px = 1280.0;
    let ns_per_px = total as f64 / view_w_px;

    let row = p.slices.visible_in_row(TrackId(0), 0, 0, total);
    let mut rendered = 0;
    for i in row.start..row.end {
        let dur = p.slices.dur_ns[i as usize];
        let w_px = dur as f64 / ns_per_px;
        if w_px >= 1.0 {
            rendered += 1;
        }
    }
    // With slices 500ns wide and total ~1e9 ns, ns_per_px ~= 781250, w_px = 0.00064 — all culled.
    assert!(rendered < 100, "sub-pixel cull failed: rendered {rendered}");
}
