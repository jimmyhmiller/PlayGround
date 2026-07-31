//! Merging every track onto one (the `m` key) must not take a stack apart.
//!
//! The failure this guards against: a per-slice row pack places each slice on
//! the first row that is free *at that slice's start*, which has nothing to do
//! with the slice's parent. A child then lands above its own parent and two
//! threads' frames interleave into one apparent stack — the vertical gaps that
//! make a merged graph look like it never merged.
//!
//! What merging does guarantee is checked below: nesting survives, a stack
//! occupies one contiguous band of rows, nothing overlaps within a row, and
//! stacks that don't overlap in time share rows.

use flame_core::{Profile, ProfileBuilder, TrackKind};
use flame_render::renderer::{
    build_left_heavy_layout, build_left_heavy_merged, build_single_track_profile,
};

/// A short-lived thread plus a long-lived one — the arrangement that breaks a
/// naive pack, because the short thread's row frees up early and the long
/// thread's *children* then fit in it.
///
/// ```text
/// short:  [ s_root ]                       (0..10)
/// long:   [ l_root ................... ]   (0..1000)
///             [ l_mid ............ ]       (100..900)
///                [ l_leaf ]                (200..400)
/// ```
fn short_and_long_threads() -> Profile {
    let mut b = ProfileBuilder::new();
    let cat = flame_core::CategoryId::DEFAULT;
    let proc = b.add_process(1, "p");

    let short = b.add_thread(Some(proc), 1, "short");
    let short_track = b.add_track(TrackKind::Thread(short), "short", None);
    let n = b.intern_string("s_root");
    b.add_complete_slice(short_track, 0, 0, 10, n, cat, None);

    let long = b.add_thread(Some(proc), 2, "long");
    let long_track = b.add_track(TrackKind::Thread(long), "long", None);
    for (depth, name, start, dur) in [
        (0u16, "l_root", 0u64, 1000u64),
        (1, "l_mid", 100, 800),
        (2, "l_leaf", 200, 200),
    ] {
        let n = b.intern_string(name);
        b.add_complete_slice(long_track, depth, start, dur, n, cat, None);
    }
    b.finish()
}

/// Every slice, keyed by name, as `(depth, start, end)`.
fn placed(p: &Profile) -> std::collections::HashMap<String, (u16, u64, u64)> {
    (0..p.slices.len())
        .map(|i| {
            (
                p.strings.get(p.slices.name[i]).to_string(),
                (
                    p.slices.depth[i],
                    p.slices.start_ns[i],
                    p.slices.start_ns[i] + p.slices.dur_ns[i],
                ),
            )
        })
        .collect()
}

#[test]
fn merging_keeps_children_under_their_parents() {
    let merged = build_single_track_profile(&short_and_long_threads());
    let at = placed(&merged);
    let root = at["l_root"].0;
    assert_eq!(at["l_mid"].0, root + 1, "l_mid should sit directly under l_root: {at:?}");
    assert_eq!(at["l_leaf"].0, root + 2, "l_leaf should sit under l_mid: {at:?}");
}

/// A whole stack lands in one contiguous band of rows, and a stack that
/// outlives another gets the lower band. Rows a stack does not currently reach
/// stay empty — that is a thread being shallow, not a hole in a stack.
#[test]
fn each_stack_lands_in_its_own_contiguous_band() {
    let merged = build_single_track_profile(&short_and_long_threads());
    let at = placed(&merged);
    let long_rows = [at["l_root"].0, at["l_mid"].0, at["l_leaf"].0];
    assert_eq!(long_rows, [0, 1, 2], "the long-lived stack takes the bottom band: {at:?}");
    assert!(
        at["s_root"].0 >= 3,
        "the short stack must not land inside the long stack's band: {at:?}"
    );
}

#[test]
fn merging_never_overlaps_two_slices_in_one_row() {
    let merged = build_single_track_profile(&short_and_long_threads());
    let n = merged.slices.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if merged.slices.depth[i] != merged.slices.depth[j] {
                continue;
            }
            let (a0, a1) = (
                merged.slices.start_ns[i],
                merged.slices.start_ns[i] + merged.slices.dur_ns[i],
            );
            let (b0, b1) = (
                merged.slices.start_ns[j],
                merged.slices.start_ns[j] + merged.slices.dur_ns[j],
            );
            assert!(
                a1 <= b0 || b1 <= a0,
                "slices {i} and {j} overlap on row {}",
                merged.slices.depth[i]
            );
        }
    }
}

/// Two threads running the same code. Aggregated + merged is the view that
/// actually folds them together, so `main` must appear once with both threads'
/// time in it, not once per thread.
#[test]
fn aggregating_in_merged_mode_folds_every_thread_into_one_tree() {
    let mut b = ProfileBuilder::new();
    let cat = flame_core::CategoryId::DEFAULT;
    let proc = b.add_process(1, "p");
    let main = b.intern_string("main");
    let work = b.intern_string("work");
    for (tid, name, work_dur) in [(1i64, "worker-1", 50u64), (2, "worker-2", 30)] {
        let t = b.add_thread(Some(proc), tid, name);
        let track = b.add_track(TrackKind::Thread(t), name, None);
        b.add_complete_slice(track, 0, 0, 100, main, cat, None);
        b.add_complete_slice(track, 1, 0, work_dur, work, cat, None);
    }
    let p = b.finish();

    let (agg, range, rows) = build_left_heavy_merged(&p);
    assert_eq!(rows, vec![2], "one track, two rows deep");
    assert_eq!(range.1, 200, "the merged x-axis spans both threads' time");
    let named: Vec<(u16, u64)> = (0..agg.len())
        .filter(|&i| agg.name[i] == main)
        .map(|i| (agg.depth[i], agg.dur_ns[i]))
        .collect();
    assert_eq!(named, vec![(0, 200)], "both threads' main should be one bar");
    let worked: Vec<(u16, u64)> = (0..agg.len())
        .filter(|&i| agg.name[i] == work)
        .map(|i| (agg.depth[i], agg.dur_ns[i]))
        .collect();
    assert_eq!(worked, vec![(1, 80)], "and its child likewise");
    assert!((0..agg.len()).all(|i| agg.track[i] == flame_core::TrackId(0)));

    // Unmerged, the same profile still aggregates per track.
    let (per_track, _, rows) = build_left_heavy_layout(&p);
    assert_eq!(rows, vec![2, 2]);
    assert_eq!(
        (0..per_track.len()).filter(|&i| per_track.name[i] == main).count(),
        2,
        "multi-track mode keeps one main bar per thread"
    );
}

/// Independent stacks that don't overlap in time should share rows — that
/// compaction is the whole point of merging.
#[test]
fn merging_reuses_a_row_once_a_stack_has_finished() {
    let mut b = ProfileBuilder::new();
    let cat = flame_core::CategoryId::DEFAULT;
    let proc = b.add_process(1, "p");
    for (tid, name, start) in [(1i64, "first", 0u64), (2, "second", 100)] {
        let t = b.add_thread(Some(proc), tid, name);
        let track = b.add_track(TrackKind::Thread(t), name, None);
        let root = b.intern_string(&format!("{name}_root"));
        let kid = b.intern_string(&format!("{name}_kid"));
        b.add_complete_slice(track, 0, start, 50, root, cat, None);
        b.add_complete_slice(track, 1, start + 10, 20, kid, cat, None);
    }
    let merged = build_single_track_profile(&b.finish());
    let at = placed(&merged);
    assert_eq!(at["first_root"].0, at["second_root"].0, "disjoint stacks share a row: {at:?}");
    assert_eq!(at["first_kid"].0, at["first_root"].0 + 1);
    assert_eq!(at["second_kid"].0, at["second_root"].0 + 1);
    assert_eq!(merged.tracks.len(), 1);
}
