//! Merging holds up on the real multi-track sample traces, not just on
//! hand-built two-thread fixtures.

use flame_core::{Profile, ProfileBuilder, TraceSource, TrackId};
use flame_render::renderer::{build_left_heavy_merged, build_single_track_profile};

const SOURCES: &[&dyn TraceSource] = &[
    &flame_format_firefox::FirefoxSource,
    &flame_format_chrome::ChromeSource,
    &flame_format_speedscope::SpeedscopeSource,
    &flame_format_folded::FoldedSource,
];

fn load(path: &str) -> Profile {
    let bytes = std::fs::read(path).unwrap();
    let name = path.rsplit('/').next().unwrap();
    let src = SOURCES
        .iter()
        .find(|s| s.detect(&bytes, Some(name)))
        .unwrap_or_else(|| panic!("no source matched {path}"));
    let mut b = ProfileBuilder::new();
    src.load(&bytes, &mut b).unwrap();
    b.finish()
}

/// Direct children of slice `i`: same track, one row down, contained in it.
fn children(p: &Profile, i: usize) -> Vec<usize> {
    let track = p.slices.track[i];
    let depth = p.slices.depth[i];
    let s = p.slices.start_ns[i];
    let e = s + p.slices.dur_ns[i];
    let row = p.slices.visible_in_row(track, depth + 1, s, e.max(s + 1));
    (row.start..row.end)
        .map(|c| c as usize)
        .filter(|&c| {
            let cs = p.slices.start_ns[c];
            cs >= s && cs + p.slices.dur_ns[c] <= e
        })
        .collect()
}

/// Merged slices keyed back to their source by (name, start, dur) — the merge
/// leaves times untouched, so that triple identifies a slice across the two
/// profiles. Triples that occur more than once can't be told apart and are
/// dropped rather than guessed at.
fn depth_by_identity(
    orig: &Profile,
    merged: &Profile,
) -> std::collections::HashMap<(u32, u64, u64), u16> {
    assert_eq!(
        merged.slices.len(),
        orig.slices.len(),
        "merging must not drop or invent slices"
    );
    let mut out: std::collections::HashMap<(u32, u64, u64), u16> = std::collections::HashMap::new();
    let mut ambiguous: std::collections::HashSet<(u32, u64, u64)> = std::collections::HashSet::new();
    for i in 0..merged.slices.len() {
        let key = (
            merged.slices.name[i].0,
            merged.slices.start_ns[i],
            merged.slices.dur_ns[i],
        );
        if out.insert(key, merged.slices.depth[i]).is_some() {
            ambiguous.insert(key);
        }
    }
    for key in ambiguous {
        out.remove(&key);
    }
    out
}

fn assert_merge_is_sane(path: &str, min_pairs: usize) {
    let orig = load(path);
    assert!(orig.tracks.len() > 1, "{path} should have several tracks");
    let merged = build_single_track_profile(&orig);
    assert_eq!(merged.tracks.len(), 1);

    let depth_of = depth_by_identity(&orig, &merged);
    let key = |p: &Profile, i: usize| {
        (p.slices.name[i].0, p.slices.start_ns[i], p.slices.dur_ns[i])
    };

    // Every parent/child pair keeps its one-row relationship.
    let mut checked = 0usize;
    for i in 0..orig.slices.len() {
        let Some(&pd) = depth_of.get(&key(&orig, i)) else { continue };
        for c in children(&orig, i) {
            let Some(&cd) = depth_of.get(&key(&orig, c)) else { continue };
            assert_eq!(
                cd,
                pd + 1,
                "{path}: child {} landed on row {cd} under a parent on row {pd}",
                orig.strings.get(orig.slices.name[c])
            );
            checked += 1;
        }
    }
    assert!(
        checked >= min_pairs,
        "{path}: only {checked} parent/child pairs checked, expected at least {min_pairs}"
    );

    // No two merged slices share a row and a moment.
    for (_, row) in merged.slices.rows.iter() {
        let mut spans: Vec<(u64, u64)> = (row.start..row.end)
            .map(|i| {
                let i = i as usize;
                (
                    merged.slices.start_ns[i],
                    merged.slices.start_ns[i] + merged.slices.dur_ns[i],
                )
            })
            .collect();
        spans.sort();
        for w in spans.windows(2) {
            assert!(w[0].1 <= w[1].0, "{path}: overlap on a merged row: {w:?}");
        }
    }
}

#[test]
fn merge_chrome_complex() {
    assert_merge_is_sane("../../samples/complex.chrome.json", 100);
}

#[test]
fn merge_firefox_three_threads() {
    // A small fixture: most of its slices share a (name, start, dur) triple
    // with another and are skipped as ambiguous, leaving few checkable pairs.
    assert_merge_is_sane("../../samples/firefox-processed-3.json", 4);
}

#[test]
fn merged_aggregation_covers_every_track() {
    let orig = load("../../samples/complex.chrome.json");
    let (agg, range, rows) = build_left_heavy_merged(&orig);
    assert_eq!(rows.len(), 1, "merged aggregation emits a single track");
    assert!((0..agg.len()).all(|i| agg.track[i] == TrackId(0)));

    // The merged tree must account for every root's time, from every track —
    // the old per-track aggregation of the packed profile lost all but one band.
    let root_total: u64 = (0..orig.slices.len())
        .filter(|&i| orig.slices.depth[i] == 0)
        .map(|i| orig.slices.dur_ns[i])
        .sum();
    assert_eq!(range.1, root_total);
}
