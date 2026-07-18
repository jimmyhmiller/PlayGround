//! Low-memory thesis guards, isolated in their own test binary.
//!
//! These read process-wide allocation counters ([`diffpack::memory`]), so they
//! must not run concurrently with other tests that allocate. As an integration
//! test this is a separate process, and the single test below runs all
//! measurements sequentially on one thread, so the deltas are clean. The
//! allocator counts every byte exactly, so with 3-5x headroom over measured
//! values these fail on a real regression rather than flaking.
//!
//! Guarding: a build must not hoard transient state, the resident graph must be
//! compact, repeated edits must not accumulate memory, and teardown must release
//! the graph. Keep green when adding the plugin host or any deep integration.

use diffpack::bundle_benchmark::run_bundle_scale_memory;

#[test]
fn the_incremental_graph_stays_low_memory() {
    // Resident graph is compact and the build does not hoard transient ASTs.
    let build = run_bundle_scale_memory(800, 4, 1, false).unwrap();
    assert!(
        build.bytes_per_module < 16_000.0,
        "resident cost is {:.0} bytes/module (measured ~3.5 KB); the graph must stay compact",
        build.bytes_per_module
    );
    let peak_per_module = build.build_peak_bytes as f64 / build.modules as f64;
    assert!(
        peak_per_module < 24_000.0,
        "build peak is {peak_per_module:.0} bytes/module; the build must not hoard transient state",
    );

    // 200 edits to one leaf module must re-transform only that module and must
    // not accumulate memory (measured growth ~0.2 KB; a per-revision leak would
    // be hundreds of KB).
    let edited = run_bundle_scale_memory(400, 4, 200, false).unwrap();
    assert_eq!(
        edited.transformed_per_edit_max, 1,
        "each leaf edit must re-transform exactly one module"
    );
    // Incremental emit: each leaf edit re-renders exactly one chunk, and the
    // render cache stays bounded to the live chunk set (a per-edit revision leak
    // would grow it with the edit count).
    assert_eq!(
        edited.rendered_chunks_per_edit_max, 1,
        "each leaf edit must re-render exactly one chunk, not the whole bundle"
    );
    assert!(
        edited.render_cache_entries <= 1,
        "the render cache grew to {} entries over 200 edits; it must stay bounded to the live chunk set",
        edited.render_cache_entries
    );
    assert!(
        edited.retained_growth_over_edits_bytes < 256 * 1024,
        "200 edits grew retained memory by {} bytes; edits must not accumulate memory",
        edited.retained_growth_over_edits_bytes
    );

    // Dropping the bundler releases the graph (measured residual ~2-3% of the
    // graph's resident cost).
    assert!(
        edited.retained_after_drop_bytes < edited.retained_after_build_bytes / 4,
        "after dropping the bundler, {} bytes remain vs {} held by the graph; teardown must release it",
        edited.retained_after_drop_bytes,
        edited.retained_after_build_bytes
    );
}
