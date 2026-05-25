//! OpenTelemetry span JSONL loader.
//!
//! Input format: one JSON span per line. Each line carries
//! `traceId`, `spanId`, optional `parentSpanId`, `name`, `kind`,
//! `startEpochNanos`, `endEpochNanos`, `service`, and an optional
//! `attributes` map. Files emitted by different processes (per-pid
//! JSONL) can be concatenated; the loader joins spans by `traceId`
//! regardless of which file they came from.
//!
//! Layout: one track per trace, ordered by trace start time. The
//! track name combines the trace's root span name with a short
//! trace-id prefix. `service` becomes the `Category` (so renderer
//! colors spans by service via the name-hash palette). Depth comes
//! from the parent-chain length within the trace; if two siblings
//! overlap in time at the same tree depth, the later one is bumped
//! down one row so the flame-graph "no overlap per row" invariant
//! holds.

use ahash::AHashMap;
use flame_core::{
    CategoryId, LoadError, LoadResult, ProfileBuilder, TraceSource, TrackKind,
};
use serde::Deserialize;

pub struct OtelSource;

pub const SOURCE: &dyn TraceSource = &OtelSource;

#[derive(Deserialize, Debug)]
struct RawSpan {
    #[serde(rename = "traceId")]
    trace_id: String,
    #[serde(rename = "spanId")]
    span_id: String,
    #[serde(rename = "parentSpanId", default)]
    parent_span_id: Option<String>,
    #[serde(default)]
    name: String,
    // CLIENT / SERVER / INTERNAL / PRODUCER / CONSUMER. Currently unused by
    // the flame view; the service-map view (TBD) will pair CLIENT with SERVER
    // across services to derive edges.
    #[allow(dead_code)]
    #[serde(default)]
    kind: String,
    #[serde(rename = "startEpochNanos")]
    start_epoch_nanos: u64,
    #[serde(rename = "endEpochNanos")]
    end_epoch_nanos: u64,
    #[serde(default)]
    service: String,
    /// Free-form per-span key/value bag. Captured as JSON values so we don't
    /// lose ints/bools; rendered to strings at attr-write time. Uses std
    /// HashMap because serde's derive doesn't cover ahash containers.
    #[serde(default)]
    attributes: std::collections::HashMap<String, serde_json::Value>,
}

/// JSON value → string for attribute storage. Strings drop their quotes,
/// numbers and bools stringify, null/objects/arrays serialize via serde_json.
fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

impl TraceSource for OtelSource {
    fn name(&self) -> &'static str {
        "OpenTelemetry JSONL"
    }

    fn detect(&self, input: &[u8], filename: Option<&str>) -> bool {
        // Filename hints: otel-traces-*.jsonl or anything ending .jsonl that
        // looks like spans.
        if let Some(name) = filename {
            let lower = name.to_ascii_lowercase();
            if lower.contains("otel-traces") || lower.ends_with(".jsonl") {
                // Also peek to make sure it's actually span-shaped.
                let head = &input[..input.len().min(4096)];
                let s = std::str::from_utf8(head).unwrap_or("");
                if s.contains("\"traceId\"") && s.contains("\"spanId\"") {
                    return true;
                }
            }
        }
        // Content sniff for pasted/no-name input: first non-whitespace must be
        // '{' and the head must contain the canonical OTel field names.
        let head = &input[..input.len().min(4096)];
        let mut i = 0;
        while i < head.len() && head[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= head.len() || head[i] != b'{' {
            return false;
        }
        let s = std::str::from_utf8(head).unwrap_or("");
        s.contains("\"traceId\"")
            && s.contains("\"spanId\"")
            && (s.contains("\"startEpochNanos\"") || s.contains("\"endEpochNanos\""))
    }

    fn load(&self, input: &[u8], builder: &mut ProfileBuilder) -> LoadResult<()> {
        // Parse every non-blank line as a span. We tolerate per-line parse
        // errors (warn + skip) rather than failing the whole file, because
        // half-written tail lines are common when a producer is still running.
        let text = std::str::from_utf8(input)
            .map_err(|e| LoadError::Parse(format!("otel jsonl: utf8: {e}")))?;

        let mut spans: Vec<RawSpan> = Vec::new();
        let mut line_no: usize = 0;
        let mut bad_lines = 0usize;
        for line in text.lines() {
            line_no += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<RawSpan>(trimmed) {
                Ok(s) => spans.push(s),
                Err(e) => {
                    bad_lines += 1;
                    if bad_lines <= 5 {
                        log::warn!("otel jsonl: line {line_no}: {e}");
                    }
                }
            }
        }
        if bad_lines > 5 {
            log::warn!("otel jsonl: {bad_lines} total bad lines (only first 5 logged)");
        }
        if spans.is_empty() {
            return Err(LoadError::Parse(
                "otel jsonl: no parseable spans found".into(),
            ));
        }

        // Time rebase: the absolute Unix-epoch nanos are huge and the renderer
        // uses durations from t=0. Subtract the global min so timestamps are
        // small and the viewport defaults are usable.
        let t0 = spans.iter().map(|s| s.start_epoch_nanos).min().unwrap_or(0);

        // Group spans by trace, preserving the first-seen order. Per-trace
        // ordering and depth assignment happens below.
        let mut trace_order: Vec<String> = Vec::new();
        let mut trace_idx: AHashMap<String, usize> = AHashMap::new();
        let mut by_trace: Vec<Vec<usize>> = Vec::new();
        for (i, s) in spans.iter().enumerate() {
            let idx = if let Some(&j) = trace_idx.get(&s.trace_id) {
                j
            } else {
                let j = by_trace.len();
                trace_idx.insert(s.trace_id.clone(), j);
                trace_order.push(s.trace_id.clone());
                by_trace.push(Vec::new());
                j
            };
            by_trace[idx].push(i);
        }

        // Service → CategoryId. The Category's color_idx is left at the
        // sentinel (u16::MAX) so the renderer falls back to its name-hash
        // palette — that gives a stable distinct color per service name.
        let mut service_cat: AHashMap<String, CategoryId> = AHashMap::new();
        let default_cat = builder.intern_category("otel");

        // Single synthetic process so every trace-track is grouped under it.
        let proc_id = builder.add_process(0, "otel");

        // Sort traces by their earliest start so the vertical layout is
        // chronological. Each trace becomes its own Track.
        let mut trace_starts: Vec<(usize, u64)> = by_trace
            .iter()
            .enumerate()
            .map(|(i, ix)| {
                let min = ix.iter().map(|&j| spans[j].start_epoch_nanos).min().unwrap_or(0);
                (i, min)
            })
            .collect();
        trace_starts.sort_by_key(|&(_, t)| t);

        for (trace_idx, _start) in trace_starts {
            let span_ix = &by_trace[trace_idx];
            let trace_id = &trace_order[trace_idx];

            // Build spanId -> local-index map for parent lookups.
            let mut local: AHashMap<&str, usize> = AHashMap::with_capacity(span_ix.len());
            for (k, &gi) in span_ix.iter().enumerate() {
                local.insert(spans[gi].span_id.as_str(), k);
            }

            // Tree-depth via memoized parent walk. Cycles or missing parents
            // collapse to depth 0 (parent in another file we didn't load is
            // the common case — the orphan still renders at the top of its
            // trace's track).
            let mut tree_depth: Vec<u16> = vec![u16::MAX; span_ix.len()];
            for k in 0..span_ix.len() {
                resolve_depth(k, span_ix, &spans, &local, &mut tree_depth);
            }

            // Resolve effective attributes per span = own attrs unioned with
            // ancestor attrs (own wins on conflict). Two-pass DP: root-down
            // would require topological order, so we recurse with memoization
            // instead. Result is per-span owned `AHashMap<String, String>`.
            let mut resolved_attrs: Vec<Option<AHashMap<String, String>>> =
                (0..span_ix.len()).map(|_| None).collect();
            for k in 0..span_ix.len() {
                resolve_attrs(k, span_ix, &spans, &local, &mut resolved_attrs);
            }

            // Find the root span for this trace's track label. Prefer one with
            // no parent in this trace; fall back to the earliest-starting span.
            let root_local = span_ix
                .iter()
                .enumerate()
                .find(|(_, &gi)| {
                    spans[gi]
                        .parent_span_id
                        .as_deref()
                        .map(|p| !local.contains_key(p))
                        .unwrap_or(true)
                })
                .map(|(k, _)| k)
                .unwrap_or(0);
            let root_name = &spans[span_ix[root_local]].name;
            let short_tid = trace_id.get(..8).unwrap_or(trace_id.as_str());
            let track_label = if root_name.is_empty() {
                format!("trace {short_tid}")
            } else {
                format!("{root_name} ({short_tid})")
            };
            let thread = builder.add_thread(Some(proc_id), trace_idx as i64, &track_label);
            let track = builder.add_track(TrackKind::Thread(thread), &track_label, None);

            // Overlap resolution. Iterate spans in start order; if proposed
            // depth row's last-end > this span's start, bump depth by 1 and
            // retry. This keeps the flame-graph "no overlap in row" invariant
            // for the renderer's row-range index.
            let mut order: Vec<usize> = (0..span_ix.len()).collect();
            order.sort_by_key(|&k| spans[span_ix[k]].start_epoch_nanos);
            let mut last_end_per_depth: Vec<u64> = Vec::new();

            for k in order {
                let gi = span_ix[k];
                let s = &spans[gi];
                let start_ns = s.start_epoch_nanos.saturating_sub(t0);
                // OTel spans with start == end (instant events) get a 1ns
                // floor so they're still visible/picky.
                let dur_ns = s
                    .end_epoch_nanos
                    .saturating_sub(s.start_epoch_nanos)
                    .max(1);
                let end_ns = start_ns + dur_ns;

                let mut depth = tree_depth[k] as usize;
                while depth < last_end_per_depth.len()
                    && last_end_per_depth[depth] > start_ns
                {
                    depth += 1;
                }
                if depth >= last_end_per_depth.len() {
                    last_end_per_depth.resize(depth + 1, 0);
                }
                last_end_per_depth[depth] = end_ns;

                let name_id = builder.intern_string(if s.name.is_empty() {
                    "(unnamed)"
                } else {
                    &s.name
                });
                let category = if s.service.is_empty() {
                    default_cat
                } else if let Some(&c) = service_cat.get(&s.service) {
                    c
                } else {
                    let c = builder.intern_category(&s.service);
                    service_cat.insert(s.service.clone(), c);
                    c
                };

                // Translate the resolved attribute bag into builder-interned
                // (key_idx, value_string_id) pairs.
                let mut attrs_vec: Vec<(u16, flame_core::StringId)> =
                    match resolved_attrs[k].as_ref() {
                        Some(map) => map
                            .iter()
                            .map(|(k_name, v_str)| {
                                let kidx = builder.intern_attr_key(k_name);
                                let vid = builder.intern_string(v_str);
                                (kidx, vid)
                            })
                            .collect(),
                        None => Vec::new(),
                    };
                // Always pin trace_id / span_id / parent_span_id as attrs so
                // downstream views (sequence diagram, service map) can walk
                // per-trace structure without the loader-private maps.
                let tid_key = builder.intern_attr_key("trace_id");
                let tid_val = builder.intern_string(&s.trace_id);
                attrs_vec.push((tid_key, tid_val));
                let sid_key = builder.intern_attr_key("span_id");
                let sid_val = builder.intern_string(&s.span_id);
                attrs_vec.push((sid_key, sid_val));
                if let Some(p) = s.parent_span_id.as_deref() {
                    let pkey = builder.intern_attr_key("parent_span_id");
                    let pval = builder.intern_string(p);
                    attrs_vec.push((pkey, pval));
                }
                // Span kind (CLIENT / SERVER / INTERNAL / …) gates request /
                // response arrow rendering on the sequence view.
                if !s.kind.is_empty() {
                    let kkey = builder.intern_attr_key("span_kind");
                    let kval = builder.intern_string(&s.kind);
                    attrs_vec.push((kkey, kval));
                }
                // Service is already the slice's Category, but exposing it as
                // an attr lets it be used as the lifeline grouping key on the
                // sequence view via the standard picker.
                if !s.service.is_empty() {
                    let svc_key = builder.intern_attr_key("service");
                    let svc_val = builder.intern_string(&s.service);
                    attrs_vec.push((svc_key, svc_val));
                }

                builder.add_complete_slice_with_attrs(
                    track,
                    depth as u16,
                    start_ns,
                    dur_ns,
                    name_id,
                    category,
                    None,
                    attrs_vec,
                );
            }
        }

        Ok(())
    }
}

/// DP: build the effective attribute bag for `k` = its own attrs unioned with
/// the resolved bag of its parent (own-keys win on conflict). Memoized.
/// Missing parents stop the walk.
fn resolve_attrs(
    k: usize,
    span_ix: &[usize],
    spans: &[RawSpan],
    local: &AHashMap<&str, usize>,
    memo: &mut [Option<AHashMap<String, String>>],
) {
    if memo[k].is_some() {
        return;
    }
    // Insert a placeholder first so cycles (shouldn't exist in OTel data but
    // be paranoid) don't recurse forever — placeholder is just the span's own
    // attrs, with no inheritance.
    let gi = span_ix[k];
    let own: AHashMap<String, String> = spans[gi]
        .attributes
        .iter()
        .map(|(k_name, v)| (k_name.clone(), value_to_string(v)))
        .collect();
    memo[k] = Some(own.clone());

    let parent_local = spans[gi]
        .parent_span_id
        .as_deref()
        .and_then(|p| local.get(p).copied())
        .filter(|&pk| pk != k);

    if let Some(pk) = parent_local {
        resolve_attrs(pk, span_ix, spans, local, memo);
        let parent_bag = memo[pk].as_ref().cloned().unwrap_or_default();
        // Inherit any parent keys this span doesn't already have.
        let merged = memo[k].as_mut().expect("just-set");
        for (kk, vv) in parent_bag {
            merged.entry(kk).or_insert(vv);
        }
    }
}

fn resolve_depth(
    k: usize,
    span_ix: &[usize],
    spans: &[RawSpan],
    local: &AHashMap<&str, usize>,
    memo: &mut [u16],
) -> u16 {
    if memo[k] != u16::MAX {
        return memo[k];
    }
    // Sentinel during recursion to break cycles.
    memo[k] = 0;
    let gi = span_ix[k];
    let depth = match spans[gi].parent_span_id.as_deref() {
        Some(p) => match local.get(p) {
            Some(&pk) if pk != k => resolve_depth(pk, span_ix, spans, local, memo).saturating_add(1),
            _ => 0,
        },
        None => 0,
    };
    memo[k] = depth;
    depth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_basic() {
        let input = br#"{"traceId":"abc","spanId":"def","name":"foo","kind":"INTERNAL","startEpochNanos":1000,"endEpochNanos":2000,"durationNanos":1000,"service":"svc"}"#;
        assert!(OtelSource.detect(input, Some("otel-traces-foo-123.jsonl")));
        assert!(OtelSource.detect(input, None));
    }

    #[test]
    fn parse_single_span() {
        let input = br#"{"traceId":"a","spanId":"b","name":"root","kind":"SERVER","startEpochNanos":1000,"endEpochNanos":3000,"durationNanos":2000,"service":"svc"}"#;
        let mut b = ProfileBuilder::new();
        OtelSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 1);
        assert_eq!(p.slices.start_ns[0], 0);
        assert_eq!(p.slices.dur_ns[0], 2000);
        assert_eq!(p.slices.depth[0], 0);
    }

    #[test]
    fn parent_chain_sets_depth() {
        let input = br#"{"traceId":"t","spanId":"r","name":"root","kind":"SERVER","startEpochNanos":1000,"endEpochNanos":5000,"durationNanos":4000,"service":"svc"}
{"traceId":"t","spanId":"c","parentSpanId":"r","name":"child","kind":"INTERNAL","startEpochNanos":2000,"endEpochNanos":3000,"durationNanos":1000,"service":"svc"}
{"traceId":"t","spanId":"g","parentSpanId":"c","name":"grand","kind":"CLIENT","startEpochNanos":2100,"endEpochNanos":2200,"durationNanos":100,"service":"svc"}"#;
        let mut b = ProfileBuilder::new();
        OtelSource.load(input, &mut b).unwrap();
        let p = b.finish();
        // Sorted by (track, depth, start): root(0), child(1), grand(2)
        assert_eq!(p.slices.len(), 3);
        let depths: Vec<u16> = (0..p.slices.len()).map(|i| p.slices.depth[i]).collect();
        assert_eq!(depths, vec![0, 1, 2]);
    }

    #[test]
    fn cross_file_join_by_trace_id() {
        // Two "files" concatenated. Same traceId, different services. Child
        // in the second file should attach to parent in the first.
        let input = br#"{"traceId":"t","spanId":"r","name":"root","kind":"SERVER","startEpochNanos":1000,"endEpochNanos":5000,"durationNanos":4000,"service":"a"}
{"traceId":"t","spanId":"c","parentSpanId":"r","name":"child","kind":"INTERNAL","startEpochNanos":2000,"endEpochNanos":3000,"durationNanos":1000,"service":"b"}"#;
        let mut b = ProfileBuilder::new();
        OtelSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 2);
        assert_eq!(p.tracks.len(), 1, "both spans should land on the trace's single track");
        // Two services -> two categories (plus the default "otel" + builder's "default").
        let svc_names: Vec<&str> = p
            .categories
            .iter()
            .map(|c| p.strings.get(c.name))
            .collect();
        assert!(svc_names.contains(&"a"));
        assert!(svc_names.contains(&"b"));
    }

    #[test]
    fn orphan_span_attaches_at_depth_zero() {
        // parentSpanId points to a span we don't have. Should not panic; the
        // span renders at depth 0.
        let input = br#"{"traceId":"t","spanId":"c","parentSpanId":"missing","name":"orphan","kind":"INTERNAL","startEpochNanos":1000,"endEpochNanos":2000,"durationNanos":1000,"service":"svc"}"#;
        let mut b = ProfileBuilder::new();
        OtelSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 1);
        assert_eq!(p.slices.depth[0], 0);
    }

    #[test]
    fn overlapping_siblings_get_distinct_rows() {
        // Two siblings of root that overlap in time. The later one should be
        // bumped to depth 2 so the flame-graph row invariant holds.
        let input = br#"{"traceId":"t","spanId":"r","name":"root","kind":"SERVER","startEpochNanos":1000,"endEpochNanos":5000,"durationNanos":4000,"service":"svc"}
{"traceId":"t","spanId":"a","parentSpanId":"r","name":"a","kind":"INTERNAL","startEpochNanos":2000,"endEpochNanos":4000,"durationNanos":2000,"service":"svc"}
{"traceId":"t","spanId":"b","parentSpanId":"r","name":"b","kind":"INTERNAL","startEpochNanos":2500,"endEpochNanos":3500,"durationNanos":1000,"service":"svc"}"#;
        let mut b = ProfileBuilder::new();
        OtelSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 3);
        // Sorted by (track, depth, start): root(d=0), a(d=1), b(d=2)
        assert_eq!(p.slices.depth, vec![0, 1, 2]);
    }

    #[test]
    fn attributes_inherit_from_parent_chain() {
        // Root has user_id; mid-level has nothing extra; leaf has its own
        // user_id (which should win). The middle span should inherit user_id
        // from root.
        let input = br#"{"traceId":"t","spanId":"r","name":"root","kind":"SERVER","startEpochNanos":1000,"endEpochNanos":9000,"durationNanos":8000,"service":"svc","attributes":{"user_id":"u-root","client":"c1"}}
{"traceId":"t","spanId":"m","parentSpanId":"r","name":"mid","kind":"INTERNAL","startEpochNanos":2000,"endEpochNanos":8000,"durationNanos":6000,"service":"svc"}
{"traceId":"t","spanId":"l","parentSpanId":"m","name":"leaf","kind":"CLIENT","startEpochNanos":3000,"endEpochNanos":4000,"durationNanos":1000,"service":"svc","attributes":{"user_id":"u-leaf"}}"#;
        let mut b = ProfileBuilder::new();
        OtelSource.load(input, &mut b).unwrap();
        let mut p = b.finish();
        assert_eq!(p.slices.len(), 3);

        let user_key = p.strings.intern("user_id");
        let client_key = p.strings.intern("client");

        // Spans are SoA-sorted by (track, depth, start); for this single
        // trace that's root(d=0), mid(d=1), leaf(d=2).
        let user_at = |i: u32| -> String {
            p.attrs
                .get(i, user_key)
                .map(|sid| p.strings.get(sid).to_string())
                .unwrap_or_default()
        };
        let client_at = |i: u32| -> String {
            p.attrs
                .get(i, client_key)
                .map(|sid| p.strings.get(sid).to_string())
                .unwrap_or_default()
        };

        assert_eq!(user_at(0), "u-root", "root keeps its own user_id");
        assert_eq!(user_at(1), "u-root", "mid inherits user_id from root");
        assert_eq!(user_at(2), "u-leaf", "leaf's own user_id wins over root's");

        assert_eq!(client_at(0), "c1");
        assert_eq!(client_at(1), "c1", "mid inherits client from root");
        assert_eq!(client_at(2), "c1", "leaf inherits client from root (not set on leaf)");
    }

    #[test]
    fn tolerates_blank_and_garbage_lines() {
        let input = b"\n\n{\"not\": \"a span\"}\n{\"traceId\":\"t\",\"spanId\":\"r\",\"name\":\"root\",\"kind\":\"SERVER\",\"startEpochNanos\":1000,\"endEpochNanos\":2000,\"durationNanos\":1000,\"service\":\"svc\"}\n";
        let mut b = ProfileBuilder::new();
        OtelSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 1);
    }
}
