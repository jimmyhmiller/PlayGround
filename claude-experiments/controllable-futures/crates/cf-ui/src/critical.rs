//! Critical-path analysis over the reconstructed span tree.
//!
//! Treats a span as the sum of its own self-time plus the time of its
//! single longest descendant chain (children assumed sequential, which
//! matches tracing's enter-exit semantics for single-threaded futures
//! and is a reasonable upper-bound elsewhere). The critical path is
//! the longest such root-to-leaf chain across the whole tree.
//!
//! In English: "if you sped up these spans, the build finishes
//! earlier". Anything not on the critical path is parallel or
//! overlapped work that doesn't extend wall-clock duration.

use cf_runtime::{Event, EventKind};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct PathSpan {
    pub span_id: u64,
    pub name: String,
    pub depth: usize,
    pub duration_ns: u64,
    /// Self-time = duration minus the sum of child durations (clamped).
    /// For a leaf this equals duration; for a parent it's the time the
    /// parent did its own work between/around its children.
    pub self_ns: u64,
}

#[derive(Default, Clone)]
pub struct CriticalPath {
    pub spans: Vec<PathSpan>,
    pub total_ns: u64,
}

/// Compute the critical path from a chronological event slice. Returns
/// an empty result if there are no usable span pairs.
pub fn compute(events: &[Event]) -> CriticalPath {
    // Pair SpanEnter/SpanExit by span_id to get [name, depth, start, end]
    // for each span.
    struct Built {
        name: String,
        parent_id: Option<u64>,
        depth: usize,
        start: Instant,
        end: Instant,
    }
    let mut starts: HashMap<u64, (String, Option<u64>, usize, Instant)> = HashMap::new();
    let mut spans: HashMap<u64, Built> = HashMap::new();
    let mut depth_of: HashMap<u64, usize> = HashMap::new();
    for e in events {
        match &e.kind {
            EventKind::SpanEnter {
                span_id,
                name,
                parent_id,
                ..
            } => {
                let depth = parent_id
                    .and_then(|p| depth_of.get(&p).copied())
                    .map(|d| d + 1)
                    .unwrap_or(0);
                depth_of.insert(*span_id, depth);
                starts.insert(*span_id, ((*name).to_string(), *parent_id, depth, e.at));
            }
            EventKind::SpanExit { span_id } => {
                if let Some((name, parent_id, depth, start)) = starts.remove(span_id) {
                    spans.insert(
                        *span_id,
                        Built {
                            name,
                            parent_id,
                            depth,
                            start,
                            end: e.at,
                        },
                    );
                }
            }
            _ => {}
        }
    }
    if spans.is_empty() {
        return CriticalPath::default();
    }

    // Build child map.
    let mut children: HashMap<u64, Vec<u64>> = HashMap::new();
    for (&id, b) in &spans {
        if let Some(p) = b.parent_id {
            children.entry(p).or_default().push(id);
        }
    }

    // Compute critical-chain for each span (longest root-to-leaf
    // duration through this subtree). Iterative bottom-up: process
    // spans in deepest-first order so children are computed before
    // parents.
    let mut order: Vec<u64> = spans.keys().copied().collect();
    order.sort_by_key(|id| std::cmp::Reverse(spans[id].depth));

    /// (chain_length_ns, longest_child_id, self_ns)
    type CritEntry = (u64, Option<u64>, u64);
    let mut crit: HashMap<u64, CritEntry> = HashMap::new();

    for id in &order {
        let b = &spans[id];
        let self_dur = (b.end - b.start).as_nanos() as u64;
        let kids = children.get(id).cloned().unwrap_or_default();
        let mut max_child_chain: u64 = 0;
        let mut max_child: Option<u64> = None;
        let mut sum_children: u64 = 0;
        for k in &kids {
            if let Some(kb) = spans.get(k) {
                sum_children =
                    sum_children.saturating_add((kb.end - kb.start).as_nanos() as u64);
            }
            if let Some((c, _, _)) = crit.get(k) {
                if *c > max_child_chain {
                    max_child_chain = *c;
                    max_child = Some(*k);
                }
            }
        }
        let self_ns = self_dur.saturating_sub(sum_children);
        // Critical chain through this span = self_ns + longest child chain.
        let chain = self_ns.saturating_add(max_child_chain);
        crit.insert(*id, (chain, max_child, self_ns));
    }

    // Find the root with the longest chain.
    let mut best_root: Option<u64> = None;
    let mut best_chain: u64 = 0;
    for (&id, b) in &spans {
        if b.parent_id.is_none() {
            let chain = crit.get(&id).map(|(c, _, _)| *c).unwrap_or(0);
            if chain > best_chain {
                best_chain = chain;
                best_root = Some(id);
            }
        }
    }
    let Some(mut cur) = best_root else {
        return CriticalPath::default();
    };

    // Walk down the path collecting nodes.
    let mut path: Vec<PathSpan> = Vec::new();
    let mut total_ns: u64 = 0;
    loop {
        let b = &spans[&cur];
        let entry = crit.get(&cur).copied().unwrap_or((0, None, 0));
        let dur = (b.end - b.start).as_nanos() as u64;
        path.push(PathSpan {
            span_id: cur,
            name: b.name.clone(),
            depth: b.depth,
            duration_ns: dur,
            self_ns: entry.2,
        });
        total_ns = total_ns.saturating_add(entry.2);
        match entry.1 {
            Some(next) => cur = next,
            None => break,
        }
    }
    CriticalPath {
        spans: path,
        total_ns,
    }
}
