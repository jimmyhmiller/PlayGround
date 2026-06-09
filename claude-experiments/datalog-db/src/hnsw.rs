//! HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour
//! index — Malkov & Yashunin, 2018.
//!
//! Brute-force kNN is O(n·d) per query; fine below ~1M vectors but linear.
//! HNSW builds a multi-layer navigable graph that gives ~O(log n) search with
//! high recall. The index is held in memory and built from the stored vectors
//! (which live durably in EAVT) — so it's a cache, not a separate source of
//! truth, and never needs its own persistence/corruption story.
//!
//! Level assignment uses a *seeded deterministic* PRNG so a given set of
//! inserts always yields the same graph — reproducible, no wall-clock or
//! global RNG.

use crate::datom::EntityId;
use crate::query::VectorMetric;

/// A small deterministic PRNG (SplitMix64) for level assignment. Seeded per
/// index so builds are reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform f64 in (0, 1].
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa; +1 / +1 keeps it in (0, 1] so ln() is finite.
        let v = (self.next_u64() >> 11) as f64;
        (v + 1.0) / 9007199254740992.0
    }
}

/// Build/search parameters. Defaults follow the common HNSW recommendations.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Max neighbours per node on layers > 0.
    pub m: usize,
    /// Max neighbours per node on layer 0 (typically 2*M).
    pub m0: usize,
    /// Candidate-list width during construction.
    pub ef_construction: usize,
    /// Level-generation normalization factor (1 / ln(M)).
    pub ml: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

/// A node: its vector and per-layer adjacency (layer 0 at index 0).
struct Node {
    entity: EntityId,
    vector: Vec<f32>,
    /// neighbours[layer] = neighbour node indices on that layer.
    neighbours: Vec<Vec<usize>>,
}

/// An in-memory HNSW index over f32 vectors keyed by `EntityId`.
pub struct Hnsw {
    nodes: Vec<Node>,
    entry: Option<usize>,
    max_layer: usize,
    metric: VectorMetric,
    params: HnswParams,
    rng: SplitMix64,
}

impl Hnsw {
    pub fn new(metric: VectorMetric, params: HnswParams, seed: u64) -> Self {
        Self {
            nodes: Vec::new(),
            entry: None,
            max_layer: 0,
            metric,
            params,
            // Avoid a zero seed (SplitMix64 still works, but vary it).
            rng: SplitMix64(seed ^ 0xA5A5_5A5A_DEAD_BEEF),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Distance as "smaller = closer" for graph navigation. The query
    /// `VectorMetric::score` returns "higher = closer", so we negate it. NaN
    /// (e.g. dimension mismatch) sorts as farthest.
    #[inline]
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        let s = self.metric.score(a, b);
        if s.is_nan() {
            f32::INFINITY
        } else {
            -s
        }
    }

    fn random_level(&mut self) -> usize {
        // L = floor(-ln(U) * ml)
        let u = self.rng.next_unit();
        (-u.ln() * self.params.ml).floor() as usize
    }

    /// Insert a vector. Duplicate entity ids are allowed (the caller dedups);
    /// nothing here assumes uniqueness.
    pub fn insert(&mut self, entity: EntityId, vector: Vec<f32>) {
        let level = self.random_level();
        let node_idx = self.nodes.len();
        self.nodes.push(Node {
            entity,
            vector,
            neighbours: vec![Vec::new(); level + 1],
        });

        let entry = match self.entry {
            None => {
                // First node becomes the entry point.
                self.entry = Some(node_idx);
                self.max_layer = level;
                return;
            }
            Some(e) => e,
        };

        let q = self.nodes[node_idx].vector.clone();
        let mut ep = entry;

        // Phase 1: descend from the top down to level+1 with greedy ef=1 search.
        let top = self.max_layer;
        let mut lc = top;
        while lc > level {
            ep = self.greedy_nearest(&q, ep, lc);
            if lc == 0 {
                break;
            }
            lc -= 1;
        }

        // Phase 2: from min(top, level) down to 0, ef-search and connect.
        let mut lc = top.min(level);
        loop {
            let ef = self.params.ef_construction;
            let mut candidates = self.search_layer(&q, &[ep], ef, lc);
            // Choose M (or M0 on layer 0) neighbours by the heuristic.
            let m = if lc == 0 { self.params.m0 } else { self.params.m };
            let selected = self.select_neighbours(&q, &mut candidates, m);

            // Connect node -> selected, and selected -> node (pruning to bound).
            self.nodes[node_idx].neighbours[lc] = selected.clone();
            for &nbr in &selected {
                self.nodes[nbr].neighbours[lc].push(node_idx);
                let bound = if lc == 0 { self.params.m0 } else { self.params.m };
                if self.nodes[nbr].neighbours[lc].len() > bound {
                    self.prune(nbr, lc, bound);
                }
            }

            // Re-seed the entry for the next-lower layer with the closest found.
            if let Some(&(_, best)) = candidates.first() {
                ep = best;
            }
            if lc == 0 {
                break;
            }
            lc -= 1;
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry = Some(node_idx);
        }
    }

    /// Greedy single-best descent on one layer: walk to a strictly-closer
    /// neighbour until none improves.
    fn greedy_nearest(&self, q: &[f32], start: usize, layer: usize) -> usize {
        let mut cur = start;
        let mut cur_d = self.dist(q, &self.nodes[cur].vector);
        loop {
            let mut improved = false;
            for &nbr in &self.nodes[cur].neighbours[layer] {
                let d = self.dist(q, &self.nodes[nbr].vector);
                if d < cur_d {
                    cur_d = d;
                    cur = nbr;
                    improved = true;
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// ef-bounded best-first search on `layer`, seeded by `entry_points`.
    /// Returns (dist, node) sorted ascending by distance.
    fn search_layer(
        &self,
        q: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, usize)> {
        use std::collections::HashSet;

        let mut visited: HashSet<usize> = HashSet::new();
        // Candidate frontier (min-by-dist) and result set (we keep a sorted Vec
        // since ef is small; a heap is overkill at these sizes).
        let mut candidates: Vec<(f32, usize)> = Vec::new();
        let mut results: Vec<(f32, usize)> = Vec::new();

        for &ep in entry_points {
            let d = self.dist(q, &self.nodes[ep].vector);
            visited.insert(ep);
            candidates.push((d, ep));
            results.push((d, ep));
        }
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        while let Some((cd, c)) = pop_min(&mut candidates) {
            // Stop when the closest candidate is worse than the current worst
            // result (and we already have ef results).
            let worst = results.last().map(|x| x.0).unwrap_or(f32::INFINITY);
            if cd > worst && results.len() >= ef {
                break;
            }
            for &nbr in &self.nodes[c].neighbours[layer] {
                if !visited.insert(nbr) {
                    continue;
                }
                let d = self.dist(q, &self.nodes[nbr].vector);
                let worst = results.last().map(|x| x.0).unwrap_or(f32::INFINITY);
                if d < worst || results.len() < ef {
                    insert_sorted(&mut candidates, (d, nbr));
                    insert_sorted(&mut results, (d, nbr));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        results
    }

    /// Neighbour selection: the simple "keep the M closest" heuristic. (The
    /// fancier diversity heuristic helps recall marginally; the closest-M rule
    /// is standard, simple, and works well.)
    fn select_neighbours(
        &self,
        _q: &[f32],
        candidates: &mut [(f32, usize)],
        m: usize,
    ) -> Vec<usize> {
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.iter().take(m).map(|&(_, n)| n).collect()
    }

    /// Prune a node's neighbour list on `layer` back to the `bound` closest.
    fn prune(&mut self, node: usize, layer: usize, bound: usize) {
        let v = self.nodes[node].vector.clone();
        let mut scored: Vec<(f32, usize)> = self.nodes[node].neighbours[layer]
            .iter()
            .map(|&n| (self.dist(&v, &self.nodes[n].vector), n))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(bound);
        self.nodes[node].neighbours[layer] = scored.into_iter().map(|(_, n)| n).collect();
    }

    /// Search for the `k` nearest entities to `query`. `ef` controls
    /// accuracy/speed (larger = better recall, slower); it's clamped to be at
    /// least `k`. Returns (entity, similarity) best-first, where similarity is
    /// the metric's "higher = closer" score (matching the brute-force path).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(EntityId, f32)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let ef = ef.max(k).max(1);

        // Descend greedily from the top layer to layer 1.
        let mut ep = entry;
        let mut lc = self.max_layer;
        while lc > 0 {
            ep = self.greedy_nearest(query, ep, lc);
            lc -= 1;
        }
        // ef-search on layer 0.
        let mut results = self.search_layer(query, &[ep], ef, 0);
        results.truncate(k);
        results
            .into_iter()
            .map(|(d, n)| (self.nodes[n].entity, -d)) // back to "higher = closer"
            .collect()
    }
}

/// Pop the minimum-distance element from a distance-sorted-ascending Vec.
fn pop_min(v: &mut Vec<(f32, usize)>) -> Option<(f32, usize)> {
    if v.is_empty() {
        None
    } else {
        Some(v.remove(0))
    }
}

/// Insert keeping the Vec sorted ascending by distance.
fn insert_sorted(v: &mut Vec<(f32, usize)>, item: (f32, usize)) {
    let pos = v
        .binary_search_by(|probe| probe.0.partial_cmp(&item.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or_else(|e| e);
    v.insert(pos, item);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_topk(vecs: &[(EntityId, Vec<f32>)], q: &[f32], k: usize, m: VectorMetric) -> Vec<EntityId> {
        let mut scored: Vec<(f32, EntityId)> =
            vecs.iter().map(|(e, v)| (m.score(q, v), *e)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(k).map(|(_, e)| e).collect()
    }

    #[test]
    fn high_recall_vs_brute_force() {
        // Deterministic pseudo-random vectors via SplitMix64.
        let mut rng = SplitMix64(12345);
        let dim = 32;
        let n = 500;
        let vecs: Vec<(EntityId, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|_| (rng.next_unit() as f32) * 2.0 - 1.0)
                    .collect();
                (i as EntityId + 1, v)
            })
            .collect();

        let metric = VectorMetric::Cosine;
        let mut index = Hnsw::new(metric, HnswParams::default(), 42);
        for (e, v) in &vecs {
            index.insert(*e, v.clone());
        }
        assert_eq!(index.len(), n);

        // Recall@10 over several queries should be high.
        let mut hits = 0usize;
        let mut total = 0usize;
        for qi in 0..20 {
            let q = &vecs[qi * 13 % n].1;
            let truth = brute_topk(&vecs, q, 10, metric);
            let got: Vec<EntityId> = index.search(q, 10, 64).into_iter().map(|(e, _)| e).collect();
            for t in &truth {
                if got.contains(t) {
                    hits += 1;
                }
                total += 1;
            }
        }
        let recall = hits as f32 / total as f32;
        assert!(recall > 0.90, "recall@10 was {recall:.3}, expected > 0.90");
    }

    #[test]
    fn finds_exact_match_first() {
        let mut index = Hnsw::new(VectorMetric::Cosine, HnswParams::default(), 7);
        index.insert(1, vec![1.0, 0.0, 0.0]);
        index.insert(2, vec![0.0, 1.0, 0.0]);
        index.insert(3, vec![0.0, 0.0, 1.0]);
        let r = index.search(&[1.0, 0.0, 0.0], 1, 16);
        assert_eq!(r[0].0, 1);
        assert!((r[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_index() {
        let index = Hnsw::new(VectorMetric::Dot, HnswParams::default(), 1);
        assert!(index.search(&[1.0, 2.0], 5, 16).is_empty());
    }
}
