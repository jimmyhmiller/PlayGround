//! A real little workload: a load -> index -> query search pipeline, with each
//! stage (and each query) tagged with `meta!` so the recording can be pivoted by
//! subsystem / query / term.
//!
//!   MEMSCOPE_RECORD=/tmp/pipe.mscope cargo run -p demo --release --bin pipeline
//!   memscope flamegraph /tmp/pipe.mscope --no-std --group-by subsystem
//!   memscope flamegraph /tmp/pipe.mscope --no-std --filter subsystem=query --group-by term
//!
//! Everything is deterministic (a tiny LCG), so runs are reproducible.

use std::collections::HashMap;

use memscope::{MemScope, Mode};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

/// Deterministic pseudo-random generator (no external crates, no clock).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.next() as usize) % xs.len()]
    }
}

struct Doc {
    id: u64,
    text: String,
}

const VOCAB: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "memory", "alloc", "heap", "stack", "vector",
    "string", "index", "query", "token", "graph", "node", "edge", "scope", "phase", "trace",
];

/// LOAD: synthesize a corpus of documents.
fn load_corpus(n: usize) -> Vec<Doc> {
    let mut rng = Lcg(0x1234_5678);
    let mut docs = Vec::with_capacity(n);
    for id in 0..n as u64 {
        let words = 12 + (rng.next() as usize % 24);
        let mut text = String::new();
        for _ in 0..words {
            text.push_str(rng.pick(VOCAB));
            text.push(' ');
        }
        docs.push(Doc { id, text });
    }
    docs
}

/// INDEX: tokenize every document into an inverted index term -> doc ids.
fn build_index(docs: &[Doc]) -> HashMap<String, Vec<u64>> {
    let mut index: HashMap<String, Vec<u64>> = HashMap::new();
    for doc in docs {
        for tok in doc.text.split_whitespace() {
            let postings = index.entry(tok.to_string()).or_default();
            if postings.last() != Some(&doc.id) {
                postings.push(doc.id);
            }
        }
    }
    index
}

/// QUERY: intersect the postings of several terms (a tiny AND search).
fn search(index: &HashMap<String, Vec<u64>>, terms: &[&str]) -> Vec<u64> {
    let mut sets: Vec<&Vec<u64>> = terms.iter().filter_map(|t| index.get(*t)).collect();
    if sets.is_empty() {
        return Vec::new();
    }
    sets.sort_by_key(|s| s.len());
    let mut hits: Vec<u64> = sets[0].clone();
    for s in &sets[1..] {
        let set: std::collections::HashSet<u64> = s.iter().copied().collect();
        hits.retain(|id| set.contains(id));
    }
    hits
}

fn main() {
    memscope::set_mode(Mode::Full);
    if let Ok(path) = std::env::var("MEMSCOPE_RECORD") {
        memscope::record_to_file(&path).expect("failed to start recording");
    }

    // LOAD ----------------------------------------------------------------
    let docs = {
        let _m = memscope::meta!(subsystem = "load");
        load_corpus(4000)
    };

    // INDEX ---------------------------------------------------------------
    let index = {
        let _m = memscope::meta!(subsystem = "index");
        build_index(&docs)
    };

    // QUERY ---------------------------------------------------------------
    let queries: &[&[&str]] = &[
        &["memory", "alloc"],
        &["graph", "node", "edge"],
        &["alpha"],
        &["query", "index", "token"],
        &["string", "vector", "heap"],
    ];
    let mut total_hits = 0usize;
    {
        let _m = memscope::meta!(subsystem = "query");
        // Run each query many times so query-stage allocation is substantial.
        for round in 0..200u64 {
            for (qid, terms) in queries.iter().enumerate() {
                let _q = memscope::meta!(query = qid as u64, term = terms[0], round = round);
                let hits = search(&index, terms);
                total_hits += hits.len();
                std::hint::black_box(&hits);
            }
        }
    }

    // Flush the reconstructor so stats() is exact, then report.
    let s = memscope::stats();
    println!(
        "pipeline done: {} docs, {} index terms, {} query hits",
        docs.len(),
        index.len(),
        total_hits
    );
    println!(
        "  tracked: {} allocations, {} total allocated, {} KiB still live",
        s.total_allocs,
        human(s.total_alloc_bytes),
        s.live_bytes / 1024,
    );
    if let Ok(path) = std::env::var("MEMSCOPE_RECORD") {
        // Give the pump a moment to flush the tail to disk.
        std::thread::sleep(std::time::Duration::from_millis(100));
        println!("  recording written to {path}");
        println!("  try: memscope flamegraph {path} --no-std --group-by subsystem");
    }
}

fn human(b: u64) -> String {
    const U: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let (mut v, mut i) = (b as f64, 0);
    while v >= 1024.0 && i < 3 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", U[i])
}
