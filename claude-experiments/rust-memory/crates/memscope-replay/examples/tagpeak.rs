//! Fold a recording's event stream, reconstructing the live set and a per-thread
//! metadata-scope stack, to answer: at the instant total live bytes peaks, how
//! many of those bytes were allocated under `region=<TAG>` (default
//! `parse_ast`)? That tagged-live-at-peak number is the AST a full magic-string
//! codegen path could drop from the peak. Also reports the best-case floor
//! max_t(total - tagged) — the new peak if the tagged bytes were never resident.
//!
//! Run: cargo run --release -p memscope-replay --example tagpeak -- <file.mscope> [tag]

use std::collections::{HashMap, HashSet};

use memscope_proto::EventKind;
use memscope_replay::{read_recording_raw, stream_events};

fn mb(b: u64) -> f64 {
    b as f64 / 1_048_576.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let file = args.get(1).expect("usage: tagpeak <file.mscope> [tag1,tag2,...]");
    let tag = args.get(2).map(|s| s.as_str()).unwrap_or("parse_ast");
    let wanted: Vec<&str> = tag.split(',').map(|s| s.trim()).collect();

    // Which meta-context ids carry region=<one of the wanted tags>.
    let rec = read_recording_raw(file).expect("read_recording_raw");
    let tag_ids: HashSet<u32> = rec
        .meta
        .iter()
        .filter(|(_, kvs)| {
            kvs.iter()
                .any(|(k, v)| k == "region" && wanted.contains(&v.as_str()))
        })
        .map(|(id, _)| *id)
        .collect();
    eprintln!(
        "meta contexts: {} total, {} carry region in {:?}",
        rec.meta.len(),
        tag_ids.len(),
        wanted
    );

    struct L {
        size: u64,
        tagged: bool,
    }
    let mut live: HashMap<u64, L> = HashMap::new();
    let mut stacks: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut total: u64 = 0;
    let mut tagged: u64 = 0;

    let mut peak_total: u64 = 0;
    let mut tagged_at_peak: u64 = 0;
    // Best-case floor: max over time of (total - tagged).
    let mut peak_net: u64 = 0;
    let mut total_at_peak_net: u64 = 0;

    let mut tagged_alloc_bytes: u64 = 0; // lifetime sum, sanity

    let remove = |live: &mut HashMap<u64, L>, total: &mut u64, tagged: &mut u64, addr: u64| {
        if let Some(old) = live.remove(&addr) {
            *total -= old.size;
            if old.tagged {
                *tagged -= old.size;
            }
        }
    };

    for e in stream_events(file).expect("stream_events") {
        match e.kind {
            EventKind::MetaEnter => {
                stacks.entry(e.thread).or_default().push(e.site);
            }
            EventKind::MetaExit => {
                if let Some(s) = stacks.get_mut(&e.thread) {
                    s.pop();
                }
            }
            EventKind::Alloc | EventKind::ReallocGrow => {
                remove(&mut live, &mut total, &mut tagged, e.addr);
                let is_tagged = stacks
                    .get(&e.thread)
                    .and_then(|s| s.last())
                    .map(|id| tag_ids.contains(id))
                    .unwrap_or(false);
                live.insert(e.addr, L { size: e.size, tagged: is_tagged });
                total += e.size;
                if is_tagged {
                    tagged += e.size;
                    tagged_alloc_bytes += e.size;
                }
                if total > peak_total {
                    peak_total = total;
                    tagged_at_peak = tagged;
                }
                let net = total - tagged;
                if net > peak_net {
                    peak_net = net;
                    total_at_peak_net = total;
                }
            }
            EventKind::Dealloc => {
                remove(&mut live, &mut total, &mut tagged, e.addr);
            }
            EventKind::Mark => {}
        }
    }

    println!("── {file}  (tag: region={tag}) ──");
    println!("peak total live heap        : {:8.1} MB", mb(peak_total));
    println!(
        "  └ tagged ({}) live at peak : {:8.1} MB  ({:.1}% of peak)  ← droppable from the peak",
        tag,
        mb(tagged_at_peak),
        100.0 * tagged_at_peak as f64 / peak_total as f64
    );
    println!(
        "  └ untagged at peak          : {:8.1} MB",
        mb(peak_total - tagged_at_peak)
    );
    println!();
    println!(
        "best-case new peak = max_t(total-tagged): {:8.1} MB  (at an instant whose total was {:.1} MB)",
        mb(peak_net),
        mb(total_at_peak_net)
    );
    println!(
        "  → reduction vs current peak: {:.1} MB ({:.1}%)",
        mb(peak_total - peak_net),
        100.0 * (peak_total - peak_net) as f64 / peak_total as f64
    );
    println!();
    println!("(lifetime tagged alloc bytes, sanity: {:.1} MB)", mb(tagged_alloc_bytes));
}
