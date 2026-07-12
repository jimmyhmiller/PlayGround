//! Loading real graphs from disk. Supports whitespace-separated edge lists
//! (`u v` per line, `#`/`//` comments ignored) — the de-facto format for SNAP,
//! Konect, and most graph datasets. Node ids may be arbitrary integers; they
//! are remapped to a compact `0..n` range.

use crate::graph::{Graph, NodeId};
use ahash::AHashMap;
use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Load an edge list. Arbitrary integer node ids are compacted; the returned
/// `id_map` (compact id -> original id) is provided for labeling if needed.
pub fn load_edge_list(path: impl AsRef<Path>) -> Result<(Graph, Vec<u64>)> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file)? };
    parse_edge_list_bytes(&mmap)
}

/// Parse an edge list from raw bytes. Broken out for testability.
pub fn parse_edge_list_bytes(bytes: &[u8]) -> Result<(Graph, Vec<u64>)> {
    let mut remap: AHashMap<u64, NodeId> = AHashMap::with_capacity(1 << 16);
    let mut originals: Vec<u64> = Vec::new();
    let mut edges: Vec<[NodeId; 2]> = Vec::new();

    let mut intern = |orig: u64, remap: &mut AHashMap<u64, NodeId>, originals: &mut Vec<u64>| -> NodeId {
        if let Some(&id) = remap.get(&orig) {
            id
        } else {
            let id = originals.len() as NodeId;
            originals.push(orig);
            remap.insert(orig, id);
            id
        }
    };

    for line in bytes.split(|&b| b == b'\n') {
        let line = trim(line);
        if line.is_empty() || line[0] == b'#' || (line.len() >= 2 && &line[0..2] == b"//") {
            continue;
        }
        let mut it = line.split(|b| b.is_ascii_whitespace()).filter(|s| !s.is_empty());
        let (Some(a), Some(b)) = (it.next(), it.next()) else {
            continue; // skip malformed lines rather than aborting the whole load
        };
        let (Some(ua), Some(ub)) = (parse_u64(a), parse_u64(b)) else {
            continue;
        };
        let ia = intern(ua, &mut remap, &mut originals);
        let ib = intern(ub, &mut remap, &mut originals);
        if ia != ib {
            edges.push([ia, ib]);
        }
    }

    let n = originals.len() as u64;
    Ok((Graph::new(n, edges), originals))
}

#[inline]
fn trim(mut s: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = s {
        if first.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    while let [rest @ .., last] = s {
        if last.is_ascii_whitespace() || *last == b'\r' {
            s = rest;
        } else {
            break;
        }
    }
    s
}

#[inline]
fn parse_u64(s: &[u8]) -> Option<u64> {
    if s.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    for &c in s {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add((c - b'0') as u64)?;
    }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_with_comments_and_remap() {
        let data = b"# comment\n100 200\n200 300\n// another\n100 300\n";
        let (g, originals) = parse_edge_list_bytes(data).unwrap();
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.num_edges(), 3);
        // Originals preserved in first-seen order.
        assert_eq!(originals, vec![100, 200, 300]);
    }

    #[test]
    fn skips_self_loops_and_malformed() {
        let data = b"5 5\n1 2\ngarbage line here\n3\n2 3\n";
        let (g, _) = parse_edge_list_bytes(data).unwrap();
        // self-loop 5-5 dropped; "3" malformed; only 1-2 and 2-3 kept.
        assert_eq!(g.num_edges(), 2);
    }
}
