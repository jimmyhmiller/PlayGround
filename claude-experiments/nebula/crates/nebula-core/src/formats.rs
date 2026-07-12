//! Multi-format graph loading. Detects the format from the file extension (with
//! a content sniff as backup) and dispatches to the right parser. Node ids of any
//! kind — sparse integers or arbitrary strings — are compacted to a dense
//! `0..n` range, and the original labels are returned for display/picking.
//!
//! Supported:
//!   - **edge list** (`.txt .edges .el`) and **CSV** (`.csv`): `u v` per line
//!   - **Matrix Market** (`.mtx`): coordinate format, 1-indexed
//!   - **DIMACS** (`.gr .dimacs .col`): `p ...` / `e u v` / `a u v w`
//!   - **GML** (`.gml`): `edge [ source A target B ]`
//!   - **DOT / Graphviz** (`.dot .gv`): `a -- b` / `a -> b`
//!   - **JSON** (`.json`): d3-style `{nodes, links}` or `{edges:[[u,v]]}`
//!   - **adjacency list** (lines like `node: n1 n2 n3`)

use crate::graph::{Graph, NodeId};
use ahash::AHashMap;
use anyhow::{anyhow, Context, Result};
use std::path::Path;

/// A loaded graph plus the original node labels (indexed by compact id).
pub struct Loaded {
    pub graph: Graph,
    pub labels: Vec<String>,
    /// Optional per-node attributes (key/value), indexed by compact id. Populated
    /// by formats that carry node metadata (currently JSON); empty otherwise.
    pub attrs: Vec<Vec<(String, String)>>,
    pub format: &'static str,
    /// True if the source declared itself directed (informational only; we store
    /// an undirected edge list regardless).
    pub directed: bool,
}

/// Interns arbitrary string labels into compact `NodeId`s in first-seen order.
#[derive(Default)]
struct Interner {
    map: AHashMap<String, NodeId>,
    labels: Vec<String>,
}

impl Interner {
    fn intern(&mut self, key: &str) -> NodeId {
        if let Some(&id) = self.map.get(key) {
            id
        } else {
            let id = self.labels.len() as NodeId;
            self.labels.push(key.to_string());
            self.map.insert(key.to_string(), id);
            id
        }
    }
    /// Ensure at least `n` nodes exist even if some are isolated (no edges).
    fn ensure_count(&mut self, n: u64) {
        while (self.labels.len() as u64) < n {
            let id = self.labels.len();
            let key = id.to_string();
            self.map.insert(key.clone(), id as NodeId);
            self.labels.push(key);
        }
    }
}

pub fn load(path: impl AsRef<Path>) -> Result<Loaded> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    load_bytes(&bytes, &ext)
}

/// Parse already-read bytes, choosing a parser by extension then content sniff.
pub fn load_bytes(bytes: &[u8], ext: &str) -> Result<Loaded> {
    let text = String::from_utf8_lossy(bytes);
    let format = detect(ext, &text);
    match format {
        "mtx" => parse_matrix_market(&text),
        "dimacs" => parse_dimacs(&text),
        "gml" => parse_gml(&text),
        "dot" => parse_dot(&text),
        "json" => parse_json(&text),
        "adjacency" => parse_adjacency(&text),
        _ => parse_edge_list(&text),
    }
}

fn detect(ext: &str, text: &str) -> &'static str {
    match ext {
        "mtx" => return "mtx",
        "gr" | "dimacs" | "col" => return "dimacs",
        "gml" => return "gml",
        "dot" | "gv" => return "dot",
        "json" => return "json",
        _ => {}
    }
    // Content sniff on the first meaningful bytes.
    let head: String = text.chars().take(512).collect();
    let trimmed = head.trim_start();
    if trimmed.starts_with("%%MatrixMarket") {
        "mtx"
    } else if trimmed.starts_with('{') || trimmed.starts_with('[') {
        "json"
    } else if trimmed.starts_with("graph") || trimmed.starts_with("digraph") || trimmed.contains("->") || trimmed.contains("--") {
        // could be dot; also GML starts with "graph [".
        if trimmed.contains('[') && (trimmed.contains("node") || trimmed.contains("edge") || trimmed.starts_with("graph [")) {
            "gml"
        } else {
            "dot"
        }
    } else if text.lines().take(20).any(|l| {
        let l = l.trim();
        !l.is_empty() && !l.starts_with('#') && l.contains(':')
    }) {
        "adjacency"
    } else {
        "edgelist"
    }
}

// --------------------------------------------------------------------------

fn finish(mut interner: Interner, edges: Vec<[NodeId; 2]>, declared_nodes: Option<u64>, format: &'static str, directed: bool) -> Loaded {
    if let Some(n) = declared_nodes {
        interner.ensure_count(n);
    }
    let n = interner.labels.len() as u64;
    Loaded {
        graph: Graph::new(n, edges),
        labels: interner.labels,
        attrs: Vec::new(),
        format,
        directed,
    }
}

fn parse_edge_list(text: &str) -> Result<Loaded> {
    let mut it = Interner::default();
    let mut edges = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") || line.starts_with('%') {
            continue;
        }
        let mut parts = line.split(|c: char| c.is_whitespace() || c == ',' || c == ';').filter(|s| !s.is_empty());
        let (Some(a), Some(b)) = (parts.next(), parts.next()) else { continue };
        let ia = it.intern(a);
        let ib = it.intern(b);
        if ia != ib {
            edges.push([ia, ib]);
        }
    }
    Ok(finish(it, edges, None, "edge-list", false))
}

fn parse_matrix_market(text: &str) -> Result<Loaded> {
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("");
    let symmetric = header.to_ascii_lowercase().contains("symmetric");
    let mut it = Interner::default();
    let mut edges = Vec::new();
    let mut dims: Option<(u64, u64)> = None;
    for line in lines {
        let l = line.trim();
        if l.is_empty() || l.starts_with('%') {
            continue;
        }
        let mut p = l.split_whitespace();
        if dims.is_none() {
            // "rows cols nnz"
            let rows: u64 = p.next().and_then(|s| s.parse().ok()).ok_or_else(|| anyhow!("mtx: bad size line"))?;
            let cols: u64 = p.next().and_then(|s| s.parse().ok()).unwrap_or(rows);
            dims = Some((rows, cols));
            continue;
        }
        let (Some(a), Some(b)) = (p.next(), p.next()) else { continue };
        // 1-indexed.
        let (Ok(ua), Ok(ub)) = (a.parse::<u64>(), b.parse::<u64>()) else { continue };
        if ua == 0 || ub == 0 {
            continue;
        }
        let ia = it.intern(&(ua - 1).to_string());
        let ib = it.intern(&(ub - 1).to_string());
        if ia != ib {
            edges.push([ia, ib]);
        }
    }
    let n = dims.map(|(r, c)| r.max(c));
    let _ = symmetric; // we always treat as undirected
    Ok(finish(it, edges, n, "matrix-market", false))
}

fn parse_dimacs(text: &str) -> Result<Loaded> {
    let mut it = Interner::default();
    let mut edges = Vec::new();
    let mut declared = None;
    let mut directed = false;
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('c') {
            continue;
        }
        let mut p = l.split_whitespace();
        match p.next() {
            Some("p") => {
                // "p edge N M" or "p sp N M"
                let ty = p.next().unwrap_or("");
                directed = ty.eq_ignore_ascii_case("sp");
                if let Some(n) = p.next().and_then(|s| s.parse::<u64>().ok()) {
                    declared = Some(n);
                }
            }
            Some("e") | Some("a") => {
                let (Some(a), Some(b)) = (p.next(), p.next()) else { continue };
                let (Ok(ua), Ok(ub)) = (a.parse::<u64>(), b.parse::<u64>()) else { continue };
                if ua == 0 || ub == 0 {
                    continue;
                }
                let ia = it.intern(&(ua - 1).to_string());
                let ib = it.intern(&(ub - 1).to_string());
                if ia != ib {
                    edges.push([ia, ib]);
                }
            }
            _ => {}
        }
    }
    Ok(finish(it, edges, declared, "dimacs", directed))
}

fn parse_gml(text: &str) -> Result<Loaded> {
    // Tokenize on whitespace and brackets; scan for `edge [ ... source X target Y ... ]`.
    let mut it = Interner::default();
    let mut edges = Vec::new();
    let toks: Vec<&str> = text
        .split(|c: char| c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == "edge" {
            // Find source/target within the following bracket block.
            let mut src: Option<String> = None;
            let mut dst: Option<String> = None;
            let mut j = i + 1;
            let mut depth = 0i32;
            while j < toks.len() {
                match toks[j] {
                    "[" => depth += 1,
                    "]" => {
                        depth -= 1;
                        if depth <= 0 {
                            break;
                        }
                    }
                    "source" => {
                        if let Some(v) = toks.get(j + 1) {
                            src = Some(v.trim_matches('"').to_string());
                        }
                    }
                    "target" => {
                        if let Some(v) = toks.get(j + 1) {
                            dst = Some(v.trim_matches('"').to_string());
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if let (Some(a), Some(b)) = (src, dst) {
                let ia = it.intern(&a);
                let ib = it.intern(&b);
                if ia != ib {
                    edges.push([ia, ib]);
                }
            }
            i = j;
        }
        i += 1;
    }
    Ok(finish(it, edges, None, "gml", false))
}

fn parse_dot(text: &str) -> Result<Loaded> {
    let mut it = Interner::default();
    let mut edges = Vec::new();
    let directed = text.contains("digraph");
    // Work on the body between the outermost braces, so the `digraph G {` header
    // and trailing `}` don't leak into node ids. Statements are `;`-separated.
    let body = match (text.find('{'), text.rfind('}')) {
        (Some(a), Some(b)) if b > a => &text[a + 1..b],
        _ => text,
    };
    for raw in body.split(';') {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        {
            let stmt = line;
            if !stmt.contains("->") && !stmt.contains("--") {
                continue;
            }
            // Normalize edge ops to a single delimiter and split into a chain.
            let norm = stmt.replace("->", "\u{1}").replace("--", "\u{1}");
            let nodes: Vec<String> = norm
                .split('\u{1}')
                .map(|s| clean_dot_id(s))
                .filter(|s| !s.is_empty())
                .collect();
            for w in nodes.windows(2) {
                let ia = it.intern(&w[0]);
                let ib = it.intern(&w[1]);
                if ia != ib {
                    edges.push([ia, ib]);
                }
            }
        }
    }
    Ok(finish(it, edges, None, "dot", directed))
}

fn clean_dot_id(s: &str) -> String {
    // Drop attribute blocks `[...]` and surrounding quotes/whitespace.
    let s = s.split('[').next().unwrap_or("").trim();
    s.trim_matches('"').trim().to_string()
}

fn parse_json(text: &str) -> Result<Loaded> {
    let v: serde_json::Value = serde_json::from_str(text).context("parsing JSON graph")?;
    let mut it = Interner::default();
    let mut edges = Vec::new();

    let id_of = |val: &serde_json::Value| -> Option<String> {
        match val {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            _ => None,
        }
    };

    // Format a scalar JSON value for display as an attribute; skip containers.
    let scalar = |val: &serde_json::Value| -> Option<String> {
        match val {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(b.to_string()),
            serde_json::Value::Null => Some("null".to_string()),
            _ => None,
        }
    };

    // Register explicit nodes first (preserves isolated nodes + ordering), and
    // capture their scalar fields as per-node attributes (keyed by compact id).
    let mut raw_attrs: Vec<(NodeId, Vec<(String, String)>)> = Vec::new();
    if let Some(nodes) = v.get("nodes").and_then(|n| n.as_array()) {
        for node in nodes {
            let key = node
                .get("id")
                .and_then(&id_of)
                .or_else(|| id_of(node));
            if let Some(k) = key {
                let cid = it.intern(&k);
                if let Some(obj) = node.as_object() {
                    let a: Vec<(String, String)> = obj
                        .iter()
                        .filter(|(k, _)| k.as_str() != "id" && k.as_str() != "children")
                        .filter_map(|(k, v)| scalar(v).map(|s| (k.clone(), s)))
                        .collect();
                    if !a.is_empty() {
                        raw_attrs.push((cid, a));
                    }
                }
            }
        }
    }

    // Edges may be under "links" (d3), "edges", with source/target or as pairs.
    let edge_arrays = ["links", "edges"];
    for key in edge_arrays {
        if let Some(arr) = v.get(key).and_then(|e| e.as_array()) {
            for e in arr {
                let (a, b) = match e {
                    serde_json::Value::Array(pair) if pair.len() >= 2 => {
                        (id_of(&pair[0]), id_of(&pair[1]))
                    }
                    serde_json::Value::Object(_) => (
                        e.get("source").and_then(&id_of),
                        e.get("target").and_then(&id_of),
                    ),
                    _ => (None, None),
                };
                if let (Some(a), Some(b)) = (a, b) {
                    let ia = it.intern(&a);
                    let ib = it.intern(&b);
                    if ia != ib {
                        edges.push([ia, ib]);
                    }
                }
            }
        }
    }
    let mut loaded = finish(it, edges, None, "json", false);
    if !raw_attrs.is_empty() {
        let mut attrs = vec![Vec::new(); loaded.labels.len()];
        for (cid, a) in raw_attrs {
            if let Some(slot) = attrs.get_mut(cid as usize) {
                *slot = a;
            }
        }
        loaded.attrs = attrs;
    }
    Ok(loaded)
}

fn parse_adjacency(text: &str) -> Result<Loaded> {
    let mut it = Interner::default();
    let mut edges = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }
        // "node: n1 n2 n3"  or  "node n1 n2 n3"
        let (head, rest) = if let Some((h, r)) = line.split_once(':') {
            (h.trim(), r)
        } else {
            let mut parts = line.splitn(2, char::is_whitespace);
            let h = parts.next().unwrap_or("").trim();
            (h, parts.next().unwrap_or(""))
        };
        if head.is_empty() {
            continue;
        }
        let src = it.intern(head);
        for nb in rest.split(|c: char| c.is_whitespace() || c == ',').filter(|s| !s.is_empty()) {
            let dst = it.intern(nb);
            if src != dst {
                edges.push([src, dst]);
            }
        }
    }
    Ok(finish(it, edges, None, "adjacency-list", false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_and_parse_mtx() {
        let s = "%%MatrixMarket matrix coordinate pattern symmetric\n% a comment\n3 3 2\n1 2\n2 3\n";
        let l = load_bytes(s.as_bytes(), "mtx").unwrap();
        assert_eq!(l.format, "matrix-market");
        assert_eq!(l.graph.num_nodes(), 3);
        assert_eq!(l.graph.num_edges(), 2);
    }

    #[test]
    fn parse_dot_chain() {
        let s = "digraph G { a -> b -> c; a -> c; }";
        let l = load_bytes(s.as_bytes(), "dot").unwrap();
        assert_eq!(l.format, "dot");
        assert_eq!(l.graph.num_nodes(), 3);
        assert_eq!(l.graph.num_edges(), 3);
        assert!(l.directed);
    }

    #[test]
    fn parse_gml_edges() {
        let s = "graph [ node [ id 0 ] node [ id 1 ] node [ id 2 ] edge [ source 0 target 1 ] edge [ source 1 target 2 ] ]";
        let l = load_bytes(s.as_bytes(), "gml").unwrap();
        assert_eq!(l.format, "gml");
        assert_eq!(l.graph.num_edges(), 2);
    }

    #[test]
    fn parse_json_d3() {
        let s = r#"{"nodes":[{"id":"a"},{"id":"b"},{"id":"c"}],"links":[{"source":"a","target":"b"},{"source":"b","target":"c"}]}"#;
        let l = load_bytes(s.as_bytes(), "json").unwrap();
        assert_eq!(l.format, "json");
        assert_eq!(l.graph.num_nodes(), 3);
        assert_eq!(l.graph.num_edges(), 2);
        assert_eq!(l.labels[0], "a");
    }

    #[test]
    fn parse_dimacs_graph() {
        let s = "c comment\np edge 4 3\ne 1 2\ne 2 3\ne 3 4\n";
        let l = load_bytes(s.as_bytes(), "gr").unwrap();
        assert_eq!(l.graph.num_nodes(), 4);
        assert_eq!(l.graph.num_edges(), 3);
    }

    #[test]
    fn parse_adjacency_list() {
        let s = "a: b c\nb: c\n";
        let l = load_bytes(s.as_bytes(), "adj").unwrap();
        assert_eq!(l.format, "adjacency-list");
        assert_eq!(l.graph.num_edges(), 3);
    }
}
