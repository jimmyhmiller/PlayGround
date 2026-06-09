//! Full-text search: tokenization and BM25 ranking.
//!
//! A `fulltext` string field is tokenized into terms and stored in an inverted
//! index (term → postings of `(entity, term_frequency)`), plus per-document
//! field lengths and corpus-level stats. Queries tokenize the same way, walk
//! the postings, and rank with Okapi BM25.
//!
//! The tokenizer is intentionally simple and dependency-free: Unicode-aware
//! lowercasing, split on non-alphanumeric, a small English stopword set, and a
//! light suffix stemmer so `recursion`/`recursive`/`recurse` collapse toward a
//! shared stem. It is NOT a linguistic stemmer — it trades precision for the
//! recall that makes "search" feel different from substring matching.

use std::collections::HashMap;

/// Standard BM25 term-frequency saturation parameter.
pub const BM25_K1: f32 = 1.2;
/// Standard BM25 length-normalization parameter.
pub const BM25_B: f32 = 0.75;

/// A compact English stopword list. Kept small on purpose — over-aggressive
/// stoplists hurt phrase-like queries.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with", "we", "you", "i", "he", "she", "him", "her", "its", "our", "from",
];

/// Tokenize text into a list of normalized terms (in order, with duplicates —
/// the caller counts frequencies). Lowercases, splits on non-alphanumeric,
/// drops stopwords and 1-char tokens, and stems.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            push_term(&mut out, std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        push_term(&mut out, cur);
    }
    out
}

fn push_term(out: &mut Vec<String>, term: String) {
    if term.len() < 2 {
        return;
    }
    if STOPWORDS.contains(&term.as_str()) {
        return;
    }
    out.push(stem(&term));
}

/// A deliberately light suffix stemmer. Strips a handful of common English
/// inflections so morphological variants share a posting list. Order matters:
/// longer suffixes first. Only applied when enough stem remains (≥3 chars).
pub fn stem(term: &str) -> String {
    // Only stem ASCII-alphabetic words; leave numbers / unicode tokens alone.
    if !term.bytes().all(|b| b.is_ascii_alphabetic()) {
        return term.to_string();
    }
    let strip = |w: &str, suffix: &str, keep: usize| -> Option<String> {
        if w.len() > suffix.len() + keep - 1 && w.ends_with(suffix) {
            Some(w[..w.len() - suffix.len()].to_string())
        } else {
            None
        }
    };
    let mut w = term.to_string();
    // Common English inflections, longest suffix first so e.g. "ization" wins
    // over "ion". `keep` is the minimum stem length to leave behind.
    for (suf, keep) in [
        ("ization", 4),
        ("iveness", 4),
        ("fulness", 4),
        ("ousness", 4),
        ("ational", 5),
        ("tional", 4),
        ("ically", 4),
        ("ation", 4),
        ("ings", 3),
        ("ness", 3),
        ("ment", 4),
        ("ions", 3),
        ("ing", 3),
        ("ies", 3),
        ("ied", 3),
        ("ive", 3),
        ("ion", 3),
        ("ity", 3),
        ("ers", 3),
        ("er", 3),
        ("ed", 3),
        ("es", 3),
        ("ly", 3),
        ("s", 3),
    ] {
        if let Some(stemmed) = strip(&w, suf, keep) {
            w = stemmed;
            break;
        }
    }
    // Normalize a trailing silent "e": "compute" → "comput" so it shares a stem
    // with "computing"/"computational" (which lose the "e" when "ing"/"ation"
    // is stripped). Only when enough stem remains.
    if w.len() > 4 && w.ends_with('e') {
        w.pop();
    }
    w
}

/// Count term frequencies in a tokenized field. Returns (term → freq, total
/// token count). The total length is the BM25 document length.
pub fn term_frequencies(text: &str) -> (HashMap<String, u32>, u32) {
    let toks = tokenize(text);
    let len = toks.len() as u32;
    let mut tf: HashMap<String, u32> = HashMap::new();
    for t in toks {
        *tf.entry(t).or_insert(0) += 1;
    }
    (tf, len)
}

/// Okapi BM25 contribution of a single query term to a document's score.
///
/// * `tf`     — frequency of the term in the document field.
/// * `doc_len`— length (token count) of the document field.
/// * `avgdl`  — average field length across the corpus.
/// * `n_docs` — number of documents that have this field.
/// * `df`     — number of documents containing the term (document frequency).
pub fn bm25_term_score(tf: u32, doc_len: u32, avgdl: f32, n_docs: u64, df: u64) -> f32 {
    if tf == 0 || df == 0 {
        return 0.0;
    }
    // IDF with the standard BM25 +0.5 smoothing; clamp at 0 so ultra-common
    // terms can't push a negative score.
    let idf = (((n_docs as f32 - df as f32 + 0.5) / (df as f32 + 0.5)) + 1.0).ln();
    let tf = tf as f32;
    let norm = tf * (BM25_K1 + 1.0)
        / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len as f32 / avgdl.max(1.0))));
    idf.max(0.0) * norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_basics() {
        let toks = tokenize("The Recursive Functions of LISP!");
        // "the" and "of" are stopwords and must be dropped.
        assert!(!toks.contains(&"the".to_string()));
        assert!(!toks.contains(&"of".to_string()));
        // "recursive" and "functions" survive (in stemmed form). The query
        // term stems the same way, so searching finds them.
        assert!(toks.contains(&stem("recursive")));
        assert!(toks.contains(&stem("functions")));
        // lowercased.
        assert!(toks.contains(&stem("lisp")));
    }

    #[test]
    fn stemming_collapses_variants() {
        // The point isn't linguistic correctness, just that variants share a stem.
        let a = stem("recursion");
        let b = stem("recursive");
        let c = stem("recurse");
        // At least two of the three should collapse together.
        let same = (a == b) as u8 + (b == c) as u8 + (a == c) as u8;
        assert!(same >= 1, "got {a}/{b}/{c}");
    }

    #[test]
    fn bm25_rewards_rarer_terms() {
        // A term in 1 of 1000 docs should outscore one in 500 of 1000.
        let rare = bm25_term_score(1, 100, 100.0, 1000, 1);
        let common = bm25_term_score(1, 100, 100.0, 1000, 500);
        assert!(rare > common);
    }
}
