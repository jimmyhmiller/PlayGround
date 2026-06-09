//! `datalog-embed` — embed a text field of an entity type into a vector field,
//! entirely in Rust (candle, CPU). No Python.
//!
//! Reads `(entity, <text-field>)` for every entity of a type, embeds each text
//! with a local BERT/sentence-transformer model, and asserts the resulting
//! `vector(N)` back onto the same entity. The vector field must already be
//! defined with the model's dimension (bge-small-en-v1.5 → `vector(384)`).
//!
//! Usage:
//!   datalog-embed [--host H:P] --type Chapter --text body --vector embedding \
//!                 [--model BAAI/bge-small-en-v1.5] [--batch 16] [--limit N] \
//!                 [--skip-existing]
//!
//! Build with `--features embed`.

use datalog_db::client::Client;
use datalog_db::embed::{Embedder, Pooling};
use serde_json::json;

struct Args {
    host: String,
    entity_type: String,
    text_field: String,
    vector_field: String,
    model: String,
    revision: Option<String>,
    batch: usize,
    limit: Option<usize>,
    skip_existing: bool,
    /// When set, embed this text and run a nearest-neighbour search instead of
    /// embedding the corpus. Prints ranked results.
    query: Option<String>,
    /// k for --query mode.
    k: usize,
    /// Field to print for each search hit (defaults to the text field).
    show_field: Option<String>,
    /// Hybrid mode: fuse BM25 over the (fulltext) text field with vector kNN
    /// over the vector field, via RRF. Requires the text field to be fulltext.
    hybrid: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        host: "127.0.0.1:5557".to_string(),
        entity_type: String::new(),
        text_field: String::new(),
        vector_field: String::new(),
        model: "BAAI/bge-small-en-v1.5".to_string(),
        revision: None,
        batch: 32,
        limit: None,
        skip_existing: false,
        query: None,
        k: 10,
        show_field: None,
        hybrid: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut next = || it.next().ok_or_else(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--host" => a.host = next()?,
            "--type" => a.entity_type = next()?,
            "--text" => a.text_field = next()?,
            "--vector" => a.vector_field = next()?,
            "--model" => a.model = next()?,
            "--revision" => a.revision = Some(next()?),
            "--batch" => a.batch = next()?.parse().map_err(|_| "bad --batch".to_string())?,
            "--limit" => a.limit = Some(next()?.parse().map_err(|_| "bad --limit".to_string())?),
            "--skip-existing" => a.skip_existing = true,
            "--query" => a.query = Some(next()?),
            "--k" => a.k = next()?.parse().map_err(|_| "bad --k".to_string())?,
            "--show" => a.show_field = Some(next()?),
            "--hybrid" => a.hybrid = true,
            "-h" | "--help" => return Err(HELP.to_string()),
            other => return Err(format!("unknown flag: {other}\n{HELP}")),
        }
    }
    if a.entity_type.is_empty() || a.text_field.is_empty() || a.vector_field.is_empty() {
        return Err(format!("--type, --text and --vector are required\n{HELP}"));
    }
    Ok(a)
}

const HELP: &str = "datalog-embed --type T --text FIELD --vector FIELD [--model ID] \
[--host H:P] [--batch N] [--limit N] [--skip-existing]";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args = parse_args().map_err(|e| anyhow::anyhow!(e))?;

    eprintln!("loading model {} (CPU) …", args.model);
    let t0 = std::time::Instant::now();
    let embedder = Embedder::from_hub(&args.model, args.revision.as_deref())?;
    eprintln!(
        "model loaded in {:.1}s — embedding dim {}",
        t0.elapsed().as_secs_f32(),
        embedder.dim
    );

    let mut client = Client::connect(&args.host)
        .map_err(|e| anyhow::anyhow!("connect to {}: {e}", args.host))?;

    // --- Query mode: embed the query text, run a (vector / hybrid) search. ---
    if let Some(qtext) = &args.query {
        let qvec = embedder.embed(qtext, Pooling::Mean)?;
        let show = args.show_field.clone().unwrap_or_else(|| args.text_field.clone());

        // Build the where-clause: vector-only, or hybrid (BM25 over the text
        // field fused with vector kNN over the vector field).
        let mut where_obj = json!({
            "bind": "?e",
            "type": args.entity_type,
            show.clone(): "?show",
            args.vector_field.clone(): {
                "near": qvec, "k": args.k, "score": "?sim", "metric": "cosine"
            }
        });
        if args.hybrid {
            where_obj[args.text_field.clone()] = json!({ "search": qtext, "k": args.k });
        }
        let query = json!({
            "find": ["?show", "?sim"],
            "where": [where_obj],
            "order_by": [{ "var": "?sim", "desc": true }]
        });
        let result = client.query(&query).map_err(|e| anyhow::anyhow!("query: {e}"))?;
        let mode = if args.hybrid { "hybrid (BM25+vector, RRF)" } else { "vector" };
        println!("query: {qtext:?}  [{mode}]  (top {})", args.k);
        for row in &result.rows {
            let sim = row[1].as_f64().unwrap_or(0.0);
            let text = row[0].as_str().unwrap_or("");
            let snippet: String = text.chars().take(80).collect();
            println!("  {sim:+.4}  {snippet}");
        }
        return Ok(());
    }

    // Fetch the work list in PAGES ordered by entity id. Pulling every body in
    // one query overflows the 64 MB wire limit at corpus scale, so we page
    // through with limit/offset and embed each page before fetching the next.
    // (Bodies are large; keep the page small enough that a page's worth of text
    // stays well under the protocol limit.)
    const PAGE: usize = 128;
    let mut work: Vec<(u64, String)> = Vec::new();
    let mut offset = 0usize;
    loop {
        let mut where_fields = json!({
            "bind": "?e",
            "type": args.entity_type,
            args.text_field.clone(): "?text",
        });
        let mut find = vec![json!("?e"), json!("?text")];
        if args.skip_existing {
            where_fields[args.vector_field.clone()] = json!("?vec");
            find.push(json!("?vec"));
        }
        let query = json!({
            "find": find,
            "where": [where_fields],
            "order_by": ["?e"],
            "limit": PAGE,
            "offset": offset,
        });
        let result = client
            .query(&query)
            .map_err(|e| anyhow::anyhow!("query (offset {offset}): {e}"))?;
        let got = result.rows.len();
        for row in &result.rows {
            let eid = row[0].get("ref").and_then(|r| r.as_u64());
            let text = row[1].as_str();
            let (eid, text) = match (eid, text) {
                (Some(e), Some(t)) => (e, t.to_string()),
                _ => continue,
            };
            if args.skip_existing {
                let has_vec = row.get(2).map(|v| !v.is_null()).unwrap_or(false);
                if has_vec {
                    continue;
                }
            }
            work.push((eid, text));
        }
        offset += got;
        if got < PAGE {
            break; // last page
        }
        if let Some(limit) = args.limit {
            if work.len() >= limit {
                break;
            }
        }
    }
    if let Some(limit) = args.limit {
        work.truncate(limit);
    }
    eprintln!("{} entities to embed", work.len());

    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    for chunk in work.chunks(args.batch) {
        let texts: Vec<String> = chunk.iter().map(|(_, t)| t.clone()).collect();
        let vectors = embedder.embed_batch(&texts, Pooling::Mean)?;

        // One transaction per batch: update each entity's vector field by id.
        let ops: Vec<serde_json::Value> = chunk
            .iter()
            .zip(vectors.iter())
            .map(|((eid, _), vec)| {
                json!({
                    "assert": args.entity_type,
                    "entity": eid,
                    "data": { args.vector_field.clone(): { "vec": vec } }
                })
            })
            .collect();
        client
            .transact(ops)
            .map_err(|e| anyhow::anyhow!("transact: {e}"))?;

        done += chunk.len();
        let rate = done as f32 / t0.elapsed().as_secs_f32().max(1e-3);
        eprint!("\r  embedded {done}/{} ({rate:.1}/s)   ", work.len());
    }
    eprintln!(
        "\ndone: embedded {} entities in {:.1}s",
        done,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
