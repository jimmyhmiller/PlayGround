//! Integration tests for the five features added for the paper-audiobooks
//! corpus: string `contains`, unknown-operator errors, list/cardinality-many
//! fields, list membership via `contains`, and composite-unique upsert.

use std::sync::Arc;

use datalog_db::db::Database;
use datalog_db::query::Query;
use datalog_db::storage::rocksdb_backend::RocksDbStorage;
use datalog_db::tx::TxOp;
use serde_json::json;

fn mem_db() -> (Arc<Database>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
    let db = Database::open(storage).unwrap();
    (Arc::new(db), dir)
}

fn define(db: &Database, req: serde_json::Value) {
    let type_def = datalog_db::db::parse_define_request(&req).expect("parse define");
    db.define_type(type_def).expect("define type");
}

fn assert_ops(db: &Database, ops: Vec<serde_json::Value>) -> Vec<u64> {
    let parsed: Vec<TxOp> = ops.iter().map(|o| TxOp::from_json(o).unwrap()).collect();
    db.transact(parsed).expect("transact").entity_ids
}

/// Like `assert_ops` but returns the transaction id (for `as_of` tests).
fn assert_ops_tx(db: &Database, ops: Vec<serde_json::Value>) -> u64 {
    let parsed: Vec<TxOp> = ops.iter().map(|o| TxOp::from_json(o).unwrap()).collect();
    db.transact(parsed).expect("transact").tx_id
}

fn run(db: &Database, q: serde_json::Value) -> Vec<Vec<serde_json::Value>> {
    let query = Query::from_json(&q).expect("parse query");
    let res = db.query(&query).expect("run query");
    res.rows
        .into_iter()
        .map(|row| row.into_iter().map(value_to_json).collect())
        .collect()
}

fn value_to_json(v: datalog_db::datom::Value) -> serde_json::Value {
    use datalog_db::datom::Value;
    match v {
        Value::String(s) => json!(s.to_string()),
        Value::I64(n) => json!(n),
        Value::F64(f) => json!(f),
        Value::Bool(b) => json!(b),
        Value::Ref(r) => json!({ "ref": r }),
        Value::List(items) => serde_json::Value::Array(items.into_iter().map(value_to_json).collect()),
        Value::Null => serde_json::Value::Null,
        other => json!(format!("{}", other)),
    }
}

#[test]
fn contains_filters_substrings() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(
        &db,
        json!({"entity_type": "Doc", "fields": [
            {"name": "title", "type": "string", "required": true},
            {"name": "body", "type": "string"},
        ]}),
    );
    assert_ops(
        &db,
        vec![
            json!({"assert": "Doc", "data": {"title": "Recursion", "body": "we discuss recursion here"}}),
            json!({"assert": "Doc", "data": {"title": "Types", "body": "propositions as types"}}),
            json!({"assert": "Doc", "data": {"title": "Both", "body": "recursion and types"}}),
        ],
    );

    // contains on body
    let rows = run(
        &db,
        json!({"find": ["?t"], "where": [
            {"bind": "?d", "type": "Doc", "title": "?t", "body": {"contains": "recursion"}}
        ]}),
    );
    let mut titles: Vec<String> = rows.iter().map(|r| r[0].as_str().unwrap().to_string()).collect();
    titles.sort();
    assert_eq!(titles, vec!["Both", "Recursion"]);

    // starts_with / ends_with
    let rows = run(
        &db,
        json!({"find": ["?t"], "where": [
            {"bind": "?d", "type": "Doc", "title": "?t", "body": {"starts_with": "propositions"}}
        ]}),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], json!("Types"));
}

#[test]
fn unknown_operator_is_an_error() {
    // A typo'd operator must NOT silently match everything.
    let bad = json!({"find": ["?t"], "where": [
        {"bind": "?d", "type": "Doc", "title": "?t", "body": {"contians": "x"}}
    ]});
    let err = Query::from_json(&bad).unwrap_err();
    assert!(err.contains("contians") || err.contains("unknown"), "got: {}", err);
}

#[test]
fn list_fields_store_and_query() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(
        &db,
        json!({"entity_type": "Book", "fields": [
            {"name": "title", "type": "string", "required": true},
            {"name": "subjects", "type": "[string]"},
        ]}),
    );
    assert_ops(
        &db,
        vec![
            json!({"assert": "Book", "data": {"title": "A", "subjects": ["logic", "computation"]}}),
            json!({"assert": "Book", "data": {"title": "B", "subjects": ["history"]}}),
        ],
    );

    // The list round-trips as a JSON array.
    let rows = run(
        &db,
        json!({"find": ["?t", "?s"], "where": [
            {"bind": "?b", "type": "Book", "title": "?t", "subjects": "?s"}
        ], "order_by": ["?t"]}),
    );
    assert_eq!(rows[0][1], json!(["logic", "computation"]));

    // List membership via contains.
    let rows = run(
        &db,
        json!({"find": ["?t"], "where": [
            {"bind": "?b", "type": "Book", "title": "?t", "subjects": {"contains": "logic"}}
        ]}),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], json!("A"));
}

#[test]
fn composite_unique_upserts() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(
        &db,
        json!({"entity_type": "Chapter", "fields": [
            {"name": "doc", "type": "i64", "required": true},
            {"name": "idx", "type": "i64", "required": true},
            {"name": "title", "type": "string"},
        ], "unique_keys": [["doc", "idx"]]}),
    );

    // First assert creates the entity.
    let ids1 = assert_ops(
        &db,
        vec![json!({"assert": "Chapter", "data": {"doc": 1, "idx": 0, "title": "Intro"}})],
    );
    // Re-asserting the SAME (doc, idx) must update in place, not duplicate.
    let ids2 = assert_ops(
        &db,
        vec![json!({"assert": "Chapter", "data": {"doc": 1, "idx": 0, "title": "Introduction"}})],
    );
    assert_eq!(ids1, ids2, "composite-key re-assert should upsert the same entity");

    // A different (doc, idx) is a new entity.
    let ids3 = assert_ops(
        &db,
        vec![json!({"assert": "Chapter", "data": {"doc": 1, "idx": 1, "title": "Next"}})],
    );
    assert_ne!(ids1, ids3);

    // Exactly two chapters exist, and chapter (1,0) has the updated title.
    let rows = run(
        &db,
        json!({"find": ["?t", "?i"], "where": [
            {"bind": "?c", "type": "Chapter", "title": "?t", "idx": "?i"}
        ], "order_by": ["?i"]}),
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], json!("Introduction"));
    assert_eq!(rows[1][0], json!("Next"));
}

#[test]
fn cardinality_many_membership_and_fanout() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(
        &db,
        json!({"entity_type": "Paper", "fields": [
            {"name": "title", "type": "string", "required": true},
            {"name": "tags", "type": "string", "cardinality": "many", "indexed": true},
        ]}),
    );
    assert_ops(
        &db,
        vec![
            json!({"assert": "Paper", "data": {"title": "A", "tags": ["logic", "types", "proof"]}}),
            json!({"assert": "Paper", "data": {"title": "B", "tags": ["logic", "history"]}}),
            json!({"assert": "Paper", "data": {"title": "C", "tags": ["cooking"]}}),
        ],
    );

    // Indexed membership: every paper tagged "logic".
    let rows = run(
        &db,
        json!({"find": ["?t"], "where": [
            {"bind": "?p", "type": "Paper", "title": "?t", "tags": "logic"}
        ], "order_by": ["?t"]}),
    );
    let titles: Vec<&str> = rows.iter().map(|r| r[0].as_str().unwrap()).collect();
    assert_eq!(titles, vec!["A", "B"]);

    // Membership via the `contains` operator behaves the same on a many field.
    let rows = run(
        &db,
        json!({"find": ["?t"], "where": [
            {"bind": "?p", "type": "Paper", "title": "?t", "tags": {"contains": "cooking"}}
        ]}),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], json!("C"));

    // Fan-out bind: each (paper, tag) pair is its own row.
    let rows = run(
        &db,
        json!({"find": ["?t", "?tag"], "where": [
            {"bind": "?p", "type": "Paper", "title": "?t", "tags": "?tag"}
        ], "order_by": ["?t", "?tag"]}),
    );
    // A has 3 tags, B has 2, C has 1 = 6 rows.
    assert_eq!(rows.len(), 6);
    assert_eq!(rows[0], vec![json!("A"), json!("logic")]);
    assert_eq!(rows[1], vec![json!("A"), json!("proof")]);
    assert_eq!(rows[2], vec![json!("A"), json!("types")]);

    // count(*) grouped by tag — aggregate over the fan-out.
    let rows = run(
        &db,
        json!({"find": ["?tag", {"agg": "count", "var": "?p"}], "where": [
            {"bind": "?p", "type": "Paper", "tags": "?tag"}
        ], "order_by": ["?tag"]}),
    );
    // "logic" appears on 2 papers.
    let logic = rows.iter().find(|r| r[0] == json!("logic")).unwrap();
    assert_eq!(logic[1], json!(2));
}

#[test]
fn cardinality_many_refs_reverse_lookup() {
    // A many ref field should support "who points at X" via the entity graph.
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "name", "type": "string", "required": true, "unique": true},
    ]}));
    define(&db, json!({"entity_type": "Paper", "fields": [
        {"name": "title", "type": "string", "required": true},
        {"name": "cites", "type": "ref(Doc)", "cardinality": "many", "indexed": true},
    ]}));

    let d = assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"name": "seminal"}}),
        json!({"assert": "Doc", "data": {"name": "other"}}),
    ]);
    let (seminal, other) = (d[0], d[1]);

    assert_ops(&db, vec![
        json!({"assert": "Paper", "data": {"title": "P1", "cites": [{"ref": seminal}, {"ref": other}]}}),
        json!({"assert": "Paper", "data": {"title": "P2", "cites": [{"ref": seminal}]}}),
        json!({"assert": "Paper", "data": {"title": "P3", "cites": [{"ref": other}]}}),
    ]);

    // Reverse lookup: which papers cite the seminal doc? (membership on a many ref)
    let rows = run(&db, json!({"find": ["?t"], "where": [
        {"bind": "?p", "type": "Paper", "title": "?t", "cites": {"ref": seminal}}
    ], "order_by": ["?t"]}));
    let titles: Vec<&str> = rows.iter().map(|r| r[0].as_str().unwrap()).collect();
    assert_eq!(titles, vec!["P1", "P2"]);
}

#[test]
fn cardinality_many_retract_single_value() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Item", "fields": [
        {"name": "name", "type": "string", "required": true, "unique": true},
        {"name": "tags", "type": "string", "cardinality": "many", "indexed": true},
    ]}));
    let ids = assert_ops(&db, vec![
        json!({"assert": "Item", "data": {"name": "x", "tags": ["a", "b", "c"]}}),
    ]);
    let eid = ids[0];

    // Retract a single value from the set, leaving the others.
    let _ = assert_ops(&db, vec![
        json!({"retract": "Item", "entity": eid, "fields": ["tags"], "values": ["b"]}),
    ]);

    // After retracting "b", membership on "b" is empty but "a"/"c" remain.
    let on_b = run(&db, json!({"find": ["?n"], "where": [
        {"bind": "?i", "type": "Item", "name": "?n", "tags": "b"}
    ]}));
    let on_a = run(&db, json!({"find": ["?n"], "where": [
        {"bind": "?i", "type": "Item", "name": "?n", "tags": "a"}
    ]}));
    assert_eq!(on_b.len(), 0, "tag b should be gone");
    assert_eq!(on_a.len(), 1, "tag a should remain");
}

#[test]
fn cardinality_many_as_of_time_travel() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "name", "type": "string", "required": true, "unique": true},
        {"name": "tags", "type": "string", "cardinality": "many", "indexed": true},
    ]}));

    // tx1: doc with tags [a, b].
    let tx1 = assert_ops_tx(&db, vec![
        json!({"assert": "Doc", "data": {"name": "x", "tags": ["a", "b"]}}),
    ]);
    let eid = run(&db, json!({"find": ["?d"], "where": [
        {"bind": "?d", "type": "Doc", "name": "x"}
    ]}))[0][0].clone();
    let eid = eid.get("ref").unwrap().as_u64().unwrap();

    // tx2: add tag c.
    let tx2 = assert_ops_tx(&db, vec![
        json!({"assert": "Doc", "entity": eid, "data": {"tags": ["c"]}}),
    ]);
    // tx3: retract tag a.
    let tx3 = assert_ops_tx(&db, vec![
        json!({"retract": "Doc", "entity": eid, "fields": ["tags"], "values": ["a"]}),
    ]);

    let tags_at = |asof: u64| -> Vec<String> {
        let mut t: Vec<String> = run(&db, json!({"find": ["?tag"], "where": [
            {"bind": "?d", "type": "Doc", "tags": "?tag"}
        ], "as_of": asof}))
            .into_iter()
            .map(|r| r[0].as_str().unwrap().to_string())
            .collect();
        t.sort();
        t
    };

    // History reconstructed per-tx: [a,b] -> [a,b,c] -> [b,c].
    assert_eq!(tags_at(tx1), vec!["a", "b"]);
    assert_eq!(tags_at(tx2), vec!["a", "b", "c"]);
    assert_eq!(tags_at(tx3), vec!["b", "c"]);

    // Membership lookup as_of must also respect history: "a" existed at tx1/tx2
    // but not tx3.
    let has_a = |asof: u64| -> usize {
        run(&db, json!({"find": ["?n"], "where": [
            {"bind": "?d", "type": "Doc", "name": "?n", "tags": "a"}
        ], "as_of": asof})).len()
    };
    assert_eq!(has_a(tx1), 1);
    assert_eq!(has_a(tx2), 1);
    assert_eq!(has_a(tx3), 0, "tag a was retracted by tx3");
}

#[test]
fn cardinality_many_retract_entity_removes_all_values() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "name", "type": "string", "required": true, "unique": true},
        {"name": "tags", "type": "string", "cardinality": "many", "indexed": true},
    ]}));
    let ids = assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"name": "x", "tags": ["a", "b", "c"]}}),
        json!({"assert": "Doc", "data": {"name": "y", "tags": ["a"]}}),
    ]);
    let x = ids[0];

    // Retract the whole entity x. All of x's many-values must disappear from
    // the membership index, but y's "a" must remain.
    let parsed = vec![TxOp::from_json(&json!({"retract_entity": "Doc", "entity": x})).unwrap()];
    db.transact(parsed).expect("retract entity");

    for tag in ["a", "b", "c"] {
        let rows = run(&db, json!({"find": ["?n"], "where": [
            {"bind": "?d", "type": "Doc", "name": "?n", "tags": tag}
        ]}));
        let names: Vec<&str> = rows.iter().map(|r| r[0].as_str().unwrap()).collect();
        // x must be gone for every one of its tags; y still has "a".
        assert!(!names.contains(&"x"), "x should be fully retracted (tag {})", tag);
        if tag == "a" {
            assert_eq!(names, vec!["y"], "y should still be tagged a");
        }
    }
}

#[test]
fn vector_search_topk_ranked() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "name", "type": "string", "required": true, "unique": true},
        {"name": "embedding", "type": "vector(3)"},
    ]}));

    // Three docs with simple 3-d embeddings.
    assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"name": "x", "embedding": {"vec": [1.0, 0.0, 0.0]}}}),
        json!({"assert": "Doc", "data": {"name": "y", "embedding": {"vec": [0.0, 1.0, 0.0]}}}),
        json!({"assert": "Doc", "data": {"name": "z", "embedding": {"vec": [0.9, 0.1, 0.0]}}}),
    ]);

    // Query near [1,0,0] — x (identical) then z (close) then y (orthogonal).
    let rows = run(&db, json!({"find": ["?n", "?sim"], "where": [
        {"bind": "?d", "type": "Doc", "name": "?n",
         "embedding": {"near": [1.0, 0.0, 0.0], "k": 2, "score": "?sim", "metric": "cosine"}}
    ], "order_by": [{"var": "?sim", "desc": true}]}));

    assert_eq!(rows.len(), 2, "k=2 should return 2");
    assert_eq!(rows[0][0], json!("x"), "x is the closest (cosine 1.0)");
    assert_eq!(rows[1][0], json!("z"), "z is second");
    // x's cosine similarity is ~1.0.
    let sim_x = rows[0][1].as_f64().unwrap();
    assert!((sim_x - 1.0).abs() < 1e-5, "x cosine ~1.0, got {}", sim_x);
}

#[test]
fn vector_search_with_prefilter() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "name", "type": "string", "required": true, "unique": true},
        {"name": "kind", "type": "string", "indexed": true},
        {"name": "embedding", "type": "vector(2)"},
    ]}));
    assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"name": "a", "kind": "paper", "embedding": {"vec": [1.0, 0.0]}}}),
        json!({"assert": "Doc", "data": {"name": "b", "kind": "book",  "embedding": {"vec": [1.0, 0.0]}}}),
        json!({"assert": "Doc", "data": {"name": "c", "kind": "paper", "embedding": {"vec": [0.0, 1.0]}}}),
    ]);

    // Nearest paper to [1,0] — must skip the (identical) book b.
    let rows = run(&db, json!({"find": ["?n", "?s"], "where": [
        {"bind": "?d", "type": "Doc", "name": "?n", "kind": "paper",
         "embedding": {"near": [1.0, 0.0], "k": 5, "score": "?s"}}
    ], "order_by": [{"var": "?s", "desc": true}]}));
    let names: Vec<&str> = rows.iter().map(|r| r[0].as_str().unwrap()).collect();
    assert_eq!(names, vec!["a", "c"], "only papers, a closest");
}

#[test]
fn vector_dimension_mismatch_errors() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "name", "type": "string", "required": true, "unique": true},
        {"name": "embedding", "type": "vector(4)"},
    ]}));
    // Asserting a 3-d vector into a vector(4) field must error.
    let ops = vec![TxOp::from_json(
        &json!({"assert": "Doc", "data": {"name": "x", "embedding": {"vec": [1.0, 2.0, 3.0]}}})
    ).unwrap()];
    let res = db.transact(ops);
    assert!(res.is_err(), "dimension mismatch on assert should error");
}

#[test]
fn fulltext_bm25_ranked_search() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "title", "type": "string", "required": true, "unique": true},
        {"name": "body", "type": "string", "fulltext": true},
    ]}));
    assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"title": "a",
            "body": "recursive functions compute over recursive data structures"}}),
        json!({"assert": "Doc", "data": {"title": "b",
            "body": "the cat sat on the mat in the sun"}}),
        json!({"assert": "Doc", "data": {"title": "c",
            "body": "a function returns a value; functional programming avoids state"}}),
    ]);

    // Search "recursive functions" — doc a (both terms, high tf) ranks top.
    let rows = run(&db, json!({"find": ["?t", "?s"], "where": [
        {"bind": "?d", "type": "Doc", "title": "?t",
         "body": {"search": "recursive functions", "k": 5, "score": "?s"}}
    ], "order_by": [{"var": "?s", "desc": true}]}));
    assert!(!rows.is_empty(), "should find matches");
    assert_eq!(rows[0][0], json!("a"), "doc a is the best match");
    // b (no query terms) must not appear.
    let titles: Vec<&str> = rows.iter().map(|r| r[0].as_str().unwrap()).collect();
    assert!(!titles.contains(&"b"), "doc b has none of the terms");
    // Scores are positive and descending.
    let s0 = rows[0][1].as_f64().unwrap();
    assert!(s0 > 0.0);
}

#[test]
fn fulltext_stemming_matches_variants() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "title", "type": "string", "required": true, "unique": true},
        {"name": "body", "type": "string", "fulltext": true},
    ]}));
    assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"title": "x", "body": "computational models of computing"}}),
    ]);
    // Query "compute" should match via stemming ("computing"/"computational").
    let rows = run(&db, json!({"find": ["?t", "?s"], "where": [
        {"bind": "?d", "type": "Doc", "title": "?t", "body": {"search": "compute", "score": "?s"}}
    ], "order_by": [{"var": "?s", "desc": true}]}));
    assert_eq!(rows.len(), 1, "stemmed query should match");
    assert_eq!(rows[0][0], json!("x"));
}

#[test]
fn fulltext_updates_on_overwrite() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "title", "type": "string", "required": true, "unique": true},
        {"name": "body", "type": "string", "fulltext": true},
    ]}));
    let ids = assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"title": "x", "body": "alpha beta gamma"}}),
    ]);
    let eid = ids[0];
    // Overwrite the body — old terms must be de-indexed, new ones indexed.
    db.transact(vec![TxOp::from_json(
        &json!({"assert": "Doc", "entity": eid, "data": {"body": "delta epsilon"}})
    ).unwrap()]).unwrap();

    let hits = |term: &str| run(&db, json!({"find": ["?t"], "where": [
        {"bind": "?d", "type": "Doc", "title": "?t", "body": {"search": term, "score": "?s"}}
    ]})).len();
    assert_eq!(hits("alpha"), 0, "old term must be de-indexed");
    assert_eq!(hits("delta"), 1, "new term must be indexed");
}

#[test]
fn fulltext_purge_clears_index() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "title", "type": "string", "required": true, "unique": true},
        {"name": "body", "type": "string", "fulltext": true},
    ]}));
    assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"title": "x", "body": "uniqueterm appears here"}}),
    ]);
    // Sanity: indexed.
    let hits = |term: &str| {
        let q = json!({"find": ["?t", "?s"], "where": [
            {"bind": "?d", "type": "Doc", "title": "?t", "body": {"search": term, "score": "?s"}}
        ], "order_by": [{"var": "?s", "desc": true}]});
        db.query(&Query::from_json(&q).unwrap()).unwrap().rows.len()
    };
    assert_eq!(hits("uniqueterm"), 1);

    // Hard-purge the type; the inverted index must be gone too.
    db.drop_type("Doc", true).unwrap();
    // Redefine and re-add a different doc; the old term must NOT resurface and
    // BM25 stats must be consistent (a fresh search scores > 0).
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "title", "type": "string", "required": true, "unique": true},
        {"name": "body", "type": "string", "fulltext": true},
    ]}));
    assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"title": "y", "body": "different content entirely"}}),
    ]);
    assert_eq!(hits("uniqueterm"), 0, "old postings must be purged");

    let q = json!({"find": ["?t", "?s"], "where": [
        {"bind": "?d", "type": "Doc", "title": "?t", "body": {"search": "different", "score": "?s"}}
    ], "order_by": [{"var": "?s", "desc": true}]});
    let rows = db.query(&Query::from_json(&q).unwrap()).unwrap().rows;
    assert_eq!(rows.len(), 1);
    // Score must be a finite positive (stats not corrupted by the purge).
    if let datalog_db::datom::Value::F64(s) = &rows[0][1] {
        assert!(*s > 0.0, "BM25 score should be positive after purge+reindex, got {s}");
    } else {
        panic!("expected a score");
    }
}

#[test]
fn hybrid_rrf_fuses_lexical_and_vector() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "title", "type": "string", "required": true, "unique": true},
        {"name": "body", "type": "string", "fulltext": true},
        {"name": "emb", "type": "vector(3)"},
    ]}));
    // a: strong lexical match for "alpha", vector far from query.
    // b: weak/no lexical, vector identical to query.
    // c: irrelevant on both.
    assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"title": "a", "body": "alpha alpha alpha beta", "emb": {"vec": [0.0, 0.0, 1.0]}}}),
        json!({"assert": "Doc", "data": {"title": "b", "body": "completely unrelated words", "emb": {"vec": [1.0, 0.0, 0.0]}}}),
        json!({"assert": "Doc", "data": {"title": "c", "body": "nothing here matches", "emb": {"vec": [0.0, 1.0, 0.0]}}}),
    ]);

    // Hybrid: search "alpha" (favors a) + near [1,0,0] (favors b). RRF should
    // surface BOTH a and b above c.
    let rows = run(&db, json!({"find": ["?t", "?s"], "where": [{
        "bind": "?d", "type": "Doc", "title": "?t",
        "body": {"search": "alpha", "k": 10},
        "emb": {"near": [1.0, 0.0, 0.0], "k": 10, "score": "?s"}
    }], "order_by": [{"var": "?s", "desc": true}]}));

    let titles: Vec<&str> = rows.iter().map(|r| r[0].as_str().unwrap()).collect();
    assert!(titles.contains(&"a"), "lexical winner must appear: {titles:?}");
    assert!(titles.contains(&"b"), "vector winner must appear: {titles:?}");
    // a and b (each #1 in one modality) must both outrank c.
    let pos = |t: &str| titles.iter().position(|x| *x == t);
    if let Some(pc) = pos("c") {
        assert!(pos("a").unwrap() < pc && pos("b").unwrap() < pc,
            "a and b should rank above c: {titles:?}");
    }
    // Fused score is positive.
    assert!(rows[0][1].as_f64().unwrap() > 0.0);
}

#[test]
fn hybrid_with_prefilter() {
    let (db, _dir) = mem_db();
    let db = &*db;
    define(&db, json!({"entity_type": "Doc", "fields": [
        {"name": "title", "type": "string", "required": true, "unique": true},
        {"name": "kind", "type": "string", "indexed": true},
        {"name": "body", "type": "string", "fulltext": true},
        {"name": "emb", "type": "vector(2)"},
    ]}));
    assert_ops(&db, vec![
        json!({"assert": "Doc", "data": {"title": "p", "kind": "paper", "body": "alpha topic", "emb": {"vec": [1.0, 0.0]}}}),
        json!({"assert": "Doc", "data": {"title": "q", "kind": "book", "body": "alpha topic", "emb": {"vec": [1.0, 0.0]}}}),
    ]);
    // A shared pre-filter (kind=paper) must constrain BOTH modalities.
    let rows = run(&db, json!({"find": ["?t", "?s"], "where": [{
        "bind": "?d", "type": "Doc", "title": "?t", "kind": "paper",
        "body": {"search": "alpha", "k": 10},
        "emb": {"near": [1.0, 0.0], "k": 10, "score": "?s"}
    }], "order_by": [{"var": "?s", "desc": true}]}));
    let titles: Vec<&str> = rows.iter().map(|r| r[0].as_str().unwrap()).collect();
    assert_eq!(titles, vec!["p"], "only the paper, not the book");
}
