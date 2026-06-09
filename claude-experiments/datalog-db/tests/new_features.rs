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
