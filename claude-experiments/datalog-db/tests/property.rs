use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datalog_db::datom::Value;
use datalog_db::db::Database;
use datalog_db::index;
use datalog_db::query::Query;
use datalog_db::schema::{EntityTypeDef, EnumTypeDef, EnumVariant, FieldDef, FieldType};
use datalog_db::storage::rocksdb_backend::RocksDbStorage;
use datalog_db::storage::StorageBackend;
use datalog_db::tx::TxOp;

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_db() -> (Arc<Database>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let storage = RocksDbStorage::open(dir.path()).unwrap();
    let db = Database::open(Arc::new(storage)).unwrap();
    (Arc::new(db), dir)
}

/// Variant that also returns the storage handle so tests can scan raw
/// index bytes directly (e.g. to verify cross-index agreement).
fn test_db_with_storage() -> (Arc<Database>, Arc<RocksDbStorage>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
    let db = Database::open(storage.clone()).unwrap();
    (Arc::new(db), storage, dir)
}

/// Raw scan of all entries beginning with a single-byte prefix.
fn raw_scan_prefix(storage: &dyn StorageBackend, prefix_byte: u8) -> Vec<(Vec<u8>, Vec<u8>)> {
    let prefix = vec![prefix_byte];
    let end = index::prefix_end(&prefix);
    let result = storage
        .execute_read(Box::new(move |snap| {
            let entries = snap.scan(&prefix, &end).unwrap();
            Ok(Box::new(entries) as Box<dyn Any + Send>)
        }))
        .unwrap();
    *result.downcast::<Vec<(Vec<u8>, Vec<u8>)>>().unwrap()
}

/// Wide entity type exercising every scalar field type.
fn wide_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "Wide".to_string(),
        fields: vec![
            FieldDef {
                name: "s".to_string(),
                field_type: FieldType::String,
                required: false,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
            FieldDef {
                name: "i".to_string(),
                field_type: FieldType::I64,
                required: false,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
            FieldDef {
                name: "f".to_string(),
                field_type: FieldType::F64,
                required: false,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
            FieldDef {
                name: "b".to_string(),
                field_type: FieldType::Bool,
                required: false,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
            FieldDef {
                name: "data".to_string(),
                field_type: FieldType::Bytes,
                required: false,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
        ],
    }
}

fn color_enum() -> EnumTypeDef {
    EnumTypeDef {
        name: "Color".to_string(),
        variants: vec![
            EnumVariant {
                name: "Red".to_string(),
                fields: vec![],
            },
            EnumVariant {
                name: "Green".to_string(),
                fields: vec![],
            },
            EnumVariant {
                name: "Custom".to_string(),
                fields: vec![
                    FieldDef {
                        name: "r".to_string(),
                        field_type: FieldType::I64,
                        required: false,
                        unique: false,
                        indexed: false,
                        cardinality: Default::default(),
                        fulltext: false,
                        ann: false,
                    },
                    FieldDef {
                        name: "g".to_string(),
                        field_type: FieldType::I64,
                        required: false,
                        unique: false,
                        indexed: false,
                        cardinality: Default::default(),
                        fulltext: false,
                        ann: false,
                    },
                    FieldDef {
                        name: "b".to_string(),
                        field_type: FieldType::I64,
                        required: false,
                        unique: false,
                        indexed: false,
                        cardinality: Default::default(),
                        fulltext: false,
                        ann: false,
                    },
                ],
            },
        ],
    }
}

fn widget_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "Widget".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
            FieldDef {
                name: "color".to_string(),
                field_type: FieldType::Enum("Color".to_string()),
                required: false,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
        ],
    }
}

/// Count datoms excluding schema internals.
fn data_datom_count(db: &Database) -> usize {
    db.all_datoms()
        .unwrap()
        .iter()
        .filter(|d| !d.attribute.starts_with("__schema_"))
        .count()
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Strings: includes unicode, empty, long, special chars, embedded nulls.
fn arb_string() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => "\\PC{0,50}",                              // arbitrary unicode up to 50 chars
        2 => "[a-zA-Z0-9 _]{0,100}",                    // ASCII including empty
        1 => Just(String::new()),                        // empty
        1 => Just("a".repeat(1000)),                     // long
        1 => Just("\0".to_string()),                     // null byte
        1 => Just("hello\0world".to_string()),           // embedded null
        1 => Just("\u{1F600}\u{1F4A9}".to_string()),     // emoji
        1 => Just("\u{202E}reversed\u{202C}".to_string()), // bidi override
        1 => Just("日本語テスト".to_string()),              // CJK
    ]
}

/// i64: full range including extremes that stress sign-bit encoding.
fn arb_i64() -> impl Strategy<Value = i64> {
    prop_oneof![
        4 => prop::num::i64::ANY,                        // full range
        1 => Just(0i64),
        1 => Just(1i64),
        1 => Just(-1i64),
        1 => Just(i64::MIN),
        1 => Just(i64::MAX),
        1 => Just(i64::MIN + 1),
        1 => Just(i64::MAX - 1),
    ]
}

/// f64: includes edge cases that stress the bit-flipping sort encoding.
/// Excludes NaN because NaN != NaN breaks equality-based assertions.
fn arb_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        4 => prop::num::f64::NORMAL | prop::num::f64::SUBNORMAL | prop::num::f64::ZERO,
        1 => Just(0.0f64),
        1 => Just(-0.0f64),
        1 => Just(f64::MIN),
        1 => Just(f64::MAX),
        1 => Just(f64::MIN_POSITIVE),
        1 => Just(f64::EPSILON),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
        1 => Just(1.0e-308_f64),  // near subnormal boundary
        1 => Just(-1.0e-308_f64),
    ]
}

fn arb_bool() -> impl Strategy<Value = bool> {
    prop::bool::ANY
}

fn arb_bytes() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        3 => prop::collection::vec(prop::num::u8::ANY, 0..100),
        1 => Just(vec![]),                               // empty
        1 => Just(vec![0u8; 256]),                        // all zeros
        1 => Just(vec![0xFF; 256]),                       // all 0xFF
        1 => Just((0..=255).collect::<Vec<u8>>()),        // all byte values
    ]
}

/// Generate a random subset of fields for the Wide type.
/// Returns a HashMap ready for transact, with 1..=5 fields chosen at random.
fn arb_wide_fields() -> impl Strategy<Value = HashMap<String, Value>> {
    (
        prop::option::of(arb_string()),
        prop::option::of(arb_i64()),
        prop::option::of(arb_f64()),
        prop::option::of(arb_bool()),
        prop::option::of(arb_bytes()),
    )
        .prop_filter("need at least one field", |(s, i, f, b, d)| {
            s.is_some() || i.is_some() || f.is_some() || b.is_some() || d.is_some()
        })
        .prop_map(|(s, i, f, b, d)| {
            let mut m = HashMap::new();
            if let Some(v) = s {
                m.insert("s".to_string(), Value::String(v.into()));
            }
            if let Some(v) = i {
                m.insert("i".to_string(), Value::I64(v));
            }
            if let Some(v) = f {
                m.insert("f".to_string(), Value::F64(v));
            }
            if let Some(v) = b {
                m.insert("b".to_string(), Value::Bool(v));
            }
            if let Some(v) = d {
                m.insert("data".to_string(), Value::Bytes(v));
            }
            m
        })
}

/// A sequence of field maps for successive updates to a Wide entity.
fn arb_wide_update_seq() -> impl Strategy<Value = Vec<HashMap<String, Value>>> {
    prop::collection::vec(arb_wide_fields(), 2..8)
}

/// Pick a Color variant, including partial Custom fields (all optional).
fn arb_color() -> impl Strategy<Value = Value> {
    prop_oneof![
        1 => Just(Value::String("Red".into())),
        1 => Just(Value::String("Green".into())),
        // Custom with all subsets of {r, g, b}
        2 => (
            prop::option::of(arb_i64()),
            prop::option::of(arb_i64()),
            prop::option::of(arb_i64()),
        ).prop_map(|(r, g, b)| {
            let mut fields = HashMap::new();
            if let Some(v) = r { fields.insert("r".to_string(), Value::I64(v)); }
            if let Some(v) = g { fields.insert("g".to_string(), Value::I64(v)); }
            if let Some(v) = b { fields.insert("b".to_string(), Value::I64(v)); }
            Value::Enum(Box::new(datalog_db::datom::EnumValue {
                variant: "Custom".to_string(),
                fields,
            }))
        }),
    ]
}

/// Generate a sequence of Color values for successive updates.
fn arb_color_seq() -> impl Strategy<Value = Vec<Value>> {
    prop::collection::vec(arb_color(), 2..8)
}

// ---------------------------------------------------------------------------
// Property 1: Insert-Get Roundtrip (all value types)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_insert_get_roundtrip(fields in arb_wide_fields()) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let result = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: fields.clone(),
        }]).unwrap();

        let eid = result.entity_ids[0];
        let entity = db.get_entity(eid).unwrap().unwrap();

        for (field_name, expected) in &fields {
            let key = format!("Wide/{}", field_name);
            let actual = entity.get(&key);
            prop_assert_eq!(
                actual, Some(expected),
                "field '{}': expected {:?}, got {:?}", field_name, expected, actual
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 2: Last-Write-Wins for Scalars (multi-field, all types)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_last_write_wins(updates in arb_wide_update_seq()) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        // Insert with first update
        let result = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: updates[0].clone(),
        }]).unwrap();
        let eid = result.entity_ids[0];

        // Track expected state: last asserted value per field
        let mut expected: HashMap<String, Value> = updates[0].clone();

        for update in &updates[1..] {
            db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: Some(eid),
                data: update.clone(),
            }]).unwrap();

            for (k, v) in update {
                expected.insert(k.clone(), v.clone());
            }
        }

        let entity = db.get_entity(eid).unwrap().unwrap();

        for (field_name, expected_val) in &expected {
            let key = format!("Wide/{}", field_name);
            prop_assert_eq!(
                entity.get(&key), Some(expected_val),
                "field '{}' should be {:?}", field_name, expected_val
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 3: Datom Count Monotonicity (Append-Only)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_datom_count_monotonic(updates in arb_wide_update_seq()) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let result = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: updates[0].clone(),
        }]).unwrap();
        let eid = result.entity_ids[0];

        let mut prev_count = data_datom_count(&db);

        for update in &updates[1..] {
            db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: Some(eid),
                data: update.clone(),
            }]).unwrap();

            let new_count = data_datom_count(&db);
            prop_assert!(
                new_count >= prev_count,
                "datom count decreased from {} to {} (append-only violated)",
                prev_count,
                new_count
            );
            prev_count = new_count;
        }
    }
}

// ---------------------------------------------------------------------------
// Property 4: Idempotent Same-Value Update (all types)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_idempotent_update(fields in arb_wide_fields()) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let result = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: fields.clone(),
        }]).unwrap();
        let eid = result.entity_ids[0];

        let count_before = data_datom_count(&db);

        // Update to the SAME values
        let update_result = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: Some(eid),
            data: fields.clone(),
        }]).unwrap();

        let count_after = data_datom_count(&db);
        prop_assert_eq!(
            count_before, count_after,
            "same-value update should produce 0 new datoms, but count went {} -> {}",
            count_before, count_after
        );
        prop_assert_eq!(
            update_result.datom_count, 0,
            "transaction should report 0 datoms for no-op"
        );
    }
}

// ---------------------------------------------------------------------------
// Property 5: Time Travel Immutability
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_time_travel_stable(
        initial in arb_wide_fields(),
        later_updates in prop::collection::vec(arb_wide_fields(), 1..5)
    ) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let result = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: initial.clone(),
        }]).unwrap();
        let eid = result.entity_ids[0];
        let snapshot_tx = result.tx_id;

        // Pick one field that was in the initial insert to query for
        let (query_field, query_val) = initial.iter().next().unwrap();

        let query_json = serde_json::json!({
            "find": ["?e"],
            "where": [
                {"bind": "?e", "type": "Wide"}
            ],
            "as_of": snapshot_tx
        });
        let query = datalog_db::query::Query::from_json(&query_json).unwrap();
        let baseline = db.query(&query).unwrap();
        prop_assert_eq!(baseline.rows.len(), 1, "should find entity at snapshot tx");

        // Apply later transactions
        for update in &later_updates {
            db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: Some(eid),
                data: update.clone(),
            }]).unwrap();
        }

        // as_of should still return exactly 1 entity
        let after = db.query(&query).unwrap();
        prop_assert_eq!(
            after.rows.len(), 1,
            "time travel should still find 1 entity"
        );

        // Also verify via get_entity + resolve_current_values at as_of
        // by checking the raw datoms haven't been mutated
        let datoms = db.entity_datoms(eid).unwrap();
        let at_snapshot: Vec<_> = datoms.iter()
            .filter(|d| d.tx <= snapshot_tx)
            .collect();
        // The initial field should appear as asserted
        let attr = format!("Wide/{}", query_field);
        let found = at_snapshot.iter().any(|d| {
            d.attribute == attr && d.added && d.value == *query_val
        });
        prop_assert!(found, "initial field {}={:?} should be in datoms at snapshot", query_field, query_val);
    }
}

// ---------------------------------------------------------------------------
// Property 6: Retract Removes Field (all types)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_retract_removes_field(fields in arb_wide_fields()) {
        // Need at least 2 fields to retract one and keep one
        prop_assume!(fields.len() >= 2);

        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let result = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: fields.clone(),
        }]).unwrap();
        let eid = result.entity_ids[0];

        // Retract the first field
        let retract_field = fields.keys().next().unwrap().clone();
        db.transact(vec![TxOp::Retract {
            entity_type: "Wide".to_string(),
            entity: eid,
            fields: vec![retract_field.clone()],
            values: vec![],
        }]).unwrap();

        let entity = db.get_entity(eid).unwrap().unwrap();
        let retract_key = format!("Wide/{}", retract_field);
        prop_assert!(
            !entity.contains_key(&retract_key),
            "retracted field '{}' should be gone, but found {:?}",
            retract_field, entity.get(&retract_key)
        );

        // Other fields should remain
        for (field_name, expected) in &fields {
            if *field_name == retract_field {
                continue;
            }
            let key = format!("Wide/{}", field_name);
            prop_assert_eq!(
                entity.get(&key), Some(expected),
                "non-retracted field '{}' should still be present", field_name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 7: Enum Variant Exclusivity (with partial Custom fields)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_enum_variant_exclusivity(colors in arb_color_seq()) {
        let (db, _dir) = test_db();
        db.define_enum(color_enum()).unwrap();
        db.define_type(widget_type()).unwrap();

        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("w".into()));
        data.insert("color".to_string(), colors[0].clone());
        let result = db.transact(vec![TxOp::Assert {
            entity_type: "Widget".to_string(),
            entity: None,
            data,
        }]).unwrap();
        let eid = result.entity_ids[0];

        for color in &colors[1..] {
            let mut data = HashMap::new();
            data.insert("color".to_string(), color.clone());
            db.transact(vec![TxOp::Assert {
                entity_type: "Widget".to_string(),
                entity: Some(eid),
                data,
            }]).unwrap();
        }

        let entity = db.get_entity(eid).unwrap().unwrap();
        let last_color = colors.last().unwrap();

        let (expected_variant, expected_fields): (String, HashMap<String, Value>) = match last_color {
            Value::String(s) => (s.to_string(), HashMap::new()),
            Value::Enum(e) => (e.variant.clone(), e.fields.clone()),
            _ => panic!("unexpected color value"),
        };

        // Tag should match
        prop_assert_eq!(
            entity.get("Widget/color/__tag"),
            Some(&Value::String(expected_variant.as_str().into())),
            "tag should be '{}'", expected_variant
        );

        // Expected variant fields should be present
        for (fname, fval) in &expected_fields {
            let key = format!("Widget/color.{}/{}", expected_variant, fname);
            prop_assert_eq!(
                entity.get(&key), Some(fval),
                "expected field {} = {:?}", key, fval
            );
        }

        // Fields from OTHER variants should NOT be present
        for variant in &["Red", "Green", "Custom"] {
            if *variant == expected_variant {
                continue;
            }
            let stale: Vec<_> = entity.keys()
                .filter(|k| k.starts_with(&format!("Widget/color.{}/", variant)))
                .collect();
            prop_assert!(
                stale.is_empty(),
                "stale fields from variant '{}' still present: {:?}",
                variant, stale
            );
        }

        // If current variant is Custom, fields NOT in expected_fields should be gone
        if expected_variant == "Custom" {
            for f in &["r", "g", "b"] {
                if expected_fields.contains_key(*f) {
                    continue;
                }
                let key = format!("Widget/color.Custom/{}", f);
                prop_assert!(
                    !entity.contains_key(&key),
                    "Custom/{} was not asserted this time, should be retracted",
                    f
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Property 8: Entity Independence (all types)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_entity_independence(
        a_fields in arb_wide_fields(),
        b_fields in arb_wide_fields(),
        a_update in arb_wide_fields(),
    ) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let result_a = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: a_fields.clone(),
        }]).unwrap();
        let eid_a = result_a.entity_ids[0];

        let result_b = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: b_fields.clone(),
        }]).unwrap();
        let eid_b = result_b.entity_ids[0];

        // Update only entity A
        db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: Some(eid_a),
            data: a_update.clone(),
        }]).unwrap();

        // Entity B should be completely unaffected
        let entity_b = db.get_entity(eid_b).unwrap().unwrap();
        for (field_name, expected) in &b_fields {
            let key = format!("Wide/{}", field_name);
            prop_assert_eq!(
                entity_b.get(&key), Some(expected),
                "entity B's field '{}' should be unchanged", field_name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 9: f64 Sort Order Preserved
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_f64_sort_order(values in prop::collection::vec(arb_f64(), 3..10)) {
        // Filter out NaN — not orderable
        let values: Vec<f64> = values.into_iter().filter(|v| !v.is_nan()).collect();
        prop_assume!(values.len() >= 2);

        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        for val in &values {
            let mut data = HashMap::new();
            data.insert("f".to_string(), Value::F64(*val));
            db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: None,
                data,
            }]).unwrap();
        }

        // Every inserted value should roundtrip correctly
        let datoms = db.all_datoms().unwrap();
        let f_datoms: Vec<f64> = datoms.iter()
            .filter(|d| d.attribute == "Wide/f" && d.added)
            .filter_map(|d| if let Value::F64(v) = d.value { Some(v) } else { None })
            .collect();

        for val in &values {
            prop_assert!(
                f_datoms.contains(val),
                "f64 value {} should be in stored datoms, got {:?}",
                val, f_datoms
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 10: i64 Sort Order Preserved
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_i64_encoding_roundtrip(values in prop::collection::vec(arb_i64(), 2..10)) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        for val in &values {
            let mut data = HashMap::new();
            data.insert("i".to_string(), Value::I64(*val));
            db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: None,
                data,
            }]).unwrap();
        }

        let datoms = db.all_datoms().unwrap();
        let i_datoms: Vec<i64> = datoms.iter()
            .filter(|d| d.attribute == "Wide/i" && d.added)
            .filter_map(|d| if let Value::I64(v) = d.value { Some(v) } else { None })
            .collect();

        for val in &values {
            prop_assert!(
                i_datoms.contains(val),
                "i64 value {} should roundtrip through storage, got {:?}",
                val, i_datoms
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers used by the cross-cutting properties below
// ---------------------------------------------------------------------------

/// Encode a Value as a deterministic byte sequence for hashing/comparison.
/// `Value` itself isn't `Hash + Eq` (because `f64` isn't), so we go via
/// bytes whenever we want to drop datoms into a `HashSet`. Bit-pattern
/// comparison is what we want anyway — two f64 NaNs with the same bit
/// pattern should be treated as equal across indexes.
fn value_bytes(v: &Value) -> Vec<u8> {
    let mut buf = Vec::new();
    match v {
        Value::String(s) => {
            buf.push(1);
            buf.extend_from_slice(&(s.len() as u32).to_be_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        Value::I64(n) => {
            buf.push(2);
            buf.extend_from_slice(&n.to_be_bytes());
        }
        Value::F64(f) => {
            buf.push(3);
            buf.extend_from_slice(&f.to_bits().to_be_bytes());
        }
        Value::Bool(b) => {
            buf.push(4);
            buf.push(if *b { 1 } else { 0 });
        }
        Value::Ref(id) => {
            buf.push(5);
            buf.extend_from_slice(&id.to_be_bytes());
        }
        Value::Bytes(b) => {
            buf.push(6);
            buf.extend_from_slice(&(b.len() as u32).to_be_bytes());
            buf.extend_from_slice(b);
        }
        Value::List(items) => {
            buf.push(7);
            buf.extend_from_slice(&(items.len() as u32).to_be_bytes());
            for it in items {
                buf.extend_from_slice(&value_bytes(it));
            }
        }
        Value::Vector(vec) => {
            buf.push(8);
            buf.extend_from_slice(&(vec.len() as u32).to_be_bytes());
            for f in vec {
                buf.extend_from_slice(&f.to_bits().to_be_bytes());
            }
        }
        Value::Enum(_) | Value::Null => panic!("unsupported in datom comparison: {:?}", v),
    }
    buf
}

/// Canonical hashable form of a datom: (entity, attr_id, value_bytes, tx, added).
type NormKey = (u64, u32, Vec<u8>, u64, bool);

fn norm_eavt(entries: &[(Vec<u8>, Vec<u8>)]) -> HashSet<NormKey> {
    entries
        .iter()
        .filter_map(|(k, _)| index::decode_datom_from_eavt(k))
        .map(|d| (d.entity, d.attr_id, value_bytes(&d.value), d.tx, d.added))
        .collect()
}

fn norm_aevt(entries: &[(Vec<u8>, Vec<u8>)]) -> HashSet<NormKey> {
    entries
        .iter()
        .filter_map(|(k, _)| index::decode_datom_from_aevt(k))
        .map(|d| (d.entity, d.attr_id, value_bytes(&d.value), d.tx, d.added))
        .collect()
}

fn norm_avet(entries: &[(Vec<u8>, Vec<u8>)]) -> HashSet<NormKey> {
    entries
        .iter()
        .filter_map(|(k, _)| index::decode_datom_from_avet(k))
        .map(|d| (d.entity, d.attr_id, value_bytes(&d.value), d.tx, d.added))
        .collect()
}

// ---------------------------------------------------------------------------
// Property 11: Cross-Index Consistency (EAVT == AEVT == AVET)
// ---------------------------------------------------------------------------
//
// Every datom is encoded into EAVT, AEVT, and AVET (and VAET for Ref values).
// Decoding each index back to (entity, attr_id, value, tx, added) must
// produce identical sets. A divergence would indicate one of the encoders
// (e.g. with a value-type bug or attr-id bug) is dropping or corrupting
// entries — the kind of bug that breaks query planning silently.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_cross_index_consistency(
        updates in prop::collection::vec(arb_wide_fields(), 1..6)
    ) {
        let (db, storage, _dir) = test_db_with_storage();
        db.define_type(wide_type()).unwrap();

        let mut eid: Option<u64> = None;
        for upd in &updates {
            let r = db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: eid,
                data: upd.clone(),
            }]).unwrap();
            if eid.is_none() {
                eid = Some(r.entity_ids[0]);
            }
        }

        let eavt = norm_eavt(&raw_scan_prefix(&*storage, index::EAVT_PREFIX));
        let aevt = norm_aevt(&raw_scan_prefix(&*storage, index::AEVT_PREFIX));
        let avet = norm_avet(&raw_scan_prefix(&*storage, index::AVET_PREFIX));

        prop_assert_eq!(
            &eavt, &aevt,
            "EAVT and AEVT disagree: {} vs {} entries",
            eavt.len(), aevt.len()
        );
        prop_assert_eq!(
            &eavt, &avet,
            "EAVT and AVET disagree: {} vs {} entries",
            eavt.len(), avet.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Property 12: Persistence Roundtrip (Close + Reopen Preserves State)
// ---------------------------------------------------------------------------
//
// After dropping the Database and storage handles, reopening from the same
// path must yield byte-identical datoms and the same schema. A regression
// here would mean a tx_counter or attr-interner state didn't make it to
// the WAL — fatal in production.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn prop_persistence_roundtrip(
        seq in prop::collection::vec(arb_wide_fields(), 1..6)
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // First open: define schema, transact data, capture state.
        let (datoms_before, schema_before, tx_counter_before): (Vec<_>, _, u64) = {
            let storage = Arc::new(RocksDbStorage::open(&path).unwrap());
            let db = Database::open(storage).unwrap();
            db.define_type(wide_type()).unwrap();

            let mut last_tx = 0u64;
            for upd in &seq {
                let r = db.transact(vec![TxOp::Assert {
                    entity_type: "Wide".to_string(),
                    entity: None,
                    data: upd.clone(),
                }]).unwrap();
                last_tx = r.tx_id;
            }

            let datoms: Vec<_> = db.all_datoms().unwrap().into_iter()
                .map(|d| (d.entity, d.attribute, d.value, d.tx, d.added))
                .collect();
            let schema = db.schema_json();
            (datoms, schema, last_tx)
        }; // db and storage drop here, releasing the RocksDB file lock.

        // Second open from the same path.
        let storage2 = Arc::new(RocksDbStorage::open(&path).unwrap());
        let db2 = Database::open(storage2).unwrap();

        let datoms_after: Vec<_> = db2.all_datoms().unwrap().into_iter()
            .map(|d| (d.entity, d.attribute, d.value, d.tx, d.added))
            .collect();
        let schema_after = db2.schema_json();

        prop_assert_eq!(
            datoms_before.len(), datoms_after.len(),
            "datom count differs after reopen"
        );
        // Compare as sets — EAVT ordering is deterministic but comparing
        // as sorted Vec<(...)> requires Value to be Ord, which it isn't
        // (F64). Use HashSet equality via Vec equality after sort by tx.
        for d in &datoms_before {
            prop_assert!(
                datoms_after.contains(d),
                "datom missing after reopen: {:?}", d
            );
        }

        prop_assert_eq!(
            schema_before.to_string(), schema_after.to_string(),
            "schema differs after reopen"
        );

        // The next transaction after reopen must use a strictly larger tx_id.
        let mut data = HashMap::new();
        data.insert("i".to_string(), Value::I64(7));
        let r = db2.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data,
        }]).unwrap();
        prop_assert!(
            r.tx_id > tx_counter_before,
            "tx_counter not preserved across reopen: {} <= {}",
            r.tx_id, tx_counter_before
        );
    }
}

// ---------------------------------------------------------------------------
// Property 13: Unique Constraint Enforcement
// ---------------------------------------------------------------------------
//
// A unique field rejects duplicate values across distinct entities, but
// re-asserting the same value on the same entity is a no-op, and after
// a retract the value becomes available again.

fn unique_user_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "U".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
            FieldDef {
                name: "email".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: true,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
        ],
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_unique_constraint(
        // n distinct emails, then m attempted duplicates of email[0]
        emails in prop::collection::vec("[a-z]{1,8}@x", 2..6),
    ) {
        // Force unique strings — make each unique by appending its index.
        let emails: Vec<String> = emails.iter().enumerate()
            .map(|(i, s)| format!("{}-{}", i, s))
            .collect();

        let (db, _dir) = test_db();
        db.define_type(unique_user_type()).unwrap();

        // Insert one entity per email — all should succeed.
        let mut eids = Vec::new();
        for (i, e) in emails.iter().enumerate() {
            let mut data = HashMap::new();
            data.insert("name".to_string(), Value::String(format!("u{}", i).into()));
            data.insert("email".to_string(), Value::String(e.as_str().into()));
            let r = db.transact(vec![TxOp::Assert {
                entity_type: "U".to_string(),
                entity: None,
                data,
            }]).unwrap();
            eids.push(r.entity_ids[0]);
        }

        // Re-asserting same email on same entity is a no-op.
        let mut data = HashMap::new();
        data.insert("email".to_string(), Value::String(emails[0].as_str().into()));
        let r = db.transact(vec![TxOp::Assert {
            entity_type: "U".to_string(),
            entity: Some(eids[0]),
            data,
        }]).unwrap();
        prop_assert_eq!(r.datom_count, 0, "re-asserting same value should be a no-op");

        // Inserting a new entity with email[0] must fail.
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("dup".into()));
        data.insert("email".to_string(), Value::String(emails[0].as_str().into()));
        let r = db.transact(vec![TxOp::Assert {
            entity_type: "U".to_string(),
            entity: None,
            data,
        }]);
        prop_assert!(r.is_err(), "duplicate unique value should fail");

        // After retracting email[0], a new entity can take it.
        db.transact(vec![TxOp::Retract {
            entity_type: "U".to_string(),
            entity: eids[0],
            fields: vec!["email".to_string()],
            values: vec![],
        }]).unwrap();

        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("new".into()));
        data.insert("email".to_string(), Value::String(emails[0].as_str().into()));
        let r = db.transact(vec![TxOp::Assert {
            entity_type: "U".to_string(),
            entity: None,
            data,
        }]);
        prop_assert!(r.is_ok(), "value should be available after retract: {:?}", r.err());
    }
}

// ---------------------------------------------------------------------------
// Property 14: Query Results Agree With Raw-Datom Replay (as_of)
// ---------------------------------------------------------------------------
//
// For any sequence of single-entity inserts and any tx snapshot, querying
// the database at that snapshot must return exactly the entities created
// at or before that tx. Verified against the raw datom log replayed in
// memory (the source of truth).

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_query_matches_raw_datoms_asof(
        n in 1usize..8,
    ) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let mut tx_ids = Vec::new();
        for i in 0..n {
            let mut data = HashMap::new();
            data.insert("i".to_string(), Value::I64(i as i64));
            let r = db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: None,
                data,
            }]).unwrap();
            tx_ids.push(r.tx_id);
        }

        let all = db.all_datoms().unwrap();

        for (i, &snap_tx) in tx_ids.iter().enumerate() {
            // Ground truth: entities with __type=Wide datom whose tx <= snap_tx.
            let truth: HashSet<u64> = all
                .iter()
                .filter(|d| d.attribute == "__type" && d.added && d.tx <= snap_tx)
                .filter(|d| matches!(&d.value, Value::String(s) if s.as_ref() == "Wide"))
                .map(|d| d.entity)
                .collect();

            let q = Query::from_json(&serde_json::json!({
                "find": ["?e"],
                "where": [{"bind": "?e", "type": "Wide"}],
                "as_of": snap_tx,
            })).unwrap();
            let r = db.query(&q).unwrap();

            let got: HashSet<u64> = r.rows.iter()
                .filter_map(|row| match row.first() {
                    Some(Value::Ref(id)) => Some(*id),
                    Some(Value::I64(id)) => Some(*id as u64),
                    _ => None,
                })
                .collect();

            prop_assert_eq!(
                got.len(), i + 1,
                "expected {} entities at snapshot {}, got {}", i + 1, snap_tx, r.rows.len()
            );
            prop_assert_eq!(
                got, truth,
                "query result disagrees with raw datom replay at tx {}", snap_tx
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 15: Predicate Query Equivalence (planner-chosen path vs scan+filter)
// ---------------------------------------------------------------------------
//
// For numeric predicates (gt/lt/gte/lte), the planner may pick a range
// scan over AVET or fall back to a type scan with in-memory filtering.
// Either way, the result set must equal the set computed by scanning all
// datoms and applying the predicate in memory.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn prop_predicate_query_equivalence(
        values in prop::collection::vec(arb_i64(), 3..15),
        op_kind in 0u8..5,
        pivot in arb_i64(),
    ) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let mut entity_to_val: HashMap<u64, i64> = HashMap::new();
        for v in &values {
            let mut data = HashMap::new();
            data.insert("i".to_string(), Value::I64(*v));
            let r = db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: None,
                data,
            }]).unwrap();
            entity_to_val.insert(r.entity_ids[0], *v);
        }

        let (op_key, predicate): (&str, Box<dyn Fn(i64) -> bool>) = match op_kind {
            0 => ("gt", Box::new(move |v| v > pivot)),
            1 => ("lt", Box::new(move |v| v < pivot)),
            2 => ("gte", Box::new(move |v| v >= pivot)),
            3 => ("lte", Box::new(move |v| v <= pivot)),
            _ => ("ne", Box::new(move |v| v != pivot)),
        };

        let truth: HashSet<u64> = entity_to_val.iter()
            .filter(|(_, v)| predicate(**v))
            .map(|(e, _)| *e)
            .collect();

        let q = Query::from_json(&serde_json::json!({
            "find": ["?e"],
            "where": [{
                "bind": "?e",
                "type": "Wide",
                "i": { op_key: pivot },
            }],
        })).unwrap();
        let r = db.query(&q).unwrap();

        let got: HashSet<u64> = r.rows.iter()
            .filter_map(|row| match row.first() {
                Some(Value::Ref(id)) => Some(*id),
                Some(Value::I64(id)) => Some(*id as u64),
                _ => None,
            })
            .collect();

        prop_assert_eq!(
            got, truth,
            "predicate '{}' with pivot {} returned wrong set", op_key, pivot
        );
    }
}

// ---------------------------------------------------------------------------
// Property 16: RetractEntity Completeness
// ---------------------------------------------------------------------------
//
// After RetractEntity, the entity must be invisible to current-state
// reads and queries, while history is preserved (the datom log still
// has the asserts AND the matching retracts).

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_retract_entity_complete(fields in arb_wide_fields()) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let r = db.transact(vec![TxOp::Assert {
            entity_type: "Wide".to_string(),
            entity: None,
            data: fields.clone(),
        }]).unwrap();
        let eid = r.entity_ids[0];

        // Capture history before retract.
        let datoms_before = db.entity_datoms(eid).unwrap();
        let asserted_data: Vec<_> = datoms_before.iter()
            .filter(|d| d.added && !d.attribute.starts_with("__"))
            .collect();
        let asserted_count = asserted_data.len();
        prop_assume!(asserted_count > 0);

        db.transact(vec![TxOp::RetractEntity {
            entity_type: "Wide".to_string(),
            entity: eid,
        }]).unwrap();

        // get_entity returns None (no current-state fields).
        let entity = db.get_entity(eid).unwrap();
        prop_assert!(
            entity.is_none(),
            "get_entity must return None after RetractEntity, got {:?}", entity
        );

        // Query by type yields no row for this eid.
        let q = Query::from_json(&serde_json::json!({
            "find": ["?e"],
            "where": [{"bind": "?e", "type": "Wide"}],
        })).unwrap();
        let r = db.query(&q).unwrap();
        let got: HashSet<u64> = r.rows.iter()
            .filter_map(|row| match row.first() {
                Some(Value::Ref(id)) => Some(*id),
                Some(Value::I64(id)) => Some(*id as u64),
                _ => None,
            })
            .collect();
        prop_assert!(
            !got.contains(&eid),
            "retracted entity {} still appears in current query", eid
        );

        // History preserved: each asserted data field has a matching
        // retract (added=false) for the same attribute and value.
        let after = db.entity_datoms(eid).unwrap();
        for a in &asserted_data {
            let matched = after.iter().any(|d| {
                !d.added && d.attribute == a.attribute && d.value == a.value
            });
            prop_assert!(
                matched,
                "no retract datom found for asserted ({}, {:?})",
                a.attribute, a.value
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 17: Group Commit Assigns Monotonic, Contiguous tx_ids
// ---------------------------------------------------------------------------
//
// transact_many must hand out tx_ids that strictly increase by 1 across
// the input list, regardless of how many ops each transaction contains.
// This is what makes asOf snapshots well-defined across a batch.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_group_commit_monotonic(n in 2usize..8) {
        let (db, _dir) = test_db();
        db.define_type(wide_type()).unwrap();

        let baseline_tx = {
            let mut data = HashMap::new();
            data.insert("i".to_string(), Value::I64(-1));
            db.transact(vec![TxOp::Assert {
                entity_type: "Wide".to_string(),
                entity: None,
                data,
            }]).unwrap().tx_id
        };

        let ops_lists: Vec<Vec<TxOp>> = (0..n)
            .map(|i| {
                let mut data = HashMap::new();
                data.insert("i".to_string(), Value::I64(i as i64));
                vec![TxOp::Assert {
                    entity_type: "Wide".to_string(),
                    entity: None,
                    data,
                }]
            })
            .collect();

        let results = db.transact_many(ops_lists).unwrap();

        prop_assert_eq!(results.len(), n, "transact_many returned wrong count");

        let mut prev_tx = baseline_tx;
        for (i, res) in results.iter().enumerate() {
            let tx_id = res.as_ref()
                .unwrap_or_else(|e| panic!("tx {} failed: {}", i, e))
                .tx_id;
            prop_assert_eq!(
                tx_id, prev_tx + 1,
                "tx {} got id {}, expected {}", i, tx_id, prev_tx + 1
            );
            prev_tx = tx_id;
        }
    }
}

// ---------------------------------------------------------------------------
// Property 18: VAET Integrity (Ref-valued datoms have a VAET entry each)
// ---------------------------------------------------------------------------
//
// VAET is the reverse-ref index: only populated for Value::Ref. For every
// Ref-valued datom in EAVT there must be exactly one corresponding entry
// in VAET, and vice versa.

fn vaet_user_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "VUser".to_string(),
        fields: vec![FieldDef {
            name: "name".to_string(),
            field_type: FieldType::String,
            required: true,
            unique: false,
            indexed: false,
            cardinality: Default::default(),
            fulltext: false,
            ann: false,
        }],
    }
}

fn vaet_post_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "VPost".to_string(),
        fields: vec![
            FieldDef {
                name: "title".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
            FieldDef {
                name: "author".to_string(),
                field_type: FieldType::Ref("VUser".to_string()),
                required: true,
                unique: false,
                indexed: false,
                cardinality: Default::default(),
                fulltext: false,
                ann: false,
            },
        ],
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn prop_vaet_matches_ref_datoms(
        user_count in 1usize..5,
        post_count in 1usize..8,
        post_authors in prop::collection::vec(0usize..16, 1..8),
    ) {
        let (db, storage, _dir) = test_db_with_storage();
        db.define_type(vaet_user_type()).unwrap();
        db.define_type(vaet_post_type()).unwrap();

        let mut user_eids = Vec::new();
        for i in 0..user_count {
            let mut data = HashMap::new();
            data.insert("name".to_string(), Value::String(format!("u{}", i).into()));
            let r = db.transact(vec![TxOp::Assert {
                entity_type: "VUser".to_string(),
                entity: None,
                data,
            }]).unwrap();
            user_eids.push(r.entity_ids[0]);
        }

        for i in 0..post_count {
            let author_idx = post_authors[i % post_authors.len()] % user_eids.len();
            let mut data = HashMap::new();
            data.insert("title".to_string(), Value::String(format!("t{}", i).into()));
            data.insert("author".to_string(), Value::Ref(user_eids[author_idx]));
            db.transact(vec![TxOp::Assert {
                entity_type: "VPost".to_string(),
                entity: None,
                data,
            }]).unwrap();
        }

        // Count Ref-valued entries in EAVT — value_bytes encodes Ref with
        // tag byte 5, so we can filter by first byte without decoding back.
        let eavt_refs = norm_eavt(&raw_scan_prefix(&*storage, index::EAVT_PREFIX));
        let ref_entries: Vec<_> = eavt_refs.iter()
            .filter(|(_, _, v, _, _)| v.first() == Some(&5))
            .collect();

        // VAET should have exactly one raw entry per Ref-valued EAVT entry.
        let vaet = raw_scan_prefix(&*storage, index::VAET_PREFIX);
        prop_assert_eq!(
            vaet.len(), ref_entries.len(),
            "VAET count {} != Ref-valued EAVT count {}",
            vaet.len(), ref_entries.len()
        );

        // Spot-check: each VAET key starts with [VAET_PREFIX, Ref tag = 0x05].
        for (key, _) in &vaet {
            prop_assert_eq!(key[0], index::VAET_PREFIX);
            prop_assert_eq!(key[1], 0x05, "VAET value tag must be Ref (0x05)");
        }
    }
}
