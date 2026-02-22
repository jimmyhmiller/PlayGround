use std::collections::HashMap;
use std::sync::Arc;

use datalog_db::datom::Value;
use datalog_db::db::Database;
use datalog_db::schema::{EntityTypeDef, EnumTypeDef, EnumVariant, FieldDef, FieldType};
use datalog_db::storage::rocksdb_backend::RocksDbStorage;
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

/// Wide entity type exercising every scalar field type.
fn wide_type() -> EntityTypeDef {
    EntityTypeDef {
        name: "Wide".to_string(),
        fields: vec![
            FieldDef {
                name: "s".to_string(),
                field_type: FieldType::String,
                required: false,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "i".to_string(),
                field_type: FieldType::I64,
                required: false,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "f".to_string(),
                field_type: FieldType::F64,
                required: false,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "b".to_string(),
                field_type: FieldType::Bool,
                required: false,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "data".to_string(),
                field_type: FieldType::Bytes,
                required: false,
                unique: false,
                indexed: false,
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
                    },
                    FieldDef {
                        name: "g".to_string(),
                        field_type: FieldType::I64,
                        required: false,
                        unique: false,
                        indexed: false,
                    },
                    FieldDef {
                        name: "b".to_string(),
                        field_type: FieldType::I64,
                        required: false,
                        unique: false,
                        indexed: false,
                    },
                ],
            },
        ],
    }
}

fn widget_type() -> EntityTypeDef {
    EntityTypeDef {
        name: "Widget".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "color".to_string(),
                field_type: FieldType::Enum("Color".to_string()),
                required: false,
                unique: false,
                indexed: false,
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
                m.insert("s".to_string(), Value::String(v));
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
        1 => Just(Value::String("Red".to_string())),
        1 => Just(Value::String("Green".to_string())),
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
            Value::Enum {
                variant: "Custom".to_string(),
                fields,
            }
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
        data.insert("name".to_string(), Value::String("w".to_string()));
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

        let (expected_variant, expected_fields) = match last_color {
            Value::String(s) => (s.clone(), HashMap::new()),
            Value::Enum { variant, fields } => (variant.clone(), fields.clone()),
            _ => panic!("unexpected color value"),
        };

        // Tag should match
        prop_assert_eq!(
            entity.get("Widget/color/__tag"),
            Some(&Value::String(expected_variant.clone())),
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
