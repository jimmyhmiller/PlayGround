use std::collections::HashMap;
use std::sync::Arc;

use datalog_db::datom::Value;
use datalog_db::db::Database;
use datalog_db::query::Query;
use datalog_db::schema::{EntityTypeDef, EnumTypeDef, EnumVariant, FieldDef, FieldType};
use datalog_db::storage::rocksdb_backend::RocksDbStorage;
use datalog_db::tx::TxOp;

/// Create a temporary database for testing.
fn test_db() -> (Arc<Database>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let storage = RocksDbStorage::open(dir.path()).unwrap();
    let storage = Arc::new(storage);
    let db = Database::open(storage).unwrap();
    (Arc::new(db), dir)
}

fn user_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "User".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: true,
            },
            FieldDef {
                name: "age".to_string(),
                field_type: FieldType::I64,
                required: false,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "email".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: true,
                indexed: true,
            },
        ],
    }
}

fn post_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "Post".to_string(),
        fields: vec![
            FieldDef {
                name: "title".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "author".to_string(),
                field_type: FieldType::Ref("User".to_string()),
                required: true,
                unique: false,
                indexed: true,
            },
        ],
    }
}

#[test]
fn test_define_type() {
    let (db, _dir) = test_db();

    let tx_id = db.define_type(user_type()).unwrap();
    assert!(tx_id > 0);
}

#[test]
fn test_transact_insert() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();

    assert!(result.tx_id > 0);
    assert_eq!(result.entity_ids.len(), 1);
    assert!(result.entity_ids[0] > 0);
    assert_eq!(result.datom_count, 4); // name + age + email + __type
}

#[test]
fn test_transact_and_get_entity() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert("age".to_string(), Value::I64(25));
    data.insert("email".to_string(), Value::String("bob@example.com".into()));

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();

    let eid = result.entity_ids[0];
    let entity = db.get_entity(eid).unwrap().unwrap();

    assert_eq!(entity.get("User/name"), Some(&Value::String("Bob".into())));
    assert_eq!(entity.get("User/age"), Some(&Value::I64(25)));
    assert_eq!(
        entity.get("__type"),
        Some(&Value::String("User".into()))
    );
}

#[test]
fn test_transact_update() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Charlie".into()));
    data.insert("age".to_string(), Value::I64(20));
    data.insert("email".to_string(), Value::String("charlie@example.com".into()));

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    // Update age
    let mut update_data = HashMap::new();
    update_data.insert("age".to_string(), Value::I64(21));

    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: Some(eid),
        data: update_data,
    }])
    .unwrap();

    let entity = db.get_entity(eid).unwrap().unwrap();
    // Should show latest age
    assert_eq!(entity.get("User/age"), Some(&Value::I64(21)));
    // Name should still be there
    assert_eq!(
        entity.get("User/name"),
        Some(&Value::String("Charlie".into()))
    );
}

#[test]
fn test_transact_retract() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Diana".into()));
    data.insert("age".to_string(), Value::I64(40));
    data.insert(
        "email".to_string(),
        Value::String("diana@example.com".into()),
    );

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    // Retract email
    db.transact(vec![TxOp::Retract {
        entity_type: "User".to_string(),
        entity: eid,
        fields: vec!["email".to_string()],
    }])
    .unwrap();

    let entity = db.get_entity(eid).unwrap().unwrap();
    assert_eq!(
        entity.get("User/name"),
        Some(&Value::String("Diana".into()))
    );
    assert_eq!(entity.get("User/age"), Some(&Value::I64(40)));
    assert!(entity.get("User/email").is_none()); // retracted
}

#[test]
fn test_schema_validation_unknown_type() {
    let (db, _dir) = test_db();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Eve".into()));

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "NonExistent".to_string(),
            entity: None,
            data,
        }]);

    assert!(result.is_err());
}

#[test]
fn test_schema_validation_type_mismatch() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Frank".into()));
    data.insert("age".to_string(), Value::String("not a number".into())); // wrong type

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }]);

    assert!(result.is_err());
}

#[test]
fn test_schema_validation_missing_required() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("age".to_string(), Value::I64(25)); // missing required "name"

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }]);

    assert!(result.is_err());
}

#[test]
fn test_basic_query() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert two users
    for (name, age, email) in [("Alice", 30, "alice@example.com"), ("Bob", 25, "bob@example.com")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.into()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Query all users
    let query_json = serde_json::json!({
        "find": ["?name", "?age"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name", "age": "?age"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();

    assert_eq!(result.columns, vec!["?name", "?age"]);
    assert_eq!(result.rows.len(), 2);

    // Check both users are present (order may vary)
    let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::String("Alice".into())));
    assert!(names.contains(&&Value::String("Bob".into())));
}

#[test]
fn test_query_with_constant_filter() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    for (name, age, email) in [("Alice", 30, "alice@example.com"), ("Bob", 25, "bob@example.com"), ("Charlie", 35, "charlie@example.com")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.into()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Query for Bob specifically
    let query_json = serde_json::json!({
        "find": ["?age"],
        "where": [
            {"bind": "?u", "type": "User", "name": "Bob", "age": "?age"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::I64(25));
}

#[test]
fn test_query_with_predicate() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    for (name, age, email) in [("Alice", 30, "alice@example.com"), ("Bob", 25, "bob@example.com"), ("Charlie", 35, "charlie@example.com")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.into()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Query users older than 27
    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name", "age": {"gt": 27}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();

    assert_eq!(result.rows.len(), 2);
    let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::String("Alice".into())));
    assert!(names.contains(&&Value::String("Charlie".into())));
}

#[test]
fn test_query_with_join() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    // Create users
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));
    let alice_result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let alice_id = alice_result.entity_ids[0];

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert("age".to_string(), Value::I64(25));
    data.insert("email".to_string(), Value::String("bob@example.com".into()));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Create posts by Alice
    for title in ["First Post", "Second Post"] {
        let mut data = HashMap::new();
        data.insert("title".to_string(), Value::String(title.into()));
        data.insert("author".to_string(), Value::Ref(alice_id));
        db.transact(vec![TxOp::Assert {
            entity_type: "Post".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Query: find all post titles by Alice
    let query_json = serde_json::json!({
        "find": ["?name", "?title"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();

    // Alice has 2 posts, Bob has 0 — so 2 result rows
    assert_eq!(result.rows.len(), 2);
    for row in &result.rows {
        assert_eq!(row[0], Value::String("Alice".into()));
    }
    let titles: Vec<&Value> = result.rows.iter().map(|r| &r[1]).collect();
    assert!(titles.contains(&&Value::String("First Post".into())));
    assert!(titles.contains(&&Value::String("Second Post".into())));
}

#[test]
fn test_time_travel_query() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert user at tx1
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));
    let result1 = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result1.entity_ids[0];
    let tx1 = result1.tx_id;

    // Update age at tx2
    let mut data = HashMap::new();
    data.insert("age".to_string(), Value::I64(31));
    let result2 = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: Some(eid),
            data,
        }])
        .unwrap();
    let _tx2 = result2.tx_id;

    // Query as-of tx1: should see age 30
    let query_json = serde_json::json!({
        "find": ["?age"],
        "where": [
            {"bind": "?u", "type": "User", "name": "Alice", "age": "?age"}
        ],
        "as_of": tx1
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::I64(30));

    // Query latest: should see age 31
    let query_json = serde_json::json!({
        "find": ["?age"],
        "where": [
            {"bind": "?u", "type": "User", "name": "Alice", "age": "?age"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::I64(31));
}

#[test]
fn test_schema_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();

    // Open, define type, close
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();
        db.define_type(user_type()).unwrap();

        // Transact a user
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("Persist".into()));
        data.insert("email".to_string(), Value::String("persist@example.com".into()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Reopen and verify schema is loaded
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();

        // Should be able to transact without re-defining
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("Persist2".into()));
        data.insert("email".to_string(), Value::String("persist2@example.com".into()));
        let result = db
            .transact(vec![TxOp::Assert {
                entity_type: "User".to_string(),
                entity: None,
                data,
            }]);
        assert!(result.is_ok());

        // And query should work
        let query_json = serde_json::json!({
            "find": ["?name"],
            "where": [
                {"bind": "?u", "type": "User", "name": "?name"}
            ]
        });
        let query = Query::from_json(&query_json).unwrap();
        let result = db.query(&query).unwrap();
        assert_eq!(result.rows.len(), 2); // both users
    }
}

#[test]
fn test_tcp_protocol_roundtrip() {
    use datalog_db::protocol;
    use std::net::TcpListener;

    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    // Server thread
    let server_handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        protocol::server_handshake(&mut stream).unwrap();

        // Read one message
        let msg = protocol::read_message(&mut stream).unwrap();
        assert_eq!(msg.request_id, 1);

        // Send response
        let response = serde_json::json!({"status": "ok", "data": "hello"});
        protocol::write_message(&mut stream, msg.request_id, &response).unwrap();
    });

    // Client
    let mut client = std::net::TcpStream::connect(addr).unwrap();
    protocol::client_handshake(&mut client).unwrap();

    // Send a message
    let payload = serde_json::json!({"type": "status"});
    protocol::write_message(&mut client, 1, &payload).unwrap();

    // Read response
    let response = protocol::read_message(&mut client).unwrap();
    assert_eq!(response.request_id, 1);
    assert_eq!(response.payload["status"], "ok");

    server_handle.join().unwrap();
}

#[test]
fn test_full_server_define_transact_query() {
    use datalog_db::protocol;
    use std::net::TcpListener;

    let (db, _dir) = test_db();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let db_clone = db.clone();
    let server_handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        protocol::server_handshake(&mut stream).unwrap();

        // Process messages until client disconnects
        loop {
            let msg = match protocol::read_message(&mut stream) {
                Ok(msg) => msg,
                Err(_) => break,
            };

            let request_type = msg
                .payload
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("");

            let response = match request_type {
                "define" => {
                    let type_def =
                        datalog_db::db::parse_define_request(&msg.payload).unwrap();
                    let name = type_def.name.clone();
                    let tx_id = db_clone.define_type(type_def).unwrap();
                    serde_json::json!({"status": "ok", "data": {"tx_id": tx_id, "entity_type": name}})
                }
                "transact" => {
                    let ops: Vec<datalog_db::tx::TxOp> = msg
                        .payload
                        .get("ops")
                        .unwrap()
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|o| datalog_db::tx::TxOp::from_json(o).unwrap())
                        .collect();
                    let result = db_clone.transact(ops).unwrap();
                    serde_json::json!({
                        "status": "ok",
                        "data": {
                            "tx_id": result.tx_id,
                            "entity_ids": result.entity_ids,
                        }
                    })
                }
                "query" => {
                    let query =
                        datalog_db::query::Query::from_json(&msg.payload).unwrap();
                    let result = db_clone.query(&query).unwrap();
                    let rows: Vec<Vec<serde_json::Value>> = result
                        .rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|v| match v {
                                    Value::String(s) => serde_json::json!(s),
                                    Value::I64(n) => serde_json::json!(n),
                                    _ => serde_json::json!(null),
                                })
                                .collect()
                        })
                        .collect();
                    serde_json::json!({
                        "status": "ok",
                        "data": {"columns": result.columns, "rows": rows}
                    })
                }
                _ => serde_json::json!({"status": "error", "error": "unknown type"}),
            };

            protocol::write_message(&mut stream, msg.request_id, &response).unwrap();
        }
    });

    // Client side
    let mut client = std::net::TcpStream::connect(addr).unwrap();
    protocol::client_handshake(&mut client).unwrap();

    // 1. Define User type
    let define_payload = serde_json::json!({
        "type": "define",
        "entity_type": "User",
        "fields": [
            {"name": "name", "type": "string", "required": true},
            {"name": "age", "type": "i64"}
        ]
    });
    protocol::write_message(&mut client, 1, &define_payload).unwrap();
    let resp = protocol::read_message(&mut client).unwrap();
    assert_eq!(resp.payload["status"], "ok");

    // 2. Transact: insert users
    let transact_payload = serde_json::json!({
        "type": "transact",
        "ops": [
            {"assert": "User", "data": {"name": "Alice", "age": 30}},
            {"assert": "User", "data": {"name": "Bob", "age": 25}}
        ]
    });
    protocol::write_message(&mut client, 2, &transact_payload).unwrap();
    let resp = protocol::read_message(&mut client).unwrap();
    assert_eq!(resp.payload["status"], "ok");
    assert_eq!(resp.payload["data"]["entity_ids"].as_array().unwrap().len(), 2);

    // 3. Query all users
    let query_payload = serde_json::json!({
        "type": "query",
        "find": ["?name", "?age"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name", "age": "?age"}
        ]
    });
    protocol::write_message(&mut client, 3, &query_payload).unwrap();
    let resp = protocol::read_message(&mut client).unwrap();
    assert_eq!(resp.payload["status"], "ok");
    let rows = resp.payload["data"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);

    // Drop client to close connection, ending the server loop
    drop(client);
    server_handle.join().unwrap();
}

// --- Enum (sum type) tests ---

fn shape_enum() -> EnumTypeDef {
    EnumTypeDef {
        name: "Shape".to_string(),
        variants: vec![
            EnumVariant {
                name: "Circle".to_string(),
                fields: vec![FieldDef {
                    name: "radius".to_string(),
                    field_type: FieldType::F64,
                    required: true,
                    unique: false,
                    indexed: false,
                }],
            },
            EnumVariant {
                name: "Rect".to_string(),
                fields: vec![
                    FieldDef {
                        name: "w".to_string(),
                        field_type: FieldType::F64,
                        required: true,
                        unique: false,
                        indexed: false,
                    },
                    FieldDef {
                        name: "h".to_string(),
                        field_type: FieldType::F64,
                        required: true,
                        unique: false,
                        indexed: false,
                    },
                ],
            },
            EnumVariant {
                name: "Point".to_string(),
                fields: vec![],
            },
        ],
    }
}

fn drawing_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "Drawing".to_string(),
        fields: vec![
            FieldDef {
                name: "label".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "shape".to_string(),
                field_type: FieldType::Enum("Shape".to_string()),
                required: true,
                unique: false,
                indexed: false,
            },
        ],
    }
}

#[test]
fn test_define_enum() {
    let (db, _dir) = test_db();
    let tx_id = db.define_enum(shape_enum()).unwrap();
    assert!(tx_id > 0);
}

#[test]
fn test_enum_insert_and_query() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert a circle
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("my circle".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]) })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    assert!(result.tx_id > 0);
    // __type + label + shape.__tag + shape.Circle/radius = 4 datoms
    assert_eq!(result.datom_count, 4);

    // Query: find all circles with their radius
    let query_json = serde_json::json!({
        "find": ["?label", "?r"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Circle", "radius": "?r"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("my circle".into()));
    assert_eq!(result.rows[0][1], Value::F64(5.0));
}

#[test]
fn test_enum_unit_variant() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert a Point (unit variant — no fields)
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("origin".into()));
    data.insert("shape".to_string(), Value::String("Point".into()));
    db.transact(vec![TxOp::Assert {
        entity_type: "Drawing".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Query for points using constant match
    let query_json = serde_json::json!({
        "find": ["?label"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Point"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("origin".into()));
}

#[test]
fn test_enum_match_filters_variants() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert one of each variant
    let drawings = vec![
        (
            "my circle",
            Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]) })),
        ),
        (
            "my rect",
            Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Rect".to_string(), fields: HashMap::from([
                    ("w".to_string(), Value::F64(10.0)),
                    ("h".to_string(), Value::F64(20.0)),
                ]) })),
        ),
        ("origin", Value::String("Point".into())),
    ];

    for (label, shape) in drawings {
        let mut data = HashMap::new();
        data.insert("label".to_string(), Value::String(label.into()));
        data.insert("shape".to_string(), shape);
        db.transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Query only circles
    let query_json = serde_json::json!({
        "find": ["?label", "?r"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Circle", "radius": "?r"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("my circle".into()));
    assert_eq!(result.rows[0][1], Value::F64(5.0));

    // Query only rects
    let query_json = serde_json::json!({
        "find": ["?label", "?w", "?h"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Rect", "w": "?w", "h": "?h"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("my rect".into()));
    assert_eq!(result.rows[0][1], Value::F64(10.0));
    assert_eq!(result.rows[0][2], Value::F64(20.0));

    // Query only points
    let query_json = serde_json::json!({
        "find": ["?label"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Point"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("origin".into()));
}

#[test]
fn test_enum_bind_variant_tag() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    for (label, shape) in [
        (
            "c1",
            Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(1.0))]) })),
        ),
        ("p1", Value::String("Point".into())),
    ] {
        let mut data = HashMap::new();
        data.insert("label".to_string(), Value::String(label.into()));
        data.insert("shape".to_string(), shape);
        db.transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Bind the variant tag to a variable
    let query_json = serde_json::json!({
        "find": ["?label", "?variant"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label", "shape": "?variant"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 2);

    let mut tags: Vec<_> = result
        .rows
        .iter()
        .map(|r| {
            (
                r[0].clone(),
                r[1].clone(),
            )
        })
        .collect();
    tags.sort_by(|a, b| format!("{}", a.0).cmp(&format!("{}", b.0)));

    assert_eq!(tags[0].0, Value::String("c1".into()));
    assert_eq!(tags[0].1, Value::String("Circle".into()));
    assert_eq!(tags[1].0, Value::String("p1".into()));
    assert_eq!(tags[1].1, Value::String("Point".into()));
}

#[test]
fn test_enum_variant_change() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert as Circle
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("morphing".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]) })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    // Change to Rect
    let mut data = HashMap::new();
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Rect".to_string(), fields: HashMap::from([
                ("w".to_string(), Value::F64(10.0)),
                ("h".to_string(), Value::F64(20.0)),
            ]) })),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "Drawing".to_string(),
        entity: Some(eid),
        data,
    }])
    .unwrap();

    // Should NOT match as Circle anymore
    let query_json = serde_json::json!({
        "find": ["?label"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Circle", "radius": "?r"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 0);

    // Should match as Rect
    let query_json = serde_json::json!({
        "find": ["?label", "?w", "?h"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Rect", "w": "?w", "h": "?h"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][1], Value::F64(10.0));
    assert_eq!(result.rows[0][2], Value::F64(20.0));
}

#[test]
fn test_enum_same_variant_stale_field_retraction() {
    let (db, _dir) = test_db();

    // Enum with optional fields
    let status_enum = EnumTypeDef {
        name: "Status".to_string(),
        variants: vec![
            EnumVariant {
                name: "Active".to_string(),
                fields: vec![],
            },
            EnumVariant {
                name: "Suspended".to_string(),
                fields: vec![
                    FieldDef {
                        name: "reason".to_string(),
                        field_type: FieldType::String,
                        required: false,
                        unique: false,
                        indexed: false,
                    },
                    FieldDef {
                        name: "until".to_string(),
                        field_type: FieldType::I64,
                        required: false,
                        unique: false,
                        indexed: false,
                    },
                ],
            },
        ],
    };
    db.define_enum(status_enum).unwrap();

    let account_type = EntityTypeDef {
        unique_keys: vec![],
        name: "Account".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "status".to_string(),
                field_type: FieldType::Enum("Status".to_string()),
                required: true,
                unique: false,
                indexed: false,
            },
        ],
    };
    db.define_type(account_type).unwrap();

    // Insert with Suspended{reason: "TOS", until: 1000}
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert(
        "status".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Suspended".to_string(), fields: HashMap::from([
                ("reason".to_string(), Value::String("TOS".into())),
                ("until".to_string(), Value::I64(1000)),
            ]) })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Account".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    // Update to Suspended{reason: "Spam"} — no 'until' field
    let mut data = HashMap::new();
    data.insert(
        "status".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Suspended".to_string(), fields: HashMap::from([("reason".to_string(), Value::String("Spam".into()))]) })),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "Account".to_string(),
        entity: Some(eid),
        data,
    }])
    .unwrap();

    // Query: 'until' should NOT leak through from the first assertion
    let query_json = serde_json::json!({
        "find": ["?reason", "?until"],
        "where": [
            {"bind": "?a", "type": "Account",
             "status": {"match": "Suspended", "reason": "?reason", "until": "?until"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    // Should find 0 rows because 'until' was retracted
    assert_eq!(result.rows.len(), 0);

    // But querying just for reason should work
    let query_json = serde_json::json!({
        "find": ["?reason"],
        "where": [
            {"bind": "?a", "type": "Account",
             "status": {"match": "Suspended", "reason": "?reason"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Spam".into()));
}

#[test]
fn test_enum_retract_whole_field() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("gone".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(3.0))]) })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    // Retract the shape field entirely
    db.transact(vec![TxOp::Retract {
        entity_type: "Drawing".to_string(),
        entity: eid,
        fields: vec!["shape".to_string()],
    }])
    .unwrap();

    // Query should find no circles
    let query_json = serde_json::json!({
        "find": ["?label"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Circle", "radius": "?r"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 0);
}

#[test]
fn test_enum_schema_validation() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Wrong variant name
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("bad".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue {
            variant: "Triangle".to_string(), // doesn't exist
            fields: HashMap::new(),
        })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }]);
    assert!(result.is_err());

    // Wrong field type in variant
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("bad".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue {
            variant: "Circle".to_string(),
            fields: HashMap::from([(
                "radius".to_string(),
                Value::String("not a number".into()),
            )]),
        })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }]);
    assert!(result.is_err());

    // Missing required variant field
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("bad".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue {
            variant: "Rect".to_string(),
            fields: HashMap::from([("w".to_string(), Value::F64(10.0))]), // missing 'h'
        })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }]);
    assert!(result.is_err());
}

#[test]
fn test_enum_json_wire_format() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert via JSON (as it would come over the wire)
    let op_json = serde_json::json!({
        "assert": "Drawing",
        "data": {
            "label": "wire circle",
            "shape": {"Circle": {"radius": 7.5}}
        }
    });
    let op = TxOp::from_json(&op_json).unwrap();
    db.transact(vec![op]).unwrap();

    // Also test unit variant over the wire
    let op_json = serde_json::json!({
        "assert": "Drawing",
        "data": {
            "label": "wire point",
            "shape": "Point"
        }
    });
    let op = TxOp::from_json(&op_json).unwrap();
    db.transact(vec![op]).unwrap();

    // Query to verify
    let query_json = serde_json::json!({
        "find": ["?label", "?r"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Circle", "radius": "?r"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("wire circle".into()));
    assert_eq!(result.rows[0][1], Value::F64(7.5));
}

#[test]
fn test_enum_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();

    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();
        db.define_enum(shape_enum()).unwrap();
        db.define_type(drawing_type()).unwrap();

        let mut data = HashMap::new();
        data.insert("label".to_string(), Value::String("persistent".into()));
        data.insert(
            "shape".to_string(),
            Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(42.0))]) })),
        );
        db.transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Reopen
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();

        // Should be able to insert without re-defining
        let mut data = HashMap::new();
        data.insert("label".to_string(), Value::String("after reopen".into()));
        data.insert(
            "shape".to_string(),
            Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Rect".to_string(), fields: HashMap::from([
                    ("w".to_string(), Value::F64(1.0)),
                    ("h".to_string(), Value::F64(2.0)),
                ]) })),
        );
        db.transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }])
        .unwrap();

        // Query the circle from the first session
        let query_json = serde_json::json!({
            "find": ["?label", "?r"],
            "where": [
                {"bind": "?d", "type": "Drawing", "label": "?label",
                 "shape": {"match": "Circle", "radius": "?r"}}
            ]
        });
        let query = Query::from_json(&query_json).unwrap();
        let result = db.query(&query).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("persistent".into()));
        assert_eq!(result.rows[0][1], Value::F64(42.0));
    }
}

// --- Snapshot Tests ---
// These tests show the exact datom state before and after each transaction,
// verifying that retraction semantics work correctly for append-only storage.

/// Format all datoms into a readable snapshot string.
fn snapshot_datoms(datoms: &[datalog_db::datom::Datom]) -> String {
    let mut lines: Vec<String> = datoms
        .iter()
        .filter(|d| !d.attribute.starts_with("__schema_"))
        .map(|d| {
            format!(
                "[e={} {:30} {:20} tx={} {}]",
                d.entity,
                d.attribute,
                d.value,
                d.tx,
                if d.added { "assert" } else { "retract" }
            )
        })
        .collect();
    lines.sort();
    lines.join("\n")
}

#[test]
fn test_snapshot_scalar_insert() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Before: no data datoms
    let before = db.all_datoms().unwrap();
    assert_eq!(
        snapshot_datoms(&before),
        "",
        "before insert, no data datoms"
    );

    // Insert Alice
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    let after = db.all_datoms().unwrap();
    insta::assert_snapshot!("scalar_insert", snapshot_datoms(&after));
}

#[test]
fn test_snapshot_scalar_update() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    let after_insert = db.all_datoms().unwrap();
    insta::assert_snapshot!("scalar_update__after_insert", snapshot_datoms(&after_insert));

    // Update age 30 -> 31 (should retract 30, assert 31)
    let mut data = HashMap::new();
    data.insert("age".to_string(), Value::I64(31));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: Some(eid),
        data,
    }])
    .unwrap();

    let after_update = db.all_datoms().unwrap();
    insta::assert_snapshot!("scalar_update__after_update", snapshot_datoms(&after_update));
}

#[test]
fn test_snapshot_enum_insert() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("my circle".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]) })),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "Drawing".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    let datoms = db.all_datoms().unwrap();
    insta::assert_snapshot!("enum_insert", snapshot_datoms(&datoms));
}

#[test]
fn test_snapshot_enum_variant_change() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert Circle
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("shape1".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]) })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Drawing".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    let after_insert = db.all_datoms().unwrap();
    insta::assert_snapshot!(
        "enum_variant_change__after_insert",
        snapshot_datoms(&after_insert)
    );

    // Change Circle -> Rect (retracts tag + Circle/radius, asserts tag + Rect/w,h)
    let mut data = HashMap::new();
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Rect".to_string(), fields: HashMap::from([
                ("w".to_string(), Value::F64(10.0)),
                ("h".to_string(), Value::F64(20.0)),
            ]) })),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "Drawing".to_string(),
        entity: Some(eid),
        data,
    }])
    .unwrap();

    let after_change = db.all_datoms().unwrap();
    insta::assert_snapshot!(
        "enum_variant_change__after_change",
        snapshot_datoms(&after_change)
    );
}

#[test]
fn test_snapshot_enum_same_variant_update() {
    let (db, _dir) = test_db();

    let status_enum = EnumTypeDef {
        name: "Status".to_string(),
        variants: vec![
            EnumVariant {
                name: "Active".to_string(),
                fields: vec![],
            },
            EnumVariant {
                name: "Suspended".to_string(),
                fields: vec![
                    FieldDef {
                        name: "reason".to_string(),
                        field_type: FieldType::String,
                        required: false,
                        unique: false,
                        indexed: false,
                    },
                    FieldDef {
                        name: "until".to_string(),
                        field_type: FieldType::I64,
                        required: false,
                        unique: false,
                        indexed: false,
                    },
                ],
            },
        ],
    };
    db.define_enum(status_enum).unwrap();

    let account_type = EntityTypeDef {
        unique_keys: vec![],
        name: "Account".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "status".to_string(),
                field_type: FieldType::Enum("Status".to_string()),
                required: true,
                unique: false,
                indexed: false,
            },
        ],
    };
    db.define_type(account_type).unwrap();

    // Insert: Suspended{reason: "TOS", until: 1000}
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert(
        "status".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Suspended".to_string(), fields: HashMap::from([
                ("reason".to_string(), Value::String("TOS".into())),
                ("until".to_string(), Value::I64(1000)),
            ]) })),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Account".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    let after_insert = db.all_datoms().unwrap();
    insta::assert_snapshot!(
        "enum_same_variant_update__after_insert",
        snapshot_datoms(&after_insert)
    );

    // Update: Suspended{reason: "Spam"} — drops 'until', changes 'reason'
    let mut data = HashMap::new();
    data.insert(
        "status".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Suspended".to_string(), fields: HashMap::from([("reason".to_string(), Value::String("Spam".into()))]) })),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "Account".to_string(),
        entity: Some(eid),
        data,
    }])
    .unwrap();

    let after_update = db.all_datoms().unwrap();
    insta::assert_snapshot!(
        "enum_same_variant_update__after_update",
        snapshot_datoms(&after_update)
    );
}

// --- Concurrent stress test ---

#[test]
fn test_concurrent_stress() {
    let (db, _dir) = test_db();
    db.define_type(EntityTypeDef {
        unique_keys: vec![],
        name: "Counter".to_string(),
        fields: vec![
            FieldDef {
                name: "label".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "value".to_string(),
                field_type: FieldType::I64,
                required: true,
                unique: false,
                indexed: false,
            },
        ],
    })
    .unwrap();

    let mut handles = Vec::new();
    for i in 0..50 {
        let db = db.clone();
        handles.push(std::thread::spawn(move || {
            // Insert
            let mut data = HashMap::new();
            data.insert("label".to_string(), Value::String(format!("task_{}", i).into()));
            data.insert("value".to_string(), Value::I64(i));
            let result = db
                .transact(vec![TxOp::Assert {
                    entity_type: "Counter".to_string(),
                    entity: None,
                    data,
                }])
                .unwrap();

            // Query back
            let query_json = serde_json::json!({
                "find": ["?label"],
                "where": [
                    {"bind": "?c", "type": "Counter", "label": format!("task_{}", i)}
                ]
            });
            let query = Query::from_json(&query_json).unwrap();
            let qr = db.query(&query).unwrap();
            assert!(!qr.rows.is_empty(), "task {} should find its entity", i);

            result.entity_ids[0]
        }));
    }

    let mut all_eids = Vec::new();
    for h in handles {
        all_eids.push(h.join().unwrap());
    }

    // All 50 entities should exist
    let query_json = serde_json::json!({
        "find": ["?label", "?value"],
        "where": [
            {"bind": "?c", "type": "Counter", "label": "?label", "value": "?value"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 50);
}

/// Regression for the query-cache read-after-write race: a thread inserts
/// an entity, then immediately queries it back. With a cold per-type cache,
/// a concurrent thread's stale load could poison the cache and hide the just
/// committed write. Each round uses a fresh DB (cold cache) since that is
/// exactly the window the bug lived in; looping raises detection probability.
#[test]
fn test_concurrent_read_after_write_no_stale_cache() {
    for round in 0..12 {
        let (db, _dir) = test_db();
        db.define_type(EntityTypeDef {
            unique_keys: vec![],
            name: "Counter".to_string(),
            fields: vec![
                FieldDef {
                    name: "label".to_string(),
                    field_type: FieldType::String,
                    required: true,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "value".to_string(),
                    field_type: FieldType::I64,
                    required: true,
                    unique: false,
                    indexed: false,
                },
            ],
        })
        .unwrap();

        let mut handles = Vec::new();
        for i in 0..40 {
            let db = db.clone();
            handles.push(std::thread::spawn(move || {
                let mut data = HashMap::new();
                data.insert("label".to_string(), Value::String(format!("c_{}", i).into()));
                data.insert("value".to_string(), Value::I64(i));
                db.transact(vec![TxOp::Assert {
                    entity_type: "Counter".to_string(),
                    entity: None,
                    data,
                }])
                .unwrap();

                let q = Query::from_json(&serde_json::json!({
                    "find": ["?c"],
                    "where": [{"bind": "?c", "type": "Counter", "label": format!("c_{}", i)}]
                }))
                .unwrap();
                let qr = db.query(&q).unwrap();
                assert!(
                    !qr.rows.is_empty(),
                    "round {round}: thread {i} could not read back its own write"
                );
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}

// --- Unique constraint tests ---

#[test]
fn test_unique_basic() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert user with unique email
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Try to insert another user with the same email — should fail
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Unique constraint violated"), "got: {}", err);
}

#[test]
fn test_unique_same_entity_update() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let eid = result.entity_ids[0];

    // Re-assert the same unique value on the same entity — should be ok (no-op)
    let mut data = HashMap::new();
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: Some(eid),
            data,
        }]);
    assert!(result.is_ok());
}

#[test]
fn test_unique_after_retract() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Alice claims the email
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let alice_eid = result.entity_ids[0];

    // Retract Alice's email
    db.transact(vec![TxOp::Retract {
        entity_type: "User".to_string(),
        entity: alice_eid,
        fields: vec!["email".to_string()],
    }])
    .unwrap();

    // Bob can now take the email
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }]);
    assert!(result.is_ok());
}

#[test]
fn test_unique_update_to_taken_value() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Alice
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Bob with different email
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert(
        "email".to_string(),
        Value::String("bob@example.com".into()),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let bob_eid = result.entity_ids[0];

    // Bob tries to take Alice's email — should fail
    let mut data = HashMap::new();
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: Some(bob_eid),
            data,
        }]);
    assert!(result.is_err());
}

#[test]
fn test_unique_within_single_tx() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Two inserts in one transaction with same unique email — should fail
    let mut data1 = HashMap::new();
    data1.insert("name".to_string(), Value::String("Alice".into()));
    data1.insert(
        "email".to_string(),
        Value::String("shared@example.com".into()),
    );
    let mut data2 = HashMap::new();
    data2.insert("name".to_string(), Value::String("Bob".into()));
    data2.insert(
        "email".to_string(),
        Value::String("shared@example.com".into()),
    );
    let result = db
        .transact(vec![
            TxOp::Assert {
                entity_type: "User".to_string(),
                entity: None,
                data: data1,
            },
            TxOp::Assert {
                entity_type: "User".to_string(),
                entity: None,
                data: data2,
            },
        ]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Unique constraint violated"), "got: {}", err);
}

#[test]
fn test_unique_different_types_independent() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Define a second type with its own unique field
    db.define_type(EntityTypeDef {
        unique_keys: vec![],
        name: "Admin".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "email".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: true,
                indexed: true,
            },
        ],
    })
    .unwrap();

    // User with email
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Admin with same email value — should succeed (different type → different attribute)
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice Admin".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Admin".to_string(),
            entity: None,
            data,
        }]);
    assert!(result.is_ok());
}

// --- Index-based query optimization tests ---

#[test]
fn test_query_constant_uses_index() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert 100 users, one named "target"
    for i in 0..100 {
        let mut data = HashMap::new();
        let name = if i == 42 {
            "target".to_string()
        } else {
            format!("user_{}", i)
        };
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("age".to_string(), Value::I64(i));
        data.insert("email".to_string(), Value::String(format!("user{}@example.com", i).into()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Query by exact constant — should use AVET index
    let query_json = serde_json::json!({
        "find": ["?age"],
        "where": [
            {"bind": "?u", "type": "User", "name": "target", "age": "?age"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::I64(42));
}

#[test]
fn test_query_bound_variable_uses_index() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    // Create Alice
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));
    let alice_result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let alice_id = alice_result.entity_ids[0];

    // Create Bob
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert("email".to_string(), Value::String("bob@example.com".into()));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Create posts — some by Alice, some by Bob
    for i in 0..10 {
        let mut data = HashMap::new();
        data.insert("title".to_string(), Value::String(format!("Post {}", i).into()));
        data.insert("author".to_string(), Value::Ref(alice_id));
        db.transact(vec![TxOp::Assert {
            entity_type: "Post".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Join query: second clause should use bound ?u to index into Post/author
    let query_json = serde_json::json!({
        "find": ["?title"],
        "where": [
            {"bind": "?u", "type": "User", "name": "Alice"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 10);
    for row in &result.rows {
        // All titles should be strings starting with "Post "
        if let Value::String(s) = &row[0] {
            assert!(s.starts_with("Post "));
        } else {
            panic!("expected string title");
        }
    }
}

#[test]
fn test_query_enum_field_still_works() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert drawings
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("circle1".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Circle".to_string(), fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]) })),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "Drawing".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("rect1".into()));
    data.insert(
        "shape".to_string(),
        Value::Enum(Box::new(datalog_db::datom::EnumValue { variant: "Rect".to_string(), fields: HashMap::from([
                ("w".to_string(), Value::F64(10.0)),
                ("h".to_string(), Value::F64(20.0)),
            ]) })),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "Drawing".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Query with constant on enum field — should fall back to scan and still work
    let query_json = serde_json::json!({
        "find": ["?label", "?r"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Circle", "radius": "?r"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("circle1".into()));
    assert_eq!(result.rows[0][1], Value::F64(5.0));
}

#[test]
fn test_query_multiple_constants() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert several users
    for (name, age, email) in [
        ("Alice", 30, "alice@example.com"),
        ("Bob", 25, "bob@example.com"),
        ("Charlie", 30, "charlie@example.com"),
    ] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.into()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Query with two constants — intersection of AVET lookups
    let query_json = serde_json::json!({
        "find": ["?email"],
        "where": [
            {"bind": "?u", "type": "User", "name": "Alice", "age": 30, "email": "?email"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        result.rows[0][0],
        Value::String("alice@example.com".into())
    );
}

#[test]
fn test_query_predicate_falls_back() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    for (name, age, email) in [("Alice", 30, "alice@example.com"), ("Bob", 25, "bob@example.com"), ("Charlie", 35, "charlie@example.com")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.into()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Predicate pattern — can't use AVET index, should fall back to scan
    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name", "age": {"gt": 27}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 2);
    let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::String("Alice".into())));
    assert!(names.contains(&&Value::String("Charlie".into())));
}

// --- Retract entity (soft delete) tests ---

#[test]
fn test_retract_entity() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert two users
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let alice_eid = result.entity_ids[0];
    let insert_tx = result.tx_id;

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert("age".to_string(), Value::I64(25));
    data.insert("email".to_string(), Value::String("bob@example.com".into()));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Retract Alice entirely
    db.transact(vec![TxOp::RetractEntity {
        entity_type: "User".to_string(),
        entity: alice_eid,
    }])
    .unwrap();

    // get_entity should return None (all attributes retracted)
    let entity = db.get_entity(alice_eid).unwrap();
    assert!(entity.is_none(), "retracted entity should return None");

    // Query should only find Bob
    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Bob".into()));

    // Time-travel query as_of the insert tx should still show Alice
    let query_json = serde_json::json!({
        "find": ["?name", "?age"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name", "age": "?age"}
        ],
        "as_of": insert_tx
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::String("Alice".into())));
}

// --- Wall-clock timestamp tests ---

#[test]
fn test_transact_returns_timestamp() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));

    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();

    let after = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    assert!(result.timestamp_ms >= before, "timestamp_ms {} should be >= before {}", result.timestamp_ms, before);
    assert!(result.timestamp_ms <= after, "timestamp_ms {} should be <= after {}", result.timestamp_ms, after);
}

#[test]
fn test_as_of_time_between_transactions() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert Alice
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));
    let result1 = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let _ts1 = result1.timestamp_ms;

    // Small delay to ensure distinct timestamps
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Get a midpoint timestamp
    let midpoint = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    std::thread::sleep(std::time::Duration::from_millis(10));

    // Insert Bob
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".into()));
    data.insert("age".to_string(), Value::I64(25));
    data.insert("email".to_string(), Value::String("bob@example.com".into()));
    let result2 = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let _ts2 = result2.timestamp_ms;

    // Query as_of_time at midpoint — should only see Alice
    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"}
        ],
        "as_of_time": midpoint
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alice".into()));

    // Query latest — should see both
    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn test_as_of_time_before_all_transactions() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Capture time before any data transactions
    let before_all = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    // Subtract 1000ms to be safely before the define_type tx
    let before_all = before_all.saturating_sub(1000);

    // Insert Alice
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".into()));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Query as_of_time before everything — should return no data
    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"}
        ],
        "as_of_time": before_all
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 0, "should find no data before all transactions");
}

// --- Query planner tests ---

#[test]
fn test_hash_join_correctness() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    // Insert 20 users
    for i in 0..20 {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(format!("user_{}", i).into()));
        data.insert("age".to_string(), Value::I64(20 + i));
        data.insert(
            "email".to_string(),
            Value::String(format!("user_{}@example.com", i).into()),
        );
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Get entity IDs for first 5 users to create posts
    let all_users_json = serde_json::json!({
        "find": ["?u", "?name"],
        "where": [{"bind": "?u", "type": "User", "name": "?name"}]
    });
    let all_users = db.query(&Query::from_json(&all_users_json).unwrap()).unwrap();
    let user_ids: Vec<u64> = all_users.rows.iter().filter_map(|row| {
        if let Value::Ref(id) = &row[0] { Some(*id) } else { None }
    }).collect();

    // Create 3 posts for each of the first 5 users
    for &uid in &user_ids[..5] {
        for j in 0..3 {
            let mut data = HashMap::new();
            data.insert(
                "title".to_string(),
                Value::String(format!("Post {} by user {}", j, uid).into()),
            );
            data.insert("author".to_string(), Value::Ref(uid));
            db.transact(vec![TxOp::Assert {
                entity_type: "Post".to_string(),
                entity: None,
                data,
            }])
            .unwrap();
        }
    }

    // Join query — should find 15 results (5 users × 3 posts each)
    let query_json = serde_json::json!({
        "find": ["?name", "?title"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 15);

    // All rows should have valid name and title
    for row in &result.rows {
        if let Value::String(name) = &row[0] {
            assert!(name.starts_with("user_"));
        } else {
            panic!("expected string name");
        }
        if let Value::String(title) = &row[1] {
            assert!(title.starts_with("Post "));
        } else {
            panic!("expected string title");
        }
    }
}

#[test]
fn test_explain_returns_plan() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    // Insert some data so counts are non-zero
    for i in 0..5 {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(format!("user_{}", i).into()));
        data.insert(
            "email".to_string(),
            Value::String(format!("user_{}@example.com", i).into()),
        );
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    let query_json = serde_json::json!({
        "find": ["?name", "?title"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();

    let plan = db.explain(&query).unwrap();
    let display = format!("{}", plan);
    let json = plan.to_json();

    // Display should contain Project and scan types
    assert!(display.contains("Project"), "display should show Project: {}", display);
    assert!(display.contains("Scan"), "display should show Scan: {}", display);
    assert!(display.contains("User"), "display should mention User: {}", display);
    assert!(display.contains("Post"), "display should mention Post: {}", display);

    // JSON should have expected structure
    assert_eq!(json["node"], "Project");
    assert!(json["variables"].as_array().unwrap().contains(&serde_json::json!("?name")));
    assert!(json["variables"].as_array().unwrap().contains(&serde_json::json!("?title")));
    assert!(json["input"].is_object());
}

#[test]
fn test_explain_via_query_flag() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Query with explain: true — should NOT execute, just return plan info
    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [{"bind": "?u", "type": "User", "name": "?name"}],
        "explain": true
    });
    let query = Query::from_json(&query_json).unwrap();
    assert!(query.explain);

    // The explain method should work
    let plan = db.explain(&query).unwrap();
    let display = format!("{}", plan);
    assert!(display.contains("Project"));
}

#[test]
fn test_explain_wire_protocol() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    let server = datalog_db::server::Server::bind("127.0.0.1:0", db).unwrap();
    let addr = server.local_addr().unwrap().to_string();
    std::thread::spawn(move || { let _ = server.run(); });
    std::thread::sleep(std::time::Duration::from_millis(50));

    let mut client = datalog_db::client::Client::connect(&addr).unwrap();

    let result = client.explain(&serde_json::json!({
        "find": ["?name", "?title"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"}
        ]
    })).unwrap();

    assert!(!result.display.is_empty(), "display should not be empty");
    assert!(result.display.contains("Project"), "display: {}", result.display);
    assert!(result.plan.is_object(), "plan should be JSON object");
    assert_eq!(result.plan["node"], "Project");
}

#[test]
fn test_reorder_preserves_results() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    // Insert users
    let mut user_ids = Vec::new();
    for i in 0..10 {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(format!("user_{}", i).into()));
        data.insert("age".to_string(), Value::I64(20 + i));
        data.insert(
            "email".to_string(),
            Value::String(format!("user_{}@example.com", i).into()),
        );
        let result = db
            .transact(vec![TxOp::Assert {
                entity_type: "User".to_string(),
                entity: None,
                data,
            }])
            .unwrap();
        user_ids.push(result.entity_ids[0]);
    }

    // Insert posts for first 3 users
    for &uid in &user_ids[..3] {
        let mut data = HashMap::new();
        data.insert("title".to_string(), Value::String(format!("Post by {}", uid).into()));
        data.insert("author".to_string(), Value::Ref(uid));
        db.transact(vec![TxOp::Assert {
            entity_type: "Post".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Query with clauses in "natural" order (User first, then Post)
    let q1_json = serde_json::json!({
        "find": ["?name", "?title"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"}
        ]
    });
    let q1 = Query::from_json(&q1_json).unwrap();
    let result1 = db.query(&q1).unwrap();

    // Query with clauses in reversed order (Post first, then User)
    let q2_json = serde_json::json!({
        "find": ["?name", "?title"],
        "where": [
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"},
            {"bind": "?u", "type": "User", "name": "?name"}
        ]
    });
    let q2 = Query::from_json(&q2_json).unwrap();
    let result2 = db.query(&q2).unwrap();

    // Both should produce same number of results
    assert_eq!(result1.rows.len(), result2.rows.len());
    assert_eq!(result1.rows.len(), 3);

    // Collect and sort for comparison
    let mut names1: Vec<String> = result1.rows.iter().map(|r| format!("{}", r[0])).collect();
    let mut names2: Vec<String> = result2.rows.iter().map(|r| format!("{}", r[0])).collect();
    names1.sort();
    names2.sort();
    assert_eq!(names1, names2);
}

#[test]
fn test_hash_join_empty_result() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    // Insert users but NO posts
    for i in 0..5 {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(format!("user_{}", i).into()));
        data.insert(
            "email".to_string(),
            Value::String(format!("user_{}@example.com", i).into()),
        );
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Join should return empty
    let query_json = serde_json::json!({
        "find": ["?name", "?title"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 0);
}

#[test]
fn test_multi_clause_join() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    // Define a Comment type
    db.define_type(EntityTypeDef {
        unique_keys: vec![],
        name: "Comment".to_string(),
        fields: vec![
            FieldDef {
                name: "body".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "post".to_string(),
                field_type: FieldType::Ref("Post".to_string()),
                required: true,
                unique: false,
                indexed: true,
            },
            FieldDef {
                name: "commenter".to_string(),
                field_type: FieldType::Ref("User".to_string()),
                required: true,
                unique: false,
                indexed: true,
            },
        ],
    })
    .unwrap();

    // Insert a user
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    let alice_result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let alice_id = alice_result.entity_ids[0];

    // Insert a post
    let mut data = HashMap::new();
    data.insert("title".to_string(), Value::String("Hello".into()));
    data.insert("author".to_string(), Value::Ref(alice_id));
    let post_result = db
        .transact(vec![TxOp::Assert {
            entity_type: "Post".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let post_id = post_result.entity_ids[0];

    // Insert a comment
    let mut data = HashMap::new();
    data.insert("body".to_string(), Value::String("Great post!".into()));
    data.insert("post".to_string(), Value::Ref(post_id));
    data.insert("commenter".to_string(), Value::Ref(alice_id));
    db.transact(vec![TxOp::Assert {
        entity_type: "Comment".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // 3-way join: User → Post → Comment
    let query_json = serde_json::json!({
        "find": ["?name", "?title", "?body"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"},
            {"bind": "?c", "type": "Comment", "post": "?p", "body": "?body"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Alice".into()));
    assert_eq!(result.rows[0][1], Value::String("Hello".into()));
    assert_eq!(result.rows[0][2], Value::String("Great post!".into()));

    // Verify the plan builds a left-deep tree
    let plan = db.explain(&query).unwrap();
    let display = format!("{}", plan);
    assert!(display.contains("Project"), "plan: {}", display);
    let json = plan.to_json();
    // Should have 3 scans in the tree — count Scan nodes in JSON
    let json_str = serde_json::to_string(&json).unwrap();
    let scan_count = json_str.matches("\"node\":\"Scan\"").count();
    assert_eq!(scan_count, 3, "should have 3 scan nodes: {}", display);
}

#[test]
fn test_explain_single_clause() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [{"bind": "?u", "type": "User", "name": "?name"}]
    });
    let query = Query::from_json(&query_json).unwrap();
    let plan = db.explain(&query).unwrap();
    let json = plan.to_json();

    // Single clause → Project wrapping a Scan (no Join)
    assert_eq!(json["node"], "Project");
    assert_eq!(json["input"]["node"], "Scan");
    assert_eq!(json["input"]["type"], "User");
}

#[test]
fn test_explain_constant_uses_index_lookup() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert data so type count > 0
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    let query_json = serde_json::json!({
        "find": ["?age"],
        "where": [{"bind": "?u", "type": "User", "name": "Alice", "age": "?age"}]
    });
    let query = Query::from_json(&query_json).unwrap();
    let plan = db.explain(&query).unwrap();
    let json = plan.to_json();

    assert_eq!(json["input"]["node"], "Scan");
    let strategy = json["input"]["strategy"].as_str().unwrap();
    assert!(strategy.contains("IndexLookup"), "strategy should be IndexLookup, got: {}", strategy);
}

#[test]
fn test_explain_range_uses_range_scan() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert data so type count > 0
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".into()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".into()),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    let query_json = serde_json::json!({
        "find": ["?name"],
        "where": [{"bind": "?u", "type": "User", "name": "?name", "age": {"gt": 25}}]
    });
    let query = Query::from_json(&query_json).unwrap();
    let plan = db.explain(&query).unwrap();
    let json = plan.to_json();

    assert_eq!(json["input"]["node"], "Scan");
    let strategy = json["input"]["strategy"].as_str().unwrap();
    assert!(strategy.contains("RangeScan"), "strategy should be RangeScan, got: {}", strategy);
}

#[test]
fn test_explain_join_shows_join_vars() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    let query_json = serde_json::json!({
        "find": ["?name", "?title"],
        "where": [
            {"bind": "?u", "type": "User", "name": "?name"},
            {"bind": "?p", "type": "Post", "author": "?u", "title": "?title"}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let plan = db.explain(&query).unwrap();
    let json = plan.to_json();

    // The input should be a join node
    let input = &json["input"];
    let join_on = input["join_on"].as_array().unwrap();
    assert!(
        join_on.contains(&serde_json::json!("?u")),
        "join should be on ?u, got: {:?}",
        join_on
    );
}

// --- Storage durability options ---

mod durability {
    use super::*;
    use datalog_db::storage::{Durability, StorageOptions};

    fn open_at(path: &std::path::Path, durability: Durability) -> Arc<Database> {
        let opts = StorageOptions {
            durability,
            ..StorageOptions::default()
        };
        let storage = RocksDbStorage::open_with(path, opts).unwrap();
        Arc::new(Database::open(Arc::new(storage)).unwrap())
    }

    fn assert_round_trip(durability: Durability) {
        let dir = tempfile::tempdir().unwrap();

        // Write under the requested durability and close cleanly.
        {
            let db = open_at(dir.path(), durability);
            db.define_type(user_type()).unwrap();

            let mut data = HashMap::new();
            data.insert("name".to_string(), Value::String("Alice".into()));
            data.insert(
                "email".to_string(),
                Value::String("alice@example.com".into()),
            );
            db.transact(vec![TxOp::Assert {
                entity_type: "User".to_string(),
                entity: None,
                data,
            }])
            .unwrap();
        }

        // Reopen and verify the data is still there. A clean close should
        // make every durability mode equivalent — only the per-commit
        // WAL/fsync semantics differ on crash, which we can't simulate
        // portably.
        let db = open_at(dir.path(), durability);
        let query = Query::from_json(&serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?u", "type": "User", "name": "?name"}]
        }))
        .unwrap();
        let result = db.query(&query).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("Alice".into()));
    }

    #[test]
    fn buffered_round_trip() {
        assert_round_trip(Durability::Buffered);
    }

    #[test]
    fn sync_round_trip() {
        assert_round_trip(Durability::Sync);
    }

    #[test]
    fn memory_only_round_trip() {
        // Clean close flushes the memtable to SSTs, so even MemoryOnly
        // (WAL-off) data survives a graceful drop. This only checks the
        // option plumbs through; crash-loss behavior is documented, not
        // tested here.
        assert_round_trip(Durability::MemoryOnly);
    }

    #[test]
    fn default_is_buffered() {
        assert_eq!(Durability::default(), Durability::Buffered);
        assert_eq!(StorageOptions::default().durability, Durability::Buffered);
    }

    #[test]
    fn open_default_equivalent_to_buffered() {
        // RocksDbStorage::open should be a thin wrapper over
        // open_with(StorageOptions::default()). Verify both paths work.
        let dir = tempfile::tempdir().unwrap();
        let storage = RocksDbStorage::open(dir.path()).unwrap();
        let db = Arc::new(Database::open(Arc::new(storage)).unwrap());
        db.define_type(user_type()).unwrap();
        // If we got here without a panic, the default path works.
        let _ = db;
    }
}

// --- Storage tuning knobs (block cache, compression, bloom filter, memtable) ---

mod tuning {
    use super::*;
    use datalog_db::storage::{Compression, StorageOptions};

    fn open_with(opts: StorageOptions) -> (Arc<Database>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = RocksDbStorage::open_with(dir.path(), opts).unwrap();
        let db = Arc::new(Database::open(Arc::new(storage)).unwrap());
        (db, dir)
    }

    fn assert_round_trip(opts: StorageOptions) {
        let (db, _dir) = open_with(opts);
        db.define_type(user_type()).unwrap();

        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("Alice".into()));
        data.insert(
            "email".to_string(),
            Value::String("alice@example.com".into()),
        );
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();

        let query = Query::from_json(&serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?u", "type": "User", "name": "?name"}]
        }))
        .unwrap();
        let result = db.query(&query).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("Alice".into()));
    }

    #[test]
    fn compression_none() {
        assert_round_trip(StorageOptions {
            compression: Compression::None,
            ..StorageOptions::default()
        });
    }

    #[test]
    fn compression_snappy() {
        assert_round_trip(StorageOptions {
            compression: Compression::Snappy,
            ..StorageOptions::default()
        });
    }

    #[test]
    fn compression_lz4() {
        assert_round_trip(StorageOptions {
            compression: Compression::Lz4,
            ..StorageOptions::default()
        });
    }

    #[test]
    fn compression_zstd() {
        assert_round_trip(StorageOptions {
            compression: Compression::Zstd,
            ..StorageOptions::default()
        });
    }

    #[test]
    fn bloom_filter_disabled() {
        assert_round_trip(StorageOptions {
            bloom_filter_bits_per_key: None,
            ..StorageOptions::default()
        });
    }

    #[test]
    fn block_cache_disabled() {
        // 0 means "don't override RocksDB's built-in default" — should
        // still work, just with less RAM dedicated to the cache.
        assert_round_trip(StorageOptions {
            block_cache_bytes: 0,
            ..StorageOptions::default()
        });
    }

    #[test]
    fn small_write_buffer() {
        // Tiny memtable forces frequent SST flushes — exercises that
        // code path and verifies data survives across flushes.
        assert_round_trip(StorageOptions {
            write_buffer_size: 64 * 1024,
            ..StorageOptions::default()
        });
    }

    #[test]
    fn default_constants_are_sensible() {
        // Guardrails on the public default. These numbers can change;
        // the assertion is just that they're at least a sane order of
        // magnitude — catches accidental reverts to RocksDB's tiny 8 MB
        // block cache or to bloom filters being disabled.
        let opts = StorageOptions::default();
        assert!(opts.block_cache_bytes >= 16 * 1024 * 1024);
        assert!(opts.bloom_filter_bits_per_key.is_some());
        assert_eq!(opts.compression, Compression::Lz4);
    }
}

// --- Query cache policy ---

mod cache_policy {
    use super::*;
    use datalog_db::cache::CachePolicy;
    use datalog_db::db::DatabaseOptions;

    fn open(cache: CachePolicy) -> (Arc<Database>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let opts = DatabaseOptions {
            cache,
            ..DatabaseOptions::default()
        };
        let db = Database::open_with(storage, opts).unwrap();
        (Arc::new(db), dir)
    }

    fn insert_user(db: &Database, name: &str, email: &str) {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("email".to_string(), Value::String(email.into()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    fn run_user_query(db: &Database) -> usize {
        let query = Query::from_json(&serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?u", "type": "User", "name": "?name"}]
        }))
        .unwrap();
        db.query(&query).unwrap().rows.len()
    }

    #[test]
    fn default_policy_is_unbounded() {
        assert_eq!(CachePolicy::default(), CachePolicy::Unbounded);
    }

    #[test]
    fn unbounded_caches_after_query() {
        let (db, _dir) = open(CachePolicy::Unbounded);
        db.define_type(user_type()).unwrap();
        insert_user(&db, "Alice", "alice@example.com");

        // No cache hit yet.
        assert_eq!(run_user_query(&db), 1);
        // A query should have warmed the cache.
        // (We can't reach into Database to inspect the cache directly,
        // but the query result correctness is the contract — the
        // dedicated cache_policy_disabled test below proves the
        // uncached path still works.)
    }

    #[test]
    fn disabled_policy_still_returns_correct_results() {
        // The critical contract: queries must return the same answer
        // whether the cache is on or off. Otherwise disabling the cache
        // for memory reasons would silently change behavior.
        let (db, _dir) = open(CachePolicy::None);
        db.define_type(user_type()).unwrap();

        insert_user(&db, "Alice", "alice@example.com");
        insert_user(&db, "Bob", "bob@example.com");
        insert_user(&db, "Carol", "carol@example.com");

        assert_eq!(run_user_query(&db), 3);
        // Run it twice to exercise the "no cache to hit" path
        // repeatedly — the result must still be stable.
        assert_eq!(run_user_query(&db), 3);
    }

    #[test]
    fn bounded_zero_behaves_like_disabled() {
        let (db, _dir) = open(CachePolicy::Bounded { max_types: 0 });
        db.define_type(user_type()).unwrap();
        insert_user(&db, "Alice", "alice@example.com");
        assert_eq!(run_user_query(&db), 1);
    }

    #[test]
    fn bounded_evicts_lru_type() {
        // Use a 1-type cache so any new query evicts the previous type.
        let (db, _dir) = open(CachePolicy::Bounded { max_types: 1 });
        db.define_type(user_type()).unwrap();
        db.define_type(post_type()).unwrap();

        // Insert one user and one post so each type has data to load.
        insert_user(&db, "Alice", "alice@example.com");

        let mut post_data = HashMap::new();
        post_data.insert("title".to_string(), Value::String("Hello".into()));
        // post_type requires an author ref — find Alice's id.
        let users = db
            .query(
                &Query::from_json(&serde_json::json!({
                    "find": ["?u"],
                    "where": [{"bind": "?u", "type": "User"}]
                }))
                .unwrap(),
            )
            .unwrap();
        let alice_id = match &users.rows[0][0] {
            Value::Ref(id) => *id,
            _ => panic!("expected ref"),
        };
        post_data.insert("author".to_string(), Value::Ref(alice_id));
        db.transact(vec![TxOp::Assert {
            entity_type: "Post".to_string(),
            entity: None,
            data: post_data,
        }])
        .unwrap();

        // Query User → cache holds {User}.
        assert_eq!(run_user_query(&db), 1);

        // Query Post → cache should evict User to make room for Post.
        let post_query = Query::from_json(&serde_json::json!({
            "find": ["?title"],
            "where": [{"bind": "?p", "type": "Post", "title": "?title"}]
        }))
        .unwrap();
        assert_eq!(db.query(&post_query).unwrap().rows.len(), 1);

        // Query User again → must still return correct data. With
        // max_types=1 the User cache was evicted, so this exercises the
        // reload path.
        assert_eq!(run_user_query(&db), 1);
    }

    #[test]
    fn bounded_keeps_recently_used() {
        // max_types=2, query A then B then A then C → C evicts B (LRU).
        // Hard to assert directly without cache inspection, but we can
        // at least verify all queries return correct results regardless
        // of which one happens to be a hit vs a miss.
        let (db, _dir) = open(CachePolicy::Bounded { max_types: 2 });
        db.define_type(user_type()).unwrap();
        db.define_type(post_type()).unwrap();

        insert_user(&db, "Alice", "alice@example.com");
        for _ in 0..5 {
            assert_eq!(run_user_query(&db), 1);
        }
    }
}

// --- Group commit (transact_many) ---

mod group_commit {
    use super::*;

    fn assert_user(name: &str, email: &str) -> TxOp {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("email".to_string(), Value::String(email.into()));
        TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }
    }

    fn count_users(db: &Database) -> usize {
        let query = Query::from_json(&serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?u", "type": "User", "name": "?name"}]
        }))
        .unwrap();
        db.query(&query).unwrap().rows.len()
    }

    #[test]
    fn empty_group_returns_empty_vec() {
        let (db, _dir) = test_db();
        let results = db.transact_many(vec![]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn all_success_group_commits_all() {
        let (db, _dir) = test_db();
        db.define_type(user_type()).unwrap();

        let results = db
            .transact_many(vec![
                vec![assert_user("Alice", "alice@example.com")],
                vec![assert_user("Bob", "bob@example.com")],
                vec![assert_user("Carol", "carol@example.com")],
            ])
            .unwrap();

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "expected success, got {:?}", r);
        }
        assert_eq!(count_users(&db), 3);
    }

    #[test]
    fn tx_ids_are_strictly_increasing_in_group() {
        let (db, _dir) = test_db();
        db.define_type(user_type()).unwrap();

        let results = db
            .transact_many(vec![
                vec![assert_user("A", "a@x.com")],
                vec![assert_user("B", "b@x.com")],
                vec![assert_user("C", "c@x.com")],
            ])
            .unwrap();

        let tx_ids: Vec<u64> = results
            .iter()
            .map(|r| r.as_ref().unwrap().tx_id)
            .collect();
        assert_eq!(tx_ids.len(), 3);
        assert!(
            tx_ids[0] < tx_ids[1] && tx_ids[1] < tx_ids[2],
            "tx_ids should be strictly increasing in group order, got {:?}",
            tx_ids
        );
    }

    #[test]
    fn failing_tx_does_not_affect_siblings() {
        let (db, _dir) = test_db();
        db.define_type(user_type()).unwrap();

        // Middle tx violates the schema (unknown field) — it should
        // fail, but the first and third must still commit. Two users
        // must be visible.
        let mut bad_data = HashMap::new();
        bad_data.insert(
            "this_field_does_not_exist".to_string(),
            Value::String("oops".into()),
        );

        let results = db
            .transact_many(vec![
                vec![assert_user("Alice", "alice@example.com")],
                vec![TxOp::Assert {
                    entity_type: "User".to_string(),
                    entity: None,
                    data: bad_data,
                }],
                vec![assert_user("Carol", "carol@example.com")],
            ])
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err(), "middle tx should have failed");
        assert!(results[2].is_ok());
        assert_eq!(count_users(&db), 2);
    }

    #[test]
    fn unique_constraint_visible_across_group() {
        // Two transactions in the same group both try to insert the
        // same unique email. The second must see the first's pending
        // write via the overlay and fail with a UniqueViolation.
        let (db, _dir) = test_db();
        db.define_type(user_type()).unwrap();

        let results = db
            .transact_many(vec![
                vec![assert_user("Alice", "shared@example.com")],
                vec![assert_user("Bob", "shared@example.com")],
            ])
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok(), "first tx should succeed");
        assert!(
            results[1].is_err(),
            "second tx should fail on unique constraint, got {:?}",
            results[1]
        );
        // Exactly one user persisted.
        assert_eq!(count_users(&db), 1);
    }

    #[test]
    fn as_of_includes_first_committed_tx_in_group() {
        // Group commits txs A, B, C. asOf(A.tx_id) should return only
        // tx A's data, even though B and C committed atomically in the
        // same batch. Verifies tx_id ordering is preserved.
        let (db, _dir) = test_db();
        db.define_type(user_type()).unwrap();

        let results = db
            .transact_many(vec![
                vec![assert_user("A", "a@x.com")],
                vec![assert_user("B", "b@x.com")],
                vec![assert_user("C", "c@x.com")],
            ])
            .unwrap();

        let tx_a = results[0].as_ref().unwrap().tx_id;

        let query = Query::from_json(&serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?u", "type": "User", "name": "?name"}],
            "as_of": tx_a,
        }))
        .unwrap();
        let result = db.query(&query).unwrap();
        assert_eq!(
            result.rows.len(),
            1,
            "asOf the first tx should return only that tx's data, got {} rows",
            result.rows.len()
        );
    }

    #[test]
    fn group_durably_persists_across_reopen() {
        // The whole point of group commit is one atomic WriteBatch.
        // Verify that after a clean drop + reopen, all members of a
        // group are present together.
        let dir = tempfile::tempdir().unwrap();
        {
            let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
            let db = Database::open(storage).unwrap();
            db.define_type(user_type()).unwrap();
            let results = db
                .transact_many(vec![
                    vec![assert_user("A", "a@x.com")],
                    vec![assert_user("B", "b@x.com")],
                    vec![assert_user("C", "c@x.com")],
                ])
                .unwrap();
            for r in &results {
                assert!(r.is_ok());
            }
        }
        // Reopen and check.
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();
        assert_eq!(count_users(&db), 3);
    }
}

// --- Background writer thread (auto-batching group commit) ---

mod writer_thread {
    use super::*;
    use datalog_db::db::{DatabaseOptions, GroupCommitConfig};
    use std::sync::Arc as StdArc;
    use std::thread;
    use std::time::Duration;

    fn open_with_writer(config: GroupCommitConfig) -> (StdArc<Database>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let opts = DatabaseOptions {
            group_commit: Some(config),
            ..DatabaseOptions::default()
        };
        let db = Database::open_with(storage, opts).unwrap();
        (StdArc::new(db), dir)
    }

    fn count_users(db: &Database) -> usize {
        let query = Query::from_json(&serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?u", "type": "User", "name": "?name"}]
        }))
        .unwrap();
        db.query(&query).unwrap().rows.len()
    }

    fn assert_user(name: &str, email: &str) -> TxOp {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("email".to_string(), Value::String(email.into()));
        TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }
    }

    #[test]
    fn single_transact_with_writer_works() {
        // 1-element batch path through the writer must produce the same
        // result as the sync path.
        let (db, _dir) = open_with_writer(GroupCommitConfig::default());
        db.define_type(user_type()).unwrap();

        let result = db
            .transact(vec![assert_user("Alice", "alice@example.com")])
            .unwrap();
        assert!(result.tx_id > 0);
        assert_eq!(count_users(&db), 1);
    }

    #[test]
    fn concurrent_transacts_all_succeed() {
        // Spawn many threads each issuing one transact(). All must
        // succeed. With a large window the writer should coalesce a
        // bunch of them into one group commit.
        let (db, _dir) = open_with_writer(GroupCommitConfig {
            max_batch_size: 32,
            max_window: Duration::from_millis(20),
        });
        db.define_type(user_type()).unwrap();

        let n = 50;
        let mut handles = Vec::new();
        for i in 0..n {
            let db = db.clone();
            handles.push(thread::spawn(move || {
                db.transact(vec![assert_user(
                    &format!("user{}", i),
                    &format!("user{}@example.com", i),
                )])
            }));
        }

        let mut tx_ids = Vec::new();
        for h in handles {
            let r = h.join().unwrap().unwrap();
            tx_ids.push(r.tx_id);
        }
        assert_eq!(tx_ids.len(), n);
        // All tx_ids must be unique.
        tx_ids.sort();
        tx_ids.dedup();
        assert_eq!(tx_ids.len(), n);
        assert_eq!(count_users(&db), n);
    }

    #[test]
    fn concurrent_unique_constraint_enforced() {
        // Many threads race to insert the same unique email. With the
        // writer auto-batching, only one must succeed; the rest get
        // UniqueViolation. Critical correctness property: the overlay's
        // read-your-prior-writes makes this safe even when all threads
        // submit "simultaneously".
        let (db, _dir) = open_with_writer(GroupCommitConfig {
            max_batch_size: 32,
            max_window: Duration::from_millis(20),
        });
        db.define_type(user_type()).unwrap();

        let n = 20;
        let mut handles = Vec::new();
        for i in 0..n {
            let db = db.clone();
            handles.push(thread::spawn(move || {
                db.transact(vec![assert_user(
                    &format!("racer{}", i),
                    "shared@example.com",
                )])
            }));
        }

        let mut ok_count = 0;
        let mut err_count = 0;
        for h in handles {
            match h.join().unwrap() {
                Ok(_) => ok_count += 1,
                Err(_) => err_count += 1,
            }
        }
        assert_eq!(ok_count, 1, "exactly one tx should win the unique race");
        assert_eq!(err_count, n - 1);
        assert_eq!(count_users(&db), 1);
    }

    #[test]
    fn writer_shutdown_on_drop() {
        // After dropping the Database, the writer thread must exit
        // cleanly. We can't directly observe thread exit, but if the
        // join hangs, this test will hang too — which is the assertion.
        let (db, dir) = open_with_writer(GroupCommitConfig::default());
        db.define_type(user_type()).unwrap();
        db.transact(vec![assert_user("Alice", "a@x.com")]).unwrap();
        drop(db);
        // If we got here, Drop completed (writer thread joined).
        // Reopen to verify the data is still there.
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();
        let query = Query::from_json(&serde_json::json!({
            "find": ["?n"],
            "where": [{"bind": "?u", "type": "User", "name": "?n"}]
        }))
        .unwrap();
        assert_eq!(db.query(&query).unwrap().rows.len(), 1);
    }
}

// --- Attribute interning ---

mod attribute_interning {
    use super::*;
    use datalog_db::intern::AttrInterner;

    /// Touch a typical schema and a few asserts so the attribute table
    /// has interesting content to inspect.
    fn populate_db(db: &Database) {
        db.define_type(user_type()).unwrap();
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("Alice".into()));
        data.insert(
            "email".to_string(),
            Value::String("alice@example.com".into()),
        );
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    #[test]
    fn attribute_ids_are_persisted_and_reused_across_reopens() {
        // Open, populate, drop. Then reopen and assert the interner
        // loaded the same IDs from storage.
        let dir = tempfile::tempdir().unwrap();
        let attrs_first_open: Vec<(String, u32)>;
        {
            let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
            let db = Database::open(storage.clone()).unwrap();
            populate_db(&db);

            // Inspect via a fresh interner loaded from the same storage —
            // proves what's actually persisted, not just what's in
            // memory.
            let inspector = AttrInterner::new();
            inspector.load_from_storage(&*storage).unwrap();
            // The populate inserts a User with `name` and `email` set
            // (age is optional, left unset). So we expect at least
            // __schema_type, __type, User/name, User/email — four
            // distinct attributes. age is not interned because no datom
            // uses it.
            assert!(
                inspector.len() >= 4,
                "expected at least 4 interned attrs, got {}",
                inspector.len()
            );

            attrs_first_open = ["__schema_type", "__type", "User/name", "User/email"]
                .iter()
                .filter_map(|n| inspector.lookup(n).map(|id| (n.to_string(), id)))
                .collect();
            assert_eq!(attrs_first_open.len(), 4);
        }

        // Reopen with a fresh storage handle.
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let inspector = AttrInterner::new();
        inspector.load_from_storage(&*storage).unwrap();

        for (name, expected_id) in &attrs_first_open {
            assert_eq!(
                inspector.lookup(name),
                Some(*expected_id),
                "attr id for `{}` did not survive reopen",
                name
            );
        }
    }

    #[test]
    fn writes_do_not_allocate_duplicate_ids() {
        // Multiple transacts touching the same attributes must reuse
        // the existing IDs. The interner length must equal the number
        // of distinct attributes, not the number of writes.
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage.clone()).unwrap();
        populate_db(&db);
        let after_first: usize;
        {
            let inspector = AttrInterner::new();
            inspector.load_from_storage(&*storage).unwrap();
            after_first = inspector.len();
        }

        // Five more inserts on the same type — all reuse User/name,
        // User/email, User/age, __type. No new attributes.
        for i in 0..5 {
            let mut data = HashMap::new();
            data.insert("name".to_string(), Value::String(format!("user{}", i).into()));
            data.insert(
                "email".to_string(),
                Value::String(format!("u{}@x.com", i).into()),
            );
            db.transact(vec![TxOp::Assert {
                entity_type: "User".to_string(),
                entity: None,
                data,
            }])
            .unwrap();
        }

        let inspector = AttrInterner::new();
        inspector.load_from_storage(&*storage).unwrap();
        assert_eq!(
            inspector.len(),
            after_first,
            "no new attribute IDs should have been allocated"
        );
    }

    #[test]
    fn unique_attribute_names_get_distinct_ids() {
        // Sanity check that the forward + reverse maps are coherent —
        // every name maps to a unique id and vice versa.
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage.clone()).unwrap();
        populate_db(&db);

        let inspector = AttrInterner::new();
        inspector.load_from_storage(&*storage).unwrap();

        // Collect (name, id) for the well-known attrs.
        let attrs = ["__schema_type", "__type", "User/name", "User/email"];
        let mut seen_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for name in attrs {
            let id = inspector.lookup(name).expect("attr not interned");
            assert!(
                seen_ids.insert(id),
                "attr id {} was assigned to multiple names",
                id
            );
            // Reverse lookup matches.
            assert_eq!(inspector.name_of(id).as_deref(), Some(name));
        }
    }
}

// ---------------------------------------------------------------------------
// Backup / checkpoint tests
// ---------------------------------------------------------------------------

mod backups {
    use super::*;
    use datalog_db::backup::{
        create_checkpoint, list_checkpoints, prune_checkpoints, spawn_backup_scheduler,
        BackupSchedulerConfig,
    };
    use std::time::Duration;

    fn alice_assert() -> TxOp {
        let mut data = HashMap::new();
        data.insert("name".into(), Value::String("Alice".into()));
        data.insert("email".into(), Value::String("alice@x".into()));
        TxOp::Assert {
            entity_type: "User".into(),
            entity: None,
            data,
        }
    }

    fn bob_assert() -> TxOp {
        let mut data = HashMap::new();
        data.insert("name".into(), Value::String("Bob".into()));
        data.insert("email".into(), Value::String("bob@x".into()));
        TxOp::Assert {
            entity_type: "User".into(),
            entity: None,
            data,
        }
    }

    fn count_users(db: &Database) -> usize {
        let q = Query::from_json(&serde_json::json!({
            "find": ["?u", "?name"],
            "where": [{"bind": "?u", "type": "User", "name": "?name"}],
        }))
        .unwrap();
        db.query(&q).unwrap().rows.len()
    }

    fn names(db: &Database) -> Vec<String> {
        let q = Query::from_json(&serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?u", "type": "User", "name": "?name"}],
        }))
        .unwrap();
        let result = db.query(&q).unwrap();
        let mut out: Vec<String> = result
            .rows
            .iter()
            .filter_map(|r| match &r[0] {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn checkpoint_captures_point_in_time() {
        let parent = tempfile::tempdir().unwrap();
        let data_dir = parent.path().join("data");
        let backup_dir = parent.path().join("backups");

        // Phase 1 — write Alice, checkpoint, write Bob, then drop the live DB.
        let checkpoint_path = {
            let storage = Arc::new(RocksDbStorage::open(&data_dir).unwrap());
            let db = Arc::new(Database::open(storage).unwrap());
            db.define_type(user_type()).unwrap();
            db.transact(vec![alice_assert()]).unwrap();

            let path = create_checkpoint(&db, &backup_dir).unwrap();
            assert!(path.starts_with(&backup_dir));
            assert!(path.exists());

            db.transact(vec![bob_assert()]).unwrap();
            assert_eq!(count_users(&db), 2);

            path
        };

        // Phase 2 — open the checkpoint as a fresh DB. Schema and the
        // pre-checkpoint data must be there; post-checkpoint data must not.
        let storage = Arc::new(RocksDbStorage::open(&checkpoint_path).unwrap());
        let restored = Database::open(storage).unwrap();
        assert_eq!(count_users(&restored), 1);
        assert_eq!(names(&restored), vec!["Alice".to_string()]);
    }

    #[test]
    fn list_and_prune_oldest_first() {
        let parent = tempfile::tempdir().unwrap();
        let data_dir = parent.path().join("data");
        let backup_dir = parent.path().join("backups");

        let storage = Arc::new(RocksDbStorage::open(&data_dir).unwrap());
        let db = Arc::new(Database::open(storage).unwrap());
        db.define_type(user_type()).unwrap();

        // Take three checkpoints with a small sleep so the millisecond-
        // resolution names differ. The collision-suffix branch would
        // make the test pass even without the sleep, but we want to
        // exercise the timestamp-sorted path that production uses.
        let mut taken = Vec::new();
        for _ in 0..3 {
            taken.push(create_checkpoint(&db, &backup_dir).unwrap());
            std::thread::sleep(Duration::from_millis(5));
        }

        let listed = list_checkpoints(&backup_dir).unwrap();
        assert_eq!(listed.len(), 3);
        // Listing is oldest-first.
        assert_eq!(listed, taken);

        // Retain 1 → two oldest removed.
        let removed = prune_checkpoints(&backup_dir, 1).unwrap();
        assert_eq!(removed, 2);
        let remaining = list_checkpoints(&backup_dir).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0], taken[2]);
    }

    #[test]
    fn scheduler_takes_periodic_checkpoints() {
        let parent = tempfile::tempdir().unwrap();
        let data_dir = parent.path().join("data");
        let backup_dir = parent.path().join("backups");

        let storage = Arc::new(RocksDbStorage::open(&data_dir).unwrap());
        let db = Arc::new(Database::open(storage).unwrap());
        db.define_type(user_type()).unwrap();
        db.transact(vec![alice_assert()]).unwrap();

        let handle = spawn_backup_scheduler(
            db.clone(),
            BackupSchedulerConfig {
                root: backup_dir.clone(),
                interval: Duration::from_millis(80),
                retain: 2,
            },
        );

        // Give the scheduler enough wall-time to produce >2 checkpoints so
        // pruning has work to do.
        std::thread::sleep(Duration::from_millis(500));
        handle.stop();

        let listed = list_checkpoints(&backup_dir).unwrap();
        assert!(
            !listed.is_empty(),
            "expected at least one scheduled checkpoint, got none"
        );
        assert!(
            listed.len() <= 2,
            "retain=2 violated, got {} checkpoints",
            listed.len()
        );

        // Each retained checkpoint should be openable and contain Alice.
        for path in &listed {
            let storage = Arc::new(RocksDbStorage::open(path).unwrap());
            let restored = Database::open(storage).unwrap();
            assert_eq!(names(&restored), vec!["Alice".to_string()]);
        }
    }
}

// ---------------------------------------------------------------------------
// Regression tests for query correctness: plan-cache value sensitivity,
// numeric type coercion, and bind+filter patterns.
// ---------------------------------------------------------------------------

/// A type with both an f64 and an i64 field for numeric-coercion tests.
fn product_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "Product".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: true,
            },
            FieldDef {
                name: "price".to_string(),
                field_type: FieldType::F64,
                required: false,
                unique: false,
                indexed: false,
            },
            FieldDef {
                name: "qty".to_string(),
                field_type: FieldType::I64,
                required: false,
                unique: false,
                indexed: false,
            },
        ],
    }
}

fn seed_products(db: &Database) {
    // Widget: price 9.99, qty 100 | Gadget: price 19.5, qty 5
    for (name, price, qty) in [("Widget", 9.99_f64, 100_i64), ("Gadget", 19.5, 5)] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("price".to_string(), Value::F64(price));
        data.insert("qty".to_string(), Value::I64(qty));
        db.transact(vec![TxOp::Assert {
            entity_type: "Product".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }
}

fn names_of(result: &datalog_db::query::executor::QueryResult, col: usize) -> Vec<String> {
    let mut out: Vec<String> = result
        .rows
        .iter()
        .map(|r| match &r[col] {
            Value::String(s) => s.to_string(),
            other => format!("{:?}", other),
        })
        .collect();
    out.sort();
    out
}

fn run_query(db: &Database, json: serde_json::Value) -> datalog_db::query::executor::QueryResult {
    let query = Query::from_json(&json).unwrap();
    db.query(&query).unwrap()
}

/// Two queries with the same shape but different predicate literals must
/// return different results. Regression for the plan-cache key erasing
/// literal values while the cached plan baked them in.
#[test]
fn test_plan_cache_is_value_sensitive() {
    let (db, _dir) = test_db();
    db.define_type(product_type()).unwrap();
    seed_products(&db);

    let q = |bound: i64| {
        serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?p", "type": "Product", "name": "?name", "qty": {"gt": bound}}]
        })
    };

    // Run the small bound first to populate the plan cache.
    assert_eq!(names_of(&run_query(&db, q(10)), 0), vec!["Widget"]);
    // Same shape, larger bound — must NOT reuse the qty>10 result.
    assert_eq!(names_of(&run_query(&db, q(50)), 0), vec!["Widget"]);
    assert_eq!(names_of(&run_query(&db, q(200)), 0), Vec::<String>::new());
    assert_eq!(names_of(&run_query(&db, q(3)), 0), vec!["Gadget", "Widget"]);
}

/// An f64 field compared against an integer literal must coerce and match,
/// rather than silently returning nothing.
#[test]
fn test_numeric_coercion_f64_field_int_literal() {
    let (db, _dir) = test_db();
    db.define_type(product_type()).unwrap();
    seed_products(&db);

    let q = |op: &str, v: i64| {
        serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?p", "type": "Product", "name": "?name", "price": {op: v}}]
        })
    };
    assert_eq!(names_of(&run_query(&db, q("gt", 10)), 0), vec!["Gadget"]);
    assert_eq!(names_of(&run_query(&db, q("lt", 10)), 0), vec!["Widget"]);
    assert_eq!(names_of(&run_query(&db, q("gte", 19)), 0), vec!["Gadget"]);
}

/// An i64 field compared against a fractional float literal must use the
/// operator-preserving integer bound (e.g. `qty > 10.5` ⇔ `qty > 10`).
#[test]
fn test_numeric_coercion_i64_field_float_literal() {
    let (db, _dir) = test_db();
    db.define_type(product_type()).unwrap();
    seed_products(&db);

    let q = |op: &str, v: f64| {
        serde_json::json!({
            "find": ["?name"],
            "where": [{"bind": "?p", "type": "Product", "name": "?name", "qty": {op: v}}]
        })
    };
    assert_eq!(names_of(&run_query(&db, q("gt", 10.5)), 0), vec!["Widget"]);
    assert_eq!(names_of(&run_query(&db, q("lt", 5.5)), 0), vec!["Gadget"]);
    assert_eq!(names_of(&run_query(&db, q("lte", 5.0)), 0), vec!["Gadget"]);
    // qty == 100.0 should match the integer 100.
    let eq = serde_json::json!({
        "find": ["?name"],
        "where": [{"bind": "?p", "type": "Product", "name": "?name", "qty": 100.0}]
    });
    assert_eq!(names_of(&run_query(&db, eq), 0), vec!["Widget"]);
}

/// `{"var": "?x", "gt": n}` must both bind ?x and filter by the predicate.
#[test]
fn test_bound_predicate_binds_and_filters() {
    let (db, _dir) = test_db();
    db.define_type(product_type()).unwrap();
    seed_products(&db);

    let result = run_query(
        &db,
        serde_json::json!({
            "find": ["?name", "?qty"],
            "where": [{
                "bind": "?p", "type": "Product",
                "name": "?name", "qty": {"var": "?qty", "gt": 10}
            }]
        }),
    );
    assert_eq!(result.columns, vec!["?name", "?qty"]);
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Widget".into()));
    assert_eq!(result.rows[0][1], Value::I64(100));
}

/// Extra products for ordering tests, including a price tie (Anvil & Zeta).
fn seed_products_for_ordering(db: &Database) {
    for (name, price, qty) in [
        ("Widget", 9.99_f64, 100_i64),
        ("Gadget", 19.5, 5),
        ("Anvil", 5.0, 50),
        ("Zeta", 5.0, 7),
        ("Bolt", 0.25, 1000),
    ] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("price".to_string(), Value::F64(price));
        data.insert("qty".to_string(), Value::I64(qty));
        db.transact(vec![TxOp::Assert {
            entity_type: "Product".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }
}

fn ordered_names(result: &datalog_db::query::executor::QueryResult) -> Vec<String> {
    // Keep result order (do NOT sort) so we verify the query's ordering.
    result
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.to_string(),
            other => format!("{:?}", other),
        })
        .collect()
}

#[test]
fn test_order_by_ascending_and_descending() {
    let (db, _dir) = test_db();
    db.define_type(product_type()).unwrap();
    seed_products_for_ordering(&db);

    let asc = run_query(
        &db,
        serde_json::json!({
            "find": ["?name", "?price"],
            "where": [{"bind": "?p", "type": "Product", "name": "?name", "price": "?price"}],
            "order_by": ["?price"]
        }),
    );
    // Anvil/Zeta tie at 5.0 — stable sort keeps insertion order (Anvil first).
    assert_eq!(
        ordered_names(&asc),
        ["Bolt", "Anvil", "Zeta", "Widget", "Gadget"]
            .map(String::from)
            .to_vec()
    );

    let desc = run_query(
        &db,
        serde_json::json!({
            "find": ["?name", "?price"],
            "where": [{"bind": "?p", "type": "Product", "name": "?name", "price": "?price"}],
            "order_by": [{"var": "?price", "desc": true}]
        }),
    );
    assert_eq!(
        ordered_names(&desc),
        ["Gadget", "Widget", "Anvil", "Zeta", "Bolt"]
            .map(String::from)
            .to_vec()
    );
}

#[test]
fn test_order_by_multi_key() {
    let (db, _dir) = test_db();
    db.define_type(product_type()).unwrap();
    seed_products_for_ordering(&db);

    // price asc, then name desc — breaks the 5.0 tie as Zeta before Anvil.
    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["?name", "?price"],
            "where": [{"bind": "?p", "type": "Product", "name": "?name", "price": "?price"}],
            "order_by": ["?price", {"var": "?name", "desc": true}]
        }),
    );
    assert_eq!(
        ordered_names(&r),
        ["Bolt", "Zeta", "Anvil", "Widget", "Gadget"]
            .map(String::from)
            .to_vec()
    );
}

#[test]
fn test_limit_and_offset() {
    let (db, _dir) = test_db();
    db.define_type(product_type()).unwrap();
    seed_products_for_ordering(&db);

    let base = serde_json::json!({
        "find": ["?name", "?qty"],
        "where": [{"bind": "?p", "type": "Product", "name": "?name", "qty": "?qty"}],
        "order_by": [{"var": "?qty", "desc": true}]
    });

    let mut limited = base.clone();
    limited["limit"] = serde_json::json!(2);
    assert_eq!(
        ordered_names(&run_query(&db, limited)),
        ["Bolt", "Widget"].map(String::from).to_vec()
    );

    let mut paged = base.clone();
    paged["limit"] = serde_json::json!(2);
    paged["offset"] = serde_json::json!(1);
    assert_eq!(
        ordered_names(&run_query(&db, paged)),
        ["Widget", "Anvil"].map(String::from).to_vec()
    );

    // limit 0 yields nothing.
    let mut zero = base.clone();
    zero["limit"] = serde_json::json!(0);
    assert!(run_query(&db, zero).rows.is_empty());
}

fn employee_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "Employee".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: true,
            },
            FieldDef {
                name: "dept".to_string(),
                field_type: FieldType::String,
                required: false,
                unique: false,
                indexed: true,
            },
            FieldDef {
                name: "salary".to_string(),
                field_type: FieldType::I64,
                required: false,
                unique: false,
                indexed: false,
            },
        ],
    }
}

fn seed_employees(db: &Database) {
    for (name, dept, salary) in [
        ("Alice", "eng", 100_i64),
        ("Bob", "eng", 120),
        ("Carol", "sales", 90),
        ("Dave", "sales", 95),
        ("Eve", "sales", 80),
    ] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("dept".to_string(), Value::String(dept.into()));
        data.insert("salary".to_string(), Value::I64(salary));
        db.transact(vec![TxOp::Assert {
            entity_type: "Employee".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }
}

fn project_type() -> EntityTypeDef {
    EntityTypeDef {
        unique_keys: vec![],
        name: "Project".to_string(),
        fields: vec![
            FieldDef {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                unique: false,
                indexed: true,
            },
            FieldDef {
                name: "lead".to_string(),
                field_type: FieldType::Ref("Employee".to_string()),
                required: false,
                unique: false,
                indexed: true,
            },
        ],
    }
}

/// Seed employees and return a name -> entity id map.
fn seed_employees_with_ids(db: &Database) -> HashMap<String, u64> {
    let mut ids = HashMap::new();
    for (name, dept, salary) in [
        ("Alice", "eng", 100_i64),
        ("Bob", "eng", 120),
        ("Carol", "sales", 90),
        ("Dave", "sales", 95),
        ("Eve", "sales", 80),
    ] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.into()));
        data.insert("dept".to_string(), Value::String(dept.into()));
        data.insert("salary".to_string(), Value::I64(salary));
        let r = db
            .transact(vec![TxOp::Assert {
                entity_type: "Employee".to_string(),
                entity: None,
                data,
            }])
            .unwrap();
        ids.insert(name.to_string(), r.entity_ids[0]);
    }
    ids
}

#[test]
fn test_or_disjunction() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    seed_employees(&db);

    // dept == eng OR salary < 85  => Alice, Bob (eng) + Eve (80)
    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["?name"],
            "where": [
                {"bind": "?e", "type": "Employee", "name": "?name"},
                {"or": [
                    {"bind": "?e", "type": "Employee", "dept": "eng"},
                    {"bind": "?e", "type": "Employee", "salary": {"lt": 85}}
                ]}
            ]
        }),
    );
    assert_eq!(names_of(&r, 0), vec!["Alice", "Bob", "Eve"]);
}

#[test]
fn test_not_anti_join() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    db.define_type(project_type()).unwrap();
    let ids = seed_employees_with_ids(&db);

    // Alice and Carol lead projects.
    for (pname, lead) in [("Apollo", "Alice"), ("Zephyr", "Carol")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(pname.into()));
        data.insert("lead".to_string(), Value::Ref(ids[lead]));
        db.transact(vec![TxOp::Assert {
            entity_type: "Project".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Employees leading no project => Bob, Dave, Eve.
    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["?name"],
            "where": [
                {"bind": "?e", "type": "Employee", "name": "?name"},
                {"not": {"bind": "?p", "type": "Project", "lead": "?e"}}
            ]
        }),
    );
    assert_eq!(names_of(&r, 0), vec!["Bob", "Dave", "Eve"]);
}

#[test]
fn test_or_and_not_combined() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    db.define_type(project_type()).unwrap();
    let ids = seed_employees_with_ids(&db);
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Apollo".into()));
    data.insert("lead".to_string(), Value::Ref(ids["Alice"]));
    db.transact(vec![TxOp::Assert {
        entity_type: "Project".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // (eng OR sales) AND not-leading => everyone except Alice (who leads).
    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["?name"],
            "where": [
                {"bind": "?e", "type": "Employee", "name": "?name"},
                {"or": [
                    {"bind": "?e", "type": "Employee", "dept": "eng"},
                    {"bind": "?e", "type": "Employee", "dept": "sales"}
                ]},
                {"not": {"bind": "?p", "type": "Project", "lead": "?e"}}
            ]
        }),
    );
    assert_eq!(names_of(&r, 0), vec!["Bob", "Carol", "Dave", "Eve"]);
}

#[test]
fn test_redefine_type_compatibility() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();

    // Additive: re-state existing fields unchanged, add a new one -> OK.
    let mut additive = employee_type();
    additive.fields.push(FieldDef {
        name: "title".to_string(),
        field_type: FieldType::String,
        required: false,
        unique: false,
        indexed: false,
    });
    assert!(db.define_type(additive.clone()).is_ok());

    // Idempotent: redefining identically -> OK.
    assert!(db.define_type(additive.clone()).is_ok());

    // Dropping a field -> rejected.
    let dropped = EntityTypeDef {
        unique_keys: vec![],
        name: "Employee".to_string(),
        fields: vec![FieldDef {
            name: "name".to_string(),
            field_type: FieldType::String,
            required: true,
            unique: false,
            indexed: true,
        }],
    };
    let err = db.define_type(dropped).unwrap_err().to_string();
    assert!(err.contains("missing from the new definition"), "got: {err}");

    // Changing a field's type -> rejected.
    let mut retyped = additive.clone();
    retyped
        .fields
        .iter_mut()
        .find(|f| f.name == "salary")
        .unwrap()
        .field_type = FieldType::String;
    let err = db.define_type(retyped).unwrap_err().to_string();
    assert!(err.contains("changed definition"), "got: {err}");

    // Changing a modifier (indexed) -> rejected.
    let mut remod = additive.clone();
    remod
        .fields
        .iter_mut()
        .find(|f| f.name == "dept")
        .unwrap()
        .indexed = false;
    let err = db.define_type(remod).unwrap_err().to_string();
    assert!(err.contains("changed definition"), "got: {err}");
}

#[test]
fn test_filter_by_constant_entity_ref() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    db.define_type(project_type()).unwrap();
    let ids = seed_employees_with_ids(&db);
    for (pname, lead) in [("Apollo", "Alice"), ("Zephyr", "Carol")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(pname.into()));
        data.insert("lead".to_string(), Value::Ref(ids[lead]));
        db.transact(vec![TxOp::Assert {
            entity_type: "Project".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // Exact ref constant: projects led by Alice -> Apollo.
    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["?n"],
            "where": [{"bind": "?p", "type": "Project", "name": "?n", "lead": {"ref": ids["Alice"]}}]
        }),
    );
    assert_eq!(names_of(&r, 0), vec!["Apollo"]);

    // != ref constant: everything except Alice's projects -> Zephyr.
    let r2 = run_query(
        &db,
        serde_json::json!({
            "find": ["?n"],
            "where": [{"bind": "?p", "type": "Project", "name": "?n", "lead": {"ne": {"ref": ids["Alice"]}}}]
        }),
    );
    assert_eq!(names_of(&r2, 0), vec!["Zephyr"]);
}

#[test]
fn test_double_negation_cancels() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    db.define_type(project_type()).unwrap();
    let ids = seed_employees_with_ids(&db);
    for (pname, lead) in [("Apollo", "Alice"), ("Zephyr", "Carol")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(pname.into()));
        data.insert("lead".to_string(), Value::Ref(ids[lead]));
        db.transact(vec![TxOp::Assert {
            entity_type: "Project".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    }

    // not { not { leads a project } } == leads a project => Alice, Carol.
    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["?name"],
            "where": [
                {"bind": "?e", "type": "Employee", "name": "?name"},
                {"not": {"not": {"bind": "?p", "type": "Project", "lead": "?e"}}}
            ]
        }),
    );
    assert_eq!(names_of(&r, 0), vec!["Alice", "Carol"]);
}

#[test]
fn test_unbound_find_variable_is_null() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    seed_employees(&db);

    // ?missing never appears in the where clause -> projects as Null, not a
    // sentinel string that would collide with real data.
    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["?name", "?missing"],
            "where": [{"bind": "?e", "type": "Employee", "name": "?name"}]
        }),
    );
    assert_eq!(r.rows.len(), 5);
    assert!(
        r.rows.iter().all(|row| row[1] == Value::Null),
        "unbound var must be Null, got {:?}",
        r.rows
    );
}

#[test]
fn test_negation_only_clause_errors() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    seed_employees(&db);

    let query = Query::from_json(&serde_json::json!({
        "find": ["?name"],
        "where": [{"not": {"bind": "?e", "type": "Employee", "name": "?name"}}]
    }))
    .unwrap();
    let err = db.query(&query).unwrap_err();
    assert!(
        err.to_string().contains("positive clause"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_aggregate_global() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    seed_employees(&db);

    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["count(*)", "sum(?sal)", "min(?sal)", "max(?sal)", "avg(?sal)"],
            "where": [{"bind": "?e", "type": "Employee", "salary": "?sal"}]
        }),
    );
    assert_eq!(
        r.columns,
        vec!["count(*)", "sum(?sal)", "min(?sal)", "max(?sal)", "avg(?sal)"]
    );
    assert_eq!(r.rows.len(), 1);
    let row = &r.rows[0];
    assert_eq!(row[0], Value::I64(5)); // count
    assert_eq!(row[1], Value::I64(485)); // sum 100+120+90+95+80
    assert_eq!(row[2], Value::I64(80)); // min
    assert_eq!(row[3], Value::I64(120)); // max
    assert_eq!(row[4], Value::F64(97.0)); // avg 485/5
}

#[test]
fn test_aggregate_group_by() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    seed_employees(&db);

    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["?dept", "count(?e)", "max(?sal)", "sum(?sal)"],
            "where": [{"bind": "?e", "type": "Employee", "dept": "?dept", "salary": "?sal"}],
            "order_by": ["?dept"]
        }),
    );
    assert_eq!(r.columns, vec!["?dept", "count(?e)", "max(?sal)", "sum(?sal)"]);
    assert_eq!(r.rows.len(), 2);
    // eng: 2 employees, max 120, sum 220
    assert_eq!(r.rows[0][0], Value::String("eng".into()));
    assert_eq!(r.rows[0][1], Value::I64(2));
    assert_eq!(r.rows[0][2], Value::I64(120));
    assert_eq!(r.rows[0][3], Value::I64(220));
    // sales: 3 employees, max 95, sum 265
    assert_eq!(r.rows[1][0], Value::String("sales".into()));
    assert_eq!(r.rows[1][1], Value::I64(3));
    assert_eq!(r.rows[1][2], Value::I64(95));
    assert_eq!(r.rows[1][3], Value::I64(265));
}

#[test]
fn test_aggregate_count_over_empty_is_zero() {
    let (db, _dir) = test_db();
    db.define_type(employee_type()).unwrap();
    seed_employees(&db);

    let r = run_query(
        &db,
        serde_json::json!({
            "find": ["count(*)"],
            "where": [{"bind": "?e", "type": "Employee", "dept": "nonexistent"}]
        }),
    );
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0], Value::I64(0));
}

#[test]
fn test_order_by_unknown_variable_errors() {
    let (db, _dir) = test_db();
    db.define_type(product_type()).unwrap();
    seed_products(&db);

    let query = Query::from_json(&serde_json::json!({
        "find": ["?name"],
        "where": [{"bind": "?p", "type": "Product", "name": "?name", "price": "?price"}],
        "order_by": ["?price"]
    }))
    .unwrap();
    let err = db.query(&query).unwrap_err();
    assert!(
        err.to_string().contains("order_by variable '?price'"),
        "unexpected error: {err}"
    );
}

// --- Drop / purge (soft + hard) -----------------------------------------

/// Insert one User, returning its entity id.
fn insert_one_user(db: &Database, name: &str, age: i64, email: &str) -> u64 {
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String(name.into()));
    data.insert("age".to_string(), Value::I64(age));
    data.insert("email".to_string(), Value::String(email.into()));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap()
    .entity_ids[0]
}

fn query_all_users(db: &Database) -> datalog_db::query::executor::QueryResult {
    let query = Query::from_json(&serde_json::json!({
        "find": ["?name"],
        "where": [{"bind": "?u", "type": "User", "name": "?name"}]
    }))
    .unwrap();
    db.query(&query).unwrap()
}

fn schema_has_type(db: &Database, name: &str) -> bool {
    db.schema_json()["types"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["name"] == name)
}

fn schema_has_enum(db: &Database, name: &str) -> bool {
    db.schema_json()["enums"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["name"] == name)
}

#[test]
fn test_soft_drop_hides_type_but_keeps_data() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    insert_one_user(&db, "Alice", 30, "alice@example.com");

    let datoms_before = db.all_datoms().unwrap().len();

    let r = db.drop_type("User", false).unwrap();
    assert!(!r.hard);
    assert!(r.tx_id.is_some());

    // Gone from schema and no longer queryable.
    assert!(!schema_has_type(&db, "User"));
    let q = Query::from_json(&serde_json::json!({
        "find": ["?name"],
        "where": [{"bind": "?u", "type": "User", "name": "?name"}]
    }))
    .unwrap();
    assert!(db.query(&q).is_err(), "dropped type must not be queryable");

    // But every data datom is still on disk (soft drop adds a schema
    // retraction datom, so the count only grows).
    assert!(db.all_datoms().unwrap().len() >= datoms_before);

    // Re-defining the exact type makes the old data visible again.
    db.define_type(user_type()).unwrap();
    assert!(schema_has_type(&db, "User"));
    let res = query_all_users(&db);
    assert_eq!(res.rows.len(), 1);
    assert_eq!(res.rows[0][0], Value::String("Alice".into()));
}

#[test]
fn test_soft_drop_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();
        db.define_type(user_type()).unwrap();
        insert_one_user(&db, "Alice", 30, "alice@example.com");
        db.drop_type("User", false).unwrap();
        assert!(!schema_has_type(&db, "User"));
    }
    // The retraction must win on reload — not be resurrected by the
    // still-present `added` definition datom.
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();
        assert!(!schema_has_type(&db, "User"), "soft drop must survive reopen");

        // Re-define and confirm the data is intact.
        db.define_type(user_type()).unwrap();
        assert_eq!(query_all_users(&db).rows.len(), 1);
    }
}

#[test]
fn test_hard_purge_deletes_all_datoms() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    insert_one_user(&db, "Alice", 30, "alice@example.com");
    insert_one_user(&db, "Bob", 25, "bob@example.com");

    let r = db.drop_type("User", true).unwrap();
    assert!(r.hard);
    assert!(r.tx_id.is_none());
    assert_eq!(r.entities_purged, 2);
    assert!(r.datoms_deleted > 0);

    assert!(!schema_has_type(&db, "User"));

    // No User datom of any kind survives in any index.
    let remaining = db.all_datoms().unwrap();
    assert!(
        remaining
            .iter()
            .all(|d| !d.attribute.starts_with("User/")
                && !(d.attribute == "__type" && d.value == Value::String("User".into()))),
        "hard purge left User datoms behind: {:?}",
        remaining
    );

    // Re-define from scratch: the old rows are truly gone.
    db.define_type(user_type()).unwrap();
    assert_eq!(query_all_users(&db).rows.len(), 0);
}

#[test]
fn test_hard_purge_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();
        db.define_type(user_type()).unwrap();
        insert_one_user(&db, "Alice", 30, "alice@example.com");
        db.drop_type("User", true).unwrap();
    }
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).unwrap();
        assert!(!schema_has_type(&db, "User"));
        db.define_type(user_type()).unwrap();
        assert_eq!(query_all_users(&db).rows.len(), 0);
    }
}

#[test]
fn test_hard_purge_refused_when_referenced() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap(); // Post.author: ref(User)

    let err = db.drop_type("User", true).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("still referenced"), "unexpected: {msg}");
    assert!(msg.contains("Post.author"), "should name referrer: {msg}");

    // Soft drop is allowed even when referenced, but warns.
    let r = db.drop_type("User", false).unwrap();
    assert!(!r.warnings.is_empty(), "soft drop should warn about referrers");
    assert!(r.warnings[0].contains("Post.author"));
}

#[test]
fn test_hard_purge_reports_dangling_refs() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    let uid = insert_one_user(&db, "Alice", 30, "alice@example.com");
    let mut post = HashMap::new();
    post.insert("title".to_string(), Value::String("Hello".into()));
    post.insert("author".to_string(), Value::Ref(uid));
    db.transact(vec![TxOp::Assert {
        entity_type: "Post".to_string(),
        entity: None,
        data: post,
    }])
    .unwrap();

    // Soft-drop Post so it stops being a *live* schema referrer (its data,
    // incl. the author ref, stays on disk), then hard purge User.
    db.drop_type("Post", false).unwrap();
    let r = db.drop_type("User", true).unwrap();
    assert_eq!(r.entities_purged, 1);
    assert_eq!(r.dangling_refs, 1, "the post still refs the purged user");
}

#[test]
fn test_soft_then_hard_enum() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    assert!(schema_has_enum(&db, "Shape"));

    // Soft drop, reopen-equivalent in-memory check, then re-define.
    let r = db.drop_enum("Shape", false).unwrap();
    assert!(!r.hard && r.tx_id.is_some());
    assert!(!schema_has_enum(&db, "Shape"));
    db.define_enum(shape_enum()).unwrap();
    assert!(schema_has_enum(&db, "Shape"));

    // Hard purge removes the definition datoms entirely.
    let r = db.drop_enum("Shape", true).unwrap();
    assert!(r.hard);
    assert_eq!(r.entities_purged, 0, "an enum owns no data entities");
    assert!(r.datoms_deleted >= 1, "schema-def datoms should be deleted");
    assert!(!schema_has_enum(&db, "Shape"));
}

#[test]
fn test_hard_purge_enum_refused_when_used() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();

    // A type with a field of the enum.
    let widget = EntityTypeDef {
        unique_keys: vec![],
        name: "Widget".to_string(),
        fields: vec![FieldDef {
            name: "shape".to_string(),
            field_type: FieldType::Enum("Shape".to_string()),
            required: false,
            unique: false,
            indexed: false,
        }],
    };
    db.define_type(widget).unwrap();

    let err = db.drop_enum("Shape", true).unwrap_err();
    assert!(err.to_string().contains("still referenced"), "{err}");
    assert!(err.to_string().contains("Widget.shape"));
}

#[test]
fn test_drop_unknown_type_errors() {
    let (db, _dir) = test_db();
    assert!(db.drop_type("Nope", false).is_err());
    assert!(db.drop_type("Nope", true).is_err());
    assert!(db.drop_enum("Nope", false).is_err());
}

#[test]
fn test_cannot_drop_reserved() {
    let (db, _dir) = test_db();
    assert!(db.drop_type("__type", false).is_err());
    assert!(db.drop_type("__schema_type", true).is_err());
}
