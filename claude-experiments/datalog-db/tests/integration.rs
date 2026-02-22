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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
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
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert("age".to_string(), Value::I64(25));
    data.insert("email".to_string(), Value::String("bob@example.com".to_string()));

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();

    let eid = result.entity_ids[0];
    let entity = db.get_entity(eid).unwrap().unwrap();

    assert_eq!(entity.get("User/name"), Some(&Value::String("Bob".to_string())));
    assert_eq!(entity.get("User/age"), Some(&Value::I64(25)));
    assert_eq!(
        entity.get("__type"),
        Some(&Value::String("User".to_string()))
    );
}

#[test]
fn test_transact_update() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Charlie".to_string()));
    data.insert("age".to_string(), Value::I64(20));
    data.insert("email".to_string(), Value::String("charlie@example.com".to_string()));

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
        Some(&Value::String("Charlie".to_string()))
    );
}

#[test]
fn test_transact_retract() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Diana".to_string()));
    data.insert("age".to_string(), Value::I64(40));
    data.insert(
        "email".to_string(),
        Value::String("diana@example.com".to_string()),
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
        Some(&Value::String("Diana".to_string()))
    );
    assert_eq!(entity.get("User/age"), Some(&Value::I64(40)));
    assert!(entity.get("User/email").is_none()); // retracted
}

#[test]
fn test_schema_validation_unknown_type() {
    let (db, _dir) = test_db();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Eve".to_string()));

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
    data.insert("name".to_string(), Value::String("Frank".to_string()));
    data.insert("age".to_string(), Value::String("not a number".to_string())); // wrong type

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
        data.insert("name".to_string(), Value::String(name.to_string()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.to_string()));
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
    assert!(names.contains(&&Value::String("Alice".to_string())));
    assert!(names.contains(&&Value::String("Bob".to_string())));
}

#[test]
fn test_query_with_constant_filter() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    for (name, age, email) in [("Alice", 30, "alice@example.com"), ("Bob", 25, "bob@example.com"), ("Charlie", 35, "charlie@example.com")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.to_string()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.to_string()));
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
        data.insert("name".to_string(), Value::String(name.to_string()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.to_string()));
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
    assert!(names.contains(&&Value::String("Alice".to_string())));
    assert!(names.contains(&&Value::String("Charlie".to_string())));
}

#[test]
fn test_query_with_join() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();
    db.define_type(post_type()).unwrap();

    // Create users
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));
    let alice_result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .unwrap();
    let alice_id = alice_result.entity_ids[0];

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert("age".to_string(), Value::I64(25));
    data.insert("email".to_string(), Value::String("bob@example.com".to_string()));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Create posts by Alice
    for title in ["First Post", "Second Post"] {
        let mut data = HashMap::new();
        data.insert("title".to_string(), Value::String(title.to_string()));
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
        assert_eq!(row[0], Value::String("Alice".to_string()));
    }
    let titles: Vec<&Value> = result.rows.iter().map(|r| &r[1]).collect();
    assert!(titles.contains(&&Value::String("First Post".to_string())));
    assert!(titles.contains(&&Value::String("Second Post".to_string())));
}

#[test]
fn test_time_travel_query() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert user at tx1
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));
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
        data.insert("name".to_string(), Value::String("Persist".to_string()));
        data.insert("email".to_string(), Value::String("persist@example.com".to_string()));
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
        data.insert("name".to_string(), Value::String("Persist2".to_string()));
        data.insert("email".to_string(), Value::String("persist2@example.com".to_string()));
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
    data.insert("label".to_string(), Value::String("my circle".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Circle".to_string(),
            fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]),
        },
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
    assert_eq!(result.rows[0][0], Value::String("my circle".to_string()));
    assert_eq!(result.rows[0][1], Value::F64(5.0));
}

#[test]
fn test_enum_unit_variant() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert a Point (unit variant — no fields)
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("origin".to_string()));
    data.insert("shape".to_string(), Value::String("Point".to_string()));
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
    assert_eq!(result.rows[0][0], Value::String("origin".to_string()));
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
            Value::Enum {
                variant: "Circle".to_string(),
                fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]),
            },
        ),
        (
            "my rect",
            Value::Enum {
                variant: "Rect".to_string(),
                fields: HashMap::from([
                    ("w".to_string(), Value::F64(10.0)),
                    ("h".to_string(), Value::F64(20.0)),
                ]),
            },
        ),
        ("origin", Value::String("Point".to_string())),
    ];

    for (label, shape) in drawings {
        let mut data = HashMap::new();
        data.insert("label".to_string(), Value::String(label.to_string()));
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
    assert_eq!(result.rows[0][0], Value::String("my circle".to_string()));
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
    assert_eq!(result.rows[0][0], Value::String("my rect".to_string()));
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
    assert_eq!(result.rows[0][0], Value::String("origin".to_string()));
}

#[test]
fn test_enum_bind_variant_tag() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    for (label, shape) in [
        (
            "c1",
            Value::Enum {
                variant: "Circle".to_string(),
                fields: HashMap::from([("radius".to_string(), Value::F64(1.0))]),
            },
        ),
        ("p1", Value::String("Point".to_string())),
    ] {
        let mut data = HashMap::new();
        data.insert("label".to_string(), Value::String(label.to_string()));
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

    assert_eq!(tags[0].0, Value::String("c1".to_string()));
    assert_eq!(tags[0].1, Value::String("Circle".to_string()));
    assert_eq!(tags[1].0, Value::String("p1".to_string()));
    assert_eq!(tags[1].1, Value::String("Point".to_string()));
}

#[test]
fn test_enum_variant_change() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    // Insert as Circle
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("morphing".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Circle".to_string(),
            fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]),
        },
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
        Value::Enum {
            variant: "Rect".to_string(),
            fields: HashMap::from([
                ("w".to_string(), Value::F64(10.0)),
                ("h".to_string(), Value::F64(20.0)),
            ]),
        },
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
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert(
        "status".to_string(),
        Value::Enum {
            variant: "Suspended".to_string(),
            fields: HashMap::from([
                ("reason".to_string(), Value::String("TOS".to_string())),
                ("until".to_string(), Value::I64(1000)),
            ]),
        },
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
        Value::Enum {
            variant: "Suspended".to_string(),
            fields: HashMap::from([("reason".to_string(), Value::String("Spam".to_string()))]),
        },
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
    assert_eq!(result.rows[0][0], Value::String("Spam".to_string()));
}

#[test]
fn test_enum_retract_whole_field() {
    let (db, _dir) = test_db();
    db.define_enum(shape_enum()).unwrap();
    db.define_type(drawing_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("gone".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Circle".to_string(),
            fields: HashMap::from([("radius".to_string(), Value::F64(3.0))]),
        },
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
    data.insert("label".to_string(), Value::String("bad".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Triangle".to_string(), // doesn't exist
            fields: HashMap::new(),
        },
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
    data.insert("label".to_string(), Value::String("bad".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Circle".to_string(),
            fields: HashMap::from([("radius".to_string(), Value::String("not a number".to_string()))]),
        },
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
    data.insert("label".to_string(), Value::String("bad".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Rect".to_string(),
            fields: HashMap::from([("w".to_string(), Value::F64(10.0))]), // missing 'h'
        },
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
    assert_eq!(result.rows[0][0], Value::String("wire circle".to_string()));
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
        data.insert("label".to_string(), Value::String("persistent".to_string()));
        data.insert(
            "shape".to_string(),
            Value::Enum {
                variant: "Circle".to_string(),
                fields: HashMap::from([("radius".to_string(), Value::F64(42.0))]),
            },
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
        data.insert("label".to_string(), Value::String("after reopen".to_string()));
        data.insert(
            "shape".to_string(),
            Value::Enum {
                variant: "Rect".to_string(),
                fields: HashMap::from([
                    ("w".to_string(), Value::F64(1.0)),
                    ("h".to_string(), Value::F64(2.0)),
                ]),
            },
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
        assert_eq!(result.rows[0][0], Value::String("persistent".to_string()));
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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));
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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));
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
    data.insert("label".to_string(), Value::String("my circle".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Circle".to_string(),
            fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]),
        },
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
    data.insert("label".to_string(), Value::String("shape1".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Circle".to_string(),
            fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]),
        },
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
        Value::Enum {
            variant: "Rect".to_string(),
            fields: HashMap::from([
                ("w".to_string(), Value::F64(10.0)),
                ("h".to_string(), Value::F64(20.0)),
            ]),
        },
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
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert(
        "status".to_string(),
        Value::Enum {
            variant: "Suspended".to_string(),
            fields: HashMap::from([
                ("reason".to_string(), Value::String("TOS".to_string())),
                ("until".to_string(), Value::I64(1000)),
            ]),
        },
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
        Value::Enum {
            variant: "Suspended".to_string(),
            fields: HashMap::from([("reason".to_string(), Value::String("Spam".to_string()))]),
        },
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
            data.insert("label".to_string(), Value::String(format!("task_{}", i)));
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

// --- Unique constraint tests ---

#[test]
fn test_unique_basic() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert user with unique email
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Try to insert another user with the same email — should fail
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
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
        Value::String("alice@example.com".to_string()),
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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
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
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Bob with different email
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("bob@example.com".to_string()),
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
        Value::String("alice@example.com".to_string()),
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
    data1.insert("name".to_string(), Value::String("Alice".to_string()));
    data1.insert(
        "email".to_string(),
        Value::String("shared@example.com".to_string()),
    );
    let mut data2 = HashMap::new();
    data2.insert("name".to_string(), Value::String("Bob".to_string()));
    data2.insert(
        "email".to_string(),
        Value::String("shared@example.com".to_string()),
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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Admin with same email value — should succeed (different type → different attribute)
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice Admin".to_string()));
    data.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
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
        data.insert("name".to_string(), Value::String(name));
        data.insert("age".to_string(), Value::I64(i));
        data.insert("email".to_string(), Value::String(format!("user{}@example.com", i)));
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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));
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
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert("email".to_string(), Value::String("bob@example.com".to_string()));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    // Create posts — some by Alice, some by Bob
    for i in 0..10 {
        let mut data = HashMap::new();
        data.insert("title".to_string(), Value::String(format!("Post {}", i)));
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
    data.insert("label".to_string(), Value::String("circle1".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Circle".to_string(),
            fields: HashMap::from([("radius".to_string(), Value::F64(5.0))]),
        },
    );
    db.transact(vec![TxOp::Assert {
        entity_type: "Drawing".to_string(),
        entity: None,
        data,
    }])
    .unwrap();

    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("rect1".to_string()));
    data.insert(
        "shape".to_string(),
        Value::Enum {
            variant: "Rect".to_string(),
            fields: HashMap::from([
                ("w".to_string(), Value::F64(10.0)),
                ("h".to_string(), Value::F64(20.0)),
            ]),
        },
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
    assert_eq!(result.rows[0][0], Value::String("circle1".to_string()));
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
        data.insert("name".to_string(), Value::String(name.to_string()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.to_string()));
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
        Value::String("alice@example.com".to_string())
    );
}

#[test]
fn test_query_predicate_falls_back() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    for (name, age, email) in [("Alice", 30, "alice@example.com"), ("Bob", 25, "bob@example.com"), ("Charlie", 35, "charlie@example.com")] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.to_string()));
        data.insert("age".to_string(), Value::I64(age));
        data.insert("email".to_string(), Value::String(email.to_string()));
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
    assert!(names.contains(&&Value::String("Alice".to_string())));
    assert!(names.contains(&&Value::String("Charlie".to_string())));
}

// --- Retract entity (soft delete) tests ---

#[test]
fn test_retract_entity() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    // Insert two users
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));
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
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert("age".to_string(), Value::I64(25));
    data.insert("email".to_string(), Value::String("bob@example.com".to_string()));
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
    assert_eq!(result.rows[0][0], Value::String("Bob".to_string()));

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
    assert!(names.contains(&&Value::String("Alice".to_string())));
}

// --- Wall-clock timestamp tests ---

#[test]
fn test_transact_returns_timestamp() {
    let (db, _dir) = test_db();
    db.define_type(user_type()).unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));

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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));
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
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert("age".to_string(), Value::I64(25));
    data.insert("email".to_string(), Value::String("bob@example.com".to_string()));
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
    assert_eq!(result.rows[0][0], Value::String("Alice".to_string()));

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
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    data.insert("email".to_string(), Value::String("alice@example.com".to_string()));
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
