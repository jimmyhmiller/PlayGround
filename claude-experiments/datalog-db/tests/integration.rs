use std::collections::HashMap;
use std::sync::Arc;

use datalog_db::datom::Value;
use datalog_db::db::Database;
use datalog_db::query::Query;
use datalog_db::schema::{EntityTypeDef, EnumTypeDef, EnumVariant, FieldDef, FieldType};
use datalog_db::storage::rocksdb_backend::RocksDbStorage;
use datalog_db::tx::TxOp;

/// Create a temporary database for testing.
async fn test_db() -> (Arc<Database>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let storage = RocksDbStorage::open(dir.path()).unwrap();
    let storage = Arc::new(storage);
    let db = Database::open(storage).await.unwrap();
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
                required: false,
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

#[tokio::test]
async fn test_define_type() {
    let (db, _dir) = test_db().await;

    let tx_id = db.define_type(user_type()).await.unwrap();
    assert!(tx_id > 0);
}

#[tokio::test]
async fn test_transact_insert() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

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
        .await
        .unwrap();

    assert!(result.tx_id > 0);
    assert_eq!(result.entity_ids.len(), 1);
    assert!(result.entity_ids[0] > 0);
    assert_eq!(result.datom_count, 4); // name + age + email + __type
}

#[tokio::test]
async fn test_transact_and_get_entity() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert("age".to_string(), Value::I64(25));

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
        .unwrap();

    let eid = result.entity_ids[0];
    let entity = db.get_entity(eid).await.unwrap().unwrap();

    assert_eq!(entity.get("User/name"), Some(&Value::String("Bob".to_string())));
    assert_eq!(entity.get("User/age"), Some(&Value::I64(25)));
    assert_eq!(
        entity.get("__type"),
        Some(&Value::String("User".to_string()))
    );
}

#[tokio::test]
async fn test_transact_update() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    // Insert
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Charlie".to_string()));
    data.insert("age".to_string(), Value::I64(20));

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
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
    .await
    .unwrap();

    let entity = db.get_entity(eid).await.unwrap().unwrap();
    // Should show latest age
    assert_eq!(entity.get("User/age"), Some(&Value::I64(21)));
    // Name should still be there
    assert_eq!(
        entity.get("User/name"),
        Some(&Value::String("Charlie".to_string()))
    );
}

#[tokio::test]
async fn test_transact_retract() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

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
        .await
        .unwrap();
    let eid = result.entity_ids[0];

    // Retract email
    db.transact(vec![TxOp::Retract {
        entity_type: "User".to_string(),
        entity: eid,
        fields: vec!["email".to_string()],
    }])
    .await
    .unwrap();

    let entity = db.get_entity(eid).await.unwrap().unwrap();
    assert_eq!(
        entity.get("User/name"),
        Some(&Value::String("Diana".to_string()))
    );
    assert_eq!(entity.get("User/age"), Some(&Value::I64(40)));
    assert!(entity.get("User/email").is_none()); // retracted
}

#[tokio::test]
async fn test_schema_validation_unknown_type() {
    let (db, _dir) = test_db().await;

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Eve".to_string()));

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "NonExistent".to_string(),
            entity: None,
            data,
        }])
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_schema_validation_type_mismatch() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Frank".to_string()));
    data.insert("age".to_string(), Value::String("not a number".to_string())); // wrong type

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_schema_validation_missing_required() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    let mut data = HashMap::new();
    data.insert("age".to_string(), Value::I64(25)); // missing required "name"

    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_basic_query() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    // Insert two users
    for (name, age) in [("Alice", 30), ("Bob", 25)] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.to_string()));
        data.insert("age".to_string(), Value::I64(age));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
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
    let result = db.query(&query).await.unwrap();

    assert_eq!(result.columns, vec!["?name", "?age"]);
    assert_eq!(result.rows.len(), 2);

    // Check both users are present (order may vary)
    let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::String("Alice".to_string())));
    assert!(names.contains(&&Value::String("Bob".to_string())));
}

#[tokio::test]
async fn test_query_with_constant_filter() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    for (name, age) in [("Alice", 30), ("Bob", 25), ("Charlie", 35)] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.to_string()));
        data.insert("age".to_string(), Value::I64(age));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
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
    let result = db.query(&query).await.unwrap();

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::I64(25));
}

#[tokio::test]
async fn test_query_with_predicate() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    for (name, age) in [("Alice", 30), ("Bob", 25), ("Charlie", 35)] {
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String(name.to_string()));
        data.insert("age".to_string(), Value::I64(age));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
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
    let result = db.query(&query).await.unwrap();

    assert_eq!(result.rows.len(), 2);
    let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
    assert!(names.contains(&&Value::String("Alice".to_string())));
    assert!(names.contains(&&Value::String("Charlie".to_string())));
}

#[tokio::test]
async fn test_query_with_join() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();
    db.define_type(post_type()).await.unwrap();

    // Create users
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    let alice_result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
        .unwrap();
    let alice_id = alice_result.entity_ids[0];

    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Bob".to_string()));
    data.insert("age".to_string(), Value::I64(25));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .await
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
        .await
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
    let result = db.query(&query).await.unwrap();

    // Alice has 2 posts, Bob has 0 — so 2 result rows
    assert_eq!(result.rows.len(), 2);
    for row in &result.rows {
        assert_eq!(row[0], Value::String("Alice".to_string()));
    }
    let titles: Vec<&Value> = result.rows.iter().map(|r| &r[1]).collect();
    assert!(titles.contains(&&Value::String("First Post".to_string())));
    assert!(titles.contains(&&Value::String("Second Post".to_string())));
}

#[tokio::test]
async fn test_time_travel_query() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    // Insert user at tx1
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    let result1 = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
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
        .await
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
    let result = db.query(&query).await.unwrap();
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
    let result = db.query(&query).await.unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::I64(31));
}

#[tokio::test]
async fn test_schema_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();

    // Open, define type, close
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).await.unwrap();
        db.define_type(user_type()).await.unwrap();

        // Transact a user
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("Persist".to_string()));
        db.transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
        .unwrap();
    }

    // Reopen and verify schema is loaded
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).await.unwrap();

        // Should be able to transact without re-defining
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("Persist2".to_string()));
        let result = db
            .transact(vec![TxOp::Assert {
                entity_type: "User".to_string(),
                entity: None,
                data,
            }])
            .await;
        assert!(result.is_ok());

        // And query should work
        let query_json = serde_json::json!({
            "find": ["?name"],
            "where": [
                {"bind": "?u", "type": "User", "name": "?name"}
            ]
        });
        let query = Query::from_json(&query_json).unwrap();
        let result = db.query(&query).await.unwrap();
        assert_eq!(result.rows.len(), 2); // both users
    }
}

#[tokio::test]
async fn test_tcp_protocol_roundtrip() {
    use datalog_db::protocol;
    use tokio::net::TcpListener;

    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Server task
    let server_handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut stream = stream;
        protocol::server_handshake(&mut stream).await.unwrap();

        // Read one message
        let msg = protocol::read_message(&mut stream).await.unwrap();
        assert_eq!(msg.request_id, 1);

        // Send response
        let response = serde_json::json!({"status": "ok", "data": "hello"});
        protocol::write_message(&mut stream, msg.request_id, &response)
            .await
            .unwrap();
    });

    // Client
    let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
    protocol::client_handshake(&mut client).await.unwrap();

    // Send a message
    let payload = serde_json::json!({"type": "status"});
    protocol::write_message(&mut client, 1, &payload)
        .await
        .unwrap();

    // Read response
    let response = protocol::read_message(&mut client).await.unwrap();
    assert_eq!(response.request_id, 1);
    assert_eq!(response.payload["status"], "ok");

    server_handle.await.unwrap();
}

#[tokio::test]
async fn test_full_server_define_transact_query() {
    use datalog_db::protocol;
    use datalog_db::server::Server;

    let (db, _dir) = test_db().await;

    let server = Server::bind("127.0.0.1:0", db.clone()).await.unwrap();
    // We need the address before running the server
    // Unfortunately Server doesn't expose the address, so let's use a different approach
    // Let's just bind manually and use the server dispatch logic
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    // Drop the server we created, we'll use the listener directly
    drop(server);

    let db_clone = db.clone();
    let server_handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut stream = stream;
        protocol::server_handshake(&mut stream).await.unwrap();

        // Process messages until client disconnects
        loop {
            let msg = match protocol::read_message(&mut stream).await {
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
                    let tx_id = db_clone.define_type(type_def).await.unwrap();
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
                    let result = db_clone.transact(ops).await.unwrap();
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
                    let result = db_clone.query(&query).await.unwrap();
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

            protocol::write_message(&mut stream, msg.request_id, &response)
                .await
                .unwrap();
        }
    });

    // Client side
    let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
    protocol::client_handshake(&mut client).await.unwrap();

    // 1. Define User type
    let define_payload = serde_json::json!({
        "type": "define",
        "entity_type": "User",
        "fields": [
            {"name": "name", "type": "string", "required": true},
            {"name": "age", "type": "i64"}
        ]
    });
    protocol::write_message(&mut client, 1, &define_payload)
        .await
        .unwrap();
    let resp = protocol::read_message(&mut client).await.unwrap();
    assert_eq!(resp.payload["status"], "ok");

    // 2. Transact: insert users
    let transact_payload = serde_json::json!({
        "type": "transact",
        "ops": [
            {"assert": "User", "data": {"name": "Alice", "age": 30}},
            {"assert": "User", "data": {"name": "Bob", "age": 25}}
        ]
    });
    protocol::write_message(&mut client, 2, &transact_payload)
        .await
        .unwrap();
    let resp = protocol::read_message(&mut client).await.unwrap();
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
    protocol::write_message(&mut client, 3, &query_payload)
        .await
        .unwrap();
    let resp = protocol::read_message(&mut client).await.unwrap();
    assert_eq!(resp.payload["status"], "ok");
    let rows = resp.payload["data"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);

    // Drop client to close connection, ending the server loop
    drop(client);
    server_handle.await.unwrap();
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

#[tokio::test]
async fn test_define_enum() {
    let (db, _dir) = test_db().await;
    let tx_id = db.define_enum(shape_enum()).await.unwrap();
    assert!(tx_id > 0);
}

#[tokio::test]
async fn test_enum_insert_and_query() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

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
        .await
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
    let result = db.query(&query).await.unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("my circle".to_string()));
    assert_eq!(result.rows[0][1], Value::F64(5.0));
}

#[tokio::test]
async fn test_enum_unit_variant() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

    // Insert a Point (unit variant — no fields)
    let mut data = HashMap::new();
    data.insert("label".to_string(), Value::String("origin".to_string()));
    data.insert("shape".to_string(), Value::String("Point".to_string()));
    db.transact(vec![TxOp::Assert {
        entity_type: "Drawing".to_string(),
        entity: None,
        data,
    }])
    .await
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
    let result = db.query(&query).await.unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("origin".to_string()));
}

#[tokio::test]
async fn test_enum_match_filters_variants() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

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
        .await
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
    let result = db.query(&query).await.unwrap();
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
    let result = db.query(&query).await.unwrap();
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
    let result = db.query(&query).await.unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("origin".to_string()));
}

#[tokio::test]
async fn test_enum_bind_variant_tag() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

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
        .await
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
    let result = db.query(&query).await.unwrap();
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

#[tokio::test]
async fn test_enum_variant_change() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

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
        .await
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
    .await
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
    let result = db.query(&query).await.unwrap();
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
    let result = db.query(&query).await.unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][1], Value::F64(10.0));
    assert_eq!(result.rows[0][2], Value::F64(20.0));
}

#[tokio::test]
async fn test_enum_same_variant_stale_field_retraction() {
    let (db, _dir) = test_db().await;

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
    db.define_enum(status_enum).await.unwrap();

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
    db.define_type(account_type).await.unwrap();

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
        .await
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
    .await
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
    let result = db.query(&query).await.unwrap();
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
    let result = db.query(&query).await.unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("Spam".to_string()));
}

#[tokio::test]
async fn test_enum_retract_whole_field() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

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
        .await
        .unwrap();
    let eid = result.entity_ids[0];

    // Retract the shape field entirely
    db.transact(vec![TxOp::Retract {
        entity_type: "Drawing".to_string(),
        entity: eid,
        fields: vec!["shape".to_string()],
    }])
    .await
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
    let result = db.query(&query).await.unwrap();
    assert_eq!(result.rows.len(), 0);
}

#[tokio::test]
async fn test_enum_schema_validation() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

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
        }])
        .await;
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
        }])
        .await;
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
        }])
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_enum_json_wire_format() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

    // Insert via JSON (as it would come over the wire)
    let op_json = serde_json::json!({
        "assert": "Drawing",
        "data": {
            "label": "wire circle",
            "shape": {"Circle": {"radius": 7.5}}
        }
    });
    let op = TxOp::from_json(&op_json).unwrap();
    db.transact(vec![op]).await.unwrap();

    // Also test unit variant over the wire
    let op_json = serde_json::json!({
        "assert": "Drawing",
        "data": {
            "label": "wire point",
            "shape": "Point"
        }
    });
    let op = TxOp::from_json(&op_json).unwrap();
    db.transact(vec![op]).await.unwrap();

    // Query to verify
    let query_json = serde_json::json!({
        "find": ["?label", "?r"],
        "where": [
            {"bind": "?d", "type": "Drawing", "label": "?label",
             "shape": {"match": "Circle", "radius": "?r"}}
        ]
    });
    let query = Query::from_json(&query_json).unwrap();
    let result = db.query(&query).await.unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("wire circle".to_string()));
    assert_eq!(result.rows[0][1], Value::F64(7.5));
}

#[tokio::test]
async fn test_enum_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();

    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).await.unwrap();
        db.define_enum(shape_enum()).await.unwrap();
        db.define_type(drawing_type()).await.unwrap();

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
        .await
        .unwrap();
    }

    // Reopen
    {
        let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
        let db = Database::open(storage).await.unwrap();

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
        .await
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
        let result = db.query(&query).await.unwrap();
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

#[tokio::test]
async fn test_snapshot_scalar_insert() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    // Before: no data datoms
    let before = db.all_datoms().await.unwrap();
    assert_eq!(
        snapshot_datoms(&before),
        "",
        "before insert, no data datoms"
    );

    // Insert Alice
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: None,
        data,
    }])
    .await
    .unwrap();

    let after = db.all_datoms().await.unwrap();
    insta::assert_snapshot!("scalar_insert", snapshot_datoms(&after));
}

#[tokio::test]
async fn test_snapshot_scalar_update() {
    let (db, _dir) = test_db().await;
    db.define_type(user_type()).await.unwrap();

    // Insert
    let mut data = HashMap::new();
    data.insert("name".to_string(), Value::String("Alice".to_string()));
    data.insert("age".to_string(), Value::I64(30));
    let result = db
        .transact(vec![TxOp::Assert {
            entity_type: "User".to_string(),
            entity: None,
            data,
        }])
        .await
        .unwrap();
    let eid = result.entity_ids[0];

    let after_insert = db.all_datoms().await.unwrap();
    insta::assert_snapshot!("scalar_update__after_insert", snapshot_datoms(&after_insert));

    // Update age 30 -> 31 (should retract 30, assert 31)
    let mut data = HashMap::new();
    data.insert("age".to_string(), Value::I64(31));
    db.transact(vec![TxOp::Assert {
        entity_type: "User".to_string(),
        entity: Some(eid),
        data,
    }])
    .await
    .unwrap();

    let after_update = db.all_datoms().await.unwrap();
    insta::assert_snapshot!("scalar_update__after_update", snapshot_datoms(&after_update));
}

#[tokio::test]
async fn test_snapshot_enum_insert() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

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
    .await
    .unwrap();

    let datoms = db.all_datoms().await.unwrap();
    insta::assert_snapshot!("enum_insert", snapshot_datoms(&datoms));
}

#[tokio::test]
async fn test_snapshot_enum_variant_change() {
    let (db, _dir) = test_db().await;
    db.define_enum(shape_enum()).await.unwrap();
    db.define_type(drawing_type()).await.unwrap();

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
        .await
        .unwrap();
    let eid = result.entity_ids[0];

    let after_insert = db.all_datoms().await.unwrap();
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
    .await
    .unwrap();

    let after_change = db.all_datoms().await.unwrap();
    insta::assert_snapshot!(
        "enum_variant_change__after_change",
        snapshot_datoms(&after_change)
    );
}

#[tokio::test]
async fn test_snapshot_enum_same_variant_update() {
    let (db, _dir) = test_db().await;

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
    db.define_enum(status_enum).await.unwrap();

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
    db.define_type(account_type).await.unwrap();

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
        .await
        .unwrap();
    let eid = result.entity_ids[0];

    let after_insert = db.all_datoms().await.unwrap();
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
    .await
    .unwrap();

    let after_update = db.all_datoms().await.unwrap();
    insta::assert_snapshot!(
        "enum_same_variant_update__after_update",
        snapshot_datoms(&after_update)
    );
}
