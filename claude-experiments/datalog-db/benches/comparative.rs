//! Comparative macro-benchmark: datalog-db vs PostgreSQL
//!
//! Runs identical workloads over real network connections to both databases
//! and prints a side-by-side timing comparison.
//!
//! Usage:
//!   cargo bench --bench comparative
//!
//! Requires PostgreSQL running locally at localhost:5432 with a `postgres`
//! superuser (or adjust PG_CONN below).

use std::sync::Arc;
use std::time::{Duration, Instant};

use datalog_db::client::Client;
use datalog_db::db::Database;
use datalog_db::schema::{EntityTypeDef, FieldDef, FieldType};
use datalog_db::server::Server;
use datalog_db::storage::rocksdb_backend::RocksDbStorage;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const N: usize = 1000;
const PG_CONN: &str = "host=localhost user=postgres";
const MIXED_THREADS: usize = 4;
const MIXED_OPS_PER_THREAD: usize = 500;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // -- Connect PostgreSQL (required) -------------------------------------
    let bench_db_name = format!("datalog_bench_{}", std::process::id());
    let mut pg_client = setup_postgres(&bench_db_name);

    // -- Start datalog-db --------------------------------------------------
    let tmp = tempfile::tempdir().expect("tempdir");
    let storage = RocksDbStorage::open(tmp.path()).expect("open rocksdb");
    let db = Database::open(Arc::new(storage)).expect("open db");
    let db = Arc::new(db);

    let server = Server::bind("127.0.0.1:0", db).expect("bind");
    let addr = server.local_addr().expect("local_addr");
    let addr_str = addr.to_string();

    // Run server in background thread
    let server_handle = std::thread::spawn(move || {
        let _ = server.run();
    });

    // Give server a moment to start accepting
    std::thread::sleep(Duration::from_millis(50));

    // -- Connect datalog-db client -----------------------------------------
    let mut dl_client = Client::connect(&addr_str).expect("connect datalog-db");
    setup_datalog_schema(&mut dl_client);

    // -- Run workloads -----------------------------------------------------
    println!();
    println!("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    println!("  datalog-db vs PostgreSQL Benchmark");
    println!("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    println!();
    println!(
        "{:<28} {:>12} {:>12} {:>8}",
        "Workload", "datalog-db", "PostgreSQL", "Ratio"
    );
    println!(
        "{:\u{2500}<28} {:\u{2500}>12} {:\u{2500}>12} {:\u{2500}>8}",
        "", "", "", ""
    );

    struct WorkloadResult {
        name: String,
        dl_dur: Duration,
        pg_dur: Duration,
    }

    let mut results: Vec<WorkloadResult> = Vec::new();

    // 1. Single inserts
    {
        let dl = run_workload(|| datalog_single_inserts(&addr_str, N));
        let pg = run_workload(|| pg_single_inserts(&mut pg_client, N));
        results.push(WorkloadResult {
            name: format!("Single inserts ({})", N),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 2. Batch insert
    {
        let dl = run_workload(|| datalog_batch_insert(&addr_str, N));
        let pg = run_workload(|| pg_batch_insert(&mut pg_client, N));
        results.push(WorkloadResult {
            name: format!("Batch insert ({})", N),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 3. Point query
    {
        let dl = run_workload(|| datalog_point_query(&addr_str, N));
        let pg = run_workload(|| pg_point_query(&mut pg_client, N));
        results.push(WorkloadResult {
            name: format!("Point query ({}x)", N),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 4. Range query
    {
        let dl = run_workload(|| datalog_range_query(&addr_str, N));
        let pg = run_workload(|| pg_range_query(&mut pg_client, N));
        results.push(WorkloadResult {
            name: format!("Range query ({}x)", N),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 5. Join query
    {
        datalog_insert_posts(&addr_str);
        pg_insert_posts(&mut pg_client);

        let dl = run_workload(|| datalog_join_query(&addr_str, N));
        let pg = run_workload(|| pg_join_query(&mut pg_client, N));
        results.push(WorkloadResult {
            name: format!("Join query ({}x)", N),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 6. Mixed read/write
    {
        let dl = run_workload(|| datalog_mixed(&addr_str, MIXED_THREADS, MIXED_OPS_PER_THREAD));
        let pg = run_workload(|| pg_mixed(&bench_db_name, MIXED_THREADS, MIXED_OPS_PER_THREAD));
        let total_ops = MIXED_THREADS * MIXED_OPS_PER_THREAD;
        results.push(WorkloadResult {
            name: format!("Mixed {}-thread ({} ops)", MIXED_THREADS, total_ops),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // -- Complex benchmarks setup ------------------------------------------
    println!("  Populating complex data...");
    datalog_insert_complex_data(&addr_str);
    pg_insert_complex_data(&mut pg_client);
    println!("  Done.");
    println!();

    // 7. 3-Way Join Chain (User → Post → Comment)
    {
        let dl = run_workload(|| datalog_three_way_join(&addr_str, 500));
        let pg = run_workload(|| pg_three_way_join(&mut pg_client, 500));
        results.push(WorkloadResult {
            name: "3-Way Join (500x)".into(),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 8. 4-Way Star Join (Order → User + Product)
    {
        let dl = run_workload(|| datalog_four_way_star_join(&addr_str, 500));
        let pg = run_workload(|| pg_four_way_star_join(&mut pg_client, 500));
        results.push(WorkloadResult {
            name: "4-Way Star Join (500x)".into(),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 9. Large Fan-Out Join
    {
        let dl = run_workload(|| datalog_large_fanout_join(&addr_str, 200));
        let pg = run_workload(|| pg_large_fanout_join(&mut pg_client, 200));
        results.push(WorkloadResult {
            name: "Large Fan-Out (200x)".into(),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 10. Graph 2-Hop (Employee → Manager → Skip-Level)
    {
        let dl = run_workload(|| datalog_graph_two_hop(&addr_str, 500));
        let pg = run_workload(|| pg_graph_two_hop(&mut pg_client, 500));
        results.push(WorkloadResult {
            name: "Graph 2-Hop (500x)".into(),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 11. Selective Multi-Predicate Scan
    {
        let dl = run_workload(|| datalog_selective_scan(&addr_str, 1000));
        let pg = run_workload(|| pg_selective_scan(&mut pg_client, 1000));
        results.push(WorkloadResult {
            name: "Selective Scan (1000x)".into(),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 12. Many-to-Many Join (Post ↔ Tag)
    {
        let dl = run_workload(|| datalog_many_to_many_join(&addr_str, 500));
        let pg = run_workload(|| pg_many_to_many_join(&mut pg_client, 500));
        results.push(WorkloadResult {
            name: "Many-to-Many Join (500x)".into(),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // 13. Unfiltered Large Scan
    {
        let dl = run_workload(|| datalog_unfiltered_scan(&addr_str, 500));
        let pg = run_workload(|| pg_unfiltered_scan(&mut pg_client, 500));
        results.push(WorkloadResult {
            name: "Unfiltered Scan (500x)".into(),
            dl_dur: dl,
            pg_dur: pg,
        });
    }

    // -- Print results -----------------------------------------------------
    for r in &results {
        let dl_s = format!("{:.3}s", r.dl_dur.as_secs_f64());
        let ratio = r.dl_dur.as_secs_f64() / r.pg_dur.as_secs_f64();
        let pg_s = format!("{:.3}s", r.pg_dur.as_secs_f64());
        let ratio_s = format!("{:.2}x", ratio);
        println!("{:<28} {:>12} {:>12} {:>8}", r.name, dl_s, pg_s, ratio_s);
    }

    println!();

    // -- Cleanup -----------------------------------------------------------
    drop(dl_client);
    drop(pg_client);
    cleanup_postgres(&bench_db_name);

    // Server thread will exit when TcpListener is dropped
    drop(server_handle);
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

fn run_workload<F: FnOnce()>(f: F) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

// ---------------------------------------------------------------------------
// datalog-db schema setup
// ---------------------------------------------------------------------------

fn setup_datalog_schema(client: &mut Client) {
    client
        .define_type(&EntityTypeDef {
            name: "User".into(),
            fields: vec![
                FieldDef {
                    name: "name".into(),
                    field_type: FieldType::String,
                    required: true,
                    unique: false,
                    indexed: true,
                },
                FieldDef {
                    name: "age".into(),
                    field_type: FieldType::I64,
                    required: false,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "email".into(),
                    field_type: FieldType::String,
                    required: true,
                    unique: true,
                    indexed: false,
                },
            ],
        })
        .expect("define User");

    client
        .define_type(&EntityTypeDef {
            name: "Post".into(),
            fields: vec![
                FieldDef {
                    name: "title".into(),
                    field_type: FieldType::String,
                    required: true,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "body".into(),
                    field_type: FieldType::String,
                    required: false,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "author_id".into(),
                    field_type: FieldType::Ref("User".into()),
                    required: true,
                    unique: false,
                    indexed: true,
                },
            ],
        })
        .expect("define Post");

    client
        .define_type(&EntityTypeDef {
            name: "Comment".into(),
            fields: vec![
                FieldDef {
                    name: "body".into(),
                    field_type: FieldType::String,
                    required: true,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "post_id".into(),
                    field_type: FieldType::Ref("Post".into()),
                    required: true,
                    unique: false,
                    indexed: true,
                },
                FieldDef {
                    name: "author_id".into(),
                    field_type: FieldType::Ref("User".into()),
                    required: true,
                    unique: false,
                    indexed: true,
                },
            ],
        })
        .expect("define Comment");

    client
        .define_type(&EntityTypeDef {
            name: "Product".into(),
            fields: vec![
                FieldDef {
                    name: "name".into(),
                    field_type: FieldType::String,
                    required: true,
                    unique: true,
                    indexed: false,
                },
                FieldDef {
                    name: "price".into(),
                    field_type: FieldType::I64,
                    required: true,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "category".into(),
                    field_type: FieldType::String,
                    required: true,
                    unique: false,
                    indexed: true,
                },
            ],
        })
        .expect("define Product");

    client
        .define_type(&EntityTypeDef {
            name: "Order".into(),
            fields: vec![
                FieldDef {
                    name: "user_id".into(),
                    field_type: FieldType::Ref("User".into()),
                    required: true,
                    unique: false,
                    indexed: true,
                },
                FieldDef {
                    name: "product_id".into(),
                    field_type: FieldType::Ref("Product".into()),
                    required: true,
                    unique: false,
                    indexed: true,
                },
                FieldDef {
                    name: "quantity".into(),
                    field_type: FieldType::I64,
                    required: true,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "total".into(),
                    field_type: FieldType::I64,
                    required: true,
                    unique: false,
                    indexed: false,
                },
            ],
        })
        .expect("define Order");

    client
        .define_type(&EntityTypeDef {
            name: "Employee".into(),
            fields: vec![
                FieldDef {
                    name: "name".into(),
                    field_type: FieldType::String,
                    required: true,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "department".into(),
                    field_type: FieldType::String,
                    required: true,
                    unique: false,
                    indexed: true,
                },
                FieldDef {
                    name: "salary".into(),
                    field_type: FieldType::I64,
                    required: true,
                    unique: false,
                    indexed: false,
                },
                FieldDef {
                    name: "manager_id".into(),
                    field_type: FieldType::Ref("Employee".into()),
                    required: false,
                    unique: false,
                    indexed: true,
                },
            ],
        })
        .expect("define Employee");

    client
        .define_type(&EntityTypeDef {
            name: "Tag".into(),
            fields: vec![FieldDef {
                name: "name".into(),
                field_type: FieldType::String,
                required: true,
                unique: true,
                indexed: false,
            }],
        })
        .expect("define Tag");

    client
        .define_type(&EntityTypeDef {
            name: "PostTag".into(),
            fields: vec![
                FieldDef {
                    name: "post_id".into(),
                    field_type: FieldType::Ref("Post".into()),
                    required: true,
                    unique: false,
                    indexed: true,
                },
                FieldDef {
                    name: "tag_id".into(),
                    field_type: FieldType::Ref("Tag".into()),
                    required: true,
                    unique: false,
                    indexed: true,
                },
            ],
        })
        .expect("define PostTag");
}

// ---------------------------------------------------------------------------
// PostgreSQL setup
// ---------------------------------------------------------------------------

fn setup_postgres(bench_db: &str) -> postgres::Client {
    // Create the benchmark database
    let mut admin = postgres::Client::connect(PG_CONN, postgres::NoTls)
        .expect("PostgreSQL connection failed. This benchmark requires PostgreSQL running at localhost:5432");
    let _ = admin.execute(&format!("DROP DATABASE IF EXISTS {}", bench_db), &[]);
    admin
        .execute(&format!("CREATE DATABASE {}", bench_db), &[])
        .expect("create bench db");
    drop(admin);

    // Connect to the benchmark database
    let conn_str = format!("{} dbname={}", PG_CONN, bench_db);
    let mut client = postgres::Client::connect(&conn_str, postgres::NoTls).expect("pg connect");

    client
        .batch_execute(
            "CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                age INT,
                email TEXT UNIQUE NOT NULL
            );
            CREATE TABLE posts (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT,
                author_id INT NOT NULL REFERENCES users(id)
            );
            CREATE INDEX ON users(name);
            CREATE INDEX ON posts(author_id);

            CREATE TABLE comments (
                id SERIAL PRIMARY KEY,
                body TEXT NOT NULL,
                post_id INT NOT NULL REFERENCES posts(id),
                author_id INT NOT NULL REFERENCES users(id)
            );
            CREATE INDEX ON comments(post_id);
            CREATE INDEX ON comments(author_id);

            CREATE TABLE products (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                price INT NOT NULL,
                category TEXT NOT NULL
            );
            CREATE INDEX ON products(category);

            CREATE TABLE orders (
                id SERIAL PRIMARY KEY,
                user_id INT NOT NULL REFERENCES users(id),
                product_id INT NOT NULL REFERENCES products(id),
                quantity INT NOT NULL,
                total INT NOT NULL
            );
            CREATE INDEX ON orders(user_id);
            CREATE INDEX ON orders(product_id);

            CREATE TABLE employees (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary INT NOT NULL,
                manager_id INT REFERENCES employees(id)
            );
            CREATE INDEX ON employees(department);
            CREATE INDEX ON employees(manager_id);

            CREATE TABLE tags (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE post_tags (
                id SERIAL PRIMARY KEY,
                post_id INT NOT NULL REFERENCES posts(id),
                tag_id INT NOT NULL REFERENCES tags(id)
            );
            CREATE INDEX ON post_tags(post_id);
            CREATE INDEX ON post_tags(tag_id);",
        )
        .expect("create pg tables");

    client
}

fn cleanup_postgres(bench_db: &str) {
    if let Ok(mut admin) = postgres::Client::connect(PG_CONN, postgres::NoTls) {
        let _ = admin.execute(&format!("DROP DATABASE IF EXISTS {}", bench_db), &[]);
    }
}

// ---------------------------------------------------------------------------
// datalog-db workloads
// ---------------------------------------------------------------------------

fn datalog_single_inserts(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for i in 0..n {
        client
            .transact(vec![serde_json::json!({
                "assert": "User",
                "data": {
                    "name": format!("user_{}", i),
                    "age": (i % 100) as i64,
                    "email": format!("user_{}@single.test", i),
                }
            })])
            .expect("single insert");
    }
}

fn datalog_batch_insert(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    let ops: Vec<serde_json::Value> = (0..n)
        .map(|i| {
            serde_json::json!({
                "assert": "User",
                "data": {
                    "name": format!("batch_user_{}", i),
                    "age": (i % 100) as i64,
                    "email": format!("batch_{}@batch.test", i),
                }
            })
        })
        .collect();
    client.transact(ops).expect("batch insert");
}

fn datalog_point_query(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for i in 0..n {
        let name = format!("user_{}", i % N);
        client
            .query(&serde_json::json!({
                "find": ["?name", "?age"],
                "where": [{
                    "bind": "?u",
                    "type": "User",
                    "name": name,
                }]
            }))
            .expect("point query");
    }
}

fn datalog_range_query(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?name"],
                "where": [{
                    "bind": "?u",
                    "type": "User",
                    "age": {"gt": 50},
                }]
            }))
            .expect("range query");
    }
}

fn datalog_insert_posts(addr: &str) {
    let mut client = Client::connect(addr).expect("connect");
    // Insert some posts referencing existing users (entity IDs from single inserts start at 9 - after 8 schema entities)
    let ops: Vec<serde_json::Value> = (0..100)
        .map(|i| {
            serde_json::json!({
                "assert": "Post",
                "data": {
                    "title": format!("Post {}", i),
                    "body": format!("Body of post {}", i),
                    "author_id": {"ref": (i % 50) + 9},
                }
            })
        })
        .collect();
    client.transact(ops).expect("insert posts");
}

fn datalog_join_query(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?name", "?title"],
                "where": [
                    {
                        "bind": "?u",
                        "type": "User",
                        "name": "?name",
                    },
                    {
                        "bind": "?p",
                        "type": "Post",
                        "title": "?title",
                        "author_id": "?u",
                    }
                ]
            }))
            .expect("join query");
    }
}

fn datalog_mixed(addr: &str, threads: usize, ops_per_thread: usize) {
    let addr = addr.to_string();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let addr = addr.clone();
            std::thread::spawn(move || {
                let mut client = Client::connect(&addr).expect("connect");
                for i in 0..ops_per_thread {
                    if i % 5 == 0 {
                        // 20% writes
                        client
                            .transact(vec![serde_json::json!({
                                "assert": "User",
                                "data": {
                                    "name": format!("mixed_t{}_i{}", t, i),
                                    "age": (i % 100) as i64,
                                    "email": format!("mixed_t{}_i{}@mix.test", t, i),
                                }
                            })])
                            .expect("mixed write");
                    } else {
                        // 80% reads
                        let name = format!("user_{}", i % N);
                        client
                            .query(&serde_json::json!({
                                "find": ["?name", "?age"],
                                "where": [{
                                    "bind": "?u",
                                    "type": "User",
                                    "name": name,
                                }]
                            }))
                            .expect("mixed read");
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread join");
    }
}

// ---------------------------------------------------------------------------
// datalog-db complex data population
// ---------------------------------------------------------------------------

const SCHEMA_ENTITY_COUNT: u64 = 8; // User, Post, Comment, Product, Order, Employee, Tag, PostTag
const DEPARTMENTS: [&str; 5] = ["Engineering", "Sales", "Marketing", "Finance", "Operations"];
const CATEGORIES: [&str; 5] = ["Electronics", "Books", "Clothing", "Food", "Tools"];

fn datalog_insert_complex_data(addr: &str) {
    let mut client = Client::connect(addr).expect("connect");

    // User entity IDs: single inserts created IDs (SCHEMA+1)..(SCHEMA+1000),
    // batch inserts created IDs (SCHEMA+1001)..(SCHEMA+2000)
    let user_ids: Vec<u64> = ((SCHEMA_ENTITY_COUNT + 1)..=(SCHEMA_ENTITY_COUNT + 2000)).collect();

    // Insert 5000 posts (batched in chunks of 500)
    let mut post_ids: Vec<u64> = Vec::with_capacity(5000);
    for chunk_start in (0..5000usize).step_by(500) {
        let chunk_end = (chunk_start + 500).min(5000);
        let ops: Vec<serde_json::Value> = (chunk_start..chunk_end)
            .map(|i| {
                serde_json::json!({
                    "assert": "Post",
                    "data": {
                        "title": format!("Complex Post {}", i),
                        "body": format!("Body of complex post {}", i),
                        "author_id": {"ref": user_ids[i % user_ids.len()]}
                    }
                })
            })
            .collect();
        let result = client.transact(ops).expect("insert complex posts");
        post_ids.extend(&result.entity_ids);
    }

    // Insert 20000 comments (batched in chunks of 500)
    for chunk_start in (0..20000usize).step_by(500) {
        let chunk_end = (chunk_start + 500).min(20000);
        let ops: Vec<serde_json::Value> = (chunk_start..chunk_end)
            .map(|i| {
                serde_json::json!({
                    "assert": "Comment",
                    "data": {
                        "body": format!("Comment body {}", i),
                        "post_id": {"ref": post_ids[i % post_ids.len()]},
                        "author_id": {"ref": user_ids[i % user_ids.len()]}
                    }
                })
            })
            .collect();
        client.transact(ops).expect("insert comments");
    }

    // Insert 500 products
    let ops: Vec<serde_json::Value> = (0..500)
        .map(|i| {
            serde_json::json!({
                "assert": "Product",
                "data": {
                    "name": format!("Product {}", i),
                    "price": (10 + (i * 7) % 990) as i64,
                    "category": CATEGORIES[i % CATEGORIES.len()]
                }
            })
        })
        .collect();
    let product_result = client.transact(ops).expect("insert products");
    let product_ids = &product_result.entity_ids;

    // Insert 10000 orders (batched in chunks of 500)
    for chunk_start in (0..10000usize).step_by(500) {
        let chunk_end = (chunk_start + 500).min(10000);
        let ops: Vec<serde_json::Value> = (chunk_start..chunk_end)
            .map(|i| {
                let qty = (i % 10 + 1) as i64;
                let price = (10 + (i * 7) % 990) as i64;
                serde_json::json!({
                    "assert": "Order",
                    "data": {
                        "user_id": {"ref": user_ids[i % user_ids.len()]},
                        "product_id": {"ref": product_ids[i % product_ids.len()]},
                        "quantity": qty,
                        "total": qty * price
                    }
                })
            })
            .collect();
        client.transact(ops).expect("insert orders");
    }

    // Insert 2000 employees as a tree (depth ~5)
    // Level 0: 1 CEO (no manager)
    let ceo_ops = vec![serde_json::json!({
        "assert": "Employee",
        "data": {
            "name": "CEO_0",
            "department": DEPARTMENTS[0],
            "salary": 200000i64
        }
    })];
    let ceo_result = client.transact(ceo_ops).expect("insert CEO");
    let mut prev_level_ids = ceo_result.entity_ids.clone();
    let mut all_employee_ids: Vec<u64> = ceo_result.entity_ids.clone();
    let mut remaining = 1999usize;

    // Build tree: each level has ~fan_factor children per parent
    let level_sizes = [5usize, 25, 125, 1844]; // total = 1999
    for (level, &size) in level_sizes.iter().enumerate() {
        let count = size.min(remaining);
        if count == 0 {
            break;
        }
        let ops: Vec<serde_json::Value> = (0..count)
            .map(|i| {
                let manager = prev_level_ids[i % prev_level_ids.len()];
                let dept = DEPARTMENTS[(level + i) % DEPARTMENTS.len()];
                let salary = (50000 + ((4 - level.min(3)) * 30000) + (i * 1000) % 20000) as i64;
                serde_json::json!({
                    "assert": "Employee",
                    "data": {
                        "name": format!("Emp_L{}_{}", level + 1, i),
                        "department": dept,
                        "salary": salary,
                        "manager_id": {"ref": manager}
                    }
                })
            })
            .collect();
        let result = client.transact(ops).expect("insert employees");
        prev_level_ids = result.entity_ids.clone();
        all_employee_ids.extend(&result.entity_ids);
        remaining -= count;
    }
    let _ = all_employee_ids; // used for reference

    // Insert 50 tags
    let ops: Vec<serde_json::Value> = (0..50)
        .map(|i| {
            let tag_name = match i {
                0 => "rust".to_string(),
                1 => "python".to_string(),
                2 => "javascript".to_string(),
                _ => format!("tag_{}", i),
            };
            serde_json::json!({
                "assert": "Tag",
                "data": { "name": tag_name }
            })
        })
        .collect();
    let tag_result = client.transact(ops).expect("insert tags");
    let tag_ids = &tag_result.entity_ids;

    // Insert 15000 post_tags (batched in chunks of 500)
    for chunk_start in (0..15000usize).step_by(500) {
        let chunk_end = (chunk_start + 500).min(15000);
        let ops: Vec<serde_json::Value> = (chunk_start..chunk_end)
            .map(|i| {
                serde_json::json!({
                    "assert": "PostTag",
                    "data": {
                        "post_id": {"ref": post_ids[i % post_ids.len()]},
                        "tag_id": {"ref": tag_ids[i % tag_ids.len()]}
                    }
                })
            })
            .collect();
        client.transact(ops).expect("insert post_tags");
    }
}

fn pg_insert_complex_data(client: &mut postgres::Client) {
    // Insert 5000 posts
    {
        let mut txn = client.transaction().expect("pg begin");
        for i in 0..5000 {
            txn.execute(
                "INSERT INTO posts (title, body, author_id) VALUES ($1, $2, $3)",
                &[
                    &format!("Complex Post {}", i),
                    &format!("Body of complex post {}", i),
                    &((i % 2000 + 1) as i32),
                ],
            )
            .expect("pg insert complex post");
        }
        txn.commit().expect("pg commit posts");
    }

    // Insert 20000 comments
    {
        // PG post IDs: first 100 from simple bench (1..100), then 5000 complex (101..5100)
        let mut txn = client.transaction().expect("pg begin");
        for i in 0..20000 {
            txn.execute(
                "INSERT INTO comments (body, post_id, author_id) VALUES ($1, $2, $3)",
                &[
                    &format!("Comment body {}", i),
                    &((i % 5000 + 101) as i32), // reference complex posts (PG IDs 101..5100)
                    &((i % 2000 + 1) as i32),
                ],
            )
            .expect("pg insert comment");
        }
        txn.commit().expect("pg commit comments");
    }

    // Insert 500 products
    {
        let mut txn = client.transaction().expect("pg begin");
        for i in 0..500 {
            txn.execute(
                "INSERT INTO products (name, price, category) VALUES ($1, $2, $3)",
                &[
                    &format!("Product {}", i),
                    &((10 + (i * 7) % 990) as i32),
                    &CATEGORIES[i % CATEGORIES.len()],
                ],
            )
            .expect("pg insert product");
        }
        txn.commit().expect("pg commit products");
    }

    // Insert 10000 orders
    {
        let mut txn = client.transaction().expect("pg begin");
        for i in 0..10000 {
            let qty = (i % 10 + 1) as i32;
            let price = (10 + (i * 7) % 990) as i32;
            txn.execute(
                "INSERT INTO orders (user_id, product_id, quantity, total) VALUES ($1, $2, $3, $4)",
                &[
                    &((i % 2000 + 1) as i32),
                    &((i % 500 + 1) as i32),
                    &qty,
                    &(qty * price),
                ],
            )
            .expect("pg insert order");
        }
        txn.commit().expect("pg commit orders");
    }

    // Insert 2000 employees as a tree
    {
        let mut txn = client.transaction().expect("pg begin");
        // CEO (no manager)
        txn.execute(
            "INSERT INTO employees (name, department, salary) VALUES ($1, $2, $3)",
            &[&"CEO_0", &DEPARTMENTS[0], &200000i32],
        )
        .expect("pg insert CEO");

        let mut emp_id = 2i32; // next employee ID
        let level_sizes = [5usize, 25, 125, 1844];
        let mut prev_level_start = 1i32;
        let mut prev_level_count = 1usize;

        for (level, &size) in level_sizes.iter().enumerate() {
            for i in 0..size {
                let manager_id = prev_level_start + (i % prev_level_count) as i32;
                let dept = DEPARTMENTS[(level + i) % DEPARTMENTS.len()];
                let salary =
                    (50000 + ((4 - level.min(3)) * 30000) + (i * 1000) % 20000) as i32;
                txn.execute(
                    "INSERT INTO employees (name, department, salary, manager_id) VALUES ($1, $2, $3, $4)",
                    &[
                        &format!("Emp_L{}_{}", level + 1, i),
                        &dept,
                        &salary,
                        &manager_id,
                    ],
                )
                .expect("pg insert employee");
                emp_id += 1;
            }
            prev_level_start = emp_id - size as i32;
            prev_level_count = size;
        }
        let _ = emp_id;
        txn.commit().expect("pg commit employees");
    }

    // Insert 50 tags
    {
        let mut txn = client.transaction().expect("pg begin");
        for i in 0..50 {
            let tag_name = match i {
                0 => "rust".to_string(),
                1 => "python".to_string(),
                2 => "javascript".to_string(),
                _ => format!("tag_{}", i),
            };
            txn.execute(
                "INSERT INTO tags (name) VALUES ($1)",
                &[&tag_name],
            )
            .expect("pg insert tag");
        }
        txn.commit().expect("pg commit tags");
    }

    // Insert 15000 post_tags
    {
        // PG post IDs for complex posts: 101..5100, tag IDs: 1..50
        let mut txn = client.transaction().expect("pg begin");
        for i in 0..15000 {
            txn.execute(
                "INSERT INTO post_tags (post_id, tag_id) VALUES ($1, $2)",
                &[
                    &((i % 5000 + 101) as i32),
                    &((i % 50 + 1) as i32),
                ],
            )
            .expect("pg insert post_tag");
        }
        txn.commit().expect("pg commit post_tags");
    }
}

// ---------------------------------------------------------------------------
// datalog-db complex workloads
// ---------------------------------------------------------------------------

/// 3-Way Join Chain: User → Post → Comment
fn datalog_three_way_join(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?name", "?title", "?body"],
                "where": [
                    {"bind": "?u", "type": "User", "name": "user_42"},
                    {"bind": "?p", "type": "Post", "author_id": "?u", "title": "?title"},
                    {"bind": "?c", "type": "Comment", "post_id": "?p", "body": "?body"}
                ]
            }))
            .expect("3-way join");
    }
}

/// 4-Way Star Join: Order → User + Product (filtered by category)
fn datalog_four_way_star_join(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?uname", "?pname", "?qty", "?total"],
                "where": [
                    {"bind": "?pr", "type": "Product", "category": "Electronics", "name": "?pname"},
                    {"bind": "?o", "type": "Order", "product_id": "?pr", "user_id": "?u", "quantity": "?qty", "total": "?total"},
                    {"bind": "?u", "type": "User", "name": "?uname"}
                ]
            }))
            .expect("4-way star join");
    }
}

/// Large Fan-Out Join: all User → Post pairs (5000+ rows)
fn datalog_large_fanout_join(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?name", "?title"],
                "where": [
                    {"bind": "?u", "type": "User", "name": "?name"},
                    {"bind": "?p", "type": "Post", "author_id": "?u", "title": "?title"}
                ]
            }))
            .expect("large fanout join");
    }
}

/// Graph 2-Hop: Employee → Manager → Skip-Level Manager
fn datalog_graph_two_hop(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?ename", "?mname", "?sname"],
                "where": [
                    {"bind": "?e", "type": "Employee", "department": "Engineering", "name": "?ename", "manager_id": "?m"},
                    {"bind": "?m", "type": "Employee", "name": "?mname", "manager_id": "?s"},
                    {"bind": "?s", "type": "Employee", "name": "?sname"}
                ]
            }))
            .expect("graph 2-hop");
    }
}

/// Selective Multi-Predicate Scan: salary > 80000 AND department = Engineering
fn datalog_selective_scan(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?name", "?salary"],
                "where": [{
                    "bind": "?e",
                    "type": "Employee",
                    "department": "Engineering",
                    "salary": {"gt": 80000},
                    "name": "?name"
                }]
            }))
            .expect("selective scan");
    }
}

/// Many-to-Many Join: Post ↔ Tag via PostTag (find posts tagged "rust")
fn datalog_many_to_many_join(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?title"],
                "where": [
                    {"bind": "?t", "type": "Tag", "name": "rust"},
                    {"bind": "?pt", "type": "PostTag", "tag_id": "?t", "post_id": "?p"},
                    {"bind": "?p", "type": "Post", "title": "?title"}
                ]
            }))
            .expect("many-to-many join");
    }
}

/// Unfiltered Large Scan: all employees
fn datalog_unfiltered_scan(addr: &str, n: usize) {
    let mut client = Client::connect(addr).expect("connect");
    for _ in 0..n {
        client
            .query(&serde_json::json!({
                "find": ["?name", "?dept", "?salary"],
                "where": [{
                    "bind": "?e",
                    "type": "Employee",
                    "name": "?name",
                    "department": "?dept",
                    "salary": "?salary"
                }]
            }))
            .expect("unfiltered scan");
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL workloads
// ---------------------------------------------------------------------------

fn pg_single_inserts(client: &mut postgres::Client, n: usize) {
    for i in 0..n {
        client
            .execute(
                "INSERT INTO users (name, age, email) VALUES ($1, $2, $3)",
                &[
                    &format!("user_{}", i),
                    &((i % 100) as i32),
                    &format!("user_{}@single.test", i),
                ],
            )
            .expect("pg single insert");
    }
}

fn pg_batch_insert(client: &mut postgres::Client, n: usize) {
    let mut txn = client.transaction().expect("pg begin");
    for i in 0..n {
        txn.execute(
            "INSERT INTO users (name, age, email) VALUES ($1, $2, $3)",
            &[
                &format!("batch_user_{}", i),
                &((i % 100) as i32),
                &format!("batch_{}@batch.test", i),
            ],
        )
        .expect("pg batch insert row");
    }
    txn.commit().expect("pg commit");
}

fn pg_point_query(client: &mut postgres::Client, n: usize) {
    for i in 0..n {
        let name = format!("user_{}", i % N);
        client
            .query("SELECT name, age FROM users WHERE name = $1", &[&name])
            .expect("pg point query");
    }
}

fn pg_range_query(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query("SELECT name FROM users WHERE age > $1", &[&50i32])
            .expect("pg range query");
    }
}

fn pg_insert_posts(client: &mut postgres::Client) {
    let mut txn = client.transaction().expect("pg begin");
    for i in 0..100 {
        txn.execute(
            "INSERT INTO posts (title, body, author_id) VALUES ($1, $2, $3)",
            &[
                &format!("Post {}", i),
                &format!("Body of post {}", i),
                &((i % 50) + 1i32),
            ],
        )
        .expect("pg insert post");
    }
    txn.commit().expect("pg commit posts");
}

fn pg_join_query(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query(
                "SELECT u.name, p.title FROM users u JOIN posts p ON p.author_id = u.id",
                &[],
            )
            .expect("pg join query");
    }
}

fn pg_mixed(bench_db: &str, threads: usize, ops_per_thread: usize) {
    let conn_str = format!("{} dbname={}", PG_CONN, bench_db);
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let conn_str = conn_str.clone();
            std::thread::spawn(move || {
                let mut client =
                    postgres::Client::connect(&conn_str, postgres::NoTls).expect("pg connect");
                for i in 0..ops_per_thread {
                    if i % 5 == 0 {
                        // 20% writes
                        client
                            .execute(
                                "INSERT INTO users (name, age, email) VALUES ($1, $2, $3)",
                                &[
                                    &format!("mixed_t{}_i{}", t, i),
                                    &((i % 100) as i32),
                                    &format!("mixed_t{}_i{}@mix.test", t, i),
                                ],
                            )
                            .expect("pg mixed write");
                    } else {
                        // 80% reads
                        let name = format!("user_{}", i % N);
                        client
                            .query(
                                "SELECT name, age FROM users WHERE name = $1",
                                &[&name],
                            )
                            .expect("pg mixed read");
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread join");
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL complex workloads
// ---------------------------------------------------------------------------

fn pg_three_way_join(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query(
                "SELECT u.name, p.title, c.body
                 FROM users u
                 JOIN posts p ON p.author_id = u.id
                 JOIN comments c ON c.post_id = p.id
                 WHERE u.name = $1",
                &[&"user_42"],
            )
            .expect("pg 3-way join");
    }
}

fn pg_four_way_star_join(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query(
                "SELECT u.name, pr.name, o.quantity, o.total
                 FROM orders o
                 JOIN users u ON o.user_id = u.id
                 JOIN products pr ON o.product_id = pr.id
                 WHERE pr.category = $1",
                &[&"Electronics"],
            )
            .expect("pg 4-way star join");
    }
}

fn pg_large_fanout_join(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query(
                "SELECT u.name, p.title FROM users u JOIN posts p ON p.author_id = u.id",
                &[],
            )
            .expect("pg large fanout join");
    }
}

fn pg_graph_two_hop(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query(
                "SELECT e.name, m.name, s.name
                 FROM employees e
                 JOIN employees m ON e.manager_id = m.id
                 JOIN employees s ON m.manager_id = s.id
                 WHERE e.department = $1",
                &[&"Engineering"],
            )
            .expect("pg graph 2-hop");
    }
}

fn pg_selective_scan(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query(
                "SELECT name, salary FROM employees WHERE salary > $1 AND department = $2",
                &[&80000i32, &"Engineering"],
            )
            .expect("pg selective scan");
    }
}

fn pg_many_to_many_join(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query(
                "SELECT p.title
                 FROM posts p
                 JOIN post_tags pt ON pt.post_id = p.id
                 JOIN tags t ON pt.tag_id = t.id
                 WHERE t.name = $1",
                &[&"rust"],
            )
            .expect("pg many-to-many join");
    }
}

fn pg_unfiltered_scan(client: &mut postgres::Client, n: usize) {
    for _ in 0..n {
        client
            .query(
                "SELECT name, department, salary FROM employees",
                &[],
            )
            .expect("pg unfiltered scan");
    }
}
