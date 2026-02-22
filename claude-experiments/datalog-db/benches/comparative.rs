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
            CREATE INDEX ON posts(author_id);",
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
    // Insert some posts referencing existing users (entity IDs from single inserts start at 3 - after 2 schema entities)
    let ops: Vec<serde_json::Value> = (0..100)
        .map(|i| {
            serde_json::json!({
                "assert": "Post",
                "data": {
                    "title": format!("Post {}", i),
                    "body": format!("Body of post {}", i),
                    "author_id": {"ref": (i % 50) + 3},
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
