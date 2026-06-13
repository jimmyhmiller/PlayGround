# Rust Client Guide

The `datalog_db::client::Client` is the in-process Rust interface to a
running `datalog-db` server. It speaks the same TCP / framed-JSON wire
protocol as `datalog-cli`, with one ergonomic method per request type
and typed result structs.

This document covers the things rustdoc on the individual methods does
not: the shape of the JSON you pass to `transact` / `query`, the error
model, and how to use the client from more than one thread.

## Starting a server

```sh
cargo run --release --bin datalog-db -- \
    --data-dir ./datalog-data \
    --bind 127.0.0.1:5557
```

The server is a long-running process. Data lives in RocksDB under
`--data-dir`. The wire protocol is plain TCP with **shared-token auth** but
no TLS, so it should still only be bound to a loopback or trusted interface.

### Auth

Auth is **required by default**. Start the server with a shared bearer token
via `--auth-token <secret>` or the `DATALOG_AUTH_TOKEN` env var (the env var
keeps the secret out of `ps`):

```sh
DATALOG_AUTH_TOKEN=s3cret cargo run --release --bin datalog-db -- \
    --data-dir ./datalog-data --bind 127.0.0.1:5557
```

To run an unauthenticated server you must opt out explicitly with `--no-auth`
— the server refuses to start with neither a token nor `--no-auth`, so an
open server is always a deliberate choice.

## Connecting

```rust
use datalog_db::client::Client;

// Auth-enabled server: present the shared token.
let mut client = Client::connect_with_token("127.0.0.1:5557", "s3cret")?;

// --no-auth server: token-free connect (equivalent to an empty token).
let mut client = Client::connect("127.0.0.1:5557")?;
```

Both open a `TcpStream`, set `TCP_NODELAY`, perform the magic+version
handshake (protocol version 2) followed by a token frame, and return the
live `Client`. A bad or missing token fails the handshake with
`ProtocolError::AuthFailed`. The connection stays open for the lifetime of
the value; drop the `Client` to close it.

## Defining schema

```rust
use datalog_db::schema::{EntityTypeDef, EnumTypeDef, EnumVariant, FieldDef, FieldType};

client.define_type(&EntityTypeDef {
    name: "User".into(),
    fields: vec![
        FieldDef { name: "name".into(),  field_type: FieldType::String,
                   required: true,  unique: false, indexed: false },
        FieldDef { name: "email".into(), field_type: FieldType::String,
                   required: true,  unique: true,  indexed: true  },
        FieldDef { name: "age".into(),   field_type: FieldType::I64,
                   required: false, unique: false, indexed: false },
        FieldDef { name: "status".into(), field_type: FieldType::Enum("UserStatus".into()),
                   required: false, unique: false, indexed: false },
    ],
})?;

client.define_enum(&EnumTypeDef {
    name: "UserStatus".into(),
    variants: vec![
        EnumVariant { name: "Active".into(),    fields: vec![] },
        EnumVariant { name: "Suspended".into(), fields: vec![
            FieldDef { name: "reason".into(), field_type: FieldType::String,
                       required: true, unique: false, indexed: false },
        ]},
    ],
})?;
```

Constraints worth remembering:

- A `unique` field must also be `required`. Otherwise `define_type`
  returns `ClientError::Server("unique field 'X' must also be required")`.
- `Ref(target)` and `Enum(target)` reference another defined type by
  name. Define the target first.

## Transactions

`Client::transact` takes a `Vec<serde_json::Value>` — one op per element.
There are three op shapes:

```rust
use serde_json::json;

// Insert (entity id auto-assigned, returned via TransactResult.entity_ids).
json!({ "assert": "User", "data": { "name": "Alice", "email": "a@b.com", "age": 30 } });

// Update an existing entity. Only the listed fields change.
json!({ "assert": "User", "entity": 42, "data": { "age": 31 } });

// Retract specific fields.
json!({ "retract": "User", "entity": 42, "fields": ["age", "email"] });

// Retract a whole entity (soft delete — history is preserved).
json!({ "retract_entity": "User", "entity": 42 });
```

Value encoding inside a `data` object:

| Lang type        | JSON encoding                                  |
|------------------|------------------------------------------------|
| `String`         | `"hello"`                                      |
| `I64`            | `42`                                           |
| `F64`            | `1.5`                                          |
| `Bool`           | `true`                                         |
| `Ref(_)`         | `{ "ref": <entity_id> }`                       |
| `Enum(_)`        | `{ "VariantName": { "field": value, ... } }`   |
| `Bytes`          | `{ "bytes": "<base64>" }`                      |

Multiple ops in one `transact` call are a single atomic transaction;
they share one `tx_id` and are committed together.

```rust
let tx = client.transact(vec![
    json!({ "assert": "User", "data": { "name": "Alice", "email": "a@b.com" } }),
    json!({ "assert": "User", "data": { "name": "Bob",   "email": "b@c.com" } }),
])?;
assert_eq!(tx.entity_ids.len(), 2);
```

`TransactResult` exposes `tx_id`, `entity_ids` (in op order, asserts
only), `datom_count`, and `timestamp_ms`.

## Queries

`Client::query` takes a `serde_json::Value` with `find`, `where`, and
optionally `as_of` / `as_of_time`. The `"type": "query"` wrapper is
added for you.

```rust
let result = client.query(&json!({
    "find":  ["?u", "?name", "?age"],
    "where": [{
        "bind": "?u",
        "type": "User",
        "name": "?name",
        "age":  { "gt": 25 },
    }],
}))?;

for row in &result.rows {
    // row[0] is {"ref": N} for entity ?u; row[1] / row[2] are concrete values.
}
```

Pattern types inside a where clause (per field):

- `"?var"` — bind to a variable.
- A bare JSON scalar (string, number, bool) — exact match.
- `{"ref": N}` — match a specific entity reference.
- `{"gt": V}`, `{"lt": V}`, `{"gte": V}`, `{"lte": V}`, `{"ne": V}` — predicates.
- `{"match": "VariantName", "field": pattern, ...}` — match an enum value.

Multiple where clauses are joined on shared variables. To traverse a
ref, use the same variable for the ref's value and the next clause's
`bind`:

```rust
client.query(&json!({
    "find":  ["?author_name", "?title"],
    "where": [
        { "bind": "?p", "type": "Post", "author": "?a", "title": "?title" },
        { "bind": "?a", "type": "User", "name": "?author_name" },
    ],
}))?;
```

Time travel:

- `"as_of": <tx_id>` — view of the database as of that tx.
- `"as_of_time": <unix_millis | "YYYY-MM-DDTHH:MM:SSZ">` — view as of
  the latest tx at or before that time.

## Explain

`Client::explain` is the same shape as `Client::query` but returns the
plan instead of executing. `ExplainResult.plan` is the structured plan
JSON; `ExplainResult.display` is a pretty-printed string.

## Status and schema

```rust
let st = client.status()?;          // server name + version
let sch = client.schema()?;         // SchemaResult { types: Value, enums: Value }
```

`SchemaResult` returns the raw JSON to avoid coupling clients to the
internal schema types; deserialize into `EntityTypeDef` / `EnumTypeDef`
via `serde_json::from_value` if needed.

## Error model

Every method returns `Result<T, ClientError>`:

- `ClientError::Server(String)` — the server replied
  `{"status":"error","error":"..."}`. The string is the server message
  verbatim; show it to the user.
- `ClientError::Protocol(ProtocolError)` — IO failure, wrong magic
  number, unsupported version, JSON-decoding failure on the wire.
- `ClientError::UnexpectedResponse(String)` — the server replied with a
  shape this client doesn't understand. This shouldn't happen against a
  matching server version; if you see it, the server is newer than the
  client.

After a `Protocol` error the `Client` is in an unknown state; drop it
and reconnect.

## Concurrency

`Client` owns one `TcpStream` and is *not* `Sync`. Two options:

1. **One client per thread.** Recommended for throughput — the server
   supports parallel writes and per-connection processing.
2. **`Arc<Mutex<Client>>`** if you must share. The wire protocol is
   request/response with sequence numbers, but the methods here read
   the response immediately after writing the request, so concurrent
   `send` calls on one `Client` would interleave bytes.

## Raw send

If a future server endpoint isn't yet wrapped, drop down to
`send_raw(payload) -> Result<serde_json::Value>`. It performs the
framing and request-id assignment but does no status checking — you
inspect the response yourself.

## See also

- `src/bin/cli.rs` — `datalog-cli`, including `datalog-cli agent` which
  prints an LLM-targeted usage guide for the CLI surface.
- `tests/integration.rs` — runnable end-to-end examples that exercise
  the client against a real server.
- `docs/conceptual-overview.md` — the data model itself (entities,
  refs, enums, time travel).
