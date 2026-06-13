//! Fuzz / robustness tests for the wire-handshake parsers.
//!
//! These throw random and malformed bytes at the authentication code and assert
//! two invariants that must hold for ANY input:
//!   1. It never panics (no unwrap/slice/try_into blowups on attacker bytes).
//!   2. It never authenticates garbage (no `Ok` / authenticated outcome without
//!      a valid credential).
//!
//! We use proptest (already a dev-dependency) rather than cargo-fuzz so this
//! runs on stable and in the normal `cargo test` suite. proptest's shrinking
//! also yields a minimal reproducer when something does break.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use datalog_db::auth::scram::{ScramServer, Verifier};
use datalog_db::db::Database;
use datalog_db::protocol::{self, ServerAuth};
use datalog_db::storage::rocksdb_backend::RocksDbStorage;

use proptest::prelude::*;

fn test_db() -> (Arc<Database>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(RocksDbStorage::open(dir.path()).unwrap());
    (Arc::new(Database::open(storage).unwrap()), dir)
}

proptest! {
    // The SCRAM server message parsers must never panic on arbitrary input, and
    // must never report `ok == true` for an input the attacker controls (it
    // cannot forge a proof without the password).
    #[test]
    fn scram_server_never_panics_never_authenticates(
        client_first in ".{0,300}",
        client_final in ".{0,300}",
    ) {
        let verifier = Verifier::create("the-real-password");
        let mut server = ScramServer::new(verifier);

        // server_first may legitimately error on malformed input — just must
        // not panic. If it succeeds, feed the (attacker-controlled) client_final
        // to server_final; it must never authenticate.
        if server.server_first(&client_first).is_ok() {
            if let Ok((_server_final, ok)) = server.server_final(&client_final) {
                prop_assert!(!ok, "garbage client-final must never authenticate");
            }
        }
    }

    // Same, but the attacker has observed a *real* server-first (correct nonce
    // echo) and only forges the proof `p=`. Still must never authenticate.
    #[test]
    fn scram_forged_proof_with_valid_nonce_rejected(
        proof_b64 in "[A-Za-z0-9+/=]{0,80}",
    ) {
        let verifier = Verifier::create("pw");
        let mut server = ScramServer::new(verifier);
        let sf = server.server_first("n,,n=alice,r=clientnonce").unwrap();
        // Extract the combined nonce the server expects.
        let combined = sf.split(',').next().unwrap().strip_prefix("r=").unwrap();
        let client_final = format!("c=biws,r={combined},p={proof_b64}");
        if let Ok((_sf, ok)) = server.server_final(&client_final) {
            prop_assert!(!ok, "forged proof must never authenticate");
        }
    }
}

/// Drive the FULL server_handshake over a real loopback socket with arbitrary
/// bytes. Asserts the handshake thread never panics and never returns an
/// authenticated outcome. Runs a batch of random payloads against fresh
/// connections (kept modest so the suite stays fast).
#[test]
fn server_handshake_random_bytes_never_panics_or_authenticates() {
    let (db, _dir) = test_db();
    db.put_scram_verifier("alice", &Verifier::create("secret"))
        .unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    // A deterministic-but-varied set of hostile payloads.
    let payloads = hostile_payloads();
    let n = payloads.len();

    let db2 = db.clone();
    let server = std::thread::spawn(move || {
        for _ in 0..n {
            let (mut stream, _) = listener.accept().unwrap();
            // Mirror Server::run's handshake deadline: a truncated/slow payload
            // (e.g. a frame header claiming bytes the client never sends) must
            // not block the handshake forever. A short timeout turns it into a
            // clean Err instead of a hang. (In production Server::run sets this;
            // here we drive server_handshake directly so we set it ourselves.)
            stream
                .set_read_timeout(Some(std::time::Duration::from_millis(500)))
                .unwrap();
            let store = DbStore(db2.clone());
            let auth = ServerAuth {
                expected_token: Some(b"the-token".to_vec()),
                users: Some(&store),
                mock_key: b"fuzz-mock-key",
                allow_no_auth: false,
            };
            // The contract: this returns Result; on Ok it must carry a real
            // identity (which random bytes can't produce). It must NOT panic.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                protocol::server_handshake(&mut stream, &auth)
            }));
            match outcome {
                Err(_) => panic!("server_handshake PANICKED on hostile input"),
                Ok(Ok(auth_outcome)) => {
                    // The only way to legitimately get Ok here with these
                    // payloads would be a correct token; none of the hostile
                    // payloads send "the-token", so any Ok is a bypass.
                    panic!(
                        "server_handshake AUTHENTICATED hostile input: {:?}",
                        auth_outcome
                    );
                }
                Ok(Err(_)) => { /* expected: rejected cleanly */ }
            }
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
    });

    for p in payloads {
        if let Ok(mut c) = TcpStream::connect(addr) {
            // Bound the client read too, so a payload where the server is still
            // reading (truncated frame) doesn't block the client's drain.
            let _ = c.set_read_timeout(Some(std::time::Duration::from_millis(800)));
            let _ = c.write_all(&p);
            let _ = c.flush();
            let mut buf = [0u8; 64];
            let _ = c.read(&mut buf);
        }
    }
    server.join().unwrap();
}

/// A spread of malformed handshakes: good magic/version but garbage after;
/// truncated frames; oversized length prefixes; valid method header + wrong
/// token; SCRAM with junk messages; etc.
fn hostile_payloads() -> Vec<Vec<u8>> {
    let magic = 0xDA7A_1061u32.to_be_bytes();
    let v3 = 3u32.to_be_bytes();
    let frame = |b: &[u8]| {
        let mut f = (b.len() as u32).to_be_bytes().to_vec();
        f.extend_from_slice(b);
        f
    };
    let mut out: Vec<Vec<u8>> = Vec::new();

    // Pure garbage.
    out.push(vec![]);
    out.push(vec![0u8; 4]);
    out.push(vec![0xFF; 64]);

    // Good magic+version, then nothing / truncated frame length.
    out.push([magic, v3].concat());
    out.push([&magic[..], &v3[..], &[0x00, 0x00, 0x10, 0x00]].concat()); // claims 4096 bytes, sends none

    // Good prefix, oversized frame length (must hit the MAX_AUTH_FRAME cap, not OOM).
    out.push([&magic[..], &v3[..], &0xFFFF_FFFFu32.to_be_bytes()[..]].concat());

    // Good prefix + unknown method.
    {
        let mut p = [&magic[..], &v3[..]].concat();
        p.extend(frame(br#"{"method":"banana"}"#));
        out.push(p);
    }
    // Non-JSON method frame.
    {
        let mut p = [&magic[..], &v3[..]].concat();
        p.extend(frame(b"not json at all"));
        out.push(p);
    }
    // token method + WRONG token.
    {
        let mut p = [&magic[..], &v3[..]].concat();
        p.extend(frame(br#"{"method":"token"}"#));
        p.extend(frame(b"wrong-token"));
        out.push(p);
    }
    // token method + empty token (server requires "the-token").
    {
        let mut p = [&magic[..], &v3[..]].concat();
        p.extend(frame(br#"{"method":"token"}"#));
        p.extend(frame(b""));
        out.push(p);
    }
    // scram method + junk SASL messages.
    {
        let mut p = [&magic[..], &v3[..]].concat();
        p.extend(frame(br#"{"method":"scram","user":"alice"}"#));
        p.extend(frame(b"\xff\xff not a sasl message"));
        out.push(p);
    }
    // scram for an unknown user (mock path) + junk.
    {
        let mut p = [&magic[..], &v3[..]].concat();
        p.extend(frame(br#"{"method":"scram","user":"ghost"}"#));
        p.extend(frame(b"n,,n=ghost,r=abc"));
        p.extend(frame(b"c=biws,r=wrongnonce,p=AAAA"));
        out.push(p);
    }
    // Wrong magic.
    out.push([&[0xDE, 0xAD, 0xBE, 0xEF][..], &v3[..]].concat());
    // Wrong version (v2).
    out.push([&magic[..], &2u32.to_be_bytes()[..]].concat());

    out
}

/// Minimal UserStore over the test DB.
struct DbStore(Arc<Database>);
impl protocol::UserStore for DbStore {
    fn lookup(&self, username: &str) -> Result<Option<Verifier>, String> {
        self.0.get_scram_verifier(username).map_err(|e| e.to_string())
    }
}
