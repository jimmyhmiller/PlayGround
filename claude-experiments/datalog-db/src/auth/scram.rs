//! SCRAM-SHA-256 (RFC 5802 / RFC 7677) — the pure cryptographic core.
//!
//! This module knows nothing about sockets, TLS, or our wire framing. It
//! implements the key derivations and the four SASL messages as functions over
//! byte slices and strings, plus a small server- and client-side state machine
//! that the protocol layer drives by feeding it message bytes. That separation
//! keeps the crypto independently testable against the published RFC vectors.
//!
//! ## What SCRAM gives us
//!
//! The password never crosses the wire. The server stores only a *verifier*
//! (`salt`, `iterations`, `StoredKey`, `ServerKey`) — never the password and
//! never `ClientKey`/`SaltedPassword`. A stolen verifier does not let an
//! attacker authenticate without also mounting an offline PBKDF2 attack, and
//! the mutual `ServerSignature` step means a server that lacks `ServerKey`
//! cannot impersonate the real one.
//!
//! ## Channel binding
//!
//! We run SCRAM *inside* a TLS tunnel and do **not** implement
//! `tls-server-end-point` channel binding. The client therefore advertises the
//! GS2 header `n,,` ("I do not support channel binding"), whose base64 is the
//! literal `biws`. The server enforces `c=biws` in the client-final message:
//! that is the standard downgrade protection and is cheap, so we do it even
//! though TLS already protects the channel.
//!
//! ## Deliberate simplification (v1)
//!
//! We do **not** apply SASLprep (RFC 4013) to the password — it is treated as
//! raw UTF-8 bytes. This is internally consistent as long as the same
//! (non-)normalization is applied at user creation and at authentication, which
//! it is (both go through [`Verifier::create`]). It only matters for non-ASCII
//! passwords; if PostgreSQL wire-interop with such passwords is ever needed,
//! add a `stringprep` pass in exactly one place. The `n=` username field *is*
//! escaped per the ABNF (`=` → `=3D`, `,` → `=2C`).

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// Output size of SHA-256 / HMAC-SHA-256, in bytes.
const HASH_LEN: usize = 32;

/// RFC 7677 mandates a minimum of 4096 PBKDF2 iterations for SCRAM-SHA-256.
/// A verifier with fewer is rejected on load — it is not safe to honor.
pub const MIN_ITERATIONS: u32 = 4096;

/// Default iteration count for newly created verifiers. Well above the RFC
/// floor; stored per-user so it can be raised later without breaking existing
/// users.
pub const DEFAULT_ITERATIONS: u32 = 15_000;

/// Upper bound on the iteration count a CLIENT will honor from a server-supplied
/// server-first message. Without this, a malicious or MITM'd server could send
/// `i=4294967295` and pin the client's CPU for hours computing PBKDF2 (the
/// client must compute it before it can even verify the server's signature).
/// Generous enough for any legitimate policy; refuses absurd costs.
pub const MAX_ITERATIONS: u32 = 1_000_000;

/// Salt length in bytes for new verifiers.
const SALT_LEN: usize = 16;

/// Minimum random bytes behind a nonce (base64-encoded, no padding). 18 bytes
/// → 24 base64 chars, comfortably unique and free of the `,` delimiter.
const NONCE_RAND_LEN: usize = 18;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScramError {
    #[error("malformed SCRAM message: {0}")]
    Malformed(String),
    #[error("unsupported channel binding (expected 'n,,' / c=biws)")]
    ChannelBinding,
    #[error("server nonce does not extend client nonce")]
    NonceMismatch,
    #[error("authentication failed")]
    AuthFailed,
    #[error("server signature verification failed")]
    ServerSignature,
    #[error("iteration count {0} below the RFC 7677 minimum of {min}", min = MIN_ITERATIONS)]
    WeakIterations(u32),
    #[error("invalid base64 in SCRAM message")]
    Base64,
}

type Result<T> = std::result::Result<T, ScramError>;

// ---------------------------------------------------------------------------
// Key derivations (RFC 5802 §3)
// ---------------------------------------------------------------------------

fn hmac(key: &[u8], msg: &[u8]) -> [u8; HASH_LEN] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(msg);
    mac.finalize().into_bytes().into()
}

fn h(data: &[u8]) -> [u8; HASH_LEN] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn xor(a: &[u8; HASH_LEN], b: &[u8; HASH_LEN]) -> [u8; HASH_LEN] {
    let mut out = [0u8; HASH_LEN];
    for i in 0..HASH_LEN {
        out[i] = a[i] ^ b[i];
    }
    out
}

/// `SaltedPassword = PBKDF2(HMAC-SHA-256, password, salt, i, dkLen=HASH_LEN)`.
fn salted_password(password: &[u8], salt: &[u8], iterations: u32) -> [u8; HASH_LEN] {
    let mut out = [0u8; HASH_LEN];
    pbkdf2::pbkdf2_hmac::<Sha256>(password, salt, iterations, &mut out);
    out
}

/// The keys derived from a salted password. `client_key`/`server_key` are
/// transient (used during a handshake); only `stored_key` (= H(client_key)) and
/// `server_key` are persisted in the verifier.
struct DerivedKeys {
    client_key: [u8; HASH_LEN],
    stored_key: [u8; HASH_LEN],
    server_key: [u8; HASH_LEN],
}

fn derive_keys(salted: &[u8; HASH_LEN]) -> DerivedKeys {
    let client_key = hmac(salted, b"Client Key");
    let stored_key = h(&client_key);
    let server_key = hmac(salted, b"Server Key");
    DerivedKeys {
        client_key,
        stored_key,
        server_key,
    }
}

// ---------------------------------------------------------------------------
// Verifier — what the server stores per user
// ---------------------------------------------------------------------------

/// A stored SCRAM verifier. Persisted (base64 fields) in the DB meta keyspace.
/// Contains no password-equivalent secret beyond what SCRAM requires.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct Verifier {
    /// Struct version, for forward migration of this record's shape.
    pub version: u8,
    pub iterations: u32,
    /// base64(salt)
    pub salt: String,
    /// base64(StoredKey = H(ClientKey))
    pub stored_key: String,
    /// base64(ServerKey = HMAC(SaltedPassword, "Server Key"))
    pub server_key: String,
}

impl Verifier {
    /// Compute a fresh verifier for `password` using a random salt and the
    /// default iteration count. `rand_bytes` fills the salt — pass a CSPRNG
    /// (see [`fill_random`]); injectable so tests are deterministic.
    pub fn create(password: &str) -> Verifier {
        let mut salt = [0u8; SALT_LEN];
        fill_random(&mut salt);
        Verifier::create_with(password, &salt, DEFAULT_ITERATIONS)
    }

    /// Deterministic constructor used by `create` and by tests/RFC vectors.
    pub fn create_with(password: &str, salt: &[u8], iterations: u32) -> Verifier {
        let salted = salted_password(password.as_bytes(), salt, iterations);
        let keys = derive_keys(&salted);
        Verifier {
            version: 1,
            iterations,
            salt: B64.encode(salt),
            stored_key: B64.encode(keys.stored_key),
            server_key: B64.encode(keys.server_key),
        }
    }

    fn stored_key_bytes(&self) -> Result<[u8; HASH_LEN]> {
        decode_fixed(&self.stored_key)
    }
    fn server_key_bytes(&self) -> Result<[u8; HASH_LEN]> {
        decode_fixed(&self.server_key)
    }

    /// Reject a verifier whose iteration count is below the RFC floor — honoring
    /// it would silently weaken every login for that user.
    pub fn validate(&self) -> Result<()> {
        if self.iterations < MIN_ITERATIONS {
            return Err(ScramError::WeakIterations(self.iterations));
        }
        Ok(())
    }
}

fn decode_fixed(s: &str) -> Result<[u8; HASH_LEN]> {
    let v = B64.decode(s).map_err(|_| ScramError::Base64)?;
    let arr: [u8; HASH_LEN] = v.as_slice().try_into().map_err(|_| ScramError::Base64)?;
    Ok(arr)
}

// ---------------------------------------------------------------------------
// Username escaping (RFC 5802 §5.1: the `n=` value)
// ---------------------------------------------------------------------------

/// Escape a username for the `n=` field: `=` → `=3D`, `,` → `=2C`.
pub fn escape_username(name: &str) -> String {
    name.replace('=', "=3D").replace(',', "=2C")
}

/// Reverse [`escape_username`]. Order matters: decode `=2C`/`=3D` in one pass so
/// a literal `=3D` in the source can't be mis-decoded.
pub fn unescape_username(s: &str) -> Result<String> {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'=' {
            match s.get(i..i + 3) {
                Some("=3D") => {
                    out.push('=');
                    i += 3;
                }
                Some("=2C") => {
                    out.push(',');
                    i += 3;
                }
                _ => return Err(ScramError::Malformed("bad =XX escape in username".into())),
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Nonce / randomness
// ---------------------------------------------------------------------------

/// Fill `buf` from the OS CSPRNG. Panics only if the OS RNG is unavailable,
/// which on this platform does not happen in practice.
pub fn fill_random(buf: &mut [u8]) {
    getrandom::getrandom(buf).expect("OS CSPRNG unavailable");
}

/// A fresh nonce: base64 (no padding, so it contains no `=` and never the `,`
/// delimiter) of `NONCE_RAND_LEN` random bytes.
pub fn make_nonce() -> String {
    let mut raw = [0u8; NONCE_RAND_LEN];
    fill_random(&mut raw);
    base64::engine::general_purpose::STANDARD_NO_PAD.encode(raw)
}

// ---------------------------------------------------------------------------
// Message parsing helpers
// ---------------------------------------------------------------------------

/// Split a SCRAM message `a=...,b=...,c=...` into its `key=value` attributes.
/// Values may themselves contain `=` (e.g. base64 padding), so we split each
/// comma-separated field only at the *first* `=`.
///
/// **Duplicate attribute keys are rejected.** Otherwise an attacker could smuggle
/// a second `c=`/`r=`/`p=` and exploit the disagreement between attribute lookup
/// (which would otherwise take the *first*) and the AuthMessage construction in
/// `server_final` (which splits at the *last* `,p=`). Rejecting duplicates makes
/// the parse unambiguous so those two views can never diverge.
fn parse_attrs(msg: &str) -> Result<Vec<(char, String)>> {
    let mut out: Vec<(char, String)> = Vec::new();
    for field in msg.split(',') {
        if field.is_empty() {
            continue;
        }
        let mut chars = field.chars();
        let key = chars
            .next()
            .ok_or_else(|| ScramError::Malformed("empty attribute".into()))?;
        let rest = chars.as_str();
        let val = rest
            .strip_prefix('=')
            .ok_or_else(|| ScramError::Malformed(format!("attribute '{key}' missing '='")))?;
        if out.iter().any(|(k, _)| *k == key) {
            return Err(ScramError::Malformed(format!("duplicate attribute '{key}'")));
        }
        out.push((key, val.to_string()));
    }
    Ok(out)
}

fn attr<'a>(attrs: &'a [(char, String)], key: char) -> Result<&'a str> {
    attrs
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v.as_str())
        .ok_or_else(|| ScramError::Malformed(format!("missing attribute '{key}'")))
}

// ---------------------------------------------------------------------------
// Server-side state machine
// ---------------------------------------------------------------------------

/// Drives the server half of a SCRAM exchange. The protocol layer reads frames
/// from the socket and feeds them here; this returns the bytes to send back.
pub struct ScramServer {
    verifier: Verifier,
    /// Set once the verifier is a mock (unknown user); the proof will fail but
    /// the exchange runs to completion so the timing/shape matches a real user.
    is_mock: bool,
    // Carried between steps to build AuthMessage.
    client_first_bare: String,
    server_first: String,
    server_nonce: String,
    client_nonce: String,
}

impl ScramServer {
    /// Begin an exchange against a known user's `verifier`.
    pub fn new(verifier: Verifier) -> Self {
        Self {
            verifier,
            is_mock: false,
            client_first_bare: String::new(),
            server_first: String::new(),
            server_nonce: String::new(),
            client_nonce: String::new(),
        }
    }

    /// Begin an exchange for an *unknown* user, producing a mock verifier that
    /// is indistinguishable from a real one to a remote attacker.
    ///
    /// Critically, this derives the salt and keys **directly via HMAC** under a
    /// server-only `mock_key` — it does NOT run PBKDF2. A real user's handshake
    /// also runs no server-side PBKDF2 (the stored verifier already holds the
    /// derived keys), so both paths perform the same cheap HMAC work. An earlier
    /// version ran PBKDF2 here, which made unknown-user logins measurably slower
    /// than known-user ones — i.e. the anti-enumeration mock was itself a timing
    /// oracle. Deriving keys without PBKDF2 closes that.
    ///
    /// `iterations` is reported as [`DEFAULT_ITERATIONS`] so the server-first
    /// message looks identical to a real user's. The keys are unpredictable to
    /// anyone without `mock_key`, so the client proof can never verify.
    pub fn mock(username: &str, mock_key: &[u8]) -> Self {
        // Salt: stable per username (like a real user's stored salt), domain-
        // separated from the key derivations below.
        let salt = hmac(mock_key, &[b"mock-salt:", username.as_bytes()].concat());
        // StoredKey/ServerKey derived directly from mock_key+username — no
        // PBKDF2. Distinct domain tags so the two keys differ.
        let stored_key = hmac(mock_key, &[b"mock-stored:", username.as_bytes()].concat());
        let server_key = hmac(mock_key, &[b"mock-server:", username.as_bytes()].concat());
        let verifier = Verifier {
            version: 1,
            iterations: DEFAULT_ITERATIONS,
            salt: B64.encode(&salt[..SALT_LEN]),
            stored_key: B64.encode(stored_key),
            server_key: B64.encode(server_key),
        };
        let mut s = Self::new(verifier);
        s.is_mock = true;
        s
    }

    /// Process `client-first-message` (`n,,n=user,r=cnonce` or a `y,,`/no-CB
    /// variant) and produce `server-first-message`. `*_nonce` randomness comes
    /// from [`make_nonce`].
    pub fn server_first(&mut self, client_first: &str) -> Result<String> {
        // GS2 header: we accept only the no-channel-binding forms. The header is
        // the part up to the second comma: "n,," or "y,," (no `a=` authzid).
        // A client demanding CB ("p=...") is rejected.
        let (gs2, bare) = split_gs2(client_first)?;
        if gs2.starts_with("p=") {
            return Err(ScramError::ChannelBinding);
        }
        self.client_first_bare = bare.to_string();

        let attrs = parse_attrs(bare)?;
        self.client_nonce = attr(&attrs, 'r')?.to_string();

        self.server_nonce = make_nonce();
        let combined = format!("{}{}", self.client_nonce, self.server_nonce);
        let salt_b64 = &self.verifier.salt;
        self.server_first = format!("r={combined},s={salt_b64},i={}", self.verifier.iterations);
        Ok(self.server_first.clone())
    }

    /// Process `client-final-message` (`c=biws,r=...,p=proof`). On success
    /// returns `(server-final = "v=...", authenticated_ok)`. `authenticated_ok`
    /// is false for a mock user or a bad proof — callers MUST treat false as a
    /// failure; the `v=` is still well-formed so as not to leak which it was.
    pub fn server_final(&mut self, client_final: &str) -> Result<(String, bool)> {
        let attrs = parse_attrs(client_final)?;

        // Downgrade protection: channel-binding data MUST be exactly base64("n,,")
        // = "biws", matching the header the client sent in client-first.
        if attr(&attrs, 'c')? != "biws" {
            return Err(ScramError::ChannelBinding);
        }

        // The client must echo our combined nonce verbatim.
        let combined = format!("{}{}", self.client_nonce, self.server_nonce);
        if attr(&attrs, 'r')? != combined {
            return Err(ScramError::NonceMismatch);
        }

        let proof_b64 = attr(&attrs, 'p')?;
        let proof = B64.decode(proof_b64).map_err(|_| ScramError::Base64)?;
        let proof: [u8; HASH_LEN] = proof.as_slice().try_into().map_err(|_| ScramError::Base64)?;

        // client-final-without-proof = everything up to ",p="
        let cfwp = client_final
            .rsplit_once(",p=")
            .map(|(head, _)| head)
            .ok_or_else(|| ScramError::Malformed("client-final missing ,p=".into()))?;

        let auth_message =
            format!("{},{},{}", self.client_first_bare, self.server_first, cfwp);

        let stored_key = self.verifier.stored_key_bytes()?;
        let client_sig = hmac(&stored_key, auth_message.as_bytes());
        // RecoveredClientKey = ClientProof XOR ClientSignature
        let recovered = xor(&proof, &client_sig);
        // Auth holds iff H(RecoveredClientKey) == StoredKey, in constant time.
        let recovered_stored = h(&recovered);
        let ok = recovered_stored.ct_eq(&stored_key).into() && !self.is_mock;

        // ServerSignature is always computed honestly so a *real* user who
        // authenticated gets a verifiable mutual-auth signature. For a mock or
        // failed auth the value is still well-formed (computed over the mock
        // server_key), keeping the message shape uniform.
        let server_key = self.verifier.server_key_bytes()?;
        let server_sig = hmac(&server_key, auth_message.as_bytes());
        let server_final = format!("v={}", B64.encode(server_sig));

        Ok((server_final, ok))
    }
}

/// Split a client-first-message into its GS2 header and the client-first-bare.
/// GS2 header is the first two comma-separated fields (cbind-flag, optional
/// authzid). Returns `(gs2_header, bare)`.
fn split_gs2(client_first: &str) -> Result<(&str, &str)> {
    // cbind-flag is one of: "n", "y", or "p=name". authzid is "" or "a=...".
    // So the header ends after the second comma.
    let mut comma_idxs = client_first.match_indices(',');
    let _first = comma_idxs
        .next()
        .ok_or_else(|| ScramError::Malformed("client-first missing GS2 header".into()))?;
    let second = comma_idxs
        .next()
        .ok_or_else(|| ScramError::Malformed("client-first missing GS2 header".into()))?;
    let header_end = second.0 + 1; // include the second comma
    let gs2 = &client_first[..header_end];
    let bare = &client_first[header_end..];
    Ok((gs2, bare))
}

// ---------------------------------------------------------------------------
// Client-side state machine
// ---------------------------------------------------------------------------

/// Drives the client half of a SCRAM exchange.
pub struct ScramClient {
    username: String,
    password: String,
    client_nonce: String,
    client_first_bare: String,
    // Filled after server-first is processed.
    auth_message: String,
    server_key: Option<[u8; HASH_LEN]>,
}

impl ScramClient {
    pub fn new(username: &str, password: &str) -> Self {
        Self {
            username: username.to_string(),
            password: password.to_string(),
            client_nonce: String::new(),
            client_first_bare: String::new(),
            auth_message: String::new(),
            server_key: None,
        }
    }

    /// Produce `client-first-message` (`n,,n=user,r=cnonce`).
    pub fn client_first(&mut self) -> String {
        self.client_nonce = make_nonce();
        self.client_first_bare = format!(
            "n={},r={}",
            escape_username(&self.username),
            self.client_nonce
        );
        format!("n,,{}", self.client_first_bare)
    }

    /// Given `server-first-message`, produce `client-final-message`.
    pub fn client_final(&mut self, server_first: &str) -> Result<String> {
        let attrs = parse_attrs(server_first)?;
        let combined = attr(&attrs, 'r')?.to_string();
        if !combined.starts_with(&self.client_nonce) {
            return Err(ScramError::NonceMismatch);
        }
        let salt = B64.decode(attr(&attrs, 's')?).map_err(|_| ScramError::Base64)?;
        let iterations: u32 = attr(&attrs, 'i')?
            .parse()
            .map_err(|_| ScramError::Malformed("bad iteration count".into()))?;
        // Bound the server-supplied cost on BOTH ends. Below the floor weakens
        // our own hashing; above the ceiling lets a rogue server burn our CPU
        // (we compute PBKDF2 here, before we can verify the server is genuine).
        if iterations < MIN_ITERATIONS || iterations > MAX_ITERATIONS {
            return Err(ScramError::WeakIterations(iterations));
        }

        let salted = salted_password(self.password.as_bytes(), &salt, iterations);
        let keys = derive_keys(&salted);

        // client-final-without-proof, c=biws (base64 of the "n,," GS2 header).
        let cfwp = format!("c=biws,r={combined}");
        self.auth_message =
            format!("{},{},{}", self.client_first_bare, server_first, cfwp);

        let client_sig = hmac(&keys.stored_key, self.auth_message.as_bytes());
        let proof = xor(&keys.client_key, &client_sig);
        self.server_key = Some(keys.server_key);

        Ok(format!("{cfwp},p={}", B64.encode(proof)))
    }

    /// Verify the server's `server-final-message` (`v=ServerSignature`). This is
    /// mutual authentication: a server that does not know `ServerKey` cannot
    /// produce a matching signature. MUST be called before trusting the server.
    pub fn verify_server_final(&self, server_final: &str) -> Result<()> {
        let attrs = parse_attrs(server_final)?;
        if let Ok(err) = attr(&attrs, 'e') {
            return Err(ScramError::Malformed(format!("server error: {err}")));
        }
        let sig = B64
            .decode(attr(&attrs, 'v')?)
            .map_err(|_| ScramError::Base64)?;
        let expected = hmac(
            self.server_key.as_ref().ok_or(ScramError::ServerSignature)?,
            self.auth_message.as_bytes(),
        );
        if sig.ct_eq(&expected).into() {
            Ok(())
        } else {
            Err(ScramError::ServerSignature)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC 7677 §3 worked example: user "user", password "pencil",
    // salt = base64 "W22ZaJ0SNY7soEsUEjb6gQ==", iterations 4096.
    // Expected ClientProof / ServerSignature are published there.
    const RFC_SALT_B64: &str = "W22ZaJ0SNY7soEsUEjb6gQ==";
    const RFC_CLIENT_NONCE: &str = "rOprNGfwEbeRWgbNEkqO";
    const RFC_SERVER_NONCE: &str = "%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0";
    const RFC_EXPECTED_PROOF: &str = "dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ=";
    const RFC_EXPECTED_SERVER_SIG: &str = "6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4=";

    #[test]
    fn rfc7677_vector_proof_and_server_signature() {
        let salt = B64.decode(RFC_SALT_B64).unwrap();
        let verifier = Verifier::create_with("pencil", &salt, 4096);

        // Reconstruct the exact RFC messages with fixed nonces.
        let client_first_bare = format!("n=user,r={RFC_CLIENT_NONCE}");
        let combined = format!("{RFC_CLIENT_NONCE}{RFC_SERVER_NONCE}");
        let server_first = format!("r={combined},s={RFC_SALT_B64},i=4096");
        let cfwp = format!("c=biws,r={combined}");
        let auth_message = format!("{client_first_bare},{server_first},{cfwp}");

        // Client side: compute proof.
        let salted = salted_password(b"pencil", &salt, 4096);
        let keys = derive_keys(&salted);
        let client_sig = hmac(&keys.stored_key, auth_message.as_bytes());
        let proof = xor(&keys.client_key, &client_sig);
        assert_eq!(B64.encode(proof), RFC_EXPECTED_PROOF, "ClientProof mismatch");

        // Server side: verify the proof against the stored verifier.
        let stored_key = verifier.stored_key_bytes().unwrap();
        let client_sig2 = hmac(&stored_key, auth_message.as_bytes());
        let recovered = xor(&proof, &client_sig2);
        assert_eq!(h(&recovered), stored_key, "server proof verification");

        // ServerSignature.
        let server_key = verifier.server_key_bytes().unwrap();
        let server_sig = hmac(&server_key, auth_message.as_bytes());
        assert_eq!(
            B64.encode(server_sig),
            RFC_EXPECTED_SERVER_SIG,
            "ServerSignature mismatch"
        );
    }

    #[test]
    fn full_exchange_success() {
        let verifier = Verifier::create("hunter2");
        let mut client = ScramClient::new("alice", "hunter2");
        let mut server = ScramServer::new(verifier);

        let cf = client.client_first();
        let sf = server.server_first(&cf).unwrap();
        let cfin = client.client_final(&sf).unwrap();
        let (sfin, ok) = server.server_final(&cfin).unwrap();
        assert!(ok, "valid password should authenticate");
        client.verify_server_final(&sfin).unwrap();
    }

    #[test]
    fn wrong_password_fails_but_completes() {
        let verifier = Verifier::create("correct-horse");
        let mut client = ScramClient::new("alice", "wrong-password");
        let mut server = ScramServer::new(verifier);

        let cf = client.client_first();
        let sf = server.server_first(&cf).unwrap();
        let cfin = client.client_final(&sf).unwrap();
        let (_sfin, ok) = server.server_final(&cfin).unwrap();
        assert!(!ok, "wrong password must not authenticate");
    }

    #[test]
    fn mock_user_never_authenticates() {
        let mock_key = b"server-only-secret";
        let mut client = ScramClient::new("ghost", "anything");
        let mut server = ScramServer::mock("ghost", mock_key);

        let cf = client.client_first();
        let sf = server.server_first(&cf).unwrap();
        let cfin = client.client_final(&sf).unwrap();
        let (_sfin, ok) = server.server_final(&cfin).unwrap();
        assert!(!ok, "mock user must never authenticate");
    }

    #[test]
    fn mock_salt_is_stable_per_username() {
        // Enumeration resistance: the same unknown username yields the same
        // server-first salt across attempts (like a real user would).
        let mock_key = b"server-only-secret";
        let s1 = ScramServer::mock("ghost", mock_key).verifier.salt;
        let s2 = ScramServer::mock("ghost", mock_key).verifier.salt;
        let s3 = ScramServer::mock("other", mock_key).verifier.salt;
        assert_eq!(s1, s2, "same username → same mock salt");
        assert_ne!(s1, s3, "different username → different mock salt");
    }

    #[test]
    fn mock_runs_no_pbkdf2_and_matches_real_message_shape() {
        // C1 regression: the mock path must NOT run PBKDF2 (it derives keys via
        // HMAC), and its server-first must look like a real user's (same iters,
        // base64 salt of the same length). We can't time here, but we can assert
        // the structural equivalence that makes the timing equal.
        let mut mock = ScramServer::mock("ghost", b"k");
        let mut real = ScramServer::new(Verifier::create("pw"));
        let sf_mock = mock.server_first("n,,n=ghost,r=abc").unwrap();
        let sf_real = real.server_first("n,,n=alice,r=abc").unwrap();
        // Both advertise DEFAULT_ITERATIONS and a 16-byte (base64) salt.
        assert!(sf_mock.contains(&format!("i={DEFAULT_ITERATIONS}")));
        assert!(sf_real.contains(&format!("i={DEFAULT_ITERATIONS}")));
        let salt_len = |sf: &str| {
            B64.decode(sf.split(',').find(|f| f.starts_with("s=")).unwrap().strip_prefix("s=").unwrap())
                .unwrap()
                .len()
        };
        assert_eq!(salt_len(&sf_mock), SALT_LEN);
        assert_eq!(salt_len(&sf_real), SALT_LEN);
    }

    #[test]
    fn client_rejects_out_of_range_iterations() {
        // C2 regression: a malicious server-first with an absurd or too-low `i`
        // must be refused before the client runs PBKDF2.
        let mut client = ScramClient::new("alice", "pw");
        let _cf = client.client_first();
        let nonce = &client.client_nonce.clone();
        // Too high.
        let sf_high = format!("r={nonce}srv,s={},i={}", B64.encode(b"saltsaltsaltsalt"), MAX_ITERATIONS + 1);
        assert_eq!(
            client.client_final(&sf_high).unwrap_err(),
            ScramError::WeakIterations(MAX_ITERATIONS + 1)
        );
        // Too low.
        let sf_low = format!("r={nonce}srv,s={},i=100", B64.encode(b"saltsaltsaltsalt"));
        assert_eq!(client.client_final(&sf_low).unwrap_err(), ScramError::WeakIterations(100));
    }

    #[test]
    fn duplicate_attributes_rejected() {
        // H2 regression: a message with a duplicate key must be refused so the
        // first-match lookup can't disagree with the AuthMessage split.
        assert!(parse_attrs("r=a,r=b").is_err());
        assert!(parse_attrs("c=biws,r=x,p=y,p=z").is_err());
        // A normal message still parses.
        assert!(parse_attrs("c=biws,r=x,p=y").is_ok());
    }

    #[test]
    fn tampered_nonce_rejected() {
        let verifier = Verifier::create("pw");
        let mut client = ScramClient::new("alice", "pw");
        let mut server = ScramServer::new(verifier);
        let cf = client.client_first();
        let _sf = server.server_first(&cf).unwrap();
        // Forge a client-final with a nonce that doesn't match.
        let forged = "c=biws,r=totally-wrong-nonce,p=AAAA";
        let err = server.server_final(forged).unwrap_err();
        assert_eq!(err, ScramError::NonceMismatch);
    }

    #[test]
    fn channel_binding_downgrade_rejected() {
        let verifier = Verifier::create("pw");
        let mut client = ScramClient::new("alice", "pw");
        let mut server = ScramServer::new(verifier);
        let cf = client.client_first();
        let sf = server.server_first(&cf).unwrap();
        // Client sends a c= that isn't biws.
        let combined = sf
            .split(',')
            .next()
            .unwrap()
            .strip_prefix("r=")
            .unwrap();
        let bad = format!("c=YmFk,r={combined},p=AAAA");
        let err = server.server_final(&bad).unwrap_err();
        assert_eq!(err, ScramError::ChannelBinding);
    }

    #[test]
    fn client_rejects_bad_server_signature() {
        let verifier = Verifier::create("pw");
        let mut client = ScramClient::new("alice", "pw");
        let mut server = ScramServer::new(verifier);
        let cf = client.client_first();
        let sf = server.server_first(&cf).unwrap();
        let _cfin = client.client_final(&sf).unwrap();
        // A forged server-final with a wrong v=.
        let forged = format!("v={}", B64.encode([0u8; HASH_LEN]));
        assert_eq!(
            client.verify_server_final(&forged).unwrap_err(),
            ScramError::ServerSignature
        );
    }

    #[test]
    fn username_escaping_roundtrip() {
        for name in ["alice", "a=b", "a,b", "a=,=b", "=2C=3D"] {
            let esc = escape_username(name);
            assert!(!esc.contains(','));
            assert_eq!(unescape_username(&esc).unwrap(), name);
        }
    }

    #[test]
    fn weak_iterations_rejected() {
        let mut v = Verifier::create_with("pw", b"saltsaltsaltsalt", 4096);
        v.validate().unwrap();
        v.iterations = 1000;
        assert_eq!(v.validate().unwrap_err(), ScramError::WeakIterations(1000));
    }

    #[test]
    fn verifier_serde_roundtrip() {
        let v = Verifier::create("pw");
        let json = serde_json::to_vec(&v).unwrap();
        let back: Verifier = serde_json::from_slice(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn nonces_are_unique_and_comma_free() {
        let a = make_nonce();
        let b = make_nonce();
        assert_ne!(a, b);
        assert!(!a.contains(','));
        assert!(!a.contains('='));
    }
}
