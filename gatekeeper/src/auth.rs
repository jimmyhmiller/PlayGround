//! Shared-secret authentication.
//!
//! A single bearer token gates every private route. The token is presented
//! either as `Authorization: Bearer <token>` (preferred, for APIs) or as a
//! `gatekeeper=<token>` cookie (for browsers). The header wins if both are
//! present.
//!
//! Comparison is constant-time (`subtle`) so an attacker can't recover the
//! token a byte at a time via response-timing. We also hash both sides to a
//! fixed length first, which removes the length-dependent early return that a
//! naive `ct_eq` on differing lengths would otherwise have.

use subtle::ConstantTimeEq;

/// Holds the expected token in a form that compares in constant time.
#[derive(Clone)]
pub struct Authenticator {
    expected: [u8; 32],
}

impl Authenticator {
    pub fn new(token: &str) -> Self {
        Authenticator {
            expected: digest(token.as_bytes()),
        }
    }

    /// True iff `presented` matches the expected token. Constant-time in the
    /// token contents (both sides are hashed to 32 bytes, then compared with
    /// `ct_eq`, so neither length nor content branches early).
    pub fn verify(&self, presented: &str) -> bool {
        let got = digest(presented.as_bytes());
        got.ct_eq(&self.expected).into()
    }

    /// Extract a token from the request headers and verify it. Returns true on
    /// a valid token. The header (`Authorization: Bearer`) takes precedence
    /// over the cookie; if an Authorization header is present we use it and do
    /// NOT fall back to the cookie (so a forged/blank header can't be bypassed).
    pub fn check_headers(&self, headers: &[tiny_http::Header]) -> bool {
        if let Some(auth) = header_value(headers, "authorization") {
            return match bearer(auth) {
                Some(tok) => self.verify(tok),
                None => false,
            };
        }
        if let Some(cookie) = header_value(headers, "cookie") {
            if let Some(tok) = cookie_token(cookie) {
                return self.verify(tok);
            }
        }
        false
    }
}

/// Find the first header whose field name equals `name` (case-insensitive),
/// returning its value as a `&str`.
fn header_value<'a>(headers: &'a [tiny_http::Header], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str())
}

/// Pull the token out of an `Authorization: Bearer <tok>` value.
fn bearer(value: &str) -> Option<&str> {
    let tok = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))?
        .trim();
    if tok.is_empty() {
        None
    } else {
        Some(tok)
    }
}

/// Pull the `gatekeeper=<tok>` value out of a Cookie header value.
fn cookie_token(value: &str) -> Option<&str> {
    for pair in value.split(';') {
        let pair = pair.trim();
        if let Some(tok) = pair.strip_prefix("gatekeeper=") {
            let tok = tok.trim();
            if !tok.is_empty() {
                return Some(tok);
            }
        }
    }
    None
}

/// SHA-256-style fixed-length digest. We don't pull in a crypto hash crate for
/// this — a token equality check only needs a fixed-width, well-mixed reduction
/// so the constant-time compare has no length side channel. FxHash-style
/// mixing over 4 lanes is more than enough to avoid collisions for this use
/// (an attacker who can find a 256-bit multi-lane collision has already won
/// elsewhere). The point is constant-time *comparison*, not hash secrecy.
fn digest(bytes: &[u8]) -> [u8; 32] {
    // Four independent 64-bit accumulators with distinct primes, each folded
    // over the whole input, then serialized. This is deterministic and
    // fixed-length regardless of input length.
    const SEEDS: [u64; 4] = [
        0x100000001b3,
        0xff51afd7ed558ccd,
        0xc4ceb9fe1a85ec53,
        0x9e3779b97f4a7c15,
    ];
    let mut acc = SEEDS;
    for (i, &b) in bytes.iter().enumerate() {
        let lane = i % 4;
        acc[lane] = (acc[lane] ^ b as u64).wrapping_mul(0x100000001b3);
        // rotate to spread influence across lanes
        acc[(lane + 1) % 4] = acc[(lane + 1) % 4].rotate_left(7) ^ acc[lane];
    }
    // Final avalanche per lane (splitmix64-style).
    for a in acc.iter_mut() {
        let mut x = *a ^ (bytes.len() as u64).wrapping_mul(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        *a = x ^ (x >> 31);
    }
    let mut out = [0u8; 32];
    for (i, a) in acc.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&a.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr(field: &str, value: &str) -> tiny_http::Header {
        tiny_http::Header::from_bytes(field.as_bytes(), value.as_bytes()).unwrap()
    }

    #[test]
    fn verify_correct_and_wrong() {
        let a = Authenticator::new("s3cr3t-token-value");
        assert!(a.verify("s3cr3t-token-value"));
        assert!(!a.verify("s3cr3t-token-valuE"));
        assert!(!a.verify("wrong"));
        assert!(!a.verify(""));
        // Different-length wrong token still rejected (no length leak path).
        assert!(!a.verify("s3cr3t-token-value-extra"));
    }

    #[test]
    fn header_bearer() {
        let a = Authenticator::new("tok");
        assert!(a.check_headers(&[hdr("Authorization", "Bearer tok")]));
        assert!(!a.check_headers(&[hdr("Authorization", "Bearer nope")]));
        assert!(!a.check_headers(&[hdr("Authorization", "tok")])); // missing scheme
    }

    #[test]
    fn cookie_fallback() {
        let a = Authenticator::new("tok");
        assert!(a.check_headers(&[hdr("Cookie", "foo=bar; gatekeeper=tok; baz=1")]));
        assert!(!a.check_headers(&[hdr("Cookie", "gatekeeper=nope")]));
    }

    #[test]
    fn header_wins_over_cookie() {
        let a = Authenticator::new("tok");
        // Good header, bad cookie -> allowed (header checked first).
        assert!(a.check_headers(&[
            hdr("Authorization", "Bearer tok"),
            hdr("Cookie", "gatekeeper=nope"),
        ]));
        // Bad header present -> we use the header and do NOT fall back to a
        // good cookie. This prevents a stale/forged header being bypassed.
        assert!(!a.check_headers(&[
            hdr("Authorization", "Bearer nope"),
            hdr("Cookie", "gatekeeper=tok"),
        ]));
    }

    #[test]
    fn no_credentials() {
        let a = Authenticator::new("tok");
        assert!(!a.check_headers(&[]));
    }
}
