//! Transport layer: a stream that is either a plain TCP socket or a rustls TLS
//! stream, plus helpers to build rustls server/client configs.
//!
//! TLS here is **wrapped-socket** (RethinkDB-style): the whole stream is TLS
//! from byte zero, established before any protocol bytes. There is no in-band
//! STARTTLS negotiation — a connection is TLS or plaintext for its whole life,
//! decided by which constructor the peer uses. The [`crate::protocol`] handshake
//! and framing run unchanged inside the tunnel because they are generic over
//! `Read`/`Write`, which both variants implement.
//!
//! `set_nodelay`, `peer_addr`, and read/write timeouts must be set on the raw
//! [`TcpStream`] *before* wrapping — they are not exposed by the TLS stream and
//! rustls calls straight through to the underlying socket, so the socket
//! options keep applying.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::sync::Arc;
use std::sync::Once;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::{ClientConnection, ServerConnection, StreamOwned};

/// rustls 0.23 requires a process-wide default `CryptoProvider`. We install the
/// `ring` provider exactly once; a second call is a harmless no-op (rustls
/// returns Err if one is already installed, which we ignore).
static INSTALL_PROVIDER: Once = Once::new();

pub fn ensure_crypto_provider() {
    INSTALL_PROVIDER.call_once(|| {
        // Ignore the error: another component (or an earlier call) may have
        // installed a provider already, which is fine.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Server-side connection: plaintext or TLS. Implements `Read`/`Write` so the
/// protocol layer treats both uniformly.
pub enum ServerStream {
    Plain(TcpStream),
    Tls(Box<StreamOwned<ServerConnection, TcpStream>>),
}

/// Client-side connection: plaintext or TLS.
pub enum ClientStream {
    Plain(TcpStream),
    Tls(Box<StreamOwned<ClientConnection, TcpStream>>),
}

impl Read for ServerStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            ServerStream::Plain(s) => s.read(buf),
            ServerStream::Tls(s) => s.read(buf),
        }
    }
}
impl Write for ServerStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            ServerStream::Plain(s) => s.write(buf),
            ServerStream::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            ServerStream::Plain(s) => s.flush(),
            ServerStream::Tls(s) => s.flush(),
        }
    }
}

impl Read for ClientStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            ClientStream::Plain(s) => s.read(buf),
            ClientStream::Tls(s) => s.read(buf),
        }
    }
}
impl Write for ClientStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            ClientStream::Plain(s) => s.write(buf),
            ClientStream::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            ClientStream::Plain(s) => s.flush(),
            ClientStream::Tls(s) => s.flush(),
        }
    }
}

impl ServerStream {
    /// Wrap an accepted socket in TLS using `config`. The TLS handshake itself
    /// completes lazily on the first `read`/`write` (driven by the protocol
    /// handshake), so this does not block here.
    pub fn accept_tls(sock: TcpStream, config: Arc<rustls::ServerConfig>) -> io::Result<Self> {
        let conn = ServerConnection::new(config)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("tls: {e}")))?;
        Ok(ServerStream::Tls(Box::new(StreamOwned::new(conn, sock))))
    }
}

// ---------------------------------------------------------------------------
// Config builders
// ---------------------------------------------------------------------------

/// Build a server config from PEM cert chain + private key files. The cert file
/// may contain a chain (leaf first); the key is the first PKCS#8/SEC1/PKCS#1
/// key found.
pub fn server_config(
    cert_pem: &Path,
    key_pem: &Path,
) -> io::Result<Arc<rustls::ServerConfig>> {
    ensure_crypto_provider();

    let certs = load_certs(cert_pem)?;
    let key = load_key(key_pem)?;

    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("tls config: {e}")))?;
    Ok(Arc::new(config))
}

/// Build a client config that trusts the certificate(s) in `ca_pem`.
///
/// The PEM may be either a proper CA certificate (the server presents a leaf
/// chained to it) **or** a single self-signed server certificate, which is the
/// common convenience for an internal service. To support both — and because
/// webpki refuses to treat a self-signed leaf as a CA root
/// (`CaUsedAsEndEntity`) — we install a verifier that accepts a connection iff
/// the leaf the server presents *exactly equals* one of the trusted certs
/// (certificate pinning), falling back to standard webpki CA-chain validation
/// otherwise. Pinning is strictly stronger than CA trust: only the exact
/// pinned cert is accepted. The hostname (`server_name`) is still required to
/// match a SAN for the CA-chain path.
pub fn client_config(ca_pem: Option<&Path>) -> io::Result<Arc<rustls::ClientConfig>> {
    ensure_crypto_provider();

    let path = ca_pem.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "TLS client requires a CA/cert PEM (--ca) to trust the server certificate",
        )
    })?;
    let trusted = load_certs(path)?;

    // Try to build a webpki root store too, for the proper-CA-chain case.
    let mut roots = rustls::RootCertStore::empty();
    for cert in &trusted {
        // Ignore per-cert add errors (a self-signed leaf may not be a valid
        // CA root — the pinning path covers it).
        let _ = roots.add(cert.clone());
    }

    let verifier = Arc::new(PinningVerifier {
        pinned: trusted,
        webpki: if roots.is_empty() {
            None
        } else {
            rustls::client::WebPkiServerVerifier::builder(Arc::new(roots))
                .build()
                .ok()
        },
    });

    let config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_no_client_auth();
    Ok(Arc::new(config))
}

/// A `ServerCertVerifier` that accepts the server's leaf cert if it exactly
/// matches a pinned cert, otherwise defers to standard webpki CA validation.
#[derive(Debug)]
struct PinningVerifier {
    pinned: Vec<CertificateDer<'static>>,
    webpki: Option<Arc<rustls::client::WebPkiServerVerifier>>,
}

impl rustls::client::danger::ServerCertVerifier for PinningVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        // Exact-pin match: the presented leaf is byte-identical to a trusted
        // cert. This is the self-signed-server case.
        if self.pinned.iter().any(|c| c.as_ref() == end_entity.as_ref()) {
            return Ok(rustls::client::danger::ServerCertVerified::assertion());
        }
        // Otherwise require a valid CA chain to a trusted root.
        match &self.webpki {
            Some(v) => v
                .verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now),
            None => Err(rustls::Error::General(
                "server certificate does not match the pinned certificate".into(),
            )),
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        // Delegate signature checks to the default crypto provider's schemes.
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Establish a client-side TLS stream over `sock`, validating the server cert
/// against `config` for `server_name` (used for SNI and cert hostname check).
pub fn connect_tls(
    sock: TcpStream,
    config: Arc<rustls::ClientConfig>,
    server_name: &str,
) -> io::Result<ClientStream> {
    let name: ServerName<'static> = ServerName::try_from(server_name.to_string())
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid server name for TLS: {server_name:?}"),
            )
        })?;
    let conn = ClientConnection::new(config, name)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("tls: {e}")))?;
    Ok(ClientStream::Tls(Box::new(StreamOwned::new(conn, sock))))
}

fn load_certs(path: &Path) -> io::Result<Vec<CertificateDer<'static>>> {
    let data = std::fs::read(path)?;
    let mut reader = io::BufReader::new(&data[..]);
    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut reader)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if certs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no certificates found in {}", path.display()),
        ));
    }
    Ok(certs)
}

fn load_key(path: &Path) -> io::Result<PrivateKeyDer<'static>> {
    let data = std::fs::read(path)?;
    let mut reader = io::BufReader::new(&data[..]);
    rustls_pemfile::private_key(&mut reader)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no private key found in {}", path.display()),
        )
    })
}
