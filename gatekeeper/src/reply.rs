//! A uniform in-memory reply, so every handler (static, proxy, auth-deny) can
//! produce the same type and a single function emits it. Bodies are buffered in
//! memory — fine for a personal gateway; not built for huge streaming payloads.

pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        Reply {
            status,
            headers: Vec::new(),
            body,
        }
    }

    /// A bare status with a short text body.
    pub fn status(status: u16, msg: &str) -> Self {
        Reply::new(status, msg.as_bytes().to_vec())
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    /// Convert to a tiny_http response and send it on the given request.
    pub fn respond(self, request: tiny_http::Request) -> std::io::Result<()> {
        let mut resp = tiny_http::Response::from_data(self.body)
            .with_status_code(tiny_http::StatusCode(self.status));
        for (name, value) in &self.headers {
            if let Ok(h) = tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes()) {
                resp = resp.with_header(h);
            }
        }
        request.respond(resp)
    }
}
