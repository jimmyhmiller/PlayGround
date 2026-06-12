//! Self-description for a gatekeeper function: declare your endpoints, their
//! query params, and examples, so the gate's `/describe` catalog can
//! show callers what your function does.
//!
//! You build a [`Description`] in a `#[describe]`-annotated function; the macro
//! emits the `gk_describe` ABI symbol that returns it as JSON. Everything here is
//! builder-style and serializes itself (no serde dependency — the SDK stays
//! light), producing the JSON shape documented on `gatekeeper_abi::GK_DESCRIBE_SYMBOL`.
//!
//! ```ignore
//! use gatekeeper_fn::{describe, Description, Endpoint, Param};
//!
//! #[describe]
//! fn describe() -> Description {
//!     Description::new("analytics", "Website-visit analytics")
//!         .endpoint(
//!             Endpoint::get("/timeline", "per-page views over time")
//!                 .param(Param::int("days", "window: last N days"))
//!                 .param(Param::int("n", "how many pages").default("10"))
//!                 .example("/timeline?days=7&n=3")
//!                 .returns("{ series: [{ path, label, points }] }"),
//!         )
//! }
//! ```

/// The whole function's description: a name, a one-line summary, and its
/// endpoints.
pub struct Description {
    name: String,
    summary: String,
    endpoints: Vec<Endpoint>,
}

impl Description {
    pub fn new(name: impl Into<String>, summary: impl Into<String>) -> Self {
        Description {
            name: name.into(),
            summary: summary.into(),
            endpoints: Vec::new(),
        }
    }

    /// Add an endpoint (builder style).
    pub fn endpoint(mut self, e: Endpoint) -> Self {
        self.endpoints.push(e);
        self
    }

    /// Serialize to the JSON the gate expects (see `GK_DESCRIBE_SYMBOL`).
    pub fn to_json(&self) -> String {
        let mut s = String::from("{");
        push_kv_str(&mut s, "name", &self.name);
        s.push(',');
        push_kv_str(&mut s, "summary", &self.summary);
        s.push_str(",\"endpoints\":[");
        for (i, e) in self.endpoints.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            e.write_json(&mut s);
        }
        s.push_str("]}");
        s
    }
}

/// One endpoint: a sub-path (relative to the function's route prefix), the HTTP
/// methods it accepts, a summary, its query params, and optional example +
/// return-shape sketch.
pub struct Endpoint {
    path: String,
    methods: Vec<String>,
    summary: String,
    params: Vec<Param>,
    example: Option<String>,
    returns: Option<String>,
}

impl Endpoint {
    /// An endpoint with explicit methods.
    pub fn new(path: impl Into<String>, summary: impl Into<String>, methods: &[&str]) -> Self {
        Endpoint {
            path: path.into(),
            methods: methods.iter().map(|m| m.to_string()).collect(),
            summary: summary.into(),
            params: Vec::new(),
            example: None,
            returns: None,
        }
    }
    /// A `GET` endpoint (the common case).
    pub fn get(path: impl Into<String>, summary: impl Into<String>) -> Self {
        Endpoint::new(path, summary, &["GET"])
    }
    /// A `POST` endpoint.
    pub fn post(path: impl Into<String>, summary: impl Into<String>) -> Self {
        Endpoint::new(path, summary, &["POST"])
    }
    /// Add a query param (builder style).
    pub fn param(mut self, p: Param) -> Self {
        self.params.push(p);
        self
    }
    /// Set an example request string (builder style).
    pub fn example(mut self, e: impl Into<String>) -> Self {
        self.example = Some(e.into());
        self
    }
    /// Sketch the response shape (builder style).
    pub fn returns(mut self, r: impl Into<String>) -> Self {
        self.returns = Some(r.into());
        self
    }

    fn write_json(&self, s: &mut String) {
        s.push('{');
        push_kv_str(s, "path", &self.path);
        s.push_str(",\"methods\":[");
        for (i, m) in self.methods.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            push_json_str(s, m);
        }
        s.push(']');
        s.push(',');
        push_kv_str(s, "summary", &self.summary);
        s.push_str(",\"params\":[");
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            p.write_json(s);
        }
        s.push(']');
        if let Some(e) = &self.example {
            s.push(',');
            push_kv_str(s, "example", e);
        }
        if let Some(r) = &self.returns {
            s.push(',');
            push_kv_str(s, "returns", r);
        }
        s.push('}');
    }
}

/// One query param: name, type, whether required, an optional default, and a
/// description.
pub struct Param {
    name: String,
    ty: String,
    required: bool,
    default: Option<String>,
    description: String,
}

impl Param {
    /// A param with an explicit type name (e.g. "int", "string", "epoch-ms").
    pub fn new(
        name: impl Into<String>,
        ty: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Param {
            name: name.into(),
            ty: ty.into(),
            required: false,
            default: None,
            description: description.into(),
        }
    }
    /// An integer param.
    pub fn int(name: impl Into<String>, description: impl Into<String>) -> Self {
        Param::new(name, "int", description)
    }
    /// A string param.
    pub fn string(name: impl Into<String>, description: impl Into<String>) -> Self {
        Param::new(name, "string", description)
    }
    /// Mark this param as required (builder style).
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }
    /// Give a default value shown in the catalog (builder style).
    pub fn default(mut self, d: impl Into<String>) -> Self {
        self.default = Some(d.into());
        self
    }

    fn write_json(&self, s: &mut String) {
        s.push('{');
        push_kv_str(s, "name", &self.name);
        s.push(',');
        push_kv_str(s, "type", &self.ty);
        s.push_str(",\"required\":");
        s.push_str(if self.required { "true" } else { "false" });
        s.push_str(",\"default\":");
        match &self.default {
            Some(d) => push_json_str(s, d),
            None => s.push_str("null"),
        }
        s.push(',');
        push_kv_str(s, "description", &self.description);
        s.push('}');
    }
}

// ---- minimal JSON string writing (no serde) ----

fn push_kv_str(s: &mut String, key: &str, value: &str) {
    push_json_str(s, key);
    s.push(':');
    push_json_str(s, value);
}

/// Write a JSON string literal (with quotes) for `v`, escaping per RFC 8259.
fn push_json_str(s: &mut String, v: &str) {
    s.push('"');
    for c in v.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                s.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => s.push(c),
        }
    }
    s.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_expected_shape() {
        let d = Description::new("demo", "a demo function").endpoint(
            Endpoint::get("/x", "does x")
                .param(Param::int("n", "how many").default("10"))
                .param(Param::string("q", "the query").required())
                .example("/x?n=5&q=hi")
                .returns("{ ok: true }"),
        );
        let j = d.to_json();
        assert!(j.contains("\"name\":\"demo\""));
        assert!(j.contains("\"path\":\"/x\""));
        assert!(j.contains("\"methods\":[\"GET\"]"));
        assert!(j.contains("\"name\":\"n\",\"type\":\"int\",\"required\":false,\"default\":\"10\""));
        assert!(j.contains("\"name\":\"q\",\"type\":\"string\",\"required\":true,\"default\":null"));
        assert!(j.contains("\"example\":\"/x?n=5&q=hi\""));
        // Valid JSON round-trips through serde_json (used in tests only).
        let _: serde_json::Value = serde_json::from_str(&j).expect("valid JSON");
    }

    #[test]
    fn escapes_special_chars() {
        let d = Description::new("a\"b", "line\nbreak");
        let j = d.to_json();
        assert!(j.contains("\"name\":\"a\\\"b\""));
        assert!(j.contains("\"summary\":\"line\\nbreak\""));
        let _: serde_json::Value = serde_json::from_str(&j).expect("valid JSON");
    }
}
