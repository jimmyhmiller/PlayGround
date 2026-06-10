use std::io::{self, BufRead, Write};
use std::net::TcpStream;

use datalog_db::protocol;

// --- Client ---

struct Client {
    stream: TcpStream,
    next_id: u64,
}

impl Client {
    fn connect(addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut stream = TcpStream::connect(addr)?;
        protocol::client_handshake(&mut stream)?;
        Ok(Self { stream, next_id: 1 })
    }

    fn send(&mut self, payload: serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let id = self.next_id;
        self.next_id += 1;
        protocol::write_message(&mut self.stream, id, &payload)?;
        let msg = protocol::read_message(&mut self.stream)?;
        Ok(msg.payload)
    }
}

// --- Parser ---

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn skip_ws(&mut self) {
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn expect(&mut self, ch: char) -> Result<(), String> {
        self.skip_ws();
        match self.advance() {
            Some(c) if c == ch => Ok(()),
            Some(c) => Err(format!("expected '{}', got '{}'", ch, c)),
            None => Err(format!("expected '{}', got end of input", ch)),
        }
    }

    fn read_ident(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        let bytes = self.input.as_bytes();
        while self.pos < bytes.len() && (bytes[self.pos].is_ascii_alphanumeric() || bytes[self.pos] == b'_') {
            self.pos += 1;
        }
        if self.pos == start {
            Err(format!("expected identifier, got {:?}", self.peek()))
        } else {
            Ok(self.input[start..self.pos].to_string())
        }
    }

    /// Peek the next identifier without consuming input. Returns "" if the
    /// next non-whitespace token is not an identifier.
    fn peek_ident(&mut self) -> String {
        let saved = self.pos;
        let ident = self.read_ident().unwrap_or_default();
        self.pos = saved;
        ident
    }

    fn read_string_literal(&mut self) -> Result<String, String> {
        self.skip_ws();
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('\\') => match self.advance() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('\\') => s.push('\\'),
                    Some('"') => s.push('"'),
                    Some(c) => { s.push('\\'); s.push(c); }
                    None => return Err("unterminated string escape".into()),
                },
                Some('"') => return Ok(s),
                Some(c) => s.push(c),
                None => return Err("unterminated string literal".into()),
            }
        }
    }

    fn read_number(&mut self) -> Result<serde_json::Value, String> {
        self.skip_ws();
        let start = self.pos;
        let bytes = self.input.as_bytes();
        if self.pos < bytes.len() && bytes[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.pos < bytes.len() && bytes[self.pos] == b'.' {
            is_float = true;
            self.pos += 1;
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let num_str = &self.input[start..self.pos];
        if num_str.is_empty() || num_str == "-" {
            return Err(format!("expected number at position {}", start));
        }
        if is_float {
            let f: f64 = num_str.parse().map_err(|_| format!("invalid float: {}", num_str))?;
            Ok(serde_json::json!(f))
        } else {
            let i: i64 = num_str.parse().map_err(|_| format!("invalid integer: {}", num_str))?;
            Ok(serde_json::json!(i))
        }
    }

    fn read_field_type(&mut self) -> Result<String, String> {
        self.skip_ws();
        // List type: `[elem]` — a cardinality-many field.
        if self.peek() == Some('[') {
            self.advance();
            let elem = self.read_field_type()?;
            self.skip_ws();
            self.expect(']')?;
            return Ok(format!("[{}]", elem));
        }
        let ident = self.read_ident()?;
        match ident.as_str() {
            "string" | "i64" | "f64" | "bool" | "bytes" => Ok(ident),
            "ref" | "enum" => {
                self.expect('(')?;
                let inner = self.read_ident()?;
                self.expect(')')?;
                Ok(format!("{}({})", ident, inner))
            }
            "list" => {
                self.expect('(')?;
                let inner = self.read_field_type()?;
                self.skip_ws();
                self.expect(')')?;
                Ok(format!("list({})", inner))
            }
            "vector" => {
                self.expect('(')?;
                self.skip_ws();
                let dim = self.read_number()?;
                let n = dim
                    .as_u64()
                    .ok_or("vector dimension must be a positive integer")?;
                self.skip_ws();
                self.expect(')')?;
                Ok(format!("vector({})", n))
            }
            other => Err(format!("unknown field type: '{}'", other)),
        }
    }

    /// Read a value in assert/data context: string, number, bool, #ref, enum
    /// variant, or a `[...]` list literal.
    fn read_value(&mut self) -> Result<serde_json::Value, String> {
        self.skip_ws();
        match self.peek() {
            Some('"') => {
                let s = self.read_string_literal()?;
                Ok(serde_json::Value::String(s))
            }
            Some('[') => {
                // List literal: [v1, v2, ...]. Empty `[]` is allowed.
                self.advance();
                let mut items = Vec::new();
                loop {
                    self.skip_ws();
                    if self.peek() == Some(']') {
                        self.advance();
                        break;
                    }
                    items.push(self.read_value()?);
                    self.skip_ws();
                    match self.peek() {
                        Some(',') => {
                            self.advance();
                        }
                        Some(']') => {
                            self.advance();
                            break;
                        }
                        other => {
                            return Err(format!("expected ',' or ']' in list, got {:?}", other))
                        }
                    }
                }
                Ok(serde_json::Value::Array(items))
            }
            Some('#') => {
                self.advance();
                let num = self.read_number()?;
                let id = num.as_u64().ok_or("ref entity id must be a positive integer")?;
                Ok(serde_json::json!({"ref": id}))
            }
            Some(c) if c == '-' || c.is_ascii_digit() => self.read_number(),
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let ident = self.read_ident()?;
                match ident.as_str() {
                    "true" => Ok(serde_json::Value::Bool(true)),
                    "false" => Ok(serde_json::Value::Bool(false)),
                    _ => {
                        self.skip_ws();
                        if self.peek() == Some('{') {
                            let fields = self.read_kv_block()?;
                            let mut obj = serde_json::Map::new();
                            obj.insert(ident, fields);
                            Ok(serde_json::Value::Object(obj))
                        } else {
                            Ok(serde_json::Value::String(ident))
                        }
                    }
                }
            }
            other => Err(format!("expected value, got {:?}", other)),
        }
    }

    /// Read { key: value, ... } block.
    fn read_kv_block(&mut self) -> Result<serde_json::Value, String> {
        self.expect('{')?;
        let mut map = serde_json::Map::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.advance();
            return Ok(serde_json::Value::Object(map));
        }
        loop {
            let key = self.read_ident()?;
            self.expect(':')?;
            let val = self.read_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(',') => { self.advance(); }
                Some('}') => { self.advance(); break; }
                other => return Err(format!("expected ',' or '}}', got {:?}", other)),
            }
        }
        Ok(serde_json::Value::Object(map))
    }

    /// Read a pattern for a where clause field.
    fn read_pattern(&mut self) -> Result<serde_json::Value, String> {
        self.skip_ws();
        match self.peek() {
            Some('?') => {
                self.advance();
                let name = self.read_ident()?;
                let var = format!("?{}", name);
                // A variable may be followed by a comparison operator to both
                // bind and filter in one clause field, e.g. `age: ?age > 25`.
                self.skip_ws();
                if let Some(op_key) = self.try_read_cmp_op() {
                    let val = self.read_pattern_value()?;
                    Ok(serde_json::json!({ "var": var, op_key: val }))
                } else {
                    Ok(serde_json::Value::String(var))
                }
            }
            Some('"') => {
                let s = self.read_string_literal()?;
                Ok(serde_json::Value::String(s))
            }
            Some('#') => {
                self.advance();
                let num = self.read_number()?;
                let id = num.as_u64().ok_or("ref entity id must be a positive integer")?;
                Ok(serde_json::json!({"ref": id}))
            }
            Some('>') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    let val = self.read_pattern_value()?;
                    Ok(serde_json::json!({"gte": val}))
                } else {
                    let val = self.read_pattern_value()?;
                    Ok(serde_json::json!({"gt": val}))
                }
            }
            Some('<') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    let val = self.read_pattern_value()?;
                    Ok(serde_json::json!({"lte": val}))
                } else {
                    let val = self.read_pattern_value()?;
                    Ok(serde_json::json!({"lt": val}))
                }
            }
            Some('!') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                } else {
                    return Err("expected '=' after '!'".into());
                }
                let val = self.read_pattern_value()?;
                Ok(serde_json::json!({"ne": val}))
            }
            Some(c) if c == '-' || c.is_ascii_digit() => self.read_number(),
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let ident = self.read_ident()?;
                match ident.as_str() {
                    "true" => Ok(serde_json::Value::Bool(true)),
                    "false" => Ok(serde_json::Value::Bool(false)),
                    // Word operators for string search: `body: contains "x"`.
                    "contains" | "starts_with" | "ends_with" => {
                        let val = self.read_pattern_value()?;
                        Ok(serde_json::json!({ ident: val }))
                    }
                    _ => Ok(serde_json::Value::String(ident)),
                }
            }
            other => Err(format!("expected pattern, got {:?}", other)),
        }
    }

    /// If the next non-whitespace token is a comparison operator, consume it
    /// and return its JSON predicate key ("gt", "gte", "lt", "lte", "ne").
    /// Leaves the parser untouched and returns None if no operator is present.
    fn try_read_cmp_op(&mut self) -> Option<&'static str> {
        self.skip_ws();
        match self.peek() {
            Some('>') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Some("gte")
                } else {
                    Some("gt")
                }
            }
            Some('<') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Some("lte")
                } else {
                    Some("lt")
                }
            }
            Some('!') if self.input[self.pos..].starts_with("!=") => {
                self.advance();
                self.advance();
                Some("ne")
            }
            // Word operators for string search, e.g. `body: ?b contains "x"`.
            Some(c) if c.is_ascii_alphabetic() => {
                let rest = &self.input[self.pos..];
                for op in ["contains", "starts_with", "ends_with"] {
                    if rest.starts_with(op) {
                        // Only consume it as an operator if a word boundary
                        // follows (so we don't eat an identifier prefix).
                        let after = rest[op.len()..].chars().next();
                        if after.map_or(true, |c| !c.is_ascii_alphanumeric() && c != '_') {
                            for _ in 0..op.len() {
                                self.advance();
                            }
                            return Some(match op {
                                "contains" => "contains",
                                "starts_with" => "starts_with",
                                _ => "ends_with",
                            });
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Read a simple value for a predicate operand.
    fn read_pattern_value(&mut self) -> Result<serde_json::Value, String> {
        self.skip_ws();
        match self.peek() {
            Some('"') => {
                let s = self.read_string_literal()?;
                Ok(serde_json::Value::String(s))
            }
            Some(c) if c == '-' || c.is_ascii_digit() => self.read_number(),
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let ident = self.read_ident()?;
                match ident.as_str() {
                    "true" => Ok(serde_json::Value::Bool(true)),
                    "false" => Ok(serde_json::Value::Bool(false)),
                    _ => Ok(serde_json::Value::String(ident)),
                }
            }
            other => Err(format!("expected value, got {:?}", other)),
        }
    }

    fn try_keyword(&mut self, kw: &str) -> bool {
        self.skip_ws();
        let saved = self.pos;
        if let Ok(ident) = self.read_ident() {
            if ident == kw {
                return true;
            }
        }
        self.pos = saved;
        false
    }
}

// --- Command parsing ---

/// A parsed command. Most commands resolve to a server payload (`Send`),
/// but some (`List`) require client-side expansion against the schema,
/// and a few (`Agent`, `Help`) need no server at all.
enum Cmd {
    Send(serde_json::Value),
    List { type_name: String, limit: Option<usize> },
    Agent,
    Help,
}

fn parse_command(input: &str) -> Result<Cmd, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("empty command".into());
    }

    let mut p = Parser::new(input);
    let keyword = p.read_ident()?;

    match keyword.as_str() {
        "status" => Ok(Cmd::Send(serde_json::json!({"type": "status"}))),
        "schema" => Ok(Cmd::Send(serde_json::json!({"type": "schema"}))),
        "define" => {
            let payload = if p.try_keyword("enum") {
                parse_define_enum(&mut p)?
            } else {
                parse_define(&mut p)?
            };
            Ok(Cmd::Send(payload))
        }
        "drop" => parse_drop(&mut p, false),
        "purge" => parse_drop(&mut p, true),
        "assert" => Ok(Cmd::Send(parse_assert(&mut p)?)),
        "retract" => Ok(Cmd::Send(parse_retract(&mut p)?)),
        "find" => Ok(Cmd::Send(parse_find(&mut p)?)),
        "list" => parse_list(&mut p),
        "backup" => parse_backup(&mut p),
        "agent" => Ok(Cmd::Agent),
        "help" => Ok(Cmd::Help),
        "json" => {
            let rest = p.rest().trim();
            let val: serde_json::Value = serde_json::from_str(rest)
                .map_err(|e| format!("invalid JSON: {}", e))?;
            Ok(Cmd::Send(val))
        }
        other => Err(format!("unknown command: '{}'. Type 'help' for usage.", other)),
    }
}

/// drop User           — soft-drop a type (definition retracted, data kept)
/// drop enum Status    — soft-drop an enum
/// purge User          — hard-purge a type (definition + all datoms deleted)
/// purge enum Status   — hard-purge an enum
fn parse_drop(p: &mut Parser, hard: bool) -> Result<Cmd, String> {
    let is_enum = p.try_keyword("enum");
    let name = p.read_ident()?;
    let req_type = if is_enum { "drop_enum" } else { "drop_type" };
    Ok(Cmd::Send(serde_json::json!({
        "type": req_type,
        "name": name,
        "hard": hard,
    })))
}

/// backup now
/// backup list
fn parse_backup(p: &mut Parser) -> Result<Cmd, String> {
    let sub = p.read_ident()?;
    match sub.as_str() {
        "now" => Ok(Cmd::Send(serde_json::json!({"type": "backup_now"}))),
        "list" => Ok(Cmd::Send(serde_json::json!({"type": "backup_list"}))),
        other => Err(format!(
            "unknown backup subcommand: '{}'. Use 'backup now' or 'backup list'.",
            other
        )),
    }
}

/// list User
/// list User limit 50
fn parse_list(p: &mut Parser) -> Result<Cmd, String> {
    let type_name = p.read_ident()?;
    let limit = if p.try_keyword("limit") {
        let n = p.read_number()?;
        Some(n.as_u64().ok_or("limit must be a non-negative integer")? as usize)
    } else {
        None
    };
    Ok(Cmd::List { type_name, limit })
}

/// define User { name: string required, age: i64, email: string unique indexed }
fn parse_define(p: &mut Parser) -> Result<serde_json::Value, String> {
    let type_name = p.read_ident()?;
    p.expect('{')?;

    let mut fields = Vec::new();
    p.skip_ws();
    if p.peek() != Some('}') {
        loop {
            let field_name = p.read_ident()?;
            p.expect(':')?;
            let field_type = p.read_field_type()?;

            let mut required = false;
            let mut unique = false;
            let mut indexed = false;
            let mut many = false;
            let mut fulltext = false;
            let mut ann = false;

            loop {
                p.skip_ws();
                let saved = p.pos;
                if let Ok(modifier) = p.read_ident() {
                    match modifier.as_str() {
                        "required" => required = true,
                        "unique" => unique = true,
                        "indexed" => indexed = true,
                        "many" => many = true,
                        "fulltext" => fulltext = true,
                        "ann" => ann = true,
                        _ => { p.pos = saved; break; }
                    }
                } else {
                    break;
                }
            }

            let mut field = serde_json::json!({
                "name": field_name,
                "type": field_type,
            });
            if required { field["required"] = serde_json::json!(true); }
            if unique { field["unique"] = serde_json::json!(true); }
            if indexed { field["indexed"] = serde_json::json!(true); }
            if many { field["cardinality"] = serde_json::json!("many"); }
            if fulltext { field["fulltext"] = serde_json::json!(true); }
            if ann { field["ann"] = serde_json::json!(true); }
            fields.push(field);

            p.skip_ws();
            match p.peek() {
                Some(',') => { p.advance(); }
                Some('}') => break,
                other => return Err(format!("expected ',' or '}}' in define, got {:?}", other)),
            }
        }
    }
    p.expect('}')?;

    // Optional trailing composite unique keys: `unique (doc, idx)`, repeatable.
    let mut unique_keys: Vec<Vec<String>> = Vec::new();
    loop {
        p.skip_ws();
        let saved = p.pos;
        match p.read_ident() {
            Ok(kw) if kw == "unique" => {
                p.skip_ws();
                p.expect('(')?;
                let mut key_fields = Vec::new();
                loop {
                    let fname = p.read_ident()?;
                    key_fields.push(fname);
                    p.skip_ws();
                    match p.peek() {
                        Some(',') => { p.advance(); }
                        Some(')') => { p.advance(); break; }
                        other => return Err(format!(
                            "expected ',' or ')' in unique key, got {:?}", other)),
                    }
                }
                unique_keys.push(key_fields);
            }
            _ => { p.pos = saved; break; }
        }
    }

    let mut req = serde_json::json!({
        "type": "define",
        "entity_type": type_name,
        "fields": fields,
    });
    if !unique_keys.is_empty() {
        req["unique_keys"] = serde_json::json!(unique_keys);
    }
    Ok(req)
}

/// define enum Status { Active, Suspended { reason: string } }
fn parse_define_enum(p: &mut Parser) -> Result<serde_json::Value, String> {
    let enum_name = p.read_ident()?;
    p.expect('{')?;

    let mut variants = Vec::new();
    p.skip_ws();
    if p.peek() != Some('}') {
        loop {
            let variant_name = p.read_ident()?;
            p.skip_ws();

            let variant = if p.peek() == Some('{') {
                p.expect('{')?;
                let mut fields = Vec::new();
                p.skip_ws();
                if p.peek() != Some('}') {
                    loop {
                        let field_name = p.read_ident()?;
                        p.expect(':')?;
                        let field_type = p.read_field_type()?;

                        let mut required = false;
                        loop {
                            p.skip_ws();
                            let saved = p.pos;
                            if let Ok(m) = p.read_ident() {
                                if m == "required" { required = true; }
                                else { p.pos = saved; break; }
                            } else { break; }
                        }

                        let mut field = serde_json::json!({
                            "name": field_name,
                            "type": field_type,
                        });
                        if required { field["required"] = serde_json::json!(true); }
                        fields.push(field);

                        p.skip_ws();
                        match p.peek() {
                            Some(',') => { p.advance(); }
                            Some('}') => break,
                            other => return Err(format!("expected ',' or '}}' in variant fields, got {:?}", other)),
                        }
                    }
                }
                p.expect('}')?;
                serde_json::json!({"name": variant_name, "fields": fields})
            } else {
                serde_json::json!({"name": variant_name})
            };

            variants.push(variant);

            p.skip_ws();
            match p.peek() {
                Some(',') => { p.advance(); }
                Some('}') => break,
                other => return Err(format!("expected ',' or '}}' in enum variants, got {:?}", other)),
            }
        }
    }
    p.expect('}')?;

    Ok(serde_json::json!({
        "type": "define_enum",
        "enum_name": enum_name,
        "variants": variants,
    }))
}

/// assert User { name: "Alice", age: 30 }
/// assert User #42 { age: 31 }
fn parse_assert(p: &mut Parser) -> Result<serde_json::Value, String> {
    let type_name = p.read_ident()?;
    p.skip_ws();

    let entity = if p.peek() == Some('#') {
        p.advance();
        let num = p.read_number()?;
        Some(num.as_u64().ok_or("entity id must be a positive integer")?)
    } else {
        None
    };

    let data = p.read_kv_block()?;

    let mut op = serde_json::json!({
        "assert": type_name,
        "data": data,
    });
    if let Some(eid) = entity {
        op["entity"] = serde_json::json!(eid);
    }

    Ok(serde_json::json!({
        "type": "transact",
        "ops": [op],
    }))
}

/// retract User #42 [email, age]   — retract specific fields
/// retract User #42                — retract entire entity (soft delete)
fn parse_retract(p: &mut Parser) -> Result<serde_json::Value, String> {
    let type_name = p.read_ident()?;
    p.expect('#')?;
    let num = p.read_number()?;
    let entity_id = num.as_u64().ok_or("entity id must be a positive integer")?;

    p.skip_ws();
    if p.peek() == Some('[') {
        // Field-level retract
        p.advance(); // consume '['
        let mut fields = Vec::new();
        p.skip_ws();
        if p.peek() != Some(']') {
            loop {
                let field_name = p.read_ident()?;
                fields.push(serde_json::Value::String(field_name));
                p.skip_ws();
                match p.peek() {
                    Some(',') => { p.advance(); }
                    Some(']') => break,
                    other => return Err(format!("expected ',' or ']', got {:?}", other)),
                }
            }
        }
        p.expect(']')?;

        Ok(serde_json::json!({
            "type": "transact",
            "ops": [{
                "retract": type_name,
                "entity": entity_id,
                "fields": fields,
            }],
        }))
    } else {
        // Whole-entity retract (soft delete)
        Ok(serde_json::json!({
            "type": "transact",
            "ops": [{
                "retract_entity": type_name,
                "entity": entity_id,
            }],
        }))
    }
}

/// find ?name, ?age where ?u: User { name: ?name, age: > 25 }
/// find ?name where ?u: User { name: ?name } as_of 100
fn parse_find(p: &mut Parser) -> Result<serde_json::Value, String> {
    // Read find elements: plain variables (?x) or aggregates (count(?x),
    // sum(?p), count(*)). Stops at the `where` keyword.
    let mut find_vars = Vec::new();
    loop {
        p.skip_ws();
        if p.peek() == Some('?') {
            p.advance();
            let name = p.read_ident()?;
            find_vars.push(serde_json::Value::String(format!("?{}", name)));
        } else {
            let saved = p.pos;
            let ident = p.read_ident()?;
            if ident == "where" {
                break;
            }
            // Aggregate form: ident '(' (?var | '*') ')'
            p.skip_ws();
            if p.peek() == Some('(') {
                p.advance();
                p.skip_ws();
                // Optional `distinct` modifier: count(distinct ?x).
                let mut distinct = false;
                if p.peek_ident() == "distinct" {
                    p.read_ident()?;
                    distinct = true;
                }
                p.skip_ws();
                let arg = match p.peek() {
                    Some('*') => {
                        p.advance();
                        "*".to_string()
                    }
                    Some('?') => {
                        p.advance();
                        format!("?{}", p.read_ident()?)
                    }
                    other => {
                        return Err(format!(
                            "expected variable or '*' in {}(...), got {:?}",
                            ident, other
                        ))
                    }
                };
                p.skip_ws();
                if p.peek() != Some(')') {
                    return Err(format!("expected ')' to close {}(...)", ident));
                }
                p.advance();
                let inner = if distinct {
                    format!("distinct {}", arg)
                } else {
                    arg
                };
                find_vars.push(serde_json::Value::String(format!("{}({})", ident, inner)));
            } else {
                p.pos = saved;
                return Err(format!("expected '?', aggregate, or 'where', got '{}'", ident));
            }
        }
        p.skip_ws();
        if p.peek() == Some(',') {
            p.advance();
        }
    }

    if find_vars.is_empty() {
        return Err("find requires at least one variable or aggregate".into());
    }

    // Parse where clauses: a sequence of clause items (implicit AND). Each
    // item is a pattern (?v: Type { ... }), an `or { } { }`, or a `not { }`.
    let mut where_clauses = Vec::new();
    loop {
        p.skip_ws();
        let starts_clause =
            p.peek() == Some('?') || matches!(p.peek_ident().as_str(), "or" | "not" | "and");
        if !starts_clause {
            break;
        }
        where_clauses.push(parse_clause(p)?);
        p.skip_ws();
        if p.peek() == Some(',') {
            p.advance();
        }
    }
    if where_clauses.is_empty() {
        return Err("expected at least one where clause".into());
    }

    let mut query = serde_json::json!({
        "type": "query",
        "find": find_vars,
        "where": where_clauses,
    });

    // Check for as_of or as_of_time
    if p.try_keyword("as_of_time") {
        p.skip_ws();
        if p.peek() == Some('"') {
            let iso = p.read_string_literal()?;
            let ms = parse_iso8601_to_millis(&iso)
                .map_err(|e| format!("invalid ISO 8601 timestamp: {}", e))?;
            query["as_of_time"] = serde_json::json!(ms);
        } else {
            let ms = p.read_number()?;
            query["as_of_time"] = ms;
        }
    } else if p.try_keyword("as_of") {
        let tx_id = p.read_number()?;
        query["as_of"] = tx_id;
    }

    // Optional: order by ?v [asc|desc], ... | limit N | offset N
    if p.try_keyword("order") {
        if !p.try_keyword("by") {
            return Err("expected 'by' after 'order'".into());
        }
        let mut keys = Vec::new();
        loop {
            p.skip_ws();
            // An order key is either a plain `?var` or an aggregate expression
            // (`count(?v)`, `count(distinct ?v)`, `avg(?x)`, `count(*)`). The
            // aggregate form must exactly match a find aggregate's output
            // label, so the executor can resolve it to that column.
            let var = if p.peek() == Some('?') {
                p.advance();
                format!("?{}", p.read_ident()?)
            } else {
                let func = p.read_ident().map_err(|_| {
                    "expected '?var' or an aggregate in 'order by'".to_string()
                })?;
                p.skip_ws();
                if p.peek() != Some('(') {
                    return Err(format!(
                        "expected '(' after '{}' in 'order by' aggregate",
                        func
                    ));
                }
                p.advance();
                p.skip_ws();
                let mut distinct = false;
                if p.peek_ident() == "distinct" {
                    p.read_ident()?;
                    distinct = true;
                }
                p.skip_ws();
                let arg = match p.peek() {
                    Some('*') => {
                        p.advance();
                        "*".to_string()
                    }
                    Some('?') => {
                        p.advance();
                        format!("?{}", p.read_ident()?)
                    }
                    other => {
                        return Err(format!(
                            "expected variable or '*' in order by {}(...), got {:?}",
                            func, other
                        ))
                    }
                };
                p.skip_ws();
                if p.peek() != Some(')') {
                    return Err(format!("expected ')' to close order by {}(...)", func));
                }
                p.advance();
                let inner = if distinct {
                    format!("distinct {}", arg)
                } else {
                    arg
                };
                format!("{}({})", func, inner)
            };
            // Optional direction; default ascending.
            let desc = if p.try_keyword("desc") {
                true
            } else {
                p.try_keyword("asc");
                false
            };
            keys.push(serde_json::json!({ "var": var, "desc": desc }));
            p.skip_ws();
            if p.peek() == Some(',') {
                p.advance();
                continue;
            }
            break;
        }
        query["order_by"] = serde_json::Value::Array(keys);
    }
    if p.try_keyword("limit") {
        query["limit"] = p.read_number()?;
    }
    if p.try_keyword("offset") {
        query["offset"] = p.read_number()?;
    }

    Ok(query)
}

/// Parse one clause item: a pattern (`?v: Type { ... }`), an `or { } { } ...`,
/// a `not { }`, or an `and { }`.
fn parse_clause(p: &mut Parser) -> Result<serde_json::Value, String> {
    p.skip_ws();
    if p.peek() == Some('?') {
        return parse_pattern_clause(p);
    }
    let ident = p.read_ident()?;
    match ident.as_str() {
        "or" => {
            let mut branches = Vec::new();
            loop {
                p.skip_ws();
                if p.peek() != Some('{') {
                    break;
                }
                branches.push(parse_clause_group(p)?);
            }
            if branches.len() < 2 {
                return Err("'or' requires at least two { ... } groups".into());
            }
            Ok(serde_json::json!({ "or": branches }))
        }
        "and" => Ok(serde_json::json!({ "and": parse_clause_group_items(p)? })),
        "not" => Ok(serde_json::json!({ "not": parse_clause_group(p)? })),
        other => Err(format!("expected '?', 'or', or 'not', got '{}'", other)),
    }
}

/// A `{ ... }` group of one or more clause items (implicit AND). Returns a
/// single clause if there's exactly one item, else a JSON array (an `And`).
fn parse_clause_group(p: &mut Parser) -> Result<serde_json::Value, String> {
    let mut items = parse_clause_group_items(p)?;
    if items.len() == 1 {
        Ok(items.remove(0))
    } else {
        Ok(serde_json::Value::Array(items))
    }
}

/// Parse the items inside a `{ ... }` clause group.
fn parse_clause_group_items(p: &mut Parser) -> Result<Vec<serde_json::Value>, String> {
    p.expect('{')?;
    let mut items = Vec::new();
    loop {
        p.skip_ws();
        if p.peek() == Some('}') {
            break;
        }
        items.push(parse_clause(p)?);
        p.skip_ws();
        if p.peek() == Some(',') {
            p.advance();
        }
    }
    p.expect('}')?;
    if items.is_empty() {
        return Err("clause group { } must contain at least one clause".into());
    }
    Ok(items)
}

/// Parse a single pattern clause: `?v: Type { field: pattern, ... }`.
fn parse_pattern_clause(p: &mut Parser) -> Result<serde_json::Value, String> {
    p.expect('?')?;
    let bind_name = p.read_ident()?;
    let bind = format!("?{}", bind_name);
    p.expect(':')?;
    let entity_type = p.read_ident()?;

    p.expect('{')?;
    let mut clause = serde_json::json!({
        "bind": bind,
        "type": entity_type,
    });

    p.skip_ws();
    if p.peek() != Some('}') {
        loop {
            let field_name = p.read_ident()?;
            p.expect(':')?;
            let pattern = p.read_pattern()?;
            clause[&field_name] = pattern;

            p.skip_ws();
            match p.peek() {
                Some(',') => {
                    p.advance();
                }
                Some('}') => break,
                other => {
                    return Err(format!("expected ',' or '}}' in where clause, got {:?}", other))
                }
            }
        }
    }
    p.expect('}')?;
    Ok(clause)
}

/// Parse a subset of ISO 8601: "YYYY-MM-DDTHH:MM:SSZ"
fn parse_iso8601_to_millis(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.len() != 20 || !s.ends_with('Z') {
        return Err(format!("expected format YYYY-MM-DDTHH:MM:SSZ, got '{}'", s));
    }
    let b = s.as_bytes();
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return Err(format!("expected format YYYY-MM-DDTHH:MM:SSZ, got '{}'", s));
    }

    let year: i64 = s[0..4].parse().map_err(|_| "invalid year")?;
    let month: i64 = s[5..7].parse().map_err(|_| "invalid month")?;
    let day: i64 = s[8..10].parse().map_err(|_| "invalid day")?;
    let hour: i64 = s[11..13].parse().map_err(|_| "invalid hour")?;
    let min: i64 = s[14..16].parse().map_err(|_| "invalid minute")?;
    let sec: i64 = s[17..19].parse().map_err(|_| "invalid second")?;

    if !(1..=12).contains(&month) { return Err("month out of range".into()); }
    if !(1..=31).contains(&day) { return Err("day out of range".into()); }
    if !(0..=23).contains(&hour) { return Err("hour out of range".into()); }
    if !(0..=59).contains(&min) { return Err("minute out of range".into()); }
    if !(0..=59).contains(&sec) { return Err("second out of range".into()); }

    // Days from civil date to unix epoch (algorithm from Howard Hinnant)
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;

    let total_secs = days * 86400 + hour * 3600 + min * 60 + sec;
    Ok((total_secs * 1000) as u64)
}

// --- Response formatting ---

fn format_response(resp: &serde_json::Value) -> String {
    let status = resp.get("status").and_then(|s| s.as_str()).unwrap_or("unknown");

    if status == "error" {
        let error = resp.get("error").and_then(|e| e.as_str())
            .or_else(|| resp.get("error").map(|e| e.to_string()).as_deref().map(|_| ""))
            .unwrap_or("unknown error");
        return format!("ERROR: {}\n", error);
    }

    let data = match resp.get("data") {
        Some(d) => d,
        None => return "OK\n".to_string(),
    };

    // Query result (has columns + rows)
    if let (Some(columns), Some(rows)) = (data.get("columns"), data.get("rows")) {
        return format_query_result(columns, rows);
    }

    // Schema result (has types + enums)
    if data.get("types").is_some() && data.get("enums").is_some() {
        return format_schema(data);
    }

    // Non-query: format as key=value summary
    let mut parts = vec!["OK".to_string()];
    if let Some(obj) = data.as_object() {
        for (k, v) in obj {
            parts.push(format!("{}={}", k, format_summary_value(v)));
        }
    }
    format!("{}\n", parts.join(" "))
}

fn format_query_result(columns: &serde_json::Value, rows: &serde_json::Value) -> String {
    let cols: Vec<&str> = columns.as_array()
        .map(|a| a.iter().map(|v| v.as_str().unwrap_or("?")).collect())
        .unwrap_or_default();
    let row_data: Vec<Vec<String>> = rows.as_array()
        .map(|a| a.iter().map(|row| {
            row.as_array()
                .map(|r| r.iter().map(|v| format_cell_value(v)).collect())
                .unwrap_or_default()
        }).collect())
        .unwrap_or_default();

    if cols.is_empty() {
        return "(no columns)\n".to_string();
    }

    // Calculate column widths
    let mut widths: Vec<usize> = cols.iter().map(|c| c.len()).collect();
    for row in &row_data {
        for (i, val) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(val.len());
            }
        }
    }

    let mut out = String::new();

    // Header
    let header: Vec<String> = cols.iter().enumerate()
        .map(|(i, c)| format!("{:width$}", c, width = widths[i]))
        .collect();
    out.push_str(&header.join(" | "));
    out.push('\n');

    // Separator
    let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    out.push_str(&sep.join("-+-"));
    out.push('\n');

    // Rows
    if row_data.is_empty() {
        out.push_str("(no results)\n");
    } else {
        for row in &row_data {
            let cells: Vec<String> = row.iter().enumerate()
                .map(|(i, v)| {
                    let w = widths.get(i).copied().unwrap_or(0);
                    format!("{:width$}", v, width = w)
                })
                .collect();
            out.push_str(&cells.join(" | "));
            out.push('\n');
        }
        out.push_str(&format!("({} row{})\n", row_data.len(), if row_data.len() == 1 { "" } else { "s" }));
    }

    out
}

fn format_schema(data: &serde_json::Value) -> String {
    let mut out = String::new();

    out.push_str("Entity types:\n");
    if let Some(types) = data.get("types").and_then(|t| t.as_array()) {
        if types.is_empty() {
            out.push_str("  (none)\n");
        }
        for type_def in types {
            let name = type_def.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            out.push_str(&format!("  {}\n", name));
            if let Some(fields) = type_def.get("fields").and_then(|f| f.as_array()) {
                for field in fields {
                    let fname = field.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    let ftype = format_field_type(field.get("type"));
                    let mut mods = Vec::new();
                    if field.get("required").and_then(|r| r.as_bool()).unwrap_or(false) {
                        mods.push("required");
                    }
                    if field.get("unique").and_then(|u| u.as_bool()).unwrap_or(false) {
                        mods.push("unique");
                    }
                    if field.get("indexed").and_then(|i| i.as_bool()).unwrap_or(false) {
                        mods.push("indexed");
                    }
                    if field.get("cardinality").and_then(|c| c.as_str()) == Some("many") {
                        mods.push("many");
                    }
                    if field.get("fulltext").and_then(|x| x.as_bool()).unwrap_or(false) {
                        mods.push("fulltext");
                    }
                    if field.get("ann").and_then(|x| x.as_bool()).unwrap_or(false) {
                        mods.push("ann");
                    }
                    let mod_str = if mods.is_empty() { String::new() } else { format!(" {}", mods.join(" ")) };
                    out.push_str(&format!("    {}: {}{}\n", fname, ftype, mod_str));
                }
            }
            // Composite unique keys.
            if let Some(keys) = type_def.get("unique_keys").and_then(|k| k.as_array()) {
                for key in keys {
                    if let Some(fields) = key.as_array() {
                        let names: Vec<String> = fields
                            .iter()
                            .filter_map(|f| f.as_str().map(String::from))
                            .collect();
                        if !names.is_empty() {
                            out.push_str(&format!("    unique ({})\n", names.join(", ")));
                        }
                    }
                }
            }
        }
    } else {
        out.push_str("  (none)\n");
    }

    out.push('\n');

    out.push_str("Enums:\n");
    if let Some(enums) = data.get("enums").and_then(|e| e.as_array()) {
        if enums.is_empty() {
            out.push_str("  (none)\n");
        }
        for enum_def in enums {
            let name = enum_def.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            out.push_str(&format!("  {}\n", name));
            if let Some(variants) = enum_def.get("variants").and_then(|v| v.as_array()) {
                for variant in variants {
                    let vname = variant.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    let fields = variant.get("fields").and_then(|f| f.as_array());
                    match fields {
                        Some(f) if !f.is_empty() => {
                            let field_strs: Vec<String> = f.iter().map(|fd| {
                                let fn_ = fd.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                                let ft = format_field_type(fd.get("type"));
                                format!("{}: {}", fn_, ft)
                            }).collect();
                            out.push_str(&format!("    {} {{ {} }}\n", vname, field_strs.join(", ")));
                        }
                        _ => out.push_str(&format!("    {}\n", vname)),
                    }
                }
            }
        }
    } else {
        out.push_str("  (none)\n");
    }

    out
}

fn format_field_type(v: Option<&serde_json::Value>) -> String {
    match v {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(obj)) => {
            if let Some(target) = obj.get("ref").and_then(|r| r.as_str()) {
                format!("ref({})", target)
            } else if let Some(target) = obj.get("enum").and_then(|e| e.as_str()) {
                format!("enum({})", target)
            } else if let Some(elem) = obj.get("list") {
                format!("[{}]", format_field_type(Some(elem)))
            } else if let Some(dim) = obj.get("vector").and_then(|d| d.as_u64()) {
                format!("vector({})", dim)
            } else {
                format!("{}", serde_json::Value::Object(obj.clone()))
            }
        }
        Some(other) => other.to_string(),
        None => "?".to_string(),
    }
}

/// Format a value for display in a table cell (strings are quoted).
fn format_cell_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => format!("\"{}\"", s),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(|v| format_cell_value(v)).collect();
            format!("[{}]", items.join(", "))
        }
        serde_json::Value::Object(obj) => {
            if let Some(id) = obj.get("ref").and_then(|r| r.as_u64()) {
                format!("#{}", id)
            } else {
                serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
            }
        }
    }
}

/// Format a value for display in a summary line (strings are unquoted).
fn format_summary_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(|v| format_summary_value(v)).collect();
            format!("[{}]", items.join(","))
        }
        _ => format_cell_value(v),
    }
}

// --- Help ---

fn print_help() {
    print!(r#"datalog - command-line client for datalog-db

USAGE:
  datalog [--host HOST:PORT] [--json]              Interactive REPL
  datalog [--host HOST:PORT] [--json] COMMAND      One-shot command
  datalog agent                                    Print LLM/agent usage guide

GLOBAL FLAGS:
  --host HOST:PORT    Server address (default 127.0.0.1:5557)
  --json              Print server responses as raw JSON (good for scripts/LLMs)

ONE-SHOT COMMANDS:
  status                                      Server status
  schema                                      List current schema
  query  'find ?x where ...'                  Run a query
  list   Type [limit N]                       List all entities of a type
  define 'Type {{ field: type, ... }}'          Define entity type
  drop   Type                                 Soft-drop type (keep data + history)
  drop   enum Name                            Soft-drop enum
  purge  Type                                 Hard-purge type (delete all datoms)
  purge  enum Name                            Hard-purge enum
  assert 'Type {{ field: value, ... }}'         Assert entity
  retract 'Type #ID [field, ...]'             Retract fields
  retract 'Type #ID'                          Retract entire entity
  backup now                                  Take an immediate checkpoint
  backup list                                 List existing checkpoints
  json   '{{...}}'                              Send raw JSON request
  agent                                       Print LLM/agent usage guide

DSL COMMANDS (REPL):
  status
  schema
  list User
  list User limit 50
  backup now
  backup list
  define User {{ name: string required, age: i64 }}
  define enum Status {{ Active, Suspended {{ reason: string }} }}
  drop User
  drop enum Status
  purge User
  purge enum Status
  assert User {{ name: "Alice", age: 30 }}
  assert User #42 {{ age: 31 }}
  retract User #42 [email, age]
  retract User #42
  find ?name, ?age where ?u: User {{ name: ?name, age: > 25 }}
  find ?name, ?age where ?u: User {{ name: ?name, age: ?age > 25 }}
  find ?name, ?age where ?u: User {{ name: ?name, age: ?age }} order by ?age desc
  find ?name where ?u: User {{ name: ?name }} order by ?name limit 10 offset 20
  find count(*), avg(?age) where ?u: User {{ age: ?age }}
  find ?dept, count(?e), max(?sal) where ?e: Employee {{ dept: ?dept, salary: ?sal }}
  find ?path, count(?v), count(distinct ?ip) where ?v: PageView {{ path: ?path, ip: ?ip }} order by count(?v) desc limit 20
  find ?name where ?e: Employee {{ name: ?name }} or {{ ?e: Employee {{ dept: "eng" }} }} {{ ?e: Employee {{ salary: < 85 }} }}
  find ?name where ?e: Employee {{ name: ?name }} not {{ ?p: Project {{ lead: ?e }} }}
  find ?n where ?u: User {{ name: ?n }} as_of 100
  find ?n where ?u: User {{ name: ?n }} as_of_time 1740192000000
  find ?n where ?u: User {{ name: ?n }} as_of_time "2025-02-22T02:00:00Z"
  json {{"type": "status"}}
  agent
  help
  quit

FIELD TYPES:
  string, i64, f64, bool, bytes, ref(Type), enum(EnumName)

FIELD MODIFIERS:
  required, unique, indexed
  (note: 'unique' also requires 'required')

PATTERN OPERATORS (in find):
  ?var                Variable binding
  > N, < N, >= N      Comparisons (filter only)
  <= N, != N
  ?var > N            Bind AND filter the same field (e.g. age: ?age > 25)
  "string"            Exact match
  #N                  Entity reference

RESULT SHAPING (after the where clauses):
  order by ?v [asc|desc], ...   Sort rows; key is a find variable OR an
                                aggregate label, e.g. `order by count(?v) desc`
  limit N                       Cap row count (after ordering)
  offset N                      Skip N rows (after ordering)

AGGREGATES (in find):
  count(*), count(?v), count(distinct ?v),
  sum(?v), avg(?v), min(?v), max(?v)
  Plain find variables become the GROUP BY key; with none, the whole
  result is one group. `count(distinct ?v)` counts distinct non-null values
  (e.g. unique visitors per page: count(distinct ?ip)). Order a top-N
  leaderboard by the aggregate itself: `... order by count(?v) desc limit 20`.

DISJUNCTION / NEGATION (in where, nestable):
  or  {{ <clauses> }} {{ <clauses> }} ...   Union of bindings from each group
  not {{ <clauses> }}                       Keep rows the group does NOT match
  and {{ <clauses> }}                       Explicit conjunction group
  (a `not` must sit alongside a positive clause that binds its variables)
"#);
}

fn print_agent_guide() {
    print!(r#"datalog — guide for LLMs and scripted agents
==================================================

This is a Datalog-style database with EAVT/AEVT/AVET/VAET indexes, history,
and a typed schema. The CLI talks to a long-running server over TCP.

QUICK START
-----------
1. Start the server in a separate process:
     datalog-db --data-dir ./datalog-data --bind 127.0.0.1:5557
2. Talk to it from the CLI:
     datalog --json status
3. Use `--json` for every call you intend to parse. Without it, results are
   printed as ASCII tables meant for humans.

WORKFLOW
--------
The flow is always: define schema -> assert entities -> query / list.

  # 1. Define an entity type (a "collection").
  datalog --json define 'User {{ name: string required, email: string unique required indexed, age: i64 }}'

  # 2. Insert entities. The server returns the assigned entity id.
  datalog --json assert 'User {{ name: "Alice", age: 30, email: "a@b.com" }}'
  # -> {{"status":"ok","data":{{"tx_id":2,"entity_ids":[3],...}}}}

  # 3. Update by referencing the entity id with `#`.
  datalog --json assert 'User #3 {{ age: 31 }}'

  # 4. List a type. Equivalent to `find` over every field.
  datalog --json list User
  datalog --json list User limit 20

  # 5. Query with patterns and predicates. A bare predicate (age: > 25)
  #    filters without binding; `age: ?age > 25` both binds ?age and filters.
  datalog --json query 'find ?u, ?name where ?u: User {{ name: ?name, age: > 25 }}'
  datalog --json query 'find ?name, ?age where ?u: User {{ name: ?name, age: ?age > 25 }}'

  # Sort and paginate with order by / limit / offset (order vars must be in find).
  datalog --json query 'find ?name, ?age where ?u: User {{ name: ?name, age: ?age }} order by ?age desc limit 10'

  # Aggregate. Plain find vars are the GROUP BY key; aggregates: count/sum/avg/min/max.
  datalog --json query 'find ?dept, count(?e), avg(?sal) where ?e: Employee {{ dept: ?dept, salary: ?sal }}'

  # Disjunction / negation (JSON: {{"or": [clause, ...]}} and {{"not": clause}}; nestable).
  datalog --json query 'find ?name where ?e: Employee {{ name: ?name }} not {{ ?p: Project {{ lead: ?e }} }}'

  # 6. Time travel: as_of <tx_id> or as_of_time <unix_ms | ISO8601>.
  datalog --json query 'find ?n where ?u: User {{ name: ?n }} as_of 2'

  # 7. Retract a field or whole entity.
  datalog --json retract 'User #3 [age]'
  datalog --json retract 'User #3'

  # 8. Drop a whole type/enum. `drop` is soft, `purge` is hard (see below).
  datalog --json drop User           # soft: hide the type, keep data + history
  datalog --json purge User          # hard: delete the type and all its datoms
  datalog --json drop enum Status
  datalog --json purge enum Status

DROPPING TYPES (soft vs hard)
-----------------------------
Two ways to remove a type or enum from the schema:

* `drop` (soft) retracts the schema definition as a normal transaction. The
  type vanishes from `schema` and can no longer be queried, but every datom,
  index entry and history record is kept untouched. The drop is itself
  recorded in history, and re-`define`-ing the exact type makes all the old
  data visible and queryable again. Reversible. Use this by default.

* `purge` (hard) deletes the schema definition AND every datom of every
  entity of that type from all indexes (EAVT/AEVT/AVET/VAET + current
  mirrors). Irreversible; it destroys history for those entities. Take a
  `backup now` first if unsure.

  Send `{{"type":"drop_type","name":"User","hard":true}}` for a JSON purge;
  `drop_enum` for enums. `hard` defaults to false.

A hard purge is refused (error) while another live type/enum still
references the target via a `ref`/`enum` field — drop or redefine those
first. A purge response reports `entities_purged`, `datoms_deleted`, and
`dangling_refs` (other entities that still point at a purged entity; those
are left untouched, never silently rewritten).

RESPONSE SHAPE
--------------
Every `--json` response is one of:
  {{"status":"ok","data": ...}}
  {{"status":"error","error":"..."}}

Query results have shape:
  {{"status":"ok","data":{{"columns":["?u","?name"],"rows":[[{{"ref":2}},"Alice"], ...]}}}}

GOTCHAS (these will save you trial and error)
---------------------------------------------
* `unique` fields must also be marked `required`.
* `Unknown entity type: X` means a prior `define` failed silently for you;
  call `schema` to verify what is actually defined.
* Entity refs are `#N` in the DSL and `{{"ref": N}}` in JSON.
* In assert, every field whose declared type is `required` must be present.
* In one-shot mode, quote the whole command argument with single quotes so
  the shell does not eat the braces or `?`.
* The REPL accepts the same commands without the outer quoting.

BACKUPS
-------
If the server was started with `--backup-dir`, automatic checkpoint
backups run on a schedule (default hourly, keeping the last 24). You
can also trigger and inspect them on demand:

  datalog --json backup now       # take an immediate checkpoint
  datalog --json backup list      # list existing checkpoints

A checkpoint is a hard-link snapshot of the live DB at a point in time.
Restore is "stop the server, point `--data-dir` at one of the listed
backup paths, restart". Take a `backup now` before any destructive
batch you're not 100% sure of — restore costs nothing if you don't
need it, and seconds if you do.

If `backup now` returns `backups not configured`, the server was
started without `--backup-dir`; ask the operator to enable it.

COMMAND REFERENCE
-----------------
  status                              Probe server is up.
  schema                              Dump all defined types + enums.
  list <Type> [limit N]               All entities of a type (with all fields).
  define '<Type> {{ ... }}'             Add a type.
  define enum '<Name> {{ ... }}'        Add an enum type.
  drop <Type> | drop enum <Name>      Soft-drop (retract def, keep data+history).
  purge <Type> | purge enum <Name>    Hard-purge (delete def + all datoms).
  assert '<Type> {{ ... }}'             New entity.
  assert '<Type> #ID {{ ... }}'         Update existing entity.
  retract '<Type> #ID [f1, f2]'       Retract specific fields.
  retract '<Type> #ID'                Retract entire entity (soft delete).
  query '<find-expr>'                 Run a find query.
  backup now                          Take an immediate checkpoint.
  backup list                         List existing checkpoints.
  json '<raw-json>'                   Send a raw request payload.

If you need any of the above details inline, run `datalog --help`.
"#);
}

// --- Main ---

fn print_response(resp: &serde_json::Value, json_mode: bool) {
    if json_mode {
        match serde_json::to_string(resp) {
            Ok(s) => println!("{}", s),
            Err(_) => println!("{}", resp),
        }
    } else {
        print!("{}", format_response(resp));
    }
}

/// Execute a `Cmd::List <Type>` by fetching the schema, building a find
/// query that pulls every field, and sending it. `limit`, if given, is
/// applied client-side by truncating the result rows.
fn execute_list(
    client: &mut Client,
    type_name: &str,
    limit: Option<usize>,
) -> Result<serde_json::Value, String> {
    let schema_resp = client
        .send(serde_json::json!({"type": "schema"}))
        .map_err(|e| format!("schema lookup failed: {}", e))?;

    if schema_resp.get("status").and_then(|s| s.as_str()) != Some("ok") {
        return Ok(schema_resp);
    }

    let types = schema_resp
        .pointer("/data/types")
        .and_then(|t| t.as_array())
        .ok_or_else(|| "schema response missing /data/types".to_string())?;

    let entity_type = types
        .iter()
        .find(|t| t.get("name").and_then(|n| n.as_str()) == Some(type_name))
        .ok_or_else(|| format!("unknown entity type: '{}'. Try `schema` to see what is defined.", type_name))?;

    let fields: Vec<String> = entity_type
        .get("fields")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|f| f.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let mut find_vars: Vec<serde_json::Value> = Vec::with_capacity(1 + fields.len());
    find_vars.push(serde_json::Value::String("?e".to_string()));
    for f in &fields {
        find_vars.push(serde_json::Value::String(format!("?{}", f)));
    }

    let mut where_clause = serde_json::Map::new();
    where_clause.insert("bind".into(), serde_json::Value::String("?e".to_string()));
    where_clause.insert("type".into(), serde_json::Value::String(type_name.to_string()));
    for f in &fields {
        where_clause.insert(f.clone(), serde_json::Value::String(format!("?{}", f)));
    }

    let payload = serde_json::json!({
        "type": "query",
        "find": find_vars,
        "where": [serde_json::Value::Object(where_clause)],
    });

    let mut resp = client.send(payload).map_err(|e| format!("query failed: {}", e))?;

    if let Some(limit_n) = limit {
        if let Some(rows) = resp.pointer_mut("/data/rows").and_then(|r| r.as_array_mut()) {
            if rows.len() > limit_n {
                rows.truncate(limit_n);
            }
        }
    }

    Ok(resp)
}

/// Run a single parsed command against the server. Returns Err only on
/// fatal connection problems; protocol-level errors come back inside `resp`.
fn execute_cmd(client: &mut Client, cmd: Cmd, json_mode: bool) -> Result<(), String> {
    match cmd {
        Cmd::Send(payload) => match client.send(payload) {
            Ok(resp) => {
                print_response(&resp, json_mode);
                Ok(())
            }
            Err(e) => Err(format!("connection error: {}", e)),
        },
        Cmd::List { type_name, limit } => match execute_list(client, &type_name, limit) {
            Ok(resp) => {
                print_response(&resp, json_mode);
                Ok(())
            }
            Err(e) => {
                if json_mode {
                    let err = serde_json::json!({"status": "error", "error": e});
                    println!("{}", err);
                } else {
                    println!("ERROR: {}", e);
                }
                Ok(())
            }
        },
        Cmd::Agent => {
            print_agent_guide();
            Ok(())
        }
        Cmd::Help => {
            print_help();
            Ok(())
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut host = "127.0.0.1:5557".to_string();
    let mut json_mode = false;
    let mut remaining = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                if i + 1 < args.len() {
                    host = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("--host requires a value");
                    std::process::exit(1);
                }
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                remaining.push(args[i].clone());
                i += 1;
            }
        }
    }

    // `agent` is a no-server command; handle it before opening a connection
    // so an LLM can read the guide on a fresh machine.
    if remaining.first().map(|s| s.as_str()) == Some("agent") {
        print_agent_guide();
        return;
    }

    let mut client = match Client::connect(&host) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to connect to {}: {}", host, e);
            std::process::exit(1);
        }
    };

    if remaining.is_empty() {
        // REPL mode
        eprintln!("Connected to {}", host);
        eprintln!("Type 'help' for commands, 'agent' for the LLM guide, 'quit' to exit.");
        repl(&mut client, json_mode);
    } else {
        // One-shot mode
        let command = match remaining[0].as_str() {
            "status" | "schema" => remaining[0].clone(),
            "list" => {
                if remaining.len() < 2 {
                    eprintln!("'list' requires a type name");
                    std::process::exit(1);
                }
                // Re-join the rest so `list User limit 50` parses the same
                // way the REPL would parse it.
                let mut s = String::from("list");
                for piece in &remaining[1..] {
                    s.push(' ');
                    s.push_str(piece);
                }
                s
            }
            "backup" => {
                if remaining.len() < 2 {
                    eprintln!("'backup' requires a subcommand: 'now' or 'list'");
                    std::process::exit(1);
                }
                let mut s = String::from("backup");
                for piece in &remaining[1..] {
                    s.push(' ');
                    s.push_str(piece);
                }
                s
            }
            // `drop`/`purge` take bare, possibly multi-word args
            // (`drop User`, `purge enum Status`) so re-join like backup/list.
            "drop" | "purge" => {
                if remaining.len() < 2 {
                    eprintln!("'{}' requires a type or enum name", remaining[0]);
                    std::process::exit(1);
                }
                remaining.join(" ")
            }
            "query" => {
                if remaining.len() < 2 {
                    eprintln!("'query' requires an argument");
                    std::process::exit(1);
                }
                remaining[1].clone()
            }
            "define" | "assert" | "retract" | "json" => {
                if remaining.len() < 2 {
                    eprintln!("'{}' requires an argument", remaining[0]);
                    std::process::exit(1);
                }
                format!("{} {}", remaining[0], remaining[1])
            }
            other => {
                eprintln!("Unknown subcommand: '{}'. Use --help for usage.", other);
                std::process::exit(1);
            }
        };

        match parse_command(&command) {
            Ok(cmd) => {
                if let Err(e) = execute_cmd(&mut client, cmd, json_mode) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("Parse error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn repl(client: &mut Client, json_mode: bool) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        let _ = stdout.flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        }

        let line = line.trim();
        if line.is_empty() { continue; }

        if line == "quit" || line == "exit" {
            break;
        }

        match parse_command(line) {
            Ok(cmd) => {
                if let Err(e) = execute_cmd(client, cmd, json_mode) {
                    eprintln!("{}", e);
                    break;
                }
            }
            Err(e) => eprintln!("Parse error: {}", e),
        }
    }
}
