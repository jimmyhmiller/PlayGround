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
        let ident = self.read_ident()?;
        match ident.as_str() {
            "string" | "i64" | "f64" | "bool" | "bytes" => Ok(ident),
            "ref" | "enum" => {
                self.expect('(')?;
                let inner = self.read_ident()?;
                self.expect(')')?;
                Ok(format!("{}({})", ident, inner))
            }
            other => Err(format!("unknown field type: '{}'", other)),
        }
    }

    /// Read a value in assert/data context: string, number, bool, #ref, or enum variant.
    fn read_value(&mut self) -> Result<serde_json::Value, String> {
        self.skip_ws();
        match self.peek() {
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
                Ok(serde_json::Value::String(format!("?{}", name)))
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
                    _ => Ok(serde_json::Value::String(ident)),
                }
            }
            other => Err(format!("expected pattern, got {:?}", other)),
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

fn parse_command(input: &str) -> Result<serde_json::Value, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("empty command".into());
    }

    let mut p = Parser::new(input);
    let keyword = p.read_ident()?;

    match keyword.as_str() {
        "status" => Ok(serde_json::json!({"type": "status"})),
        "schema" => Ok(serde_json::json!({"type": "schema"})),
        "define" => {
            if p.try_keyword("enum") {
                parse_define_enum(&mut p)
            } else {
                parse_define(&mut p)
            }
        }
        "assert" => parse_assert(&mut p),
        "retract" => parse_retract(&mut p),
        "find" => parse_find(&mut p),
        "json" => {
            let rest = p.rest().trim();
            let val: serde_json::Value = serde_json::from_str(rest)
                .map_err(|e| format!("invalid JSON: {}", e))?;
            Ok(val)
        }
        other => Err(format!("unknown command: '{}'. Type 'help' for usage.", other)),
    }
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

            loop {
                p.skip_ws();
                let saved = p.pos;
                if let Ok(modifier) = p.read_ident() {
                    match modifier.as_str() {
                        "required" => required = true,
                        "unique" => unique = true,
                        "indexed" => indexed = true,
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

    Ok(serde_json::json!({
        "type": "define",
        "entity_type": type_name,
        "fields": fields,
    }))
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
    // Read find variables
    let mut find_vars = Vec::new();
    loop {
        p.skip_ws();
        if p.peek() == Some('?') {
            p.advance();
            let name = p.read_ident()?;
            find_vars.push(serde_json::Value::String(format!("?{}", name)));
            p.skip_ws();
            if p.peek() == Some(',') {
                p.advance();
            }
        } else {
            // Should be "where"
            let saved = p.pos;
            let ident = p.read_ident()?;
            if ident == "where" {
                break;
            } else {
                p.pos = saved;
                return Err(format!("expected '?' or 'where', got '{}'", ident));
            }
        }
    }

    if find_vars.is_empty() {
        return Err("find requires at least one variable".into());
    }

    // Parse where clauses
    let mut where_clauses = Vec::new();
    loop {
        p.skip_ws();
        if p.peek() != Some('?') {
            return Err(format!("expected '?' to start where clause, got {:?}", p.peek()));
        }
        p.advance();
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
                    Some(',') => { p.advance(); }
                    Some('}') => break,
                    other => return Err(format!("expected ',' or '}}' in where clause, got {:?}", other)),
                }
            }
        }
        p.expect('}')?;
        where_clauses.push(clause);

        // Check for more clauses (comma followed by ?) or end
        p.skip_ws();
        if p.peek() == Some(',') {
            // Peek ahead to see if next non-ws char is '?' (another clause)
            let saved = p.pos;
            p.advance(); // skip comma
            p.skip_ws();
            if p.peek() == Some('?') {
                continue;
            }
            // Not another clause, restore
            p.pos = saved;
        }
        break;
    }

    let mut query = serde_json::json!({
        "type": "query",
        "find": find_vars,
        "where": where_clauses,
    });

    // Check for as_of
    if p.try_keyword("as_of") {
        let tx_id = p.read_number()?;
        query["as_of"] = tx_id;
    }

    Ok(query)
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
                    let mod_str = if mods.is_empty() { String::new() } else { format!(" {}", mods.join(" ")) };
                    out.push_str(&format!("    {}: {}{}\n", fname, ftype, mod_str));
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
    print!(r#"datalog-cli - command-line client for datalog-db

USAGE:
  datalog-cli [--host HOST:PORT]              Interactive REPL
  datalog-cli [--host HOST:PORT] COMMAND      One-shot command

ONE-SHOT COMMANDS:
  status                                      Server status
  schema                                      List current schema
  query  'find ?x where ...'                  Run a query
  define 'Type {{ field: type, ... }}'          Define entity type
  assert 'Type {{ field: value, ... }}'         Assert entity
  retract 'Type #ID [field, ...]'             Retract fields
  retract 'Type #ID'                         Retract entire entity
  json   '{{...}}'                              Send raw JSON

DSL COMMANDS (REPL):
  status
  schema
  define User {{ name: string required, age: i64 }}
  define enum Status {{ Active, Suspended {{ reason: string }} }}
  assert User {{ name: "Alice", age: 30 }}
  assert User #42 {{ age: 31 }}
  retract User #42 [email, age]
  retract User #42
  find ?name, ?age where ?u: User {{ name: ?name, age: > 25 }}
  find ?n where ?u: User {{ name: ?n }} as_of 100
  json {{"type": "status"}}
  help
  quit

FIELD TYPES:
  string, i64, f64, bool, bytes, ref(Type), enum(EnumName)

FIELD MODIFIERS:
  required, unique, indexed

PATTERN OPERATORS (in find):
  ?var                Variable binding
  > N, < N, >= N      Comparisons
  <= N, != N
  "string"            Exact match
  #N                  Entity reference
"#);
}

// --- Main ---

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut host = "127.0.0.1:5557".to_string();
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
        eprintln!("Type 'help' for commands, 'quit' to exit.");
        repl(&mut client);
    } else {
        // One-shot mode
        let command = match remaining[0].as_str() {
            "status" | "schema" => remaining[0].clone(),
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
            Ok(payload) => {
                match client.send(payload) {
                    Ok(response) => print!("{}", format_response(&response)),
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("Parse error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn repl(client: &mut Client) {
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

        match line {
            "quit" | "exit" => break,
            "help" => { print_help(); continue; }
            _ => {}
        }

        match parse_command(line) {
            Ok(payload) => {
                match client.send(payload) {
                    Ok(response) => print!("{}", format_response(&response)),
                    Err(e) => {
                        eprintln!("Connection error: {}", e);
                        break;
                    }
                }
            }
            Err(e) => eprintln!("Parse error: {}", e),
        }
    }
}
