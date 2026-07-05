//! `coil cimport <header.h>` — generate Coil FFI bindings (externs + defstructs +
//! consts) from a C header by walking clang's JSON AST. A REAL header import via
//! clang's own parser (real C grammar, includes, typedef desugaring) — Coil NEVER
//! hand-rolls a C parser (that would be the hack: a partial/wrong parse → wrong
//! bindings → silent ABI corruption). THE CARDINAL: refuse unmappable constructs
//! with a clear skip — NEVER emit a silent-wrong binding (a wrong width/layout
//! silently corrupts the ABI).
//!
//! Phase 1: functions, scalars, pointers, simple structs.
//! Phase 2: typedefs (resolved through clang's desugaring), enums (the type → its
//! integer width; the constants → `const` defs), and object-like `#define`
//! constant macros (a second `clang -dM` pass — they aren't in the AST).
//! Unions: mapped to `:layout explicit` (every member `:at 0`) — coil's own
//! overlapping-storage representation, so a union is a REAL ABI-faithful binding
//! with named/typed member access, not a blob or a refusal. Bitfields are still
//! refused (faithful bit-offset import needs clang's `-fdump-record-layouts`).

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::process::{Command, Stdio};

/// Resolution tables gathered from the whole translation unit (system headers
/// included), so a typedef/enum *used* from the target header resolves even when
/// it's *defined* elsewhere (e.g. `size_t` from `<stddef.h>`).
#[derive(Default)]
struct Ctx {
    /// typedef name → its underlying clang `qualType` (resolved recursively).
    typedefs: HashMap<String, String>,
    /// enum tag name → the Coil integer type it lowers to (C enums are `int`
    /// unless a fixed underlying type is given).
    enums: HashMap<String, String>,
}

/// Parse `header` via clang and return a `.coil` bindings module.
pub fn cimport_header(header: &str) -> Result<String, String> {
    let ast = run_clang_ast(header)?;
    let empty = Vec::new();
    let top = ast["inner"].as_array().unwrap_or(&empty);

    // Pre-pass: build the typedef + enum-type tables from the WHOLE AST (not just
    // the target header), so resolution reaches definitions in system includes.
    let mut ctx = Ctx::default();
    collect_types(top, &mut ctx);

    let mut structs = String::new();
    let mut externs = String::new();
    let mut consts = String::new();
    let mut skipped: Vec<String> = Vec::new();
    // De-dup by name: C allows redeclaration, and clang lists a builtin libc
    // function (e.g. sqrt/labs) twice — emit ONE binding each.
    let mut seen: HashSet<String> = HashSet::new();
    // clang reports a node's source file only when it CHANGES; track it so we emit
    // ONLY this header's declarations and skip the flood of system-include decls.
    let mut from_header = false;

    for d in top {
        if let Some(file) = d.get("loc").and_then(|l| l.get("file")).and_then(|f| f.as_str()) {
            from_header = same_file(file, header);
        }
        if !from_header {
            continue;
        }
        match d["kind"].as_str() {
            Some("FunctionDecl") => {
                if !seen.insert(format!("fn:{}", name_of(d))) {
                    continue; // already emitted (redeclaration / builtin duplicate)
                }
                match function_extern(d, &ctx) {
                    Ok(s) => externs.push_str(&s),
                    Err(why) => skipped.push(format!("fn {} — {why}", name_of(d))),
                }
            }
            Some("RecordDecl") => {
                // Only a struct we actually emit (or refuse) claims its name in
                // `seen`. A forward declaration — common when a `typedef struct X
                // X;` precedes the definition — must NOT, or it would dedup away
                // the real definition that follows.
                match record_defstruct(d, &ctx) {
                    Ok(Some(s)) => {
                        if seen.insert(format!("struct:{}", name_of(d))) {
                            structs.push_str(&s);
                        }
                    }
                    Ok(None) => {} // forward decl / anonymous — nothing to emit
                    Err(why) => {
                        if seen.insert(format!("struct:{}", name_of(d))) {
                            skipped.push(format!("struct {} — {why}", name_of(d)));
                        }
                    }
                }
            }
            Some("EnumDecl") => match enum_consts(d, &mut seen) {
                Ok(s) => consts.push_str(&s),
                Err(why) => skipped.push(format!("enum {} — {why}", name_of(d))),
            },
            _ => {}
        }
    }

    // `#define` object-like constant macros — a separate preprocessor pass (they
    // never reach the AST). Refuse function-like / non-literal macros.
    match collect_defines(header) {
        Ok(defs) => {
            for (name, value) in defs {
                if seen.insert(format!("const:{name}")) {
                    consts.push_str(&format!("(const {name} {value})\n"));
                }
            }
        }
        Err(why) => skipped.push(format!("#define scan — {why}")),
    }

    // Surface refusals LOUDLY (stderr) and in the file — never a silent-wrong binding.
    for s in &skipped {
        eprintln!("cimport: SKIPPED (unmappable) — {s}");
    }
    let mut body = format!(
        "; Generated by `coil cimport {header}` — DO NOT EDIT.\n\
         ; A clang-parsed C binding (real header import). Audit before trusting the ABI.\n\
         (module cbindings)\n\n"
    );
    if !skipped.is_empty() {
        body.push_str("; SKIPPED (unmappable — left out rather than emit a wrong binding):\n");
        for s in &skipped {
            body.push_str(&format!(";   {s}\n"));
        }
        body.push('\n');
    }
    for (section, nonempty) in [(&consts, !consts.is_empty()), (&structs, !structs.is_empty())] {
        body.push_str(section);
        if nonempty {
            body.push('\n');
        }
    }
    body.push_str(&externs);
    Ok(body)
}

/// Run clang's JSON AST dump on `header`.
fn run_clang_ast(header: &str) -> Result<Value, String> {
    let out = Command::new("clang")
        .args(["-Xclang", "-ast-dump=json", "-fsyntax-only", "-x", "c", header])
        .output()
        .map_err(|e| format!("failed to run clang (is it installed?): {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "clang failed to parse '{header}':\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("parsing clang AST JSON: {e}"))
}

/// Gather typedef and enum-type resolution tables from every top-level decl.
fn collect_types(top: &[Value], ctx: &mut Ctx) {
    for d in top {
        match d["kind"].as_str() {
            Some("TypedefDecl") => {
                if let (Some(name), Some(under)) = (
                    d["name"].as_str(),
                    d["type"]["qualType"].as_str(),
                ) {
                    ctx.typedefs.insert(name.to_string(), under.to_string());
                }
            }
            Some("EnumDecl") => {
                if let Some(name) = d["name"].as_str() {
                    // C enums are `int`; a fixed underlying type (rare in C) overrides.
                    let ty = d["fixedUnderlyingType"]["qualType"]
                        .as_str()
                        .and_then(|q| map_type(q, ctx).ok())
                        .unwrap_or_else(|| "i32".to_string());
                    ctx.enums.insert(name.to_string(), ty);
                }
            }
            _ => {}
        }
    }
}

fn name_of(d: &Value) -> String {
    d["name"].as_str().unwrap_or("<anonymous>").to_string()
}

/// clang echoes the path we passed for the target header; system includes differ.
/// Match on the path or its file name (clang may normalize the path).
fn same_file(file: &str, header: &str) -> bool {
    use std::path::Path;
    file == header || Path::new(file).file_name() == Path::new(header).file_name()
}

fn function_extern(d: &Value, ctx: &Ctx) -> Result<String, String> {
    let name = d["name"].as_str().ok_or("no name")?;
    let empty = Vec::new();
    let mut params = Vec::new();
    for p in d["inner"].as_array().unwrap_or(&empty) {
        if p["kind"].as_str() == Some("ParmVarDecl") {
            let ty = p["type"]["qualType"].as_str().ok_or("a parameter has no type")?;
            params.push(map_type(ty, ctx)?);
        }
    }
    // The function's qualType is "RET (PARAMS)"; the return type is everything
    // before the first '(' (correct for ordinary functions; fn-pointer-returning
    // ones are refused below when their return fails to map).
    let qt = d["type"]["qualType"].as_str().ok_or("no function type")?;
    let ret = map_return(qt.split('(').next().unwrap_or("").trim(), ctx)?;
    let mut ps = params.join(" ");
    if qt.contains("...") {
        if !ps.is_empty() {
            ps.push(' ');
        }
        ps.push_str("...");
    }
    Ok(format!("(extern {name} :cc c [{ps}] (-> {ret}))\n"))
}

fn record_defstruct(d: &Value, ctx: &Ctx) -> Result<Option<String>, String> {
    // A C union is OVERLAPPING storage: every member sits at offset 0, sizeof is
    // the widest member (rounded to the max alignment). Coil expresses exactly this
    // with `:layout explicit` — every field `:at 0` — so a union maps to a REAL,
    // ABI-faithful binding with named + typed member access (not a blob, not a
    // refusal). See examples/explicit-layout.coil: "overlapping offsets make a union".
    let is_union = d.get("tagUsed").and_then(|t| t.as_str()) == Some("union");
    let name = match d["name"].as_str() {
        Some(n) => n,
        None => return Ok(None), // anonymous struct/union — Phase 2+
    };
    if d.get("completeDefinition").and_then(|c| c.as_bool()) != Some(true) {
        return Ok(None); // forward declaration — nothing to emit
    }
    let empty = Vec::new();
    let mut fields = Vec::new();
    // Union layout is computed from the members (standard C rules): sizeof =
    // round_up(max member size, max member align), align = max member align.
    let mut max_size: i64 = 0;
    let mut max_align: i64 = 1;
    for f in d["inner"].as_array().unwrap_or(&empty) {
        if f["kind"].as_str() == Some("FieldDecl") {
            // A bitfield packs into sub-byte bit ranges — emitting it as a
            // full-width field would corrupt the layout (silent-wrong). Refuse it
            // (faithful bitfield import needs clang's record-layout — next stage).
            if f.get("isBitfield").and_then(|b| b.as_bool()) == Some(true) {
                return Err(if is_union {
                    "union with a bitfield member (Phase 2+ — needs record layout)".to_string()
                } else {
                    "contains a bitfield (packed layout — Phase 2+ — needs record layout)".to_string()
                });
            }
            let fname = f["name"].as_str().ok_or("an unnamed field (anon struct/union)")?;
            let fty = f["type"]["qualType"].as_str().ok_or("a field has no type")?;
            let mapped = map_type(fty, ctx)?;
            if is_union {
                // We compute the union's layout ourselves, so we must know each
                // member's size/align. Scalars + pointers: known. Aggregate members
                // (nested struct/union/array) need clang's record layout — refuse
                // rather than guess (the cardinal: never a wrong layout).
                let (sz, al) = scalar_size_align(&mapped).ok_or_else(|| {
                    format!("union member '{fname}' has aggregate type '{mapped}' (Phase 2 — needs record layout)")
                })?;
                if sz > max_size {
                    max_size = sz;
                }
                if al > max_align {
                    max_align = al;
                }
                fields.push(format!("({fname} {mapped} :at 0)"));
            } else {
                fields.push(format!("({fname} {mapped})"));
            }
        }
    }
    if is_union {
        let size = ((max_size + max_align - 1) / max_align) * max_align;
        Ok(Some(format!(
            "(defstruct {name} :layout explicit :size {size} :align {max_align} [{}])\n",
            fields.join(" ")
        )))
    } else {
        Ok(Some(format!("(defstruct {name} [{}])\n", fields.join(" "))))
    }
}

/// Size and alignment (bytes) of a mapped Coil scalar/pointer type, for computing
/// union layout. `None` for aggregate/unknown types (whose layout we don't compute
/// — those unions are refused rather than mis-bound).
fn scalar_size_align(t: &str) -> Option<(i64, i64)> {
    if t.starts_with("(ptr") || t.starts_with("(fnptr") {
        return Some((8, 8)); // all target pointers are 8 bytes
    }
    match t {
        "i8" | "u8" => Some((1, 1)),
        "i16" | "u16" => Some((2, 2)),
        "i32" | "u32" | "f32" => Some((4, 4)),
        "i64" | "u64" | "f64" => Some((8, 8)),
        _ => None,
    }
}

/// Emit `(const NAME VALUE)` for each enumerator. Values follow C rules: an
/// explicit initializer sets the value, an implicit one is the previous + 1.
fn enum_consts(d: &Value, seen: &mut HashSet<String>) -> Result<String, String> {
    let empty = Vec::new();
    let mut out = String::new();
    let mut next: i64 = 0;
    for c in d["inner"].as_array().unwrap_or(&empty) {
        if c["kind"].as_str() != Some("EnumConstantDecl") {
            continue;
        }
        let name = c["name"].as_str().ok_or("an unnamed enumerator")?;
        // An explicit initializer evaluates to a ConstantExpr carrying the folded
        // integer `value`; an implicit one continues the running counter.
        let value = match enumerator_value(c) {
            Some(v) => v,
            None if c.get("inner").is_some() => {
                // had an initializer we couldn't fold to an integer — refuse the
                // whole enum rather than guess a wrong value.
                return Err(format!("enumerator '{name}' has a non-integer value"));
            }
            None => next,
        };
        next = value + 1;
        if seen.insert(format!("const:{name}")) {
            out.push_str(&format!("(const {name} {value})\n"));
        }
    }
    Ok(out)
}

/// The folded integer value of an enumerator's explicit initializer, if any.
fn enumerator_value(c: &Value) -> Option<i64> {
    fn find(node: &Value) -> Option<i64> {
        if let Some(v) = node.get("value").and_then(|v| v.as_str()) {
            if let Ok(n) = v.parse::<i64>() {
                return Some(n);
            }
        }
        for child in node.get("inner").and_then(|i| i.as_array())? {
            if let Some(n) = find(child) {
                return Some(n);
            }
        }
        None
    }
    // Only the initializer subtree counts (an EnumConstantDecl with no inner is
    // implicit); scan it for the first foldable integer value.
    let inner = c.get("inner")?.as_array()?;
    inner.iter().find_map(find)
}

/// Return-type mapping (allows `void`, which `map_type` rejects as a value type).
fn map_return(c: &str, ctx: &Ctx) -> Result<String, String> {
    if normalize(c) == "void" {
        return Ok("void".to_string());
    }
    map_type(c, ctx)
}

/// Map a C type string (clang `qualType`) to a Coil type — ABI-faithful, refusing
/// what it can't map correctly (the cardinal: never guess a binding). Typedefs
/// resolve through `ctx`; enums map to their integer width.
fn map_type(c: &str, ctx: &Ctx) -> Result<String, String> {
    map_type_d(c, ctx, 0)
}

fn map_type_d(c: &str, ctx: &Ctx, depth: usize) -> Result<String, String> {
    if depth > 64 {
        return Err(format!("typedef chain too deep resolving '{c}' (cycle?)"));
    }
    let s = normalize(c);
    if let Some(base) = s.strip_suffix('*') {
        let base = base.trim();
        if base == "void" {
            return Ok("(ptr i8)".to_string()); // opaque pointer convention
        }
        if base.contains('(') {
            return Err(format!("function-pointer type '{c}' (Phase 2+)"));
        }
        // `struct X *` etc. recurse; an unmappable pointee is refused upward.
        return Ok(format!("(ptr {})", map_type_d(base, ctx, depth + 1)?));
    }
    let mapped = match s.as_str() {
        "char" | "signed char" => "i8",
        "unsigned char" => "u8",
        "short" | "short int" => "i16",
        "unsigned short" | "unsigned short int" => "u16",
        "int" | "signed int" | "signed" => "i32",
        "unsigned int" | "unsigned" => "u32",
        "long" | "long int" => "i64",
        "unsigned long" | "unsigned long int" => "u64",
        "long long" | "long long int" => "i64",
        "unsigned long long" | "unsigned long long int" => "u64",
        "float" => "f32",
        "double" => "f64",
        // C `_Bool` is a 1-byte value at the ABI; map to `u8` (unambiguous) rather
        // than Coil's `bool` (i1), whose C-ABI width would be a guess.
        "_Bool" | "bool" => "u8",
        other if other.starts_with("struct ") => {
            return Ok(other.trim_start_matches("struct ").trim().to_string());
        }
        // A `union X` names an emitted `:layout explicit` binding (see
        // record_defstruct) — resolve to its name, exactly like `struct X`.
        other if other.starts_with("union ") => {
            return Ok(other.trim_start_matches("union ").trim().to_string());
        }
        other if other.starts_with("enum ") => {
            let tag = other.trim_start_matches("enum ").trim();
            // A known enum uses its width; an undeclared/forward enum is still `int`.
            return Ok(ctx.enums.get(tag).cloned().unwrap_or_else(|| "i32".to_string()));
        }
        // A typedef name resolves to its underlying type (recursively).
        other if ctx.typedefs.contains_key(other) => {
            return map_type_d(&ctx.typedefs[other], ctx, depth + 1);
        }
        // `void` as a value/param type, unions, function pointers, attribute'd
        // (e.g. SIMD-vector) builtins, `long double`, etc. — NOT guessed.
        other => return Err(format!("unsupported C type '{other}'")),
    };
    Ok(mapped.to_string())
}

fn normalize(c: &str) -> String {
    let mut s = c.trim().to_string();
    for q in ["const ", "volatile ", "restrict "] {
        s = s.replace(q, "");
    }
    s.trim().to_string()
}

/// Object-like `#define` constant macros from the header, as `(name, value)`
/// pairs ready to emit as `(const name value)`. Function-like macros and
/// non-literal bodies are skipped (refuse-not-guess). System/builtin macros are
/// excluded by diffing against an empty-translation-unit baseline.
fn collect_defines(header: &str) -> Result<Vec<(String, String)>, String> {
    let baseline: HashSet<String> = clang_dump_macros(&["-dM", "-E", "-x", "c", "-"], true)?
        .lines()
        .filter_map(|l| parse_define(l).map(|(n, _)| n))
        .collect();
    let mut out = Vec::new();
    for line in clang_dump_macros(&["-dM", "-E", "-x", "c", header], false)?.lines() {
        if let Some((name, body)) = parse_define(line) {
            if baseline.contains(&name) {
                continue; // a builtin / system macro, not from this header
            }
            if let Some(value) = c_literal(&body) {
                out.push((name, value));
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn clang_dump_macros(args: &[&str], null_stdin: bool) -> Result<String, String> {
    let mut cmd = Command::new("clang");
    cmd.args(args);
    if null_stdin {
        cmd.stdin(Stdio::null());
    }
    let out = cmd.output().map_err(|e| format!("failed to run clang -dM: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "clang -dM failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parse a `#define` line into `(name, body)`. Returns `None` for a function-like
/// macro (`NAME(...)` with no space before `(`) or a malformed line.
fn parse_define(line: &str) -> Option<(String, String)> {
    let rest = line.strip_prefix("#define ")?;
    let name_len = rest
        .find(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
        .unwrap_or(rest.len());
    let (name, after) = rest.split_at(name_len);
    if name.is_empty() {
        return None;
    }
    if after.starts_with('(') {
        return None; // function-like macro — not a constant
    }
    Some((name.to_string(), after.trim().to_string()))
}

/// Interpret a `#define` body as a Coil constant literal, if it's a plain integer
/// or float literal. Returns the Coil source text to emit, or `None` to skip
/// (strings, identifiers, expressions — anything we won't evaluate).
fn c_literal(body: &str) -> Option<String> {
    let b = body.trim();
    if b.is_empty() {
        return None;
    }
    // A float literal: a decimal point or exponent, no hex prefix. Emit normalized
    // text that is unambiguously a Coil float (force a '.' if formatting drops it).
    let lower = b.to_ascii_lowercase();
    let looks_float = !lower.starts_with("0x")
        && (b.contains('.') || lower.contains('e'))
        && lower.trim_end_matches(['f', 'l']).parse::<f64>().is_ok();
    if looks_float {
        let x: f64 = lower.trim_end_matches(['f', 'l']).parse().ok()?;
        let mut s = format!("{x}");
        if !s.contains('.') && !s.contains('e') {
            s.push_str(".0");
        }
        return Some(s);
    }
    // An integer literal: optional sign, decimal / 0x-hex / 0-octal, with C width
    // suffixes (u/l) stripped. Emit the decimal value.
    let (neg, digits) = match b.strip_prefix('-') {
        Some(rest) => (true, rest.trim_start()),
        None => (false, b.strip_prefix('+').unwrap_or(b)),
    };
    let digits = digits.trim_end_matches(['u', 'U', 'l', 'L']);
    let mag: i64 = if let Some(hex) = digits.strip_prefix("0x").or_else(|| digits.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()?
    } else if digits.len() > 1 && digits.starts_with('0') && digits.bytes().all(|c| (b'0'..=b'7').contains(&c)) {
        i64::from_str_radix(digits, 8).ok()?
    } else {
        digits.parse::<i64>().ok()?
    };
    Some(format!("{}", if neg { -mag } else { mag }))
}
