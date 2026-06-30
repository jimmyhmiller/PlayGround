//! Canonical normalization of textual LLVM IR (the `coil emit-ir` output), used
//! by the `coil dump-ir` oracle for the self-host codegen gate.
//!
//! Why this exists: `coil emit-ir` is NOT byte-stable run-to-run. Codegen
//! registers globals and emits functions in hash-map iteration order, so the
//! POSITIONAL numbering of anonymous globals (`@g.N`, `@str.N`, `@cstr.N`,
//! `@__coil_llvm_ir_N`), the numbering of LLVM attribute groups (`#N`), and the
//! top-level emission order of types/globals/declares/defines all vary between
//! runs — while the actual content is identical. A byte-diff gate is therefore
//! impossible on the raw text. `normalize` cancels exactly that incidental
//! variation and nothing else, so the diff is on STRUCTURE:
//!   1. attribute groups renumbered by content (sorted),
//!   2. positional globals renumbered per-family by first-encounter in a
//!      canonical (function-name-sorted) walk,
//!   3. top-level blocks sorted into a fixed category order, then by symbol name.
//!
//! This is the SAME normalization the self-host codegen output must pass through,
//! so both sides are compared on equal footing. It is a pure text transform — it
//! never changes which instructions/operands/types appear, only incidental names
//! and order. Verified to make every codegen-emitting corpus file byte-stable
//! across repeated `emit-ir` runs.

/// Normalize textual LLVM IR into a canonical, run-stable form.
pub fn normalize(text: &str) -> String {
    let text = renumber_attrs(text);
    let (header, blocks) = parse_blocks(&text);
    let mapping = build_renumber(&blocks);
    // Apply the positional-global renumbering to every block (name + body).
    let mut nblocks: Vec<(u8, String, Vec<String>)> = blocks
        .into_iter()
        .map(|(cat, name, body)| {
            let nname = apply_renumber(&name, &mapping);
            let nbody = body.iter().map(|l| apply_renumber(l, &mapping)).collect();
            (cat, nname, nbody)
        })
        .collect();
    nblocks.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    let mut out = String::new();
    for l in &header {
        out.push_str(l);
        out.push('\n');
    }
    out.push('\n');
    for (_, _, body) in &nblocks {
        for l in body {
            out.push_str(l);
            out.push('\n');
        }
        out.push('\n');
    }
    // Match the Python prototype: single trailing newline.
    while out.ends_with("\n\n") {
        out.pop();
    }
    out
}

// ---- attribute groups: renumber `#N` by content (sorted by body text) --------

fn renumber_attrs(text: &str) -> String {
    // Collect `attributes #N = {BODY}` definitions.
    let mut defs: Vec<(u64, String)> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("attributes #") {
            if let Some(eq) = rest.find(" = ") {
                if let Ok(n) = rest[..eq].parse::<u64>() {
                    defs.push((n, rest[eq + 3..].to_string()));
                }
            }
        }
    }
    if defs.is_empty() {
        return text.to_string();
    }
    // Canonical order = sort by body content.
    let mut order: Vec<&(u64, String)> = defs.iter().collect();
    order.sort_by(|a, b| a.1.cmp(&b.1));
    // remap[old] = new
    let mut remap: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for (new, (old, _)) in order.iter().enumerate() {
        remap.insert(*old, new as u64);
    }
    // Replace every `#<digits>` token (word-bounded).
    replace_attr_refs(text, &remap)
}

fn replace_attr_refs(text: &str, remap: &std::collections::HashMap<u64, u64>) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'#' && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            // word boundary: next char must not be an identifier char
            let boundary = j >= b.len() || !is_ident_byte(b[j]);
            if boundary {
                let n: u64 = text[i + 1..j].parse().unwrap();
                if let Some(new) = remap.get(&n) {
                    out.push('#');
                    out.push_str(&new.to_string());
                    i = j;
                    continue;
                }
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

// ---- block parsing -----------------------------------------------------------

const CAT_TYPE: u8 = 0;
const CAT_GLOBAL: u8 = 1;
const CAT_DECLARE: u8 = 2;
const CAT_DEFINE: u8 = 3;
const CAT_MISC: u8 = 4;

/// Returns (header lines, blocks). Each block is (category, sort-name, lines).
fn parse_blocks(text: &str) -> (Vec<String>, Vec<(u8, String, Vec<String>)>) {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut i = 0;
    let mut header = Vec::new();
    while i < lines.len() && !lines[i].is_empty() {
        header.push(lines[i].to_string());
        i += 1;
    }
    let mut blocks = Vec::new();
    while i < lines.len() {
        let l = lines[i];
        if l.is_empty() {
            i += 1;
            continue;
        }
        if l.starts_with('%') && l.contains("= type") {
            let name = first_token(l);
            blocks.push((CAT_TYPE, name, vec![l.to_string()]));
            i += 1;
        } else if l.starts_with('@') {
            let name = first_token(l);
            blocks.push((CAT_GLOBAL, name, vec![l.to_string()]));
            i += 1;
        } else if l.starts_with("declare") {
            let name = at_symbol(l).unwrap_or_else(|| l.to_string());
            blocks.push((CAT_DECLARE, name, vec![l.to_string()]));
            i += 1;
        } else if l.starts_with("define") {
            let name = at_symbol(l).unwrap_or_else(|| l.to_string());
            let mut body = vec![l.to_string()];
            i += 1;
            while i < lines.len() && lines[i] != "}" {
                body.push(lines[i].to_string());
                i += 1;
            }
            if i < lines.len() {
                body.push(lines[i].to_string()); // the closing }
                i += 1;
            }
            blocks.push((CAT_DEFINE, name, body));
        } else {
            blocks.push((CAT_MISC, l.to_string(), vec![l.to_string()]));
            i += 1;
        }
    }
    (header, blocks)
}

fn first_token(l: &str) -> String {
    l.split(' ').next().unwrap_or(l).to_string()
}

/// First `@symbol` in a line: `@"quoted"` or `@word` (chars `[A-Za-z0-9_.$-]`).
fn at_symbol(l: &str) -> Option<String> {
    let b = l.as_bytes();
    let at = l.find('@')?;
    let start = at;
    let mut j = at + 1;
    if j < b.len() && b[j] == b'"' {
        j += 1;
        while j < b.len() && b[j] != b'"' {
            j += 1;
        }
        if j < b.len() {
            j += 1; // closing quote
        }
    } else {
        while j < b.len() && is_sym_byte(b[j]) {
            j += 1;
        }
    }
    Some(l[start..j].to_string())
}

fn is_sym_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'$' | b'-')
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

// ---- positional-global renumbering ------------------------------------------

/// A positional global token starting at byte `i` (which must be `@`):
/// `@g.N` `@str.N` `@cstr.N` (family) or `@__coil_llvm_ir_N` (llvmir). Returns
/// (family, old-number, byte-index just past the number).
fn scan_positional(s: &str, i: usize) -> Option<(&'static str, u64, usize)> {
    let b = s.as_bytes();
    if b[i] != b'@' {
        return None;
    }
    let rest = &s[i + 1..];
    let families: [(&str, &'static str); 3] =
        [("g.", "g"), ("str.", "str"), ("cstr.", "cstr")];
    for (pre, fam) in families {
        if let Some(after) = rest.strip_prefix(pre) {
            if let Some((num, len)) = leading_number(after) {
                let end = i + 1 + pre.len() + len;
                if end >= b.len() || !is_ident_byte(b[end]) {
                    return Some((fam, num, end));
                }
            }
        }
    }
    if let Some(after) = rest.strip_prefix("__coil_llvm_ir_") {
        if let Some((num, len)) = leading_number(after) {
            let end = i + 1 + "__coil_llvm_ir_".len() + len;
            if end >= b.len() || !is_ident_byte(b[end]) {
                return Some(("llvmir", num, end));
            }
        }
    }
    None
}

fn leading_number(s: &str) -> Option<(u64, usize)> {
    let b = s.as_bytes();
    let mut j = 0;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    if j == 0 {
        return None;
    }
    Some((s[..j].parse().ok()?, j))
}

type RenumMap = std::collections::HashMap<(&'static str, u64), u64>;

fn is_pos_define_name(name: &str) -> bool {
    name.strip_prefix("@__coil_llvm_ir_")
        .map(|d| !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit()))
        .unwrap_or(false)
}

/// Build the per-family canonical renumbering by walking define bodies in a
/// canonical order: stable-named defines sorted by name first (so the
/// `@__coil_llvm_ir_N` helpers get canonical numbers by first call-site), then
/// the positional defines in that canonical order, then global definitions.
fn build_renumber(blocks: &[(u8, String, Vec<String>)]) -> RenumMap {
    let mut counters: std::collections::HashMap<&'static str, u64> =
        std::collections::HashMap::new();
    let mut mapping: RenumMap = std::collections::HashMap::new();
    let assign = |line: &str, mapping: &mut RenumMap, counters: &mut std::collections::HashMap<&'static str, u64>| {
        let mut i = 0;
        let bl = line.as_bytes();
        while i < bl.len() {
            if bl[i] == b'@' {
                if let Some((fam, old, end)) = scan_positional(line, i) {
                    mapping.entry((fam, old)).or_insert_with(|| {
                        let c = counters.entry(fam).or_insert(0);
                        let v = *c;
                        *c += 1;
                        v
                    });
                    i = end;
                    continue;
                }
            }
            i += 1;
        }
    };

    // 1. stable-named defines, sorted by name.
    let mut stable: Vec<&(u8, String, Vec<String>)> = blocks
        .iter()
        .filter(|(c, n, _)| *c == CAT_DEFINE && !is_pos_define_name(n))
        .collect();
    stable.sort_by(|a, b| a.1.cmp(&b.1));
    for (_, _, body) in &stable {
        for l in body.iter() {
            assign(l, &mut mapping, &mut counters);
        }
    }
    // 2. positional (llvmir) defines, ordered by their now-canonical number.
    let mut pos: Vec<&(u8, String, Vec<String>)> = blocks
        .iter()
        .filter(|(c, n, _)| *c == CAT_DEFINE && is_pos_define_name(n))
        .collect();
    let pnum = |n: &str| -> u64 {
        let d: u64 = n["@__coil_llvm_ir_".len()..].parse().unwrap();
        *mapping.get(&("llvmir", d)).unwrap_or(&u64::MAX)
    };
    pos.sort_by_key(|b| pnum(&b.1));
    for (_, _, body) in &pos {
        for l in body.iter() {
            assign(l, &mut mapping, &mut counters);
        }
    }
    // 3. global definition lines (pick up any not referenced in a body).
    for (c, _, body) in blocks {
        if *c == CAT_GLOBAL {
            for l in body.iter() {
                assign(l, &mut mapping, &mut counters);
            }
        }
    }
    mapping
}

fn apply_renumber(line: &str, mapping: &RenumMap) -> String {
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'@' {
            if let Some((fam, old, end)) = scan_positional(line, i) {
                if let Some(new) = mapping.get(&(fam, old)) {
                    if fam == "llvmir" {
                        out.push_str("@__coil_llvm_ir_");
                    } else {
                        out.push('@');
                        out.push_str(fam);
                        out.push('.');
                    }
                    out.push_str(&new.to_string());
                    i = end;
                    continue;
                }
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}
