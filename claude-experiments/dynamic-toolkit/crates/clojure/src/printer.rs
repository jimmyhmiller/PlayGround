//! Value → text. Used by the REPL and by tests.

use crate::collections::{
    array_count, array_get, is_array, is_keyword, is_list, is_map, is_record, is_set,
    is_string, is_var, is_vector, keyword_sym_id, record_field, record_field_count,
    record_type_name, set_backing, string_bytes, vector_iter,
};
use crate::namespace::{map_count, map_entry, var_sym};
use crate::symbols::SymbolTable;
use crate::value::*;

pub fn print(v: u64, sym: &SymbolTable) -> String {
    let mut out = String::new();
    write(&mut out, v, sym);
    out
}

/// Records get printed by asking the protocol-membership table what
/// the receiver claims:
///   - `IVector` → `[a b c]`
///   - `IMap` → `{k v k v}` (per-MapEntry walked via -nth 0/1)
///   - `ISeq`/`ISeqable` → `(a b c)` walked via `seq_iter`
///   - anything else → debug form `#TypeName{:f v :g w}`.
///
/// We never special-case PList/Cons/PersistentVector/MapEntry by
/// name — `seq_iter` already dispatches through the protocol for any
/// non-built-in shape, so a future deftype that implements ISeq
/// prints correctly without touching this function.
fn write_record(out: &mut String, v: u64, sym: &SymbolTable) {
    let type_sym_v = record_type_name(v);
    if !is_sym_id(type_sym_v) {
        out.push_str(&format!("#<record 0x{:016x}>", v));
        return;
    }
    let (ivector, imap, iseq) = crate::host::with_host(|h| {
        (h.ivector_sym, h.imap_sym, h.iseq_sym)
    });
    if crate::protocol::type_satisfies(ivector, v) {
        out.push('[');
        let mut first = true;
        for x in crate::collections::seq_iter(v) {
            if !first { out.push(' '); }
            first = false;
            write(out, x, sym);
        }
        out.push(']');
        return;
    }
    if crate::protocol::type_satisfies(imap, v) {
        // MapEntry-shaped seq: each step yields a 2-element seq we
        // walk to read [k v]. Same protocol path as above; we just
        // emit `{k v k v}` punctuation.
        out.push('{');
        let mut first = true;
        for entry in crate::collections::seq_iter(v) {
            if !first { out.push_str(", "); }
            first = false;
            let mut iter = crate::collections::seq_iter(entry);
            let k = iter.next().unwrap_or(NIL);
            let val = iter.next().unwrap_or(NIL);
            write(out, k, sym);
            out.push(' ');
            write(out, val, sym);
        }
        out.push('}');
        return;
    }
    if crate::protocol::type_satisfies(iseq, v) {
        out.push('(');
        let mut first = true;
        for x in crate::collections::seq_iter(v) {
            if !first { out.push(' '); }
            first = false;
            write(out, x, sym);
        }
        out.push(')');
        return;
    }
    let type_id = as_sym_id(type_sym_v);
    let name = sym.name(type_id);
    write_record_debug(out, v, sym, &name);
}

fn write_record_debug(out: &mut String, v: u64, sym: &SymbolTable, name: &str) {
    out.push('#');
    out.push_str(name);
    out.push('{');
    let n = record_field_count(v);
    let fields = crate::host::with_host(|h| {
        let map = h.deftype_fields.lock().unwrap();
        let type_id = as_sym_id(record_type_name(v));
        map.get(&type_id).cloned()
    });
    for i in 0..n {
        if i > 0 { out.push(' '); }
        if let Some(ref fs) = fields {
            if let Some(&fid) = fs.get(i) {
                out.push(':');
                out.push_str(&sym.name(fid));
                out.push(' ');
            }
        }
        write(out, record_field(v, i), sym);
    }
    out.push('}');
}

/// Like `print`, but uses Clojure's `(str x)` semantics: strings are
/// written bare (no surrounding quotes / no escape sequences) and nil
/// becomes the empty string. Used by the runtime `__str_concat` extern
/// that core.clj's `defn str` calls under the hood.
pub fn str_repr(v: u64, sym: &SymbolTable) -> String {
    if is_nil(v) {
        return String::new();
    }
    if is_ptr(v) && crate::collections::is_string(v) {
        let bytes = crate::collections::string_bytes(v);
        return String::from_utf8_lossy(bytes).into_owned();
    }
    print(v, sym)
}

fn write(out: &mut String, v: u64, sym: &SymbolTable) {
    if is_nil(v) {
        out.push_str("nil");
    } else if v == TRUE {
        out.push_str("true");
    } else if v == FALSE {
        out.push_str("false");
    } else if is_number(v) {
        let n = as_number(v);
        if n.fract() == 0.0 && n.is_finite() && n.abs() < 1e16 {
            out.push_str(&format!("{}", n as i64));
        } else {
            out.push_str(&format!("{}", n));
        }
    } else if is_sym_id(v) {
        out.push_str(&sym.name(as_sym_id(v)));
    } else if is_ptr(v) {
        if is_string(v) {
            let bytes = string_bytes(v);
            out.push('"');
            for &b in bytes {
                match b {
                    b'"' => out.push_str("\\\""),
                    b'\\' => out.push_str("\\\\"),
                    b'\n' => out.push_str("\\n"),
                    b'\t' => out.push_str("\\t"),
                    b'\r' => out.push_str("\\r"),
                    c if (0x20..=0x7e).contains(&c) => out.push(c as char),
                    c => out.push_str(&format!("\\x{:02x}", c)),
                }
            }
            out.push('"');
        } else if is_keyword(v) {
            out.push(':');
            out.push_str(&sym.name(keyword_sym_id(v)));
        } else if is_list(v) {
            out.push('(');
            let mut first = true;
            for x in list_iter(v) {
                if !first {
                    out.push(' ');
                }
                first = false;
                write(out, x, sym);
            }
            out.push(')');
        } else if is_vector(v) {
            out.push('[');
            let mut first = true;
            for x in vector_iter(v) {
                if !first {
                    out.push(' ');
                }
                first = false;
                write(out, x, sym);
            }
            out.push(']');
        } else if is_map(v) {
            out.push('{');
            let n = map_count(v) as usize;
            for i in 0..n {
                if i > 0 {
                    out.push_str(", ");
                }
                let (k, val) = map_entry(v, i);
                write(out, k, sym);
                out.push(' ');
                write(out, val, sym);
            }
            out.push('}');
        } else if is_set(v) {
            out.push_str("#{");
            let backing = set_backing(v);
            let mut first = true;
            for x in vector_iter(backing) {
                if !first {
                    out.push(' ');
                }
                first = false;
                write(out, x, sym);
            }
            out.push('}');
        } else if is_array(v) {
            // Mutable native array — print with an explicit prefix so
            // it's distinguishable from a persistent vector at the REPL.
            out.push_str("#array[");
            let n = array_count(v);
            for i in 0..n {
                if i > 0 { out.push(' '); }
                write(out, array_get(v, i), sym);
            }
            out.push(']');
        } else if is_var(v) {
            out.push_str("#'");
            let s = var_sym(v);
            if is_sym_id(s) {
                out.push_str(&sym.name(as_sym_id(s)));
            } else {
                out.push_str(&format!("0x{:016x}", v));
            }
        } else if is_record(v) {
            write_record(out, v, sym);
        } else {
            out.push_str(&format!("#<obj 0x{:016x}>", v));
        }
    } else {
        out.push_str(&format!("#<unknown 0x{:016x}>", v));
    }
}
