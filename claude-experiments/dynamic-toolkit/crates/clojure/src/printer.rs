//! Value → text. Used by the REPL and by tests.

use crate::collections::{
    array_count, array_get, is_array, is_keyword, is_list, is_map, is_set, is_string,
    is_var, is_vector, keyword_sym_id, set_backing, string_bytes, vector_iter,
};
use crate::namespace::{map_count, map_entry, var_sym};
use crate::symbols::SymbolTable;
use crate::value::*;

pub fn print(v: u64, sym: &SymbolTable) -> String {
    let mut out = String::new();
    write(&mut out, v, sym);
    out
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
        out.push_str(sym.name(as_sym_id(v)));
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
            out.push_str(sym.name(keyword_sym_id(v)));
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
                out.push_str(sym.name(as_sym_id(s)));
            } else {
                out.push_str(&format!("0x{:016x}", v));
            }
        } else {
            out.push_str(&format!("#<obj 0x{:016x}>", v));
        }
    } else {
        out.push_str(&format!("#<unknown 0x{:016x}>", v));
    }
}
