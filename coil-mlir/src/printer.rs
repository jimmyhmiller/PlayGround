//! Printing `Val` back to coil source text. Round-trips structurally: reading
//! the output yields an equal `Val` (reader-macro forms print in their expanded
//! `(quote …)` shape, which re-reads to the same structure).

use crate::value::Val;
use std::fmt::Write;

pub fn print(val: &Val) -> String {
    let mut out = String::new();
    write_val(&mut out, val);
    out
}

fn write_val(out: &mut String, val: &Val) {
    match val {
        Val::Unit => out.push_str("#<unit>"),
        Val::Nil => out.push_str("nil"),
        Val::Bool(b) => write!(out, "{b}").unwrap(),
        Val::Int(n) => write!(out, "{n}").unwrap(),
        Val::Float(f) => {
            // Ensure a float always reads back as a float.
            if f.fract() == 0.0 && f.is_finite() {
                write!(out, "{f:.1}").unwrap();
            } else {
                write!(out, "{f}").unwrap();
            }
        }
        Val::Str(s) => write_string(out, s),
        Val::Sym(s) => out.push_str(s),
        Val::Keyword(s) => {
            out.push(':');
            out.push_str(s);
        }
        Val::TypeLit(s) => {
            out.push('!');
            out.push_str(s);
        }
        Val::AttrLit(s) => {
            out.push('#');
            out.push_str(s);
        }
        Val::List(items) => write_seq(out, items, '(', ')'),
        Val::Vec(items) => write_seq(out, items, '[', ']'),
        Val::Map(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                write_val(out, k);
                out.push(' ');
                write_val(out, v);
            }
            out.push('}');
        }
    }
}

fn write_seq(out: &mut String, items: &[Val], open: char, close: char) {
    out.push(open);
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        write_val(out, item);
    }
    out.push(close);
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            other => out.push(other),
        }
    }
    out.push('"');
}
