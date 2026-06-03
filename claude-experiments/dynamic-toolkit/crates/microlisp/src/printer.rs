//! cons-tree → text. Used by the REPL and by tests checking macroexpansion.

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
        out.push_str("#t");
    } else if v == FALSE {
        out.push_str("#f");
    } else if is_number(v) {
        let n = as_number(v);
        if n.fract() == 0.0 && n.is_finite() && n.abs() < 1e16 {
            out.push_str(&format!("{}", n as i64));
        } else {
            out.push_str(&format!("{}", n));
        }
    } else if is_symbol(v) {
        out.push_str(sym.name(as_symbol_id(v)));
    } else if is_cons(v) {
        out.push('(');
        write_list_body(out, v, sym);
        out.push(')');
    } else {
        out.push_str(&format!("#<obj 0x{:016x}>", v));
    }
}

fn write_list_body(out: &mut String, mut v: u64, sym: &SymbolTable) {
    let mut first = true;
    while is_cons(v) {
        if !first {
            out.push(' ');
        }
        write(out, car(v), sym);
        first = false;
        v = cdr(v);
    }
    if !is_nil(v) {
        out.push_str(" . ");
        write(out, v, sym);
    }
}
