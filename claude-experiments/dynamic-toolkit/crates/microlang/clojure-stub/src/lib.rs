//! The live SECOND consumer. Its only job is to keep the library/language split
//! honest: it compiles against the exact same core public API as the Scheme
//! frontend, so if a core change bends toward Scheme, this stops compiling.
//!
//! It is deliberately skeletal — a Clojure-flavored `defn` desugaring on top of
//! the core reader. It does not need to be a real Clojure; it needs to be a
//! genuinely different frontend that pulls on the same interfaces.

use microlang::{CodeSpace, Runtime, Val, ValueModel};

pub fn run<M: ValueModel>(rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
    // The s-expression STRUCTURE reader is reusable core; only the surface
    // sugar is this frontend's concern.
    let forms = rt.read_all(src);
    let mut last = rt.encode(Val::Nil);
    for f in forms {
        let core = desugar(rt, f);
        last = rt.eval_top(cs, core);
    }
    last
}

fn sym<M: ValueModel>(rt: &mut Runtime<M>, n: &str) -> u64 {
    let s = rt.intern(n);
    rt.encode(Val::Sym(s))
}

fn desugar<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    // (defn name [params] body...) -> (def name (fn (params) body...))
    if let Some((h, _)) = rt.as_cons(form) {
        if let Val::Sym(s) = rt.decode(h) {
            if rt.sym_name(s) == "defn" {
                let items = rt.list_to_vec(form);
                let name = items[1];
                let params = items[2]; // `[params]` reads as a list via the core reader
                let mut fn_form = vec![sym(rt, "fn"), params];
                fn_form.extend_from_slice(&items[3..]);
                let lam = rt.vec_to_list(&fn_form);
                let def = sym(rt, "def");
                return rt.vec_to_list(&[def, name, lam]);
            }
        }
    }
    form
}
