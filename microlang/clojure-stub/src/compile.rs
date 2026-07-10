//! Clojure -> `Ir`, DIRECTLY. This frontend does NOT ride the toolkit's s-expr
//! `analyze`/special-forms — it compiles fully-expanded Clojure forms to the
//! toolkit's neutral `Ir` itself, owning name resolution, its own core forms
//! (`fn`/`let`/`if`/`do`/`def`/`quote`/`set!`/`defmethod`), prim mapping, and
//! protocol dispatch. The toolkit's job starts at `Ir`.

use std::collections::HashMap;
use std::rc::Rc;

use microlang::ir::{Ir, Prim};
use microlang::value::Sym;
use microlang::{Runtime, Val, ValueModel};

/// Compile-time state that persists across top-level forms.
pub struct Compiler {
    /// Lexical scope, innermost frame last (like `analyze`'s, but frontend-owned).
    scope: Vec<Vec<Sym>>,
    /// Protocol method names seen so far -> a call to one is a dispatch site.
    methods: std::collections::HashSet<Sym>,
    /// Monotonic dispatch-site id.
    site: usize,
    /// The names THIS frontend treats as primitives (its choice, not the
    /// toolkit's). User-facing `+`/`first`/… are ordinary globals in
    /// `clojure.core`; only these low-level names map to a `Prim`.
    prims: HashMap<Sym, Prim>,
}

impl Compiler {
    pub fn new<M: ValueModel>(rt: &mut Runtime<M>) -> Self {
        use Prim::*;
        let mut prims = HashMap::new();
        for (name, p) in [
            ("%add", Add), ("%sub", Sub), ("%mul", Mul), ("%lt", Lt), ("%num-eq", Eq),
            ("%first", First), ("%rest", Rest), ("%cons", Cons),
            ("record", Record), ("field", Field), ("type-of", TypeOf), ("nfields", NFields), ("throw", Throw),
            ("nil?", IsNil), ("list", List), ("println", Println), ("gc", Gc),
            // Mutable 1-slot cell backing atoms (a real mutable array, unlike the
            // immutable list-backed clojure `vector`).
            ("%cell", Vector), ("%cell-ref", VectorRef), ("%cell-set!", VectorSet),
        ] {
            prims.insert(rt.intern(name), p);
        }
        Compiler { scope: Vec::new(), methods: std::collections::HashSet::new(), site: 0, prims }
    }

    /// Compile one fully-expanded top-level form to `Ir`.
    pub fn compile<M: ValueModel>(&mut self, rt: &mut Runtime<M>, form: u64) -> Ir {
        match rt.decode(form) {
            Val::Int(_) | Val::Float(_) | Val::Bool(_) | Val::Nil => Ir::Const(rt.intern_const(form)),
            Val::Sym(s) => match self.resolve_local(s) {
                Some((up, idx)) => Ir::Local { up, idx },
                None => Ir::Global(s),
            },
            Val::Ref(_) => {
                if rt.as_cons(form).is_some() {
                    self.compile_list(rt, form)
                } else {
                    // keyword / string / char / quoted-collection literal: self-eval
                    Ir::Const(rt.intern_const(form))
                }
            }
        }
    }

    fn resolve_local(&self, sym: Sym) -> Option<(u16, u16)> {
        for (up, frame) in self.scope.iter().rev().enumerate() {
            if let Some(idx) = frame.iter().rposition(|&s| s == sym) {
                return Some((up as u16, idx as u16));
            }
        }
        None
    }

    fn name<M: ValueModel>(&self, rt: &Runtime<M>, bits: u64) -> Option<Sym> {
        match rt.decode(bits) {
            Val::Sym(s) => Some(s),
            _ => None,
        }
    }

    fn compile_list<M: ValueModel>(&mut self, rt: &mut Runtime<M>, form: u64) -> Ir {
        let items = rt.list_to_vec(form);
        // A symbolic head may be a core special form, a prim, a method, or a call.
        if let Some(hs) = self.name(rt, items[0]) {
            match rt.sym_name(hs) {
                "quote" => return Ir::Const(rt.intern_const(items[1])),
                "if" => {
                    let c = self.compile(rt, items[1]);
                    let t = self.compile(rt, items[2]);
                    let e = if items.len() > 3 {
                        self.compile(rt, items[3])
                    } else {
                        let nil = rt.encode(Val::Nil);
                        Ir::Const(rt.intern_const(nil))
                    };
                    return Ir::If(Box::new(c), Box::new(t), Box::new(e));
                }
                "do" => return Ir::Do(items[1..].iter().map(|&f| self.compile(rt, f)).collect()),
                "def" => {
                    let n = self.name(rt, items[1]).expect("def: name must be a symbol");
                    // value-less `(def x)` declares an unbound var (bound to nil).
                    let init = if items.len() > 2 {
                        self.compile(rt, items[2])
                    } else {
                        let nil = rt.encode(Val::Nil);
                        Ir::Const(rt.intern_const(nil))
                    };
                    return Ir::Def { name: n, init: Box::new(init) };
                }
                "fn" | "fn*" => return self.compile_fn(rt, &items),
                "let" | "let*" => return self.compile_let(rt, &items),
                "try*" => return self.compile_try(rt, &items),
                "set!" => {
                    let n = self.name(rt, items[1]).expect("set!: target must be a symbol");
                    let val = Box::new(self.compile(rt, items[2]));
                    return match self.resolve_local(n) {
                        Some((up, idx)) => Ir::SetLocal { up, idx, val },
                        None => Ir::SetGlobal { name: n, val },
                    };
                }
                "defmethod" => {
                    let m = self.name(rt, items[1]).expect("defmethod: method name");
                    let ty = self.name(rt, items[2]).expect("defmethod: type name");
                    self.methods.insert(m);
                    let imp = self.compile(rt, items[3]);
                    return Ir::DefMethod { name: m, ty, imp: Box::new(imp) };
                }
                _ => {
                    // A binding shadows a prim; a prim is otherwise its default.
                    let shadowed = self.resolve_local(hs).is_some() || rt.globals.contains_key(&hs);
                    if !shadowed {
                        if let Some(&p) = self.prims.get(&hs) {
                            let args = items[1..].iter().map(|&f| self.compile(rt, f)).collect();
                            return Ir::Prim(p, args);
                        }
                    }
                    if self.methods.contains(&hs) {
                        let site = self.fresh_site();
                        let args = items[1..].iter().map(|&f| self.compile(rt, f)).collect();
                        return Ir::Dispatch { site, method: hs, args };
                    }
                }
            }
        }
        // ordinary call: (f args...)
        let f = self.compile(rt, items[0]);
        let args = items[1..].iter().map(|&a| self.compile(rt, a)).collect();
        Ir::Call(Box::new(f), args)
    }

    fn compile_fn<M: ValueModel>(&mut self, rt: &mut Runtime<M>, items: &[u64]) -> Ir {
        let params = rt.list_to_vec(items[1]);
        let mut names = Vec::new();
        let mut variadic = false;
        let mut i = 0;
        while i < params.len() {
            let s = self.name(rt, params[i]).expect("fn: params must be symbols");
            if rt.sym_name(s) == "&" {
                let rest = self.name(rt, params[i + 1]).expect("fn: variadic param");
                names.push(rest);
                variadic = true;
                break;
            }
            names.push(s);
            i += 1;
        }
        let nparams = if variadic { names.len() - 1 } else { names.len() };
        self.scope.push(names);
        let body: Vec<Ir> = items[2..].iter().map(|&f| self.compile(rt, f)).collect();
        self.scope.pop();
        Ir::Lambda { nparams, variadic, body: Rc::new(Ir::Do(body)) }
    }

    /// `(try* body EXC dispatch finally)` — the fixed shape the expander lowers
    /// `try`/`catch`/`finally` to (see `desugar_try`). `body`/`finally` compile in
    /// the current scope; `dispatch` compiles with `EXC` bound in a fresh frame, so
    /// the thrown value lands at `Local{up:0,idx:0}` (matching `Ir::Try`). A `nil`
    /// `EXC`/`dispatch` means no catch; a `nil` `finally` means no finally.
    fn compile_try<M: ValueModel>(&mut self, rt: &mut Runtime<M>, items: &[u64]) -> Ir {
        let body = Box::new(self.compile(rt, items[1]));
        let catch = match self.name(rt, items[2]) {
            Some(exc) => {
                self.scope.push(vec![exc]);
                let handler = self.compile(rt, items[3]);
                self.scope.pop();
                Some(Box::new(handler))
            }
            None => None, // items[2] is nil -> no catch clause
        };
        let finally = if matches!(rt.decode(items[4]), Val::Nil) {
            None
        } else {
            Some(Box::new(self.compile(rt, items[4])))
        };
        Ir::Try { body, catch, finally }
    }

    fn compile_let<M: ValueModel>(&mut self, rt: &mut Runtime<M>, items: &[u64]) -> Ir {
        let binds = rt.list_to_vec(items[1]);
        self.scope.push(Vec::new());
        let mut inits = Vec::new();
        let mut i = 0;
        while i + 1 < binds.len() {
            let name = self.name(rt, binds[i]).expect("let: binding name must be a symbol");
            inits.push(self.compile(rt, binds[i + 1]));
            self.scope.last_mut().unwrap().push(name);
            i += 2;
        }
        let body: Vec<Ir> = items[2..].iter().map(|&f| self.compile(rt, f)).collect();
        self.scope.pop();
        Ir::Let(inits, Box::new(Ir::Do(body)))
    }

    fn fresh_site(&mut self) -> usize {
        let s = self.site;
        self.site += 1;
        s
    }
}
