//! Closure conversion: chain-scoped `Ir` -> FLAT `Ir`.
//!
//! Frontends lower surface syntax to chain-scoped `Ir`: `Local { up, idx }`
//! resolved against a run-time chain of frames (one per `let` / lambda / catch),
//! and `Lambda` closing over the WHOLE chain. That shape is easy to produce but
//! expensive to run: every call allocates a linked frame, every outer-variable
//! read hops the chain, and a closure keeps its entire lexical ancestry alive.
//!
//! This pass rewrites a top-level form into the FLAT shape the execution tiers
//! actually run — the classic flat-closure conversion:
//!
//!   * One activation frame per CALL, sized `nslots` (params, rest arg, and
//!     every `let`/catch binding of the function body — assigned monotonically,
//!     no reuse, so a captured continuation can safely re-enter a sibling
//!     scope). All `Local`/`SetLocal` end up `up == 0`.
//!   * `Let` disappears: each init becomes a `SetLocal` store into its slot.
//!   * `Lambda` carries an explicit capture list: the VALUES a closure needs
//!     are copied out of the enclosing activation (or the enclosing closure's
//!     own captures, transitively) at creation time. `Capture(i)` reads them.
//!   * Assignment conversion: a binding that is both ASSIGNED (`set!`) and
//!     CROSSES a lambda boundary is boxed in a cell (an `Obj::Atom`), so
//!     mutation stays visible through value-copied captures. Reads become
//!     `%atom-get`, writes `%atom-set`, creation wraps in `%atom-new`.
//!     (Clojure never assigns locals, so this triggers only for Scheme
//!     `set!`/`letrec`-style code.)
//!   * `Try`'s catch binding is re-homed to an activation slot (`cslot`) of
//!     the same frame — no fresh 1-slot frame.
//!   * A top-level form that needs slots is wrapped in an immediate call of a
//!     0-ary lambda, so every `eval_ir(ir, &None)` call site works unchanged.
//!
//! The pass runs ONCE per top-level form, at each frontend's compile
//! chokepoint. Feeding it already-flat Ir (it contains `Capture`) is a loud
//! error — that always indicates a double conversion.

use std::sync::Arc;

use crate::ir::{CapSrc, Ir, Prim};

/// Identity of one binding site, assigned in traversal order. The boxing
/// pre-pass and the rewrite walk push frames at the same nodes in the same
/// order, so ids agree between them.
type BindingId = u32;

// ───────────────────────── pre-pass: which bindings need a cell ─────────────

#[derive(Default, Clone)]
struct PreInfo {
    /// Some `SetLocal` targets this binding.
    assigned: bool,
    /// Some read/write of this binding crosses a lambda boundary.
    crossed: bool,
}

struct Pre {
    info: Vec<PreInfo>,
    /// (owning fn depth, binding ids), innermost frame last.
    frames: Vec<(usize, Vec<BindingId>)>,
    fn_depth: usize,
}

impl Pre {
    fn new_binding(&mut self) -> BindingId {
        let id = self.info.len() as BindingId;
        self.info.push(PreInfo::default());
        id
    }

    fn touch(&mut self, up: u16, idx: u16, write: bool) {
        let fi = self
            .frames
            .len()
            .checked_sub(1 + up as usize)
            .unwrap_or_else(|| panic!("flatten: Local up={up} escapes all frames"));
        let (owner_depth, ids) = &self.frames[fi];
        let id = ids[idx as usize] as usize;
        if write {
            self.info[id].assigned = true;
        }
        if *owner_depth != self.fn_depth {
            self.info[id].crossed = true;
        }
    }

    fn walk(&mut self, ir: &Ir) {
        match ir {
            Ir::Const(_) | Ir::Quote(_) | Ir::Global(_) => {}
            Ir::Capture(_) => panic!("flatten: input Ir is already flattened (contains Capture)"),
            Ir::Local { up, idx } => self.touch(*up, *idx, false),
            Ir::SetLocal { up, idx, val } => {
                self.walk(val);
                self.touch(*up, *idx, true);
            }
            Ir::SetGlobal { val, .. } => self.walk(val),
            Ir::If(c, t, e) => {
                self.walk(c);
                self.walk(t);
                self.walk(e);
            }
            Ir::Do(xs) => xs.iter().for_each(|x| self.walk(x)),
            Ir::Def { init, .. } => self.walk(init),
            Ir::Let(inits, body) => {
                self.frames.push((self.fn_depth, Vec::new()));
                for init in inits {
                    self.walk(init);
                    let id = self.new_binding();
                    self.frames.last_mut().unwrap().1.push(id);
                }
                self.walk(body);
                self.frames.pop();
            }
            Ir::Lambda { nparams, variadic, body, .. } => {
                self.fn_depth += 1;
                let n = nparams + *variadic as usize;
                let ids: Vec<BindingId> = (0..n).map(|_| self.new_binding()).collect();
                self.frames.push((self.fn_depth, ids));
                self.walk(body);
                self.frames.pop();
                self.fn_depth -= 1;
            }
            Ir::Call(f, args) => {
                self.walk(f);
                args.iter().for_each(|a| self.walk(a));
            }
            Ir::Prim(_, args) => args.iter().for_each(|a| self.walk(a)),
            Ir::DefMethod { imp, .. } => self.walk(imp),
            Ir::Dispatch { args, .. } => args.iter().for_each(|a| self.walk(a)),
            Ir::FieldGet { obj, .. } => self.walk(obj),
            Ir::Try { body, catch, finally, .. } => {
                self.walk(body);
                if let Some(c) = catch {
                    let id = self.new_binding();
                    self.frames.push((self.fn_depth, vec![id]));
                    self.walk(c);
                    self.frames.pop();
                }
                if let Some(f) = finally {
                    self.walk(f);
                }
            }
        }
    }
}

// ───────────────────────── main pass: the rewrite ───────────────────────────

#[derive(Clone, Copy)]
struct Binding {
    id: BindingId,
    slot: u16,
    boxed: bool,
}

struct FnState {
    /// Monotonic slot allocator; final value = the function's `nslots`.
    next_slot: u32,
    /// Captured bindings, in capture-array order, with where each one's value
    /// comes from in the ENCLOSING function's terms.
    caps: Vec<(BindingId, CapSrc)>,
}

struct Flat<'a> {
    info: &'a [PreInfo],
    next_id: BindingId,
    /// (owning fn INDEX into `fns`, bindings), innermost frame last.
    frames: Vec<(usize, Vec<Binding>)>,
    /// Enclosing function states; index 0 is the top-level pseudo-function.
    fns: Vec<FnState>,
}

impl<'a> Flat<'a> {
    fn cur_fn(&self) -> usize {
        self.fns.len() - 1
    }

    fn new_binding(&mut self) -> Binding {
        let id = self.next_id;
        self.next_id += 1;
        let f = self.fns.last_mut().unwrap();
        let slot = f.next_slot;
        f.next_slot += 1;
        assert!(slot <= u16::MAX as u32, "flatten: function needs more than 65536 slots");
        let inf = &self.info[id as usize];
        Binding { id, slot: slot as u16, boxed: inf.assigned && inf.crossed }
    }

    /// Ensure `fns[target]` captures binding `id` (owned by `fns[owner]`, at
    /// activation slot `slot` there); the capture index in `fns[target]`.
    fn ensure_cap(&mut self, target: usize, owner: usize, id: BindingId, slot: u16) -> u16 {
        if let Some(i) = self.fns[target].caps.iter().position(|(bid, _)| *bid == id) {
            return i as u16;
        }
        let src = if target - 1 == owner {
            CapSrc::Slot(slot)
        } else {
            CapSrc::Cap(self.ensure_cap(target - 1, owner, id, slot))
        };
        self.fns[target].caps.push((id, src));
        (self.fns[target].caps.len() - 1) as u16
    }

    /// The location expression for a chain reference, plus whether it is boxed.
    fn resolve(&mut self, up: u16, idx: u16) -> (Ir, bool) {
        let fi = self
            .frames
            .len()
            .checked_sub(1 + up as usize)
            .unwrap_or_else(|| panic!("flatten: Local up={up} escapes all frames"));
        let (owner_fn, b) = {
            let fr = &self.frames[fi];
            (fr.0, fr.1[idx as usize])
        };
        if owner_fn == self.cur_fn() {
            (Ir::Local { up: 0, idx: b.slot }, b.boxed)
        } else {
            let ci = self.ensure_cap(self.cur_fn(), owner_fn, b.id, b.slot);
            (Ir::Capture(ci), b.boxed)
        }
    }

    /// Prologue stores that re-box assigned-and-captured entry bindings (params,
    /// the rest arg, a catch binding): slot := (cell slot-value).
    fn box_prologue(slots: &[u16], body: Ir) -> Ir {
        if slots.is_empty() {
            return body;
        }
        let mut seq: Vec<Ir> = slots
            .iter()
            .map(|&s| Ir::SetLocal {
                up: 0,
                idx: s,
                val: Box::new(Ir::Prim(Prim::AtomNew, vec![Ir::Local { up: 0, idx: s }])),
            })
            .collect();
        seq.push(body);
        Ir::Do(seq)
    }

    fn walk(&mut self, ir: &Ir) -> Ir {
        match ir {
            Ir::Const(id) => Ir::Const(*id),
            Ir::Quote(id) => Ir::Quote(*id),
            Ir::Global(s) => Ir::Global(*s),
            Ir::Capture(_) => panic!("flatten: input Ir is already flattened (contains Capture)"),
            Ir::Local { up, idx } => {
                let (loc, boxed) = self.resolve(*up, *idx);
                if boxed {
                    Ir::Prim(Prim::AtomGet, vec![loc])
                } else {
                    loc
                }
            }
            Ir::SetLocal { up, idx, val } => {
                let v = self.walk(val);
                let (loc, boxed) = self.resolve(*up, *idx);
                if boxed {
                    Ir::Prim(Prim::AtomSet, vec![loc, v])
                } else {
                    match loc {
                        Ir::Local { up: 0, idx } => {
                            Ir::SetLocal { up: 0, idx, val: Box::new(v) }
                        }
                        _ => unreachable!(
                            "flatten: unboxed assignment crossed a lambda boundary"
                        ),
                    }
                }
            }
            Ir::SetGlobal { name, val } => {
                Ir::SetGlobal { name: *name, val: Box::new(self.walk(val)) }
            }
            Ir::If(c, t, e) => Ir::If(
                Box::new(self.walk(c)),
                Box::new(self.walk(t)),
                Box::new(self.walk(e)),
            ),
            Ir::Do(xs) => Ir::Do(xs.iter().map(|x| self.walk(x)).collect()),
            Ir::Def { name, init } => {
                Ir::Def { name: *name, init: Box::new(self.walk(init)) }
            }
            Ir::Let(inits, body) => {
                let fi = self.cur_fn();
                self.frames.push((fi, Vec::new()));
                let mut seq = Vec::with_capacity(inits.len() + 1);
                for init in inits {
                    // The init is walked BEFORE its binding exists (matching the
                    // frontends: a let binding is not in scope in its own init).
                    let v = self.walk(init);
                    let b = self.new_binding();
                    let val = if b.boxed {
                        Box::new(Ir::Prim(Prim::AtomNew, vec![v]))
                    } else {
                        Box::new(v)
                    };
                    seq.push(Ir::SetLocal { up: 0, idx: b.slot, val });
                    self.frames.last_mut().unwrap().1.push(b);
                }
                seq.push(self.walk(body));
                self.frames.pop();
                Ir::Do(seq)
            }
            Ir::Lambda { nparams, variadic, body, .. } => {
                self.fns.push(FnState { next_slot: 0, caps: Vec::new() });
                let fnum = self.cur_fn();
                let n = nparams + *variadic as usize;
                let mut bindings = Vec::with_capacity(n);
                let mut boxed_entries = Vec::new();
                for _ in 0..n {
                    let b = self.new_binding();
                    if b.boxed {
                        boxed_entries.push(b.slot);
                    }
                    bindings.push(b);
                }
                self.frames.push((fnum, bindings));
                let body2 = self.walk(body);
                self.frames.pop();
                let fnst = self.fns.pop().unwrap();
                let body3 = Self::box_prologue(&boxed_entries, body2);
                Ir::Lambda {
                    nparams: *nparams,
                    variadic: *variadic,
                    nslots: fnst.next_slot as u16,
                    captures: fnst.caps.into_iter().map(|(_, src)| src).collect(),
                    body: Arc::new(body3),
                }
            }
            Ir::Call(f, args) => Ir::Call(
                Box::new(self.walk(f)),
                args.iter().map(|a| self.walk(a)).collect(),
            ),
            Ir::Prim(op, args) => {
                Ir::Prim(*op, args.iter().map(|a| self.walk(a)).collect())
            }
            Ir::DefMethod { name, ty, imp } => Ir::DefMethod {
                name: *name,
                ty: *ty,
                imp: Box::new(self.walk(imp)),
            },
            Ir::Dispatch { site, method, args } => Ir::Dispatch {
                site: *site,
                method: *method,
                args: args.iter().map(|a| self.walk(a)).collect(),
            },
            Ir::FieldGet { site, field, obj } => Ir::FieldGet {
                site: *site,
                field: *field,
                obj: Box::new(self.walk(obj)),
            },
            Ir::Try { body, catch, finally, .. } => {
                let body2 = Box::new(self.walk(body));
                let (catch2, cslot) = match catch {
                    None => (None, 0),
                    Some(c) => {
                        let b = self.new_binding();
                        self.frames.push((self.cur_fn(), vec![b]));
                        let c2 = self.walk(c);
                        self.frames.pop();
                        let boxed = if b.boxed { vec![b.slot] } else { vec![] };
                        (Some(Box::new(Self::box_prologue(&boxed, c2))), b.slot)
                    }
                };
                let finally2 = finally.as_ref().map(|f| Box::new(self.walk(f)));
                Ir::Try { body: body2, catch: catch2, finally: finally2, cslot }
            }
        }
    }
}

/// Flatten one top-level form. See the module doc. If the top level itself
/// needs activation slots (a top-level `let`/`try`), the result is wrapped in
/// an immediate call of a 0-ary lambda so the caller's `eval_ir(ir, &None)`
/// contract is unchanged.
pub fn flatten(ir: &Ir) -> Ir {
    let mut pre = Pre { info: Vec::new(), frames: Vec::new(), fn_depth: 0 };
    pre.walk(ir);
    let mut fl = Flat {
        info: &pre.info,
        next_id: 0,
        frames: Vec::new(),
        fns: vec![FnState { next_slot: 0, caps: Vec::new() }],
    };
    let out = fl.walk(ir);
    let top = fl.fns.pop().unwrap();
    assert!(top.caps.is_empty(), "flatten: top level cannot capture");
    debug_assert_eq!(fl.next_id as usize, pre.info.len(), "flatten: pass binding drift");
    if top.next_slot == 0 {
        out
    } else {
        Ir::Call(
            Box::new(Ir::Lambda {
                nparams: 0,
                variadic: false,
                nslots: top.next_slot as u16,
                captures: Vec::new(),
                body: Arc::new(out),
            }),
            Vec::new(),
        )
    }
}
