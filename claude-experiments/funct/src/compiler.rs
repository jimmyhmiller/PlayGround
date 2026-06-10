//! AST → bytecode compiler.
//!
//! Mutable locals (`let mut`) always live in a Cell so closures that capture
//! them share the slot. Calls in tail position become TailCall (bounded
//! frames for self/mutual recursion in tail position).

use crate::ast::{Arm, BinOp, Expr, ExprKind, FnDef, InterpPart, Item, Pattern, Program, Stmt, UnOp, VariantCtor};
use crate::bytecode::{CaptureSrc, Const, FnProto, Instr, Pat};
use std::collections::{HashMap, HashSet};

/// Program-level context the compiler reads/extends. Owned by the engine.
pub struct ProgramCtx {
    pub fn_ids: HashMap<String, u32>,
    /// number of registered fn protos (engine owns the actual Vec)
    pub fn_count: u32,
    pub global_ids: HashMap<String, u32>,
    pub global_names: Vec<String>,
    /// Globals visible inside modules without an import: prelude + natives +
    /// host-registered values. Module code can NOT see other plain globals.
    pub shared: HashSet<u32>,
}

impl ProgramCtx {
    pub fn ensure_global(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.global_ids.get(name) {
            return id;
        }
        let id = self.global_names.len() as u32;
        self.global_names.push(name.to_string());
        self.global_ids.insert(name.to_string(), id);
        id
    }
}

/// Mangle a top-level name with its module path: "math/vec::lerp".
pub fn module_global_name(prefix: Option<&str>, name: &str) -> String {
    match prefix {
        Some(p) => format!("{}::{}", p, name),
        None => name.to_string(),
    }
}

pub struct CompiledProgram {
    /// new/replaced protos to install: (fn_id, proto)
    pub protos: Vec<(u32, FnProto)>,
    /// after install, set globals[gslot] = Closure { fn_id }
    pub fn_globals: Vec<(u32, u32)>,
    /// fn id of the top-level init/main proto, if there is any top-level code
    pub main: Option<u32>,
    /// `export`ed top-level names: (plain name, global slot)
    pub exports: Vec<(String, u32)>,
    /// global names of `#[test]` functions, in declaration order
    pub tests: Vec<String>,
}

pub fn compile_program(
    ctx: &mut ProgramCtx,
    prog: &Program,
    prefix: Option<&str>,
) -> Result<CompiledProgram, String> {
    let mut c = Compiler {
        ctx,
        fns: Vec::new(),
        protos: Vec::new(),
        prefix: prefix.map(|s| s.to_string()),
    };

    // Pass 1: declare all item names as globals (forward references work).
    for item in &prog.items {
        match item {
            Item::Fn(f) => {
                let full = c.full_name(&f.name);
                c.ctx.ensure_global(&full);
            }
            Item::Let { pattern, .. } => {
                let mut names = Vec::new();
                pattern.bound_names(&mut names);
                for n in &names {
                    let full = c.full_name(n);
                    c.ctx.ensure_global(&full);
                }
            }
            _ => {}
        }
    }

    // Pass 2: compile fn defs.
    let mut fn_globals = Vec::new();
    let mut exports = Vec::new();
    let mut tests = Vec::new();
    for item in &prog.items {
        if let Item::Fn(f) = item {
            let full = c.full_name(&f.name);
            let fn_id = c.alloc_fn_id(&full);
            let proto = c.compile_fn(f, &full)?;
            c.protos.push((fn_id, proto));
            let gslot = c.ctx.ensure_global(&full);
            fn_globals.push((gslot, fn_id));
            if f.exported {
                exports.push((f.name.clone(), gslot));
            }
            if f.attrs.iter().any(|a| a == "test") {
                tests.push(full.clone());
            }
        }
    }

    // Pass 3: top-level lets & expressions form the main proto.
    let has_top_code = prog.items.iter().any(|i| matches!(i, Item::Let { .. } | Item::Expr(_)));
    let main = if has_top_code {
        let full = c.full_name("__main__");
        let fn_id = c.alloc_fn_id(&full);
        let proto = c.compile_main(prog, &mut exports)?;
        c.protos.push((fn_id, proto));
        Some(fn_id)
    } else {
        None
    };

    Ok(CompiledProgram { protos: c.protos, fn_globals, main, exports, tests })
}

#[derive(Clone, Copy)]
enum Resolved {
    Local { slot: u16, cell: bool },
    Upval { idx: u16, cell: bool },
    Global(u32),
}

struct LocalEntry {
    name: String,
    slot: u16,
    cell: bool,
}

/// Per-loop compile state for break/continue.
struct LoopFrame {
    /// operand-stack depth at the loop's entry (what break/continue restore to)
    entry_depth: i32,
    /// absolute jump target for `continue` (cond check / IterNext)
    continue_target: u32,
    /// Jump sites to patch to the loop's end for `break`
    break_jumps: Vec<usize>,
}

struct FnCtx {
    name: String,
    arity: u8,
    code: Vec<Instr>,
    consts: Vec<Const>,
    pats: Vec<Pat>,
    lines: Vec<u32>,
    scopes: Vec<Vec<LocalEntry>>,
    next_local: u16,
    max_locals: u16,
    /// (name, capture-in-parent-terms, is_cell)
    upvals: Vec<(String, CaptureSrc, bool)>,
    cur_line: u32,
    /// simulated operand-stack depth at the current emission point; used by
    /// break/continue to pop partial operands (e.g. a match subject) before
    /// jumping out of the loop
    depth: i32,
    loops: Vec<LoopFrame>,
}

impl FnCtx {
    fn new(name: &str, arity: u8) -> FnCtx {
        FnCtx {
            name: name.to_string(),
            arity,
            code: Vec::new(),
            consts: Vec::new(),
            pats: Vec::new(),
            lines: Vec::new(),
            scopes: vec![Vec::new()],
            next_local: 0,
            max_locals: 0,
            upvals: Vec::new(),
            cur_line: 0,
            depth: 0,
            loops: Vec::new(),
        }
    }

    fn finish(self) -> FnProto {
        FnProto {
            name: self.name,
            arity: self.arity,
            num_locals: self.max_locals,
            num_upvals: self.upvals.len() as u16,
            code: self.code,
            consts: self.consts,
            pats: self.pats,
            lines: self.lines,
        }
    }
}

struct Compiler<'a> {
    ctx: &'a mut ProgramCtx,
    /// stack of nested function contexts; innermost last
    fns: Vec<FnCtx>,
    protos: Vec<(u32, FnProto)>,
    /// module path when compiling a module file (names get mangled)
    prefix: Option<String>,
}

impl<'a> Compiler<'a> {
    fn full_name(&self, name: &str) -> String {
        module_global_name(self.prefix.as_deref(), name)
    }

    /// Module-aware global lookup: a module sees its own (mangled) globals
    /// first, then only `shared` globals (prelude/natives/host values).
    fn lookup_global(&self, name: &str) -> Option<u32> {
        if let Some(p) = &self.prefix {
            if let Some(&g) = self.ctx.global_ids.get(&format!("{}::{}", p, name)) {
                return Some(g);
            }
            return self.ctx.global_ids.get(name).copied().filter(|g| self.ctx.shared.contains(g));
        }
        self.ctx.global_ids.get(name).copied()
    }

    fn alloc_fn_id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ctx.fn_ids.get(name) {
            return id; // hot reload: reuse the id, replace the proto
        }
        let id = self.ctx.fn_count;
        self.ctx.fn_count += 1;
        self.ctx.fn_ids.insert(name.to_string(), id);
        id
    }

    // ----- emit helpers (operate on innermost FnCtx) -----

    fn f(&mut self) -> &mut FnCtx {
        self.fns.last_mut().expect("no active fn ctx")
    }

    fn emit(&mut self, i: Instr) -> usize {
        let line = self.f().cur_line;
        let f = self.f();
        f.depth += instr_effect(&f.consts, &i);
        f.code.push(i);
        f.lines.push(line);
        f.code.len() - 1
    }

    fn here(&mut self) -> u32 {
        self.f().code.len() as u32
    }

    fn patch(&mut self, at: usize) {
        let target = self.here();
        let f = self.f();
        match &mut f.code[at] {
            Instr::Jump(t)
            | Instr::JumpIfFalse(t)
            | Instr::JumpIfFalsePeek(t)
            | Instr::JumpIfTruePeek(t) => *t = target,
            Instr::MatchPat { fail, .. } => *fail = target,
            Instr::IterNext { end, .. } => *end = target,
            other => panic!("patch on non-jump instr {:?}", other),
        }
    }

    fn konst(&mut self, c: Const) -> u32 {
        let f = self.f();
        if let Some(i) = f.consts.iter().position(|x| x == &c) {
            return i as u32;
        }
        f.consts.push(c);
        (f.consts.len() - 1) as u32
    }

    fn alloc_local(&mut self) -> u16 {
        let f = self.f();
        let slot = f.next_local;
        f.next_local += 1;
        if f.next_local > f.max_locals {
            f.max_locals = f.next_local;
        }
        slot
    }

    fn declare(&mut self, name: &str, slot: u16, cell: bool) {
        let f = self.f();
        f.scopes
            .last_mut()
            .unwrap()
            .push(LocalEntry { name: name.to_string(), slot, cell });
    }

    fn begin_scope(&mut self) {
        self.f().scopes.push(Vec::new());
    }

    fn end_scope(&mut self) {
        self.f().scopes.pop();
        // NOTE: slots are bump-allocated per fn and not reused; num_locals is
        // the high-water mark. Simple > optimal, and fine for v0.1.
    }

    fn err<T>(&self, line: u32, msg: &str) -> Result<T, String> {
        Err(format!("line {}: {}", line, msg))
    }

    // ----- name resolution -----

    fn resolve_in(&mut self, fn_idx: usize, name: &str) -> Option<Resolved> {
        // local?
        {
            let f = &self.fns[fn_idx];
            for scope in f.scopes.iter().rev() {
                for e in scope.iter().rev() {
                    if e.name == name {
                        return Some(Resolved::Local { slot: e.slot, cell: e.cell });
                    }
                }
            }
            // already-captured upval?
            for (i, (n, _, cell)) in f.upvals.iter().enumerate() {
                if n == name {
                    return Some(Resolved::Upval { idx: i as u16, cell: *cell });
                }
            }
        }
        if fn_idx == 0 {
            return self.lookup_global(name).map(Resolved::Global);
        }
        match self.resolve_in(fn_idx - 1, name)? {
            Resolved::Local { slot, cell } => {
                let f = &mut self.fns[fn_idx];
                f.upvals.push((name.to_string(), CaptureSrc::Local(slot), cell));
                Some(Resolved::Upval { idx: (f.upvals.len() - 1) as u16, cell })
            }
            Resolved::Upval { idx, cell } => {
                let f = &mut self.fns[fn_idx];
                f.upvals.push((name.to_string(), CaptureSrc::Upval(idx), cell));
                Some(Resolved::Upval { idx: (f.upvals.len() - 1) as u16, cell })
            }
            Resolved::Global(g) => Some(Resolved::Global(g)),
        }
    }

    fn resolve(&mut self, name: &str) -> Option<Resolved> {
        self.resolve_in(self.fns.len() - 1, name)
    }

    // ----- functions -----

    fn compile_fn(&mut self, def: &FnDef, full_name: &str) -> Result<FnProto, String> {
        self.compile_function(full_name, &def.params, &def.body, def.line)
    }

    fn compile_function(
        &mut self,
        name: &str,
        params: &[Pattern],
        body: &Expr,
        line: u32,
    ) -> Result<FnProto, String> {
        if params.len() > u8::MAX as usize {
            return self.err(line, "too many parameters");
        }
        let mut fctx = FnCtx::new(name, params.len() as u8);
        fctx.cur_line = line;
        self.fns.push(fctx);

        // Parameters occupy slots 0..arity. Simple names bind directly;
        // pattern params get a hidden slot + destructuring prologue.
        let mut destructure: Vec<(u16, &Pattern)> = Vec::new();
        for p in params {
            let slot = self.alloc_local();
            match p {
                Pattern::Bind(n) => self.declare(n, slot, false),
                _ => destructure.push((slot, p)),
            }
        }
        for (slot, p) in destructure {
            self.f().cur_line = line;
            self.emit(Instr::LoadLocal(slot));
            let pat = self.compile_pattern(p)?;
            let pidx = self.add_pat(pat);
            let m = self.emit(Instr::MatchPat { pat: pidx, fail: 0 });
            let j = self.emit(Instr::Jump(0));
            self.patch(m);
            let msg = self.konst(Const::Name(format!("parameter pattern did not match in fn {}", name)));
            self.emit(Instr::Fault(msg));
            self.patch(j);
            self.emit(Instr::Pop); // subject
        }

        self.compile_expr(body, true)?;
        self.emit(Instr::Ret);
        Ok(self.fns.pop().unwrap().finish())
    }

    fn compile_main(
        &mut self,
        prog: &Program,
        exports: &mut Vec<(String, u32)>,
    ) -> Result<FnProto, String> {
        let main_name = self.full_name("__main__");
        let mut fctx = FnCtx::new(&main_name, 0);
        fctx.cur_line = 1;
        self.fns.push(fctx);

        let top_items: Vec<&Item> = prog
            .items
            .iter()
            .filter(|i| matches!(i, Item::Let { .. } | Item::Expr(_)))
            .collect();
        let last = top_items.len().saturating_sub(1);
        let mut left_value = false;
        for (i, item) in top_items.iter().enumerate() {
            let is_last = i == last;
            match item {
                Item::Let { pattern, expr, exported, line } => {
                    self.f().cur_line = *line;
                    self.compile_expr(expr, false)?;
                    // bind via pattern into locals, then copy to globals
                    self.begin_scope();
                    let pat = self.compile_pattern(pattern)?;
                    let pidx = self.add_pat(pat);
                    let m = self.emit(Instr::MatchPat { pat: pidx, fail: 0 });
                    let j = self.emit(Instr::Jump(0));
                    self.patch(m);
                    let msg = self.konst(Const::Name("top-level let pattern did not match".into()));
                    self.emit(Instr::Fault(msg));
                    self.patch(j);
                    self.emit(Instr::Pop);
                    let mut names = Vec::new();
                    pattern.bound_names(&mut names);
                    for n in &names {
                        let r = self.resolve(n).expect("just bound");
                        if let Resolved::Local { slot, .. } = r {
                            self.emit(Instr::LoadLocal(slot));
                            let full = self.full_name(n);
                            let g = self.ctx.ensure_global(&full);
                            self.emit(Instr::StoreGlobal(g));
                            if *exported {
                                exports.push((n.clone(), g));
                            }
                        }
                    }
                    self.end_scope();
                    left_value = false;
                }
                Item::Expr(e) => {
                    self.f().cur_line = e.line;
                    self.compile_expr(e, false)?;
                    if is_last {
                        left_value = true;
                    } else {
                        self.emit(Instr::Pop);
                    }
                }
                _ => unreachable!(),
            }
        }
        if !left_value {
            self.emit(Instr::Unit);
        }
        self.emit(Instr::Ret);
        Ok(self.fns.pop().unwrap().finish())
    }

    // ----- statements -----

    /// Statements always net zero operand-stack effect; pinning the depth at
    /// each statement boundary keeps the simulation exact even past dead
    /// code after break/continue/return.
    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        let d0 = self.f().depth;
        self.compile_stmt_inner(stmt)?;
        self.f().depth = d0;
        Ok(())
    }

    fn compile_stmt_inner(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let { mutable, pattern, expr, line } => {
                self.f().cur_line = *line;
                self.compile_expr(expr, false)?;
                if *mutable {
                    let name = match pattern {
                        Pattern::Bind(n) => n.clone(),
                        _ => return self.err(*line, "`let mut` requires a plain name"),
                    };
                    self.emit(Instr::NewCell);
                    let slot = self.alloc_local();
                    self.emit(Instr::StoreLocal(slot));
                    self.declare(&name, slot, true);
                } else if let Pattern::Bind(n) = pattern {
                    let slot = self.alloc_local();
                    self.emit(Instr::StoreLocal(slot));
                    self.declare(n, slot, false);
                } else {
                    let pat = self.compile_pattern(pattern)?;
                    let pidx = self.add_pat(pat);
                    let m = self.emit(Instr::MatchPat { pat: pidx, fail: 0 });
                    let j = self.emit(Instr::Jump(0));
                    self.patch(m);
                    let msg = self.konst(Const::Name("let pattern did not match".into()));
                    self.emit(Instr::Fault(msg));
                    self.patch(j);
                    self.emit(Instr::Pop);
                }
                Ok(())
            }
            Stmt::Assign { name, expr, line } => {
                self.f().cur_line = *line;
                match self.resolve(name) {
                    Some(Resolved::Local { slot, cell: true }) => {
                        self.emit(Instr::LoadLocal(slot));
                        self.compile_expr(expr, false)?;
                        self.emit(Instr::CellSet);
                        Ok(())
                    }
                    Some(Resolved::Upval { idx, cell: true }) => {
                        self.emit(Instr::LoadUpval(idx));
                        self.compile_expr(expr, false)?;
                        self.emit(Instr::CellSet);
                        Ok(())
                    }
                    Some(Resolved::Local { .. }) | Some(Resolved::Upval { .. }) => self.err(
                        *line,
                        &format!("cannot assign to immutable binding `{}` (use `let mut`)", name),
                    ),
                    Some(Resolved::Global(_)) => self.err(
                        *line,
                        &format!("cannot assign to top-level binding `{}` (top level is immutable; use an atom)", name),
                    ),
                    None => self.err(*line, &format!("unknown variable `{}`", name)),
                }
            }
            Stmt::While { cond, body, line } => {
                self.f().cur_line = *line;
                let start = self.here();
                let entry_depth = self.f().depth;
                self.compile_expr(cond, false)?;
                let exit = self.emit(Instr::JumpIfFalse(0));
                self.f().loops.push(LoopFrame {
                    entry_depth,
                    continue_target: start,
                    break_jumps: Vec::new(),
                });
                self.compile_expr(body, false)?;
                self.emit(Instr::Pop);
                self.emit(Instr::Jump(start));
                self.patch(exit);
                let frame = self.f().loops.pop().unwrap();
                for j in frame.break_jumps {
                    self.patch(j);
                }
                self.f().depth = entry_depth;
                Ok(())
            }
            Stmt::For { pattern, iter, body, line } => {
                self.f().cur_line = *line;
                self.compile_expr(iter, false)?;
                let iter_slot = self.alloc_local();
                self.emit(Instr::StoreLocal(iter_slot));
                let zero = self.konst(Const::Int(0));
                self.emit(Instr::Const(zero));
                let idx_slot = self.alloc_local();
                self.emit(Instr::StoreLocal(idx_slot));
                let start = self.here();
                let entry_depth = self.f().depth;
                let next = self.emit(Instr::IterNext { iter: iter_slot, idx: idx_slot, end: 0 });
                self.begin_scope();
                let pat = self.compile_pattern(pattern)?;
                let pidx = self.add_pat(pat);
                let m = self.emit(Instr::MatchPat { pat: pidx, fail: 0 });
                let j = self.emit(Instr::Jump(0));
                self.patch(m);
                let msg = self.konst(Const::Name("for-loop pattern did not match element".into()));
                self.emit(Instr::Fault(msg));
                self.patch(j);
                self.emit(Instr::Pop);
                self.f().loops.push(LoopFrame {
                    entry_depth,
                    continue_target: start,
                    break_jumps: Vec::new(),
                });
                self.compile_expr(body, false)?;
                self.emit(Instr::Pop);
                self.end_scope();
                self.emit(Instr::Jump(start));
                self.patch(next);
                let frame = self.f().loops.pop().unwrap();
                for jb in frame.break_jumps {
                    self.patch(jb);
                }
                self.f().depth = entry_depth;
                Ok(())
            }
            Stmt::Return { expr, line } => {
                self.f().cur_line = *line;
                match expr {
                    Some(e) => self.compile_expr(e, true)?,
                    None => {
                        self.emit(Instr::Unit);
                    }
                }
                self.emit(Instr::Ret);
                Ok(())
            }
            Stmt::Break { line } => {
                self.f().cur_line = *line;
                let (pops, _target) = self.loop_cleanup(*line, "break")?;
                for _ in 0..pops {
                    self.emit(Instr::Pop);
                }
                let j = self.emit(Instr::Jump(0));
                self.f().loops.last_mut().unwrap().break_jumps.push(j);
                Ok(())
            }
            Stmt::Continue { line } => {
                self.f().cur_line = *line;
                let (pops, target) = self.loop_cleanup(*line, "continue")?;
                for _ in 0..pops {
                    self.emit(Instr::Pop);
                }
                self.emit(Instr::Jump(target));
                Ok(())
            }
            Stmt::Expr(e) => {
                self.compile_expr(e, false)?;
                self.emit(Instr::Pop);
                Ok(())
            }
        }
    }

    /// Pops needed to restore the enclosing loop's entry depth (partial
    /// operands like a match subject), plus the continue target.
    fn loop_cleanup(&mut self, line: u32, what: &str) -> Result<(i32, u32), String> {
        let depth = self.f().depth;
        match self.f().loops.last() {
            Some(frame) => {
                let pops = depth - frame.entry_depth;
                debug_assert!(pops >= 0, "operand depth simulation went negative");
                Ok((pops, frame.continue_target))
            }
            None => self.err(line, &format!("`{}` outside of a loop", what)),
        }
    }

    // ----- expressions -----

    /// Every expression produces exactly one value: pin the simulated depth
    /// to entry+1 on the way out so branch merges and dead code (after
    /// break/continue/return) can't drift the simulation.
    fn compile_expr(&mut self, e: &Expr, tail: bool) -> Result<(), String> {
        let d0 = self.f().depth;
        self.compile_expr_inner(e, tail)?;
        self.f().depth = d0 + 1;
        Ok(())
    }

    fn compile_expr_inner(&mut self, e: &Expr, tail: bool) -> Result<(), String> {
        self.f().cur_line = e.line;
        match &e.kind {
            ExprKind::Unit => {
                self.emit(Instr::Unit);
            }
            ExprKind::Bool(true) => {
                self.emit(Instr::True);
            }
            ExprKind::Bool(false) => {
                self.emit(Instr::False);
            }
            ExprKind::Int(i) => {
                let k = self.konst(Const::Int(*i));
                self.emit(Instr::Const(k));
            }
            ExprKind::Float(f) => {
                let k = self.konst(Const::Float(*f));
                self.emit(Instr::Const(k));
            }
            ExprKind::Str(s) => {
                let k = self.konst(Const::Str(s.clone()));
                self.emit(Instr::Const(k));
            }
            ExprKind::Interp(parts) => {
                let empty = self.konst(Const::Str(String::new()));
                self.emit(Instr::Const(empty));
                for part in parts {
                    match part {
                        InterpPart::Lit(s) => {
                            let k = self.konst(Const::Str(s.clone()));
                            self.emit(Instr::Const(k));
                        }
                        InterpPart::Expr(inner) => {
                            // str(inner)
                            let g = self
                                .ctx
                                .global_ids
                                .get("str")
                                .copied()
                                .ok_or_else(|| format!("line {}: `str` is not defined (prelude missing?)", e.line))?;
                            self.emit(Instr::LoadGlobal(g));
                            self.compile_expr(inner, false)?;
                            self.f().cur_line = e.line;
                            self.emit(Instr::Call(1));
                        }
                    }
                    self.emit(Instr::Add);
                }
            }
            ExprKind::Ident(name) => {
                if name == "_" {
                    return self.err(e.line, "`_` is only meaningful in patterns and pipe holes");
                }
                match self.resolve(name) {
                    Some(Resolved::Local { slot, cell }) => {
                        self.emit(Instr::LoadLocal(slot));
                        if cell {
                            self.emit(Instr::CellGet);
                        }
                    }
                    Some(Resolved::Upval { idx, cell }) => {
                        self.emit(Instr::LoadUpval(idx));
                        if cell {
                            self.emit(Instr::CellGet);
                        }
                    }
                    Some(Resolved::Global(g)) => {
                        self.emit(Instr::LoadGlobal(g));
                    }
                    None => {
                        if self.prefix.is_some() && self.ctx.global_ids.contains_key(name.as_str()) {
                            return self.err(
                                e.line,
                                &format!(
                                    "unknown variable `{}` — it exists outside this module; \
                                     modules only see their own definitions, imports, and \
                                     natives (add `import {{ {} }} from \"...\"`)",
                                    name, name
                                ),
                            );
                        }
                        return self.err(e.line, &format!("unknown variable `{}`", name));
                    }
                }
            }
            ExprKind::Variant { tag, payload } => match payload {
                VariantCtor::Unit => {
                    let t = self.konst(Const::Name(tag.clone()));
                    self.emit(Instr::MakeVariantUnit { tag: t });
                }
                VariantCtor::Positional(args) => {
                    for a in args {
                        self.compile_expr(a, false)?;
                    }
                    let t = self.konst(Const::Name(tag.clone()));
                    self.f().cur_line = e.line;
                    self.emit(Instr::MakeVariantPos { tag: t, count: args.len() as u16 });
                }
                VariantCtor::Named(fields) => {
                    for (_, v) in fields {
                        self.compile_expr(v, false)?;
                    }
                    let t = self.konst(Const::Name(tag.clone()));
                    let names = self.konst(Const::Names(fields.iter().map(|(n, _)| n.clone()).collect()));
                    self.f().cur_line = e.line;
                    self.emit(Instr::MakeVariantNamed { tag: t, names });
                }
            },
            ExprKind::List(items) => {
                for it in items {
                    self.compile_expr(it, false)?;
                }
                self.f().cur_line = e.line;
                self.emit(Instr::MakeList(items.len() as u16));
            }
            ExprKind::Tuple(items) => {
                for it in items {
                    self.compile_expr(it, false)?;
                }
                self.f().cur_line = e.line;
                self.emit(Instr::MakeTuple(items.len() as u16));
            }
            ExprKind::Record { spread, fields } => {
                if let Some(base) = spread {
                    self.compile_expr(base, false)?;
                }
                for (_, v) in fields {
                    self.compile_expr(v, false)?;
                }
                let names = self.konst(Const::Names(fields.iter().map(|(n, _)| n.clone()).collect()));
                self.f().cur_line = e.line;
                if spread.is_some() {
                    self.emit(Instr::RecordUpdate(names));
                } else {
                    self.emit(Instr::MakeRecord(names));
                }
            }
            ExprKind::Lambda { params, body } => {
                self.compile_lambda(params, body, e.line)?;
            }
            ExprKind::Call { callee, args } => {
                if args.len() > u8::MAX as usize {
                    return self.err(e.line, "too many arguments");
                }
                self.compile_expr(callee, false)?;
                for a in args {
                    self.compile_expr(a, false)?;
                }
                self.f().cur_line = e.line;
                if tail {
                    self.emit(Instr::TailCall(args.len() as u8));
                } else {
                    self.emit(Instr::Call(args.len() as u8));
                }
            }
            ExprKind::MethodCall { recv, name, args } => {
                self.compile_expr(recv, false)?;
                for a in args {
                    self.compile_expr(a, false)?;
                }
                let n = self.konst(Const::Name(name.clone()));
                // UFCS fallback function resolved with module-aware scoping at
                // compile time (a record field still wins at runtime)
                let global = self.lookup_global(name);
                self.f().cur_line = e.line;
                self.emit(Instr::Invoke { name: n, global, argc: args.len() as u8 });
            }
            ExprKind::Field { recv, name } => {
                self.compile_expr(recv, false)?;
                let n = self.konst(Const::Name(name.clone()));
                self.f().cur_line = e.line;
                self.emit(Instr::GetField(n));
            }
            ExprKind::Index { recv, index } => {
                self.compile_expr(recv, false)?;
                self.compile_expr(index, false)?;
                self.f().cur_line = e.line;
                self.emit(Instr::Index);
            }
            ExprKind::Unary { op, operand } => {
                self.compile_expr(operand, false)?;
                self.f().cur_line = e.line;
                self.emit(match op {
                    UnOp::Neg => Instr::Neg,
                    UnOp::Not => Instr::Not,
                });
            }
            ExprKind::Binary { op, lhs, rhs } => {
                self.compile_expr(lhs, false)?;
                self.compile_expr(rhs, false)?;
                self.f().cur_line = e.line;
                self.emit(match op {
                    BinOp::Add => Instr::Add,
                    BinOp::Sub => Instr::Sub,
                    BinOp::Mul => Instr::Mul,
                    BinOp::Div => Instr::Div,
                    BinOp::Mod => Instr::Mod,
                    BinOp::Pow => Instr::Pow,
                    BinOp::Eq => Instr::Eq,
                    BinOp::Ne => Instr::Ne,
                    BinOp::Lt => Instr::Lt,
                    BinOp::Le => Instr::Le,
                    BinOp::Gt => Instr::Gt,
                    BinOp::Ge => Instr::Ge,
                });
            }
            ExprKind::And(a, b) => {
                self.compile_expr(a, false)?;
                let j = self.emit(Instr::JumpIfFalsePeek(0));
                self.emit(Instr::Pop);
                self.compile_expr(b, false)?;
                self.patch(j);
            }
            ExprKind::Or(a, b) => {
                self.compile_expr(a, false)?;
                let j = self.emit(Instr::JumpIfTruePeek(0));
                self.emit(Instr::Pop);
                self.compile_expr(b, false)?;
                self.patch(j);
            }
            ExprKind::Range { lo, hi, inclusive } => {
                self.compile_expr(lo, false)?;
                self.compile_expr(hi, false)?;
                self.f().cur_line = e.line;
                self.emit(Instr::MakeRange { inclusive: *inclusive });
            }
            ExprKind::If { cond, then, els } => {
                self.compile_expr(cond, false)?;
                let jelse = self.emit(Instr::JumpIfFalse(0));
                let d_branch = self.f().depth;
                self.compile_expr(then, tail)?;
                let jend = self.emit(Instr::Jump(0));
                self.patch(jelse);
                self.f().depth = d_branch; // else path enters at pre-branch depth
                match els {
                    Some(b) => self.compile_expr(b, tail)?,
                    None => {
                        self.emit(Instr::Unit);
                    }
                }
                self.patch(jend);
            }
            ExprKind::Match { subject, arms } => {
                self.compile_match(subject, arms, tail)?;
            }
            ExprKind::Block(stmts, tail_expr) => {
                self.begin_scope();
                for s in stmts {
                    self.compile_stmt(s)?;
                }
                match tail_expr {
                    Some(te) => self.compile_expr(te, tail)?,
                    None => {
                        self.emit(Instr::Unit);
                    }
                }
                self.end_scope();
            }
            ExprKind::Try(inner) => {
                self.compile_expr(inner, false)?;
                self.f().cur_line = e.line;
                self.emit(Instr::Try);
            }
            ExprKind::Deref(inner) => {
                self.compile_expr(inner, false)?;
                self.f().cur_line = e.line;
                self.emit(Instr::Deref);
            }
        }
        Ok(())
    }

    fn compile_match(&mut self, subject: &Expr, arms: &[Arm], tail: bool) -> Result<(), String> {
        self.compile_expr(subject, false)?;
        let d_subject = self.f().depth; // subject is on the stack here
        let mut end_jumps = Vec::new();
        let mut next_arm: Option<usize> = None;
        for arm in arms {
            if let Some(j) = next_arm.take() {
                self.patch(j);
            }
            self.f().depth = d_subject; // every arm starts with just the subject
            self.f().cur_line = arm.line;
            self.begin_scope();
            let pat = self.compile_pattern(&arm.pattern)?;
            let pidx = self.add_pat(pat);
            let m = self.emit(Instr::MatchPat { pat: pidx, fail: 0 });
            next_arm = Some(m);
            if let Some(g) = &arm.guard {
                self.compile_expr(g, false)?;
                let gj = self.emit(Instr::JumpIfFalse(0));
                // guard failed -> fall to next arm; both jumps target same place
                // we patch both: keep a vec
                // (JumpIfFalse pops the bool, subject still on stack)
                self.emit(Instr::Pop); // subject (guard passed)
                self.compile_expr(&arm.body, tail)?;
                end_jumps.push(self.emit(Instr::Jump(0)));
                // patch guard-fail to here (next arm follows)
                self.patch(gj);
                // we need next_arm (pattern fail) AND guard fail to land here.
                // pattern-fail jump still pending in next_arm — it will be
                // patched at the start of the next arm, which is right here.
            } else {
                self.emit(Instr::Pop); // subject
                self.compile_expr(&arm.body, tail)?;
                end_jumps.push(self.emit(Instr::Jump(0)));
            }
            self.end_scope();
        }
        if let Some(j) = next_arm.take() {
            self.patch(j);
        }
        // no arm matched
        self.f().depth = d_subject;
        self.emit(Instr::Pop); // subject
        let msg = self.konst(Const::Name("no pattern matched in `match`".into()));
        self.emit(Instr::Fault(msg));
        for j in end_jumps {
            self.patch(j);
        }
        Ok(())
    }

    fn add_pat(&mut self, p: Pat) -> u32 {
        let f = self.f();
        f.pats.push(p);
        (f.pats.len() - 1) as u32
    }

    /// Compile an AST pattern: allocate slots for bound names and declare them
    /// in the current scope. Or-pattern alternatives share slots by name.
    fn compile_pattern(&mut self, p: &Pattern) -> Result<Pat, String> {
        let mut slots: HashMap<String, u16> = HashMap::new();
        let pat = self.compile_pattern_inner(p, &mut slots)?;
        Ok(pat)
    }

    fn slot_for(&mut self, name: &str, slots: &mut HashMap<String, u16>) -> u16 {
        if let Some(&s) = slots.get(name) {
            return s;
        }
        let s = self.alloc_local();
        self.declare(name, s, false);
        slots.insert(name.to_string(), s);
        s
    }

    fn compile_pattern_inner(
        &mut self,
        p: &Pattern,
        slots: &mut HashMap<String, u16>,
    ) -> Result<Pat, String> {
        Ok(match p {
            Pattern::Wildcard => Pat::Wildcard,
            Pattern::Bind(n) => Pat::Bind(self.slot_for(n, slots)),
            Pattern::LitInt(i) => Pat::LitInt(*i),
            Pattern::LitFloat(f) => Pat::LitFloat(*f),
            Pattern::LitStr(s) => Pat::LitStr(s.clone()),
            Pattern::LitBool(b) => Pat::LitBool(*b),
            Pattern::LitUnit => Pat::LitUnit,
            Pattern::VariantPos { tag, items } => Pat::VariantPos {
                tag: tag.clone(),
                items: items
                    .iter()
                    .map(|i| self.compile_pattern_inner(i, slots))
                    .collect::<Result<_, _>>()?,
            },
            Pattern::VariantNamed { tag, fields, rest } => Pat::VariantNamed {
                tag: tag.clone(),
                fields: fields
                    .iter()
                    .map(|(n, p)| Ok((n.clone(), self.compile_pattern_inner(p, slots)?)))
                    .collect::<Result<_, String>>()?,
                rest: *rest,
            },
            Pattern::Record { fields, rest } => Pat::Record {
                fields: fields
                    .iter()
                    .map(|(n, p)| Ok((n.clone(), self.compile_pattern_inner(p, slots)?)))
                    .collect::<Result<_, String>>()?,
                rest: *rest,
            },
            Pattern::Tuple(items) => Pat::Tuple(
                items
                    .iter()
                    .map(|i| self.compile_pattern_inner(i, slots))
                    .collect::<Result<_, _>>()?,
            ),
            Pattern::List { items, rest } => Pat::List {
                items: items
                    .iter()
                    .map(|i| self.compile_pattern_inner(i, slots))
                    .collect::<Result<_, _>>()?,
                rest: match rest {
                    None => None,
                    Some(None) => Some(None),
                    Some(Some(n)) => Some(Some(self.slot_for(n, slots))),
                },
            },
            Pattern::Range { lo, hi, inclusive } => Pat::Range { lo: *lo, hi: *hi, inclusive: *inclusive },
            Pattern::Or(alts) => Pat::Or(
                alts.iter()
                    .map(|a| self.compile_pattern_inner(a, slots))
                    .collect::<Result<_, _>>()?,
            ),
            Pattern::As(inner, name) => {
                let pi = self.compile_pattern_inner(inner, slots)?;
                Pat::As(Box::new(pi), self.slot_for(name, slots))
            }
        })
    }
}

// Lambdas need access to the child FnCtx's upvals at MakeClosure time, so
// they are handled here rather than inside compile_expr's big match (which
// can't easily thread the data). We rewrite ExprKind::Lambda before the
// main match via this specialization.
impl<'a> Compiler<'a> {
    fn compile_lambda(&mut self, params: &[Pattern], body: &Expr, line: u32) -> Result<(), String> {
        if params.len() > u8::MAX as usize {
            return self.err(line, "too many parameters");
        }
        let mut fctx = FnCtx::new("<lambda>", params.len() as u8);
        fctx.cur_line = line;
        self.fns.push(fctx);

        let mut destructure: Vec<(u16, &Pattern)> = Vec::new();
        for p in params {
            let slot = self.alloc_local();
            match p {
                Pattern::Bind(n) => self.declare(n, slot, false),
                _ => destructure.push((slot, p)),
            }
        }
        for (slot, p) in destructure {
            self.f().cur_line = line;
            self.emit(Instr::LoadLocal(slot));
            let pat = self.compile_pattern(p)?;
            let pidx = self.add_pat(pat);
            let m = self.emit(Instr::MatchPat { pat: pidx, fail: 0 });
            let j = self.emit(Instr::Jump(0));
            self.patch(m);
            let msg = self.konst(Const::Name("parameter pattern did not match in lambda".into()));
            self.emit(Instr::Fault(msg));
            self.patch(j);
            self.emit(Instr::Pop);
        }
        self.compile_expr(body, true)?;
        self.emit(Instr::Ret);

        let child = self.fns.pop().unwrap();
        let captures: Vec<CaptureSrc> = child.upvals.iter().map(|(_, c, _)| *c).collect();
        let proto = child.finish();
        let fn_id = self.ctx.fn_count;
        self.ctx.fn_count += 1;
        self.protos.push((fn_id, proto));
        self.f().cur_line = line;
        self.emit(Instr::MakeClosure { fn_id, captures });
        Ok(())
    }
}

/// Net operand-stack effect of one instruction on its fall-through path
/// (branch targets are handled by explicit depth resets at compile sites).
fn instr_effect(consts: &[Const], i: &Instr) -> i32 {
    use Instr as I;
    match i {
        I::Const(_)
        | I::Unit
        | I::True
        | I::False
        | I::LoadLocal(_)
        | I::LoadUpval(_)
        | I::LoadGlobal(_)
        | I::Dup
        | I::MakeVariantUnit { .. }
        | I::MakeClosure { .. }
        | I::IterNext { .. } => 1,
        I::StoreLocal(_)
        | I::StoreGlobal(_)
        | I::Pop
        | I::JumpIfFalse(_)
        | I::Index
        | I::MakeRange { .. }
        | I::Ret
        | I::Add
        | I::Sub
        | I::Mul
        | I::Div
        | I::Mod
        | I::Pow
        | I::Eq
        | I::Ne
        | I::Lt
        | I::Le
        | I::Gt
        | I::Ge => -1,
        I::CellSet => -2,
        I::NewCell
        | I::CellGet
        | I::GetField(_)
        | I::Neg
        | I::Not
        | I::Jump(_)
        | I::JumpIfFalsePeek(_)
        | I::JumpIfTruePeek(_)
        | I::MatchPat { .. }
        | I::Deref
        | I::Try
        | I::Fault(_)
        | I::Nop => 0,
        I::MakeList(n) | I::MakeTuple(n) => 1 - *n as i32,
        I::MakeVariantPos { count, .. } => 1 - *count as i32,
        I::MakeRecord(k) | I::MakeVariantNamed { names: k, .. } => match &consts[*k as usize] {
            Const::Names(ns) => 1 - ns.len() as i32,
            _ => 0,
        },
        I::RecordUpdate(k) => match &consts[*k as usize] {
            Const::Names(ns) => -(ns.len() as i32),
            _ => 0,
        },
        I::Call(argc) | I::TailCall(argc) | I::Invoke { argc, .. } => -(*argc as i32),
    }
}
