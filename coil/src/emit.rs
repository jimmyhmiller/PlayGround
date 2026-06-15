//! Codegen: core forms → MLIR, via the `Backend` trait.
//!
//! This is the "mapping" — the part `lispier` smeared across the parser,
//! expander, post-passes, and `ir_gen`. Here it is one symbol-table-directed
//! walk (ordinary AOT codegen, not interpretation — see AOT.md). It consumes
//! *core* forms: op-calls, `let`, `do`, `block`, `region`, and the explicit
//! `op` form. Surface sugar (`defn`, `(: v t)`, control flow) is the expander's
//! job and lands later; the emitter assumes it has already run.
//!
//! Forms understood:
//!   (op "d.op" :operands [..] :results [..] :attrs {..} :regions [..] :successors [..])
//!   (d.op {attrs}? operand* region*)         ; terse op-call, results inferred
//!   (let [name expr ..] body..)
//!   (do body..)
//!   (region body..) | (region (block ..)..)
//!   (block ^name [(: p t)..] body..)
//!   symbol                                    ; resolves to a bound SSA value
//!   !type / iN                                ; a type (in type position)

use crate::backend::{Backend, BackendError, Handle, NamedAttr, ResultTypes};
use crate::printer;
use crate::value::Val;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct EmitError(pub String);

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "emit error: {}", self.0)
    }
}
impl std::error::Error for EmitError {}

impl From<BackendError> for EmitError {
    fn from(e: BackendError) -> Self {
        EmitError(e.0)
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, EmitError> {
    Err(EmitError(msg.into()))
}

pub struct Emitter<'b, B: Backend> {
    b: &'b mut B,
    /// Lexical scopes mapping names → SSA value handles.
    scopes: Vec<HashMap<String, Handle>>,
    /// The block currently being inserted into.
    cur_block: Option<Handle>,
}

impl<'b, B: Backend> Emitter<'b, B> {
    pub fn new(backend: &'b mut B) -> Self {
        Emitter {
            b: backend,
            scopes: vec![HashMap::new()],
            cur_block: None,
        }
    }

    /// Emit a whole program: create a module, then emit each top-level form into
    /// its body. Returns the module handle.
    pub fn emit_module(&mut self, forms: &[Val]) -> Result<Handle, EmitError> {
        let module = self.b.create_module()?;
        let body = self.b.module_body(module)?;
        self.enter_block(body)?;
        for form in forms {
            self.emit_node(form)?;
        }
        Ok(module)
    }

    // --- scope helpers ---------------------------------------------------

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    fn bind(&mut self, name: &str, h: Handle) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), h);
    }
    fn lookup(&self, name: &str) -> Option<Handle> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }
    fn enter_block(&mut self, block: Handle) -> Result<(), EmitError> {
        self.cur_block = Some(block);
        self.b.set_insertion_end(block)?;
        Ok(())
    }

    // --- nodes -----------------------------------------------------------

    /// Emit a form in statement/value position. Returns its single result value,
    /// if any (used when the form appears as an operand or `let` initializer).
    fn emit_node(&mut self, form: &Val) -> Result<Option<Handle>, EmitError> {
        match form {
            Val::List(items) => self.emit_list(items),
            Val::Sym(name) => match self.lookup(name) {
                Some(h) => Ok(Some(h)),
                None => err(format!("unbound symbol `{name}`")),
            },
            other => err(format!("cannot emit {}", printer::print(other))),
        }
    }

    fn emit_list(&mut self, items: &[Val]) -> Result<Option<Handle>, EmitError> {
        let head = match items.first() {
            Some(Val::Sym(s)) => s.as_ref(),
            _ => return err("cannot emit a list whose head is not a symbol"),
        };
        match head {
            "let" => self.emit_let(&items[1..]),
            "do" => self.emit_do(&items[1..]),
            // (: v t) in value position → a typed constant. The reader/expander
            // leave it; the emitter realizes it as arith.constant (no magic —
            // it lowers to a visible op). `(: name type)` in *parameter* position
            // is handled separately by parse_params.
            ":" => self.emit_typed_const(items),
            "op" => self.emit_explicit_op(&items[1..]).map(first_result),
            "region" | "block" => {
                err(format!("`{head}` is only valid inside an op's :regions"))
            }
            // any dotted symbol is a terse op-call
            name if name.contains('.') => self.emit_op_call(name, &items[1..]).map(first_result),
            other => err(format!("unknown core form `{other}`")),
        }
    }

    /// `(: v t)` → `arith.constant` producing a value of type `t`.
    fn emit_typed_const(&mut self, items: &[Val]) -> Result<Option<Handle>, EmitError> {
        if items.len() != 3 {
            return err("(: value type) takes exactly a value and a type");
        }
        let ty = self.resolve_type(&items[2])?;
        let attr = NamedAttr {
            name: "value".to_string(),
            value: printer::print(&items[1]),
        };
        let results = self
            .b
            .build_op("arith.constant", &[], ResultTypes::Explicit(vec![ty]), &[attr], &[], &[])?;
        Ok(results.into_iter().next())
    }

    fn emit_do(&mut self, body: &[Val]) -> Result<Option<Handle>, EmitError> {
        let mut last = None;
        for f in body {
            last = self.emit_node(f)?;
        }
        Ok(last)
    }

    /// `(let [n e n e ...] body...)` — sequential bindings, value is last body.
    fn emit_let(&mut self, args: &[Val]) -> Result<Option<Handle>, EmitError> {
        let bindings = match args.first() {
            Some(Val::Vec(v)) => v.clone(),
            _ => return err("let requires a binding vector"),
        };
        if bindings.len() % 2 != 0 {
            return err("let binding vector must have an even number of forms");
        }
        self.push_scope();
        let mut i = 0;
        while i < bindings.len() {
            let name = match &bindings[i] {
                Val::Sym(s) => s.clone(),
                other => {
                    self.pop_scope();
                    return err(format!("let binding name must be a symbol, got {}", printer::print(other)));
                }
            };
            let value = match self.emit_node(&bindings[i + 1])? {
                Some(h) => h,
                None => {
                    self.pop_scope();
                    return err(format!("let initializer for `{name}` produced no value"));
                }
            };
            self.bind(&name, value);
            i += 2;
        }
        let result = self.emit_do(&args[1..]);
        self.pop_scope();
        result
    }

    // --- operations ------------------------------------------------------

    /// `(op "name" :operands [..] :results [..] :attrs {..} :regions [..] :successors [..])`
    fn emit_explicit_op(&mut self, args: &[Val]) -> Result<Vec<Handle>, EmitError> {
        let name = match args.first() {
            Some(Val::Str(s)) => s.to_string(),
            Some(Val::Sym(s)) => s.to_string(),
            _ => return err("(op …) requires an operation name string"),
        };
        let kw = parse_keyword_args(&args[1..])?;

        let operand_forms = kw.get_vec("operands");
        let region_forms = kw.get_vec("regions");
        let successor_forms = kw.get_vec("successors");
        let attr_map = kw.get_map("attrs");
        let results = match kw.get("results") {
            None => ResultTypes::Infer,
            Some(Val::Sym(s)) if s.as_ref() == ":infer" => ResultTypes::Infer,
            Some(Val::Vec(ts)) => ResultTypes::Explicit(self.resolve_types(ts)?),
            Some(other) => return err(format!(":results must be a type vector, got {}", printer::print(other))),
        };

        self.build(&name, operand_forms, results, attr_map, region_forms, successor_forms)
    }

    /// `(d.op {attrs}? operand* region*)` — terse; results inferred.
    fn emit_op_call(&mut self, name: &str, args: &[Val]) -> Result<Vec<Handle>, EmitError> {
        // optional leading attribute map
        let (attrs, rest): (&[(Val, Val)], &[Val]) = match args.first() {
            Some(Val::Map(m)) => (m, &args[1..]),
            _ => (&[], args),
        };
        // trailing region forms vs operands
        let regions: Vec<Val> = rest.iter().filter(|v| is_region_form(v)).cloned().collect();
        let operands: Vec<Val> = rest.iter().filter(|v| !is_region_form(v)).cloned().collect();

        self.build(name, &operands, ResultTypes::Infer, attrs, &regions, &[])
    }

    /// The shared op-construction path used by both explicit and terse forms.
    fn build(
        &mut self,
        name: &str,
        operand_forms: &[Val],
        results: ResultTypes,
        attr_map: &[(Val, Val)],
        region_forms: &[Val],
        successor_forms: &[Val],
    ) -> Result<Vec<Handle>, EmitError> {
        // 1. operands are evaluated in the *current* block, before regions.
        let mut operands = Vec::with_capacity(operand_forms.len());
        for f in operand_forms {
            match self.emit_node(f)? {
                Some(h) => operands.push(h),
                None => return err(format!("operand produced no value: {}", printer::print(f))),
            }
        }

        // 2. successors (block references) resolve to bound block handles.
        let mut successors = Vec::with_capacity(successor_forms.len());
        for f in successor_forms {
            let sym = f.as_sym().ok_or_else(|| EmitError("successor must be a block reference".into()))?;
            let h = self.lookup(sym).ok_or_else(|| EmitError(format!("unknown successor `{sym}`")))?;
            successors.push(h);
        }

        // 3. attributes.
        let attrs = self.convert_attrs(attr_map)?;

        // 4. regions — built now; emitting their bodies moves the insertion
        //    point, so save and restore it around region construction.
        let saved = self.cur_block;
        let mut regions = Vec::with_capacity(region_forms.len());
        for rf in region_forms {
            regions.push(self.emit_region(rf)?);
        }
        if let Some(b) = saved {
            self.enter_block(b)?;
        }

        // 5. build the op at the (restored) insertion point.
        let results = self.b.build_op(name, &operands, results, &attrs, &regions, &successors)?;
        Ok(results)
    }

    fn emit_region(&mut self, form: &Val) -> Result<Handle, EmitError> {
        let items = match form {
            Val::List(items) if form.head_sym() == Some("region") => &items[1..],
            _ => return err("expected a (region …) form"),
        };
        let region = self.b.create_region()?;
        let explicit_blocks = items.iter().all(|v| v.head_sym() == Some("block"));
        if !items.is_empty() && explicit_blocks {
            for blk in items {
                self.emit_block(region, blk)?;
            }
        } else {
            // implicit single entry block with no arguments
            let block = self.b.create_block(region, &[])?;
            self.push_scope();
            self.enter_block(block)?;
            self.emit_do(items)?;
            self.pop_scope();
        }
        Ok(region)
    }

    /// `(block ^name [(: p t) ...] body...)`
    fn emit_block(&mut self, region: Handle, form: &Val) -> Result<Handle, EmitError> {
        let items = match form {
            Val::List(items) => items,
            _ => return err("expected a (block …) form"),
        };
        // items[0] = block, items[1] = ^name, items[2] = params vector
        if items.len() < 3 {
            return err("block needs a name and a parameter vector");
        }
        let block_name = items[1].as_sym().ok_or_else(|| EmitError("block name must be a ^label".into()))?;
        let params = match &items[2] {
            Val::Vec(p) => p.clone(),
            _ => return err("block parameters must be a vector"),
        };
        let (names, types) = self.parse_params(&params)?;
        let block = self.b.create_block(region, &types)?;

        // the block label is itself bound (for successors), in the *outer* scope.
        self.bind(block_name, block);

        self.push_scope();
        self.enter_block(block)?;
        for (i, n) in names.iter().enumerate() {
            let arg = self.b.block_arg(block, i)?;
            self.bind(n, arg);
        }
        self.emit_do(&items[3..])?;
        self.pop_scope();
        Ok(block)
    }

    // --- types & attributes ---------------------------------------------

    fn parse_params(&mut self, params: &[Val]) -> Result<(Vec<String>, Vec<Handle>), EmitError> {
        let mut names = Vec::new();
        let mut types = Vec::new();
        for p in params {
            // (: name type)
            let items = match p {
                Val::List(items) if p.head_sym() == Some(":") && items.len() == 3 => items,
                _ => return err(format!("parameter must be (: name type), got {}", printer::print(p))),
            };
            let name = items[1].as_sym().ok_or_else(|| EmitError("parameter name must be a symbol".into()))?;
            names.push(name.to_string());
            types.push(self.resolve_type(&items[2])?);
        }
        Ok((names, types))
    }

    fn resolve_types(&mut self, forms: &[Val]) -> Result<Vec<Handle>, EmitError> {
        forms.iter().map(|t| self.resolve_type(t)).collect()
    }

    fn resolve_type(&mut self, form: &Val) -> Result<Handle, EmitError> {
        match form {
            Val::TypeLit(t) => Ok(self.b.parse_type(t)?),
            Val::Sym(s) => {
                // iN integer types are built directly; everything else parses.
                if let Some(rest) = s.strip_prefix('i') {
                    if let Ok(w) = rest.parse::<u32>() {
                        return Ok(self.b.integer_type(w, true)?);
                    }
                }
                if let Some(rest) = s.strip_prefix('u') {
                    if let Ok(w) = rest.parse::<u32>() {
                        return Ok(self.b.integer_type(w, false)?);
                    }
                }
                Ok(self.b.parse_type(s)?)
            }
            other => err(format!("not a type: {}", printer::print(other))),
        }
    }

    /// Convert an attribute map to `NamedAttr`s. Keyword keys become identifiers;
    /// values are rendered to text (the real backend parses them).
    fn convert_attrs(&mut self, pairs: &[(Val, Val)]) -> Result<Vec<NamedAttr>, EmitError> {
        let mut out = Vec::with_capacity(pairs.len());
        for (k, v) in pairs {
            let name = match k {
                Val::Keyword(s) => s.to_string(),
                Val::Sym(s) => s.to_string(),
                Val::Str(s) => s.to_string(),
                other => return err(format!("attribute key must be a keyword/symbol, got {}", printer::print(other))),
            };
            out.push(NamedAttr {
                name,
                value: printer::print(v),
            });
        }
        Ok(out)
    }
}

fn first_result(results: Vec<Handle>) -> Option<Handle> {
    results.into_iter().next()
}

fn is_region_form(v: &Val) -> bool {
    v.head_sym() == Some("region")
}

// --- keyword-argument parsing for the explicit (op …) form -----------------

struct KwArgs {
    map: HashMap<String, Val>,
}

impl KwArgs {
    fn get(&self, key: &str) -> Option<&Val> {
        self.map.get(key)
    }
    fn get_vec(&self, key: &str) -> &[Val] {
        match self.map.get(key) {
            Some(Val::Vec(v)) => v,
            _ => &[],
        }
    }
    fn get_map(&self, key: &str) -> &[(Val, Val)] {
        match self.map.get(key) {
            Some(Val::Map(m)) => m,
            _ => &[],
        }
    }
}

/// Parse `:key value :key value …` into a map.
fn parse_keyword_args(args: &[Val]) -> Result<KwArgs, EmitError> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let key = match &args[i] {
            Val::Keyword(k) => k.to_string(),
            other => return err(format!("expected a :keyword, got {}", printer::print(other))),
        };
        let value = args
            .get(i + 1)
            .ok_or_else(|| EmitError(format!("keyword :{key} has no value")))?
            .clone();
        map.insert(key, value);
        i += 2;
    }
    Ok(KwArgs { map })
}
