//! An MLIR-style control-flow graph.
//!
//! Design choices mirror MLIR:
//!  - **SSA values** (`Value`): every value has exactly one definition (an
//!    instruction result or a block argument).
//!  - **Block arguments are phi nodes.** A branch to a block passes a list of
//!    operands; the block's parameters are the merge points. There is no
//!    explicit `phi` op (exactly MLIR's model).
//!  - **Blocks end in a terminator** that names its successors and the operands
//!    passed to each successor's block arguments.
//!
//! Before SSA construction the CFG still uses `ReadVar`/`WriteVar` ops over
//! named variable slots (MLIR's pre-`mem2reg` "memory" form). SSA construction
//! removes those and introduces block arguments.

use std::collections::HashMap;

/// An SSA value id. Unique within a `Cfg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(pub u32);

/// A basic-block id. `Cfg::blocks[id.0]` is the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// A named-variable slot id (pre-SSA). Interned from `(name, def_scope)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub u32);

/// A compiled function as a CFG.
#[derive(Debug, Clone)]
pub struct Cfg {
    /// Parameter values, in order (definitions live at the entry block).
    pub params: Vec<Value>,
    /// Source names of the parameters, in order.
    pub param_names: Vec<String>,
    /// The compiled function's name, if it is a declaration.
    pub fn_name: Option<String>,
    pub blocks: Vec<Block>,
    pub entry: BlockId,
    /// Interned variable names, indexed by `VarId`.
    pub var_names: Vec<String>,
    next_value: u32,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub id: BlockId,
    /// Block arguments (phi nodes). Empty before SSA construction.
    pub params: Vec<Value>,
    pub instrs: Vec<Instr>,
    pub term: Term,
}

/// An instruction: an optional SSA result plus an operation.
#[derive(Debug, Clone)]
pub struct Instr {
    pub result: Option<Value>,
    pub op: Op,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    Undef,
    Null,
    Bool(bool),
    /// f64 bits, so NaN/-0 stay structural.
    Num(u64),
    Str(String),
}

impl Const {
    pub fn num(f: f64) -> Const {
        Const::Num(f.to_bits())
    }
}

/// Operations. A focused subset that grows as the front-end does. Pre-SSA forms
/// (`ReadVar`/`WriteVar`) are eliminated by SSA construction.
#[derive(Debug, Clone)]
pub enum Op {
    /// A constant.
    Const(Const),
    /// Read the current value of a variable slot (pre-SSA only).
    ReadVar(VarId),
    /// Store into a variable slot (pre-SSA only). No result.
    WriteVar(VarId, Value),
    /// Binary expression.
    Bin(BinOp, Value, Value),
    /// Unary expression.
    Un(UnOp, Value),
    /// Read a global / free identifier by name (unresolved binding).
    Global(String),
    /// `callee(args...)`.
    Call { callee: Value, args: Vec<Value> },
    /// `obj.prop` (static) or `obj[prop]` (computed) — `prop_value` is the key.
    Member { obj: Value, prop: MemberKey },
    /// `obj.prop = value` / `obj[prop] = value`. A **mutation** of `obj`.
    /// Result (if any) is `value`.
    StoreMember { obj: Value, prop: MemberKey, value: Value },
    /// An object literal: (key, value) pairs.
    MakeObject(Vec<(PropKey, Value)>),
    /// An array literal.
    MakeArray(Vec<Value>),
}

#[derive(Debug, Clone)]
pub enum MemberKey {
    /// `obj.name`
    Static(String),
    /// `obj[expr]`
    Computed(Value),
}

#[derive(Debug, Clone)]
pub enum PropKey {
    Ident(String),
    Computed(Value),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, Ne, StrictEq, StrictNe,
    Lt, Le, Gt, Ge,
    BitAnd, BitOr, BitXor, Shl, Shr, UShr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg, Pos, Not, BitNot, TypeOf, Void,
}

/// A block terminator. Successors carry the operands passed to that successor's
/// block arguments (MLIR-style).
#[derive(Debug, Clone)]
pub enum Term {
    /// Unconditional branch.
    Br(BlockId, Vec<Value>),
    /// Conditional branch on a value's truthiness.
    CondBr {
        cond: Value,
        then_block: BlockId,
        then_args: Vec<Value>,
        else_block: BlockId,
        else_args: Vec<Value>,
    },
    /// Return (optionally a value).
    Ret(Option<Value>),
    /// No successor and not a return (e.g. after `throw`); unreachable fallthrough.
    Unreachable,
}

impl Term {
    /// Successor block ids (no operands).
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Term::Br(b, _) => vec![*b],
            Term::CondBr { then_block, else_block, .. } => vec![*then_block, *else_block],
            Term::Ret(_) | Term::Unreachable => vec![],
        }
    }
}

impl Op {
    /// The SSA values this op uses as operands.
    pub fn operands(&self) -> Vec<Value> {
        match self {
            Op::Bin(_, a, b) => vec![*a, *b],
            Op::Un(_, a) => vec![*a],
            Op::WriteVar(_, v) => vec![*v],
            Op::Call { callee, args } => {
                let mut v = vec![*callee];
                v.extend(args.iter().copied());
                v
            }
            Op::Member { obj, prop } => {
                let mut v = vec![*obj];
                if let MemberKey::Computed(c) = prop {
                    v.push(*c);
                }
                v
            }
            Op::StoreMember { obj, prop, value } => {
                let mut v = vec![*obj];
                if let MemberKey::Computed(c) = prop {
                    v.push(*c);
                }
                v.push(*value);
                v
            }
            Op::MakeArray(e) => e.clone(),
            Op::MakeObject(props) => {
                let mut v = Vec::new();
                for (k, val) in props {
                    if let PropKey::Computed(c) = k {
                        v.push(*c);
                    }
                    v.push(*val);
                }
                v
            }
            Op::Const(_) | Op::Global(_) | Op::ReadVar(_) => vec![],
        }
    }
}

impl Cfg {
    pub fn new() -> Cfg {
        Cfg {
            params: Vec::new(),
            param_names: Vec::new(),
            fn_name: None,
            blocks: Vec::new(),
            entry: BlockId(0),
            var_names: Vec::new(),
            next_value: 0,
        }
    }

    pub fn fresh_value(&mut self) -> Value {
        let v = Value(self.next_value);
        self.next_value += 1;
        v
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(Block { id, params: Vec::new(), instrs: Vec::new(), term: Term::Unreachable });
        id
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }
    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.0 as usize]
    }

    /// Predecessors of each block (computed on demand).
    pub fn predecessors(&self) -> HashMap<BlockId, Vec<BlockId>> {
        let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for b in &self.blocks {
            for s in b.term.successors() {
                preds.entry(s).or_default().push(b.id);
            }
        }
        preds
    }

    /// Append an instruction with a fresh result to a block; returns the result.
    pub fn push(&mut self, block: BlockId, op: Op) -> Value {
        let v = self.fresh_value();
        self.block_mut(block).instrs.push(Instr { result: Some(v), op });
        v
    }

    /// Append an instruction with no result (e.g. `WriteVar`).
    pub fn push_effect(&mut self, block: BlockId, op: Op) {
        self.block_mut(block).instrs.push(Instr { result: None, op });
    }
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg::new()
    }
}
