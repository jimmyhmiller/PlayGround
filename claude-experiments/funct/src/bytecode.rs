//! Bytecode. Everything here is plain serializable data — protos (and thus
//! whole programs) can be written to disk and reloaded.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Const {
    Int(i64),
    Float(f64),
    Str(String),
    /// record keys / variant field names / method & tag names
    Name(String),
    Names(Vec<String>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CaptureSrc {
    Local(u16),
    Upval(u16),
}

// `Copy` so the dispatch loop can read an instruction with a trivial register
// copy instead of a clone — the `MakeClosure` capture list lives in a side
// table on `FnProto` (`closure_captures`) rather than inline, to keep this so.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Instr {
    // constants & locals
    Const(u32),
    Unit,
    True,
    False,
    LoadLocal(u16),
    StoreLocal(u16),
    LoadUpval(u16),
    LoadGlobal(u32),
    StoreGlobal(u32),
    /// Wrap top of stack in a fresh Cell (for `let mut`).
    NewCell,
    /// Deref the Cell on top of stack.
    CellGet,
    /// stack: [cell, value] -> set cell, push nothing
    CellSet,
    // data
    MakeList(u16),
    MakeTuple(u16),
    /// const idx -> Names; pops len(names) values
    MakeRecord(u32),
    /// stack: [base-record, v1..vn]; const idx -> Names
    RecordUpdate(u32),
    /// const idx -> Name
    GetField(u32),
    Index,
    /// tag const idx, positional payload count
    MakeVariantPos {
        tag: u32,
        count: u16,
    },
    /// tag const idx, field-names const idx
    MakeVariantNamed {
        tag: u32,
        names: u32,
    },
    /// bare tag (no payload)
    MakeVariantUnit {
        tag: u32,
    },
    // operators
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Neg,
    Not,
    /// pops hi, lo; pushes Range
    MakeRange {
        inclusive: bool,
    },
    // control (absolute targets within the proto)
    Jump(u32),
    JumpIfFalse(u32),
    /// peek (don't pop); used for `and`/`or` short circuit
    JumpIfFalsePeek(u32),
    JumpIfTruePeek(u32),
    /// Peek subject at top of stack, try pattern `pat` (index into proto
    /// patterns). On match: write bindings into locals, fall through.
    /// On fail: jump to `fail`.
    MatchPat {
        pat: u32,
        fail: u32,
    },
    /// `captures` indexes `FnProto::closure_captures` (kept out of line so
    /// `Instr` stays `Copy`).
    MakeClosure {
        fn_id: u32,
        captures: u32,
    },
    Call(u8),
    TailCall(u8),
    /// UFCS: stack [recv, a1..an]; name const idx. `global` is the fallback
    /// function's slot resolved at compile time with module-aware scoping
    /// (a record field named `name` still wins at runtime).
    Invoke {
        name: u32,
        global: Option<u32>,
        argc: u8,
    },
    Ret,
    // atoms / cells
    Deref,
    // errors
    /// `?`: Ok(v)/Some(v) -> v; Err/None -> return from fn; else fault
    Try,
    /// raise fault with message const idx
    Fault(u32),
    // misc
    Pop,
    Dup,
    /// Iterate local[iter] at index local[idx]: if exhausted jump `end`,
    /// else push next element and increment idx.
    IterNext {
        iter: u16,
        idx: u16,
        end: u32,
    },
    Nop,
}

/// A compiled pattern. Bindings carry pre-assigned local slots.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Pat {
    Wildcard,
    Bind(u16),
    LitInt(i64),
    LitFloat(f64),
    LitStr(String),
    LitBool(bool),
    LitUnit,
    VariantPos {
        tag: String,
        items: Vec<Pat>,
    },
    VariantNamed {
        tag: String,
        fields: Vec<(String, Pat)>,
        rest: bool,
    },
    Record {
        fields: Vec<(String, Pat)>,
        rest: bool,
    },
    Tuple(Vec<Pat>),
    List {
        items: Vec<Pat>,
        rest: Option<Option<u16>>,
    },
    Range {
        lo: i64,
        hi: i64,
        inclusive: bool,
    },
    Or(Vec<Pat>),
    As(Box<Pat>, u16),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnProto {
    pub name: String,
    pub arity: u8,
    pub num_locals: u16,
    pub num_upvals: u16,
    pub code: Vec<Instr>,
    pub consts: Vec<Const>,
    pub pats: Vec<Pat>,
    /// source line per instruction (parallel to `code`)
    pub lines: Vec<u32>,
    /// out-of-line capture lists for `MakeClosure` (index = the instr's field)
    pub closure_captures: Vec<Vec<CaptureSrc>>,
}

impl FnProto {
    pub fn line_at(&self, ip: usize) -> u32 {
        self.lines.get(ip).copied().unwrap_or(0)
    }
}
