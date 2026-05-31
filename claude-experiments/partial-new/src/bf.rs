//! A Brainfuck client for the generic engine.
//!
//! This is the *first Futamura projection* in miniature: we partially evaluate
//! a Brainfuck interpreter with respect to a fixed BF program. The program text
//! (and therefore the program counter and the instruction dispatch) is STATIC,
//! so the whole interpreter dispatch loop reduces away at specialization time.
//! The tape, the data pointer, and I/O are DYNAMIC, so they residualize into a
//! straight-line + loops "compiled" program.
//!
//! Two binding-time wins are visible in the output:
//!   * Dispatch removal: the residual has no opcode switch and no `pc`; it is
//!     pure tape arithmetic. (Futamura projection 1.)
//!   * Partially-static pointer: the data pointer is `dynamic_base + static_offset`,
//!     so runs of `+ - < >` coalesce into single offset-addressed updates and a
//!     single pointer move. (Classic BF optimizer output, for free.)

use std::collections::BTreeMap;
use std::fmt;

use crate::engine::{Client, Step};

/// Residual instruction. All addressing is relative to the dynamic data pointer.
#[derive(Debug)]
pub enum Op {
    /// `tape[dp + offset] += delta` (mod 256)
    AddCell { offset: i64, delta: i64 },
    /// `dp += n`
    MovePtr(i64),
    /// `putchar(tape[dp])`
    Output,
    /// `tape[dp] = getchar()`
    Input,
}

/// Residual branch condition.
#[derive(Debug)]
pub enum Cond {
    /// `tape[dp] == 0`
    CellIsZero,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::AddCell { offset, delta } => write!(f, "tape[dp+{offset}] += {delta}"),
            Op::MovePtr(n) => write!(f, "dp += {n}"),
            Op::Output => write!(f, "output tape[dp]"),
            Op::Input => write!(f, "input -> tape[dp]"),
        }
    }
}

impl fmt::Display for Cond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cond::CellIsZero => write!(f, "tape[dp] == 0"),
        }
    }
}

/// Partially-static interpreter state.
#[derive(Clone)]
pub struct State {
    /// Program counter into the static program. STATIC.
    pub pc: usize,
    /// Static offset of the logical data pointer from the last materialized dp.
    pub offset: i64,
    /// Pending, not-yet-residualized cell deltas, keyed by static offset.
    pub cells: BTreeMap<i64, i64>,
}

impl State {
    pub fn start() -> Self {
        State {
            pc: 0,
            offset: 0,
            cells: BTreeMap::new(),
        }
    }
    fn flushed_at(pc: usize) -> Self {
        State {
            pc,
            offset: 0,
            cells: BTreeMap::new(),
        }
    }
    fn is_flushed(&self) -> bool {
        self.offset == 0 && self.cells.is_empty()
    }
}

pub struct Bf {
    /// The program, filtered to the eight commands.
    prog: Vec<u8>,
    /// `jump[i]` is the matching bracket index for a `[` or `]` at `i`.
    jump: Vec<usize>,
}

impl Bf {
    pub fn new(src: &str) -> Self {
        let prog: Vec<u8> = src
            .bytes()
            .filter(|b| matches!(b, b'+' | b'-' | b'<' | b'>' | b'.' | b',' | b'[' | b']'))
            .collect();
        let mut jump = vec![0usize; prog.len()];
        let mut stack = Vec::new();
        for (i, &c) in prog.iter().enumerate() {
            match c {
                b'[' => stack.push(i),
                b']' => {
                    let open = stack.pop().expect("unmatched ] in BF source");
                    jump[i] = open;
                    jump[open] = i;
                }
                _ => {}
            }
        }
        assert!(stack.is_empty(), "unmatched [ in BF source");
        Bf { prog, jump }
    }

    pub fn prog_len(&self) -> usize {
        self.prog.len()
    }

    /// Emit all pending static deltas and pointer movement into the residual,
    /// returning the state to fully-materialized (offset 0, no pending cells).
    fn flush(s: &mut State, out: &mut Vec<Op>) {
        for (&offset, &delta) in &s.cells {
            if delta != 0 {
                out.push(Op::AddCell { offset, delta });
            }
        }
        if s.offset != 0 {
            out.push(Op::MovePtr(s.offset));
        }
        s.cells.clear();
        s.offset = 0;
    }
}

impl Client for Bf {
    type State = State;
    type Key = usize; // the program counter alone; keyed states are always flushed
    type Op = Op;
    type Cond = Cond;

    fn key(&self, s: &State) -> usize {
        debug_assert!(
            s.is_flushed(),
            "key() requires a flushed state; got offset={} pending={}",
            s.offset,
            s.cells.len()
        );
        s.pc
    }

    fn step(&self, s: &mut State, out: &mut Vec<Op>, at_entry: bool) -> Step<Self> {
        if s.pc >= self.prog.len() {
            Bf::flush(s, out);
            return Step::Halt;
        }
        match self.prog[s.pc] {
            b'+' => {
                *s.cells.entry(s.offset).or_insert(0) += 1;
                s.pc += 1;
                Step::Continue
            }
            b'-' => {
                *s.cells.entry(s.offset).or_insert(0) -= 1;
                s.pc += 1;
                Step::Continue
            }
            b'>' => {
                s.offset += 1;
                s.pc += 1;
                Step::Continue
            }
            b'<' => {
                s.offset -= 1;
                s.pc += 1;
                Step::Continue
            }
            b'.' => {
                Bf::flush(s, out); // bring dp to the logical cell, then emit
                out.push(Op::Output);
                s.pc += 1;
                Step::Continue
            }
            b',' => {
                Bf::flush(s, out);
                out.push(Op::Input);
                s.pc += 1;
                Step::Continue
            }
            b'[' => {
                // A `[` is a loop header: it must start its own block so the
                // back-edge from the matching `]` does not re-run preceding ops.
                Bf::flush(s, out);
                if !at_entry {
                    // End the current block; restart fresh at this `[`.
                    return Step::Jump(State::flushed_at(s.pc));
                }
                // We are at the header. Branch on the (dynamic) current cell:
                // zero -> past the loop, nonzero -> into the body.
                let after = State::flushed_at(self.jump[s.pc] + 1);
                let body = State::flushed_at(s.pc + 1);
                Step::Branch {
                    cond: Cond::CellIsZero,
                    t: after,
                    f: body,
                }
            }
            b']' => {
                // Back-edge to the matching header; memoization ties the knot.
                Bf::flush(s, out);
                Step::Jump(State::flushed_at(self.jump[s.pc]))
            }
            other => unreachable!("non-command byte {other} in filtered program"),
        }
    }
}

// ---------------------------------------------------------------------------
// Oracles: a plain BF interpreter and a residual interpreter, used to prove the
// partial evaluation is semantics-preserving.
// ---------------------------------------------------------------------------

const TAPE: usize = 30_000;
const STEP_BUDGET: u64 = 5_000_000_000;

/// Reference Brainfuck interpreter (the ground truth).
pub fn run_reference(src: &str, input: &[u8]) -> Vec<u8> {
    let bf = Bf::new(src);
    let mut tape = vec![0u8; TAPE];
    let mut dp = 0usize;
    let mut pc = 0usize;
    let mut out = Vec::new();
    let mut inp = input.iter().copied();
    let mut budget = STEP_BUDGET;
    while pc < bf.prog.len() {
        budget -= 1;
        if budget == 0 {
            panic!("reference interpreter exceeded step budget");
        }
        match bf.prog[pc] {
            b'+' => tape[dp] = tape[dp].wrapping_add(1),
            b'-' => tape[dp] = tape[dp].wrapping_sub(1),
            b'>' => dp += 1,
            b'<' => dp -= 1,
            b'.' => out.push(tape[dp]),
            b',' => tape[dp] = inp.next().unwrap_or(0),
            b'[' => {
                if tape[dp] == 0 {
                    pc = bf.jump[pc];
                }
            }
            b']' => {
                if tape[dp] != 0 {
                    pc = bf.jump[pc];
                }
            }
            _ => unreachable!(),
        }
        pc += 1;
    }
    out
}

/// Execute the residual program produced by the engine.
pub fn run_residual(prog: &crate::residual::Program<Op, Cond>, input: &[u8]) -> Vec<u8> {
    use crate::residual::Terminator;
    let mut tape = vec![0u8; TAPE];
    let mut dp = 0i64;
    let mut out = Vec::new();
    let mut inp = input.iter().copied();
    let mut bid = prog.entry;
    let mut budget = STEP_BUDGET;
    loop {
        let block = &prog.blocks[bid.0];
        for op in &block.ops {
            match op {
                Op::AddCell { offset, delta } => {
                    let idx = (dp + offset) as usize;
                    let v = tape[idx] as i64 + delta;
                    tape[idx] = v.rem_euclid(256) as u8;
                }
                Op::MovePtr(n) => dp += n,
                Op::Output => out.push(tape[dp as usize]),
                Op::Input => tape[dp as usize] = inp.next().unwrap_or(0),
            }
        }
        match &block.term {
            Terminator::Halt => break,
            Terminator::Br(b) => bid = *b,
            Terminator::Cond {
                cond: Cond::CellIsZero,
                t,
                f,
            } => bid = if tape[dp as usize] == 0 { *t } else { *f },
            Terminator::Unset => panic!("reached unset terminator in b{}", bid.0),
        }
        budget -= 1;
        if budget == 0 {
            panic!("residual interpreter exceeded step budget");
        }
    }
    out
}
