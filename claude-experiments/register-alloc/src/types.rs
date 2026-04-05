//! Core types used throughout the register allocator.

use std::fmt;

/// A virtual register — the pre-allocation name for a value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(pub u32);

/// A physical register on the target machine.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(pub u16);

/// A register class groups physical registers that are interchangeable
/// for a given purpose (e.g., "general purpose", "float", "vector").
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegClass(pub u8);

/// An operand is either a virtual or physical register.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    Virtual(VReg),
    Physical(PReg),
}

/// A basic block identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// An instruction identifier, unique within a function.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(pub u32);

/// A stack slot for spilled values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpillSlot(pub u32);

/// A program point — identifies a position in the program for liveness.
/// Even numbers are "before" an instruction, odd numbers are "after".
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgPoint(pub u32);

impl ProgPoint {
    pub fn before(inst: InstId) -> Self {
        ProgPoint(inst.0 * 2)
    }

    pub fn after(inst: InstId) -> Self {
        ProgPoint(inst.0 * 2 + 1)
    }

    pub fn inst(self) -> InstId {
        InstId(self.0 / 2)
    }

    pub fn is_before(self) -> bool {
        self.0 % 2 == 0
    }

    pub fn is_after(self) -> bool {
        self.0 % 2 == 1
    }
}

/// How an operand is used by an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperandKind {
    /// The instruction reads this register.
    Use,
    /// The instruction writes this register.
    Def,
    /// The instruction reads and writes this register (e.g., `add r0, r0, r1`).
    UseDef,
    /// Early def: defined before the instruction reads its inputs.
    /// This means the def's live range starts before the instruction,
    /// so it can't share a register with any input.
    EarlyDef,
}

/// A constraint on which physical register(s) an operand can be assigned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperandConstraint {
    /// Any register in the given class.
    RegClass(RegClass),
    /// Must be this exact physical register.
    FixedReg(PReg),
    /// Must be assigned the same physical register as the operand at
    /// the given index in the same instruction. Used for two-address
    /// instructions like x86 `add dst, src` where dst is also an input.
    Tied(usize),
    /// Can be a register or a stack slot (memory operand).
    RegOrStack(RegClass),
    /// Reuses the allocation of the input at the given index.
    /// Like Tied but the allocator inserts a move if needed.
    Reuse(usize),
}

/// A fully described operand: its register, how it's used, and where it can live.
#[derive(Clone, Debug)]
pub struct Operand {
    pub reg: Reg,
    pub kind: OperandKind,
    pub constraint: OperandConstraint,
}

/// A live range: a half-open interval [start, end).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRange {
    pub start: ProgPoint,
    pub end: ProgPoint,
}

impl LiveRange {
    pub fn new(start: ProgPoint, end: ProgPoint) -> Self {
        debug_assert!(start.0 <= end.0);
        LiveRange { start, end }
    }

    pub fn contains(&self, point: ProgPoint) -> bool {
        self.start.0 <= point.0 && point.0 < self.end.0
    }

    pub fn intersects(&self, other: &LiveRange) -> bool {
        self.start.0 < other.end.0 && other.start.0 < self.end.0
    }
}

// ---- Debug/Display impls ----

impl fmt::Debug for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Debug for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p{}", self.0)
    }
}

impl fmt::Display for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p{}", self.0)
    }
}

impl fmt::Debug for RegClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rc{}", self.0)
    }
}

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Debug for InstId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "inst{}", self.0)
    }
}

impl fmt::Debug for SpillSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot{}", self.0)
    }
}

impl fmt::Debug for ProgPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inst = self.inst();
        if self.is_before() {
            write!(f, "{:?}-before", inst)
        } else {
            write!(f, "{:?}-after", inst)
        }
    }
}

impl fmt::Debug for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::Virtual(v) => write!(f, "{:?}", v),
            Reg::Physical(p) => write!(f, "{:?}", p),
        }
    }
}
