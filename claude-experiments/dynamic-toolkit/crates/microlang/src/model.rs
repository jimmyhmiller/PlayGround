//! The value-model axis, carved at **immediacy**.
//!
//! `Repr` is pure bit-layout: it knows which categories are immediate and how
//! to pack/unpack the immediate ones. It never touches the heap (boxing is the
//! runtime's job, because it needs to allocate). Two real impls, `LowBit` and
//! `NanBox`, differ in exactly one interesting way: which numeric category is
//! immediate. That single `is_immediate` bit is the whole axis.
//!
//! `ValueModel` adds the host semantics the rest of the toolkit needs
//! (equality, hashing, truthiness, type names) and picks its `Repr`. `dynlang`
//! and the collection library would be generic over `ValueModel`; here the
//! tree-walk executor is.

use crate::value::{Cat, HeapId, RawTag, Sym, Val};

/// Bit-level representation. The load-bearing method is `is_immediate`.
pub trait Repr: Copy + 'static {
    const NAME: &'static str;

    /// Which categories live inline (no heap indirection). This is the ONE
    /// thing that flips between LowBit and NanBox.
    fn is_immediate(cat: Cat) -> bool;

    /// What is physically in this word?
    fn tag_of(bits: u64) -> RawTag;

    // Decoders (preconditions: the matching `tag_of`).
    fn imm_int(bits: u64) -> i64;
    fn imm_float(bits: u64) -> f64;
    fn as_bool(bits: u64) -> bool;
    fn as_sym(bits: u64) -> Sym;
    fn as_ref(bits: u64) -> HeapId;

    // Encoders for immediates. `enc_int`/`enc_float` are only called when the
    // category is immediate for this Repr; otherwise the runtime boxes.
    fn enc_int(i: i64) -> u64;
    fn enc_float(f: f64) -> u64;
    fn enc_bool(b: bool) -> u64;
    fn enc_nil() -> u64;
    fn enc_sym(s: Sym) -> u64;
    fn enc_ref(id: HeapId) -> u64;
}

/// The full value model: a `Repr` plus host semantics. Everything above the
/// executor is generic over this.
pub trait ValueModel: 'static + Sized {
    type R: Repr;

    fn type_name(v: Val) -> &'static str {
        match v {
            Val::Int(_) => "int",
            Val::Float(_) => "float",
            Val::Bool(_) => "bool",
            Val::Nil => "nil",
            Val::Sym(_) => "symbol",
            Val::Ref(_) => "object",
        }
    }

    /// Clojure-style truthiness: only `nil` and `false` are falsey.
    fn truthy(v: Val) -> bool {
        !matches!(v, Val::Nil | Val::Bool(false))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// LowBit: integer-primary. Low 3 bits are the tag. Integers are immediate
// (63-bit-ish here we keep it simple with a 61-bit payload); floats are boxed.
// This is what a Clojure-shaped language wants.
// ─────────────────────────────────────────────────────────────────────────

const LB_TAG_BITS: u64 = 3;
const LB_TAG_MASK: u64 = 0b111;
const LB_INT: u64 = 0b000;
const LB_REF: u64 = 0b001;
const LB_BOOL: u64 = 0b010;
const LB_NIL: u64 = 0b011;
const LB_SYM: u64 = 0b100;
// (0b101 = float slot: reserved, unused because LowBit boxes floats)

#[derive(Copy, Clone)]
pub struct LowBit;

impl Repr for LowBit {
    const NAME: &'static str = "LowBit";

    fn is_immediate(cat: Cat) -> bool {
        // Integers are immediate; floats are not.
        matches!(cat, Cat::Int | Cat::Bool | Cat::Nil | Cat::Sym | Cat::Ref)
    }

    fn tag_of(bits: u64) -> RawTag {
        match bits & LB_TAG_MASK {
            LB_INT => RawTag::Int,
            LB_REF => RawTag::Ref,
            LB_BOOL => RawTag::Bool,
            LB_NIL => RawTag::Nil,
            LB_SYM => RawTag::Sym,
            other => panic!("LowBit: bad tag {other:#b}"),
        }
    }

    fn imm_int(bits: u64) -> i64 {
        // arithmetic shift preserves sign
        (bits as i64) >> LB_TAG_BITS
    }
    fn imm_float(_bits: u64) -> f64 {
        unreachable!("LowBit has no immediate float")
    }
    fn as_bool(bits: u64) -> bool {
        (bits >> LB_TAG_BITS) & 1 == 1
    }
    fn as_sym(bits: u64) -> Sym {
        (bits >> LB_TAG_BITS) as Sym
    }
    fn as_ref(bits: u64) -> HeapId {
        (bits >> LB_TAG_BITS) as HeapId
    }

    fn enc_int(i: i64) -> u64 {
        ((i << LB_TAG_BITS) as u64) | LB_INT
    }
    fn enc_float(_f: f64) -> u64 {
        unreachable!("LowBit boxes floats; runtime allocates")
    }
    fn enc_bool(b: bool) -> u64 {
        ((b as u64) << LB_TAG_BITS) | LB_BOOL
    }
    fn enc_nil() -> u64 {
        LB_NIL
    }
    fn enc_sym(s: Sym) -> u64 {
        ((s as u64) << LB_TAG_BITS) | LB_SYM
    }
    fn enc_ref(id: HeapId) -> u64 {
        ((id as u64) << LB_TAG_BITS) | LB_REF
    }
}

pub struct LowBitModel;
impl ValueModel for LowBitModel {
    type R = LowBit;
}

// ─────────────────────────────────────────────────────────────────────────
// NanBox: float-primary. A non-NaN f64 is stored verbatim (immediate float).
// Non-floats are encoded in a reserved quiet-NaN payload space, tagged in the
// high payload bits. Integers are NOT immediate here (this simple scheme boxes
// them) — which is precisely why NaN-boxing is a poor fit for integer-heavy
// code. Production NaN-box variants often store small ints inline; the point
// the sketch makes is the *asymmetry*, not this exact choice.
// ─────────────────────────────────────────────────────────────────────────

// Quiet-NaN pattern: exponent all ones + top mantissa bit. We use the bits
// above that as a small tag, and the low 48 as payload.
const NB_QNAN: u64 = 0x7ff8_0000_0000_0000;
const NB_TAG_SHIFT: u64 = 47;
const NB_TAG_MASK: u64 = 0b111 << NB_TAG_SHIFT;
const NB_PAYLOAD_MASK: u64 = (1 << 47) - 1;
const NB_REF: u64 = 0b001 << NB_TAG_SHIFT;
const NB_BOOL: u64 = 0b010 << NB_TAG_SHIFT;
const NB_NIL: u64 = 0b011 << NB_TAG_SHIFT;
const NB_SYM: u64 = 0b100 << NB_TAG_SHIFT;

#[derive(Copy, Clone)]
pub struct NanBox;

impl NanBox {
    fn is_boxed_bits(bits: u64) -> bool {
        // Our tagged (non-float) values are exactly the words that carry the
        // quiet-NaN pattern with a nonzero tag.
        (bits & NB_QNAN) == NB_QNAN && (bits & NB_TAG_MASK) != 0
    }
}

impl Repr for NanBox {
    const NAME: &'static str = "NanBox";

    fn is_immediate(cat: Cat) -> bool {
        // Floats are immediate; integers are not.
        matches!(cat, Cat::Float | Cat::Bool | Cat::Nil | Cat::Sym | Cat::Ref)
    }

    fn tag_of(bits: u64) -> RawTag {
        if !Self::is_boxed_bits(bits) {
            return RawTag::Float; // any real double (incl. non-tagged NaNs)
        }
        match bits & NB_TAG_MASK {
            NB_REF => RawTag::Ref,
            NB_BOOL => RawTag::Bool,
            NB_NIL => RawTag::Nil,
            NB_SYM => RawTag::Sym,
            other => panic!("NanBox: bad tag {:#b}", other >> NB_TAG_SHIFT),
        }
    }

    fn imm_int(_bits: u64) -> i64 {
        unreachable!("NanBox has no immediate int")
    }
    fn imm_float(bits: u64) -> f64 {
        f64::from_bits(bits)
    }
    fn as_bool(bits: u64) -> bool {
        (bits & NB_PAYLOAD_MASK) != 0
    }
    fn as_sym(bits: u64) -> Sym {
        (bits & NB_PAYLOAD_MASK) as Sym
    }
    fn as_ref(bits: u64) -> HeapId {
        (bits & NB_PAYLOAD_MASK) as HeapId
    }

    fn enc_int(_i: i64) -> u64 {
        unreachable!("NanBox boxes ints; runtime allocates")
    }
    fn enc_float(f: f64) -> u64 {
        // Canonicalize any NaN so it never collides with a tagged value.
        if f.is_nan() {
            NB_QNAN
        } else {
            f.to_bits()
        }
    }
    fn enc_bool(b: bool) -> u64 {
        NB_QNAN | NB_BOOL | (b as u64)
    }
    fn enc_nil() -> u64 {
        NB_QNAN | NB_NIL
    }
    fn enc_sym(s: Sym) -> u64 {
        NB_QNAN | NB_SYM | (s as u64 & NB_PAYLOAD_MASK)
    }
    fn enc_ref(id: HeapId) -> u64 {
        NB_QNAN | NB_REF | (id as u64 & NB_PAYLOAD_MASK)
    }
}

pub struct NanBoxModel;
impl ValueModel for NanBoxModel {
    type R = NanBox;
}

// ─────────────────────────────────────────────────────────────────────────
// HighBit: a THIRD representation, to test that the value axis is genuinely
// free. Tag in the TOP 3 bits, payload in the low 61 — the mirror image of
// LowBit, and a completely different layout from NaN-boxing. Integer-primary
// (ints immediate, floats boxed). Added with ZERO changes to the collector,
// the interpreter, the compiler, or either backend: everything reaches the
// representation only through the `Repr` trait.
// ─────────────────────────────────────────────────────────────────────────

const HB_SHIFT: u64 = 61;
const HB_MASK: u64 = (1 << 61) - 1;
const HB_INT: u64 = 0;
const HB_REF: u64 = 1 << HB_SHIFT;
const HB_BOOL: u64 = 2 << HB_SHIFT;
const HB_NIL: u64 = 3 << HB_SHIFT;
const HB_SYM: u64 = 4 << HB_SHIFT;

#[derive(Copy, Clone)]
pub struct HighBit;

impl Repr for HighBit {
    const NAME: &'static str = "HighBit";

    fn is_immediate(cat: Cat) -> bool {
        matches!(cat, Cat::Int | Cat::Bool | Cat::Nil | Cat::Sym | Cat::Ref)
    }

    fn tag_of(bits: u64) -> RawTag {
        match bits >> HB_SHIFT {
            HB_INT => RawTag::Int,
            1 => RawTag::Ref,
            2 => RawTag::Bool,
            3 => RawTag::Nil,
            4 => RawTag::Sym,
            other => panic!("HighBit: bad tag {other}"),
        }
    }

    fn imm_int(bits: u64) -> i64 {
        // sign-extend the low 61 bits
        let v = bits & HB_MASK;
        ((v << 3) as i64) >> 3
    }
    fn imm_float(_bits: u64) -> f64 {
        unreachable!("HighBit boxes floats")
    }
    fn as_bool(bits: u64) -> bool {
        (bits & 1) == 1
    }
    fn as_sym(bits: u64) -> Sym {
        (bits & HB_MASK) as Sym
    }
    fn as_ref(bits: u64) -> HeapId {
        (bits & HB_MASK) as HeapId
    }

    fn enc_int(i: i64) -> u64 {
        (i as u64) & HB_MASK // top 3 bits cleared => tag 0
    }
    fn enc_float(_f: f64) -> u64 {
        unreachable!("HighBit boxes floats; runtime allocates")
    }
    fn enc_bool(b: bool) -> u64 {
        HB_BOOL | (b as u64)
    }
    fn enc_nil() -> u64 {
        HB_NIL
    }
    fn enc_sym(s: Sym) -> u64 {
        HB_SYM | (s as u64 & HB_MASK)
    }
    fn enc_ref(id: HeapId) -> u64 {
        HB_REF | (id as u64 & HB_MASK)
    }
}

pub struct HighBitModel;
impl ValueModel for HighBitModel {
    type R = HighBit;
}
