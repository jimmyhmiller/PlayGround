use crate::ir::*;
use std::fmt;

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Eq => write!(f, "eq"),
            CmpOp::Ne => write!(f, "ne"),
            CmpOp::Slt => write!(f, "slt"),
            CmpOp::Sle => write!(f, "sle"),
            CmpOp::Sgt => write!(f, "sgt"),
            CmpOp::Sge => write!(f, "sge"),
            CmpOp::Ult => write!(f, "ult"),
            CmpOp::Ule => write!(f, "ule"),
            CmpOp::Ugt => write!(f, "ugt"),
            CmpOp::Uge => write!(f, "uge"),
        }
    }
}

impl fmt::Display for OverflowOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OverflowOp::SAdd => write!(f, "sadd"),
            OverflowOp::SSub => write!(f, "ssub"),
            OverflowOp::SMul => write!(f, "smul"),
            OverflowOp::UAdd => write!(f, "uadd"),
            OverflowOp::USub => write!(f, "usub"),
            OverflowOp::UMul => write!(f, "umul"),
        }
    }
}

fn fmt_args(args: &[Value]) -> String {
    args.iter()
        .map(|v| format!("{v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inst::Iconst(ty, val) => write!(f, "iconst.{ty} {val}"),
            Inst::F64Const(val) => write!(f, "f64const {val}"),

            Inst::Add(a, b) => write!(f, "add {a}, {b}"),
            Inst::Sub(a, b) => write!(f, "sub {a}, {b}"),
            Inst::Mul(a, b) => write!(f, "mul {a}, {b}"),
            Inst::SDiv(a, b) => write!(f, "sdiv {a}, {b}"),
            Inst::UDiv(a, b) => write!(f, "udiv {a}, {b}"),

            Inst::FAdd(a, b) => write!(f, "fadd {a}, {b}"),
            Inst::FSub(a, b) => write!(f, "fsub {a}, {b}"),
            Inst::FMul(a, b) => write!(f, "fmul {a}, {b}"),
            Inst::FDiv(a, b) => write!(f, "fdiv {a}, {b}"),

            Inst::And(a, b) => write!(f, "and {a}, {b}"),
            Inst::Or(a, b) => write!(f, "or {a}, {b}"),
            Inst::Xor(a, b) => write!(f, "xor {a}, {b}"),
            Inst::Shl(a, b) => write!(f, "shl {a}, {b}"),
            Inst::LShr(a, b) => write!(f, "lshr {a}, {b}"),
            Inst::AShr(a, b) => write!(f, "ashr {a}, {b}"),

            Inst::Neg(v) => write!(f, "neg {v}"),
            Inst::FNeg(v) => write!(f, "fneg {v}"),
            Inst::Not(v) => write!(f, "not {v}"),

            Inst::Icmp(op, a, b) => write!(f, "icmp.{op} {a}, {b}"),
            Inst::Fcmp(op, a, b) => write!(f, "fcmp.{op} {a}, {b}"),

            Inst::Sext(v, ty) => write!(f, "sext {v} -> {ty}"),
            Inst::Zext(v, ty) => write!(f, "zext {v} -> {ty}"),
            Inst::Trunc(v, ty) => write!(f, "trunc {v} -> {ty}"),
            Inst::IntToFloat(v) => write!(f, "int_to_float {v}"),
            Inst::FloatToInt(v) => write!(f, "float_to_int {v}"),
            Inst::Bitcast(v, ty) => write!(f, "bitcast {v} -> {ty}"),

            Inst::Load(ty, addr, off) => write!(f, "load.{ty} [{addr} + {off}]"),
            Inst::Store(val, addr, off) => write!(f, "store {val}, [{addr} + {off}]"),

            Inst::TagOf(v) => write!(f, "tag_of {v}"),
            Inst::Payload(v) => write!(f, "payload {v}"),
            Inst::MakeTagged(tag, v) => write!(f, "make_tagged #{tag}, {v}"),
            Inst::IsTag(v, tag) => write!(f, "is_tag {v}, #{tag}"),

            Inst::Select(c, t, e) => write!(f, "select {c}, {t}, {e}"),

            Inst::OverflowCheck(op, a, b) => write!(f, "overflow_check.{op} {a}, {b}"),
            Inst::Guard(cond, deopt, live) => {
                write!(f, "guard {cond}, deopt#{}", deopt.0)?;
                if !live.is_empty() {
                    write!(f, " [{}]", fmt_args(live))?;
                }
                Ok(())
            }

            Inst::Safepoint(live) => {
                write!(f, "safepoint")?;
                if !live.is_empty() {
                    write!(f, " [{}]", fmt_args(live))?;
                }
                Ok(())
            }

            Inst::Call(fref, args) => write!(f, "call @f{}({})", fref.0, fmt_args(args)),
            Inst::CallIndirect(callee, args, _) => {
                write!(f, "call_indirect {callee}({})", fmt_args(args))
            }
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Ret(v) => write!(f, "ret {v}"),
            Terminator::RetVoid => write!(f, "ret void"),
            Terminator::Jump(target, args) => {
                if args.is_empty() {
                    write!(f, "jump {target}")
                } else {
                    write!(f, "jump {target}({})", fmt_args(args))
                }
            }
            Terminator::BrIf {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
            } => {
                write!(f, "br_if {cond}, {then_block}")?;
                if !then_args.is_empty() {
                    write!(f, "({})", fmt_args(then_args))?;
                }
                write!(f, ", {else_block}")?;
                if !else_args.is_empty() {
                    write!(f, "({})", fmt_args(else_args))?;
                }
                Ok(())
            }
            Terminator::Switch {
                val,
                cases,
                default_block,
                default_args,
            } => {
                write!(f, "switch {val}")?;
                for (case_val, block, args) in cases {
                    write!(f, ", {case_val} => {block}")?;
                    if !args.is_empty() {
                        write!(f, "({})", fmt_args(args))?;
                    }
                }
                write!(f, ", default => {default_block}")?;
                if !default_args.is_empty() {
                    write!(f, "({})", fmt_args(default_args))?;
                }
                Ok(())
            }
            Terminator::Invoke { func, args, normal, normal_args, exception, exception_args } => {
                write!(f, "invoke @f{}({}), {normal}", func.0, fmt_args(args))?;
                if !normal_args.is_empty() {
                    write!(f, "({})", fmt_args(normal_args))?;
                }
                write!(f, ", {exception}")?;
                if !exception_args.is_empty() {
                    write!(f, "({})", fmt_args(exception_args))?;
                }
                Ok(())
            }
            Terminator::InvokeIndirect { callee, args, normal, normal_args, exception, exception_args, .. } => {
                write!(f, "invoke_indirect {callee}({}), {normal}", fmt_args(args))?;
                if !normal_args.is_empty() {
                    write!(f, "({})", fmt_args(normal_args))?;
                }
                write!(f, ", {exception}")?;
                if !exception_args.is_empty() {
                    write!(f, "({})", fmt_args(exception_args))?;
                }
                Ok(())
            }
            Terminator::Unreachable => write!(f, "unreachable"),
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Function header
        write!(f, "fn {}(", self.name)?;
        for (i, ty) in self.sig.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{ty}")?;
        }
        write!(f, ")")?;
        if let Some(ret) = self.sig.ret {
            write!(f, " -> {ret}")?;
        }
        writeln!(f, " {{")?;

        // Extern funcs
        for (i, ext) in self.extern_funcs.iter().enumerate() {
            write!(f, "  declare @f{i} = {}(", ext.name)?;
            for (j, ty) in ext.sig.params.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{ty}")?;
            }
            write!(f, ")")?;
            if let Some(ret) = ext.sig.ret {
                write!(f, " -> {ret}")?;
            }
            writeln!(f)?;
        }
        if !self.extern_funcs.is_empty() {
            writeln!(f)?;
        }

        // Blocks
        for (bi, block) in self.blocks.iter().enumerate() {
            write!(f, "  bb{bi}")?;
            if !block.params.is_empty() {
                write!(f, "(")?;
                for (i, (v, ty)) in block.params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}: {ty}")?;
                }
                write!(f, ")")?;
            }
            writeln!(f, ":")?;

            for inst_node in &block.insts {
                if let Some(v) = inst_node.value {
                    let ty = self.value_type(v);
                    writeln!(f, "    {v}: {ty} = {}", inst_node.inst)?;
                } else {
                    writeln!(f, "    {}", inst_node.inst)?;
                }
            }
            writeln!(f, "    {}", block.terminator)?;
        }

        write!(f, "}}")
    }
}
