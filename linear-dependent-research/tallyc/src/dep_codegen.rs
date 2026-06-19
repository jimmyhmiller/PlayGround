//! A NATIVE backend for the dependent core (`dep.rs`), behind the `llvm` feature.
//!
//! This is the bridge that makes the dependent+linear front end run end to end:
//! a checked `dep::Term` is lowered to LLVM and JIT-executed, with no
//! intermediate normalization — eliminators become real loops, so the recursion
//! happens in native code, not in the type checker's evaluator.
//!
//! Scope (first slice): the runtime fragment over `Nat`-like datatypes (a
//! nullary "zero" constructor and a single-recursive-argument "successor"),
//! represented as `i64`. Types, indices, and proofs are ERASED (they never
//! reach a runtime position in a checked term). General (boxed) datatypes and
//! the memory postulates are future work.

use crate::dep::{Signature, Term};
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::types::IntType;
use inkwell::values::{FunctionValue, IntValue};
use inkwell::OptimizationLevel;

/// Is `data` a `Nat`-like family (no params/indices; one nullary constructor and
/// one constructor with a single recursive argument)? If so, return
/// `(zero_ctor, succ_ctor)` names.
fn nat_like(sig: &Signature, data: &str) -> Option<(String, String)> {
    let d = sig.data(data)?;
    if !d.params.is_empty() || !d.indices.is_empty() || d.ctors.len() != 2 {
        return None;
    }
    let mut zero = None;
    let mut succ = None;
    for c in &d.ctors {
        if c.args.is_empty() {
            zero = Some(c.name.clone());
        } else if c.args.len() == 1 && matches!(&c.args[0].1, Term::Data(dn, _) if dn == data) {
            succ = Some(c.name.clone());
        }
    }
    Some((zero?, succ?))
}

struct DepCg<'c, 'a> {
    ctx: &'c Context,
    i64t: IntType<'c>,
    builder: &'a inkwell::builder::Builder<'c>,
    sig: &'a Signature,
}

impl<'c, 'a> DepCg<'c, 'a> {
    /// Compile a term to an `i64`, given `env` (the value of each de Bruijn var,
    /// innermost last) and the enclosing function (for fresh basic blocks).
    fn compile(&self, f: FunctionValue<'c>, env: &[IntValue<'c>], t: &Term) -> Result<IntValue<'c>, String> {
        match t {
            Term::NatLit(n) => Ok(self.i64t.const_int(*n, false)),
            Term::Zero => Ok(self.i64t.const_int(0, false)),
            Term::Suc(x) => {
                let v = self.compile(f, env, x)?;
                Ok(self.builder.build_int_add(v, self.i64t.const_int(1, false), "suc").unwrap())
            }
            Term::Add(a, b) => {
                let x = self.compile(f, env, a)?;
                let y = self.compile(f, env, b)?;
                Ok(self.builder.build_int_add(x, y, "add").unwrap())
            }
            Term::Var(i) => env
                .get(env.len().wrapping_sub(1).wrapping_sub(*i))
                .copied()
                .ok_or_else(|| format!("unbound runtime variable #{i}")),
            Term::Ann(e, _) => self.compile(f, env, e),
            Term::NatElim(_p, z, s, scrut) => self.compile_fold(f, env, z, s, scrut),
            Term::Constr(name, args) => {
                // a Nat-like constructor: zero ↦ 0, succ x ↦ x+1
                let data = self
                    .sig
                    .datas
                    .iter()
                    .find(|d| d.ctors.iter().any(|c| &c.name == name))
                    .ok_or_else(|| format!("unknown constructor `{name}`"))?;
                let (zero, _succ) = nat_like(self.sig, &data.name)
                    .ok_or_else(|| format!("`{}` is not a Nat-like datatype (only those compile so far)", data.name))?;
                if *name == zero {
                    Ok(self.i64t.const_int(0, false))
                } else {
                    // succ: its single argument is the predecessor
                    let pred = self.compile(f, env, &args[args.len() - 1])?;
                    Ok(self.builder.build_int_add(pred, self.i64t.const_int(1, false), "succ").unwrap())
                }
            }
            Term::Elim(data, _motive, methods, scrut) => {
                let (zero, _succ) = nat_like(self.sig, data)
                    .ok_or_else(|| format!("elim over `{data}`: only Nat-like datatypes compile so far"))?;
                let decl = self.sig.data(data).unwrap();
                let zidx = decl.ctors.iter().position(|c| c.name == zero).unwrap();
                let sidx = 1 - zidx;
                self.compile_fold(f, env, &methods[zidx], &methods[sidx], scrut)
            }
            Term::App(_, _) => {
                // β-reduce a fully-applied spine: (λ…λ. body) a₁ … aₙ
                let (head, args) = flatten_app(t);
                let mut body = strip_ann(head);
                let mut env2 = env.to_vec();
                for a in &args {
                    match body {
                        Term::Lam(inner) => {
                            let v = self.compile(f, env, a)?;
                            env2.push(v);
                            body = strip_ann(inner);
                        }
                        _ => return Err("application of a non-function in runtime code".into()),
                    }
                }
                self.compile(f, &env2, body)
            }
            Term::Const(c) => Err(format!("cannot run the abstract postulate `{c}` (no native impl yet)")),
            other => Err(format!("not a runtime value: {other:?}")),
        }
    }

    /// `elim z s n` as a native loop:  acc = z; for k in 0..n { acc = s k acc }.
    fn compile_fold(&self, f: FunctionValue<'c>, env: &[IntValue<'c>], z: &Term, s: &Term, scrut: &Term) -> Result<IntValue<'c>, String> {
        let zv = self.compile(f, env, z)?;
        let nv = self.compile(f, env, scrut)?;

        // s = λk. λih. body
        let s_body = match strip_ann(s) {
            Term::Lam(b1) => match strip_ann(b1) {
                Term::Lam(b2) => &**b2,
                _ => return Err("eliminator step is not a 2-argument function".into()),
            },
            _ => return Err("eliminator step is not a function".into()),
        };

        let entry = self.builder.get_insert_block().unwrap();
        let cond = self.ctx.append_basic_block(f, "fold.cond");
        let body = self.ctx.append_basic_block(f, "fold.body");
        let exit = self.ctx.append_basic_block(f, "fold.exit");
        self.builder.build_unconditional_branch(cond).unwrap();

        self.builder.position_at_end(cond);
        let k_phi = self.builder.build_phi(self.i64t, "k").unwrap();
        let acc_phi = self.builder.build_phi(self.i64t, "acc").unwrap();
        k_phi.add_incoming(&[(&self.i64t.const_int(0, false), entry)]);
        acc_phi.add_incoming(&[(&zv, entry)]);
        let k_val = k_phi.as_basic_value().into_int_value();
        let acc_val = acc_phi.as_basic_value().into_int_value();
        let more = self
            .builder
            .build_int_compare(inkwell::IntPredicate::ULT, k_val, nv, "more")
            .unwrap();
        self.builder.build_conditional_branch(more, body, exit).unwrap();

        self.builder.position_at_end(body);
        let mut env2 = env.to_vec();
        env2.push(k_val); // k
        env2.push(acc_val); // ih
        let next_acc = self.compile(f, &env2, s_body)?;
        let next_k = self.builder.build_int_add(k_val, self.i64t.const_int(1, false), "k.next").unwrap();
        let body_end = self.builder.get_insert_block().unwrap();
        k_phi.add_incoming(&[(&next_k, body_end)]);
        acc_phi.add_incoming(&[(&next_acc, body_end)]);
        self.builder.build_unconditional_branch(cond).unwrap();

        self.builder.position_at_end(exit);
        Ok(acc_val)
    }
}

fn strip_ann(t: &Term) -> &Term {
    match t {
        Term::Ann(e, _) => strip_ann(e),
        other => other,
    }
}

fn flatten_app(t: &Term) -> (&Term, Vec<&Term>) {
    let mut args = Vec::new();
    let mut head = t;
    while let Term::App(f, a) = head {
        args.push(&**a);
        head = f;
    }
    args.reverse();
    (head, args)
}

/// JIT-compile and run a closed `Nat`-valued term, returning the `i64` it
/// evaluates to. Eliminators run as native loops (no pre-normalization).
pub fn run_nat(sig: &Signature, main: &Term) -> Result<i64, String> {
    let ctx = Context::create();
    let module = ctx.create_module("tally_dep");
    let builder = ctx.create_builder();
    let i64t = ctx.i64_type();

    let f = module.add_function("tally_dep_main", i64t.fn_type(&[], false), None);
    let bb = ctx.append_basic_block(f, "entry");
    builder.position_at_end(bb);

    let cg = DepCg { ctx: &ctx, i64t, builder: &builder, sig };
    let result = cg.compile(f, &[], main)?;
    builder.build_return(Some(&result)).unwrap();

    if f.verify(true) {
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .map_err(|e| e.to_string())?;
        unsafe {
            let func: JitFunction<unsafe extern "C" fn() -> i64> =
                ee.get_function("tally_dep_main").map_err(|e| e.to_string())?;
            Ok(func.call())
        }
    } else {
        Err("generated LLVM function failed verification".into())
    }
}

#[cfg(test)]
mod tests {
    use crate::rust_surface;

    fn run(src: &str) -> i64 {
        let prog = rust_surface::check_program(src).unwrap_or_else(|e| panic!("{e:?}"));
        let (_, _, body) = prog.defs.iter().find(|(n, _, _)| n == "main").expect("no main");
        super::run_nat(&prog.sig, body).unwrap_or_else(|e| panic!("{e}"))
    }

    const NAT: &str = r#"
enum Nat { Zero : Nat, Succ : Nat -> Nat }
add : Nat -> Nat -> Nat
fn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }
mul : Nat -> Nat -> Nat
fn mul(m, n) { match m { Zero => Zero, Succ(k) => add(n, mul(k, n)) } }
"#;

    #[test]
    fn nat_add_runs_natively() {
        // add 2 3 — the eliminator runs as a native loop, returning 5
        let src = format!("{NAT}\nmain : Nat\nfn main() {{ add(Succ(Succ(Zero)), Succ(Succ(Succ(Zero)))) }}\n");
        assert_eq!(run(&src), 5);
    }

    #[test]
    fn nat_mul_runs_natively() {
        // mul 3 4 = 12 — nested eliminators (mul calls add), all native
        let src = format!(
            "{NAT}\nmain : Nat\nfn main() {{ mul(Succ(Succ(Succ(Zero))), Succ(Succ(Succ(Succ(Zero))))) }}\n"
        );
        assert_eq!(run(&src), 12);
    }
}
