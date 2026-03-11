use criterion::{criterion_group, criterion_main, Criterion, black_box};
use dynir::*;

fn build_fib() -> Function {
    let mut b = FunctionBuilder::new("fib", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let base_bb = b.create_block(&[Type::I64]);
    let rec_bb = b.create_block(&[Type::I64]);

    let two = b.iconst(Type::I64, 2);
    let cmp = b.icmp(CmpOp::Slt, n, two);
    b.br_if(cmp, base_bb, &[n], rec_bb, &[n]);

    b.switch_to_block(base_bb);
    let base_n = b.block_param(base_bb, 0);
    b.ret(base_n);

    b.switch_to_block(rec_bb);
    let rec_n = b.block_param(rec_bb, 0);

    let fib_func = b.declare_func(
        "fib",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );

    let one = b.iconst(Type::I64, 1);
    let n1 = b.sub(rec_n, one);
    let r1 = b.call(fib_func, &[n1]).unwrap();

    let two2 = b.iconst(Type::I64, 2);
    let n2 = b.sub(rec_n, two2);
    let r2 = b.call(fib_func, &[n2]).unwrap();

    let result = b.add(r1, r2);
    b.ret(result);

    b.build()
}

fn build_linear_chain(n: usize) -> Function {
    let mut b = FunctionBuilder::new("chain", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let mut v = b.block_param(entry, 0);

    for i in 0..n {
        let c = b.iconst(Type::I64, i as i64);
        v = b.add(v, c);
    }
    b.ret(v);
    b.build()
}

fn build_multi_block(n_blocks: usize) -> Function {
    let mut b = FunctionBuilder::new("multi", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    let mut blocks = Vec::new();
    for _ in 0..n_blocks {
        blocks.push(b.create_block(&[Type::I64]));
    }

    b.jump(blocks[0], &[x]);

    for i in 0..n_blocks {
        b.switch_to_block(blocks[i]);
        let p = b.block_param(blocks[i], 0);
        let c = b.iconst(Type::I64, 1);
        let v = b.add(p, c);

        if i + 1 < n_blocks {
            b.jump(blocks[i + 1], &[v]);
        } else {
            b.ret(v);
        }
    }

    b.build()
}

fn build_diamond() -> Function {
    let mut b = FunctionBuilder::new("diamond", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    let then_bb = b.create_block(&[Type::I64]);
    let else_bb = b.create_block(&[Type::I64]);
    let merge_bb = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    let cmp = b.icmp(CmpOp::Eq, x, zero);
    b.br_if(cmp, then_bb, &[x], else_bb, &[x]);

    b.switch_to_block(then_bb);
    let tv = b.block_param(then_bb, 0);
    let one = b.iconst(Type::I64, 1);
    let r1 = b.add(tv, one);
    b.jump(merge_bb, &[r1]);

    b.switch_to_block(else_bb);
    let ev = b.block_param(else_bb, 0);
    let two = b.iconst(Type::I64, 2);
    let r2 = b.mul(ev, two);
    b.jump(merge_bb, &[r2]);

    b.switch_to_block(merge_bb);
    let m = b.block_param(merge_bb, 0);
    b.ret(m);

    b.build()
}

fn bench_build(c: &mut Criterion) {
    c.bench_function("build_fib", |b| {
        b.iter(|| black_box(build_fib()))
    });

    c.bench_function("build_linear_100", |b| {
        b.iter(|| black_box(build_linear_chain(100)))
    });

    c.bench_function("build_linear_1000", |b| {
        b.iter(|| black_box(build_linear_chain(1000)))
    });

    c.bench_function("build_multi_10_blocks", |b| {
        b.iter(|| black_box(build_multi_block(10)))
    });

    c.bench_function("build_multi_100_blocks", |b| {
        b.iter(|| black_box(build_multi_block(100)))
    });

    c.bench_function("build_diamond", |b| {
        b.iter(|| black_box(build_diamond()))
    });
}

fn bench_verify(c: &mut Criterion) {
    let fib = build_fib();
    let linear_100 = build_linear_chain(100);
    let linear_1000 = build_linear_chain(1000);
    let multi_100 = build_multi_block(100);

    c.bench_function("verify_fib", |b| {
        b.iter(|| verify(black_box(&fib)).unwrap())
    });

    c.bench_function("verify_linear_100", |b| {
        b.iter(|| verify(black_box(&linear_100)).unwrap())
    });

    c.bench_function("verify_linear_1000", |b| {
        b.iter(|| verify(black_box(&linear_1000)).unwrap())
    });

    c.bench_function("verify_multi_100", |b| {
        b.iter(|| verify(black_box(&multi_100)).unwrap())
    });
}

fn bench_display(c: &mut Criterion) {
    let fib = build_fib();
    let linear_1000 = build_linear_chain(1000);

    c.bench_function("display_fib", |b| {
        b.iter(|| black_box(fib.to_string()))
    });

    c.bench_function("display_linear_1000", |b| {
        b.iter(|| black_box(linear_1000.to_string()))
    });
}

criterion_group!(benches, bench_build, bench_verify, bench_display);
criterion_main!(benches);
