// Rust equivalent of examples/small_alloc.gcr — many small, short-lived heap
// allocations. `black_box(Box::new(..))` forces a real malloc/free per iteration
// (without it LLVM elides the non-escaping Box to the stack — gc-rust can't
// elide its opaque ai_gc_alloc_* call, so this keeps the comparison fair).
struct Pair { a: i64, b: i64 }

fn main() {
    let mut i = 0i64;
    let mut acc = 0i64;
    while i < 5000000 {
        let p = std::hint::black_box(Box::new(Pair { a: i, b: i + 1 }));
        acc = acc + p.a + p.b;
        i = i + 1;
    }
    println!("{}", acc);
}
