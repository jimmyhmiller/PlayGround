//! Verifies DWARF type-layout extraction: prints the flattened pointer fields
//! of a few known types from this binary's own DWARF.
//!
//! Expected: Holder has pointer fields at the offsets of its String / Vec / Box
//! / raw-pointer members (each std container flattens down to its data pointer).

use memscope_symbols::TypeOracle;

#[derive(Debug)]
#[allow(dead_code)]
struct Inner {
    a: u64,
    b: u64,
}

#[derive(Debug)]
#[allow(dead_code)]
struct Holder {
    id: u64,                 // offset 0  — scalar, no pointer
    name: String,            // a data pointer somewhere in here
    data: Vec<u32>,          // a data pointer
    child: Option<Box<Inner>>, // niche-optimized nullable pointer
    raw: *const Inner,       // a raw pointer
    flag: bool,
}

fn main() {
    // Force the types into the binary by constructing values.
    let inner = Inner { a: 1, b: 2 };
    let h = Holder {
        id: 7,
        name: "hello".to_string(),
        data: vec![1u32, 2, 3],
        child: Some(Box::new(Inner { a: 3, b: 4 })),
        raw: &inner,
        flag: true,
    };
    std::hint::black_box(&h);

    let oracle = TypeOracle::for_current_process().expect("oracle");
    println!("indexed types: {}", oracle.layout().type_count());

    for name in [
        "Holder",
        "String",
        "Vec<u32, alloc::alloc::Global>",
        "Inner",
    ] {
        match oracle.layout().pointer_fields(name) {
            Some(fields) => {
                println!(
                    "\n{name}  (size {:?})",
                    oracle.layout().size_of(name)
                );
                if fields.is_empty() {
                    println!("  (no pointer fields — leaf type)");
                }
                for f in fields {
                    println!(
                        "  +{:#04x}  -> {}{}",
                        f.offset,
                        f.pointee.as_deref().unwrap_or("?"),
                        if f.in_variant { "  [in enum variant]" } else { "" }
                    );
                }
            }
            None => println!("\n{name}: NOT FOUND in layout index"),
        }
    }
}
