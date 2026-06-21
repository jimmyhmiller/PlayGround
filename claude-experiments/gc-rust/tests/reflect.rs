//! Regression tests for runtime reflection metadata and the heap-exploration
//! tooling built on it: the `emit reflect` view (compiler side), the in-language
//! `type_id_of` / `type_name_of` builtins (JIT), and the `dump_heap_text`
//! renderer (runtime side).

use gcrust::codegen::{jit_run_i64_gc, layouts_to_type_meta};
use gcrust::compile::parse_with_prelude;
use gcrust::gc::{FieldTy, TypeKind};
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;

fn lower(src: &str) -> gcrust::core::CoreProgram {
    let (module, _) = parse_with_prelude(src).expect("parse");
    let resolved = resolve_module(module).expect("resolve");
    lower_program(&resolved.globals).expect("lower")
}

fn run(src: &str) -> i64 {
    jit_run_i64_gc(&lower(src), false).expect("run")
}

/// `type_id_of(x)` reads the object header, and `type_name_of(x)` resolves it
/// back through the metadata table — so the id and name must be consistent, and
/// distinct types must differ.
#[test]
fn type_id_and_name_consistent() {
    // Encode the two type names' first bytes + whether the ids differ into the
    // returned i64 so a wrong answer (not just a crash) fails the test.
    let r = run(r#"
        struct Point { x: i64, y: i64 }
        struct Other { a: i64 }
        fn main() -> i64 {
            let p = Point { x: 1, y: 2 };
            let o = Other { a: 3 };
            let same_name = if str_eq(type_name_of(p), "Point") { 1 } else { 0 };
            let other_name = if str_eq(type_name_of(o), "Other") { 1 } else { 0 };
            let differ = if type_id_of(p) == type_id_of(o) { 0 } else { 1 };
            // type_name_of must agree with a fresh instance of the same type.
            let p2 = Point { x: 9, y: 9 };
            let stable = if type_id_of(p) == type_id_of(p2) { 1 } else { 0 };
            same_name + other_name * 10 + differ * 100 + stable * 1000
        }
    "#);
    assert_eq!(r, 1111);
}

/// In-language structural field iteration: `field_count`/`field_name`/
/// `field_kind`/`field_i64` let a program walk any heap object's fields,
/// including an enum's active-variant payload.
#[test]
fn in_language_field_iteration() {
    // Struct: field_count + field_i64 decode each scalar field.
    let r = run(r#"
        struct Mixed { a: i64, b: bool }
        fn main() -> i64 {
            let m = Mixed { a: 40, b: true };
            // a=40 (kind 1), b=true→1 (kind 3): sum values + count.
            field_i64(m, 0) + field_i64(m, 1) + field_count(m)   // 40 + 1 + 2 = 43
        }
    "#);
    assert_eq!(r, 43);

    // field_kind: int=1, bool=3, ref=0.
    let kinds = run(r#"
        struct Mixed { a: i64, b: bool, c: String }
        fn main() -> i64 {
            let m = Mixed { a: 1, b: false, c: "x" };
            field_kind(m, 0) * 100 + field_kind(m, 1) * 10 + field_kind(m, 2)   // 1,3,0 → 130
        }
    "#);
    assert_eq!(kinds, 130);

    // Enum: field_count reflects the *active* variant's payload arity.
    let arity = run(r#"
        enum Shape { Circle(i64), Rect(i64, i64) }
        fn main() -> i64 {
            let a = Shape::Circle(1);
            let b = Shape::Rect(2, 3);
            field_count(a) * 10 + field_count(b)   // 1*10 + 2 = 12
        }
    "#);
    assert_eq!(arity, 12);
}

/// The compiler-built metadata table carries struct field names, offsets, and
/// types; `layouts_to_type_meta` stamps the `type_id` from table position.
#[test]
fn struct_metadata_has_named_typed_fields() {
    let prog = lower(r#"
        struct Point { x: i64, y: i64 }
        fn main() -> i64 { let p = Point { x: 1, y: 2 }; p.x }
    "#);
    let metas = layouts_to_type_meta(&prog);
    let point = metas
        .iter()
        .find(|m| m.name == "Point")
        .expect("Point in metadata");
    // type_id is stamped from table position.
    assert_eq!(point.type_id as usize, metas.iter().position(|m| m.name == "Point").unwrap());
    let TypeKind::Struct { fields } = &point.kind else {
        panic!("Point should be a struct, got {:?}", point.kind);
    };
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["x", "y"]);
    // Pointer-free struct: both i64 scalars, after the 16-byte header.
    assert_eq!(fields[0].offset, 16);
    assert_eq!(fields[1].offset, 24);
    assert!(matches!(fields[0].ty, FieldTy::Scalar(_)));
}

/// Enum metadata records each variant's tag and payload fields, and a pointer
/// payload's `FieldTy::Ref` names the target type (here, the recursive enum).
#[test]
fn enum_metadata_has_variants_and_ref_targets() {
    let prog = lower(r#"
        enum List { Nil, Cons(i64, List) }
        fn main() -> i64 {
            let l = List::Cons(1, List::Nil);   // force the layout to be built
            match l { List::Nil => 0, List::Cons(h, _) => h }
        }
    "#);
    let metas = layouts_to_type_meta(&prog);
    let list = metas.iter().find(|m| m.name == "List").expect("List in metadata");
    let list_id = list.type_id;
    let TypeKind::Enum { variants, .. } = &list.kind else {
        panic!("List should be an enum");
    };
    assert_eq!(variants.len(), 2);
    let nil = variants.iter().find(|v| v.name == "Nil").unwrap();
    assert!(nil.fields.is_empty());
    let cons = variants.iter().find(|v| v.name == "Cons").unwrap();
    assert_eq!(cons.fields.len(), 2);
    // Cons.1 is the recursive `List` pointer — its Ref target is List's own id.
    let tail = &cons.fields[1];
    assert_eq!(tail.ty, FieldTy::Ref(list_id));
}

/// Construction under a real collection: a value held across heavy allocation
/// churn must survive relocation, and objects built while the heap churns must be
/// wired correctly (this is the bug class the ANF pass + `gen_alloc`'s
/// post-allocation reload fix close — a GC value cached in a register across the
/// allocation safepoint would otherwise go stale). Runs on a tiny semi-space so
/// the program's own garbage forces collections mid-construction.
#[test]
fn held_values_survive_collection_during_construction() {
    use gcrust::codegen::{jit_run_i64_mode, GcRunMode};
    let prog = lower(r#"
        fn main() -> i64 {
            // `keep` (0..99) is held across all the churn below.
            let keep: Vec<i64> = vec_new();
            let mut k = keep;
            let mut i = 0;
            while i < 100 { k = vec_push(k, i); i = i + 1; }
            // Churn: build and discard many vecs to force space-full collections
            // while `k` stays live (a frame root that must relocate correctly).
            let mut j = 0;
            while j < 80 {
                let tmp: Vec<i64> = vec_new();
                let mut t = tmp;
                let mut m = 0;
                while m < 60 { t = vec_push(t, m); m = m + 1; }
                j = j + 1;
            }
            // Sum `keep` — 0+1+…+99 = 4950 iff it survived relocation intact.
            let mut s = 0;
            let mut p = 0;
            while p < vec_len(k) { s = s + vec_get(k, p); p = p + 1; }
            s
        }
    "#);
    // 64 KiB semi-space: the ~80×60 vec churn (plus reallocations) far exceeds it,
    // forcing many collections, while the live set stays tiny.
    let r = jit_run_i64_mode(&prog, GcRunMode::SemiSpace(64 << 10)).expect("run");
    assert_eq!(r, 4950);
}

/// Stack rooting for value-with-ref locals: an *indirect* frame root (the
/// address of a ref embedded in a value-local's alloca) must keep that ref alive
/// AND get relocated in place across a collection. Built at the runtime level: a
/// child reachable ONLY through an indirect root must survive and be fixed up.
#[test]
fn indirect_frame_root_relocates_value_interior_ref() {
    use gcrust::gc::{Full, ObjHeader, IdentityPtrPolicy, PtrPolicy, TypeInfo};
    use gcrust::runtime::{self, Frame, FrameOrigin, RuntimeContext};

    // One leaf type; small semi-space so `force_collect` actually moves objects.
    let leaf = TypeInfo::for_header(Full::SIZE).with_type_id(0);
    let mut ctx = RuntimeContext::new(1 << 16, vec![leaf]);
    let thread = ctx.thread_ptr();
    runtime::set_current_thread(thread);

    // Allocate a child. Its identity (pre-GC pointer) is captured via a sentinel
    // we can re-find: we read its type_id after relocation to confirm it's live.
    let child = unsafe { runtime::ai_gc_alloc_fixed(thread, 0, 0) };
    assert!(!child.is_null());

    // Simulate a value-with-ref local's alloca interior: a stack word holding the
    // child pointer. The frame's single INDIRECT slot points at it.
    let mut interior: u64 = child as u64;

    // FrameOrigin with 0 direct, 1 indirect root.
    let origin = FrameOrigin { num_roots: 0, num_indirect: 1, name: std::ptr::null() };
    // Frame memory: [parent][origin][indirect0]  (matches the #[repr(C)] Frame
    // header + trailing slots the GC walker reads).
    let frame_buf: [u64; 3] = [
        0,                                              // parent = null
        &origin as *const FrameOrigin as u64,           // origin
        &mut interior as *mut u64 as u64,               // indirect[0] → &interior
    ];

    unsafe {
        (*thread).top_frame = frame_buf.as_ptr() as *mut Frame;
        ctx.force_collect(&*thread);
        (*thread).top_frame = std::ptr::null_mut();
    }

    // The interior ref was traced (child survived) and fixed up in place.
    let relocated = IdentityPtrPolicy::try_decode_ptr(interior).expect("interior still a pointer");
    assert!(
        ctx.heap().from_space_contains(relocated),
        "indirect root not relocated → value-local interior ref would dangle"
    );
    let tid = unsafe { ctx.heap().obj_type_id(relocated) };
    assert_eq!(tid, 0, "relocated object is not the child leaf type");
    assert_eq!(ctx.heap().collections(), 1);
}

/// POD `#[value]` aggregates are flattened into a heap struct's raw region:
/// stored on construction, loaded on access, and reachable via nested fields.
#[test]
fn value_struct_field_roundtrips_in_heap() {
    let r = run(r#"
        #[value] struct Point { x: i64, y: i64 }
        struct Line { a: Point, b: Point }
        fn main() -> i64 {
            let l = Line { a: Point { x: 3, y: 4 }, b: Point { x: 5, y: 6 } };
            l.a.x + l.a.y + l.b.x + l.b.y
        }
    "#);
    assert_eq!(r, 18);
}

/// A `#[value]` STRUCT containing a GC reference can now be flattened into a heap
/// object: the embedded ref is traced via `interior_ptrs` (heap side) and rooted
/// via indirect frame roots (stack side). It constructs and reads correctly.
#[test]
fn value_struct_with_ref_works_as_heap_field() {
    let r = run(r#"
        struct Node { n: i64 }
        #[value] struct Holder { node: Node, k: i64 }
        struct Outer { h: Holder }
        fn main() -> i64 {
            let o = Outer { h: Holder { node: Node { n: 5 }, k: 7 } };
            o.h.node.n + o.h.k
        }
    "#);
    assert_eq!(r, 12);
}

/// A `#[value]` ENUM carrying a reference works as a heap field via the
/// pointers-first layout: every variant's ref payloads share the leading slots
/// at fixed offsets, so the embedded ref is GC-traceable. The ref survives a real
/// collection (held across churn on a tiny heap that forces space-full GCs).
#[test]
fn value_enum_with_ref_heap_field_works_and_survives_gc() {
    use gcrust::codegen::{jit_run_i64_mode, GcRunMode};
    let prog = lower(r#"
        struct Node { n: i64 }
        #[value] enum Cell { Empty, Full(Node) }
        struct Board { c: Cell }
        fn main() -> i64 {
            let b = Board { c: Cell::Full(Node { n: 1234 }) };
            // Churn garbage to force collections while `b` (holding a value-enum
            // ref) stays live; the embedded ref must trace + relocate.
            let mut i = 0;
            while i < 4000 { let g = Node { n: i }; i = i + 1; }
            match b.c { Cell::Empty => 0, Cell::Full(node) => node.n }
        }
    "#);
    // Plain JIT first (correctness), then a tiny heap to force real GCs.
    assert_eq!(jit_run_i64_mode(&prog, GcRunMode::Generational).expect("run"), 1234);
    assert_eq!(jit_run_i64_mode(&prog, GcRunMode::SemiSpace(64 << 10)).expect("run gc"), 1234);
}

/// A value-enum-with-ref held as a bare LOCAL (not inside a heap object) is
/// rooted via indirect frame roots on its alloca's leading ref slot, so it
/// survives a collection too.
#[test]
fn value_enum_ref_local_survives_gc() {
    use gcrust::codegen::{jit_run_i64_mode, GcRunMode};
    let prog = lower(r#"
        struct Node { n: i64 }
        #[value] enum Cell { Empty, Full(Node) }
        fn churn(n: i64) -> i64 { let mut i = 0; while i < n { let g = Node { n: i }; i = i + 1; } i }
        fn main() -> i64 {
            let c = Cell::Full(Node { n: 777 });   // value-enum-ref local
            let _ = churn(4000);                     // force collections while c is live
            match c { Cell::Empty => 0, Cell::Full(node) => node.n }
        }
    "#);
    assert_eq!(jit_run_i64_mode(&prog, GcRunMode::SemiSpace(64 << 10)).expect("run"), 777);
}

/// Embedded ref offsets in a value-enum heap field land in the shared leading
/// slots (pointers-first), so `Board`'s `interior_ptrs` points at the flattened
/// Cell's ref slot (offset 16 = header for a single value field).
#[test]
fn value_enum_ref_interior_ptr_offset() {
    let prog = lower(r#"
        struct Node { n: i64 }
        #[value] enum Cell { Empty, Full(Node) }
        struct Board { c: Cell }
        fn main() -> i64 { let b = Board { c: Cell::Full(Node { n: 1 }) }; 0 }
    "#);
    let board = prog.layouts.iter().find(|l| l.name == "Board").expect("Board layout");
    assert_eq!(board.interior_ptrs, vec![16u16]);
}

/// The computed `interior_ptrs` locate the embedded ref: `Outer { h: Holder { node, k } }`
/// flattens Holder at the header, so `node`'s ref sits at offset 16.
#[test]
fn interior_ptrs_locate_embedded_ref() {
    let prog = lower(r#"
        struct Node { n: i64 }
        #[value] struct Holder { node: Node, k: i64 }
        struct Outer { h: Holder }
        fn main() -> i64 { let o = Outer { h: Holder { node: Node { n: 1 }, k: 2 } }; o.h.k }
    "#);
    let outer = prog.layouts.iter().find(|l| l.name == "Outer").expect("Outer layout");
    assert_eq!(outer.interior_ptrs, vec![16u16], "the embedded node ref is at offset 16");
}

/// The heap dump recursively renders a flattened value field using the value
/// metadata table (value-relative offsets composed with the field's base).
#[test]
fn heap_dump_renders_flattened_value_field() {
    use gcrust::gc::{Full, ObjHeader};
    use gcrust::gc::{FieldMeta, ScalarKind, TypeInfo, TypeKind, TypeMeta, ValueMeta};
    use gcrust::runtime::{self, RuntimeContext};

    let holder_ti = TypeInfo::for_header(Full::SIZE).with_type_id(0).with_raw_bytes(16);
    let holder_meta = TypeMeta {
        type_id: 0,
        name: "Holder".into(),
        kind: TypeKind::Struct {
            fields: vec![FieldMeta { name: "p".into(), offset: 16, ty: FieldTy::Value(0) }],
        },
    };
    let point_vm = ValueMeta {
        value_id: 0,
        name: "Point".into(),
        kind: TypeKind::Struct {
            fields: vec![
                FieldMeta { name: "x".into(), offset: 0, ty: FieldTy::Scalar(ScalarKind::I64) },
                FieldMeta { name: "y".into(), offset: 8, ty: FieldTy::Scalar(ScalarKind::I64) },
            ],
        },
    };

    let mut ctx = RuntimeContext::new_generational(1 << 20, 8 << 20, vec![holder_ti]);
    ctx.heap().set_type_meta(vec![holder_meta]);
    ctx.heap().set_value_meta(vec![point_vm]);
    let thread = ctx.thread_ptr();
    runtime::set_current_thread(thread);
    let h = unsafe { runtime::ai_gc_alloc_fixed(thread, 0, 0) };
    unsafe {
        (h.add(16) as *mut i64).write(3); // Point.x (value-relative 0)
        (h.add(24) as *mut i64).write(4); // Point.y (value-relative 8)
    }
    let dump = unsafe { gcrust::gc::dump_heap_text(ctx.heap()) };
    assert!(dump.contains("Holder { p: Point { x: 3, y: 4 } }"), "dump was:\n{dump}");
}

/// The runtime heap dump renders the live object graph with names, decoded
/// scalar field values, and pointer fields resolved to other objects' ids. Built
/// at the runtime level (no LLVM): hand-shape two types, allocate them, wire a
/// pointer between them, install metadata, and render.
#[test]
fn heap_dump_renders_object_graph() {
    use gcrust::gc::{FieldMeta, ScalarKind, TypeInfo, TypeKind, TypeMeta};
    use gcrust::gc::{Full, ObjHeader};
    use gcrust::runtime::{self, RuntimeContext};

    // type 0 "Point": two raw i64 fields (x@16, y@24 after the 16-byte header).
    let point_ti = TypeInfo::for_header(Full::SIZE).with_type_id(0).with_raw_bytes(16);
    // type 1 "Boxed": one GC pointer field (inner@16) referencing a Point.
    let boxed_ti = TypeInfo::for_header(Full::SIZE).with_type_id(1).with_fields(1);

    let point_meta = TypeMeta {
        type_id: 0,
        name: "Point".into(),
        kind: TypeKind::Struct {
            fields: vec![
                FieldMeta { name: "x".into(), offset: 16, ty: FieldTy::Scalar(ScalarKind::I64) },
                FieldMeta { name: "y".into(), offset: 24, ty: FieldTy::Scalar(ScalarKind::I64) },
            ],
        },
    };
    let boxed_meta = TypeMeta {
        type_id: 1,
        name: "Boxed".into(),
        kind: TypeKind::Struct {
            fields: vec![FieldMeta { name: "inner".into(), offset: 16, ty: FieldTy::Ref(0) }],
        },
    };

    let mut ctx = RuntimeContext::new_generational(1 << 20, 8 << 20, vec![point_ti, boxed_ti]);
    ctx.heap().set_type_meta(vec![point_meta, boxed_meta]);
    let thread = ctx.thread_ptr();
    runtime::set_current_thread(thread);

    // Allocate a Point and write x=3, y=4. (No GC is triggered for two small
    // allocations, so raw pointers stay valid without rooting.)
    let point = unsafe { runtime::ai_gc_alloc_fixed(thread, 0, 0) };
    unsafe {
        (point.add(16) as *mut i64).write(3);
        (point.add(24) as *mut i64).write(4);
    }
    // Allocate a Boxed whose `inner` points at the Point.
    let boxed = unsafe { runtime::ai_gc_alloc_fixed(thread, 1, 0) };
    unsafe { (boxed.add(16) as *mut *mut u8).write(point) };

    // Root `boxed` for real — the snapshot uses the REAL GC root set (not an
    // in-degree-0 proxy), so an object must be held by an actual root to count as
    // reachable. A permanent-extra RootSource is the simplest GC root here.
    struct Root(std::cell::Cell<u64>);
    impl gcrust::gc::RootSource for Root {
        fn scan_roots(&self, v: &mut dyn FnMut(*mut u64)) {
            v(self.0.as_ptr());
        }
    }
    let root = Root(std::cell::Cell::new(boxed as u64));
    unsafe { ctx.heap().register_permanent_extra(&root as *const dyn gcrust::gc::RootSource) };

    let dump = unsafe { gcrust::gc::dump_heap_text(ctx.heap()) };
    // Point with decoded scalar fields.
    assert!(dump.contains("Point { x: 3, y: 4 }"), "dump was:\n{dump}");
    // Boxed's pointer field resolves to the Point's object id (#0).
    assert!(dump.contains("Boxed { inner: #0 }"), "dump was:\n{dump}");

    // The JSON snapshot of the same heap: summary, histogram, and the edge.
    let json = unsafe { gcrust::gc::dump_heap_json(ctx.heap()) };
    assert!(json.contains("\"objects\": 2"), "json was:\n{json}");
    assert!(json.contains("\"Boxed\""), "json was:\n{json}");
    assert!(json.contains("\"Point\""), "json was:\n{json}");
    // Boxed (id 1) references Point (id 0): an edge "refs": [0].
    assert!(json.contains("\"refs\": [0]"), "json was:\n{json}");
    // Real-rooted reachability: rooted Boxed reaches the Point → both reachable.
    assert!(json.contains("\"reachable_objects\": 2"), "json was:\n{json}");
    assert!(json.contains("\"reachable_bytes\""), "json was:\n{json}");
    assert!(json.contains("\"reachable\": true"), "json was:\n{json}");
    // Versioned snapshot header with the measured STW pause.
    assert!(json.contains("\"snapshot_pause_ns\""), "json was:\n{json}");
}
