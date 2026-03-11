//! Tests that prove the IR can express real dynamic-language patterns.
//!
//! Each test represents a pattern a JIT compiler needs. If we can build
//! and verify it, the IR is expressive enough for that pattern. Where
//! we find gaps, we document them.

use crate::*;

// ════════════════════════════════════════════════════════════════
// Pattern 1: Polymorphic inline cache (PIC)
//
// The bread and butter of dynamic language JITs. Check the tag of
// a value, take a fast path for the expected type, slow path otherwise.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_pic_add() {
    // add(a, b): if both are ints, do fast add; else call slow_add
    let mut b = FunctionBuilder::new("pic_add", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    let fast_bb = b.create_block(&[Type::I64, Type::I64]);
    let slow_bb = b.create_block(&[Type::I64, Type::I64]);

    let a_is_int = b.is_tag(a, 1); // tag 1 = int
    b.br_if(a_is_int, fast_bb, &[a, bv], slow_bb, &[a, bv]);

    // Fast path: both ints
    b.switch_to_block(fast_bb);
    let fa = b.block_param(fast_bb, 0);
    let fb = b.block_param(fast_bb, 1);

    let check2_bb = b.create_block(&[Type::I64, Type::I64]);
    let fb_is_int = b.is_tag(fb, 1);
    b.br_if(fb_is_int, check2_bb, &[fa, fb], slow_bb, &[fa, fb]);

    b.switch_to_block(check2_bb);
    let ca = b.block_param(check2_bb, 0);
    let cb = b.block_param(check2_bb, 1);
    let pa = b.payload(ca);
    let pb = b.payload(cb);
    let sum = b.add(pa, pb);
    let result = b.make_tagged(1, sum);
    b.ret(result);

    // Slow path
    b.switch_to_block(slow_bb);
    let sa = b.block_param(slow_bb, 0);
    let sb = b.block_param(slow_bb, 1);
    let slow_add = b.declare_func("slow_add", Signature {
        params: vec![Type::I64, Type::I64],
        ret: Some(Type::I64),
    });
    let result = b.call(slow_add, &[sa, sb]).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 2: Object property access
//
// Load a field from a heap object at a known offset.
// obj_ptr = payload(tagged_obj)
// field = load [obj_ptr + offset]
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_property_access() {
    let mut b = FunctionBuilder::new("get_x", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let obj = b.block_param(entry, 0);

    // Extract pointer from tagged value
    let ptr = b.payload(obj);

    // Load field at offset 8 (skip header)
    let field = b.load(Type::I64, ptr, 8);

    b.ret(field);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 3: Closure call
//
// A closure is a heap object: [header | code_ptr | env_ptr]
// To call: load the code pointer, load the env, call_indirect
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_closure_call() {
    let mut b = FunctionBuilder::new("call_closure", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let closure = b.block_param(entry, 0);
    let arg = b.block_param(entry, 1);

    let ptr = b.payload(closure);
    let code_ptr = b.load(Type::Ptr, ptr, 8);  // offset 8: code pointer
    let env = b.load(Type::I64, ptr, 16);       // offset 16: environment

    // Call the closure: fn(env, arg) -> result
    let result = b.call_indirect(code_ptr, &[env, arg], Some(Type::I64)).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 4: Array access with bounds check
//
// array = payload(tagged_arr)
// len = load [array + 0]
// if index < len: load element; else: call out_of_bounds
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_array_access() {
    let mut b = FunctionBuilder::new("array_get", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arr = b.block_param(entry, 0);
    let index = b.block_param(entry, 1);

    let in_bounds_bb = b.create_block(&[Type::I64, Type::I64]); // (arr_ptr, index)
    let oob_bb = b.create_block(&[]);

    let arr_ptr = b.payload(arr);
    let idx = b.payload(index);
    let len = b.load(Type::I64, arr_ptr, 0);
    let in_bounds = b.icmp(CmpOp::Ult, idx, len);
    b.br_if(in_bounds, in_bounds_bb, &[arr_ptr, idx], oob_bb, &[]);

    // In bounds: load element
    b.switch_to_block(in_bounds_bb);
    let ptr = b.block_param(in_bounds_bb, 0);
    let i = b.block_param(in_bounds_bb, 1);
    let eight = b.iconst(Type::I64, 8);
    let byte_offset = b.mul(i, eight);
    let elem_addr = b.add(ptr, byte_offset);
    let elem = b.load(Type::I64, elem_addr, 8); // +8 to skip length
    b.ret(elem);

    // Out of bounds
    b.switch_to_block(oob_bb);
    let trap = b.declare_func("out_of_bounds", Signature { params: vec![], ret: None });
    b.call(trap, &[]);
    b.unreachable();

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 5: Numeric accumulator loop
//
// sum = 0
// for i in 0..n:
//   val = array[i]
//   if is_int(val): sum += payload(val)
//   else: sum += call to_int(val)
// return make_tagged(INT, sum)
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_accumulator_loop() {
    let mut b = FunctionBuilder::new("sum_array", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arr_ptr = b.block_param(entry, 0); // raw pointer to data
    let n = b.block_param(entry, 1);       // raw i64 count

    let loop_bb = b.create_block(&[Type::I64, Type::I64, Type::I64]); // (i, sum, arr_ptr)
    let body_bb = b.create_block(&[Type::I64, Type::I64, Type::I64]); // (i, sum, arr_ptr)
    let fast_bb = b.create_block(&[Type::I64, Type::I64, Type::I64, Type::I64]); // (i, sum, arr, elem)
    let slow_bb = b.create_block(&[Type::I64, Type::I64, Type::I64, Type::I64]); // (i, sum, arr, elem)
    let cont_bb = b.create_block(&[Type::I64, Type::I64, Type::I64]); // (i, new_sum, arr)
    let exit_bb = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[zero, zero, arr_ptr]);

    // Loop header
    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let sum = b.block_param(loop_bb, 1);
    let arr = b.block_param(loop_bb, 2);
    let cmp = b.icmp(CmpOp::Slt, i, n);
    b.br_if(cmp, body_bb, &[i, sum, arr], exit_bb, &[sum]);

    // Body: load element
    b.switch_to_block(body_bb);
    let bi = b.block_param(body_bb, 0);
    let bsum = b.block_param(body_bb, 1);
    let barr = b.block_param(body_bb, 2);
    let eight = b.iconst(Type::I64, 8);
    let offset = b.mul(bi, eight);
    let addr = b.add(barr, offset);
    let elem = b.load(Type::I64, addr, 0);
    let is_int = b.is_tag(elem, 1);
    b.br_if(is_int, fast_bb, &[bi, bsum, barr, elem], slow_bb, &[bi, bsum, barr, elem]);

    // Fast: extract int payload
    b.switch_to_block(fast_bb);
    let fi = b.block_param(fast_bb, 0);
    let fsum = b.block_param(fast_bb, 1);
    let farr = b.block_param(fast_bb, 2);
    let felem = b.block_param(fast_bb, 3);
    let val = b.payload(felem);
    let new_sum = b.add(fsum, val);
    b.jump(cont_bb, &[fi, new_sum, farr]);

    // Slow: call to_int
    b.switch_to_block(slow_bb);
    let si = b.block_param(slow_bb, 0);
    let ssum = b.block_param(slow_bb, 1);
    let sarr = b.block_param(slow_bb, 2);
    let selem = b.block_param(slow_bb, 3);
    let to_int = b.declare_func("to_int", Signature {
        params: vec![Type::I64],
        ret: Some(Type::I64),
    });
    let int_val = b.call(to_int, &[selem]).unwrap();
    let new_sum2 = b.add(ssum, int_val);
    b.jump(cont_bb, &[si, new_sum2, sarr]);

    // Continue: increment i
    b.switch_to_block(cont_bb);
    let ci = b.block_param(cont_bb, 0);
    let csum = b.block_param(cont_bb, 1);
    let carr = b.block_param(cont_bb, 2);
    let one = b.iconst(Type::I64, 1);
    let next_i = b.add(ci, one);
    b.jump(loop_bb, &[next_i, csum, carr]);

    // Exit
    b.switch_to_block(exit_bb);
    let final_sum = b.block_param(exit_bb, 0);
    let result = b.make_tagged(1, final_sum);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 6: Object allocation + field initialization
//
// obj = call gc_alloc(size)
// store type_id, [obj + 0]
// store field1, [obj + 8]
// store field2, [obj + 16]
// call write_barrier(obj)
// return make_tagged(OBJ_TAG, obj)
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_object_allocation() {
    let mut b = FunctionBuilder::new("make_point", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    let y = b.block_param(entry, 1);

    let gc_alloc = b.declare_func("gc_alloc", Signature {
        params: vec![Type::I64],
        ret: Some(Type::Ptr),
    });
    let write_barrier = b.declare_func("write_barrier", Signature {
        params: vec![Type::Ptr],
        ret: None,
    });

    let size = b.iconst(Type::I64, 24); // header + 2 fields
    let obj = b.call(gc_alloc, &[size]).unwrap();

    // Store type descriptor
    let type_id = b.iconst(Type::I64, 42); // Point type ID
    b.store(type_id, obj, 0);

    // Store fields
    b.store(x, obj, 8);
    b.store(y, obj, 16);

    // Write barrier
    b.call(write_barrier, &[obj]);

    // Tag and return
    let result = b.make_tagged(3, obj); // tag 3 = object
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 7: Multi-way type dispatch
//
// With only br_if we need a chain of if/else blocks.
// A `switch` terminator would be much better for dispatching
// on 5+ type tags. See gap analysis at end of file.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_type_dispatch_chain_v2() {
    let mut b = FunctionBuilder::new("to_string", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let val = b.block_param(entry, 0);

    let int_to_str = b.declare_func("int_to_str", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });
    let float_to_str = b.declare_func("float_to_str", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });
    let bool_to_str = b.declare_func("bool_to_str", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });
    let obj_to_str = b.declare_func("obj_to_str", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });

    let int_bb = b.create_block(&[Type::I64]);
    let check_float = b.create_block(&[Type::I64]);
    let float_bb = b.create_block(&[Type::I64]);
    let check_bool = b.create_block(&[Type::I64]);
    let bool_bb = b.create_block(&[Type::I64]);
    let obj_bb = b.create_block(&[Type::I64]); // default

    // tag(val) == 1?
    let tag = b.tag_of(val);
    let one = b.iconst(Type::I32, 1);
    let is_int = b.icmp(CmpOp::Eq, tag, one);
    b.br_if(is_int, int_bb, &[val], check_float, &[val]);

    b.switch_to_block(int_bb);
    let v = b.block_param(int_bb, 0);
    let r = b.call(int_to_str, &[v]).unwrap();
    b.ret(r);

    // tag == 2?
    b.switch_to_block(check_float);
    let v = b.block_param(check_float, 0);
    let tag = b.tag_of(v);
    let two = b.iconst(Type::I32, 2);
    let is_float = b.icmp(CmpOp::Eq, tag, two);
    b.br_if(is_float, float_bb, &[v], check_bool, &[v]);

    b.switch_to_block(float_bb);
    let v = b.block_param(float_bb, 0);
    let r = b.call(float_to_str, &[v]).unwrap();
    b.ret(r);

    // tag == 3?
    b.switch_to_block(check_bool);
    let v = b.block_param(check_bool, 0);
    let tag = b.tag_of(v);
    let three = b.iconst(Type::I32, 3);
    let is_bool = b.icmp(CmpOp::Eq, tag, three);
    b.br_if(is_bool, bool_bb, &[v], obj_bb, &[v]);

    b.switch_to_block(bool_bb);
    let v = b.block_param(bool_bb, 0);
    let r = b.call(bool_to_str, &[v]).unwrap();
    b.ret(r);

    // default: object
    b.switch_to_block(obj_bb);
    let v = b.block_param(obj_bb, 0);
    let r = b.call(obj_to_str, &[v]).unwrap();
    b.ret(r);

    let func = b.build();
    verify(&func).unwrap();

    // NOTE: This works but is 7 blocks for 4-way dispatch.
    // A Switch terminator would collapse this to 1 block + 4 targets.
    assert_eq!(func.blocks.len(), 7);
}

// ════════════════════════════════════════════════════════════════
// Pattern 8: Float unboxing and arithmetic
//
// If both args are floats, unbox, do float math, rebox.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_float_arithmetic() {
    let mut b = FunctionBuilder::new("float_mul", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    // Extract payloads and bitcast to f64
    let a_bits = b.payload(a);
    let b_bits = b.payload(bv);
    let a_f = b.bitcast(a_bits, Type::F64);
    let b_f = b.bitcast(b_bits, Type::F64);

    // Float multiply
    let result_f = b.fmul(a_f, b_f);

    // Rebox
    let result_bits = b.bitcast(result_f, Type::I64);
    let result = b.make_tagged(2, result_bits);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 9: Method dispatch via vtable
//
// obj_ptr = payload(obj)
// vtable = load [obj_ptr + 0]   // first field is vtable ptr
// method = load [vtable + method_offset]
// result = call_indirect method(obj, args...)
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_vtable_dispatch() {
    let mut b = FunctionBuilder::new("call_method", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let obj = b.block_param(entry, 0);
    let arg = b.block_param(entry, 1);

    let obj_ptr = b.payload(obj);
    let vtable = b.load(Type::Ptr, obj_ptr, 0);
    // Method at slot 2 (offset 16)
    let method_ptr = b.load(Type::Ptr, vtable, 16);
    let result = b.call_indirect(method_ptr, &[obj, arg], Some(Type::I64)).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 10: String concatenation
//
// Calling a runtime function with two tagged string values.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_string_concat() {
    let mut b = FunctionBuilder::new("str_concat", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    let concat_fn = b.declare_func("runtime_str_concat", Signature {
        params: vec![Type::I64, Type::I64],
        ret: Some(Type::I64),
    });
    let result = b.call(concat_fn, &[a, bv]).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 11: Exception handling / non-local return
//
// PROBLEM: We have no way to represent try/catch or setjmp/longjmp
// directly. We can model it as calls to runtime functions that use
// setjmp/longjmp, but we can't express "if this call throws, jump
// to this block" in the IR.
//
// For now, runtime calls that throw just don't return. This is
// sufficient for many designs (Lua, early JS engines) but won't
// work for languages that need fine-grained exception handling.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_exception_via_runtime() {
    // try { result = might_throw(x) } catch { result = default }
    // We model this as: call try_might_throw which returns (value, did_throw)
    let mut b = FunctionBuilder::new("try_call", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    // Runtime handles the try/catch and returns a pair encoded as two values
    let try_fn = b.declare_func("try_might_throw", Signature {
        params: vec![Type::I64],
        ret: Some(Type::I64), // returns result (or sentinel on throw)
    });
    let did_throw_fn = b.declare_func("last_call_threw", Signature {
        params: vec![],
        ret: Some(Type::I8),
    });

    let result = b.call(try_fn, &[x]).unwrap();
    let threw = b.call(did_throw_fn, &[]).unwrap();

    let ok_bb = b.create_block(&[Type::I64]);
    let err_bb = b.create_block(&[]);

    b.br_if(threw, err_bb, &[], ok_bb, &[result]);

    b.switch_to_block(ok_bb);
    let v = b.block_param(ok_bb, 0);
    b.ret(v);

    b.switch_to_block(err_bb);
    let default = b.iconst(Type::I64, 0);
    let nil = b.make_tagged(0, default); // tag 0 = nil
    b.ret(nil);

    let func = b.build();
    verify(&func).unwrap();

    // NOTE: This works but is clunky. A proper invoke/landingpad
    // or try_call instruction would be cleaner.
}

// ════════════════════════════════════════════════════════════════
// Pattern 12: Deoptimization / side exit
//
// PROBLEM: In a tracing or method JIT, you want to "bail out"
// to the interpreter when a type guard fails. This needs:
// 1. A way to capture the current state (all live values)
// 2. A way to transfer to the interpreter
//
// We can model this as a call to a deopt function with all live
// values, but there's no first-class guard instruction.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_deopt_guard() {
    let mut b = FunctionBuilder::new("optimized_add", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    let fast_bb = b.create_block(&[Type::I64, Type::I64]);
    let deopt_bb = b.create_block(&[Type::I64, Type::I64]);

    // Guard: both must be ints
    let a_is_int = b.is_tag(a, 1);
    b.br_if(a_is_int, fast_bb, &[a, bv], deopt_bb, &[a, bv]);

    // Fast path
    b.switch_to_block(fast_bb);
    let fa = b.block_param(fast_bb, 0);
    let fb = b.block_param(fast_bb, 1);
    let pa = b.payload(fa);
    let pb = b.payload(fb);
    let sum = b.add(pa, pb);
    let result = b.make_tagged(1, sum);
    b.ret(result);

    // Deopt: call interpreter with bytecode offset + live values
    b.switch_to_block(deopt_bb);
    let da = b.block_param(deopt_bb, 0);
    let db = b.block_param(deopt_bb, 1);
    let deopt = b.declare_func("deoptimize", Signature {
        params: vec![Type::I64, Type::I64, Type::I64], // (bc_offset, a, b)
        ret: Some(Type::I64),
    });
    let bc_offset = b.iconst(Type::I64, 42); // bytecode offset for this point
    let result = b.call(deopt, &[bc_offset, da, db]).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();

    // NOTE: Works, but a first-class Guard instruction would:
    // 1. Be more optimizable (guards can be hoisted/merged)
    // 2. Carry deopt metadata implicitly
    // 3. Not need explicit block args for live values
}

// ════════════════════════════════════════════════════════════════
// Pattern 13: Hash map lookup (complex control flow)
//
// key_hash = call hash(key)
// bucket = key_hash & (capacity - 1)
// loop:
//   entry = load [table + bucket * 16]
//   if entry == empty: return nil
//   if key_eq(entry.key, key): return entry.value
//   bucket = (bucket + 1) & (capacity - 1)
//   goto loop
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_hash_lookup_clean() {
    let mut b = FunctionBuilder::new("hash_get", &[Type::Ptr, Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let table = b.block_param(entry, 0);
    let capacity = b.block_param(entry, 1);
    let key = b.block_param(entry, 2);

    let hash_fn = b.declare_func("hash", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });
    let key_eq_fn = b.declare_func("key_eq", Signature {
        params: vec![Type::I64, Type::I64], ret: Some(Type::I8),
    });

    let hash = b.call(hash_fn, &[key]).unwrap();
    let one = b.iconst(Type::I64, 1);
    let mask = b.sub(capacity, one);
    let start_bucket = b.and(hash, mask);

    let probe_bb = b.create_block(&[Type::I64]); // bucket
    let check_key_bb = b.create_block(&[Type::I64, Type::I64]); // (bucket, entry_key)
    let found_bb = b.create_block(&[Type::I64]); // value
    let not_found_bb = b.create_block(&[]);
    let advance_bb = b.create_block(&[Type::I64]); // bucket

    b.jump(probe_bb, &[start_bucket]);

    // Probe: load entry, check empty
    b.switch_to_block(probe_bb);
    let bucket = b.block_param(probe_bb, 0);
    let sixteen = b.iconst(Type::I64, 16);
    let byte_off = b.mul(bucket, sixteen);
    let entry_addr = b.add(table, byte_off);
    let entry_key = b.load(Type::I64, entry_addr, 0);
    let is_empty = b.is_tag(entry_key, 0);
    b.br_if(is_empty, not_found_bb, &[], check_key_bb, &[bucket, entry_key]);

    // Check key equality
    b.switch_to_block(check_key_bb);
    let ck_bucket = b.block_param(check_key_bb, 0);
    let ck_ekey = b.block_param(check_key_bb, 1);
    let eq = b.call(key_eq_fn, &[ck_ekey, key]).unwrap();
    // If equal, load value and return; else advance
    let load_val_bb = b.create_block(&[Type::I64]); // bucket
    b.br_if(eq, load_val_bb, &[ck_bucket], advance_bb, &[ck_bucket]);

    // Load value from matched entry
    b.switch_to_block(load_val_bb);
    let lv_bucket = b.block_param(load_val_bb, 0);
    let lv_sixteen = b.iconst(Type::I64, 16);
    let lv_off = b.mul(lv_bucket, lv_sixteen);
    let lv_addr = b.add(table, lv_off);
    let value = b.load(Type::I64, lv_addr, 8); // value at offset 8
    b.jump(found_bb, &[value]);

    // Advance to next bucket
    b.switch_to_block(advance_bb);
    let adv_bucket = b.block_param(advance_bb, 0);
    let adv_one = b.iconst(Type::I64, 1);
    let next = b.add(adv_bucket, adv_one);
    let wrapped = b.and(next, mask);
    b.jump(probe_bb, &[wrapped]);

    // Found
    b.switch_to_block(found_bb);
    let result = b.block_param(found_bb, 0);
    b.ret(result);

    // Not found: return nil
    b.switch_to_block(not_found_bb);
    let nil_pay = b.iconst(Type::I64, 0);
    let nil = b.make_tagged(0, nil_pay);
    b.ret(nil);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 14: Varargs / rest parameters
//
// Can we represent a function that takes a variable number of args?
// Answer: not directly. The caller would pack args into an array
// and pass the array + count. This is fine — most dynamic languages
// do this anyway.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_varargs() {
    // apply(fn, args_array, argc) -> result
    let mut b = FunctionBuilder::new("apply", &[Type::I64, Type::Ptr, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let func_val = b.block_param(entry, 0);
    let args = b.block_param(entry, 1);
    let argc = b.block_param(entry, 2);

    // Just delegate to a runtime function
    let rt_apply = b.declare_func("runtime_apply", Signature {
        params: vec![Type::I64, Type::Ptr, Type::I64],
        ret: Some(Type::I64),
    });
    let result = b.call(rt_apply, &[func_val, args, argc]).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

// ════════════════════════════════════════════════════════════════
// Pattern 15: Multi-way type dispatch with Switch (FIXED)
//
// Previously required 7 blocks with br_if chains.
// Now: 1 entry block + 4 handler blocks = 5 blocks.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_type_dispatch_switch() {
    let mut b = FunctionBuilder::new("to_string", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let val = b.block_param(entry, 0);

    let int_to_str = b.declare_func("int_to_str", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });
    let float_to_str = b.declare_func("float_to_str", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });
    let bool_to_str = b.declare_func("bool_to_str", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });
    let obj_to_str = b.declare_func("obj_to_str", Signature {
        params: vec![Type::I64], ret: Some(Type::I64),
    });

    let int_bb = b.create_block(&[Type::I64]);
    let float_bb = b.create_block(&[Type::I64]);
    let bool_bb = b.create_block(&[Type::I64]);
    let obj_bb = b.create_block(&[Type::I64]); // default

    let tag = b.tag_of(val);
    b.switch(
        tag,
        &[
            (1, int_bb, &[val]),
            (2, float_bb, &[val]),
            (3, bool_bb, &[val]),
        ],
        obj_bb, &[val],
    );

    b.switch_to_block(int_bb);
    let v = b.block_param(int_bb, 0);
    let r = b.call(int_to_str, &[v]).unwrap();
    b.ret(r);

    b.switch_to_block(float_bb);
    let v = b.block_param(float_bb, 0);
    let r = b.call(float_to_str, &[v]).unwrap();
    b.ret(r);

    b.switch_to_block(bool_bb);
    let v = b.block_param(bool_bb, 0);
    let r = b.call(bool_to_str, &[v]).unwrap();
    b.ret(r);

    b.switch_to_block(obj_bb);
    let v = b.block_param(obj_bb, 0);
    let r = b.call(obj_to_str, &[v]).unwrap();
    b.ret(r);

    let func = b.build();
    verify(&func).unwrap();

    // 5 blocks: entry + 4 handlers. Down from 7 with br_if chains!
    assert_eq!(func.blocks.len(), 5);

    let s = func.to_string();
    assert!(s.contains("switch"));
}

// ════════════════════════════════════════════════════════════════
// Pattern 16: Float comparison
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_float_comparison() {
    let mut b = FunctionBuilder::new("fmax", &[Type::F64, Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    let cmp = b.fcmp(CmpOp::Sgt, a, bv);
    let result = b.select(cmp, a, bv);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
    assert!(func.to_string().contains("fcmp.sgt"));
}

// ════════════════════════════════════════════════════════════════
// Pattern 17: Unary negation and bitwise not
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_unary_ops() {
    let mut b = FunctionBuilder::new("negate_and_flip", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    let negated = b.neg(x);
    let flipped = b.not(negated);
    b.ret(flipped);

    let func = b.build();
    verify(&func).unwrap();
    let s = func.to_string();
    assert!(s.contains("neg"));
    assert!(s.contains("not"));
}

#[test]
fn pattern_float_negate() {
    let mut b = FunctionBuilder::new("fneg_test", &[Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    let neg = b.fneg(x);
    b.ret(neg);

    let func = b.build();
    verify(&func).unwrap();
    assert!(func.to_string().contains("fneg"));
}

// ════════════════════════════════════════════════════════════════
// Pattern 18: Float-aware polymorphic add
//
// if is_int(a) && is_int(b): int add
// elif is_float(a) && is_float(b): float add with fcmp
// else: slow path
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_polymorphic_add_with_floats() {
    let mut b = FunctionBuilder::new("poly_add", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    let slow_bb = b.create_block(&[Type::I64, Type::I64]);
    let int_bb = b.create_block(&[Type::I64, Type::I64]);
    let float_bb = b.create_block(&[Type::I64, Type::I64]);

    // Dispatch on tag of a
    let a_tag = b.tag_of(a);
    b.switch(
        a_tag,
        &[
            (1, int_bb, &[a, bv]),   // tag 1 = int
            (2, float_bb, &[a, bv]), // tag 2 = float
        ],
        slow_bb, &[a, bv],
    );

    // Int path
    b.switch_to_block(int_bb);
    let ia = b.block_param(int_bb, 0);
    let ib = b.block_param(int_bb, 1);
    let pa = b.payload(ia);
    let pb = b.payload(ib);
    let sum = b.add(pa, pb);
    let result = b.make_tagged(1, sum);
    b.ret(result);

    // Float path
    b.switch_to_block(float_bb);
    let fa = b.block_param(float_bb, 0);
    let fb = b.block_param(float_bb, 1);
    let fa_bits = b.payload(fa);
    let fb_bits = b.payload(fb);
    let fa_f = b.bitcast(fa_bits, Type::F64);
    let fb_f = b.bitcast(fb_bits, Type::F64);
    let fsum = b.fadd(fa_f, fb_f);
    let result_bits = b.bitcast(fsum, Type::I64);
    let result = b.make_tagged(2, result_bits);
    b.ret(result);

    // Slow path
    b.switch_to_block(slow_bb);
    let sa = b.block_param(slow_bb, 0);
    let sb = b.block_param(slow_bb, 1);
    let slow_add = b.declare_func("slow_add", Signature {
        params: vec![Type::I64, Type::I64],
        ret: Some(Type::I64),
    });
    let result = b.call(slow_add, &[sa, sb]).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
    assert!(func.to_string().contains("switch"));
}

// ════════════════════════════════════════════════════════════════
// Pattern 19: Overflow-checked fixnum arithmetic
//
// Add two fixnums, check for overflow, promote to bignum if needed.
// Uses OverflowCheck instruction paired with regular Add.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_overflow_checked_add() {
    let mut b = FunctionBuilder::new("fixnum_add", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    let fast_bb = b.create_block(&[Type::I64]); // result
    let slow_bb = b.create_block(&[Type::I64, Type::I64]); // (a, b) for bignum

    // Extract payloads
    let pa = b.payload(a);
    let pb = b.payload(bv);

    // Do the add and check overflow
    let sum = b.add(pa, pb);
    let overflowed = b.overflow_check(OverflowOp::SAdd, pa, pb);

    // Branch on overflow
    b.br_if(overflowed, slow_bb, &[pa, pb], fast_bb, &[sum]);

    // Fast: no overflow, rebox as fixnum
    b.switch_to_block(fast_bb);
    let result = b.block_param(fast_bb, 0);
    let tagged = b.make_tagged(1, result);
    b.ret(tagged);

    // Slow: overflow, call bignum add
    b.switch_to_block(slow_bb);
    let sa = b.block_param(slow_bb, 0);
    let sb = b.block_param(slow_bb, 1);
    let bignum_add = b.declare_func("bignum_add", Signature {
        params: vec![Type::I64, Type::I64],
        ret: Some(Type::I64),
    });
    let result = b.call(bignum_add, &[sa, sb]).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();

    let s = func.to_string();
    assert!(s.contains("overflow_check.sadd"));
}

// ════════════════════════════════════════════════════════════════
// Pattern 20: Invoke with exception handling
//
// Call a function that may throw; normal path continues,
// exception path handles the error. No need for runtime
// setjmp/longjmp protocol — the IR expresses it directly.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_invoke_exception() {
    let mut b = FunctionBuilder::new("try_call", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    let might_throw = b.declare_func("might_throw", Signature {
        params: vec![Type::I64],
        ret: Some(Type::I64),
    });

    let ok_bb = b.create_block(&[Type::I64]); // receives return value
    let err_bb = b.create_block(&[]);

    b.invoke(might_throw, &[x], ok_bb, &[], err_bb, &[]);

    // Normal path: return the result
    b.switch_to_block(ok_bb);
    let result = b.block_param(ok_bb, 0);
    b.ret(result);

    // Exception path: return nil
    b.switch_to_block(err_bb);
    let zero = b.iconst(Type::I64, 0);
    let nil = b.make_tagged(0, zero);
    b.ret(nil);

    let func = b.build();
    verify(&func).unwrap();

    let s = func.to_string();
    assert!(s.contains("invoke @f0"));
    // Much cleaner than the old pattern_exception_via_runtime!
}

#[test]
fn pattern_invoke_indirect_exception() {
    let mut b = FunctionBuilder::new("try_call_indirect", &[Type::Ptr, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let callee = b.block_param(entry, 0);
    let arg = b.block_param(entry, 1);

    let ok_bb = b.create_block(&[Type::I64]);
    let err_bb = b.create_block(&[]);

    b.invoke_indirect(callee, &[arg], Some(Type::I64), ok_bb, &[], err_bb, &[]);

    b.switch_to_block(ok_bb);
    let result = b.block_param(ok_bb, 0);
    b.ret(result);

    b.switch_to_block(err_bb);
    let zero = b.iconst(Type::I64, 0);
    let nil = b.make_tagged(0, zero);
    b.ret(nil);

    let func = b.build();
    verify(&func).unwrap();
    assert!(func.to_string().contains("invoke_indirect"));
}

// ════════════════════════════════════════════════════════════════
// Pattern 21: Guard with deoptimization metadata
//
// First-class guard: if the condition is false, deoptimize
// with captured live values. No explicit deopt blocks needed.
// ════════════════════════════════════════════════════════════════

#[test]
fn pattern_guard_deopt() {
    let mut b = FunctionBuilder::new("optimized_add", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    // Create deopt point with bytecode offset
    let deopt = b.create_deopt(42, "add @ bytecode offset 42");

    // Guard: both must be ints (deopt if not)
    let a_is_int = b.is_tag(a, 1);
    b.guard(a_is_int, deopt, &[a, bv]);

    let b_is_int = b.is_tag(bv, 1);
    b.guard(b_is_int, deopt, &[a, bv]);

    // Fast path: both are ints, no explicit deopt block needed
    let pa = b.payload(a);
    let pb = b.payload(bv);
    let sum = b.add(pa, pb);
    let result = b.make_tagged(1, sum);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();

    let s = func.to_string();
    assert!(s.contains("guard"));
    assert!(s.contains("deopt#0"));

    // Check deopt metadata is stored
    assert_eq!(func.deopt_info.len(), 1);
    assert_eq!(func.deopt_info[0].resume_point, 42);

    // Only 1 block! Compare to pattern_deopt_guard which needs 3 blocks.
    assert_eq!(func.blocks.len(), 1);
}

// ════════════════════════════════════════════════════════════════
// All expressiveness gaps are now closed:
//   ✓ Overflow checking (OverflowCheck instruction)
//   ✓ Invoke / exception edges (Invoke, InvokeIndirect terminators)
//   ✓ Guard / deopt metadata (Guard instruction + DeoptInfo)
// ════════════════════════════════════════════════════════════════
