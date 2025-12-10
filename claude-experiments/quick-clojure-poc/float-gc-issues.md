• - GC correctness: float objects are allocated without marking the header opaque, so get_heap_references will scan the 64-bit payload and treat any “pointer-looking” bits as heap references. That can cause spurious marking or
    crashes when the GC follows random addresses. The string allocator sets header.opaque = true, but allocate_float does not (src/gc_runtime.rs:397-410 with traversal in src/gc/types.rs:325-332). Mark floats opaque (or otherwise
    skip scanning them) before shipping.
  - Type safety for arithmetic: the polymorphic arithmetic path only distinguishes “tag 0 int” vs “everything else”, and the “everything else” branch blindly LoadFloats the operand (src/compiler.rs:1666-1778). Passing a bool,
    string, or nil into +/-/*// will be routed down the float path and attempt to read float bits from a non-float pointer (untag -> small integer -> load from offset 8), leading to undefined memory reads/crashes instead of a type
    error.
  - Tests: good coverage of happy-path literals and mixed int/float ops, but there’s no coverage for negative floats, infinities/NaN, or mixed-type error handling (bool/string/nil in arithmetic). Adding those would catch the issues
    above and tighten float semantics (tests/test_floats.rs).