//! Regression tests for GC bugs.

#[cfg(test)]
mod tests {
    use dynobj::*;
    use dynvalue::{NanBox, Value};

    use crate::{PtrPolicy, SemiSpace, alloc_obj};

    struct NanBoxTag0;

    impl PtrPolicy for NanBoxTag0 {
        fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
            let v = Value::<NanBox>::from_bits(bits);
            if v.has_tag(0) && bits != Value::<NanBox>::tagged(0, 0).to_bits() {
                Some(v.payload() as *mut u8)
            } else {
                None
            }
        }

        fn encode_ptr(ptr: *mut u8) -> u64 {
            Value::<NanBox>::tagged(0, ptr as u64).to_bits()
        }
    }

    /// Regression test: the forwarding pointer scheme uses bit 0 of the u64
    /// word at `type_id_offset` to mark forwarded objects. But the Compact
    /// header stores a u16 type_id there. If the type_id is odd (1, 3, 5...),
    /// bit 0 is set, and `check_forwarded` misinterprets the unforwarded
    /// object as already forwarded — returning `(type_id & !1)` as the
    /// forwarding address (e.g., type_id=3 → forwarding addr = 2).
    ///
    /// In Lox, type_id 3 = Class. The GC rewrites a valid class pointer
    /// `0x7ffc0006040000d8` into `0x7ffc000000000002` (payload = 2), crashing.
    #[test]
    fn odd_type_id_not_misread_as_forwarding() {
        // Type table with 2 types. Type 1 has an odd type_id.
        static TYPES: [TypeInfo; 2] = [
            TypeInfo::for_header(Compact::SIZE).with_type_id(0), // type 0 (even)
            TypeInfo::for_header(Compact::SIZE)
                .with_fields(2)
                .with_type_id(1), // type 1 (odd)
        ];

        let mut gc = SemiSpace::new::<Compact>(4096);

        // Allocate an object with type_id 1 (odd)
        let obj = gc.alloc_obj::<Compact>(&TYPES[1], 0);
        assert!(!obj.is_null());

        // Verify the type_id is actually odd
        let tid = unsafe { read_type_id(obj, Compact::TYPE_ID_OFFSET) };
        assert_eq!(tid, 1, "type_id should be 1");

        // Write a child pointer into field 0
        let child = gc.alloc_obj::<Compact>(&TYPES[0], 0);
        assert!(!child.is_null());
        unsafe {
            write_value_field(
                obj,
                &TYPES[1],
                0,
                Value::<NanBox>::from_bits(NanBoxTag0::encode_ptr(child)),
            );
        }

        // Root the odd-type object
        use std::cell::Cell;
        struct SingleRoot(Cell<u64>);
        impl RootSource for SingleRoot {
            fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
                visitor(self.0.as_ptr());
            }
        }
        let root = SingleRoot(Cell::new(NanBoxTag0::encode_ptr(obj)));

        // Collect — with the bug, check_forwarded misreads type_id=1 as a
        // forwarding pointer to address 0, corrupting the root slot.
        unsafe { gc.collect::<NanBoxTag0>(&TYPES, &mut [&root]) };

        // The root should now point into to-space (the new from-space)
        let new_obj = NanBoxTag0::try_decode_ptr(root.0.get())
            .expect("root should still decode as a pointer after collection");
        assert!(
            gc.contains(new_obj as *const u8),
            "root should point into the new from-space, got {:?} (payload {:#x})",
            new_obj,
            new_obj as u64,
        );
    }
}
