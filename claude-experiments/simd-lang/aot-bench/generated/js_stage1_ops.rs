// Auto-generated bindings for js_stage1_ops.simd
// Do not edit — regenerate with: simd-lang compile

#[link(name = "js_stage1_ops", kind = "static")]
extern "C" {
    #[link_name = "js_stage1_ops"]
    fn __simd_js_stage1_ops(input_alloc: *mut u8, input_align: *mut u8, input_offset: i64, input_size: i64, input_stride: i64, start_masks_alloc: *mut u64, start_masks_align: *mut u64, start_masks_offset: i64, start_masks_size: i64, start_masks_stride: i64, word_masks_alloc: *mut u64, word_masks_align: *mut u64, word_masks_offset: i64, word_masks_size: i64, word_masks_stride: i64, op_ext_masks_alloc: *mut u64, op_ext_masks_align: *mut u64, op_ext_masks_offset: i64, op_ext_masks_size: i64, op_ext_masks_stride: i64) -> f32;
}

/// `js_stage1_ops` from js_stage1_ops.simd
pub fn js_stage1_ops(input: &mut [u8], start_masks: &mut [u64], word_masks: &mut [u64], op_ext_masks: &mut [u64]) -> i32 {
    unsafe {
        f32::to_bits(__simd_js_stage1_ops(input.as_mut_ptr(), input.as_mut_ptr(), 0, input.len() as i64, 1, start_masks.as_mut_ptr(), start_masks.as_mut_ptr(), 0, start_masks.len() as i64, 1, word_masks.as_mut_ptr(), word_masks.as_mut_ptr(), 0, word_masks.len() as i64, 1, op_ext_masks.as_mut_ptr(), op_ext_masks.as_mut_ptr(), 0, op_ext_masks.len() as i64, 1)) as i32
    }
}

