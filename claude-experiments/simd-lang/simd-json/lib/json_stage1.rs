//! Auto-generated bindings for json_stage1.simd
//! Do not edit — regenerate with: simd-lang compile

#[link(name = "json_stage1", kind = "static")]
extern "C" {
    #[link_name = "json_stage1"]
    fn __simd_json_stage1(input_alloc: *mut u8, input_align: *mut u8, input_offset: i64, input_size: i64, input_stride: i64, positions_alloc: *mut i32, positions_align: *mut i32, positions_offset: i64, positions_size: i64, positions_stride: i64) -> f32;
}

/// `json_stage1` from json_stage1.simd
pub fn json_stage1(input: &mut [u8], positions: &mut [i32]) -> i32 {
    unsafe {
        f32::to_bits(__simd_json_stage1(input.as_mut_ptr(), input.as_mut_ptr(), 0, input.len() as i64, 1, positions.as_mut_ptr(), positions.as_mut_ptr(), 0, positions.len() as i64, 1)) as i32
    }
}

