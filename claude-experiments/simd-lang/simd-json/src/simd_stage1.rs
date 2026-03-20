#[link(name = "json_stage1", kind = "static")]
extern "C" {
    #[link_name = "json_stage1"]
    fn __simd_json_stage1(
        input_alloc: *const u8, input_align: *const u8,
        input_offset: i64, input_size: i64, input_stride: i64,
        positions_alloc: *mut i32, positions_align: *mut i32,
        positions_offset: i64, positions_size: i64, positions_stride: i64,
    ) -> f32;

    #[link_name = "validate_utf8"]
    fn __simd_validate_utf8(
        input_alloc: *const u8, input_align: *const u8,
        input_offset: i64, input_size: i64, input_stride: i64,
    ) -> u128;
}

pub fn find_structural(input: &[u8], positions: &mut [i32]) -> usize {
    unsafe {
        f32::to_bits(__simd_json_stage1(
            input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
            positions.as_mut_ptr(), positions.as_mut_ptr(), 0, positions.len() as i64, 1,
        )) as usize
    }
}

pub fn validate_utf8(input: &[u8]) -> bool {
    unsafe {
        __simd_validate_utf8(
            input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
        ) == 0
    }
}
