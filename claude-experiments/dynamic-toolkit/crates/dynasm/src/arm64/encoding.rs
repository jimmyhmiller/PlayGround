/// Truncate an immediate value to WIDTH bits, asserting no information is lost.
pub fn truncate_imm<T: Into<i32>, const WIDTH: usize>(imm: T) -> u32 {
    let value: i32 = imm.into();
    let masked = (value as u32) & ((1 << WIDTH) - 1);

    if value >= 0 {
        assert_eq!(value as u32, masked, "positive immediate {value} doesn't fit in {WIDTH} bits");
    } else {
        assert_eq!(
            value as u32,
            masked | (u32::MAX << WIDTH),
            "negative immediate {value} doesn't fit in {WIDTH} bits"
        );
    }

    masked
}
