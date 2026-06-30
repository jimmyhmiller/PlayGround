// Takeuchi — deeply nested non-tail recursion. Matches tak.c / tak.coil.
const c = @cImport(@cInclude("stdio.h"));
fn tak(x: i64, y: i64, z: i64) i64 {
    if (!(y < x)) return z;
    return tak(tak(x - 1, y, z), tak(y - 1, z, x), tak(z - 1, x, y));
}
export fn main() c_int {
    _ = c.printf("%ld\n", tak(24, 16, 8));
    return 0;
}
