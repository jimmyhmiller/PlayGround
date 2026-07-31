// Struct-by-value on the hot path. Matches structcall.c / .coil.
const c = @cImport(@cInclude("stdio.h"));
const V3 = struct { x: i64, y: i64, z: i64 };
fn dot(a: V3, b: V3) i64 { return a.x *% b.x +% a.y *% b.y +% a.z *% b.z; }
export fn main() c_int {
    var v = V3{ .x = 0, .y = 2, .z = 3 };
    var acc: i64 = 0;
    var i: i64 = 0;
    while (i < 100000000) : (i += 1) { v.x = i; acc +%= dot(v, v); }
    _ = c.printf("%ld\n", acc);
    return 0;
}
