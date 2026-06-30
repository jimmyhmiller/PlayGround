// Real Zig slice indexed on the hot path. Matches slicesum.c / .coil.
const c = @cImport(@cInclude("stdio.h"));
export fn main() c_int {
    var a: [1024]i64 = undefined;
    for (0..1024) |i| a[i] = @as(i64, @intCast(i)) *% 2654435761;
    const s: []i64 = a[0..];
    var acc: i64 = 0;
    var i: usize = 0;
    while (i < 200000000) : (i += 1) acc +%= s[i & 1023];
    _ = c.printf("%ld\n", acc);
    return 0;
}
