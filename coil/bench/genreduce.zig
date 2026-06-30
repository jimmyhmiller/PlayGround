// Zig comptime-generic reduce on the hot path. Matches genreduce.c / .coil.
const c = @cImport(@cInclude("stdio.h"));
fn gsum(comptime T: type, a: [*]const T, n: usize) T {
    var acc: T = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) acc +%= a[i & 1023];
    return acc;
}
export fn main() c_int {
    var a: [1024]i64 = undefined;
    for (0..1024) |i| a[i] = @as(i64, @intCast(i)) *% 2654435761;
    _ = c.printf("%ld\n", gsum(i64, &a, 200000000));
    return 0;
}
