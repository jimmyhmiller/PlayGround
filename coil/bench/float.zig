// Harmonic sum — double divide+add throughput. Matches float.c / float.coil.
const c = @cImport(@cInclude("stdio.h"));
export fn main() c_int {
    var s: f64 = 0.0;
    const n: i64 = 200000000;
    var i: i64 = 1;
    while (i <= n) : (i += 1) s += 1.0 / @as(f64, @floatFromInt(i));
    _ = c.printf("%.12f\n", s);
    return 0;
}
