// Tight integer multiply-accumulate loop. Matches loop.c / loop.coil.
const c = @cImport(@cInclude("stdio.h"));
export fn main() c_int {
    var acc: u64 = 0;
    const n: u64 = 300000000;
    var i: u64 = 0;
    while (i < n) : (i += 1) acc = acc *% 1000003 +% i;
    _ = c.printf("%lu\n", acc);
    return 0;
}
