// Fill + sum a 20M-element heap array. Matches memory.c / memory.coil.
const c = @cImport(@cInclude("stdio.h"));
const std = @import("std");
export fn main() c_int {
    const n: usize = 20000000;
    const a = std.heap.c_allocator.alloc(u64, n) catch return 1;
    for (0..n) |i| a[i] = @as(u64, i) *% 2654435761;
    var s: u64 = 0;
    for (0..n) |i| s +%= a[i];
    _ = c.printf("%lu\n", s);
    return 0;
}
