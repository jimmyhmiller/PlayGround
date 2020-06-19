const std = @import("std");

const Node = struct {
    val: usize,
    parent: ?*Node,
    child: ?*Node,
    left: ?*Node,
    right: ?*Node,
};

// I need to test out making a big tree and doing operations over it.
// Ultimately I want to see if I can recreate what I've made in rust
// and see how they compare.

pub fn main() !void {
    var root = Node{
        .val = 0,
        .parent = null,
        .child = null,
        .left = null,
        .right = null,
    };
    // Need to understand pointers better for this part.
    var parent: *Node = &root;
    var i: usize = 0;
    while (i < 20) {
        i += 1;
        var child = Node{
            .val = i,
            .parent = parent,
            .child = null,
            .left = null,
            .right = null,
        };
        parent.child = &child;
        parent = &child;
    }
    const stdout = std.io.getStdOut().outStream();
    try stdout.print("{}\n", .{root});
}
