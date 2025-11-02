const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Reader = mlir_lisp.Reader;
const Value = mlir_lisp.Value;
const Tokenizer = mlir_lisp.Tokenizer;

// Import the C API functions we need
const c_api = mlir_lisp.c_api;
const c_api_transform = mlir_lisp.c_api_transform;

/// Walk the tree and expand any list that starts with "call"
fn walkAndExpandCalls(allocator: std.mem.Allocator, value: *Value) !*Value {
    switch (value.type) {
        .list => {
            const list = value.data.list.slice();
            if (list.len > 0) {
                // Check if first element is "call"
                const first = list[0];
                if (first.type == .identifier) {
                    const atom = first.data.atom;
                    if (std.mem.eql(u8, atom, "call")) {
                        // This is a call form - we need to expand it using the C API
                        std.debug.print("Found call form, transforming...\n", .{});

                        // Create a C allocator
                        const c_allocator = c_api.allocator_create_c() orelse return error.AllocationFailed;
                        defer c_api.allocator_destroy(c_allocator);

                        // Convert the Value pointer to opaque pointer for C API
                        const call_expr_opaque: ?*anyopaque = @ptrCast(value);

                        // Call the transformation (returns optional, not error union)
                        const operation_opaque = c_api_transform.transformCallToOperation(c_allocator, call_expr_opaque);

                        // Convert back to Value pointer
                        if (operation_opaque) |op| {
                            const operation: *Value = @ptrCast(@alignCast(op));

                            // Print the transformed operation
                            var buf = std.ArrayList(u8){};
                            defer buf.deinit(allocator);
                            try operation.print(buf.writer(allocator));
                            std.debug.print("Transformed to: {s}\n", .{buf.items});

                            return operation;
                        }

                        return value;
                    }
                }
            }

            // Not a call form, recursively walk children
            // Note: We're not modifying the tree here, just walking it
            for (list) |child| {
                _ = try walkAndExpandCalls(allocator, child);
            }

            return value;
        },
        .vector => {
            const vec = value.data.vector.slice();
            for (vec) |child| {
                _ = try walkAndExpandCalls(allocator, child);
            }
            return value;
        },
        .map => {
            const map_vec = value.data.map.slice();
            for (map_vec) |child| {
                _ = try walkAndExpandCalls(allocator, child);
            }
            return value;
        },
        else => {
            // Atoms don't need walking
            return value;
        },
    }
}

test "walk tree and find call forms" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source =
        \\(module
        \\  (operation (name func.func)
        \\    (regions [
        \\      (region (blocks [
        \\        (block (operations [
        \\          (call @test i64)
        \\          (call @foo i32)
        \\        ]))
        \\      ]))
        \\    ]))
        \\)
    ;

    // Parse the source
    var tok = Tokenizer.init(allocator, source);
    var rdr = try Reader.init(allocator, &tok);
    const value = try rdr.read();

    // Walk the tree and find call forms
    _ = try walkAndExpandCalls(allocator, value);
}
