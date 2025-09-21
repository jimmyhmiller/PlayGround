const std = @import("std");
const testing = std.testing;

const StructWithAllocator = struct {
    allocator: std.mem.Allocator,
    data: []u8,

    fn init(allocator: std.mem.Allocator) !StructWithAllocator {
        return .{
            .allocator = allocator,
            .data = try allocator.alloc(u8, 100),
        };
    }

    fn deinit(self: *StructWithAllocator) void {
        self.allocator.free(self.data);
    }
};

const StructWithArena = struct {
    arena: std.heap.ArenaAllocator,
    data: []u8,

    fn init(backing_allocator: std.mem.Allocator) !StructWithArena {
        var arena = std.heap.ArenaAllocator.init(backing_allocator);
        const allocator = arena.allocator();
        return .{
            .arena = arena,
            .data = try allocator.alloc(u8, 100),
        };
    }

    fn deinit(self: *StructWithArena) void {
        self.arena.deinit();
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Testing Allocator Assignment ===\n\n", .{});

    // Test 1: Passing Allocator interface (safe)
    std.debug.print("Test 1: Passing Allocator interface\n", .{});
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        var obj = try StructWithAllocator.init(arena.allocator());
        defer obj.deinit();

        @memset(obj.data, 'A');
        std.debug.print("  Allocated and filled buffer with 'A'\n", .{});
        std.debug.print("  First 10 bytes: {s}\n", .{obj.data[0..10]});
    }
    std.debug.print("  ✓ Successfully freed through Allocator interface\n\n", .{});

    // Test 2: Copying ArenaAllocator struct (problematic!)
    std.debug.print("Test 2: Copying ArenaAllocator struct (problematic)\n", .{});
    {
        var obj = try StructWithArena.init(allocator);
        defer obj.deinit();

        @memset(obj.data, 'B');
        std.debug.print("  Allocated and filled buffer with 'B'\n", .{});
        std.debug.print("  First 10 bytes: {s}\n", .{obj.data[0..10]});

        // This demonstrates the problem: obj.arena is a COPY of the original arena
        // The allocation was made on the original, but we're trying to free from the copy
    }
    std.debug.print("  ⚠️  ArenaAllocator was copied - may cause issues\n\n", .{});

    // Test 3: Proper pattern - store pointer to ArenaAllocator
    std.debug.print("Test 3: Storing pointer to ArenaAllocator (correct)\n", .{});
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        const CorrectStruct = struct {
            arena: *std.heap.ArenaAllocator,
            data: []u8,
        };

        var obj = CorrectStruct{
            .arena = &arena,
            .data = try arena.allocator().alloc(u8, 100),
        };

        @memset(obj.data, 'C');
        std.debug.print("  Allocated and filled buffer with 'C'\n", .{});
        std.debug.print("  First 10 bytes: {s}\n", .{obj.data[0..10]});
    }
    std.debug.print("  ✓ Stored pointer to ArenaAllocator - safe pattern\n\n", .{});

    // Test 4: Demonstrate the actual problem with copies
    std.debug.print("Test 4: Demonstrating allocation mismatch\n", .{});
    {
        var original_arena = std.heap.ArenaAllocator.init(allocator);
        defer original_arena.deinit();

        // Make a copy (this is what happens when assigning to struct field)
        var copied_arena = original_arena;
        defer copied_arena.deinit();

        // Allocate from original
        const data1 = try original_arena.allocator().alloc(u8, 50);
        @memset(data1, 'X');

        // Allocate from copy
        const data2 = try copied_arena.allocator().alloc(u8, 50);
        @memset(data2, 'Y');

        std.debug.print("  Allocated from original: {s}\n", .{data1[0..5]});
        std.debug.print("  Allocated from copy: {s}\n", .{data2[0..5]});
        std.debug.print("  ⚠️  Both arenas will try to free the same memory!\n", .{});
    }

    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("• Storing std.mem.Allocator interface: SAFE ✓\n", .{});
    std.debug.print("• Copying ArenaAllocator struct: DANGEROUS ✗\n", .{});
    std.debug.print("• Storing *ArenaAllocator pointer: SAFE ✓\n", .{});
    std.debug.print("\nThe article's concern is valid for ArenaAllocator structs,\n", .{});
    std.debug.print("but passing the Allocator interface is perfectly safe!\n", .{});
}

test "allocator interface vs arena copy" {
    const allocator = testing.allocator;

    // Safe: Using Allocator interface
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        var obj = try StructWithAllocator.init(arena.allocator());
        defer obj.deinit();

        try testing.expect(obj.data.len == 100);
    }

    // Problematic: Copying ArenaAllocator
    {
        var obj = try StructWithArena.init(allocator);
        defer obj.deinit();

        try testing.expect(obj.data.len == 100);
    }
}
