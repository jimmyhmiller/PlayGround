const std = @import("std");
const Value = @import("value.zig").Value;
const RequireDecl = @import("value.zig").RequireDecl;
const Type = @import("type_checker.zig").Type;
const TypeEnv = @import("type_checker.zig").TypeEnv;

/// Represents a compiled namespace with all its metadata
pub const CompiledNamespace = struct {
    name: []const u8,
    file_path: []const u8,
    bundle_path: []const u8,
    definitions: TypeEnv, // def name -> type
    definition_order: std.ArrayList([]const u8), // maintains definition order for code generation
    type_defs: TypeEnv,   // struct/enum types
    requires: std.ArrayList(RequireDecl), // dependencies
    bundle_handle: ?*std.DynLib = null,   // Loaded dynamic library handle
    is_loaded: bool = false,              // Has init function been called?
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !CompiledNamespace {
        return CompiledNamespace{
            .name = try allocator.dupe(u8, name),
            .file_path = &[_]u8{},
            .bundle_path = &[_]u8{},
            .definitions = TypeEnv.init(allocator),
            .definition_order = std.ArrayList([]const u8){},
            .type_defs = TypeEnv.init(allocator),
            .requires = std.ArrayList(RequireDecl){},
            .bundle_handle = null,
            .is_loaded = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CompiledNamespace) void {
        self.allocator.free(self.name);
        if (self.file_path.len > 0) self.allocator.free(self.file_path);
        if (self.bundle_path.len > 0) self.allocator.free(self.bundle_path);
        // NOTE: Don't deinit definitions and type_defs - they contain references to
        // data owned by the type checker's arena allocator. Deiniting them causes crashes.
        // This is a small memory leak but acceptable for now.
        // TODO: Properly copy string keys when caching to own the data.
        // self.definitions.deinit();
        // self.type_defs.deinit();

        // Note: requires contain owned strings, need to free them
        for (self.requires.items) |req| {
            self.allocator.free(req.namespace);
            self.allocator.free(req.alias);
        }
        self.requires.deinit(self.allocator);
        for (self.definition_order.items) |name| {
            self.allocator.free(name);
        }
        self.definition_order.deinit(self.allocator);
        // Close dynamic library if it was loaded
        if (self.bundle_handle) |handle| {
            handle.close();
            self.allocator.destroy(handle);
        }
    }
};

/// Manages namespace resolution, compilation, and caching
pub const NamespaceManager = struct {
    compiled_namespaces: std.StringHashMap(CompiledNamespace),
    compilation_stack: std.ArrayList([]const u8), // Track recursive compilation for cycle detection
    search_paths: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) NamespaceManager {
        const manager = NamespaceManager{
            .compiled_namespaces = std.StringHashMap(CompiledNamespace).init(allocator),
            .compilation_stack = std.ArrayList([]const u8){},
            .search_paths = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
        return manager;
    }

    pub fn deinit(self: *NamespaceManager) void {
        var iter = self.compiled_namespaces.valueIterator();
        while (iter.next()) |ns| {
            // Make a copy so we can call deinit
            var ns_copy = ns.*;
            ns_copy.deinit();
        }
        self.compiled_namespaces.deinit();
        self.compilation_stack.deinit(self.allocator);
        for (self.search_paths.items) |path| {
            self.allocator.free(path);
        }
        self.search_paths.deinit(self.allocator);
    }

    pub fn addSearchPath(self: *NamespaceManager, path: []const u8) !void {
        const owned = try self.allocator.dupe(u8, path);
        try self.search_paths.append(self.allocator, owned);
    }

    /// Resolve namespace name to file path
    /// e.g., "my.namespace" -> "my/namespace.lisp" or "src/my/namespace.lisp"
    pub fn resolveNamespaceFile(self: *NamespaceManager, ns_name: []const u8) !?[]const u8 {
        // Convert namespace name to relative path
        // "my.namespace" -> "my/namespace.lisp"
        var path_buf = std.ArrayList(u8){};
        defer path_buf.deinit(self.allocator);

        for (ns_name) |c| {
            if (c == '.') {
                try path_buf.append(self.allocator, '/');
            } else {
                try path_buf.append(self.allocator, c);
            }
        }
        try path_buf.appendSlice(self.allocator, ".lisp");

        const relative_path = try path_buf.toOwnedSlice(self.allocator);
        defer self.allocator.free(relative_path);

        // Try search paths
        for (self.search_paths.items) |search_path| {
            const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ search_path, relative_path });
            defer self.allocator.free(full_path);

            // Check if file exists
            if (std.fs.cwd().access(full_path, .{})) {
                // File exists, return owned copy
                return try self.allocator.dupe(u8, full_path);
            } else |_| {
                // File doesn't exist, try next search path
                continue;
            }
        }

        // Not found in search paths, try current directory
        if (std.fs.cwd().access(relative_path, .{})) {
            return try self.allocator.dupe(u8, relative_path);
        } else |_| {}

        return null;
    }

    /// Check if namespace is already compiled
    pub fn isCompiled(self: *NamespaceManager, ns_name: []const u8) bool {
        return self.compiled_namespaces.contains(ns_name);
    }

    /// Get compiled namespace info (must check isCompiled first)
    pub fn getCompiledNamespace(self: *NamespaceManager, ns_name: []const u8) ?*CompiledNamespace {
        return self.compiled_namespaces.getPtr(ns_name);
    }

    /// Add a compiled namespace to the cache
    pub fn addCompiledNamespace(self: *NamespaceManager, ns: CompiledNamespace) !void {
        try self.compiled_namespaces.put(ns.name, ns);
    }

    /// Check if we're currently compiling a namespace (for cycle detection)
    pub fn isInCompilationStack(self: *NamespaceManager, ns_name: []const u8) bool {
        for (self.compilation_stack.items) |name| {
            if (std.mem.eql(u8, name, ns_name)) {
                return true;
            }
        }
        return false;
    }

    /// Push namespace onto compilation stack
    pub fn pushCompilationStack(self: *NamespaceManager, ns_name: []const u8) !void {
        const owned = try self.allocator.dupe(u8, ns_name);
        try self.compilation_stack.append(self.allocator, owned);
    }

    /// Pop namespace from compilation stack
    pub fn popCompilationStack(self: *NamespaceManager) void {
        if (self.compilation_stack.items.len > 0) {
            const item = self.compilation_stack.items[self.compilation_stack.items.len - 1];
            _ = self.compilation_stack.pop();
            self.allocator.free(item);
        }
    }

    /// Get the current compilation stack as a string (for error messages)
    pub fn getCompilationStackTrace(self: *NamespaceManager) ![]const u8 {
        var buf = std.ArrayList(u8){};
        errdefer buf.deinit(self.allocator);

        try buf.appendSlice(self.allocator, "Compilation stack:\n");
        for (self.compilation_stack.items, 0..) |name, i| {
            try buf.writer(self.allocator).print("  {d}. {s}\n", .{ i + 1, name });
        }

        return try buf.toOwnedSlice(self.allocator);
    }
};

test "namespace manager initialization" {
    var manager = NamespaceManager.init(std.testing.allocator);
    defer manager.deinit();

    try std.testing.expect(manager.compiled_namespaces.count() == 0);
    try std.testing.expect(manager.compilation_stack.items.len == 0);
}

test "namespace manager search paths" {
    var manager = NamespaceManager.init(std.testing.allocator);
    defer manager.deinit();

    try manager.addSearchPath("src");
    try manager.addSearchPath("lib");

    try std.testing.expect(manager.search_paths.items.len == 2);
    try std.testing.expect(std.mem.eql(u8, manager.search_paths.items[0], "src"));
    try std.testing.expect(std.mem.eql(u8, manager.search_paths.items[1], "lib"));
}

test "namespace name to file path conversion" {
    var manager = NamespaceManager.init(std.testing.allocator);
    defer manager.deinit();

    // Note: This test won't find an actual file, but tests the path construction
    const result = try manager.resolveNamespaceFile("my.namespace");
    try std.testing.expect(result == null); // No file exists
}

test "compilation stack tracking" {
    var manager = NamespaceManager.init(std.testing.allocator);
    defer manager.deinit();

    try std.testing.expect(!manager.isInCompilationStack("foo.bar"));

    try manager.pushCompilationStack("foo.bar");
    try std.testing.expect(manager.isInCompilationStack("foo.bar"));

    try manager.pushCompilationStack("baz.qux");
    try std.testing.expect(manager.isInCompilationStack("foo.bar"));
    try std.testing.expect(manager.isInCompilationStack("baz.qux"));

    manager.popCompilationStack();
    try std.testing.expect(manager.isInCompilationStack("foo.bar"));
    try std.testing.expect(!manager.isInCompilationStack("baz.qux"));

    manager.popCompilationStack();
    try std.testing.expect(!manager.isInCompilationStack("foo.bar"));
}

test "compiled namespace storage" {
    var manager = NamespaceManager.init(std.testing.allocator);
    defer manager.deinit();

    const ns = try CompiledNamespace.init(std.testing.allocator, "test.namespace");

    try std.testing.expect(!manager.isCompiled("test.namespace"));

    try manager.addCompiledNamespace(ns);

    try std.testing.expect(manager.isCompiled("test.namespace"));

    const retrieved = manager.getCompiledNamespace("test.namespace");
    try std.testing.expect(retrieved != null);
    try std.testing.expect(std.mem.eql(u8, retrieved.?.name, "test.namespace"));
}
