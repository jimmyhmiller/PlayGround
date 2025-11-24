const std = @import("std");

/// Value types for the reader
pub const ValueType = enum(u8) {
    List,
    Vector,
    Map,
    Symbol,
    String,
    Number,
    Keyword,
    Boolean,
    Nil,
};

/// Namespace information for symbols
pub const Namespace = struct {
    name: []const u8,
    alias: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, alias: ?[]const u8) !*Namespace {
        const ns = try allocator.create(Namespace);
        ns.* = .{
            .name = try allocator.dupe(u8, name),
            .alias = if (alias) |a| try allocator.dupe(u8, a) else null,
        };
        return ns;
    }

    pub fn deinit(self: *Namespace, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        if (self.alias) |a| allocator.free(a);
        allocator.destroy(self);
    }
};

/// Symbol with namespace tracking
pub const Symbol = struct {
    name: []const u8,
    namespace: ?*Namespace,
    /// True if this symbol uses slash notation (alias/name)
    uses_alias: bool,
    /// True if this symbol uses dot notation (namespace.name)
    uses_dot: bool,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, namespace: ?*Namespace) !*Symbol {
        const sym = try allocator.create(Symbol);
        sym.* = .{
            .name = try allocator.dupe(u8, name),
            .namespace = namespace,
            .uses_alias = false,
            .uses_dot = false,
        };
        return sym;
    }

    pub fn deinit(self: *Symbol, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.destroy(self);
    }

    /// Get the full qualified name (namespace.name or alias/name)
    pub fn getQualifiedName(self: *const Symbol, allocator: std.mem.Allocator) ![]const u8 {
        if (self.namespace) |ns| {
            if (self.uses_alias) {
                if (ns.alias) |alias| {
                    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ alias, self.name });
                }
            }
            if (self.uses_dot) {
                return std.fmt.allocPrint(allocator, "{s}.{s}", .{ ns.name, self.name });
            }
            // Default to dot notation
            return std.fmt.allocPrint(allocator, "{s}.{s}", .{ ns.name, self.name });
        }
        return allocator.dupe(u8, self.name);
    }
};

/// Main value type
pub const Value = struct {
    type: ValueType,
    data: union {
        list: std.ArrayList(*Value),
        vector: std.ArrayList(*Value),
        map: std.StringHashMap(*Value),
        symbol: *Symbol,
        string: [:0]const u8,
        number: f64,
        keyword: []const u8,
        boolean: bool,
        nil: void,
    },
    allocator: std.mem.Allocator,

    pub fn createList(allocator: std.mem.Allocator) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .List,
            .data = .{ .list = std.ArrayList(*Value){} },
            .allocator = allocator,
        };
        return val;
    }

    pub fn createVector(allocator: std.mem.Allocator) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .Vector,
            .data = .{ .vector = std.ArrayList(*Value){} },
            .allocator = allocator,
        };
        return val;
    }

    pub fn createMap(allocator: std.mem.Allocator) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .Map,
            .data = .{ .map = std.StringHashMap(*Value).init(allocator) },
            .allocator = allocator,
        };
        return val;
    }

    pub fn createSymbol(allocator: std.mem.Allocator, symbol: *Symbol) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .Symbol,
            .data = .{ .symbol = symbol },
            .allocator = allocator,
        };
        return val;
    }

    pub fn createString(allocator: std.mem.Allocator, str: []const u8) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .String,
            .data = .{ .string = try allocator.dupeZ(u8, str) },
            .allocator = allocator,
        };
        return val;
    }

    pub fn createNumber(allocator: std.mem.Allocator, num: f64) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .Number,
            .data = .{ .number = num },
            .allocator = allocator,
        };
        return val;
    }

    pub fn createKeyword(allocator: std.mem.Allocator, kw: []const u8) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .Keyword,
            .data = .{ .keyword = try allocator.dupe(u8, kw) },
            .allocator = allocator,
        };
        return val;
    }

    pub fn createBoolean(allocator: std.mem.Allocator, b: bool) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .Boolean,
            .data = .{ .boolean = b },
            .allocator = allocator,
        };
        return val;
    }

    pub fn createNil(allocator: std.mem.Allocator) !*Value {
        const val = try allocator.create(Value);
        val.* = .{
            .type = .Nil,
            .data = .{ .nil = {} },
            .allocator = allocator,
        };
        return val;
    }

    pub fn deinit(self: *Value) void {
        switch (self.type) {
            .List => {
                for (self.data.list.items) |item| {
                    item.deinit();
                }
                self.data.list.deinit(self.allocator);
            },
            .Vector => {
                for (self.data.vector.items) |item| {
                    item.deinit();
                }
                self.data.vector.deinit(self.allocator);
            },
            .Map => {
                var it = self.data.map.iterator();
                while (it.next()) |entry| {
                    self.allocator.free(entry.key_ptr.*);
                    entry.value_ptr.*.deinit();
                }
                self.data.map.deinit();
            },
            .Symbol => {
                self.data.symbol.deinit(self.allocator);
            },
            .String => {
                self.allocator.free(self.data.string);
            },
            .Keyword => {
                self.allocator.free(self.data.keyword);
            },
            .Number, .Boolean, .Nil => {},
        }
        self.allocator.destroy(self);
    }

    /// Add item to list
    pub fn listAppend(self: *Value, item: *Value) !void {
        std.debug.assert(self.type == .List);
        try self.data.list.append(self.allocator, item);
    }

    /// Add item to vector
    pub fn vectorAppend(self: *Value, item: *Value) !void {
        std.debug.assert(self.type == .Vector);
        try self.data.vector.append(self.allocator, item);
    }

    /// Add key-value to map
    pub fn mapPut(self: *Value, key: []const u8, value: *Value) !void {
        std.debug.assert(self.type == .Map);
        const owned_key = try self.allocator.dupe(u8, key);
        try self.data.map.put(owned_key, value);
    }
};

// C API exports
export fn lispier_value_create_list() ?*Value {
    const allocator = std.heap.c_allocator;
    return Value.createList(allocator) catch null;
}

export fn lispier_value_create_vector() ?*Value {
    const allocator = std.heap.c_allocator;
    return Value.createVector(allocator) catch null;
}

export fn lispier_value_create_map() ?*Value {
    const allocator = std.heap.c_allocator;
    return Value.createMap(allocator) catch null;
}

export fn lispier_value_create_string(str: [*:0]const u8) ?*Value {
    const allocator = std.heap.c_allocator;
    const slice = std.mem.span(str);
    return Value.createString(allocator, slice) catch null;
}

export fn lispier_value_create_number(num: f64) ?*Value {
    const allocator = std.heap.c_allocator;
    return Value.createNumber(allocator, num) catch null;
}

export fn lispier_value_create_keyword(kw: [*:0]const u8) ?*Value {
    const allocator = std.heap.c_allocator;
    const slice = std.mem.span(kw);
    return Value.createKeyword(allocator, slice) catch null;
}

export fn lispier_value_create_boolean(b: bool) ?*Value {
    const allocator = std.heap.c_allocator;
    return Value.createBoolean(allocator, b) catch null;
}

export fn lispier_value_create_nil() ?*Value {
    const allocator = std.heap.c_allocator;
    return Value.createNil(allocator) catch null;
}

export fn lispier_value_destroy(val: ?*Value) void {
    if (val) |v| {
        v.deinit();
    }
}

export fn lispier_value_get_type(val: ?*Value) ValueType {
    if (val) |v| {
        return v.type;
    }
    return .Nil;
}

export fn lispier_value_list_append(list: ?*Value, item: ?*Value) bool {
    if (list == null or item == null) return false;
    list.?.listAppend(item.?) catch return false;
    return true;
}

export fn lispier_value_vector_append(vector: ?*Value, item: ?*Value) bool {
    if (vector == null or item == null) return false;
    vector.?.vectorAppend(item.?) catch return false;
    return true;
}

export fn lispier_value_map_put(map: ?*Value, key: [*:0]const u8, value: ?*Value) bool {
    if (map == null or value == null) return false;
    const slice = std.mem.span(key);
    map.?.mapPut(slice, value.?) catch return false;
    return true;
}

export fn lispier_value_get_string(val: ?*Value) [*:0]const u8 {
    if (val) |v| {
        if (v.type == .String) {
            return v.data.string.ptr;
        }
    }
    return "";
}

export fn lispier_value_get_number(val: ?*Value) f64 {
    if (val) |v| {
        if (v.type == .Number) {
            return v.data.number;
        }
    }
    return 0.0;
}

export fn lispier_value_get_boolean(val: ?*Value) bool {
    if (val) |v| {
        if (v.type == .Boolean) {
            return v.data.boolean;
        }
    }
    return false;
}
