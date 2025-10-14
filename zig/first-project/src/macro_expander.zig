const std = @import("std");
const Value = @import("value.zig").Value;
const createSymbol = @import("value.zig").createSymbol;
const createString = @import("value.zig").createString;
const createMacro = @import("value.zig").createMacro;
const PersistentLinkedList = @import("collections/linked_list.zig").PersistentLinkedList;
const PersistentVector = @import("collections/vector.zig").PersistentVector;

pub const MacroExpandError = error{
    InvalidMacroDefinition,
    UnboundMacro,
    ArgumentCountMismatch,
    OutOfMemory,
    InvalidSyntax,
};

/// Environment for macro definitions
pub const MacroEnv = struct {
    macros: std.StringHashMap(*Value),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MacroEnv {
        return MacroEnv{
            .macros = std.StringHashMap(*Value).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MacroEnv) void {
        self.macros.deinit();
    }

    pub fn put(self: *MacroEnv, name: []const u8, macro_val: *Value) !void {
        try self.macros.put(name, macro_val);
    }

    pub fn get(self: *MacroEnv, name: []const u8) ?*Value {
        return self.macros.get(name);
    }

    pub fn contains(self: *MacroEnv, name: []const u8) bool {
        return self.macros.contains(name);
    }
};

/// Macro expander that walks the AST and expands macro calls
pub const MacroExpander = struct {
    env: MacroEnv,
    allocator: std.mem.Allocator,
    gensym_counter: u64,

    pub fn init(allocator: std.mem.Allocator) MacroExpander {
        return MacroExpander{
            .env = MacroEnv.init(allocator),
            .allocator = allocator,
            .gensym_counter = 0,
        };
    }

    pub fn deinit(self: *MacroExpander) void {
        self.env.deinit();
    }

    /// Generate a unique symbol with an optional prefix
    pub fn gensym(self: *MacroExpander, prefix: []const u8) !*Value {
        const count = self.gensym_counter;
        self.gensym_counter += 1;

        const sym_name = try std.fmt.allocPrint(
            self.allocator,
            "{s}__{d}",
            .{ prefix, count }
        );

        return try createSymbol(self.allocator, sym_name);
    }

    /// Main entry point: expand all macros in an expression
    pub fn expand(self: *MacroExpander, expr: *Value) MacroExpandError!*Value {
        // Handle different value types
        if (expr.isList()) {
            return try self.expandList(expr);
        } else if (expr.isVector()) {
            return try self.expandVector(expr);
        } else {
            // Atoms don't need expansion
            return expr;
        }
    }

    /// macroexpand: Perform one step of macro expansion
    /// If the expression is a macro call, expand it once (but don't recursively expand the result)
    /// Otherwise, return the expression unchanged
    pub fn expandOne(self: *MacroExpander, expr: *Value) MacroExpandError!*Value {
        if (!expr.isList()) {
            return expr;
        }

        const list = expr.list;
        if (list.isEmpty()) {
            return expr;
        }

        const first = list.value orelse return expr;

        // Check if this is a macro call
        if (first.isSymbol()) {
            if (self.env.get(first.symbol)) |macro_def| {
                // This is a macro call - expand it once, but don't recursively expand
                return try self.expandMacroCallOnce(macro_def, list.next);
            }
        }

        // Not a macro call - return unchanged
        return expr;
    }

    /// macroexpand-all: Recursively expand all macros (same as expand)
    /// This is provided as an explicit alias for clarity
    pub fn expandAll(self: *MacroExpander, expr: *Value) MacroExpandError!*Value {
        return try self.expand(expr);
    }

    /// Expand macros in a list
    fn expandList(self: *MacroExpander, expr: *Value) MacroExpandError!*Value {
        const list = expr.list;

        // Empty list
        if (list.isEmpty()) {
            return expr;
        }

        const first = list.value orelse return expr;

        // Check if this is a special form that should not be macro expanded
        if (first.isSymbol()) {
            // defmacro - register and return
            if (std.mem.eql(u8, first.symbol, "defmacro")) {
                return try self.handleDefmacro(list);
            }

            // ns and require - pass through unchanged
            if (std.mem.eql(u8, first.symbol, "ns") or std.mem.eql(u8, first.symbol, "require")) {
                return expr;
            }

            // Check if this is a macro call
            if (self.env.get(first.symbol)) |macro_def| {
                // This is a macro call - expand it
                return try self.expandMacroCall(macro_def, list.next);
            }
        }

        // Not a macro - recursively expand subexpressions
        return try self.expandListElements(list);
    }

    /// Handle (defmacro name [params] body)
    fn handleDefmacro(self: *MacroExpander, list: *const PersistentLinkedList(*Value)) MacroExpandError!*Value {
        // Skip 'defmacro'
        var current = list.next;
        if (current == null) return MacroExpandError.InvalidMacroDefinition;

        // Get macro name
        const name_node = current.?;
        if (name_node.value == null or !name_node.value.?.isSymbol()) {
            return MacroExpandError.InvalidMacroDefinition;
        }
        const macro_name = name_node.value.?.symbol;
        current = name_node.next;

        // Get parameter list
        if (current == null) return MacroExpandError.InvalidMacroDefinition;
        const params_node = current.?;
        if (params_node.value == null or !params_node.value.?.isVector()) {
            return MacroExpandError.InvalidMacroDefinition;
        }
        const params = params_node.value.?.vector;
        current = params_node.next;

        // Get body (rest of the list)
        if (current == null) return MacroExpandError.InvalidMacroDefinition;
        const body_node = current.?;
        if (body_node.value == null) return MacroExpandError.InvalidMacroDefinition;
        const body = body_node.value.?;

        // Create and register the macro
        const macro_val = try createMacro(self.allocator, macro_name, params, body);
        try self.env.put(macro_name, macro_val);

        // Return the macro definition itself (won't be executed)
        return macro_val;
    }

    /// Expand a macro call by substituting parameters with arguments (with recursive expansion)
    fn expandMacroCall(self: *MacroExpander, macro_def: *Value, args: ?*const PersistentLinkedList(*Value)) MacroExpandError!*Value {
        if (!macro_def.isMacro()) return MacroExpandError.InvalidMacroDefinition;

        const macro_data = macro_def.macro_def;
        const params = macro_data.params;
        const body = macro_data.body;

        // Collect arguments into a slice
        var arg_list = std.ArrayList(*Value){};
        defer arg_list.deinit(self.allocator);

        var current = args;
        while (current) |node| {
            if (node.value) |v| {
                try arg_list.append(self.allocator, v);
            }
            current = node.next;
        }

        // Check argument count
        if (arg_list.items.len != params.len()) {
            return MacroExpandError.ArgumentCountMismatch;
        }

        // Create substitution map
        var subst_map = std.StringHashMap(*Value).init(self.allocator);
        defer subst_map.deinit();

        const param_slice = params.slice();
        for (param_slice, 0..) |param, i| {
            if (!param.isSymbol()) return MacroExpandError.InvalidMacroDefinition;
            try subst_map.put(param.symbol, arg_list.items[i]);
        }

        // Substitute parameters in body
        const expanded_body = try self.substitute(body, &subst_map);

        // Recursively expand the result
        return try self.expand(expanded_body);
    }

    /// Expand a macro call once without recursive expansion (for macroexpand)
    fn expandMacroCallOnce(self: *MacroExpander, macro_def: *Value, args: ?*const PersistentLinkedList(*Value)) MacroExpandError!*Value {
        if (!macro_def.isMacro()) return MacroExpandError.InvalidMacroDefinition;

        const macro_data = macro_def.macro_def;
        const params = macro_data.params;
        const body = macro_data.body;

        // Collect arguments into a slice
        var arg_list = std.ArrayList(*Value){};
        defer arg_list.deinit(self.allocator);

        var current = args;
        while (current) |node| {
            if (node.value) |v| {
                try arg_list.append(self.allocator, v);
            }
            current = node.next;
        }

        // Check argument count
        if (arg_list.items.len != params.len()) {
            return MacroExpandError.ArgumentCountMismatch;
        }

        // Create substitution map
        var subst_map = std.StringHashMap(*Value).init(self.allocator);
        defer subst_map.deinit();

        const param_slice = params.slice();
        for (param_slice, 0..) |param, i| {
            if (!param.isSymbol()) return MacroExpandError.InvalidMacroDefinition;
            try subst_map.put(param.symbol, arg_list.items[i]);
        }

        // Substitute parameters in body - but don't recursively expand
        return try self.substitute(body, &subst_map);
    }

    /// Substitute symbols in an expression based on a map
    fn substitute(self: *MacroExpander, expr: *Value, subst_map: *std.StringHashMap(*Value)) MacroExpandError!*Value {
        if (expr.isSymbol()) {
            // Replace if in substitution map
            if (subst_map.get(expr.symbol)) |replacement| {
                return replacement;
            }
            return expr;
        } else if (expr.isList()) {
            const list = expr.list;

            // Check for special forms
            if (!list.isEmpty() and list.value != null and list.value.?.isSymbol()) {
                const first_sym = list.value.?.symbol;

                // Handle syntax-quote
                if (std.mem.eql(u8, first_sym, "syntax-quote")) {
                    return try self.expandSyntaxQuote(list.next, subst_map);
                }

                // Handle unquote (should error if not inside syntax-quote)
                if (std.mem.eql(u8, first_sym, "unquote")) {
                    return MacroExpandError.InvalidSyntax;
                }

                // Handle unquote-splicing (should error if not inside syntax-quote)
                if (std.mem.eql(u8, first_sym, "unquote-splicing")) {
                    return MacroExpandError.InvalidSyntax;
                }

                // Handle gensym: (gensym) or (gensym "prefix")
                if (std.mem.eql(u8, first_sym, "gensym")) {
                    return try self.handleGensym(list.next);
                }

                // Handle let: (let [bindings] body)
                // Evaluate let at expansion time by extending the substitution map
                if (std.mem.eql(u8, first_sym, "let")) {
                    return try self.handleExpansionTimeLet(list.next, subst_map);
                }
            }

            // Recursively substitute in list elements
            return try self.substituteList(expr.list, subst_map);
        } else if (expr.isVector()) {
            return try self.substituteVector(expr.vector, subst_map);
        } else {
            // Other values stay as-is
            return expr;
        }
    }

    /// Handle (gensym) or (gensym "prefix") calls
    fn handleGensym(self: *MacroExpander, args: ?*const PersistentLinkedList(*Value)) MacroExpandError!*Value {
        // No arguments: (gensym) - use default prefix "G"
        if (args == null or args.?.value == null) {
            return try self.gensym("G");
        }

        const first_arg = args.?.value.?;

        // One argument: (gensym "prefix") - must be a string
        if (first_arg.isString()) {
            return try self.gensym(first_arg.string);
        } else if (first_arg.isSymbol()) {
            // Also allow symbols as prefix
            return try self.gensym(first_arg.symbol);
        } else {
            return MacroExpandError.InvalidSyntax;
        }
    }

    /// Handle expansion-time let: (let [bindings] body)
    /// Evaluates the let at macro expansion time by extending the substitution map
    /// and returning the substituted body (not a let form)
    fn handleExpansionTimeLet(self: *MacroExpander, args: ?*const PersistentLinkedList(*Value), subst_map: *std.StringHashMap(*Value)) MacroExpandError!*Value {
        // Get bindings vector
        if (args == null or args.?.value == null) {
            return MacroExpandError.InvalidSyntax;
        }

        const bindings_node = args.?;
        const bindings_val = bindings_node.value.?;

        if (!bindings_val.isVector()) {
            return MacroExpandError.InvalidSyntax;
        }

        const bindings = bindings_val.vector;

        // Get body
        if (bindings_node.next == null or bindings_node.next.?.value == null) {
            return MacroExpandError.InvalidSyntax;
        }

        const body = bindings_node.next.?.value.?;

        // Create extended substitution map
        var extended_map = std.StringHashMap(*Value).init(self.allocator);
        defer extended_map.deinit();

        // Copy existing bindings
        var iter = subst_map.iterator();
        while (iter.next()) |entry| {
            try extended_map.put(entry.key_ptr.*, entry.value_ptr.*);
        }

        // Process bindings: [name1 (: Type1) value1 name2 (: Type2) value2 ...]
        // OR simplified: [name1 value1 name2 value2 ...]
        const bindings_slice = bindings.slice();
        var i: usize = 0;

        while (i < bindings_slice.len) {
            // Get binding name
            if (i >= bindings_slice.len) break;
            const name_val = bindings_slice[i];

            if (!name_val.isSymbol()) {
                return MacroExpandError.InvalidSyntax;
            }

            const name = name_val.symbol;
            i += 1;

            // Check if next element is a type annotation (: Type)
            var value_expr: *Value = undefined;
            if (i < bindings_slice.len and bindings_slice[i].isList()) {
                const maybe_annotation = bindings_slice[i].list;
                if (!maybe_annotation.isEmpty() and maybe_annotation.value != null and
                    maybe_annotation.value.?.isSymbol() and
                    std.mem.eql(u8, maybe_annotation.value.?.symbol, ":"))
                {
                    // Skip type annotation
                    i += 1;
                }
            }

            // Get value expression
            if (i >= bindings_slice.len) {
                return MacroExpandError.InvalidSyntax;
            }

            value_expr = bindings_slice[i];
            i += 1;

            // Substitute the value expression with current bindings
            const substituted_value = try self.substitute(value_expr, &extended_map);

            // Add binding to extended map
            try extended_map.put(name, substituted_value);
        }

        // Substitute body with extended map
        return try self.substitute(body, &extended_map);
    }

    /// Substitute in a list
    fn substituteList(self: *MacroExpander, list: *const PersistentLinkedList(*Value), subst_map: *std.StringHashMap(*Value)) MacroExpandError!*Value {
        if (list.isEmpty()) {
            const new_list = try PersistentLinkedList(*Value).empty(self.allocator);
            const val = try self.allocator.create(Value);
            val.* = Value{ .list = new_list };
            return val;
        }

        // Build new list with substitutions
        var elements: [64]*Value = undefined;
        var count: usize = 0;
        var current = list;

        while (!current.isEmpty() and count < 64) {
            if (current.value) |v| {
                elements[count] = try self.substitute(v, subst_map);
                count += 1;
            }
            if (current.next) |next| {
                current = next;
            } else {
                break;
            }
        }

        // Rebuild list
        var new_list = try PersistentLinkedList(*Value).empty(self.allocator);
        var i = count;
        while (i > 0) {
            i -= 1;
            new_list = try new_list.push(self.allocator, elements[i]);
        }

        const val = try self.allocator.create(Value);
        val.* = Value{ .list = new_list };
        return val;
    }

    /// Substitute in a vector
    fn substituteVector(self: *MacroExpander, vec: PersistentVector(*Value), subst_map: *std.StringHashMap(*Value)) MacroExpandError!*Value {
        var new_vec = PersistentVector(*Value).init(self.allocator, null);
        const slice = vec.slice();

        for (slice) |elem| {
            const substituted = try self.substitute(elem, subst_map);
            new_vec = try new_vec.push(substituted);
        }

        const val = try self.allocator.create(Value);
        val.* = Value{ .vector = new_vec };
        return val;
    }

    /// Recursively expand all elements in a list
    fn expandListElements(self: *MacroExpander, list: *const PersistentLinkedList(*Value)) MacroExpandError!*Value {
        if (list.isEmpty()) {
            const new_list = try PersistentLinkedList(*Value).empty(self.allocator);
            const val = try self.allocator.create(Value);
            val.* = Value{ .list = new_list };
            return val;
        }

        var elements: [64]*Value = undefined;
        var count: usize = 0;
        var current = list;

        while (!current.isEmpty() and count < 64) {
            if (current.value) |v| {
                elements[count] = try self.expand(v);
                count += 1;
            }
            if (current.next) |next| {
                current = next;
            } else {
                break;
            }
        }

        // Rebuild list
        var new_list = try PersistentLinkedList(*Value).empty(self.allocator);
        var i = count;
        while (i > 0) {
            i -= 1;
            new_list = try new_list.push(self.allocator, elements[i]);
        }

        const val = try self.allocator.create(Value);
        val.* = Value{ .list = new_list };
        return val;
    }

    /// Recursively expand all elements in a vector
    fn expandVector(self: *MacroExpander, expr: *Value) MacroExpandError!*Value {
        const vec = expr.vector;
        var new_vec = PersistentVector(*Value).init(self.allocator, null);
        const slice = vec.slice();

        for (slice) |elem| {
            const expanded = try self.expand(elem);
            new_vec = try new_vec.push(expanded);
        }

        const val = try self.allocator.create(Value);
        val.* = Value{ .vector = new_vec };
        return val;
    }

    /// Expand syntax-quote form: `(syntax-quote expr)
    /// Inside syntax-quote, unquote (~) and unquote-splicing (~@) are processed
    fn expandSyntaxQuote(self: *MacroExpander, args: ?*const PersistentLinkedList(*Value), subst_map: *std.StringHashMap(*Value)) MacroExpandError!*Value {
        if (args == null or args.?.value == null) {
            return MacroExpandError.InvalidSyntax;
        }

        const expr = args.?.value.?;

        // Create a map for auto-gensym (sym#) tracking within this syntax-quote
        var auto_gensym_map = std.StringHashMap(*Value).init(self.allocator);
        defer auto_gensym_map.deinit();

        return try self.processSyntaxQuote(expr, subst_map, &auto_gensym_map);
    }

    /// Process an expression inside syntax-quote
    fn processSyntaxQuote(self: *MacroExpander, expr: *Value, subst_map: *std.StringHashMap(*Value), auto_gensym_map: *std.StringHashMap(*Value)) MacroExpandError!*Value {
        if (expr.isSymbol()) {
            // Check if this is an auto-gensym symbol (ends with #)
            const sym_name = expr.symbol;
            if (sym_name.len > 1 and sym_name[sym_name.len - 1] == '#') {
                // Get or create gensym for this symbol
                if (auto_gensym_map.get(sym_name)) |existing| {
                    return existing;
                } else {
                    // Generate new unique symbol using the name without #
                    const prefix = sym_name[0..sym_name.len - 1];
                    const new_sym = try self.gensym(prefix);
                    try auto_gensym_map.put(sym_name, new_sym);
                    return new_sym;
                }
            }
            // Regular symbol - return as-is
            return expr;
        } else if (expr.isList()) {
            const list = expr.list;

            // Check if this is an unquote, unquote-splicing, or gensym form
            if (!list.isEmpty() and list.value != null and list.value.?.isSymbol()) {
                const first_sym = list.value.?.symbol;

                // Handle unquote: ~expr
                if (std.mem.eql(u8, first_sym, "unquote")) {
                    if (list.next == null or list.next.?.value == null) {
                        return MacroExpandError.InvalidSyntax;
                    }
                    // Substitute the unquoted expression
                    return try self.substitute(list.next.?.value.?, subst_map);
                }

                // Handle unquote-splicing: ~@expr (only valid inside a list)
                if (std.mem.eql(u8, first_sym, "unquote-splicing")) {
                    // This should be handled at the list level, not here
                    return MacroExpandError.InvalidSyntax;
                }

                // Handle gensym inside syntax-quote: (gensym) or (gensym "prefix")
                if (std.mem.eql(u8, first_sym, "gensym")) {
                    return try self.handleGensym(list.next);
                }
            }

            // Recursively process list elements, handling unquote-splicing
            return try self.processSyntaxQuoteList(list, subst_map, auto_gensym_map);
        } else if (expr.isVector()) {
            return try self.processSyntaxQuoteVector(expr.vector, subst_map, auto_gensym_map);
        } else {
            // Atoms are returned as-is
            return expr;
        }
    }

    /// Process a list inside syntax-quote, handling unquote-splicing
    fn processSyntaxQuoteList(self: *MacroExpander, list: *const PersistentLinkedList(*Value), subst_map: *std.StringHashMap(*Value), auto_gensym_map: *std.StringHashMap(*Value)) MacroExpandError!*Value {
        if (list.isEmpty()) {
            const new_list = try PersistentLinkedList(*Value).empty(self.allocator);
            const val = try self.allocator.create(Value);
            val.* = Value{ .list = new_list };
            return val;
        }

        var elements: [64]*Value = undefined;
        var count: usize = 0;
        var current = list;

        while (!current.isEmpty() and count < 64) {
            if (current.value) |v| {
                // Check if this is an unquote-splicing form
                if (v.isList() and !v.list.isEmpty() and v.list.value != null and v.list.value.?.isSymbol()) {
                    const first_sym = v.list.value.?.symbol;
                    if (std.mem.eql(u8, first_sym, "unquote-splicing")) {
                        // Get the expression to splice
                        if (v.list.next == null or v.list.next.?.value == null) {
                            return MacroExpandError.InvalidSyntax;
                        }
                        const splice_expr = v.list.next.?.value.?;

                        // Substitute and expand the splice expression
                        const splice_result = try self.substitute(splice_expr, subst_map);

                        // The result should be a list - splice its elements
                        if (!splice_result.isList()) {
                            return MacroExpandError.InvalidSyntax;
                        }

                        // Add each element from the spliced list
                        var splice_current: *const PersistentLinkedList(*Value) = splice_result.list;
                        while (!splice_current.isEmpty() and count < 64) {
                            if (splice_current.value) |splice_elem| {
                                elements[count] = splice_elem;
                                count += 1;
                            }
                            if (splice_current.next) |next| {
                                splice_current = next;
                            } else {
                                break;
                            }
                        }
                    } else {
                        // Regular element - process it
                        elements[count] = try self.processSyntaxQuote(v, subst_map, auto_gensym_map);
                        count += 1;
                    }
                } else {
                    // Regular element - process it
                    elements[count] = try self.processSyntaxQuote(v, subst_map, auto_gensym_map);
                    count += 1;
                }
            }

            if (current.next) |next| {
                current = next;
            } else {
                break;
            }
        }

        // Rebuild list
        var new_list = try PersistentLinkedList(*Value).empty(self.allocator);
        var i = count;
        while (i > 0) {
            i -= 1;
            new_list = try new_list.push(self.allocator, elements[i]);
        }

        const val = try self.allocator.create(Value);
        val.* = Value{ .list = new_list };
        return val;
    }

    /// Process a vector inside syntax-quote
    fn processSyntaxQuoteVector(self: *MacroExpander, vec: PersistentVector(*Value), subst_map: *std.StringHashMap(*Value), auto_gensym_map: *std.StringHashMap(*Value)) MacroExpandError!*Value {
        var new_vec = PersistentVector(*Value).init(self.allocator, null);
        const slice = vec.slice();

        for (slice) |elem| {
            const processed = try self.processSyntaxQuote(elem, subst_map, auto_gensym_map);
            new_vec = try new_vec.push(processed);
        }

        const val = try self.allocator.create(Value);
        val.* = Value{ .vector = new_vec };
        return val;
    }
};

// Tests
test "macro expander basic creation" {
    var expander = MacroExpander.init(std.testing.allocator);
    defer expander.deinit();

    // Just verify it initializes
    try std.testing.expect(true);
}

test "macro definition parsing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Create (defmacro foo [x] x)
    var list = try PersistentLinkedList(*Value).empty(allocator);

    // Body: just x
    const body = try createSymbol(allocator, "x");
    list = try list.push(allocator, body);

    // Params: [x]
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    list = try list.push(allocator, params_val);

    // Name: foo
    const name = try createSymbol(allocator, "foo");
    list = try list.push(allocator, name);

    // defmacro symbol
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    list = try list.push(allocator, defmacro_sym);

    const expr = try allocator.create(Value);
    expr.* = Value{ .list = list };

    const result = try expander.expand(expr);

    // Should return a macro value
    try std.testing.expect(result.isMacro());
    try std.testing.expect(std.mem.eql(u8, result.macro_def.name, "foo"));
}

test "macro expansion simple" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // First, define a macro: (defmacro identity [x] x)
    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    const body = try createSymbol(allocator, "x");
    def_list = try def_list.push(allocator, body);

    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);

    const name = try createSymbol(allocator, "identity");
    def_list = try def_list.push(allocator, name);

    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);

    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Now use the macro: (identity 42)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 42 };
    call_list = try call_list.push(allocator, arg);

    const identity_sym = try createSymbol(allocator, "identity");
    call_list = try call_list.push(allocator, identity_sym);

    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to 42
    try std.testing.expect(result.isInt());
    try std.testing.expect(result.int == 42);
}

test "syntax-quote with unquote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Define: (defmacro test-macro [x] `(+ ~x 1))
    // Build body: `(+ ~x 1) which is (syntax-quote (+ (unquote x) 1))

    // First build (unquote x)
    var unquote_list = try PersistentLinkedList(*Value).empty(allocator);
    const x_sym = try createSymbol(allocator, "x");
    unquote_list = try unquote_list.push(allocator, x_sym);
    const unquote_sym = try createSymbol(allocator, "unquote");
    unquote_list = try unquote_list.push(allocator, unquote_sym);
    const unquote_val = try allocator.create(Value);
    unquote_val.* = Value{ .list = unquote_list };

    // Build (+ ~x 1)
    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    const one = try allocator.create(Value);
    one.* = Value{ .int = 1 };
    plus_list = try plus_list.push(allocator, one);
    plus_list = try plus_list.push(allocator, unquote_val);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    // Build (syntax-quote (+ ~x 1))
    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, plus_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    // Build macro definition: (defmacro test-macro [x] `(+ ~x 1))
    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-macro");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call the macro: (test-macro 10)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 10 };
    call_list = try call_list.push(allocator, arg);
    const test_macro_sym = try createSymbol(allocator, "test-macro");
    call_list = try call_list.push(allocator, test_macro_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Result should be (+ 10 1)
    try std.testing.expect(result.isList());
    const result_list = result.list;
    try std.testing.expect(result_list.value.?.isSymbol());
    try std.testing.expect(std.mem.eql(u8, result_list.value.?.symbol, "+"));

    const second = result_list.next.?.value.?;
    try std.testing.expect(second.isInt());
    try std.testing.expect(second.int == 10);

    const third = result_list.next.?.next.?.value.?;
    try std.testing.expect(third.isInt());
    try std.testing.expect(third.int == 1);
}

test "syntax-quote without unquote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Define: (defmacro test-macro [x] `(+ 5 6))
    // Build body: `(+ 5 6) which is (syntax-quote (+ 5 6))

    // Build (+ 5 6)
    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    const six = try allocator.create(Value);
    six.* = Value{ .int = 6 };
    plus_list = try plus_list.push(allocator, six);
    const five = try allocator.create(Value);
    five.* = Value{ .int = 5 };
    plus_list = try plus_list.push(allocator, five);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    // Build (syntax-quote (+ 5 6))
    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, plus_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    // Build macro definition
    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-macro");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call the macro: (test-macro 999)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 999 };
    call_list = try call_list.push(allocator, arg);
    const test_macro_sym = try createSymbol(allocator, "test-macro");
    call_list = try call_list.push(allocator, test_macro_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Result should be (+ 5 6) - argument ignored since body doesn't use it
    try std.testing.expect(result.isList());
    const result_list = result.list;
    try std.testing.expect(result_list.value.?.isSymbol());
    try std.testing.expect(std.mem.eql(u8, result_list.value.?.symbol, "+"));

    const second = result_list.next.?.value.?;
    try std.testing.expect(second.isInt());
    try std.testing.expect(second.int == 5);

    const third = result_list.next.?.next.?.value.?;
    try std.testing.expect(third.isInt());
    try std.testing.expect(third.int == 6);
}

// ============================================================================
// Comprehensive Quote/Unquote Tests
// ============================================================================

// Helper to get list element at index
fn getListElement(list_val: *Value, index: usize) ?*Value {
    if (!list_val.isList()) return null;
    var current: *const PersistentLinkedList(*Value) = list_val.list;
    var i: usize = 0;
    while (i < index) : (i += 1) {
        if (current.next) |next| {
            current = next;
        } else {
            return null;
        }
    }
    return current.value;
}

test "quote/unquote: basic literal preservation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-literal [x] `42)
    const forty_two = try allocator.create(Value);
    forty_two.* = Value{ .int = 42 };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, forty_two);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-literal");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-literal 999)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 999 };
    call_list = try call_list.push(allocator, arg);
    const test_sym = try createSymbol(allocator, "test-literal");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    try std.testing.expect(result.isInt());
    try std.testing.expectEqual(@as(i64, 42), result.int);
}

test "quote/unquote: single unquote substitution" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-unquote [x] `~x)
    var unquote_list = try PersistentLinkedList(*Value).empty(allocator);
    const x_sym = try createSymbol(allocator, "x");
    unquote_list = try unquote_list.push(allocator, x_sym);
    const unquote_sym = try createSymbol(allocator, "unquote");
    unquote_list = try unquote_list.push(allocator, unquote_sym);
    const unquote_val = try allocator.create(Value);
    unquote_val.* = Value{ .list = unquote_list };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, unquote_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-unquote");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-unquote 777)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 777 };
    call_list = try call_list.push(allocator, arg);
    const test_sym = try createSymbol(allocator, "test-unquote");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    try std.testing.expect(result.isInt());
    try std.testing.expectEqual(@as(i64, 777), result.int);
}

test "quote/unquote: multiple unquotes in list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-multi [a b] `(+ ~a ~b))
    var unquote_a_list = try PersistentLinkedList(*Value).empty(allocator);
    const a_sym = try createSymbol(allocator, "a");
    unquote_a_list = try unquote_a_list.push(allocator, a_sym);
    const unquote_sym1 = try createSymbol(allocator, "unquote");
    unquote_a_list = try unquote_a_list.push(allocator, unquote_sym1);
    const unquote_a_val = try allocator.create(Value);
    unquote_a_val.* = Value{ .list = unquote_a_list };

    var unquote_b_list = try PersistentLinkedList(*Value).empty(allocator);
    const b_sym = try createSymbol(allocator, "b");
    unquote_b_list = try unquote_b_list.push(allocator, b_sym);
    const unquote_sym2 = try createSymbol(allocator, "unquote");
    unquote_b_list = try unquote_b_list.push(allocator, unquote_sym2);
    const unquote_b_val = try allocator.create(Value);
    unquote_b_val.* = Value{ .list = unquote_b_list };

    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    plus_list = try plus_list.push(allocator, unquote_b_val);
    plus_list = try plus_list.push(allocator, unquote_a_val);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, plus_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_a = try createSymbol(allocator, "a");
    params_vec = try params_vec.push(param_a);
    const param_b = try createSymbol(allocator, "b");
    params_vec = try params_vec.push(param_b);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-multi");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-multi 10 20)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg2 = try allocator.create(Value);
    arg2.* = Value{ .int = 20 };
    call_list = try call_list.push(allocator, arg2);
    const arg1 = try allocator.create(Value);
    arg1.* = Value{ .int = 10 };
    call_list = try call_list.push(allocator, arg1);
    const test_sym = try createSymbol(allocator, "test-multi");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to (+ 10 20)
    try std.testing.expect(result.isList());
    const first = getListElement(result, 0).?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, first.symbol, "+"));

    const second = getListElement(result, 1).?;
    try std.testing.expect(second.isInt());
    try std.testing.expectEqual(@as(i64, 10), second.int);

    const third = getListElement(result, 2).?;
    try std.testing.expect(third.isInt());
    try std.testing.expectEqual(@as(i64, 20), third.int);
}

test "quote/unquote: unquote-splicing in list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-splice [xs] `(+ ~@xs))
    var splice_list = try PersistentLinkedList(*Value).empty(allocator);
    const xs_sym = try createSymbol(allocator, "xs");
    splice_list = try splice_list.push(allocator, xs_sym);
    const splice_sym = try createSymbol(allocator, "unquote-splicing");
    splice_list = try splice_list.push(allocator, splice_sym);
    const splice_val = try allocator.create(Value);
    splice_val.* = Value{ .list = splice_list };

    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    plus_list = try plus_list.push(allocator, splice_val);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, plus_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_xs = try createSymbol(allocator, "xs");
    params_vec = try params_vec.push(param_xs);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-splice");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-splice (1 2 3))
    var arg_list = try PersistentLinkedList(*Value).empty(allocator);
    const three = try allocator.create(Value);
    three.* = Value{ .int = 3 };
    arg_list = try arg_list.push(allocator, three);
    const two = try allocator.create(Value);
    two.* = Value{ .int = 2 };
    arg_list = try arg_list.push(allocator, two);
    const one = try allocator.create(Value);
    one.* = Value{ .int = 1 };
    arg_list = try arg_list.push(allocator, one);
    const arg_list_val = try allocator.create(Value);
    arg_list_val.* = Value{ .list = arg_list };

    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    call_list = try call_list.push(allocator, arg_list_val);
    const test_sym = try createSymbol(allocator, "test-splice");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to (+ 1 2 3)
    try std.testing.expect(result.isList());
    const first = getListElement(result, 0).?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, first.symbol, "+"));

    const second = getListElement(result, 1).?;
    try std.testing.expect(second.isInt());
    try std.testing.expectEqual(@as(i64, 1), second.int);

    const third = getListElement(result, 2).?;
    try std.testing.expect(third.isInt());
    try std.testing.expectEqual(@as(i64, 2), third.int);

    const fourth = getListElement(result, 3).?;
    try std.testing.expect(fourth.isInt());
    try std.testing.expectEqual(@as(i64, 3), fourth.int);
}

test "quote/unquote: mixed quoted and unquoted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-mixed [x] `(+ 10 ~x 20))
    var unquote_list = try PersistentLinkedList(*Value).empty(allocator);
    const x_sym = try createSymbol(allocator, "x");
    unquote_list = try unquote_list.push(allocator, x_sym);
    const unquote_sym = try createSymbol(allocator, "unquote");
    unquote_list = try unquote_list.push(allocator, unquote_sym);
    const unquote_val = try allocator.create(Value);
    unquote_val.* = Value{ .list = unquote_list };

    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    const twenty = try allocator.create(Value);
    twenty.* = Value{ .int = 20 };
    plus_list = try plus_list.push(allocator, twenty);
    plus_list = try plus_list.push(allocator, unquote_val);
    const ten = try allocator.create(Value);
    ten.* = Value{ .int = 10 };
    plus_list = try plus_list.push(allocator, ten);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, plus_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-mixed");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-mixed 5)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 5 };
    call_list = try call_list.push(allocator, arg);
    const test_sym = try createSymbol(allocator, "test-mixed");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to (+ 10 5 20)
    try std.testing.expect(result.isList());
    const first = getListElement(result, 0).?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, first.symbol, "+"));

    const second = getListElement(result, 1).?;
    try std.testing.expect(second.isInt());
    try std.testing.expectEqual(@as(i64, 10), second.int);

    const third = getListElement(result, 2).?;
    try std.testing.expect(third.isInt());
    try std.testing.expectEqual(@as(i64, 5), third.int);

    const fourth = getListElement(result, 3).?;
    try std.testing.expect(fourth.isInt());
    try std.testing.expectEqual(@as(i64, 20), fourth.int);
}

test "quote/unquote: error - unquote outside syntax-quote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro bad-macro [x] (unquote x))  -- no syntax-quote!
    var unquote_list = try PersistentLinkedList(*Value).empty(allocator);
    const x_sym = try createSymbol(allocator, "x");
    unquote_list = try unquote_list.push(allocator, x_sym);
    const unquote_sym = try createSymbol(allocator, "unquote");
    unquote_list = try unquote_list.push(allocator, unquote_sym);
    const unquote_val = try allocator.create(Value);
    unquote_val.* = Value{ .list = unquote_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, unquote_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "bad-macro");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (bad-macro 42)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 42 };
    call_list = try call_list.push(allocator, arg);
    const test_sym = try createSymbol(allocator, "bad-macro");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = expander.expand(call_expr);
    try std.testing.expectError(error.InvalidSyntax, result);
}

test "quote/unquote: error - unquote-splicing outside syntax-quote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro bad-splice [xs] (unquote-splicing xs))  -- no syntax-quote!
    var splice_list = try PersistentLinkedList(*Value).empty(allocator);
    const xs_sym = try createSymbol(allocator, "xs");
    splice_list = try splice_list.push(allocator, xs_sym);
    const splice_sym = try createSymbol(allocator, "unquote-splicing");
    splice_list = try splice_list.push(allocator, splice_sym);
    const splice_val = try allocator.create(Value);
    splice_val.* = Value{ .list = splice_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, splice_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_xs = try createSymbol(allocator, "xs");
    params_vec = try params_vec.push(param_xs);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "bad-splice");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call with a list argument
    var arg_list = try PersistentLinkedList(*Value).empty(allocator);
    const one = try allocator.create(Value);
    one.* = Value{ .int = 1 };
    arg_list = try arg_list.push(allocator, one);
    const arg_list_val = try allocator.create(Value);
    arg_list_val.* = Value{ .list = arg_list };

    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    call_list = try call_list.push(allocator, arg_list_val);
    const test_sym = try createSymbol(allocator, "bad-splice");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = expander.expand(call_expr);
    try std.testing.expectError(error.InvalidSyntax, result);
}

test "quote/unquote: nested list structures" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-nested [x] `(+ (* 2 ~x) 3))
    var unquote_list = try PersistentLinkedList(*Value).empty(allocator);
    const x_sym = try createSymbol(allocator, "x");
    unquote_list = try unquote_list.push(allocator, x_sym);
    const unquote_sym = try createSymbol(allocator, "unquote");
    unquote_list = try unquote_list.push(allocator, unquote_sym);
    const unquote_val = try allocator.create(Value);
    unquote_val.* = Value{ .list = unquote_list };

    var mult_list = try PersistentLinkedList(*Value).empty(allocator);
    mult_list = try mult_list.push(allocator, unquote_val);
    const two = try allocator.create(Value);
    two.* = Value{ .int = 2 };
    mult_list = try mult_list.push(allocator, two);
    const mult_sym = try createSymbol(allocator, "*");
    mult_list = try mult_list.push(allocator, mult_sym);
    const mult_val = try allocator.create(Value);
    mult_val.* = Value{ .list = mult_list };

    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    const three = try allocator.create(Value);
    three.* = Value{ .int = 3 };
    plus_list = try plus_list.push(allocator, three);
    plus_list = try plus_list.push(allocator, mult_val);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, plus_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-nested");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-nested 5)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 5 };
    call_list = try call_list.push(allocator, arg);
    const test_sym = try createSymbol(allocator, "test-nested");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to (+ (* 2 5) 3)
    try std.testing.expect(result.isList());
    const first = getListElement(result, 0).?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, first.symbol, "+"));

    const second = getListElement(result, 1).?;
    try std.testing.expect(second.isList());
    const mult_first = getListElement(second, 0).?;
    try std.testing.expect(mult_first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, mult_first.symbol, "*"));
    const mult_second = getListElement(second, 1).?;
    try std.testing.expectEqual(@as(i64, 2), mult_second.int);
    const mult_third = getListElement(second, 2).?;
    try std.testing.expectEqual(@as(i64, 5), mult_third.int);

    const third = getListElement(result, 2).?;
    try std.testing.expectEqual(@as(i64, 3), third.int);
}

test "quote/unquote: unquote-splicing with empty list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-empty-splice [xs] `(+ ~@xs 10))
    var splice_list = try PersistentLinkedList(*Value).empty(allocator);
    const xs_sym = try createSymbol(allocator, "xs");
    splice_list = try splice_list.push(allocator, xs_sym);
    const splice_sym = try createSymbol(allocator, "unquote-splicing");
    splice_list = try splice_list.push(allocator, splice_sym);
    const splice_val = try allocator.create(Value);
    splice_val.* = Value{ .list = splice_list };

    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    const ten = try allocator.create(Value);
    ten.* = Value{ .int = 10 };
    plus_list = try plus_list.push(allocator, ten);
    plus_list = try plus_list.push(allocator, splice_val);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, plus_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_xs = try createSymbol(allocator, "xs");
    params_vec = try params_vec.push(param_xs);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-empty-splice");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-empty-splice ())
    const empty_list = try PersistentLinkedList(*Value).empty(allocator);
    const empty_list_val = try allocator.create(Value);
    empty_list_val.* = Value{ .list = empty_list };

    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    call_list = try call_list.push(allocator, empty_list_val);
    const test_sym = try createSymbol(allocator, "test-empty-splice");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to (+ 10) - empty list spliced in
    try std.testing.expect(result.isList());
    const first = getListElement(result, 0).?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, first.symbol, "+"));

    const second = getListElement(result, 1).?;
    try std.testing.expectEqual(@as(i64, 10), second.int);

    const third = getListElement(result, 2);
    try std.testing.expect(third == null);
}

test "gensym: basic unique symbol generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Generate two symbols and verify they're different
    const sym1 = try expander.gensym("x");
    const sym2 = try expander.gensym("x");

    try std.testing.expect(sym1.isSymbol());
    try std.testing.expect(sym2.isSymbol());
    try std.testing.expect(!std.mem.eql(u8, sym1.symbol, sym2.symbol));

    // Verify format: prefix__N
    try std.testing.expect(std.mem.startsWith(u8, sym1.symbol, "x__"));
    try std.testing.expect(std.mem.startsWith(u8, sym2.symbol, "x__"));
}

test "gensym: in macro without arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-gensym [] (gensym))
    var gensym_call_list = try PersistentLinkedList(*Value).empty(allocator);
    const gensym_sym = try createSymbol(allocator, "gensym");
    gensym_call_list = try gensym_call_list.push(allocator, gensym_sym);
    const gensym_call_val = try allocator.create(Value);
    gensym_call_val.* = Value{ .list = gensym_call_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, gensym_call_val);
    const params_vec = PersistentVector(*Value).init(allocator, null);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-gensym");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-gensym)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const test_sym = try createSymbol(allocator, "test-gensym");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should return a unique symbol with default prefix "G"
    try std.testing.expect(result.isSymbol());
    try std.testing.expect(std.mem.startsWith(u8, result.symbol, "G__"));
}

test "gensym: in macro with string prefix" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-gensym-prefix [] (gensym "temp"))
    var gensym_call_list = try PersistentLinkedList(*Value).empty(allocator);
    const prefix = try createString(allocator, "temp");
    gensym_call_list = try gensym_call_list.push(allocator, prefix);
    const gensym_sym = try createSymbol(allocator, "gensym");
    gensym_call_list = try gensym_call_list.push(allocator, gensym_sym);
    const gensym_call_val = try allocator.create(Value);
    gensym_call_val.* = Value{ .list = gensym_call_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, gensym_call_val);
    const params_vec = PersistentVector(*Value).init(allocator, null);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-gensym-prefix");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-gensym-prefix)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const test_sym = try createSymbol(allocator, "test-gensym-prefix");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should return a unique symbol with "temp" prefix
    try std.testing.expect(result.isSymbol());
    try std.testing.expect(std.mem.startsWith(u8, result.symbol, "temp__"));
}

test "gensym: in syntax-quote creates unique symbols" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro let-unique [x val body] `(let [(gensym "tmp") ~val] ~body))
    // Build (gensym "tmp")
    var gensym_list = try PersistentLinkedList(*Value).empty(allocator);
    const tmp_str = try createString(allocator, "tmp");
    gensym_list = try gensym_list.push(allocator, tmp_str);
    const gensym_sym = try createSymbol(allocator, "gensym");
    gensym_list = try gensym_list.push(allocator, gensym_sym);
    const gensym_val = try allocator.create(Value);
    gensym_val.* = Value{ .list = gensym_list };

    // Build ~val
    var unquote_val_list = try PersistentLinkedList(*Value).empty(allocator);
    const val_sym = try createSymbol(allocator, "val");
    unquote_val_list = try unquote_val_list.push(allocator, val_sym);
    const unquote_sym1 = try createSymbol(allocator, "unquote");
    unquote_val_list = try unquote_val_list.push(allocator, unquote_sym1);
    const unquote_val = try allocator.create(Value);
    unquote_val.* = Value{ .list = unquote_val_list };

    // Build [(gensym "tmp") ~val]
    var binding_vec = PersistentVector(*Value).init(allocator, null);
    binding_vec = try binding_vec.push(gensym_val);
    binding_vec = try binding_vec.push(unquote_val);
    const binding_vec_val = try allocator.create(Value);
    binding_vec_val.* = Value{ .vector = binding_vec };

    // Build ~body
    var unquote_body_list = try PersistentLinkedList(*Value).empty(allocator);
    const body_sym = try createSymbol(allocator, "body");
    unquote_body_list = try unquote_body_list.push(allocator, body_sym);
    const unquote_sym2 = try createSymbol(allocator, "unquote");
    unquote_body_list = try unquote_body_list.push(allocator, unquote_sym2);
    const unquote_body = try allocator.create(Value);
    unquote_body.* = Value{ .list = unquote_body_list };

    // Build (let [(gensym "tmp") ~val] ~body)
    var let_list = try PersistentLinkedList(*Value).empty(allocator);
    let_list = try let_list.push(allocator, unquote_body);
    let_list = try let_list.push(allocator, binding_vec_val);
    const let_sym = try createSymbol(allocator, "let");
    let_list = try let_list.push(allocator, let_sym);
    const let_val = try allocator.create(Value);
    let_val.* = Value{ .list = let_list };

    // Build `(let [(gensym "tmp") ~val] ~body)
    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, let_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    // Build macro definition
    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const param_val = try createSymbol(allocator, "val");
    params_vec = try params_vec.push(param_val);
    const param_body = try createSymbol(allocator, "body");
    params_vec = try params_vec.push(param_body);
    const params_val_vec = try allocator.create(Value);
    params_val_vec.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val_vec);
    const macro_name = try createSymbol(allocator, "let-unique");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (let-unique x 42 x)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg_body = try createSymbol(allocator, "x");
    call_list = try call_list.push(allocator, arg_body);
    const arg_val = try allocator.create(Value);
    arg_val.* = Value{ .int = 42 };
    call_list = try call_list.push(allocator, arg_val);
    const arg_x = try createSymbol(allocator, "x");
    call_list = try call_list.push(allocator, arg_x);
    const test_sym = try createSymbol(allocator, "let-unique");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to (let [tmp__N 42] x)
    try std.testing.expect(result.isList());
    const first = getListElement(result, 0).?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, first.symbol, "let"));

    const second = getListElement(result, 1).?;
    try std.testing.expect(second.isVector());
    const binding_slice = second.vector.slice();
    try std.testing.expectEqual(@as(usize, 2), binding_slice.len);

    // First element should be a gensym'd symbol
    try std.testing.expect(binding_slice[0].isSymbol());
    try std.testing.expect(std.mem.startsWith(u8, binding_slice[0].symbol, "tmp__"));

    // Second element should be 42
    try std.testing.expect(binding_slice[1].isInt());
    try std.testing.expectEqual(@as(i64, 42), binding_slice[1].int);
}

test "gensym: multiple calls generate different symbols" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro two-gensyms [] `[(gensym "a") (gensym "b")])
    var gensym1_list = try PersistentLinkedList(*Value).empty(allocator);
    const a_str = try createString(allocator, "a");
    gensym1_list = try gensym1_list.push(allocator, a_str);
    const gensym_sym1 = try createSymbol(allocator, "gensym");
    gensym1_list = try gensym1_list.push(allocator, gensym_sym1);
    const gensym1_val = try allocator.create(Value);
    gensym1_val.* = Value{ .list = gensym1_list };

    var gensym2_list = try PersistentLinkedList(*Value).empty(allocator);
    const b_str = try createString(allocator, "b");
    gensym2_list = try gensym2_list.push(allocator, b_str);
    const gensym_sym2 = try createSymbol(allocator, "gensym");
    gensym2_list = try gensym2_list.push(allocator, gensym_sym2);
    const gensym2_val = try allocator.create(Value);
    gensym2_val.* = Value{ .list = gensym2_list };

    var vec = PersistentVector(*Value).init(allocator, null);
    vec = try vec.push(gensym1_val);
    vec = try vec.push(gensym2_val);
    const vec_val = try allocator.create(Value);
    vec_val.* = Value{ .vector = vec };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, vec_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    const params_vec = PersistentVector(*Value).init(allocator, null);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "two-gensyms");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (two-gensyms)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const test_sym = try createSymbol(allocator, "two-gensyms");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to [a__N b__M] where N != M
    try std.testing.expect(result.isVector());
    const result_slice = result.vector.slice();
    try std.testing.expectEqual(@as(usize, 2), result_slice.len);

    try std.testing.expect(result_slice[0].isSymbol());
    try std.testing.expect(result_slice[1].isSymbol());

    try std.testing.expect(std.mem.startsWith(u8, result_slice[0].symbol, "a__"));
    try std.testing.expect(std.mem.startsWith(u8, result_slice[1].symbol, "b__"));

    // Verify they're different symbols
    try std.testing.expect(!std.mem.eql(u8, result_slice[0].symbol, result_slice[1].symbol));
}

test "auto-gensym: basic x# usage with same symbol" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro let-test [] `(let [x# 5] (+ x# x#)))
    // Both x# should resolve to the same unique symbol

    // Build x# symbol
    const x_hash = try createSymbol(allocator, "x#");

    // Build 5
    const five = try allocator.create(Value);
    five.* = Value{ .int = 5 };

    // Build [x# 5]
    var binding_vec = PersistentVector(*Value).init(allocator, null);
    binding_vec = try binding_vec.push(x_hash);
    binding_vec = try binding_vec.push(five);
    const binding_vec_val = try allocator.create(Value);
    binding_vec_val.* = Value{ .vector = binding_vec };

    // Build second x# and third x#
    const x_hash2 = try createSymbol(allocator, "x#");
    const x_hash3 = try createSymbol(allocator, "x#");

    // Build (+ x# x#)
    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    plus_list = try plus_list.push(allocator, x_hash3);
    plus_list = try plus_list.push(allocator, x_hash2);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    // Build (let [x# 5] (+ x# x#))
    var let_list = try PersistentLinkedList(*Value).empty(allocator);
    let_list = try let_list.push(allocator, plus_val);
    let_list = try let_list.push(allocator, binding_vec_val);
    const let_sym = try createSymbol(allocator, "let");
    let_list = try let_list.push(allocator, let_sym);
    const let_val = try allocator.create(Value);
    let_val.* = Value{ .list = let_list };

    // Build `(let [x# 5] (+ x# x#))
    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, let_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    // Build macro definition
    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    const params_vec = PersistentVector(*Value).init(allocator, null);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "let-test");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (let-test)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const test_sym = try createSymbol(allocator, "let-test");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to (let [x__N 5] (+ x__N x__N))
    try std.testing.expect(result.isList());
    const first = getListElement(result, 0).?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, first.symbol, "let"));

    const second = getListElement(result, 1).?;
    try std.testing.expect(second.isVector());
    const binding_slice = second.vector.slice();
    try std.testing.expectEqual(@as(usize, 2), binding_slice.len);

    // First element should be a gensym'd symbol
    try std.testing.expect(binding_slice[0].isSymbol());
    try std.testing.expect(std.mem.startsWith(u8, binding_slice[0].symbol, "x__"));
    const gensym_name = binding_slice[0].symbol;

    // Check the plus expression
    const third = getListElement(result, 2).?;
    try std.testing.expect(third.isList());
    const plus_first = getListElement(third, 0).?;
    try std.testing.expect(plus_first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, plus_first.symbol, "+"));

    // Both x# in the + expression should have the same gensym name
    const plus_second = getListElement(third, 1).?;
    try std.testing.expect(plus_second.isSymbol());
    try std.testing.expect(std.mem.eql(u8, plus_second.symbol, gensym_name));

    const plus_third = getListElement(third, 2).?;
    try std.testing.expect(plus_third.isSymbol());
    try std.testing.expect(std.mem.eql(u8, plus_third.symbol, gensym_name));
}

test "auto-gensym: different symbols get different gensyms" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-multi [] `(+ x# y#))
    const x_hash = try createSymbol(allocator, "x#");
    const y_hash = try createSymbol(allocator, "y#");

    var plus_list = try PersistentLinkedList(*Value).empty(allocator);
    plus_list = try plus_list.push(allocator, y_hash);
    plus_list = try plus_list.push(allocator, x_hash);
    const plus_sym = try createSymbol(allocator, "+");
    plus_list = try plus_list.push(allocator, plus_sym);
    const plus_val = try allocator.create(Value);
    plus_val.* = Value{ .list = plus_list };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, plus_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    const params_vec = PersistentVector(*Value).init(allocator, null);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-multi");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const test_sym = try createSymbol(allocator, "test-multi");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to (+ x__N y__M) where N != M
    try std.testing.expect(result.isList());
    const first = getListElement(result, 0).?;
    try std.testing.expect(first.isSymbol());
    try std.testing.expect(std.mem.eql(u8, first.symbol, "+"));

    const second = getListElement(result, 1).?;
    try std.testing.expect(second.isSymbol());
    try std.testing.expect(std.mem.startsWith(u8, second.symbol, "x__"));

    const third = getListElement(result, 2).?;
    try std.testing.expect(third.isSymbol());
    try std.testing.expect(std.mem.startsWith(u8, third.symbol, "y__"));

    // Verify they're different
    try std.testing.expect(!std.mem.eql(u8, second.symbol, third.symbol));
}

test "auto-gensym: scoped to each syntax-quote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // Define first macro: (defmacro m1 [] `x#)
    const x_hash1 = try createSymbol(allocator, "x#");

    var sq_list1 = try PersistentLinkedList(*Value).empty(allocator);
    sq_list1 = try sq_list1.push(allocator, x_hash1);
    const sq_sym1 = try createSymbol(allocator, "syntax-quote");
    sq_list1 = try sq_list1.push(allocator, sq_sym1);
    const sq_val1 = try allocator.create(Value);
    sq_val1.* = Value{ .list = sq_list1 };

    var def_list1 = try PersistentLinkedList(*Value).empty(allocator);
    def_list1 = try def_list1.push(allocator, sq_val1);
    const params_vec1 = PersistentVector(*Value).init(allocator, null);
    const params_val1 = try allocator.create(Value);
    params_val1.* = Value{ .vector = params_vec1 };
    def_list1 = try def_list1.push(allocator, params_val1);
    const macro_name1 = try createSymbol(allocator, "m1");
    def_list1 = try def_list1.push(allocator, macro_name1);
    const defmacro_sym1 = try createSymbol(allocator, "defmacro");
    def_list1 = try def_list1.push(allocator, defmacro_sym1);
    const def_expr1 = try allocator.create(Value);
    def_expr1.* = Value{ .list = def_list1 };

    _ = try expander.expand(def_expr1);

    // Define second macro: (defmacro m2 [] `x#)
    const x_hash2 = try createSymbol(allocator, "x#");

    var sq_list2 = try PersistentLinkedList(*Value).empty(allocator);
    sq_list2 = try sq_list2.push(allocator, x_hash2);
    const sq_sym2 = try createSymbol(allocator, "syntax-quote");
    sq_list2 = try sq_list2.push(allocator, sq_sym2);
    const sq_val2 = try allocator.create(Value);
    sq_val2.* = Value{ .list = sq_list2 };

    var def_list2 = try PersistentLinkedList(*Value).empty(allocator);
    def_list2 = try def_list2.push(allocator, sq_val2);
    const params_vec2 = PersistentVector(*Value).init(allocator, null);
    const params_val2 = try allocator.create(Value);
    params_val2.* = Value{ .vector = params_vec2 };
    def_list2 = try def_list2.push(allocator, params_val2);
    const macro_name2 = try createSymbol(allocator, "m2");
    def_list2 = try def_list2.push(allocator, macro_name2);
    const defmacro_sym2 = try createSymbol(allocator, "defmacro");
    def_list2 = try def_list2.push(allocator, defmacro_sym2);
    const def_expr2 = try allocator.create(Value);
    def_expr2.* = Value{ .list = def_list2 };

    _ = try expander.expand(def_expr2);

    // Call both macros and verify they generate different gensyms
    var call_list1 = try PersistentLinkedList(*Value).empty(allocator);
    const test_sym1 = try createSymbol(allocator, "m1");
    call_list1 = try call_list1.push(allocator, test_sym1);
    const call_expr1 = try allocator.create(Value);
    call_expr1.* = Value{ .list = call_list1 };

    const result1 = try expander.expand(call_expr1);

    var call_list2 = try PersistentLinkedList(*Value).empty(allocator);
    const test_sym2 = try createSymbol(allocator, "m2");
    call_list2 = try call_list2.push(allocator, test_sym2);
    const call_expr2 = try allocator.create(Value);
    call_expr2.* = Value{ .list = call_list2 };

    const result2 = try expander.expand(call_expr2);

    // Both should be symbols starting with x__
    try std.testing.expect(result1.isSymbol());
    try std.testing.expect(std.mem.startsWith(u8, result1.symbol, "x__"));

    try std.testing.expect(result2.isSymbol());
    try std.testing.expect(std.mem.startsWith(u8, result2.symbol, "x__"));

    // But they should be different symbols (each syntax-quote gets fresh gensyms)
    try std.testing.expect(!std.mem.eql(u8, result1.symbol, result2.symbol));
}

test "quote/unquote: vector with unquote" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    // (defmacro test-vec [x] `[1 ~x 3])
    var vec = PersistentVector(*Value).init(allocator, null);
    const one = try allocator.create(Value);
    one.* = Value{ .int = 1 };
    vec = try vec.push(one);

    var unquote_list = try PersistentLinkedList(*Value).empty(allocator);
    const x_sym = try createSymbol(allocator, "x");
    unquote_list = try unquote_list.push(allocator, x_sym);
    const unquote_sym = try createSymbol(allocator, "unquote");
    unquote_list = try unquote_list.push(allocator, unquote_sym);
    const unquote_val = try allocator.create(Value);
    unquote_val.* = Value{ .list = unquote_list };

    vec = try vec.push(unquote_val);
    const three = try allocator.create(Value);
    three.* = Value{ .int = 3 };
    vec = try vec.push(three);

    const vec_val = try allocator.create(Value);
    vec_val.* = Value{ .vector = vec };

    var sq_list = try PersistentLinkedList(*Value).empty(allocator);
    sq_list = try sq_list.push(allocator, vec_val);
    const sq_sym = try createSymbol(allocator, "syntax-quote");
    sq_list = try sq_list.push(allocator, sq_sym);
    const sq_val = try allocator.create(Value);
    sq_val.* = Value{ .list = sq_list };

    var def_list = try PersistentLinkedList(*Value).empty(allocator);
    def_list = try def_list.push(allocator, sq_val);
    var params_vec = PersistentVector(*Value).init(allocator, null);
    const param_x = try createSymbol(allocator, "x");
    params_vec = try params_vec.push(param_x);
    const params_val = try allocator.create(Value);
    params_val.* = Value{ .vector = params_vec };
    def_list = try def_list.push(allocator, params_val);
    const macro_name = try createSymbol(allocator, "test-vec");
    def_list = try def_list.push(allocator, macro_name);
    const defmacro_sym = try createSymbol(allocator, "defmacro");
    def_list = try def_list.push(allocator, defmacro_sym);
    const def_expr = try allocator.create(Value);
    def_expr.* = Value{ .list = def_list };

    _ = try expander.expand(def_expr);

    // Call: (test-vec 2)
    var call_list = try PersistentLinkedList(*Value).empty(allocator);
    const arg = try allocator.create(Value);
    arg.* = Value{ .int = 2 };
    call_list = try call_list.push(allocator, arg);
    const test_sym = try createSymbol(allocator, "test-vec");
    call_list = try call_list.push(allocator, test_sym);
    const call_expr = try allocator.create(Value);
    call_expr.* = Value{ .list = call_list };

    const result = try expander.expand(call_expr);

    // Should expand to [1 2 3]
    try std.testing.expect(result.isVector());
    const result_slice = result.vector.slice();
    try std.testing.expectEqual(@as(usize, 3), result_slice.len);
    try std.testing.expectEqual(@as(i64, 1), result_slice[0].int);
    try std.testing.expectEqual(@as(i64, 2), result_slice[1].int);
    try std.testing.expectEqual(@as(i64, 3), result_slice[2].int);
}
