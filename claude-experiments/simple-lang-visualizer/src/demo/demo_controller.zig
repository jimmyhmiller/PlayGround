const std = @import("std");
const ts = @import("../types/type_system.zig");

/// Demo stage identifier
pub const DemoStage = struct {
    id: usize,
    description: []const u8,
    action: StageAction,

    pub const StageAction = union(enum) {
        add_type: AddTypeAction,
        add_struct_field: AddStructFieldAction,
        add_enum_variant: AddEnumVariantAction,
        add_function_param: AddFunctionParamAction,
        add_tuple_element: AddTupleElementAction,
        complete: void,

        pub const AddTypeAction = struct {
            type_id: []const u8,
            type_name: []const u8,
            type_kind: ts.TypeKind,
            position: ts.Vec2,
            color: []const u8,
        };

        pub const AddStructFieldAction = struct {
            type_id: []const u8,
            field_name: []const u8,
            field_type: []const u8,
            description: []const u8,
        };

        pub const AddEnumVariantAction = struct {
            type_id: []const u8,
            variant_name: []const u8,
            payload: ?[]const u8,
            description: []const u8,
        };

        pub const AddFunctionParamAction = struct {
            type_id: []const u8,
            param_name: []const u8,
            param_type: []const u8,
        };

        pub const AddTupleElementAction = struct {
            type_id: []const u8,
            element_type: []const u8,
            description: []const u8,
        };
    };
};

/// Demo scenario with multiple stages
pub const Demo = struct {
    name: []const u8,
    description: []const u8,
    stages: []DemoStage,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Demo) void {
        self.allocator.free(self.stages);
    }
};

/// Demo controller manages progression through stages
pub const DemoController = struct {
    current_demo: ?*Demo,
    current_stage: usize,
    auto_play: bool,
    auto_play_timer: f32,
    auto_play_delay: f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) DemoController {
        return .{
            .current_demo = null,
            .current_stage = 0,
            .auto_play = false,
            .auto_play_timer = 0.0,
            .auto_play_delay = 2.0, // 2 seconds between stages
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DemoController) void {
        if (self.current_demo) |demo| {
            demo.deinit();
            self.allocator.destroy(demo);
        }
    }

    /// Load a demo
    pub fn loadDemo(self: *DemoController, demo: *Demo) void {
        if (self.current_demo) |old_demo| {
            old_demo.deinit();
            self.allocator.destroy(old_demo);
        }
        self.current_demo = demo;
        self.current_stage = 0;
        self.auto_play = false;
        self.auto_play_timer = 0.0;
    }

    /// Get current stage
    pub fn getCurrentStage(self: *const DemoController) ?*const DemoStage {
        if (self.current_demo) |demo| {
            if (self.current_stage < demo.stages.len) {
                return &demo.stages[self.current_stage];
            }
        }
        return null;
    }

    /// Advance to next stage
    pub fn nextStage(self: *DemoController) bool {
        if (self.current_demo) |demo| {
            if (self.current_stage < demo.stages.len - 1) {
                self.current_stage += 1;
                self.auto_play_timer = 0.0;
                return true;
            }
        }
        return false;
    }

    /// Go back to previous stage
    pub fn previousStage(self: *DemoController) bool {
        if (self.current_stage > 0) {
            self.current_stage -= 1;
            self.auto_play_timer = 0.0;
            return true;
        }
        return false;
    }

    /// Reset to beginning
    pub fn reset(self: *DemoController) void {
        self.current_stage = 0;
        self.auto_play = false;
        self.auto_play_timer = 0.0;
    }

    /// Toggle auto-play mode
    pub fn toggleAutoPlay(self: *DemoController) void {
        self.auto_play = !self.auto_play;
        self.auto_play_timer = 0.0;
    }

    /// Update auto-play timer
    pub fn update(self: *DemoController, delta_time: f32) void {
        if (self.auto_play) {
            self.auto_play_timer += delta_time;
            if (self.auto_play_timer >= self.auto_play_delay) {
                if (!self.nextStage()) {
                    // Reached end, stop auto-play
                    self.auto_play = false;
                }
                self.auto_play_timer = 0.0;
            }
        }
    }

    /// Check if at the end of the demo
    pub fn isComplete(self: *const DemoController) bool {
        if (self.current_demo) |demo| {
            return self.current_stage >= demo.stages.len - 1;
        }
        return true;
    }

    /// Get progress (0.0 to 1.0)
    pub fn getProgress(self: *const DemoController) f32 {
        if (self.current_demo) |demo| {
            if (demo.stages.len == 0) return 1.0;
            return @as(f32, @floatFromInt(self.current_stage)) / @as(f32, @floatFromInt(demo.stages.len - 1));
        }
        return 0.0;
    }
};

/// Create a Result<T, E> enum demo
pub fn createResultDemo(allocator: std.mem.Allocator) !*Demo {
    const stages = try allocator.alloc(DemoStage, 4);

    stages[0] = .{
        .id = 0,
        .description = "Empty canvas - press SPACE to begin",
        .action = .{ .complete = {} },
    };

    stages[1] = .{
        .id = 1,
        .description = "Create Result enum - the box grows from center",
        .action = .{
            .add_type = .{
                .type_id = "result",
                .type_name = "Result<T, E>",
                .type_kind = .@"enum",
                .position = ts.Vec2.init(500, 300),
                .color = "#48BB78",
            },
        },
    };

    stages[2] = .{
        .id = 2,
        .description = "Add Ok variant - box expands to make room",
        .action = .{
            .add_enum_variant = .{
                .type_id = "result",
                .variant_name = "Ok",
                .payload = "T",
                .description = "Success case with value of type T",
            },
        },
    };

    stages[3] = .{
        .id = 3,
        .description = "Add Err variant - box grows again for new variant",
        .action = .{
            .add_enum_variant = .{
                .type_id = "result",
                .variant_name = "Err",
                .payload = "E",
                .description = "Error case with error of type E",
            },
        },
    };

    const demo = try allocator.create(Demo);
    demo.* = .{
        .name = "Result Enum",
        .description = "Building a Result<T, E> algebraic data type",
        .stages = stages,
        .allocator = allocator,
    };

    return demo;
}

/// Create a User struct demo
pub fn createUserStructDemo(allocator: std.mem.Allocator) !*Demo {
    const stages = try allocator.alloc(DemoStage, 6);

    stages[0] = .{
        .id = 0,
        .description = "Start: Empty canvas",
        .action = .{ .complete = {} },
    };

    stages[1] = .{
        .id = 1,
        .description = "Create User struct",
        .action = .{
            .add_type = .{
                .type_id = "user",
                .type_name = "User",
                .type_kind = .@"struct",
                .position = ts.Vec2.init(400, 300),
                .color = "#E85D75",
            },
        },
    };

    stages[2] = .{
        .id = 2,
        .description = "Add 'name' field",
        .action = .{
            .add_struct_field = .{
                .type_id = "user",
                .field_name = "name",
                .field_type = "String",
                .description = "User's full name",
            },
        },
    };

    stages[3] = .{
        .id = 3,
        .description = "Add 'email' field",
        .action = .{
            .add_struct_field = .{
                .type_id = "user",
                .field_name = "email",
                .field_type = "String",
                .description = "User's email address",
            },
        },
    };

    stages[4] = .{
        .id = 4,
        .description = "Add 'age' field",
        .action = .{
            .add_struct_field = .{
                .type_id = "user",
                .field_name = "age",
                .field_type = "Int",
                .description = "User's age in years",
            },
        },
    };

    stages[5] = .{
        .id = 5,
        .description = "Complete! User struct with 3 fields",
        .action = .{ .complete = {} },
    };

    const demo = try allocator.create(Demo);
    demo.* = .{
        .name = "User Struct",
        .description = "Building a User struct type",
        .stages = stages,
        .allocator = allocator,
    };

    return demo;
}

/// Create a comprehensive Task Management System demo showing all type kinds
pub fn createTaskManagementDemo(allocator: std.mem.Allocator) !*Demo {
    const stages = try allocator.alloc(DemoStage, 25);

    stages[0] = .{
        .id = 0,
        .description = "Task Management System - A real-world application",
        .action = .{ .complete = {} },
    };

    // Primitives in top-left
    stages[1] = .{
        .id = 1,
        .description = "Primitives: String - the building blocks",
        .action = .{
            .add_type = .{
                .type_id = "string",
                .type_name = "String",
                .type_kind = .primitive,
                .position = ts.Vec2.init(100, 100),
                .color = "#8B9FD9",
            },
        },
    };

    stages[2] = .{
        .id = 2,
        .description = "Primitives: Int for numeric values",
        .action = .{
            .add_type = .{
                .type_id = "int",
                .type_name = "Int",
                .type_kind = .primitive,
                .position = ts.Vec2.init(100, 200),
                .color = "#8B9FD9",
            },
        },
    };

    stages[3] = .{
        .id = 3,
        .description = "Primitives: Bool for flags",
        .action = .{
            .add_type = .{
                .type_id = "bool",
                .type_name = "Bool",
                .type_kind = .primitive,
                .position = ts.Vec2.init(100, 300),
                .color = "#8B9FD9",
            },
        },
    };

    // Enum: Priority in top-center
    stages[4] = .{
        .id = 4,
        .description = "Enum: Priority levels for tasks",
        .action = .{
            .add_type = .{
                .type_id = "priority",
                .type_name = "Priority",
                .type_kind = .@"enum",
                .position = ts.Vec2.init(400, 80),
                .color = "#8BD99F",
            },
        },
    };

    stages[5] = .{
        .id = 5,
        .description = "Priority::Low variant",
        .action = .{
            .add_enum_variant = .{
                .type_id = "priority",
                .variant_name = "Low",
                .payload = null,
                .description = "Low priority",
            },
        },
    };

    stages[6] = .{
        .id = 6,
        .description = "Priority::Medium variant",
        .action = .{
            .add_enum_variant = .{
                .type_id = "priority",
                .variant_name = "Medium",
                .payload = null,
                .description = "Medium priority",
            },
        },
    };

    stages[7] = .{
        .id = 7,
        .description = "Priority::High variant",
        .action = .{
            .add_enum_variant = .{
                .type_id = "priority",
                .variant_name = "High",
                .payload = null,
                .description = "High priority",
            },
        },
    };

    // Enum: TaskStatus in top-right
    stages[8] = .{
        .id = 8,
        .description = "Enum: TaskStatus with payloads",
        .action = .{
            .add_type = .{
                .type_id = "status",
                .type_name = "TaskStatus",
                .type_kind = .@"enum",
                .position = ts.Vec2.init(850, 80),
                .color = "#8BD99F",
            },
        },
    };

    stages[9] = .{
        .id = 9,
        .description = "TaskStatus::Todo variant",
        .action = .{
            .add_enum_variant = .{
                .type_id = "status",
                .variant_name = "Todo",
                .payload = null,
                .description = "Not started yet",
            },
        },
    };

    stages[10] = .{
        .id = 10,
        .description = "TaskStatus::InProgress with assignee",
        .action = .{
            .add_enum_variant = .{
                .type_id = "status",
                .variant_name = "InProgress",
                .payload = "String",
                .description = "Currently being worked on",
            },
        },
    };

    stages[11] = .{
        .id = 11,
        .description = "TaskStatus::Done with completion date",
        .action = .{
            .add_enum_variant = .{
                .type_id = "status",
                .variant_name = "Done",
                .payload = "Int",
                .description = "Completed (timestamp)",
            },
        },
    };

    // Struct: Task in center
    stages[12] = .{
        .id = 12,
        .description = "Struct: Task - the main data structure",
        .action = .{
            .add_type = .{
                .type_id = "task",
                .type_name = "Task",
                .type_kind = .@"struct",
                .position = ts.Vec2.init(550, 350),
                .color = "#D98B9F",
            },
        },
    };

    stages[13] = .{
        .id = 13,
        .description = "Task.id field",
        .action = .{
            .add_struct_field = .{
                .type_id = "task",
                .field_name = "id",
                .field_type = "Int",
                .description = "Unique identifier",
            },
        },
    };

    stages[14] = .{
        .id = 14,
        .description = "Task.title field",
        .action = .{
            .add_struct_field = .{
                .type_id = "task",
                .field_name = "title",
                .field_type = "String",
                .description = "Task title",
            },
        },
    };

    stages[15] = .{
        .id = 15,
        .description = "Task.description field (optional)",
        .action = .{
            .add_struct_field = .{
                .type_id = "task",
                .field_name = "description",
                .field_type = "String?",
                .description = "Optional description",
            },
        },
    };

    stages[16] = .{
        .id = 16,
        .description = "Task.priority field",
        .action = .{
            .add_struct_field = .{
                .type_id = "task",
                .field_name = "priority",
                .field_type = "Priority",
                .description = "Task priority",
            },
        },
    };

    stages[17] = .{
        .id = 17,
        .description = "Task.status field",
        .action = .{
            .add_struct_field = .{
                .type_id = "task",
                .field_name = "status",
                .field_type = "TaskStatus",
                .description = "Current status",
            },
        },
    };

    stages[18] = .{
        .id = 18,
        .description = "Task.tags field (array)",
        .action = .{
            .add_struct_field = .{
                .type_id = "task",
                .field_name = "tags",
                .field_type = "String[]",
                .description = "Categorization tags",
            },
        },
    };

    // Tuple: Coordinates in bottom-left
    stages[19] = .{
        .id = 19,
        .description = "Tuple: Coordinates for task position on board",
        .action = .{
            .add_type = .{
                .type_id = "coords",
                .type_name = "Coordinates",
                .type_kind = .tuple,
                .position = ts.Vec2.init(150, 550),
                .color = "#B98BD9",
            },
        },
    };

    stages[20] = .{
        .id = 20,
        .description = "Coordinates: x position",
        .action = .{
            .add_tuple_element = .{
                .type_id = "coords",
                .element_type = "Int",
                .description = "X coordinate",
            },
        },
    };

    stages[21] = .{
        .id = 21,
        .description = "Coordinates: y position",
        .action = .{
            .add_tuple_element = .{
                .type_id = "coords",
                .element_type = "Int",
                .description = "Y coordinate",
            },
        },
    };

    // Function: filterTasks in bottom-right
    stages[22] = .{
        .id = 22,
        .description = "Function: filterTasks - filter by priority",
        .action = .{
            .add_type = .{
                .type_id = "filter",
                .type_name = "filterTasks",
                .type_kind = .function,
                .position = ts.Vec2.init(750, 580),
                .color = "#D9B98B",
            },
        },
    };

    stages[23] = .{
        .id = 23,
        .description = "filterTasks parameter: tasks array",
        .action = .{
            .add_function_param = .{
                .type_id = "filter",
                .param_name = "tasks",
                .param_type = "Task[]",
            },
        },
    };

    stages[24] = .{
        .id = 24,
        .description = "Complete! A real task management type system",
        .action = .{ .complete = {} },
    };

    const demo = try allocator.create(Demo);
    demo.* = .{
        .name = "Task Management System",
        .description = "Real-world application showing all type kinds",
        .stages = stages,
        .allocator = allocator,
    };

    return demo;
}
