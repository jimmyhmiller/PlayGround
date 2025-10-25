const std = @import("std");
const ts = @import("../types/type_system.zig");

/// Theme mode
pub const ThemeMode = enum {
    light,
    dark,
};

/// Color scheme for a specific type kind
pub const TypeColors = struct {
    primary: ts.Color,      // Main color for the type
    secondary: ts.Color,    // Lighter shade for backgrounds
    accent: ts.Color,       // Darker shade for outlines/borders
    text: ts.Color,         // Text color
    text_header: ts.Color,  // Header text color (usually white or very light)

    pub fn init(hex: []const u8) !TypeColors {
        const primary = try ts.Color.fromHex(hex);

        // Generate secondary (much lighter, fully opaque to hide shadow)
        const secondary = ts.Color.init(
            if (primary.r <= 180) primary.r + 75 else 255,
            if (primary.g <= 180) primary.g + 75 else 255,
            if (primary.b <= 180) primary.b + 75 else 255,
            255,  // Fully opaque to hide shadow underneath
        );

        // Generate accent (darker for borders)
        const accent = ts.Color.init(
            if (primary.r >= 40) primary.r - 40 else 0,
            if (primary.g >= 40) primary.g - 40 else 0,
            if (primary.b >= 40) primary.b - 40 else 0,
            255,
        );

        return .{
            .primary = primary,
            .secondary = secondary,
            .accent = accent,
            .text = ts.Color.init(45, 55, 72, 255),          // Darker, more readable
            .text_header = ts.Color.init(255, 255, 255, 255),
        };
    }
};

/// Complete theme configuration
pub const Theme = struct {
    mode: ThemeMode,
    background: ts.Color,
    grid: ts.Color,

    // Type-specific colors
    primitive: TypeColors,
    struct_type: TypeColors,
    enum_type: TypeColors,
    function_type: TypeColors,
    tuple_type: TypeColors,
    optional_type: TypeColors,

    // Validation colors
    valid_reference: ts.Color,
    invalid_reference: ts.Color,
    warning: ts.Color,

    // UI colors
    ui_text: ts.Color,
    ui_text_secondary: ts.Color,
    ui_button: ts.Color,
    ui_button_hover: ts.Color,
    ui_button_active: ts.Color,

    pub fn light() !Theme {
        return .{
            .mode = .light,
            .background = ts.Color.init(252, 252, 253, 255),
            .grid = ts.Color.init(240, 240, 242, 255),

            // Very subtle, muted pastels - easier on the eyes
            .primitive = try TypeColors.init("#8B9FD9"),      // Muted blue
            .struct_type = try TypeColors.init("#D98B9F"),    // Muted rose
            .enum_type = try TypeColors.init("#8BD99F"),      // Muted mint
            .function_type = try TypeColors.init("#D9B98B"),  // Muted amber
            .tuple_type = try TypeColors.init("#B98BD9"),     // Muted lavender
            .optional_type = try TypeColors.init("#9FA8B8"),  // Muted slate

            // Validation
            .valid_reference = ts.Color.init(34, 197, 94, 255),    // Green
            .invalid_reference = ts.Color.init(239, 68, 68, 255),  // Red
            .warning = ts.Color.init(251, 191, 36, 255),           // Yellow

            // UI
            .ui_text = ts.Color.init(30, 30, 40, 255),
            .ui_text_secondary = ts.Color.init(100, 100, 110, 255),
            .ui_button = ts.Color.init(100, 116, 139, 255),
            .ui_button_hover = ts.Color.init(71, 85, 105, 255),
            .ui_button_active = ts.Color.init(51, 65, 85, 255),
        };
    }

    pub fn dark() !Theme {
        return .{
            .mode = .dark,
            .background = ts.Color.init(17, 24, 39, 255),
            .grid = ts.Color.init(31, 41, 55, 255),

            // Brighter, more saturated colors for dark mode
            .primitive = try TypeColors.init("#7C9EFF"),
            .struct_type = try TypeColors.init("#FF6B9D"),
            .enum_type = try TypeColors.init("#6EE7B7"),
            .function_type = try TypeColors.init("#FBBF24"),
            .tuple_type = try TypeColors.init("#F472B6"),
            .optional_type = try TypeColors.init("#CBD5E1"),

            // Validation (brighter for dark bg)
            .valid_reference = ts.Color.init(74, 222, 128, 255),
            .invalid_reference = ts.Color.init(248, 113, 113, 255),
            .warning = ts.Color.init(252, 211, 77, 255),

            // UI
            .ui_text = ts.Color.init(241, 245, 249, 255),
            .ui_text_secondary = ts.Color.init(148, 163, 184, 255),
            .ui_button = ts.Color.init(71, 85, 105, 255),
            .ui_button_hover = ts.Color.init(100, 116, 139, 255),
            .ui_button_active = ts.Color.init(148, 163, 184, 255),
        };
    }

    /// Get colors for a specific type kind
    pub fn colorsForType(self: *const Theme, kind: ts.TypeKind) TypeColors {
        return switch (kind) {
            .primitive => self.primitive,
            .@"struct" => self.struct_type,
            .@"enum" => self.enum_type,
            .function => self.function_type,
            .tuple => self.tuple_type,
            .optional => self.optional_type,
            .recursive => self.struct_type, // Treat like struct for now
        };
    }

    /// Toggle between light and dark mode
    pub fn toggle(self: *Theme) !void {
        self.* = if (self.mode == .light)
            try Theme.dark()
        else
            try Theme.light();
    }
};

/// Global theme instance
var current_theme: ?Theme = null;

/// Get the current theme (initializes to light if not set)
pub fn getCurrentTheme() !*Theme {
    if (current_theme == null) {
        current_theme = try Theme.light();
    }
    return &current_theme.?;
}

/// Toggle the global theme
pub fn toggleTheme() !void {
    const theme = try getCurrentTheme();
    try theme.toggle();
}
