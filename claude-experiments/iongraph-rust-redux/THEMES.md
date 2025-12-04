# CodeGraph Themes and Configuration

## Overview

CodeGraph supports customizable themes and layout configurations, allowing different compilers and users to personalize the appearance of their visualizations.

## Configuration Files

Configuration files are stored in the `config/` directory and use TOML format.

### Themes

Themes control colors and visual appearance:
- `config/ion-default.toml` - Original IonGraph colors for SpiderMonkey
- `config/dark.toml` - Dark theme for late-night debugging
- `config/llvm.toml` - LLVM-inspired color scheme

### Layouts

Layout configurations control spacing and sizing:
- `config/layout-default.toml` - Standard spacing (original IonGraph)
- `config/layout-compact.toml` - Tighter spacing for large graphs

## Theme File Format

### Complete Example

```toml
[metadata]
name = "My Custom Theme"
description = "A beautiful custom color scheme"
compiler = "ion"  # optional: target compiler

[blocks]
header = "#0c0c0d"        # Block header background
loop_header = "#1fa411"   # Loop header background
backedge = "#ff6600"      # Backedge block background
border = "#000000"        # Block border color
text = "#ffffff"          # Text color

[instruction_attributes]
# Map instruction attributes to colors
Movable = "#1048af"
Guard = "#000000"
RecoveredOnBailout = "#444444"

[heatmap]
enabled = true
hot = "#ff849e"          # Hot execution path color
cool = "#ffe546"         # Cool execution path color
threshold = 0.2          # Hot/cool threshold (0.0-1.0)

[arrows]
normal = "#000000"       # Normal forward edge
backedge = "#ff0000"     # Loop backedge
loop_header = "#1fa411"  # Loop header edge
```

### Required Fields

Only these sections are required (all have sensible defaults):
- `[blocks]` - Block colors
- `[arrows]` - Arrow colors

All other sections are optional.

### Color Format

Colors must be in hex format: `#RRGGBB`

## Layout File Format

### Complete Example

```toml
[blocks]
margin_x = 20.0      # Horizontal margin between blocks
margin_y = 30.0      # Vertical margin between layers
padding = 20.0       # Internal padding
gap = 44.0           # Gap between blocks in same layer

[arrows]
radius = 12.0              # Arrow curve radius
track_padding = 36.0       # Track padding for routing
joint_spacing = 16.0       # Spacing between joints
port_start = 16.0          # Port starting position
port_spacing = 60.0        # Spacing between ports
header_pushdown = 16.0     # Arrow pushdown for headers

[text]
character_width = 7.2      # Character width estimate
line_height = 16.0         # Line height
font_family = "monospace"  # Font family
font_size = 12.0           # Font size in pixels

[backedge]
loop_margin = 7.0          # Loop-specific margin
backedge_margin = 7.0      # Backedge-specific margin
```

## Using Themes

### From Rust Code

```rust
use iongraph_rust_redux::config::Theme;

// Load theme from file
let theme = Theme::load("config/ion-default.toml")?;

// Use theme colors
let header_color = theme.block_header_color();
let loop_color = theme.loop_header_color();

// Get instruction attribute color
if let Some(color) = theme.instruction_attribute_color("Movable") {
    println!("Movable instructions are: {}", color);
}
```

### From Command Line

Test themes with the `test_themes` tool:

```bash
# Test a single theme
cargo run --release --bin test_themes config/dark.toml

# Test theme + layout combination
cargo run --release --bin test_themes config/llvm.toml config/layout-compact.toml
```

## Creating Custom Themes

### Step 1: Copy an Existing Theme

```bash
cp config/ion-default.toml config/my-theme.toml
```

### Step 2: Edit Colors

Open `config/my-theme.toml` and modify:

```toml
[metadata]
name = "My Theme"
description = "My custom color scheme"

[blocks]
header = "#2c3e50"        # Dark blue-gray
loop_header = "#27ae60"   # Green
backedge = "#e74c3c"      # Red

[instruction_attributes]
Movable = "#3498db"       # Blue
Guard = "#f39c12"         # Orange
```

### Step 3: Test Your Theme

```bash
cargo run --bin test_themes config/my-theme.toml
```

## Compiler-Specific Themes

Different compilers can have their own default themes:

### Ion (SpiderMonkey)
- **Attributes**: Movable, Guard, RecoveredOnBailout, InWorklist
- **Theme**: `config/ion-default.toml`

### LLVM
- **Attributes**: nounwind, readonly, noalias, inbounds, nsw, nuw
- **Theme**: `config/llvm.toml`

### Custom Compilers

When creating a theme for a new compiler:

1. Identify the instruction attributes used by your compiler
2. Choose colors that help distinguish different attribute types
3. Set the `compiler` field in metadata
4. Map common semantic attributes to appropriate colors

## Theme Design Tips

### Color Accessibility

- Use sufficient contrast between text and background
- Avoid relying solely on color to convey information
- Test with colorblindness simulators

### Semantic Grouping

Group related attributes by color family:
- **Safety/Correctness**: Guards, checks → Red/Orange
- **Optimization Hints**: Movable, readonly → Blue/Cyan
- **Loop Information**: Loop headers, backedges → Green

### Dark vs Light Themes

**Dark Themes:**
- Use lighter text (#d4d4d4 or similar)
- Darker block backgrounds (#1e1e1e to #2a2a2a)
- Medium-brightness accents for loop headers

**Light Themes:**
- Use darker text (#333333 or similar)
- Lighter block backgrounds (#f5f5f5 to #ffffff)
- High-contrast accents for important elements

## Built-in Themes

### Ion Default
- **Best for**: SpiderMonkey JIT development
- **Style**: Original IonGraph colors, high contrast
- **Colors**: Black headers, bright green loops, orange backedges

### Dark
- **Best for**: Late-night debugging sessions
- **Style**: VSCode-inspired dark theme
- **Colors**: Dark gray base, cyan loops, warm accents

### LLVM
- **Best for**: LLVM compiler development
- **Style**: LLVM tool-inspired colors
- **Colors**: Gray base, bright blue loops, amber backedges

## Future Extensions

Planned theme features:

- **CSS Export**: Generate CSS from theme files
- **Theme Inheritance**: Extend existing themes
- **Dynamic Theming**: Switch themes at runtime
- **Color Schemes**: Predefined color palettes (solarized, nord, etc.)
