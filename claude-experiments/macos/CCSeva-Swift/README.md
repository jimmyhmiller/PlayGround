# CCSeva Swift

A native macOS Swift port of CCSeva - A menu bar application for real-time Claude Code usage tracking.

## Attribution

This project is a Swift port of the original [CCSeva](https://github.com/Iamshankhadeep/ccseva) by Iamshankhadeep.

**Original CCSeva:**
- Repository: https://github.com/Iamshankhadeep/ccseva
- Author: Iamshankhadeep
- License: MIT
- Technology: Electron + React + TypeScript

**CCSeva Swift:**
- Technology: Native Swift + SwiftUI
- Platform: macOS 13.0+
- License: MIT

## Purpose

Track and monitor Claude Code usage in real-time through a native macOS menu bar application. Features include:

- Real-time token usage monitoring
- Menu bar percentage/cost display
- Usage analytics and trends
- Notifications at usage thresholds
- Native macOS integration

## Development

This is a step-by-step port focusing on native performance and macOS integration while maintaining the core functionality of the original CCSeva.

### Prerequisites

CCSeva Swift reads Claude usage data directly from your `~/.claude` directory. No external dependencies required!

### Quick Start

```bash
# Clone or navigate to the project directory
cd CCSeva-Swift

# Build and run as proper macOS app (RECOMMENDED)
./run

# Alternative: Build then launch manually
./build-app.sh
open CCSeva.app

# To stop the app
pkill CCSeva
```

**Important:** 
- Reads Claude usage data directly from `~/.claude` directory
- For the menu bar item to appear properly, use the `./run` or `./build-app.sh` methods
- Make sure you've used Claude Code at least once to generate usage data

### Features Implemented

- ✅ **Menu bar integration** - Native macOS menu bar app with popover
- ✅ **Real-time data display** - Toggles between percentage and cost every 3 seconds
- ✅ **Native Claude usage reading** - Directly reads JSONL files from ~/.claude (no external dependencies)
- ✅ **Dashboard view** - Usage progress, stats grid, model breakdown
- ✅ **Analytics view** - Velocity info, weekly usage, predictions, reset information
- ✅ **Settings view** - Configuration options
- ✅ **Auto-refresh** - Updates every 30 seconds like original CCSeva
- ✅ **Swift Package Manager** - Easy building with `./run` command
- ✅ **Proper app bundle** - Native macOS .app structure for menu bar access

### Project Structure

```
CCSeva-Swift/
├── Package.swift                    # Swift Package Manager configuration
├── Sources/
│   └── CCSeva/
│       ├── main.swift              # App entry point and NSApplication setup
│       ├── ContentView.swift       # SwiftUI interface with tabs
│       ├── CCUsageService.swift    # Native Claude usage service
│       ├── ClaudeUsageReader.swift # Direct JSONL file parser
│       └── UsageModels.swift       # Data models for usage statistics
├── Sources/TestMenuBar/
│   └── main.swift                  # Test app for debugging menu bar issues
├── build-app.sh                    # Script to create proper .app bundle
├── run                             # Quick run script
├── CCSeva.entitlements            # macOS permissions
└── README.md
```

### How It Works

1. **Menu Bar Display**: Shows real-time usage percentage/cost that updates every 30 seconds
2. **Data Fetching**: Directly reads JSONL files from `~/.claude/projects/` to get real Claude usage data
3. **Dashboard**: Comprehensive usage overview with progress bars, stats, and model breakdown
4. **Analytics**: Detailed velocity tracking, weekly trends, predictions, and reset information
5. **Native Integration**: Pure Swift/SwiftUI with proper macOS app bundle for menu bar access

## Credits

- Original CCSeva concept and design: [Iamshankhadeep](https://github.com/Iamshankhadeep)
- Swift port implementation: This project
- Native Claude file parsing: This project

## License

MIT License - See LICENSE file for details.