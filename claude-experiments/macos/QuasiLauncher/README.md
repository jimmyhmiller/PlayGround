# QuasiLauncher 🚀

A macOS quasi-modal application launcher inspired by Humanized Enso. QuasiLauncher provides lightning-fast access to applications, files, and system commands through an elegant overlay interface activated by holding the Escape key.

## Features ✨

### Quasi-Modal Interface
- **Hold Escape** to activate the command overlay
- **Type commands** while holding the key
- **Release to execute** the selected command
- **Transparent overlay** with glass morphism effects

### Smart Command Recognition
- **Application launching**: `open safari`, `open terminal`
- **File operations**: `open documents`, `open downloads`
- **Web search**: `search swift programming`, `google macos development`
- **System controls**: `volume up`, `brightness down`, `sleep`
- **Calculations**: `calculate 15 * 24 + 100`

### Advanced Features
- **Menu bar interface** with real-time status indicators
- **Fuzzy matching** for applications and files
- **Real-time suggestions** with keyboard navigation
- **Recent files integration** from Documents, Downloads, Desktop
- **System command execution** via AppleScript
- **Spotlight-like file search** across user directories
- **Debug information** and accessibility status monitoring

## Installation 📦

### Prerequisites
- macOS 13.0 or later
- Xcode Command Line Tools
- Swift 5.9+

### Build from Source

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd QuasiLauncher
   ```

2. **Build the application**:
   ```bash
   ./build.sh
   ```

3. **Install the app**:
   ```bash
   cp -r QuasiLauncher.app /Applications/
   ```

4. **Launch QuasiLauncher**:
   ```bash
   open /Applications/QuasiLauncher.app
   ```

### Permissions Setup

QuasiLauncher requires specific permissions to function properly:

1. **Accessibility Access**:
   - Go to **System Preferences** > **Security & Privacy** > **Privacy** > **Accessibility**
   - Click the lock to make changes
   - Add QuasiLauncher and enable it

2. **Input Monitoring** (for global key detection):
   - Go to **System Preferences** > **Security & Privacy** > **Privacy** > **Input Monitoring**
   - Add QuasiLauncher and enable it
   - **Required for Escape key detection**

3. **Apple Events** (for system commands):
   - The app will request permission when first using system commands
   - Grant access to control System Events, Finder, and System Preferences

### Important: Entitlements and Code Signing

The app is automatically built with proper entitlements and code signing:
- `com.apple.security.device.input-monitoring` - Required for global key monitoring
- `com.apple.security.automation.apple-events` - Required for system control
- Self-signed certificate for development use

## Usage 🎯

### Menu Bar Interface

QuasiLauncher adds a status icon to your menu bar with live status indicators:

- **🔒** - Accessibility permissions needed (red)
- **◊** - Ready and waiting (system accent color)
- **⌨️** - Escape key detected (orange)
- **🚀** - Command overlay active (blue)

**Left-click** the menu bar icon to see:
- Real-time status of accessibility permissions
- Hotkey detection state
- Overlay visibility status
- Key press counter for debugging
- Quick toggle and settings buttons

**Right-click** the menu bar icon for:
- Toggle Overlay
- Settings...
- Quit QuasiLauncher

### Basic Commands

#### Application Launching
- `open safari` - Launch Safari
- `open terminal` - Launch Terminal
- `open` + partial app name for suggestions

#### File Operations
- `open documents` - Open Documents folder
- `open downloads` - Open Downloads folder
- Type any filename to search recent files

#### Web Search
- `search macos development` - Google search
- `google swift programming` - Google search
- `youtube swift tutorials` - YouTube search

#### System Commands
- `volume up/down` - Adjust system volume
- `brightness up/down` - Adjust screen brightness
- `sleep` - Put system to sleep
- `lock screen` - Lock the screen
- `wifi toggle` - Toggle WiFi
- `empty trash` - Empty the Trash

#### Calculations
- `calculate 15 * 24` - Basic math
- `calc 100 + 25 * 3` - Expression evaluation

### Navigation
- **Arrow keys** to navigate suggestions
- **Enter** to execute selected command
- **Escape** to cancel
- **Release Caps Lock** to execute top suggestion

## Architecture 🏗️

### Core Components

```
QuasiLauncher/
├── Sources/
│   ├── App/
│   │   ├── main.swift              # App entry point
│   │   └── AppDelegate.swift       # App lifecycle management
│   ├── Core/
│   │   ├── GlobalHotkeyManager.swift   # Caps Lock monitoring
│   │   ├── OverlayWindowManager.swift  # Transparent window system
│   │   └── CommandEngine.swift         # Command parsing & execution
│   ├── UI/
│   │   ├── CommandOverlay.swift    # Main overlay interface
│   │   └── SettingsView.swift      # Settings panel
│   └── Services/
│       ├── ApplicationService.swift    # App discovery & launching
│       ├── FileService.swift          # File search & operations
│       └── SystemService.swift        # System command execution
```

### Key Technologies
- **SwiftUI** - Modern declarative UI framework
- **Carbon APIs** - Low-level keyboard event monitoring
- **NSWorkspace** - Application and file launching
- **AppleScript** - System command execution
- **Core Graphics** - Event tap management

## Development 🛠️

### Building for Development

```bash
# Build and run directly
swift run

# Build release version
swift build -c release

# Build app bundle
./build.sh
```

### Code Structure

- **Singleton Pattern** for core services
- **Publisher/Subscriber** for real-time updates
- **Protocol-oriented** command system
- **Async/await** for file operations
- **Combine** for reactive UI updates

### Adding New Commands

1. Extend `CommandEngine.getSuggestions(for:)` with new patterns
2. Add execution logic in `CommandEngine.executeCommand(_:)`
3. Create service methods in appropriate service class
4. Update command suggestions with new icons and descriptions

## Customization ⚙️

### Settings Panel
Access via the Settings scene (⌘,):

- **Interaction Mode**: Quasi-modal vs. sticky modal
- **Hotkey Selection**: Escape, Caps Lock, Option, Control, Command
- **Overlay Appearance**: Opacity and visual effects
- **Command Preferences**: Default search engines, file locations

### Configuration
Settings are stored in `UserDefaults` with keys:
- `isQuasiModalEnabled` - Interaction mode
- `selectedHotkey` - Activation key
- `overlayOpacity` - UI transparency

## Troubleshooting 🔧

### Common Issues

#### Escape Key Not Working
- Check the menu bar icon status - it should show **⌨️** when Escape is pressed
- Ensure Accessibility permissions are granted (icon will be **🔒** if not)
- View the debug counter in the menu bar popover to see if key events are detected
- Check Console.app for debug logs showing "Key event" and "Escape key" messages
- Restart the application after permission changes

#### Commands Not Executing
- Verify Apple Events permissions
- Check Console.app for error messages
- Ensure target applications are installed

#### Performance Issues
- Application cache refreshes every 5 minutes
- File search limited to 3 directory levels
- Suggestion results capped at 8 items

### Debug Mode
Enable verbose logging:
```bash
# Set debug environment variable
export QUASI_DEBUG=1
./QuasiLauncher
```

## Contributing 🤝

We welcome contributions! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with clear commit messages
4. Test thoroughly on macOS 13+
5. Submit a pull request

### Areas for Contribution
- Additional system commands
- Plugin architecture for custom commands
- Alternative activation methods
- Performance optimizations
- UI/UX improvements

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- **Humanized Enso** - Original inspiration for quasi-modal interaction
- **Jef Raskin** - Pioneer of humane interface design
- **Apple** - SwiftUI and macOS frameworks
- **Open Source Community** - Various libraries and inspiration

## Roadmap 🗺️

### Version 1.1
- [ ] Plugin system for custom commands
- [ ] Command history and learning
- [ ] Multiple hotkey support
- [ ] Customizable themes

### Version 1.2
- [ ] Natural language processing
- [ ] Cloud sync for settings
- [ ] Integration with more system APIs
- [ ] Voice command support

---

**Built with ❤️ using Swift and SwiftUI**