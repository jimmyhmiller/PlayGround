#!/bin/bash

# Build script for Claude Code Manager

set -e

echo "Building Claude Code Manager..."

# Clean previous builds
swift package clean

# Build the project
swift build -c release

# Create app bundle directory
mkdir -p ClaudeCodeManager.app/Contents/MacOS
mkdir -p ClaudeCodeManager.app/Contents/Resources

# Copy executable
cp .build/release/ClaudeCodeManager ClaudeCodeManager.app/Contents/MacOS/

# Create Info.plist
cat > ClaudeCodeManager.app/Contents/Info.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>ClaudeCodeManager</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.claudecodemanager</string>
    <key>CFBundleName</key>
    <string>Claude Code Manager</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo "Build complete! App bundle created at ClaudeCodeManager.app"
echo "To run: open ClaudeCodeManager.app"