#!/bin/bash

echo "🔨 Building CCSeva.app bundle..."

# Build the Swift package
swift build -c release

# Create app bundle structure
APP_NAME="CCSeva.app"
rm -rf "$APP_NAME"
mkdir -p "$APP_NAME/Contents/MacOS"
mkdir -p "$APP_NAME/Contents/Resources"

# Copy executable
cp .build/release/CCSeva "$APP_NAME/Contents/MacOS/"

# Create Info.plist
cat > "$APP_NAME/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>CCSeva</string>
    <key>CFBundleIdentifier</key>
    <string>com.ccsevaswift.CCSeva</string>
    <key>CFBundleName</key>
    <string>CCSeva</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo "✅ Created $APP_NAME"
echo "🚀 To run: open $APP_NAME"
echo "🛑 To stop: pkill CCSeva"