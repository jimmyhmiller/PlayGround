#!/bin/bash
set -e

ICON_STYLE="${1:-new}"

if [[ "$ICON_STYLE" != "new" && "$ICON_STYLE" != "old" ]]; then
    echo "Usage: ./build-app.sh [new|old]"
    echo "  new - Use AppIcon.icon with Short/Long/Medium bars (default)"
    echo "  old - Use AppIcon-original.icon with Long/Short/Medium bars"
    exit 1
fi

# Build release version
echo "Building release..."
swift build -c release

# Copy executable to app bundle
echo "Creating app bundle..."
cp .build/release/Ease Ease.app/Contents/MacOS/

# Select icon source based on style
if [[ "$ICON_STYLE" == "new" ]]; then
    ICON_SOURCE="Ease/AppIcon.icon"
    echo "Using new icon: Short / Long / Medium"
else
    ICON_SOURCE="Ease/AppIcon-original.icon"
    echo "Using old icon: Long / Short / Medium"
fi

# Create MenuBarConfig.plist with matching proportions
if [[ "$ICON_STYLE" == "new" ]]; then
    cat > Ease.app/Contents/Resources/MenuBarConfig.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>barWidths</key>
    <array>
        <real>7</real>
        <real>14</real>
        <real>11</real>
    </array>
</dict>
</plist>
EOF
else
    cat > Ease.app/Contents/Resources/MenuBarConfig.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>barWidths</key>
    <array>
        <real>14</real>
        <real>8</real>
        <real>11</real>
    </array>
</dict>
</plist>
EOF
fi

# Compile .icon file for macOS Tahoe with dark mode support
echo "Compiling icon with dark mode support..."
xcrun actool "$ICON_SOURCE" \
    --compile Ease.app/Contents/Resources \
    --app-icon AppIcon \
    --enable-on-demand-resources NO \
    --development-region en \
    --target-device mac \
    --platform macosx \
    --enable-icon-stack-fallback-generation=disabled \
    --include-all-app-icons \
    --minimum-deployment-target 14.0 \
    --output-partial-info-plist /tmp/icon-partial.plist

# Create legacy .icns for backward compatibility
echo "Creating legacy icon..."
mkdir -p /tmp/legacy.iconset
LIGHT_SOURCE="$(find /Users/jimmyhmiller/Downloads/icon -name '*-macOS-Default-1024x1024@2x.png' | head -1)"
if [ -n "$LIGHT_SOURCE" ]; then
    sips -z 16 16 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_16x16.png >/dev/null
    sips -z 32 32 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_16x16@2x.png >/dev/null
    sips -z 32 32 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_32x32.png >/dev/null
    sips -z 64 64 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_32x32@2x.png >/dev/null
    sips -z 128 128 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_128x128.png >/dev/null
    sips -z 256 256 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_128x128@2x.png >/dev/null
    sips -z 256 256 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_256x256.png >/dev/null
    sips -z 512 512 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_256x256@2x.png >/dev/null
    sips -z 512 512 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_512x512.png >/dev/null
    sips -z 1024 1024 "$LIGHT_SOURCE" --out /tmp/legacy.iconset/icon_512x512@2x.png >/dev/null
    iconutil -c icns /tmp/legacy.iconset -o Ease.app/Contents/Resources/AppIcon.icns
fi

# Add rpath for Frameworks directory
echo "Setting rpath..."
install_name_tool -add_rpath @executable_path/../Frameworks Ease.app/Contents/MacOS/Ease 2>/dev/null || true

# Copy Sparkle framework to app bundle
echo "Copying Sparkle framework..."
mkdir -p Ease.app/Contents/Frameworks
SPARKLE_PATH=$(find .build -name "Sparkle.framework" -type d | head -1)
if [ -n "$SPARKLE_PATH" ]; then
    rm -rf Ease.app/Contents/Frameworks/Sparkle.framework
    cp -R "$SPARKLE_PATH" Ease.app/Contents/Frameworks/
else
    echo "Warning: Sparkle.framework not found in build directory"
fi

# Clean up old Tracker executable if present
rm -f Ease.app/Contents/MacOS/Tracker

# Sign the app (replace with your identity if you have one)
# To find your identity: security find-identity -v -p codesigning
IDENTITY="${CODESIGN_IDENTITY:-}"

if [ -n "$IDENTITY" ]; then
    echo "Signing with identity: $IDENTITY"
    codesign --force --deep --sign "$IDENTITY" Ease.app
else
    echo "No CODESIGN_IDENTITY set, signing for local use only..."
    codesign --force --deep --sign - Ease.app
fi

echo "App bundle created at Ease.app"

# Create DMG with nice installer layout
echo "Creating DMG..."
rm -f Ease.dmg

# Check if create-dmg is available
if command -v create-dmg &> /dev/null; then
    create-dmg \
        --volname "Ease" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "Ease.app" 150 185 \
        --hide-extension "Ease.app" \
        --app-drop-link 450 185 \
        Ease.dmg \
        Ease.app
else
    echo "Error: create-dmg not found. Install it with: brew install create-dmg"
    exit 1
fi

echo "Done! Ease.dmg created"
echo ""
echo "To distribute publicly, you need to:"
echo "1. Sign with a Developer ID certificate (requires Apple Developer account)"
echo "2. Notarize the app with: xcrun notarytool submit Ease.dmg --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID --password YOUR_APP_SPECIFIC_PASSWORD"
echo "3. Staple the notarization: xcrun stapler staple Ease.dmg"
