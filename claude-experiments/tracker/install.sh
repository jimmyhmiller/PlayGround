#!/bin/bash
set -e

ICON_STYLE="${1:-new}"

if [[ "$ICON_STYLE" != "new" && "$ICON_STYLE" != "old" ]]; then
    echo "Usage: ./install.sh [new|old]"
    echo "  new - Use AppIcon.icon with Short/Long/Medium bars (default)"
    echo "  old - Use AppIcon-original.icon with Long/Short/Medium bars"
    exit 1
fi

echo "Building with $ICON_STYLE icon..."

# Build
swift build || exit 1

# Kill any running instance
pkill -f "Ease.app" 2>/dev/null || true
sleep 0.5

# Copy executable
cp .build/debug/Ease Ease.app/Contents/MacOS/Ease

# Select icon source
if [[ "$ICON_STYLE" == "new" ]]; then
    ICON_SOURCE="Ease/AppIcon.icon"
    echo "Using new icon: Short / Long / Medium"
else
    ICON_SOURCE="Ease/AppIcon-original.icon"
    echo "Using old icon: Long / Short / Medium"
fi

# Remove old icons and copy selected one
rm -rf Ease.app/Contents/Resources/AppIcon.icon
rm -f Ease.app/Contents/Resources/AppIcon.icns
cp -R "$ICON_SOURCE" Ease.app/Contents/Resources/AppIcon.icon

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

# Copy Sparkle framework
mkdir -p Ease.app/Contents/Frameworks
cp -R .build/arm64-apple-macosx/debug/Sparkle.framework Ease.app/Contents/Frameworks/

# Fix rpath
install_name_tool -add_rpath @executable_path/../Frameworks Ease.app/Contents/MacOS/Ease 2>/dev/null || true

# Re-sign
codesign --force --deep --sign - Ease.app

echo "Done! Launching..."
open Ease.app
