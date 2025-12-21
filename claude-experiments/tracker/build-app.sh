#!/bin/bash
set -e

# Build release version
echo "Building release..."
swift build -c release

# Copy executable to app bundle
echo "Creating app bundle..."
cp .build/release/Ease Ease.app/Contents/MacOS/

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
