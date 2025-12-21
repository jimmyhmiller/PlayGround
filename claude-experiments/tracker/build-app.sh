#!/bin/bash
set -e

# Build release version
echo "Building release..."
swift build -c release

# Copy executable to app bundle
echo "Creating app bundle..."
cp .build/release/Tracker Tracker.app/Contents/MacOS/

# Sign the app (replace with your identity if you have one)
# To find your identity: security find-identity -v -p codesigning
IDENTITY="${CODESIGN_IDENTITY:-}"

if [ -n "$IDENTITY" ]; then
    echo "Signing with identity: $IDENTITY"
    codesign --force --deep --sign "$IDENTITY" Tracker.app
else
    echo "No CODESIGN_IDENTITY set, signing for local use only..."
    codesign --force --deep --sign - Tracker.app
fi

echo "App bundle created at Tracker.app"

# Create DMG
echo "Creating DMG..."
rm -f Tracker.dmg
hdiutil create -volname "Tracker" -srcfolder Tracker.app -ov -format UDZO Tracker.dmg

echo "Done! Tracker.dmg created"
echo ""
echo "To distribute publicly, you need to:"
echo "1. Sign with a Developer ID certificate (requires Apple Developer account)"
echo "2. Notarize the app with: xcrun notarytool submit Tracker.dmg --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID --password YOUR_APP_SPECIFIC_PASSWORD"
echo "3. Staple the notarization: xcrun stapler staple Tracker.dmg"
