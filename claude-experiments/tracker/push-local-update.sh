#!/bin/bash
# Push a local update for testing Sparkle auto-updates
#
# Usage:
#   ./push-local-update.sh           # Auto-increment build number
#   ./push-local-update.sh 1.2.0     # Set specific version (auto-increments build)
#   ./push-local-update.sh 1.2.0 5   # Set specific version and build number
#
# This script:
#   1. Increments the version in Info.plist
#   2. Builds the app
#   3. Creates Ease.zip for the update
#   4. Updates appcast.xml with new version info
#   5. Starts the local update server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INFO_PLIST="Ease.app/Contents/Info.plist"
APPCAST="local-updates/appcast.xml"
SIGN_UPDATE=".build/artifacts/sparkle/Sparkle/bin/sign_update"

# Get current version info
CURRENT_SHORT_VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$INFO_PLIST")
CURRENT_BUILD=$(/usr/libexec/PlistBuddy -c "Print :CFBundleVersion" "$INFO_PLIST")

# Determine new version
if [ -n "$1" ]; then
    NEW_SHORT_VERSION="$1"
else
    NEW_SHORT_VERSION="$CURRENT_SHORT_VERSION"
fi

if [ -n "$2" ]; then
    NEW_BUILD="$2"
else
    NEW_BUILD=$((CURRENT_BUILD + 1))
fi

echo "=== Pushing Local Update ==="
echo "Current: v$CURRENT_SHORT_VERSION (build $CURRENT_BUILD)"
echo "New:     v$NEW_SHORT_VERSION (build $NEW_BUILD)"
echo ""

# Update Info.plist
echo "Updating Info.plist..."
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $NEW_SHORT_VERSION" "$INFO_PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $NEW_BUILD" "$INFO_PLIST"

# Build the app
echo "Building release..."
swift build -c release

# Copy executable to app bundle
echo "Updating app bundle..."
cp .build/release/Ease Ease.app/Contents/MacOS/

# Add rpath for Frameworks directory
install_name_tool -add_rpath @executable_path/../Frameworks Ease.app/Contents/MacOS/Ease 2>/dev/null || true

# Copy Sparkle framework to app bundle
SPARKLE_PATH=$(find .build -name "Sparkle.framework" -type d | head -1)
if [ -n "$SPARKLE_PATH" ]; then
    rm -rf Ease.app/Contents/Frameworks/Sparkle.framework
    cp -R "$SPARKLE_PATH" Ease.app/Contents/Frameworks/
fi

# Sign the app
echo "Signing app..."
codesign --force --deep --sign - Ease.app

# Create zip for update
echo "Creating Ease.zip..."
rm -f local-updates/Ease.zip
cd Ease.app/..
zip -r local-updates/Ease.zip Ease.app
cd "$SCRIPT_DIR"

# Get zip size
ZIP_SIZE=$(stat -f%z local-updates/Ease.zip)

# Sign the update with EdDSA
echo "Signing update with EdDSA..."
SIGNATURE=$("$SIGN_UPDATE" -p local-updates/Ease.zip)
echo "Signature: $SIGNATURE"

# Update appcast.xml
echo "Updating appcast.xml..."
PUB_DATE=$(date -R)

cat > "$APPCAST" << EOF
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <channel>
        <title>Ease Updates (Local Dev)</title>
        <link>http://localhost:8080/appcast.xml</link>
        <description>Local development update feed for Ease</description>
        <language>en</language>
        <item>
            <title>Version $NEW_SHORT_VERSION</title>
            <description><![CDATA[
                <h2>What's New in $NEW_SHORT_VERSION</h2>
                <ul>
                    <li>Local development build</li>
                </ul>
            ]]></description>
            <pubDate>$PUB_DATE</pubDate>
            <sparkle:version>$NEW_BUILD</sparkle:version>
            <sparkle:shortVersionString>$NEW_SHORT_VERSION</sparkle:shortVersionString>
            <sparkle:minimumSystemVersion>13.0</sparkle:minimumSystemVersion>
            <enclosure url="http://localhost:8080/Ease.zip"
                       sparkle:version="$NEW_BUILD"
                       sparkle:shortVersionString="$NEW_SHORT_VERSION"
                       sparkle:edSignature="$SIGNATURE"
                       length="$ZIP_SIZE"
                       type="application/octet-stream"/>
        </item>
    </channel>
</rss>
EOF

echo ""
echo "=== Update Ready ==="
echo "Version: $NEW_SHORT_VERSION (build $NEW_BUILD)"
echo "Zip size: $ZIP_SIZE bytes"
echo ""
echo "Starting local server on http://localhost:8080"
echo ""
echo "To apply update:"
echo "  1. In app: Right-click menu bar icon → Update Channel → Local Dev"
echo "  2. In app: Right-click menu bar icon → Check for Updates..."
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd local-updates
python3 -m http.server 8080
