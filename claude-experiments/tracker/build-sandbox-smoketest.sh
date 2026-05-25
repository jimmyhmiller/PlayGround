#!/bin/bash
#
# build-sandbox-smoketest.sh — Build the MAS variant (sandboxed, no Sparkle)
# but sign with the local Development identity so we can run and test it
# *before* obtaining the App Store distribution cert.
#
# This catches sandbox-related runtime failures (CloudKit, NSSavePanel,
# file access) that you'd otherwise discover only after a rejected MAS
# submission.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP="Ease-Sandbox.app"
IDENTITY="${CODESIGN_IDENTITY:-Apple Development: jimmyhmiller@gmail.com (NPAKY9T8P8)}"

echo "→ Cleaning"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

echo "→ Building (EASE_MAS=1, debug)"
rm -rf .build
EASE_MAS=1 swift build

echo "→ Assembling bundle"
cp ".build/debug/Ease" "$APP/Contents/MacOS/Ease"
chmod +x "$APP/Contents/MacOS/Ease"
cp Info.MAS.plist "$APP/Contents/Info.plist"

cat > "$APP/Contents/Resources/MenuBarConfig.plist" << 'EOF'
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

echo "→ Compiling icon"
xcrun actool Ease/AppIcon.icon \
    --compile "$APP/Contents/Resources" \
    --app-icon AppIcon \
    --enable-on-demand-resources NO \
    --development-region en \
    --target-device mac \
    --platform macosx \
    --enable-icon-stack-fallback-generation=disabled \
    --include-all-app-icons \
    --minimum-deployment-target 14.0 \
    --output-partial-info-plist /tmp/icon-partial-smoke.plist >/dev/null

LIGHT_SOURCE="$SCRIPT_DIR/icons/source/icon-1024.png"
rm -rf /tmp/legacy.iconset
mkdir -p /tmp/legacy.iconset
for sz in 16 32 128 256 512; do
    sips -z $sz $sz "$LIGHT_SOURCE" --out "/tmp/legacy.iconset/icon_${sz}x${sz}.png" >/dev/null
    sips -z $((sz*2)) $((sz*2)) "$LIGHT_SOURCE" --out "/tmp/legacy.iconset/icon_${sz}x${sz}@2x.png" >/dev/null
done
iconutil -c icns /tmp/legacy.iconset -o "$APP/Contents/Resources/AppIcon.icns"

# Embed local development provisioning profile (gives CloudKit access)
PROFILE_PATH="$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles/62bd6a42-5425-4524-a997-49a1662bb6d7.provisionprofile"
if [ -f "$PROFILE_PATH" ]; then
    cp "$PROFILE_PATH" "$APP/Contents/embedded.provisionprofile"
fi

echo "→ Signing with $IDENTITY (sandbox entitlements)"
codesign --force --options runtime \
    --entitlements Ease.MAS.entitlements \
    --sign "$IDENTITY" \
    "$APP"

codesign --verify --deep --strict --verbose=2 "$APP"

echo
echo "✓ Sandboxed app built at $APP"
echo "  Launch: open $APP"
echo "  Logs:   log stream --predicate 'process == \"Ease\"' --level debug"
