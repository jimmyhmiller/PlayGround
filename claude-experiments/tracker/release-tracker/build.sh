#!/bin/bash
#
# build.sh — Build ReleaseTracker.app with the WidgetKit extension embedded.
#
# Produces:
#   ReleaseTracker.app
#     └─ Contents/PlugIns/ReleaseTrackerWidget.appex
#
# After building, install with:
#   cp -R ReleaseTracker.app /Applications/
#   open /Applications/ReleaseTracker.app
#
# Then add the widget to your desktop: right-click desktop → Edit Widgets,
# search for "Ease Release", drag onto desktop.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP="ReleaseTracker.app"
APPEX="ReleaseTrackerWidget.appex"

IDENTITY="${CODESIGN_IDENTITY:-Apple Development: jimmyhmiller@gmail.com (NPAKY9T8P8)}"

echo "→ Cleaning"
rm -rf "$APP"

echo "→ Building release binaries"
swift build -c release

echo "→ Assembling host app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/PlugIns"
cp .build/release/ReleaseTracker "$APP/Contents/MacOS/ReleaseTracker"
cp Resources/Info.host.plist "$APP/Contents/Info.plist"

echo "→ Assembling widget extension"
WIDGET_PATH="$APP/Contents/PlugIns/$APPEX"
mkdir -p "$WIDGET_PATH/Contents/MacOS"
cp .build/release/ReleaseTrackerWidget "$WIDGET_PATH/Contents/MacOS/ReleaseTrackerWidget"
cp Resources/Info.widget.plist "$WIDGET_PATH/Contents/Info.plist"

echo "→ Signing widget extension"
codesign --force --options runtime \
    --entitlements Resources/ReleaseTrackerWidget.entitlements \
    --sign "$IDENTITY" \
    "$WIDGET_PATH"

echo "→ Signing host app (covers the embedded widget)"
codesign --force --options runtime \
    --entitlements Resources/ReleaseTracker.entitlements \
    --sign "$IDENTITY" \
    "$APP"

echo "→ Verifying"
codesign --verify --deep --strict --verbose=2 "$APP"

echo
echo "✓ Built $APP"
echo
echo "Next steps:"
echo "  1. cp -R $APP /Applications/"
echo "  2. open /Applications/$APP"
echo "     (this registers the widget with the system)"
echo "  3. Right-click desktop → Edit Widgets → search 'Ease Release'"
echo "  4. Drag the widget onto your desktop."
