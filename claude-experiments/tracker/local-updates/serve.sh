#!/bin/bash
# Serve local updates for testing Sparkle auto-updates
#
# Usage:
#   ./local-updates/serve.sh
#
# This starts a simple HTTP server on port 8080 serving the local-updates directory.
# Make sure to:
# 1. Build a new version with a higher version number
# 2. Create Ease.zip from the app bundle
# 3. Update appcast.xml with the new version info
# 4. Switch to "Local Dev" channel in the app
# 5. Click "Check for Updates..."

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting local update server on http://localhost:8080"
echo ""
echo "To test updates:"
echo "  1. Build app with higher version: Update Info.plist CFBundleVersion and CFBundleShortVersionString"
echo "  2. Create zip: cd Ease.app/.. && zip -r local-updates/Ease.zip Ease.app"
echo "  3. Update local-updates/appcast.xml with new version"
echo "  4. In app: Right-click menu bar icon → Update Channel → Local Dev"
echo "  5. In app: Right-click menu bar icon → Check for Updates..."
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$SCRIPT_DIR"
python3 -m http.server 8080
