#!/bin/bash

# QuasiLauncher Build Script
set -e

echo "🚀 Building QuasiLauncher..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf .build
rm -rf QuasiLauncher.app

# Build the Swift package
echo "🔨 Building Swift package..."
swift build -c release

# Create app bundle structure
echo "📦 Creating app bundle..."
mkdir -p QuasiLauncher.app/Contents/MacOS
mkdir -p QuasiLauncher.app/Contents/Resources

# Copy executable
echo "📋 Copying executable..."
cp .build/release/QuasiLauncher QuasiLauncher.app/Contents/MacOS/

# Copy Info.plist
echo "📋 Copying Info.plist..."
cp Info.plist QuasiLauncher.app/Contents/

# Copy entitlements
echo "📋 Copying entitlements..."
cp QuasiLauncher.entitlements QuasiLauncher.app/Contents/

# Create a simple icon (you can replace this with a proper .icns file)
echo "🎨 Creating placeholder icon..."
mkdir -p QuasiLauncher.app/Contents/Resources
# You would normally copy an .icns file here
# cp icon.icns QuasiLauncher.app/Contents/Resources/

# Sign the app with entitlements using a proper certificate
echo "🔐 Signing app with entitlements..."

# Try to find a valid code signing certificate
CERT_NAME="QuasiLauncher Developer"
if security find-certificate -c "$CERT_NAME" >/dev/null 2>&1; then
    echo "Using certificate: $CERT_NAME"
    SIGN_IDENTITY="$CERT_NAME"
else
    # Fallback: try to find any Developer ID or development certificate
    SIGN_IDENTITY=$(security find-identity -v -p codesigning | grep -E "Developer ID Application|Apple Development|Mac Developer" | head -1 | awk -F'"' '{print $2}')
    
    if [ -z "$SIGN_IDENTITY" ]; then
        echo "⚠️  No code signing certificate found!"
        echo "Creating self-signed certificate..."
        ./create-certificate.sh
        SIGN_IDENTITY="QuasiLauncher Developer"
    else
        echo "Using certificate: $SIGN_IDENTITY"
    fi
fi

# Sign with proper identity
codesign --force --deep --sign "$SIGN_IDENTITY" --entitlements QuasiLauncher.entitlements QuasiLauncher.app

# Verify the signature
echo "🔍 Verifying signature..."
codesign -dv QuasiLauncher.app 2>&1 | grep -E "Signature|TeamIdentifier|Identifier"

echo "✅ Build complete! QuasiLauncher.app is ready."
echo ""
echo "To run the app:"
echo "  open QuasiLauncher.app"
echo ""
echo "To install for development:"
echo "  cp -r QuasiLauncher.app /Applications/"
echo ""
echo "⚠️  Remember to grant accessibility permissions in System Preferences > Security & Privacy > Privacy > Accessibility"