#!/bin/bash

# Create a self-signed certificate for code signing
echo "🔐 Setting up code signing for QuasiLauncher..."

# Certificate name
CERT_NAME="QuasiLauncher Developer"

# Check if certificate already exists
if security find-certificate -c "$CERT_NAME" >/dev/null 2>&1; then
    echo "✅ Certificate '$CERT_NAME' already exists"
    exit 0
fi

echo ""
echo "📝 A code signing certificate is required for accessibility permissions to work."
echo ""
echo "Please create one manually in Keychain Access:"
echo ""
echo "1. Open Keychain Access (Cmd+Space, type 'Keychain Access')"
echo ""
echo "2. From the menu bar: Keychain Access > Certificate Assistant > Create a Certificate..."
echo ""
echo "3. In the dialog:"
echo "   • Name: QuasiLauncher Developer"
echo "   • Identity Type: Self Signed Root"
echo "   • Certificate Type: Code Signing"
echo "   • ✅ CHECK 'Let me override defaults'"
echo ""
echo "4. Click Continue through all screens (accept all defaults)"
echo ""
echo "5. Once done, run ./build.sh again"
echo ""
echo "Press Enter to open Keychain Access..."
read

# Open Keychain Access
open "/Applications/Utilities/Keychain Access.app"

echo ""
echo "After creating the certificate, run ./build.sh again"
exit 1