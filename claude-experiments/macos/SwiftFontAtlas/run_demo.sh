#!/bin/bash

# SwiftFontAtlas Demo Runner
# This script builds and runs the demo application

set -e

echo "🚀 SwiftFontAtlas Demo"
echo "====================="

# Check if we're in the right directory
if [ ! -f "Package.swift" ]; then
    echo "❌ Error: Please run this script from the SwiftFontAtlas directory"
    exit 1
fi

echo "📦 Building SwiftFontAtlas library..."
swift build

if [ $? -eq 0 ]; then
    echo "✅ Library built successfully"
else
    echo "❌ Failed to build library"
    exit 1
fi

echo ""
echo "🎨 Running FontAtlas Demo App..."
echo "   - Use different fonts and sizes"
echo "   - Try the 'Prerender ASCII' button"
echo "   - Test custom text rendering"
echo "   - Watch the atlas visualization"
echo "   - Try the stress test to see atlas growth"
echo ""

# Run the demo app
swift -I .build/debug FontAtlasDemoApp.swift

echo ""
echo "👋 Demo completed!"